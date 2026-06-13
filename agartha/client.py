import asyncio
import inspect
import os
import time
from collections.abc import AsyncIterator

from agartha.config import get_default_temperature, get_max_tool_iterations
from agartha.events import (
    REQUEST_COMPLETE,
    REQUEST_ERROR,
    REQUEST_START,
    STREAM_CHUNK,
    STREAM_COMPLETE,
    STREAM_ERROR,
    STREAM_START,
    TOOL_COMPLETE,
    TOOL_ERROR,
    TOOL_START,
    EventBus,
    Handler,
    RequestCompleteEvent,
    RequestErrorEvent,
    RequestStartEvent,
    StreamChunkEvent,
    StreamCompleteEvent,
    StreamErrorEvent,
    StreamStartEvent,
    ToolCompleteEvent,
    ToolErrorEvent,
    ToolStartEvent,
)
from agartha.extensions.llm.builder import (
    CircuitBreakerConfig,
    CostTrackingConfig,
    ProviderConfig,
    RetryConfig,
    build_provider,
)
from agartha.extensions.llm.cost_tracking import CostTrackingProvider, UsageStats
from agartha.extensions.llm.registry import LLM_PROVIDER_KIND
from agartha.extensions.memory import InMemorySessionStore
from agartha.llm.contracts import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
)
from agartha.memory import Session, SessionStore
from agartha.middleware import Middleware
from agartha.registry import REGISTRY, list_kind
from agartha.tools.registry import dispatch_tool


def _config_from_env(
    *,
    with_retry: bool,
    with_circuit_breaker: bool,
    with_cost_tracking: bool,
) -> ProviderConfig:
    """Resolve LLM_PROVIDER + per-provider env vars into a ProviderConfig.

    Used by `Agartha()` when no explicit provider or config is given.
    """
    name = os.getenv("LLM_PROVIDER", "openai")
    specs = REGISTRY[LLM_PROVIDER_KIND]
    if name not in specs:
        raise ValueError(
            f"Unknown LLM_PROVIDER={name!r}. "
            f"Known: {list_kind(LLM_PROVIDER_KIND)}."
        )
    meta = specs[name].metadata

    api_key = None
    if env := meta.get("api_key_env"):
        api_key = os.getenv(env)
        if not api_key:
            raise RuntimeError(
                f"Set {env} to use the {name!r} provider."
            )

    default_model = None
    if env := meta.get("model_env"):
        default_model = os.getenv(env)

    return ProviderConfig(
        name=name,
        api_key=api_key,
        default_model=default_model,
        retry=RetryConfig() if with_retry else None,
        circuit_breaker=CircuitBreakerConfig() if with_circuit_breaker else None,
        cost_tracking=CostTrackingConfig() if with_cost_tracking else None,
    )


class Agartha:
    """Top-level entry point. Hides provider construction, resilience wiring,
    and request/response plumbing.

    Three construction modes through one constructor:

        Agartha()
            Auto-detect from env. Reads LLM_PROVIDER and the matching
            *_API_KEY / *_MODEL vars. Wraps the provider in retry +
            circuit-breaker + cost-tracking by default.

        Agartha(config=ProviderConfig(...))
            Build the same decorator stack from an explicit config object.

        Agartha(provider=some_provider)
            Use a pre-built provider as-is. You're in control of any
            decorator wrapping; the `with_*` flags are ignored.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        config: ProviderConfig | None = None,
        with_retry: bool = True,
        with_circuit_breaker: bool = True,
        with_cost_tracking: bool = True,
        events: EventBus | None = None,
        memory_store: SessionStore | None = None,
        middleware: list[Middleware] | None = None,
    ):
        if provider is not None and config is not None:
            raise ValueError("Pass either `provider` or `config`, not both.")

        # Share an EventBus across clients by passing the same instance.
        # Default: each client gets its own.
        self.events = events or EventBus()

        # Pipeline middleware: before_request / on_tool_call / after_response
        # hooks the loop calls into. Empty by default (pure pass-through).
        self._middleware = list(middleware or [])

        # Session store + a per-client cache so `client.session("id")` always
        # returns the same Session object (in-process consistency).
        self._memory_store: SessionStore = memory_store or InMemorySessionStore()
        self._sessions: dict[str, Session] = {}

        if provider is not None:
            self._provider = provider
            return

        if config is None:
            config = _config_from_env(
                with_retry=with_retry,
                with_circuit_breaker=with_circuit_breaker,
                with_cost_tracking=with_cost_tracking,
            )
        self._provider = build_provider(config)

    # --- Sessions ---------------------------------------------------------

    def session(self, session_id: str, *, system: str | None = None) -> Session:
        """Get or create a stateful conversation session.

        Same `session_id` returns the same Session object within this client
        (state stays in sync). `system` only applies on first creation; later
        calls with a different `system` value are ignored.

        For cross-process sharing (multiple workers), pass a shared
        `memory_store` to each `Agartha` — they'll see the same persisted
        history even with separate in-memory caches.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(
                session_id=session_id,
                store=self._memory_store,
                client=self,
                system=system,
            )
        return self._sessions[session_id]

    # --- Observability shortcuts -----------------------------------------

    def on(self, event: str, handler: Handler | None = None):
        """Subscribe a handler (sync or async) to a lifecycle event.

        Two forms:

            client.on("request.complete", my_handler)        # direct call
            @client.on("request.complete")                   # decorator
            def my_handler(event): ...
        """
        return self.events.on(event, handler)

    def off(self, event: str, handler: Handler) -> None:
        """Unsubscribe a handler from an event. No-op if it wasn't registered."""
        self.events.off(event, handler)

    # --- Internal emit helpers ------------------------------------------

    async def _complete_with_events(self, request: LLMRequest) -> LLMResponse:
        """Wrap one provider.complete() call in request.start/complete/error events."""
        await self.events.emit(REQUEST_START, RequestStartEvent(request=request))
        start = time.perf_counter()
        try:
            response = await self._provider.complete(request)
        except BaseException as e:
            duration = time.perf_counter() - start
            await self.events.emit(REQUEST_ERROR, RequestErrorEvent(
                request=request, error=e, duration_seconds=duration,
            ))
            raise
        duration = time.perf_counter() - start
        await self.events.emit(REQUEST_COMPLETE, RequestCompleteEvent(
            request=request, response=response, duration_seconds=duration,
        ))
        return response

    async def _dispatch_with_events(self, tc) -> str:
        """Wrap one tool dispatch in tool.start/complete/error events.

        Returns a *content* string suitable for a tool message. Errors are
        converted to content (so the LLM can recover) — they don't propagate.
        """
        await self.events.emit(TOOL_START, ToolStartEvent(call=tc))
        start = time.perf_counter()
        try:
            result = await dispatch_tool(tc.name, tc.arguments)
        except Exception as e:
            duration = time.perf_counter() - start
            await self.events.emit(TOOL_ERROR, ToolErrorEvent(
                call=tc, error=e, duration_seconds=duration,
            ))
            return f"Error executing {tc.name}: {e}"
        duration = time.perf_counter() - start
        await self.events.emit(TOOL_COMPLETE, ToolCompleteEvent(
            call=tc, result=result, duration_seconds=duration,
        ))
        return str(result)

    @property
    def usage(self) -> UsageStats:
        """Cumulative token usage and cost across all calls made via this client.

        Returns an empty `UsageStats` if cost tracking is not enabled in the
        decorator stack — so the property is always callable, you just see zeros.
        """
        # Walk the decorator chain looking for a CostTrackingProvider.
        provider = self._provider
        while True:
            if isinstance(provider, CostTrackingProvider):
                return provider.stats
            inner = getattr(provider, "inner_provider", None)
            if inner is None:
                return UsageStats()
            provider = inner

    @staticmethod
    def _to_messages(prompt: str | list[Message]) -> list[Message]:
        """Accept either a string (single user turn) or a full message list."""
        if isinstance(prompt, str):
            return [Message(role="user", content=prompt)]
        return prompt

    async def _apply_before_request(self, request: LLMRequest) -> LLMRequest:
        """Thread the request through each middleware's `before_request` hook."""
        for mw in self._middleware:
            result = mw.before_request(request)
            if inspect.isawaitable(result):
                result = await result
            request = result
        return request

    async def _apply_after_response(
        self, request: LLMRequest, response: LLMResponse
    ) -> LLMResponse:
        """Thread the final answer through each middleware's `after_response`
        hook. A hook may return a modified response or raise to reject."""
        for mw in self._middleware:
            result = mw.after_response(request, response)
            if inspect.isawaitable(result):
                result = await result
            response = result
        return response

    async def _tool_override(self, tc) -> str | None:
        """Return the first non-None `on_tool_call` decision, or None to run
        the tool normally."""
        for mw in self._middleware:
            result = mw.on_tool_call(tc)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None

    async def _dispatch_with_middleware(self, tc) -> str:
        """Dispatch a tool, unless middleware overrides the result first.

        An override skips execution but still emits tool.start/complete so the
        decision stays observable.
        """
        override = await self._tool_override(tc)
        if override is None:
            return await self._dispatch_with_events(tc)
        await self.events.emit(TOOL_START, ToolStartEvent(call=tc))
        await self.events.emit(TOOL_COMPLETE, ToolCompleteEvent(
            call=tc, result=override, duration_seconds=0.0,
        ))
        return override

    async def complete(
        self,
        prompt: str | list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[str] | None = None,
        max_tool_iterations: int | None = None,
    ) -> LLMResponse:
        """Full response — text, model, token counts.

        `prompt` accepts either a string (treated as a single user turn) or
        a list of `Message` objects for multi-turn conversations.

        If `tools` is provided, runs the tool-calling loop: the LLM may
        request tool invocations, which Agartha dispatches and feeds back
        as new messages, up to `max_tool_iterations` round-trips before
        returning the most recent response.

        Unspecified `temperature` reads `AGARTHA_DEFAULT_TEMPERATURE`
        (falls back to 0.7). Unspecified `max_tool_iterations` reads
        `AGARTHA_MAX_TOOL_ITERATIONS` (falls back to 10).
        """
        if temperature is None:
            temperature = get_default_temperature()
        if max_tool_iterations is None:
            max_tool_iterations = get_max_tool_iterations()
        messages = self._to_messages(prompt)

        # Fast path: no tools → single round-trip.
        if not tools:
            req = await self._apply_before_request(LLMRequest(
                messages=messages,
                model=model,
                temperature=temperature,
            ))
            resp = await self._complete_with_events(req)
            return await self._apply_after_response(req, resp)

        # Tool loop: each iteration is one LLM call. If the LLM emits
        # tool_calls, dispatch them, append the results as messages, and
        # call again. Stop when the LLM produces a response with no more
        # tool calls, or when the iteration cap is hit.
        response: LLMResponse | None = None
        request: LLMRequest | None = None
        for _ in range(max_tool_iterations + 1):
            request = await self._apply_before_request(LLMRequest(
                messages=messages,
                model=model,
                temperature=temperature,
                tools=tools,
            ))
            response = await self._complete_with_events(request)
            if not response.tool_calls:
                return await self._apply_after_response(request, response)

            messages.append(Message(
                role="assistant",
                content=response.text or None,
                tool_calls=response.tool_calls,
            ))
            # Dispatch every tool call in this turn concurrently, then append
            # results in the original order. Independent calls (notably
            # parallel sub-agent fan-out) run at the same time; the stable
            # ordering keeps the message sequence deterministic.
            contents = await asyncio.gather(*(
                self._dispatch_with_middleware(tc) for tc in response.tool_calls
            ))
            for tc, content in zip(response.tool_calls, contents, strict=True):
                messages.append(Message(
                    role="tool",
                    content=content,
                    tool_call_id=tc.id,
                ))

        # Hit the iteration cap — return the last response (likely still
        # asking for tools, but the caller said "give up after N"). The loop
        # always runs at least once, so both are set; guard explicitly
        # rather than assert so the check survives `python -O`.
        if response is None or request is None:
            raise RuntimeError("tool loop produced no response")
        return await self._apply_after_response(request, response)

    async def ask(
        self,
        question: str | list[Message],
        *,
        tools: list[str] | None = None,
    ) -> str:
        """Text-only convenience over `complete()`. Returns just the answer.

        Supports tool calling — pass `tools=["name", ...]` and Agartha runs
        the loop, returning the final assistant text after all tool calls
        are resolved.
        """
        response = await self.complete(question, tools=tools)
        return response.text

    async def stream(
        self,
        prompt: str | list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[str] | None = None,
        max_tool_iterations: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream of rich chunks — delta text + metadata.

        Fires `stream.start` ONCE at the beginning, `stream.chunk` per
        text chunk yielded to the user, and `stream.complete` when the
        whole user-facing stream ends. If `tools` is provided, the LLM
        may call tools mid-stream; those happen invisibly to the user
        (no chunks emitted for tool-call control flow). Tool dispatch
        fires the normal `tool.*` events.

        Errors propagate; `stream.error` fires with `chunk_count` set
        to however many user-facing text chunks made it out first.
        """
        if temperature is None:
            temperature = get_default_temperature()
        if max_tool_iterations is None:
            max_tool_iterations = get_max_tool_iterations()
        messages = self._to_messages(prompt)

        # `initial_request` is what the user-facing stream events reference.
        # Internal tool-loop iterations build their own LLMRequest objects.
        initial_request = LLMRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            tools=tools,
        )

        await self.events.emit(STREAM_START, StreamStartEvent(request=initial_request))
        start = time.perf_counter()
        chunk_count = 0

        try:
            if not tools:
                # Fast path — single provider session, no tool loop.
                async for chunk in self._provider.stream(initial_request):
                    chunk_count += 1
                    await self.events.emit(STREAM_CHUNK, StreamChunkEvent(
                        request=initial_request, chunk=chunk,
                    ))
                    yield chunk
            else:
                # Tool-aware loop: each iteration is one provider.stream()
                # session. Text chunks pass through to the user; chunks
                # carrying tool_calls are consumed internally (dispatched,
                # results appended to messages, next iteration starts).
                for _ in range(max_tool_iterations + 1):
                    request = LLMRequest(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        tools=tools,
                    )

                    session_tool_calls: list = []
                    session_text = ""

                    async for chunk in self._provider.stream(request):
                        if chunk.tool_calls:
                            # Control-flow chunk — accumulate, don't show user.
                            session_tool_calls.extend(chunk.tool_calls)
                            continue
                        chunk_count += 1
                        session_text += chunk.text
                        await self.events.emit(STREAM_CHUNK, StreamChunkEvent(
                            request=initial_request, chunk=chunk,
                        ))
                        yield chunk

                    if not session_tool_calls:
                        # LLM produced its final answer; user-facing stream is done.
                        break

                    # Append the assistant turn (text it streamed + tool_calls
                    # it requested), then dispatch each tool and append results.
                    messages.append(Message(
                        role="assistant",
                        content=session_text or None,
                        tool_calls=session_tool_calls,
                    ))
                    for tc in session_tool_calls:
                        content = await self._dispatch_with_events(tc)
                        messages.append(Message(
                            role="tool",
                            content=content,
                            tool_call_id=tc.id,
                        ))
        except BaseException as e:
            duration = time.perf_counter() - start
            await self.events.emit(STREAM_ERROR, StreamErrorEvent(
                request=initial_request, error=e,
                duration_seconds=duration, chunk_count=chunk_count,
            ))
            raise

        duration = time.perf_counter() - start
        await self.events.emit(STREAM_COMPLETE, StreamCompleteEvent(
            request=initial_request,
            duration_seconds=duration,
            chunk_count=chunk_count,
        ))

    async def stream_text(
        self,
        prompt: str | list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[str] | None = None,
        max_tool_iterations: int | None = None,
    ) -> AsyncIterator[str]:
        """Text-only convenience over `stream()`. Yields just text deltas.

        Supports tool calling: pass `tools=["name", ...]` and Agartha runs
        the tool loop invisibly, yielding only the LLM's final-answer text.
        """
        async for chunk in self.stream(
            prompt,
            model=model,
            temperature=temperature,
            tools=tools,
            max_tool_iterations=max_tool_iterations,
        ):
            if chunk.text:
                yield chunk.text
