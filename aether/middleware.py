"""Middleware — the pipeline the framework calls into at each decision point.

A `Middleware` plugs into the request/tool/response loop. Override only the
hooks you need; the rest are pass-through. Each hook may act on its input and
return a (possibly changed) value, so middleware *shapes* control flow rather
than merely observing it (that's the EventBus's job):

    before_request(request)            -> request     # inject context, redact, rewrite
    on_tool_call(call)                 -> str | None   # approve / deny / override a tool
    after_response(request, response)  -> response     # gate, annotate, transform the answer

Hooks may be sync or async; the client awaits a hook's result only when it is
awaitable. Middleware run in order; `before_request` and `after_response`
thread the value through the chain, while the first non-None `on_tool_call`
decision wins.

This is the seam the Agent runtime, tool-approval policies, sub-agent routing,
and automatic context management all build on — they're middleware on this one
pipeline. (The retry / circuit-breaker / cost-tracking layers are the
*provider-level* analog: decorators around the raw call, where per-attempt
semantics belong.)

Configure via `Aether(middleware=[...])`.
"""
import inspect
import re
from collections.abc import Awaitable, Callable

from aether.llm.contracts import LLMRequest, LLMResponse, ToolCall


class Middleware:
    """Base class for pipeline middleware. Subclass and override the hooks you
    need — unoverridden hooks are pass-through, so a result-only guard need
    only define `after_response`."""

    # Hooks may be overridden sync or async — hence the `| Awaitable[...]`
    # return types. The client awaits a hook's result only when it's awaitable.

    def before_request(
        self, request: LLMRequest
    ) -> LLMRequest | Awaitable[LLMRequest]:
        """Called before every provider request (each tool-loop iteration).

        Return the request unchanged or a modified copy — e.g. inject a system
        message, redact content, or add retrieved context.
        """
        return request

    def on_tool_call(self, call: ToolCall) -> str | None | Awaitable[str | None]:
        """Called before each tool is dispatched.

        Return None to let the tool run normally, or a string to *override* the
        result without running it — denying a tool ("Blocked by policy."),
        gating it behind your own approval, or supplying a canned result. The
        first non-None decision across the chain wins.
        """
        return None

    def after_response(
        self, request: LLMRequest, response: LLMResponse
    ) -> LLMResponse | Awaitable[LLMResponse]:
        """Called on the final answer of complete()/ask() (not on the
        intermediate tool-calling rounds). Return the response unchanged, a
        modified copy, or raise to reject."""
        return response


# Verifier signature for GroundingGuard: given the answer text and the source
# documents, return (optionally via awaitable) whether the answer is grounded.
Verifier = Callable[[str, list[str]], bool | Awaitable[bool]]


DEFAULT_REFUSAL = (
    "I can only answer using the provided documents, and I couldn't find "
    "support for that there."
)

# Tiny stopword set so the default overlap heuristic isn't trivially satisfied
# by filler words that appear in any text.
_STOPWORDS = frozenset(
    "a an the and or but if then is are was were be been being of to in on for "
    "with as by at from this that these those it its i you he she they we".split()
)


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS
    }


def _overlap_grounded(answer: str, sources: list[str], threshold: float) -> bool:
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return True  # nothing substantive to ground
    source_tokens: set[str] = set()
    for s in sources:
        source_tokens |= _tokens(s)
    overlap = answer_tokens & source_tokens
    return len(overlap) / len(answer_tokens) >= threshold


class GroundingGuard(Middleware):
    """Reject answers not supported by a known set of source documents.

    For RAG: pass the retrieved chunks as `sources`. If the model's answer
    isn't grounded in them, the answer text is replaced with `refusal` (tool
    calls are cleared too). Implemented as the `after_response` hook of the
    middleware pipeline.

    The default check is a transparent token-overlap heuristic — adequate for
    catching blatant fabrication, but for production pass your own `verifier`
    (e.g. an LLM judge); it may be sync or async:

        async def judge(answer, sources): ...    # call your model here
        Aether(middleware=[GroundingGuard(chunks, verifier=judge)])
    """

    def __init__(
        self,
        sources: list[str],
        *,
        verifier: Verifier | None = None,
        refusal: str = DEFAULT_REFUSAL,
        threshold: float = 0.5,
    ):
        self.sources = list(sources)
        self.verifier = verifier
        self.refusal = refusal
        self.threshold = threshold

    async def after_response(
        self, request: LLMRequest, response: LLMResponse
    ) -> LLMResponse:
        text = response.text.strip()
        if not text:
            return response

        if self.verifier is not None:
            verdict = self.verifier(text, self.sources)
            if inspect.isawaitable(verdict):
                verdict = await verdict
            grounded = bool(verdict)
        else:
            grounded = _overlap_grounded(text, self.sources, self.threshold)

        if grounded:
            return response
        return response.model_copy(update={"text": self.refusal, "tool_calls": []})
