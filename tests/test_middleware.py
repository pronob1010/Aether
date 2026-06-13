"""Middleware pipeline — before_request / on_tool_call / after_response."""
import pytest

from agartha import Agartha, GroundingGuard, Message, Middleware
from agartha.extensions.llm.fake import FakeProvider
from agartha.llm.contracts import LLMResponse, ToolCall
from agartha.registry import REGISTRY

# --- after_response ------------------------------------------------------


class _Uppercase(Middleware):
    async def after_response(self, request, response):
        return response.model_copy(update={"text": response.text.upper()})


class _Tagger(Middleware):
    """A sync hook — the client awaits a hook result only when it's awaitable."""
    def after_response(self, request, response):
        return response.model_copy(update={"text": response.text + " [checked]"})


@pytest.mark.asyncio
async def test_after_response_transforms_final_answer():
    client = Agartha(FakeProvider(canned_response="hello"), middleware=[_Uppercase()])
    assert (await client.complete("hi")).text == "HELLO"


@pytest.mark.asyncio
async def test_sync_and_chained_hooks_run_in_order():
    client = Agartha(FakeProvider(canned_response="hello"),
                    middleware=[_Uppercase(), _Tagger()])
    assert (await client.complete("hi")).text == "HELLO [checked]"


@pytest.mark.asyncio
async def test_after_response_applies_to_ask():
    client = Agartha(FakeProvider(canned_response="hello"), middleware=[_Uppercase()])
    assert await client.ask("hi") == "HELLO"


@pytest.mark.asyncio
async def test_no_middleware_is_passthrough():
    client = Agartha(FakeProvider(canned_response="hello"))
    assert (await client.complete("hi")).text == "hello"


@pytest.mark.asyncio
async def test_after_response_can_veto_by_raising():
    class _Reject(Middleware):
        async def after_response(self, request, response):
            raise ValueError("blocked")

    client = Agartha(FakeProvider(canned_response="x"), middleware=[_Reject()])
    with pytest.raises(ValueError, match="blocked"):
        await client.complete("hi")


# --- before_request ------------------------------------------------------


@pytest.mark.asyncio
async def test_before_request_can_rewrite_the_request():
    class _InjectContext(Middleware):
        def before_request(self, request):
            msgs = [Message(role="system", content="injected"), *request.messages]
            return request.model_copy(update={"messages": msgs})

    fake = FakeProvider(canned_response="ok")
    client = Agartha(fake, middleware=[_InjectContext()])
    await client.complete("hi")
    sent = fake.calls[0].messages
    assert sent[0].role == "system" and sent[0].content == "injected"


# --- on_tool_call --------------------------------------------------------


@pytest.mark.asyncio
async def test_on_tool_call_can_deny_a_tool_without_running_it():
    ran: list[str] = []

    class _DenyDelete(Middleware):
        def on_tool_call(self, call):
            if call.name == "delete_everything":
                return "Blocked by policy."
            return None

    @register_helper("delete_everything")
    def delete_everything() -> str:
        ran.append("yes")
        return "deleted"

    fake = FakeProvider(responses=[
        LLMResponse(text="", model="fake-model", input_tokens=1, output_tokens=1,
                    tool_calls=[ToolCall(id="t1", name="delete_everything", arguments={})]),
        LLMResponse(text="done", model="fake-model", input_tokens=1, output_tokens=1),
    ])
    try:
        client = Agartha(fake, middleware=[_DenyDelete()])
        result = await client.complete("clean up", tools=["delete_everything"])
    finally:
        del REGISTRY["tool"]["delete_everything"]

    assert ran == []                     # the tool never executed
    assert result.text == "done"
    # The override string was fed back as the tool result.
    tool_msgs = [m for m in fake.calls[1].messages if m.role == "tool"]
    assert tool_msgs[0].content == "Blocked by policy."


@pytest.mark.asyncio
async def test_on_tool_call_none_lets_the_tool_run():
    ran: list[str] = []

    @register_helper("noop")
    def noop() -> str:
        ran.append("yes")
        return "real result"

    fake = FakeProvider(responses=[
        LLMResponse(text="", model="fake-model", input_tokens=1, output_tokens=1,
                    tool_calls=[ToolCall(id="t1", name="noop", arguments={})]),
        LLMResponse(text="ok", model="fake-model", input_tokens=1, output_tokens=1),
    ])
    try:
        client = Agartha(fake, middleware=[Middleware()])  # base = all pass-through
        await client.complete("go", tools=["noop"])
    finally:
        del REGISTRY["tool"]["noop"]

    assert ran == ["yes"]
    tool_msgs = [m for m in fake.calls[1].messages if m.role == "tool"]
    assert tool_msgs[0].content == "real result"


# Helper so the tool-registering tests read cleanly.
def register_helper(name):
    from agartha import register_tool
    return register_tool(name=name)


# --- after_response only runs on the final answer ------------------------


@pytest.mark.asyncio
async def test_after_response_runs_only_on_final_answer_not_tool_rounds():
    seen: list[str] = []

    class _Recorder(Middleware):
        async def after_response(self, request, response):
            seen.append(response.text)
            return response

    @register_helper("now")
    def now() -> str:
        return "noon"

    fake = FakeProvider(responses=[
        LLMResponse(text="", model="fake-model", input_tokens=1, output_tokens=1,
                    tool_calls=[ToolCall(id="t1", name="now", arguments={})]),
        LLMResponse(text="final answer", model="fake-model",
                    input_tokens=1, output_tokens=1),
    ])
    try:
        client = Agartha(fake, middleware=[_Recorder()])
        result = await client.complete("what time?", tools=["now"])
    finally:
        del REGISTRY["tool"]["now"]

    assert result.text == "final answer"
    assert seen == ["final answer"]


# --- GroundingGuard (an after_response middleware) -----------------------

SOURCES = [
    "The Eiffel Tower is located in Paris and was completed in 1889.",
    "It was designed by the engineer Gustave Eiffel.",
]


@pytest.mark.asyncio
async def test_grounding_guard_passes_supported_answer():
    client = Agartha(
        FakeProvider(canned_response="The Eiffel Tower was completed in 1889 in Paris."),
        middleware=[GroundingGuard(SOURCES)],
    )
    assert "1889" in (await client.complete("when?")).text


@pytest.mark.asyncio
async def test_grounding_guard_refuses_unsupported_answer():
    guard = GroundingGuard(SOURCES)
    client = Agartha(
        FakeProvider(canned_response="Mount Kilimanjaro is the tallest mountain in Africa."),
        middleware=[guard],
    )
    assert (await client.complete("tell me something")).text == guard.refusal


@pytest.mark.asyncio
async def test_grounding_guard_custom_refusal_and_clears_tool_calls():
    guard = GroundingGuard(SOURCES, refusal="Not in the docs.")
    response = LLMResponse(
        text="Completely unrelated invented fact about quantum llamas.",
        model="fake-model", input_tokens=1, output_tokens=1,
        tool_calls=[ToolCall(id="t", name="x", arguments={})],
    )
    out = await guard.after_response(request=None, response=response)
    assert out.text == "Not in the docs."
    assert out.tool_calls == []


@pytest.mark.asyncio
async def test_grounding_guard_accepts_custom_async_verifier():
    calls: list[str] = []

    async def judge(answer: str, sources: list[str]) -> bool:
        calls.append(answer)
        return "Paris" in answer  # pretend LLM judge

    guard = GroundingGuard(SOURCES, verifier=judge)
    grounded = LLMResponse(text="Paris is the city.", model="m",
                           input_tokens=1, output_tokens=1)
    ungrounded = LLMResponse(text="Tokyo is the city.", model="m",
                             input_tokens=1, output_tokens=1)
    assert (await guard.after_response(None, grounded)).text == "Paris is the city."
    assert (await guard.after_response(None, ungrounded)).text == guard.refusal
    assert len(calls) == 2
