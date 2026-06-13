"""Response middleware — user-authored result layers + GroundingGuard."""
import pytest

from aether import Aether, GroundingGuard
from aether.extensions.llm.fake import FakeProvider
from aether.llm.contracts import LLMResponse, ToolCall

# --- The mechanism -------------------------------------------------------

class _Uppercase:
    async def process(self, request, response):
        return response.model_copy(update={"text": response.text.upper()})


class _Tagger:
    """A sync middleware — the client supports both sync and async process."""
    def process(self, request, response):
        return response.model_copy(update={"text": response.text + " [checked]"})


@pytest.mark.asyncio
async def test_middleware_transforms_final_answer():
    client = Aether(FakeProvider(canned_response="hello"),
                    response_middleware=[_Uppercase()])
    result = await client.complete("hi")
    assert result.text == "HELLO"


@pytest.mark.asyncio
async def test_sync_and_chained_middleware_run_in_order():
    client = Aether(FakeProvider(canned_response="hello"),
                    response_middleware=[_Uppercase(), _Tagger()])
    result = await client.complete("hi")
    assert result.text == "HELLO [checked]"


@pytest.mark.asyncio
async def test_middleware_applies_to_ask():
    client = Aether(FakeProvider(canned_response="hello"),
                    response_middleware=[_Uppercase()])
    assert await client.ask("hi") == "HELLO"


@pytest.mark.asyncio
async def test_no_middleware_is_passthrough():
    client = Aether(FakeProvider(canned_response="hello"))
    assert (await client.complete("hi")).text == "hello"


@pytest.mark.asyncio
async def test_middleware_can_veto_by_raising():
    class _Reject:
        async def process(self, request, response):
            raise ValueError("blocked")

    client = Aether(FakeProvider(canned_response="x"),
                    response_middleware=[_Reject()])
    with pytest.raises(ValueError, match="blocked"):
        await client.complete("hi")


@pytest.mark.asyncio
async def test_middleware_runs_only_on_final_answer_not_tool_rounds():
    """In a tool loop, middleware sees the final answer once — not the
    intermediate tool-calling response."""
    seen: list[str] = []

    class _Recorder:
        async def process(self, request, response):
            seen.append(response.text)
            return response

    fake = FakeProvider(responses=[
        LLMResponse(text="", model="fake-model", input_tokens=1, output_tokens=1,
                    tool_calls=[ToolCall(id="t1", name="now", arguments={})]),
        LLMResponse(text="final answer", model="fake-model",
                    input_tokens=1, output_tokens=1),
    ])

    from aether import register_tool
    from aether.registry import REGISTRY

    @register_tool(name="now")
    def now() -> str:
        return "noon"

    try:
        client = Aether(fake, response_middleware=[_Recorder()])
        result = await client.complete("what time?", tools=["now"])
    finally:
        del REGISTRY["tool"]["now"]

    assert result.text == "final answer"
    assert seen == ["final answer"]  # the tool-round response was not passed in


# --- GroundingGuard ------------------------------------------------------

SOURCES = [
    "The Eiffel Tower is located in Paris and was completed in 1889.",
    "It was designed by the engineer Gustave Eiffel.",
]


@pytest.mark.asyncio
async def test_grounding_guard_passes_supported_answer():
    guard = GroundingGuard(SOURCES)
    client = Aether(
        FakeProvider(canned_response="The Eiffel Tower was completed in 1889 in Paris."),
        response_middleware=[guard],
    )
    result = await client.complete("when was it completed?")
    assert "1889" in result.text  # untouched


@pytest.mark.asyncio
async def test_grounding_guard_refuses_unsupported_answer():
    guard = GroundingGuard(SOURCES)
    client = Aether(
        FakeProvider(canned_response="Mount Kilimanjaro is the tallest mountain in Africa."),
        response_middleware=[guard],
    )
    result = await client.complete("tell me something")
    assert result.text == guard.refusal


@pytest.mark.asyncio
async def test_grounding_guard_custom_refusal_and_clears_tool_calls():
    guard = GroundingGuard(SOURCES, refusal="Not in the docs.")
    response = LLMResponse(
        text="Completely unrelated invented fact about quantum llamas.",
        model="fake-model", input_tokens=1, output_tokens=1,
        tool_calls=[ToolCall(id="t", name="x", arguments={})],
    )
    out = await guard.process(request=None, response=response)
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
    assert (await guard.process(None, grounded)).text == "Paris is the city."
    assert (await guard.process(None, ungrounded)).text == guard.refusal
    assert len(calls) == 2
