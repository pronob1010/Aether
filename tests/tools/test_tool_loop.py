"""End-to-end tool calling: LLM emits ToolCall, Aether dispatches it,
result flows back, LLM produces final answer."""
import pytest
from aether import Aether, register_tool
from aether.llm.contracts import LLMResponse, ToolCall
from aether.extensions.llm.fake import FakeProvider
from aether.registry import REGISTRY


@pytest.fixture
def cleanup_registry():
    original = {kind: dict(specs) for kind, specs in REGISTRY.items()}
    yield
    REGISTRY.clear()
    for kind, specs in original.items():
        REGISTRY[kind] = specs


# --- Helpers ------------------------------------------------------------

def _assistant_with_call(name: str, arguments: dict, call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        text="",
        model="fake-model",
        input_tokens=1,
        output_tokens=1,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


def _assistant_final(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="fake-model", input_tokens=1, output_tokens=1)


# --- Basic single-tool flow --------------------------------------------

@pytest.mark.asyncio
async def test_single_tool_call_round_trip(cleanup_registry):
    @register_tool()
    def add(a: int, b: int) -> int:
        return a + b

    fake = FakeProvider(responses=[
        _assistant_with_call("add", {"a": 2, "b": 3}),
        _assistant_final("The answer is 5."),
    ])
    client = Aether(fake)
    answer = await client.ask("What is 2+3?", tools=["add"])
    assert answer == "The answer is 5."
    # Two LLM calls: initial + after tool result
    assert len(fake.calls) == 2
    # Second call must contain assistant turn with tool_calls AND tool result
    second = fake.calls[1].messages
    assert second[-2].role == "assistant"
    assert second[-2].tool_calls[0].name == "add"
    assert second[-1].role == "tool"
    assert second[-1].content == "5"


@pytest.mark.asyncio
async def test_async_tool_function_works(cleanup_registry):
    @register_tool()
    async def fetch(url: str) -> str:
        return f"got {url}"

    fake = FakeProvider(responses=[
        _assistant_with_call("fetch", {"url": "example.com"}),
        _assistant_final("Done."),
    ])
    client = Aether(fake)
    answer = await client.ask("fetch it", tools=["fetch"])
    assert answer == "Done."
    assert fake.calls[1].messages[-1].content == "got example.com"


@pytest.mark.asyncio
async def test_tool_call_with_no_arguments(cleanup_registry):
    @register_tool()
    def heartbeat() -> str:
        return "ok"

    fake = FakeProvider(responses=[
        _assistant_with_call("heartbeat", {}),
        _assistant_final("Alive."),
    ])
    client = Aether(fake)
    answer = await client.ask("status", tools=["heartbeat"])
    assert answer == "Alive."


# --- Multi-step / iteration cap ----------------------------------------

@pytest.mark.asyncio
async def test_loop_handles_multiple_tool_round_trips(cleanup_registry):
    @register_tool()
    def step() -> str:
        return "step-done"

    fake = FakeProvider(responses=[
        _assistant_with_call("step", {}, call_id="c1"),
        _assistant_with_call("step", {}, call_id="c2"),
        _assistant_with_call("step", {}, call_id="c3"),
        _assistant_final("All done."),
    ])
    client = Aether(fake)
    answer = await client.ask("loop", tools=["step"])
    assert answer == "All done."
    assert len(fake.calls) == 4


@pytest.mark.asyncio
async def test_iteration_cap_stops_runaway_loop(cleanup_registry):
    """LLM keeps requesting tools forever; cap kicks in and we return."""
    @register_tool()
    def forever() -> str:
        return "again"

    # 100 tool-call responses (more than the cap)
    fake = FakeProvider(responses=[
        _assistant_with_call("forever", {}, call_id=f"c{i}") for i in range(100)
    ])
    client = Aether(fake)
    response = await client.complete("loop please", tools=["forever"], max_tool_iterations=3)
    # 3 iterations + 1 initial = 4 LLM calls
    assert len(fake.calls) == 4
    # Response still has tool_calls — we hit the cap, didn't resolve.
    assert len(response.tool_calls) == 1


# --- Error handling ----------------------------------------------------

@pytest.mark.asyncio
async def test_tool_error_is_reported_back_to_llm(cleanup_registry):
    @register_tool()
    def broken() -> str:
        raise ValueError("kaboom")

    fake = FakeProvider(responses=[
        _assistant_with_call("broken", {}),
        _assistant_final("I see the tool failed."),
    ])
    client = Aether(fake)
    answer = await client.ask("call broken", tools=["broken"])
    assert answer == "I see the tool failed."
    # The tool's error was passed to the LLM as the tool result, not raised.
    tool_msg = fake.calls[1].messages[-1]
    assert tool_msg.role == "tool"
    assert "kaboom" in tool_msg.content


@pytest.mark.asyncio
async def test_no_tools_passed_means_no_loop(cleanup_registry):
    """When `tools=None`, complete() makes exactly one provider call."""
    fake = FakeProvider(canned_response="straight answer")
    client = Aether(fake)
    answer = await client.ask("hi")
    assert answer == "straight answer"
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_complete_forwards_tools_to_provider(cleanup_registry):
    @register_tool()
    def t() -> str:
        return ""

    fake = FakeProvider(responses=[_assistant_final("done")])
    client = Aether(fake)
    await client.complete("go", tools=["t"])
    assert fake.calls[0].tools == ["t"]


# --- Parallel tool calls in one turn ----------------------------------

@pytest.mark.asyncio
async def test_multiple_tool_calls_in_single_turn(cleanup_registry):
    """LLM emits two tool calls at once; both get dispatched before re-prompting."""
    @register_tool()
    def add(a: int, b: int) -> int:
        return a + b

    @register_tool()
    def mul(a: int, b: int) -> int:
        return a * b

    fake = FakeProvider(responses=[
        LLMResponse(
            text="",
            model="fake-model",
            input_tokens=1,
            output_tokens=1,
            tool_calls=[
                ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}),
                ToolCall(id="c2", name="mul", arguments={"a": 4, "b": 5}),
            ],
        ),
        _assistant_final("Sum=5, product=20."),
    ])
    client = Aether(fake)
    answer = await client.ask("do both", tools=["add", "mul"])
    assert answer == "Sum=5, product=20."
    # Two tool result messages should land in the second LLM call
    second = fake.calls[1].messages
    tool_msgs = [m for m in second if m.role == "tool"]
    assert len(tool_msgs) == 2
    assert {m.content for m in tool_msgs} == {"5", "20"}
