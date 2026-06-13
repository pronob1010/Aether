"""Agent runtime — declarative config, run loop, and sub-agent delegation."""
import asyncio

import pytest

from agartha import Agent, Message
from agartha.extensions.llm.fake import FakeProvider
from agartha.llm.contracts import LLMResponse, ToolCall
from agartha.registry import REGISTRY


def _tool_call(name, **args):
    return ToolCall(id=f"t-{name}", name=name, arguments=args)


def _turn(*tool_calls, text=""):
    return LLMResponse(text=text, model="fake-model", input_tokens=1,
                       output_tokens=1, tool_calls=list(tool_calls))

# --- Basic run -----------------------------------------------------------

@pytest.mark.asyncio
async def test_run_prepends_instructions_as_system_turn():
    fake = FakeProvider(canned_response="hi there")
    agent = Agent("greeter", instructions="Be friendly.", provider=fake)
    await agent.run("hello")
    sent = fake.calls[0].messages
    assert sent[0].role == "system" and sent[0].content == "Be friendly."
    assert sent[1].role == "user" and sent[1].content == "hello"


@pytest.mark.asyncio
async def test_run_without_instructions_has_no_system_turn():
    fake = FakeProvider(canned_response="ok")
    agent = Agent("bare", provider=fake)
    await agent.run("hello")
    assert [m.role for m in fake.calls[0].messages] == ["user"]


@pytest.mark.asyncio
async def test_run_text_returns_answer():
    agent = Agent("a", provider=FakeProvider(canned_response="the answer"))
    assert await agent.run_text("q") == "the answer"


@pytest.mark.asyncio
async def test_run_accepts_message_list():
    fake = FakeProvider(canned_response="ok")
    agent = Agent("a", instructions="sys", provider=fake)
    await agent.run([Message(role="user", content="one"),
                     Message(role="assistant", content="two"),
                     Message(role="user", content="three")])
    roles = [m.role for m in fake.calls[0].messages]
    assert roles == ["system", "user", "assistant", "user"]


def test_client_and_provider_are_mutually_exclusive():
    from agartha import Agartha
    with pytest.raises(ValueError, match="not both"):
        Agent("x", client=Agartha(FakeProvider()), provider=FakeProvider())


# --- Tools ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_runs_the_tool_loop():
    ran: list[str] = []

    from agartha import register_tool

    @register_tool(name="ping")
    def ping() -> str:
        ran.append("yes")
        return "pong"

    fake = FakeProvider(responses=[
        LLMResponse(text="", model="fake-model", input_tokens=1, output_tokens=1,
                    tool_calls=[ToolCall(id="t1", name="ping", arguments={})]),
        LLMResponse(text="done", model="fake-model", input_tokens=1, output_tokens=1),
    ])
    try:
        agent = Agent("a", tools=["ping"], provider=fake)
        result = await agent.run_text("go")
    finally:
        del REGISTRY["tool"]["ping"]

    assert ran == ["yes"]
    assert result == "done"


# --- Sub-agent delegation ------------------------------------------------

@pytest.mark.asyncio
async def test_as_tool_registers_and_delegates_to_subagent():
    child_fake = FakeProvider(canned_response="child answer")
    child = Agent("researcher", instructions="research things", provider=child_fake)

    parent_fake = FakeProvider(responses=[
        LLMResponse(
            text="", model="fake-model", input_tokens=1, output_tokens=1,
            tool_calls=[ToolCall(id="t1", name="researcher",
                                 arguments={"task": "find X"})],
        ),
        LLMResponse(text="parent done", model="fake-model",
                    input_tokens=1, output_tokens=1),
    ])
    parent = Agent("coordinator", provider=parent_fake)

    tool_name = child.as_tool()
    assert tool_name == "researcher"
    parent.tools = [tool_name]

    try:
        result = await parent.run_text("delegate the research")
    finally:
        del REGISTRY["tool"]["researcher"]

    # Parent produced the final answer.
    assert result == "parent done"
    # The child agent actually ran, receiving the delegated task as its prompt.
    assert len(child_fake.calls) == 1
    assert child_fake.calls[0].messages[-1].content == "find X"
    # The child's answer was fed back to the parent as the tool result.
    tool_msgs = [m for m in parent_fake.calls[1].messages if m.role == "tool"]
    assert tool_msgs[0].content == "child answer"


def test_as_tool_custom_name_and_description():
    agent = Agent("worker", provider=FakeProvider())
    name = agent.as_tool(name="do_work", description="Does the work.")
    try:
        from agartha import get_tool
        spec = get_tool("do_work")
        assert name == "do_work"
        assert spec.schema["description"] == "Does the work."
        # The delegated tool exposes a single `task` string argument.
        assert set(spec.schema["parameters"]["properties"]) == {"task"}
    finally:
        del REGISTRY["tool"]["do_work"]


# --- Delegation depth guard ----------------------------------------------

@pytest.mark.asyncio
async def test_delegation_depth_guard_stops_recursion():
    """A -> B -> C with max_depth=1: A may delegate to B, but B's delegation
    to C is refused, so C never runs."""
    c_fake = FakeProvider(canned_response="C answer")
    c = Agent("C", provider=c_fake)

    b_fake = FakeProvider(responses=[
        _turn(_tool_call("C", task="sub")),   # B tries to delegate deeper
        _turn(text="B answer"),
    ])
    b = Agent("B", provider=b_fake, tools=[c.as_tool(max_depth=1)])

    a_fake = FakeProvider(responses=[
        _turn(_tool_call("B", task="work")),  # A delegates to B (allowed)
        _turn(text="A answer"),
    ])
    a = Agent("A", provider=a_fake, tools=[b.as_tool(max_depth=1)])

    try:
        result = await a.run_text("go")
    finally:
        del REGISTRY["tool"]["B"]
        del REGISTRY["tool"]["C"]

    assert result == "A answer"
    assert len(c_fake.calls) == 0   # C was never reached — depth limit held
    # B saw the refusal as the tool result for its delegation attempt.
    b_tool_results = [m.content for m in b_fake.calls[1].messages if m.role == "tool"]
    assert "depth limit" in b_tool_results[0]


@pytest.mark.asyncio
async def test_delegation_depth_allows_within_limit():
    """With the default limit, a single level of delegation works."""
    child_fake = FakeProvider(canned_response="child answer")
    child = Agent("worker", provider=child_fake)
    parent = Agent("boss", provider=FakeProvider(responses=[
        _turn(_tool_call("worker", task="do it")),
        _turn(text="boss answer"),
    ]))
    parent.tools = [child.as_tool()]
    try:
        result = await parent.run_text("delegate")
    finally:
        del REGISTRY["tool"]["worker"]
    assert result == "boss answer"
    assert len(child_fake.calls) == 1


# --- Parallel fan-out ----------------------------------------------------

class _Gate:
    """A tiny barrier: every arrival blocks until `n` have arrived. If the
    callers were dispatched sequentially, the first arrival blocks forever."""
    def __init__(self, n: int):
        self.n = n
        self.count = 0
        self.event = asyncio.Event()

    async def arrive_and_wait(self) -> None:
        self.count += 1
        if self.count >= self.n:
            self.event.set()
        await self.event.wait()


class _ConcurrentProvider:
    def __init__(self, text: str, gate: _Gate):
        self.text = text
        self.gate = gate
        self.calls: list = []

    async def complete(self, request):
        self.calls.append(request)
        await self.gate.arrive_and_wait()
        return LLMResponse(text=self.text, model="fake-model",
                           input_tokens=1, output_tokens=1)

    async def stream(self, request):  # pragma: no cover - unused
        raise NotImplementedError
        yield


@pytest.mark.asyncio
async def test_parallel_subagent_fan_out_runs_concurrently():
    """A coordinator that emits two delegation calls in one turn runs them
    concurrently — the shared gate only releases once both have started, so a
    sequential dispatch would deadlock (caught by the timeout)."""
    gate = _Gate(2)
    cp1 = _ConcurrentProvider("c1 done", gate)
    cp2 = _ConcurrentProvider("c2 done", gate)
    c1 = Agent("c1", provider=cp1)
    c2 = Agent("c2", provider=cp2)

    parent_fake = FakeProvider(responses=[
        _turn(_tool_call("c1", task="a"), _tool_call("c2", task="b")),
        _turn(text="parent done"),
    ])
    parent = Agent("coordinator", provider=parent_fake)
    parent.tools = [c1.as_tool(), c2.as_tool()]

    try:
        result = await asyncio.wait_for(parent.run_text("fan out"), timeout=3.0)
    finally:
        del REGISTRY["tool"]["c1"]
        del REGISTRY["tool"]["c2"]

    assert result == "parent done"
    assert len(cp1.calls) == 1 and len(cp2.calls) == 1
    # Results appended in the original tool-call order despite concurrency.
    tool_msgs = [m.content for m in parent_fake.calls[1].messages if m.role == "tool"]
    assert tool_msgs == ["c1 done", "c2 done"]
