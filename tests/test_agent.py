"""Agent runtime — declarative config, run loop, and sub-agent delegation."""
import pytest

from agartha import Agent, Message
from agartha.extensions.llm.fake import FakeProvider
from agartha.llm.contracts import LLMResponse, ToolCall
from agartha.registry import REGISTRY

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
