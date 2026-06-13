"""Concurrent vs sequential dispatch of multiple tool calls in one turn."""
import asyncio

import pytest

from agartha import Agartha, register_tool
from agartha.extensions.llm.fake import FakeProvider
from agartha.llm.contracts import LLMResponse, ToolCall
from agartha.registry import REGISTRY


def _two_call_then_done(events_unused=None):
    return FakeProvider(responses=[
        LLMResponse(
            text="", model="fake-model", input_tokens=1, output_tokens=1,
            tool_calls=[ToolCall(id="t1", name="t1", arguments={}),
                        ToolCall(id="t2", name="t2", arguments={})],
        ),
        LLMResponse(text="done", model="fake-model", input_tokens=1, output_tokens=1),
    ])


def _register_overlap_probe_tools(events: list[str]):
    """Two async tools that record enter/exit with a yield in between, so
    overlap is observable in the event order."""
    @register_tool(name="t1")
    async def t1() -> str:
        events.append("t1:enter")
        await asyncio.sleep(0.01)
        events.append("t1:exit")
        return "r1"

    @register_tool(name="t2")
    async def t2() -> str:
        events.append("t2:enter")
        await asyncio.sleep(0.01)
        events.append("t2:exit")
        return "r2"


@pytest.mark.asyncio
async def test_parallel_is_default_and_overlaps():
    events: list[str] = []
    _register_overlap_probe_tools(events)
    try:
        result = await Agartha(_two_call_then_done()).complete(
            "go", tools=["t1", "t2"],
        )
    finally:
        del REGISTRY["tool"]["t1"]
        del REGISTRY["tool"]["t2"]
    assert result.text == "done"
    # Both tools entered before either exited -> they ran concurrently.
    assert events[:2] == ["t1:enter", "t2:enter"]


@pytest.mark.asyncio
async def test_sequential_when_disabled_no_overlap():
    events: list[str] = []
    _register_overlap_probe_tools(events)
    try:
        await Agartha(_two_call_then_done()).complete(
            "go", tools=["t1", "t2"], parallel_tools=False,
        )
    finally:
        del REGISTRY["tool"]["t1"]
        del REGISTRY["tool"]["t2"]
    # Each tool fully completes before the next starts.
    assert events == ["t1:enter", "t1:exit", "t2:enter", "t2:exit"]


@pytest.mark.asyncio
async def test_env_var_can_default_to_sequential(monkeypatch):
    monkeypatch.setenv("AGARTHA_PARALLEL_TOOLS", "0")
    events: list[str] = []
    _register_overlap_probe_tools(events)
    try:
        # No parallel_tools kwarg -> reads the env default (sequential).
        await Agartha(_two_call_then_done()).complete("go", tools=["t1", "t2"])
    finally:
        del REGISTRY["tool"]["t1"]
        del REGISTRY["tool"]["t2"]
    assert events == ["t1:enter", "t1:exit", "t2:enter", "t2:exit"]


@pytest.mark.asyncio
async def test_results_appended_in_order_regardless_of_mode():
    fake = _two_call_then_done()
    events: list[str] = []
    _register_overlap_probe_tools(events)
    try:
        await Agartha(fake).complete("go", tools=["t1", "t2"], parallel_tools=True)
    finally:
        del REGISTRY["tool"]["t1"]
        del REGISTRY["tool"]["t2"]
    tool_msgs = [m.content for m in fake.calls[1].messages if m.role == "tool"]
    assert tool_msgs == ["r1", "r2"]
