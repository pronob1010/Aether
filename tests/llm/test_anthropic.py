"""Anthropic (Claude) provider adapter.

complete()/stream() are exercised against a fake injected client so no SDK or
network is required; the pure translation helpers are tested directly.
"""
from types import SimpleNamespace

import pytest

from agartha import register_tool
from agartha.extensions.llm.anthropic import (
    AnthropicProvider,
    _split_system,
    _to_anthropic_messages,
    _tools_payload,
)
from agartha.extensions.llm.registry import LLM_PROVIDER_KIND
from agartha.llm.contracts import LLMRequest, Message, ToolCall
from agartha.registry import REGISTRY, list_kind
from agartha.tools.registry import dispatch_tool  # noqa: F401 (registry side effects)

# --- Registration --------------------------------------------------------

def test_anthropic_provider_is_registered():
    assert "anthropic" in list_kind(LLM_PROVIDER_KIND)


def test_anthropic_metadata_points_at_env_vars():
    meta = REGISTRY[LLM_PROVIDER_KIND]["anthropic"].metadata
    assert meta["api_key_env"] == "ANTHROPIC_API_KEY"
    assert meta["model_env"] == "ANTHROPIC_MODEL"


# --- Message translation -------------------------------------------------

def test_split_system_lifts_system_turns():
    system, conversation = _split_system([
        Message(role="system", content="be terse"),
        Message(role="user", content="hi"),
    ])
    assert system == "be terse"
    assert [m.role for m in conversation] == ["user"]


def test_to_anthropic_messages_plain_turns():
    out = _to_anthropic_messages([
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ])
    assert out == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_to_anthropic_messages_tool_use_and_result():
    out = _to_anthropic_messages([
        Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="t1", name="add", arguments={"a": 1})],
        ),
        Message(role="tool", content="3", tool_call_id="t1"),
    ])
    assert out[0]["role"] == "assistant"
    assert out[0]["content"][0] == {
        "type": "tool_use", "id": "t1", "name": "add", "input": {"a": 1},
    }
    assert out[1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "3"}],
    }


@pytest.fixture
def cleanup_registry():
    original = {kind: dict(specs) for kind, specs in REGISTRY.items()}
    yield
    REGISTRY.clear()
    for kind, specs in original.items():
        REGISTRY[kind] = specs


def test_tools_payload_uses_input_schema(cleanup_registry):
    @register_tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    payload = _tools_payload(["add"])
    assert payload[0]["name"] == "add"
    assert payload[0]["description"] == "Add two numbers"
    assert set(payload[0]["input_schema"]["properties"]) == {"a", "b"}


def test_tools_payload_none_when_empty():
    assert _tools_payload(None) is None
    assert _tools_payload([]) is None


# --- complete() against a fake client ------------------------------------

class _FakeMessages:
    def __init__(self, response=None, events=None):
        self._response = response
        self._events = events or []
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        events = self._events

        class _Ctx:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *exc):
                return False

            async def __aiter__(self_inner):
                for ev in events:
                    yield ev

        return _Ctx()


class _FakeClient:
    def __init__(self, response=None, events=None):
        self.messages = _FakeMessages(response=response, events=events)


@pytest.mark.asyncio
async def test_complete_returns_text_and_usage():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Paris.")],
        model="claude-opus-4-8",
        usage=SimpleNamespace(input_tokens=12, output_tokens=3),
    )
    provider = AnthropicProvider(client=_FakeClient(response=response))
    result = await provider.complete(LLMRequest(
        messages=[Message(role="user", content="capital of France?")],
    ))
    assert result.text == "Paris."
    assert result.model == "claude-opus-4-8"
    assert result.input_tokens == 12
    assert result.output_tokens == 3
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_complete_parses_tool_use_block():
    response = SimpleNamespace(
        content=[SimpleNamespace(
            type="tool_use", id="t1", name="add", input={"a": 1, "b": 2},
        )],
        model="claude-opus-4-8",
        usage=SimpleNamespace(input_tokens=20, output_tokens=10),
    )
    provider = AnthropicProvider(client=_FakeClient(response=response))
    result = await provider.complete(LLMRequest(
        messages=[Message(role="user", content="add 1 and 2")],
        tools=None,
    ))
    assert result.text == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "add"
    assert result.tool_calls[0].arguments == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_complete_requires_max_tokens_and_omits_temperature():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        model="claude-opus-4-8",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    client = _FakeClient(response=response)
    provider = AnthropicProvider(client=client, default_max_tokens=512)
    await provider.complete(LLMRequest(
        messages=[Message(role="system", content="sys"),
                  Message(role="user", content="hi")],
        temperature=0.9,
    ))
    kwargs = client.messages.last_kwargs
    assert kwargs["max_tokens"] == 512        # required param is set
    assert kwargs["system"] == "sys"          # system lifted to top level
    assert "temperature" not in kwargs        # never forwarded (400 on Opus 4.8)


# --- stream() against a fake client --------------------------------------

@pytest.mark.asyncio
async def test_stream_text_then_usage():
    events = [
        SimpleNamespace(type="message_start",
                        message=SimpleNamespace(usage=SimpleNamespace(input_tokens=5))),
        SimpleNamespace(type="content_block_delta", index=0,
                        delta=SimpleNamespace(type="text_delta", text="Hel")),
        SimpleNamespace(type="content_block_delta", index=0,
                        delta=SimpleNamespace(type="text_delta", text="lo")),
        SimpleNamespace(type="message_delta",
                        usage=SimpleNamespace(output_tokens=2)),
    ]
    provider = AnthropicProvider(client=_FakeClient(events=events))
    chunks = [c async for c in provider.stream(LLMRequest(
        messages=[Message(role="user", content="hi")],
    ))]
    text = "".join(c.text for c in chunks)
    assert text == "Hello"
    final = chunks[-1]
    assert final.input_tokens == 5
    assert final.output_tokens == 2
    assert final.tool_calls == []


@pytest.mark.asyncio
async def test_stream_accumulates_tool_use_input_json():
    events = [
        SimpleNamespace(type="message_start",
                        message=SimpleNamespace(usage=SimpleNamespace(input_tokens=8))),
        SimpleNamespace(type="content_block_start", index=0,
                        content_block=SimpleNamespace(type="tool_use", id="t1", name="add")),
        SimpleNamespace(type="content_block_delta", index=0,
                        delta=SimpleNamespace(type="input_json_delta", partial_json='{"a": 1,')),
        SimpleNamespace(type="content_block_delta", index=0,
                        delta=SimpleNamespace(type="input_json_delta", partial_json=' "b": 2}')),
        SimpleNamespace(type="message_delta",
                        usage=SimpleNamespace(output_tokens=4)),
    ]
    provider = AnthropicProvider(client=_FakeClient(events=events))
    chunks = [c async for c in provider.stream(LLMRequest(
        messages=[Message(role="user", content="add 1 and 2")],
    ))]
    final = chunks[-1]
    assert final.finish_reason == "tool_calls"
    assert len(final.tool_calls) == 1
    assert final.tool_calls[0].name == "add"
    assert final.tool_calls[0].arguments == {"a": 1, "b": 2}
