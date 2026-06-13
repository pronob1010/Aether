"""Session memory subsystem — stateful conversations via Aether.session()."""
import asyncio
import pytest
from aether import Aether, Message, Session, register_tool
from aether.llm.contracts import LLMResponse, ToolCall
from aether.extensions.llm.fake import FakeProvider
from aether.extensions.memory import InMemorySessionStore
from aether.registry import REGISTRY


@pytest.fixture
def cleanup_registry():
    original = {kind: dict(specs) for kind, specs in REGISTRY.items()}
    yield
    REGISTRY.clear()
    for kind, specs in original.items():
        REGISTRY[kind] = specs


# --- Basic round-trip ---------------------------------------------------

@pytest.mark.asyncio
async def test_session_remembers_prior_turn():
    fake = FakeProvider(canned_response="ok")
    client = Aether(fake)
    session = client.session("alice")
    await session.ask("first")
    await session.ask("second")
    # Second LLM call should see both user turns + the first assistant turn
    second_call = fake.calls[1].messages
    assert [m.role for m in second_call] == ["user", "assistant", "user"]
    assert second_call[0].content == "first"
    assert second_call[2].content == "second"


@pytest.mark.asyncio
async def test_session_history_grows_after_each_turn():
    fake = FakeProvider(canned_response="reply")
    client = Aether(fake)
    session = client.session("bob")
    await session.ask("hi")
    history = await session.history()
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[1].content == "reply"


@pytest.mark.asyncio
async def test_same_session_id_returns_same_object():
    """In-process consistency: two .session('alice') calls share state."""
    client = Aether(FakeProvider())
    a = client.session("alice")
    b = client.session("alice")
    assert a is b


@pytest.mark.asyncio
async def test_different_session_ids_are_isolated():
    fake = FakeProvider(canned_response="ok")
    client = Aether(fake)
    a = client.session("alice")
    b = client.session("bob")
    await a.ask("alice's message")
    await b.ask("bob's message")
    assert "bob's message" not in [m.content for m in await a.history()]
    assert "alice's message" not in [m.content for m in await b.history()]


# --- System prompt ------------------------------------------------------

@pytest.mark.asyncio
async def test_system_prompt_prepended_on_first_turn():
    fake = FakeProvider(canned_response="ok")
    client = Aether(fake)
    session = client.session("c", system="You are terse.")
    await session.ask("hi")
    sent = fake.calls[0].messages
    assert sent[0].role == "system"
    assert sent[0].content == "You are terse."


@pytest.mark.asyncio
async def test_system_prompt_persists_across_clear():
    fake = FakeProvider(canned_response="ok")
    client = Aether(fake)
    session = client.session("d", system="be brief")
    await session.ask("hi")
    await session.clear()
    history = await session.history()
    assert len(history) == 1
    assert history[0].role == "system"


@pytest.mark.asyncio
async def test_clear_without_system_leaves_empty_history():
    client = Aether(FakeProvider())
    session = client.session("e")
    await session.ask("hi")
    await session.clear()
    assert await session.history() == []


# --- Tool calling -------------------------------------------------------

@pytest.mark.asyncio
async def test_session_supports_tool_calling(cleanup_registry):
    """Tool messages from the loop should NOT end up in session history —
    only the final assistant text is persisted."""
    @register_tool()
    def add(a: int, b: int) -> int:
        return a + b

    fake = FakeProvider(responses=[
        LLMResponse(
            text="", model="fake-model", input_tokens=1, output_tokens=1,
            tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})],
        ),
        LLMResponse(text="The answer is 5.", model="fake-model",
                    input_tokens=1, output_tokens=1),
    ])
    client = Aether(fake)
    session = client.session("calc")
    answer = await session.ask("what is 2+3?", tools=["add"])
    assert answer == "The answer is 5."

    history = await session.history()
    # NO tool turns in history — only user + final assistant.
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[1].content == "The answer is 5."


# --- Streaming ----------------------------------------------------------

@pytest.mark.asyncio
async def test_session_stream_text_persists_accumulated_response():
    fake = FakeProvider(canned_response="hello world")
    client = Aether(fake)
    session = client.session("s")
    pieces = []
    async for delta in session.stream_text("hi"):
        pieces.append(delta)
    history = await session.history()
    # The persisted assistant message is the full streamed text.
    assert history[-1].content == "hello world"


@pytest.mark.asyncio
async def test_session_stream_second_turn_sees_first():
    fake = FakeProvider(canned_response="b")
    client = Aether(fake)
    session = client.session("t")
    async for _ in session.stream_text("first"):
        pass
    async for _ in session.stream_text("second"):
        pass
    second_call = fake.calls[1].messages
    assert [m.role for m in second_call] == ["user", "assistant", "user"]


# --- Error rollback -----------------------------------------------------

class _Boom:
    async def complete(self, request):
        raise ValueError("provider down")
    async def stream(self, request):
        raise ValueError("provider down")
        yield  # noqa: unreachable


@pytest.mark.asyncio
async def test_complete_error_rolls_back_user_message():
    """If the LLM call fails, the dangling user message must NOT stick around."""
    client = Aether(_Boom())
    session = client.session("x")
    with pytest.raises(ValueError):
        await session.ask("hello")
    assert await session.history() == []


@pytest.mark.asyncio
async def test_stream_error_rolls_back_user_message():
    client = Aether(_Boom())
    session = client.session("y")
    with pytest.raises(ValueError):
        async for _ in session.stream_text("hi"):
            pass
    assert await session.history() == []


# --- Persistence via store ---------------------------------------------

@pytest.mark.asyncio
async def test_history_persists_across_client_restart():
    """A new Aether sharing the same store sees the prior session's history."""
    store = InMemorySessionStore()
    client_a = Aether(FakeProvider(canned_response="reply"), memory_store=store)
    session_a = client_a.session("z")
    await session_a.ask("first")

    # Simulate a new process — fresh client, same store
    client_b = Aether(FakeProvider(canned_response="other"), memory_store=store)
    session_b = client_b.session("z")
    history = await session_b.history()
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[0].content == "first"


@pytest.mark.asyncio
async def test_in_memory_store_round_trip():
    store = InMemorySessionStore()
    await store.save("foo", [Message(role="user", content="hi")])
    assert await store.exists("foo")
    loaded = await store.load("foo")
    assert loaded[0].content == "hi"
    await store.delete("foo")
    assert not await store.exists("foo")
    assert await store.load("foo") == []


# --- Custom store via Protocol -----------------------------------------

@pytest.mark.asyncio
async def test_custom_store_works_via_protocol():
    """Any duck-typed store works as long as it satisfies the Protocol."""
    class DictBackedStore:
        def __init__(self):
            self.data: dict[str, list[Message]] = {}
        async def load(self, sid): return list(self.data.get(sid, []))
        async def save(self, sid, msgs): self.data[sid] = list(msgs)
        async def delete(self, sid): self.data.pop(sid, None)
        async def exists(self, sid): return sid in self.data

    store = DictBackedStore()
    client = Aether(FakeProvider(canned_response="ok"), memory_store=store)
    session = client.session("custom")
    await session.ask("hi")
    assert "custom" in store.data


# --- Concurrent calls (per-session lock) --------------------------------

@pytest.mark.asyncio
async def test_concurrent_asks_serialize_via_per_session_lock():
    """Two concurrent asks on the same session must produce a consistent
    history: [user1, assistant1, user2, assistant2] or the symmetric order,
    NOT interleaved garbage."""
    fake = FakeProvider(canned_response="r")
    client = Aether(fake)
    session = client.session("concurrent")
    await asyncio.gather(
        session.ask("a"),
        session.ask("b"),
    )
    history = await session.history()
    roles = [m.role for m in history]
    assert roles == ["user", "assistant", "user", "assistant"]


# --- Manual injection --------------------------------------------------

@pytest.mark.asyncio
async def test_add_message_appends_to_history():
    fake = FakeProvider()
    client = Aether(fake)
    session = client.session("inject")
    await session.add_message(Message(role="system", content="be polite"))
    await session.ask("hi")
    sent = fake.calls[0].messages
    assert sent[0].role == "system"
    assert sent[0].content == "be polite"


# --- Session typing -----------------------------------------------------

@pytest.mark.asyncio
async def test_session_object_typed_as_Session():
    client = Aether(FakeProvider())
    session = client.session("typed")
    assert isinstance(session, Session)
