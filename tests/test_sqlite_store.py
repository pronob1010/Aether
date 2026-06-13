"""Durable SQLite-backed SessionStore."""
import pytest

from agartha import Agartha, Message
from agartha.extensions.llm.fake import FakeProvider
from agartha.extensions.memory import SQLiteSessionStore
from agartha.llm.contracts import ToolCall
from agartha.memory.contracts import SessionStore


def _store(tmp_path) -> SQLiteSessionStore:
    return SQLiteSessionStore(str(tmp_path / "sessions.db"))


def test_satisfies_sessionstore_protocol(tmp_path):
    assert isinstance(_store(tmp_path), SessionStore)


@pytest.mark.asyncio
async def test_save_and_load_round_trip(tmp_path):
    store = _store(tmp_path)
    messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]
    await store.save("alice", messages)
    loaded = await store.load("alice")
    assert loaded == messages


@pytest.mark.asyncio
async def test_load_missing_returns_empty(tmp_path):
    assert await store_load_missing(tmp_path) == []


async def store_load_missing(tmp_path):
    return await _store(tmp_path).load("nobody")


@pytest.mark.asyncio
async def test_exists_and_delete(tmp_path):
    store = _store(tmp_path)
    assert await store.exists("bob") is False
    await store.save("bob", [Message(role="user", content="hi")])
    assert await store.exists("bob") is True
    await store.delete("bob")
    assert await store.exists("bob") is False
    assert await store.load("bob") == []


@pytest.mark.asyncio
async def test_save_overwrites(tmp_path):
    store = _store(tmp_path)
    await store.save("k", [Message(role="user", content="first")])
    await store.save("k", [Message(role="user", content="second")])
    loaded = await store.load("k")
    assert len(loaded) == 1
    assert loaded[0].content == "second"


@pytest.mark.asyncio
async def test_round_trip_preserves_tool_calls(tmp_path):
    store = _store(tmp_path)
    messages = [
        Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="t1", name="add", arguments={"a": 1, "b": 2})],
        ),
        Message(role="tool", content="3", tool_call_id="t1"),
    ]
    await store.save("s", messages)
    loaded = await store.load("s")
    assert loaded == messages


@pytest.mark.asyncio
async def test_durable_across_instances(tmp_path):
    """A fresh store pointed at the same file sees previously-saved data —
    this is the property InMemorySessionStore lacks."""
    path = str(tmp_path / "shared.db")
    await SQLiteSessionStore(path).save("carol", [Message(role="user", content="persist me")])
    reopened = SQLiteSessionStore(path)
    loaded = await reopened.load("carol")
    assert loaded[0].content == "persist me"


@pytest.mark.asyncio
async def test_works_as_agartha_memory_store(tmp_path):
    """End-to-end: sessions persisted via the SQLite store survive a new
    Agartha client built on the same database file."""
    path = str(tmp_path / "agent.db")
    fake = FakeProvider(canned_response="ok")

    client1 = Agartha(fake, memory_store=SQLiteSessionStore(path))
    session1 = client1.session("dave")
    await session1.ask("remember this")

    # New client, new in-process Session cache, same database.
    client2 = Agartha(fake, memory_store=SQLiteSessionStore(path))
    session2 = client2.session("dave")
    history = await session2.history()
    assert any(m.content == "remember this" for m in history)
