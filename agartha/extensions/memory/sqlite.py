"""Durable `SessionStore` backed by SQLite.

Unlike `InMemorySessionStore`, this survives process restarts and is shared
across processes pointing at the same database file (SQLite handles the file
locking; WAL mode improves read/write concurrency). It satisfies the same
`agartha.memory.SessionStore` protocol, so it drops in via
`Agartha(memory_store=SQLiteSessionStore("sessions.db"))`.

`sqlite3` is synchronous, so each operation runs in a worker thread via
`asyncio.to_thread` to avoid blocking the event loop. Messages are stored as
a JSON-serialized list in a single row keyed by session id.
"""
import asyncio
import json
import sqlite3
from contextlib import closing

from agartha.llm.contracts import Message


class SQLiteSessionStore:
    """A `SessionStore` that persists sessions to a SQLite database file.

    Pass ":memory:" for an ephemeral database (note: a private in-memory DB is
    not shared across connections, so prefer the default file path for real
    persistence).
    """

    def __init__(self, path: str = "agartha_sessions.db"):
        self._path = str(path)
        self._initialize()

    def _initialize(self) -> None:
        with closing(sqlite3.connect(self._path)) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                "  session_id TEXT PRIMARY KEY,"
                "  messages   TEXT NOT NULL"
                ")"
            )

    # --- async SessionStore protocol ------------------------------------

    async def load(self, session_id: str) -> list[Message]:
        return await asyncio.to_thread(self._load_sync, session_id)

    async def save(self, session_id: str, messages: list[Message]) -> None:
        # Serialize on the calling thread (cheap, and snapshots the list so a
        # later mutation of the caller's list can't change what we persist).
        payload = json.dumps([m.model_dump() for m in messages])
        await asyncio.to_thread(self._save_sync, session_id, payload)

    async def delete(self, session_id: str) -> None:
        await asyncio.to_thread(self._delete_sync, session_id)

    async def exists(self, session_id: str) -> bool:
        return await asyncio.to_thread(self._exists_sync, session_id)

    # --- blocking implementations (run in a worker thread) --------------

    def _load_sync(self, session_id: str) -> list[Message]:
        with closing(sqlite3.connect(self._path)) as conn:
            row = conn.execute(
                "SELECT messages FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return []
        return [Message(**data) for data in json.loads(row[0])]

    def _save_sync(self, session_id: str, payload: str) -> None:
        with closing(sqlite3.connect(self._path)) as conn, conn:
            conn.execute(
                "INSERT INTO sessions (session_id, messages) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET messages = excluded.messages",
                (session_id, payload),
            )

    def _delete_sync(self, session_id: str) -> None:
        with closing(sqlite3.connect(self._path)) as conn, conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def _exists_sync(self, session_id: str) -> bool:
        with closing(sqlite3.connect(self._path)) as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row is not None
