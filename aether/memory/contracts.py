"""Storage contract for chat sessions.

A `SessionStore` is anything that can `load`, `save`, `delete`, and check
`exists` for a session by ID. The framework ships an in-memory default
(`aether.extensions.memory.InMemorySessionStore`); users plug in Redis,
SQLite, Postgres, etc. by implementing this Protocol.
"""
from typing import Protocol, runtime_checkable
from aether.llm.contracts import Message


@runtime_checkable
class SessionStore(Protocol):
    async def load(self, session_id: str) -> list[Message]:
        """Return the persisted message list for `session_id`, or [] if absent."""
        ...

    async def save(self, session_id: str, messages: list[Message]) -> None:
        """Replace the persisted message list. Implementations should copy
        the list to defend against later in-place mutation by the caller."""
        ...

    async def delete(self, session_id: str) -> None:
        """Remove the session entirely. No-op if it doesn't exist."""
        ...

    async def exists(self, session_id: str) -> bool:
        """Return True iff a session with this ID has been saved at least once."""
        ...
