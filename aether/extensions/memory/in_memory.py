"""Dict-backed session store. Single-process, no persistence."""
from aether.llm.contracts import Message


class InMemorySessionStore:
    """The default `SessionStore` — keeps everything in a Python dict.

    Survives only the lifetime of the process. For multi-process / multi-host
    deployments, implement `aether.memory.SessionStore` against Redis or SQL.
    """

    def __init__(self):
        self._sessions: dict[str, list[Message]] = {}

    async def load(self, session_id: str) -> list[Message]:
        return list(self._sessions.get(session_id, []))

    async def save(self, session_id: str, messages: list[Message]) -> None:
        # Copy on save so a later mutation of the caller's list can't
        # silently change what we have persisted.
        self._sessions[session_id] = list(messages)

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def exists(self, session_id: str) -> bool:
        return session_id in self._sessions
