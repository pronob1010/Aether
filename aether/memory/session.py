"""A `Session` is a stateful conversation with the LLM.

Wraps an `Aether` client and a `SessionStore`. Each `ask` / `complete` /
`stream*` call appends to the persisted message history and the next call
sees the full prior conversation — no need for the caller to thread
`messages` through their code.

Correctness rules baked in:
  1. Each turn passes a COPY of the history to `Aether.complete()`,
     because the underlying tool loop mutates its messages list and we
     don't want internal tool dispatches polluting the session history.
  2. The user message is appended BEFORE the LLM call. If the LLM call
     fails, the user message is rolled back so the next turn doesn't
     replay a half-finished exchange.
  3. A per-session `asyncio.Lock` serializes concurrent calls on the
     same session — last-write-wins clobbering would otherwise corrupt
     the history.
"""
import asyncio
from typing import TYPE_CHECKING, AsyncIterator, Any
from aether.llm.contracts import LLMResponse, LLMStreamChunk, Message
from aether.memory.contracts import SessionStore

if TYPE_CHECKING:
    from aether.client import Aether


class Session:
    """One stateful conversation. Created via `Aether.session(id)`."""

    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        client: "Aether",
        system: str | None = None,
    ):
        self.id = session_id
        self._store = store
        self._client = client
        self._system = system
        self._messages: list[Message] = []
        self._loaded = False
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        existing = await self._store.load(self.id)
        if existing:
            self._messages = list(existing)
        elif self._system:
            self._messages = [Message(role="system", content=self._system)]
        self._loaded = True

    # --- Inspection -------------------------------------------------------

    async def history(self) -> list[Message]:
        """A copy of the current persisted conversation."""
        await self._ensure_loaded()
        return list(self._messages)

    async def clear(self) -> None:
        """Wipe the conversation. Keeps the system prompt if one was set."""
        async with self._lock:
            self._messages = (
                [Message(role="system", content=self._system)]
                if self._system else []
            )
            self._loaded = True
            await self._store.save(self.id, self._messages)

    # --- Manual injection (useful for tool transcripts, edits, etc.) -----

    async def add_message(self, message: Message) -> None:
        """Append a message directly. Saves to the store."""
        async with self._lock:
            await self._ensure_loaded()
            self._messages.append(message)
            await self._store.save(self.id, self._messages)

    # --- Conversation methods --------------------------------------------

    async def complete(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a user message, get the full response, persist both turns."""
        async with self._lock:
            await self._ensure_loaded()
            user_msg = Message(role="user", content=prompt)
            self._messages.append(user_msg)
            try:
                # Copy so Aether's internal tool loop doesn't mutate our list.
                response = await self._client.complete(
                    list(self._messages), **kwargs,
                )
            except BaseException:
                # Roll back the user message — the turn never completed.
                self._messages.pop()
                raise
            self._messages.append(Message(role="assistant", content=response.text))
            await self._store.save(self.id, self._messages)
            return response

    async def ask(self, question: str, **kwargs: Any) -> str:
        """Text-only convenience over `complete()`."""
        response = await self.complete(question, **kwargs)
        return response.text

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream rich chunks. Persists the accumulated assistant text at the end."""
        async with self._lock:
            await self._ensure_loaded()
            user_msg = Message(role="user", content=prompt)
            self._messages.append(user_msg)
            accumulated = ""
            try:
                async for chunk in self._client.stream(
                    list(self._messages), **kwargs,
                ):
                    accumulated += chunk.text
                    yield chunk
            except BaseException:
                # Roll back — partial assistant text is not persisted.
                self._messages.pop()
                raise
            self._messages.append(Message(role="assistant", content=accumulated))
            await self._store.save(self.id, self._messages)

    async def stream_text(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Text-only streaming convenience."""
        async for chunk in self.stream(prompt, **kwargs):
            if chunk.text:
                yield chunk.text
