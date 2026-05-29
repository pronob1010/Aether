"""Session memory subsystem.

Public surface:
  - `Session`      — stateful conversation wrapper
  - `SessionStore` — Protocol for plugging in storage backends

Sessions are created via `Aether.session(id)`. The default store is the
in-memory dict-backed `aether.extensions.memory.InMemorySessionStore`;
implement `SessionStore` to back Sessions with Redis, SQLite, etc.
"""

from aether.memory.contracts import SessionStore
from aether.memory.session import Session

__all__ = [
    "Session",
    "SessionStore",
]
