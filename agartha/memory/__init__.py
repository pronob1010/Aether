"""Session memory subsystem.

Public surface:
  - `Session`      — stateful conversation wrapper
  - `SessionStore` — Protocol for plugging in storage backends

Sessions are created via `Agartha.session(id)`. The default store is the
in-memory dict-backed `agartha.extensions.memory.InMemorySessionStore`;
implement `SessionStore` to back Sessions with Redis, SQLite, etc.
"""

from agartha.memory.contracts import SessionStore
from agartha.memory.session import Session

__all__ = [
    "Session",
    "SessionStore",
]
