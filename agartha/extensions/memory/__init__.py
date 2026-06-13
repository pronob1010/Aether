"""Built-in session store implementations.

Available:
  - `agartha.extensions.memory.in_memory.InMemorySessionStore` —
    dict-backed, single-process. The framework's default.
  - `agartha.extensions.memory.sqlite.SQLiteSessionStore` —
    durable, file-backed, survives restarts and shares across processes.

Other stores (Redis, etc.) plug in by implementing
`agartha.memory.SessionStore`.
"""

from agartha.extensions.memory.in_memory import InMemorySessionStore
from agartha.extensions.memory.sqlite import SQLiteSessionStore

__all__ = ["InMemorySessionStore", "SQLiteSessionStore"]
