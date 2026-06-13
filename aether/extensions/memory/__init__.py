"""Built-in session store implementations.

Available:
  - `aether.extensions.memory.in_memory.InMemorySessionStore` —
    dict-backed, single-process. The framework's default.
  - `aether.extensions.memory.sqlite.SQLiteSessionStore` —
    durable, file-backed, survives restarts and shares across processes.

Other stores (Redis, etc.) plug in by implementing
`aether.memory.SessionStore`.
"""

from aether.extensions.memory.in_memory import InMemorySessionStore
from aether.extensions.memory.sqlite import SQLiteSessionStore

__all__ = ["InMemorySessionStore", "SQLiteSessionStore"]
