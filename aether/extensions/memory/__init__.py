"""Built-in session store implementations.

Available:
  - `aether.extensions.memory.in_memory.InMemorySessionStore` —
    dict-backed, single-process. The framework's default.

Other stores (Redis, SQLite, etc.) plug in by implementing
`aether.memory.SessionStore`.
"""

from aether.extensions.memory.in_memory import InMemorySessionStore

__all__ = ["InMemorySessionStore"]
