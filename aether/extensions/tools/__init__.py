"""Reference tools shipped with Aether.

Each module here defines one or more `@register_tool`-decorated callables.
Importing the module triggers registration as a side effect.

Available:
  - `aether.extensions.tools.time` — `get_current_time`
  - `aether.extensions.tools.http` — `http_get`
  - `aether.extensions.tools.file` — `read_file`

Or import this package to register all three at once:

    import aether.extensions.tools  # registers time, http, file
"""

# Trigger registration for all built-in tools.
from aether.extensions.tools import time, http, file  # noqa: F401
