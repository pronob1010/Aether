"""Reference tools shipped with Agartha.

Each module here defines one or more `@register_tool`-decorated callables.
Importing the module triggers registration as a side effect.

Available:
  - `agartha.extensions.tools.time` — `get_current_time`
  - `agartha.extensions.tools.http` — `http_get`
  - `agartha.extensions.tools.file` — `read_file`

Or import this package to register all three at once:

    import agartha.extensions.tools  # registers time, http, file
"""

# Trigger registration for all built-in tools.
from agartha.extensions.tools import time, http, file  # noqa: F401
