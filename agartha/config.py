"""Runtime configuration sourced from environment variables.

Each accessor reads its env var on every call (not at import time) so that
tests can `monkeypatch.setenv` and see the new value without re-importing.
Invalid values (unparseable numbers, etc.) fall back to the documented
default — better to silently degrade than to crash a long-running agent
because someone fat-fingered a number in an env var.
"""
import os


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Tool loop ----------------------------------------------------------

def get_max_tool_iterations() -> int:
    """Default cap on the tool-dispatch loop in `Agartha.complete()`.

    Override the per-process default by setting `AGARTHA_MAX_TOOL_ITERATIONS`.
    Override per-call by passing `max_tool_iterations=N` to `complete()`.
    Falls back to 10 if the env var is unset or unparseable.
    """
    return _int_env("AGARTHA_MAX_TOOL_ITERATIONS", 10)


def get_max_delegation_depth() -> int:
    """How many nested levels of sub-agent delegation are allowed.

    A top-level run is depth 0; each `Agent.as_tool()` delegation goes one
    level deeper. When the limit is reached, further delegation is refused with
    an error string (the model sees it) rather than recursing without bound.

    Override via `AGARTHA_MAX_DELEGATION_DEPTH`, or per-tool with
    `Agent.as_tool(max_depth=N)`. Falls back to 3.
    """
    return _int_env("AGARTHA_MAX_DELEGATION_DEPTH", 3)


# --- LLM request defaults ----------------------------------------------

def get_default_temperature() -> float:
    """Default sampling temperature used when callers don't pass one.

    Override via `AGARTHA_DEFAULT_TEMPERATURE`. Per-call kwarg wins.
    Falls back to 0.7 (matches the long-standing OpenAI default).
    """
    return _float_env("AGARTHA_DEFAULT_TEMPERATURE", 0.7)


# --- Reference tool limits ---------------------------------------------

def get_http_tool_timeout() -> float:
    """Default timeout (seconds) for the built-in `http_get` tool.

    Override via `AGARTHA_HTTP_TOOL_TIMEOUT`. Falls back to 10.0.
    """
    return _float_env("AGARTHA_HTTP_TOOL_TIMEOUT", 10.0)


def get_http_tool_max_bytes() -> int:
    """Max bytes the built-in `http_get` returns before truncating.

    Override via `AGARTHA_HTTP_TOOL_MAX_BYTES`. Falls back to 100_000
    (large enough for most responses, small enough not to flood the
    LLM's context window).
    """
    return _int_env("AGARTHA_HTTP_TOOL_MAX_BYTES", 100_000)


def get_file_tool_max_bytes() -> int:
    """Max bytes the built-in `read_file` returns before truncating.

    Override via `AGARTHA_FILE_TOOL_MAX_BYTES`. Falls back to 200_000.
    """
    return _int_env("AGARTHA_FILE_TOOL_MAX_BYTES", 200_000)


# --- Reference tool security gates -------------------------------------

def get_http_tool_allow_private() -> bool:
    """Whether `http_get` may reach private / loopback / link-local addresses.

    Defaults to False: requests resolving to internal IP ranges — including
    the cloud metadata endpoint 169.254.169.254 — are refused to mitigate
    SSRF when the URL is chosen by the LLM. Set
    `AGARTHA_HTTP_TOOL_ALLOW_PRIVATE=1` to permit them (e.g. local development
    hitting localhost).
    """
    return _bool_env("AGARTHA_HTTP_TOOL_ALLOW_PRIVATE", False)


def get_http_tool_allowed_hosts() -> list[str] | None:
    """Optional allowlist of hostnames `http_get` may contact.

    Set `AGARTHA_HTTP_TOOL_ALLOWED_HOSTS` to a comma-separated list; any host
    not on it is refused. When unset (None) all hosts are allowed, subject to
    the private-address check above.
    """
    raw = os.getenv("AGARTHA_HTTP_TOOL_ALLOWED_HOSTS")
    if not raw:
        return None
    return [h.strip().lower() for h in raw.split(",") if h.strip()]


def get_file_tool_root() -> str | None:
    """Optional directory that `read_file` is confined to.

    Set `AGARTHA_FILE_TOOL_ROOT` to sandbox reads: any path resolving outside
    this directory (after following symlinks) is refused, blocking traversal
    such as `../../etc/passwd`. When unset (None), `read_file` may read any
    path the process can access.
    """
    return os.getenv("AGARTHA_FILE_TOOL_ROOT") or None
