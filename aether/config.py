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


# --- Tool loop ----------------------------------------------------------

def get_max_tool_iterations() -> int:
    """Default cap on the tool-dispatch loop in `Aether.complete()`.

    Override the per-process default by setting `AETHER_MAX_TOOL_ITERATIONS`.
    Override per-call by passing `max_tool_iterations=N` to `complete()`.
    Falls back to 10 if the env var is unset or unparseable.
    """
    return _int_env("AETHER_MAX_TOOL_ITERATIONS", 10)


# --- LLM request defaults ----------------------------------------------

def get_default_temperature() -> float:
    """Default sampling temperature used when callers don't pass one.

    Override via `AETHER_DEFAULT_TEMPERATURE`. Per-call kwarg wins.
    Falls back to 0.7 (matches the long-standing OpenAI default).
    """
    return _float_env("AETHER_DEFAULT_TEMPERATURE", 0.7)


# --- Reference tool limits ---------------------------------------------

def get_http_tool_timeout() -> float:
    """Default timeout (seconds) for the built-in `http_get` tool.

    Override via `AETHER_HTTP_TOOL_TIMEOUT`. Falls back to 10.0.
    """
    return _float_env("AETHER_HTTP_TOOL_TIMEOUT", 10.0)


def get_http_tool_max_bytes() -> int:
    """Max bytes the built-in `http_get` returns before truncating.

    Override via `AETHER_HTTP_TOOL_MAX_BYTES`. Falls back to 100_000
    (large enough for most responses, small enough not to flood the
    LLM's context window).
    """
    return _int_env("AETHER_HTTP_TOOL_MAX_BYTES", 100_000)


def get_file_tool_max_bytes() -> int:
    """Max bytes the built-in `read_file` returns before truncating.

    Override via `AETHER_FILE_TOOL_MAX_BYTES`. Falls back to 200_000.
    """
    return _int_env("AETHER_FILE_TOOL_MAX_BYTES", 200_000)
