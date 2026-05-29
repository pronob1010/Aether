"""Tool registry — built on the generic `aether.registry`.

A tool is just a Python callable (sync or async) the LLM can invoke during
a `complete()` call. Registration auto-generates the JSON Schema the LLM
needs to know how to call it.
"""
import inspect
from dataclasses import dataclass
from typing import Any, Callable
from aether.registry import get as _registry_get, list_kind, register_lazy
from aether.tools.schema import tool_schema


TOOL_KIND = "tool"


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool — the callable plus its JSON Schema."""
    func: Callable
    schema: dict[str, Any]


def register_tool(
    name: str | None = None,
    *,
    description: str | None = None,
    **metadata: Any,
):
    """Decorator that registers a function as a callable tool.

    `name` defaults to the function's __name__.
    `description` defaults to the docstring's first paragraph.
    `**metadata` is forwarded into the registry spec for future consumers.

    Example:
        @register_tool(description="Get the current time")
        def get_current_time(timezone: str = "UTC") -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        if not callable(func):
            raise TypeError(f"register_tool expects a callable, got {type(func).__name__}")
        resolved_name = name or func.__name__
        schema = tool_schema(func, name=resolved_name, description=description)
        spec = ToolSpec(func=func, schema=schema)
        # The generic registry's `register_lazy` accepts any factory output —
        # we hand back the ToolSpec directly (no need to fake a class wrapper).
        register_lazy(TOOL_KIND, resolved_name, lambda: spec, **metadata)
        return func
    return decorator


def get_tool(name: str) -> ToolSpec:
    """Look up a registered tool by name. Raises KeyError if unknown."""
    return _registry_get(TOOL_KIND, name).factory()


def list_tools() -> list[str]:
    """Names of all registered tools."""
    return list_kind(TOOL_KIND)


async def dispatch_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Invoke a registered tool with parsed arguments.

    Handles both sync and async tool functions transparently.
    """
    spec = get_tool(name)
    result = spec.func(**arguments)
    if inspect.isawaitable(result):
        result = await result
    return result
