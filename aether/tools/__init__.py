"""Tool registration and schema generation.

Public surface:
  - `register_tool`  — decorator for any callable (sync or async)
  - `get_tool`       — look up a registered tool by name
  - `list_tools`     — names of all registered tools
  - `tool_schema`    — JSON Schema for one tool (used by LLM providers)
  - `dispatch_tool`  — invoke a registered tool with parsed arguments

Aether's tool calling layers on `aether.llm` — the LLM contract (Message,
ToolCall, LLMResponse.tool_calls) is what carries this across providers.
"""

from aether.tools.registry import (
    TOOL_KIND,
    ToolSpec,
    register_tool,
    get_tool,
    list_tools,
    dispatch_tool,
)
from aether.tools.schema import tool_schema

__all__ = [
    "TOOL_KIND",
    "ToolSpec",
    "register_tool",
    "get_tool",
    "list_tools",
    "dispatch_tool",
    "tool_schema",
]
