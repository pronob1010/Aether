"""Auto-generate JSON Schema for a Python function.

Extracts:
  - parameter names and types from the signature
  - per-parameter descriptions from a Google-style docstring's `Args:` block
  - which parameters are required (no default value)

Output shape matches OpenAI's function-calling format. Gemini and Anthropic
accept the same shape with minor wrapping done by their provider adapters.
"""
import inspect
import re
import types
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints


_PYTHON_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(py_type: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment.

    Handles primitives, Optional[T], list[T], and dict.
    Falls back to "string" for unknown types — better than crashing.
    """
    origin = get_origin(py_type)

    # Optional[T] / Union[T, None] / T | None → schema for T (LLM ignores nullability).
    # Must check Union origin explicitly — `list[str]` also has args, but it
    # is NOT a Union and should fall through to the array case.
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(py_type) if a is not type(None)]
        if len(args) == 1:
            return _python_type_to_json_schema(args[0])

    # list[T] → array with items
    if origin in (list, tuple):
        item_args = get_args(py_type)
        items = _python_type_to_json_schema(item_args[0]) if item_args else {"type": "string"}
        return {"type": "array", "items": items}

    # dict → object (LLM doesn't get inner shape; that's fine for most tools)
    if origin is dict:
        return {"type": "object"}

    return {"type": _PYTHON_TO_JSON_TYPE.get(py_type, "string")}


def _parse_args_block(docstring: str | None) -> dict[str, str]:
    """Pull per-parameter descriptions from a Google-style 'Args:' block.

        Args:
            city: City name.
            units: 'celsius' or 'fahrenheit'.
    """
    if not docstring:
        return {}
    # Find the Args: section and slurp until a blank line or the next
    # un-indented section heading (Returns:, Raises:, Yields:, ...).
    # Parameter lines inside Args: are indented, so requiring NO leading
    # whitespace before the next heading avoids stopping at the first param.
    match = re.search(
        r"(?:Args|Arguments|Parameters):\s*\n(.+?)(?=\n\s*\n|\n\w+:|\Z)",
        docstring,
        re.DOTALL,
    )
    if not match:
        return {}
    body = match.group(1)
    descriptions: dict[str, str] = {}
    for line in body.splitlines():
        m = re.match(r"\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", line)
        if m:
            descriptions[m.group(1)] = m.group(2).strip()
    return descriptions


def tool_schema(
    func: Callable,
    *,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Build a JSON Schema description of `func` for LLM function calling.

    `name` defaults to the function's __name__.
    `description` defaults to the docstring's first line.
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    arg_descriptions = _parse_args_block(inspect.getdoc(func))

    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        prop = _python_type_to_json_schema(hints.get(param_name, str))
        if param_name in arg_descriptions:
            prop["description"] = arg_descriptions[param_name]
        properties[param_name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    final_description = description
    if final_description is None:
        doc = inspect.getdoc(func)
        if doc:
            final_description = doc.split("\n\n")[0].strip()

    schema: dict[str, Any] = {
        "name": name or func.__name__,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
    if final_description:
        schema["description"] = final_description
    return schema
