"""File-reading tool. Returns text from a path on the local filesystem."""
from pathlib import Path
from aether import register_tool
from aether.config import get_file_tool_max_bytes


@register_tool(description="Read a text file from the local filesystem.")
def read_file(path: str, encoding: str = "utf-8") -> str:
    """Read the contents of a file and return it as text.

    Args:
        path: Filesystem path to a text file.
        encoding: Text encoding. Defaults to 'utf-8'. Decoding errors are
            replaced (no crash on weird bytes).
    """
    p = Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    if not p.is_file():
        return f"Error: not a file: {path}"
    try:
        data = p.read_bytes()
    except PermissionError:
        return f"Error: permission denied: {path}"
    max_bytes = get_file_tool_max_bytes()
    if len(data) > max_bytes:
        truncated = data[:max_bytes].decode(encoding, errors="replace")
        return truncated + f"\n\n[truncated — {len(data)} bytes total]"
    return data.decode(encoding, errors="replace")
