"""HTTP GET tool. Uses `httpx` if available (async-friendly), falls back to urllib."""
from aether import register_tool
from aether.config import get_http_tool_max_bytes, get_http_tool_timeout


@register_tool(description="Fetch the body of an HTTP(S) URL via GET. Returns text.")
async def http_get(url: str, timeout: float | None = None) -> str:
    """Send an HTTP GET request and return the response body.

    Args:
        url: Full URL starting with http:// or https://.
        timeout: Request timeout in seconds. Omit to use the framework
            default (AETHER_HTTP_TOOL_TIMEOUT env, falls back to 10s).
    """
    if not url.startswith(("http://", "https://")):
        return f"Error: url must start with http:// or https://, got {url!r}."
    if timeout is None:
        timeout = get_http_tool_timeout()
    max_bytes = get_http_tool_max_bytes()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            text = response.text
    except ImportError:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as f:  # noqa: S310
            text = f.read(max_bytes + 1).decode("utf-8", errors="replace")
    if len(text) > max_bytes:
        return text[:max_bytes] + f"\n\n[truncated — {len(text)} bytes total]"
    return text
