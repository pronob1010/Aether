"""HTTP GET tool. Uses `httpx` if available (async-friendly), falls back to urllib."""
from aether import register_tool


_MAX_BYTES = 100_000  # truncate huge responses so they don't blow the LLM context


@register_tool(description="Fetch the body of an HTTP(S) URL via GET. Returns text.")
async def http_get(url: str, timeout: float = 10.0) -> str:
    """Send an HTTP GET request and return the response body.

    Args:
        url: Full URL starting with http:// or https://.
        timeout: Request timeout in seconds. Defaults to 10.
    """
    if not url.startswith(("http://", "https://")):
        return f"Error: url must start with http:// or https://, got {url!r}."
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            text = response.text
    except ImportError:
        # Fallback for environments without httpx — sync urllib, but called
        # from async context. Acceptable for a reference tool.
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as f:  # noqa: S310
            text = f.read(_MAX_BYTES + 1).decode("utf-8", errors="replace")
    if len(text) > _MAX_BYTES:
        return text[:_MAX_BYTES] + f"\n\n[truncated — {len(text)} bytes total]"
    return text
