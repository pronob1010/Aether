"""HTTP GET tool. Uses `httpx` if available (async-friendly), falls back to urllib.

Security: the URL here is typically chosen by the LLM, so this tool is an SSRF
sink. By default it refuses URLs that resolve to private / loopback / link-local
addresses (e.g. the cloud metadata endpoint 169.254.169.254) and honours an
optional host allowlist. See `aether.config` for the gates. Note this is a
best-effort check at request time and does not defend against DNS rebinding.
"""
import ipaddress
import socket
from urllib.parse import urlparse

from aether import register_tool
from aether.config import (
    get_http_tool_allow_private,
    get_http_tool_allowed_hosts,
    get_http_tool_max_bytes,
    get_http_tool_timeout,
)


def _ssrf_check(url: str) -> str | None:
    """Return an error message if `url` is disallowed, else None."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return f"Error: could not parse a host from url {url!r}."

    allowed = get_http_tool_allowed_hosts()
    if allowed is not None and host not in allowed:
        return f"Error: host {host!r} is not in AETHER_HTTP_TOOL_ALLOWED_HOSTS."

    if get_http_tool_allow_private():
        return None

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return f"Error: could not resolve host {host!r}."
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return (
                f"Error: refusing to fetch {url!r} — host {host!r} resolves to a "
                f"private/internal address ({ip}). Set "
                f"AETHER_HTTP_TOOL_ALLOW_PRIVATE=1 to allow."
            )
    return None


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
    blocked = _ssrf_check(url)
    if blocked is not None:
        return blocked
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
