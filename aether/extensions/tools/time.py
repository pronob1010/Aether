"""Current-time tool. Stdlib only — no dependencies."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from aether import register_tool


@register_tool(description="Get the current date and time in a given timezone.")
def get_current_time(tz: str = "UTC") -> str:
    """Return the current time as an ISO-8601 string.

    Args:
        tz: IANA timezone name like 'UTC', 'America/Los_Angeles',
            'Europe/Berlin'. Defaults to UTC.
    """
    if tz.upper() == "UTC":
        return datetime.now(timezone.utc).isoformat()
    try:
        return datetime.now(ZoneInfo(tz)).isoformat()
    except ZoneInfoNotFoundError:
        return f"Error: unknown timezone {tz!r}. Use an IANA name like 'America/New_York'."
