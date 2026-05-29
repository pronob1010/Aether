"""Reference tools shipped under aether.extensions.tools."""
import pytest
import tempfile
from pathlib import Path
from aether.tools import list_tools, get_tool
from aether.tools.registry import dispatch_tool


# Importing the package triggers registration of all 3 reference tools.
import aether.extensions.tools  # noqa: F401


def test_all_reference_tools_register():
    names = list_tools()
    assert "get_current_time" in names
    assert "http_get" in names
    assert "read_file" in names


def test_get_current_time_schema_describes_tz_arg():
    spec = get_tool("get_current_time")
    props = spec.schema["parameters"]["properties"]
    assert "tz" in props
    assert "IANA" in props["tz"]["description"]


# --- get_current_time ----------------------------------------------------

@pytest.mark.asyncio
async def test_get_current_time_utc():
    result = await dispatch_tool("get_current_time", {})
    # ISO 8601 UTC ends with +00:00
    assert "+00:00" in result or result.endswith("Z")


@pytest.mark.asyncio
async def test_get_current_time_explicit_tz():
    result = await dispatch_tool("get_current_time", {"tz": "America/Los_Angeles"})
    # LA offset is -07:00 or -08:00
    assert "-07:00" in result or "-08:00" in result


@pytest.mark.asyncio
async def test_get_current_time_unknown_tz_returns_error_string():
    """Tool reports error in its return value rather than raising — better UX for the LLM."""
    result = await dispatch_tool("get_current_time", {"tz": "Mars/Olympus_Mons"})
    assert "Error" in result
    assert "Mars/Olympus_Mons" in result


# --- read_file -----------------------------------------------------------

@pytest.mark.asyncio
async def test_read_file_returns_content(tmp_path):
    p = tmp_path / "hi.txt"
    p.write_text("hello world\n")
    result = await dispatch_tool("read_file", {"path": str(p)})
    assert result == "hello world\n"


@pytest.mark.asyncio
async def test_read_file_missing_returns_error_string(tmp_path):
    result = await dispatch_tool("read_file", {"path": str(tmp_path / "nope.txt")})
    assert "Error" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_read_file_on_directory_returns_error(tmp_path):
    result = await dispatch_tool("read_file", {"path": str(tmp_path)})
    assert "Error" in result
    assert "not a file" in result


# --- http_get ------------------------------------------------------------

@pytest.mark.asyncio
async def test_http_get_rejects_non_http_url():
    """Don't even try filesystem paths or ftp URLs — guard at the boundary."""
    result = await dispatch_tool("http_get", {"url": "file:///etc/passwd"})
    assert "Error" in result
    assert "http://" in result or "https://" in result


# Note: live HTTP integration test omitted — flaky on offline test runs
# and not the responsibility of this test file. Real testing happens at
# integration time.
