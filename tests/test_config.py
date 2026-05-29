"""Runtime configuration sourced from env vars (aether/config.py)."""
import pytest
from aether import Aether, register_tool
from aether.config import (
    get_default_temperature,
    get_file_tool_max_bytes,
    get_http_tool_max_bytes,
    get_http_tool_timeout,
    get_max_tool_iterations,
)
from aether.llm.contracts import LLMResponse, ToolCall
from aether.extensions.llm.fake import FakeProvider
from aether.registry import REGISTRY


@pytest.fixture
def cleanup_registry():
    original = {kind: dict(specs) for kind, specs in REGISTRY.items()}
    yield
    REGISTRY.clear()
    for kind, specs in original.items():
        REGISTRY[kind] = specs


# --- get_max_tool_iterations() --------------------------------------------

def test_falls_back_to_10_when_env_unset(monkeypatch):
    monkeypatch.delenv("AETHER_MAX_TOOL_ITERATIONS", raising=False)
    assert get_max_tool_iterations() == 10


def test_reads_env_when_set(monkeypatch):
    monkeypatch.setenv("AETHER_MAX_TOOL_ITERATIONS", "5")
    assert get_max_tool_iterations() == 5


def test_invalid_env_value_falls_back_to_10(monkeypatch):
    """Don't crash a production agent because someone fat-fingered the env var."""
    monkeypatch.setenv("AETHER_MAX_TOOL_ITERATIONS", "not_a_number")
    assert get_max_tool_iterations() == 10


def test_each_call_re_reads_env(monkeypatch):
    """No caching — testability and live reconfig both rely on this."""
    monkeypatch.setenv("AETHER_MAX_TOOL_ITERATIONS", "3")
    assert get_max_tool_iterations() == 3
    monkeypatch.setenv("AETHER_MAX_TOOL_ITERATIONS", "7")
    assert get_max_tool_iterations() == 7


# --- End-to-end through Aether.complete() ---------------------------------

def _tool_response(call_id: str) -> LLMResponse:
    return LLMResponse(
        text="",
        model="fake-model",
        input_tokens=1,
        output_tokens=1,
        tool_calls=[ToolCall(id=call_id, name="ping", arguments={})],
    )


@pytest.mark.asyncio
async def test_env_caps_the_tool_loop(monkeypatch, cleanup_registry):
    """AETHER_MAX_TOOL_ITERATIONS=2 → loop stops after 2 iterations + 1 initial."""
    monkeypatch.setenv("AETHER_MAX_TOOL_ITERATIONS", "2")

    @register_tool()
    def ping() -> str:
        return "pong"

    fake = FakeProvider(responses=[_tool_response(f"c{i}") for i in range(50)])
    client = Aether(fake)
    await client.complete("loop", tools=["ping"])
    # 2 iterations + 1 initial = 3 LLM calls
    assert len(fake.calls) == 3


@pytest.mark.asyncio
async def test_per_call_override_beats_env(monkeypatch, cleanup_registry):
    """Explicit `max_tool_iterations=N` always wins over the env default."""
    monkeypatch.setenv("AETHER_MAX_TOOL_ITERATIONS", "100")

    @register_tool()
    def ping() -> str:
        return "pong"

    fake = FakeProvider(responses=[_tool_response(f"c{i}") for i in range(50)])
    client = Aether(fake)
    await client.complete("loop", tools=["ping"], max_tool_iterations=1)
    # Caller's 1 wins over env's 100 → 1 + 1 initial = 2 calls
    assert len(fake.calls) == 2


# --- get_default_temperature() --------------------------------------------

def test_temperature_falls_back_to_0_7(monkeypatch):
    monkeypatch.delenv("AETHER_DEFAULT_TEMPERATURE", raising=False)
    assert get_default_temperature() == 0.7


def test_temperature_reads_env(monkeypatch):
    monkeypatch.setenv("AETHER_DEFAULT_TEMPERATURE", "0.0")
    assert get_default_temperature() == 0.0


def test_temperature_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("AETHER_DEFAULT_TEMPERATURE", "hot")
    assert get_default_temperature() == 0.7


@pytest.mark.asyncio
async def test_complete_uses_env_temperature(monkeypatch):
    monkeypatch.setenv("AETHER_DEFAULT_TEMPERATURE", "0.0")
    fake = FakeProvider()
    client = Aether(fake)
    await client.complete("hi")
    assert fake.calls[0].temperature == 0.0


@pytest.mark.asyncio
async def test_per_call_temperature_beats_env(monkeypatch):
    monkeypatch.setenv("AETHER_DEFAULT_TEMPERATURE", "0.0")
    fake = FakeProvider()
    client = Aether(fake)
    await client.complete("hi", temperature=0.9)
    assert fake.calls[0].temperature == 0.9


@pytest.mark.asyncio
async def test_stream_uses_env_temperature(monkeypatch):
    monkeypatch.setenv("AETHER_DEFAULT_TEMPERATURE", "0.2")
    fake = FakeProvider()
    client = Aether(fake)
    _ = [c async for c in client.stream("hi")]
    assert fake.calls[0].temperature == 0.2


# --- get_http_tool_timeout() / get_http_tool_max_bytes() / get_file_tool_max_bytes()

def test_http_timeout_falls_back(monkeypatch):
    monkeypatch.delenv("AETHER_HTTP_TOOL_TIMEOUT", raising=False)
    assert get_http_tool_timeout() == 10.0


def test_http_timeout_reads_env(monkeypatch):
    monkeypatch.setenv("AETHER_HTTP_TOOL_TIMEOUT", "3.5")
    assert get_http_tool_timeout() == 3.5


def test_http_max_bytes_falls_back(monkeypatch):
    monkeypatch.delenv("AETHER_HTTP_TOOL_MAX_BYTES", raising=False)
    assert get_http_tool_max_bytes() == 100_000


def test_http_max_bytes_reads_env(monkeypatch):
    monkeypatch.setenv("AETHER_HTTP_TOOL_MAX_BYTES", "50")
    assert get_http_tool_max_bytes() == 50


def test_file_max_bytes_falls_back(monkeypatch):
    monkeypatch.delenv("AETHER_FILE_TOOL_MAX_BYTES", raising=False)
    assert get_file_tool_max_bytes() == 200_000


def test_file_max_bytes_reads_env(monkeypatch):
    monkeypatch.setenv("AETHER_FILE_TOOL_MAX_BYTES", "50")
    assert get_file_tool_max_bytes() == 50


# --- End-to-end: env-configured truncation actually truncates ------------

@pytest.mark.asyncio
async def test_read_file_truncates_at_env_max_bytes(monkeypatch, tmp_path):
    # Register the reference tool by importing it.
    import aether.extensions.tools.file  # noqa: F401
    from aether.tools.registry import dispatch_tool

    p = tmp_path / "big.txt"
    p.write_text("x" * 500)
    monkeypatch.setenv("AETHER_FILE_TOOL_MAX_BYTES", "100")
    result = await dispatch_tool("read_file", {"path": str(p)})
    assert "truncated" in result
    assert "500 bytes total" in result
    # Body up to the truncation marker is exactly 100 chars
    body = result.split("\n\n[truncated")[0]
    assert len(body) == 100
