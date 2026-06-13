"""The shipped examples must stay importable and the offline one must run.

Examples register tools at import time, so each load is wrapped in a registry
snapshot/restore to avoid leaking into other tests.
"""
import importlib.util
from pathlib import Path

import pytest

from aether.registry import REGISTRY

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def cleanup_registry():
    original = {kind: dict(specs) for kind, specs in REGISTRY.items()}
    yield
    REGISTRY.clear()
    for kind, specs in original.items():
        REGISTRY[kind] = specs


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", [
    "offline_tool_loop",
    "tool_agent",
    "rag_grounding",
    "multi_agent",
])
def test_example_loads(name, cleanup_registry):
    # Constructing a live client happens inside each example's main(), so
    # importing the module never needs an API key.
    assert hasattr(_load(name), "main")


@pytest.mark.asyncio
async def test_offline_example_runs(cleanup_registry):
    result = await _load("offline_tool_loop").main()
    assert "42" in result
