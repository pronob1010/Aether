"""Tool registry — @register_tool decorator + dispatch."""
import pytest
from aether import register_tool, list_tools, get_tool
from aether.tools.registry import dispatch_tool, TOOL_KIND
from aether.registry import REGISTRY


@pytest.fixture
def cleanup_registry():
    original = {kind: dict(specs) for kind, specs in REGISTRY.items()}
    yield
    REGISTRY.clear()
    for kind, specs in original.items():
        REGISTRY[kind] = specs


# --- Registration --------------------------------------------------------

def test_register_uses_function_name_by_default(cleanup_registry):
    @register_tool()
    def my_func(x: int) -> int:
        return x

    assert "my_func" in list_tools()


def test_register_with_explicit_name(cleanup_registry):
    @register_tool(name="renamed")
    def the_func(): ...

    assert "renamed" in list_tools()
    assert "the_func" not in list_tools()


def test_register_stores_schema_with_spec(cleanup_registry):
    @register_tool(description="Adds two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    spec = get_tool("add")
    assert spec.schema["name"] == "add"
    assert spec.schema["description"] == "Adds two numbers"
    assert set(spec.schema["parameters"]["properties"]) == {"a", "b"}
    assert spec.schema["parameters"]["required"] == ["a", "b"]


def test_register_rejects_non_callable(cleanup_registry):
    with pytest.raises(TypeError, match="expects a callable"):
        register_tool()("not a callable")


def test_register_kinds_namespaced_from_llm_providers():
    """tool/X and llm_provider/X don't collide — proven by built-ins."""
    assert TOOL_KIND == "tool"
    # llm_provider "fake" exists (a built-in), tool "fake" doesn't conflict.
    @register_tool(name="fake")
    def fake_tool(): ...
    assert "fake" in list_tools()
    # Restore for other tests
    del REGISTRY[TOOL_KIND]["fake"]


# --- Dispatch ------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_sync_tool(cleanup_registry):
    @register_tool()
    def add(a: int, b: int) -> int:
        return a + b

    result = await dispatch_tool("add", {"a": 2, "b": 3})
    assert result == 5


@pytest.mark.asyncio
async def test_dispatch_async_tool(cleanup_registry):
    @register_tool()
    async def fetch(url: str) -> str:
        return f"fetched {url}"

    result = await dispatch_tool("fetch", {"url": "example.com"})
    assert result == "fetched example.com"


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_raises(cleanup_registry):
    with pytest.raises(KeyError, match="unknown tool"):
        await dispatch_tool("nope", {})


# --- Schema generation through @register_tool ---------------------------

def test_schema_extracts_arg_descriptions(cleanup_registry):
    @register_tool()
    def get_weather(city: str, units: str = "celsius") -> str:
        """Get weather for a city.

        Args:
            city: City name.
            units: 'celsius' or 'fahrenheit'.
        """
        return ""

    spec = get_tool("get_weather")
    props = spec.schema["parameters"]["properties"]
    assert props["city"]["description"] == "City name."
    assert props["units"]["description"] == "'celsius' or 'fahrenheit'."
