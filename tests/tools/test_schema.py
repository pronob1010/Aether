"""Auto-generated JSON Schema from Python function signatures."""
from typing import Optional
from aether.tools.schema import tool_schema


def test_basic_types_map_correctly():
    def f(s: str, n: int, x: float, ok: bool): ...
    schema = tool_schema(f)
    props = schema["parameters"]["properties"]
    assert props["s"]["type"] == "string"
    assert props["n"]["type"] == "integer"
    assert props["x"]["type"] == "number"
    assert props["ok"]["type"] == "boolean"


def test_required_vs_optional_parameters():
    def f(required_arg: str, optional_arg: int = 42): ...
    schema = tool_schema(f)
    assert schema["parameters"]["required"] == ["required_arg"]


def test_optional_type_unwraps_to_inner():
    def f(maybe: Optional[str] = None): ...
    schema = tool_schema(f)
    assert schema["parameters"]["properties"]["maybe"]["type"] == "string"


def test_list_type_becomes_array():
    def f(items: list[str]): ...
    schema = tool_schema(f)
    prop = schema["parameters"]["properties"]["items"]
    assert prop["type"] == "array"
    assert prop["items"]["type"] == "string"


def test_dict_type_becomes_object():
    def f(data: dict): ...
    schema = tool_schema(f)
    assert schema["parameters"]["properties"]["data"]["type"] == "object"


def test_unknown_type_falls_back_to_string():
    """We don't want schema generation to crash on weird types."""
    class Custom: pass
    def f(thing: Custom): ...
    schema = tool_schema(f)
    assert schema["parameters"]["properties"]["thing"]["type"] == "string"


def test_docstring_first_paragraph_becomes_description():
    def f():
        """Get the current weather.

        More details that should be ignored.
        """
    schema = tool_schema(f)
    assert schema["description"] == "Get the current weather."


def test_args_block_provides_per_param_descriptions():
    def f(city: str, units: str = "celsius"):
        """Get the weather.

        Args:
            city: City name like 'Paris' or 'SF'.
            units: 'celsius' or 'fahrenheit'.
        """
    schema = tool_schema(f)
    props = schema["parameters"]["properties"]
    assert props["city"]["description"] == "City name like 'Paris' or 'SF'."
    assert props["units"]["description"] == "'celsius' or 'fahrenheit'."


def test_explicit_name_and_description_override():
    def some_function():
        """Default desc."""
    schema = tool_schema(some_function, name="custom_name", description="custom desc")
    assert schema["name"] == "custom_name"
    assert schema["description"] == "custom desc"


def test_self_and_varargs_are_skipped():
    class C:
        def m(self, x: int, *args, **kwargs): ...
    schema = tool_schema(C.m)
    assert list(schema["parameters"]["properties"]) == ["x"]
