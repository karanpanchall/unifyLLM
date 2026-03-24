"""Tests for @tool decorator, ToolDefinition, and ToolRegistry."""

import pytest

from bridgellm.tools import ToolDefinition, ToolRegistry, tool


@tool
async def simple_tool(query: str) -> str:
    """Search for information."""
    return f"Result for: {query}"


@tool
async def multi_param(city: str, unit: str = "celsius") -> str:
    """Get weather for a city.

    Args:
        city: City name
        unit: Temperature unit
    """
    return f"{city}: 72°F"


@tool(name="custom_name", description="Custom description")
async def renamed_tool(text: str) -> str:
    return text


class TestToolDecorator:
    def test_creates_tool_definition(self):
        assert isinstance(simple_tool, ToolDefinition)
        assert simple_tool.name == "simple_tool"

    def test_extracts_description_from_docstring(self):
        assert simple_tool.description == "Search for information."

    def test_custom_name_and_description(self):
        assert renamed_tool.name == "custom_name"
        assert renamed_tool.description == "Custom description"

    def test_generates_parameters_schema(self):
        schema = simple_tool.parameters
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        assert "query" in schema["required"]

    def test_default_params_not_required(self):
        schema = multi_param.parameters
        assert "city" in schema["required"]
        assert "unit" not in schema["required"]
        assert schema["properties"]["unit"]["default"] == "celsius"

    def test_param_descriptions_from_docstring(self):
        schema = multi_param.parameters
        assert schema["properties"]["city"]["description"] == "City name"
        assert schema["properties"]["unit"]["description"] == "Temperature unit"


class TestToolDefinitionOpenAI:
    def test_as_openai_tool(self):
        definition = simple_tool.as_openai_tool()
        assert definition["type"] == "function"
        assert definition["function"]["name"] == "simple_tool"
        assert definition["function"]["description"] == "Search for information."
        assert "parameters" in definition["function"]


class TestToolDefinitionExecute:
    @pytest.mark.asyncio
    async def test_execute_async(self):
        result = await simple_tool.execute({"query": "test"})
        assert result == "Result for: test"

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        @tool
        async def ctx_tool(name: str, context: dict = None) -> str:
            return f"Hello {name}, user={context.get('user_id')}"

        result = await ctx_tool.execute({"name": "World"}, context={"user_id": "123"})
        assert "user=123" in result

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        @tool
        def sync_tool(value: int) -> str:
            return f"Got: {value}"

        result = await sync_tool.execute({"value": 42})
        assert result == "Got: 42"


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry([simple_tool, multi_param])
        assert registry.get("simple_tool") is simple_tool
        assert registry.get("multi_param") is multi_param

    def test_unknown_tool_returns_none(self):
        registry = ToolRegistry([simple_tool])
        assert registry.get("nonexistent") is None

    def test_as_openai_tools(self):
        registry = ToolRegistry([simple_tool, multi_param])
        defs = registry.as_openai_tools()
        assert len(defs) == 2
        assert all(d["type"] == "function" for d in defs)

    def test_tool_names(self):
        registry = ToolRegistry([simple_tool, multi_param])
        assert set(registry.tool_names) == {"simple_tool", "multi_param"}

    @pytest.mark.asyncio
    async def test_execute_by_name(self):
        registry = ToolRegistry([simple_tool])
        result = await registry.execute("simple_tool", {"query": "hello"})
        assert "Result for: hello" in result

    @pytest.mark.asyncio
    async def test_execute_unknown_returns_error(self):
        registry = ToolRegistry([simple_tool])
        result = await registry.execute("nonexistent", {})
        assert "Unknown tool" in result

    def test_empty_registry(self):
        registry = ToolRegistry()
        assert registry.tool_names == []
        assert registry.as_openai_tools() == []
