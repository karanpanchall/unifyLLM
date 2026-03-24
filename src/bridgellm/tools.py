"""Tool decorator and registry — auto-generate OpenAI tool definitions from functions.

The @tool decorator inspects type hints and docstrings to build JSON Schema.
ToolRegistry dispatches tool calls by name.
"""

import inspect
import logging
import re
from typing import Any, Callable, Optional, get_type_hints

logger = logging.getLogger(__name__)

# Python type → JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolDefinition:
    """A tool with its OpenAI-compatible definition and callable handler."""

    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or _extract_description(func)
        self.parameters = _build_parameters(func)

    def as_openai_tool(self) -> dict:
        """Return the OpenAI-compatible tool definition dict."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, arguments: dict, context: Any = None) -> str:
        """Execute the tool handler with given arguments.

        If the handler accepts a 'context' parameter, it is injected.
        """
        sig = inspect.signature(self.func)
        kwargs = dict(arguments)
        if "context" in sig.parameters and context is not None:
            kwargs["context"] = context
        result = self.func(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return str(result)


def tool(func: Optional[Callable] = None, *, name: Optional[str] = None, description: Optional[str] = None):
    """Decorator that converts a function into a ToolDefinition.

    Usage:
        @tool
        async def get_weather(city: str) -> str:
            '''Get weather for a city.'''
            return "72°F"

        @tool(name="search", description="Custom description")
        async def search_web(query: str) -> str:
            return "results..."
    """
    def wrapper(fn: Callable) -> ToolDefinition:
        return ToolDefinition(func=fn, name=name, description=description)

    if func is not None:
        return ToolDefinition(func=func)
    return wrapper


class ToolRegistry:
    """Registry for dispatching tool calls by name."""

    def __init__(self, tools: Optional[list[ToolDefinition]] = None):
        self._tools: dict[str, ToolDefinition] = {}
        for registered_tool in (tools or []):
            self.register(registered_tool)

    def register(self, registered_tool: ToolDefinition) -> None:
        self._tools[registered_tool.name] = registered_tool
        logger.debug("Registered tool: %s", registered_tool.name)

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def as_openai_tools(self) -> list[dict]:
        """Return all tools as OpenAI-compatible definitions."""
        return [registered_tool.as_openai_tool() for registered_tool in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(self, name: str, arguments: dict, context: Any = None) -> str:
        """Execute a tool by name. Returns result string or error message."""
        registered_tool = self._tools.get(name)
        if not registered_tool:
            return f"Error: Unknown tool '{name}'. Available: {', '.join(self._tools.keys())}"
        return await registered_tool.execute(arguments, context)


# -- Schema generation helpers --


def _extract_description(func: Callable) -> str:
    """Extract the first paragraph of a function's docstring."""
    doc = inspect.getdoc(func)
    if not doc:
        return func.__name__
    first_paragraph = doc.split("\n\n")[0].replace("\n", " ").strip()
    return first_paragraph


def _build_parameters(func: Callable) -> dict:
    """Build JSON Schema parameters from function type hints and docstring."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    param_docs = _parse_param_docs(func)

    properties: dict[str, dict] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "context", "return"):
            continue

        python_type = hints.get(param_name, str)
        json_type = _TYPE_MAP.get(python_type, "string")
        prop: dict[str, Any] = {"type": json_type}

        doc = param_docs.get(param_name)
        if doc:
            prop["description"] = doc

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = prop

    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _parse_param_docs(func: Callable) -> dict[str, str]:
    """Parse Args section from Google-style docstrings."""
    doc = inspect.getdoc(func)
    if not doc:
        return {}

    param_docs: dict[str, str] = {}
    in_args = False

    for line in doc.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if not stripped or (not stripped.startswith(" ") and ":" not in stripped and stripped.lower().startswith(("returns", "raises", "note", "example"))):
                break
            match = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", stripped)
            if match:
                param_docs[match.group(1)] = match.group(2).strip()

    return param_docs
