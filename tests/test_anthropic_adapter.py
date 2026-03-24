"""Tests for the Anthropic adapter.

Validates translation between OpenAI-shaped inputs and Anthropic's
content-block-based API. Uses mock objects matching Anthropic SDK v0.79.x shapes.
"""

import json
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bridgellm.adapters.anthropic import (
    _assemble_tools,
    _convert_tools,
    _map_stop_reason,
    _parse_content_blocks,
    _prepare_messages,
    _translate_tool_choice,
)
from bridgellm.errors import ProviderError, SDKNotInstalledError
from bridgellm.models import ToolCall


# -- Helper function tests --


class TestPrepareMessages:
    def test_extracts_single_system_message(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        system, conversation = _prepare_messages(messages)
        assert system == "You are helpful."
        assert len(conversation) == 1
        assert conversation[0]["role"] == "user"

    def test_concatenates_multiple_system_messages(self):
        messages = [
            {"role": "system", "content": "Rule one."},
            {"role": "system", "content": "Rule two."},
            {"role": "user", "content": "Hello"},
        ]
        system, conversation = _prepare_messages(messages)
        assert "Rule one." in system
        assert "Rule two." in system

    def test_no_system_messages(self):
        messages = [{"role": "user", "content": "Hello"}]
        system, conversation = _prepare_messages(messages)
        assert system == ""
        assert len(conversation) >= 1

    def test_empty_messages(self):
        system, conversation = _prepare_messages([])
        assert system == ""
        assert conversation == []

    def test_skips_empty_system_content(self):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ]
        system, conversation = _prepare_messages(messages)
        assert system == ""


class TestConvertTools:
    def test_openai_format_to_anthropic(self):
        openai_tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]
        result = _convert_tools(openai_tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get current weather"
        assert result[0]["input_schema"]["type"] == "object"

    def test_flat_format(self):
        """Some callers pass tools without the function wrapper."""
        flat_tools = [{
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        }]
        result = _convert_tools(flat_tools)
        assert result[0]["name"] == "search"


class TestParseContentBlocks:
    def test_text_only(self):
        @dataclass
        class TextBlock:
            type: str = "text"
            text: str = "Hello world"

        text, reasoning, tools = _parse_content_blocks([TextBlock()])
        assert text == "Hello world"
        assert reasoning == ""
        assert tools == []

    def test_tool_use_only(self):
        @dataclass
        class ToolUseBlock:
            type: str = "tool_use"
            id: str = "toolu_1"
            name: str = "get_weather"
            input: dict = None

            def __post_init__(self):
                if self.input is None:
                    self.input = {"city": "NYC"}

        text, reasoning, tools = _parse_content_blocks([ToolUseBlock()])
        assert text == ""
        assert len(tools) == 1
        assert tools[0].function_name == "get_weather"
        assert tools[0].arguments == {"city": "NYC"}

    def test_thinking_blocks(self):
        @dataclass
        class ThinkingBlock:
            type: str = "thinking"
            thinking: str = "Let me reason about this..."

        text, reasoning, tools = _parse_content_blocks([ThinkingBlock()])
        assert text == ""
        assert reasoning == "Let me reason about this..."

    def test_mixed_blocks(self):
        @dataclass
        class TextBlock:
            type: str = "text"
            text: str = ""

        @dataclass
        class ToolUseBlock:
            type: str = "tool_use"
            id: str = ""
            name: str = ""
            input: dict = None

            def __post_init__(self):
                if self.input is None:
                    self.input = {}

        blocks = [
            TextBlock(text="Let me check that."),
            ToolUseBlock(id="t1", name="search", input={"q": "test"}),
        ]
        text, reasoning, tools = _parse_content_blocks(blocks)
        assert "Let me check" in text
        assert len(tools) == 1

    def test_empty_blocks(self):
        text, reasoning, tools = _parse_content_blocks([])
        assert text == ""
        assert reasoning == ""
        assert tools == []


class TestAssembleAnthropicTools:
    def test_valid_json(self):
        accumulator = {
            0: {"id": "t1", "name": "fn", "arguments": '{"key": "val"}'},
        }
        result = _assemble_tools(accumulator)
        assert len(result) == 1
        assert result[0].arguments == {"key": "val"}

    def test_malformed_json(self):
        accumulator = {0: {"id": "t1", "name": "fn", "arguments": "broken"}}
        result = _assemble_tools(accumulator)
        assert result[0].arguments == {"_parse_error": True, "raw": "broken"}

    def test_empty_arguments(self):
        accumulator = {0: {"id": "t1", "name": "fn", "arguments": ""}}
        result = _assemble_tools(accumulator)
        assert result[0].arguments == {}

    def test_sorted_by_index(self):
        accumulator = {
            2: {"id": "t3", "name": "fn_c", "arguments": "{}"},
            0: {"id": "t1", "name": "fn_a", "arguments": "{}"},
            1: {"id": "t2", "name": "fn_b", "arguments": "{}"},
        }
        result = _assemble_tools(accumulator)
        assert [call.function_name for call in result] == ["fn_a", "fn_b", "fn_c"]


class TestMapStopReason:
    def test_end_turn(self):
        assert _map_stop_reason("end_turn") == "stop"

    def test_stop_sequence(self):
        assert _map_stop_reason("stop_sequence") == "stop"

    def test_max_tokens(self):
        assert _map_stop_reason("max_tokens") == "length"

    def test_tool_use(self):
        assert _map_stop_reason("tool_use") == "tool_calls"

    def test_unknown_reason_passed_through(self):
        assert _map_stop_reason("unknown_reason") == "unknown_reason"

    def test_empty_string(self):
        assert _map_stop_reason("") == ""


class TestTranslateToolChoice:
    def test_auto(self):
        assert _translate_tool_choice("auto") == {"type": "auto"}

    def test_none(self):
        assert _translate_tool_choice("none") == {"type": "none"}

    def test_required_maps_to_any(self):
        assert _translate_tool_choice("required") == {"type": "any"}

    def test_dict_passthrough(self):
        custom = {"type": "tool", "name": "my_func"}
        assert _translate_tool_choice(custom) == custom

    def test_unknown_defaults_to_auto(self):
        assert _translate_tool_choice("unknown") == {"type": "auto"}


class TestAnthropicAdapterSDKMissing:
    def test_raises_sdk_not_installed(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises((SDKNotInstalledError, ImportError)):
                from bridgellm.adapters.anthropic import _get_anthropic_module
                _get_anthropic_module()


# -- Mock SDK objects matching Anthropic response shapes --


@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class MockToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = None

    def __post_init__(self):
        if self.input is None:
            self.input = {}


@dataclass
class MockUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class MockMessage:
    content: list = None
    usage: MockUsage = None
    model: str = "claude-sonnet"
    stop_reason: str = "end_turn"

    def __post_init__(self):
        if self.content is None:
            self.content = []
        if self.usage is None:
            self.usage = MockUsage()


@dataclass
class MockModelEntry:
    id: str = ""


class _MockAsyncIter:
    """Async iterator wrapper for testing list_models."""
    def __init__(self, items):
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


# -- Adapter method tests (mocked SDK) --


def _make_adapter():
    """Create an AnthropicAdapter with a mocked SDK client."""
    from bridgellm.adapters.anthropic import AnthropicAdapter
    from bridgellm.registry import ProviderConfig

    config = ProviderConfig(
        base_url="https://api.anthropic.com",
        api_key_env="ANTHROPIC_API_KEY",
        openai_compatible=False,
    )
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    adapter._client = MagicMock()
    return adapter


class TestAnthropicAdapterComplete:
    @pytest.mark.asyncio
    async def test_returns_llm_response(self):
        adapter = _make_adapter()
        mock_msg = MockMessage(
            content=[MockTextBlock(text="Hello from Claude")],
            usage=MockUsage(input_tokens=15, output_tokens=8),
            model="claude-sonnet",
            stop_reason="end_turn",
        )
        adapter._client.messages.create = AsyncMock(return_value=mock_msg)

        result = await adapter.complete(
            model="claude-sonnet",
            messages=[
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ],
        )
        assert result.content == "Hello from Claude"
        assert result.input_tokens == 15
        assert result.output_tokens == 8
        assert result.finish_reason == "stop"

        # Verify system was extracted to top-level param
        call_kwargs = adapter._client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be helpful."
        assert all(msg["role"] != "system" for msg in call_kwargs["messages"])

    @pytest.mark.asyncio
    async def test_with_tool_use_response(self):
        adapter = _make_adapter()
        mock_msg = MockMessage(
            content=[
                MockTextBlock(text="Let me check."),
                MockToolUseBlock(id="t1", name="weather", input={"city": "NYC"}),
            ],
            stop_reason="tool_use",
        )
        adapter._client.messages.create = AsyncMock(return_value=mock_msg)

        result = await adapter.complete(model="claude-sonnet", messages=[{"role": "user", "content": "weather?"}])
        assert "Let me check" in result.content
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "weather"
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_sdk_error_wrapped(self):
        adapter = _make_adapter()
        adapter._client.messages.create = AsyncMock(side_effect=RuntimeError("API error"))

        with pytest.raises(ProviderError, match="API error"):
            await adapter.complete(model="m", messages=[{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_tools_converted_to_anthropic_format(self):
        adapter = _make_adapter()
        adapter._client.messages.create = AsyncMock(return_value=MockMessage())

        openai_tools = [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        await adapter.complete(model="m", messages=[{"role": "user", "content": "hi"}], tools=openai_tools)

        call_kwargs = adapter._client.messages.create.call_args[1]
        assert call_kwargs["tools"][0]["name"] == "search"
        assert "input_schema" in call_kwargs["tools"][0]

    @pytest.mark.asyncio
    async def test_default_max_tokens(self):
        adapter = _make_adapter()
        adapter._client.messages.create = AsyncMock(return_value=MockMessage())

        await adapter.complete(model="m", messages=[{"role": "user", "content": "hi"}])
        call_kwargs = adapter._client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096


class _MockStreamEvent:
    """Simulates events from Anthropic's streaming API."""
    def __init__(self, event_type, **kwargs):
        self.type = event_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class _MockContentBlock:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class _MockStream:
    """Simulates Anthropic's async stream context manager."""
    def __init__(self, events, final_message):
        self._events = events
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def __aiter__(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._final_message


class TestAnthropicAdapterStream:
    @pytest.mark.asyncio
    async def test_streams_text_deltas(self):
        adapter = _make_adapter()

        events = [
            _MockStreamEvent("text", text="Hello "),
            _MockStreamEvent("text", text="world"),
            _MockStreamEvent(
                "message_stop",
            ),
        ]
        final_msg = MockMessage(
            content=[MockTextBlock(text="Hello world")],
            usage=MockUsage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
        )
        mock_stream = _MockStream(events, final_msg)
        adapter._client.messages.stream = MagicMock(return_value=mock_stream)

        chunks = []
        async for chunk in adapter.stream(model="claude", messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        text_chunks = [ch for ch in chunks if ch.delta_content]
        assert text_chunks[0].delta_content == "Hello "
        assert text_chunks[1].delta_content == "world"

        final = [ch for ch in chunks if ch.finish_reason][0]
        assert final.finish_reason == "stop"
        assert final.input_tokens == 10
        assert final.output_tokens == 5

    @pytest.mark.asyncio
    async def test_streams_tool_use(self):
        adapter = _make_adapter()

        events = [
            _MockStreamEvent("text", text="Checking."),
            _MockStreamEvent(
                "content_block_start",
                index=1,
                content_block=_MockContentBlock("tool_use", id="t1", name="weather"),
            ),
            _MockStreamEvent("input_json", partial_json='{"city":'),
            _MockStreamEvent("input_json", partial_json=' "NYC"}'),
            _MockStreamEvent("message_stop"),
        ]
        final_msg = MockMessage(
            content=[MockTextBlock(text="Checking.")],
            stop_reason="tool_use",
            usage=MockUsage(input_tokens=20, output_tokens=15),
        )
        mock_stream = _MockStream(events, final_msg)
        adapter._client.messages.stream = MagicMock(return_value=mock_stream)

        chunks = []
        async for chunk in adapter.stream(model="claude", messages=[{"role": "user", "content": "weather?"}]):
            chunks.append(chunk)

        final = [ch for ch in chunks if ch.accumulated_tool_calls][0]
        assert len(final.accumulated_tool_calls) == 1
        assert final.accumulated_tool_calls[0].function_name == "weather"
        assert final.accumulated_tool_calls[0].arguments == {"city": "NYC"}
        assert final.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_stream_sdk_error_wrapped(self):
        adapter = _make_adapter()

        class _FailingStream:
            async def __aenter__(self):
                raise RuntimeError("stream failed")
            async def __aexit__(self, *args):
                pass

        adapter._client.messages.stream = MagicMock(return_value=_FailingStream())

        with pytest.raises(ProviderError, match="stream failed"):
            async for _ in adapter.stream(model="m", messages=[{"role": "user", "content": "test"}]):
                pass


class TestAnthropicAdapterEmbed:
    @pytest.mark.asyncio
    async def test_raises_not_supported(self):
        adapter = _make_adapter()
        with pytest.raises(ProviderError, match="does not provide"):
            await adapter.embed(model="m", texts=["hello"])


class TestAnthropicAdapterListModels:
    @pytest.mark.asyncio
    async def test_returns_model_info(self):
        adapter = _make_adapter()
        adapter._client.models.list = MagicMock(
            return_value=_MockAsyncIter([MockModelEntry(id="claude-3"), MockModelEntry(id="claude-4")])
        )

        models = await adapter.list_models()
        assert len(models) == 2
        assert models[0].model_id == "claude-3"
        assert models[0].provider == "anthropic"

    @pytest.mark.asyncio
    async def test_sdk_error_wrapped(self):
        adapter = _make_adapter()
        adapter._client.models.list = MagicMock(side_effect=RuntimeError("list failed"))

        with pytest.raises(ProviderError, match="list failed"):
            await adapter.list_models()
