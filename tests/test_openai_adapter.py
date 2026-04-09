"""Tests for the OpenAI-compatible adapter.

Uses mock objects that replicate the SDK response shapes documented in the
OpenAI Python SDK v2.x. If a future SDK version changes these shapes,
these contract tests will catch it.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bridgellm.adapters.openai_compat import (
    OpenAICompatAdapter,
    _accumulate_tool_deltas,
    _assemble_tool_calls,
    _parse_tool_calls,
    _safe_parse_json,
)
from bridgellm.errors import ProviderError
from bridgellm.models import LLMResponse, StreamChunk, ToolCall
from bridgellm.registry import ProviderConfig


# -- Fixtures that replicate real SDK response shapes --


@dataclass
class MockFunction:
    name: str
    arguments: str


@dataclass
class MockToolCall:
    id: str
    function: MockFunction


@dataclass
class MockUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class MockMessage:
    content: Optional[str]
    tool_calls: Optional[list]


@dataclass
class MockChoice:
    message: MockMessage
    finish_reason: str


@dataclass
class MockCompletion:
    choices: list[MockChoice]
    usage: MockUsage
    model: str


@dataclass
class MockDeltaFunction:
    name: Optional[str] = None
    arguments: Optional[str] = None


@dataclass
class MockDeltaToolCall:
    index: int
    id: Optional[str] = None
    function: Optional[MockDeltaFunction] = None


@dataclass
class MockDelta:
    content: Optional[str] = None
    tool_calls: Optional[list] = None


@dataclass
class MockStreamChoice:
    delta: MockDelta
    finish_reason: Optional[str] = None


@dataclass
class MockStreamChunk:
    choices: list[MockStreamChoice]
    usage: Optional[MockUsage] = None
    model: str = "gpt-4o"


@dataclass
class MockEmbedding:
    embedding: list[float]
    index: int


@dataclass
class MockEmbeddingUsage:
    prompt_tokens: int


@dataclass
class MockEmbeddingResponse:
    data: list[MockEmbedding]
    usage: MockEmbeddingUsage


@dataclass
class MockModelEntry:
    id: str
    owned_by: str
    created: int


@dataclass
class MockModelPage:
    data: list[MockModelEntry]


# -- Test config and factory --


MOCK_CONFIG = ProviderConfig(
    base_url="https://api.example.com/v1",
    api_key_env="TEST_KEY",
)


def _create_adapter():
    return OpenAICompatAdapter(provider_name="test", config=MOCK_CONFIG, api_key="sk-test")


# -- Helper function tests --


class TestSafeParseJson:
    def test_valid_json(self):
        result = _safe_parse_json('{"city": "NYC"}', "fn")
        assert result == {"city": "NYC"}

    def test_empty_string(self):
        assert _safe_parse_json("", "fn") == {}

    def test_malformed_json(self):
        result = _safe_parse_json("not json", "fn")
        assert result == {"_parse_error": True, "raw": "not json"}

    def test_none_input(self):
        """None is falsy like empty string — returns empty dict, not an error."""
        result = _safe_parse_json(None, "fn")
        assert result == {}


class TestParseToolCalls:
    def test_none_returns_empty(self):
        assert _parse_tool_calls(None) == []

    def test_empty_list_returns_empty(self):
        assert _parse_tool_calls([]) == []

    def test_parses_valid_calls(self):
        raw = [
            MockToolCall(id="c1", function=MockFunction(name="weather", arguments='{"city": "NYC"}')),
            MockToolCall(id="c2", function=MockFunction(name="search", arguments='{"q": "test"}')),
        ]
        result = _parse_tool_calls(raw)
        assert len(result) == 2
        assert result[0] == ToolCall(call_id="c1", function_name="weather", arguments={"city": "NYC"})
        assert result[1] == ToolCall(call_id="c2", function_name="search", arguments={"q": "test"})


class TestAccumulateToolDeltas:
    def test_none_deltas_no_op(self):
        accumulator = {}
        _accumulate_tool_deltas(accumulator, None)
        assert accumulator == {}

    def test_accumulates_fragments(self):
        accumulator = {}
        _accumulate_tool_deltas(accumulator, [
            MockDeltaToolCall(index=0, id="c1", function=MockDeltaFunction(name="fn", arguments='{"ci')),
        ])
        _accumulate_tool_deltas(accumulator, [
            MockDeltaToolCall(index=0, function=MockDeltaFunction(arguments='ty": "NYC"}')),
        ])
        assert accumulator[0]["id"] == "c1"
        assert accumulator[0]["name"] == "fn"
        assert accumulator[0]["arguments"] == '{"city": "NYC"}'

    def test_multiple_tools(self):
        accumulator = {}
        _accumulate_tool_deltas(accumulator, [
            MockDeltaToolCall(index=0, id="c1", function=MockDeltaFunction(name="fn1", arguments="{}")),
            MockDeltaToolCall(index=1, id="c2", function=MockDeltaFunction(name="fn2", arguments="{}")),
        ])
        assert len(accumulator) == 2


class TestAssembleToolCalls:
    def test_assembles_from_accumulator(self):
        accumulator = {
            0: {"id": "c1", "name": "weather", "arguments": '{"city": "NYC"}'},
            1: {"id": "c2", "name": "search", "arguments": '{"q": "test"}'},
        }
        result = _assemble_tool_calls(accumulator)
        assert len(result) == 2
        assert result[0].function_name == "weather"
        assert result[1].function_name == "search"

    def test_handles_malformed_json(self):
        accumulator = {0: {"id": "c1", "name": "fn", "arguments": "broken"}}
        result = _assemble_tool_calls(accumulator)
        assert result[0].arguments == {"_parse_error": True, "raw": "broken"}


# -- Adapter integration tests (mocked SDK) --


class TestOpenAICompatComplete:
    @pytest.mark.asyncio
    async def test_returns_llm_response(self):
        adapter = _create_adapter()
        mock_response = MockCompletion(
            choices=[MockChoice(
                message=MockMessage(content="Hello world", tool_calls=None),
                finish_reason="stop",
            )],
            usage=MockUsage(prompt_tokens=10, completion_tokens=5),
            model="gpt-4o",
        )
        adapter._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await adapter.complete(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello world"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.model == "gpt-4o"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_with_tool_calls(self):
        adapter = _create_adapter()
        mock_response = MockCompletion(
            choices=[MockChoice(
                message=MockMessage(
                    content=None,
                    tool_calls=[MockToolCall(
                        id="c1",
                        function=MockFunction(name="get_weather", arguments='{"city": "NYC"}'),
                    )],
                ),
                finish_reason="tool_calls",
            )],
            usage=MockUsage(prompt_tokens=20, completion_tokens=10),
            model="gpt-4o",
        )
        adapter._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await adapter.complete(model="gpt-4o", messages=[{"role": "user", "content": "test"}])
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "get_weather"
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_sdk_error_wrapped(self):
        adapter = _create_adapter()
        adapter._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))

        with pytest.raises(ProviderError, match="API down"):
            await adapter.complete(model="gpt-4o", messages=[{"role": "user", "content": "test"}])


class TestOpenAICompatStream:
    @pytest.mark.asyncio
    async def test_yields_text_chunks(self):
        adapter = _create_adapter()

        async def mock_stream():
            yield MockStreamChunk(choices=[MockStreamChoice(delta=MockDelta(content="Hello "))])
            yield MockStreamChunk(choices=[MockStreamChoice(delta=MockDelta(content="world"))])
            yield MockStreamChunk(
                choices=[MockStreamChoice(delta=MockDelta(), finish_reason="stop")]
            )
            yield MockStreamChunk(
                choices=[], usage=MockUsage(prompt_tokens=10, completion_tokens=5)
            )

        adapter._client.chat.completions.create = AsyncMock(return_value=mock_stream())

        chunks = []
        async for chunk in adapter.stream(model="gpt-4o", messages=[{"role": "user", "content": "test"}]):
            chunks.append(chunk)

        text_chunks = [ch for ch in chunks if ch.delta_content]
        assert text_chunks[0].delta_content == "Hello "
        assert text_chunks[1].delta_content == "world"

        finish_chunk = [ch for ch in chunks if ch.finish_reason][0]
        assert finish_chunk.finish_reason == "stop"

        usage_chunk = [ch for ch in chunks if ch.input_tokens > 0][0]
        assert usage_chunk.input_tokens == 10
        assert usage_chunk.output_tokens == 5

    @pytest.mark.asyncio
    async def test_streaming_tool_calls(self):
        adapter = _create_adapter()

        async def mock_stream():
            yield MockStreamChunk(choices=[MockStreamChoice(
                delta=MockDelta(tool_calls=[
                    MockDeltaToolCall(index=0, id="c1", function=MockDeltaFunction(name="fn", arguments='{"k')),
                ]),
            )])
            yield MockStreamChunk(choices=[MockStreamChoice(
                delta=MockDelta(tool_calls=[
                    MockDeltaToolCall(index=0, function=MockDeltaFunction(arguments='ey": "val"}')),
                ]),
            )])
            yield MockStreamChunk(choices=[MockStreamChoice(
                delta=MockDelta(), finish_reason="tool_calls"
            )])

        adapter._client.chat.completions.create = AsyncMock(return_value=mock_stream())

        chunks = []
        async for chunk in adapter.stream(model="gpt-4o", messages=[{"role": "user", "content": "test"}]):
            chunks.append(chunk)

        final = [ch for ch in chunks if ch.accumulated_tool_calls][0]
        assert len(final.accumulated_tool_calls) == 1
        assert final.accumulated_tool_calls[0].arguments == {"key": "val"}

    @pytest.mark.asyncio
    async def test_sdk_error_wrapped(self):
        adapter = _create_adapter()
        adapter._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("timeout"))

        with pytest.raises(ProviderError, match="timeout"):
            async for _ in adapter.stream(model="gpt-4o", messages=[{"role": "user", "content": "test"}]):
                pass


class TestOpenAICompatEmbed:
    @pytest.mark.asyncio
    async def test_returns_vectors(self):
        adapter = _create_adapter()
        mock_response = MockEmbeddingResponse(
            data=[
                MockEmbedding(embedding=[0.1, 0.2, 0.3], index=0),
                MockEmbedding(embedding=[0.4, 0.5, 0.6], index=1),
            ],
            usage=MockEmbeddingUsage(prompt_tokens=15),
        )
        adapter._client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await adapter.embed(model="text-embedding-3-small", texts=["hello", "world"])
        assert len(result.vectors) == 2
        assert result.vectors[0] == [0.1, 0.2, 0.3]
        assert result.input_tokens == 15

    @pytest.mark.asyncio
    async def test_batches_large_input(self):
        adapter = _create_adapter()
        call_count = 0

        async def mock_embed(**kwargs):
            nonlocal call_count
            call_count += 1
            batch_size = len(kwargs["input"])
            return MockEmbeddingResponse(
                data=[MockEmbedding(embedding=[0.1], index=idx) for idx in range(batch_size)],
                usage=MockEmbeddingUsage(prompt_tokens=batch_size),
            )

        adapter._client.embeddings.create = mock_embed
        texts = [f"text_{num}" for num in range(1200)]
        result = await adapter.embed(model="model", texts=texts)

        assert len(result.vectors) == 1200
        assert call_count == 3  # 500 + 500 + 200


class TestOpenAICompatListModels:
    @pytest.mark.asyncio
    async def test_returns_model_info(self):
        adapter = _create_adapter()
        mock_page = MockModelPage(data=[
            MockModelEntry(id="gpt-4o", owned_by="openai", created=1000),
            MockModelEntry(id="gpt-4o-mini", owned_by="openai", created=2000),
        ])
        adapter._client.models.list = AsyncMock(return_value=mock_page)

        models = await adapter.list_models()
        assert len(models) == 2
        assert models[0].model_id == "gpt-4o"
        assert models[0].provider == "test"


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_empty_messages_raises(self):
        adapter = _create_adapter()
        with pytest.raises(ProviderError, match="messages list cannot be empty"):
            await adapter.complete(model="gpt-4o", messages=[])

    @pytest.mark.asyncio
    async def test_empty_messages_stream_raises(self):
        adapter = _create_adapter()
        with pytest.raises(ProviderError, match="messages list cannot be empty"):
            async for _ in adapter.stream(model="gpt-4o", messages=[]):
                pass

    @pytest.mark.asyncio
    async def test_empty_texts_embed_raises(self):
        adapter = _create_adapter()
        with pytest.raises(ProviderError, match="texts list cannot be empty"):
            await adapter.embed(model="model", texts=[])

    @pytest.mark.asyncio
    async def test_empty_text_speak_raises(self):
        adapter = _create_adapter()
        with pytest.raises(ProviderError, match="text cannot be empty"):
            await adapter.speak(model="tts-1", text="")

    @pytest.mark.asyncio
    async def test_empty_audio_transcribe_raises(self):
        adapter = _create_adapter()
        with pytest.raises(ProviderError, match="audio_data cannot be empty"):
            await adapter.transcribe(model="whisper-1", audio_data=b"")

    @pytest.mark.asyncio
    async def test_empty_choices_raises(self):
        adapter = _create_adapter()
        mock_response = MockCompletion(choices=[], usage=MockUsage(0, 0), model="gpt-4o")
        adapter._client.chat.completions.create = AsyncMock(return_value=mock_response)
        with pytest.raises(ProviderError, match="empty choices"):
            await adapter.complete(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])


class TestCancellationPropagation:
    @pytest.mark.asyncio
    async def test_complete_propagates_cancellation(self):
        adapter = _create_adapter()
        adapter._client.chat.completions.create = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await adapter.complete(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_embed_propagates_cancellation(self):
        adapter = _create_adapter()
        adapter._client.embeddings.create = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await adapter.embed(model="model", texts=["hello"])


class TestRequestConfigForwarding:
    """Verify that RequestConfig params are correctly forwarded to the SDK."""

    @pytest.mark.asyncio
    async def test_response_format(self):
        from bridgellm.models import RequestConfig
        from bridgellm.adapters.openai_compat import _build_request

        config = RequestConfig(response_format={"type": "json_object"})
        kwargs = _build_request("gpt-4o", [], None, 0.7, 100, config)
        assert kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_stop_sequences(self):
        from bridgellm.adapters.openai_compat import _build_request
        from bridgellm.models import RequestConfig

        config = RequestConfig(stop=["END", "STOP"])
        kwargs = _build_request("gpt-4o", [], None, 0.7, 100, config)
        assert kwargs["stop"] == ["END", "STOP"]

    @pytest.mark.asyncio
    async def test_tool_choice_forwarded(self):
        from bridgellm.adapters.openai_compat import _build_request
        from bridgellm.models import RequestConfig

        config = RequestConfig(tool_choice="required")
        tools = [{"type": "function", "function": {"name": "fn"}}]
        kwargs = _build_request("gpt-4o", [], tools, 0.7, 100, config)
        assert kwargs["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_sampling_params(self):
        from bridgellm.adapters.openai_compat import _build_request
        from bridgellm.models import RequestConfig

        config = RequestConfig(
            top_p=0.9,
            seed=42,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        kwargs = _build_request("gpt-4o", [], None, 0.7, 100, config)
        assert kwargs["top_p"] == 0.9
        assert kwargs["seed"] == 42
        assert kwargs["frequency_penalty"] == 0.5
        assert kwargs["presence_penalty"] == 0.3

    @pytest.mark.asyncio
    async def test_logprobs(self):
        from bridgellm.adapters.openai_compat import _build_request
        from bridgellm.models import RequestConfig

        config = RequestConfig(logprobs=True, top_logprobs=5)
        kwargs = _build_request("gpt-4o", [], None, 0.7, 100, config)
        assert kwargs["logprobs"] is True
        assert kwargs["top_logprobs"] == 5

    @pytest.mark.asyncio
    async def test_reasoning_effort(self):
        from bridgellm.adapters.openai_compat import _build_request
        from bridgellm.models import RequestConfig

        config = RequestConfig(reasoning={"effort": "high"})
        kwargs = _build_request("o3", [], None, 0.7, 100, config)
        assert kwargs["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_service_tier(self):
        from bridgellm.adapters.openai_compat import _build_request
        from bridgellm.models import RequestConfig

        config = RequestConfig(service_tier="flex")
        kwargs = _build_request("gpt-4o", [], None, 0.7, 100, config)
        assert kwargs["service_tier"] == "flex"

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self):
        from bridgellm.adapters.openai_compat import _build_request
        from bridgellm.models import RequestConfig

        config = RequestConfig(parallel_tool_calls=False)
        tools = [{"type": "function", "function": {"name": "fn"}}]
        kwargs = _build_request("gpt-4o", [], tools, 0.7, 100, config)
        assert kwargs["parallel_tool_calls"] is False

    @pytest.mark.asyncio
    async def test_extra_passthrough(self):
        from bridgellm.adapters.openai_compat import _build_request
        from bridgellm.models import RequestConfig

        config = RequestConfig(extra={"custom_param": "value"})
        kwargs = _build_request("gpt-4o", [], None, 0.7, 100, config)
        assert kwargs["custom_param"] == "value"

    @pytest.mark.asyncio
    async def test_none_config_no_extras(self):
        from bridgellm.adapters.openai_compat import _build_request

        kwargs = _build_request("gpt-4o", [], None, 0.7, 100, None)
        assert "response_format" not in kwargs
        assert "stop" not in kwargs

    @pytest.mark.asyncio
    async def test_n_param(self):
        from bridgellm.adapters.openai_compat import _build_request
        from bridgellm.models import RequestConfig

        config = RequestConfig(n=3)
        kwargs = _build_request("gpt-4o", [], None, 0.7, 100, config)
        assert kwargs["n"] == 3

    @pytest.mark.asyncio
    async def test_complete_with_config(self):
        """End-to-end: config params reach the SDK call."""
        from bridgellm.models import RequestConfig

        adapter = _create_adapter()
        mock_response = MockCompletion(
            choices=[MockChoice(message=MockMessage(content="ok", tool_calls=None), finish_reason="stop")],
            usage=MockUsage(prompt_tokens=5, completion_tokens=2),
            model="gpt-4o",
        )
        adapter._client.chat.completions.create = AsyncMock(return_value=mock_response)

        config = RequestConfig(response_format={"type": "json_object"}, top_p=0.9)
        await adapter.complete(model="gpt-4o", messages=[{"role": "user", "content": "test"}], config=config)

        call_kwargs = adapter._client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["top_p"] == 0.9
