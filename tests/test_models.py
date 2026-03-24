"""Tests for bridgellm.models data types."""

from bridgellm.models import (
    EmbeddingResponse,
    LLMResponse,
    ModelInfo,
    StreamChunk,
    ToolCall,
)


class TestToolCall:
    def test_creation(self):
        call = ToolCall(call_id="call_1", function_name="get_weather", arguments={"city": "NYC"})
        assert call.call_id == "call_1"
        assert call.function_name == "get_weather"
        assert call.arguments == {"city": "NYC"}

    def test_frozen(self):
        call = ToolCall(call_id="call_1", function_name="fn", arguments={})
        try:
            call.call_id = "changed"
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_equality(self):
        call_a = ToolCall(call_id="c1", function_name="fn", arguments={"key": "val"})
        call_b = ToolCall(call_id="c1", function_name="fn", arguments={"key": "val"})
        assert call_a == call_b


class TestLLMResponse:
    def test_defaults(self):
        response = LLMResponse()
        assert response.content is None
        assert response.tool_calls == []
        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.model == ""
        assert response.finish_reason == ""

    def test_with_content(self):
        response = LLMResponse(
            content="Hello",
            input_tokens=10,
            output_tokens=5,
            model="gpt-4o",
            finish_reason="stop",
        )
        assert response.content == "Hello"
        assert response.model == "gpt-4o"

    def test_with_tool_calls(self):
        calls = [ToolCall(call_id="c1", function_name="fn", arguments={})]
        response = LLMResponse(tool_calls=calls, finish_reason="tool_calls")
        assert len(response.tool_calls) == 1
        assert response.finish_reason == "tool_calls"


class TestStreamChunk:
    def test_text_chunk(self):
        chunk = StreamChunk(delta_content="Hello")
        assert chunk.delta_content == "Hello"
        assert chunk.finish_reason is None
        assert chunk.accumulated_tool_calls is None

    def test_final_chunk_with_tools(self):
        calls = [ToolCall(call_id="c1", function_name="fn", arguments={})]
        chunk = StreamChunk(finish_reason="tool_calls", accumulated_tool_calls=calls)
        assert chunk.finish_reason == "tool_calls"
        assert len(chunk.accumulated_tool_calls) == 1

    def test_usage_only_chunk(self):
        chunk = StreamChunk(input_tokens=100, output_tokens=50)
        assert chunk.input_tokens == 100
        assert chunk.output_tokens == 50
        assert chunk.delta_content is None


class TestEmbeddingResponse:
    def test_creation(self):
        response = EmbeddingResponse(
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            model="text-embedding-3-small",
            input_tokens=20,
        )
        assert len(response.vectors) == 2
        assert response.model == "text-embedding-3-small"
        assert response.input_tokens == 20

    def test_frozen(self):
        response = EmbeddingResponse(vectors=[], model="m")
        try:
            response.model = "changed"
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestModelInfo:
    def test_creation(self):
        info = ModelInfo(model_id="gpt-4o", provider="openai", owned_by="openai", created=1000)
        assert info.model_id == "gpt-4o"
        assert info.provider == "openai"

    def test_defaults(self):
        info = ModelInfo(model_id="m1", provider="test")
        assert info.owned_by == ""
        assert info.created == 0
