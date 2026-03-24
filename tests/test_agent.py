"""Tests for AgentLoop — iteration control, tool execution, events, and stop conditions."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from bridgellm.agent import AgentEvent, AgentLoop, RetryPolicy
from bridgellm.client import BridgeLLM
from bridgellm.models import LLMResponse, StreamChunk, ToolCall
from bridgellm.tools import ToolDefinition, tool


@tool
async def echo_tool(message: str) -> str:
    """Echo back the message."""
    return f"Echo: {message}"


@tool
async def failing_tool(value: str) -> str:
    """A tool that always fails."""
    raise RuntimeError("Tool crashed")


def _make_llm(openai_api_key):
    """Create a BridgeLLM with mocked adapter for testing."""
    return BridgeLLM(model="openai/gpt-4o")


class TestAgentLoopBasic:
    @pytest.mark.asyncio
    async def test_no_tool_calls_stops_immediately(self, openai_api_key):
        llm = _make_llm(openai_api_key)
        llm._primary_adapter.complete = AsyncMock(return_value=LLMResponse(
            content="Hello!", tool_calls=[], finish_reason="stop",
            input_tokens=10, output_tokens=5,
        ))

        agent = AgentLoop(llm=llm, tools=[echo_tool], streaming=False)
        events = []
        async for event in agent.run(messages=[{"role": "user", "content": "Hi"}]):
            events.append(event)

        types = [ev.type for ev in events]
        assert "iteration_start" in types
        assert "usage" in types
        assert "done" in types
        done = [ev for ev in events if ev.type == "done"][0]
        assert done.finish_reason == "stop"
        assert done.total_iterations == 1

    @pytest.mark.asyncio
    async def test_tool_call_then_response(self, openai_api_key):
        llm = _make_llm(openai_api_key)
        call_count = 0

        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content=None, finish_reason="tool_calls",
                    tool_calls=[ToolCall(call_id="c1", function_name="echo_tool", arguments={"message": "test"})],
                    input_tokens=20, output_tokens=10,
                )
            return LLMResponse(content="Done!", tool_calls=[], finish_reason="stop",
                               input_tokens=15, output_tokens=8)

        llm._primary_adapter.complete = mock_complete

        agent = AgentLoop(llm=llm, tools=[echo_tool], streaming=False)
        events = []
        async for event in agent.run(messages=[{"role": "user", "content": "test"}]):
            events.append(event)

        types = [ev.type for ev in events]
        assert "tool_start" in types
        assert "tool_result" in types
        assert "done" in types
        assert call_count == 2

        tool_result = [ev for ev in events if ev.type == "tool_result"][0]
        assert "Echo: test" in tool_result.content

    @pytest.mark.asyncio
    async def test_max_iterations_stops_loop(self, openai_api_key):
        llm = _make_llm(openai_api_key)

        # Always returns tool calls — should hit max_iterations
        llm._primary_adapter.complete = AsyncMock(return_value=LLMResponse(
            content=None, finish_reason="tool_calls",
            tool_calls=[ToolCall(call_id="c1", function_name="echo_tool", arguments={"message": "loop"})],
            input_tokens=10, output_tokens=5,
        ))

        agent = AgentLoop(llm=llm, tools=[echo_tool], max_iterations=3, streaming=False)
        events = []
        async for event in agent.run(messages=[{"role": "user", "content": "test"}]):
            events.append(event)

        done = [ev for ev in events if ev.type == "done"][0]
        assert done.finish_reason == "max_iterations"
        assert done.total_iterations == 3


class TestAgentToolErrors:
    @pytest.mark.asyncio
    async def test_tool_error_skip_continues(self, openai_api_key):
        llm = _make_llm(openai_api_key)
        call_count = 0

        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content=None, finish_reason="tool_calls",
                    tool_calls=[ToolCall(call_id="c1", function_name="failing_tool", arguments={"value": "x"})],
                    input_tokens=10, output_tokens=5,
                )
            return LLMResponse(content="Recovered", tool_calls=[], finish_reason="stop",
                               input_tokens=10, output_tokens=5)

        llm._primary_adapter.complete = mock_complete

        agent = AgentLoop(llm=llm, tools=[failing_tool], on_tool_error="skip", streaming=False)
        events = []
        async for event in agent.run(messages=[{"role": "user", "content": "test"}]):
            events.append(event)

        types = [ev.type for ev in events]
        assert "tool_error" in types
        assert "done" in types

    @pytest.mark.asyncio
    async def test_tool_error_stop_ends_loop(self, openai_api_key):
        llm = _make_llm(openai_api_key)
        llm._primary_adapter.complete = AsyncMock(return_value=LLMResponse(
            content=None, finish_reason="tool_calls",
            tool_calls=[ToolCall(call_id="c1", function_name="failing_tool", arguments={"value": "x"})],
            input_tokens=10, output_tokens=5,
        ))

        agent = AgentLoop(llm=llm, tools=[failing_tool], on_tool_error="stop", streaming=False)
        events = []
        async for event in agent.run(messages=[{"role": "user", "content": "test"}]):
            events.append(event)

        done = [ev for ev in events if ev.type == "done"][0]
        assert done.finish_reason == "tool_error"


class TestAgentTimeout:
    @pytest.mark.asyncio
    async def test_timeout_stops_loop(self, openai_api_key):
        llm = _make_llm(openai_api_key)

        async def slow_complete(**kwargs):
            await asyncio.sleep(0.5)
            return LLMResponse(
                content=None, finish_reason="tool_calls",
                tool_calls=[ToolCall(call_id="c1", function_name="echo_tool", arguments={"message": "x"})],
                input_tokens=10, output_tokens=5,
            )

        llm._primary_adapter.complete = slow_complete

        agent = AgentLoop(llm=llm, tools=[echo_tool], timeout_seconds=0.3, streaming=False)
        events = []
        async for event in agent.run(messages=[{"role": "user", "content": "test"}]):
            events.append(event)

        done_events = [ev for ev in events if ev.type == "done"]
        assert len(done_events) == 1
        assert done_events[0].finish_reason in ("timeout", "stop")


class TestAgentTokenLimit:
    @pytest.mark.asyncio
    async def test_max_tokens_stops_loop(self, openai_api_key):
        llm = _make_llm(openai_api_key)
        call_count = 0

        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return LLMResponse(
                    content=None, finish_reason="tool_calls",
                    tool_calls=[ToolCall(call_id=f"c{call_count}", function_name="echo_tool", arguments={"message": "x"})],
                    input_tokens=5000, output_tokens=2000,
                )
            return LLMResponse(content="Done", tool_calls=[], finish_reason="stop",
                               input_tokens=100, output_tokens=50)

        llm._primary_adapter.complete = mock_complete

        agent = AgentLoop(llm=llm, tools=[echo_tool], max_total_tokens=10000, streaming=False)
        events = []
        async for event in agent.run(messages=[{"role": "user", "content": "test"}]):
            events.append(event)

        done = [ev for ev in events if ev.type == "done"][0]
        assert done.finish_reason == "max_tokens"


class TestRetryPolicy:
    def test_default_retryable(self):
        policy = RetryPolicy()
        assert policy.is_retryable(Exception("rate_limit error")) is True
        assert policy.is_retryable(Exception("429 too many requests")) is True
        assert policy.is_retryable(Exception("timeout")) is True
        assert policy.is_retryable(Exception("invalid request")) is False

    def test_custom_retryable(self):
        policy = RetryPolicy(retryable_check=lambda exc: "custom" in str(exc))
        assert policy.is_retryable(Exception("custom error")) is True
        assert policy.is_retryable(Exception("other error")) is False


class TestAgentStreaming:
    @pytest.mark.asyncio
    async def test_streaming_mode(self, openai_api_key):
        llm = _make_llm(openai_api_key)

        async def mock_stream(**kwargs):
            yield StreamChunk(delta_content="Hello ")
            yield StreamChunk(delta_content="world")
            yield StreamChunk(finish_reason="stop")
            yield StreamChunk(input_tokens=10, output_tokens=5)

        llm._primary_adapter.stream = mock_stream

        agent = AgentLoop(llm=llm, tools=[echo_tool], streaming=True)
        events = []
        async for event in agent.run(messages=[{"role": "user", "content": "Hi"}]):
            events.append(event)

        done = [ev for ev in events if ev.type == "done"][0]
        assert done.finish_reason == "stop"
        assert done.total_input_tokens == 10


class TestAgentParallelTools:
    @pytest.mark.asyncio
    async def test_parallel_execution(self, openai_api_key):
        @tool
        async def slow_tool(value: str) -> str:
            """Slow tool."""
            await asyncio.sleep(0.05)
            return f"done:{value}"

        llm = _make_llm(openai_api_key)
        call_count = 0

        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content=None, finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(call_id="c1", function_name="slow_tool", arguments={"value": "a"}),
                        ToolCall(call_id="c2", function_name="slow_tool", arguments={"value": "b"}),
                    ],
                    input_tokens=10, output_tokens=5,
                )
            return LLMResponse(content="Done", tool_calls=[], finish_reason="stop",
                               input_tokens=10, output_tokens=5)

        llm._primary_adapter.complete = mock_complete

        agent = AgentLoop(llm=llm, tools=[slow_tool], parallel_tool_calls=True, streaming=False)
        events = []
        async for event in agent.run(messages=[{"role": "user", "content": "test"}]):
            events.append(event)

        results = [ev for ev in events if ev.type == "tool_result"]
        assert len(results) == 2


class TestAgentContext:
    @pytest.mark.asyncio
    async def test_context_passed_to_tools(self, openai_api_key):
        @tool
        async def ctx_tool(name: str, context: dict = None) -> str:
            return f"user={context['user_id']}"

        llm = _make_llm(openai_api_key)
        call_count = 0

        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content=None, finish_reason="tool_calls",
                    tool_calls=[ToolCall(call_id="c1", function_name="ctx_tool", arguments={"name": "test"})],
                    input_tokens=10, output_tokens=5,
                )
            return LLMResponse(content="Done", tool_calls=[], finish_reason="stop",
                               input_tokens=10, output_tokens=5)

        llm._primary_adapter.complete = mock_complete

        agent = AgentLoop(
            llm=llm, tools=[ctx_tool],
            context={"user_id": "u123"},
            streaming=False,
        )
        events = []
        async for event in agent.run(messages=[{"role": "user", "content": "test"}]):
            events.append(event)

        result = [ev for ev in events if ev.type == "tool_result"][0]
        assert "user=u123" in result.content
