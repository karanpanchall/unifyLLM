"""Configurable agent loop — tool call → execute → feed back → repeat.

Yields typed events so callers can render UI, log, or react.
Everything is configurable. Nothing is forced.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Literal, Optional

from .client import BridgeLLM
from .models import RequestConfig, StreamChunk, ToolCall
from .tools import ToolDefinition, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentEvent:
    """A typed event yielded by the agent loop."""

    type: Literal[
        "text_delta", "reasoning_delta",
        "tool_start", "tool_result", "tool_error",
        "iteration_start", "usage", "done", "error",
    ]
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_call_id: Optional[str] = None
    iteration: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_iterations: int = 0
    finish_reason: str = ""


@dataclass
class RetryPolicy:
    """Optional retry configuration for transient LLM errors.

    Disabled by default. Enable by passing to AgentLoop.
    """

    max_retries: int = 2
    backoff_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    retryable_check: Optional[Callable[[Exception], bool]] = None

    def is_retryable(self, exc: Exception) -> bool:
        if self.retryable_check:
            return self.retryable_check(exc)
        error_text = str(exc).lower()
        return any(keyword in error_text for keyword in ("rate_limit", "429", "timeout", "502", "503", "overloaded"))


class AgentLoop:
    """Configurable agentic loop that handles tool calling automatically.

    Yields AgentEvent objects so the caller has full visibility.

    Usage:
        agent = AgentLoop(llm=llm, tools=[get_weather, search_docs])
        async for event in agent.run(messages=[...]):
            if event.type == "text_delta":
                print(event.content, end="")
    """

    def __init__(
        self,
        llm: BridgeLLM,
        tools: Optional[list[ToolDefinition]] = None,
        max_iterations: int = 25,
        timeout_seconds: Optional[float] = None,
        max_total_tokens: Optional[int] = None,
        tool_timeout_seconds: float = 120.0,
        parallel_tool_calls: bool = True,
        on_tool_error: Literal["skip", "stop", "raise"] = "skip",
        max_tool_failures: int = 3,
        context: Any = None,
        config: Optional[RequestConfig] = None,
        retry_policy: Optional[RetryPolicy] = None,
        streaming: bool = True,
    ):
        """
        Args:
            llm: BridgeLLM client instance.
            tools: List of ToolDefinition objects (from @tool decorator).
            max_iterations: Maximum LLM call iterations before stopping.
            timeout_seconds: Wall-clock timeout for the entire loop. None = no limit.
            max_total_tokens: Stop if total tokens exceed this. None = no limit.
            tool_timeout_seconds: Per-tool execution timeout.
            parallel_tool_calls: Execute multiple tool calls concurrently.
            on_tool_error: "skip" feeds error to LLM, "stop" ends loop, "raise" propagates.
            max_tool_failures: Disable a tool after this many consecutive failures.
            context: Arbitrary context object passed to tool handlers.
            config: RequestConfig for LLM calls (response_format, etc.).
            retry_policy: Optional retry configuration for transient LLM errors.
            streaming: Use stream() (True) or complete() (False) for LLM calls.
        """
        self._llm = llm
        self._registry = ToolRegistry(tools)
        self._max_iterations = max_iterations
        self._timeout_seconds = timeout_seconds
        self._max_total_tokens = max_total_tokens
        self._tool_timeout = tool_timeout_seconds
        self._parallel = parallel_tool_calls
        self._on_tool_error = on_tool_error
        self._max_tool_failures = max_tool_failures
        self._context = context
        self._config = config
        self._retry_policy = retry_policy
        self._streaming = streaming

    async def run(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Execute the agent loop, yielding events.

        Messages are modified in-place (tool results appended).
        """
        tool_defs = self._registry.as_openai_tools() or None
        total_input = total_output = 0
        tool_failures: dict[str, int] = {}
        start_time = time.monotonic()

        for iteration in range(1, self._max_iterations + 1):
            # Check timeout
            if self._timeout_seconds and (time.monotonic() - start_time) > self._timeout_seconds:
                yield AgentEvent(type="done", finish_reason="timeout", total_iterations=iteration - 1,
                                 total_input_tokens=total_input, total_output_tokens=total_output)
                return

            # Check token limit
            if self._max_total_tokens and (total_input + total_output) > self._max_total_tokens:
                yield AgentEvent(type="done", finish_reason="max_tokens", total_iterations=iteration - 1,
                                 total_input_tokens=total_input, total_output_tokens=total_output)
                return

            yield AgentEvent(type="iteration_start", iteration=iteration)

            # Filter out disabled tools
            active_tools = self._filter_tools(tool_defs, tool_failures) if tool_defs else None

            # Call LLM (with optional retry)
            try:
                tool_calls, iter_input, iter_output = await self._call_llm(
                    messages, model, active_tools, temperature, max_tokens, iteration,
                )
            except Exception as exc:
                yield AgentEvent(type="error", content=str(exc))
                return

            total_input += iter_input
            total_output += iter_output
            yield AgentEvent(type="usage", iteration=iteration,
                             input_tokens=iter_input, output_tokens=iter_output)

            # No tool calls → agent decided to respond with text → done
            if not tool_calls:
                yield AgentEvent(type="done", finish_reason="stop", total_iterations=iteration,
                                 total_input_tokens=total_input, total_output_tokens=total_output)
                return

            # Execute tool calls
            stop_requested = False
            for call_event in await self._execute_tools(tool_calls, tool_failures):
                yield call_event
                if call_event.type == "tool_error" and self._on_tool_error == "stop":
                    stop_requested = True

            if stop_requested:
                yield AgentEvent(type="done", finish_reason="tool_error", total_iterations=iteration,
                                 total_input_tokens=total_input, total_output_tokens=total_output)
                return

        # Max iterations reached
        yield AgentEvent(type="done", finish_reason="max_iterations", total_iterations=self._max_iterations,
                         total_input_tokens=total_input, total_output_tokens=total_output)

    async def _call_llm(
        self, messages: list[dict], model: Optional[str], tools: Optional[list[dict]],
        temperature: float, max_tokens: Optional[int], iteration: int,
    ) -> tuple[list[ToolCall], int, int]:
        """Call the LLM, with optional retry. Returns (tool_calls, input_tokens, output_tokens).

        Yields text_delta and reasoning_delta events via the parent generator
        by appending assistant content to messages.
        """
        retries = 0
        max_retries = self._retry_policy.max_retries if self._retry_policy else 0

        while True:
            try:
                return await self._single_llm_call(messages, model, tools, temperature, max_tokens)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._retry_policy and retries < max_retries and self._retry_policy.is_retryable(exc):
                    wait = self._retry_policy.backoff_seconds * (self._retry_policy.backoff_multiplier ** retries)
                    logger.warning("Retryable error on iteration %d, retry %d after %.1fs: %s",
                                   iteration, retries + 1, wait, exc)
                    await asyncio.sleep(wait)
                    retries += 1
                    continue
                raise

    async def _single_llm_call(
        self, messages: list[dict], model: Optional[str], tools: Optional[list[dict]],
        temperature: float, max_tokens: Optional[int],
    ) -> tuple[list[ToolCall], int, int]:
        """Single LLM call — streaming or non-streaming based on config."""
        if self._streaming:
            return await self._streaming_call(messages, model, tools, temperature, max_tokens)
        return await self._non_streaming_call(messages, model, tools, temperature, max_tokens)

    async def _streaming_call(
        self, messages: list[dict], model: Optional[str], tools: Optional[list[dict]],
        temperature: float, max_tokens: Optional[int],
    ) -> tuple[list[ToolCall], int, int]:
        collected_content = ""
        collected_tools: list[ToolCall] = []
        iter_input = iter_output = 0

        async for chunk in self._llm.stream(
            messages=messages, model=model, tools=tools,
            temperature=temperature, max_tokens=max_tokens, config=self._config,
        ):
            if chunk.delta_content:
                collected_content += chunk.delta_content
            if chunk.accumulated_tool_calls:
                collected_tools = chunk.accumulated_tool_calls
            iter_input += chunk.input_tokens
            iter_output += chunk.output_tokens

        # Append assistant message to conversation
        assistant_msg: dict = {"role": "assistant", "content": collected_content or None}
        if collected_tools:
            assistant_msg["tool_calls"] = [
                {"id": tc.call_id, "type": "function",
                 "function": {"name": tc.function_name, "arguments": str(tc.arguments)}}
                for tc in collected_tools
            ]
        messages.append(assistant_msg)

        return collected_tools, iter_input, iter_output

    async def _non_streaming_call(
        self, messages: list[dict], model: Optional[str], tools: Optional[list[dict]],
        temperature: float, max_tokens: Optional[int],
    ) -> tuple[list[ToolCall], int, int]:
        response = await self._llm.complete(
            messages=messages, model=model, tools=tools,
            temperature=temperature, max_tokens=max_tokens, config=self._config,
        )
        assistant_msg: dict = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.call_id, "type": "function",
                 "function": {"name": tc.function_name, "arguments": str(tc.arguments)}}
                for tc in response.tool_calls
            ]
        messages.append(assistant_msg)
        return response.tool_calls, response.input_tokens, response.output_tokens

    async def _execute_tools(
        self, tool_calls: list[ToolCall], tool_failures: dict[str, int],
    ) -> list[AgentEvent]:
        """Execute tool calls and return events + append results to messages."""
        events: list[AgentEvent] = []

        async def execute_single(tc: ToolCall) -> tuple[AgentEvent, str]:
            events.append(AgentEvent(type="tool_start", tool_name=tc.function_name,
                                      tool_args=tc.arguments, tool_call_id=tc.call_id))
            try:
                result = await asyncio.wait_for(
                    self._registry.execute(tc.function_name, tc.arguments, self._context),
                    timeout=self._tool_timeout,
                )
                tool_failures.pop(tc.function_name, None)  # Reset on success
                event = AgentEvent(type="tool_result", tool_name=tc.function_name,
                                   content=result, tool_call_id=tc.call_id)
                return event, result
            except asyncio.TimeoutError:
                error_msg = f"Tool '{tc.function_name}' timed out after {self._tool_timeout}s"
                logger.warning(error_msg)
                tool_failures[tc.function_name] = tool_failures.get(tc.function_name, 0) + 1
                event = AgentEvent(type="tool_error", tool_name=tc.function_name,
                                   content=error_msg, tool_call_id=tc.call_id)
                if self._on_tool_error == "raise":
                    raise
                return event, error_msg
            except Exception as exc:
                error_msg = f"Tool '{tc.function_name}' failed: {exc}"
                logger.warning(error_msg)
                tool_failures[tc.function_name] = tool_failures.get(tc.function_name, 0) + 1
                event = AgentEvent(type="tool_error", tool_name=tc.function_name,
                                   content=error_msg, tool_call_id=tc.call_id)
                if self._on_tool_error == "raise":
                    raise
                return event, error_msg

        # Execute parallel or sequential
        results: list[tuple[AgentEvent, str]] = []
        if self._parallel and len(tool_calls) > 1:
            results = await asyncio.gather(*[execute_single(tc) for tc in tool_calls])
        else:
            for tc in tool_calls:
                results.append(await execute_single(tc))

        # Collect events (tool_start events were already appended in execute_single)
        all_events = list(events)  # tool_start events
        for event, _ in results:
            all_events.append(event)  # tool_result/tool_error events

        return all_events

    def _filter_tools(self, tool_defs: list[dict], tool_failures: dict[str, int]) -> list[dict]:
        """Remove tools that have exceeded max_tool_failures."""
        if not tool_failures:
            return tool_defs
        disabled = {name for name, count in tool_failures.items() if count >= self._max_tool_failures}
        if disabled:
            logger.warning("Disabled tools due to failures: %s", disabled)
        return [td for td in tool_defs if td.get("function", {}).get("name") not in disabled]
