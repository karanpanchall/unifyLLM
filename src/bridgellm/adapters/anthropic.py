"""Adapter for the Anthropic Messages API.

Translates between bridgellm's OpenAI-shaped interface and Anthropic's
content-block-based API. Handles system-as-parameter, content blocks,
tool_result messages, image format conversion, extended thinking, and
the multi-event streaming protocol.

Requires: pip install bridgellm[anthropic]
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Optional

from ..errors import ProviderError, SDKNotInstalledError
from ..models import EmbeddingResponse, LLMResponse, ModelInfo, RequestConfig, StreamChunk, ToolCall
from ..registry import ProviderConfig
from ._convert import convert_messages_for_anthropic
from .base import LLMAdapter

logger = logging.getLogger(__name__)
_PROVIDER = "anthropic"


def _get_sdk():
    """Lazy-import the anthropic SDK, raising a clear error if missing."""
    try:
        import anthropic
        return anthropic
    except ImportError as exc:
        raise SDKNotInstalledError(_PROVIDER, "anthropic", "anthropic") from exc


class AnthropicAdapter(LLMAdapter):
    """Adapter for the Anthropic Messages API using the official SDK."""

    def __init__(self, config: ProviderConfig, api_key: str):
        sdk = _get_sdk()
        self._client = sdk.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        config: Optional[RequestConfig] = None,
    ) -> LLMResponse:
        if not messages:
            raise ProviderError(_PROVIDER, "messages list cannot be empty")

        system, converted = _prepare_messages(messages)
        kwargs = _build_request(model, converted, system, tools, temperature, max_tokens, config)
        try:
            message = await self._client.messages.create(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(_PROVIDER, str(exc)) from exc

        text, reasoning, tool_calls = _parse_content_blocks(getattr(message, "content", []))
        usage = getattr(message, "usage", None)
        return LLMResponse(
            content=text or None,
            reasoning_content=reasoning or None,
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) if usage else 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) if usage else 0,
            model=getattr(message, "model", model),
            finish_reason=_map_stop_reason(getattr(message, "stop_reason", "")),
        )

    async def stream(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        config: Optional[RequestConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        if not messages:
            raise ProviderError(_PROVIDER, "messages list cannot be empty")

        system, converted = _prepare_messages(messages)
        kwargs = _build_request(model, converted, system, tools, temperature, max_tokens, config)
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                tool_accumulator: dict[int, dict] = {}
                block_index = 0

                async for event in stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        block_index = getattr(event, "index", block_index)
                        if block and getattr(block, "type", "") == "tool_use":
                            tool_accumulator[block_index] = {
                                "id": getattr(block, "id", ""),
                                "name": getattr(block, "name", ""),
                                "arguments": "",
                            }

                    elif event_type == "text":
                        yield StreamChunk(delta_content=getattr(event, "text", ""))

                    elif event_type == "thinking":
                        yield StreamChunk(delta_reasoning=getattr(event, "thinking", ""))

                    elif event_type == "input_json":
                        partial = getattr(event, "partial_json", "")
                        if block_index in tool_accumulator:
                            tool_accumulator[block_index]["arguments"] += partial

                    elif event_type == "message_stop":
                        assembled = _assemble_tools(tool_accumulator) or None
                        final = await stream.get_final_message()
                        usage = getattr(final, "usage", None)
                        yield StreamChunk(
                            finish_reason=_map_stop_reason(getattr(final, "stop_reason", "")),
                            accumulated_tool_calls=assembled,
                            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) if usage else 0,
                            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) if usage else 0,
                        )
        except asyncio.CancelledError:
            raise
        except SDKNotInstalledError:
            raise
        except Exception as exc:
            raise ProviderError(_PROVIDER, str(exc)) from exc

    async def embed(self, model: str, texts: list[str], dimensions: Optional[int] = None) -> EmbeddingResponse:
        raise ProviderError(_PROVIDER, "Anthropic does not provide an embeddings API. Use an OpenAI-compatible provider.")

    async def list_models(self) -> list[ModelInfo]:
        try:
            models = []
            async for entry in self._client.models.list():
                model_id = getattr(entry, "id", None)
                if not model_id:
                    logger.warning("Skipping model entry without id: %s", entry)
                    continue
                raw_caps = getattr(entry, "capabilities", None)
                capabilities = None
                if raw_caps:
                    capabilities = raw_caps if isinstance(raw_caps, dict) else vars(raw_caps) if hasattr(raw_caps, "__dict__") else {"raw": str(raw_caps)}
                models.append(ModelInfo(
                    model_id=model_id,
                    provider=_PROVIDER,
                    owned_by="anthropic",
                    context_window=getattr(entry, "max_input_tokens", None),
                    max_output_tokens=getattr(entry, "max_tokens", None),
                    capabilities=capabilities,
                ))
            return models
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(_PROVIDER, str(exc)) from exc


# -- message preparation --


def _prepare_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Extract system prompt and convert messages for Anthropic."""
    system_parts: list[str] = []
    remaining: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if content:
                system_parts.append(content)
        else:
            remaining.append(msg)
    system = "\n\n".join(system_parts)
    converted = convert_messages_for_anthropic(remaining)
    return system, converted


# -- request building --


def _build_request(
    model: str, messages: list[dict], system: str,
    tools: Optional[list[dict]], temperature: float,
    max_tokens: Optional[int], config: Optional[RequestConfig],
) -> dict:
    """Assemble kwargs for the Anthropic SDK, translating from OpenAI conventions."""
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens or 4096,
    }

    has_thinking = config and config.reasoning and config.reasoning.get("budget_tokens")
    if not has_thinking:
        kwargs["temperature"] = temperature

    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = _convert_tools(tools)
        tool_choice = (config.tool_choice if config and config.tool_choice else "auto")
        kwargs["tool_choice"] = _translate_tool_choice(tool_choice)

    if not config:
        return kwargs

    if config.stop:
        kwargs["stop_sequences"] = config.stop
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if config.top_k is not None:
        kwargs["top_k"] = config.top_k
    if config.service_tier:
        kwargs["service_tier"] = config.service_tier
    if config.reasoning:
        budget = config.reasoning.get("budget_tokens")
        if budget:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        elif config.reasoning.get("effort") == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
    if config.response_format:
        schema = config.response_format.get("json_schema", {}).get("schema")
        if schema:
            kwargs["output_config"] = {"format": {"type": "json", "schema": schema}}
    if config.extra:
        kwargs.update(config.extra)
    return kwargs


def _convert_tools(openai_tools: list[dict]) -> list[dict]:
    return [
        {
            "name": (func := tool.get("function", tool)).get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {}),
        }
        for tool in openai_tools
    ]


def _translate_tool_choice(choice) -> dict:
    if isinstance(choice, dict):
        return choice
    mapping = {"auto": {"type": "auto"}, "none": {"type": "none"}, "required": {"type": "any"}}
    return mapping.get(choice, {"type": "auto"})


# -- response parsing --


def _parse_content_blocks(blocks: list) -> tuple[str, str, list[ToolCall]]:
    """Extract text, reasoning, and tool calls from content blocks."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in blocks:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            text_parts.append(getattr(block, "text", ""))
        elif block_type == "thinking":
            reasoning_parts.append(getattr(block, "thinking", ""))
        elif block_type == "tool_use":
            input_val = getattr(block, "input", {})
            if not isinstance(input_val, dict):
                logger.warning("Tool input is not a dict for '%s': %r", getattr(block, "name", ""), input_val)
                input_val = {"_parse_error": True, "raw": str(input_val)}
            tool_calls.append(ToolCall(
                call_id=getattr(block, "id", ""),
                function_name=getattr(block, "name", ""),
                arguments=input_val,
            ))

    return "\n".join(text_parts), "\n".join(reasoning_parts), tool_calls


def _assemble_tools(accumulator: dict[int, dict]) -> list[ToolCall]:
    assembled: list[ToolCall] = []
    for idx in sorted(accumulator):
        raw = accumulator[idx]
        try:
            arguments = json.loads(raw["arguments"]) if raw["arguments"] else {}
        except (json.JSONDecodeError, TypeError):
            logger.warning("Malformed tool JSON for '%s': %r", raw["name"], raw["arguments"])
            arguments = {"_parse_error": True, "raw": raw["arguments"]}
        assembled.append(ToolCall(call_id=raw["id"], function_name=raw["name"], arguments=arguments))
    return assembled


def _map_stop_reason(reason: str) -> str:
    return {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}.get(reason, reason or "")
