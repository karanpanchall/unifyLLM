"""Adapter for AWS Bedrock.

Uses boto3 with the Converse API for chat completions and streaming.
Requires: pip install boto3

Authentication: Uses standard AWS credential chain
(env vars AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_SESSION_TOKEN,
or IAM role, or AWS profile).
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Optional

from ..errors import ProviderError, SDKNotInstalledError
from ..models import EmbeddingResponse, LLMResponse, ModelInfo, RequestConfig, StreamChunk, ToolCall
from ..registry import ProviderConfig
from .base import LLMAdapter

logger = logging.getLogger(__name__)
_PROVIDER = "bedrock"


def _get_boto3():
    try:
        import boto3
        return boto3
    except ImportError as exc:
        raise SDKNotInstalledError(_PROVIDER, "boto3", "bedrock") from exc


class BedrockAdapter(LLMAdapter):
    """Adapter for AWS Bedrock using the Converse API."""

    def __init__(self, config: ProviderConfig, api_key: str):
        boto3 = _get_boto3()
        region = config.base_url or "us-east-1"
        self._client = boto3.client("bedrock-runtime", region_name=region)

    async def complete(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        config: Optional[RequestConfig] = None,
    ) -> LLMResponse:
        converse_messages, system = _convert_messages(messages)
        kwargs = _build_converse_request(model, converse_messages, system, tools, temperature, max_tokens)

        try:
            response = await asyncio.to_thread(self._client.converse, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(_PROVIDER, str(exc)) from exc

        output = response.get("output", {}).get("message", {})
        content_blocks = output.get("content", [])
        text, tool_calls = _parse_converse_content(content_blocks)
        usage = response.get("usage", {})

        return LLMResponse(
            content=text or None,
            tool_calls=tool_calls,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            model=model,
            finish_reason=_map_stop_reason(response.get("stopReason", "")),
        )

    async def stream(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        config: Optional[RequestConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        converse_messages, system = _convert_messages(messages)
        kwargs = _build_converse_request(model, converse_messages, system, tools, temperature, max_tokens)

        try:
            response = await asyncio.to_thread(self._client.converse_stream, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(_PROVIDER, str(exc)) from exc

        event_stream = response.get("stream", [])
        try:
            for event in event_stream:
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    text = delta.get("text")
                    if text:
                        yield StreamChunk(delta_content=text)
                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})
                    yield StreamChunk(
                        input_tokens=usage.get("inputTokens", 0),
                        output_tokens=usage.get("outputTokens", 0),
                    )
                elif "messageStop" in event:
                    yield StreamChunk(
                        finish_reason=_map_stop_reason(event["messageStop"].get("stopReason", "")),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(_PROVIDER, str(exc)) from exc

    async def embed(self, model: str, texts: list[str], dimensions: Optional[int] = None) -> EmbeddingResponse:
        raise ProviderError(_PROVIDER, "Use a dedicated embedding provider. Bedrock embeddings require model-specific request formats.")

    async def list_models(self) -> list[ModelInfo]:
        try:
            boto3 = _get_boto3()
            client = boto3.client("bedrock")
            response = await asyncio.to_thread(
                client.list_foundation_models, byOutputModality="TEXT"
            )
            return [
                ModelInfo(
                    model_id=model.get("modelId", ""),
                    provider=_PROVIDER,
                    owned_by=model.get("providerName", ""),
                )
                for model in response.get("modelSummaries", [])
            ]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(_PROVIDER, str(exc)) from exc


def _convert_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Convert OpenAI messages to Bedrock Converse format, extracting system."""
    system: list[dict] = []
    converse_messages: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            if content:
                system.append({"text": content})
        elif role in ("user", "assistant"):
            if isinstance(content, str):
                converse_messages.append({"role": role, "content": [{"text": content}]})
            elif isinstance(content, list):
                converse_messages.append({"role": role, "content": content})
        elif role == "tool":
            converse_messages.append({
                "role": "user",
                "content": [{"toolResult": {
                    "toolUseId": msg.get("tool_call_id", ""),
                    "content": [{"text": content}],
                }}],
            })

    return converse_messages, system


def _build_converse_request(
    model: str, messages: list[dict], system: list[dict],
    tools: Optional[list[dict]], temperature: Optional[float], max_tokens: Optional[int],
) -> dict:
    kwargs: dict = {"modelId": model, "messages": messages}
    if system:
        kwargs["system"] = system

    inference_config: dict = {}
    if temperature is not None:
        inference_config["temperature"] = temperature
    if max_tokens:
        inference_config["maxTokens"] = max_tokens
    if inference_config:
        kwargs["inferenceConfig"] = inference_config

    if tools:
        kwargs["toolConfig"] = {"tools": [_convert_tool(tool) for tool in tools]}
    return kwargs


def _convert_tool(openai_tool: dict) -> dict:
    func = openai_tool.get("function", openai_tool)
    return {
        "toolSpec": {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "inputSchema": {"json": func.get("parameters", {})},
        }
    }


def _parse_converse_content(blocks: list) -> tuple[str, list[ToolCall]]:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in blocks:
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(ToolCall(
                call_id=tu.get("toolUseId", ""),
                function_name=tu.get("name", ""),
                arguments=tu.get("input", {}),
            ))
    return "\n".join(text_parts), tool_calls


def _map_stop_reason(reason: str) -> str:
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(reason, reason or "")
