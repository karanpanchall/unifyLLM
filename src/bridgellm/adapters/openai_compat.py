"""Adapter for all OpenAI-compatible providers.

Covers OpenAI, Groq, Together, Fireworks, DeepSeek, Mistral, xAI,
Perplexity, OpenRouter, Cerebras, and Gemini (via compatibility endpoint).
Uses the official openai SDK with base_url override.
"""

import asyncio
import io
import json
import logging
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from .._model_caps import sanitize_params
from ..errors import ProviderError
from ..models import (
    AudioData,
    EmbeddingResponse,
    LLMResponse,
    ModelInfo,
    RequestConfig,
    StreamChunk,
    ToolCall,
    TTSResponse,
    TranscriptionResponse,
)
from ..registry import ProviderConfig
from .base import LLMAdapter

logger = logging.getLogger(__name__)

_EMBEDDING_BATCH_SIZE = 100


class OpenAICompatAdapter(LLMAdapter):
    """Single adapter serving all OpenAI-compatible providers via base_url."""

    def __init__(self, provider_name: str, config: ProviderConfig, api_key: str):
        self._provider = provider_name
        self._client = AsyncOpenAI(base_url=config.base_url, api_key=api_key)

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
            raise ProviderError(self._provider, "messages list cannot be empty")

        kwargs = _build_request(model, messages, tools, temperature, max_tokens, config)
        kwargs = sanitize_params(model, kwargs)
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(self._provider, str(exc)) from exc

        if not response.choices:
            raise ProviderError(self._provider, "API returned empty choices list")

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=getattr(choice.message, "content", None),
            reasoning_content=getattr(choice.message, "reasoning_content", None),
            audio=_parse_audio_output(getattr(choice.message, "audio", None)),
            tool_calls=_parse_tool_calls(getattr(choice.message, "tool_calls", None)),
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            cache_creation_tokens=_extract_cache_tokens(usage, "creation"),
            cache_read_tokens=_extract_cache_tokens(usage, "read"),
            model=getattr(response, "model", model),
            finish_reason=getattr(choice, "finish_reason", "") or "",
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
            raise ProviderError(self._provider, "messages list cannot be empty")

        kwargs = _build_request(model, messages, tools, temperature, max_tokens, config)
        kwargs = sanitize_params(model, kwargs)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(self._provider, str(exc)) from exc

        tool_accumulator: dict[int, dict] = {}
        total_input = total_output = 0

        try:
            async for chunk in response:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    total_input = getattr(chunk_usage, "prompt_tokens", 0) or total_input
                    total_output = getattr(chunk_usage, "completion_tokens", 0) or total_output

                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue

                delta = choice.delta
                if delta is None:
                    continue

                _accumulate_tool_deltas(tool_accumulator, getattr(delta, "tool_calls", None))
                finish = choice.finish_reason
                assembled = _assemble_tool_calls(tool_accumulator) if finish and tool_accumulator else None

                yield StreamChunk(
                    delta_content=getattr(delta, "content", None),
                    delta_reasoning=getattr(delta, "reasoning_content", None),
                    finish_reason=finish,
                    accumulated_tool_calls=assembled,
                )

            if total_input or total_output:
                yield StreamChunk(input_tokens=total_input, output_tokens=total_output)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(self._provider, str(exc)) from exc

    async def embed(self, model: str, texts: list[str], dimensions: Optional[int] = None) -> EmbeddingResponse:
        if not texts:
            raise ProviderError(self._provider, "texts list cannot be empty")

        all_vectors: list[list[float]] = []
        total_tokens = 0

        for batch_start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
            batch = texts[batch_start : batch_start + _EMBEDDING_BATCH_SIZE]
            kwargs: dict = {"model": model, "input": batch}
            if dimensions:
                kwargs["dimensions"] = dimensions
            try:
                response = await self._client.embeddings.create(**kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise ProviderError(self._provider, str(exc)) from exc

            all_vectors.extend(item.embedding for item in response.data)
            resp_usage = getattr(response, "usage", None)
            total_tokens += getattr(resp_usage, "prompt_tokens", 0) if resp_usage else 0

        return EmbeddingResponse(vectors=all_vectors, model=model, input_tokens=total_tokens)

    async def list_models(self) -> list[ModelInfo]:
        try:
            page = await self._client.models.list()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(self._provider, str(exc)) from exc

        return [_parse_model_entry(entry, self._provider) for entry in page.data]

    async def speak(
        self, model: str, text: str, voice: str = "alloy",
        response_format: str = "mp3", speed: float = 1.0,
    ) -> TTSResponse:
        if not text:
            raise ProviderError(self._provider, "text cannot be empty for TTS")
        try:
            response = await self._client.audio.speech.create(
                model=model, input=text, voice=voice,
                response_format=response_format, speed=speed,
            )
            audio_bytes = response.read()
            return TTSResponse(audio_data=audio_bytes, format=response_format)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(self._provider, str(exc)) from exc

    async def transcribe(
        self, model: str, audio_data: bytes,
        language: Optional[str] = None, response_format: str = "json",
    ) -> TranscriptionResponse:
        if not audio_data:
            raise ProviderError(self._provider, "audio_data cannot be empty")
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.wav"
        try:
            kwargs: dict = {"model": model, "file": audio_file, "response_format": response_format}
            if language:
                kwargs["language"] = language
            result = await self._client.audio.transcriptions.create(**kwargs)
            return TranscriptionResponse(
                text=getattr(result, "text", str(result)),
                language=getattr(result, "language", language),
                duration=getattr(result, "duration", None),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderError(self._provider, str(exc)) from exc
        finally:
            audio_file.close()


# -- request building --


def _build_request(
    model: str, messages: list[dict], tools: Optional[list[dict]],
    temperature: float, max_tokens: Optional[int], config: Optional[RequestConfig],
) -> dict:
    """Assemble kwargs for the OpenAI SDK call, applying all RequestConfig params."""
    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = (config.tool_choice if config and config.tool_choice else "auto")
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    if not config:
        return kwargs

    _set_if_present(kwargs, "response_format", config.response_format)
    _set_if_present(kwargs, "stop", config.stop)
    _set_if_present(kwargs, "top_p", config.top_p)
    _set_if_present(kwargs, "seed", config.seed)
    _set_if_present(kwargs, "frequency_penalty", config.frequency_penalty)
    _set_if_present(kwargs, "presence_penalty", config.presence_penalty)
    _set_if_present(kwargs, "service_tier", config.service_tier)
    _set_if_present(kwargs, "logprobs", config.logprobs)
    _set_if_present(kwargs, "top_logprobs", config.top_logprobs)
    _set_if_present(kwargs, "n", config.n)
    if config.parallel_tool_calls is not None and tools:
        kwargs["parallel_tool_calls"] = config.parallel_tool_calls
    if config.reasoning:
        effort = config.reasoning.get("effort")
        if effort:
            kwargs["reasoning_effort"] = effort
    if config.modalities:
        kwargs["modalities"] = config.modalities
    if config.audio:
        kwargs["audio"] = {"voice": config.audio.voice, "format": config.audio.format}
    if config.extra:
        kwargs.update(config.extra)
    return kwargs


def _set_if_present(kwargs: dict, key: str, value) -> None:
    if value is not None:
        kwargs[key] = value


# -- response parsing --


def _parse_model_entry(entry, provider: str) -> ModelInfo:
    """Normalize model metadata from different provider response shapes.

    Each provider uses different field names for context window:
    Groq: context_window, Together: context_length, Fireworks: contextLength,
    Mistral: max_context_length. We check all known names.
    """
    context_window = (
        getattr(entry, "context_window", None)
        or getattr(entry, "context_length", None)
        or getattr(entry, "contextLength", None)
        or getattr(entry, "max_context_length", None)
    )
    max_output = getattr(entry, "max_completion_tokens", None)

    # Together/Fireworks expose capabilities as boolean flags or type enums
    raw_caps = getattr(entry, "capabilities", None)
    capabilities = None
    if raw_caps:
        capabilities = raw_caps if isinstance(raw_caps, dict) else {"raw": str(raw_caps)}

    # OpenRouter nests limits under top_provider
    top_provider = getattr(entry, "top_provider", None)
    if top_provider:
        context_window = context_window or getattr(top_provider, "context_length", None)
        max_output = max_output or getattr(top_provider, "max_completion_tokens", None)

    return ModelInfo(
        model_id=getattr(entry, "id", ""),
        provider=provider,
        owned_by=getattr(entry, "owned_by", ""),
        created=getattr(entry, "created", 0),
        context_window=context_window,
        max_output_tokens=max_output,
        capabilities=capabilities,
    )


def _parse_audio_output(raw_audio) -> Optional[AudioData]:
    if not raw_audio:
        return None
    return AudioData(
        data=getattr(raw_audio, "data", ""),
        format=getattr(raw_audio, "format", "wav"),
        transcript=getattr(raw_audio, "transcript", None),
    )


def _extract_cache_tokens(usage, cache_type: str) -> int:
    if not usage:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if not details:
        return 0
    if cache_type == "read":
        return getattr(details, "cached_tokens", 0) or 0
    return 0


def _parse_tool_calls(raw_calls: Optional[list]) -> list[ToolCall]:
    if not raw_calls:
        return []
    result: list[ToolCall] = []
    for call in raw_calls:
        args_raw = getattr(call.function, "arguments", "")
        if not isinstance(args_raw, str):
            logger.warning("Tool arguments not a string for '%s': %r", call.function.name, args_raw)
            args_raw = str(args_raw)
        result.append(ToolCall(
            call_id=call.id,
            function_name=call.function.name,
            arguments=_safe_parse_json(args_raw, call.function.name),
        ))
    return result


def _accumulate_tool_deltas(accumulator: dict[int, dict], deltas: Optional[list]) -> None:
    if not deltas:
        return
    for delta in deltas:
        idx = delta.index
        if idx not in accumulator:
            accumulator[idx] = {"id": "", "name": "", "arguments": ""}
        if delta.id:
            accumulator[idx]["id"] = delta.id
        if delta.function:
            if delta.function.name:
                accumulator[idx]["name"] = delta.function.name
            if delta.function.arguments:
                accumulator[idx]["arguments"] += delta.function.arguments


def _assemble_tool_calls(accumulator: dict[int, dict]) -> list[ToolCall]:
    return [
        ToolCall(
            call_id=accumulator[idx]["id"],
            function_name=accumulator[idx]["name"],
            arguments=_safe_parse_json(accumulator[idx]["arguments"], accumulator[idx]["name"]),
        )
        for idx in sorted(accumulator)
    ]


def _safe_parse_json(json_string: str, function_name: str) -> dict:
    try:
        return json.loads(json_string) if json_string else {}
    except (json.JSONDecodeError, TypeError):
        logger.warning("Malformed tool call JSON for '%s': %r", function_name, json_string)
        return {"_parse_error": True, "raw": json_string}
