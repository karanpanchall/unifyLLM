"""Provider-agnostic data types for LLM requests and responses."""

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class AudioConfig:
    """Configuration for audio input/output in chat completions.

    Set on RequestConfig.audio to enable voice interactions.
    OpenAI models: gpt-4o-audio-preview, gpt-4o-mini-audio-preview.
    """

    voice: str = "alloy"
    format: str = "wav"  # wav, mp3, flac, opus, pcm16


@dataclass
class RequestConfig:
    """Optional parameters for fine-grained control over LLM requests.

    Common params (messages, tools, temperature, max_tokens) live on the
    method signature. Everything else goes here so the simple API stays simple.
    """

    response_format: Optional[dict] = None
    stop: Optional[list[str]] = None
    tool_choice: Union[str, dict, None] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    reasoning: Optional[dict] = None  # {"effort": "high"} or {"budget_tokens": 10000}
    service_tier: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    n: Optional[int] = None
    modalities: Optional[list[str]] = None  # ["text"], ["text", "audio"]
    audio: Optional[AudioConfig] = None
    extra: Optional[dict] = None  # provider-specific pass-through


@dataclass(frozen=True)
class ToolCall:
    """A tool/function call returned by the model."""

    call_id: str
    function_name: str
    arguments: dict[str, Any]


@dataclass
class AudioData:
    """Audio data returned by audio-capable models."""

    data: str  # base64-encoded audio
    format: str = "wav"
    transcript: Optional[str] = None


@dataclass
class LLMResponse:
    """Complete (non-streaming) response from any LLM provider."""

    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    audio: Optional[AudioData] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    model: str = ""
    finish_reason: str = ""


@dataclass
class StreamChunk:
    """Single chunk from a streaming LLM response.

    During streaming, `delta_content` carries text fragments and
    `delta_reasoning` carries reasoning/thinking fragments.
    On the final chunk, `accumulated_tool_calls` holds fully-assembled calls.
    """

    delta_content: Optional[str] = None
    delta_reasoning: Optional[str] = None
    finish_reason: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    accumulated_tool_calls: Optional[list[ToolCall]] = None


@dataclass(frozen=True)
class EmbeddingResponse:
    """Batch embedding result from any provider."""

    vectors: list[list[float]]
    model: str = ""
    input_tokens: int = 0


@dataclass(frozen=True)
class TTSResponse:
    """Text-to-speech result."""

    audio_data: bytes
    format: str = "mp3"


@dataclass
class TranscriptionResponse:
    """Speech-to-text transcription result."""

    text: str
    language: Optional[str] = None
    duration: Optional[float] = None


@dataclass
class ModelInfo:
    """Metadata about a model from a provider catalog.

    Fields populated depend on what the provider API returns.
    context_window, max_output_tokens, and capabilities are None
    when the provider does not expose them (e.g., OpenAI, DeepSeek).
    """

    model_id: str
    provider: str
    owned_by: str = ""
    created: int = 0
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    capabilities: Optional[dict] = None
