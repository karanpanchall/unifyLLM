"""Abstract base class for all LLM provider adapters.

Every adapter must implement the core methods. Optional methods (speak,
transcribe) have default implementations that raise ProviderError.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from ..errors import ProviderError
from ..models import (
    EmbeddingResponse,
    LLMResponse,
    ModelInfo,
    RequestConfig,
    StreamChunk,
    TTSResponse,
    TranscriptionResponse,
)


class LLMAdapter(ABC):
    """Contract that every provider adapter must fulfill."""

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        config: Optional[RequestConfig] = None,
    ) -> LLMResponse:
        """Send messages and return a complete response."""

    @abstractmethod
    async def stream(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        config: Optional[RequestConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Send messages and yield streaming chunks."""

    @abstractmethod
    async def embed(
        self,
        model: str,
        texts: list[str],
        dimensions: Optional[int] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings for a batch of texts."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Fetch available models from this provider."""

    async def speak(
        self,
        model: str,
        text: str,
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> TTSResponse:
        """Convert text to speech. Override in providers that support TTS."""
        raise ProviderError("base", "This provider does not support text-to-speech.")

    async def transcribe(
        self,
        model: str,
        audio_data: bytes,
        language: Optional[str] = None,
        response_format: str = "json",
    ) -> TranscriptionResponse:
        """Transcribe audio to text. Override in providers that support STT."""
        raise ProviderError("base", "This provider does not support transcription.")
