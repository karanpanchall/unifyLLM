"""Tests for multimodal features — audio, PDF, vision conversions."""

from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from bridgellm.adapters._convert import openai_pdf_to_anthropic
from bridgellm.adapters.openai_compat import OpenAICompatAdapter, _parse_audio_output
from bridgellm.models import (
    AudioConfig,
    AudioData,
    LLMResponse,
    RequestConfig,
    TTSResponse,
    TranscriptionResponse,
)
from bridgellm.registry import ProviderConfig


MOCK_CONFIG = ProviderConfig(base_url="https://api.example.com/v1", api_key_env="TEST_KEY")


def _create_adapter():
    return OpenAICompatAdapter(provider_name="test", config=MOCK_CONFIG, api_key="sk-test")


class TestAudioConfig:
    def test_default_values(self):
        config = AudioConfig()
        assert config.voice == "alloy"
        assert config.format == "wav"

    def test_custom_values(self):
        config = AudioConfig(voice="nova", format="mp3")
        assert config.voice == "nova"
        assert config.format == "mp3"


class TestAudioData:
    def test_creation(self):
        audio = AudioData(data="base64data", format="wav", transcript="Hello")
        assert audio.data == "base64data"
        assert audio.transcript == "Hello"


class TestParseAudioOutput:
    def test_none_returns_none(self):
        assert _parse_audio_output(None) is None

    def test_parses_audio_response(self):
        @dataclass
        class MockAudio:
            data: str = "base64audio"
            format: str = "wav"
            transcript: str = "Hello world"

        result = _parse_audio_output(MockAudio())
        assert isinstance(result, AudioData)
        assert result.data == "base64audio"
        assert result.transcript == "Hello world"


class TestAudioInRequestConfig:
    def test_modalities_forwarded(self):
        from bridgellm.adapters.openai_compat import _build_request

        config = RequestConfig(
            modalities=["text", "audio"],
            audio=AudioConfig(voice="nova", format="mp3"),
        )
        kwargs = _build_request("gpt-4o-audio-preview", [], None, 0.7, 100, config)
        assert kwargs["modalities"] == ["text", "audio"]
        assert kwargs["audio"] == {"voice": "nova", "format": "mp3"}

    def test_no_audio_no_keys(self):
        from bridgellm.adapters.openai_compat import _build_request

        kwargs = _build_request("gpt-4o", [], None, 0.7, 100, None)
        assert "modalities" not in kwargs
        assert "audio" not in kwargs


class TestPDFConversion:
    def test_anthropic_document_passthrough(self):
        block = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "PDFDATA"},
        }
        result = openai_pdf_to_anthropic(block)
        assert result == block

    def test_file_data_uri_to_document(self):
        block = {
            "type": "file",
            "file": {"url": "data:application/pdf;base64,PDFDATA"},
        }
        result = openai_pdf_to_anthropic(block)
        assert result["type"] == "document"
        assert result["source"]["media_type"] == "application/pdf"
        assert result["source"]["data"] == "PDFDATA"

    def test_file_url_to_document(self):
        block = {
            "type": "file",
            "file": {"url": "https://example.com/doc.pdf"},
        }
        result = openai_pdf_to_anthropic(block)
        assert result["type"] == "document"
        assert result["source"]["type"] == "url"

    def test_text_block_returns_none(self):
        assert openai_pdf_to_anthropic({"type": "text", "text": "hi"}) is None

    def test_empty_block_returns_none(self):
        assert openai_pdf_to_anthropic({}) is None


class TestTTSAdapter:
    @pytest.mark.asyncio
    async def test_speak_calls_sdk(self):
        adapter = _create_adapter()

        @dataclass
        class MockSpeechResponse:
            def read(self):
                return b"audio_bytes_here"

        adapter._client.audio.speech.create = AsyncMock(return_value=MockSpeechResponse())

        result = await adapter.speak(model="tts-1", text="Hello", voice="alloy")
        assert isinstance(result, TTSResponse)
        assert result.audio_data == b"audio_bytes_here"
        assert result.format == "mp3"

    @pytest.mark.asyncio
    async def test_speak_error_wrapped(self):
        adapter = _create_adapter()
        adapter._client.audio.speech.create = AsyncMock(side_effect=RuntimeError("TTS failed"))

        from bridgellm.errors import ProviderError
        with pytest.raises(ProviderError, match="TTS failed"):
            await adapter.speak(model="tts-1", text="Hello")


class TestTranscriptionAdapter:
    @pytest.mark.asyncio
    async def test_transcribe_calls_sdk(self):
        adapter = _create_adapter()

        @dataclass
        class MockTranscription:
            text: str = "Hello world"
            language: str = "en"
            duration: float = 2.5

        adapter._client.audio.transcriptions.create = AsyncMock(return_value=MockTranscription())

        result = await adapter.transcribe(model="whisper-1", audio_data=b"wav_data")
        assert isinstance(result, TranscriptionResponse)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 2.5

    @pytest.mark.asyncio
    async def test_transcribe_with_language(self):
        adapter = _create_adapter()

        @dataclass
        class MockTranscription:
            text: str = "Bonjour"
            language: Optional[str] = None
            duration: Optional[float] = None

        adapter._client.audio.transcriptions.create = AsyncMock(return_value=MockTranscription())

        result = await adapter.transcribe(model="whisper-1", audio_data=b"data", language="fr")
        call_kwargs = adapter._client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["language"] == "fr"

    @pytest.mark.asyncio
    async def test_transcribe_error_wrapped(self):
        adapter = _create_adapter()
        adapter._client.audio.transcriptions.create = AsyncMock(side_effect=RuntimeError("STT failed"))

        from bridgellm.errors import ProviderError
        with pytest.raises(ProviderError, match="STT failed"):
            await adapter.transcribe(model="whisper-1", audio_data=b"data")
