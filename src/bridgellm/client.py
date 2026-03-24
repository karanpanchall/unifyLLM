"""Main BridgeLLM client — the single entry point for all LLM operations.

Routes requests to the correct provider adapter based on the model string.
Supports multi-provider API keys, task-specific default models, fallback
chains, and config-dict initialization.
"""

import asyncio
import logging
import os
import threading
from typing import AsyncIterator, Optional

from .adapters import create_adapter
from .adapters.base import LLMAdapter
from .errors import AllProvidersFailedError
from .models import (
    EmbeddingResponse,
    LLMResponse,
    ModelInfo,
    RequestConfig,
    StreamChunk,
    TTSResponse,
    TranscriptionResponse,
)
from .registry import ProviderConfig, get_provider_config, parse_model_string, resolve_api_key

logger = logging.getLogger(__name__)

_ENV_MODEL_KEY = "BRIDGELLM_MODEL"
_DEFAULT_MODEL = "gpt-4o-mini"


class BridgeLLM:
    """Provider-agnostic LLM client.

    Initialization options:

        # Minimal — reads OPENAI_API_KEY from env
        llm = BridgeLLM()

        # Multi-provider with per-provider keys
        llm = BridgeLLM(
            model="openai/gpt-4o",
            api_keys={"openai": "sk-...", "groq": "gsk-...", "anthropic": "sk-ant-..."},
            fallback_models=["groq/llama-3.3-70b"],
            embedding_model="openai/text-embedding-3-small",
            tts_model="openai/tts-1",
        )

        # From a config dict (load from YAML/JSON/settings)
        llm = BridgeLLM.from_config({...})
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_keys: Optional[dict[str, str]] = None,
        env_var_names: Optional[dict[str, str]] = None,
        base_url: Optional[str] = None,
        fallback_models: Optional[list[str]] = None,
        system_prompt: Optional[str] = None,
        embedding_model: Optional[str] = None,
        tts_model: Optional[str] = None,
        transcription_model: Optional[str] = None,
    ):
        """
        Args:
            model: Primary model as 'provider/model' (e.g., 'openai/gpt-4o').
            api_key: Single API key used for all providers.
            api_keys: Per-provider API keys (e.g., {"openai": "sk-...", "groq": "gsk-..."}).
            env_var_names: Custom env var names per provider
                (e.g., {"openai": "MY_APP_OPENAI_KEY", "groq": "PROD_GROQ_KEY"}).
                Falls back to defaults (OPENAI_API_KEY, GROQ_API_KEY, etc.) if not set.
            base_url: Override the base URL for the primary provider.
            system_prompt: Default system prompt prepended to every call.
                Skipped if the messages already contain a system message.
                Can also be included directly in messages if preferred.
            fallback_models: Models to try if the primary fails.
            embedding_model: Default model for embed() calls.
            tts_model: Default model for speak() calls.
            transcription_model: Default model for transcribe() calls.
        """
        self._model_string = model or os.environ.get(_ENV_MODEL_KEY, _DEFAULT_MODEL)
        self._single_api_key = api_key
        self._api_keys = api_keys or {}
        self._env_var_names = env_var_names or {}
        self._base_url_override = base_url
        self._system_prompt = system_prompt
        self._fallback_models = fallback_models or []
        self._embedding_model = embedding_model
        self._tts_model = tts_model or "openai/tts-1"
        self._transcription_model = transcription_model or "openai/whisper-1"
        self._adapter_cache: dict[str, LLMAdapter] = {}
        self._cache_lock = threading.Lock()  # Thread-safe adapter creation

        # Eagerly resolve all providers that have explicit keys — fail fast on bad config
        self._primary_adapter, self._primary_model = self._resolve_adapter(self._model_string)
        self._eager_init_providers()

    def _eager_init_providers(self) -> None:
        """Initialize adapters for all providers with explicit keys.

        Validates keys and connections upfront so misconfiguration
        fails at init time, not mid-request.
        """
        for provider_name in self._api_keys:
            if provider_name not in self._adapter_cache:
                try:
                    self._resolve_adapter(f"{provider_name}/placeholder")
                except Exception as exc:
                    logger.warning("Failed to initialize provider '%s': %s", provider_name, exc)

    @classmethod
    def from_config(cls, config: dict) -> "BridgeLLM":
        """Create a client from a configuration dictionary.

        Accepts the same keys as __init__ params. Useful for loading
        from YAML, JSON, Django settings, or environment-based config.
        """
        return cls(
            model=config.get("model"),
            api_key=config.get("api_key"),
            api_keys=config.get("api_keys"),
            env_var_names=config.get("env_var_names"),
            base_url=config.get("base_url"),
            system_prompt=config.get("system_prompt"),
            fallback_models=config.get("fallback_models"),
            embedding_model=config.get("embedding_model"),
            tts_model=config.get("tts_model"),
            transcription_model=config.get("transcription_model"),
        )

    def _resolve_api_key(self, provider_name: str, config: ProviderConfig) -> str:
        """Resolve API key with priority: api_keys dict > single api_key > custom env var > default env var."""
        explicit = self._api_keys.get(provider_name) or self._single_api_key
        custom_env = self._env_var_names.get(provider_name)
        return resolve_api_key(config, explicit, custom_env)

    def _resolve_adapter(self, model_string: str) -> tuple[LLMAdapter, str]:
        """Parse a model string and return (adapter, model_name).

        Caches adapters by provider to reuse HTTP connections.
        Thread-safe via lock to prevent duplicate adapter creation.
        """
        provider_name, model_name = parse_model_string(model_string)

        if provider_name not in self._adapter_cache:
            with self._cache_lock:
                # Double-check after acquiring lock
                if provider_name not in self._adapter_cache:
                    config = get_provider_config(provider_name)

                    if self._base_url_override and config.openai_compatible:
                        config = ProviderConfig(
                            base_url=self._base_url_override,
                            api_key_env=config.api_key_env,
                            models_path=config.models_path,
                            openai_compatible=config.openai_compatible,
                        )

                    api_key = self._resolve_api_key(provider_name, config)
                    adapter = create_adapter(provider_name, config, api_key)
                    self._adapter_cache[provider_name] = adapter
                    logger.info("Initialized adapter for provider '%s'", provider_name)

        return self._adapter_cache[provider_name], model_name

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Prepend system_prompt if set and no system message already exists."""
        if not self._system_prompt:
            return messages
        has_system = any(msg.get("role") == "system" for msg in messages)
        if has_system:
            return messages
        return [{"role": "system", "content": self._system_prompt}] + messages

    # -- Chat completions --

    async def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        config: Optional[RequestConfig] = None,
    ) -> LLMResponse:
        """Send messages and return a complete response.

        Tries the specified/primary model first, then each fallback.
        """
        prepared = self._prepare_messages(messages)
        primary = model or self._model_string
        models_to_try = [primary] + self._fallback_models
        errors: list[Exception] = []

        for model_string in models_to_try:
            adapter, model_name = self._resolve_adapter(model_string)
            try:
                return await adapter.complete(
                    model=model_name, messages=prepared, tools=tools,
                    temperature=temperature, max_tokens=max_tokens, config=config,
                )
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:
                logger.warning("Completion failed for '%s': %s", model_string, exc)
                errors.append(exc)

        raise AllProvidersFailedError(errors)

    async def stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        config: Optional[RequestConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response, trying fallbacks on failure."""
        prepared = self._prepare_messages(messages)
        primary = model or self._model_string
        models_to_try = [primary] + self._fallback_models
        errors: list[Exception] = []

        for model_string in models_to_try:
            adapter, model_name = self._resolve_adapter(model_string)
            try:
                async for chunk in adapter.stream(
                    model=model_name, messages=prepared, tools=tools,
                    temperature=temperature, max_tokens=max_tokens, config=config,
                ):
                    yield chunk
                return
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:
                logger.warning("Stream failed for '%s': %s", model_string, exc)
                errors.append(exc)

        raise AllProvidersFailedError(errors)

    # -- Embeddings --

    async def embed(
        self,
        texts: list[str],
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings. Uses embedding_model default if set."""
        target = model or self._embedding_model or self._model_string
        adapter, model_name = self._resolve_adapter(target)
        return await adapter.embed(model=model_name, texts=texts, dimensions=dimensions)

    async def embed_query(self, text: str, model: Optional[str] = None, dimensions: Optional[int] = None) -> list[float]:
        """Embed a single text and return the vector directly."""
        response = await self.embed(texts=[text], model=model, dimensions=dimensions)
        return response.vectors[0]

    # -- Audio --

    async def speak(
        self,
        text: str,
        model: Optional[str] = None,
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> TTSResponse:
        """Convert text to speech audio."""
        target = model or self._tts_model
        adapter, model_name = self._resolve_adapter(target)
        return await adapter.speak(
            model=model_name, text=text, voice=voice,
            response_format=response_format, speed=speed,
        )

    async def transcribe(
        self,
        audio_data: bytes,
        model: Optional[str] = None,
        language: Optional[str] = None,
        response_format: str = "json",
    ) -> TranscriptionResponse:
        """Transcribe audio to text."""
        target = model or self._transcription_model
        adapter, model_name = self._resolve_adapter(target)
        return await adapter.transcribe(
            model=model_name, audio_data=audio_data,
            language=language, response_format=response_format,
        )

    # -- Discovery --

    @property
    def active_providers(self) -> list[str]:
        """List provider names that have been initialized."""
        return list(self._adapter_cache.keys())

    async def get_model_info(self, model: Optional[str] = None) -> Optional[ModelInfo]:
        """Fetch metadata for a specific model from its provider's API.

        Returns None if the model is not found in the provider's catalog.
        Fields like context_window and max_output_tokens are only populated
        if the provider API exposes them.
        """
        target = model or self._model_string
        adapter, model_name = self._resolve_adapter(target)
        all_models = await adapter.list_models()
        for model_info in all_models:
            if model_info.model_id == model_name:
                return model_info
        return None

    async def list_models(self, provider: Optional[str] = None) -> list[ModelInfo]:
        """List available models. Defaults to primary provider."""
        if provider:
            adapter, _ = self._resolve_adapter(f"{provider}/any")
        else:
            adapter = self._primary_adapter
        return await adapter.list_models()
