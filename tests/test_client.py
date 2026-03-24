"""Tests for the BridgeLLM client — routing, caching, and fallback logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bridgellm.client import BridgeLLM
from bridgellm.errors import AllProvidersFailedError, ProviderError, ProviderNotFoundError
from bridgellm.models import EmbeddingResponse, LLMResponse, ModelInfo, StreamChunk, ToolCall


class TestClientInit:
    def test_defaults_to_env_or_fallback(self, openai_api_key, monkeypatch):
        monkeypatch.delenv("BRIDGELLM_MODEL", raising=False)
        llm = BridgeLLM()
        # Should not raise — uses default model with openai provider

    def test_explicit_model(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")
        assert llm._model_string == "openai/gpt-4o"

    def test_env_model(self, openai_api_key, monkeypatch):
        monkeypatch.setenv("BRIDGELLM_MODEL", "groq/llama-3")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        llm = BridgeLLM()
        assert llm._model_string == "groq/llama-3"

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ProviderNotFoundError):
            BridgeLLM(model="nonexistent/model")

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            BridgeLLM(model="gpt-4o")

    def test_explicit_api_key(self):
        llm = BridgeLLM(model="openai/gpt-4o", api_key="sk-explicit")
        # Should not raise


class TestClientComplete:
    @pytest.mark.asyncio
    async def test_routes_to_adapter(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")
        mock_response = LLMResponse(content="Hello", model="gpt-4o", finish_reason="stop")
        llm._primary_adapter.complete = AsyncMock(return_value=mock_response)

        result = await llm.complete(messages=[{"role": "user", "content": "Hi"}])
        assert result.content == "Hello"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        llm = BridgeLLM(
            model="openai/gpt-4o",
            fallback_models=["groq/llama-3"],
        )

        # Primary fails
        llm._primary_adapter.complete = AsyncMock(
            side_effect=ProviderError("openai", "rate limited")
        )

        # Fallback succeeds
        fallback_response = LLMResponse(content="Fallback response", model="llama-3")
        fallback_adapter = llm._resolve_adapter("groq/llama-3")[0]
        fallback_adapter.complete = AsyncMock(return_value=fallback_response)

        result = await llm.complete(messages=[{"role": "user", "content": "Hi"}])
        assert result.content == "Fallback response"

    @pytest.mark.asyncio
    async def test_all_fail_raises(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")
        llm._primary_adapter.complete = AsyncMock(
            side_effect=ProviderError("openai", "error")
        )

        with pytest.raises(AllProvidersFailedError):
            await llm.complete(messages=[{"role": "user", "content": "test"}])


class TestClientStream:
    @pytest.mark.asyncio
    async def test_yields_chunks(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")

        async def mock_stream(**kwargs):
            yield StreamChunk(delta_content="Hello ")
            yield StreamChunk(delta_content="world")
            yield StreamChunk(finish_reason="stop")

        llm._primary_adapter.stream = mock_stream

        chunks = []
        async for chunk in llm.stream(messages=[{"role": "user", "content": "test"}]):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].delta_content == "Hello "

    @pytest.mark.asyncio
    async def test_stream_all_fail_raises(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")

        async def failing_stream(**kwargs):
            raise ProviderError("openai", "stream failed")
            yield  # Make it a generator

        llm._primary_adapter.stream = failing_stream

        with pytest.raises(AllProvidersFailedError):
            async for _ in llm.stream(messages=[{"role": "user", "content": "test"}]):
                pass


class TestClientEmbed:
    @pytest.mark.asyncio
    async def test_embed_with_primary(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")
        mock_response = EmbeddingResponse(vectors=[[0.1, 0.2]], model="emb", input_tokens=5)
        llm._primary_adapter.embed = AsyncMock(return_value=mock_response)

        result = await llm.embed(texts=["hello"])
        assert len(result.vectors) == 1

    @pytest.mark.asyncio
    async def test_embed_with_explicit_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        llm = BridgeLLM(model="openai/gpt-4o")
        mock_response = EmbeddingResponse(vectors=[[0.1]], model="emb")

        # Override the embedding adapter
        groq_adapter = llm._resolve_adapter("groq/some-emb")[0]
        groq_adapter.embed = AsyncMock(return_value=mock_response)

        result = await llm.embed(texts=["hello"], model="groq/some-emb")
        assert len(result.vectors) == 1


class TestClientListModels:
    @pytest.mark.asyncio
    async def test_lists_models(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")
        mock_models = [ModelInfo(model_id="gpt-4o", provider="openai")]
        llm._primary_adapter.list_models = AsyncMock(return_value=mock_models)

        models = await llm.list_models()
        assert len(models) == 1
        assert models[0].model_id == "gpt-4o"


class TestClientEmbedQuery:
    @pytest.mark.asyncio
    async def test_returns_single_vector(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")
        mock_response = EmbeddingResponse(vectors=[[0.1, 0.2, 0.3]], model="emb", input_tokens=5)
        llm._primary_adapter.embed = AsyncMock(return_value=mock_response)

        vector = await llm.embed_query("hello")
        assert vector == [0.1, 0.2, 0.3]


class TestClientBaseUrlOverride:
    def test_base_url_applied(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o", base_url="https://custom-proxy.com/v1")
        # Should not raise — just verifies constructor accepts base_url


class TestSystemPrompt:
    @pytest.mark.asyncio
    async def test_prepends_system_prompt(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o", system_prompt="You are a pirate.")
        mock_response = LLMResponse(content="Ahoy!", model="gpt-4o", finish_reason="stop")
        llm._primary_adapter.complete = AsyncMock(return_value=mock_response)

        await llm.complete(messages=[{"role": "user", "content": "Hello"}])

        call_kwargs = llm._primary_adapter.complete.call_args[1]
        passed_messages = call_kwargs["messages"]
        assert passed_messages[0]["role"] == "system"
        assert passed_messages[0]["content"] == "You are a pirate."
        assert passed_messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_skips_when_system_already_present(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o", system_prompt="Default persona.")
        mock_response = LLMResponse(content="ok", model="gpt-4o", finish_reason="stop")
        llm._primary_adapter.complete = AsyncMock(return_value=mock_response)

        messages = [
            {"role": "system", "content": "Custom override."},
            {"role": "user", "content": "Hello"},
        ]
        await llm.complete(messages=messages)

        call_kwargs = llm._primary_adapter.complete.call_args[1]
        passed_messages = call_kwargs["messages"]
        # Should NOT prepend default — user already has a system message
        assert len(passed_messages) == 2
        assert passed_messages[0]["content"] == "Custom override."

    @pytest.mark.asyncio
    async def test_no_system_prompt_no_change(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")  # no system_prompt
        mock_response = LLMResponse(content="ok", model="gpt-4o", finish_reason="stop")
        llm._primary_adapter.complete = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        await llm.complete(messages=messages)

        call_kwargs = llm._primary_adapter.complete.call_args[1]
        assert len(call_kwargs["messages"]) == 1

    @pytest.mark.asyncio
    async def test_system_prompt_in_stream(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o", system_prompt="Be concise.")

        captured_messages = None

        async def mock_stream(**kwargs):
            nonlocal captured_messages
            captured_messages = kwargs["messages"]
            yield StreamChunk(delta_content="ok")

        llm._primary_adapter.stream = mock_stream

        async for _ in llm.stream(messages=[{"role": "user", "content": "Hi"}]):
            pass

        assert captured_messages[0]["role"] == "system"
        assert captured_messages[0]["content"] == "Be concise."

    def test_from_config_with_system_prompt(self, openai_api_key):
        llm = BridgeLLM.from_config({
            "model": "openai/gpt-4o",
            "system_prompt": "You are a helpful tutor.",
        })
        assert llm._system_prompt == "You are a helpful tutor."


class TestAdapterCaching:
    def test_same_provider_reuses_adapter(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")
        adapter_a, _ = llm._resolve_adapter("openai/gpt-4o-mini")
        adapter_b, _ = llm._resolve_adapter("openai/gpt-5")
        assert adapter_a is adapter_b


class TestEagerProviderInit:
    def test_all_api_keys_providers_initialized(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        llm = BridgeLLM(
            model="openai/gpt-4o",
            api_keys={"openai": "sk-test", "groq": "gsk-test"},
        )
        assert "openai" in llm.active_providers
        assert "groq" in llm.active_providers

    def test_bad_provider_in_api_keys_warns_not_crashes(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # "badprovider" is not in registry — should warn, not crash
        # But the primary model init will use openai which has a key
        llm = BridgeLLM(
            model="openai/gpt-4o",
            api_keys={"openai": "sk-test", "badprovider": "bad-key"},
        )
        assert "openai" in llm.active_providers
        # badprovider should have been skipped with a warning

    def test_active_providers_property(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")
        assert "openai" in llm.active_providers
        assert isinstance(llm.active_providers, list)


class TestMultiProviderApiKeys:
    def test_api_keys_dict(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        llm = BridgeLLM(
            model="openai/gpt-4o",
            api_keys={"openai": "sk-from-dict", "groq": "gsk-from-dict"},
        )
        # Should not raise — both keys provided via dict

    def test_api_keys_dict_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        llm = BridgeLLM(
            model="openai/gpt-4o",
            api_keys={"openai": "sk-from-dict"},
        )
        # The dict key should take priority (verified by adapter init succeeding)

    def test_falls_back_to_single_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = BridgeLLM(model="openai/gpt-4o", api_key="sk-single-key")
        # Single key used as fallback for any provider

    def test_falls_back_to_env_when_no_dict_entry(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        llm = BridgeLLM(
            model="openai/gpt-4o",
            api_keys={"groq": "gsk-only-groq"},  # no openai entry
        )
        # Should use OPENAI_API_KEY env var for openai provider


class TestCustomEnvVarNames:
    def test_custom_env_var_name(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MY_APP_LLM_KEY", "sk-from-custom-env")
        llm = BridgeLLM(
            model="openai/gpt-4o",
            env_var_names={"openai": "MY_APP_LLM_KEY"},
        )
        # Should succeed — reads from MY_APP_LLM_KEY instead of OPENAI_API_KEY

    def test_falls_back_to_default_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-default")
        monkeypatch.delenv("MY_MISSING_VAR", raising=False)
        llm = BridgeLLM(
            model="openai/gpt-4o",
            env_var_names={"openai": "MY_MISSING_VAR"},
        )
        # Should succeed — MY_MISSING_VAR not found, falls back to OPENAI_API_KEY

    def test_from_config_with_env_var_names(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("PROD_OPENAI_KEY", "sk-prod")
        llm = BridgeLLM.from_config({
            "model": "openai/gpt-4o",
            "env_var_names": {"openai": "PROD_OPENAI_KEY"},
        })


class TestTaskSpecificModels:
    @pytest.mark.asyncio
    async def test_embedding_model_used(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o", embedding_model="openai/text-embedding-3-small")
        mock_response = EmbeddingResponse(vectors=[[0.1]], model="emb")
        llm._primary_adapter.embed = AsyncMock(return_value=mock_response)

        await llm.embed(texts=["test"])
        call_kwargs = llm._primary_adapter.embed.call_args[1]
        assert call_kwargs["model"] == "text-embedding-3-small"

    def test_tts_model_default(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")
        assert llm._tts_model == "openai/tts-1"

    def test_tts_model_custom(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o", tts_model="openai/gpt-4o-mini-tts")
        assert llm._tts_model == "openai/gpt-4o-mini-tts"

    def test_transcription_model_default(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")
        assert llm._transcription_model == "openai/whisper-1"


class TestFromConfig:
    def test_from_config_basic(self, openai_api_key):
        llm = BridgeLLM.from_config({"model": "openai/gpt-4o"})
        assert llm._model_string == "openai/gpt-4o"

    def test_from_config_full(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = BridgeLLM.from_config({
            "model": "openai/gpt-4o",
            "api_keys": {"openai": "sk-from-config"},
            "fallback_models": ["groq/llama-3"],
            "embedding_model": "openai/text-embedding-3-small",
            "tts_model": "openai/gpt-4o-mini-tts",
            "transcription_model": "openai/gpt-4o-mini-transcribe",
        })
        assert llm._model_string == "openai/gpt-4o"
        assert llm._fallback_models == ["groq/llama-3"]
        assert llm._embedding_model == "openai/text-embedding-3-small"
        assert llm._tts_model == "openai/gpt-4o-mini-tts"

    def test_from_config_empty_dict(self, openai_api_key):
        llm = BridgeLLM.from_config({})
        # Falls back to defaults

    def test_from_config_with_base_url(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = BridgeLLM.from_config({
            "model": "openai/gpt-4o",
            "api_key": "sk-test",
            "base_url": "https://my-proxy.com/v1",
        })


class TestPerCallModelOverride:
    @pytest.mark.asyncio
    async def test_complete_with_model_override(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")
        mock_response = LLMResponse(content="From mini", model="gpt-4o-mini")
        llm._primary_adapter.complete = AsyncMock(return_value=mock_response)

        result = await llm.complete(messages=[{"role": "user", "content": "test"}], model="openai/gpt-4o-mini")
        call_kwargs = llm._primary_adapter.complete.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_stream_with_model_override(self, openai_api_key):
        llm = BridgeLLM(model="openai/gpt-4o")

        async def mock_stream(**kwargs):
            assert kwargs["model"] == "gpt-4o-mini"
            yield StreamChunk(delta_content="ok")

        llm._primary_adapter.stream = mock_stream

        async for chunk in llm.stream(messages=[{"role": "user", "content": "test"}], model="openai/gpt-4o-mini"):
            assert chunk.delta_content == "ok"


class TestListModelsProvider:
    @pytest.mark.asyncio
    async def test_list_specific_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        llm = BridgeLLM(model="openai/gpt-4o")
        groq_adapter = llm._resolve_adapter("groq/any")[0]
        groq_adapter.list_models = AsyncMock(return_value=[
            ModelInfo(model_id="llama-3", provider="groq"),
        ])

        models = await llm.list_models(provider="groq")
        assert models[0].provider == "groq"
