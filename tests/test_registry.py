"""Tests for bridgellm.registry — provider config, parsing, and key resolution."""

import os

import pytest

from bridgellm.errors import ProviderNotFoundError
from bridgellm.registry import (
    PROVIDERS,
    ProviderConfig,
    mask_key,
    get_provider_config,
    parse_model_string,
    register_provider,
    resolve_api_key,
)


class TestParseModelString:
    def test_with_provider_prefix(self):
        provider, model = parse_model_string("groq/llama-3.3-70b")
        assert provider == "groq"
        assert model == "llama-3.3-70b"

    def test_without_prefix_defaults_to_openai(self):
        provider, model = parse_model_string("gpt-5-mini")
        assert provider == "openai"
        assert model == "gpt-5-mini"

    def test_case_insensitive_provider(self):
        provider, model = parse_model_string("GROQ/llama-3.3-70b")
        assert provider == "groq"

    def test_anthropic_prefix(self):
        provider, model = parse_model_string("anthropic/claude-sonnet-4-20250514")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-20250514"

    def test_model_with_multiple_slashes(self):
        """Some model names contain slashes (e.g., org/model-name)."""
        provider, model = parse_model_string("openrouter/meta-llama/llama-3-70b")
        assert provider == "openrouter"
        assert model == "meta-llama/llama-3-70b"

    def test_empty_string(self):
        provider, model = parse_model_string("")
        assert provider == "openai"
        assert model == ""

    def test_empty_provider_raises(self):
        with pytest.raises(ValueError, match="provider name cannot be empty"):
            parse_model_string("/gpt-4o")

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="model name cannot be empty"):
            parse_model_string("openai/")


class TestGetProviderConfig:
    def test_known_provider(self):
        config = get_provider_config("openai")
        assert config.base_url == "https://api.openai.com/v1"
        assert config.api_key_env == "OPENAI_API_KEY"
        assert config.openai_compatible is True

    def test_unknown_provider_raises(self):
        with pytest.raises(ProviderNotFoundError) as exc_info:
            get_provider_config("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_case_insensitive(self):
        config = get_provider_config("GROQ")
        assert config.api_key_env == "GROQ_API_KEY"

    def test_anthropic_not_openai_compatible(self):
        config = get_provider_config("anthropic")
        assert config.openai_compatible is False

    def test_all_providers_have_base_url(self):
        for name, config in PROVIDERS.items():
            assert config.base_url is not None, f"Provider '{name}' missing base_url"
            assert config.api_key_env, f"Provider '{name}' missing api_key_env"


class TestRegisterProvider:
    def test_register_new_provider(self):
        custom = ProviderConfig(
            base_url="https://my-llm.example.com/v1",
            api_key_env="MY_LLM_KEY",
        )
        register_provider("my_llm", custom)
        assert get_provider_config("my_llm") == custom

    def test_override_existing_provider(self):
        custom = ProviderConfig(
            base_url="https://custom-openai-proxy.com/v1",
            api_key_env="CUSTOM_KEY",
        )
        register_provider("openai", custom)
        assert get_provider_config("openai").base_url == "https://custom-openai-proxy.com/v1"

    def test_case_insensitive_registration(self):
        custom = ProviderConfig(base_url="https://test.com", api_key_env="TEST_KEY")
        register_provider("MyProvider", custom)
        config = get_provider_config("myprovider")
        assert config.base_url == "https://test.com"


class TestResolveApiKey:
    def test_explicit_key_takes_priority(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        config = get_provider_config("openai")
        key = resolve_api_key(config, explicit_key="explicit-key")
        assert key == "explicit-key"

    def test_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        config = get_provider_config("openai")
        key = resolve_api_key(config)
        assert key == "env-key"

    def test_raises_when_no_key_found(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = get_provider_config("openai")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            resolve_api_key(config)

    def test_empty_env_var_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        config = get_provider_config("openai")
        with pytest.raises(ValueError):
            resolve_api_key(config)

    def test_custom_env_var_takes_priority(self, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_KEY", "custom-key-value")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = get_provider_config("openai")
        key = resolve_api_key(config, custom_env_var="MY_CUSTOM_KEY")
        assert key == "custom-key-value"

    def test_custom_env_var_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("MY_MISSING_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "default-env-key")
        config = get_provider_config("openai")
        key = resolve_api_key(config, custom_env_var="MY_MISSING_KEY")
        assert key == "default-env-key"

    def test_explicit_key_beats_custom_env(self, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_KEY", "env-value")
        config = get_provider_config("openai")
        key = resolve_api_key(config, explicit_key="explicit", custom_env_var="MY_CUSTOM_KEY")
        assert key == "explicit"


class TestMaskKey:
    def test_standard_key(self):
        assert mask_key("sk-abc123def456xyz") == "****6xyz"

    def test_short_key_fully_masked(self):
        assert mask_key("short") == "****"

    def test_exactly_8_chars(self):
        assert mask_key("12345678") == "****5678"

    def test_empty_string(self):
        assert mask_key("") == "****"

    def test_none(self):
        assert mask_key(None) == "****"

    def test_preserves_only_last_4(self):
        masked = mask_key("sk-proj-very-long-api-key-abc123")
        assert masked.endswith("c123")
        assert masked.startswith("****")
        assert len(masked) == 8
