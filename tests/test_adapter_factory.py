"""Tests for the adapter factory — provider routing to correct adapter class."""

import pytest

from bridgellm.adapters import create_adapter
from bridgellm.adapters.openai_compat import OpenAICompatAdapter
from bridgellm.errors import SDKNotInstalledError
from bridgellm.registry import ProviderConfig

try:
    from bridgellm.adapters.anthropic import AnthropicAdapter
except SDKNotInstalledError:
    AnthropicAdapter = None


OPENAI_CONFIG = ProviderConfig(
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    openai_compatible=True,
)

ANTHROPIC_CONFIG = ProviderConfig(
    base_url="https://api.anthropic.com",
    api_key_env="ANTHROPIC_API_KEY",
    openai_compatible=False,
)


class TestCreateAdapter:
    def test_openai_compatible_returns_openai_adapter(self):
        adapter = create_adapter("openai", OPENAI_CONFIG, api_key="sk-test")
        assert isinstance(adapter, OpenAICompatAdapter)

    def test_groq_returns_openai_adapter(self):
        config = ProviderConfig(
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            openai_compatible=True,
        )
        adapter = create_adapter("groq", config, api_key="gsk-test")
        assert isinstance(adapter, OpenAICompatAdapter)

    def test_anthropic_returns_anthropic_adapter(self):
        try:
            adapter = create_adapter("anthropic", ANTHROPIC_CONFIG, api_key="sk-ant-test")
            assert isinstance(adapter, AnthropicAdapter)
        except SDKNotInstalledError:
            pytest.skip("anthropic SDK not installed")

    def test_unknown_non_compatible_raises(self):
        config = ProviderConfig(
            base_url="https://unknown.com",
            api_key_env="UNKNOWN_KEY",
            openai_compatible=False,
        )
        with pytest.raises(NotImplementedError, match="no dedicated adapter"):
            create_adapter("unknown", config, api_key="key")
