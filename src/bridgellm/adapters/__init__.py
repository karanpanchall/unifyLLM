"""Adapter factory — resolves provider name to the correct adapter class."""

from ..registry import ProviderConfig
from .base import LLMAdapter


def create_adapter(provider_name: str, config: ProviderConfig, api_key: str) -> LLMAdapter:
    """Instantiate the correct adapter for a given provider.

    OpenAI-compatible providers share a single adapter class.
    Non-compatible providers use dedicated adapters.
    """
    if config.openai_compatible:
        from .openai_compat import OpenAICompatAdapter
        return OpenAICompatAdapter(provider_name=provider_name, config=config, api_key=api_key)

    if provider_name == "anthropic":
        from .anthropic import AnthropicAdapter
        return AnthropicAdapter(config=config, api_key=api_key)

    if provider_name == "bedrock":
        from .bedrock import BedrockAdapter
        return BedrockAdapter(config=config, api_key=api_key)

    if provider_name == "vertex":
        from .vertex import VertexAdapter
        return VertexAdapter(config=config, api_key=api_key)

    raise NotImplementedError(
        f"Provider '{provider_name}' is marked as non-OpenAI-compatible "
        f"but has no dedicated adapter."
    )
