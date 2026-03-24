"""Provider registry — maps provider names to base URLs and API key env vars.

Adding a new OpenAI-compatible provider requires one dict entry. No code changes.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProviderConfig:
    """Connection details for a single LLM provider."""

    base_url: str
    api_key_env: str
    models_path: str = "/v1/models"
    openai_compatible: bool = True


# Provider catalog. Add new OpenAI-compatible providers here.
PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    "groq": ProviderConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    ),
    "together": ProviderConfig(
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
    ),
    "fireworks": ProviderConfig(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
    ),
    "deepseek": ProviderConfig(
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        models_path="/models",
    ),
    "mistral": ProviderConfig(
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
    ),
    "openrouter": ProviderConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    ),
    "cerebras": ProviderConfig(
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
    ),
    "xai": ProviderConfig(
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
    ),
    "perplexity": ProviderConfig(
        base_url="https://api.perplexity.ai",
        api_key_env="PERPLEXITY_API_KEY",
    ),
    "gemini": ProviderConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
    ),
    "anthropic": ProviderConfig(
        base_url="https://api.anthropic.com",
        api_key_env="ANTHROPIC_API_KEY",
        openai_compatible=False,
    ),
}


def mask_key(key: str) -> str:
    """Mask an API key for safe display in logs and errors.

    Shows only the last 4 characters, replacing the rest with asterisks.
    Keys shorter than 8 characters are fully masked.
    """
    if not key or len(key) < 8:
        return "****"
    return f"****{key[-4:]}"


def parse_model_string(model_string: str) -> tuple[str, str]:
    """Split 'provider/model-name' into (provider, model).

    Falls back to 'openai' when no provider prefix is present.

    Examples:
        'groq/llama-3.3-70b'      -> ('groq', 'llama-3.3-70b')
        'gpt-5-mini'              -> ('openai', 'gpt-5-mini')
        'anthropic/claude-sonnet'  -> ('anthropic', 'claude-sonnet')
    """
    if "/" in model_string:
        provider, model = model_string.split("/", 1)
        if not provider:
            raise ValueError(f"Invalid model string '{model_string}': provider name cannot be empty")
        if not model:
            raise ValueError(f"Invalid model string '{model_string}': model name cannot be empty")
        return provider.lower(), model
    return "openai", model_string


def get_provider_config(provider_name: str) -> ProviderConfig:
    """Look up provider configuration by name.

    Raises ProviderNotFoundError with available providers on miss.
    """
    from .errors import ProviderNotFoundError

    normalized = provider_name.lower()
    if normalized not in PROVIDERS:
        raise ProviderNotFoundError(provider_name, list(PROVIDERS.keys()))
    return PROVIDERS[normalized]


def register_provider(name: str, config: ProviderConfig) -> None:
    """Register a custom provider at runtime.

    Use this to add private or self-hosted endpoints without modifying source.
    """
    PROVIDERS[name.lower()] = config


def resolve_api_key(
    config: ProviderConfig,
    explicit_key: Optional[str] = None,
    custom_env_var: Optional[str] = None,
) -> str:
    """Resolve an API key from explicit value, custom env var, or default env var.

    Priority: explicit_key > custom_env_var > config.api_key_env (default).
    Raises ValueError with the expected env var name when no key is found.
    """
    if explicit_key:
        return explicit_key

    # Try custom env var name first, then the default
    env_var_name = custom_env_var or config.api_key_env
    env_value = os.environ.get(env_var_name, "")
    if env_value:
        return env_value

    # If custom was tried and failed, also try the default
    if custom_env_var and custom_env_var != config.api_key_env:
        env_value = os.environ.get(config.api_key_env, "")
        if env_value:
            return env_value

    raise ValueError(
        f"No API key found for provider. "
        f"Set the {env_var_name} environment variable or pass api_key explicitly."
    )
