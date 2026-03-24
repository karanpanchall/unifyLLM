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
    max_tokens_param: str = "max_tokens"  # "max_tokens" or "max_completion_tokens"


# Provider catalog — all OpenAI-compatible providers work with zero adapter code.
# Non-compatible providers (anthropic, bedrock, vertex) need dedicated adapters.
# Users can add custom providers at runtime via register_provider().
PROVIDERS: dict[str, ProviderConfig] = {
    # ── Tier 1: Major providers ──────────────────────────────────────────
    "openai": ProviderConfig(
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        max_tokens_param="max_completion_tokens",
    ),
    "anthropic": ProviderConfig(
        base_url="https://api.anthropic.com",
        api_key_env="ANTHROPIC_API_KEY",
        openai_compatible=False,
    ),
    "gemini": ProviderConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
    ),
    "mistral": ProviderConfig(
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
    ),
    "deepseek": ProviderConfig(
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        models_path="/models",
    ),
    "xai": ProviderConfig(
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
    ),
    "cohere": ProviderConfig(
        base_url="https://api.cohere.com/v2",
        api_key_env="COHERE_API_KEY",
    ),

    # ── Tier 2: Inference platforms ──────────────────────────────────────
    "groq": ProviderConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        max_tokens_param="max_completion_tokens",
    ),
    "together": ProviderConfig(
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
    ),
    "fireworks": ProviderConfig(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
    ),
    "cerebras": ProviderConfig(
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
    ),
    "sambanova": ProviderConfig(
        base_url="https://api.sambanova.ai/v1",
        api_key_env="SAMBANOVA_API_KEY",
    ),
    "deepinfra": ProviderConfig(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key_env="DEEPINFRA_API_KEY",
    ),
    "hyperbolic": ProviderConfig(
        base_url="https://api.hyperbolic.xyz/v1",
        api_key_env="HYPERBOLIC_API_KEY",
    ),
    "novita": ProviderConfig(
        base_url="https://api.novita.ai/v3/openai",
        api_key_env="NOVITA_API_KEY",
    ),
    "lambda": ProviderConfig(
        base_url="https://api.lambdalabs.com/v1",
        api_key_env="LAMBDA_API_KEY",
    ),
    "featherless": ProviderConfig(
        base_url="https://api.featherless.ai/v1",
        api_key_env="FEATHERLESS_API_KEY",
    ),
    "nscale": ProviderConfig(
        base_url="https://inference.api.nscale.com/v1",
        api_key_env="NSCALE_API_KEY",
    ),

    # ── Tier 3: Aggregators / Gateways ───────────────────────────────────
    "openrouter": ProviderConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    ),
    "perplexity": ProviderConfig(
        base_url="https://api.perplexity.ai",
        api_key_env="PERPLEXITY_API_KEY",
    ),

    # ── Tier 4: Enterprise cloud ─────────────────────────────────────────
    "azure": ProviderConfig(
        base_url="",  # Must be set via base_url param or AZURE_OPENAI_BASE_URL
        api_key_env="AZURE_OPENAI_API_KEY",
    ),
    "nvidia": ProviderConfig(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
    ),
    "databricks": ProviderConfig(
        base_url="",  # Must be set via base_url param (workspace-specific)
        api_key_env="DATABRICKS_API_KEY",
    ),
    "watsonx": ProviderConfig(
        base_url="",  # Must be set via base_url param (instance-specific)
        api_key_env="WATSONX_API_KEY",
    ),
    "github": ProviderConfig(
        base_url="https://models.inference.ai.azure.com",
        api_key_env="GITHUB_TOKEN",
    ),

    # ── Tier 5: Specialty / Regional ─────────────────────────────────────
    "ai21": ProviderConfig(
        base_url="https://api.ai21.com/studio/v1",
        api_key_env="AI21_API_KEY",
    ),
    "moonshot": ProviderConfig(
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
    ),
    "dashscope": ProviderConfig(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
    ),
    "volcengine": ProviderConfig(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key_env="VOLCENGINE_API_KEY",
    ),
    "friendli": ProviderConfig(
        base_url="https://inference.friendli.ai/v1",
        api_key_env="FRIENDLI_API_KEY",
    ),
    "nebius": ProviderConfig(
        base_url="https://api.studio.nebius.ai/v1",
        api_key_env="NEBIUS_API_KEY",
    ),

    # ── AWS / GCP native (non-OpenAI-compatible, need dedicated adapters)
    "bedrock": ProviderConfig(
        base_url="us-east-1",  # Region — override with base_url param
        api_key_env="AWS_ACCESS_KEY_ID",
        openai_compatible=False,
    ),
    "vertex": ProviderConfig(
        base_url="us-central1",  # Region — override with base_url param
        api_key_env="GOOGLE_CLOUD_PROJECT",  # Project ID
        openai_compatible=False,
    ),

    # ── Tier 6: Local / Self-hosted ──────────────────────────────────────
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",  # Usually not needed
    ),
    "lmstudio": ProviderConfig(
        base_url="http://localhost:1234/v1",
        api_key_env="LMSTUDIO_API_KEY",
    ),
    "vllm": ProviderConfig(
        base_url="http://localhost:8000/v1",
        api_key_env="VLLM_API_KEY",
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
