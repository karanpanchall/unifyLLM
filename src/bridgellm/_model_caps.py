"""Model capability detection — identifies reasoning models and filters unsupported params.

Reasoning models (OpenAI o-series, DeepSeek R1) reject standard sampling
parameters like temperature. This module detects them and strips unsafe
params with logged warnings instead of letting requests fail.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns that identify reasoning models requiring special handling.
# These reject temperature, top_p, frequency_penalty, presence_penalty.
_REASONING_PATTERNS = [
    re.compile(r"^o[134]\b"),       # o1, o3, o3-mini, o3-pro, o4-mini
    re.compile(r"deepseek-reasoner"),
]

# Params that reasoning models reject outright (400 error on OpenAI).
_REASONING_BLOCKED_PARAMS = {"temperature", "top_p", "frequency_penalty", "presence_penalty", "logprobs", "top_logprobs", "logit_bias"}

# OpenAI reasoning models use max_completion_tokens, not max_tokens.
_OPENAI_REASONING_PATTERNS = [
    re.compile(r"^o[134]\b"),
]


def is_reasoning_model(model_name: str) -> bool:
    """Check if a model is a reasoning model that restricts sampling params."""
    return any(pattern.search(model_name) for pattern in _REASONING_PATTERNS)


def requires_max_completion_tokens(model_name: str) -> bool:
    """Check if model uses max_completion_tokens instead of max_tokens."""
    return any(pattern.search(model_name) for pattern in _OPENAI_REASONING_PATTERNS)


def sanitize_params(model_name: str, kwargs: dict) -> dict:
    """Remove unsupported params for reasoning models, logging warnings.

    Returns a new dict with blocked params stripped. Does not mutate the input.
    """
    if not is_reasoning_model(model_name):
        return kwargs

    sanitized = dict(kwargs)
    for param in _REASONING_BLOCKED_PARAMS:
        if param in sanitized:
            logger.warning(
                "'%s' is not supported by reasoning model '%s', removing",
                param, model_name,
            )
            del sanitized[param]

    # Swap max_tokens → max_completion_tokens for OpenAI reasoning models
    if requires_max_completion_tokens(model_name) and "max_tokens" in sanitized:
        sanitized["max_completion_tokens"] = sanitized.pop("max_tokens")

    return sanitized
