"""Exception hierarchy for bridgellm.

All exceptions inherit from BridgeLLMError so callers can catch broadly
or narrowly depending on their needs.
"""


class BridgeLLMError(Exception):
    """Base exception for all bridgellm errors."""


class ProviderNotFoundError(BridgeLLMError):
    """Raised when a model string references an unknown provider."""

    def __init__(self, provider_name: str, available: list[str]):
        self.provider_name = provider_name
        self.available = available
        available_list = ", ".join(sorted(available))
        super().__init__(
            f"Unknown provider '{provider_name}'. "
            f"Available providers: {available_list}. "
            f"Register new providers with bridgellm.register_provider()."
        )


class SDKNotInstalledError(BridgeLLMError):
    """Raised when a provider requires an SDK that is not installed."""

    def __init__(self, provider_name: str, package_name: str, extras_name: str):
        self.provider_name = provider_name
        self.package_name = package_name
        super().__init__(
            f"Provider '{provider_name}' requires the '{package_name}' package. "
            f"Install it with: pip install bridgellm[{extras_name}]"
        )


class ProviderError(BridgeLLMError):
    """Wraps provider-specific SDK errors into a uniform type.

    Preserves the original exception as `__cause__` for debugging.
    """

    def __init__(self, provider_name: str, message: str, status_code: int = 0):
        self.provider_name = provider_name
        self.status_code = status_code
        super().__init__(f"[{provider_name}] {message}")


class AllProvidersFailedError(BridgeLLMError):
    """Raised when every provider in a fallback chain has failed."""

    def __init__(self, errors: list[Exception]):
        self.errors = errors
        provider_errors = "; ".join(str(err) for err in errors)
        super().__init__(
            f"All providers failed. Errors: {provider_errors}"
        )
