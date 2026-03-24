"""Tests for bridgellm.errors exception hierarchy."""

from bridgellm.errors import (
    AllProvidersFailedError,
    ProviderError,
    ProviderNotFoundError,
    SDKNotInstalledError,
    BridgeLLMError,
)


class TestBridgeLLMError:
    def test_base_exception(self):
        error = BridgeLLMError("test message")
        assert str(error) == "test message"
        assert isinstance(error, Exception)


class TestProviderNotFoundError:
    def test_message_includes_available_providers(self):
        error = ProviderNotFoundError("foobar", ["openai", "groq", "anthropic"])
        assert "foobar" in str(error)
        assert "anthropic" in str(error)
        assert "groq" in str(error)
        assert "openai" in str(error)

    def test_inherits_from_base(self):
        error = ProviderNotFoundError("x", [])
        assert isinstance(error, BridgeLLMError)

    def test_attributes(self):
        error = ProviderNotFoundError("bad_provider", ["openai"])
        assert error.provider_name == "bad_provider"
        assert error.available == ["openai"]


class TestSDKNotInstalledError:
    def test_message_includes_install_command(self):
        error = SDKNotInstalledError("anthropic", "anthropic", "anthropic")
        assert "pip install bridgellm[anthropic]" in str(error)
        assert "anthropic" in str(error)

    def test_inherits_from_base(self):
        error = SDKNotInstalledError("p", "pkg", "extra")
        assert isinstance(error, BridgeLLMError)

    def test_attributes(self):
        error = SDKNotInstalledError("anthropic", "anthropic", "anthropic")
        assert error.provider_name == "anthropic"
        assert error.package_name == "anthropic"


class TestProviderError:
    def test_message_format(self):
        error = ProviderError("openai", "rate limit exceeded", status_code=429)
        assert "[openai]" in str(error)
        assert "rate limit exceeded" in str(error)

    def test_preserves_cause(self):
        original = ValueError("original cause")
        error = ProviderError("groq", "failed")
        error.__cause__ = original
        assert error.__cause__ is original

    def test_attributes(self):
        error = ProviderError("openai", "msg", status_code=500)
        assert error.provider_name == "openai"
        assert error.status_code == 500

    def test_inherits_from_base(self):
        assert isinstance(ProviderError("p", "m"), BridgeLLMError)


class TestAllProvidersFailedError:
    def test_message_includes_all_errors(self):
        errors = [ValueError("error one"), RuntimeError("error two")]
        error = AllProvidersFailedError(errors)
        assert "error one" in str(error)
        assert "error two" in str(error)

    def test_errors_attribute(self):
        errors = [ValueError("a"), ValueError("b")]
        error = AllProvidersFailedError(errors)
        assert error.errors == errors
        assert len(error.errors) == 2

    def test_inherits_from_base(self):
        assert isinstance(AllProvidersFailedError([]), BridgeLLMError)
