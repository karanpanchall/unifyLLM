"""Shared fixtures for bridgellm tests."""

import pytest


@pytest.fixture(autouse=True)
def _clean_provider_registry():
    """Reset the provider registry after each test to prevent leaks."""
    from bridgellm.registry import PROVIDERS

    original = dict(PROVIDERS)
    yield
    PROVIDERS.clear()
    PROVIDERS.update(original)


@pytest.fixture
def openai_api_key(monkeypatch):
    """Set a fake OPENAI_API_KEY for tests that need it."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-unit-tests")
    return "sk-test-key-for-unit-tests"
