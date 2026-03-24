"""Tests for SDK version compatibility checking."""

from unittest.mock import patch

import pytest

from bridgellm.compat import (
    _get_installed_version,
    _parse_major_version,
    check_updates,
    verify_sdk_versions,
)


class TestParseVersion:
    def test_standard_version(self):
        assert _parse_major_version("2.29.0") == 2

    def test_zero_major(self):
        assert _parse_major_version("0.79.0") == 0

    def test_single_digit(self):
        assert _parse_major_version("3") == 3

    def test_invalid_version(self):
        assert _parse_major_version("abc") is None

    def test_empty_string(self):
        assert _parse_major_version("") is None


class TestGetInstalledVersion:
    def test_installed_package(self):
        """openai should be installed in the test environment."""
        version = _get_installed_version("openai")
        assert version is not None
        assert "." in version

    def test_uninstalled_package(self):
        version = _get_installed_version("nonexistent-package-xyz")
        assert version is None


class TestVerifySdkVersions:
    def test_no_warnings_for_current_versions(self):
        """Should produce no warnings for the currently installed openai SDK."""
        warnings = verify_sdk_versions()
        # The installed openai should be within the tested range
        openai_warnings = [msg for msg in warnings if "openai" in msg]
        assert len(openai_warnings) == 0

    def test_warns_on_future_major_version(self):
        with patch("bridgellm.compat._get_installed_version") as mock_version:
            mock_version.return_value = "99.0.0"
            warnings = verify_sdk_versions()
            assert any("openai" in msg for msg in warnings)


class TestCheckUpdates:
    @pytest.mark.asyncio
    async def test_returns_none_on_network_error(self):
        with patch("bridgellm.compat.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.side_effect = OSError("no network")
            result = await check_updates()
            assert result is None
