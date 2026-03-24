"""SDK version compatibility checking.

Warns at import time if installed provider SDKs are outside the tested range.
Provides a check_updates() helper for manual verification.
"""

import importlib.metadata
import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_UNIFYLLM_VERSION = "0.3.0"


@dataclass(frozen=True)
class VersionRange:
    """Minimum and maximum tested major version for a provider SDK."""

    package: str
    min_version: str
    max_major: int


# Tested SDK version ranges. Update these when testing against new releases.
TESTED_RANGES: list[VersionRange] = [
    VersionRange(package="openai", min_version="1.50.0", max_major=2),
    VersionRange(package="anthropic", min_version="0.40.0", max_major=0),
    VersionRange(package="google-genai", min_version="1.0.0", max_major=1),
]


def verify_sdk_versions() -> list[str]:
    """Check installed SDK versions against tested ranges.

    Returns a list of warning messages for any SDK outside the tested range.
    Called at import time to alert users early.
    """
    warning_messages: list[str] = []

    for version_range in TESTED_RANGES:
        installed = _get_installed_version(version_range.package)
        if installed is None:
            continue  # SDK not installed, skip

        major = _parse_major_version(installed)
        if major is None:
            continue

        if major > version_range.max_major:
            msg = (
                f"bridgellm {_UNIFYLLM_VERSION} was tested with "
                f"{version_range.package}<={version_range.max_major}.x, "
                f"but you have {installed}. "
                f"Run: pip install --upgrade bridgellm"
            )
            warning_messages.append(msg)
            warnings.warn(msg, stacklevel=2)

    return warning_messages


async def check_updates() -> Optional[str]:
    """Check PyPI for a newer version of bridgellm.

    Returns a message if an update is available, or None if current.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as http_client:
            response = await http_client.get("https://pypi.org/pypi/bridgellm/json")
            if response.status_code != 200:
                return None
            latest = response.json().get("info", {}).get("version", "")
            if latest and latest != _UNIFYLLM_VERSION:
                return (
                    f"bridgellm {latest} is available (you have {_UNIFYLLM_VERSION}). "
                    f"Run: pip install --upgrade bridgellm"
                )
    except (httpx.RequestError, httpx.HTTPStatusError, OSError, ValueError):
        logger.debug("Failed to check PyPI for bridgellm updates", exc_info=True)
    return None


def _get_installed_version(package_name: str) -> Optional[str]:
    """Read the installed version of a package, or None if not installed."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parse_major_version(version_string: str) -> Optional[int]:
    """Extract the major version number from a version string."""
    try:
        return int(version_string.split(".")[0])
    except (ValueError, IndexError):
        return None
