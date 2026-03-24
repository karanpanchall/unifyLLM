"""Adapter for Google Vertex AI.

Uses the OpenAI-compatible endpoint that Vertex AI provides, authenticated
with Google Cloud credentials instead of an API key.
Requires: pip install google-auth

Authentication: Uses Application Default Credentials (ADC).
Set up with: gcloud auth application-default login
Or set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON path.
"""

import asyncio
import logging
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from ..errors import ProviderError, SDKNotInstalledError
from ..models import EmbeddingResponse, LLMResponse, ModelInfo, RequestConfig, StreamChunk
from ..registry import ProviderConfig
from .openai_compat import OpenAICompatAdapter

logger = logging.getLogger(__name__)
_PROVIDER = "vertex"


def _get_vertex_token() -> str:
    """Get a short-lived access token from Google ADC."""
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as exc:
        raise SDKNotInstalledError(_PROVIDER, "google-auth", "vertex") from exc

    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


class VertexAdapter(OpenAICompatAdapter):
    """Vertex AI adapter using their OpenAI-compatible endpoint.

    Inherits all functionality from OpenAICompatAdapter — only overrides
    initialization to handle Google Cloud authentication.
    """

    def __init__(self, config: ProviderConfig, api_key: str):
        # api_key param is the project_id for Vertex
        project_id = api_key
        region = config.base_url or "us-central1"

        base_url = (
            f"https://{region}-aiplatform.googleapis.com/v1/"
            f"projects/{project_id}/locations/{region}/endpoints/openapi"
        )

        token = _get_vertex_token()
        self._provider = _PROVIDER
        self._client = AsyncOpenAI(base_url=base_url, api_key=token)
        self._project_id = project_id
        self._region = region
