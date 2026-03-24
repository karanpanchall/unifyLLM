"""bridgellm — Provider-agnostic LLM client with agentic capabilities.

One interface, any provider, zero wrapper libraries.

Usage:
    from bridgellm import BridgeLLM, tool, AgentLoop

    @tool
    async def get_weather(city: str) -> str:
        '''Get weather for a city.'''
        return "72°F"

    llm = BridgeLLM(model="openai/gpt-4o")
    agent = AgentLoop(llm=llm, tools=[get_weather])

    async for event in agent.run(messages=[...]):
        print(event)
"""

from .agent import AgentEvent, AgentLoop, RetryPolicy
from .budget import TokenBudget
from .client import BridgeLLM
from .compat import check_updates, verify_sdk_versions
from .errors import (
    AllProvidersFailedError,
    ProviderError,
    ProviderNotFoundError,
    SDKNotInstalledError,
    BridgeLLMError,
)
from .models import (
    AudioConfig,
    AudioData,
    EmbeddingResponse,
    LLMResponse,
    ModelInfo,
    RequestConfig,
    StreamChunk,
    TTSResponse,
    ToolCall,
    TranscriptionResponse,
)
from .registry import ProviderConfig, mask_key, register_provider
from .tools import ToolDefinition, ToolRegistry, tool

__all__ = [
    # Client
    "BridgeLLM",
    # Agent
    "AgentLoop",
    "AgentEvent",
    "RetryPolicy",
    # Tools
    "tool",
    "ToolDefinition",
    "ToolRegistry",
    # Budget
    "TokenBudget",
    # Config
    "RequestConfig",
    "AudioConfig",
    "ProviderConfig",
    "register_provider",
    # Response types
    "LLMResponse",
    "StreamChunk",
    "ToolCall",
    "AudioData",
    "EmbeddingResponse",
    "TTSResponse",
    "TranscriptionResponse",
    "ModelInfo",
    # Errors
    "BridgeLLMError",
    "ProviderError",
    "ProviderNotFoundError",
    "SDKNotInstalledError",
    "AllProvidersFailedError",
    # Utilities
    "mask_key",
    "check_updates",
    "verify_sdk_versions",
]

verify_sdk_versions()
