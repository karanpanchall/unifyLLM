# bridgellm

Provider-agnostic LLM client with agentic capabilities. One interface, any provider, zero wrapper libraries.

```
pip install bridgellm
```

Optional providers:
```
pip install bridgellm[anthropic]    # Anthropic Claude support
pip install bridgellm[gemini]       # Google Gemini native support
pip install bridgellm[all]          # All optional providers
```

---

## Quick Start

```python
from bridgellm import BridgeLLM

llm = BridgeLLM(model="openai/gpt-4o")

response = await llm.complete(
    messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}]
)
print(response.content)
```

Switch providers by changing the model string. No code changes needed.

```python
llm = BridgeLLM(model="groq/llama-3.3-70b")       # Groq
llm = BridgeLLM(model="anthropic/claude-sonnet-4")  # Anthropic
llm = BridgeLLM(model="together/llama-3.3-70b")     # Together AI
llm = BridgeLLM(model="deepseek/deepseek-chat")     # DeepSeek
```

---

## Supported Providers

| Provider | Model Prefix | Example |
|----------|-------------|---------|
| OpenAI | `openai/` or no prefix | `gpt-4o`, `openai/gpt-4o-mini` |
| Anthropic | `anthropic/` | `anthropic/claude-sonnet-4` |
| Groq | `groq/` | `groq/llama-3.3-70b` |
| Together AI | `together/` | `together/llama-3.3-70b` |
| Fireworks AI | `fireworks/` | `fireworks/llama-v3p1-70b` |
| DeepSeek | `deepseek/` | `deepseek/deepseek-chat` |
| Mistral | `mistral/` | `mistral/mistral-large-latest` |
| OpenRouter | `openrouter/` | `openrouter/anthropic/claude-sonnet` |
| Cerebras | `cerebras/` | `cerebras/llama-3.3-70b` |
| xAI | `xai/` | `xai/grok-2` |
| Perplexity | `perplexity/` | `perplexity/sonar-pro` |
| Google Gemini | `gemini/` | `gemini/gemini-2.5-flash` |

---

## Client Configuration

### API Keys

Keys are resolved in this order: `api_keys` dict → `api_key` single → custom env var → default env var.

```python
# Reads OPENAI_API_KEY from environment
llm = BridgeLLM(model="openai/gpt-4o")

# Explicit key
llm = BridgeLLM(model="openai/gpt-4o", api_key="sk-...")

# Multiple providers with per-provider keys
llm = BridgeLLM(
    model="openai/gpt-4o",
    api_keys={
        "openai": "sk-...",
        "groq": "gsk-...",
        "anthropic": "sk-ant-...",
    },
)

# Custom environment variable names
llm = BridgeLLM(
    model="openai/gpt-4o",
    env_var_names={
        "openai": "MY_APP_OPENAI_KEY",
        "groq": "PROD_GROQ_KEY",
    },
)
```

All providers in `api_keys` are validated at initialization. Bad keys fail immediately, not mid-request.

```python
print(llm.active_providers)  # ['openai', 'groq', 'anthropic']
```

### From Config Dict

Load from YAML, JSON, Django settings, or any config source.

```python
llm = BridgeLLM.from_config({
    "model": "openai/gpt-4o",
    "api_keys": {"openai": "sk-...", "groq": "gsk-..."},
    "system_prompt": "You are a helpful assistant.",
    "embedding_model": "openai/text-embedding-3-small",
    "fallback_models": ["groq/llama-3.3-70b"],
})
```

### System Prompt

Set a default persona that auto-injects into every call.

```python
llm = BridgeLLM(
    model="openai/gpt-4o",
    system_prompt="You are a senior Python developer. Be concise.",
)

# System prompt is prepended automatically
await llm.complete(messages=[{"role": "user", "content": "Review this code"}])

# User's own system message overrides the default
await llm.complete(messages=[
    {"role": "system", "content": "You are a creative writer."},
    {"role": "user", "content": "Write a poem"},
])
```

### Task-Specific Models

Set default models for different tasks.

```python
llm = BridgeLLM(
    model="openai/gpt-4o",                           # Chat
    embedding_model="openai/text-embedding-3-small",  # Embeddings
    tts_model="openai/tts-1",                         # Text-to-speech
    transcription_model="openai/whisper-1",            # Speech-to-text
)

await llm.complete(messages=[...])        # Uses gpt-4o
await llm.embed(texts=["hello"])          # Uses text-embedding-3-small
await llm.speak("Hello world")           # Uses tts-1
await llm.transcribe(audio_bytes)        # Uses whisper-1
```

### Per-Call Model Override

Use different models on any call without creating a new client.

```python
llm = BridgeLLM(model="openai/gpt-4o", api_keys={...})

fast = await llm.complete(messages=[...], model="groq/llama-3.3-70b")
smart = await llm.complete(messages=[...], model="anthropic/claude-sonnet-4")
cheap = await llm.embed(texts=[...], model="together/m2-bert-80M")
```

### Fallback Chain

Automatically try backup models if the primary fails.

```python
llm = BridgeLLM(
    model="openai/gpt-4o",
    fallback_models=["groq/llama-3.3-70b", "together/llama-3.3-70b"],
    api_keys={"openai": "sk-...", "groq": "gsk-...", "together": "tog-..."},
)

# If OpenAI fails → tries Groq → tries Together → raises AllProvidersFailedError
response = await llm.complete(messages=[...])
```

### Custom Base URL

Point at a proxy, gateway, or self-hosted endpoint.

```python
llm = BridgeLLM(
    model="openai/gpt-4o",
    base_url="https://my-proxy.example.com/v1",
    api_key="sk-...",
)
```

---

## Chat Completions

### Basic Completion

```python
response = await llm.complete(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=1024,
)

print(response.content)           # "Hello! How can I help?"
print(response.model)             # "gpt-4o-2024-08-06"
print(response.input_tokens)      # 12
print(response.output_tokens)     # 8
print(response.finish_reason)     # "stop"
```

### Streaming

```python
async for chunk in llm.stream(
    messages=[{"role": "user", "content": "Write a story"}],
    temperature=0.9,
):
    if chunk.delta_content:
        print(chunk.delta_content, end="", flush=True)
    if chunk.finish_reason:
        print(f"\nDone: {chunk.finish_reason}")
```

### RequestConfig — Advanced Parameters

```python
from bridgellm import RequestConfig

response = await llm.complete(
    messages=[...],
    config=RequestConfig(
        response_format={"type": "json_object"},
        stop=["END", "STOP"],
        tool_choice="required",
        top_p=0.9,
        seed=42,
        frequency_penalty=0.5,
        presence_penalty=0.3,
        service_tier="flex",
        logprobs=True,
        top_logprobs=5,
        reasoning={"effort": "high"},   # For reasoning models (o3, etc.)
    ),
)
```

### Tool / Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = await llm.complete(messages=[...], tools=tools)

for call in response.tool_calls:
    print(call.function_name)  # "get_weather"
    print(call.arguments)      # {"city": "Tokyo"}
    print(call.call_id)        # "call_abc123"
```

### Reasoning Models

Reasoning models (OpenAI o-series, DeepSeek R1) are automatically detected. Temperature and other unsupported parameters are stripped with a logged warning.

```python
llm = BridgeLLM(model="openai/o3-mini")

response = await llm.complete(
    messages=[...],
    temperature=0.7,  # Automatically stripped for reasoning models
    config=RequestConfig(reasoning={"effort": "high"}),
)

print(response.reasoning_content)  # Model's internal reasoning (if available)
```

### Audio Input / Output

```python
from bridgellm import AudioConfig, RequestConfig

response = await llm.complete(
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "What do you hear?"},
            {"type": "input_audio", "input_audio": {"data": base64_audio, "format": "wav"}},
        ]},
    ],
    config=RequestConfig(
        modalities=["text", "audio"],
        audio=AudioConfig(voice="nova", format="mp3"),
    ),
)

print(response.content)          # Text transcript
print(response.audio.data)       # Base64-encoded audio response
print(response.audio.transcript) # Audio transcript
```

---

## Embeddings

```python
# Batch embedding
result = await llm.embed(texts=["First document", "Second document"])
print(len(result.vectors))     # 2
print(len(result.vectors[0]))  # 1536

# Single query (convenience method)
vector = await llm.embed_query("search query")
print(len(vector))  # 1536

# Custom dimensions
result = await llm.embed(texts=["hello"], dimensions=256)
```

---

## Text-to-Speech

```python
result = await llm.speak(
    text="Hello world, this is bridgellm speaking.",
    voice="nova",
    response_format="mp3",
    speed=1.0,
)

with open("output.mp3", "wb") as f:
    f.write(result.audio_data)
```

---

## Speech-to-Text

```python
with open("recording.wav", "rb") as f:
    audio_data = f.read()

result = await llm.transcribe(audio_data=audio_data, language="en")
print(result.text)      # "Hello world"
print(result.duration)  # 2.5
```

---

## @tool Decorator

Auto-generate OpenAI tool definitions from Python functions.

```python
from bridgellm import tool

@tool
async def get_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather for a city.

    Args:
        city: City name (e.g., "New York")
        unit: Temperature unit
    """
    return f"72°F in {city}"

# Auto-generated tool definition
print(get_weather.as_openai_tool())
# {
#   "type": "function",
#   "function": {
#     "name": "get_weather",
#     "description": "Get current weather for a city.",
#     "parameters": {
#       "type": "object",
#       "properties": {
#         "city": {"type": "string", "description": "City name (e.g., \"New York\")"},
#         "unit": {"type": "string", "description": "Temperature unit", "default": "celsius"}
#       },
#       "required": ["city"]
#     }
#   }
# }
```

Custom name and description:

```python
@tool(name="search", description="Search the knowledge base")
async def search_docs(query: str) -> str:
    return "results..."
```

Tools can optionally receive a context object:

```python
@tool
async def create_note(title: str, content: str, context: dict = None) -> str:
    """Create a note for the user."""
    user_id = context["user_id"]
    # ... create note ...
    return f"Created note: {title}"
```

---

## Tool Registry

```python
from bridgellm import ToolRegistry

registry = ToolRegistry([get_weather, search_docs, create_note])

# Get OpenAI-compatible definitions for all tools
tool_defs = registry.as_openai_tools()

# Execute a tool by name
result = await registry.execute("get_weather", {"city": "NYC"}, context={"user_id": "123"})

# List registered tools
print(registry.tool_names)  # ["get_weather", "search_docs", "create_note"]
```

---

## Agent Loop

Automated tool-calling loop: LLM decides which tools to call, results are fed back, repeat until the model responds with text.

```python
from bridgellm import BridgeLLM, AgentLoop, tool

@tool
async def search(query: str) -> str:
    """Search for information."""
    return f"Found results for: {query}"

@tool
async def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

llm = BridgeLLM(model="openai/gpt-4o", system_prompt="You are a helpful assistant.")
agent = AgentLoop(llm=llm, tools=[search, calculate])

async for event in agent.run(messages=[{"role": "user", "content": "What is 2+2?"}]):
    match event.type:
        case "text_delta":
            print(event.content, end="")
        case "tool_start":
            print(f"\n  Calling {event.tool_name}({event.tool_args})")
        case "tool_result":
            print(f"  → {event.content}")
        case "tool_error":
            print(f"  Error: {event.content}")
        case "done":
            print(f"\nFinished: {event.total_iterations} iterations, "
                  f"{event.total_input_tokens + event.total_output_tokens} tokens")
```

### Agent Configuration

Every option is configurable. Nothing is forced.

```python
from bridgellm import AgentLoop, RetryPolicy, RequestConfig

agent = AgentLoop(
    llm=llm,
    tools=[search, calculate],

    # Iteration control
    max_iterations=15,               # Max LLM calls (default: 25)
    timeout_seconds=120.0,           # Wall-clock limit (default: None)
    max_total_tokens=50000,          # Total token budget (default: None)

    # Tool execution
    tool_timeout_seconds=30.0,       # Per-tool timeout (default: 120)
    parallel_tool_calls=True,        # Run tools concurrently (default: True)
    on_tool_error="skip",            # "skip", "stop", or "raise" (default: "skip")
    max_tool_failures=2,             # Disable tool after N failures (default: 3)

    # Context for tools
    context={"user_id": "123", "db": session},  # Passed to tool handlers

    # LLM configuration
    config=RequestConfig(response_format={"type": "json_object"}),
    streaming=True,                  # Use stream() or complete() (default: True)

    # Optional retry policy for transient LLM errors
    retry_policy=RetryPolicy(
        max_retries=2,
        backoff_seconds=2.0,
        backoff_multiplier=2.0,
    ),
)
```

### Non-Streaming Mode

```python
agent = AgentLoop(llm=llm, tools=[...], streaming=False)

async for event in agent.run(messages=[...]):
    if event.type == "done":
        print(f"Completed in {event.total_iterations} iterations")
```

### Custom Retry Logic

```python
agent = AgentLoop(
    llm=llm,
    tools=[...],
    retry_policy=RetryPolicy(
        max_retries=3,
        retryable_check=lambda exc: "rate_limit" in str(exc).lower(),
    ),
)
```

### Stop Conditions

The loop stops when any of these conditions is met:

| Condition | Finish Reason |
|-----------|---------------|
| Model responds with text only (no tool calls) | `"stop"` |
| `max_iterations` reached | `"max_iterations"` |
| `timeout_seconds` exceeded | `"timeout"` |
| `max_total_tokens` exceeded | `"max_tokens"` |
| Tool error with `on_tool_error="stop"` | `"tool_error"` |
| Unrecoverable LLM error | `"error"` event |

### Event Types

| Event | Fields | When |
|-------|--------|------|
| `iteration_start` | `iteration` | Each LLM call begins |
| `text_delta` | `content` | Streaming text fragment (streaming mode) |
| `reasoning_delta` | `content` | Thinking/reasoning fragment |
| `tool_start` | `tool_name`, `tool_args`, `tool_call_id` | Tool about to execute |
| `tool_result` | `tool_name`, `content`, `tool_call_id` | Tool finished |
| `tool_error` | `tool_name`, `content`, `tool_call_id` | Tool failed |
| `usage` | `input_tokens`, `output_tokens`, `iteration` | Per-iteration token usage |
| `done` | `finish_reason`, `total_iterations`, `total_input_tokens`, `total_output_tokens` | Loop complete |
| `error` | `content` | Unrecoverable error |

---

## Token Budget Management

Trim messages to fit within a context window. No hardcoded model limits — you provide the numbers.

```python
from bridgellm import TokenBudget

budget = TokenBudget(context_window=128000, headroom=4096)

# Estimate tokens
tokens = budget.estimate_tokens(messages, tools=tool_defs)
print(f"Estimated: {tokens} tokens")

# Trim to fit (removes oldest messages first, preserves system prompt)
trimmed = budget.trim_messages(messages, tools=tool_defs, preserve_first_n=2)
```

Tool call / tool result pairs are kept intact — orphaned messages are automatically removed.

---

## Model Discovery

Query provider APIs for available models and metadata.

```python
# List all models from a provider
models = await llm.list_models()
for model in models:
    print(f"{model.model_id}: context={model.context_window}, max_output={model.max_output_tokens}")

# List from a specific provider
groq_models = await llm.list_models(provider="groq")

# Get info for a specific model
info = await llm.get_model_info("anthropic/claude-sonnet-4")
if info:
    print(f"Context: {info.context_window}")       # 200000 (from Anthropic API)
    print(f"Max output: {info.max_output_tokens}")  # 16384
    print(f"Capabilities: {info.capabilities}")      # {"vision": ..., "thinking": ...}
```

Metadata fields are populated only when the provider API exposes them. OpenAI and DeepSeek return minimal metadata; Anthropic, Groq, Together, Mistral, and OpenRouter return rich metadata including context windows and capabilities.

---

## Custom Providers

Register self-hosted, private, or new providers at runtime.

```python
from bridgellm import register_provider, ProviderConfig

register_provider("my_vllm", ProviderConfig(
    base_url="http://localhost:8000/v1",
    api_key_env="MY_VLLM_KEY",
))

llm = BridgeLLM(model="my_vllm/llama-3-70b")
```

---

## Error Handling

All errors inherit from `BridgeLLMError` for broad catching, with specific types for narrow handling.

```python
from bridgellm import BridgeLLMError, ProviderError, ProviderNotFoundError, AllProvidersFailedError

try:
    response = await llm.complete(messages=[...])
except ProviderNotFoundError as exc:
    print(f"Unknown provider: {exc.provider_name}")
    print(f"Available: {exc.available}")
except AllProvidersFailedError as exc:
    print(f"All providers failed: {exc.errors}")
except ProviderError as exc:
    print(f"[{exc.provider_name}] {exc}")
    print(f"Status: {exc.status_code}")
except BridgeLLMError as exc:
    print(f"bridgellm error: {exc}")
```

### Key Masking

API keys are never logged or exposed in error messages.

```python
from bridgellm import mask_key

print(mask_key("sk-proj-abc123def456"))  # "****f456"
```

---

## SDK Version Safety

bridgellm checks installed SDK versions at import time and warns if they are outside the tested range.

```
⚠ bridgellm 0.1.0 was tested with openai<=2.x, but you have 3.1.0.
  Run: pip install --upgrade bridgellm
```

Check for library updates programmatically:

```python
from bridgellm import check_updates

message = await check_updates()
if message:
    print(message)  # "bridgellm 0.2.0 is available..."
```

---

## Concurrency

bridgellm is safe for concurrent use. One client handles multiple simultaneous calls with connection pooling and thread-safe adapter caching.

```python
import asyncio

llm = BridgeLLM(model="openai/gpt-4o")

results = await asyncio.gather(
    llm.complete(messages=[{"role": "user", "content": "Question 1"}]),
    llm.complete(messages=[{"role": "user", "content": "Question 2"}]),
    llm.complete(messages=[{"role": "user", "content": "Question 3"}], model="groq/llama-3"),
    llm.embed(texts=["hello", "world"]),
)
```
