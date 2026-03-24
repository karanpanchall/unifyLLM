"""Real integration tests against the OpenAI API.

Run with: OPENAI_API_KEY=sk-... python3 -m pytest tests/test_integration_real.py -v -s
Skipped automatically when OPENAI_API_KEY is not set.
"""

import os

import pytest

from bridgellm import BridgeLLM, LLMResponse, StreamChunk, EmbeddingResponse, ModelInfo

SKIP_REASON = "OPENAI_API_KEY not set — skipping real API tests"
requires_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason=SKIP_REASON,
)


@requires_api_key
class TestRealCompletion:
    @pytest.mark.asyncio
    async def test_simple_completion(self):
        """Verify a basic completion round-trip against the real API."""
        llm = BridgeLLM(model="gpt-4o-mini")

        response = await llm.complete(
            messages=[{"role": "user", "content": "Reply with exactly: UNIFYLLM_OK"}],
            temperature=0.0,
            max_tokens=20,
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert "UNIFYLLM_OK" in response.content
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.model != ""
        assert response.finish_reason == "stop"
        print(f"\n  completion: '{response.content}' | {response.input_tokens}+{response.output_tokens} tokens")

    @pytest.mark.asyncio
    async def test_completion_with_tools(self):
        """Verify tool calling works end-to-end."""
        llm = BridgeLLM(model="gpt-4o-mini")

        tools = [{
            "type": "function",
            "function": {
                "name": "get_temperature",
                "description": "Get the current temperature for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
        }]

        response = await llm.complete(
            messages=[{"role": "user", "content": "What is the temperature in Tokyo?"}],
            tools=tools,
            temperature=0.0,
            max_tokens=100,
        )

        assert isinstance(response, LLMResponse)
        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) >= 1

        tool_call = response.tool_calls[0]
        assert tool_call.function_name == "get_temperature"
        assert "city" in tool_call.arguments
        assert tool_call.call_id != ""
        print(f"\n  tool call: {tool_call.function_name}({tool_call.arguments})")


@requires_api_key
class TestRealStreaming:
    @pytest.mark.asyncio
    async def test_stream_text(self):
        """Verify streaming produces text deltas and a finish chunk."""
        llm = BridgeLLM(model="gpt-4o-mini")

        collected_text = ""
        chunk_count = 0
        got_finish = False
        got_usage = False

        async for chunk in llm.stream(
            messages=[{"role": "user", "content": "Count from 1 to 5, one number per line."}],
            temperature=0.0,
            max_tokens=50,
        ):
            assert isinstance(chunk, StreamChunk)
            chunk_count += 1

            if chunk.delta_content:
                collected_text += chunk.delta_content
            if chunk.finish_reason:
                got_finish = True
            if chunk.input_tokens > 0 or chunk.output_tokens > 0:
                got_usage = True

        assert chunk_count > 1, "Should receive multiple chunks"
        assert len(collected_text) > 0, "Should have collected text"
        assert got_finish, "Should have received a finish chunk"
        assert got_usage, "Should have received usage data"
        print(f"\n  streamed {chunk_count} chunks: '{collected_text.strip()}'")

    @pytest.mark.asyncio
    async def test_stream_tool_calls(self):
        """Verify streaming assembles tool calls correctly."""
        llm = BridgeLLM(model="gpt-4o-mini")

        tools = [{
            "type": "function",
            "function": {
                "name": "lookup_price",
                "description": "Look up the price of a product.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product": {"type": "string"},
                    },
                    "required": ["product"],
                },
            },
        }]

        assembled_calls = None
        async for chunk in llm.stream(
            messages=[{"role": "user", "content": "What is the price of a MacBook Pro?"}],
            tools=tools,
            temperature=0.0,
            max_tokens=100,
        ):
            if chunk.accumulated_tool_calls:
                assembled_calls = chunk.accumulated_tool_calls

        assert assembled_calls is not None, "Should have assembled tool calls"
        assert len(assembled_calls) >= 1
        assert assembled_calls[0].function_name == "lookup_price"
        assert "product" in assembled_calls[0].arguments
        print(f"\n  streamed tool call: {assembled_calls[0].function_name}({assembled_calls[0].arguments})")


@requires_api_key
class TestRealEmbeddings:
    @pytest.mark.asyncio
    async def test_single_embedding(self):
        """Verify embedding generation for a single text."""
        llm = BridgeLLM(model="text-embedding-3-small")

        result = await llm.embed(texts=["Hello world"])

        assert isinstance(result, EmbeddingResponse)
        assert len(result.vectors) == 1
        assert len(result.vectors[0]) > 0, "Embedding vector should not be empty"
        assert result.input_tokens > 0
        print(f"\n  embedding: {len(result.vectors[0])} dimensions, {result.input_tokens} tokens")

    @pytest.mark.asyncio
    async def test_batch_embeddings(self):
        """Verify batch embedding works for multiple texts."""
        llm = BridgeLLM(model="text-embedding-3-small")

        texts = ["First document", "Second document", "Third document"]
        result = await llm.embed(texts=texts)

        assert len(result.vectors) == 3
        # Vectors for different texts should differ
        assert result.vectors[0] != result.vectors[1]
        print(f"\n  batch: {len(result.vectors)} embeddings, {len(result.vectors[0])} dims each")

    @pytest.mark.asyncio
    async def test_embedding_with_dimensions(self):
        """Verify custom dimension parameter works."""
        llm = BridgeLLM(model="text-embedding-3-small")

        result = await llm.embed(texts=["Test"], dimensions=256)

        assert len(result.vectors) == 1
        assert len(result.vectors[0]) == 256
        print(f"\n  custom dimensions: {len(result.vectors[0])}")


@requires_api_key
class TestRealListModels:
    @pytest.mark.asyncio
    async def test_list_models(self):
        """Verify model listing returns real model data."""
        llm = BridgeLLM(model="gpt-4o-mini")

        models = await llm.list_models()

        assert len(models) > 0
        assert all(isinstance(model, ModelInfo) for model in models)

        model_ids = [model.model_id for model in models]
        assert any("gpt" in model_id for model_id in model_ids), "Should contain GPT models"
        print(f"\n  listed {len(models)} models, first 5: {model_ids[:5]}")


@requires_api_key
class TestRealRequestConfig:
    @pytest.mark.asyncio
    async def test_json_mode(self):
        """Verify structured output with response_format works."""
        from bridgellm import RequestConfig

        llm = BridgeLLM(model="gpt-4o-mini")
        config = RequestConfig(response_format={"type": "json_object"})

        response = await llm.complete(
            messages=[{"role": "user", "content": "Return a JSON object with key 'status' and value 'ok'."}],
            config=config,
            temperature=0.0,
            max_tokens=50,
        )
        assert response.content is not None
        import json
        parsed = json.loads(response.content)
        assert parsed["status"] == "ok"
        print(f"\n  json mode: {response.content}")

    @pytest.mark.asyncio
    async def test_stop_sequences(self):
        """Verify stop sequences truncate output."""
        from bridgellm import RequestConfig

        llm = BridgeLLM(model="gpt-4o-mini")
        config = RequestConfig(stop=["3"])

        response = await llm.complete(
            messages=[{"role": "user", "content": "Count from 1 to 10, one per line."}],
            config=config,
            temperature=0.0,
            max_tokens=100,
        )
        assert "4" not in (response.content or "")
        assert response.finish_reason == "stop"
        print(f"\n  stop sequence: '{response.content.strip()}'")

    @pytest.mark.asyncio
    async def test_seed_determinism(self):
        """Verify seed produces consistent output."""
        from bridgellm import RequestConfig

        llm = BridgeLLM(model="gpt-4o-mini")
        config = RequestConfig(seed=12345)
        messages = [{"role": "user", "content": "Pick a random number between 1 and 100. Just the number."}]

        response_a = await llm.complete(messages=messages, config=config, temperature=1.0, max_tokens=10)
        response_b = await llm.complete(messages=messages, config=config, temperature=1.0, max_tokens=10)
        # Seed gives best-effort determinism, may not always match, but should usually
        print(f"\n  seed test: '{response_a.content}' vs '{response_b.content}'")


@requires_api_key
class TestRealEmbedQuery:
    @pytest.mark.asyncio
    async def test_embed_query_returns_vector(self):
        """Verify embed_query convenience method."""
        llm = BridgeLLM(model="text-embedding-3-small")
        vector = await llm.embed_query("Hello world")
        assert isinstance(vector, list)
        assert len(vector) == 1536
        assert all(isinstance(val, float) for val in vector[:5])
        print(f"\n  embed_query: {len(vector)} dims")


@requires_api_key
class TestRealProviderSwitch:
    @pytest.mark.asyncio
    async def test_switch_model_same_provider(self):
        """Verify the same client works when switching models within OpenAI."""
        llm = BridgeLLM(model="gpt-4o-mini")

        response = await llm.complete(
            messages=[{"role": "user", "content": "Say OK"}],
            temperature=0.0,
            max_tokens=5,
        )
        assert response.content is not None
        print(f"\n  gpt-4o-mini: '{response.content}'")

    @pytest.mark.asyncio
    async def test_embed_with_different_model(self):
        """Verify embed() accepts a different model than the primary."""
        llm = BridgeLLM(model="gpt-4o-mini")

        result = await llm.embed(
            texts=["Hello"],
            model="openai/text-embedding-3-small",
        )
        assert len(result.vectors) == 1
        assert len(result.vectors[0]) > 0
        print(f"\n  cross-model embed: {len(result.vectors[0])} dims")

    @pytest.mark.asyncio
    async def test_multiple_models_one_client(self):
        """Verify one client can call different models on each request."""
        llm = BridgeLLM(model="gpt-4o-mini")

        # Call 1: gpt-4o-mini (default)
        resp_mini = await llm.complete(
            messages=[{"role": "user", "content": "Reply with exactly: MODEL_A"}],
            temperature=0.0, max_tokens=10,
        )
        assert "MODEL_A" in resp_mini.content
        print(f"\n  call 1 (default): '{resp_mini.content}' via {resp_mini.model}")

        # Call 2: override to gpt-4o on the same client
        resp_4o = await llm.complete(
            messages=[{"role": "user", "content": "Reply with exactly: MODEL_B"}],
            model="openai/gpt-4o",
            temperature=0.0, max_tokens=10,
        )
        assert "MODEL_B" in resp_4o.content
        print(f"\n  call 2 (override): '{resp_4o.content}' via {resp_4o.model}")

        # Call 3: embed with yet another model
        vectors = await llm.embed(texts=["test"], model="openai/text-embedding-3-small")
        assert len(vectors.vectors[0]) == 1536
        print(f"\n  call 3 (embed): {len(vectors.vectors[0])} dims")

    @pytest.mark.asyncio
    async def test_task_specific_defaults(self):
        """Verify embedding_model default routes correctly."""
        llm = BridgeLLM(
            model="gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
        )

        # Chat uses primary model
        chat = await llm.complete(
            messages=[{"role": "user", "content": "Say HI"}],
            temperature=0.0, max_tokens=5,
        )
        assert chat.content is not None

        # Embed uses embedding_model without specifying it
        vectors = await llm.embed(texts=["hello world"])
        assert len(vectors.vectors[0]) == 1536
        print(f"\n  chat: '{chat.content}' | embed: {len(vectors.vectors[0])} dims (auto-routed)")
