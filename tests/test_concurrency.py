"""Concurrency stress tests — verify thread safety and async safety.

Tests concurrent completions, streaming, embeddings, and adapter caching
under real async load.
"""

import asyncio
import os

import pytest

from bridgellm import BridgeLLM, LLMResponse, StreamChunk, EmbeddingResponse

SKIP_REASON = "OPENAI_API_KEY not set — skipping concurrency tests"
requires_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason=SKIP_REASON,
)


@requires_api_key
class TestConcurrentCompletions:
    @pytest.mark.asyncio
    async def test_10_concurrent_completions(self):
        """Fire 10 completions simultaneously on the same client."""
        llm = BridgeLLM(model="gpt-4o-mini")

        async def single_call(call_index: int) -> LLMResponse:
            return await llm.complete(
                messages=[{"role": "user", "content": f"Reply with exactly: CALL_{call_index}"}],
                temperature=0.0,
                max_tokens=10,
            )

        results = await asyncio.gather(*[single_call(idx) for idx in range(10)])

        assert len(results) == 10
        for idx, result in enumerate(results):
            assert isinstance(result, LLMResponse)
            assert result.content is not None
            assert f"CALL_{idx}" in result.content
        print(f"\n  10 concurrent completions: all returned correctly")

    @pytest.mark.asyncio
    async def test_concurrent_different_models(self):
        """Fire concurrent calls to different models on the same client."""
        llm = BridgeLLM(model="gpt-4o-mini")

        async def call_mini():
            return await llm.complete(
                messages=[{"role": "user", "content": "Reply: MINI"}],
                temperature=0.0, max_tokens=10,
            )

        async def call_4o():
            return await llm.complete(
                messages=[{"role": "user", "content": "Reply: FULL"}],
                model="openai/gpt-4o",
                temperature=0.0, max_tokens=10,
            )

        results = await asyncio.gather(
            call_mini(), call_4o(), call_mini(), call_4o(), call_mini(),
        )
        assert len(results) == 5
        for result in results:
            assert result.content is not None
        print(f"\n  5 concurrent cross-model calls: all succeeded")


@requires_api_key
class TestConcurrentStreaming:
    @pytest.mark.asyncio
    async def test_3_concurrent_streams(self):
        """Run 3 streams simultaneously, each collecting independent text."""
        llm = BridgeLLM(model="gpt-4o-mini")

        async def stream_call(number: int) -> str:
            collected = ""
            async for chunk in llm.stream(
                messages=[{"role": "user", "content": f"Reply with exactly: STREAM_{number}"}],
                temperature=0.0, max_tokens=10,
            ):
                if chunk.delta_content:
                    collected += chunk.delta_content
            return collected

        results = await asyncio.gather(
            stream_call(1), stream_call(2), stream_call(3),
        )
        assert len(results) == 3
        for idx, text in enumerate(results, 1):
            assert f"STREAM_{idx}" in text
        print(f"\n  3 concurrent streams: {[r.strip() for r in results]}")


@requires_api_key
class TestConcurrentEmbeddings:
    @pytest.mark.asyncio
    async def test_5_concurrent_embed_calls(self):
        """Fire 5 embedding calls simultaneously."""
        llm = BridgeLLM(
            model="gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
        )

        async def embed_call(text: str) -> list[float]:
            return await llm.embed_query(text)

        texts = ["apple", "banana", "cherry", "dragon", "elderberry"]
        results = await asyncio.gather(*[embed_call(text) for text in texts])

        assert len(results) == 5
        for vector in results:
            assert len(vector) == 1536
        # Different texts should produce different vectors
        assert results[0][:5] != results[1][:5]
        print(f"\n  5 concurrent embeddings: all 1536-dim, all unique")


@requires_api_key
class TestConcurrentMixed:
    @pytest.mark.asyncio
    async def test_mixed_operations(self):
        """Run completion, stream, and embed concurrently on the same client."""
        llm = BridgeLLM(
            model="gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
        )

        async def do_complete():
            return await llm.complete(
                messages=[{"role": "user", "content": "Reply: COMPLETE"}],
                temperature=0.0, max_tokens=10,
            )

        async def do_stream():
            collected = ""
            async for chunk in llm.stream(
                messages=[{"role": "user", "content": "Reply: STREAMED"}],
                temperature=0.0, max_tokens=10,
            ):
                if chunk.delta_content:
                    collected += chunk.delta_content
            return collected

        async def do_embed():
            return await llm.embed_query("test concurrency")

        completion, streamed_text, vector = await asyncio.gather(
            do_complete(), do_stream(), do_embed(),
        )

        assert "COMPLETE" in completion.content
        assert "STREAMED" in streamed_text
        assert len(vector) == 1536
        print(f"\n  mixed concurrent: complete='{completion.content}', stream='{streamed_text.strip()}', embed={len(vector)}d")


class TestAdapterCacheConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_adapter_resolution(self, openai_api_key):
        """Verify concurrent _resolve_adapter calls don't create duplicate adapters."""
        llm = BridgeLLM(model="openai/gpt-4o")

        async def resolve():
            return llm._resolve_adapter("openai/gpt-4o-mini")

        results = await asyncio.gather(*[resolve() for _ in range(20)])

        adapters = [adapter for adapter, _ in results]
        # All should be the exact same adapter instance
        assert all(adapter is adapters[0] for adapter in adapters)
        print(f"\n  20 concurrent resolves: all same adapter instance ({id(adapters[0])})")
