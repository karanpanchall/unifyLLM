"""Tests for TokenBudget — message trimming and tool-pair safety."""

import pytest

from bridgellm.budget import TokenBudget, _repair_tool_pairs


class TestTokenBudgetInit:
    def test_valid_init(self):
        budget = TokenBudget(context_window=128000, headroom=4096)
        assert budget.budget == 128000 - 4096

    def test_zero_context_raises(self):
        with pytest.raises(ValueError, match="positive"):
            TokenBudget(context_window=0)

    def test_negative_headroom_raises(self):
        with pytest.raises(ValueError, match="negative"):
            TokenBudget(context_window=128000, headroom=-1)


class TestEstimateTokens:
    def test_simple_message(self):
        budget = TokenBudget(context_window=128000)
        tokens = budget.estimate_tokens([{"role": "user", "content": "Hello world"}])
        assert tokens > 0

    def test_empty_messages(self):
        budget = TokenBudget(context_window=128000)
        assert budget.estimate_tokens([]) == 0

    def test_with_tools(self):
        budget = TokenBudget(context_window=128000)
        tools = [{"function": {"name": "search", "description": "Search docs", "parameters": {}}}]
        tokens_no_tools = budget.estimate_tokens([{"role": "user", "content": "Hi"}])
        tokens_with_tools = budget.estimate_tokens([{"role": "user", "content": "Hi"}], tools)
        assert tokens_with_tools > tokens_no_tools

    def test_content_block_list(self):
        budget = TokenBudget(context_window=128000)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        ]}]
        tokens = budget.estimate_tokens(messages)
        assert tokens > 170  # At least the image token estimate


class TestTrimMessages:
    def test_no_trim_when_within_budget(self):
        budget = TokenBudget(context_window=128000)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = budget.trim_messages(messages)
        assert len(result) == 2

    def test_trims_oldest_first(self):
        budget = TokenBudget(context_window=50, headroom=10)
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Old message " * 30},
            {"role": "assistant", "content": "Old reply " * 30},
            {"role": "user", "content": "Recent question"},
        ]
        result = budget.trim_messages(messages, preserve_first_n=1)
        assert result[0]["role"] == "system"
        assert result[-1]["content"] == "Recent question"
        assert len(result) < len(messages)

    def test_preserves_first_n(self):
        budget = TokenBudget(context_window=100, headroom=20)
        messages = [
            {"role": "system", "content": "Keep me."},
            {"role": "user", "content": "Long " * 50},
            {"role": "assistant", "content": "Reply " * 50},
            {"role": "user", "content": "Recent"},
        ]
        result = budget.trim_messages(messages, preserve_first_n=1)
        assert result[0]["content"] == "Keep me."


class TestRepairToolPairs:
    def test_removes_orphaned_tool_results(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "tool", "tool_call_id": "orphan_1", "content": "result"},
        ]
        result = _repair_tool_pairs(messages)
        assert len(result) == 1  # tool result removed

    def test_keeps_paired_tool_results(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        result = _repair_tool_pairs(messages)
        assert len(result) == 3

    def test_empty_messages(self):
        assert _repair_tool_pairs([]) == []

    def test_no_tool_messages(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = _repair_tool_pairs(messages)
        assert len(result) == 2
