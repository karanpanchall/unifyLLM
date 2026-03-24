"""Tests for cross-adapter conversion utilities."""

from bridgellm.adapters._convert import (
    _merge_content,
    _repair_alternation,
    convert_messages_for_anthropic,
    openai_image_to_anthropic,
)


class TestOpenAIImageToAnthropic:
    def test_base64_data_uri(self):
        block = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ", "detail": "high"},
        }
        result = openai_image_to_anthropic(block)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/jpeg"
        assert result["source"]["data"] == "/9j/4AAQ"

    def test_png_data_uri(self):
        block = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBOR"},
        }
        result = openai_image_to_anthropic(block)
        assert result["source"]["media_type"] == "image/png"

    def test_http_url(self):
        block = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"},
        }
        result = openai_image_to_anthropic(block)
        assert result["type"] == "image"
        assert result["source"]["type"] == "url"
        assert result["source"]["url"] == "https://example.com/image.jpg"

    def test_non_image_block_returns_none(self):
        block = {"type": "text", "text": "hello"}
        assert openai_image_to_anthropic(block) is None

    def test_empty_block(self):
        assert openai_image_to_anthropic({}) is None


class TestRepairAlternation:
    def test_already_alternating(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ]
        result = _repair_alternation(messages)
        assert len(result) == 3

    def test_merges_consecutive_user(self):
        messages = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]
        result = _repair_alternation(messages)
        assert len(result) == 1
        assert "First" in result[0]["content"]
        assert "Second" in result[0]["content"]

    def test_merges_consecutive_assistant(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "assistant", "content": "How can I help?"},
        ]
        result = _repair_alternation(messages)
        assert len(result) == 2

    def test_empty_list(self):
        assert _repair_alternation([]) == []

    def test_single_message(self):
        messages = [{"role": "user", "content": "Hi"}]
        assert _repair_alternation(messages) == messages

    def test_merges_content_block_lists(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "First"}]},
            {"role": "user", "content": [{"type": "text", "text": "Second"}]},
        ]
        result = _repair_alternation(messages)
        assert len(result) == 1
        assert len(result[0]["content"]) == 2


class TestConvertMessagesForAnthropic:
    def test_converts_tool_results(self):
        """Tool results become user messages with tool_result blocks.

        Since the preceding message is also user role, alternation repair
        merges them into one user message with combined content.
        """
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "tool", "tool_call_id": "call_1", "content": "result data"},
        ]
        result = convert_messages_for_anthropic(messages)
        assert len(result) == 3
        tool_msg = result[2]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "call_1"

    def test_converts_images_in_content(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
            ],
        }]
        result = convert_messages_for_anthropic(messages)
        content = result[0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        assert content[1]["source"]["data"] == "abc"

    def test_repairs_alternation_after_tool_conversion(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
        ]
        result = convert_messages_for_anthropic(messages)
        # Both tool results become user role, should be merged
        assert all(msg["role"] == "user" for msg in result)
        assert len(result) == 1  # All merged into one user message


class TestMergeContent:
    def test_two_strings(self):
        assert _merge_content("hello", "world") == "hello\nworld"

    def test_empty_existing(self):
        assert _merge_content("", "world") == "world"

    def test_string_and_list(self):
        result = _merge_content("hello", [{"type": "text", "text": "world"}])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_two_lists(self):
        result = _merge_content(
            [{"type": "text", "text": "a"}],
            [{"type": "text", "text": "b"}],
        )
        assert len(result) == 2
