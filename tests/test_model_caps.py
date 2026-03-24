"""Tests for reasoning model detection and parameter sanitization."""

from bridgellm._model_caps import is_reasoning_model, requires_max_completion_tokens, sanitize_params


class TestIsReasoningModel:
    def test_openai_o1(self):
        assert is_reasoning_model("o1") is True
        assert is_reasoning_model("o1-mini") is True
        assert is_reasoning_model("o1-preview") is True

    def test_openai_o3(self):
        assert is_reasoning_model("o3") is True
        assert is_reasoning_model("o3-mini") is True
        assert is_reasoning_model("o3-pro") is True

    def test_openai_o4(self):
        assert is_reasoning_model("o4-mini") is True

    def test_deepseek_reasoner(self):
        assert is_reasoning_model("deepseek-reasoner") is True

    def test_standard_models(self):
        assert is_reasoning_model("gpt-4o") is False
        assert is_reasoning_model("gpt-4o-mini") is False
        assert is_reasoning_model("gpt-5-mini") is False
        assert is_reasoning_model("claude-sonnet") is False
        assert is_reasoning_model("llama-3.3-70b") is False

    def test_models_containing_o_not_matched(self):
        assert is_reasoning_model("gpt-4o") is False
        assert is_reasoning_model("some-other-model") is False


class TestRequiresMaxCompletionTokens:
    def test_openai_reasoning(self):
        assert requires_max_completion_tokens("o1") is True
        assert requires_max_completion_tokens("o3-mini") is True
        assert requires_max_completion_tokens("o4-mini") is True

    def test_deepseek_not_openai(self):
        assert requires_max_completion_tokens("deepseek-reasoner") is False

    def test_standard_models(self):
        assert requires_max_completion_tokens("gpt-4o") is False


class TestSanitizeParams:
    def test_standard_model_unchanged(self):
        kwargs = {"model": "gpt-4o", "temperature": 0.7, "max_tokens": 100}
        result = sanitize_params("gpt-4o", kwargs)
        assert result == kwargs

    def test_reasoning_model_strips_temperature(self):
        kwargs = {"model": "o3-mini", "temperature": 0.7, "messages": []}
        result = sanitize_params("o3-mini", kwargs)
        assert "temperature" not in result
        assert "messages" in result

    def test_reasoning_model_strips_all_blocked(self):
        kwargs = {
            "model": "o1",
            "temperature": 0.5,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "logprobs": True,
            "messages": [],
        }
        result = sanitize_params("o1", kwargs)
        for blocked in ["temperature", "top_p", "frequency_penalty", "presence_penalty", "logprobs"]:
            assert blocked not in result
        assert "messages" in result

    def test_openai_reasoning_swaps_max_tokens(self):
        kwargs = {"model": "o3", "max_tokens": 500, "messages": []}
        result = sanitize_params("o3", kwargs)
        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 500

    def test_deepseek_reasoning_no_swap(self):
        kwargs = {"model": "deepseek-reasoner", "max_tokens": 500, "messages": []}
        result = sanitize_params("deepseek-reasoner", kwargs)
        assert result["max_tokens"] == 500
        assert "max_completion_tokens" not in result

    def test_does_not_mutate_input(self):
        kwargs = {"model": "o1", "temperature": 0.7}
        sanitize_params("o1", kwargs)
        assert "temperature" in kwargs
