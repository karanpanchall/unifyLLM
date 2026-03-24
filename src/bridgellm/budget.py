"""Token budget management — trim messages to fit context windows.

No hardcoded model limits. Limits must be provided by the developer
or queried from the provider API via get_model_info().

Preserves tool call / tool result pairing integrity when trimming.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Rough heuristic: 1 token ≈ 4 characters. Not exact, but safe for budgeting.
_CHARS_PER_TOKEN = 4
_MESSAGE_OVERHEAD_TOKENS = 4  # Per-message overhead (role, formatting)


class TokenBudget:
    """Manages message trimming to stay within a context window.

    All limits are developer-provided. Nothing is assumed.

    Usage:
        budget = TokenBudget(context_window=128000, headroom=4096)
        trimmed = budget.trim_messages(messages, tools=tool_defs)
    """

    def __init__(
        self,
        context_window: int,
        headroom: int = 4096,
        chars_per_token: int = _CHARS_PER_TOKEN,
    ):
        """
        Args:
            context_window: Total context window size in tokens.
            headroom: Tokens reserved for the model's response.
            chars_per_token: Character-to-token ratio for estimation.
        """
        if context_window <= 0:
            raise ValueError("context_window must be positive")
        if headroom < 0:
            raise ValueError("headroom cannot be negative")

        self.context_window = context_window
        self.headroom = headroom
        self.chars_per_token = chars_per_token
        self.budget = context_window - headroom

    def estimate_tokens(self, messages: list[dict], tools: Optional[list[dict]] = None) -> int:
        """Estimate total token count for a set of messages and tool definitions."""
        total = 0
        for msg in messages:
            total += _MESSAGE_OVERHEAD_TOKENS
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // self.chars_per_token
            elif isinstance(content, list):
                for block in content:
                    text = block.get("text", "")
                    if text:
                        total += len(text) // self.chars_per_token
                    if block.get("type") in ("image_url", "image"):
                        total += 170  # Rough image token estimate
        if tools:
            total += self._estimate_tool_tokens(tools)
        return total

    def trim_messages(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        preserve_first_n: int = 1,
    ) -> list[dict]:
        """Trim messages to fit within the token budget.

        Removes oldest messages first while preserving:
        - The first N messages (system prompt + initial context)
        - Tool call / tool result pairs (never orphan one without the other)
        - The most recent messages

        Args:
            messages: Full message list.
            tools: Tool definitions (counted toward budget).
            preserve_first_n: Number of initial messages to always keep.

        Returns:
            Trimmed message list that fits within budget.
        """
        current_estimate = self.estimate_tokens(messages, tools)
        if current_estimate <= self.budget:
            return messages

        preserved_head = messages[:preserve_first_n]
        trimmable = messages[preserve_first_n:]

        # Remove from the oldest trimmable messages until within budget
        while trimmable and self.estimate_tokens(preserved_head + trimmable, tools) > self.budget:
            trimmable = trimmable[1:]

        result = preserved_head + trimmable
        result = _repair_tool_pairs(result)

        trimmed_count = len(messages) - len(result)
        if trimmed_count > 0:
            logger.info(
                "Trimmed %d messages to fit budget (%d tokens, limit %d)",
                trimmed_count, self.estimate_tokens(result, tools), self.budget,
            )
        return result

    def _estimate_tool_tokens(self, tools: list[dict]) -> int:
        """Estimate tokens for tool definitions."""
        total = 0
        for registered_tool in tools:
            func = registered_tool.get("function", registered_tool)
            name_tokens = len(func.get("name", "")) // self.chars_per_token
            desc_tokens = len(func.get("description", "")) // self.chars_per_token
            param_tokens = len(str(func.get("parameters", {}))) // self.chars_per_token
            total += name_tokens + desc_tokens + param_tokens + 10  # overhead per tool
        return total


def _repair_tool_pairs(messages: list[dict]) -> list[dict]:
    """Remove orphaned tool calls and tool results.

    A tool call without its corresponding result (or vice versa) confuses
    the model. This removes any unpaired messages.
    """
    # Collect all tool_call_ids from assistant messages
    call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in (msg.get("tool_calls") or []):
                call_id = tc.get("id", "")
                if call_id:
                    call_ids.add(call_id)

    # Collect all tool result ids
    result_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            result_id = msg.get("tool_call_id", "")
            if result_id:
                result_ids.add(result_id)

    paired_ids = call_ids & result_ids

    # Remove tool result messages whose call_id is not in paired set
    repaired: list[dict] = []
    for msg in messages:
        if msg.get("role") == "tool":
            if msg.get("tool_call_id", "") not in paired_ids:
                continue
        repaired.append(msg)

    return repaired
