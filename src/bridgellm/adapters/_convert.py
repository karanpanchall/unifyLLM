"""Cross-adapter conversion utilities.

Handles image format translation and message validation that differ
between providers. Kept separate to avoid bloating individual adapters.
"""

import base64
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Matches data URIs: data:image/jpeg;base64,<data>
_DATA_URI_PATTERN = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)


def openai_image_to_anthropic(content_block: dict) -> Optional[dict]:
    """Convert an OpenAI image_url block to Anthropic image block.

    OpenAI:    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "high"}}
    Anthropic: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}

    Returns None if the block is not an image_url type.
    """
    if content_block.get("type") != "image_url":
        return None

    image_url_obj = content_block.get("image_url", {})
    url = image_url_obj.get("url", "")

    # Handle base64 data URIs
    match = _DATA_URI_PATTERN.match(url)
    if match:
        media_type = match.group(1)
        raw_data = match.group(2)
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": raw_data},
        }

    # Handle regular URLs
    if url.startswith("http"):
        return {
            "type": "image",
            "source": {"type": "url", "url": url},
        }

    logger.warning("Unrecognized image URL format: %s...", url[:80])
    return content_block


def openai_pdf_to_anthropic(content_block: dict) -> Optional[dict]:
    """Convert a PDF content block to Anthropic document format.

    bridgellm canonical: {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "..."}}
    This is already Anthropic's format. Pass through as-is, or convert from URL variant.
    Also handles: {"type": "file", "file": {"url": "data:application/pdf;base64,..."}}
    """
    block_type = content_block.get("type", "")

    if block_type == "document":
        return content_block  # Already Anthropic format

    if block_type == "file":
        file_info = content_block.get("file", {})
        url = file_info.get("url", "")
        match = _DATA_URI_PATTERN.match(url)
        if match:
            return {
                "type": "document",
                "source": {"type": "base64", "media_type": match.group(1), "data": match.group(2)},
            }
        if url.startswith("http"):
            return {"type": "document", "source": {"type": "url", "url": url}}

    return None


def convert_messages_for_anthropic(messages: list[dict]) -> list[dict]:
    """Prepare messages for the Anthropic API.

    1. Convert image_url blocks to Anthropic image format.
    2. Convert document/file blocks to Anthropic document format.
    3. Repair role alternation (Anthropic requires strict user/assistant alternation).
    4. Convert OpenAI tool result messages to Anthropic tool_result format.
    """
    converted = [_convert_single_message(msg) for msg in messages]
    return _repair_alternation(converted)


def _convert_single_message(msg: dict) -> dict:
    """Convert a single message, handling images and tool results."""
    role = msg.get("role", "")

    # Convert OpenAI tool results → Anthropic tool_result content blocks
    if role == "tool":
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }],
        }

    content = msg.get("content")
    if not isinstance(content, list):
        return msg

    # Convert image and document blocks within content arrays
    converted_parts = []
    for block in content:
        anthropic_image = openai_image_to_anthropic(block)
        if anthropic_image:
            converted_parts.append(anthropic_image)
            continue
        anthropic_doc = openai_pdf_to_anthropic(block)
        if anthropic_doc:
            converted_parts.append(anthropic_doc)
            continue
        converted_parts.append(block)

    return {**msg, "content": converted_parts}


def _repair_alternation(messages: list[dict]) -> list[dict]:
    """Merge consecutive same-role messages to satisfy Anthropic's alternation rule.

    Anthropic requires strict user → assistant → user alternation.
    Consecutive messages with the same role are merged into one message.
    """
    if not messages:
        return messages

    repaired: list[dict] = [messages[0]]

    for msg in messages[1:]:
        prev = repaired[-1]
        if msg.get("role") == prev.get("role"):
            # Merge content into the previous message
            prev_content = prev.get("content", "")
            msg_content = msg.get("content", "")
            merged = _merge_content(prev_content, msg_content)
            repaired[-1] = {**prev, "content": merged}
        else:
            repaired.append(msg)

    return repaired


def _merge_content(existing, new):
    """Merge two message content values (string or list of blocks)."""
    # Both strings: join with newline
    if isinstance(existing, str) and isinstance(new, str):
        return f"{existing}\n{new}" if existing else new

    # Normalize to lists
    existing_list = existing if isinstance(existing, list) else [{"type": "text", "text": existing}] if existing else []
    new_list = new if isinstance(new, list) else [{"type": "text", "text": new}] if new else []

    return existing_list + new_list
