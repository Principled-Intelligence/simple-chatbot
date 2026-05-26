"""Responses API support: input normalization, output-item mapping, and a small
in-memory store for `previous_response_id` chaining.

This module is intentionally separate from `agent.py` so the agent remains
focused on the chat-completions-shaped tool loop. The Responses adapter calls
the agent unchanged and translates the result.

Current contents:
  - InvalidInputError: raised when a Responses API `input` value can't be normalized.
  - normalize_input: converts the request `input` field (string or list of input
    items) into the chat-completions messages list the Agent expects.

Planned additions (Tasks 5-7):
  - Output-item mapping: translates chat-completions assistant messages and tool
    calls into Responses API output items (message, function_call,
    function_call_output, etc.).
  - build_response: assembles a full Responses API response object from the
    agent's final messages and usage metadata.
  - ResponseStore: a small in-memory store that maps response IDs to their
    message histories, enabling `previous_response_id` chaining so callers can
    resume conversations without re-sending the full history.
"""

from __future__ import annotations


class InvalidInputError(ValueError):
    """Raised when a Responses API `input` value can't be normalized."""


def normalize_input(value: str | list) -> list[dict]:
    """Convert a Responses API `input` field into chat-completions messages.

    Args:
        value: Either a plain string (treated as a single user message) or a
            list whose items may be plain strings or Responses API input item
            dicts (currently only ``type="message"`` is supported).

    Returns:
        A list of chat-completions-style message dicts, each with at least
        ``role`` and ``content`` keys.

    Raises:
        InvalidInputError: If ``value`` is not a string or list, if a list
            item is neither a string nor a dict, or if a dict item has an
            unsupported ``type``.
    """
    if isinstance(value, str):
        return [{"role": "user", "content": value}]

    if not isinstance(value, list):
        raise InvalidInputError(f"`input` must be a string or list, got {type(value).__name__}")

    messages: list[dict] = []
    for idx, item in enumerate(value):
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if isinstance(item, dict):
            item_type = item.get("type", "message")
            if item_type == "message":
                role = item.get("role")
                content = item.get("content")
                if role not in ("user", "assistant", "system"):
                    raise InvalidInputError(f"input[{idx}].role must be user/assistant/system")
                messages.append({"role": role, "content": content})
                continue
            raise InvalidInputError(f"input[{idx}].type {item_type!r} is not supported")
        raise InvalidInputError(f"input[{idx}] must be a string or dict, got {type(item).__name__}")

    return messages
