"""Responses API adapter.

Bridges the existing agent (which speaks chat-completions shape) to OpenAI's
Responses API wire format:

- `normalize_input`: convert a Responses API `input` field (string or list of
  message items) into chat-completions messages.
- `build_output_items` / `build_response`: convert a `ChatResult` into the
  ordered output array and the full Response payload, using the openai SDK's
  typed schemas.
- `ResponseStore`: in-memory mapping of response_id -> stored entry, used for
  `previous_response_id` chaining. Process-local; resets on server restart.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseUsage,
)
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from simple_chatbot.agent import ChatResult


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


def _fc_id() -> str:
    return f"fc_{uuid.uuid4().hex}"


def _fco_id() -> str:
    return f"fco_{uuid.uuid4().hex}"


def _msg_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _reasoning_id() -> str:
    return f"rs_{uuid.uuid4().hex}"


def _build_reasoning_item(text: str) -> dict:
    return ResponseReasoningItem(
        id=_reasoning_id(),
        type="reasoning",
        summary=[Summary(type="summary_text", text=text)],
    ).model_dump()


def build_output_items(result: ChatResult) -> list[dict]:
    """Convert a ChatResult into the ordered Responses API `output` array."""
    items: list[dict] = []

    for msg in result.tool_messages:
        role = msg.get("role")
        if role == "assistant":
            # Emit reasoning item BEFORE function_call items if present
            reasoning = msg.get("reasoning_content")
            if reasoning:
                items.append(_build_reasoning_item(reasoning))
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                items.append(
                    ResponseFunctionToolCall(
                        type="function_call",
                        id=_fc_id(),
                        call_id=tc["id"],
                        name=fn.get("name", ""),
                        arguments=fn.get("arguments", "") or "",
                        status="completed",
                    ).model_dump()
                )
        elif role == "tool":
            items.append(
                ResponseFunctionToolCallOutputItem(
                    type="function_call_output",
                    id=_fco_id(),
                    call_id=msg["tool_call_id"],
                    output=msg.get("content") or "",
                    status="completed",
                ).model_dump()
            )

    # Emit final-message reasoning BEFORE the final message item
    if result.final_reasoning_content:
        items.append(_build_reasoning_item(result.final_reasoning_content))

    items.append(
        ResponseOutputMessage(
            id=_msg_id(),
            type="message",
            role="assistant",
            status="completed",
            content=[
                ResponseOutputText(
                    type="output_text",
                    text=result.content or "",
                    annotations=[],
                )
            ],
        ).model_dump()
    )

    return items


def _resp_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def build_response(
    result: ChatResult,
    model: str,
    previous_response_id: str | None,
    conversation_id: str,
) -> dict:
    """Build the full Responses API response payload from a ChatResult."""
    usage = result.usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    response_obj = Response(
        id=_resp_id(),
        object="response",
        created_at=int(time.time()),
        model=model,
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        status="completed",
        previous_response_id=previous_response_id,
        usage=ResponseUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=usage.get("completion_tokens", 0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=usage.get("total_tokens", 0),
        ),
    )

    payload = response_obj.model_dump()
    payload["output"] = build_output_items(result)
    # Ensure created_at is an int (the SDK model uses float internally).
    payload["created_at"] = int(payload["created_at"])
    # Non-standard extension we add at the envelope level (same as chat completions).
    payload["conversation_id"] = conversation_id
    return payload


class ResponseStore:
    """In-memory mapping of response_id -> stored Response entry.

    Process-local; resets on server restart. Acceptable for the evaluator
    integration which drives all chained turns within a single server lifetime.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._entries: dict[str, dict] = {}

    async def put(self, response_id: str, entry: dict) -> None:
        async with self._lock:
            self._entries[response_id] = entry

    async def get(self, response_id: str) -> dict | None:
        async with self._lock:
            return self._entries.get(response_id)
