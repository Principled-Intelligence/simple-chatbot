"""Adapter that lets the agent's chat-completions loop drive the OpenAI
Responses API instead.

Why this exists: gpt-5.6 reasoning models reject `function` tools together with
reasoning on `/v1/chat/completions` (hard 400). The tool+reasoning combination is
only available on `/v1/responses`. Rather than rewrite the agent's chat-completions
loop, this module exposes `responses_acompletion(**kwargs)` with the SAME call/return
contract as `litellm.acompletion`: it accepts chat-style `messages`/`tools` and returns
a `litellm.ModelResponse` (so `choices[0].message`, `.tool_calls`, `.model_dump()`,
`reasoning_content`, `finish_reason`, and `usage` all look exactly as the agent expects).
Internally it translates to `litellm.aresponses` and translates the output back.

Multi-turn works statelessly: each round the agent rebuilds its full message list and
we reconstruct the Responses `input` from it (verified: OpenAI accepts function_call /
function_call_output items without echoing reasoning items and without previous_response_id).
"""

from __future__ import annotations

from typing import Any

import litellm
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    Usage,
)


def _tools_to_responses(tools: list[dict] | None) -> list[dict]:
    """chat-completions nested {"type":"function","function":{...}} ->
    Responses flat {"type":"function","name","description","parameters"}."""
    out: list[dict] = []
    for t in tools or []:
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            out.append(
                {
                    "type": "function",
                    "name": fn.get("name"),
                    "description": fn.get("description", "") or "",
                    "parameters": fn.get("parameters", {}) or {},
                }
            )
        else:
            out.append(t)
    return out


def _messages_to_input(messages: list[dict]) -> list[dict]:
    """chat-completions message list -> Responses `input` item list."""
    inp: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer", "user"):
            inp.append({"role": role, "content": m.get("content") or ""})
        elif role == "assistant":
            content = m.get("content")
            if content:
                inp.append({"role": "assistant", "content": content})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                inp.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id"),
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments", "") or "",
                    }
                )
        elif role == "tool":
            inp.append(
                {
                    "type": "function_call_output",
                    "call_id": m.get("tool_call_id"),
                    "output": m.get("content") or "",
                }
            )
    return inp


def _as_dict(item: Any) -> dict:
    return item if isinstance(item, dict) else item.model_dump()


def _responses_to_model_response(resp: Any, model: str) -> ModelResponse:
    """Responses output -> chat-completions ModelResponse the agent understands."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ChatCompletionMessageToolCall] = []

    for raw in resp.output or []:
        d = _as_dict(raw)
        typ = d.get("type")
        if typ == "message":
            for c in d.get("content") or []:
                cd = _as_dict(c)
                if cd.get("type") in ("output_text", "text") and cd.get("text"):
                    text_parts.append(cd["text"])
        elif typ == "function_call":
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=d.get("call_id") or d.get("id"),
                    type="function",
                    function=Function(
                        name=d.get("name"),
                        arguments=d.get("arguments", "") or "",
                    ),
                )
            )
        elif typ == "reasoning":
            for s in d.get("summary") or []:
                sd = _as_dict(s)
                if sd.get("text"):
                    reasoning_parts.append(sd["text"])

    content = "".join(text_parts) or None
    reasoning = "".join(reasoning_parts) or None
    finish_reason = "tool_calls" if tool_calls else "stop"

    message = Message(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
    )
    if reasoning is not None:
        message.reasoning_content = reasoning

    u = getattr(resp, "usage", None)
    prompt = getattr(u, "input_tokens", 0) or 0
    completion = getattr(u, "output_tokens", 0) or 0
    total = getattr(u, "total_tokens", 0) or (prompt + completion)
    usage = Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)

    mr = ModelResponse(
        model=model,
        choices=[Choices(index=0, finish_reason=finish_reason, message=message)],
    )
    mr.usage = usage
    return mr


async def responses_acompletion(**kwargs: Any) -> ModelResponse:
    """Drop-in async replacement for litellm.acompletion that routes through
    litellm.aresponses. Same accepted kwargs (model, messages, tools, tool_choice,
    api_base, sampling params) and same ModelResponse return shape."""
    model = kwargs.pop("model")
    messages = kwargs.pop("messages", [])
    tools = kwargs.pop("tools", None)
    tool_choice = kwargs.pop("tool_choice", None)

    call: dict[str, Any] = {
        "model": model,
        "input": _messages_to_input(messages),
        "drop_params": True,  # silently drop sampling params the Responses API rejects
    }
    resp_tools = _tools_to_responses(tools)
    if resp_tools:
        call["tools"] = resp_tools
        if tool_choice is not None:
            call["tool_choice"] = tool_choice
    # Forward remaining passthrough kwargs (api_base, sampling, etc.).
    for k, v in kwargs.items():
        if v is not None:
            call[k] = v

    resp = await litellm.aresponses(**call)
    return _responses_to_model_response(resp, model)
