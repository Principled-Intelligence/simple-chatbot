import json
import uuid
from dataclasses import dataclass, field


@dataclass
class _ScriptedToolCallFunction:
    name: str
    arguments: str


@dataclass
class _ScriptedToolCall:
    id: str
    function: _ScriptedToolCallFunction
    type: str = "function"


@dataclass
class _ScriptedMessage:
    role: str = "assistant"
    content: str | None = None
    tool_calls: list = field(default_factory=list)

    def model_dump(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ],
        }


@dataclass
class _ScriptedChoice:
    message: _ScriptedMessage
    finish_reason: str
    index: int = 0


@dataclass
class _ScriptedUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _ScriptedResponse:
    choices: list
    usage: _ScriptedUsage = field(default_factory=_ScriptedUsage)


def _extract_user_text(content) -> str:
    """Extract a plain string query from a chat-completions user `content`.

    Accepts either a plain string or a content-parts list (OpenAI's input
    content shape). Returns the first text-bearing part as a string.
    """
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                return str(part["text"])
        return ""
    return str(content)


async def acompletion(*, messages: list[dict], **kwargs) -> _ScriptedResponse:
    """LiteLLM-compatible scripted completion for offline testing.

    Walks `messages` from the end backwards. Dispatch rules:
    - most recent message is a tool result -> emit final echo answer
    - most recent message is a user input AND no prior tool activity is
      present in the history -> emit a search_documents tool_call
    - most recent message is a user input AND prior tool activity exists
      -> emit a direct answer (no new tool_call) so chained turns don't
      keep re-searching
    - nothing dispatchable -> emit a generic stop response

    The "first message only" tool-call policy mimics how a real agent
    behaves once it has search context already in its history.
    """
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        role = msg.get("role")
        if role == "tool":
            return _final_echo_response(msg.get("content", ""))
        if role == "user":
            if _has_prior_tool_activity(messages[:idx]):
                return _direct_answer_response(msg.get("content", ""))
            return _tool_call_response(msg.get("content", ""))
    return _fallback_response()


def _has_prior_tool_activity(messages: list[dict]) -> bool:
    """True if any earlier message represents a completed search round."""
    for m in messages:
        role = m.get("role")
        if role == "tool":
            return True
        if role == "assistant" and m.get("tool_calls"):
            return True
    return False


def _tool_call_response(user_content) -> _ScriptedResponse:
    query = _extract_user_text(user_content)
    call_id = f"call_scripted_{uuid.uuid4().hex}"
    message = _ScriptedMessage(
        content=None,
        tool_calls=[
            _ScriptedToolCall(
                id=call_id,
                function=_ScriptedToolCallFunction(
                    name="search_documents",
                    arguments=json.dumps({"query": query}),
                ),
            )
        ],
    )
    return _ScriptedResponse(
        choices=[_ScriptedChoice(message=message, finish_reason="tool_calls")],
    )


def _final_echo_response(tool_content) -> _ScriptedResponse:
    text = str(tool_content)
    message = _ScriptedMessage(
        content=f"Scripted answer based on search result: {text}",
    )
    return _ScriptedResponse(
        choices=[_ScriptedChoice(message=message, finish_reason="stop")],
    )


def _direct_answer_response(user_content) -> _ScriptedResponse:
    """Direct answer used when the thread already contains a prior search.

    The mock policy is "tool invocation only on the first message"; chained
    turns get a canned direct response that still echoes the user input so
    the trace is observable.
    """
    text = _extract_user_text(user_content)
    message = _ScriptedMessage(
        content=f"Scripted direct answer (no new search) for: {text!r}",
    )
    return _ScriptedResponse(
        choices=[_ScriptedChoice(message=message, finish_reason="stop")],
    )


def _fallback_response() -> _ScriptedResponse:
    message = _ScriptedMessage(content="ok")
    return _ScriptedResponse(
        choices=[_ScriptedChoice(message=message, finish_reason="stop")],
    )
