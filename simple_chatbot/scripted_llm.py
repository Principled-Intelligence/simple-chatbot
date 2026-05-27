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

    Walks `messages` from the end backwards and dispatches on the most recent
    user or tool message. This handles:
    - initial calls (last is user) -> emit search tool_call
    - post-tool calls (last is tool) -> emit final echo answer
    - multi-message inputs ending in assistant -> still finds the prior user
    - chained turns of any depth, where prior session_messages may interleave
      user/assistant/tool roles before the new user input

    Falls back to a generic "ok" stop response only if no user or tool
    message is present anywhere in the list.
    """
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "tool":
            tool_content = str(msg.get("content", ""))
            message = _ScriptedMessage(
                content=f"Scripted answer based on search result: {tool_content}",
            )
            return _ScriptedResponse(
                choices=[_ScriptedChoice(message=message, finish_reason="stop")],
            )
        if role == "user":
            query = _extract_user_text(msg.get("content", ""))
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

    message = _ScriptedMessage(content="ok")
    return _ScriptedResponse(
        choices=[_ScriptedChoice(message=message, finish_reason="stop")],
    )
