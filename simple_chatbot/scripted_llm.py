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


async def acompletion(*, messages: list[dict], **kwargs) -> _ScriptedResponse:
    """LiteLLM-compatible scripted completion for offline testing.

    Dispatches on the last message's role:
    - user        -> emit a search_documents tool_calls response with user text as query
    - tool        -> emit a final assistant text echoing the tool result
    - otherwise   -> emit a generic 'ok' stop response
    """
    last = messages[-1] if messages else {}
    role = last.get("role")

    if role == "user":
        # extract a string from the user content (may be a list of parts)
        content = last.get("content", "")
        if isinstance(content, list):
            # take the first input_text-ish part if available
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    content = part["text"]
                    break
            else:
                content = ""
        query = str(content)
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

    if role == "tool":
        tool_content = str(last.get("content", ""))
        message = _ScriptedMessage(content=f"Scripted answer based on search result: {tool_content}")
        return _ScriptedResponse(
            choices=[_ScriptedChoice(message=message, finish_reason="stop")],
        )

    message = _ScriptedMessage(content="ok")
    return _ScriptedResponse(
        choices=[_ScriptedChoice(message=message, finish_reason="stop")],
    )
