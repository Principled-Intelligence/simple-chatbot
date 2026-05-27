import json
import re
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
    reasoning_content: str | None = None  # NEW

    def model_dump(self) -> dict:
        d = {
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
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
        return d


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


# ---------------------------------------------------------------------------
# Marker parsing
# ---------------------------------------------------------------------------

_MARKER_RE = re.compile(r"\[(parallel|reasoning|multi-?round|error)\]", re.IGNORECASE)


@dataclass
class _Markers:
    parallel: bool = False
    reasoning: bool = False
    multi_round: bool = False
    error: bool = False

    @property
    def any_explicit(self) -> bool:
        return self.parallel or self.reasoning or self.multi_round or self.error


def _parse_markers(text: str) -> tuple[_Markers, str]:
    """Extract markers from `text` and return (markers, text_with_markers_stripped)."""
    markers = _Markers()

    def _capture(m: re.Match) -> str:
        token = m.group(1).lower().replace("-", "")
        if token == "parallel":
            markers.parallel = True
        elif token == "reasoning":
            markers.reasoning = True
        elif token == "multiround":
            markers.multi_round = True
        elif token == "error":
            markers.error = True
        return ""

    cleaned = _MARKER_RE.sub(_capture, text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Heuristic reasoning triggers (only when no explicit marker yet)
    if not markers.reasoning:
        lowered = cleaned.lower()
        if any(p in lowered for p in ["step by step", "think it through", "let's reason"]):
            markers.reasoning = True
    return markers, cleaned


# ---------------------------------------------------------------------------
# Tool selection heuristics
# ---------------------------------------------------------------------------

_CALC_KEYWORDS = re.compile(r"\b(calculat\w*|comput\w*|math)\b", re.IGNORECASE)
_CALC_OPERATOR = re.compile(r"\d\s*[\+\-\*/=]\s*\d")
_TIME_KEYWORDS = re.compile(r"\btime\b", re.IGNORECASE)
_USER_KEYWORDS = re.compile(r"\b(lookup|user_id)\b", re.IGNORECASE)


def _pick_tools(cleaned_text: str) -> list[str]:
    """Return ordered list of tool names to call based on keyword presence.

    Returns ["search_documents"] when nothing matches.
    """
    tools = []
    if _CALC_KEYWORDS.search(cleaned_text) or _CALC_OPERATOR.search(cleaned_text):
        tools.append("calculate")
    if _TIME_KEYWORDS.search(cleaned_text):
        tools.append("get_current_time")
    if _USER_KEYWORDS.search(cleaned_text):
        tools.append("lookup_user")
    if not tools:
        tools.append("search_documents")
    return tools


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------

def _build_args_for(tool_name: str, cleaned_text: str) -> dict:
    if tool_name == "calculate":
        m = re.search(r"(?:calculate|compute)\s+([^.,;!?]+)", cleaned_text, re.IGNORECASE)
        expression = m.group(1).strip() if m else cleaned_text
        return {"expression": expression}
    if tool_name == "get_current_time":
        m = re.search(r"\bin\s+([A-Za-z_/]+)", cleaned_text, re.IGNORECASE)
        if m:
            return {"tz": m.group(1)}
        return {}
    if tool_name == "lookup_user":
        # Try "lookup [user] <id>" or "user_id <id>" patterns — skip the literal word "user"
        m = re.search(r"(?:lookup\s+(?:user\s+)?|user_id\s+)(\w+)", cleaned_text, re.IGNORECASE)
        user_id = m.group(1) if m else cleaned_text
        return {"user_id": user_id}
    # default search
    return {"query": cleaned_text or "(empty)"}


# ---------------------------------------------------------------------------
# Multi-round helpers
# ---------------------------------------------------------------------------

def _count_rounds_since_latest_user(messages: list[dict]) -> int:
    """Count assistant-with-tool_calls messages since the most recent user message.

    Each such assistant message represents one round of tool calling (which may
    contain N parallel tool calls). Used by the [multi-round] marker to decide
    whether to emit another round or the final answer.
    """
    count = 0
    for m in reversed(messages):
        role = m.get("role")
        if role == "assistant" and (m.get("tool_calls") or []):
            count += 1
        elif role == "user":
            return count
    return count


def _multi_round_secondary_tool(primary_tool: str) -> str:
    """Pick a different tool for the second round so the trace shows variety."""
    rotation = ["search_documents", "calculate", "get_current_time", "lookup_user"]
    primary_idx = rotation.index(primary_tool) if primary_tool in rotation else 0
    return rotation[(primary_idx + 1) % len(rotation)]


# ---------------------------------------------------------------------------
# Reasoning text
# ---------------------------------------------------------------------------

def _reasoning_text(cleaned_text: str, tools: list[str]) -> str:
    tool_list = ", ".join(tools) if tools else "none"
    return (
        f"User asked: {cleaned_text!r}. "
        f"Planned tool(s): {tool_list}. "
        f"I'll call the selected tool(s) and combine the results."
    )


def _final_reasoning_text(cleaned_text: str, tool_output_count: int) -> str:
    return (
        f"User asked: {cleaned_text!r}. "
        f"Reviewing {tool_output_count} tool result(s) and producing the combined answer."
    )


# ---------------------------------------------------------------------------
# Tool call construction
# ---------------------------------------------------------------------------

def _make_tool_call(tool_name: str, args: dict) -> _ScriptedToolCall:
    return _ScriptedToolCall(
        id=f"call_scripted_{uuid.uuid4().hex}",
        function=_ScriptedToolCallFunction(
            name=tool_name,
            arguments=json.dumps(args),
        ),
    )


def _make_malformed_tool_call(tool_name: str) -> _ScriptedToolCall:
    """Used by [error] marker to exercise the agent's malformed-args path."""
    return _ScriptedToolCall(
        id=f"call_scripted_{uuid.uuid4().hex}",
        function=_ScriptedToolCallFunction(
            name=tool_name,
            arguments='{intentionally_malformed_json',
        ),
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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


def _find_latest_user(messages: list[dict]) -> dict | None:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m
    return None


def _collect_tool_outputs_since_latest_user(messages: list[dict]) -> list[str]:
    outputs: list[str] = []
    for m in reversed(messages):
        role = m.get("role")
        if role == "tool":
            outputs.append(str(m.get("content", "")))
        elif role == "user":
            break
    outputs.reverse()
    return outputs


def _has_prior_tool_activity(messages: list[dict]) -> bool:
    """True if any earlier message represents a completed search round."""
    for m in messages:
        role = m.get("role")
        if role == "tool":
            return True
        if role == "assistant" and m.get("tool_calls"):
            return True
    return False


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


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

async def acompletion(*, messages: list[dict], **kwargs) -> _ScriptedResponse:
    """Scripted LiteLLM-compatible completion for offline testing.

    Heuristic tool selection by keyword in the latest user input, plus
    explicit `[parallel]`, `[reasoning]`, `[multi-round]`, `[error]` markers
    that force specific trace shapes.

    Dispatches by walking messages from the end backwards:
    - tool result -> emit another tool call (multi-round) or final answer
    - user input -> pick tools per heuristic / markers, emit tool_call(s)
                    (or skip if chained-turn with prior tool activity and no
                     explicit marker overriding)
    - nothing dispatchable -> generic "ok" stop
    """
    last_user_msg = _find_latest_user(messages)
    last_user_text = _extract_user_text(last_user_msg.get("content") if last_user_msg else "")
    markers, cleaned_text = _parse_markers(last_user_text)

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        role = msg.get("role")

        if role == "tool":
            rounds = _count_rounds_since_latest_user(messages)
            if markers.multi_round and rounds < 2:
                # Emit a second-round tool call with a different tool
                primary = _pick_tools(cleaned_text)[0]
                secondary = _multi_round_secondary_tool(primary)
                tc = _make_tool_call(secondary, _build_args_for(secondary, cleaned_text))
                message = _ScriptedMessage(
                    content=None,
                    tool_calls=[tc],
                    reasoning_content=_reasoning_text(cleaned_text, [secondary]) if markers.reasoning else None,
                )
                return _ScriptedResponse(
                    choices=[_ScriptedChoice(message=message, finish_reason="tool_calls")],
                )
            # Otherwise: final answer combining tool results
            tool_outputs = _collect_tool_outputs_since_latest_user(messages)
            summary = " | ".join(tool_outputs) if tool_outputs else "(no tool results)"
            text = f"Scripted answer combining {len(tool_outputs)} tool result(s): {summary}"
            message = _ScriptedMessage(
                content=text,
                reasoning_content=(
                    _final_reasoning_text(cleaned_text, len(tool_outputs))
                    if markers.reasoning else None
                ),
            )
            return _ScriptedResponse(
                choices=[_ScriptedChoice(message=message, finish_reason="stop")],
            )

        if role == "user":
            # Markers override chained-turn heuristic
            if not markers.any_explicit and _has_prior_tool_activity(messages[:idx]):
                return _direct_answer_response(msg.get("content", ""))

            tools = _pick_tools(cleaned_text)
            if markers.parallel and len(tools) < 2:
                # Force parallelism by appending an extra tool
                rotation = ["search_documents", "calculate", "get_current_time", "lookup_user"]
                for fallback in rotation:
                    if fallback not in tools:
                        tools.append(fallback)
                        break

            if markers.error:
                # Replace the first tool call with a malformed one; keep the rest normal
                first_call = _make_malformed_tool_call(tools[0])
                rest = [_make_tool_call(t, _build_args_for(t, cleaned_text)) for t in tools[1:]]
                tool_calls = [first_call] + rest
            else:
                tool_calls = [_make_tool_call(t, _build_args_for(t, cleaned_text)) for t in tools]

            message = _ScriptedMessage(
                content=None,
                tool_calls=tool_calls,
                reasoning_content=_reasoning_text(cleaned_text, tools) if markers.reasoning else None,
            )
            return _ScriptedResponse(
                choices=[_ScriptedChoice(message=message, finish_reason="tool_calls")],
            )

    return _fallback_response()
