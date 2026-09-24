"""Tool registry and built-in tool definitions for the agent.

A `ToolDef` bundles an OpenAI tool schema with an async executor that runs
the tool and returns its result. The agent advertises every registered
tool's schema to the LLM and dispatches tool calls by name through the
registry.

Search is special: its executor also surfaces retrieved `Document`
chunks via `ToolResult.chunks`, which the agent threads into
`ChatResult.retrieved_chunks` for conversation logging. Other tools return
`chunks=[]`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from simple_chatbot.loader import Document


SEARCH_TOOL_NAME = "search_documents"

# Invalid-JSON marker for the malformed-args knob, shared by the deterministic
# engine and the evil agent. `scripted_llm` mirrors this same literal inline.
MALFORMED_TOOL_ARGS = "{intentionally_malformed_json"


SEARCH_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Search the document index for relevant information. "
            "Use this whenever the user asks something that may be answered by the loaded documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
    },
}

# Backward-compat alias
SEARCH_TOOL = SEARCH_TOOL_SCHEMA


@dataclass
class ToolResult:
    """Result of running a tool. `text` is the string returned to the LLM as
    the tool message content. `chunks` is non-empty only for search-style
    tools that surface document hits to the conversation logger. `is_error`
    flags a failed call (bad arguments, validation failure) so the Responses
    adapter can map it to a non-"completed" status; the text is still fed back
    to the model regardless."""

    text: str
    chunks: list[Document] = field(default_factory=list)
    is_error: bool = False


@dataclass
class ToolDef:
    """A tool the agent can advertise and execute."""

    schema: dict
    executor: Callable[[str], Awaitable[ToolResult]]

    @property
    def name(self) -> str:
        return self.schema["function"]["name"]


def _parse_args(tool_name: str, arguments: str | dict) -> tuple[dict, str | None]:
    """Parse a tool call's arguments into a dict. Returns (args_dict, error_text)."""
    if isinstance(arguments, dict):
        return arguments, None
    if not isinstance(arguments, str):
        return {}, f"arguments for {tool_name!r} must be a JSON object"
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON arguments for {tool_name!r}: {exc.msg}"
    if not isinstance(parsed, dict):
        return {}, f"arguments for {tool_name!r} must be a JSON object"
    return parsed, None


CALCULATE_TOOL_NAME = "calculate"
GET_CURRENT_TIME_TOOL_NAME = "get_current_time"
LOOKUP_USER_TOOL_NAME = "lookup_user"


CALCULATE_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": CALCULATE_TOOL_NAME,
        "description": (
            "Evaluate a mathematical expression and return the result. "
            "Use this for arithmetic, algebra, and numeric computations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '2+2' or '(7*8)/2'.",
                },
            },
            "required": ["expression"],
        },
    },
}


GET_CURRENT_TIME_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": GET_CURRENT_TIME_TOOL_NAME,
        "description": (
            "Return the current time. Optionally accepts a timezone name (e.g. 'UTC', "
            "'America/Los_Angeles'). If no timezone is given, returns the server's local time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tz": {
                    "type": "string",
                    "description": "IANA timezone identifier, e.g. 'UTC' or 'Europe/London'.",
                },
            },
            "required": [],
        },
    },
}


LOOKUP_USER_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": LOOKUP_USER_TOOL_NAME,
        "description": (
            "Look up a user record by ID. Returns the user's display name, email, and role."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The user ID to look up.",
                },
            },
            "required": ["user_id"],
        },
    },
}


def make_search_tool(indexer) -> ToolDef:
    """Wrap an Indexer-shaped object (real or scripted) into the search ToolDef."""

    async def executor(arguments: str) -> ToolResult:
        args, error = _parse_args(SEARCH_TOOL_NAME, arguments)
        if error:
            return ToolResult(text=f"Tool error: {error}", is_error=True)
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                text=f"Tool error: {SEARCH_TOOL_NAME!r} requires a non-empty string 'query' argument",
                is_error=True,
            )

        results = await indexer.search(query)
        if results:
            text = "\n\n".join(
                f"[{d.metadata.get('source', 'unknown')}]\n{d.text}" for d in results
            )
        else:
            text = "No relevant documents found."
        return ToolResult(text=text, chunks=results)

    return ToolDef(schema=SEARCH_TOOL_SCHEMA, executor=executor)


def make_calculate_tool() -> ToolDef:
    """Mock calculator. Echoes the expression back as if it were evaluated.

    The point is to give an evaluator a trace where `expression` appears in the
    tool arguments and reappears in the tool output, so faithfulness scoring
    can verify the assistant's final answer references the right computation.
    """

    async def executor(arguments: str) -> ToolResult:
        args, error = _parse_args(CALCULATE_TOOL_NAME, arguments)
        if error:
            return ToolResult(text=f"Tool error: {error}", is_error=True)
        expression = args.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            return ToolResult(
                text=f"Tool error: {CALCULATE_TOOL_NAME!r} requires a non-empty string 'expression' argument",
                is_error=True,
            )
        return ToolResult(text=f"[scripted] calculate({expression!r}) = <fake-numeric-result>")

    return ToolDef(schema=CALCULATE_TOOL_SCHEMA, executor=executor)


def make_get_current_time_tool() -> ToolDef:
    """Mock clock. Returns a canned timestamp, optionally tagged with the requested tz."""

    async def executor(arguments: str) -> ToolResult:
        args, error = _parse_args(GET_CURRENT_TIME_TOOL_NAME, arguments)
        if error:
            return ToolResult(text=f"Tool error: {error}", is_error=True)
        tz = args.get("tz")
        if tz is not None and not isinstance(tz, str):
            return ToolResult(
                text=f"Tool error: {GET_CURRENT_TIME_TOOL_NAME!r} 'tz' must be a string if provided",
                is_error=True,
            )
        if tz:
            return ToolResult(text=f"[scripted] current time in {tz}: 2026-05-27T12:00:00")
        return ToolResult(text="[scripted] current time: 2026-05-27T12:00:00")

    return ToolDef(schema=GET_CURRENT_TIME_TOOL_SCHEMA, executor=executor)


def make_lookup_user_tool() -> ToolDef:
    """Mock user lookup. Echoes the user_id into a canned record."""

    async def executor(arguments: str) -> ToolResult:
        args, error = _parse_args(LOOKUP_USER_TOOL_NAME, arguments)
        if error:
            return ToolResult(text=f"Tool error: {error}", is_error=True)
        user_id = args.get("user_id")
        if not isinstance(user_id, str) or not user_id.strip():
            return ToolResult(
                text=f"Tool error: {LOOKUP_USER_TOOL_NAME!r} requires a non-empty string 'user_id' argument",
                is_error=True,
            )
        return ToolResult(
            text=(
                f"[scripted] user_id={user_id!r} "
                f"display_name='Test User {user_id}' "
                f"email='{user_id}@example.test' "
                f"role='member'"
            )
        )

    return ToolDef(schema=LOOKUP_USER_TOOL_SCHEMA, executor=executor)


def scripted_tools(indexer) -> list[ToolDef]:
    """All four tools registered in scripted offline mode.

    The search tool wraps the given indexer (usually a `ScriptedIndexer`).
    The other three are pure-Python mocks with no external dependencies.
    """
    return [
        make_search_tool(indexer),
        make_calculate_tool(),
        make_get_current_time_tool(),
        make_lookup_user_tool(),
    ]
