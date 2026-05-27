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
    tools that surface document hits to the conversation logger."""

    text: str
    chunks: list[Document] = field(default_factory=list)


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


def make_search_tool(indexer) -> ToolDef:
    """Wrap an Indexer-shaped object (real or scripted) into the search ToolDef."""

    async def executor(arguments: str) -> ToolResult:
        args, error = _parse_args(SEARCH_TOOL_NAME, arguments)
        if error:
            return ToolResult(text=f"Tool error: {error}")
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                text=f"Tool error: {SEARCH_TOOL_NAME!r} requires a non-empty string 'query' argument"
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
