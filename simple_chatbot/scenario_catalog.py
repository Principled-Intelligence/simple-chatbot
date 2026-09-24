# simple_chatbot/scenario_catalog.py
"""Project a Scenario's tools into the flat Responses-API `Response.tools` shape.

NOTE the FLAT function-tool shape `{type, name, parameters, description}` — not
the Chat Completions nested `{type, function: {...}}` shape. The catalog is the
UNION of all agents' tools plus a single generated `route` tool whose enum is
every routing target in the scenario.
"""

from __future__ import annotations

from simple_chatbot.scenario import ROUTE_TOOL_NAME, Scenario


def build_responses_tools(scenario: Scenario) -> list[dict]:
    out: list[dict] = []
    for t in scenario.all_tools():
        entry: dict = {"type": "function", "name": t.name, "parameters": t.parameters}
        if t.description:
            entry["description"] = t.description
        out.append(entry)

    targets = scenario.route_targets()
    if targets:
        out.append(
            {
                "type": "function",
                "name": ROUTE_TOOL_NAME,
                "description": "Hand off the request to another agent.",
                "parameters": {
                    "type": "object",
                    "properties": {"agent": {"type": "string", "enum": targets}},
                    "required": ["agent"],
                },
            }
        )
    return out


def chat_tool_to_responses_tool(schema: dict) -> dict:
    """Convert one OpenAI Chat Completions tool schema (the nested
    `{type:"function", function:{name, description, parameters}}` shape the
    generic RAG agent advertises) into the flat Responses-API tool shape
    `{type:"function", name, parameters, description?}`.

    A tool that is NOT a plain function — e.g. an OpenAI built-in like
    `{"type": "web_search"}` — already uses the flat Responses shape, so it
    passes through unchanged. That way, the day we hand the agent a built-in
    OpenAI tool it surfaces in the catalog automatically.
    """
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        fn = schema["function"]
        entry: dict = {
            "type": "function",
            "name": fn["name"],
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }
        if fn.get("description"):
            entry["description"] = fn["description"]
        return entry
    return dict(schema)


def build_responses_tools_from_defs(tools) -> list[dict]:
    """Project a list of `ToolDef`-shaped objects (anything exposing a `.schema`
    chat-completions tool dict) into the flat Responses-API catalog shape, so the
    generic RAG agent's catalog renders identically to a fixture's."""
    return [chat_tool_to_responses_tool(t.schema) for t in tools]
