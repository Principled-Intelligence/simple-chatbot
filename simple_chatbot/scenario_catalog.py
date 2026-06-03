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
