# tests/test_scenario_catalog.py
import unittest

from simple_chatbot.scenario import Agent, Scenario, tool
from simple_chatbot.scenario_catalog import (
    build_responses_tools,
    build_responses_tools_from_defs,
    chat_tool_to_responses_tool,
)
from simple_chatbot.agent import ChatResult
from simple_chatbot.responses import build_response


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice."""
    return {"invoice_id": invoice_id}


class CatalogTests(unittest.TestCase):
    def _scenario(self):
        return Scenario(
            id="demo",
            entry="dispatcher",
            agents=[
                Agent("dispatcher", routes=["billing"]),
                Agent("billing", tools=[lookup_invoice]),
            ],
        )

    def test_function_tool_is_flat_shape(self):
        entries = build_responses_tools(self._scenario())
        inv = next(e for e in entries if e["name"] == "lookup_invoice")
        self.assertEqual(inv["type"], "function")
        self.assertEqual(inv["description"], "Look up an invoice.")
        self.assertEqual(inv["parameters"]["required"], ["invoice_id"])
        # FLAT shape (no nested "function" key like chat-completions)
        self.assertNotIn("function", inv)

    def test_route_tool_published_with_enum(self):
        entries = build_responses_tools(self._scenario())
        route = next(e for e in entries if e["name"] == "route")
        self.assertEqual(route["type"], "function")
        self.assertEqual(
            route["parameters"]["properties"]["agent"]["enum"], ["billing"]
        )
        self.assertEqual(route["parameters"]["required"], ["agent"])

    def test_no_route_tool_when_no_routes(self):
        s = Scenario(id="x", entry="a", agents=[Agent("a", tools=[lookup_invoice])])
        names = [e["name"] for e in build_responses_tools(s)]
        self.assertNotIn("route", names)


class _FakeToolDef:
    def __init__(self, schema):
        self.schema = schema


class ChatToResponsesToolTests(unittest.TestCase):
    def test_nested_function_flattened(self):
        schema = {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search the index.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
        out = chat_tool_to_responses_tool(schema)
        self.assertEqual(out["type"], "function")
        self.assertEqual(out["name"], "search_documents")
        self.assertEqual(out["description"], "Search the index.")
        self.assertEqual(out["parameters"]["required"], ["query"])
        # FLAT Responses shape — no nested "function" key.
        self.assertNotIn("function", out)

    def test_description_omitted_when_blank(self):
        out = chat_tool_to_responses_tool(
            {"type": "function", "function": {"name": "t", "parameters": {}}}
        )
        self.assertNotIn("description", out)

    def test_builtin_tool_passes_through_unchanged(self):
        # A future OpenAI built-in (already flat, not a function) must survive.
        builtin = {"type": "web_search"}
        out = chat_tool_to_responses_tool(builtin)
        self.assertEqual(out, builtin)
        self.assertIsNot(out, builtin)  # defensively copied

    def test_build_from_defs(self):
        defs = [
            _FakeToolDef(
                {"type": "function", "function": {"name": "a", "parameters": {}}}
            ),
            _FakeToolDef({"type": "web_search"}),
        ]
        out = build_responses_tools_from_defs(defs)
        self.assertEqual([e.get("name", e["type"]) for e in out], ["a", "web_search"])


class BuildResponseToolsTests(unittest.TestCase):
    def test_response_tools_populated_from_result(self):
        result = ChatResult(
            content="done",
            retrieved_chunks=[],
            responses_tools=[
                {"type": "function", "name": "route",
                 "parameters": {"type": "object", "properties": {}}}
            ],
        )
        payload = build_response(
            result=result,
            model="fixture:demo",
            previous_response_id=None,
            conversation_id="conv-1",
        )
        self.assertEqual(payload["tools"][0]["name"], "route")

    def test_response_tools_default_empty(self):
        result = ChatResult(content="done", retrieved_chunks=[])
        payload = build_response(
            result=result, model="m", previous_response_id=None, conversation_id="c"
        )
        self.assertEqual(payload["tools"], [])
