# tests/test_scenario_catalog.py
import unittest

from simple_chatbot.scenario import Agent, Scenario, tool
from simple_chatbot.scenario_catalog import build_responses_tools


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
