# tests/test_scenario_fixtures.py
import asyncio
import json
import unittest

from simple_chatbot.scenario_provider import DeterministicProvider
from simple_chatbot.scenario_orchestrator import ScenarioOrchestrator


class CsRoutingFixtureTests(unittest.TestCase):
    def _run(self):
        from simple_chatbot.fixtures.cs_routing import scenario
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat([{"role": "user", "content": "I want a refund"}]))

    def test_routes_to_billing_and_finishes(self):
        result = self._run()
        names = [
            m["tool_calls"][0]["function"]["name"]
            for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        self.assertEqual(names[0], "route")
        self.assertIn("lookup_invoice", names)
        self.assertTrue(result.content)

    def test_catalog_has_route_and_billing_tools(self):
        result = self._run()
        names = {e["name"] for e in result.responses_tools}
        self.assertIn("route", names)
        self.assertIn("lookup_invoice", names)
