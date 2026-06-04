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


class ValidityProbeFixtureTests(unittest.TestCase):
    def _run(self):
        from simple_chatbot.fixtures.validity_probe import scenario
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat([{"role": "user", "content": "probe"}]))

    def test_emits_unknown_malformed_and_required_violation(self):
        result = self._run()
        call_msgs = [m for m in result.tool_messages if m["role"] == "assistant" and m.get("tool_calls")]
        args_by_name = {
            m["tool_calls"][0]["function"]["name"]: m["tool_calls"][0]["function"]["arguments"]
            for m in call_msgs
        }
        # unknown tool present
        self.assertIn("ghost_tool", args_by_name)
        # malformed args present (invalid JSON)
        self.assertIn("{intentionally_malformed_json", args_by_name.values())
        # a required-violation call with empty args present
        self.assertIn("{}", args_by_name.values())


class ParallelFixtureTests(unittest.TestCase):
    def _run(self):
        from simple_chatbot.fixtures.parallel import scenario
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat([{"role": "user", "content": "Lisbon this weekend?"}]))

    def test_fans_out_two_calls_in_one_round(self):
        result = self._run()
        assistant_calls = [
            m["tool_calls"] for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        # exactly one assistant message carrying both calls (a single fan-out round)
        self.assertEqual(len(assistant_calls), 1)
        names = [tc["function"]["name"] for tc in assistant_calls[0]]
        self.assertEqual(names, ["get_weather", "get_events"])
        # deterministic per-call ids within the round
        ids = [tc["id"] for tc in assistant_calls[0]]
        self.assertEqual(ids, ["call_parallel_1_1", "call_parallel_1_2"])
        self.assertTrue(result.content)


class MultiRoundRoutingFixtureTests(unittest.TestCase):
    def _run(self):
        from simple_chatbot.fixtures.multi_round_routing import scenario
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat([{"role": "user", "content": "my ticket is stuck"}]))

    def test_two_route_hops_then_resolves(self):
        result = self._run()
        names = [
            m["tool_calls"][0]["function"]["name"]
            for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        self.assertEqual(names, ["route", "route", "check_status"])
        self.assertEqual(result.content, "Resolved at tier 2.")
        self.assertEqual(result.active_agent, "tier2")


class RagIgnoreFixtureTests(unittest.TestCase):
    def _run(self):
        from simple_chatbot.fixtures.rag_ignore import scenario
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat([{"role": "user", "content": "what's the refund window?"}]))

    def test_retrieves_then_ignores(self):
        result = self._run()
        # the retrieved passage is present in a tool-output item in the window
        tool_outputs = [m["content"] for m in result.tool_messages if m["role"] == "tool"]
        self.assertTrue(any("within 14 days" in out for out in tool_outputs))
        # the final answer contradicts the retrieved passage
        self.assertEqual(result.content, "Refunds are available at any time, with no deadline.")
        # the authored Final carries the ignore_retrieval tag
        from simple_chatbot.fixtures.rag_ignore import scenario
        from simple_chatbot.scenario import Final
        finals = [s for s in scenario.agent("support").script if isinstance(s, Final)]
        self.assertTrue(finals[0].ignore_retrieval)


class EscalationFixtureTests(unittest.TestCase):
    def _run(self):
        from simple_chatbot.fixtures.escalation import scenario
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat([{"role": "user", "content": "I need a human"}]))

    def test_routes_straight_to_human_terminal(self):
        result = self._run()
        names = [
            m["tool_calls"][0]["function"]["name"]
            for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        self.assertEqual(names, ["route"])  # single hop, then terminal
        self.assertEqual(result.content, "A human agent will take over from here.")
        self.assertEqual(result.active_agent, "human")
