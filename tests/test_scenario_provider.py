# tests/test_scenario_provider.py
import asyncio
import json
import unittest

from simple_chatbot.scenario import Agent, Call, Final, MalformedCall, Route, UnknownToolCall, tool
from simple_chatbot.scenario_provider import DeterministicProvider


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice."""
    return {"invoice_id": invoice_id}


def _decide(provider, agent):
    return asyncio.run(provider.decide(agent, messages=[], tool_messages=[]))


class DeterministicProviderTests(unittest.TestCase):
    def test_plays_route_then_exhausts(self):
        agent = Agent("dispatcher", routes=["billing"], script=[Route("billing")])
        p = DeterministicProvider()
        d1 = _decide(p, agent)
        self.assertIsNone(d1.final)
        self.assertEqual(d1.calls[0].name, "route")
        self.assertEqual(json.loads(d1.calls[0].arguments), {"agent": "billing"})
        # exhausted → final fallback
        d2 = _decide(p, agent)
        self.assertIsNotNone(d2.final)

    def test_call_then_final(self):
        agent = Agent(
            "billing",
            tools=[lookup_invoice],
            script=[Call(lookup_invoice, {"invoice_id": "INV-1"}), Final("done")],
        )
        p = DeterministicProvider()
        d1 = _decide(p, agent)
        self.assertEqual(d1.calls[0].name, "lookup_invoice")
        self.assertEqual(json.loads(d1.calls[0].arguments), {"invoice_id": "INV-1"})
        d2 = _decide(p, agent)
        self.assertEqual(d2.final, "done")

    def test_malformed_call_emits_invalid_json(self):
        agent = Agent("a", tools=[lookup_invoice], script=[MalformedCall(lookup_invoice)])
        d = _decide(DeterministicProvider(), agent)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(d.calls[0].arguments)

    def test_unknown_tool_call(self):
        agent = Agent("a", script=[UnknownToolCall("ghost_tool")])
        d = _decide(DeterministicProvider(), agent)
        self.assertEqual(d.calls[0].name, "ghost_tool")

    def test_required_violation_emits_empty_args(self):
        agent = Agent("a", tools=[lookup_invoice], script=[Call(lookup_invoice)])
        d = _decide(DeterministicProvider(), agent)
        self.assertEqual(json.loads(d.calls[0].arguments), {})

    def test_cursor_is_per_agent(self):
        a = Agent("a", routes=["b"], script=[Route("b")])
        b = Agent("b", tools=[lookup_invoice], script=[Final("b-done")])
        p = DeterministicProvider()
        self.assertEqual(_decide(p, a).calls[0].name, "route")
        self.assertEqual(_decide(p, b).final, "b-done")
