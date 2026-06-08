# tests/test_scenario_provider.py
import asyncio
import json
import unittest

from simple_chatbot.scenario import (
    ROUTE_TOOL_NAME,
    Agent,
    Call,
    Final,
    MalformedCall,
    Parallel,
    Route,
    UnknownToolCall,
    tool,
)
from simple_chatbot.scenario_provider import DeterministicProvider, LiveProvider


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


class _Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, arguments):
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, content="", tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, message, finish_reason):
        self.choices = [_Choice(message, finish_reason)]
        self.usage = None


def _live(captured, response=None, raises=None):
    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        if raises is not None:
            raise raises
        return response
    return LiveProvider(acompletion=fake_acompletion, model="test-model")


class LiveProviderTests(unittest.TestCase):
    def test_offers_only_active_agent_scoped_tools_nested_shape(self):
        agent = Agent(
            "billing",
            system_prompt="You handle billing.",
            tools=[lookup_invoice],
            routes=["human"],
        )
        captured: dict = {}
        provider = _live(captured, _Response(_Message(content="hi"), "stop"))
        asyncio.run(provider.decide(agent, messages=[{"role": "user", "content": "q"}], tool_messages=[]))

        offered = {t["function"]["name"] for t in captured["tools"]}
        self.assertEqual(offered, {"lookup_invoice", "route"})
        # nested Chat-Completions shape (not the flat Response.tools shape)
        inv = next(t for t in captured["tools"] if t["function"]["name"] == "lookup_invoice")
        self.assertEqual(inv["type"], "function")
        self.assertIn("parameters", inv["function"])
        # route tool enum is THIS agent's own routes
        route = next(t for t in captured["tools"] if t["function"]["name"] == ROUTE_TOOL_NAME)
        self.assertEqual(route["function"]["parameters"]["properties"]["agent"]["enum"], ["human"])
        # active agent's system prompt is prepended
        self.assertEqual(captured["messages"][0], {"role": "system", "content": "You handle billing."})

    def test_no_route_tool_when_agent_has_no_routes(self):
        agent = Agent("billing", tools=[lookup_invoice])
        captured: dict = {}
        provider = _live(captured, _Response(_Message(content="hi"), "stop"))
        asyncio.run(provider.decide(agent, messages=[], tool_messages=[]))
        offered = {t["function"]["name"] for t in captured["tools"]}
        self.assertEqual(offered, {"lookup_invoice"})

    def test_tool_call_response_maps_to_calls(self):
        agent = Agent("billing", tools=[lookup_invoice])
        resp = _Response(
            _Message(tool_calls=[_ToolCall("lookup_invoice", '{"invoice_id": "INV-9"}')]),
            "tool_calls",
        )
        d = asyncio.run(_live({}, resp).decide(agent, messages=[], tool_messages=[]))
        self.assertIsNone(d.final)
        self.assertEqual(d.calls[0].name, "lookup_invoice")
        self.assertEqual(json.loads(d.calls[0].arguments), {"invoice_id": "INV-9"})

    def test_tool_call_response_carries_intermediate_text(self):
        # A model that speaks to the user AND calls a tool in the same turn:
        # the text is carried as `text` on the decision (non-terminal), not as
        # `final`, so the orchestrator can surface it before the handoff.
        agent = Agent("billing", tools=[lookup_invoice], routes=["router"])
        resp = _Response(
            _Message(
                content="I'll send you back to the front desk.",
                tool_calls=[_ToolCall("route", '{"agent": "router"}')],
            ),
            "tool_calls",
        )
        d = asyncio.run(_live({}, resp).decide(agent, messages=[], tool_messages=[]))
        self.assertIsNone(d.final)
        self.assertEqual(d.text, "I'll send you back to the front desk.")
        self.assertEqual(d.calls[0].name, "route")

    def test_tool_call_response_without_text_has_no_text(self):
        agent = Agent("billing", tools=[lookup_invoice])
        resp = _Response(
            _Message(tool_calls=[_ToolCall("lookup_invoice", "{}")]),
            "tool_calls",
        )
        d = asyncio.run(_live({}, resp).decide(agent, messages=[], tool_messages=[]))
        self.assertIsNone(d.text)

    def test_text_response_maps_to_final(self):
        agent = Agent("billing", tools=[lookup_invoice])
        resp = _Response(_Message(content="All done.", reasoning_content="because"), "stop")
        d = asyncio.run(_live({}, resp).decide(agent, messages=[], tool_messages=[]))
        self.assertEqual(d.final, "All done.")
        self.assertEqual(d.reasoning, "because")

    def test_acompletion_exception_degrades_to_final(self):
        agent = Agent("billing", tools=[lookup_invoice])
        d = asyncio.run(_live({}, raises=RuntimeError("boom")).decide(agent, messages=[], tool_messages=[]))
        self.assertIsNotNone(d.final)
        self.assertIn("boom", d.final)
        self.assertFalse(d.calls)


class ParallelDecisionTests(unittest.TestCase):
    def test_parallel_yields_multiple_calls_in_order(self):
        agent = Agent(
            "a",
            tools=[lookup_invoice],
            script=[
                Parallel([
                    Call(lookup_invoice, {"invoice_id": "INV-1"}),
                    Call(lookup_invoice, {"invoice_id": "INV-2"}),
                ]),
            ],
        )
        d = _decide(DeterministicProvider(), agent)
        self.assertIsNone(d.final)
        self.assertEqual([c.name for c in d.calls], ["lookup_invoice", "lookup_invoice"])
        self.assertEqual(json.loads(d.calls[0].arguments), {"invoice_id": "INV-1"})
        self.assertEqual(json.loads(d.calls[1].arguments), {"invoice_id": "INV-2"})

    def test_parallel_mixes_valid_and_malformed(self):
        agent = Agent(
            "a",
            tools=[lookup_invoice],
            script=[Parallel([Call(lookup_invoice, {"invoice_id": "INV-1"}), MalformedCall(lookup_invoice)])],
        )
        d = _decide(DeterministicProvider(), agent)
        self.assertEqual(len(d.calls), 2)
        self.assertEqual(d.calls[1].arguments, "{intentionally_malformed_json")

    def test_single_call_still_one_element(self):
        agent = Agent("a", tools=[lookup_invoice], script=[Call(lookup_invoice, {"invoice_id": "INV-1"})])
        d = _decide(DeterministicProvider(), agent)
        self.assertEqual(len(d.calls), 1)
        self.assertEqual(d.calls[0].name, "lookup_invoice")
