# tests/test_golden_traces.py
import asyncio
import unittest

from simple_chatbot.responses import build_output_items
from simple_chatbot.scenario_orchestrator import ScenarioOrchestrator
from simple_chatbot.scenario_provider import DeterministicProvider


def _summarize(items: list[dict]) -> list[tuple]:
    """Compact, envelope-id-free projection of an output array.

    Keeps the fields the parent spec defines as the reproducibility contract:
    item type, deterministic call_id, tool name, raw arguments, tool output,
    and final message text. Drops random `fc_`/`fco_`/`msg_`/`rs_` ids.
    """
    out: list[tuple] = []
    for it in items:
        t = it["type"]
        if t == "function_call":
            out.append(("call", it["call_id"], it["name"], it["arguments"]))
        elif t == "function_call_output":
            out.append(("output", it["call_id"], it["output"]))
        elif t == "message":
            out.append(("message", it["content"][0]["text"]))
        elif t == "reasoning":
            out.append(("reasoning",))
        else:
            out.append((t,))
    return out


def _trace(module_name: str, user_text: str) -> list[tuple]:
    import importlib

    scenario = importlib.import_module(module_name).scenario
    orch = ScenarioOrchestrator(scenario, DeterministicProvider())
    result = asyncio.run(orch.chat([{"role": "user", "content": user_text}]))
    return _summarize(build_output_items(result))


class GoldenTraceTests(unittest.TestCase):
    def test_parallel_golden(self):
        self.assertEqual(
            _trace("simple_chatbot.fixtures.parallel", "Lisbon this weekend?"),
            [
                ("call", "call_parallel_1_1", "get_weather", '{"city": "Lisbon"}'),
                ("call", "call_parallel_1_2", "get_events", '{"city": "Lisbon"}'),
                ("output", "call_parallel_1_1", '{"city": "Lisbon", "forecast": "sunny"}'),
                ("output", "call_parallel_1_2", '{"city": "Lisbon", "events": ["jazz festival"]}'),
                ("message", "Lisbon will be sunny, and there's a jazz festival this weekend."),
            ],
        )

    def test_rag_ignore_golden(self):
        self.assertEqual(
            _trace("simple_chatbot.fixtures.rag_ignore", "what's the refund window?"),
            [
                ("call", "call_rag-ignore_1_1", "search_docs", '{"query": "refund window"}'),
                (
                    "output",
                    "call_rag-ignore_1_1",
                    '{"query": "refund window", "passages": ["Refunds are allowed within 14 days of purchase."]}',
                ),
                ("message", "Refunds are available at any time, with no deadline."),
            ],
        )

    def test_cs_routing_golden(self):
        self.assertEqual(
            _trace("simple_chatbot.fixtures.cs_routing", "I want a refund"),
            [
                ("call", "call_cs-routing_1_1", "route", '{"agent": "billing"}'),
                ("output", "call_cs-routing_1_1", '{"routed": true, "agent": "billing"}'),
                ("call", "call_cs-routing_1_2", "lookup_invoice", '{"invoice_id": "INV-1"}'),
                (
                    "output",
                    "call_cs-routing_1_2",
                    '{"invoice_id": "INV-1", "amount_due": "42.00", "status": "open"}',
                ),
                ("call", "call_cs-routing_1_3", "issue_refund", '{"invoice_id": "INV-1", "amount": 42.0}'),
                (
                    "output",
                    "call_cs-routing_1_3",
                    '{"refunded": true, "invoice_id": "INV-1", "amount": 42.0}',
                ),
                ("message", "I've refunded $42.00 against invoice INV-1."),
            ],
        )
