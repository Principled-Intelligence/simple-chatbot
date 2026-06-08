import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

import simple_chatbot.server as server
from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.conversation_state import ConversationStateStore, ConvState
from simple_chatbot.responses import ResponseStore
from simple_chatbot.scenario_orchestrator import ScenarioOrchestrator
from simple_chatbot.scenario_provider import LiveProvider
from simple_chatbot.fixtures.support_swarm import (
    scenario,
    get_order,
    cancel_order,
    get_customer,
    update_customer,
    lookup_invoice,
    issue_refund,
)


# --- State semantics: call the mock tool functions directly (no LLM) ---------

class StatefulToolTests(unittest.TestCase):
    def test_seeded_world_on_fresh_state(self):
        state = ConvState()
        self.assertEqual(get_order.func(order_id="ORD-1", state=state)["status"], "open")
        self.assertEqual(get_customer.func(state=state)["email"], "ada@example.com")
        self.assertEqual(lookup_invoice.func(invoice_id="INV-1", state=state)["status"], "open")

    def test_cancel_then_get_reflects_cancelled(self):
        state = ConvState()
        cancel_order.func(order_id="ORD-1", state=state)
        self.assertEqual(get_order.func(order_id="ORD-1", state=state)["status"], "cancelled")

    def test_update_customer_persists(self):
        state = ConvState()
        update_customer.func(field="email", value="grace@example.com", state=state)
        self.assertEqual(get_customer.func(state=state)["email"], "grace@example.com")

    def test_issue_refund_flips_invoice_status(self):
        state = ConvState()
        issue_refund.func(invoice_id="INV-1", amount=42.0, state=state)
        self.assertEqual(lookup_invoice.func(invoice_id="INV-1", state=state)["status"], "refunded")

    def test_isolation_between_states(self):
        a, b = ConvState(), ConvState()
        cancel_order.func(order_id="ORD-1", state=a)
        self.assertEqual(get_order.func(order_id="ORD-1", state=a)["status"], "cancelled")
        self.assertEqual(get_order.func(order_id="ORD-1", state=b)["status"], "open")

    def test_unknown_order_returns_error(self):
        self.assertIn("error", get_order.func(order_id="NOPE", state=ConvState()))


# --- Fake sequenced LLM for the live path ------------------------------------

class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, name, arguments):
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, content="", tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, message, finish_reason):
        self.choices = [_Choice(message, finish_reason)]
        self.usage = None


def _call(name, args):
    return _Resp(_Msg(tool_calls=[_TC(name, json.dumps(args))]), "tool_calls")


def _final(text):
    return _Resp(_Msg(content=text), "stop")


def _seq_acompletion(responses):
    it = iter(responses)

    async def fake(**kwargs):
        return next(it)

    return fake


class _FakeConversationLogger:
    async def log(self, conversation_id, messages, response, chunks) -> None:
        return None


class LiveSwarmTests(unittest.TestCase):
    def _orch(self, responses):
        provider = LiveProvider(acompletion=_seq_acompletion(responses), model="test-model")
        return ScenarioOrchestrator(scenario, provider)

    def test_router_handoff_and_stateful_tool(self):
        responses = [
            _call("route", {"agent": "order_tracking"}),
            _call("cancel_order", {"order_id": "ORD-1"}),
            _final("Your order ORD-1 is cancelled."),
        ]
        state = ConvState()
        result = asyncio.run(
            self._orch(responses).chat(
                [{"role": "user", "content": "cancel my order ORD-1"}], state=state
            )
        )
        self.assertEqual(result.active_agent, "order_tracking")
        self.assertEqual(result.content, "Your order ORD-1 is cancelled.")
        self.assertEqual(state["orders"]["ORD-1"]["status"], "cancelled")
        names = [m["name"] for m in result.tool_messages if m["role"] == "tool"]
        self.assertEqual(names, ["route", "cancel_order"])

    def test_route_back_to_router_then_other_subagent(self):
        responses = [
            _call("route", {"agent": "order_tracking"}),
            _call("route", {"agent": "router"}),
            _call("route", {"agent": "billing"}),
            _call("lookup_invoice", {"invoice_id": "INV-1"}),
            _final("Invoice INV-1 is open for $42.00."),
        ]
        result = asyncio.run(
            self._orch(responses).chat(
                [{"role": "user", "content": "actually, a billing question"}], state=ConvState()
            )
        )
        self.assertEqual(result.active_agent, "billing")
        names = [m["name"] for m in result.tool_messages if m["role"] == "tool"]
        self.assertEqual(names, ["route", "route", "route", "lookup_invoice"])


# --- Server integration: state persists across chained turns -----------------

class ServerSwarmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = (
            server._config,
            server._agent,
            server._conversation_logger,
            getattr(server, "_response_store", None),
            getattr(server, "_scenario_registry", None),
            getattr(server, "_acompletion", None),
            getattr(server, "_conversation_state_store", None),
        )
        self.addCleanup(self._restore)

        server._config = SimpleChatbotConfig(
            docs_dir=Path(self.tmp.name) / "docs",
            conversation_log_dir=Path(self.tmp.name) / "conversations",
        )
        server._agent = None
        server._conversation_logger = _FakeConversationLogger()
        server._response_store = ResponseStore()
        server._conversation_state_store = ConversationStateStore()
        server._scenario_registry = {"support-swarm": scenario}
        # Turn 1: route -> cancel -> final.  Turn 2 (resumes order_tracking): get -> final.
        server._acompletion = _seq_acompletion(
            [
                _call("route", {"agent": "order_tracking"}),
                _call("cancel_order", {"order_id": "ORD-1"}),
                _final("Cancelled ORD-1."),
                _call("get_order", {"order_id": "ORD-1"}),
                _final("ORD-1 status reported."),
            ]
        )
        self.client = TestClient(server.app)

    def _restore(self):
        (
            server._config,
            server._agent,
            server._conversation_logger,
            server._response_store,
            server._scenario_registry,
            server._acompletion,
            server._conversation_state_store,
        ) = self.old

    def test_state_persists_across_chained_turns(self):
        r1 = self.client.post(
            "/v1/responses",
            json={"model": "support-swarm", "input": "cancel ORD-1", "user": "conv-x"},
        )
        self.assertEqual(r1.status_code, 200, r1.text)
        first_id = r1.json()["id"]

        # The mutation landed in the conversation's state.
        state = asyncio.run(server._conversation_state_store.get_or_create("conv-x"))
        self.assertEqual(state["orders"]["ORD-1"]["status"], "cancelled")

        # Turn 2 resumes the same conversation; get_order sees the cancellation.
        r2 = self.client.post(
            "/v1/responses",
            json={
                "model": "support-swarm",
                "input": "what's the status now?",
                "previous_response_id": first_id,
                "user": "conv-x",
            },
        )
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertIn("cancelled", r2.text)


if __name__ == "__main__":
    unittest.main()
