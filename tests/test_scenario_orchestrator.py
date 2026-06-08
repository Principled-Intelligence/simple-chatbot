# tests/test_scenario_orchestrator.py
import asyncio
import json
import unittest

from simple_chatbot.conversation_state import ConvState
from simple_chatbot.scenario import Agent, Call, Final, MalformedCall, Route, Scenario, UnknownToolCall, tool
from simple_chatbot.scenario_provider import DeterministicProvider
from simple_chatbot.scenario_orchestrator import ESCALATED_CLOSED_MESSAGE, ScenarioOrchestrator


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice."""
    return {"invoice_id": invoice_id, "amount_due": "42.00"}


def _run(scenario, messages):
    orch = ScenarioOrchestrator(scenario, DeterministicProvider())
    return asyncio.run(orch.chat(messages))


class OrchestratorRoutingTests(unittest.TestCase):
    def _scenario(self):
        return Scenario(
            id="cs",
            entry="dispatcher",
            agents=[
                Agent("dispatcher", routes=["billing"], script=[Route("billing")]),
                Agent(
                    "billing",
                    tools=[lookup_invoice],
                    script=[Call(lookup_invoice, {"invoice_id": "INV-1"}), Final("Refunded.")],
                ),
            ],
        )

    def test_one_turn_collapses_route_and_subagent_calls(self):
        result = _run(self._scenario(), [{"role": "user", "content": "refund please"}])
        self.assertEqual(result.content, "Refunded.")
        roles_and_names = [
            (m["role"], m.get("name") or (m["tool_calls"][0]["function"]["name"] if m.get("tool_calls") else None))
            for m in result.tool_messages
        ]
        # assistant(route) → tool(route) → assistant(lookup) → tool(lookup)
        self.assertEqual(
            roles_and_names,
            [("assistant", "route"), ("tool", "route"),
             ("assistant", "lookup_invoice"), ("tool", "lookup_invoice")],
        )

    def test_route_output_and_handoff(self):
        result = _run(self._scenario(), [{"role": "user", "content": "x"}])
        route_out = next(m for m in result.tool_messages if m["role"] == "tool" and m["name"] == "route")
        self.assertEqual(json.loads(route_out["content"]), {"routed": True, "agent": "billing"})

    def test_stable_call_ids(self):
        msgs = [{"role": "user", "content": "x"}]
        r1 = _run(self._scenario(), msgs)
        r2 = _run(self._scenario(), msgs)
        ids1 = [m["tool_call_id"] for m in r1.tool_messages if m["role"] == "tool"]
        ids2 = [m["tool_call_id"] for m in r2.tool_messages if m["role"] == "tool"]
        self.assertEqual(ids1, ids2)
        self.assertEqual(ids1, ["call_cs_1_1", "call_cs_1_2"])

    def test_responses_tools_attached(self):
        result = _run(self._scenario(), [{"role": "user", "content": "x"}])
        names = {e["name"] for e in result.responses_tools}
        self.assertEqual(names, {"lookup_invoice", "route"})


class _ScriptedDecisionProvider:
    """Returns a pre-built list of ProviderDecisions, one per round."""

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self._i = 0

    async def decide(self, agent, messages, tool_messages):
        d = self._decisions[self._i]
        self._i += 1
        return d


class OrchestratorIntermediateTextTests(unittest.TestCase):
    def _scenario(self):
        front = Agent("front", routes=["billing"])
        billing = Agent("billing", tools=[lookup_invoice])
        return Scenario(id="s", entry="front", agents=[front, billing])

    def test_intermediate_text_alongside_route_stored_on_assistant_message(self):
        from simple_chatbot.scenario_provider import PlannedCall, ProviderDecision

        provider = _ScriptedDecisionProvider(
            [
                ProviderDecision(
                    text="I'll connect you to billing.",
                    calls=[PlannedCall("route", json.dumps({"agent": "billing"}))],
                ),
                ProviderDecision(final="How can I help with billing?"),
            ]
        )
        orch = ScenarioOrchestrator(self._scenario(), provider)
        result = asyncio.run(orch.chat([{"role": "user", "content": "hi"}]))

        # Terminal answer is the final text.
        self.assertEqual(result.content, "How can I help with billing?")
        # The routing assistant turn carries the non-terminal user-facing text.
        route_assistant = next(
            m for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        )
        self.assertEqual(route_assistant["content"], "I'll connect you to billing.")

    def test_no_intermediate_text_keeps_content_none(self):
        from simple_chatbot.scenario_provider import PlannedCall, ProviderDecision

        provider = _ScriptedDecisionProvider(
            [
                ProviderDecision(
                    calls=[PlannedCall("route", json.dumps({"agent": "billing"}))],
                ),
                ProviderDecision(final="done"),
            ]
        )
        orch = ScenarioOrchestrator(self._scenario(), provider)
        result = asyncio.run(orch.chat([{"role": "user", "content": "hi"}]))
        route_assistant = next(
            m for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        )
        self.assertIsNone(route_assistant["content"])


class OrchestratorTerminalTests(unittest.TestCase):
    def test_terminal_agent_emits_escalation_final(self):
        s = Scenario(
            id="esc",
            entry="dispatcher",
            agents=[
                Agent("dispatcher", routes=["human"], script=[Route("human")]),
                Agent("human", terminal=True, escalation_message="Escalating to a human."),
            ],
        )
        result = _run(s, [{"role": "user", "content": "help"}])
        self.assertEqual(result.content, "Escalating to a human.")


class OrchestratorKnobTests(unittest.TestCase):
    def test_malformed_args_surface_as_tool_error(self):
        s = Scenario(
            id="k",
            entry="a",
            agents=[Agent("a", tools=[lookup_invoice],
                          script=[MalformedCall(lookup_invoice), Final("ok")])],
        )
        result = _run(s, [{"role": "user", "content": "x"}])
        out = next(m for m in result.tool_messages if m["role"] == "tool")
        self.assertIn("Tool error", out["content"])
        # the malformed args are still present on the call (for validity scoring)
        call = next(m for m in result.tool_messages if m["role"] == "assistant")
        self.assertEqual(call["tool_calls"][0]["function"]["arguments"], "{intentionally_malformed_json")

    def test_unknown_tool_surface_as_tool_error(self):
        s = Scenario(id="k", entry="a",
                     agents=[Agent("a", script=[UnknownToolCall("ghost"), Final("ok")])])
        result = _run(s, [{"role": "user", "content": "x"}])
        out = next(m for m in result.tool_messages if m["role"] == "tool")
        self.assertIn("unsupported tool 'ghost'", out["content"])

    def test_required_violation_keeps_empty_args_and_errors(self):
        s = Scenario(id="k", entry="a",
                     agents=[Agent("a", tools=[lookup_invoice],
                                   script=[Call(lookup_invoice), Final("ok")])])
        result = _run(s, [{"role": "user", "content": "x"}])
        call = next(m for m in result.tool_messages if m["role"] == "assistant")
        self.assertEqual(call["tool_calls"][0]["function"]["arguments"], "{}")
        out = next(m for m in result.tool_messages if m["role"] == "tool")
        self.assertIn("Tool error", out["content"])


class OrchestratorResumeTests(unittest.TestCase):
    def _scenario(self):
        return Scenario(
            id="cs",
            entry="dispatcher",
            agents=[
                Agent("dispatcher", routes=["billing", "human"], script=[Route("billing")]),
                Agent(
                    "billing",
                    tools=[lookup_invoice],
                    script=[Call(lookup_invoice, {"invoice_id": "INV-1"}), Final("Refunded.")],
                ),
                Agent("human", terminal=True, escalation_message="Escalating to a human."),
            ],
        )

    def _run(self, scenario, messages, start_agent=None):
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat(messages, start_agent=start_agent))

    def test_reports_active_agent_at_turn_end(self):
        result = self._run(self._scenario(), [{"role": "user", "content": "x"}])
        self.assertEqual(result.active_agent, "billing")

    def test_resume_starts_in_given_agent_without_routing(self):
        result = self._run(
            self._scenario(), [{"role": "user", "content": "x"}], start_agent="billing"
        )
        call_names = [
            m["tool_calls"][0]["function"]["name"]
            for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        self.assertNotIn("route", call_names)  # did NOT re-enter dispatcher
        self.assertEqual(call_names[0], "lookup_invoice")
        self.assertEqual(result.content, "Refunded.")
        self.assertEqual(result.active_agent, "billing")

    def test_unknown_start_agent_falls_back_to_entry(self):
        result = self._run(
            self._scenario(), [{"role": "user", "content": "x"}], start_agent="ghost"
        )
        first_call = next(
            m["tool_calls"][0]["function"]["name"]
            for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        )
        self.assertEqual(first_call, "route")  # entry dispatcher ran

    def test_resume_into_terminal_returns_closed_message(self):
        # A prior turn escalated to human; resuming the conversation must NOT
        # replay the escalation — it returns the closed/start-a-new-chat message.
        result = self._run(
            self._scenario(), [{"role": "user", "content": "x"}], start_agent="human"
        )
        self.assertEqual(result.content, ESCALATED_CLOSED_MESSAGE)
        self.assertNotEqual(result.content, "Escalating to a human.")
        self.assertEqual(result.tool_messages, [])  # no rounds executed
        self.assertEqual(result.active_agent, "human")

    def test_chat_without_start_agent_unchanged(self):
        # default path (no start_agent) still enters the entry agent
        result = self._run(self._scenario(), [{"role": "user", "content": "x"}])
        first_call = next(
            m["tool_calls"][0]["function"]["name"]
            for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        )
        self.assertEqual(first_call, "route")


@tool
def bump(state) -> dict:
    """Increment a per-conversation counter and return it."""
    state["count"] = state.get("count", 0) + 1
    return {"count": state["count"]}


class OrchestratorStateInjectionTests(unittest.TestCase):
    def _scenario(self):
        return Scenario(
            id="counter",
            entry="worker",
            agents=[
                Agent("worker", tools=[bump], script=[Call(bump, {}), Final("done")]),
            ],
        )

    def test_stateful_tool_receives_and_mutates_injected_state(self):
        state = ConvState()
        orch = ScenarioOrchestrator(self._scenario(), DeterministicProvider())
        result = asyncio.run(orch.chat([{"role": "user", "content": "go"}], state=state))
        bump_out = next(m for m in result.tool_messages if m.get("name") == "bump")
        self.assertEqual(json.loads(bump_out["content"]), {"count": 1})
        self.assertEqual(state["count"], 1)

    def test_stateful_tool_works_without_explicit_state(self):
        orch = ScenarioOrchestrator(self._scenario(), DeterministicProvider())
        result = asyncio.run(orch.chat([{"role": "user", "content": "go"}]))
        bump_out = next(m for m in result.tool_messages if m.get("name") == "bump")
        self.assertEqual(json.loads(bump_out["content"]), {"count": 1})
