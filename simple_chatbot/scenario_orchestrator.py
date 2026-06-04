# simple_chatbot/scenario_orchestrator.py
"""Runs a Scenario into a ChatResult.

Maintains active-agent state and offers only the active agent's tools each
round. A `route` call switches the active agent. Every round appends to ONE
`tool_messages` list, which the wire layer serializes into one /v1/responses
output array (= one Spectral turn). Tool `call_id`s are deterministic
(`call_<scenario>_<turn>_<seq>`) so repeated runs are reproducible.
"""

from __future__ import annotations

import json

from loguru import logger

from simple_chatbot.agent import ChatResult
from simple_chatbot.scenario import ROUTE_TOOL_NAME, Agent as ScenarioAgent, Scenario
from simple_chatbot.scenario_catalog import build_responses_tools
from simple_chatbot.scenario_provider import DeterministicProvider, ProviderDecision
from simple_chatbot.tools import _parse_args


class ScenarioOrchestrator:
    def __init__(
        self,
        scenario: Scenario,
        provider: DeterministicProvider | None = None,
        max_rounds: int = 8,
    ) -> None:
        self.scenario = scenario
        self.provider = provider or DeterministicProvider()
        self.max_rounds = max_rounds
        self._tools_by_name = {t.name: t for t in scenario.all_tools()}

    async def chat(self, messages: list[dict], start_agent: str | None = None) -> ChatResult:
        turn_idx = sum(1 for m in messages if m.get("role") == "user")
        active = self._resolve_start(start_agent)
        tool_messages: list[dict] = []
        content = ""
        seq = 0

        if active.terminal:
            # Resuming directly into a terminal agent re-emits its escalation
            # and ends the turn (no rounds run).
            content = active.escalation_message or "(escalated)"
        else:
            for _round in range(self.max_rounds):
                decision: ProviderDecision = await self.provider.decide(
                    active, messages=messages, tool_messages=tool_messages
                )

                if decision.final is not None and not decision.calls:
                    content = decision.final
                    break

                tool_calls: list[dict] = []
                for pc in decision.calls:
                    seq += 1
                    call_id = f"call_{self.scenario.id}_{turn_idx}_{seq}"
                    tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": pc.name, "arguments": pc.arguments},
                        }
                    )

                assistant_msg: dict = {"role": "assistant", "content": None, "tool_calls": tool_calls}
                if decision.reasoning:
                    assistant_msg["reasoning_content"] = decision.reasoning
                tool_messages.append(assistant_msg)

                switch_to: str | None = None
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    raw = tc["function"]["arguments"]
                    output, target = self._execute(active, name, raw)
                    if target is not None:
                        switch_to = target
                    tool_messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "name": name, "content": output}
                    )

                if switch_to is not None:
                    active = self.scenario.agent(switch_to)
                    if active.terminal:
                        content = active.escalation_message or "(escalated)"
                        break
            else:
                content = content or "(max rounds reached)"
                logger.bind(scenario=self.scenario.id).warning(
                    "Scenario reached max rounds without a final answer"
                )

        final_messages = list(messages) + tool_messages + [{"role": "assistant", "content": content}]
        return ChatResult(
            content=content,
            retrieved_chunks=[],
            tool_messages=tool_messages,
            final_messages=final_messages,
            responses_tools=build_responses_tools(self.scenario),
            active_agent=active.name,
        )

    def _resolve_start(self, start_agent: str | None) -> ScenarioAgent:
        """Resolve the agent a turn starts in: the requested resume agent when
        valid, else the scenario's entry agent."""
        if start_agent is not None:
            try:
                return self.scenario.agent(start_agent)
            except KeyError:
                logger.bind(
                    scenario=self.scenario.id, start_agent=start_agent
                ).warning("Unknown start_agent; falling back to entry")
        return self.scenario.agent(self.scenario.entry)

    def _execute(self, active, name: str, raw: str) -> tuple[str, str | None]:
        """Return (output_text, switch_target). switch_target is set only for a
        valid route call."""
        if name == ROUTE_TOOL_NAME:
            args, err = _parse_args(ROUTE_TOOL_NAME, raw)
            if err:
                return f"Tool error: {err}", None
            target = args.get("agent")
            if target not in active.routes:
                return (
                    f"Tool error: agent {active.name!r} cannot route to {target!r}",
                    None,
                )
            return json.dumps({"routed": True, "agent": target}), target

        tool = self._tools_by_name.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools_by_name)) or "(none)"
            return f"Tool error: unsupported tool {name!r}; available tools: {available}", None

        args, err = _parse_args(name, raw)
        if err:
            return f"Tool error: {err}", None
        try:
            result = tool.func(**args)
        except Exception as exc:  # missing-required, etc. — surfaced as a tool error
            return f"Tool error: {exc}", None
        return (result if isinstance(result, str) else json.dumps(result)), None
