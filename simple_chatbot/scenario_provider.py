# simple_chatbot/scenario_provider.py
"""Per-round decision providers for the scenario orchestrator.

A provider, given the active agent and the trace so far, decides what the agent
does this round: emit tool calls, or finish with an assistant message. The
DeterministicProvider plays the active agent's authored `script` one step per
round, with a per-agent cursor (a handoff advances to a different agent whose
own script then plays from the top).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from simple_chatbot.scenario import (
    Agent,
    Call,
    Final,
    MalformedCall,
    Route,
    UnknownToolCall,
)

# Invalid-JSON string used for the malformed-args knob. Mirrors the marker the
# existing scripted_llm uses for the same purpose.
_MALFORMED_ARGS = "{intentionally_malformed_json"


@dataclass
class PlannedCall:
    """A tool call the provider wants to emit this round. `arguments` is a raw
    JSON string (intentionally invalid for the malformed-args knob)."""

    name: str
    arguments: str


@dataclass
class ProviderDecision:
    """Either `final` (end the turn) or one-or-more `calls`."""

    final: str | None = None
    calls: list[PlannedCall] = field(default_factory=list)
    reasoning: str | None = None
    ignore_retrieval: bool = False


class DeterministicProvider:
    def __init__(self) -> None:
        self._cursors: dict[str, int] = {}

    async def decide(
        self, agent: Agent, messages: list[dict], tool_messages: list[dict]
    ) -> ProviderDecision:
        cursor = self._cursors.get(agent.name, 0)
        if cursor >= len(agent.script):
            return ProviderDecision(final="(scripted end)")
        step = agent.script[cursor]
        self._cursors[agent.name] = cursor + 1

        if isinstance(step, Final):
            return ProviderDecision(final=step.text, ignore_retrieval=step.ignore_retrieval)
        if isinstance(step, Route):
            return ProviderDecision(
                calls=[PlannedCall("route", json.dumps({"agent": step.target}))]
            )
        if isinstance(step, Call):
            return ProviderDecision(
                calls=[PlannedCall(step.tool.name, json.dumps(step.args))]
            )
        if isinstance(step, MalformedCall):
            return ProviderDecision(calls=[PlannedCall(step.tool.name, _MALFORMED_ARGS)])
        if isinstance(step, UnknownToolCall):
            return ProviderDecision(
                calls=[PlannedCall(step.name, json.dumps(step.args))]
            )
        return ProviderDecision(final="(unknown step)")
