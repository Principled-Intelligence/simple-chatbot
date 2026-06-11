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
    ROUTE_TOOL_NAME,
    Agent,
    Call,
    Final,
    MalformedCall,
    Parallel,
    Route,
    UnknownToolCall,
)
from simple_chatbot.tools import MALFORMED_TOOL_ARGS as _MALFORMED_ARGS

# Keys carried on trace messages for the /v1/responses adapter that providers
# reject when an assistant message is replayed on the next round (mirrors the
# single-agent RAG path's `_sanitize_assistant_dump`). The trace itself keeps
# them; only the copy fed back to the model is stripped.
_REPLAY_UNSAFE_KEYS = (
    "reasoning_content",
    "provider_specific_fields",
    "thinking_blocks",
    "audio",
)


def _safe_for_replay(msg: dict) -> dict:
    """Return a copy of a trace message safe to re-feed to the provider."""
    if not any(k in msg for k in _REPLAY_UNSAFE_KEYS):
        return msg
    return {k: v for k, v in msg.items() if k not in _REPLAY_UNSAFE_KEYS}


def _planned_call(step) -> "PlannedCall":
    """Map a single call-family step to a PlannedCall."""
    if isinstance(step, Call):
        return PlannedCall(step.tool.name, json.dumps(step.args))
    if isinstance(step, MalformedCall):
        return PlannedCall(step.tool.name, _MALFORMED_ARGS)
    if isinstance(step, UnknownToolCall):
        return PlannedCall(step.name, json.dumps(step.args))
    raise ValueError(f"not a call-family step: {type(step).__name__}")


@dataclass
class PlannedCall:
    """A tool call the provider wants to emit this round. `arguments` is a raw
    JSON string (intentionally invalid for the malformed-args knob)."""

    name: str
    arguments: str


@dataclass
class ProviderDecision:
    """Either `final` (end the turn) or one-or-more `calls`.

    `text` is non-terminal user-facing text emitted ALONGSIDE `calls` (the agent
    says something, then hands off / calls a tool without ending its turn). It is
    distinct from `final`, which ends the turn.
    """

    final: str | None = None
    calls: list[PlannedCall] = field(default_factory=list)
    reasoning: str | None = None
    ignore_retrieval: bool = False
    text: str | None = None


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
        if isinstance(step, (Call, MalformedCall, UnknownToolCall)):
            return ProviderDecision(calls=[_planned_call(step)])
        if isinstance(step, Parallel):
            return ProviderDecision(calls=[_planned_call(s) for s in step.steps])
        return ProviderDecision(final="(unknown step)")


class LiveProvider:
    """Model-driven provider: calls `acompletion` once per round with the active
    agent's system prompt + scoped tools, ignoring the authored `script`.

    Tools are offered in the NESTED Chat-Completions shape
    `{type, function: {name, description, parameters}}` (what litellm's `tools=`
    expects) — distinct from the flat `Response.tools` catalog shape. The `route`
    tool's enum is the ACTIVE agent's own routes, not the scenario-wide union.
    """

    def __init__(
        self,
        acompletion,
        model: str,
        api_base: str | None = None,
        sampling_kwargs: dict | None = None,
    ) -> None:
        self._acompletion = acompletion
        self._model = model
        self._api_base = api_base
        self._sampling_kwargs = sampling_kwargs or {}

    def _offered_tools(self, agent: Agent) -> list[dict]:
        tools: list[dict] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in agent.tools
        ]
        if agent.routes:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": ROUTE_TOOL_NAME,
                        "description": "Hand off the request to another agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "agent": {"type": "string", "enum": list(agent.routes)}
                            },
                            "required": ["agent"],
                        },
                    },
                }
            )
        return tools

    async def decide(
        self, agent: Agent, messages: list[dict], tool_messages: list[dict]
    ) -> ProviderDecision:
        working: list[dict] = []
        if agent.system_prompt:
            working.append({"role": "system", "content": agent.system_prompt})
        working.extend(messages)
        working.extend(_safe_for_replay(m) for m in tool_messages)

        kwargs: dict = {
            "model": self._model,
            "messages": working,
            "tools": self._offered_tools(agent),
            "tool_choice": "auto",
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        kwargs.update(self._sampling_kwargs)

        try:
            response = await self._acompletion(**kwargs)
            choice = response.choices[0]
        except Exception as exc:  # live is best-effort: degrade, don't 500
            return ProviderDecision(final=f"(live provider error: {exc})")

        msg = choice.message
        reasoning = getattr(msg, "reasoning_content", None)
        tool_calls = getattr(msg, "tool_calls", None)
        if choice.finish_reason == "tool_calls" and tool_calls:
            calls = []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                calls.append(
                    PlannedCall(
                        name=getattr(fn, "name", None),
                        arguments=getattr(fn, "arguments", "{}"),
                    )
                )
            # The model may speak to the user AND call a tool in the same turn.
            # Carry that text as non-terminal `text` (empty/None -> no text).
            return ProviderDecision(
                calls=calls, reasoning=reasoning, text=(msg.content or None)
            )
        return ProviderDecision(final=(msg.content or ""), reasoning=reasoning)


# Either provider satisfies the orchestrator's `decide()` seam.
Provider = DeterministicProvider | LiveProvider
