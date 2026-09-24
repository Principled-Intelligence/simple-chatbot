# Scenario Fixture Engine — Phase 2 (Live Mode) design

**Date:** 2026-06-03
**Status:** Approved (pending spec review)
**Author:** Niccolò Campolungo / Claude
**Repo:** `simple-chatbot` (this repo only — ghost-backend is untouched)
**Parent spec:** [2026-06-03-scenario-fixture-authoring-design.md](2026-06-03-scenario-fixture-authoring-design.md) (see "Modes" + the Phasing table; this is Phase 2)
**Phase 1 plan:** [../plans/2026-06-03-scenario-fixture-engine-phase1.md](../plans/2026-06-03-scenario-fixture-engine-phase1.md)

## Goal

Add a **live, model-driven** path to the scenario fixture engine. A `LiveProvider`
calls a real model (via `litellm`) once per round with the active agent's
`system_prompt` + scoped tools, ignoring the authored `script`, and maps the
model's tool-call / final-message response into the existing `ProviderDecision`.
A mode switch (per-scenario default + server override) chooses deterministic vs
live. Deterministic mode remains the canonical, CI-friendly, reproducible path;
live mode is best-effort and exists to surface platform gaps for Spectral.

This phase also lands **active-agent persistence across user turns** (a Phase 1
open seam): a follow-up message that chains via `previous_response_id` resumes
with the agent that was in charge at the end of the prior turn, rather than
re-entering the entry agent.

## What Phase 1 already built (the seams this plugs into)

- **`scenario_provider.py`** — `PlannedCall`, `ProviderDecision(final, calls,
  reasoning, ignore_retrieval)`, and `DeterministicProvider`. The provider seam
  is `async def decide(agent, messages, tool_messages) -> ProviderDecision`.
- **`scenario_orchestrator.py`** — `ScenarioOrchestrator(scenario, provider,
  max_rounds)` owns the round loop: it calls `provider.decide(...)` once per
  round, builds the assistant `tool_calls` message + `tool` result messages,
  generates deterministic `call_id`s (`call_<scenario>_<turn>_<seq>`), switches
  the active agent on a `route` call, collapses the whole topology into one
  `tool_messages` list, and emits a `ChatResult`. **The round loop needs no
  change for live mode** — `LiveProvider` is just a different `decide()`.
- **`scenario_catalog.py`** — `build_responses_tools(scenario)` projects tools
  into the **flat** Responses-API `Response.tools` shape `{type, name,
  parameters, description}` (union of all agents' tools + one `route` tool whose
  enum is the union of all routing targets).
- **`agent.py`** — the existing single-agent path already drives `litellm`
  via an injected `acompletion` callable, with `_sanitize_assistant_dump`,
  `_sampling_kwargs`, `chat_api_base` handling, and a forced-final fallback.
  `LiveProvider` reuses these patterns. `ChatResult` is the shared return type.
- **`server.py`** — `init(config, indexer, acompletion, tools)` wires the
  `acompletion` into the `Agent`; `/v1/responses` selects a scenario by
  `body.model` and runs `ScenarioOrchestrator(scenario, DeterministicProvider())`;
  the response store persists `session_messages` keyed by response id and
  rehydrates them on a `previous_response_id` turn.

## Locked decisions (from brainstorming)

| Topic | Decision | Rationale |
|---|---|---|
| **Live engine** | **Hand-rolled `litellm` loop behind the existing `decide()` seam.** No OpenAI Agents SDK. | The seam is per-round/single-step; the SDK wants to own the whole loop + handoffs, which would mean bypassing the orchestrator and re-serializing SDK run items into our wire shape — exactly the "client-role re-serialization" the parent spec argues against. Hand-rolled keeps one orchestrator, one wire path, and a key-free CI. |
| **Active-agent persistence** | **Persist the in-charge agent across turns, in both modes.** | A routed follow-up (`previous_response_id`) must resume *as the routed agent* (its prompt + scoped tools), not re-enter the dispatcher. `previous_response_id` already restores message history; the active-agent pointer is the missing piece (it is orchestrator state, not message content). |
| **Terminal on resume** | **A resumed terminal agent ends the turn with a "conversation closed" message (`ESCALATED_CLOSED_MESSAGE`), not a replay of `escalation_message`.** | **(Revised — superseded the original "re-emit" rule, see note below.)** "Handed to a human" is a sticky, *closed* state; replaying the escalation line on every follow-up reads as a loop. Announce the escalation once, on the turn it happens; afterwards tell the user the chat is closed and to start a new one. The original direct-entry-into-terminal case (entry agent is terminal, not a resume) still emits `escalation_message`. |
| **Knobs in live mode** | **Live mode ignores `script` entirely; misbehavior knobs are deterministic-mode-only (documented).** | Simplest and most honest: no coupling between `LiveProvider` and the authored script. A "misbehavior" fixture run live simply produces well-formed, model-driven calls. |
| **Mode switch** | **Per-scenario `mode` default + server `scenario_mode` override; override wins.** | Deterministic stays the canonical default; an operator can force a whole server live without editing fixtures. |
| **Live model config** | **Reuse the `acompletion` handed to `init()` plus `config.chat_model` / `chat_api_base` / sampling params.** | The scenario id is the wire `model` (used for selection), so it cannot also name the upstream model — live mode targets the server's configured chat model. |

## Non-goals

- OpenAI Agents SDK backing (documented as a possible future alternate live
  flavor that would emit SDK-native `transfer_to_<agent>` handoff traces).
- Live-mode misbehavior knobs / prompt nudges (deterministic-only).
- Reproducible/golden traces for live mode (live is best-effort by definition).
- Phase 3 (broader knob/fixture library) and Phase 4 (ghost-backend
  `Response.tools` harvest — author-gated; **do not start without explicit
  go-ahead**). ghost-backend is not touched in this phase.

## Architecture

No new top-level component: Phase 2 adds a second `Provider` implementation, a
small amount of orchestrator/`ChatResult` state for resume, and provider
selection + persistence wiring in the server.

```
/v1/responses (model=<scenario id>)
        │
        │  resolve mode = config.scenario_mode or scenario.mode
        ▼
ScenarioOrchestrator(scenario, provider, max_rounds)
        │  active = start_agent (resumed) or scenario.entry
        │  per round: provider.decide(active, messages, tool_messages)
        │      ├── DeterministicProvider → plays script (canonical)
        │      └── LiveProvider → litellm(active.system_prompt + scoped tools)
        ▼
ChatResult(content, tool_messages, responses_tools, active_agent=<in-charge at end>)
        │
build_response → output[] + Response.tools     response store ← session_messages + active_agent
```

### Component changes

- **`scenario_provider.py` — add `LiveProvider`.**
  `LiveProvider(acompletion, model, api_base=None, sampling_kwargs=None)`,
  stateless across rounds. `decide(agent, messages, tool_messages)`:
  1. Build working messages: `[{"role":"system","content":agent.system_prompt}]
     + messages + tool_messages`. (`tool_messages` is already in clean
     chat-completion shape — assistant `tool_calls` + `tool` results — built by
     the orchestrator, so it feeds straight back.)
  2. Offer **only the active agent's scoped tools**, built in the **nested
     Chat-Completions tool shape** `{"type":"function","function":{"name",
     "description","parameters"}}` from each `ScenarioTool`, plus a `route` tool
     whose `agent` enum is **this agent's own `routes`** (present only if the
     agent has routes). NOTE: this nested shape differs from the flat
     `Response.tools` catalog shape — the catalog stays flat; the offered tools
     are nested because that is what `litellm`'s `tools=` expects (mirrors
     `ToolDef.schema`, keyed at `schema["function"]["name"]`).
  3. Call `await acompletion(model=model, messages=working, tools=offered,
     tool_choice="auto", api_base=api_base, **sampling_kwargs)`.
  4. Map the response: `finish_reason == "tool_calls"` →
     `ProviderDecision(calls=[PlannedCall(fn.name, fn.arguments_str), …],
     reasoning=reasoning_content)`; otherwise
     `ProviderDecision(final=content, reasoning=reasoning_content)`.
  5. On a `litellm`/`acompletion` exception: return a turn-ending
     `ProviderDecision(final="<error message>")` (mirrors `agent.py`'s
     forced-final fallback) so a live failure degrades gracefully rather than
     500-ing the request.

- **`agent.py` — small refactor + one field.**
  Extract the body of `Agent._sampling_kwargs` into a module-level
  `sampling_kwargs(config) -> dict` reused by both `Agent` and the server when
  constructing `LiveProvider` (identical wire kwargs from one source). Add
  `active_agent: str | None = None` to `ChatResult`.

- **`scenario_orchestrator.py` — resume + report active agent.**
  `chat(messages, start_agent: str | None = None)`:
  - `active = scenario.agent(start_agent)` when `start_agent` is a valid agent
    name, else `scenario.agent(scenario.entry)` (unknown name → entry fallback +
    a `logger.warning`).
  - If the resolved start agent is `terminal`: when it is a *resume* of the same
    terminal agent (`start_agent` matches), end the turn with
    `ESCALATED_CLOSED_MESSAGE` (the conversation is closed — see the revised
    "Terminal on resume" decision above). When it is direct *entry* into a
    terminal agent (not a resume), emit its `escalation_message` (or
    `"(escalated)"`). Either way, end the turn immediately.
  - At turn end, set `ChatResult.active_agent = active.name`.

- **`scenario.py` — mode field.**
  Add `mode: Literal["deterministic","live"] = "deterministic"` to `Scenario`.
  Pure metadata; no validation change beyond accepting the field.

- **`config.py` — server override.**
  Add `scenario_mode: Literal["deterministic","live"] | None = None` to
  `SimpleChatbotConfig` (`None` = honor the per-scenario default). CLI/env may
  populate it; no validator change.

- **`server.py` — capture acompletion, select provider, persist active agent.**
  - `init()`: store the injected `acompletion` in a module global `_acompletion`
    (today it is only handed to `Agent`).
  - `responses_create()`: when `body.model` resolves to a scenario, compute
    `resolved_mode = config.scenario_mode or scenario.mode`; build
    `LiveProvider(acompletion=_acompletion, model=config.chat_model,
    api_base=config.chat_api_base, sampling_kwargs=sampling_kwargs(config))`
    for live, else `DeterministicProvider()`. On a `previous_response_id` turn,
    read `prior_entry.get("active_agent")` and pass it as `start_agent` **only
    when the resumed request resolves to the same scenario** (same `body.model`);
    otherwise ignore it. Persist `result.active_agent` in the response-store
    entry alongside `session_messages`.

## Active-agent persistence — data flow

1. Turn 1 (`model="cs-routing"`): dispatcher routes to billing; billing answers.
   `ChatResult.active_agent == "billing"`. The store entry for the new
   response id holds `session_messages` (full history) **and**
   `active_agent="billing"`.
2. Turn 2 (`model="cs-routing"`, `previous_response_id=<turn 1>`): the server
   rehydrates `session_messages` (existing behavior) **and** reads
   `active_agent="billing"`, passing `start_agent="billing"`. The orchestrator
   starts in billing — billing's `system_prompt` + scoped tools are live from
   round 1, with full prior context already in the messages. No re-route.
3. **Self-activating:** in a scenario where no routing occurred, `active_agent`
   is just the entry agent, so resume is identical to today's behavior.
4. **Guard:** if turn 2 resolves to a different scenario (or the non-scenario
   agent path), the persisted `active_agent` is ignored; an unknown/stale agent
   name falls back to `entry`.

## Error handling & key-free CI

- Deterministic remains the default, so **CI requires no API key**.
- `LiveProvider` unit/integration tests inject a **fake `acompletion`** (the same
  technique the existing agent tests use), keeping the live path testable
  without a real key or network.
- A `LiveProvider` model-call exception degrades to a turn-ending error `final`;
  the existing `max_rounds` guard prevents runaway loops.
- Live traces are explicitly **not reproducible** — no golden snapshots for live.

## Testing

- **Unit — `LiveProvider` (fake `acompletion`):**
  - Offers only the active agent's tools + an own-`routes` `route` tool, in the
    nested Chat-Completions shape (not the flat catalog shape).
  - A `tool_calls` response maps to `ProviderDecision.calls` with raw argument
    strings preserved; a text response maps to `final`.
  - `reasoning_content` propagates to `ProviderDecision.reasoning`.
  - An `acompletion` that raises yields a graceful error `final`.
- **Unit — orchestrator resume:**
  - `start_agent` resumes mid-topology (e.g. starts in billing, no route call).
  - Unknown/stale `start_agent` falls back to `entry` (logged).
  - A resumed terminal agent ends the turn with the closed-conversation message;
    direct entry into a terminal agent still emits its escalation.
  - `ChatResult.active_agent` equals the in-charge agent at turn end.
- **Server / E2E (fake `acompletion`):**
  - `previous_response_id` round-trip resumes the routed agent (the core
    requirement) — turn 2 acts as billing without re-routing.
  - `config.scenario_mode` override and per-scenario `mode` select the provider
    (live → `LiveProvider`, default → `DeterministicProvider`).
  - The deterministic path and the full pre-existing suite stay green (no
    regressions).

## Open seams & notes

- **OpenAI Agents SDK alternate flavor.** If we later want SDK-native
  `transfer_to_<agent>` handoff traces as a distinct fixture flavor for Spectral,
  it would be a *second* live engine selected behind the mode switch — bypassing
  the orchestrator with a run-item→wire adapter. Out of scope here.
- **Live multi-call rounds.** A live model may emit parallel tool calls; the
  orchestrator already handles multiple `PlannedCall`s per round, including a
  `route` among them (it switches active agent after processing the round).
- **ghost-backend.** Untouched. Phase 4's `Response.tools` harvest remains the
  last, author-gated, async step; until it lands a fixture's catalog is
  registered via the Target's `declared_tools`.
