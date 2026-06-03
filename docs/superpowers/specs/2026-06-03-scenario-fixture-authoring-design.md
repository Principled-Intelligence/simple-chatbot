# Scenario fixture-authoring engine — design

**Date:** 2026-06-03
**Status:** Approved (pending spec review)
**Author:** Niccolò Campolungo / Claude
**Repos:** `simple-chatbot` (primary), `ghost-backend` (one small cross-repo harvest change)

## Goal

Turn simple-chatbot into a fixture-authoring tool for testing Spectral's tool
supervisors. A fixture declares one or more **test agents** with varied tool
interactions — routing/handoff, parallel calls, RAG, common tools, and
deliberate misbehavior — and the server emits a faithful Responses API trace
that Spectral ingests and evaluates. Fixtures are authored **in code** (a small
typed builder API), version-controlled, and **deterministic** so the same
fixture yields the same trace on every run.

A secondary goal: generic agentic cases (especially the live, model-driven
path) should surface gaps that drive the Spectral platform's evolution.

## Why simple-chatbot is the vehicle (decided: adjust, don't rebuild)

The hard, already-solved part is being a Responses-API **target server**:
faithful ordered output items (`function_call` → `function_call_output` →
`message`, plus `reasoning`), an in-memory response store,
`previous_response_id` multi-turn chaining, and deterministic scripted fixtures.
That is the canonical `Conversation` shape Spectral ingests. No agent framework
helps here: every framework (OpenAI Agents SDK, LangGraph, Pydantic AI, …) is a
**client** of a model/Responses endpoint — we need to **be** the endpoint.
Rebuilding the wire layer, or reaching for the OpenAI Agents SDK in its client
role, would just mean re-serializing into `/v1/responses` anyway.

## What's missing today (the brainstorm scope)

1. The real loop ([agent.py:163-273](../../../simple_chatbot/agent.py)) is a flat,
   single-agent ReAct loop over a **fixed** tool set: no routing, no handoff, no
   sub-agents. "Next step" only ever means "next tool," never "next agent."
   Routing is a first-class dimension the supervisors want to judge.
2. Scripted mode ([scripted_llm.py](../../../simple_chatbot/scripted_llm.py)) is
   deterministic keyword/marker heuristics, conceptually separate from a
   model-driven loop. We keep a deterministic path as the canonical fixture
   source but generalize it to multi-agent scenarios.
3. No structured scenario authoring yet (tools are CLI-flag + hardcoded in
   [tools.py](../../../simple_chatbot/tools.py)).
4. The tool catalog never reaches Spectral: [responses.py:188](../../../simple_chatbot/responses.py)
   builds `Response(tools=[])` and emits no catalog item, so a standalone trace
   has an empty `tool_catalog` and the validity dimension silently no-ops.

## Who consumes the traces (design must serve these)

Spectral's tool supervisors read the canonical `Conversation`: ordered items
(incl. `ToolCallItem` keyed by `call_id`), `tool_catalog` (`canonical_tool_id` +
JSON schemas), `raw_responses`, with **per-turn windowing**. Confirmed against
the consumer code:

- **Turn boundaries are `UserMessageItem` indices.** Turn *T*'s window is every
  item from user message *T* up to (not including) user message *T+1*
  ([tool_relevance_supervisor_agent.py:42-72](../../../../ghost-backend/packages/core/src/spectral/core/shogun/supervisors/tool_relevance_supervisor_agent.py)).
  So everything between two user messages is **one turn**, regardless of how many
  agents or rounds happen inside it.
- **Catalog population** has two channels Spectral reads
  ([responses_ai_service.py:174-213](../../../../ghost-backend/packages/core/src/spectral/core/shogun/ai_services/responses_ai_service.py)):
  (a) the Target's `declared_tools` (sent in the request `tools` field, seeded
  into the catalog); (b) tools harvested from an `mcp_list_tools` output item.
  Spectral does **not** currently read the spec's `Response.tools` echo.
- **Call → catalog resolution** is by `canonical_tool_id` when present, else by
  tool **name** fallback (`resolve_tool_def`).

Supervisor dimensions (current + planned, from the SP3/SP4 driving plan): tool
**relevance** (1-5 + rationale); **validity** (deterministic gate — unknown tool
→ 1, malformed args → 2, jsonschema required-class violation → 2; skips when
catalog/schema absent); later **call necessity**, **trajectory**, **output
value**, **utilization/faithfulness**.

Driving plan: `ghost-backend/docs/superpowers/specs/2026-05-29-unified-classification-and-tool-supervisors-driving-plan.md`
(SP3 = tool supervisors, SP4 = per-call dimensions).

## Locked decisions

| Topic | Decision | Rationale |
|---|---|---|
| **Authoring engine** | **Both, one model, mode switch.** A single in-memory `Scenario` model drives a deterministic replay engine (canonical) and an optional live model engine. | Reproducible fixtures for CI + a live path that surfaces platform gaps. |
| **Authoring surface** | **Code-first builder API**, not YAML. The in-memory `Scenario` model is the orchestrator's contract; a YAML loader is a deferred optional adapter over the same model. | Reuses the existing `ToolDef` code idiom; eliminates embedding JSON Schema in YAML and a `script`-step DSL; type-checked, refactorable. |
| **Tool schemas** | **Typed function → auto-generated JSON Schema; the function body is the deterministic stub.** | One definition, no duplication (what framework `@function_tool` decorators do, without the framework). Removes hand-written schema dicts. |
| **Turn model** | **One turn per user message.** A whole agent-A→B→… topology collapses into ONE `/v1/responses` output array = one Spectral turn. Routing is an inline `route` tool call. | Reuses the existing ReAct flattening; per-turn windowing sees the routing decision *and* the routed sub-agent's calls together. No new wire mechanism. |
| **Catalog channel** | **Populate the spec's `Response.tools` echo from the scenario**, and add a small `Response.tools` harvest to Spectral. | Most standards-faithful for an autonomous agent that owns its tools; self-describing traces; no per-fixture MCP framing; no manual Target registration. |
| **Scenario selection** | **`model` field = scenario id.** One server loads a fixtures package; `/v1/responses` resolves the scenario from `body.model`; `/v1/models` lists fixture ids. | Maps cleanly onto how a Spectral Target already configures a model. |
| **Framework use** | **Own deterministic engine; pluggable `Provider` seam for a framework-backed live provider (deferred).** | The misbehavior knobs deliberately emit *invalid* traces (malformed args, unknown tool, required-violation) — exactly what frameworks validate away. We own the server/serialization always. |

## Non-goals

- A YAML schema in v1 (deferred adapter over the same model).
- Each routed sub-agent as its own separately-windowed Spectral turn (would
  require a synthetic user message or chained extra request — unfaithful / extra
  machinery). v1 collapses to one turn per user message.
- Persisting the active agent across user turns. v1 **re-enters the entry
  agent** on each new user message (noted open seam).
- Guaranteeing misbehavior knobs in live mode (best-effort prompt nudges only;
  deterministic mode is the guaranteed path).
- Replacing the existing single-agent RAG path; it stays for the real chatbot.
- Building the live, framework-backed provider in Phase 1.

## Architecture

A new **scenario layer** sits beside the existing single-agent path. Both
produce a `ChatResult`, so the wire layer ([responses.py](../../../simple_chatbot/responses.py),
[server.py](../../../simple_chatbot/server.py)) is reused nearly unchanged.

```
fixtures/*.py ──import──► Scenario (in-memory model, the contract)
                              │
 /v1/responses (model=<id>) ─► ScenarioRegistry ─► ScenarioOrchestrator
                                                       │ active-agent state machine
                                                       │ per-round scoped toolset
                                                       │ handoff (route) handling
                                                       │ Provider: deterministic | live
                                                       ▼
                                       ChatResult (one flat tool_messages trace)
                                                       │
                       build_response → output[] + Response.tools (catalog)
```

### Components (each small, single-purpose)

- **`scenario.py`** — the in-memory model and builder API: `Scenario`, `Agent`,
  the `@tool` decorator, and step constructors (`Call`, `Route`, `Final`,
  `MalformedCall`, `UnknownToolCall`). Plus validation (every `routes`/`tools`
  reference resolves; `entry` exists; tool names unique).
- **`tool` decorator** — wraps a typed Python function into a `ToolDef`-shaped
  object: JSON Schema derived from the signature (via Pydantic), description from
  the docstring, the function body used as the deterministic stub (executed via
  the same arg-parsing path as [tools.py](../../../simple_chatbot/tools.py)).
- **`orchestrator.py`** — `ScenarioOrchestrator`: active-agent state machine that
  runs the round loop offering only the active agent's tools (+ its generated
  `route` tool), executes stubs, switches active agent on a `route` call, and
  appends every round to one `tool_messages` list. Emits a `ChatResult`. Reuses
  agent.py's `_sanitize_assistant_dump` and forced-final fallback.
- **Providers** — `DeterministicProvider` (plays the active agent's `script`;
  scenario-aware replacement for the LLM) and a deferred `LiveProvider` (real
  `litellm` per round with the active agent's `system_prompt` + scoped tools;
  later optionally framework-backed). The orchestrator calls a provider each
  round; a mode flag picks which.
- **`catalog.py`** — projects a scenario's tools (+ each routing agent's
  generated `route` tool) into the flat `Response.tools` shape.
- **Server wiring** — `ScenarioRegistry` imports the fixtures package →
  `{id: Scenario}`; `/v1/responses` routes to the orchestrator when `body.model`
  is a known fixture id, else falls back to the existing agent; `/v1/models`
  lists fixture ids.
- **Spectral-side (cross-repo)** — harvest `Response.tools` into `tool_catalog`
  in responses_ai_service / responses_normalization.

## Authoring surface (final)

```python
# fixtures/cs_routing.py
from simple_chatbot.scenario import Scenario, Agent, tool, Route, Call, Final

@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice by id."""          # → tool description
    return {"invoice_id": invoice_id, "amount_due": "42.00", "status": "open"}
# parameters JSON Schema ← signature; body = deterministic stub. No hand-written dict.

@tool
def issue_refund(invoice_id: str, amount: float) -> dict:
    """Issue a refund against an invoice."""
    return {"refunded": True, "invoice_id": invoice_id, "amount": amount}

scenario = Scenario(
    id="cs-routing",
    description="CS dispatcher routing to billing / human escalation.",
    entry="dispatcher",
    agents=[
        Agent("dispatcher",
              system_prompt="Route requests to a specialist; don't answer yourself.",
              routes=["billing", "human"],            # generates this agent's `route` tool
              script=[Route("billing")]),             # deterministic-mode trajectory
        Agent("billing",
              system_prompt="You handle billing and refunds.",
              tools=[lookup_invoice, issue_refund],
              script=[Call(lookup_invoice, invoice_id="INV-1"),
                      Call(issue_refund, invoice_id="INV-1", amount=42.0),
                      Final("I've refunded $42.00 against invoice INV-1.")]),
        Agent("human", terminal=True,
              escalation_message="Escalating to a human agent."),
    ],
)
```

- Tools are referenced as **symbols** (`tools=[lookup_invoice]`), not strings —
  no silent typos.
- `Route(target)` is sugar for a `route` call with `{"agent": target}`.
- `script` is the deterministic trajectory; **live mode ignores it** and lets the
  model drive from `system_prompt` + scoped tools.

## Trace serialization & turn model (the main hazard)

Each round the active provider returns either tool calls or a `Final`. Tool calls
execute their stubs; the assistant(tool_calls) message and the tool-result
messages append to one growing `tool_messages` list. A `route` call's executor
returns `{"routed": true, "agent": <target>}` and signals the orchestrator to
switch the active agent — the **next** round offers the target's tools.
Everything accrues into ONE `tool_messages` list → ONE output array via the
existing [build_output_items](../../../simple_chatbot/responses.py) (reasoning →
function_call(s) → function_call_output(s) → … → final message). One user message
in ⇒ Spectral sees ONE turn whose window holds the `route` call **and** the
routed agent's calls together. Terminal agents emit their `escalation_message` as
the final message and end the turn.

Multi-turn uses the existing `previous_response_id` chaining
([server.py:266-302](../../../simple_chatbot/server.py)); each new user message
**re-enters the entry agent** (v1 simplification; persisting the active agent
across turns is an open seam).

**Reproducibility** is defined as **stable `call_id`s + stable item ordering +
stable tool outputs**, not byte-identical envelopes. The deterministic engine
generates counter-based ids (`call_<scenario>_<turn>_<seq>`) — `call_id` is what
Spectral keys `ToolCallItem` on. The `fc_`/`fco_`/`msg_`/`rs_` envelope ids
(which Spectral does not key on) may stay random. A per-scenario `max_rounds`
guard reuses agent.py's forced-final fallback to prevent runaway loops.

## Tool catalog publication

`catalog.py` projects every scenario tool **plus each routing agent's generated
`route` tool** into flat `Response.tools` entries
`{type: "function", name, parameters, description}`; `build_response` populates
the currently-empty `tools` field. The catalog is the **union** of all agents'
tools for the scenario, so validity/relevance can resolve **any** call in the
turn — per-agent scoping only governs what's *offered* each round (trajectory
faithfulness), not what's *published*.

The `route` tool is a first-class catalog entry (schema:
`{agent: {type: string, enum: [<routes>]}}`), so supervisors can judge the
**routing decision itself** (routing to billing for a shipping question = low
relevance).

**Cross-repo (Spectral):** add harvesting of `Response.tools` into
`conversation.tool_catalog` (`fingerprint_tool(name, params)` →
`canonical_tool_id(target_id, fp)`, `source="declared"`). Function calls resolve
by name fallback; `route` resolves by name `"route"`. This is the only
ghost-backend change and is additive (the existing `declared_tools` +
`mcp_list_tools` channels are untouched).

## Misbehavior knobs (step-level, deterministic-guaranteed)

| Knob | Construct | Dimension exercised |
|---|---|---|
| malformed args | `MalformedCall(tool)` | validity → 2 (invalid JSON args) |
| unknown tool | `UnknownToolCall("xyz")` | validity → 1 (name absent from populated catalog) |
| required-arg violation | `Call(tool)` omitting a required arg | validity → 2 (jsonschema required-class) |
| irrelevant call | `Call(tool, ..., irrelevant=True)` | relevance → low (e.g. refund tool for a shipping Q) |
| ignore retrieval | `Final("...", ignore_retrieval=True)` | utilization / faithfulness |

`irrelevant` / `ignore_retrieval` are tags for test assertions; the emitted trace
is just a normal (well-formed) call to a mismatched tool or a final that ignores
retrieved chunks. In **live mode** all knobs degrade to best-effort prompt nudges
(documented; not guaranteed).

## Modes

- **Deterministic** (default for fixtures): `DeterministicProvider` plays the
  active agent's `script`. No API key, CI-friendly, reproducible.
- **Live** (deferred to Phase 2): `LiveProvider` calls `litellm` per round with
  the active agent's `system_prompt` + scoped tools, ignoring `script`. The
  `Provider` seam allows a later framework-backed implementation — the **OpenAI
  Agents SDK** is the strongest candidate (Responses-native; first-class
  handoffs serialized as `transfer_to_<agent>` tool calls; `LitellmModel` runs
  against our backend; auto-schema tools) — adapting its run items into our wire
  payload. Decided when we reach Phase 2.

Mode is chosen by a per-scenario default plus a server/env override.

## Testing

- **Unit:** scenario validation (refs resolve, entry exists, names unique,
  schemas well-formed); `@tool` schema generation from signatures; orchestrator
  state machine (handoff switches active agent; scoped tools per round);
  deterministic provider plays scripts; each misbehavior knob yields its
  expected shape.
- **Wire:** `Response.tools` flat-schema correctness; output ordering; stable
  `call_id`s; `route` call + output present and correctly ordered.
- **E2E:** drive `/v1/responses` by `model`; multi-turn chaining; mimic Spectral
  normalization to assert `tool_catalog` is populated **and** the per-turn window
  contains routing + sub-agent calls.
- **Cross-repo (Spectral):** `Response.tools` harvest test — catalog populated
  from a trace with no `mcp_list_tools` and no declared Target tools.
- **Golden traces:** snapshot deterministic output for canonical fixtures
  (`cs-routing`, `parallel`, `rag-ignore`) to lock reproducibility.

## Phasing (within this one spec)

1. **Phase 1 — canonical fixture path end-to-end:** `scenario.py` model +
   builder API + `@tool` + `orchestrator.py` + `DeterministicProvider` +
   `catalog.py` + server selection + the Spectral `Response.tools` harvest + 1–2
   canonical fixtures + tests.
2. **Phase 2 — live mode:** `LiveProvider` (litellm per round, scoped tools);
   decide whether to back it with the OpenAI Agents SDK.
3. **Phase 3 — knob/fixture library:** parallel calls, multi-round routing,
   RAG-ignore, escalation, and the remaining misbehavior fixtures.

## Open seams & notes

- **Active agent across turns.** v1 re-enters the entry agent each user message.
  Persisting it (so a billing follow-up stays in billing) can use the response
  store's `session_messages` entry — deferred.
- **YAML adapter.** The in-memory `Scenario` model is the contract; a YAML loader
  producing the same model can be added later with zero orchestrator change.
- **`Response.tools` vs Spectral harvest ordering.** The emit side (this repo)
  and the harvest side (ghost-backend) ship together in Phase 1; until the
  harvest lands, a fixture's catalog can still be registered via the Target's
  `declared_tools` as a fallback.
- **Existing scripted mode.** [scripted_llm.py](../../../simple_chatbot/scripted_llm.py)
  and its `EXHAUSTIVE_TOOL_USE` rotation remain for the single-agent scripted
  path; the scenario engine is the multi-agent successor, not a replacement of
  that mode.
