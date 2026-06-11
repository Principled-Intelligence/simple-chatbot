# Scenario Fixture Engine — Phase 3 (Knob & Fixture Library) design

**Date:** 2026-06-04
**Status:** Approved (pending spec review)
**Author:** Niccolò Campolungo / Claude
**Repo:** `simple-chatbot` (this repo only — ghost-backend is untouched)
**Branch:** `feat/agentic-behavior` (continue on it — do **not** branch off or merge to main)
**Parent spec:** [2026-06-03-scenario-fixture-authoring-design.md](2026-06-03-scenario-fixture-authoring-design.md) (see the Phasing table; this is Phase 3)
**Phase 2 design:** [2026-06-03-scenario-fixture-engine-phase2-live-mode-design.md](2026-06-03-scenario-fixture-engine-phase2-live-mode-design.md)

## Goal

Broaden the deterministic fixture engine so authored scenarios cover the full
range of tool behaviors Spectral's supervisors judge. Phase 3 adds **one new
structural construct** (`Parallel`, for multi-call fan-out rounds), **three new
intent tags** on `Call` (`redundant`, `wrong_value`, `type_mismatch`), and a
**library of canonical fixtures** authored on top of these plus existing
constructs — with **golden-trace snapshots** locking the deterministic output of
the new canonical fixtures.

This is purely additive to the deterministic path. The Phase 2 `LiveProvider` /
mode-switch seams are stable and are **not** touched. Misbehavior knobs remain
**deterministic-mode-only** (the locked decision carried forward: live mode
ignores `script` and has no prompt nudges).

## What Phases 1–2 already built (the seams this plugs into)

- **`scenario.py`** — the in-memory model + builder API: `Scenario`, `Agent`,
  `ScenarioTool`, the `@tool` decorator, the step constructors (`Call`, `Route`,
  `Final`, `MalformedCall`, `UnknownToolCall`), the `Step` union, and scenario
  validation. `Call` already carries an `irrelevant: bool` tag (precedent for the
  new tags). `Scenario.mode` (Phase 2) is untouched here.
- **`scenario_provider.py`** — `PlannedCall`, `ProviderDecision`,
  `DeterministicProvider` (plays one script step per round, per-agent cursor), and
  the Phase 2 `LiveProvider`. The deterministic provider currently maps each step
  to a **single-call** `ProviderDecision`.
- **`scenario_orchestrator.py`** — the round loop. It **already** iterates
  `decision.calls` (plural), builds one assistant message with N `tool_calls`,
  executes each, appends each tool-result, and assigns deterministic per-call ids
  `call_<scenario>_<turn>_<seq>` (incrementing `seq` per call within a round). So
  **multi-call rounds already work end-to-end** — the only missing piece is a
  deterministic *script construct* that produces a multi-call decision.
- **`scenario_catalog.py`** — `build_responses_tools(scenario)` projects the union
  of all agents' tools + one `route` tool into the flat `Response.tools` shape.
  Unaffected by Phase 3 (new tags/constructs reference existing tools).
- **`scenario_registry.py`** — auto-discovers `simple_chatbot/fixtures/*.py`
  modules exposing a module-level `scenario: Scenario`, keyed by `scenario.id`.
  New fixtures are picked up automatically.
- **`responses.py`** — `build_output_items(result)` serializes `tool_messages`
  into the ordered Responses output array (reasoning → function_call(s) →
  function_call_output(s) → message). This is what golden-trace tests snapshot.
- **Existing fixtures** — `cs-routing` (dispatcher → billing → human terminal),
  `validity-probe` (unknown-tool, malformed-args, required-violation knobs).

## Locked decisions (from brainstorming)

| Topic | Decision | Rationale |
|---|---|---|
| **Scope shape** | **Engine + fixtures.** Add the minimal new constructs, then author the fixture library on top. | Matches the parent spec's Phase 3 ("expand the knob catalog **and** the fixtures"). |
| **Parallel calls** | **New `Parallel(steps=[...])` step** wrapping call-family steps; `DeterministicProvider` maps it to a multi-`PlannedCall` decision. | The only behavior not expressible today. The orchestrator already handles multi-call rounds, so this is a provider/model addition, **not** an orchestrator change. |
| **New misbehavior knobs** | **`redundant`, `wrong_value`, `type_mismatch` as boolean tags on `Call`** (default `False`), not new step types. | They emit well-formed-looking calls with **no new wire behavior** — identical to the existing `irrelevant` tag. The author writes the offending args; the tag records intent + drives test assertions. |
| **RAG-ignore mechanism** | **Retrieval-as-tool stub.** A `search`/`retrieve` tool whose stub returns chunks as its tool output, then `Final(ignore_retrieval=True)` contradicting them. | No new orchestrator concept; the retrieved content already lives in a tool-output item inside the per-turn window Spectral reads. `ignore_retrieval` stays a test-assertion tag. |
| **type-mismatch placement** | **Extend the existing `validity_probe` fixture** with a type-mismatch step rather than a separate fixture. | `validity_probe` is the validity-knob showcase; keeping all validity classes (unknown / malformed / required / type) in one fixture is clearest. |
| **Golden traces** | **Snapshot the new canonical fixtures** (`parallel`, `rag-ignore`) and backfill `cs-routing`. | Honors the parent spec's reproducibility promise; catches accidental wire-shape regressions. Modest test code, no engine cost. |
| **Live mode** | **Untouched; knobs are deterministic-only.** | Carried-forward locked decision. A misbehavior fixture run live simply produces well-formed model-driven calls. |

## Non-goals

- Any change to the orchestrator round loop, the `LiveProvider`, the mode switch,
  active-agent persistence, or the catalog projection.
- Live-mode knob support / prompt nudges (deterministic-only, carried forward).
- Live-mode golden traces (live is best-effort, non-reproducible).
- Phase 4 (ghost-backend `Response.tools` harvest) — author-gated; **do not start**.
- A YAML authoring adapter (still deferred).
- New supervisor dimensions beyond the three knobs chosen
  (no separate `output-value` / `trajectory` knobs this phase).

## Architecture

No new top-level component. Phase 3 adds one step type + three tags to the model,
one mapping branch to the deterministic provider, and a set of fixture modules +
golden tests. Data flow is unchanged from Phase 1.

```
fixtures/parallel.py ─┐
fixtures/multi_round_routing.py ─┐
fixtures/rag_ignore.py ──────────┤  import ──► Scenario (model, with Parallel/tags)
fixtures/escalation.py ──────────┤
fixtures/relevance_probe.py ─────┘
                                      │
 /v1/responses (model=<id>) ─► registry ─► ScenarioOrchestrator
                                              │ (UNCHANGED round loop;
                                              │  already multi-call capable)
                                              ▼ DeterministicProvider
                                              │  + Parallel → multi PlannedCall
                                              ▼
                                       ChatResult ─► build_output_items ─► output[]
                                                          ▲
                                              golden-trace snapshot asserts here
```

### Component changes

- **`scenario.py` — `Parallel` step + three `Call` tags.**
  - Add `@dataclass Parallel` with `steps: list[Call | MalformedCall |
    UnknownToolCall]`. Add it to the `Step` union.
  - Add to the `Scenario.validate()` pass: a `Parallel` may contain only
    call-family steps — `Route`, `Final`, or a nested `Parallel` inside it raise a
    `ValueError` (a fan-out batch cannot hand off, finish, or nest). Validation
    runs in `Scenario.__post_init__`, so a malformed fixture fails at import.
  - Add three boolean fields to `Call` (default `False`): `redundant`,
    `wrong_value`, `type_mismatch`. Pure metadata — they do **not** change how the
    call is serialized; they document authorial intent and are asserted by tests.

- **`scenario_provider.py` — deterministic multi-call mapping.**
  Extract the existing per-step → `PlannedCall` logic (the `Call` / `MalformedCall`
  / `UnknownToolCall` branches) into a small helper `_planned_call(step)`. Add a
  `Parallel` branch in `DeterministicProvider.decide` that returns
  `ProviderDecision(calls=[_planned_call(s) for s in step.steps])`. Single-call
  steps keep returning a one-element `calls` list (behavior unchanged). The
  `LiveProvider` is untouched (it ignores `script`).

- **`fixtures/` — new fixture modules.** Each exposes a module-level
  `scenario: Scenario` and is auto-discovered by the registry. See the table below.

- **`tests/` — golden traces + per-fixture behavioral tests + unit tests.**

## New step: `Parallel`

```python
@tool
def get_weather(city: str) -> dict: ...
@tool
def get_events(city: str) -> dict: ...

Agent(
    "concierge",
    tools=[get_weather, get_events],
    script=[
        Parallel([
            Call(get_weather, {"city": "Lisbon"}),
            Call(get_events, {"city": "Lisbon"}),
        ]),
        Final("It'll be sunny; there's a jazz festival this weekend."),
    ],
)
```

The `Parallel` step yields **one** assistant message with two `tool_calls`
(`call_parallel_1_1`, `call_parallel_1_2`), then two `function_call_output`
items, in order — one Spectral turn whose window holds the whole fan-out. A
`Parallel` may include misbehavior steps (e.g. one valid `Call` + one
`MalformedCall`) to exercise per-call validity within a fan-out.

## New `Call` tags

```python
Call(lookup_invoice, {"invoice_id": "INV-1"}),
Call(lookup_invoice, {"invoice_id": "INV-1"}, redundant=True),   # call-necessity
Call(issue_refund, {"invoice_id": "INV-999", "amount": 9999.0},  # relevance edge
     wrong_value=True),                                          # schema-valid, wrong value
Call(issue_refund, {"invoice_id": "INV-1", "amount": "forty-two"},  # validity (type)
     type_mismatch=True),                                        # string where number required
```

All three emit normal, well-formed `function_call` items (same as a plain
`Call`); the tags are for authorial intent and test assertions. For
`type_mismatch`, the author supplies a schema-type-violating value — that's what a
validity supervisor scores; if the stub raises on the bad type, the tool output is
a graceful `Tool error:` (the same path the existing required-violation knob
already takes through `_execute`).

## Fixture library

| Fixture module / `id` | Topology & script | Dimension(s) exercised | New capability |
|---|---|---|---|
| `parallel.py` / `parallel` | One agent fans out 2 read calls in a single `Parallel` round, then `Final` synthesizing. | trajectory / fan-out | `Parallel` step |
| `multi_round_routing.py` / `multi-round-routing` | Dispatcher → tier-1 → tier-2 (two `Route` hops) all within one user turn, ending with the deepest agent's `Final`. | routing relevance across hops, trajectory | existing constructs |
| `rag_ignore.py` / `rag-ignore` | `search_docs` tool stub returns authoritative chunks; agent then `Final(..., ignore_retrieval=True)` contradicting them. | faithfulness / utilization | retrieval-as-tool stub |
| `escalation.py` / `escalation` | Dispatcher routes **straight to** a `terminal` human agent which emits its `escalation_message` as the final. | terminal-as-final, route-to-human relevance | existing constructs |
| `relevance_probe.py` / `relevance-probe` | One agent fires an `irrelevant` call, a `wrong_value` call, and a `redundant` call, then `Final`. | relevance + call-necessity | `irrelevant` / `wrong_value` / `redundant` tags |
| `validity_probe.py` (extend) / `validity-probe` | Add a `type_mismatch` step to the existing unknown/malformed/required sequence. | validity (all classes incl. type) | `type_mismatch` tag |

## Testing

- **Unit — `scenario.py`:** `Parallel` constructs; `Scenario.validate()` rejects
  `Route` / `Final` / nested `Parallel` inside a `Parallel` (raises `ValueError` at
  import-time via `__post_init__`); the three new `Call` tags default `False` and
  accept `True`.
- **Unit — `DeterministicProvider`:** a `Parallel` step yields a `ProviderDecision`
  with multiple `PlannedCall`s in order (including a mixed valid + malformed
  batch); single-call steps still yield a one-element `calls` list (no regression).
- **Per-fixture behavioral tests** (mirroring `tests/test_scenario_fixtures.py`):
  each new fixture runs through `ScenarioOrchestrator` + `DeterministicProvider`
  and asserts call names/order, routing hops, terminal final, and the tag-bearing
  calls. For `rag-ignore`, assert the `search_docs` tool output is present in the
  trace and the final contradicts it.
- **Golden traces** (`tests/test_golden_traces.py`): snapshot the full ordered
  output array from `build_output_items(result)` for `parallel` and `rag-ignore`,
  and backfill `cs-routing`. Assert item types, `name`s, deterministic `call_id`s
  (`call_<scenario>_<turn>_<seq>`), and tool-output contents. Envelope ids
  (`fc_`/`fco_`/`msg_`/`rs_`) that Spectral does not key on are normalized out of
  the snapshot (assert on `call_id` + ordering + content, per the parent spec's
  reproducibility definition).
- **Regression gate:** full suite green (`python -m pytest -q`) — the current 228
  tests plus all Phase 3 additions, no failures.

## Open seams & notes

- **Live-mode knobs.** Still deterministic-only. A misbehavior fixture run live
  produces well-formed, model-driven calls (documented, carried forward).
- **`Parallel` in live mode.** A live model may already emit parallel tool calls;
  the orchestrator handles multiple `PlannedCall`s per round regardless of
  provider. `Parallel` only governs the *deterministic* script.
- **Other dimensions.** `output-value` and `trajectory`-specific knobs are not
  added this phase (trajectory is already governed by script ordering; output
  value by stub bodies). They can be added later as further tags/fixtures with no
  engine change.
- **ghost-backend.** Untouched. Phase 4's `Response.tools` harvest remains the
  last, author-gated, async step.
