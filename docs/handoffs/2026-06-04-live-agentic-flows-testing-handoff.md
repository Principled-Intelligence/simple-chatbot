# Handoff — Live agentic-flows testing (design/brainstorm phase)

**Repo:** simple-chatbot (branch `feat/agentic-behavior`) · likely consumes from ghost-backend later (Phase 4 / Spectral, not started)
**Date:** 2026-06-04 · **Status:** Scenario fixture engine Phase 3 complete (committed); two server flags added (`default_fixture`, `scenario_mode`) but **uncommitted**; full suite green (255 tests). No design doc yet for the new effort — writing/shaping it is this phase's first job.

This doc is the entry point for a **new design phase**: the user wants to build *several agentic flows that test real interactions with real (live) agents*, defined by us as a **strict, expandable set of actions we can test against**. That goal has an unresolved crux (live agents are non-deterministic; the user wants strict testable actions) which must be shaped before any plan is written. This is NOT yet a campaign — whether to create one is an open decision below.

## Starter message (the user will paste this into the fresh conversation)
```
Read docs/handoffs/2026-06-04-live-agentic-flows-testing-handoff.md and start the design phase for live agentic-flows testing.
Objective: shape a testable "strict, expandable action set" for real/live agent interactions, then decide if it's a campaign.
Use superpowers:brainstorming for this phase.
First, read the two canonical docs listed in the handoff, then confirm/refute the proposed deterministic-golden-trajectory + live-conformance framing before exploring alternatives.
```

## Canonical docs to read first (in order)
1. `docs/superpowers/specs/2026-06-04-scenario-fixture-engine-phase3-knob-fixture-library-design.md` — the Phase 3 design; see "Open seams & notes" (live-mode is deterministic-only/best-effort, non-reproducible) and the fixture-library table.
2. `docs/superpowers/plans/2026-06-04-scenario-fixture-engine-phase3-knob-fixture-library.md` — what was actually built in Phase 3 (Parallel step, Call tags, fixtures, golden traces).
3. `simple_chatbot/scenario_provider.py` — both providers: `DeterministicProvider` (replays the authored script) and `LiveProvider` (model-driven, **ignores the script**).
4. `simple_chatbot/server.py` — `/v1/responses` fixture selection + mode resolution (`config.scenario_mode or scenario.mode`) + active-agent resume across turns.
5. `simple_chatbot/agent.py` — the built-in single-agent RAG bot (the no-fixture path).

## State / the change

**Phase 3 (committed on `feat/agentic-behavior`, commits `079ae7d`..`ca31889`):** deterministic scenario fixture engine. A `Scenario` = agents, each with a scoped toolset, optional `routes`, and a deterministic `script` of steps (`Call`, `MalformedCall`, `UnknownToolCall`, `Route`, `Final`, `Parallel`). Misbehavior is recorded as inert boolean tags on `Call` (`irrelevant`, `redundant`, `wrong_value`, `type_mismatch`) and `Final` (`ignore_retrieval`) — these only affect deterministic runs.

**7 fixtures exist** (`simple_chatbot/fixtures/*.py`), all `mode="deterministic"`:
- `cs-routing`: dispatcher → billing (`lookup_invoice`, `issue_refund`) → human (terminal)
- `multi-round-routing`: front_desk → tier1 → tier2 (`check_status`)
- `escalation`: dispatcher → human (terminal)
- `parallel`: concierge (`get_weather`, `get_events`) — fan-out in one round
- `rag-ignore`: support (`search_docs`) then contradicts retrieval
- `relevance-probe`: agent (`lookup_invoice`, `get_weather`) — irrelevant/wrong-value/redundant calls
- `validity-probe`: prober (`lookup_invoice`, `get_order`, `set_credit_limit`) — unknown/malformed/required/type knobs

**Two distinct "live" paths exist today** (this is the part the user wanted clarified):
- **No-fixture path → built-in RAG agent** (`agent.py`): single chat model via litellm, ONE default tool (`search` over ChromaDB), agentic loop up to `max_tool_rounds` (default 5), optional ScopeGuard. No routing/multi-agent. This is "live" in that it always calls a real LLM.
- **Fixture in live mode → `LiveProvider`**: model-driven, one LLM call per round with the active agent's `system_prompt` + scoped tools + a `route` tool (enum = that agent's own routes). **Ignores the authored script**, so the misbehavior knobs do nothing live. Multi-agent routing still works (model decides when to `route`/finish). Best-effort: on LLM error returns `"(live provider error: …)"`.
- **0 fixtures declare `mode="live"`** — live for a fixture is only reached via the new server-level `scenario_mode="live"` override.

**Two server flags added this session, NOT yet committed** (working tree: `M cli.py config.py server.py`, `?? tests/test_default_fixture.py`):
- `default_fixture` (config + `--default-fixture` + `SIMPLE_CHATBOT_DEFAULT_FIXTURE`): on `/v1/responses`, when the request names no known fixture, run this fixture instead of the RAG agent. Explicit fixture in `model` still wins. Fail-fast at startup if the id is unknown. Resume across turns preserved (stores the resolved fixture id). Tests: `tests/test_default_fixture.py` (9 tests).
- `scenario_mode` (config field already existed; added `--scenario-mode` + `SIMPLE_CHATBOT_SCENARIO_MODE`): server-level override of fixture run mode. Fail-fast on invalid value (pydantic Literal). Behavior already covered by `tests/test_scenario_endpoint.py::test_server_mode_override_selects_live_provider`.

No-key testing: `SIMPLE_CHATBOT_SCRIPTED_LLM=1` swaps in a scripted indexer + a litellm-compatible scripted `acompletion` (so even the live path runs without a real key), and leaving `--api-key` unset disables auth.

## What the next phase IS (objective + gate)
**Objective:** Brainstorm and define a *testable contract* for live agentic flows — a strict, expandable set of actions/trajectories we can assert real agents against — and decide whether the work is a campaign or a single plan.
**Done when:** a written design/brainstorm output exists (likely `docs/superpowers/specs/…` or a brainstorm note) that resolves the open questions below and decomposes the work into named plans (or one plan). No code is required in this phase.

## Decisions locked (actually agreed this session)
- `--scenario-mode` flag should exist (user approved) — implemented, pending commit.
- `default_fixture` mechanism for model-less clients (e.g. Spectral) — implemented, pending commit.
- Misbehavior knobs stay deterministic-only (carried from Phase 3 spec; live ignores the script).

## Decisions the next phase MUST make (open questions)
- **The crux:** how do you "test against a strict set of actions" when live agents are non-deterministic? Proposed (UNCONFIRMED — explore in brainstorm) framing: treat the existing deterministic fixture as the *allowed/expected trajectory* (the golden), run the *same scenario* in live mode, and assert the live agent's trajectory **conforms** (which tools called, routing decisions, terminal reached) rather than matching text byte-for-byte. "Expandable" = add a fixture; "testable" = conformance, not equality. Validate or replace this framing first.
- Where does the "strict set of actions" live — extend the scenario engine's vocabulary, or a new assertion/conformance layer on top of existing fixtures?
- What exactly is asserted (tool-call set / order / routing / reaching `Final` or terminal / arg shapes) and how strict (subset? exact sequence? allowed alternatives?).
- Which model/endpoint backs "real agents" for these tests, and how to keep them runnable offline (scripted LLM gives no-key runs but isn't a "real" agent).
- **Is it a campaign?** Likely yes (multi-plan, expandable, probably cross-repo with ghost-backend Phase 4), but unconfirmed. The user floated `/campaign`; this handoff defers that to the brainstorm's decomposition. If brainstorming yields several plans, invoke `hyperpowers:campaign` then.

## Coupling / gotchas
- `LiveProvider` deliberately discards the authored script — any "live conformance" idea must get its expected actions from the fixture *definition*, not from a live replay.
- Active-agent resume only triggers when the prior turn used the same resolved model/fixture (`server.py`). Relevant if live flows are multi-turn.
- ghost-backend / Spectral Phase 4 (`Response.tools` harvest) was "author-gated"; the user said "the backend work has been done" in conversation but that is UNCONFIRMED in this repo — verify on the ghost-backend side before assuming the consumer exists.

## Process & runner
- **Next phase skill:** `superpowers:brainstorming` (creative/design work — shape before planning). Then `superpowers:writing-plans` per plan, and `hyperpowers:campaign` if it decomposes into several.
- **Model:** Sonnet fine for brainstorming/planning; the deterministic engine work was done with subagents on Sonnet.
- **Conventions to honor:** TDD (unittest + `asyncio.run`), exact-verbatim plan code when executing plans, deterministic-only knobs. Full gate: `python -m pytest -q`.

## Suggested first move
Decide whether to commit the two pending flags first (recommended — they're green and self-contained: `feat(server): optional default fixture for model-less clients` and `feat(server): --scenario-mode CLI/env override`). Then read the two canonical docs and pressure-test the deterministic-golden-trajectory + live-conformance framing before opening the brainstorm wider.

## Pointer back
Builds directly on Phase 3 (the deterministic fixture engine + golden traces) and the two uncommitted server flags. Prior handoff context for the ghost-backend consumer side was drafted in conversation (the `Response.tools` flat-catalog contract: `{type, name, parameters, description}` + a generated `route` tool; deterministic `call_<scenario>_<turn>_<seq>` ids) but was not saved as a file in this repo.
