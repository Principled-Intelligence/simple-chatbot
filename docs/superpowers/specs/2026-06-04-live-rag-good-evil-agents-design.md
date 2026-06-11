# Live RAG good/evil agents — design

**Date:** 2026-06-04
**Status:** Approved (pending spec review)
**Author:** Niccolò Campolungo / Claude
**Repo:** `simple-chatbot` (this repo only)
**Branch:** `feat/agentic-behavior` (continue on it — do **not** branch off or merge to main)
**Handoff:** [../../handoffs/2026-06-04-live-agentic-flows-testing-handoff.md](../../handoffs/2026-06-04-live-agentic-flows-testing-handoff.md)

## Goal

Provide a way to run the **real, production RAG chatbot** (real ChromaDB retrieval,
real LLM) and an **evil counterpart** that misbehaves *some of the time* at any
point of the pipeline, producing production-shaped traces plus a ground-truth
label of *what was sabotaged where*. The purpose is to **test supervisors** (e.g.
Spectral's judges): a supervisor should flag the evil runs and pass the good ones,
and the injection labels are the ground truth to check that against.

This is the first use case. The misbehavior mechanism is built to be **expandable**:
each future agent (multi-agent routing, multi-tool validity, …) is a separate,
additive effort that reuses the same seeded policy core. It is therefore a single
plan, **not** a campaign.

## Crux & framing (resolved during brainstorming)

The handoff proposed treating the deterministic fixture as a "golden trajectory"
and asserting live runs **conform** to it.

- **Confirmed:** conformance (a predicate over the trajectory), not byte-equality,
  is the only coherent target for a non-deterministic agent.
- **Refuted:** the deterministic script is *not* a uniform "expected trajectory."
  Roughly half the existing fixtures (`relevance-probe`, `validity-probe`,
  `rag-ignore`) are **deliberate-misbehavior** scripts authored to feed supervisors;
  a well-behaved live agent conforming to them would be *failing*.
- **Reframed (and then narrowed by the user):** the actual objective is not a
  conformance checker at all. It is a **good agent + evil counterpart** pair where
  the evil twin misbehaves probabilistically, drawing from a **closed, expandable
  vocabulary of misbehavior modes** (the same knobs the deterministic engine already
  names). "Strict" = closed mode set; "testable" = seeded + labeled; "expandable" =
  add a mode / add an agent. The deterministic golden traces remain a separate,
  unchanged reproducibility lock.

## Locked decisions

| Topic | Decision | Rationale |
|---|---|---|
| **What we build** | A good/evil pair for the **real RAG bot**, evil = seeded probabilistic misbehavior injected across the pipeline. | Matches the user's restated goal: "run good agents and their evil counterparts that misbehave some of the time." |
| **Realization of "some of the time"** | **Hybrid:** forced + seeded injection as the testable backbone; prompt-steering for semantic modes as the realism arm. | Forced injection gives controlled rate + ground-truth labels; steering gives realistic agent-fault behavior. |
| **Substrate** | **Instrument the real `agent.py` pipeline**, by **composition over its existing `acompletion` + `tools` seams** — `agent.py` itself is **untouched**. | Highest production fidelity (real retrieval + real LLM); `Agent` already accepts both deps, so the evil twin is pure composition with zero production-code change. The scenario engine is reserved for later multi-agent use cases. |
| **Scope** | **RAG only this plan.** New agents (+ their evil injections) are separate, additive efforts. | YAGNI; prove the mechanism on one flagship use case. |
| **Campaign?** | **No — single plan.** Expandability is an architectural property (drop in a mode/agent), not a multi-plan graph. | No cross-plan coordination needed yet. |
| **Label strength** | Prefer **strong-label** environment-fault modes (`poison/drop_retrieval`, structural call injections) as the supervisor-fuel backbone; steered semantic modes (`ignore_retrieval`, `wrong_value`) are realistic but **weakly labeled**. | Strong labels are exact ground truth (the trace contains the sabotage); steered intent may not actually fool the model. |

## Non-goals

- The supervisor / detector itself (Spectral side). We produce labeled traces; we do
  not judge them.
- Multi-agent and other use cases (routing-relevance, tool-arg validity) — separate,
  later efforts.
- `EvilLiveProvider` / scenario-engine-based evil agents — deferred to those later
  use cases. RAG uses the `agent.py` composition path.
- `type_mismatch` and an `irrelevant`-distinct-from-`unknown` mode — both need a
  multi-tool agent; the single-tool RAG bot can't express them.
- Verifying that a *steered* semantic misbehavior actually changed the model's
  output (closing the weak-label gap needs a detector) — deferred.
- Any change to `agent.py`, the deterministic provider, the scenario orchestrator,
  the golden traces, or production behavior.

## Architecture

The feature is **additive to the live path** and built entirely by composing over
two dependencies `Agent.__init__` already accepts (`acompletion`, `tools`). The good
agent is the unmodified production bot; the evil agent is the same `Agent` class
wired with two policy-driven wrappers.

```
good:  Agent(config, indexer)                              # real, untouched
evil:  Agent(config, indexer,
             acompletion = evil_acompletion(real, policy),
             tools       = [evil_search(real_search, policy)])
                         │  policy: MisbehaviorPolicy(seed, rate, modes)
                         ▼  seeded, labeled injections across pipeline stages
                 agent.chat(messages)  →  ChatResult (real production trace)
                 policy.injections     →  [{stage, mode, turn, round, detail}, …]
                                          (ground truth for supervisor testing)
```

### New components (all new files in `simple_chatbot/`)

- **`misbehavior.py`** — `MisbehaviorConfig`, `Injection`, `MisbehaviorPolicy`.
  Pure, seeded, no I/O. The substrate-agnostic core reused by future use cases.
- **`evil_rag.py`** — the two wrappers: `evil_search(...)` (wraps a real `ToolDef`
  executor) and `evil_acompletion(...)` (wraps an `acompletion` callable), plus a
  small `build_evil_agent(config, indexer, policy)` helper that assembles the evil
  `Agent`.
- **Server wiring** (`config.py`, `cli.py`, `server.py`) — opt-in `misbehavior`
  config (rate/modes/seed); when `rate > 0`, `init()` builds the evil agent and
  surfaces injection labels on the response payload + conversation log.

## Component detail

### `MisbehaviorPolicy`

```python
@dataclass(frozen=True)
class MisbehaviorConfig:
    rate: float                 # 0.0–1.0: p of firing at each eligible site
    modes: tuple[str, ...]      # enabled mode names
    seed: int

@dataclass
class Injection:
    stage: str   # "retrieval" | "decision" | "answer"
    mode: str
    turn: int
    round: int
    detail: dict

class MisbehaviorPolicy:
    def __init__(self, config: MisbehaviorConfig) -> None: ...  # random.Random(seed)
    def maybe(self, stage: str, eligible_modes, ctx: dict) -> Injection | None: ...
    @property
    def injections(self) -> list[Injection]: ...
```

`maybe()` intersects `eligible_modes` with the enabled set; if non-empty, draws
`rng.random() < rate`; if it fires, `rng.choice` picks the mode, appends an
`Injection` to the log, and returns it. Determinism depends only on the seed and
the order of `maybe()` calls.

### Pipeline stages (mapped to `agent.py`'s existing loop)

| Stage | Site (wrapper) | Modes | Mechanism | Label |
|---|---|---|---|---|
| **retrieval** | `evil_search` executor | `poison_retrieval`, `drop_retrieval` | run real retrieval, then mutate the `ToolResult` chunks + text | **strong** |
| **decision (pre)** | `evil_acompletion` before call | `ignore_retrieval` | prepend a steering system message to `messages` | weak |
| **decision (post)** | `evil_acompletion` after call, **tool-call rounds only** | `redundant_search`, `malformed_search`, `unknown_tool` | mutate `message.tool_calls` (duplicate / corrupt-args via existing `_MALFORMED_ARGS` marker / synthesize unknown call) | **strong** |
| **answer** | `evil_acompletion`, final round | `wrong_value` | steer ("include a plausible but unsupported detail") | weak |

Structural call injections piggyback only on rounds where the model **already**
returned tool calls (we don't flip a final answer into a tool-call round) — keeps
the mutation realistic and simple.

## Data flow & determinism

- **Scripted LLM** (`SIMPLE_CHATBOT_SCRIPTED_LLM=1`, fixed responses): call sequence
  is fixed → the whole evil run is **fully reproducible**. This is the CI regime.
- **Real LLM**: responses vary → round/call counts vary → the injection *pattern*
  can differ run-to-run at the same seed. Inherent to a real agent. The seed
  reproduces the policy's *per-site decisions*, not byte-identical traces.

## Error handling & production safety

- The **good path is `agent.py` unchanged** — not a code path this feature touches.
- Evil wrappers **never break the pipeline**: a failed mutation falls back to the
  real result/response; the policy is side-effect-free apart from its own log.
- Evil is **opt-in** — active only when a `MisbehaviorConfig` is supplied; the
  default server runs the production/good agent.
- Injected content is recorded as injected (in `policy.injections` and the log) so a
  poisoned trace is never mistaken for a real outage.

## Testing

- **Unit — `MisbehaviorPolicy`:** same seed → same `maybe()` outcome sequence;
  `rate=0.0` never fires, `rate=1.0` always fires; mode filtering (only enabled +
  eligible modes fire); injection-log accuracy (stage/mode/turn/round).
- **Unit — `evil_search`:** wraps a fake real executor; `poison_retrieval` /
  `drop_retrieval` mutate chunks + text and log; `rate=0` is an identical passthrough.
- **Unit — `evil_acompletion`:** with a fake underlying `acompletion` returning a
  known response — `ignore_retrieval` prepends the steering message;
  `redundant_search` / `malformed_search` / `unknown_tool` mutate `tool_calls` on a
  tool-call round and are no-ops on a final-answer round; `rate=0` is passthrough.
- **Integration (offline, scripted LLM):** a good `Agent` → empty injection log and
  a faithful trace; an evil `Agent` at seed `S` → the trace contains exactly the
  labeled injections; reproducible across repeated runs.
- **Good/evil pair demo test:** same input through both agents; assert the good run
  uses retrieval and the evil run shows the labeled divergence. This is the
  supervisor-fuel artifact in miniature.
- **Live smoke test:** opt-in, `skip`-marked without a real key — not a CI gate.
- **Regression gate:** full suite green (`python -m pytest -q`).

## Server wiring (minimal)

Add an opt-in `misbehavior` config (rate / modes / seed) via CLI flags + env vars,
mirroring the existing `--default-fixture` / `--scenario-mode` conventions. When
`rate > 0`, `init()` builds the evil-wrapped `Agent`; the per-turn injection labels
are surfaced on the `/v1/responses` payload (a non-standard field normal clients
ignore) and in the conversation log, so a Spectral-style consumer receives
**(real trace, ground-truth label)** pairs. The primary consumer remains in-process
tests + trace capture; the server path exists to drive it like production.

## Open seams & notes

- **Weak-label gap.** Steered semantic modes (`ignore_retrieval`, `wrong_value`) are
  realistic but label *intent*, not outcome. The strong-label retrieval-fault modes
  cover the same faithfulness dimension with exact ground truth; closing the gap for
  the steered modes needs a detector (deferred).
- **Reusable core.** `MisbehaviorPolicy` is substrate-agnostic. Future multi-agent
  use cases wire the same policy into the scenario provider layer (`EvilLiveProvider`)
  instead of `agent.py`; only the thin injection-site adapters differ.
- **Mode set grows by declaration.** New modes are added to the policy's vocabulary
  and a wrapper site; `type_mismatch` and a distinct `irrelevant` mode arrive with the
  first multi-tool use case.
