# AI Service Descriptions (AISD) for fixtures + live RAG agent — design

**Date:** 2026-06-04
**Status:** Approved (pending spec review)
**Author:** Niccolò Campolungo / Claude
**Repo:** `simple-chatbot` (this repo only)
**Branch:** `feat/agentic-behavior` (continue on it — do **not** branch off or merge to main)

## Goal

Give Spectral (the conversation-steering / jailbreak tester) an accurate **AI
Service Description (AISD)** of whatever chatbot a `simple-chatbot` server is
serving, so it knows what the bot under test can actually do. Each fixture and the
live RAG agent gets its own AISD. The server **prints the active chatbot's AISD at
startup** so a human can copy-paste it into Spectral (there is no API integration
yet).

## Crux: the AISD describes capabilities, not behavior

The AISD describes the chatbot's **surface and capabilities** (its tools, domains,
routing/handoff, multi-turn behavior). It deliberately does **not** reveal
evaluation intent or whether the bot misbehaves. Two consequences, both locked:

- **Good/evil invariance.** The good RAG agent and its evil twin share one
  identical `live-rag` AISD. Toggling misbehavior flags never changes the printed
  AISD — Spectral must not be tipped off which variant it is probing.
- **Probe fixtures are described as the domain bot they emulate.** `relevance-probe`,
  `validity-probe`, and `rag-ignore` are deliberate-misbehavior scripts, but their
  AISD presents the ordinary domain bot (its tool surface), not "this misbehaves on
  purpose." The misbehavior is the thing under test, not a capability.

## Relationship to the existing `guard.service_description`

`config.guard.service_description` already exists, but it serves a **different
consumer** (ScopeGuard, to judge in-scope vs out-of-scope requests) and is a single
config value. The AISDs here are per-chatbot stored content for the **Spectral**
tester. They are kept separate; this work does not touch ScopeGuard. (A future
change could let an AISD feed the guard, but that is out of scope.)

## Locked decisions

| Topic | Decision | Rationale |
|---|---|---|
| **Storage** | Markdown files under `simple_chatbot/aisd/`, one per fixture id + `live-rag.md`. | Long-form prose is easier to author/edit/`cat` as real markdown than inline in Python; keyed by id so it mirrors the registry. |
| **Surfacing** | Print the **active default** chatbot's AISD to stdout at server startup. | Spectral has no API integration; the workflow is copy-paste. Active default = what a model-less client (Spectral) actually reaches. |
| **Selection rule** | `--default-fixture` set → that fixture's AISD; else → `live-rag.md`. | The default chatbot is exactly what Spectral hits when it cannot set `model`. |
| **Good/evil** | AISD never reads the misbehavior config; identical for good and evil. | Capability description, not behavior; must not leak the variant. |
| **Scope of output** | Active default only (not all fixtures). | Cleanest copy-paste — exactly the one Spectral will probe. |
| **Missing AISD** | Warn + print a fallback (scenario `description` / placeholder); non-fatal. | A missing description must not block the server. |

## Non-goals

- No HTTP endpoint for AISDs (copy-paste-at-startup is the only consumer now). The
  resolver is the seam a future `GET /v1/service-description/{id}` would extend.
- No change to ScopeGuard or `guard.service_description`.
- No change to the chatbot's behavior, the agent, the providers, or the misbehavior
  feature.
- Not printing every fixture's AISD at startup; no `describe` CLI subcommand (can be
  added later on the same resolver).

## Architecture

Additive: one content directory, one small resolver module, and a few lines in
`cli.serve` to print the active AISD at startup.

```
simple_chatbot/aisd/<id>.md  ─┐
simple_chatbot/aisd/live-rag.md ─┤  read by
                                 ▼
        service_description.py: load_service_description(key) / resolve_active(config, registry)
                                 │  selection rule (default_fixture else live-rag)
                                 ▼
        cli.serve (after init()): print_service_description(label, text)  ─► stdout (clean block)
```

### Component 1 — content files (`simple_chatbot/aisd/`)

Eight markdown files, keyed by fixture id, plus the reserved `live-rag` key:
`cs-routing.md`, `multi-round-routing.md`, `escalation.md`, `parallel.md`,
`rag-ignore.md`, `relevance-probe.md`, `validity-probe.md`, `live-rag.md`.

Each follows the established AISD structure:
- `## Service overview` — what the bot is, who uses it.
- `## Surface & capabilities` — actual tool names + params, routing/handoff targets,
  multi-turn behavior, anything Spectral can exercise.
- `## Out of scope` — what it cannot do; boundaries.

Authoring principles: capabilities/surface only; no evaluation/misbehavior intent;
probe fixtures described as their domain bot; `live-rag` describes the real RAG bot
(searches an indexed knowledge base via `search_documents`, agentic tool loop,
answers from retrieved documents) and is good/evil-invariant.

Per-fixture surface (from the fixture definitions):
- **cs-routing** — customer-service dispatcher routing to a billing agent
  (`lookup_invoice`, `issue_refund`) or human escalation.
- **multi-round-routing** — support triage: front desk → tier-1 → tier-2
  (`check_status`).
- **escalation** — dispatcher that escalates directly to a human agent.
- **parallel** — concierge gathering facts via `get_weather` + `get_events`.
- **rag-ignore** — support bot answering from a knowledge base via `search_docs`.
- **relevance-probe** — billing assistant with `lookup_invoice` and `get_weather`.
- **validity-probe** — account/ops assistant with `lookup_invoice`, `get_order`,
  `set_credit_limit`.
- **live-rag** — RAG assistant searching an indexed document collection via
  `search_documents`.

### Component 2 — resolver (`simple_chatbot/service_description.py`)

```python
LIVE_RAG_KEY = "live-rag"

def load_service_description(key: str) -> str | None:
    """Return the markdown AISD for a fixture id or LIVE_RAG_KEY, or None if the
    aisd/<key>.md file is absent."""

def resolve_active(config, registry) -> tuple[str, str]:
    """Return (label, markdown) for the chatbot this server serves by default:
    config.default_fixture's AISD when set, else the live-rag AISD. Never reads the
    misbehavior config. On a missing file, returns a fallback text (the registry
    scenario's one-line description, or a clear placeholder) — never raises."""

def print_service_description(label: str, text: str) -> None:
    """Write a clearly-delimited AISD block to stdout (raw markdown, bypassing
    loguru) for clean copy-paste."""
```

Reads files via `Path(__file__).parent / "aisd" / f"{key}.md"`.

### Component 3 — startup wiring (`simple_chatbot/cli.py`)

In `serve`, after `init(...)` succeeds and before `uvicorn.run(...)`: resolve the
active AISD (using the loaded registry) and `print_service_description(...)`. Also
emit a one-line loguru `info` ("Printed AI service description for <label>") for the
operational log. The block goes to stdout so it is copy-pasteable without log
formatting.

## Data flow

1. `serve` builds `config`, calls `init(config, indexer, ...)` (loads the fixture
   registry, validates `--default-fixture`).
2. `resolve_active(config, registry)` picks the key (default_fixture or `live-rag`)
   and loads `aisd/<key>.md`, falling back to the scenario description / placeholder
   if absent.
3. `print_service_description(label, text)` prints the delimited block to stdout.
4. `uvicorn.run(...)` starts serving as before.

## Error handling

- Active default's AISD file missing → loguru `warning` + print the fallback
  (registry scenario `description`, or a placeholder for `live-rag`). Server still
  starts.
- File read errors are caught and degrade to the same fallback.
- Unknown `--default-fixture` id still fail-fasts in `init()` (unchanged behavior);
  the AISD layer never sees an unknown id because `init()` already rejected it.
- Packaging note (non-blocking): files are read via a module-relative path, which
  works for source / editable installs (how this repo runs). If the package is ever
  built into a wheel, add `aisd/*.md` to package data; not required now.

## Testing

- **Resolver units** (`tests/test_service_description.py`):
  - `load_service_description` returns markdown for a known key; `None` for an
    unknown key.
  - `resolve_active` with `default_fixture="cs-routing"` returns
    `("cs-routing", <file text>)`.
  - `resolve_active` with no `default_fixture` returns `("live-rag", <file text>)`.
  - Missing-file fallback: monkeypatch the directory / use an unknown default so the
    file is absent, assert it returns the scenario `description` (or placeholder) and
    does not raise.
- **Coverage guard:** every id in `load_fixtures()` has a matching `aisd/<id>.md`,
  and `aisd/live-rag.md` exists; each file is non-empty and contains the three
  section headers (`## Service overview`, `## Surface & capabilities`,
  `## Out of scope`). Catches adding a fixture without its AISD.
- **Good/evil invariance:** `resolve_active` returns identical `live-rag` text for a
  config with misbehavior disabled vs enabled (rate 0 vs rate 1, no default_fixture).
- **Print format:** `print_service_description` writes a block containing the text
  between delimiter lines (capture stdout).
- **Regression gate:** full suite green (`python -m pytest -q`).

## Open seams & notes

- The `resolve_active` seam is exactly what a future `GET /v1/service-description/{id}`
  endpoint (or a `describe` CLI command) would call — no rework needed to add
  programmatic Spectral integration later.
- AISD content is independent of run mode (deterministic vs live) and of the
  misbehavior config; it describes the scenario's declared capability surface.
