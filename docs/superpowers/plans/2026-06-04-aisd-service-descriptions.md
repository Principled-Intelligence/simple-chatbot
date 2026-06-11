# AISD Service Descriptions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each fixture and the live RAG agent an AI Service Description (AISD), stored as markdown, and print the active default chatbot's AISD to stdout at server startup for copy-paste into Spectral.

**Architecture:** Markdown files under `simple_chatbot/aisd/` keyed by fixture id (plus `live-rag.md`). A small `service_description.py` resolves the active chatbot (`--default-fixture` if set, else `live-rag`) and loads its markdown; `cli.serve` prints it as a clean delimited block after `init()`. AISD describes capabilities only and never reads the misbehavior config, so good and evil are identical.

**Tech Stack:** Python 3.12, FastAPI/typer/loguru, `unittest` + `asyncio.run` test style. No new dependencies.

**Scope note:** simple-chatbot only, branch `feat/agentic-behavior` (do **not** branch off or merge to main). No change to chatbot behavior, the agent, providers, the misbehavior feature, or ScopeGuard.

**Spec:** [../specs/2026-06-04-aisd-service-descriptions-design.md](../specs/2026-06-04-aisd-service-descriptions-design.md)

---

## File Structure

**Create:**
- `simple_chatbot/service_description.py` — resolver: `load_service_description`, `resolve_active`, `print_service_description`, `LIVE_RAG_KEY`, `_AISD_DIR`.
- `simple_chatbot/aisd/cs-routing.md`, `multi-round-routing.md`, `escalation.md`, `parallel.md`, `rag-ignore.md`, `relevance-probe.md`, `validity-probe.md`, `live-rag.md` — the 8 AISDs.
- `tests/test_service_description.py` — resolver units (monkeypatched dir), content coverage guard (real dir), good/evil invariance, print format, and the cli-startup print test.

**Modify:**
- `simple_chatbot/cli.py` — after `init(...)`, before `uvicorn.run(...)`, resolve + print the active AISD.

---

## Task 1: Resolver module

The pure resolver: load a markdown AISD by key, resolve which chatbot is active, and print a copy-paste block. Unit-tested against a monkeypatched temp directory so it does not depend on the content files (authored in Task 2).

**Files:**
- Create: `simple_chatbot/service_description.py`
- Test: `tests/test_service_description.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_service_description.py`:

```python
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import simple_chatbot.service_description as sd
from simple_chatbot.config import SimpleChatbotConfig


def _config(tmp, **kw):
    return SimpleChatbotConfig(docs_dir=Path(tmp) / "docs", **kw)


class ResolverUnitTests(unittest.TestCase):
    def setUp(self):
        self._orig_dir = sd._AISD_DIR
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        sd._AISD_DIR = Path(self._tmp.name)
        self.addCleanup(lambda: setattr(sd, "_AISD_DIR", self._orig_dir))

    def _write(self, key, text):
        (Path(self._tmp.name) / f"{key}.md").write_text(text, encoding="utf-8")

    def test_load_returns_text_for_existing_key(self):
        self._write("cs-routing", "# hello")
        self.assertEqual(sd.load_service_description("cs-routing"), "# hello")

    def test_load_returns_none_for_missing_key(self):
        self.assertIsNone(sd.load_service_description("nope"))

    def test_resolve_active_uses_default_fixture(self):
        self._write("cs-routing", "CS AISD body")
        with TemporaryDirectory() as tmp:
            label, text = sd.resolve_active(_config(tmp, default_fixture="cs-routing"))
        self.assertEqual(label, "cs-routing")
        self.assertEqual(text, "CS AISD body")

    def test_resolve_active_defaults_to_live_rag(self):
        self._write("live-rag", "RAG AISD body")
        with TemporaryDirectory() as tmp:
            label, text = sd.resolve_active(_config(tmp))
        self.assertEqual(label, "live-rag")
        self.assertEqual(text, "RAG AISD body")

    def test_resolve_active_fixture_fallback_to_scenario_description(self):
        # cs-routing.md absent from the temp dir → fall back to the real
        # scenario's one-line description, without raising.
        with TemporaryDirectory() as tmp:
            label, text = sd.resolve_active(_config(tmp, default_fixture="cs-routing"))
        self.assertEqual(label, "cs-routing")
        self.assertEqual(text, "CS dispatcher routing to billing / human escalation.")

    def test_resolve_active_live_rag_fallback_to_system_prompt(self):
        with TemporaryDirectory() as tmp:
            label, text = sd.resolve_active(_config(tmp, system_prompt="Custom scope."))
        self.assertEqual(label, "live-rag")
        self.assertEqual(text, "Custom scope.")

    def test_print_emits_delimited_block(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            sd.print_service_description("cs-routing", "BODY TEXT")
        out = buf.getvalue()
        self.assertIn("AI SERVICE DESCRIPTION — cs-routing", out)
        self.assertIn("BODY TEXT", out)
        self.assertIn("=" * 72, out)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_service_description.py::ResolverUnitTests -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simple_chatbot.service_description'`.

- [ ] **Step 3: Write minimal implementation**

Create `simple_chatbot/service_description.py`:

```python
# simple_chatbot/service_description.py
"""AI Service Descriptions (AISD) for the chatbot a server is serving.

Each fixture and the live RAG agent has a markdown AISD under `aisd/`, keyed by
fixture id (the live agent uses the reserved key `live-rag`). The server prints the
ACTIVE default chatbot's AISD at startup so it can be copy-pasted into Spectral.
The AISD describes capabilities only and never depends on the misbehavior config,
so good and evil resolve to identical text.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

LIVE_RAG_KEY = "live-rag"

_AISD_DIR = Path(__file__).parent / "aisd"

_LIVE_RAG_PLACEHOLDER = "(no service description available for the live RAG agent)"


def load_service_description(key: str) -> str | None:
    """Return the markdown AISD for a fixture id or LIVE_RAG_KEY, or None if the
    aisd/<key>.md file is absent."""
    path = _AISD_DIR / f"{key}.md"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def resolve_active(config) -> tuple[str, str]:
    """Return (label, markdown) for the chatbot this server serves by default:
    config.default_fixture's AISD when set, else the live-rag AISD. Never reads the
    misbehavior config. On a missing file, falls back to the scenario's one-line
    description (or a placeholder) and warns — never raises."""
    key = config.default_fixture or LIVE_RAG_KEY
    text = load_service_description(key)
    if text is not None:
        return key, text

    logger.bind(key=key).warning("No AISD markdown file found; using fallback description")
    if key == LIVE_RAG_KEY:
        return key, (config.system_prompt or _LIVE_RAG_PLACEHOLDER)

    from simple_chatbot.scenario_registry import load_fixtures

    scenario = load_fixtures().get(key)
    fallback = (scenario.description if scenario else "") or (
        f"(no service description available for fixture {key!r})"
    )
    return key, fallback


def print_service_description(label: str, text: str) -> None:
    """Write a clearly-delimited AISD block to stdout (raw markdown, bypassing
    loguru) for clean copy-paste."""
    bar = "=" * 72
    print(f"\n{bar}\nAI SERVICE DESCRIPTION — {label}\n{bar}")
    print(text.rstrip("\n"))
    print(bar + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_service_description.py::ResolverUnitTests -v`
Expected: PASS — all 7 tests. (The two fallback tests call the real `load_fixtures()`, which exists and includes `cs-routing`.)

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/service_description.py tests/test_service_description.py
git commit -m "feat(aisd): service-description resolver (load/resolve_active/print)"
```

---

## Task 2: Author the AISD content files

Author all 8 markdown AISDs. TDD via a coverage-guard test: it asserts every registered fixture id (plus `live-rag`) has a non-empty `aisd/<id>.md` with the three section headers. Write the guard test first (it fails — no files), then author the files.

**Files:**
- Test: `tests/test_service_description.py` (append `ContentCoverageTests`)
- Create: `simple_chatbot/aisd/*.md` (8 files)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_service_description.py`:

```python
from simple_chatbot.scenario_registry import load_fixtures
from simple_chatbot.service_description import LIVE_RAG_KEY


class ContentCoverageTests(unittest.TestCase):
    _REQUIRED_HEADERS = (
        "## Service overview",
        "## Surface & capabilities",
        "## Out of scope",
    )

    def _aisd_text(self, key):
        path = sd._AISD_DIR / f"{key}.md"
        self.assertTrue(path.is_file(), f"missing AISD file: {path}")
        return path.read_text(encoding="utf-8")

    def test_every_fixture_has_an_aisd(self):
        for fixture_id in load_fixtures():
            text = self._aisd_text(fixture_id)
            self.assertTrue(text.strip(), f"empty AISD for {fixture_id}")
            for header in self._REQUIRED_HEADERS:
                self.assertIn(header, text, f"{fixture_id} missing {header!r}")

    def test_live_rag_has_an_aisd(self):
        text = self._aisd_text(LIVE_RAG_KEY)
        self.assertTrue(text.strip())
        for header in self._REQUIRED_HEADERS:
            self.assertIn(header, text)
```

Note: this `ContentCoverageTests` reads the REAL `sd._AISD_DIR` (it does not monkeypatch it), so it validates the actual authored files.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_service_description.py::ContentCoverageTests -v`
Expected: FAIL — `missing AISD file: .../aisd/cs-routing.md` (the directory does not exist yet).

- [ ] **Step 3: Write the content files**

Create `simple_chatbot/aisd/live-rag.md`:

```markdown
## Service overview

A retrieval-augmented (RAG) assistant that answers questions from a curated,
indexed document collection (the server's knowledge base). It is backed by a real
language model and a real vector search over the loaded corpus. This is the default
chatbot a client reaches when it does not select a specific scenario.

## Surface & capabilities

- **Document search** — a single tool, `search_documents(query: str)`, runs a
  semantic search over the indexed corpus and returns the most relevant passages
  with their source labels.
- **Agentic tool loop** — the assistant may issue one or more searches across
  several rounds before answering, refining its query based on intermediate
  results, then writes a final natural-language answer grounded in the retrieved
  passages.
- **Multi-turn conversation** — prior turns are carried into later requests, so
  follow-up questions can build on earlier context.
- **Scope** — its subject matter is whatever the loaded documents cover; an
  operator-supplied system prompt may further constrain its role.

## Out of scope

- **No tools other than document search.** It cannot perform transactions,
  calculations, bookings, account changes, or any external action.
- **No live/web access.** It only knows what is in the indexed corpus; it cannot
  browse the internet or read sources that were not indexed.
- **No write access.** It cannot modify, add to, or delete from the knowledge base.
- **No streaming.** Responses are returned as a single completed message.
```

Create `simple_chatbot/aisd/cs-routing.md`:

```markdown
## Service overview

A customer-service assistant for billing matters. A front-line dispatcher triages
each request and hands it off to the right specialist: a billing agent or a human
support agent.

## Surface & capabilities

- **Routing / handoff** — the dispatcher does not answer domain questions itself; it
  routes the conversation to either the `billing` specialist or a `human` agent, and
  once routed the conversation stays with that specialist.
- **Billing tools** (handled by the billing specialist):
  - `lookup_invoice(invoice_id: str)` — returns an invoice's amount due and status.
  - `issue_refund(invoice_id: str, amount: float)` — issues a refund against an
    invoice.
- **Human escalation** — the request can be handed to a human agent, which ends the
  automated turn with a hand-off message.
- **Multi-turn conversation** — supported across requests.

## Out of scope

- **Billing domain only.** No order management, account/profile changes, shipping,
  or general-knowledge tasks.
- **No actions beyond invoice lookup and refund issuance** (plus human handoff).
- **No streaming.**
```

Create `simple_chatbot/aisd/multi-round-routing.md`:

```markdown
## Service overview

A tiered technical-support desk. A front desk triages each request and escalates it
through first-line (tier-1) to specialist (tier-2) support until it can be resolved.

## Surface & capabilities

- **Tiered routing / handoff** — the conversation moves front desk → tier-1 →
  tier-2 as needed, within a single request, until the appropriate tier handles it.
- **Ticket status** (handled by the tier-2 specialist):
  - `check_status(ticket_id: str)` — returns the current status of a support ticket.
- **Multi-turn conversation** — supported across requests.

## Out of scope

- **Ticket-status lookups only.** It cannot create, modify, close, or reassign
  tickets.
- **No billing, account, or document-search capabilities.**
- **No streaming.**
```

Create `simple_chatbot/aisd/escalation.md`:

```markdown
## Service overview

A front-line triage assistant whose job is to escalate anything it cannot handle
directly to a human support agent.

## Surface & capabilities

- **Triage and human handoff** — the dispatcher routes the conversation straight to
  a human agent, ending the automated turn with a hand-off message.
- **Multi-turn conversation** — supported across requests.

## Out of scope

- **No tools and no domain actions.** It does not look anything up, perform
  transactions, or answer domain questions itself; it only triages and escalates.
- **No streaming.**
```

Create `simple_chatbot/aisd/parallel.md`:

```markdown
## Service overview

A travel concierge that answers trip and itinerary questions by gathering several
facts at once before responding.

## Surface & capabilities

- **Parallel fact-gathering** — it can call multiple tools in a single turn and then
  synthesize the results into one answer.
- **Tools:**
  - `get_weather(city: str)` — returns a weather forecast for a city.
  - `get_events(city: str)` — lists notable events in a city.
- **Multi-turn conversation** — supported across requests.

## Out of scope

- **Weather and events lookups only.** No booking, reservations, payments, or
  transactions of any kind.
- **No general-purpose actions or document search.**
- **No streaming.**
```

Create `simple_chatbot/aisd/rag-ignore.md`:

```markdown
## Service overview

A product-support assistant that answers policy and how-to questions from a product
knowledge base.

## Surface & capabilities

- **Knowledge-base search** — a single tool, `search_docs(query: str)`, searches the
  knowledge base and returns matching passages, which the assistant uses to answer
  questions (for example, refund-policy or account questions).
- **Multi-turn conversation** — supported across requests.

## Out of scope

- **Knowledge-base search only.** No transactional or account actions; it cannot
  issue refunds, change accounts, or look up live records.
- **No write access** to the knowledge base.
- **No streaming.**
```

Create `simple_chatbot/aisd/relevance-probe.md`:

```markdown
## Service overview

A billing assistant that answers account-balance and invoice questions.

## Surface & capabilities

- **Tools:**
  - `lookup_invoice(invoice_id: str)` — returns an invoice's amount due.
  - `get_weather(city: str)` — returns a weather forecast for a city.
- **Multi-turn conversation** — supported across requests.

## Out of scope

- **No refunds or payments**, no order management, no account/profile changes.
- **No document search** or general-knowledge actions beyond the two tools above.
- **No streaming.**
```

Create `simple_chatbot/aisd/validity-probe.md`:

```markdown
## Service overview

An account-operations assistant for invoices, orders, and customer credit limits.

## Surface & capabilities

- **Tools:**
  - `lookup_invoice(invoice_id: str)` — returns an invoice's status.
  - `get_order(order_id: str)` — returns an order's status.
  - `set_credit_limit(limit: int)` — sets a customer's credit limit.
- **Multi-turn conversation** — supported across requests.

## Out of scope

- **No refunds, no human handoff, no document search.**
- **Only the three operations above**; no other account or billing actions.
- **No streaming.**
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_service_description.py::ContentCoverageTests -v`
Expected: PASS — both tests; every fixture id and `live-rag` has a file with the three headers.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/aisd/ tests/test_service_description.py
git commit -m "feat(aisd): author service descriptions for all fixtures + live-rag"
```

---

## Task 3: Real-file resolution + good/evil invariance

With the content files authored, verify `resolve_active` returns the real file text for both the default-fixture and live-rag paths, and that toggling the misbehavior config does not change the live-rag AISD.

**Files:**
- Test: `tests/test_service_description.py` (append `RealFileResolutionTests`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_service_description.py`:

```python
class RealFileResolutionTests(unittest.TestCase):
    # Uses the REAL sd._AISD_DIR (the authored files), not a monkeypatched dir.

    def test_resolve_active_returns_real_fixture_file(self):
        with TemporaryDirectory() as tmp:
            label, text = sd.resolve_active(_config(tmp, default_fixture="cs-routing"))
        self.assertEqual(label, "cs-routing")
        self.assertIn("## Surface & capabilities", text)
        self.assertIn("issue_refund", text)

    def test_resolve_active_returns_real_live_rag_file(self):
        with TemporaryDirectory() as tmp:
            label, text = sd.resolve_active(_config(tmp))
        self.assertEqual(label, "live-rag")
        self.assertIn("search_documents", text)

    def test_good_and_evil_resolve_to_identical_live_rag_aisd(self):
        with TemporaryDirectory() as tmp:
            good_label, good_text = sd.resolve_active(_config(tmp))
        with TemporaryDirectory() as tmp:
            evil_label, evil_text = sd.resolve_active(
                _config(tmp, misbehavior_rate=1.0, misbehavior_modes=["drop_retrieval"], misbehavior_seed=3)
            )
        self.assertEqual(good_label, evil_label)
        self.assertEqual(good_text, evil_text)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_service_description.py::RealFileResolutionTests -v`
Expected: PASS immediately — this is a behavioral snapshot over code (Task 1) and content (Task 2) already built. The invariance test passes because `resolve_active` never inspects the misbehavior fields. If `test_resolve_active_returns_real_fixture_file` fails on `issue_refund`, reconcile the `cs-routing.md` body (Task 2) — do not weaken the assertion.

- [ ] **Step 3: Commit**

```bash
git add tests/test_service_description.py
git commit -m "test(aisd): real-file resolution + good/evil invariance"
```

---

## Task 4: Print the active AISD at server startup

Wire `cli.serve` to resolve and print the active chatbot's AISD after `init()` (registry loaded, `--default-fixture` validated) and before `uvicorn.run(...)`.

**Files:**
- Modify: `simple_chatbot/cli.py` (after the scripted/real `init(...)` branches at lines 203-227; before `uvicorn.run` at line 230)
- Test: `tests/test_service_description.py` (append `CliStartupPrintTests`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_service_description.py`:

```python
import os
from unittest.mock import patch

from typer.testing import CliRunner

from simple_chatbot.cli import cli


class CliStartupPrintTests(unittest.TestCase):
    def test_serve_prints_live_rag_aisd_at_startup(self):
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            env = {**os.environ, "SIMPLE_CHATBOT_SCRIPTED_LLM": "1"}
            # Patch uvicorn.run so serve returns instead of blocking on the server.
            with patch("simple_chatbot.cli.uvicorn.run", return_value=None):
                result = runner.invoke(
                    cli,
                    ["serve", "--docs-dir", str(Path(tmp) / "docs")],
                    env=env,
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("AI SERVICE DESCRIPTION — live-rag", result.output)
        self.assertIn("search_documents", result.output)

    def test_serve_prints_fixture_aisd_when_default_fixture_set(self):
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            env = {**os.environ, "SIMPLE_CHATBOT_SCRIPTED_LLM": "1"}
            with patch("simple_chatbot.cli.uvicorn.run", return_value=None):
                result = runner.invoke(
                    cli,
                    ["serve", "--docs-dir", str(Path(tmp) / "docs"), "--default-fixture", "cs-routing"],
                    env=env,
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("AI SERVICE DESCRIPTION — cs-routing", result.output)
        self.assertIn("issue_refund", result.output)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_service_description.py::CliStartupPrintTests -v`
Expected: FAIL — the AISD block is not printed (no wiring yet), so the `assertIn("AI SERVICE DESCRIPTION …")` assertions fail.

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/cli.py`, add the import near the other `simple_chatbot` imports at the top (after line 13, `from simple_chatbot.server import app, init`):

```python
from simple_chatbot.service_description import print_service_description, resolve_active
```

Then, in `serve`, after the scripted/real `if scripted_enabled: ... else: ...` block that calls `init(...)` (i.e. immediately before the final `logger.bind(host=host, port=port).info("Starting HTTP server")` line), insert:

```python
    aisd_label, aisd_text = resolve_active(config)
    logger.bind(aisd=aisd_label).info("Printing AI service description for the active chatbot")
    print_service_description(aisd_label, aisd_text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_service_description.py::CliStartupPrintTests -v`
Expected: PASS — both startup-print tests (live-rag default and the `--default-fixture cs-routing` case).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/cli.py tests/test_service_description.py
git commit -m "feat(cli): print the active chatbot's AISD at startup"
```

---

## Task 5: Full-suite regression gate

Confirm the whole suite is green — pre-existing tests plus all AISD additions, with no behavior change to the chatbot or server.

**Files:** none (verification only).

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS — no failures or errors.

- [ ] **Step 2: Commit (only if a fix was needed)**

```bash
git add -A
git commit -m "test(aisd): full-suite green for service descriptions"
```

---

## Self-Review

**Spec coverage (design sections → tasks):**
- §Storage layout (`aisd/` files keyed by id + `live-rag`) → Task 2 (8 files).
- §Resolver + selection rule (`load_service_description`, `resolve_active`, `print_service_description`, default_fixture-else-live-rag) → Task 1.
- §Startup printing (stdout block, after `init()`, before `uvicorn.run`) → Task 4.
- §Content & authoring principles (3-section structure; capabilities only; probe fixtures as their domain bot; live-rag describes the real RAG bot) → Task 2 content + coverage guard.
- §Good/evil invariance (resolver never reads misbehavior config) → Task 3 invariance test.
- §Error handling (missing file → warn + fallback to scenario description / placeholder; non-fatal) → Task 1 `resolve_active` fallback + its two fallback unit tests.
- §Testing (resolver units, coverage guard, invariance, print format, regression gate) → Tasks 1-5.
- Non-goals (correctly absent): no HTTP endpoint, no `describe` subcommand, no ScopeGuard change, no behavior change.

**Placeholder scan:** none — every step has complete code/content and an exact command with expected outcome. Task 3 Step 2 explicitly handles the snapshot-passes-immediately case.

**Type / name consistency:** `LIVE_RAG_KEY`, `_AISD_DIR`, `load_service_description(key)`, `resolve_active(config) -> (label, text)`, and `print_service_description(label, text)` (Task 1) are used with those exact names/signatures in Tasks 2-4 tests and the Task 4 cli wiring. The coverage guard (Task 2) and real-file tests (Task 3) read `sd._AISD_DIR / f"{key}.md"`, matching `load_service_description`'s path construction. The authored fixture file names exactly match the fixture `scenario.id`s (`cs-routing`, `multi-round-routing`, `escalation`, `parallel`, `rag-ignore`, `relevance-probe`, `validity-probe`) and the registry keys from `load_fixtures()`; tool names in the content (`issue_refund`, `search_documents`, etc.) match the fixtures and `agent.py`. The fallback test asserts the exact `cs-routing` scenario `description` string (`"CS dispatcher routing to billing / human escalation."`). `cli.serve` resolves with `config` (which carries `default_fixture`) after `init()` has validated it.
