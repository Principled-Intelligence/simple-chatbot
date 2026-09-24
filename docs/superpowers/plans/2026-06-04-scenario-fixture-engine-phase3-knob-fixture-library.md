# Scenario Fixture Engine — Phase 3 (Knob & Fixture Library) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `Parallel` fan-out step + three new `Call` intent tags to the deterministic scenario engine, then author a library of canonical fixtures on top, with golden-trace snapshots locking the new canonical fixtures' deterministic output.

**Architecture:** Phase 1's `ScenarioOrchestrator` round loop already emits multi-`tool_call` rounds (it iterates `decision.calls` and assigns per-call ids), so the only engine change is teaching `DeterministicProvider` to *produce* a multi-call decision from a new `Parallel` script step. Three booleans on `Call` (`redundant`, `wrong_value`, `type_mismatch`) record authorial intent with no wire-shape change (exactly like the existing `irrelevant` tag). Everything else is fixture modules (auto-discovered by the registry) and tests.

**Tech Stack:** Python 3.12, pydantic v2, FastAPI, `unittest` + `asyncio.run` (existing test style). No new dependencies.

**Scope note:** simple-chatbot only, on branch `feat/agentic-behavior` (do **not** branch off or merge to main). Deterministic-only — the Phase 2 `LiveProvider`, mode switch, active-agent persistence, orchestrator round loop, and catalog projection are **not** touched. No ghost-backend (Phase 4 is author-gated). Misbehavior knobs remain deterministic-mode-only.

**Spec:** [../specs/2026-06-04-scenario-fixture-engine-phase3-knob-fixture-library-design.md](../specs/2026-06-04-scenario-fixture-engine-phase3-knob-fixture-library-design.md)

---

## File Structure

**Modify:**
- `simple_chatbot/scenario.py` — add `@dataclass Parallel`, extend the `Step` union, add three boolean tags to `Call`, extend `validate()` to reject non-call-family steps inside a `Parallel`.
- `simple_chatbot/scenario_provider.py` — extract a module-level `_planned_call(step)` helper; add a `Parallel` branch to `DeterministicProvider.decide` returning a multi-`PlannedCall` decision.
- `simple_chatbot/fixtures/validity_probe.py` — add a `type_mismatch` step (and a numeric-param tool to violate).

**Create (fixtures):**
- `simple_chatbot/fixtures/parallel.py`
- `simple_chatbot/fixtures/multi_round_routing.py`
- `simple_chatbot/fixtures/rag_ignore.py`
- `simple_chatbot/fixtures/escalation.py`
- `simple_chatbot/fixtures/relevance_probe.py`

**Create / extend (tests):**
- (extend) `tests/test_scenario_model.py` — `Parallel` model + validation; new `Call` tags.
- (extend) `tests/test_scenario_provider.py` — `Parallel` → multi-call decision.
- (extend) `tests/test_scenario_fixtures.py` — behavioral tests for each new fixture + the `validity-probe` type-mismatch addition.
- (create) `tests/test_golden_traces.py` — full ordered-output snapshots for `parallel`, `rag-ignore`, `cs-routing`.

---

## Task 1: `Parallel` step + validation

A new script step that emits several tool calls in **one** round. It may contain only call-family steps; `validate()` (run from `Scenario.__post_init__`) rejects anything else so a malformed fixture fails at import.

**Files:**
- Modify: `simple_chatbot/scenario.py` (after `UnknownToolCall` at line 99; the `Step` union at line 106; `validate()` at line 150)
- Test: `tests/test_scenario_model.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_scenario_model.py`. First extend the existing import line to include `Parallel`:

```python
from simple_chatbot.scenario import Agent, Scenario, tool, ScenarioTool, Call, Route, Final, MalformedCall, UnknownToolCall, Parallel
```

Then append:

```python
class ParallelStepTests(unittest.TestCase):
    def _tool(self):
        @tool
        def lookup(invoice_id: str) -> dict:
            """Look up an invoice."""
            return {"invoice_id": invoice_id}
        return lookup

    def test_parallel_holds_call_family_steps(self):
        lookup = self._tool()
        p = Parallel([Call(lookup, {"invoice_id": "INV-1"}), MalformedCall(lookup)])
        self.assertEqual(len(p.steps), 2)

    def test_valid_parallel_scenario_constructs(self):
        lookup = self._tool()
        s = Scenario(
            id="ok",
            entry="a",
            agents=[Agent("a", tools=[lookup], script=[Parallel([Call(lookup, {"invoice_id": "INV-1"})])])],
        )
        self.assertEqual(s.entry, "a")

    def test_parallel_rejects_route_inside(self):
        with self.assertRaises(ValueError):
            Scenario(id="bad", entry="a", agents=[Agent("a", script=[Parallel([Route("a")])])])

    def test_parallel_rejects_final_inside(self):
        with self.assertRaises(ValueError):
            Scenario(id="bad", entry="a", agents=[Agent("a", script=[Parallel([Final("x")])])])

    def test_parallel_rejects_nested_parallel(self):
        with self.assertRaises(ValueError):
            Scenario(id="bad", entry="a", agents=[Agent("a", script=[Parallel([Parallel([])])])])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_model.py::ParallelStepTests -v`
Expected: FAIL — `ImportError: cannot import name 'Parallel' from 'simple_chatbot.scenario'`.

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/scenario.py`, add the `Parallel` dataclass immediately after the `UnknownToolCall` class (after line 103, before the `Step =` line):

```python
@dataclass
class Parallel:
    """A fan-out round: emit several tool calls in ONE assistant message.

    May contain only call-family steps (`Call`, `MalformedCall`,
    `UnknownToolCall`) — never `Route`, `Final`, or a nested `Parallel`
    (a fan-out batch cannot hand off, finish, or nest). Enforced in
    `Scenario.validate()`."""

    steps: list[Call | MalformedCall | UnknownToolCall] = field(default_factory=list)
```

Extend the `Step` union (currently line 106) to include `Parallel`:

```python
Step = Call | Route | Final | MalformedCall | UnknownToolCall | Parallel
```

In `Scenario.validate()` (line 150), append a script-validation loop at the end of the method:

```python
        for a in self.agents:
            for step in a.script:
                if isinstance(step, Parallel):
                    for inner in step.steps:
                        if not isinstance(inner, (Call, MalformedCall, UnknownToolCall)):
                            raise ValueError(
                                f"Parallel step in agent {a.name!r} may contain only "
                                f"Call/MalformedCall/UnknownToolCall, got "
                                f"{type(inner).__name__}"
                            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_model.py -v`
Expected: PASS — `ParallelStepTests` plus all pre-existing model tests.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario.py tests/test_scenario_model.py
git commit -m "feat(scenario): Parallel fan-out step + validation"
```

---

## Task 2: New `Call` intent tags (`redundant`, `wrong_value`, `type_mismatch`)

Three booleans on `Call` (default `False`), mirroring the existing `irrelevant` tag. They record authorial intent and drive test assertions; they do **not** change how the call is serialized.

**Files:**
- Modify: `simple_chatbot/scenario.py` (the `Call` dataclass, lines 66-73)
- Test: `tests/test_scenario_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scenario_model.py`:

```python
class CallTagTests(unittest.TestCase):
    def _tool(self):
        @tool
        def t(x: str) -> dict:
            """t."""
            return {"x": x}
        return t

    def test_tags_default_false(self):
        c = Call(self._tool(), {"x": "1"})
        self.assertFalse(c.redundant)
        self.assertFalse(c.wrong_value)
        self.assertFalse(c.type_mismatch)

    def test_tags_settable(self):
        c = Call(self._tool(), {"x": "1"}, redundant=True, wrong_value=True, type_mismatch=True)
        self.assertTrue(c.redundant)
        self.assertTrue(c.wrong_value)
        self.assertTrue(c.type_mismatch)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_model.py::CallTagTests -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'redundant'`.

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/scenario.py`, extend the `Call` dataclass (currently lines 66-73) by adding three fields after `irrelevant`:

```python
@dataclass
class Call:
    """A well-formed tool call. Empty `args` (omitting a required field) is the
    deterministic required-arg-violation knob."""

    tool: ScenarioTool
    args: dict = field(default_factory=dict)
    irrelevant: bool = False  # tag for relevance-dimension test assertions
    redundant: bool = False  # tag: duplicate/unnecessary call → call-necessity dim
    wrong_value: bool = False  # tag: schema-valid call, semantically wrong arg value
    type_mismatch: bool = False  # tag: arg value violates the schema's declared type
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_model.py -v`
Expected: PASS — `CallTagTests` plus all earlier model tests.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario.py tests/test_scenario_model.py
git commit -m "feat(scenario): redundant/wrong_value/type_mismatch Call tags"
```

---

## Task 3: `DeterministicProvider` plays a `Parallel` step

Teach the deterministic provider to map a `Parallel` step to a `ProviderDecision` with **multiple** `PlannedCall`s, reusing one per-step mapping helper. Single-call steps still yield a one-element list (no regression). The orchestrator already serializes multi-call rounds.

**Files:**
- Modify: `simple_chatbot/scenario_provider.py` (the scenario import block lines 16-24; `DeterministicProvider.decide` lines 54-79)
- Test: `tests/test_scenario_provider.py`

- [ ] **Step 1: Write the failing test**

Add `Parallel` to the existing scenario import in `tests/test_scenario_provider.py`:

```python
from simple_chatbot.scenario import (
    ROUTE_TOOL_NAME,
    Agent,
    Call,
    Final,
    MalformedCall,
    Parallel,
    Route,
    UnknownToolCall,
    tool,
)
```

Append (the module already defines `lookup_invoice` and the `_decide` helper):

```python
class ParallelDecisionTests(unittest.TestCase):
    def test_parallel_yields_multiple_calls_in_order(self):
        agent = Agent(
            "a",
            tools=[lookup_invoice],
            script=[
                Parallel([
                    Call(lookup_invoice, {"invoice_id": "INV-1"}),
                    Call(lookup_invoice, {"invoice_id": "INV-2"}),
                ]),
            ],
        )
        d = _decide(DeterministicProvider(), agent)
        self.assertIsNone(d.final)
        self.assertEqual([c.name for c in d.calls], ["lookup_invoice", "lookup_invoice"])
        self.assertEqual(json.loads(d.calls[0].arguments), {"invoice_id": "INV-1"})
        self.assertEqual(json.loads(d.calls[1].arguments), {"invoice_id": "INV-2"})

    def test_parallel_mixes_valid_and_malformed(self):
        agent = Agent(
            "a",
            tools=[lookup_invoice],
            script=[Parallel([Call(lookup_invoice, {"invoice_id": "INV-1"}), MalformedCall(lookup_invoice)])],
        )
        d = _decide(DeterministicProvider(), agent)
        self.assertEqual(len(d.calls), 2)
        self.assertEqual(d.calls[1].arguments, "{intentionally_malformed_json")

    def test_single_call_still_one_element(self):
        agent = Agent("a", tools=[lookup_invoice], script=[Call(lookup_invoice, {"invoice_id": "INV-1"})])
        d = _decide(DeterministicProvider(), agent)
        self.assertEqual(len(d.calls), 1)
        self.assertEqual(d.calls[0].name, "lookup_invoice")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_provider.py::ParallelDecisionTests -v`
Expected: FAIL — `Parallel` import error, then (once imported) an exhausted-script `final` fallback instead of multi-call decision.

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/scenario_provider.py`, add `Parallel` to the scenario import block (lines 16-24):

```python
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
```

Add a module-level helper just below the `_MALFORMED_ARGS` constant (after line 28):

```python
def _planned_call(step) -> "PlannedCall":
    """Map a single call-family step to a PlannedCall."""
    if isinstance(step, Call):
        return PlannedCall(step.tool.name, json.dumps(step.args))
    if isinstance(step, MalformedCall):
        return PlannedCall(step.tool.name, _MALFORMED_ARGS)
    if isinstance(step, UnknownToolCall):
        return PlannedCall(step.name, json.dumps(step.args))
    raise ValueError(f"not a call-family step: {type(step).__name__}")
```

Replace the `Call`/`MalformedCall`/`UnknownToolCall` branches in `DeterministicProvider.decide` (lines 69-78) with a combined call-family branch plus a `Parallel` branch. The full body of `decide` after the cursor handling (keep the `Final` and `Route` branches as they are at lines 63-68) becomes:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_provider.py -v`
Expected: PASS — `ParallelDecisionTests` plus the pre-existing `DeterministicProviderTests` and `LiveProviderTests` (the `Final`/`Route`/single-call behavior is unchanged).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario_provider.py tests/test_scenario_provider.py
git commit -m "feat(scenario): DeterministicProvider plays Parallel as a multi-call round"
```

---

## Task 4: `parallel` fixture

The canonical fan-out fixture: one agent issues two read calls in a single `Parallel` round, then a synthesizing `Final`.

**Files:**
- Create: `simple_chatbot/fixtures/parallel.py`
- Test: `tests/test_scenario_fixtures.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scenario_fixtures.py`:

```python
class ParallelFixtureTests(unittest.TestCase):
    def _run(self):
        from simple_chatbot.fixtures.parallel import scenario
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat([{"role": "user", "content": "Lisbon this weekend?"}]))

    def test_fans_out_two_calls_in_one_round(self):
        result = self._run()
        assistant_calls = [
            m["tool_calls"] for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        # exactly one assistant message carrying both calls (a single fan-out round)
        self.assertEqual(len(assistant_calls), 1)
        names = [tc["function"]["name"] for tc in assistant_calls[0]]
        self.assertEqual(names, ["get_weather", "get_events"])
        # deterministic per-call ids within the round
        ids = [tc["id"] for tc in assistant_calls[0]]
        self.assertEqual(ids, ["call_parallel_1_1", "call_parallel_1_2"])
        self.assertTrue(result.content)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_fixtures.py::ParallelFixtureTests -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simple_chatbot.fixtures.parallel'`.

- [ ] **Step 3: Write minimal implementation**

Create `simple_chatbot/fixtures/parallel.py`:

```python
# simple_chatbot/fixtures/parallel.py
"""An agent that fans out parallel tool calls in a single round."""

from simple_chatbot.scenario import Agent, Call, Final, Parallel, Scenario, tool


@tool
def get_weather(city: str) -> dict:
    """Get the weather forecast for a city."""
    return {"city": city, "forecast": "sunny"}


@tool
def get_events(city: str) -> dict:
    """List notable events in a city."""
    return {"city": city, "events": ["jazz festival"]}


scenario = Scenario(
    id="parallel",
    description="An agent that gathers facts via parallel tool calls in one round.",
    entry="concierge",
    agents=[
        Agent(
            "concierge",
            system_prompt="Answer travel questions; gather facts in parallel.",
            tools=[get_weather, get_events],
            script=[
                Parallel([
                    Call(get_weather, {"city": "Lisbon"}),
                    Call(get_events, {"city": "Lisbon"}),
                ]),
                Final("Lisbon will be sunny, and there's a jazz festival this weekend."),
            ],
        ),
    ],
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_fixtures.py::ParallelFixtureTests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/fixtures/parallel.py tests/test_scenario_fixtures.py
git commit -m "feat(fixtures): parallel fan-out scenario"
```

---

## Task 5: `multi-round-routing` fixture

Two `Route` hops in a single user turn: front desk → tier-1 → tier-2, ending with the deepest agent's tool call + `Final`.

**Files:**
- Create: `simple_chatbot/fixtures/multi_round_routing.py`
- Test: `tests/test_scenario_fixtures.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scenario_fixtures.py`:

```python
class MultiRoundRoutingFixtureTests(unittest.TestCase):
    def _run(self):
        from simple_chatbot.fixtures.multi_round_routing import scenario
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat([{"role": "user", "content": "my ticket is stuck"}]))

    def test_two_route_hops_then_resolves(self):
        result = self._run()
        names = [
            m["tool_calls"][0]["function"]["name"]
            for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        self.assertEqual(names, ["route", "route", "check_status"])
        self.assertEqual(result.content, "Resolved at tier 2.")
        self.assertEqual(result.active_agent, "tier2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_fixtures.py::MultiRoundRoutingFixtureTests -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `simple_chatbot/fixtures/multi_round_routing.py`:

```python
# simple_chatbot/fixtures/multi_round_routing.py
"""Front desk routes to tier-1, which routes to tier-2 — two hops, one turn."""

from simple_chatbot.scenario import Agent, Call, Final, Route, Scenario, tool


@tool
def check_status(ticket_id: str) -> dict:
    """Check the status of a support ticket."""
    return {"ticket_id": ticket_id, "status": "escalated"}


scenario = Scenario(
    id="multi-round-routing",
    description="Two-hop handoff (front desk -> tier1 -> tier2) within one user turn.",
    entry="front_desk",
    agents=[
        Agent(
            "front_desk",
            system_prompt="Triage requests and route to first-line support.",
            routes=["tier1"],
            script=[Route("tier1")],
        ),
        Agent(
            "tier1",
            system_prompt="First-line support; escalate hard cases to a specialist.",
            routes=["tier2"],
            script=[Route("tier2")],
        ),
        Agent(
            "tier2",
            system_prompt="Specialist support.",
            tools=[check_status],
            script=[
                Call(check_status, {"ticket_id": "T-9"}),
                Final("Resolved at tier 2."),
            ],
        ),
    ],
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_fixtures.py::MultiRoundRoutingFixtureTests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/fixtures/multi_round_routing.py tests/test_scenario_fixtures.py
git commit -m "feat(fixtures): multi-round-routing two-hop handoff scenario"
```

---

## Task 6: `rag-ignore` fixture

Retrieval-as-tool: a `search_docs` stub returns an authoritative passage, then the agent's `Final(ignore_retrieval=True)` contradicts it.

**Files:**
- Create: `simple_chatbot/fixtures/rag_ignore.py`
- Test: `tests/test_scenario_fixtures.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scenario_fixtures.py`:

```python
class RagIgnoreFixtureTests(unittest.TestCase):
    def _run(self):
        from simple_chatbot.fixtures.rag_ignore import scenario
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat([{"role": "user", "content": "what's the refund window?"}]))

    def test_retrieves_then_ignores(self):
        result = self._run()
        # the retrieved passage is present in a tool-output item in the window
        tool_outputs = [m["content"] for m in result.tool_messages if m["role"] == "tool"]
        self.assertTrue(any("within 14 days" in out for out in tool_outputs))
        # the final answer contradicts the retrieved passage
        self.assertEqual(result.content, "Refunds are available at any time, with no deadline.")
        # the authored Final carries the ignore_retrieval tag
        from simple_chatbot.fixtures.rag_ignore import scenario
        from simple_chatbot.scenario import Final
        finals = [s for s in scenario.agent("support").script if isinstance(s, Final)]
        self.assertTrue(finals[0].ignore_retrieval)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_fixtures.py::RagIgnoreFixtureTests -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `simple_chatbot/fixtures/rag_ignore.py`:

```python
# simple_chatbot/fixtures/rag_ignore.py
"""Agent retrieves an authoritative passage, then ignores it in its answer."""

from simple_chatbot.scenario import Agent, Call, Final, Scenario, tool


@tool
def search_docs(query: str) -> dict:
    """Search the knowledge base and return matching passages."""
    return {
        "query": query,
        "passages": ["Refunds are allowed within 14 days of purchase."],
    }


scenario = Scenario(
    id="rag-ignore",
    description="Retrieval-as-tool: agent retrieves a passage then contradicts it.",
    entry="support",
    agents=[
        Agent(
            "support",
            system_prompt="Answer using the knowledge base.",
            tools=[search_docs],
            script=[
                Call(search_docs, {"query": "refund window"}),
                Final(
                    "Refunds are available at any time, with no deadline.",
                    ignore_retrieval=True,
                ),
            ],
        ),
    ],
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_fixtures.py::RagIgnoreFixtureTests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/fixtures/rag_ignore.py tests/test_scenario_fixtures.py
git commit -m "feat(fixtures): rag-ignore retrieval-then-contradict scenario"
```

---

## Task 7: `escalation` fixture

A dispatcher routes straight to a `terminal` human agent, which emits its `escalation_message` as the turn's final.

**Files:**
- Create: `simple_chatbot/fixtures/escalation.py`
- Test: `tests/test_scenario_fixtures.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scenario_fixtures.py`:

```python
class EscalationFixtureTests(unittest.TestCase):
    def _run(self):
        from simple_chatbot.fixtures.escalation import scenario
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat([{"role": "user", "content": "I need a human"}]))

    def test_routes_straight_to_human_terminal(self):
        result = self._run()
        names = [
            m["tool_calls"][0]["function"]["name"]
            for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        self.assertEqual(names, ["route"])  # single hop, then terminal
        self.assertEqual(result.content, "A human agent will take over from here.")
        self.assertEqual(result.active_agent, "human")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_fixtures.py::EscalationFixtureTests -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `simple_chatbot/fixtures/escalation.py`:

```python
# simple_chatbot/fixtures/escalation.py
"""Dispatcher escalates straight to a human terminal agent."""

from simple_chatbot.scenario import Agent, Route, Scenario


scenario = Scenario(
    id="escalation",
    description="Dispatcher routes directly to a human terminal agent.",
    entry="dispatcher",
    agents=[
        Agent(
            "dispatcher",
            system_prompt="Escalate anything you cannot handle to a human.",
            routes=["human"],
            script=[Route("human")],
        ),
        Agent(
            "human",
            terminal=True,
            escalation_message="A human agent will take over from here.",
        ),
    ],
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_fixtures.py::EscalationFixtureTests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/fixtures/escalation.py tests/test_scenario_fixtures.py
git commit -m "feat(fixtures): escalation route-to-human-terminal scenario"
```

---

## Task 8: `relevance-probe` fixture

One agent fires an `irrelevant` call, a `wrong_value` call, and a `redundant` call for a billing question, then a `Final`. Exercises the relevance + call-necessity dimensions.

**Files:**
- Create: `simple_chatbot/fixtures/relevance_probe.py`
- Test: `tests/test_scenario_fixtures.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scenario_fixtures.py` (add `from simple_chatbot.scenario import Call` to the file's imports first):

```python
class RelevanceProbeFixtureTests(unittest.TestCase):
    def _run(self):
        from simple_chatbot.fixtures.relevance_probe import scenario
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat([{"role": "user", "content": "what's my balance?"}]))

    def test_emits_irrelevant_wrong_value_and_redundant(self):
        from simple_chatbot.fixtures.relevance_probe import scenario
        result = self._run()
        names = [
            m["tool_calls"][0]["function"]["name"]
            for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        self.assertEqual(names, ["get_weather", "lookup_invoice", "lookup_invoice"])
        # tags recorded on the authored script (in order)
        call_steps = [s for s in scenario.agent("agent").script if isinstance(s, Call)]
        self.assertTrue(call_steps[0].irrelevant)
        self.assertTrue(call_steps[1].wrong_value)
        self.assertTrue(call_steps[2].redundant)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_fixtures.py::RelevanceProbeFixtureTests -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `simple_chatbot/fixtures/relevance_probe.py`:

```python
# simple_chatbot/fixtures/relevance_probe.py
"""Fires irrelevant, wrong-value, and redundant calls for a billing question."""

from simple_chatbot.scenario import Agent, Call, Final, Scenario, tool


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice by id."""
    return {"invoice_id": invoice_id, "amount_due": "10.00"}


@tool
def get_weather(city: str) -> dict:
    """Get the weather forecast for a city."""
    return {"city": city, "forecast": "rainy"}


scenario = Scenario(
    id="relevance-probe",
    description="Deliberately poor tool choices: irrelevant, wrong-value, redundant.",
    entry="agent",
    agents=[
        Agent(
            "agent",
            system_prompt="A billing agent that deliberately makes poor tool choices.",
            tools=[lookup_invoice, get_weather],
            script=[
                Call(get_weather, {"city": "Paris"}, irrelevant=True),  # wrong tool for billing
                Call(lookup_invoice, {"invoice_id": "INV-999"}, wrong_value=True),  # wrong invoice
                Call(lookup_invoice, {"invoice_id": "INV-999"}, redundant=True),  # duplicate
                Final("Your balance is $10.00."),
            ],
        ),
    ],
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_fixtures.py::RelevanceProbeFixtureTests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/fixtures/relevance_probe.py tests/test_scenario_fixtures.py
git commit -m "feat(fixtures): relevance-probe (irrelevant/wrong-value/redundant)"
```

---

## Task 9: Extend `validity-probe` with the `type_mismatch` knob

Add a numeric-param tool and a `type_mismatch` step to the existing `validity-probe` fixture so it showcases all validity classes (unknown / malformed / required / type).

**Files:**
- Modify: `simple_chatbot/fixtures/validity_probe.py`
- Test: `tests/test_scenario_fixtures.py` (the existing `ValidityProbeFixtureTests`)

- [ ] **Step 1: Write the failing test**

Extend the existing `test_emits_unknown_malformed_and_required_violation` in `ValidityProbeFixtureTests` by appending these assertions to its body (after the existing `self.assertIn("{}", args_by_name.values())` line):

```python
        # type-mismatch knob: schema wants int, author supplies a string
        self.assertIn("set_credit_limit", args_by_name)
        self.assertEqual(json.loads(args_by_name["set_credit_limit"]), {"limit": "a lot"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_fixtures.py::ValidityProbeFixtureTests -v`
Expected: FAIL — `set_credit_limit` is not among the emitted calls.

- [ ] **Step 3: Write minimal implementation**

Edit `simple_chatbot/fixtures/validity_probe.py`. Add a numeric-param tool after `get_order`:

```python
@tool
def set_credit_limit(limit: int) -> dict:
    """Set a customer's credit limit."""
    return {"limit": limit}
```

Add `set_credit_limit` to the prober's `tools` list and insert a `type_mismatch` `Call` before the closing `Final` in its `script`:

```python
        Agent(
            "prober",
            system_prompt="A probe agent that deliberately misuses tools.",
            tools=[lookup_invoice, get_order, set_credit_limit],
            script=[
                UnknownToolCall("ghost_tool"),
                MalformedCall(lookup_invoice),
                Call(get_order),  # omits required order_id → required violation
                Call(set_credit_limit, {"limit": "a lot"}, type_mismatch=True),  # string ≠ int
                Final("Probe complete."),
            ],
        ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_fixtures.py::ValidityProbeFixtureTests -v`
Expected: PASS — the existing unknown/malformed/required assertions plus the new type-mismatch ones.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/fixtures/validity_probe.py tests/test_scenario_fixtures.py
git commit -m "feat(fixtures): add type_mismatch knob to validity-probe"
```

---

## Task 10: Golden-trace snapshots

Lock the full ordered output array (via `build_output_items`) for the new canonical fixtures `parallel` and `rag-ignore`, and backfill `cs-routing`. A compact summarizer normalizes out the random envelope ids (`fc_`/`fco_`/`msg_`) and asserts on item type, `call_id`, name, arguments, tool-output contents, and final text — per the parent spec's reproducibility definition (stable `call_id`s + ordering + outputs, not byte-identical envelopes).

**Files:**
- Create: `tests/test_golden_traces.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_golden_traces.py`:

```python
# tests/test_golden_traces.py
import asyncio
import unittest

from simple_chatbot.responses import build_output_items
from simple_chatbot.scenario_orchestrator import ScenarioOrchestrator
from simple_chatbot.scenario_provider import DeterministicProvider


def _summarize(items: list[dict]) -> list[tuple]:
    """Compact, envelope-id-free projection of an output array.

    Keeps the fields the parent spec defines as the reproducibility contract:
    item type, deterministic call_id, tool name, raw arguments, tool output,
    and final message text. Drops random `fc_`/`fco_`/`msg_`/`rs_` ids.
    """
    out: list[tuple] = []
    for it in items:
        t = it["type"]
        if t == "function_call":
            out.append(("call", it["call_id"], it["name"], it["arguments"]))
        elif t == "function_call_output":
            out.append(("output", it["call_id"], it["output"]))
        elif t == "message":
            out.append(("message", it["content"][0]["text"]))
        elif t == "reasoning":
            out.append(("reasoning",))
        else:
            out.append((t,))
    return out


def _trace(module_name: str, user_text: str) -> list[tuple]:
    import importlib

    scenario = importlib.import_module(module_name).scenario
    orch = ScenarioOrchestrator(scenario, DeterministicProvider())
    result = asyncio.run(orch.chat([{"role": "user", "content": user_text}]))
    return _summarize(build_output_items(result))


class GoldenTraceTests(unittest.TestCase):
    def test_parallel_golden(self):
        self.assertEqual(
            _trace("simple_chatbot.fixtures.parallel", "Lisbon this weekend?"),
            [
                ("call", "call_parallel_1_1", "get_weather", '{"city": "Lisbon"}'),
                ("call", "call_parallel_1_2", "get_events", '{"city": "Lisbon"}'),
                ("output", "call_parallel_1_1", '{"city": "Lisbon", "forecast": "sunny"}'),
                ("output", "call_parallel_1_2", '{"city": "Lisbon", "events": ["jazz festival"]}'),
                ("message", "Lisbon will be sunny, and there's a jazz festival this weekend."),
            ],
        )

    def test_rag_ignore_golden(self):
        self.assertEqual(
            _trace("simple_chatbot.fixtures.rag_ignore", "what's the refund window?"),
            [
                ("call", "call_rag-ignore_1_1", "search_docs", '{"query": "refund window"}'),
                (
                    "output",
                    "call_rag-ignore_1_1",
                    '{"query": "refund window", "passages": ["Refunds are allowed within 14 days of purchase."]}',
                ),
                ("message", "Refunds are available at any time, with no deadline."),
            ],
        )

    def test_cs_routing_golden(self):
        self.assertEqual(
            _trace("simple_chatbot.fixtures.cs_routing", "I want a refund"),
            [
                ("call", "call_cs-routing_1_1", "route", '{"agent": "billing"}'),
                ("output", "call_cs-routing_1_1", '{"routed": true, "agent": "billing"}'),
                ("call", "call_cs-routing_1_2", "lookup_invoice", '{"invoice_id": "INV-1"}'),
                (
                    "output",
                    "call_cs-routing_1_2",
                    '{"invoice_id": "INV-1", "amount_due": "42.00", "status": "open"}',
                ),
                ("call", "call_cs-routing_1_3", "issue_refund", '{"invoice_id": "INV-1", "amount": 42.0}'),
                (
                    "output",
                    "call_cs-routing_1_3",
                    '{"refunded": true, "invoice_id": "INV-1", "amount": 42.0}',
                ),
                ("message", "I've refunded $42.00 against invoice INV-1."),
            ],
        )
```

- [ ] **Step 2: Run test to verify it fails (or passes immediately)**

Run: `python -m pytest tests/test_golden_traces.py -v`
Expected: PASS — this test is a snapshot of behavior built in Tasks 4, 6, and the pre-existing `cs-routing`. If any tuple mismatches, the failure output shows the exact divergence; reconcile the fixture (Tasks 4/6) and this snapshot until they agree. (Authoring the test after the fixtures means it may pass on first run; that is acceptable for a pure snapshot — the value is regression-locking, and Step 2 confirms the snapshot matches the real serialized trace.)

- [ ] **Step 3: (only if Step 2 failed) reconcile**

If Step 2 failed, the mismatch is between the expected tuples here and the actual `build_output_items` output. Read the diff, confirm the fixture script in Task 4/6 is what you intend, and update the expected list here to the real serialized values (deterministic `call_id`s and `json.dumps` arg/output strings). Do not weaken `_summarize` to hide a real divergence.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_golden_traces.py -v`
Expected: PASS — all three golden traces.

- [ ] **Step 5: Commit**

```bash
git add tests/test_golden_traces.py
git commit -m "test(scenario): golden traces for parallel, rag-ignore, cs-routing"
```

---

## Task 11: Full-suite regression gate

Confirm the whole suite is green — the pre-existing 228 tests plus all Phase 3 additions, with the new fixtures auto-discovered by the registry and not breaking any `/v1/models` or registry assertions (all use `assertIn`).

**Files:** none (verification only).

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS — no failures or errors. If anything fails, fix the offending task before proceeding; this is the integration gate.

- [ ] **Step 2: Commit (only if a fix was needed)**

```bash
git add -A
git commit -m "test(scenario): Phase 3 knob & fixture library full-suite green"
```

---

## Self-Review

**Spec coverage (Phase 3 design sections):**
- §"Component changes / New step `Parallel`" — `Parallel` dataclass + `Step` union + validation → Task 1; deterministic multi-call mapping → Task 3.
- §"New `Call` tags" — `redundant`/`wrong_value`/`type_mismatch` booleans → Task 2; exercised by `relevance-probe` (Task 8) and `validity-probe` type-mismatch (Task 9).
- §"Fixture library" table — `parallel` (Task 4), `multi-round-routing` (Task 5), `rag-ignore` (Task 6), `escalation` (Task 7), `relevance-probe` (Task 8), `validity-probe` extension (Task 9).
- §"RAG-ignore mechanism" (retrieval-as-tool stub) → Task 6.
- §"Testing / Golden traces" — full ordered-array snapshots for `parallel`, `rag-ignore`, `cs-routing` → Task 10; per-fixture behavioral tests → Tasks 4-9; unit tests for `Parallel`/tags/provider → Tasks 1-3; regression gate → Task 11.
- Out of scope (correctly absent): any `LiveProvider`/mode-switch/orchestrator/catalog change, live-mode knobs, live golden traces, ghost-backend, YAML adapter, output-value/trajectory knobs.

**Placeholder scan:** none — every code step shows complete, runnable code; every run step has an exact command and expected outcome. (Task 10 Step 2 explicitly handles the "passes immediately" case for a pure snapshot test.)

**Type / name consistency:** `Parallel(steps=[...])` (Task 1) is consumed by `_planned_call` iteration in Task 3 and authored in Task 4; the `Call` tags `redundant`/`wrong_value`/`type_mismatch` (Task 2) are read in Tasks 8-9 tests; `_planned_call(step)` (Task 3) reuses `PlannedCall(name, arguments)` and `_MALFORMED_ARGS` from Phase 1 unchanged; deterministic `call_id` format `call_<scenario>_<turn>_<seq>` (orchestrator, unchanged) is asserted literally in Tasks 4 and 10; fixture `scenario.id`s (`parallel`, `multi-round-routing`, `rag-ignore`, `escalation`, `relevance-probe`) match the golden-trace `call_id` prefixes and the registry keys; tool stub return values in Tasks 4/6 match the `json.dumps` output strings asserted in Task 10. The `DeterministicProvider` `Final`/`Route` branches are preserved verbatim in Task 3 (only the call-family branches are refactored), so no Phase 1/2 behavior changes.
