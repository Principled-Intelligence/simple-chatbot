# Scenario Fixture Engine — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the canonical deterministic fixture path in simple-chatbot: code-authored multi-agent scenarios (with routing + misbehavior knobs) that emit a faithful Responses API trace, with the tool catalog published in `Response.tools`.

**Architecture:** A new scenario layer sits beside the existing single-agent path. A `Scenario` (built from a small typed code API) is run by a `ScenarioOrchestrator` that maintains active-agent state and a per-round scoped toolset; a `DeterministicProvider` plays each agent's authored `script`. The whole multi-agent trajectory collapses into one `ChatResult.tool_messages` list, which the existing wire layer serializes into one `/v1/responses` output array (= one Spectral turn). The scenario's tool catalog is projected into the `Response.tools` echo field. Scenarios are selected by the request's `model` string from a fixtures package.

**Tech Stack:** Python 3.12, pydantic v2 (schema generation), FastAPI (existing server), `unittest` + `asyncio.run` (existing test style), litellm (existing; not used in this phase's deterministic path).

**Scope note:** This plan is simple-chatbot only. The cross-repo ghost-backend `Response.tools` harvest (spec Phase 4) is explicitly out of scope and **must not be started until the author authorizes it**. Until then, a fixture's catalog can be registered via the Spectral Target's `declared_tools`, so everything here is fully verifiable standalone.

---

## File Structure

**Create:**
- `simple_chatbot/scenario.py` — the in-memory model + builder API: `ScenarioTool`, `@tool`, step types (`Call`, `Route`, `Final`, `MalformedCall`, `UnknownToolCall`), `Agent`, `Scenario`, plus `ROUTE_TOOL_NAME`.
- `simple_chatbot/scenario_catalog.py` — `build_responses_tools(scenario)` projecting tools into the flat `Response.tools` shape.
- `simple_chatbot/scenario_provider.py` — `PlannedCall`, `ProviderDecision`, `DeterministicProvider`.
- `simple_chatbot/scenario_orchestrator.py` — `ScenarioOrchestrator.chat()`.
- `simple_chatbot/scenario_registry.py` — `load_fixtures()`.
- `simple_chatbot/fixtures/__init__.py` — empty package marker.
- `simple_chatbot/fixtures/cs_routing.py` — canonical routing fixture.
- `simple_chatbot/fixtures/validity_probe.py` — misbehavior-knob fixture.
- Tests: `tests/test_scenario_model.py`, `tests/test_scenario_catalog.py`, `tests/test_scenario_provider.py`, `tests/test_scenario_orchestrator.py`, `tests/test_scenario_fixtures.py`, `tests/test_scenario_registry.py`, `tests/test_scenario_endpoint.py`.

**Modify:**
- `simple_chatbot/agent.py` — add `responses_tools: list[dict]` field to `ChatResult`.
- `simple_chatbot/responses.py` — `build_response` sets `payload["tools"]` from `result.responses_tools`.
- `simple_chatbot/server.py` — `_scenario_registry` global, scenario selection in `/v1/responses`, fixture ids in `/v1/models`.

---

## Task 1: `ScenarioTool` + `@tool` decorator

Defines a tool as a typed Python function: JSON Schema derived from the signature, description from the docstring, the body used as the deterministic stub.

**Files:**
- Create: `simple_chatbot/scenario.py`
- Test: `tests/test_scenario_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenario_model.py
import unittest

from simple_chatbot.scenario import tool, ScenarioTool


class ToolDecoratorTests(unittest.TestCase):
    def test_tool_builds_schema_from_signature(self):
        @tool
        def lookup_invoice(invoice_id: str) -> dict:
            """Look up an invoice by id."""
            return {"invoice_id": invoice_id, "status": "open"}

        self.assertIsInstance(lookup_invoice, ScenarioTool)
        self.assertEqual(lookup_invoice.name, "lookup_invoice")
        self.assertEqual(lookup_invoice.description, "Look up an invoice by id.")
        self.assertEqual(lookup_invoice.parameters["type"], "object")
        self.assertIn("invoice_id", lookup_invoice.parameters["properties"])
        self.assertEqual(
            lookup_invoice.parameters["properties"]["invoice_id"]["type"], "string"
        )
        self.assertEqual(lookup_invoice.parameters["required"], ["invoice_id"])

    def test_optional_param_not_required(self):
        @tool
        def get_time(tz: str = "UTC") -> dict:
            """Return the current time."""
            return {"tz": tz}

        self.assertNotIn("required", get_time.parameters)
        self.assertEqual(get_time.parameters["properties"]["tz"]["type"], "string")

    def test_stub_body_is_callable(self):
        @tool
        def echo(value: str) -> dict:
            """Echo."""
            return {"echoed": value}

        self.assertEqual(echo.func(value="hi"), {"echoed": "hi"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_model.py::ToolDecoratorTests -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simple_chatbot.scenario'`.

- [ ] **Step 3: Write minimal implementation**

```python
# simple_chatbot/scenario.py
"""In-memory model + code-first builder API for fixture scenarios.

A scenario declares agents, each with a scoped toolset, optional routing
targets, and a deterministic `script`. Tools are defined as typed Python
functions via `@tool`: the signature yields the JSON Schema, the docstring the
description, and the body is the deterministic stub.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

from pydantic import create_model

ROUTE_TOOL_NAME = "route"


@dataclass
class ScenarioTool:
    """A tool: JSON-Schema parameters + an async-free stub callable."""

    name: str
    description: str
    parameters: dict
    func: Callable[..., dict | str]


def tool(func: Callable[..., dict | str]) -> ScenarioTool:
    """Decorator turning a typed function into a `ScenarioTool`.

    The parameter JSON Schema is generated from the signature via pydantic;
    `title` keys pydantic adds are stripped to keep the schema close to what a
    hand-written OpenAI function schema looks like.
    """
    sig = inspect.signature(func)
    fields: dict = {}
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        annotation = p.annotation if p.annotation is not inspect.Parameter.empty else str
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[pname] = (annotation, default)

    model = create_model(f"{func.__name__}_Args", **fields)
    schema = model.model_json_schema()

    properties = {}
    for key, prop in (schema.get("properties") or {}).items():
        properties[key] = {k: v for k, v in prop.items() if k != "title"}

    parameters: dict = {"type": "object", "properties": properties}
    if schema.get("required"):
        parameters["required"] = schema["required"]

    return ScenarioTool(
        name=func.__name__,
        description=(func.__doc__ or "").strip(),
        parameters=parameters,
        func=func,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_model.py::ToolDecoratorTests -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario.py tests/test_scenario_model.py
git commit -m "feat(scenario): @tool decorator generating schema from signature"
```

---

## Task 2: Step types

The authored-trajectory step constructors, including the misbehavior knobs.

**Files:**
- Modify: `simple_chatbot/scenario.py`
- Test: `tests/test_scenario_model.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_scenario_model.py
from simple_chatbot.scenario import Call, Route, Final, MalformedCall, UnknownToolCall, tool


class StepTypeTests(unittest.TestCase):
    def _t(self):
        @tool
        def lookup_invoice(invoice_id: str) -> dict:
            """Look up an invoice."""
            return {"invoice_id": invoice_id}
        return lookup_invoice

    def test_call_holds_tool_and_args(self):
        t = self._t()
        c = Call(t, {"invoice_id": "INV-1"})
        self.assertIs(c.tool, t)
        self.assertEqual(c.args, {"invoice_id": "INV-1"})
        self.assertFalse(c.irrelevant)

    def test_call_defaults_to_empty_args(self):
        c = Call(self._t())
        self.assertEqual(c.args, {})

    def test_route_holds_target(self):
        self.assertEqual(Route("billing").target, "billing")

    def test_final_flags(self):
        f = Final("done", ignore_retrieval=True)
        self.assertEqual(f.text, "done")
        self.assertTrue(f.ignore_retrieval)

    def test_malformed_and_unknown(self):
        t = self._t()
        self.assertIs(MalformedCall(t).tool, t)
        self.assertEqual(UnknownToolCall("ghost_tool").name, "ghost_tool")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_model.py::StepTypeTests -v`
Expected: FAIL — `ImportError: cannot import name 'Call'`.

- [ ] **Step 3: Write minimal implementation**

Append to `simple_chatbot/scenario.py`:

```python
@dataclass
class Call:
    """A well-formed tool call. Empty `args` (omitting a required field) is the
    deterministic required-arg-violation knob."""

    tool: ScenarioTool
    args: dict = field(default_factory=dict)
    irrelevant: bool = False  # tag for relevance-dimension test assertions


@dataclass
class Route:
    """A handoff to another agent (serialized as a `route` tool call)."""

    target: str


@dataclass
class Final:
    """End the turn with this assistant message."""

    text: str
    ignore_retrieval: bool = False  # tag for utilization/faithfulness assertions


@dataclass
class MalformedCall:
    """Emit a call whose arguments are invalid JSON (validity knob → score 2)."""

    tool: ScenarioTool


@dataclass
class UnknownToolCall:
    """Emit a call to a tool absent from the catalog (validity knob → score 1)."""

    name: str
    args: dict = field(default_factory=dict)


Step = Call | Route | Final | MalformedCall | UnknownToolCall
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_model.py::StepTypeTests -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario.py tests/test_scenario_model.py
git commit -m "feat(scenario): step types incl. misbehavior knobs"
```

---

## Task 3: `Agent` + `Scenario` model with validation

**Files:**
- Modify: `simple_chatbot/scenario.py`
- Test: `tests/test_scenario_model.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_scenario_model.py
from simple_chatbot.scenario import Agent, Scenario, tool


class ScenarioModelTests(unittest.TestCase):
    def _tools(self):
        @tool
        def lookup_invoice(invoice_id: str) -> dict:
            """Look up an invoice."""
            return {"invoice_id": invoice_id}

        @tool
        def get_status(order_id: str) -> dict:
            """Get order status."""
            return {"order_id": order_id, "status": "shipped"}

        return lookup_invoice, get_status

    def _scenario(self):
        inv, status = self._tools()
        return Scenario(
            id="demo",
            entry="dispatcher",
            agents=[
                Agent("dispatcher", routes=["billing", "orders"]),
                Agent("billing", tools=[inv]),
                Agent("orders", tools=[status]),
            ],
        )

    def test_agent_lookup(self):
        s = self._scenario()
        self.assertEqual(s.agent("billing").name, "billing")

    def test_all_tools_is_union_deduped_by_name(self):
        s = self._scenario()
        names = sorted(t.name for t in s.all_tools())
        self.assertEqual(names, ["get_status", "lookup_invoice"])

    def test_route_targets_sorted_unique(self):
        s = self._scenario()
        self.assertEqual(s.route_targets(), ["billing", "orders"])

    def test_validate_rejects_missing_entry(self):
        inv, _ = self._tools()
        with self.assertRaises(ValueError):
            Scenario(id="x", entry="nope", agents=[Agent("a", tools=[inv])])

    def test_validate_rejects_route_to_unknown_agent(self):
        with self.assertRaises(ValueError):
            Scenario(id="x", entry="a", agents=[Agent("a", routes=["ghost"])])

    def test_validate_rejects_duplicate_agent_names(self):
        with self.assertRaises(ValueError):
            Scenario(id="x", entry="a", agents=[Agent("a"), Agent("a")])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_model.py::ScenarioModelTests -v`
Expected: FAIL — `ImportError: cannot import name 'Agent'`.

- [ ] **Step 3: Write minimal implementation**

Append to `simple_chatbot/scenario.py`:

```python
@dataclass
class Agent:
    name: str
    system_prompt: str = ""
    tools: list[ScenarioTool] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    script: list[Step] = field(default_factory=list)
    terminal: bool = False
    escalation_message: str = ""


@dataclass
class Scenario:
    id: str
    entry: str
    agents: list[Agent]
    description: str = ""

    def __post_init__(self) -> None:
        self.validate()

    def agent(self, name: str) -> Agent:
        for a in self.agents:
            if a.name == name:
                return a
        raise KeyError(f"no agent named {name!r}")

    def all_tools(self) -> list[ScenarioTool]:
        by_name: dict[str, ScenarioTool] = {}
        for a in self.agents:
            for t in a.tools:
                by_name.setdefault(t.name, t)
        return list(by_name.values())

    def route_targets(self) -> list[str]:
        targets: set[str] = set()
        for a in self.agents:
            targets.update(a.routes)
        return sorted(targets)

    def validate(self) -> None:
        names = [a.name for a in self.agents]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate agent names in scenario {self.id!r}")
        name_set = set(names)
        if self.entry not in name_set:
            raise ValueError(f"entry agent {self.entry!r} not found in scenario {self.id!r}")
        for a in self.agents:
            for target in a.routes:
                if target not in name_set:
                    raise ValueError(
                        f"agent {a.name!r} routes to unknown agent {target!r}"
                    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_model.py -v`
Expected: PASS (all classes).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario.py tests/test_scenario_model.py
git commit -m "feat(scenario): Agent + Scenario model with validation"
```

---

## Task 4: Catalog projection (`Response.tools` shape)

**Files:**
- Create: `simple_chatbot/scenario_catalog.py`
- Test: `tests/test_scenario_catalog.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenario_catalog.py
import unittest

from simple_chatbot.scenario import Agent, Scenario, tool
from simple_chatbot.scenario_catalog import build_responses_tools


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice."""
    return {"invoice_id": invoice_id}


class CatalogTests(unittest.TestCase):
    def _scenario(self):
        return Scenario(
            id="demo",
            entry="dispatcher",
            agents=[
                Agent("dispatcher", routes=["billing"]),
                Agent("billing", tools=[lookup_invoice]),
            ],
        )

    def test_function_tool_is_flat_shape(self):
        entries = build_responses_tools(self._scenario())
        inv = next(e for e in entries if e["name"] == "lookup_invoice")
        self.assertEqual(inv["type"], "function")
        self.assertEqual(inv["description"], "Look up an invoice.")
        self.assertEqual(inv["parameters"]["required"], ["invoice_id"])
        # FLAT shape (no nested "function" key like chat-completions)
        self.assertNotIn("function", inv)

    def test_route_tool_published_with_enum(self):
        entries = build_responses_tools(self._scenario())
        route = next(e for e in entries if e["name"] == "route")
        self.assertEqual(route["type"], "function")
        self.assertEqual(
            route["parameters"]["properties"]["agent"]["enum"], ["billing"]
        )
        self.assertEqual(route["parameters"]["required"], ["agent"])

    def test_no_route_tool_when_no_routes(self):
        s = Scenario(id="x", entry="a", agents=[Agent("a", tools=[lookup_invoice])])
        names = [e["name"] for e in build_responses_tools(s)]
        self.assertNotIn("route", names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_catalog.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simple_chatbot.scenario_catalog'`.

- [ ] **Step 3: Write minimal implementation**

```python
# simple_chatbot/scenario_catalog.py
"""Project a Scenario's tools into the flat Responses-API `Response.tools` shape.

NOTE the FLAT function-tool shape `{type, name, parameters, description}` — not
the Chat Completions nested `{type, function: {...}}` shape. The catalog is the
UNION of all agents' tools plus a single generated `route` tool whose enum is
every routing target in the scenario.
"""

from __future__ import annotations

from simple_chatbot.scenario import ROUTE_TOOL_NAME, Scenario


def build_responses_tools(scenario: Scenario) -> list[dict]:
    out: list[dict] = []
    for t in scenario.all_tools():
        entry: dict = {"type": "function", "name": t.name, "parameters": t.parameters}
        if t.description:
            entry["description"] = t.description
        out.append(entry)

    targets = scenario.route_targets()
    if targets:
        out.append(
            {
                "type": "function",
                "name": ROUTE_TOOL_NAME,
                "description": "Hand off the request to another agent.",
                "parameters": {
                    "type": "object",
                    "properties": {"agent": {"type": "string", "enum": targets}},
                    "required": ["agent"],
                },
            }
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_catalog.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario_catalog.py tests/test_scenario_catalog.py
git commit -m "feat(scenario): catalog projection into Response.tools shape"
```

---

## Task 5: `ChatResult.responses_tools` + `build_response` wiring

Thread the catalog through the existing wire layer so `Response.tools` is populated.

**Files:**
- Modify: `simple_chatbot/agent.py:63-72` (the `ChatResult` dataclass)
- Modify: `simple_chatbot/responses.py:170-205` (`build_response`)
- Test: `tests/test_scenario_catalog.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_scenario_catalog.py
from simple_chatbot.agent import ChatResult
from simple_chatbot.responses import build_response


class BuildResponseToolsTests(unittest.TestCase):
    def test_response_tools_populated_from_result(self):
        result = ChatResult(
            content="done",
            retrieved_chunks=[],
            responses_tools=[
                {"type": "function", "name": "route",
                 "parameters": {"type": "object", "properties": {}}}
            ],
        )
        payload = build_response(
            result=result,
            model="fixture:demo",
            previous_response_id=None,
            conversation_id="conv-1",
        )
        self.assertEqual(payload["tools"][0]["name"], "route")

    def test_response_tools_default_empty(self):
        result = ChatResult(content="done", retrieved_chunks=[])
        payload = build_response(
            result=result, model="m", previous_response_id=None, conversation_id="c"
        )
        self.assertEqual(payload["tools"], [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_catalog.py::BuildResponseToolsTests -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'responses_tools'`.

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/agent.py`, add the field to `ChatResult` (after `final_reasoning_content`):

```python
    final_reasoning_content: str | None = None
    responses_tools: list[dict] = field(default_factory=list)
```

In `simple_chatbot/responses.py`, inside `build_response`, after the existing `payload["created_at"] = int(payload["created_at"])` line and before/around the `conversation_id` line, add:

```python
    payload["tools"] = result.responses_tools
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_catalog.py tests/test_responses_mapping.py tests/test_responses_endpoint.py -v`
Expected: PASS (new tests pass; existing responses tests still green — they assert `tools` is `[]` by default, which still holds).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/agent.py simple_chatbot/responses.py tests/test_scenario_catalog.py
git commit -m "feat(responses): populate Response.tools from ChatResult.responses_tools"
```

---

## Task 6: `DeterministicProvider`

Plays the active agent's `script` one step per round, tracking a per-agent cursor.

**Files:**
- Create: `simple_chatbot/scenario_provider.py`
- Test: `tests/test_scenario_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenario_provider.py
import asyncio
import json
import unittest

from simple_chatbot.scenario import Agent, Call, Final, MalformedCall, Route, UnknownToolCall, tool
from simple_chatbot.scenario_provider import DeterministicProvider


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice."""
    return {"invoice_id": invoice_id}


def _decide(provider, agent):
    return asyncio.run(provider.decide(agent, messages=[], tool_messages=[]))


class DeterministicProviderTests(unittest.TestCase):
    def test_plays_route_then_exhausts(self):
        agent = Agent("dispatcher", routes=["billing"], script=[Route("billing")])
        p = DeterministicProvider()
        d1 = _decide(p, agent)
        self.assertIsNone(d1.final)
        self.assertEqual(d1.calls[0].name, "route")
        self.assertEqual(json.loads(d1.calls[0].arguments), {"agent": "billing"})
        # exhausted → final fallback
        d2 = _decide(p, agent)
        self.assertIsNotNone(d2.final)

    def test_call_then_final(self):
        agent = Agent(
            "billing",
            tools=[lookup_invoice],
            script=[Call(lookup_invoice, {"invoice_id": "INV-1"}), Final("done")],
        )
        p = DeterministicProvider()
        d1 = _decide(p, agent)
        self.assertEqual(d1.calls[0].name, "lookup_invoice")
        self.assertEqual(json.loads(d1.calls[0].arguments), {"invoice_id": "INV-1"})
        d2 = _decide(p, agent)
        self.assertEqual(d2.final, "done")

    def test_malformed_call_emits_invalid_json(self):
        agent = Agent("a", tools=[lookup_invoice], script=[MalformedCall(lookup_invoice)])
        d = _decide(DeterministicProvider(), agent)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(d.calls[0].arguments)

    def test_unknown_tool_call(self):
        agent = Agent("a", script=[UnknownToolCall("ghost_tool")])
        d = _decide(DeterministicProvider(), agent)
        self.assertEqual(d.calls[0].name, "ghost_tool")

    def test_required_violation_emits_empty_args(self):
        agent = Agent("a", tools=[lookup_invoice], script=[Call(lookup_invoice)])
        d = _decide(DeterministicProvider(), agent)
        self.assertEqual(json.loads(d.calls[0].arguments), {})

    def test_cursor_is_per_agent(self):
        a = Agent("a", routes=["b"], script=[Route("b")])
        b = Agent("b", tools=[lookup_invoice], script=[Final("b-done")])
        p = DeterministicProvider()
        self.assertEqual(_decide(p, a).calls[0].name, "route")
        self.assertEqual(_decide(p, b).final, "b-done")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simple_chatbot.scenario_provider'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_provider.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario_provider.py tests/test_scenario_provider.py
git commit -m "feat(scenario): DeterministicProvider playing agent scripts"
```

---

## Task 7: `ScenarioOrchestrator`

The round loop: active-agent state, scoped execution, routing, stable `call_id`s, knob handling. Produces a `ChatResult` whose `tool_messages` serialize into one output array.

**Files:**
- Create: `simple_chatbot/scenario_orchestrator.py`
- Test: `tests/test_scenario_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenario_orchestrator.py
import asyncio
import json
import unittest

from simple_chatbot.scenario import Agent, Call, Final, MalformedCall, Route, Scenario, UnknownToolCall, tool
from simple_chatbot.scenario_provider import DeterministicProvider
from simple_chatbot.scenario_orchestrator import ScenarioOrchestrator


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice."""
    return {"invoice_id": invoice_id, "amount_due": "42.00"}


def _run(scenario, messages):
    orch = ScenarioOrchestrator(scenario, DeterministicProvider())
    return asyncio.run(orch.chat(messages))


class OrchestratorRoutingTests(unittest.TestCase):
    def _scenario(self):
        return Scenario(
            id="cs",
            entry="dispatcher",
            agents=[
                Agent("dispatcher", routes=["billing"], script=[Route("billing")]),
                Agent(
                    "billing",
                    tools=[lookup_invoice],
                    script=[Call(lookup_invoice, {"invoice_id": "INV-1"}), Final("Refunded.")],
                ),
            ],
        )

    def test_one_turn_collapses_route_and_subagent_calls(self):
        result = _run(self._scenario(), [{"role": "user", "content": "refund please"}])
        self.assertEqual(result.content, "Refunded.")
        roles_and_names = [
            (m["role"], m.get("name") or (m["tool_calls"][0]["function"]["name"] if m.get("tool_calls") else None))
            for m in result.tool_messages
        ]
        # assistant(route) → tool(route) → assistant(lookup) → tool(lookup)
        self.assertEqual(
            roles_and_names,
            [("assistant", "route"), ("tool", "route"),
             ("assistant", "lookup_invoice"), ("tool", "lookup_invoice")],
        )

    def test_route_output_and_handoff(self):
        result = _run(self._scenario(), [{"role": "user", "content": "x"}])
        route_out = next(m for m in result.tool_messages if m["role"] == "tool" and m["name"] == "route")
        self.assertEqual(json.loads(route_out["content"]), {"routed": True, "agent": "billing"})

    def test_stable_call_ids(self):
        msgs = [{"role": "user", "content": "x"}]
        r1 = _run(self._scenario(), msgs)
        r2 = _run(self._scenario(), msgs)
        ids1 = [m["tool_call_id"] for m in r1.tool_messages if m["role"] == "tool"]
        ids2 = [m["tool_call_id"] for m in r2.tool_messages if m["role"] == "tool"]
        self.assertEqual(ids1, ids2)
        self.assertEqual(ids1, ["call_cs_1_1", "call_cs_1_2"])

    def test_responses_tools_attached(self):
        result = _run(self._scenario(), [{"role": "user", "content": "x"}])
        names = {e["name"] for e in result.responses_tools}
        self.assertEqual(names, {"lookup_invoice", "route"})


class OrchestratorTerminalTests(unittest.TestCase):
    def test_terminal_agent_emits_escalation_final(self):
        s = Scenario(
            id="esc",
            entry="dispatcher",
            agents=[
                Agent("dispatcher", routes=["human"], script=[Route("human")]),
                Agent("human", terminal=True, escalation_message="Escalating to a human."),
            ],
        )
        result = _run(s, [{"role": "user", "content": "help"}])
        self.assertEqual(result.content, "Escalating to a human.")


class OrchestratorKnobTests(unittest.TestCase):
    def test_malformed_args_surface_as_tool_error(self):
        s = Scenario(
            id="k",
            entry="a",
            agents=[Agent("a", tools=[lookup_invoice],
                          script=[MalformedCall(lookup_invoice), Final("ok")])],
        )
        result = _run(s, [{"role": "user", "content": "x"}])
        out = next(m for m in result.tool_messages if m["role"] == "tool")
        self.assertIn("Tool error", out["content"])
        # the malformed args are still present on the call (for validity scoring)
        call = next(m for m in result.tool_messages if m["role"] == "assistant")
        self.assertEqual(call["tool_calls"][0]["function"]["arguments"], "{intentionally_malformed_json")

    def test_unknown_tool_surface_as_tool_error(self):
        s = Scenario(id="k", entry="a",
                     agents=[Agent("a", script=[UnknownToolCall("ghost"), Final("ok")])])
        result = _run(s, [{"role": "user", "content": "x"}])
        out = next(m for m in result.tool_messages if m["role"] == "tool")
        self.assertIn("unsupported tool 'ghost'", out["content"])

    def test_required_violation_keeps_empty_args_and_errors(self):
        s = Scenario(id="k", entry="a",
                     agents=[Agent("a", tools=[lookup_invoice],
                                   script=[Call(lookup_invoice), Final("ok")])])
        result = _run(s, [{"role": "user", "content": "x"}])
        call = next(m for m in result.tool_messages if m["role"] == "assistant")
        self.assertEqual(call["tool_calls"][0]["function"]["arguments"], "{}")
        out = next(m for m in result.tool_messages if m["role"] == "tool")
        self.assertIn("Tool error", out["content"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simple_chatbot.scenario_orchestrator'`.

- [ ] **Step 3: Write minimal implementation**

```python
# simple_chatbot/scenario_orchestrator.py
"""Runs a Scenario into a ChatResult.

Maintains active-agent state and offers only the active agent's tools each
round. A `route` call switches the active agent. Every round appends to ONE
`tool_messages` list, which the wire layer serializes into one /v1/responses
output array (= one Spectral turn). Tool `call_id`s are deterministic
(`call_<scenario>_<turn>_<seq>`) so repeated runs are reproducible.
"""

from __future__ import annotations

import json

from loguru import logger

from simple_chatbot.agent import ChatResult
from simple_chatbot.scenario import ROUTE_TOOL_NAME, Scenario
from simple_chatbot.scenario_catalog import build_responses_tools
from simple_chatbot.scenario_provider import DeterministicProvider, ProviderDecision
from simple_chatbot.tools import _parse_args


class ScenarioOrchestrator:
    def __init__(
        self,
        scenario: Scenario,
        provider: DeterministicProvider | None = None,
        max_rounds: int = 8,
    ) -> None:
        self.scenario = scenario
        self.provider = provider or DeterministicProvider()
        self.max_rounds = max_rounds
        self._tools_by_name = {t.name: t for t in scenario.all_tools()}

    async def chat(self, messages: list[dict]) -> ChatResult:
        turn_idx = sum(1 for m in messages if m.get("role") == "user")
        active = self.scenario.agent(self.scenario.entry)
        tool_messages: list[dict] = []
        content = ""
        seq = 0

        for _round in range(self.max_rounds):
            decision: ProviderDecision = await self.provider.decide(
                active, messages=messages, tool_messages=tool_messages
            )

            if decision.final is not None and not decision.calls:
                content = decision.final
                break

            tool_calls: list[dict] = []
            for pc in decision.calls:
                seq += 1
                call_id = f"call_{self.scenario.id}_{turn_idx}_{seq}"
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": pc.name, "arguments": pc.arguments},
                    }
                )

            assistant_msg: dict = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            if decision.reasoning:
                assistant_msg["reasoning_content"] = decision.reasoning
            tool_messages.append(assistant_msg)

            switch_to: str | None = None
            for tc in tool_calls:
                name = tc["function"]["name"]
                raw = tc["function"]["arguments"]
                output, target = self._execute(active, name, raw)
                if target is not None:
                    switch_to = target
                tool_messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "name": name, "content": output}
                )

            if switch_to is not None:
                active = self.scenario.agent(switch_to)
                if active.terminal:
                    content = active.escalation_message or "(escalated)"
                    break

        else:
            content = content or "(max rounds reached)"
            logger.bind(scenario=self.scenario.id).warning(
                "Scenario reached max rounds without a final answer"
            )

        final_messages = list(messages) + tool_messages + [{"role": "assistant", "content": content}]
        return ChatResult(
            content=content,
            retrieved_chunks=[],
            tool_messages=tool_messages,
            final_messages=final_messages,
            responses_tools=build_responses_tools(self.scenario),
        )

    def _execute(self, active, name: str, raw: str) -> tuple[str, str | None]:
        """Return (output_text, switch_target). switch_target is set only for a
        valid route call."""
        if name == ROUTE_TOOL_NAME:
            args, err = _parse_args(ROUTE_TOOL_NAME, raw)
            if err:
                return f"Tool error: {err}", None
            target = args.get("agent")
            if target not in active.routes:
                return (
                    f"Tool error: agent {active.name!r} cannot route to {target!r}",
                    None,
                )
            return json.dumps({"routed": True, "agent": target}), target

        tool = self._tools_by_name.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools_by_name)) or "(none)"
            return f"Tool error: unsupported tool {name!r}; available tools: {available}", None

        args, err = _parse_args(name, raw)
        if err:
            return f"Tool error: {err}", None
        try:
            result = tool.func(**args)
        except Exception as exc:  # missing-required, etc. — surfaced as a tool error
            return f"Tool error: {exc}", None
        return (result if isinstance(result, str) else json.dumps(result)), None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_orchestrator.py -v`
Expected: PASS (all classes).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario_orchestrator.py tests/test_scenario_orchestrator.py
git commit -m "feat(scenario): orchestrator with routing, knobs, stable call_ids"
```

---

## Task 8: Canonical `cs-routing` fixture

**Files:**
- Create: `simple_chatbot/fixtures/__init__.py`
- Create: `simple_chatbot/fixtures/cs_routing.py`
- Test: `tests/test_scenario_fixtures.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenario_fixtures.py
import asyncio
import json
import unittest

from simple_chatbot.scenario_provider import DeterministicProvider
from simple_chatbot.scenario_orchestrator import ScenarioOrchestrator


class CsRoutingFixtureTests(unittest.TestCase):
    def _run(self):
        from simple_chatbot.fixtures.cs_routing import scenario
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat([{"role": "user", "content": "I want a refund"}]))

    def test_routes_to_billing_and_finishes(self):
        result = self._run()
        names = [
            m["tool_calls"][0]["function"]["name"]
            for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        self.assertEqual(names[0], "route")
        self.assertIn("lookup_invoice", names)
        self.assertTrue(result.content)

    def test_catalog_has_route_and_billing_tools(self):
        result = self._run()
        names = {e["name"] for e in result.responses_tools}
        self.assertIn("route", names)
        self.assertIn("lookup_invoice", names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_fixtures.py::CsRoutingFixtureTests -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simple_chatbot.fixtures'`.

- [ ] **Step 3: Write minimal implementation**

```python
# simple_chatbot/fixtures/__init__.py
"""Fixture scenarios. Each module exposes a module-level `scenario: Scenario`."""
```

```python
# simple_chatbot/fixtures/cs_routing.py
"""Customer-service dispatcher routing to billing or human escalation."""

from simple_chatbot.scenario import Agent, Call, Final, Route, Scenario, tool


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice by id."""
    return {"invoice_id": invoice_id, "amount_due": "42.00", "status": "open"}


@tool
def issue_refund(invoice_id: str, amount: float) -> dict:
    """Issue a refund against an invoice."""
    return {"refunded": True, "invoice_id": invoice_id, "amount": amount}


scenario = Scenario(
    id="cs-routing",
    description="CS dispatcher routing to billing / human escalation.",
    entry="dispatcher",
    agents=[
        Agent(
            "dispatcher",
            system_prompt="Route requests to a specialist; don't answer domain questions yourself.",
            routes=["billing", "human"],
            script=[Route("billing")],
        ),
        Agent(
            "billing",
            system_prompt="You handle billing and refunds.",
            tools=[lookup_invoice, issue_refund],
            script=[
                Call(lookup_invoice, {"invoice_id": "INV-1"}),
                Call(issue_refund, {"invoice_id": "INV-1", "amount": 42.0}),
                Final("I've refunded $42.00 against invoice INV-1."),
            ],
        ),
        Agent(
            "human",
            terminal=True,
            escalation_message="Escalating to a human agent.",
        ),
    ],
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_fixtures.py::CsRoutingFixtureTests -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/fixtures/__init__.py simple_chatbot/fixtures/cs_routing.py tests/test_scenario_fixtures.py
git commit -m "feat(fixtures): canonical cs-routing scenario"
```

---

## Task 9: `validity-probe` misbehavior fixture

A fixture that fires each validity knob, so the deterministic invalid-trace path is exercised end-to-end.

**Files:**
- Create: `simple_chatbot/fixtures/validity_probe.py`
- Test: `tests/test_scenario_fixtures.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_scenario_fixtures.py
class ValidityProbeFixtureTests(unittest.TestCase):
    def _run(self):
        from simple_chatbot.fixtures.validity_probe import scenario
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat([{"role": "user", "content": "probe"}]))

    def test_emits_unknown_malformed_and_required_violation(self):
        result = self._run()
        call_msgs = [m for m in result.tool_messages if m["role"] == "assistant" and m.get("tool_calls")]
        args_by_name = {
            m["tool_calls"][0]["function"]["name"]: m["tool_calls"][0]["function"]["arguments"]
            for m in call_msgs
        }
        # unknown tool present
        self.assertIn("ghost_tool", args_by_name)
        # malformed args present (invalid JSON)
        self.assertIn("{intentionally_malformed_json", args_by_name.values())
        # a required-violation call with empty args present
        self.assertIn("{}", args_by_name.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_fixtures.py::ValidityProbeFixtureTests -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simple_chatbot.fixtures.validity_probe'`.

- [ ] **Step 3: Write minimal implementation**

```python
# simple_chatbot/fixtures/validity_probe.py
"""A single-agent scenario that fires each deterministic validity knob:
unknown tool (→1), malformed args (→2), required-arg violation (→2)."""

from simple_chatbot.scenario import (
    Agent,
    Call,
    Final,
    MalformedCall,
    Scenario,
    UnknownToolCall,
    tool,
)


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice by id."""
    return {"invoice_id": invoice_id, "status": "open"}


scenario = Scenario(
    id="validity-probe",
    description="Fires unknown-tool, malformed-args, and required-violation knobs.",
    entry="prober",
    agents=[
        Agent(
            "prober",
            system_prompt="A probe agent that deliberately misuses tools.",
            tools=[lookup_invoice],
            script=[
                UnknownToolCall("ghost_tool"),
                MalformedCall(lookup_invoice),
                Call(lookup_invoice),  # omits required invoice_id → required violation
                Final("Probe complete."),
            ],
        ),
    ],
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_fixtures.py -v`
Expected: PASS (all fixture tests).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/fixtures/validity_probe.py tests/test_scenario_fixtures.py
git commit -m "feat(fixtures): validity-probe misbehavior scenario"
```

---

## Task 10: `ScenarioRegistry` (load fixtures package)

**Files:**
- Create: `simple_chatbot/scenario_registry.py`
- Test: `tests/test_scenario_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenario_registry.py
import unittest

from simple_chatbot.scenario import Scenario
from simple_chatbot.scenario_registry import load_fixtures


class RegistryTests(unittest.TestCase):
    def test_loads_known_fixtures_keyed_by_id(self):
        registry = load_fixtures()
        self.assertIn("cs-routing", registry)
        self.assertIn("validity-probe", registry)
        self.assertIsInstance(registry["cs-routing"], Scenario)

    def test_returns_empty_for_missing_package(self):
        self.assertEqual(load_fixtures("simple_chatbot.does_not_exist"), {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simple_chatbot.scenario_registry'`.

- [ ] **Step 3: Write minimal implementation**

```python
# simple_chatbot/scenario_registry.py
"""Discover fixture scenarios by importing a package and collecting module-level
`scenario: Scenario` attributes. The scenario id is the request `model` selector.
"""

from __future__ import annotations

import importlib
import pkgutil

from loguru import logger

from simple_chatbot.scenario import Scenario


def load_fixtures(package_name: str = "simple_chatbot.fixtures") -> dict[str, Scenario]:
    try:
        pkg = importlib.import_module(package_name)
    except ModuleNotFoundError:
        logger.bind(package=package_name).warning("Fixtures package not found")
        return {}

    registry: dict[str, Scenario] = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        module = importlib.import_module(f"{package_name}.{mod.name}")
        scenario = getattr(module, "scenario", None)
        if isinstance(scenario, Scenario):
            registry[scenario.id] = scenario
    logger.bind(count=len(registry), ids=sorted(registry)).info("Loaded fixtures")
    return registry
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_registry.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario_registry.py tests/test_scenario_registry.py
git commit -m "feat(scenario): fixture registry loading the fixtures package"
```

---

## Task 11: Server wiring — selection by `model` + `/v1/models`

`/v1/responses` runs the orchestrator when `body.model` is a known fixture id, else the existing agent. `/v1/models` lists fixture ids.

**Files:**
- Modify: `simple_chatbot/server.py` (globals, `init`, `responses_create`, `list_models`)
- Test: `tests/test_scenario_endpoint.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenario_endpoint.py
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

import simple_chatbot.server as server
from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.scenario_registry import load_fixtures


class _FakeConversationLogger:
    async def log(self, conversation_id, messages, response, chunks) -> None:
        return None


class _FakeAgent:
    async def chat(self, messages):
        from simple_chatbot.agent import ChatResult
        return ChatResult(content="agent-path", retrieved_chunks=[],
                          final_messages=list(messages) + [{"role": "assistant", "content": "agent-path"}])


class ScenarioEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = (
            server._config, server._agent, server._conversation_logger,
            getattr(server, "_response_store", None), getattr(server, "_scenario_registry", None),
        )
        self.addCleanup(self._restore)

        server._config = SimpleChatbotConfig(
            docs_dir=Path(self.tmp.name) / "docs",
            conversation_log_dir=Path(self.tmp.name) / "conversations",
        )
        server._agent = _FakeAgent()
        server._conversation_logger = _FakeConversationLogger()
        from simple_chatbot.responses import ResponseStore
        server._response_store = ResponseStore()
        server._scenario_registry = load_fixtures()
        self.client = TestClient(server.app)

    def _restore(self):
        (server._config, server._agent, server._conversation_logger,
         server._response_store, server._scenario_registry) = self.old

    def test_known_model_runs_scenario(self):
        resp = self.client.post("/v1/responses", json={"model": "cs-routing", "input": "refund"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        types = [it["type"] for it in body["output"]]
        # routing + sub-agent calls collapse into one output array
        self.assertIn("function_call", types)
        self.assertEqual(types[-1], "message")
        # catalog published in Response.tools
        names = {t["name"] for t in body["tools"]}
        self.assertIn("route", names)
        self.assertIn("lookup_invoice", names)

    def test_unknown_model_falls_back_to_agent(self):
        resp = self.client.post("/v1/responses", json={"model": "not-a-fixture", "input": "hi"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["output"][-1]["content"][0]["text"], "agent-path")

    def test_models_lists_fixtures(self):
        resp = self.client.get("/v1/models")
        ids = {m["id"] for m in resp.json()["data"]}
        self.assertIn("cs-routing", ids)
        self.assertIn("validity-probe", ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_endpoint.py -v`
Expected: FAIL — `test_known_model_runs_scenario` returns the agent path (no `function_call` items) / `tools` empty, and `test_models_lists_fixtures` misses fixture ids.

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/server.py`:

(a) Add the global near the other globals (after `_response_store`):

```python
_response_store: ResponseStore | None = None
_scenario_registry: dict | None = None
```

(b) Add imports near the top (with the other `simple_chatbot` imports):

```python
from simple_chatbot.scenario_orchestrator import ScenarioOrchestrator
from simple_chatbot.scenario_provider import DeterministicProvider
from simple_chatbot.scenario_registry import load_fixtures
```

(c) In `init`, after `_response_store = ResponseStore()`, add:

```python
    global _scenario_registry
    _scenario_registry = load_fixtures()
```

(Declare `_scenario_registry` in the existing `global` statement at the top of `init` alongside the others as well.)

(d) In `responses_create`, replace the single line `result = await _require_agent().chat(messages)` with scenario selection:

```python
        registry = _scenario_registry or {}
        scenario = registry.get(body.model)
        if scenario is not None:
            orchestrator = ScenarioOrchestrator(scenario, DeterministicProvider())
            result = await orchestrator.chat(messages)
        else:
            result = await _require_agent().chat(messages)
```

(e) In `list_models`, return fixtures alongside the base model:

```python
@app.get("/v1/models")
def list_models():
    model_id = _config.chat_model if _config else "unknown"
    created = int(time.time())
    data = [{"id": model_id, "object": "model", "created": created, "owned_by": "simple-chatbot"}]
    for fixture_id in sorted(_scenario_registry or {}):
        data.append({"id": fixture_id, "object": "model", "created": created, "owned_by": "simple-chatbot-fixture"})
    logger.bind(model_id=model_id, fixture_count=len(_scenario_registry or {})).debug("Returning model list")
    return {"object": "list", "data": data}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_endpoint.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/server.py tests/test_scenario_endpoint.py
git commit -m "feat(server): select scenario fixtures by model; list them in /v1/models"
```

---

## Task 12: End-to-end — catalog populated + per-turn window (Spectral-shape assertion)

Drives `/v1/responses` and asserts the trace ingests the way Spectral reads it: a populated catalog from `Response.tools`, and a single turn whose ordered items contain the routing decision followed by the routed agent's calls.

**Files:**
- Test: `tests/test_scenario_endpoint.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_scenario_endpoint.py
class ScenarioTraceShapeTests(ScenarioEndpointTests):
    def test_turn_window_has_route_then_subagent_calls(self):
        resp = self.client.post("/v1/responses", json={"model": "cs-routing", "input": "refund"})
        body = resp.json()
        # Emulate Spectral's per-turn windowing: one user message in => one turn;
        # the output array IS that turn's window. Extract ordered tool-call names.
        call_names = [
            it["name"] for it in body["output"] if it["type"] == "function_call"
        ]
        self.assertEqual(call_names[0], "route")  # routing decision is in-window
        self.assertIn("lookup_invoice", call_names)  # routed sub-agent's call too
        # function_call_output items are paired and present
        out_types = [it["type"] for it in body["output"]]
        self.assertIn("function_call_output", out_types)

    def test_catalog_resolves_calls_by_name(self):
        resp = self.client.post("/v1/responses", json={"model": "cs-routing", "input": "refund"})
        body = resp.json()
        catalog_names = {t["name"] for t in body["tools"]}
        call_names = {it["name"] for it in body["output"] if it["type"] == "function_call"}
        # every emitted call resolves to a catalog entry by name (validity precondition)
        self.assertTrue(call_names.issubset(catalog_names))
```

- [ ] **Step 2: Run test to verify it fails (then passes — no new code)**

Run: `python -m pytest tests/test_scenario_endpoint.py::ScenarioTraceShapeTests -v`
Expected: PASS immediately — Tasks 7 + 11 already produce this shape. (If it fails, the orchestrator/serialization has a defect to fix before proceeding; this task is the integration gate.)

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS — all new tests plus the pre-existing suite green (no regressions in `test_responses_*`, `test_agent_*`, `test_scripted_mode`).

- [ ] **Step 4: Commit**

```bash
git add tests/test_scenario_endpoint.py
git commit -m "test(scenario): e2e trace-shape + catalog-resolution gate"
```

---

## Self-Review

**Spec coverage (Phase 1 items):**
- Scenario model + builder API → Tasks 1–3.
- `@tool` typed-function schema generation → Task 1.
- Step types incl. misbehavior knobs → Task 2 (constructs), Tasks 7/9 (behavior).
- Orchestrator (active-agent state, scoped offering, routing, stable call_ids, one-turn collapse) → Task 7.
- DeterministicProvider → Task 6.
- Catalog publication via `Response.tools` → Tasks 4 + 5.
- Server selection by `model`; `/v1/models` listing → Task 11.
- Canonical fixtures → Tasks 8 (cs-routing) + 9 (validity-probe).
- Reproducibility (stable call_ids) → Task 7 (`test_stable_call_ids`).
- E2E Spectral-shape assertion (catalog populated + per-turn window) → Task 12.
- Out of scope (correctly absent): live mode (Phase 2), broader knob/fixture library (Phase 3), ghost-backend harvest (Phase 4, author-gated).

**Type consistency check:** `ScenarioTool(name, description, parameters, func)`, `Agent(name, system_prompt, tools, routes, script, terminal, escalation_message)`, `Scenario(id, entry, agents, description)`, `PlannedCall(name, arguments)`, `ProviderDecision(final, calls, reasoning, ignore_retrieval)`, `DeterministicProvider.decide(agent, messages, tool_messages)`, `ScenarioOrchestrator(scenario, provider, max_rounds).chat(messages)`, `build_responses_tools(scenario)`, `load_fixtures(package_name)` — all used consistently across tasks. `ChatResult.responses_tools` added in Task 5 and consumed in Tasks 7/11/12. `ROUTE_TOOL_NAME = "route"` used in scenario.py, scenario_catalog.py, scenario_orchestrator.py.

**Placeholder scan:** none — every code step contains complete, runnable code; every run step has an exact command and expected outcome.

**Note on `_parse_args` reuse:** Task 7 imports `_parse_args` from `simple_chatbot.tools`. It is module-private by name but same-package reuse is intentional (the orchestrator deliberately mirrors the agent's arg-parsing + tool-error wording so traces are consistent across both paths).
