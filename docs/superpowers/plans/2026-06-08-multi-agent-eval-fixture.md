# Live Multi-Agent Eval Fixture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live-mode router→subagent fixture whose mock tools carry conversation-scoped, mutating state, so ghost-backend's supervisors can evaluate a real router+subagent+tool path.

**Architecture:** Reuse the existing scenario engine (`Agent`/`route`/`active_agent` handoff + `LiveProvider`). Add a server-level `ConversationStateStore` keyed by `conversation_id` that hands each turn a mutable `ConvState`. Tools opt into state by declaring a `state` parameter, which `@tool` excludes from the JSON schema and the orchestrator injects at execution. A new `support_swarm` fixture wires a router and three subagents (+ human escalation) in `live` mode.

**Tech Stack:** Python 3.12, FastAPI, LiteLLM, unittest. Package `simple_chatbot/`, tests `tests/`. Run tests with `uv run python -m unittest <module> -v`.

**Spec:** `docs/superpowers/specs/2026-06-08-multi-agent-eval-fixture-design.md`

---

## File Structure

- Create: `simple_chatbot/conversation_state.py` — `ConvState` + `ConversationStateStore` (LRU-bounded, async-lock-guarded). One responsibility: per-conversation mock world storage.
- Modify: `simple_chatbot/scenario.py` — `ScenarioTool.wants_state` field; `tool()` skips a `state` param and flags it.
- Modify: `simple_chatbot/scenario_orchestrator.py` — thread `state` through `chat()` → `_execute()`, inject into stateful tools.
- Modify: `simple_chatbot/server.py` — module-level `_conversation_state_store`; fetch per-conversation state and pass to `orchestrator.chat`.
- Create: `simple_chatbot/fixtures/support_swarm.py` — the fixture + its stateful mock tools.
- Create: `tests/test_conversation_state.py`
- Create: `tests/test_support_swarm.py`
- Modify: `tests/test_scenario_model.py` — `@tool` state-opt-in tests.
- Modify: `tests/test_scenario_orchestrator.py` — state-injection test.

---

## Task 1: Conversation state store

**Files:**
- Create: `simple_chatbot/conversation_state.py`
- Test: `tests/test_conversation_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_conversation_state.py`:

```python
import asyncio
import unittest

from simple_chatbot.conversation_state import ConversationStateStore, ConvState


def _run(coro):
    return asyncio.run(coro)


class ConversationStateStoreTests(unittest.TestCase):
    def test_get_or_create_returns_same_instance_per_id(self):
        store = ConversationStateStore()
        a = _run(store.get_or_create("conv-1"))
        a["orders"] = {"ORD-1": "open"}
        b = _run(store.get_or_create("conv-1"))
        self.assertIs(a, b)
        self.assertEqual(b["orders"], {"ORD-1": "open"})

    def test_distinct_ids_get_distinct_states(self):
        store = ConversationStateStore()
        a = _run(store.get_or_create("conv-1"))
        b = _run(store.get_or_create("conv-2"))
        self.assertIsNot(a, b)

    def test_returns_convstate_instances(self):
        store = ConversationStateStore()
        self.assertIsInstance(_run(store.get_or_create("conv-1")), ConvState)

    def test_lru_evicts_oldest_beyond_cap(self):
        store = ConversationStateStore(max_conversations=2)
        first = _run(store.get_or_create("conv-1"))
        first["seen"] = True
        _run(store.get_or_create("conv-2"))
        _run(store.get_or_create("conv-3"))  # evicts conv-1 (oldest)
        # conv-1 is recreated fresh, so the prior mutation is gone
        self.assertNotIn("seen", _run(store.get_or_create("conv-1")))

    def test_access_refreshes_recency(self):
        store = ConversationStateStore(max_conversations=2)
        _run(store.get_or_create("conv-1"))
        _run(store.get_or_create("conv-2"))
        _run(store.get_or_create("conv-1"))  # touch conv-1 → conv-2 now oldest
        marked = _run(store.get_or_create("conv-1"))
        marked["kept"] = True
        _run(store.get_or_create("conv-3"))  # evicts conv-2, not conv-1
        self.assertTrue(_run(store.get_or_create("conv-1")).get("kept"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_conversation_state -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simple_chatbot.conversation_state'`.

- [ ] **Step 3: Write minimal implementation**

Create `simple_chatbot/conversation_state.py`:

```python
"""Conversation-scoped mock state for fixture tools.

Stateful fixture tools (order tracking, customer data, billing) need a small
mutable "world" that persists across a conversation's turns yet stays isolated
across parallel conversations. `ConversationStateStore` maps a conversation id
to one `ConvState` (a plain dict), created lazily and bounded by an LRU cap so
the test harness can't grow without limit. Process-local; resets on restart,
like `ResponseStore`.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict


class ConvState(dict):
    """One conversation's mutable mock world. A thin dict subclass so a tool can
    `state.setdefault("orders", {...})` to lazily seed its own slice."""


class ConversationStateStore:
    """Maps conversation_id -> ConvState, lazily created and LRU-bounded."""

    def __init__(self, max_conversations: int = 512) -> None:
        self._lock = asyncio.Lock()
        self._states: "OrderedDict[str, ConvState]" = OrderedDict()
        self._max = max_conversations

    async def get_or_create(self, conversation_id: str) -> ConvState:
        async with self._lock:
            state = self._states.get(conversation_id)
            if state is None:
                state = ConvState()
                self._states[conversation_id] = state
            self._states.move_to_end(conversation_id)
            while len(self._states) > self._max:
                self._states.popitem(last=False)
            return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_conversation_state -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/conversation_state.py tests/test_conversation_state.py
git commit -m "feat: add conversation-scoped state store for fixture tools"
```

---

## Task 2: `@tool` opt-in for injected state

**Files:**
- Modify: `simple_chatbot/scenario.py:21-63` (`ScenarioTool` dataclass + `tool()` decorator)
- Test: `tests/test_scenario_model.py` (add to `ToolDecoratorTests`)

- [ ] **Step 1: Write the failing test**

Add these two methods inside the `ToolDecoratorTests` class in `tests/test_scenario_model.py`:

```python
    def test_state_param_excluded_from_schema_and_flagged(self):
        @tool
        def get_order(order_id: str, state) -> dict:
            """Look up an order."""
            return {"order_id": order_id}

        self.assertTrue(get_order.wants_state)
        self.assertNotIn("state", get_order.parameters["properties"])
        self.assertEqual(get_order.parameters["required"], ["order_id"])

    def test_tool_without_state_is_not_flagged(self):
        @tool
        def ping(message: str) -> dict:
            """Echo."""
            return {"message": message}

        self.assertFalse(ping.wants_state)
        self.assertIn("message", ping.parameters["properties"])
```

Confirm `from simple_chatbot.scenario import ... tool` is already imported at the top of the file (it is, used by existing tests).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_scenario_model.ToolDecoratorTests -v`
Expected: FAIL with `AttributeError: 'ScenarioTool' object has no attribute 'wants_state'`.

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/scenario.py`, add the `wants_state` field to `ScenarioTool` (the dataclass at lines 21-28):

```python
@dataclass
class ScenarioTool:
    """A tool: JSON-Schema parameters + an async-free stub callable."""

    name: str
    description: str
    parameters: dict
    func: Callable[..., dict | str]
    wants_state: bool = False
```

Then update the `tool()` decorator body (lines 38-63) to skip and flag a `state` parameter:

```python
    sig = inspect.signature(func)
    fields: dict = {}
    wants_state = False
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        if pname == "state":
            # Injected at execution from the conversation's ConvState; never
            # part of the advertised JSON schema.
            wants_state = True
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
        wants_state=wants_state,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_scenario_model -v`
Expected: PASS (all `ToolDecoratorTests` plus the rest of the module).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario.py tests/test_scenario_model.py
git commit -m "feat: @tool excludes and flags an injected 'state' param"
```

---

## Task 3: Orchestrator injects conversation state into stateful tools

**Files:**
- Modify: `simple_chatbot/scenario_orchestrator.py:36-104` (`chat`), `:118-145` (`_execute`)
- Test: `tests/test_scenario_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Add this test class to `tests/test_scenario_orchestrator.py` (the imports `ConvState` and `LiveProvider` are new — add them). At the top of the file the existing imports include `asyncio`, `json`, `unittest`, `Agent`, `Call`, `Final`, `Route`, `Scenario`, `tool`, `DeterministicProvider`, `ScenarioOrchestrator`. Add:

```python
from simple_chatbot.conversation_state import ConvState
```

Then add:

```python
@tool
def bump(state) -> dict:
    """Increment a per-conversation counter and return it."""
    state["count"] = state.get("count", 0) + 1
    return {"count": state["count"]}


class OrchestratorStateInjectionTests(unittest.TestCase):
    def _scenario(self):
        return Scenario(
            id="counter",
            entry="worker",
            agents=[
                Agent("worker", tools=[bump], script=[Call(bump, {}), Final("done")]),
            ],
        )

    def test_stateful_tool_receives_and_mutates_injected_state(self):
        state = ConvState()
        orch = ScenarioOrchestrator(self._scenario(), DeterministicProvider())
        result = asyncio.run(orch.chat([{"role": "user", "content": "go"}], state=state))
        # The tool's output reflects the injected state...
        bump_out = next(m for m in result.tool_messages if m.get("name") == "bump")
        self.assertEqual(json.loads(bump_out["content"]), {"count": 1})
        # ...and the mutation is visible on the passed-in ConvState.
        self.assertEqual(state["count"], 1)

    def test_stateful_tool_works_without_explicit_state(self):
        # No state passed: a throwaway ConvState is used, tool still runs.
        orch = ScenarioOrchestrator(self._scenario(), DeterministicProvider())
        result = asyncio.run(orch.chat([{"role": "user", "content": "go"}]))
        bump_out = next(m for m in result.tool_messages if m.get("name") == "bump")
        self.assertEqual(json.loads(bump_out["content"]), {"count": 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_scenario_orchestrator.OrchestratorStateInjectionTests -v`
Expected: FAIL — `chat()` does not accept a `state` keyword (`TypeError`), and `_execute` does not inject state.

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/scenario_orchestrator.py`, add the import near the top (after the existing `from simple_chatbot.scenario import ...` line):

```python
from simple_chatbot.conversation_state import ConvState
```

Change the `chat` signature (line 36) and create a fallback state at the top of the method (right after the signature, before `turn_idx = ...`):

```python
    async def chat(
        self,
        messages: list[dict],
        start_agent: str | None = None,
        state: ConvState | None = None,
    ) -> ChatResult:
        conv_state = state if state is not None else ConvState()
        turn_idx = sum(1 for m in messages if m.get("role") == "user")
```

In the per-call loop, pass `conv_state` to `_execute`. Change the call site (currently line 78) from:

```python
                    output, target = self._execute(active, name, raw)
```

to:

```python
                    output, target = self._execute(active, name, raw, conv_state)
```

Update `_execute` (line 118) to accept and inject state. Replace the method's signature and tool-call block:

```python
    def _execute(self, active, name: str, raw: str, state: ConvState) -> tuple[str, str | None]:
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
            if tool.wants_state:
                result = tool.func(state=state, **args)
            else:
                result = tool.func(**args)
        except Exception as exc:  # missing-required, etc. — surfaced as a tool error
            return f"Tool error: {exc}", None
        return (result if isinstance(result, str) else json.dumps(result)), None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_scenario_orchestrator -v`
Expected: PASS — new injection tests plus all existing orchestrator tests (stateless tools unaffected).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario_orchestrator.py tests/test_scenario_orchestrator.py
git commit -m "feat: orchestrator injects conversation state into stateful tools"
```

---

## Task 4: Wire the state store through the server

**Files:**
- Modify: `simple_chatbot/server.py` (imports, module globals, `init`, `/v1/responses` scenario branch)

This task has no standalone test; it is covered end-to-end by the server integration test in Task 5. The change is small and mechanical.

- [ ] **Step 1: Add the import**

In `simple_chatbot/server.py`, add to the imports block (next to the other `simple_chatbot` imports, e.g. after the `conversation_logger` import on line 15):

```python
from simple_chatbot.conversation_state import ConversationStateStore
```

- [ ] **Step 2: Add the module global**

After the existing module globals (the block ending at line 40 with `_misbehavior_policy`), add:

```python
# Per-conversation mock-tool state for scenario fixtures. Defaulted at import so
# tests that poke server globals (rather than calling init) still find a store.
_conversation_state_store: ConversationStateStore = ConversationStateStore()
```

- [ ] **Step 3: Reset it in `init()`**

In `init()`, add `_conversation_state_store` to the `global` declaration (line 96) and reassign a fresh store next to where `_response_store` is created (line 126):

```python
    global _config, _agent, _conversation_logger, _response_store, _scenario_registry, _acompletion, _misbehavior_policy, _conversation_state_store
```

and, right after `_response_store = ResponseStore()`:

```python
    _conversation_state_store = ConversationStateStore()
```

- [ ] **Step 4: Pass state into the orchestrator**

In the `/v1/responses` handler, inside `if scenario is not None:`, locate the `start_agent = (...)` assignment and the `result = await orchestrator.chat(messages, start_agent=start_agent)` call (lines 374-379). Replace that `result = ...` line with:

```python
            state = await _conversation_state_store.get_or_create(conversation_id)
            result = await orchestrator.chat(
                messages, start_agent=start_agent, state=state
            )
```

- [ ] **Step 5: Run the existing server/scenario tests to verify nothing regressed**

Run: `uv run python -m unittest tests.test_scenario_endpoint tests.test_responses_endpoint -v`
Expected: PASS (the import-time default store means the scenario branch resolves a store even though these tests poke globals).

- [ ] **Step 6: Commit**

```bash
git add simple_chatbot/server.py
git commit -m "feat: thread per-conversation state store into scenario orchestration"
```

---

## Task 5: The `support_swarm` fixture + its tests

**Files:**
- Create: `simple_chatbot/fixtures/support_swarm.py`
- Test: `tests/test_support_swarm.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_support_swarm.py`:

```python
import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

import simple_chatbot.server as server
from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.conversation_state import ConversationStateStore, ConvState
from simple_chatbot.responses import ResponseStore
from simple_chatbot.scenario_orchestrator import ScenarioOrchestrator
from simple_chatbot.scenario_provider import LiveProvider
from simple_chatbot.fixtures.support_swarm import (
    scenario,
    get_order,
    cancel_order,
    get_customer,
    update_customer,
    lookup_invoice,
    issue_refund,
)


# --- State semantics: call the mock tool functions directly (no LLM) ---------

class StatefulToolTests(unittest.TestCase):
    def test_seeded_world_on_fresh_state(self):
        state = ConvState()
        self.assertEqual(get_order.func(order_id="ORD-1", state=state)["status"], "open")
        self.assertEqual(get_customer.func(state=state)["email"], "ada@example.com")
        self.assertEqual(lookup_invoice.func(invoice_id="INV-1", state=state)["status"], "open")

    def test_cancel_then_get_reflects_cancelled(self):
        state = ConvState()
        cancel_order.func(order_id="ORD-1", state=state)
        self.assertEqual(get_order.func(order_id="ORD-1", state=state)["status"], "cancelled")

    def test_update_customer_persists(self):
        state = ConvState()
        update_customer.func(field="email", value="grace@example.com", state=state)
        self.assertEqual(get_customer.func(state=state)["email"], "grace@example.com")

    def test_issue_refund_flips_invoice_status(self):
        state = ConvState()
        issue_refund.func(invoice_id="INV-1", amount=42.0, state=state)
        self.assertEqual(lookup_invoice.func(invoice_id="INV-1", state=state)["status"], "refunded")

    def test_isolation_between_states(self):
        a, b = ConvState(), ConvState()
        cancel_order.func(order_id="ORD-1", state=a)
        self.assertEqual(get_order.func(order_id="ORD-1", state=a)["status"], "cancelled")
        self.assertEqual(get_order.func(order_id="ORD-1", state=b)["status"], "open")

    def test_unknown_order_returns_error(self):
        self.assertIn("error", get_order.func(order_id="NOPE", state=ConvState()))


# --- Fake sequenced LLM for the live path ------------------------------------

class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, name, arguments):
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, content="", tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, message, finish_reason):
        self.choices = [_Choice(message, finish_reason)]
        self.usage = None


def _call(name, args):
    return _Resp(_Msg(tool_calls=[_TC(name, json.dumps(args))]), "tool_calls")


def _final(text):
    return _Resp(_Msg(content=text), "stop")


def _seq_acompletion(responses):
    it = iter(responses)

    async def fake(**kwargs):
        return next(it)

    return fake


class LiveSwarmTests(unittest.TestCase):
    def _orch(self, responses):
        provider = LiveProvider(acompletion=_seq_acompletion(responses), model="test-model")
        return ScenarioOrchestrator(scenario, provider)

    def test_router_handoff_and_stateful_tool(self):
        responses = [
            _call("route", {"agent": "order_tracking"}),
            _call("cancel_order", {"order_id": "ORD-1"}),
            _final("Your order ORD-1 is cancelled."),
        ]
        state = ConvState()
        result = asyncio.run(
            self._orch(responses).chat(
                [{"role": "user", "content": "cancel my order ORD-1"}], state=state
            )
        )
        self.assertEqual(result.active_agent, "order_tracking")
        self.assertEqual(result.content, "Your order ORD-1 is cancelled.")
        self.assertEqual(state["orders"]["ORD-1"]["status"], "cancelled")
        names = [m["name"] for m in result.tool_messages if m["role"] == "tool"]
        self.assertEqual(names, ["route", "cancel_order"])

    def test_route_back_to_router_then_other_subagent(self):
        responses = [
            _call("route", {"agent": "order_tracking"}),
            _call("route", {"agent": "router"}),
            _call("route", {"agent": "billing"}),
            _call("lookup_invoice", {"invoice_id": "INV-1"}),
            _final("Invoice INV-1 is open for $42.00."),
        ]
        result = asyncio.run(
            self._orch(responses).chat(
                [{"role": "user", "content": "actually, a billing question"}], state=ConvState()
            )
        )
        self.assertEqual(result.active_agent, "billing")
        names = [m["name"] for m in result.tool_messages if m["role"] == "tool"]
        self.assertEqual(names, ["route", "route", "route", "lookup_invoice"])


# --- Server integration: state persists across chained turns -----------------

class ServerSwarmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = (
            server._config,
            server._agent,
            server._conversation_logger,
            getattr(server, "_response_store", None),
            getattr(server, "_scenario_registry", None),
            getattr(server, "_acompletion", None),
            getattr(server, "_conversation_state_store", None),
        )
        self.addCleanup(self._restore)

        server._config = SimpleChatbotConfig(
            docs_dir=Path(self.tmp.name) / "docs",
            conversation_log_dir=Path(self.tmp.name) / "conversations",
        )
        server._agent = None
        server._conversation_logger = _FakeConversationLogger()
        server._response_store = ResponseStore()
        server._conversation_state_store = ConversationStateStore()
        server._scenario_registry = {"support-swarm": scenario}
        # Turn 1: route → cancel → final.  Turn 2 (resumes order_tracking): get → final.
        server._acompletion = _seq_acompletion(
            [
                _call("route", {"agent": "order_tracking"}),
                _call("cancel_order", {"order_id": "ORD-1"}),
                _final("Cancelled ORD-1."),
                _call("get_order", {"order_id": "ORD-1"}),
                _final("ORD-1 status reported."),
            ]
        )
        self.client = TestClient(server.app)

    def _restore(self):
        (
            server._config,
            server._agent,
            server._conversation_logger,
            server._response_store,
            server._scenario_registry,
            server._acompletion,
            server._conversation_state_store,
        ) = self.old

    def test_state_persists_across_chained_turns(self):
        r1 = self.client.post(
            "/v1/responses",
            json={"model": "support-swarm", "input": "cancel ORD-1", "user": "conv-x"},
        )
        self.assertEqual(r1.status_code, 200, r1.text)
        first_id = r1.json()["id"]

        # The mutation landed in the conversation's state.
        state = asyncio.run(server._conversation_state_store.get_or_create("conv-x"))
        self.assertEqual(state["orders"]["ORD-1"]["status"], "cancelled")

        # Turn 2 resumes the same conversation; get_order sees the cancellation.
        r2 = self.client.post(
            "/v1/responses",
            json={
                "model": "support-swarm",
                "input": "what's the status now?",
                "previous_response_id": first_id,
                "user": "conv-x",
            },
        )
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertIn("cancelled", r2.text)


class _FakeConversationLogger:
    async def log(self, conversation_id, messages, response, chunks) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_support_swarm -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'simple_chatbot.fixtures.support_swarm'`.

- [ ] **Step 3: Write the fixture**

Create `simple_chatbot/fixtures/support_swarm.py`:

```python
"""Customer-support swarm: a router that hands off, statefully, to specialized
subagents (order tracking, customer data, billing) or escalates to a human.

Runs in `live` mode — the LLM drives routing and tool use. The tools are mocks
whose state lives in a per-conversation `ConvState` (injected via the `state`
param), so a change made on one turn is visible on later turns of the same
conversation while staying isolated across parallel conversations. Each tool
lazily seeds its own slice of the world via `state.setdefault(...)`.
"""

from simple_chatbot.scenario import Agent, Scenario, tool


def _orders(state):
    return state.setdefault(
        "orders",
        {"ORD-1": {"order_id": "ORD-1", "item": "Wireless headphones", "status": "open"}},
    )


@tool
def get_order(order_id: str, state) -> dict:
    """Look up an order's current status by id."""
    order = _orders(state).get(order_id)
    if order is None:
        return {"error": f"no order {order_id!r}"}
    return dict(order)


@tool
def cancel_order(order_id: str, state) -> dict:
    """Cancel an order by id."""
    orders = _orders(state)
    order = orders.get(order_id)
    if order is None:
        return {"error": f"no order {order_id!r}"}
    order["status"] = "cancelled"
    return dict(order)


def _customer(state):
    return state.setdefault(
        "customer",
        {"customer_id": "CUST-1", "email": "ada@example.com", "address": "1 Analytical Way"},
    )


@tool
def get_customer(state) -> dict:
    """Return the current customer profile."""
    return dict(_customer(state))


@tool
def update_customer(field: str, value: str, state) -> dict:
    """Update a customer profile field ('email' or 'address')."""
    customer = _customer(state)
    if field not in ("email", "address"):
        return {"error": f"unknown field {field!r}"}
    customer[field] = value
    return dict(customer)


def _invoice(state):
    return state.setdefault(
        "invoice", {"invoice_id": "INV-1", "amount_due": "42.00", "status": "open"}
    )


@tool
def lookup_invoice(invoice_id: str, state) -> dict:
    """Look up an invoice by id."""
    return dict(_invoice(state))


@tool
def issue_refund(invoice_id: str, amount: float, state) -> dict:
    """Issue a refund against an invoice."""
    invoice = _invoice(state)
    invoice["status"] = "refunded"
    return {"refunded": True, "invoice_id": invoice_id, "amount": amount}


scenario = Scenario(
    id="support-swarm",
    description=(
        "Router dispatching to order-tracking / customer-data / billing "
        "subagents, with human escalation."
    ),
    entry="router",
    mode="live",
    agents=[
        Agent(
            "router",
            system_prompt=(
                "You are the front-desk router. Identify the user's intent and hand off to "
                "the right specialist with the `route` tool: order_tracking for order status "
                "or cancellation, customer_data for profile changes, billing for invoices or "
                "refunds, human for anything else. Do not answer domain questions yourself."
            ),
            routes=["order_tracking", "customer_data", "billing", "human"],
        ),
        Agent(
            "order_tracking",
            system_prompt=(
                "You handle order status and cancellations. Use your tools. If the request is "
                "out of scope, route back to the router."
            ),
            tools=[get_order, cancel_order],
            routes=["router"],
        ),
        Agent(
            "customer_data",
            system_prompt=(
                "You handle customer profile reads and updates. Use your tools. If the request "
                "is out of scope, route back to the router."
            ),
            tools=[get_customer, update_customer],
            routes=["router"],
        ),
        Agent(
            "billing",
            system_prompt=(
                "You handle invoices and refunds. Use your tools. If the request is out of "
                "scope, route back to the router."
            ),
            tools=[lookup_invoice, issue_refund],
            routes=["router"],
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

Run: `uv run python -m unittest tests.test_support_swarm -v`
Expected: PASS (all `StatefulToolTests`, `LiveSwarmTests`, `ServerSwarmTests`).

- [ ] **Step 5: Run the full suite**

Run: `uv run python -m unittest discover -s tests`
Expected: PASS — including `tests.test_scenario_registry` (auto-discovers the new fixture) and `tests.test_scenario_catalog` (the new fixture's tools appear in the catalog without a `state` property).

- [ ] **Step 6: Commit**

```bash
git add simple_chatbot/fixtures/support_swarm.py tests/test_support_swarm.py
git commit -m "feat: add support-swarm live multi-agent fixture with stateful mock tools"
```

---

## Self-Review

**Spec coverage:**

- Conversation state store (spec §1) → Task 1.
- `@tool` opt-in for state (spec §2) → Task 2.
- Orchestrator injection + `state=None` fallback (spec §3) → Task 3.
- Server threading via `conversation_id` (spec §4) → Task 4.
- `support_swarm` fixture: router + order_tracking + customer_data + billing + human, live mode, route-back, lazy seeding (spec §5) → Task 5.
- Tests — state mutation/persistence/isolation/seeding, live handoff+injection, route-back, store LRU/lazy (spec §6) → Tasks 1 and 5.
- Non-goals respected: no new mode literal/CLI flag; no ghost-backend changes; in-memory only.

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code and exact commands.

**Type/name consistency:** `ConvState`, `ConversationStateStore.get_or_create`, `ScenarioTool.wants_state`, `chat(messages, start_agent, state)`, `_execute(active, name, raw, state)`, and the fixture tool names (`get_order`, `cancel_order`, `get_customer`, `update_customer`, `lookup_invoice`, `issue_refund`) are used identically across tasks and tests. Fixture id `support-swarm` matches the model name used in `ServerSwarmTests`.

**Cross-turn note:** `ServerSwarmTests` fixes `conversation_id` via the request `user` field so the state store key is stable across the two chained turns; turn 2 resumes `order_tracking` via the persisted `active_agent`, so its first decision is `get_order` (no extra route), matching the 5-element fake-LLM sequence.
