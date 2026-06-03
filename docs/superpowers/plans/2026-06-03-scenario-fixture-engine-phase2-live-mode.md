# Scenario Fixture Engine — Phase 2 (Live Mode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live, model-driven path to the scenario fixture engine: a `LiveProvider` that calls `litellm` per round with the active agent's `system_prompt` + scoped tools (ignoring the authored `script`), a per-scenario/server **mode switch** to choose deterministic vs live, and **active-agent persistence** so a `previous_response_id` follow-up resumes as the routed agent.

**Architecture:** Phase 1's `ScenarioOrchestrator` already owns the round loop and calls `provider.decide(agent, messages, tool_messages)` once per round. Phase 2 adds a second provider (`LiveProvider`) behind that same seam — the loop is unchanged. The orchestrator gains a `start_agent` parameter and reports the in-charge agent at turn end via a new `ChatResult.active_agent`; the server persists that name in the response store and passes it back on chained turns. A mode flag (per-scenario default, server override) selects the provider.

**Tech Stack:** Python 3.12, litellm (existing; injected as an `acompletion` callable, faked in tests), pydantic v2, FastAPI, `unittest` + `asyncio.run` (existing test style).

**Scope note:** simple-chatbot only. Deterministic mode stays the canonical, key-free CI path. Live-mode tests inject a fake `acompletion` — no real API key in CI. ghost-backend is **not touched** (Phase 4 harvest remains author-gated). Live mode ignores `script` entirely; misbehavior knobs are deterministic-only.

**Spec:** [../specs/2026-06-03-scenario-fixture-engine-phase2-live-mode-design.md](../specs/2026-06-03-scenario-fixture-engine-phase2-live-mode-design.md)

---

## File Structure

**Modify:**
- `simple_chatbot/agent.py` — extract a module-level `sampling_kwargs(config)` helper (reused by `Agent` and the server); add `active_agent: str | None` to `ChatResult`.
- `simple_chatbot/scenario.py` — add `mode: Literal["deterministic","live"]` to `Scenario`.
- `simple_chatbot/config.py` — add `scenario_mode: Literal["deterministic","live"] | None` to `SimpleChatbotConfig`.
- `simple_chatbot/scenario_orchestrator.py` — `chat(messages, start_agent=None)`: resume resolution, terminal-on-resume, set `ChatResult.active_agent`.
- `simple_chatbot/scenario_provider.py` — add `LiveProvider`.
- `simple_chatbot/server.py` — capture `_acompletion` global; select provider by resolved mode; persist + resume `active_agent`.

**Create (tests):**
- `tests/test_sampling_kwargs.py`
- (extend) `tests/test_scenario_model.py`, `tests/test_scenario_orchestrator.py`, `tests/test_scenario_provider.py`, `tests/test_scenario_endpoint.py`
- `tests/test_config_scenario_mode.py`

---

## Task 1: Extract `sampling_kwargs(config)` helper

`Agent._sampling_kwargs` builds the litellm sampling kwargs from config. The live provider needs the identical mapping, so lift it to a module-level function and have `Agent` delegate. Pure refactor — no behavior change.

**Files:**
- Modify: `simple_chatbot/agent.py:97-113`
- Test: `tests/test_sampling_kwargs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sampling_kwargs.py
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simple_chatbot.agent import sampling_kwargs
from simple_chatbot.config import SimpleChatbotConfig


class SamplingKwargsTests(unittest.TestCase):
    def test_only_set_params_are_included(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(
                docs_dir=Path(tmp) / "docs", temperature=0.5, gen_top_k=40
            )
            kwargs = sampling_kwargs(config)
        # explicitly set params present under their wire names
        self.assertEqual(kwargs["temperature"], 0.5)
        self.assertEqual(kwargs["top_k"], 40)  # gen_top_k -> top_k
        # unset params absent (not None-valued)
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("presence_penalty", kwargs)

    def test_empty_when_nothing_set(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(docs_dir=Path(tmp) / "docs")
            self.assertEqual(sampling_kwargs(config), {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sampling_kwargs.py -v`
Expected: FAIL — `ImportError: cannot import name 'sampling_kwargs' from 'simple_chatbot.agent'`.

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/agent.py`, add a module-level function (place it just above the `ChatResult` dataclass, after the `FINAL_ANSWER_INSTRUCTION` / `_sanitize_assistant_dump` block):

```python
def sampling_kwargs(config: SimpleChatbotConfig) -> dict:
    """Return only the sampling params the user set explicitly (skip None).

    Shared by the single-agent `Agent` and the scenario `LiveProvider` so both
    forward identical wire kwargs to litellm.
    """
    # attr name on SimpleChatbotConfig  ->  key litellm/OpenAI expects on the wire
    field_map = {
        "temperature": "temperature",
        "top_p": "top_p",
        "gen_top_k": "top_k",
        "min_p": "min_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "repetition_penalty": "repetition_penalty",
    }
    return {
        wire: getattr(config, attr)
        for attr, wire in field_map.items()
        if getattr(config, attr) is not None
    }
```

Then replace the body of the existing `Agent._sampling_kwargs` method (lines 97-113) so it delegates:

```python
    def _sampling_kwargs(self) -> dict:
        """Return only the sampling params the user set explicitly (skip None)."""
        return sampling_kwargs(self.config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sampling_kwargs.py tests/test_agent_tool_calls.py tests/test_agent_usage_and_final_messages.py -v`
Expected: PASS — new tests pass and the existing agent tests still pass (delegation preserved behavior).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/agent.py tests/test_sampling_kwargs.py
git commit -m "refactor(agent): extract module-level sampling_kwargs(config) helper"
```

---

## Task 2: `Scenario.mode` field

A per-scenario default mode. Pure metadata read by the server when selecting a provider.

**Files:**
- Modify: `simple_chatbot/scenario.py` (the `Scenario` dataclass)
- Test: `tests/test_scenario_model.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_scenario_model.py
from simple_chatbot.scenario import Agent, Scenario, tool


class ScenarioModeTests(unittest.TestCase):
    def test_mode_defaults_to_deterministic(self):
        s = Scenario(id="m", entry="a", agents=[Agent("a")])
        self.assertEqual(s.mode, "deterministic")

    def test_mode_accepts_live(self):
        s = Scenario(id="m", entry="a", agents=[Agent("a")], mode="live")
        self.assertEqual(s.mode, "live")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_model.py::ScenarioModeTests -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'mode'`.

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/scenario.py`, add the import at the top (with the other typing imports):

```python
from typing import Callable, Literal
```

Add the `mode` field to the `Scenario` dataclass (after `description`):

```python
@dataclass
class Scenario:
    id: str
    entry: str
    agents: list[Agent]
    description: str = ""
    mode: Literal["deterministic", "live"] = "deterministic"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_model.py -v`
Expected: PASS — all model tests including the new `ScenarioModeTests`.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario.py tests/test_scenario_model.py
git commit -m "feat(scenario): per-scenario mode field (deterministic|live)"
```

---

## Task 3: `config.scenario_mode` server override

A server-level override that wins over the per-scenario default when set.

**Files:**
- Modify: `simple_chatbot/config.py` (the `SimpleChatbotConfig` model)
- Test: `tests/test_config_scenario_mode.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_scenario_mode.py
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simple_chatbot.config import SimpleChatbotConfig


class ScenarioModeConfigTests(unittest.TestCase):
    def test_defaults_to_none(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(docs_dir=Path(tmp) / "docs")
            self.assertIsNone(config.scenario_mode)

    def test_accepts_live(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(
                docs_dir=Path(tmp) / "docs", scenario_mode="live"
            )
            self.assertEqual(config.scenario_mode, "live")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_scenario_mode.py -v`
Expected: FAIL — pydantic rejects the unknown field `scenario_mode`.

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/config.py`, add the field to `SimpleChatbotConfig` (after `max_tool_rounds` / `system_prompt`). `Literal` is already imported in this file:

```python
    # Scenario engine: server-level override of a scenario's own `mode`.
    # None = honor each scenario's per-scenario default.
    scenario_mode: Literal["deterministic", "live"] | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config_scenario_mode.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/config.py tests/test_config_scenario_mode.py
git commit -m "feat(config): scenario_mode server override"
```

---

## Task 4: Orchestrator resume (`start_agent`) + `ChatResult.active_agent`

The orchestrator can start a turn in a given agent (resumed from a prior turn), re-emits the escalation when resumed directly into a terminal agent, falls back to `entry` on an unknown name, and reports the in-charge agent at turn end.

**Files:**
- Modify: `simple_chatbot/agent.py` (the `ChatResult` dataclass, lines 63-73)
- Modify: `simple_chatbot/scenario_orchestrator.py` (the `chat` method, lines 36-99)
- Test: `tests/test_scenario_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_scenario_orchestrator.py
class OrchestratorResumeTests(unittest.TestCase):
    def _scenario(self):
        return Scenario(
            id="cs",
            entry="dispatcher",
            agents=[
                Agent("dispatcher", routes=["billing", "human"], script=[Route("billing")]),
                Agent(
                    "billing",
                    tools=[lookup_invoice],
                    script=[Call(lookup_invoice, {"invoice_id": "INV-1"}), Final("Refunded.")],
                ),
                Agent("human", terminal=True, escalation_message="Escalating to a human."),
            ],
        )

    def _run(self, scenario, messages, start_agent=None):
        orch = ScenarioOrchestrator(scenario, DeterministicProvider())
        return asyncio.run(orch.chat(messages, start_agent=start_agent))

    def test_reports_active_agent_at_turn_end(self):
        result = self._run(self._scenario(), [{"role": "user", "content": "x"}])
        self.assertEqual(result.active_agent, "billing")

    def test_resume_starts_in_given_agent_without_routing(self):
        result = self._run(
            self._scenario(), [{"role": "user", "content": "x"}], start_agent="billing"
        )
        call_names = [
            m["tool_calls"][0]["function"]["name"]
            for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        self.assertNotIn("route", call_names)  # did NOT re-enter dispatcher
        self.assertEqual(call_names[0], "lookup_invoice")
        self.assertEqual(result.content, "Refunded.")
        self.assertEqual(result.active_agent, "billing")

    def test_unknown_start_agent_falls_back_to_entry(self):
        result = self._run(
            self._scenario(), [{"role": "user", "content": "x"}], start_agent="ghost"
        )
        first_call = next(
            m["tool_calls"][0]["function"]["name"]
            for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        )
        self.assertEqual(first_call, "route")  # entry dispatcher ran

    def test_resume_into_terminal_reemits_escalation(self):
        result = self._run(
            self._scenario(), [{"role": "user", "content": "x"}], start_agent="human"
        )
        self.assertEqual(result.content, "Escalating to a human.")
        self.assertEqual(result.tool_messages, [])  # no rounds executed
        self.assertEqual(result.active_agent, "human")

    def test_chat_without_start_agent_unchanged(self):
        # default path (no start_agent) still enters the entry agent
        result = self._run(self._scenario(), [{"role": "user", "content": "x"}])
        first_call = next(
            m["tool_calls"][0]["function"]["name"]
            for m in result.tool_messages
            if m["role"] == "assistant" and m.get("tool_calls")
        )
        self.assertEqual(first_call, "route")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_orchestrator.py::OrchestratorResumeTests -v`
Expected: FAIL — `chat()` has no `start_agent` parameter / `ChatResult` has no `active_agent`.

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/agent.py`, add the field to `ChatResult` (after `responses_tools`):

```python
    responses_tools: list[dict] = field(default_factory=list)
    active_agent: str | None = None
```

In `simple_chatbot/scenario_orchestrator.py`, replace the entire `chat` method (lines 36-99) with this version (adds `start_agent`, a `_resolve_start` helper, the terminal-on-resume branch, and `active_agent` on the result):

```python
    async def chat(self, messages: list[dict], start_agent: str | None = None) -> ChatResult:
        turn_idx = sum(1 for m in messages if m.get("role") == "user")
        active = self._resolve_start(start_agent)
        tool_messages: list[dict] = []
        content = ""
        seq = 0

        if active.terminal:
            # Resuming directly into a terminal agent re-emits its escalation
            # and ends the turn (no rounds run).
            content = active.escalation_message or "(escalated)"
        else:
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
            active_agent=active.name,
        )

    def _resolve_start(self, start_agent: str | None):
        """Resolve the agent a turn starts in: the requested resume agent when
        valid, else the scenario's entry agent."""
        if start_agent:
            try:
                return self.scenario.agent(start_agent)
            except KeyError:
                logger.bind(
                    scenario=self.scenario.id, start_agent=start_agent
                ).warning("Unknown start_agent; falling back to entry")
        return self.scenario.agent(self.scenario.entry)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_orchestrator.py -v`
Expected: PASS — the new `OrchestratorResumeTests` plus all pre-existing orchestrator tests (routing, terminal, knobs) stay green.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/agent.py simple_chatbot/scenario_orchestrator.py tests/test_scenario_orchestrator.py
git commit -m "feat(scenario): orchestrator resume via start_agent + report active_agent"
```

---

## Task 5: `LiveProvider`

A model-driven provider behind the existing `decide()` seam. Offers only the active agent's scoped tools (in the nested Chat-Completions shape litellm expects) plus an own-`routes` `route` tool, calls the injected `acompletion`, and maps the response into a `ProviderDecision`. Ignores `script`.

**Files:**
- Modify: `simple_chatbot/scenario_provider.py`
- Test: `tests/test_scenario_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_scenario_provider.py
from simple_chatbot.scenario import ROUTE_TOOL_NAME
from simple_chatbot.scenario_provider import LiveProvider


class _Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, arguments):
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, content="", tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, message, finish_reason):
        self.choices = [_Choice(message, finish_reason)]
        self.usage = None


def _live(captured, response=None, raises=None):
    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        if raises is not None:
            raise raises
        return response
    return LiveProvider(acompletion=fake_acompletion, model="test-model")


class LiveProviderTests(unittest.TestCase):
    def test_offers_only_active_agent_scoped_tools_nested_shape(self):
        agent = Agent(
            "billing",
            system_prompt="You handle billing.",
            tools=[lookup_invoice],
            routes=["human"],
        )
        captured: dict = {}
        provider = _live(captured, _Response(_Message(content="hi"), "stop"))
        asyncio.run(provider.decide(agent, messages=[{"role": "user", "content": "q"}], tool_messages=[]))

        offered = {t["function"]["name"] for t in captured["tools"]}
        self.assertEqual(offered, {"lookup_invoice", "route"})
        # nested Chat-Completions shape (not the flat Response.tools shape)
        inv = next(t for t in captured["tools"] if t["function"]["name"] == "lookup_invoice")
        self.assertEqual(inv["type"], "function")
        self.assertIn("parameters", inv["function"])
        # route tool enum is THIS agent's own routes
        route = next(t for t in captured["tools"] if t["function"]["name"] == ROUTE_TOOL_NAME)
        self.assertEqual(route["function"]["parameters"]["properties"]["agent"]["enum"], ["human"])
        # active agent's system prompt is prepended
        self.assertEqual(captured["messages"][0], {"role": "system", "content": "You handle billing."})

    def test_no_route_tool_when_agent_has_no_routes(self):
        agent = Agent("billing", tools=[lookup_invoice])
        captured: dict = {}
        provider = _live(captured, _Response(_Message(content="hi"), "stop"))
        asyncio.run(provider.decide(agent, messages=[], tool_messages=[]))
        offered = {t["function"]["name"] for t in captured["tools"]}
        self.assertEqual(offered, {"lookup_invoice"})

    def test_tool_call_response_maps_to_calls(self):
        agent = Agent("billing", tools=[lookup_invoice])
        resp = _Response(
            _Message(tool_calls=[_ToolCall("lookup_invoice", '{"invoice_id": "INV-9"}')]),
            "tool_calls",
        )
        d = asyncio.run(_live({}, resp).decide(agent, messages=[], tool_messages=[]))
        self.assertIsNone(d.final)
        self.assertEqual(d.calls[0].name, "lookup_invoice")
        self.assertEqual(json.loads(d.calls[0].arguments), {"invoice_id": "INV-9"})

    def test_text_response_maps_to_final(self):
        agent = Agent("billing", tools=[lookup_invoice])
        resp = _Response(_Message(content="All done.", reasoning_content="because"), "stop")
        d = asyncio.run(_live({}, resp).decide(agent, messages=[], tool_messages=[]))
        self.assertEqual(d.final, "All done.")
        self.assertEqual(d.reasoning, "because")

    def test_acompletion_exception_degrades_to_final(self):
        agent = Agent("billing", tools=[lookup_invoice])
        d = asyncio.run(_live({}, raises=RuntimeError("boom")).decide(agent, messages=[], tool_messages=[]))
        self.assertIsNotNone(d.final)
        self.assertIn("boom", d.final)
        self.assertFalse(d.calls)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_provider.py::LiveProviderTests -v`
Expected: FAIL — `ImportError: cannot import name 'LiveProvider'`.

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/scenario_provider.py`, add `ROUTE_TOOL_NAME` to the existing scenario import and append the `LiveProvider` class:

```python
from simple_chatbot.scenario import (
    ROUTE_TOOL_NAME,
    Agent,
    Call,
    Final,
    MalformedCall,
    Route,
    UnknownToolCall,
)
```

```python
class LiveProvider:
    """Model-driven provider: calls `acompletion` once per round with the active
    agent's system prompt + scoped tools, ignoring the authored `script`.

    Tools are offered in the NESTED Chat-Completions shape
    `{type, function: {name, description, parameters}}` (what litellm's `tools=`
    expects) — distinct from the flat `Response.tools` catalog shape. The `route`
    tool's enum is the ACTIVE agent's own routes, not the scenario-wide union.
    """

    def __init__(
        self,
        acompletion,
        model: str,
        api_base: str | None = None,
        sampling_kwargs: dict | None = None,
    ) -> None:
        self._acompletion = acompletion
        self._model = model
        self._api_base = api_base
        self._sampling_kwargs = sampling_kwargs or {}

    def _offered_tools(self, agent: Agent) -> list[dict]:
        tools: list[dict] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in agent.tools
        ]
        if agent.routes:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": ROUTE_TOOL_NAME,
                        "description": "Hand off the request to another agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "agent": {"type": "string", "enum": list(agent.routes)}
                            },
                            "required": ["agent"],
                        },
                    },
                }
            )
        return tools

    async def decide(
        self, agent: Agent, messages: list[dict], tool_messages: list[dict]
    ) -> ProviderDecision:
        working: list[dict] = []
        if agent.system_prompt:
            working.append({"role": "system", "content": agent.system_prompt})
        working.extend(messages)
        working.extend(tool_messages)

        kwargs: dict = {
            "model": self._model,
            "messages": working,
            "tools": self._offered_tools(agent),
            "tool_choice": "auto",
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        kwargs.update(self._sampling_kwargs)

        try:
            response = await self._acompletion(**kwargs)
        except Exception as exc:  # live is best-effort: degrade, don't 500
            return ProviderDecision(final=f"(live provider error: {exc})")

        choice = response.choices[0]
        msg = choice.message
        reasoning = getattr(msg, "reasoning_content", None)
        tool_calls = getattr(msg, "tool_calls", None)
        if choice.finish_reason == "tool_calls" and tool_calls:
            calls = []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                calls.append(
                    PlannedCall(
                        name=getattr(fn, "name", None),
                        arguments=getattr(fn, "arguments", "{}"),
                    )
                )
            return ProviderDecision(calls=calls, reasoning=reasoning)
        return ProviderDecision(final=(msg.content or ""), reasoning=reasoning)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_provider.py -v`
Expected: PASS — `LiveProviderTests` plus the pre-existing `DeterministicProviderTests`.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scenario_provider.py tests/test_scenario_provider.py
git commit -m "feat(scenario): LiveProvider driving litellm per round behind decide() seam"
```

---

## Task 6: Server wiring — provider selection by mode + active-agent persistence

`/v1/responses` selects `LiveProvider` vs `DeterministicProvider` from the resolved mode, passes the resumed agent on chained turns, and persists the in-charge agent.

**Files:**
- Modify: `simple_chatbot/server.py` (imports, globals, `init`, `responses_create`)
- Test: `tests/test_scenario_endpoint.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_scenario_endpoint.py

# These fakes mirror litellm's response object shape for the live path.
class _LiveFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _LiveToolCall:
    def __init__(self, name, arguments):
        self.function = _LiveFunction(name, arguments)


class _LiveMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _LiveChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _LiveResponse:
    def __init__(self, message, finish_reason):
        self.choices = [_LiveChoice(message, finish_reason)]
        self.usage = None


class ScenarioModeAndResumeTests(ScenarioEndpointTests):
    def test_deterministic_resume_stays_in_routed_agent(self):
        # turn 1: dispatcher routes to billing, billing answers
        r1 = self.client.post("/v1/responses", json={"model": "cs-routing", "input": "refund"})
        self.assertEqual(r1.status_code, 200, r1.text)
        rid = r1.json()["id"]
        # turn 2: chained — must resume in billing (no route call), per the requirement
        r2 = self.client.post(
            "/v1/responses",
            json={"model": "cs-routing", "input": "and the receipt?", "previous_response_id": rid},
        )
        self.assertEqual(r2.status_code, 200, r2.text)
        call_names = [it["name"] for it in r2.json()["output"] if it["type"] == "function_call"]
        self.assertNotIn("route", call_names)
        self.assertEqual(call_names[0], "lookup_invoice")

    def test_server_mode_override_selects_live_provider(self):
        # force the whole server live; inject a fake acompletion that finalizes immediately
        server._config.scenario_mode = "live"

        async def fake_acompletion(**kwargs):
            return _LiveResponse(_LiveMessage(content="live answer"), "stop")

        server._acompletion = fake_acompletion
        resp = self.client.post("/v1/responses", json={"model": "cs-routing", "input": "refund"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["output"][-1]["content"][0]["text"], "live answer")
        # catalog still published in live mode
        self.assertIn("route", {t["name"] for t in body["tools"]})
```

Also update the `setUp`/`_restore` in the existing `ScenarioEndpointTests` to save and restore `server._acompletion` so the live-override test cannot leak into other tests. Change the `self.old = (...)` tuple and `_restore` accordingly:

```python
        self.old = (
            server._config, server._agent, server._conversation_logger,
            getattr(server, "_response_store", None),
            getattr(server, "_scenario_registry", None),
            getattr(server, "_acompletion", None),
        )
```

```python
    def _restore(self):
        (server._config, server._agent, server._conversation_logger,
         server._response_store, server._scenario_registry,
         server._acompletion) = self.old
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_endpoint.py::ScenarioModeAndResumeTests -v`
Expected: FAIL — `test_deterministic_resume_stays_in_routed_agent` sees a `route` call on turn 2 (re-enters dispatcher), and `test_server_mode_override_selects_live_provider` fails (no `_acompletion` global / no live selection).

- [ ] **Step 3: Write minimal implementation**

In `simple_chatbot/server.py`:

(a) Add imports near the other imports (top of file):

```python
import litellm

from simple_chatbot.agent import Agent, sampling_kwargs
from simple_chatbot.scenario_provider import DeterministicProvider, LiveProvider
```

(The existing `from simple_chatbot.agent import Agent` and `from simple_chatbot.scenario_provider import DeterministicProvider` lines should be replaced by these — do not duplicate the imports.)

(b) Add the `_acompletion` global (after `_scenario_registry`):

```python
_scenario_registry: dict | None = None
_acompletion: Callable[..., Awaitable[Any]] | None = None
```

(c) In `init`, add `_acompletion` to the `global` statement and capture it:

```python
    global _config, _agent, _conversation_logger, _response_store, _scenario_registry, _acompletion
```

and, where the body sets the other globals (e.g. right after `_agent = Agent(...)`):

```python
    _acompletion = acompletion
```

(d) In `responses_create`, replace the scenario-selection block (currently):

```python
        registry = _scenario_registry or {}
        scenario = registry.get(body.model)
        if scenario is not None:
            orchestrator = ScenarioOrchestrator(scenario, DeterministicProvider())
            result = await orchestrator.chat(messages)
        else:
            result = await _require_agent().chat(messages)
```

with:

```python
        registry = _scenario_registry or {}
        scenario = registry.get(body.model)
        if scenario is not None:
            resolved_mode = (_config.scenario_mode if _config else None) or scenario.mode
            if resolved_mode == "live":
                provider = LiveProvider(
                    acompletion=_acompletion or litellm.acompletion,
                    model=_require_config().chat_model,
                    api_base=_require_config().chat_api_base,
                    sampling_kwargs=sampling_kwargs(_require_config()),
                )
            else:
                provider = DeterministicProvider()
            orchestrator = ScenarioOrchestrator(scenario, provider)
            start_agent = prior_entry.get("active_agent") if prior_entry else None
            result = await orchestrator.chat(messages, start_agent=start_agent)
        else:
            result = await _require_agent().chat(messages)
```

(e) In `responses_create`, persist the in-charge agent — add `active_agent` to the response-store `put` dict (in the existing `await _require_response_store().put(...)` call):

```python
        await _require_response_store().put(
            payload["id"],
            {
                "response_id": payload["id"],
                "previous_response_id": body.previous_response_id,
                "session_messages": list(result.final_messages),
                "response_json": payload,
                "conversation_id": conversation_id,
                "active_agent": result.active_agent,
            },
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_endpoint.py -v`
Expected: PASS — the new `ScenarioModeAndResumeTests` plus all pre-existing endpoint tests (`test_known_model_runs_scenario`, `test_unknown_model_falls_back_to_agent`, `test_models_lists_fixtures`, `ScenarioTraceShapeTests`).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/server.py tests/test_scenario_endpoint.py
git commit -m "feat(server): mode-based provider selection + active-agent persistence"
```

---

## Task 7: Full-suite regression gate

Confirm the whole suite is green — no regressions in the Phase 1 scenario tests or the pre-existing agent/responses/scripted suites.

**Files:** none (verification only).

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS — all Phase 1 tests (207) plus the new Phase 2 tests, with no failures or errors. If anything fails, fix the offending task before proceeding; this is the integration gate.

- [ ] **Step 2: Commit (only if a fix was needed)**

```bash
git add -A
git commit -m "test(scenario): Phase 2 live-mode full-suite green"
```

---

## Self-Review

**Spec coverage (Phase 2 design sections):**
- §2 `LiveProvider` (per-round litellm, scoped tools nested shape, own-routes `route` tool, response→decision mapping, graceful error) → Task 5.
- §2 small refactor: module-level `sampling_kwargs` → Task 1.
- §3 active-agent persistence (orchestrator `start_agent`, `ChatResult.active_agent`, terminal-on-resume, entry fallback) → Task 4; server persist + resume → Task 6.
- §4 mode switch (`Scenario.mode`, `config.scenario_mode`, override precedence, provider selection) → Tasks 2, 3, 6.
- §5 key-free CI / graceful live errors → fake `acompletion` throughout; error degradation in Task 5 (`test_acompletion_exception_degrades_to_final`).
- §6 testing (LiveProvider unit, orchestrator resume, server/E2E mode + resume) → Tasks 4, 5, 6; regression gate → Task 7.
- Out of scope (correctly absent): OpenAI Agents SDK, live-mode knob nudges, golden live traces, ghost-backend harvest.

**Placeholder scan:** none — every code step contains complete, runnable code; every run step has an exact command and expected outcome.

**Type consistency check:** `sampling_kwargs(config) -> dict` (Task 1) consumed in Task 6; `Scenario(..., mode=...)` (Task 2) read in Task 6; `SimpleChatbotConfig.scenario_mode` (Task 3) read in Task 6; `ChatResult.active_agent` (Task 4) set by orchestrator (Task 4) and consumed by server persist/resume (Task 6); `ScenarioOrchestrator.chat(messages, start_agent=None)` (Task 4) called with `start_agent` in Task 6; `LiveProvider(acompletion, model, api_base=None, sampling_kwargs=None).decide(agent, messages, tool_messages) -> ProviderDecision` (Task 5) constructed in Task 6; `PlannedCall(name, arguments)` and `ProviderDecision(final, calls, reasoning, ignore_retrieval)` reused unchanged from Phase 1; `ROUTE_TOOL_NAME` imported in Task 5 from `scenario.py`. The live-path fake response shape (`_Message.tool_calls`, `_Choice.finish_reason`, `response.choices[0]`) matches what `LiveProvider.decide` reads and mirrors the existing `tests/test_agent_tool_calls.py` fakes.

**Note on deterministic resume test (Task 6):** turn 1 of `cs-routing` ends in-charge as `billing` (dispatcher routes once, billing finalizes), so the stored `active_agent` is `"billing"`; turn 2 resumes there and replays billing's script from the top (a fresh `DeterministicProvider` per request, as today), emitting `lookup_invoice` first and no `route` — exactly the persistence requirement, verifiable without an API key.
