# Live multi-agent eval fixture with conversation-scoped mock tools

Date: 2026-06-08
Status: Approved (design)

## Problem

We want a new evaluation flow where a **routing agent** (hit first) hands off,
statefully, to one of several **specialized subagents** (e.g. order tracking vs.
customer-data change), each of which uses tools. The point is **not** useful
agents — it is to emit realistic live router→subagent traces with believable
tool usage so the ghost-backend testing platform's supervisors can evaluate
whether a given router+subagent path was correctly exercised.

The scenario-fixture engine already provides almost everything:

- `Scenario` = a set of `Agent`s, each with its own `system_prompt`, scoped
  `tools`, and `routes`. See `simple_chatbot/scenario.py`.
- `ScenarioOrchestrator` runs a single loop with one shared `tool_messages`
  list and one `active` agent; a `route` tool call swaps the active agent. The
  active agent is returned as `active_agent` and resumed on the next turn via
  `start_agent`, i.e. **stateful handoff already works**. See
  `simple_chatbot/scenario_orchestrator.py` and `simple_chatbot/server.py`
  (lines ~358–379, 415–426).
- `LiveProvider` drives each round with the active agent's prompt + scoped
  tools via the configured LLM — i.e. **real, model-driven subagents** already
  exist in `live` mode. See `simple_chatbot/scenario_provider.py`.
- The orchestrator serializes `route` and domain tool calls into
  `function_call` / `function_call_output` output items, which ghost-backend
  normalizes into `ToolCallItem` / `ToolOutputItem`. The **routing path is
  already visible in the trace** (route calls + each subagent's domain-tool
  calls), and the tool catalog is emitted via `build_responses_tools`
  (`simple_chatbot/scenario_catalog.py`).

**The one gap:** tools are stateless stub functions. The orchestrator runs
`tool.func(**args)` (`scenario_orchestrator.py:142`) with no conversation
context, so a mutation like "I changed this customer's email" has nowhere to
live and cannot be reflected on a later turn. Mock state must be keyed by
conversation and survive across turns, while staying isolated across parallel
conversations.

## Goals

- Conversation-scoped, mutating mock state for fixture tools.
- A fixture modeling a router + 3 specialized subagents + human escalation,
  running in `live` mode so the LLM genuinely drives routing and tool use.
- Parallel-safe: concurrent conversations never share state.
- State semantics remain testable without an LLM (the mock tools are plain
  functions, unit-testable directly against a `ConvState`).

## Non-goals (YAGNI)

- No new eval **mode** literal or CLI flag — reuse `live` + the existing
  `route` / `active_agent` handoff.
- No ghost-backend changes. A routing-path supervisor, if wanted, is built
  there separately; this work only ensures simple-chatbot emits a clean,
  evaluable trace.
- No persistence beyond in-memory; tools stay mock (no real backends).

## Design decisions (confirmed)

- **Delegation model:** stateful handoff (swarm) — control transfers to a
  subagent and sticks (via `active_agent`) until handed back.
- **State mechanism:** conversation-scoped state dict injected into tools
  (Approach A below).
- **Topology:** route-back — router ↔ subagents. Each subagent can hand control
  back to the router, which can re-dispatch to a different subagent on a later
  turn. (No direct subagent↔subagent edges.)
- **Subagents:** order-tracking, customer-data, and billing/refunds, plus a
  terminal human-escalation agent.

## Approach A: conversation-scoped state dict injected into tools

A server-level store maps `conversation_id -> ConvState` (a mutable "world").
Each turn fetches its `ConvState` and threads it into tool execution. A tool
opts in to state by declaring a `state` parameter, which `@tool` excludes from
the generated JSON schema and the orchestrator injects at call time. Stateless
tools are unchanged; the wire/catalog shape ghost-backend sees never includes
`state`.

Per-conversation isolation makes parallel conversations collision-free by
construction. The mechanism is orthogonal to the provider, so it works in both
`live` and `deterministic` modes.

(Rejected: Approach B — tools as stateful objects/closures — breaks the clean
function→schema `@tool` pattern and adds per-fixture boilerplate. Approach C —
reconstruct state by replaying prior tool calls from `session_messages` — is
brittle and over-engineered for a mock.)

## Components

### 1. Conversation state store — new `simple_chatbot/conversation_state.py`

- `ConvState` — a thin `dict` subclass representing one conversation's world.
- `ConversationStateStore` — `conversation_id -> ConvState`, lazily created via
  `get_or_create(conversation_id)`, async-lock guarded (mirrors
  `ResponseStore`), with an LRU cap (default 512 conversations) so the harness
  cannot grow unbounded. Server-level singleton, sibling to `ResponseStore`.

### 2. `@tool` opt-in for state — `simple_chatbot/scenario.py`

- The `tool` decorator skips a parameter named `state` (alongside the existing
  `self` skip) when building the parameter schema, and records
  `wants_state=True` on the resulting `ScenarioTool` (new dataclass field,
  default `False`).
- `ScenarioTool` gains `wants_state: bool = False`.

### 3. Orchestrator injection — `simple_chatbot/scenario_orchestrator.py`

- `chat(messages, start_agent=None, state: ConvState | None = None)`.
- In `_execute`, after `_parse_args` succeeds, if `tool.wants_state`, call
  `tool.func(state=state, **args)`; otherwise `tool.func(**args)` as today.
- Route handling, call-id determinism, and trace shape are unchanged.
- When `state is None` (e.g. a caller that doesn't supply one), a stateful tool
  still executes against a throwaway `ConvState()` so behavior is well-defined.

### 4. Server threading — `simple_chatbot/server.py`

- Add a module-level `_conversation_state_store` initialized in `init()`.
- In the scenario branch, fetch
  `state = _conversation_state_store.get_or_create(conversation_id)` and pass
  it to `orchestrator.chat(messages, start_agent=start_agent, state=state)`.
- `conversation_id` is already computed above and is stable across
  `previous_response_id` chaining, so state persists across a session's turns.

### 5. New fixture — `simple_chatbot/fixtures/support_swarm.py`, `mode="live"`

Agents:

- **router** (entry): `routes=["order_tracking","customer_data","billing","human"]`,
  no domain tools; system prompt instructs it to dispatch, not answer.
- **order_tracking**: stateful tools `get_order(order_id, state)`,
  `cancel_order(order_id, state)` (status mutates `open` → `cancelled`);
  `routes=["router"]`.
- **customer_data**: stateful tools `get_customer(state)`,
  `update_customer(field, value, state)` (email/address mutate);
  `routes=["router"]`.
- **billing**: `lookup_invoice`/`issue_refund` reused as stateful (invoice flips
  to `refunded`); `routes=["router"]`.
- **human**: `terminal=True` with an escalation message, like `cs_routing`.

Details:

- Every subagent can `route("router")` (route-back topology).
- Live-only: agents carry no authored `script` (the fixture is exercised
  through `live` mode, where scripts are ignored anyway). State semantics are
  covered by direct tool unit tests and a fake-`acompletion` live test instead.
- Tools lazily seed their slice of the world via `state.setdefault(...)` (one
  known order, one customer, one invoice), so a fresh conversation starts from a
  believable state.

### 6. Tests — `tests/test_support_swarm.py` (unittest, matching existing style)

State semantics (no LLM, no orchestrator — call the tool functions directly):

- **Mutation:** `cancel_order` then `get_order` against one `ConvState` reflects
  `cancelled`.
- **Cross-turn persistence:** a mutation written to a `ConvState` is visible to a
  later call using the same `ConvState`.
- **Isolation:** two distinct `ConvState`s do not interfere.
- **Seeding:** a fresh `ConvState` yields the seeded order/customer/invoice.

Live path (fake `acompletion`, as in existing provider tests):

- **Handoff + injection:** the router routes to a subagent, whose stateful tool
  call reads/writes the turn's `ConvState` and the trace reflects it.
- **Route-back:** subagent → router → a different subagent is reachable across
  scripted fake-LLM rounds.

Store:

- **`ConversationStateStore`:** lazy `get_or_create` returns the same instance
  per id; LRU cap evicts the oldest beyond the cap.

## Data flow

```
POST /v1/responses
  → conversation_id (stable across previous_response_id)
  → ConvState = store.get_or_create(conversation_id)
  → orchestrator.chat(messages, start_agent=active_agent, state=ConvState)
      → live router emits a `route` call → active agent switches
        (persisted as active_agent for the next turn)
      → subagent's live tool calls read/write ConvState
  → trace serialized into `output` items (function_call / function_call_output)
  → ghost-backend normalizes `route` + domain tool calls into the path
    its supervisors score
```

## Risks / notes

- The state store is mutable server-global state, like `ResponseStore`; the LRU
  cap bounds memory. Acceptable for a test harness.
- `live` mode requires a configured chat model/api at runtime; tests inject a
  fake `acompletion`, consistent with existing provider tests.
- If a future need arises for subagent↔subagent handoff or a declared
  scenario-level initial state, both are additive and out of scope here.
