# Responses API endpoint — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `POST /v1/responses` endpoint (plus `GET /v1/responses/{id}`) that speaks OpenAI's Responses API wire format alongside the existing `/v1/chat/completions`, with the full ordered tool-call trace inlined in `output`.

**Architecture:** Thin Responses adapter on top of the existing Agent. The Agent gets additive changes (token-usage accumulation, expose `final_messages`, don't double-prepend system prompt). A new `simple_chatbot/responses.py` module owns request parsing, ChatResult-to-output-item mapping, and an in-memory `ResponseStore` keyed by `response_id` for multi-turn via `previous_response_id`.

**Tech Stack:** Python 3.12+, FastAPI, Pydantic v2, `openai` SDK (typed schema only — no HTTP client use), LiteLLM (existing), unittest + FastAPI TestClient.

**Spec:** [`docs/superpowers/specs/2026-05-26-responses-api-endpoint-design.md`](../specs/2026-05-26-responses-api-endpoint-design.md)

---

## File Structure

| File                                          | Purpose                                                   |
| --------------------------------------------- | --------------------------------------------------------- |
| `pyproject.toml`                              | Add `openai>=1.50.0` dependency                           |
| `simple_chatbot/agent.py`                     | Modify: usage accumulation, `final_messages`, skip-dup-system |
| `simple_chatbot/responses.py` (new)           | Request model, input normalization, output mapping, ResponseStore |
| `simple_chatbot/server.py`                    | Modify: rename auth helper, add `/v1/responses` endpoints |
| `tests/test_agent_usage_and_final_messages.py` (new) | Agent additive-change tests                          |
| `tests/test_responses_mapping.py` (new)       | Pure-unit tests for normalization and output mapping      |
| `tests/test_responses_endpoint.py` (new)      | FastAPI TestClient tests for the new endpoint             |
| `README.md`                                   | Document the new endpoint                                 |
| `AGENTS.md`                                   | List `responses.py` in modules                            |
| `smoke_test.py`                               | Add a Responses API smoke probe                           |

**Note on SDK types:** `ResponseFunctionToolCallOutputItem` IS in the `openai.types.responses.ResponseOutputItem` union, so inlining call+output in `output` is fully SDK-typed. The `Response.output` annotation accepts it directly.

---

## Task 1: Add `openai` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml` under `[project].dependencies`, add `"openai>=1.50.0"` so the list reads:

```toml
dependencies = [
    "litellm>=1.83.4",
    "chromadb>=1.5.7",
    "fastapi>=0.135.3",
    "uvicorn[standard]>=0.44.0",
    "pypdf>=6.10.0",
    "python-docx>=1.2.0",
    "typer>=0.12.0",
    "pydantic>=2.7.0",
    "loguru>=0.7.3",
    "orbitals>=0.1.6",
    "openai>=1.50.0",
]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: success, `openai` resolved (it may already be present as a transitive dep but we want it declared).

- [ ] **Step 3: Smoke-import the types we'll use**

Run:
```bash
uv run python -c "from openai.types.responses import Response, ResponseFunctionToolCall, ResponseOutputMessage, ResponseUsage; from openai.types.responses.response_function_tool_call_output_item import ResponseFunctionToolCallOutputItem; from openai.types.responses.response_output_text import ResponseOutputText; from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add openai SDK for Responses API typed schemas"
```

---

## Task 2: Agent — accumulate token usage across rounds

**Files:**
- Modify: `simple_chatbot/agent.py:66-73` (ChatResult dataclass) and the chat loop
- Create: `tests/test_agent_usage_and_final_messages.py`

The Agent currently logs `response.usage` per round but discards it. We need to sum across all LiteLLM calls (including the forced-final call) and surface the totals.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_usage_and_final_messages.py` with this initial content:

```python
import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from simple_chatbot.agent import Agent
from simple_chatbot.config import SimpleChatbotConfig


class _FakeIndexer:
    def __init__(self) -> None:
        self.search_calls: list[str] = []

    def document_count(self) -> int:
        return 1

    async def search(self, query: str) -> list:
        self.search_calls.append(query)
        return []


class _Function:
    def __init__(self, name, arguments) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id, name, arguments) -> None:
        self.id = id
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, content="", tool_calls=None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self) -> dict:
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ],
        }


class _Choice:
    def __init__(self, message, finish_reason) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt, completion, total) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class _Response:
    def __init__(self, message, finish_reason, usage=None) -> None:
        self.choices = [_Choice(message, finish_reason)]
        self.usage = usage


def _config(tmp: str) -> SimpleChatbotConfig:
    return SimpleChatbotConfig(docs_dir=Path(tmp) / "docs")


class AgentUsageTests(unittest.TestCase):
    def test_usage_sums_across_all_litellm_calls(self):
        with TemporaryDirectory() as tmp:
            agent = Agent(_config(tmp), _FakeIndexer())
            responses = [
                _Response(
                    _Message(tool_calls=[_ToolCall("call_1", "search_documents", '{"query": "a"}')]),
                    "tool_calls",
                    usage=_Usage(10, 5, 15),
                ),
                _Response(_Message(content="done"), "stop", usage=_Usage(20, 7, 27)),
            ]
            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                result = asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))

            self.assertEqual(result.usage["prompt_tokens"], 30)
            self.assertEqual(result.usage["completion_tokens"], 12)
            self.assertEqual(result.usage["total_tokens"], 42)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m unittest tests.test_agent_usage_and_final_messages.AgentUsageTests -v`
Expected: FAIL — `AttributeError: 'ChatResult' object has no attribute 'usage'` (or similar).

- [ ] **Step 3: Add usage field to ChatResult and accumulate in the loop**

In `simple_chatbot/agent.py`, change the `ChatResult` dataclass (lines 66-72) to:

```python
@dataclass
class ChatResult:
    content: str
    retrieved_chunks: list[Document]
    blocked_by_guard: bool = False
    tools: list[dict] = field(default_factory=list)
    tool_messages: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    final_messages: list[dict] = field(default_factory=list)
```

Then in `Agent.chat()`, just before the `for round_num` loop (right after `last_content = ""` around line 130), add:

```python
usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
```

Inside the loop, after the existing `usage = response.usage` block (around line 154-160), accumulate:

```python
if usage:
    usage_totals["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
    usage_totals["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    usage_totals["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
```

Update every existing `return ChatResult(...)` in `agent.py` to pass `usage=usage_totals`. There are three of them (guard-blocked branch, empty-KB branch, success branch, max-rounds branch). For the guard-blocked and empty-KB branches, `usage_totals` doesn't exist yet — pass `usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}` directly OR move `usage_totals` initialization above the guard check.

Recommended: initialize `usage_totals` at the very top of `chat()`, immediately after the log line, so all branches can reference it.

For the forced-final call (around line 237-242), accumulate too if a usage object comes back. Replace the existing block:

```python
try:
    final_response = await litellm.acompletion(**forced_response_kwargs)
    final_usage = final_response.usage
    if final_usage:
        usage_totals["prompt_tokens"] += getattr(final_usage, "prompt_tokens", 0) or 0
        usage_totals["completion_tokens"] += getattr(final_usage, "completion_tokens", 0) or 0
        usage_totals["total_tokens"] += getattr(final_usage, "total_tokens", 0) or 0
    final_response = final_response.choices[0].message.content or ""
except Exception as exc:
    logger.bind(error=str(exc)).warning("Forced final response call failed")
    final_response = ""
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run python -m unittest tests.test_agent_usage_and_final_messages.AgentUsageTests -v`
Expected: PASS

- [ ] **Step 5: Re-run the full unit test suite to confirm nothing regressed**

Run: `uv run python -m unittest discover -s tests`
Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add simple_chatbot/agent.py tests/test_agent_usage_and_final_messages.py
git commit -m "feat(agent): accumulate token usage across LiteLLM calls"
```

---

## Task 3: Agent — expose `final_messages` and skip duplicate system prompt

**Files:**
- Modify: `simple_chatbot/agent.py`
- Modify: `tests/test_agent_usage_and_final_messages.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_usage_and_final_messages.py`:

```python
class AgentFinalMessagesTests(unittest.TestCase):
    def test_final_messages_includes_final_assistant_when_no_tools(self):
        with TemporaryDirectory() as tmp:
            agent = Agent(_config(tmp), _FakeIndexer())
            responses = [_Response(_Message(content="hello"), "stop", usage=None)]
            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                result = asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))

            # Should contain the user input plus the final assistant message
            roles = [m["role"] for m in result.final_messages]
            self.assertEqual(roles[-1], "assistant")
            self.assertEqual(result.final_messages[-1]["content"], "hello")
            self.assertIn({"role": "user", "content": "hi"}, result.final_messages)

    def test_final_messages_includes_tool_loop_in_order(self):
        with TemporaryDirectory() as tmp:
            agent = Agent(_config(tmp), _FakeIndexer())
            responses = [
                _Response(
                    _Message(tool_calls=[_ToolCall("call_1", "search_documents", '{"query": "a"}')]),
                    "tool_calls",
                ),
                _Response(_Message(content="answered"), "stop"),
            ]
            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                result = asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))

            roles = [m["role"] for m in result.final_messages]
            # user, assistant(with tool_calls), tool, assistant(final)
            self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
            self.assertEqual(result.final_messages[-1]["content"], "answered")


class AgentSystemPromptTests(unittest.TestCase):
    def test_does_not_double_prepend_system_prompt(self):
        with TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            cfg = cfg.model_copy(update={"system_prompt": "You are a bot."})
            agent = Agent(cfg, _FakeIndexer())
            responses = [_Response(_Message(content="ok"), "stop")]

            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                # Caller already includes a system message — agent must not add another.
                asyncio.run(agent.chat([
                    {"role": "system", "content": "You are a bot."},
                    {"role": "user", "content": "hi"},
                ]))

            sent_messages = completion.await_args_list[0].kwargs["messages"]
            system_count = sum(1 for m in sent_messages if m.get("role") == "system")
            self.assertEqual(system_count, 1)

    def test_prepends_system_prompt_when_not_already_present(self):
        with TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            cfg = cfg.model_copy(update={"system_prompt": "You are a bot."})
            agent = Agent(cfg, _FakeIndexer())
            responses = [_Response(_Message(content="ok"), "stop")]

            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))

            sent_messages = completion.await_args_list[0].kwargs["messages"]
            self.assertEqual(sent_messages[0]["role"], "system")
            self.assertEqual(sent_messages[0]["content"], "You are a bot.")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest tests.test_agent_usage_and_final_messages -v`
Expected: the new tests FAIL — `final_messages` is empty / system prompt is duplicated.

- [ ] **Step 3: Implement skip-duplicate-system + final_messages**

In `simple_chatbot/agent.py`, find the block that prepends the system prompt (around line 124-128):

```python
if self.config.system_prompt:
    logger.bind(system_prompt_chars=len(self.config.system_prompt)).info("Prepending system prompt")
    working.append({"role": "system", "content": self.config.system_prompt})

working.extend(messages)
```

Replace with:

```python
already_has_system = bool(messages) and messages[0].get("role") == "system"
if self.config.system_prompt and not already_has_system:
    logger.bind(system_prompt_chars=len(self.config.system_prompt)).info("Prepending system prompt")
    working.append({"role": "system", "content": self.config.system_prompt})

working.extend(messages)
```

Then, in EACH branch that returns a `ChatResult`, also build and pass `final_messages`:

- **Guard-blocked branch** (around line 105-110): `final_messages` is the original `messages` (no tool loop happened). Pass `final_messages=list(messages)`.
- **Empty KB branch** (around line 114-118): same — `final_messages=list(messages)`.
- **Success branch** (around line 219-224): append the final assistant message to `working`:

```python
final_messages = list(working) + [{"role": "assistant", "content": last_content}]
return ChatResult(
    content=last_content,
    retrieved_chunks=all_chunks,
    tools=[SEARCH_TOOL],
    tool_messages=tool_messages,
    usage=usage_totals,
    final_messages=final_messages,
)
```

- **Max-rounds branch** (around line 245-250): same pattern, append the forced-final response:

```python
final_messages = list(working) + [{"role": "assistant", "content": final_response}]
return ChatResult(
    content=final_response,
    retrieved_chunks=all_chunks,
    tools=[SEARCH_TOOL],
    tool_messages=tool_messages,
    usage=usage_totals,
    final_messages=final_messages,
)
```

- [ ] **Step 4: Run all agent tests**

Run: `uv run python -m unittest tests.test_agent_usage_and_final_messages tests.test_agent_tool_calls -v`
Expected: all pass.

- [ ] **Step 5: Full suite check**

Run: `uv run python -m unittest discover -s tests`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add simple_chatbot/agent.py tests/test_agent_usage_and_final_messages.py
git commit -m "feat(agent): expose final_messages and avoid duplicate system prompt"
```

---

## Task 4: `responses.py` — input normalization

**Files:**
- Create: `simple_chatbot/responses.py`
- Create: `tests/test_responses_mapping.py`

Goal: convert the request's `input` field (string or list of input items) into the chat-completions messages list the Agent expects.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_responses_mapping.py`:

```python
import unittest

from simple_chatbot.responses import normalize_input, InvalidInputError


class NormalizeInputTests(unittest.TestCase):
    def test_string_input_becomes_user_message(self):
        self.assertEqual(
            normalize_input("hello"),
            [{"role": "user", "content": "hello"}],
        )

    def test_list_of_strings_becomes_user_messages(self):
        self.assertEqual(
            normalize_input(["hi", "again"]),
            [
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "again"},
            ],
        )

    def test_message_item_with_role_user(self):
        self.assertEqual(
            normalize_input([{"type": "message", "role": "user", "content": "hi"}]),
            [{"role": "user", "content": "hi"}],
        )

    def test_message_item_with_role_assistant(self):
        self.assertEqual(
            normalize_input([{"type": "message", "role": "assistant", "content": "ok"}]),
            [{"role": "assistant", "content": "ok"}],
        )

    def test_message_item_with_content_parts(self):
        # Content parts passed through unchanged
        parts = [{"type": "input_text", "text": "hello"}]
        self.assertEqual(
            normalize_input([{"type": "message", "role": "user", "content": parts}]),
            [{"role": "user", "content": parts}],
        )

    def test_mixed_list(self):
        self.assertEqual(
            normalize_input([
                "first",
                {"type": "message", "role": "assistant", "content": "second"},
                {"type": "message", "role": "user", "content": "third"},
            ]),
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ],
        )

    def test_unknown_item_type_raises(self):
        with self.assertRaises(InvalidInputError):
            normalize_input([{"type": "made_up_type", "blob": 1}])

    def test_non_string_non_dict_item_raises(self):
        with self.assertRaises(InvalidInputError):
            normalize_input([42])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m unittest tests.test_responses_mapping -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simple_chatbot.responses'`.

- [ ] **Step 3: Implement the module skeleton**

Create `simple_chatbot/responses.py`:

```python
"""Responses API support: input normalization, output-item mapping, and a small
in-memory store for `previous_response_id` chaining.

This module is intentionally separate from `agent.py` so the agent remains
focused on the chat-completions-shaped tool loop. The Responses adapter calls
the agent unchanged and translates the result.
"""

from __future__ import annotations


class InvalidInputError(ValueError):
    """Raised when a Responses API `input` value can't be normalized."""


def normalize_input(value: str | list) -> list[dict]:
    """Convert a Responses API `input` field into chat-completions messages."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}]

    if not isinstance(value, list):
        raise InvalidInputError(f"`input` must be a string or list, got {type(value).__name__}")

    messages: list[dict] = []
    for idx, item in enumerate(value):
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if isinstance(item, dict):
            item_type = item.get("type", "message")
            if item_type == "message":
                role = item.get("role")
                content = item.get("content")
                if role not in ("user", "assistant", "system"):
                    raise InvalidInputError(f"input[{idx}].role must be user/assistant/system")
                messages.append({"role": role, "content": content})
                continue
            raise InvalidInputError(f"input[{idx}].type {item_type!r} is not supported")
        raise InvalidInputError(f"input[{idx}] must be a string or dict, got {type(item).__name__}")

    return messages
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_responses_mapping -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/responses.py tests/test_responses_mapping.py
git commit -m "feat(responses): add input normalization"
```

---

## Task 5: `responses.py` — output items mapping

**Files:**
- Modify: `simple_chatbot/responses.py`
- Modify: `tests/test_responses_mapping.py`

Goal: convert a `ChatResult` to the ordered `output` array of `function_call`, `function_call_output`, and `message` items.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_responses_mapping.py`:

```python
from simple_chatbot.agent import ChatResult
from simple_chatbot.responses import build_output_items


class _Doc:
    # ChatResult.retrieved_chunks holds Document instances; we don't need them here.
    pass


def _chat_result(
    content: str,
    tool_messages: list[dict] | None = None,
) -> ChatResult:
    return ChatResult(
        content=content,
        retrieved_chunks=[],
        tools=[],
        tool_messages=tool_messages or [],
    )


class BuildOutputItemsTests(unittest.TestCase):
    def test_message_only_when_no_tool_calls(self):
        items = build_output_items(_chat_result("hello"))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["type"], "message")
        self.assertEqual(items[0]["role"], "assistant")
        self.assertEqual(items[0]["content"][0]["text"], "hello")
        self.assertEqual(items[0]["content"][0]["type"], "output_text")

    def test_single_tool_call_pair_and_message(self):
        tool_messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "search_documents", "arguments": '{"query": "x"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "name": "search_documents",
                "content": "RESULT",
            },
        ]
        items = build_output_items(_chat_result("final", tool_messages))
        types = [it["type"] for it in items]
        self.assertEqual(types, ["function_call", "function_call_output", "message"])
        self.assertEqual(items[0]["call_id"], "call_abc")
        self.assertEqual(items[0]["name"], "search_documents")
        self.assertEqual(items[0]["arguments"], '{"query": "x"}')
        self.assertEqual(items[1]["call_id"], "call_abc")
        self.assertEqual(items[1]["output"], "RESULT")
        self.assertEqual(items[2]["content"][0]["text"], "final")

    def test_multiple_tool_calls_interleaved(self):
        tool_messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search_documents", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "search_documents", "content": "R1"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c2", "type": "function", "function": {"name": "search_documents", "arguments": "{}"}},
                    {"id": "c3", "type": "function", "function": {"name": "search_documents", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "name": "search_documents", "content": "R2"},
            {"role": "tool", "tool_call_id": "c3", "name": "search_documents", "content": "R3"},
        ]
        items = build_output_items(_chat_result("done", tool_messages))
        types = [it["type"] for it in items]
        self.assertEqual(
            types,
            [
                "function_call", "function_call_output",
                "function_call", "function_call", "function_call_output", "function_call_output",
                "message",
            ],
        )
        # call_id linkage preserved
        call_ids = [it["call_id"] for it in items if it["type"] in ("function_call", "function_call_output")]
        self.assertEqual(call_ids, ["c1", "c1", "c2", "c3", "c2", "c3"])

    def test_tool_error_appears_in_function_call_output(self):
        tool_messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search_documents", "arguments": "{not"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "search_documents", "content": "Tool error: bad JSON"},
        ]
        items = build_output_items(_chat_result("recovered", tool_messages))
        fco = next(it for it in items if it["type"] == "function_call_output")
        self.assertIn("Tool error", fco["output"])

    def test_each_item_has_unique_stable_id(self):
        items = build_output_items(_chat_result("hi"))
        ids = [it["id"] for it in items]
        # Message item id should start with `msg_`
        self.assertTrue(ids[0].startswith("msg_"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest tests.test_responses_mapping -v`
Expected: FAIL — `build_output_items` not defined.

- [ ] **Step 3: Implement `build_output_items`**

Add to `simple_chatbot/responses.py`:

```python
import uuid

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
)
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_text import ResponseOutputText

from simple_chatbot.agent import ChatResult


def _fc_id() -> str:
    return f"fc_{uuid.uuid4().hex}"


def _fco_id() -> str:
    return f"fco_{uuid.uuid4().hex}"


def _msg_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def build_output_items(result: ChatResult) -> list[dict]:
    """Convert a ChatResult into the ordered Responses API `output` array."""
    items: list[dict] = []

    for msg in result.tool_messages:
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                items.append(
                    ResponseFunctionToolCall(
                        type="function_call",
                        id=_fc_id(),
                        call_id=tc["id"],
                        name=fn.get("name", ""),
                        arguments=fn.get("arguments", "") or "",
                        status="completed",
                    ).model_dump()
                )
        elif role == "tool":
            items.append(
                ResponseFunctionToolCallOutputItem(
                    type="function_call_output",
                    id=_fco_id(),
                    call_id=msg["tool_call_id"],
                    output=msg.get("content") or "",
                    status="completed",
                ).model_dump()
            )

    items.append(
        ResponseOutputMessage(
            id=_msg_id(),
            type="message",
            role="assistant",
            status="completed",
            content=[
                ResponseOutputText(
                    type="output_text",
                    text=result.content or "",
                    annotations=[],
                )
            ],
        ).model_dump()
    )

    return items
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_responses_mapping -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/responses.py tests/test_responses_mapping.py
git commit -m "feat(responses): map ChatResult to Responses API output items"
```

---

## Task 6: `responses.py` — assemble the full Response

**Files:**
- Modify: `simple_chatbot/responses.py`
- Modify: `tests/test_responses_mapping.py`

Goal: assemble the complete Response JSON payload from a `ChatResult`, including id, usage rename, status, and echoed `previous_response_id`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_responses_mapping.py`:

```python
from simple_chatbot.responses import build_response


class BuildResponseTests(unittest.TestCase):
    def test_basic_envelope_fields(self):
        result = ChatResult(
            content="hi",
            retrieved_chunks=[],
            usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        )
        payload = build_response(
            result=result,
            model="openai/gpt-5.4-nano",
            previous_response_id=None,
            conversation_id="conv_abc",
        )
        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["model"], "openai/gpt-5.4-nano")
        self.assertEqual(payload["conversation_id"], "conv_abc")
        self.assertIsNone(payload["previous_response_id"])
        self.assertTrue(payload["id"].startswith("resp_"))
        self.assertIsInstance(payload["created_at"], int)
        # Output: just the final message item
        self.assertEqual([it["type"] for it in payload["output"]], ["message"])

    def test_usage_renamed_to_openai_fields(self):
        result = ChatResult(
            content="hi",
            retrieved_chunks=[],
            usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        )
        payload = build_response(result, "m", None, "conv")
        self.assertEqual(payload["usage"]["input_tokens"], 10)
        self.assertEqual(payload["usage"]["output_tokens"], 4)
        self.assertEqual(payload["usage"]["total_tokens"], 14)

    def test_previous_response_id_echoed(self):
        payload = build_response(
            ChatResult(content="x", retrieved_chunks=[]),
            "m",
            previous_response_id="resp_prior",
            conversation_id="conv",
        )
        self.assertEqual(payload["previous_response_id"], "resp_prior")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest tests.test_responses_mapping.BuildResponseTests -v`
Expected: FAIL — `build_response` not defined.

- [ ] **Step 3: Implement `build_response`**

Append to `simple_chatbot/responses.py`:

```python
import time

from openai.types.responses import Response, ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails


def _resp_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def build_response(
    result: ChatResult,
    model: str,
    previous_response_id: str | None,
    conversation_id: str,
) -> dict:
    """Build the full Responses API response payload from a ChatResult."""
    usage = result.usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    response_obj = Response(
        id=_resp_id(),
        object="response",
        created_at=int(time.time()),
        model=model,
        output=[],  # filled by model_dump merge below
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        status="completed",
        previous_response_id=previous_response_id,
        usage=ResponseUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=usage.get("completion_tokens", 0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=usage.get("total_tokens", 0),
        ),
    )

    payload = response_obj.model_dump()
    payload["output"] = build_output_items(result)
    # Non-standard extension we add at the envelope level (same as chat completions).
    payload["conversation_id"] = conversation_id
    return payload
```

Note: we set `output=[]` on the Response constructor to satisfy validation, then overwrite via `model_dump()` because the items in `build_output_items` are already dumped dicts. This avoids re-validating the union types.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_responses_mapping -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/responses.py tests/test_responses_mapping.py
git commit -m "feat(responses): assemble full Response payload from ChatResult"
```

---

## Task 7: `responses.py` — in-memory `ResponseStore`

**Files:**
- Modify: `simple_chatbot/responses.py`
- Modify: `tests/test_responses_mapping.py`

Goal: a thread-safe (asyncio-lock) dict keyed by response_id, storing what we need for chain reconstruction.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_responses_mapping.py`:

```python
import asyncio

from simple_chatbot.responses import ResponseStore


class ResponseStoreTests(unittest.TestCase):
    def test_put_then_get(self):
        entry = {
            "response_id": "resp_1",
            "previous_response_id": None,
            "session_messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
            "response_json": {"id": "resp_1"},
            "conversation_id": "conv_1",
        }

        async def scenario():
            store = ResponseStore()
            await store.put("resp_1", entry)
            return await store.get("resp_1")

        got = asyncio.run(scenario())
        self.assertEqual(got, entry)

    def test_get_missing_returns_none(self):
        async def scenario():
            store = ResponseStore()
            return await store.get("nope")

        self.assertIsNone(asyncio.run(scenario()))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest tests.test_responses_mapping.ResponseStoreTests -v`
Expected: FAIL — `ResponseStore` not defined.

- [ ] **Step 3: Implement `ResponseStore`**

Append to `simple_chatbot/responses.py`:

```python
import asyncio


class ResponseStore:
    """In-memory mapping of response_id -> stored Response entry.

    Process-local; resets on server restart. Acceptable for the evaluator
    integration which drives all chained turns within a single server lifetime.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._entries: dict[str, dict] = {}

    async def put(self, response_id: str, entry: dict) -> None:
        async with self._lock:
            self._entries[response_id] = entry

    async def get(self, response_id: str) -> dict | None:
        async with self._lock:
            return self._entries.get(response_id)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_responses_mapping -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/responses.py tests/test_responses_mapping.py
git commit -m "feat(responses): add in-memory ResponseStore for chain lookups"
```

---

## Task 8: server — rename auth helper to `_require_auth`

**Files:**
- Modify: `simple_chatbot/server.py`

Trivial rename so the helper is endpoint-agnostic. No test changes expected — the existing tests reach via the endpoint, not the helper directly.

- [ ] **Step 1: Rename in `server.py`**

In `simple_chatbot/server.py`, rename `_require_chat_auth` to `_require_auth`:

```python
def _require_auth(request: Request) -> None:
    ...
```

And update the call site in `chat_completions()` (line 131) to `_require_auth(request)`. Also update the log line inside the helper that says `path="/v1/chat/completions"` — make it more generic:

```python
logger.bind(path=request.url.path).warning("Rejected unauthorized request")
```

- [ ] **Step 2: Run all tests**

Run: `uv run python -m unittest discover -s tests`
Expected: all PASS (the auth tests don't depend on the helper's name).

- [ ] **Step 3: Commit**

```bash
git add simple_chatbot/server.py
git commit -m "refactor(server): rename _require_chat_auth to _require_auth"
```

---

## Task 9: server — add `POST /v1/responses` (single-turn path)

**Files:**
- Modify: `simple_chatbot/server.py`
- Create: `tests/test_responses_endpoint.py`

Goal: wire up the endpoint with single-turn behavior (no `previous_response_id` handling yet — that's Task 10).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_responses_endpoint.py`:

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

import simple_chatbot.server as server
from simple_chatbot.agent import ChatResult
from simple_chatbot.config import SimpleChatbotConfig


class _FakeAgent:
    def __init__(self, *, content="ok", tool_messages=None, usage=None) -> None:
        self.calls: list[list[dict]] = []
        self._content = content
        self._tool_messages = tool_messages or []
        self._usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async def chat(self, messages: list[dict]) -> ChatResult:
        self.calls.append(list(messages))
        # final_messages = the input we received plus the final assistant message
        final = list(messages) + [{"role": "assistant", "content": self._content}]
        return ChatResult(
            content=self._content,
            retrieved_chunks=[],
            tool_messages=self._tool_messages,
            usage=self._usage,
            final_messages=final,
        )


class _FakeConversationLogger:
    async def log(self, conversation_id, messages, response, chunks) -> None:
        return None


class ResponsesEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = (server._config, server._agent, server._conversation_logger, getattr(server, "_response_store", None))
        self.addCleanup(self._restore)

        self.agent = _FakeAgent()
        server._config = SimpleChatbotConfig(
            docs_dir=Path(self.tmp.name) / "docs",
            conversation_log_dir=Path(self.tmp.name) / "conversations",
        )
        server._agent = self.agent
        server._conversation_logger = _FakeConversationLogger()
        # Force a fresh response store for each test
        from simple_chatbot.responses import ResponseStore
        server._response_store = ResponseStore()
        self.client = TestClient(server.app)

    def _restore(self) -> None:
        server._config, server._agent, server._conversation_logger, store = self.old
        server._response_store = store

    def test_single_turn_string_input(self):
        response = self.client.post("/v1/responses", json={"input": "hello"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["object"], "response")
        self.assertEqual(body["status"], "completed")
        self.assertTrue(body["id"].startswith("resp_"))
        # Output: single message item
        types = [it["type"] for it in body["output"]]
        self.assertEqual(types, ["message"])
        self.assertEqual(body["output"][0]["content"][0]["text"], "ok")
        # Agent saw the normalized message
        self.assertEqual(self.agent.calls[0], [{"role": "user", "content": "hello"}])

    def test_single_turn_list_input(self):
        response = self.client.post(
            "/v1/responses",
            json={"input": [{"type": "message", "role": "user", "content": "hello"}]},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        types = [it["type"] for it in body["output"]]
        self.assertEqual(types, ["message"])
        self.assertEqual(self.agent.calls[0], [{"role": "user", "content": "hello"}])

    def test_single_turn_with_tool_call_trace(self):
        self.agent = _FakeAgent(
            content="final answer",
            tool_messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "search_documents", "arguments": '{"query": "x"}'}},
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1",
                 "name": "search_documents", "content": "RESULT"},
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        server._agent = self.agent
        response = self.client.post("/v1/responses", json={"input": "x"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        types = [it["type"] for it in body["output"]]
        self.assertEqual(types, ["function_call", "function_call_output", "message"])
        self.assertEqual(body["output"][0]["call_id"], "call_1")
        self.assertEqual(body["output"][1]["call_id"], "call_1")
        self.assertEqual(body["output"][2]["content"][0]["text"], "final answer")
        self.assertEqual(body["usage"]["input_tokens"], 10)
        self.assertEqual(body["usage"]["output_tokens"], 5)
        self.assertEqual(body["usage"]["total_tokens"], 15)

    def test_streaming_rejected(self):
        response = self.client.post("/v1/responses", json={"input": "x", "stream": True})
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]["error"]
        self.assertEqual(detail["param"], "stream")

    def test_unknown_input_item_type_returns_400(self):
        response = self.client.post(
            "/v1/responses",
            json={"input": [{"type": "not_a_real_thing"}]},
        )
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]["error"]
        self.assertEqual(detail["param"], "input")

    def test_auth_required_when_configured(self):
        server._config = SimpleChatbotConfig(
            docs_dir=Path(self.tmp.name) / "docs",
            conversation_log_dir=Path(self.tmp.name) / "conversations",
            api_key="secret",
        )
        no_auth = self.client.post("/v1/responses", json={"input": "x"})
        self.assertEqual(no_auth.status_code, 401)

        bearer = self.client.post(
            "/v1/responses",
            json={"input": "x"},
            headers={"Authorization": "Bearer secret"},
        )
        self.assertEqual(bearer.status_code, 200)

        xkey = self.client.post(
            "/v1/responses",
            json={"input": "x"},
            headers={"X-API-Key": "secret"},
        )
        self.assertEqual(xkey.status_code, 200)

    def test_tools_field_is_accepted_but_ignored(self):
        response = self.client.post(
            "/v1/responses",
            json={"input": "x", "tools": [{"type": "function", "function": {"name": "irrelevant"}}]},
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest tests.test_responses_endpoint -v`
Expected: FAIL — endpoint doesn't exist (404).

- [ ] **Step 3: Wire up the endpoint in `server.py`**

In `simple_chatbot/server.py`:

1. Near the other imports, add:

```python
from simple_chatbot.responses import (
    InvalidInputError,
    ResponseStore,
    build_response,
    normalize_input,
)
```

2. Add a module-level store global next to the others (line 17-19 area):

```python
_response_store: ResponseStore | None = None
```

3. In `init()`, initialize the store after `_agent` is created:

```python
global _config, _agent, _conversation_logger, _response_store
...
_response_store = ResponseStore()
```

4. Add the Pydantic request model below `ChatCompletionRequest`:

```python
class ResponsesRequest(BaseModel):
    model: str | None = None
    input: str | list = ""
    previous_response_id: str | None = None
    tools: list | None = None
    tool_choice: object = None
    user: str | None = None
    stream: bool = False
    # All other Responses API fields are accepted but ignored.
    model_config = {"extra": "allow"}
```

5. Add the endpoint below `chat_completions`:

```python
@app.post("/v1/responses")
async def responses_create(request: Request, body: ResponsesRequest):
    _require_auth(request)

    if body.stream:
        logger.bind(stream=body.stream).warning("Rejected streaming response request")
        raise HTTPException(
            status_code=400,
            detail=_openai_error(
                "stream=true is not supported by this server yet",
                "invalid_request_error",
                "stream",
            ),
        )

    try:
        new_messages = normalize_input(body.input)
    except InvalidInputError as exc:
        raise HTTPException(
            status_code=400,
            detail=_openai_error(str(exc), "invalid_request_error", "input"),
        ) from exc

    # Single-turn path — chain handling lands in Task 10.
    messages = new_messages

    conversation_id = (
        body.user
        or request.headers.get("x-conversation-id")
        or derive_conversation_id(messages)
    )

    with logger.contextualize(conversation_id=conversation_id):
        logger.bind(
            model=body.model,
            message_count=len(messages),
            previous_response_id=body.previous_response_id,
        ).info("Response create started")

        start = time.perf_counter()
        result = await _agent.chat(messages)
        elapsed_ms = (time.perf_counter() - start) * 1000

        payload = build_response(
            result=result,
            model=_config.chat_model,
            previous_response_id=body.previous_response_id,
            conversation_id=conversation_id,
        )

        logger.bind(
            response_id=payload["id"],
            duration_ms=round(elapsed_ms, 1),
            output_item_count=len(payload["output"]),
        ).info("Response create finished")

        await _conversation_logger.log(
            conversation_id=conversation_id,
            messages=messages,
            response=result.content,
            chunks=[{"text": d.text, **d.metadata} for d in result.retrieved_chunks],
        )

        await _response_store.put(
            payload["id"],
            {
                "response_id": payload["id"],
                "previous_response_id": body.previous_response_id,
                "session_messages": list(result.final_messages),
                "response_json": payload,
                "conversation_id": conversation_id,
            },
        )

        return payload
```

- [ ] **Step 4: Run the test file**

Run: `uv run python -m unittest tests.test_responses_endpoint -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run python -m unittest discover -s tests`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add simple_chatbot/server.py tests/test_responses_endpoint.py
git commit -m "feat(server): add POST /v1/responses (single-turn)"
```

---

## Task 10: server — `previous_response_id` chain

**Files:**
- Modify: `simple_chatbot/server.py`
- Modify: `tests/test_responses_endpoint.py`

Goal: when `previous_response_id` is supplied, prepend the stored `session_messages` from that response so the next agent call sees full history. Inherit `conversation_id`. Validate the id and the input.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_responses_endpoint.py`:

```python
class ResponsesChainTests(ResponsesEndpointTests):
    def test_chained_turn_prepends_prior_session_messages(self):
        first = self.client.post("/v1/responses", json={"input": "hi"})
        self.assertEqual(first.status_code, 200)
        first_body = first.json()
        first_id = first_body["id"]
        first_conv = first_body["conversation_id"]

        # Snapshot the messages the agent saw on turn 1
        turn1_messages = list(self.agent.calls[-1])

        second = self.client.post(
            "/v1/responses",
            json={"input": "follow-up", "previous_response_id": first_id},
        )
        self.assertEqual(second.status_code, 200, second.text)
        second_body = second.json()
        self.assertEqual(second_body["previous_response_id"], first_id)
        # Chained turn inherits the conversation_id
        self.assertEqual(second_body["conversation_id"], first_conv)

        # Turn 2's messages: prior session_messages + new user input
        turn2_messages = self.agent.calls[-1]
        # Should start with the prior turn's user + assistant
        self.assertEqual(turn2_messages[0], {"role": "user", "content": "hi"})
        self.assertEqual(turn2_messages[1], {"role": "assistant", "content": "ok"})
        # Then the new user input
        self.assertEqual(turn2_messages[-1], {"role": "user", "content": "follow-up"})

    def test_unknown_previous_response_id_returns_404(self):
        response = self.client.post(
            "/v1/responses",
            json={"input": "x", "previous_response_id": "resp_does_not_exist"},
        )
        self.assertEqual(response.status_code, 404)
        detail = response.json()["detail"]["error"]
        self.assertEqual(detail["param"], "previous_response_id")

    def test_previous_response_id_with_empty_input_returns_400(self):
        first = self.client.post("/v1/responses", json={"input": "hi"}).json()
        response = self.client.post(
            "/v1/responses",
            json={"input": "", "previous_response_id": first["id"]},
        )
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]["error"]
        self.assertEqual(detail["param"], "input")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest tests.test_responses_endpoint.ResponsesChainTests -v`
Expected: FAIL — chain logic not implemented.

- [ ] **Step 3: Implement chain handling**

In `simple_chatbot/server.py`, inside the `responses_create` function, replace the single-turn assembly section (the part that reads `messages = new_messages` and derives `conversation_id`) with chain-aware logic:

```python
    try:
        new_messages = normalize_input(body.input)
    except InvalidInputError as exc:
        raise HTTPException(
            status_code=400,
            detail=_openai_error(str(exc), "invalid_request_error", "input"),
        ) from exc

    prior_entry = None
    if body.previous_response_id:
        prior_entry = await _response_store.get(body.previous_response_id)
        if prior_entry is None:
            raise HTTPException(
                status_code=404,
                detail=_openai_error(
                    f"previous_response_id {body.previous_response_id!r} not found",
                    "invalid_request_error",
                    "previous_response_id",
                ),
            )
        if not new_messages:
            raise HTTPException(
                status_code=400,
                detail=_openai_error(
                    "input must contain at least one item when previous_response_id is set",
                    "invalid_request_error",
                    "input",
                ),
            )

    if prior_entry is not None:
        messages = list(prior_entry["session_messages"]) + new_messages
        conversation_id = prior_entry["conversation_id"]
    else:
        messages = new_messages
        conversation_id = (
            body.user
            or request.headers.get("x-conversation-id")
            or derive_conversation_id(messages)
        )
```

- [ ] **Step 4: Run the chain tests**

Run: `uv run python -m unittest tests.test_responses_endpoint -v`
Expected: all PASS, including the new ChainTests.

- [ ] **Step 5: Full suite**

Run: `uv run python -m unittest discover -s tests`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add simple_chatbot/server.py tests/test_responses_endpoint.py
git commit -m "feat(server): support previous_response_id chain for multi-turn"
```

---

## Task 11: server — `GET /v1/responses/{response_id}`

**Files:**
- Modify: `simple_chatbot/server.py`
- Modify: `tests/test_responses_endpoint.py`

Goal: retrieve a stored Response by id.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_responses_endpoint.py`:

```python
class ResponsesRetrieveTests(ResponsesEndpointTests):
    def test_get_returns_stored_response(self):
        created = self.client.post("/v1/responses", json={"input": "hi"}).json()
        got = self.client.get(f"/v1/responses/{created['id']}")
        self.assertEqual(got.status_code, 200)
        self.assertEqual(got.json(), created)

    def test_get_unknown_id_returns_404(self):
        got = self.client.get("/v1/responses/resp_unknown")
        self.assertEqual(got.status_code, 404)

    def test_get_respects_auth(self):
        server._config = SimpleChatbotConfig(
            docs_dir=Path(self.tmp.name) / "docs",
            conversation_log_dir=Path(self.tmp.name) / "conversations",
            api_key="secret",
        )
        created = self.client.post(
            "/v1/responses",
            json={"input": "hi"},
            headers={"Authorization": "Bearer secret"},
        ).json()

        no_auth = self.client.get(f"/v1/responses/{created['id']}")
        self.assertEqual(no_auth.status_code, 401)

        with_auth = self.client.get(
            f"/v1/responses/{created['id']}",
            headers={"Authorization": "Bearer secret"},
        )
        self.assertEqual(with_auth.status_code, 200)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest tests.test_responses_endpoint.ResponsesRetrieveTests -v`
Expected: FAIL — endpoint doesn't exist.

- [ ] **Step 3: Add the GET endpoint**

In `simple_chatbot/server.py`, below `responses_create`, add:

```python
@app.get("/v1/responses/{response_id}")
async def responses_retrieve(response_id: str, request: Request):
    _require_auth(request)
    entry = await _response_store.get(response_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=_openai_error(
                f"response {response_id!r} not found",
                "invalid_request_error",
                "response_id",
            ),
        )
    return entry["response_json"]
```

- [ ] **Step 4: Run the tests**

Run: `uv run python -m unittest tests.test_responses_endpoint -v`
Expected: all PASS.

- [ ] **Step 5: Full suite check**

Run: `uv run python -m unittest discover -s tests`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add simple_chatbot/server.py tests/test_responses_endpoint.py
git commit -m "feat(server): add GET /v1/responses/{response_id}"
```

---

## Task 12: Documentation and smoke test

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `smoke_test.py`

- [ ] **Step 1: Update `AGENTS.md`**

In `AGENTS.md`, in the "Primary modules" list under "Project Overview", add a bullet for `responses.py` after the `agent.py` line:

```markdown
- `simple_chatbot/responses.py`: Responses API request parsing, output-item mapping, and in-memory response store.
```

Update `simple_chatbot/server.py` line description to mention the new endpoints:

```markdown
- `simple_chatbot/server.py`: FastAPI `/v1/models`, `/v1/chat/completions`, and `/v1/responses` endpoints.
```

- [ ] **Step 2: Update `README.md`**

Add a section after the existing Chat Completions docs (find the most natural spot — likely after the "API" or "Usage" section). Insert:

````markdown
### Responses API

In addition to `/v1/chat/completions`, the server exposes
`POST /v1/responses` and `GET /v1/responses/{id}` implementing OpenAI's
Responses API wire format. Every tool call and tool result appears in the
response's `output` array, in the order it happened, so downstream evaluators
can inspect the full trace.

**Single turn:**

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input": "What does the doc say about onboarding?"}'
```

Response (abbreviated):

```json
{
  "id": "resp_...",
  "object": "response",
  "status": "completed",
  "output": [
    {"type": "function_call", "call_id": "call_1", "name": "search_documents", "arguments": "{\"query\": \"onboarding\"}"},
    {"type": "function_call_output", "call_id": "call_1", "output": "..."},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "...", "annotations": []}]}
  ],
  "usage": {"input_tokens": 123, "output_tokens": 45, "total_tokens": 168}
}
```

**Multi-turn:** Pass the prior response's `id` as `previous_response_id`. Only
new input items are needed; the server reconstructs prior context:

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input": "tell me more", "previous_response_id": "resp_..."}'
```

**Notes:**
- `tools` and `tool_choice` in the request are accepted but ignored — the
  server always exposes its built-in `search_documents` tool.
- Streaming (`stream: true`) is not supported.
- Response state is stored in-memory; `previous_response_id` chains do not
  survive a server restart.
````

- [ ] **Step 3: Add a Responses probe to `smoke_test.py`**

In `smoke_test.py`, inside `main()` after the existing chat-completions block (after the `print("PASS")` line is too late — put it before the success print), add:

```python
        # Responses API probe
        resp_payload = {"input": args.message}
        print(f"→ POST {base}/v1/responses (input={args.message!r})")
        resp_body = _post_json(f"{base}/v1/responses", resp_payload, timeout=args.timeout)
        assert resp_body.get("object") == "response", f"unexpected /v1/responses response: {resp_body}"
        output_items = resp_body.get("output") or []
        message_items = [it for it in output_items if it.get("type") == "message"]
        assert message_items, "no message item in output"
        text = message_items[-1]["content"][0]["text"]
        assert text, "empty assistant text"
        print(f"  ok — response_id={resp_body['id']}, output_item_count={len(output_items)}")
        print(f"  assistant: {text!r}")
```

- [ ] **Step 4: Run all unit tests one last time**

Run: `uv run python -m unittest discover -s tests`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md AGENTS.md smoke_test.py
git commit -m "docs: document Responses API endpoint and add smoke probe"
```

---

## Self-Review Notes

After all tasks complete, verify:

- The spec's "Output items mapping" section matches Task 5's implementation exactly.
- `function_call.call_id` equals the matching `function_call_output.call_id` in tests (Task 5 and Task 9 both assert this).
- `previous_response_id` chains use stored `session_messages` from the prior entry as the prefix (Task 10).
- `ChatResult.usage` is summed across all rounds including the forced-final call (Task 2).
- `_require_auth` (renamed in Task 8) is called on both `POST /v1/responses` and `GET /v1/responses/{id}` (Tasks 9 and 11).
- LiteLLM `prompt_tokens` / `completion_tokens` are renamed to `input_tokens` / `output_tokens` in the Response payload (Task 6).
- No new module imports anything that wasn't added in Task 1.
