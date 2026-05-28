# Exhaustive tool-use mode — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `EXHAUSTIVE_TOOL_USE` env var that makes scripted offline mode showcase every mocked tool and every trace shape over a 5-turn rotation, deterministic-by-turn-index.

**Architecture:** Single-file change to `simple_chatbot/scripted_llm.py`. A new turn-index-based rotation lookup synthesizes `(markers, forced_tools)` overrides that feed into the existing parallel/multi-round/error/reasoning dispatch unchanged. Explicit bracketed markers in the user's text still win.

**Tech Stack:** Python 3.12+, unittest, `unittest.mock.patch.dict` for env-var isolation.

**Spec:** [`docs/superpowers/specs/2026-05-28-exhaustive-tool-use-mode-design.md`](../specs/2026-05-28-exhaustive-tool-use-mode-design.md)

---

## File Structure

| File                                          | Purpose                                                   |
| --------------------------------------------- | --------------------------------------------------------- |
| `simple_chatbot/scripted_llm.py`              | Modify: add env-var reader, `_count_user_turns`, `_exhaustive_overrides`, rotation table, wire into `acompletion()`. Repurpose unused `any_explicit` property into a `bracketed` field. |
| `tests/test_scripted_mode.py`                 | Modify: add `ExhaustiveModeTests` class                   |
| `README.md`                                   | Modify: paragraph in "Multi-tool evaluator harness" section |
| `AGENTS.md`                                   | Modify: bullet under `scripted_llm.py` summary line       |

---

## Task 1: Replace unused `any_explicit` with a `bracketed` field on `_Markers`

The existing `_Markers.any_explicit` property is unused (verified via grep). It conflates bracketed markers with the "step by step" heuristic reasoning trigger — using it to gate exhaustive overrides would let heuristic reasoning falsely suppress them. Replace it with a `bracketed` field set by `_parse_markers` only when an actual `[...]` marker matched.

**Files:**
- Modify: `simple_chatbot/scripted_llm.py:72-107`
- Test: `tests/test_scripted_mode.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_scripted_mode.py` inside `class MarkerDispatchTests`:

```python
    def test_parse_markers_sets_bracketed_when_bracket_present(self):
        from simple_chatbot.scripted_llm import _parse_markers
        markers, _ = _parse_markers("[parallel] hello")
        self.assertTrue(markers.bracketed)

    def test_parse_markers_bracketed_false_without_brackets(self):
        from simple_chatbot.scripted_llm import _parse_markers
        markers, _ = _parse_markers("hello world")
        self.assertFalse(markers.bracketed)

    def test_parse_markers_heuristic_reasoning_does_not_set_bracketed(self):
        from simple_chatbot.scripted_llm import _parse_markers
        markers, _ = _parse_markers("explain step by step")
        self.assertTrue(markers.reasoning)
        self.assertFalse(markers.bracketed)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/nick/projects/simple-chatbot
uv run python -m unittest tests.test_scripted_mode.MarkerDispatchTests.test_parse_markers_sets_bracketed_when_bracket_present -v
```

Expected: FAIL with `AttributeError: '_Markers' object has no attribute 'bracketed'`.

- [ ] **Step 3: Modify `_Markers` and `_parse_markers`**

In `simple_chatbot/scripted_llm.py`, replace the `_Markers` dataclass (lines 72-81) and `_parse_markers` function (lines 84-107) with:

```python
@dataclass
class _Markers:
    parallel: bool = False
    reasoning: bool = False
    multi_round: bool = False
    error: bool = False
    # True iff at least one bracketed marker (e.g. "[parallel]") appeared in
    # the user's text. Heuristic phrases like "step by step" do NOT set this.
    bracketed: bool = False


def _parse_markers(text: str) -> tuple[_Markers, str]:
    """Extract markers from `text` and return (markers, text_with_markers_stripped)."""
    markers = _Markers()

    def _capture(m: re.Match) -> str:
        markers.bracketed = True
        token = m.group(1).lower().replace("-", "")
        if token == "parallel":
            markers.parallel = True
        elif token == "reasoning":
            markers.reasoning = True
        elif token == "multiround":
            markers.multi_round = True
        elif token == "error":
            markers.error = True
        return ""

    cleaned = _MARKER_RE.sub(_capture, text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Heuristic reasoning triggers (only when no explicit marker yet)
    if not markers.reasoning:
        lowered = cleaned.lower()
        if any(p in lowered for p in ["step by step", "think it through", "let's reason"]):
            markers.reasoning = True
    return markers, cleaned
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m unittest tests.test_scripted_mode -v
```

Expected: all existing tests still pass + the 3 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scripted_llm.py tests/test_scripted_mode.py
git commit -m "$(cat <<'EOF'
refactor(scripted_llm): track bracketed markers separately from heuristic

Replaces the unused any_explicit property with a bracketed field so
downstream code can distinguish a user-typed [marker] from a heuristic
phrase like "step by step".

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `_is_exhaustive()` env-var reader

Read the env var on each dispatch so tests can flip it with `patch.dict`.

**Files:**
- Modify: `simple_chatbot/scripted_llm.py` (add at top, near other imports)
- Test: `tests/test_scripted_mode.py`

- [ ] **Step 1: Write the failing test**

Add a new class to `tests/test_scripted_mode.py` (at the bottom, before `if __name__ == "__main__":`):

```python
class ExhaustiveModeHelperTests(unittest.TestCase):
    def test_is_exhaustive_true_when_env_set_to_1(self):
        import os
        from unittest.mock import patch
        from simple_chatbot.scripted_llm import _is_exhaustive
        with patch.dict(os.environ, {"EXHAUSTIVE_TOOL_USE": "1"}):
            self.assertTrue(_is_exhaustive())

    def test_is_exhaustive_true_for_truthy_values(self):
        import os
        from unittest.mock import patch
        from simple_chatbot.scripted_llm import _is_exhaustive
        for value in ["1", "true", "TRUE", "yes", "on", "  true  "]:
            with patch.dict(os.environ, {"EXHAUSTIVE_TOOL_USE": value}):
                self.assertTrue(_is_exhaustive(), f"expected truthy for {value!r}")

    def test_is_exhaustive_false_for_unset_or_falsy(self):
        import os
        from unittest.mock import patch
        from simple_chatbot.scripted_llm import _is_exhaustive
        # Unset
        env_no_var = {k: v for k, v in os.environ.items() if k != "EXHAUSTIVE_TOOL_USE"}
        with patch.dict(os.environ, env_no_var, clear=True):
            self.assertFalse(_is_exhaustive())
        # Empty / falsy values
        for value in ["", "0", "false", "no", "off", "garbage"]:
            with patch.dict(os.environ, {"EXHAUSTIVE_TOOL_USE": value}):
                self.assertFalse(_is_exhaustive(), f"expected falsy for {value!r}")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m unittest tests.test_scripted_mode.ExhaustiveModeHelperTests -v
```

Expected: FAIL with `ImportError: cannot import name '_is_exhaustive'`.

- [ ] **Step 3: Add the helper**

At the top of `simple_chatbot/scripted_llm.py`, add `import os` to the imports (the file currently imports `json`, `re`, `uuid`). After the imports and before the dataclasses, add:

```python
def _is_exhaustive() -> bool:
    """Return True if EXHAUSTIVE_TOOL_USE is set to a truthy value.

    Read per-dispatch so tests can flip it with `patch.dict(os.environ, ...)`.
    """
    return os.environ.get("EXHAUSTIVE_TOOL_USE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m unittest tests.test_scripted_mode.ExhaustiveModeHelperTests -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scripted_llm.py tests/test_scripted_mode.py
git commit -m "$(cat <<'EOF'
feat(scripted_llm): add EXHAUSTIVE_TOOL_USE env var reader

Reads the env var per-dispatch so tests can flip it without
module-reload gymnastics. No wiring yet — the flag has no effect
until the rotation lands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `_count_user_turns()` helper

**Files:**
- Modify: `simple_chatbot/scripted_llm.py` (add near other `_count_*` helpers)
- Test: `tests/test_scripted_mode.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_scripted_mode.py` inside the `ExhaustiveModeHelperTests` class added in Task 2:

```python
    def test_count_user_turns_empty_messages(self):
        from simple_chatbot.scripted_llm import _count_user_turns
        self.assertEqual(_count_user_turns([]), 0)

    def test_count_user_turns_one_user(self):
        from simple_chatbot.scripted_llm import _count_user_turns
        self.assertEqual(_count_user_turns([{"role": "user", "content": "hi"}]), 1)

    def test_count_user_turns_mixed_roles(self):
        from simple_chatbot.scripted_llm import _count_user_turns
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "content": "r"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "y"},
            {"role": "user", "content": "c"},
        ]
        self.assertEqual(_count_user_turns(msgs), 3)

    def test_count_user_turns_ignores_unknown_roles(self):
        from simple_chatbot.scripted_llm import _count_user_turns
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "a"},
            {"role": "developer", "content": "d"},
        ]
        self.assertEqual(_count_user_turns(msgs), 1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m unittest tests.test_scripted_mode.ExhaustiveModeHelperTests -v
```

Expected: 4 new tests FAIL with `ImportError: cannot import name '_count_user_turns'`.

- [ ] **Step 3: Add the helper**

In `simple_chatbot/scripted_llm.py`, in the multi-round helpers section (just after `_count_rounds_since_latest_user`, around line 178), add:

```python
def _count_user_turns(messages: list[dict]) -> int:
    """Count `user`-role messages in the input. Used to index the exhaustive rotation."""
    return sum(1 for m in messages if m.get("role") == "user")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m unittest tests.test_scripted_mode.ExhaustiveModeHelperTests -v
```

Expected: all 7 tests pass (3 from Task 2 + 4 from Task 3).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scripted_llm.py tests/test_scripted_mode.py
git commit -m "$(cat <<'EOF'
feat(scripted_llm): add _count_user_turns helper for exhaustive rotation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add the rotation table and `_exhaustive_overrides()` lookup

The 5-slot rotation that drives turn-by-turn shape and tool selection.

**Files:**
- Modify: `simple_chatbot/scripted_llm.py`
- Test: `tests/test_scripted_mode.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_scripted_mode.py` inside `ExhaustiveModeHelperTests`:

```python
    def test_exhaustive_overrides_slot_0_all_parallel(self):
        from simple_chatbot.scripted_llm import _exhaustive_overrides
        markers, tools = _exhaustive_overrides(0)
        self.assertTrue(markers.parallel)
        self.assertTrue(markers.reasoning)
        self.assertFalse(markers.multi_round)
        self.assertFalse(markers.error)
        self.assertEqual(
            sorted(tools),
            sorted(["search_documents", "calculate", "get_current_time", "lookup_user"]),
        )

    def test_exhaustive_overrides_slot_1_multi_round(self):
        from simple_chatbot.scripted_llm import _exhaustive_overrides
        markers, tools = _exhaustive_overrides(1)
        self.assertTrue(markers.multi_round)
        self.assertTrue(markers.reasoning)
        self.assertFalse(markers.parallel)
        self.assertFalse(markers.error)
        self.assertEqual(tools, ["search_documents"])

    def test_exhaustive_overrides_slot_2_error(self):
        from simple_chatbot.scripted_llm import _exhaustive_overrides
        markers, tools = _exhaustive_overrides(2)
        self.assertTrue(markers.error)
        self.assertTrue(markers.reasoning)
        self.assertEqual(tools, ["lookup_user"])

    def test_exhaustive_overrides_slot_3_parallel_pair(self):
        from simple_chatbot.scripted_llm import _exhaustive_overrides
        markers, tools = _exhaustive_overrides(3)
        self.assertTrue(markers.parallel)
        self.assertTrue(markers.reasoning)
        self.assertEqual(tools, ["calculate", "get_current_time"])

    def test_exhaustive_overrides_slot_4_plain_single(self):
        from simple_chatbot.scripted_llm import _exhaustive_overrides
        markers, tools = _exhaustive_overrides(4)
        self.assertTrue(markers.reasoning)
        self.assertFalse(markers.parallel)
        self.assertFalse(markers.multi_round)
        self.assertFalse(markers.error)
        self.assertEqual(tools, ["search_documents"])

    def test_exhaustive_overrides_wraps_modulo_5(self):
        from simple_chatbot.scripted_llm import _exhaustive_overrides
        for base in [0, 1, 2, 3, 4]:
            m_a, t_a = _exhaustive_overrides(base)
            m_b, t_b = _exhaustive_overrides(base + 5)
            self.assertEqual((m_a.parallel, m_a.reasoning, m_a.multi_round, m_a.error),
                             (m_b.parallel, m_b.reasoning, m_b.multi_round, m_b.error))
            self.assertEqual(t_a, t_b)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m unittest tests.test_scripted_mode.ExhaustiveModeHelperTests -v
```

Expected: 6 new tests FAIL with `ImportError: cannot import name '_exhaustive_overrides'`.

- [ ] **Step 3: Add the rotation table and lookup**

In `simple_chatbot/scripted_llm.py`, just after `_count_user_turns` from Task 3, add:

```python
# Exhaustive-mode rotation: 5 slots, indexed by user-turn index (mod 5).
# Each slot = (marker keyword args, forced tool list). The first user turn
# (single user message in history) is slot 0; subsequent turns advance.
_EXHAUSTIVE_ROTATION: list[tuple[dict, list[str]]] = [
    # slot 0: all-parallel + reasoning, every tool fires once
    (
        {"parallel": True, "reasoning": True},
        ["search_documents", "calculate", "get_current_time", "lookup_user"],
    ),
    # slot 1: multi-round + reasoning; round 1 = search, round 2 = calculate
    # (the existing _multi_round_secondary_tool helper rotates search -> calculate)
    (
        {"multi_round": True, "reasoning": True},
        ["search_documents"],
    ),
    # slot 2: errored single tool call (malformed JSON args) + reasoning
    (
        {"error": True, "reasoning": True},
        ["lookup_user"],
    ),
    # slot 3: parallel pair + reasoning
    (
        {"parallel": True, "reasoning": True},
        ["calculate", "get_current_time"],
    ),
    # slot 4: plain single tool + reasoning (baseline shape)
    (
        {"reasoning": True},
        ["search_documents"],
    ),
]


def _exhaustive_overrides(turn_idx: int) -> tuple[_Markers, list[str]]:
    """Return synthesized (markers, forced_tools) for the given user-turn index."""
    slot = turn_idx % len(_EXHAUSTIVE_ROTATION)
    marker_kwargs, tools = _EXHAUSTIVE_ROTATION[slot]
    return _Markers(**marker_kwargs), list(tools)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m unittest tests.test_scripted_mode.ExhaustiveModeHelperTests -v
```

Expected: 13 tests pass (the helpers from earlier tasks + the 6 new ones).

- [ ] **Step 5: Commit**

```bash
git add simple_chatbot/scripted_llm.py tests/test_scripted_mode.py
git commit -m "$(cat <<'EOF'
feat(scripted_llm): add 5-slot exhaustive rotation table and lookup

Five rotation slots cover all four mocked tools and all four trace
shapes (parallel, multi-round, errored, reasoning). _exhaustive_overrides
turns a user-turn index into a Markers struct plus a forced tool list.
Not wired into dispatch yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Wire exhaustive overrides into `acompletion()`

Apply the rotation at the top of `acompletion`, threaded through the user-message and tool-result branches. Explicit bracketed markers (`markers.bracketed`) suppress the override so testers can still force shapes per-turn.

**Files:**
- Modify: `simple_chatbot/scripted_llm.py` (the `acompletion` function, lines 281-360)
- Test: `tests/test_scripted_mode.py`

- [ ] **Step 1: Write the failing tests**

Add a new class to `tests/test_scripted_mode.py` (before `if __name__ == "__main__":`):

```python
class ExhaustiveModeDispatchTests(unittest.TestCase):
    """End-to-end tests for EXHAUSTIVE_TOOL_USE=1 routing through acompletion()."""

    def _exhaustive_env(self):
        import os
        from unittest.mock import patch
        return patch.dict(os.environ, {"EXHAUSTIVE_TOOL_USE": "1"})

    def test_first_user_turn_emits_all_four_tools_in_parallel(self):
        with self._exhaustive_env():
            resp = asyncio.run(scripted_acompletion(
                messages=[{"role": "user", "content": "anything"}],
            ))
        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        names = sorted(tc.function.name for tc in choice.message.tool_calls)
        self.assertEqual(
            names,
            sorted(["search_documents", "calculate", "get_current_time", "lookup_user"]),
        )
        self.assertIsNotNone(choice.message.reasoning_content)

    def test_second_user_turn_multi_round_emits_round_two_on_tool_result(self):
        # Two user messages in history → turn index 1 → multi-round.
        # Final user message present, then a tool result.
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "search_documents", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "R1"},
        ]
        with self._exhaustive_env():
            resp = asyncio.run(scripted_acompletion(messages=msgs))
        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(len(choice.message.tool_calls), 1)
        # Secondary tool after search_documents is calculate per the rotation helper.
        self.assertEqual(choice.message.tool_calls[0].function.name, "calculate")

    def test_third_user_turn_emits_malformed_args(self):
        # Three user messages → turn index 2 → errored shape.
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "third"},
        ]
        with self._exhaustive_env():
            resp = asyncio.run(scripted_acompletion(messages=msgs))
        tc = resp.choices[0].message.tool_calls[0]
        self.assertEqual(tc.function.name, "lookup_user")
        import json as _json
        with self.assertRaises(_json.JSONDecodeError):
            _json.loads(tc.function.arguments)

    def test_fifth_user_turn_emits_single_search_with_reasoning(self):
        # Five user messages → turn index 4 → plain single search_documents.
        msgs = []
        for i in range(5):
            msgs.append({"role": "user", "content": f"q{i}"})
            if i < 4:
                msgs.append({"role": "assistant", "content": f"a{i}"})
        with self._exhaustive_env():
            resp = asyncio.run(scripted_acompletion(messages=msgs))
        choice = resp.choices[0]
        self.assertEqual(len(choice.message.tool_calls), 1)
        self.assertEqual(choice.message.tool_calls[0].function.name, "search_documents")
        self.assertIsNotNone(choice.message.reasoning_content)

    def test_sixth_user_turn_wraps_back_to_slot_0(self):
        # Six user messages → turn index 5 → slot 0 → all four parallel.
        msgs = []
        for i in range(6):
            msgs.append({"role": "user", "content": f"q{i}"})
            if i < 5:
                msgs.append({"role": "assistant", "content": f"a{i}"})
        with self._exhaustive_env():
            resp = asyncio.run(scripted_acompletion(messages=msgs))
        names = sorted(tc.function.name for tc in resp.choices[0].message.tool_calls)
        self.assertEqual(
            names,
            sorted(["search_documents", "calculate", "get_current_time", "lookup_user"]),
        )

    def test_explicit_bracketed_marker_overrides_exhaustive(self):
        # Even on turn 0 (which would be all-parallel), an explicit [error]
        # marker should yield the existing error shape: single tool with
        # malformed args, defaulting to search_documents per heuristic.
        with self._exhaustive_env():
            resp = asyncio.run(scripted_acompletion(
                messages=[{"role": "user", "content": "[error] hello"}],
            ))
        tcs = resp.choices[0].message.tool_calls
        # Existing [error] behavior: first call is malformed.
        import json as _json
        with self.assertRaises(_json.JSONDecodeError):
            _json.loads(tcs[0].function.arguments)
        # Should NOT be all four tools — the override yielded to the explicit marker.
        self.assertLess(len(tcs), 4)

    def test_env_var_off_does_not_change_default_behavior(self):
        # Env var unset → existing scripted heuristic applies: single
        # search_documents call for an arbitrary message.
        import os
        from unittest.mock import patch
        env_no_var = {k: v for k, v in os.environ.items() if k != "EXHAUSTIVE_TOOL_USE"}
        with patch.dict(os.environ, env_no_var, clear=True):
            resp = asyncio.run(scripted_acompletion(
                messages=[{"role": "user", "content": "tell me about elephants"}],
            ))
        tcs = resp.choices[0].message.tool_calls
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0].function.name, "search_documents")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m unittest tests.test_scripted_mode.ExhaustiveModeDispatchTests -v
```

Expected: most fail because the override isn't wired in yet. The `test_env_var_off_does_not_change_default_behavior` may pass (existing behavior). The others fail.

- [ ] **Step 3: Wire the override into `acompletion()`**

In `simple_chatbot/scripted_llm.py`, replace the entire `acompletion` function (lines 281-360 in the current file) with:

```python
async def acompletion(*, messages: list[dict], **kwargs) -> _ScriptedResponse:
    """Scripted LiteLLM-compatible completion for offline testing.

    Heuristic tool selection by keyword in the latest user input, plus
    explicit `[parallel]`, `[reasoning]`, `[multi-round]`, `[error]` markers
    that force specific trace shapes.

    When `EXHAUSTIVE_TOOL_USE` is truthy and the user text contains no
    bracketed marker, a 5-slot rotation indexed by user-turn count overrides
    both markers and tool selection so a typical conversation cycles through
    every tool and every shape.

    Dispatches by walking messages from the end backwards:
    - tool result -> emit another tool call (multi-round) or final answer
    - user input -> pick tools per heuristic / markers / exhaustive override,
                    emit tool_call(s)
    - nothing dispatchable -> generic "ok" stop
    """
    last_user_msg = _find_latest_user(messages)
    last_user_text = _extract_user_text(last_user_msg.get("content") if last_user_msg else "")
    markers, cleaned_text = _parse_markers(last_user_text)

    # Exhaustive-mode override: synthesize markers + forced tools from the
    # rotation when the env var is set AND the user did not type a bracketed
    # marker (those still win, so testers can force a shape per-turn).
    forced_tools: list[str] | None = None
    if _is_exhaustive() and not markers.bracketed:
        turn_idx = max(0, _count_user_turns(messages) - 1)
        markers, forced_tools = _exhaustive_overrides(turn_idx)

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        role = msg.get("role")

        if role == "tool":
            rounds = _count_rounds_since_latest_user(messages)
            if markers.multi_round and rounds < 2:
                # Emit a second-round tool call with a different tool
                primary = (
                    forced_tools[0] if forced_tools else _pick_tools(cleaned_text)[0]
                )
                secondary = _multi_round_secondary_tool(primary)
                tc = _make_tool_call(secondary, _build_args_for(secondary, cleaned_text))
                message = _ScriptedMessage(
                    content=None,
                    tool_calls=[tc],
                    reasoning_content=_reasoning_text(cleaned_text, [secondary]) if markers.reasoning else None,
                )
                return _ScriptedResponse(
                    choices=[_ScriptedChoice(message=message, finish_reason="tool_calls")],
                )
            # Otherwise: final answer combining tool results
            tool_outputs = _collect_tool_outputs_since_latest_user(messages)
            summary = " | ".join(tool_outputs) if tool_outputs else "(no tool results)"
            text = f"Scripted answer combining {len(tool_outputs)} tool result(s): {summary}"
            message = _ScriptedMessage(
                content=text,
                reasoning_content=(
                    _final_reasoning_text(cleaned_text, len(tool_outputs))
                    if markers.reasoning else None
                ),
            )
            return _ScriptedResponse(
                choices=[_ScriptedChoice(message=message, finish_reason="stop")],
            )

        if role == "user":
            tools = forced_tools if forced_tools is not None else _pick_tools(cleaned_text)
            if markers.parallel and len(tools) < 2:
                # Force parallelism by appending an extra tool
                rotation = ["search_documents", "calculate", "get_current_time", "lookup_user"]
                for fallback in rotation:
                    if fallback not in tools:
                        tools.append(fallback)
                        break

            if markers.error:
                # Replace the first tool call with a malformed one; keep the rest normal
                first_call = _make_malformed_tool_call(tools[0])
                rest = [_make_tool_call(t, _build_args_for(t, cleaned_text)) for t in tools[1:]]
                tool_calls = [first_call] + rest
            else:
                tool_calls = [_make_tool_call(t, _build_args_for(t, cleaned_text)) for t in tools]

            message = _ScriptedMessage(
                content=None,
                tool_calls=tool_calls,
                reasoning_content=_reasoning_text(cleaned_text, tools) if markers.reasoning else None,
            )
            return _ScriptedResponse(
                choices=[_ScriptedChoice(message=message, finish_reason="tool_calls")],
            )

    return _fallback_response()
```

The diff vs the current code:
1. Added `forced_tools` block right after `_parse_markers`.
2. In the tool-result branch, `primary` now uses `forced_tools[0]` when present.
3. In the user-message branch, `tools = forced_tools if forced_tools is not None else _pick_tools(cleaned_text)`.

Everything else (parallel padding, error path, reasoning rendering) is unchanged.

- [ ] **Step 4: Run all scripted-mode tests to verify they pass**

```bash
uv run python -m unittest tests.test_scripted_mode -v
```

Expected: every test passes, including the new `ExhaustiveModeDispatchTests` class and all pre-existing tests (the rotation must not regress default behavior).

- [ ] **Step 5: Run the full test suite to verify no other tests broke**

```bash
uv run python -m unittest discover -s tests
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add simple_chatbot/scripted_llm.py tests/test_scripted_mode.py
git commit -m "$(cat <<'EOF'
feat(scripted_llm): wire EXHAUSTIVE_TOOL_USE rotation into acompletion

When EXHAUSTIVE_TOOL_USE is truthy and the user did not type a
bracketed marker, a 5-slot rotation indexed by user-turn count
overrides both markers and tool selection so a typical conversation
cycles through every mocked tool and every supported trace shape
(parallel, multi-round, errored, reasoning). Explicit bracketed
markers still win.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Document `EXHAUSTIVE_TOOL_USE` in README and AGENTS.md

**Files:**
- Modify: `README.md` (around line 340, "Multi-tool evaluator harness" section)
- Modify: `AGENTS.md` (the `simple_chatbot/scripted_llm.py` bullet, around line 13)

- [ ] **Step 1: Read the current README section to find the exact insertion point**

```bash
sed -n '325,360p' README.md
```

Note the existing paragraphs about `SIMPLE_CHATBOT_SCRIPTED_LLM` and the marker list. The new paragraph goes immediately after the marker description.

- [ ] **Step 2: Add the README paragraph**

In `README.md`, locate the "Multi-tool evaluator harness" section. Immediately after the existing description of the four bracketed markers, add a new subsection:

```markdown
**Exhaustive showcase mode:** Set `EXHAUSTIVE_TOOL_USE=1` alongside
`SIMPLE_CHATBOT_SCRIPTED_LLM=1` to make every scripted user turn cycle through
a fixed 5-slot rotation that exercises every mocked tool and every trace
shape over a typical conversation. By construction, the first user turn fires
all four tools in parallel and subsequent turns rotate through multi-round,
errored, parallel-pair, and plain-single shapes:

| User turn (mod 5) | Shape          | Tools                                                          |
|-------------------|----------------|----------------------------------------------------------------|
| 0                 | All-parallel   | `search_documents`, `calculate`, `get_current_time`, `lookup_user` |
| 1                 | Multi-round    | round 1 `search_documents`, round 2 `calculate`                |
| 2                 | Errored call   | `lookup_user` with malformed JSON args                          |
| 3                 | Parallel pair  | `calculate`, `get_current_time`                                |
| 4                 | Plain single   | `search_documents`                                             |

Explicit bracketed markers (`[parallel]`, `[multi-round]`, `[error]`,
`[reasoning]`) in the user's text still take precedence over the rotation, so
a tester can force any specific shape on any individual turn.
```

(Use the existing surrounding prose voice — the surrounding section is
descriptive paragraphs, no bullet lists for the framing.)

- [ ] **Step 3: Add the AGENTS.md bullet**

In `AGENTS.md`, locate the `simple_chatbot/scripted_llm.py` line under "Primary modules":

```markdown
- `simple_chatbot/scripted_llm.py`: Scripted LLM dispatch with marker-driven trace shapes (`[parallel]`, `[reasoning]`, `[multi-round]`, `[error]`) and keyword heuristics for tool selection.
```

Replace it with:

```markdown
- `simple_chatbot/scripted_llm.py`: Scripted LLM dispatch with marker-driven trace shapes (`[parallel]`, `[reasoning]`, `[multi-round]`, `[error]`), keyword heuristics for tool selection, and an `EXHAUSTIVE_TOOL_USE` env-var rotation that cycles through every tool and trace shape over five turns.
```

- [ ] **Step 4: Sanity-check the docs render**

```bash
grep -n "EXHAUSTIVE_TOOL_USE" README.md AGENTS.md
```

Expected: a handful of hits showing the env-var name in both files.

- [ ] **Step 5: Commit**

```bash
git add README.md AGENTS.md
git commit -m "$(cat <<'EOF'
docs: document EXHAUSTIVE_TOOL_USE rotation in README and AGENTS.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Final verification

- [ ] **Step 1: Run the full test suite**

```bash
uv run python -m unittest discover -s tests
```

Expected: all tests pass.

- [ ] **Step 2: Manual sanity check via scripted server** (optional but recommended)

```bash
EXHAUSTIVE_TOOL_USE=1 SIMPLE_CHATBOT_SCRIPTED_LLM=1 uv run simple-chatbot serve --docs-dir /tmp/empty --port 15078 &
sleep 2
BASE_URL=http://localhost:15078 ./chain_demo.sh
kill %1
```

Expected: each of the three turns shown by `chain_demo.sh` shows different tool shapes — turn 1 should list four parallel tool calls; turn 2 should show multi-round; turn 3 should show the errored shape.

- [ ] **Step 3: Skim `git log` to confirm a clean history**

```bash
git log --oneline -8
```

Expected: 6 commits in this branch following the plan, with clear messages.

---

## Notes for the executor

- Keep the existing `_Markers` field order intact in Task 1: parallel, reasoning, multi_round, error first, then the new `bracketed`. Several tests (and the `MarkerDispatchTests` already in the file) construct `_Markers` only via keyword arguments, so adding a field at the end is safe.
- Don't add new fields to `_Markers` between Task 1 and Task 5 — the rotation table assumes `parallel/reasoning/multi_round/error` are the only marker keywords.
- `_pick_tools` and `_build_args_for` stay untouched throughout. The forced-tools list bypasses `_pick_tools` for exhaustive turns but the existing key-word heuristic is still used when exhaustive is off.
- All tests use `unittest.mock.patch.dict(os.environ, …)` because `_is_exhaustive()` reads `os.environ` on every call. Do NOT cache the env var at module import — that would break testing.
