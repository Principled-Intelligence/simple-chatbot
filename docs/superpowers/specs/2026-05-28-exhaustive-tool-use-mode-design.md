# Exhaustive tool-use mode — design

**Date:** 2026-05-28
**Status:** Approved (pending spec review)
**Author:** Niccolò Campolungo / Claude

## Goal

Add an `EXHAUSTIVE_TOOL_USE` environment variable that, when set, makes scripted
offline mode showcase every mocked tool and every supported trace shape over a
typical ~5-turn conversation. Useful as a one-knob demo / evaluator-fixture
generator that exercises the full surface of the scripted dispatch without the
caller having to remember and combine markers.

## Why this matters

Scripted mode already mocks four tools (`search_documents`, `calculate`,
`get_current_time`, `lookup_user`) and supports four trace-shape markers
(`[parallel]`, `[reasoning]`, `[multi-round]`, `[error]`). Today, a caller who
wants to see all of them must hand-craft a multi-turn conversation with the
right markers and keyword triggers in each turn. That's tedious and easy to get
wrong.

`EXHAUSTIVE_TOOL_USE` flips a single switch: a fresh conversation walked through
~5 turns will, by construction, hit every tool and every trace shape.

## Non-goals

- True randomness. The rotation is deterministic-by-turn-index so tests can
  assert specific shapes per turn.
- A new demo shell script. The existing `chain_demo.sh` is enough to drive a
  multi-turn conversation against the server with the env var set.
- Changing the real (non-scripted) LLM path. This is a scripted-mode-only knob.
- Auto-enabling whenever `SIMPLE_CHATBOT_SCRIPTED_LLM` is set. Exhaustive mode
  is opt-in, separate from scripted mode being on.

## Approach

Read `EXHAUSTIVE_TOOL_USE` once at module import in `scripted_llm.py`. When
truthy, every user-turn dispatch synthesizes a fixed `(markers, tools)`
override based on a deterministic 5-turn rotation indexed by the number of
`user`-role messages in the message history. The existing parallel /
multi-round / error / reasoning machinery then handles the rest of the dispatch
unchanged.

Considered and rejected:
- True random shape selection. Loses test predictability; the user explicitly
  said no randomness.
- Tracking turn state out-of-band. Stateless dispatch via the messages list is
  consistent with how the rest of `scripted_llm.py` works.
- Replacing existing markers wholesale. Explicit markers in the user's text
  must still override exhaustive mode so testers can force a specific shape on
  any turn.

## The rotation

Turn index `n` = `max(0, count_of_user_messages - 1)`. The first user turn
(`messages` ends with a single user message) is index 0, the second user turn
is index 1, and so on. Effective rotation slot = `n % 5`.

| `n % 5` | Shape | Synthesized markers | Forced tools |
|--------|------|--------------------|--------------|
| 0 | All-parallel | `parallel`, `reasoning` | `search_documents`, `calculate`, `get_current_time`, `lookup_user` (all 4) |
| 1 | Multi-round | `multi_round`, `reasoning` | round 1: `search_documents`; round 2: `calculate` (via existing rotation helper) |
| 2 | Errored call | `error`, `reasoning` | `lookup_user` (single tool, malformed JSON args) |
| 3 | Parallel pair | `parallel`, `reasoning` | `calculate`, `get_current_time` |
| 4 | Plain single | `reasoning` | `search_documents` |

Coverage claims:
- Every mocked tool appears in turn 0 alone. Other turns add variety, not
  coverage.
- Every trace shape (parallel, multi-round, errored, reasoning, plain) appears
  at least once across the 5 rotation slots.
- A conversation longer than 5 turns continues to vary via the modulo.

## Override semantics

If the latest user message contains any explicit marker
(`[parallel]`, `[reasoning]`, `[multi-round]`, `[error]`), exhaustive mode
yields. The existing marker parser runs and its markers + the existing keyword
heuristic for tool selection take over for that turn. This lets testers force a
specific shape on any single turn even when exhaustive mode is globally on.

Heuristic reasoning triggers ("step by step" etc.) do *not* count as explicit
markers for this purpose — they're too loose. Only the bracketed markers
suppress exhaustive override.

## Module structure

All changes live in `simple_chatbot/scripted_llm.py`:

- **New module-level constant:** `_EXHAUSTIVE = os.environ.get("EXHAUSTIVE_TOOL_USE", "").strip().lower() in {"1", "true", "yes", "on"}`
- **New helper:** `_count_user_turns(messages: list[dict]) -> int` — counts
  `user`-role messages in the list.
- **New helper:** `_exhaustive_overrides(turn_idx: int) -> tuple[_Markers, list[str]]` —
  returns the synthesized markers + forced tools for `turn_idx % 5`.
- **Dispatch wiring in `acompletion()`:**
  - At the top of the user-message branch, if `_EXHAUSTIVE` and the parsed
    markers have no explicit marker set (`not markers.any_explicit`), replace
    `markers` and the result of `_pick_tools(cleaned_text)` with the rotation's
    overrides for this turn.
  - In the tool-result branch, also replace `markers` from the rotation so
    multi-round turns correctly emit round 2. (The rotation's index counts
    user turns, so a tool-result message in turn idx 1 still resolves to the
    multi-round shape.)

No changes to `tools.py`, `agent.py`, `cli.py`, or any other module. The
scripted indexer / tool executors already accept the kinds of calls this
rotation emits.

## Error handling

- Env var absent / empty / unrecognized value (e.g. `"0"`, `"no"`, garbage):
  exhaustive mode disabled, scripted dispatch behaves exactly as today. No
  warnings, no startup noise.
- Conversation with zero user messages (only assistant or tool messages): the
  existing fallback path produces `"ok"`, no rotation invoked.

## Testing

Add a new `ExhaustiveModeTests` class in `tests/test_scripted_mode.py`:

- First user turn (idx 0) with env var on → 4 parallel tool calls covering all
  4 tool names.
- Second user turn (idx 1) with env var on and a tool result in history → emits
  a second tool call (multi-round round 2).
- Third user turn (idx 2) with env var on → emits a tool call whose arguments
  do *not* parse as valid JSON.
- Fifth user turn (idx 4) with env var on → emits a single `search_documents`
  call with `reasoning_content` present.
- Sixth user turn (idx 5, slot 0) with env var on → matches first turn's shape
  (rotation wraps).
- Env var on + explicit `[parallel]` marker → behaves like today's `[parallel]`
  marker, not the rotation. Confirms overrides apply.
- Env var off (default) → existing scripted behavior unchanged.

All tests use `unittest.mock.patch.dict(os.environ, {"EXHAUSTIVE_TOOL_USE": "1"})`
or `clear=False` to flip the env var per-test. The module-level constant is
re-read in tests via either a small `_is_exhaustive()` callable indirection or
by patching the module attribute directly — pick whichever is cleaner during
implementation.

## Documentation

- `README.md`: add a short paragraph in the "Multi-tool evaluator harness"
  section that introduces the env var, lists the rotation table, and notes the
  override semantics.
- `AGENTS.md`: one bullet under the `simple_chatbot/scripted_llm.py` summary
  line mentioning the new env var.

No changes to other docs.
