import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simple_chatbot.agent import Agent
from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.scripted_indexer import ScriptedIndexer
from simple_chatbot.scripted_llm import acompletion as scripted_acompletion


def _config(tmp: str) -> SimpleChatbotConfig:
    return SimpleChatbotConfig(docs_dir=Path(tmp) / "docs")


class ScriptedAcompletionTests(unittest.TestCase):
    def test_user_message_emits_search_tool_call(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "alpha bravo"}],
        ))
        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        tc = choice.message.tool_calls[0]
        self.assertEqual(tc.function.name, "search_documents")
        self.assertIn("alpha bravo", tc.function.arguments)
        self.assertTrue(tc.id.startswith("call_scripted_"))

    def test_tool_message_emits_echo_stop(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[
                {"role": "user", "content": "alpha"},
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "tool_call_id": "x", "name": "search_documents", "content": "RESULT-X"},
            ],
        ))
        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertIn("RESULT-X", choice.message.content)
        self.assertIn("combining", choice.message.content)
        self.assertEqual(choice.message.tool_calls, [])

    def test_fallback_message_role_emits_generic_stop(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "assistant", "content": "previous answer"}],
        ))
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual(resp.choices[0].message.content, "ok")

    def test_dispatch_uses_most_recent_user_when_list_ends_with_assistant(self):
        # Client sends a multi-message list ending in assistant.
        # Should still emit a tool_call using the most recent user message.
        resp = asyncio.run(scripted_acompletion(
            messages=[
                {"role": "user", "content": "first user"},
                {"role": "assistant", "content": "earlier assistant reply"},
            ],
        ))
        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        tc = choice.message.tool_calls[0]
        self.assertEqual(tc.function.name, "search_documents")
        self.assertIn("first user", tc.function.arguments)

    def test_dispatch_picks_latest_user_when_multiple_user_messages(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[
                {"role": "user", "content": "older user msg"},
                {"role": "assistant", "content": "older reply"},
                {"role": "user", "content": "latest user msg"},
            ],
        ))
        tc = resp.choices[0].message.tool_calls[0]
        self.assertIn("latest user msg", tc.function.arguments)
        self.assertNotIn("older user msg", tc.function.arguments)

    def test_dispatch_prefers_tool_over_earlier_user(self):
        # When the list is [user, assistant_with_tc, tool], the tool is the
        # most recent dispatchable message — we should emit the final answer
        # rather than starting a new search.
        resp = asyncio.run(scripted_acompletion(
            messages=[
                {"role": "user", "content": "original question"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
                {"role": "tool", "tool_call_id": "x", "name": "search_documents", "content": "TOOL RESULT"},
            ],
        ))
        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertIn("TOOL RESULT", choice.message.content)
        self.assertIn("combining", choice.message.content)

    def test_dispatch_falls_back_when_no_user_or_tool(self):
        # No user, no tool — only assistant. Should still produce a valid
        # stop response with content "ok" rather than crashing.
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "assistant", "content": "lonely assistant"}],
        ))
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual(resp.choices[0].message.content, "ok")

    def test_dispatch_falls_back_when_messages_empty(self):
        resp = asyncio.run(scripted_acompletion(messages=[]))
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual(resp.choices[0].message.content, "ok")

    def test_subsequent_user_still_dispatches_through_heuristic(self):
        # Multi-tool harness: a follow-up user message after a prior tool
        # round must still be routed through tool-selection heuristics rather
        # than collapsing to a "direct answer" — different turns may need
        # different tools.
        resp = asyncio.run(scripted_acompletion(
            messages=[
                {"role": "user", "content": "calculate 2+2"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
                {"role": "tool", "tool_call_id": "x", "name": "calculate", "content": "= 4"},
                {"role": "assistant", "content": "earlier scripted answer"},
                {"role": "user", "content": "lookup user fred"},
            ],
        ))
        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(len(choice.message.tool_calls), 1)
        self.assertEqual(choice.message.tool_calls[0].function.name, "lookup_user")
        self.assertIn("fred", choice.message.tool_calls[0].function.arguments)

    def test_user_after_plain_assistant_still_emits_tool_call(self):
        # Pre-existing assistant message with no tool_calls: latest user input
        # still triggers a tool call as usual.
        resp = asyncio.run(scripted_acompletion(
            messages=[
                {"role": "user", "content": "older"},
                {"role": "assistant", "content": "plain reply, no tool"},
                {"role": "user", "content": "latest"},
            ],
        ))
        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        tc = choice.message.tool_calls[0]
        self.assertIn("latest", tc.function.arguments)


class ScriptedIndexerTests(unittest.TestCase):
    def test_search_echoes_query(self):
        idx = ScriptedIndexer()
        results = asyncio.run(idx.search("what is X?"))
        self.assertEqual(len(results), 1)
        self.assertIn("what is X?", results[0].text)
        self.assertEqual(results[0].metadata["source"], "scripted")

    def test_document_count_is_one(self):
        self.assertEqual(ScriptedIndexer().document_count(), 1)


class ScriptedModeIntegrationTests(unittest.TestCase):
    def test_agent_chat_end_to_end_with_both_stubs(self):
        with TemporaryDirectory() as tmp:
            agent = Agent(
                _config(tmp),
                ScriptedIndexer(),
                acompletion=scripted_acompletion,
            )
            result = asyncio.run(agent.chat([{"role": "user", "content": "hello world"}]))

            self.assertIn("Scripted answer", result.content)
            self.assertIn("hello world", result.content)
            roles = [m["role"] for m in result.tool_messages]
            self.assertEqual(roles, ["assistant", "tool"])
            assistant_msg = result.tool_messages[0]
            self.assertEqual(len(assistant_msg["tool_calls"]), 1)
            self.assertEqual(
                assistant_msg["tool_calls"][0]["function"]["name"],
                "search_documents",
            )


class ScriptedModeChainedTurnTests(unittest.TestCase):
    def test_chained_turn_still_dispatches_through_heuristic(self):
        # Drive a fake "chain" at the Agent level: pass turn 1's
        # final_messages plus a new user turn as turn 2's input. Turn 2 must
        # go through the normal tool-selection heuristic; the harness exposes
        # multiple distinct tools, so chained turns may legitimately need to
        # call a different tool than the prior turn.
        with TemporaryDirectory() as tmp:
            agent = Agent(
                _config(tmp),
                ScriptedIndexer(),
                acompletion=scripted_acompletion,
            )
            turn1 = asyncio.run(agent.chat([{"role": "user", "content": "what?"}]))
            self.assertEqual(
                [m["role"] for m in turn1.tool_messages],
                ["assistant", "tool"],
            )

            turn2_input = list(turn1.final_messages) + [
                {"role": "user", "content": "lookup user alice"},
            ]
            turn2 = asyncio.run(agent.chat(turn2_input))

            # Turn 2 must have its own tool round
            self.assertEqual(
                [m["role"] for m in turn2.tool_messages],
                ["assistant", "tool"],
            )
            tool_call = turn2.tool_messages[0]["tool_calls"][0]
            self.assertEqual(tool_call["function"]["name"], "lookup_user")
            self.assertIn("alice", tool_call["function"]["arguments"])


class MarkerDispatchTests(unittest.TestCase):
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

    def test_parallel_marker_emits_multiple_tool_calls(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "[parallel] tell me about user alice"}],
        ))
        tcs = resp.choices[0].message.tool_calls
        self.assertGreaterEqual(len(tcs), 2, f"expected >=2 tool calls, got {len(tcs)}")

    def test_reasoning_marker_emits_reasoning_content(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "[reasoning] hi"}],
        ))
        self.assertIsNotNone(resp.choices[0].message.reasoning_content)

    def test_error_marker_emits_malformed_args(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "[error] anything"}],
        ))
        tc = resp.choices[0].message.tool_calls[0]
        # Args are JSON-encoded; for error marker they should NOT parse as valid JSON.
        import json as _json
        with self.assertRaises(_json.JSONDecodeError):
            _json.loads(tc.function.arguments)

    def test_step_by_step_phrase_triggers_reasoning_heuristic(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "explain step by step"}],
        ))
        self.assertIsNotNone(resp.choices[0].message.reasoning_content)

    def test_parallel_marker_works_on_chained_turn(self):
        # Follow-up turn with [parallel] should still produce >=2 tool calls.
        resp = asyncio.run(scripted_acompletion(
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
                {"role": "tool", "tool_call_id": "x", "content": "R"},
                {"role": "assistant", "content": "prior answer"},
                {"role": "user", "content": "[parallel] second"},
            ],
        ))
        self.assertGreaterEqual(len(resp.choices[0].message.tool_calls), 2)


class HeuristicToolSelectionTests(unittest.TestCase):
    def test_calculate_keyword_picks_calculate_tool(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "calculate 2+2"}],
        ))
        tc = resp.choices[0].message.tool_calls[0]
        self.assertEqual(tc.function.name, "calculate")
        self.assertIn("2+2", tc.function.arguments)

    def test_calculate_without_operator_still_picks_calculate(self):
        # Regression: 'calculate' (inflected) must match even without digits/operators.
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "please calculate the cost of doing nothing"}],
        ))
        tc = resp.choices[0].message.tool_calls[0]
        self.assertEqual(tc.function.name, "calculate")

    def test_time_keyword_picks_time_tool(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "what time is it"}],
        ))
        tc = resp.choices[0].message.tool_calls[0]
        self.assertEqual(tc.function.name, "get_current_time")

    def test_time_in_location_extracts_tz(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "time in Europe/London"}],
        ))
        tc = resp.choices[0].message.tool_calls[0]
        self.assertEqual(tc.function.name, "get_current_time")
        self.assertIn("Europe/London", tc.function.arguments)

    def test_lookup_user_keyword_picks_lookup_tool(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "lookup user alice"}],
        ))
        tc = resp.choices[0].message.tool_calls[0]
        self.assertEqual(tc.function.name, "lookup_user")
        self.assertIn("alice", tc.function.arguments)

    def test_default_falls_back_to_search(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "tell me about elephants"}],
        ))
        tc = resp.choices[0].message.tool_calls[0]
        self.assertEqual(tc.function.name, "search_documents")

    def test_multiple_keywords_emit_multiple_tool_calls(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "calculate 5+5 and show me the time"}],
        ))
        names = {tc.function.name for tc in resp.choices[0].message.tool_calls}
        self.assertIn("calculate", names)
        self.assertIn("get_current_time", names)


class MultiRoundTests(unittest.TestCase):
    def test_first_round_emits_one_tool_call(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "[multi-round] research deeply"}],
        ))
        self.assertEqual(len(resp.choices[0].message.tool_calls), 1)
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")

    def test_second_round_emits_another_tool_call(self):
        # Simulate the agent's view after round 1's tool ran.
        resp = asyncio.run(scripted_acompletion(
            messages=[
                {"role": "user", "content": "[multi-round] research deeply"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "search_documents", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "R1"},
            ],
        ))
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertEqual(len(resp.choices[0].message.tool_calls), 1)

    def test_third_round_emits_final_answer(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[
                {"role": "user", "content": "[multi-round] research deeply"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "search_documents", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "R1"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c2", "type": "function",
                                 "function": {"name": "calculate", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c2", "content": "R2"},
            ],
        ))
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertIsNone(resp.choices[0].message.tool_calls or None)

    def test_parallel_plus_multi_round_does_not_collapse(self):
        # Round 1 with [parallel] + [multi-round]: should emit 2 parallel tool calls.
        r1 = asyncio.run(scripted_acompletion(
            messages=[{"role": "user", "content": "[parallel] [multi-round] calculate and lookup user bob"}],
        ))
        self.assertEqual(r1.choices[0].finish_reason, "tool_calls")
        self.assertGreaterEqual(len(r1.choices[0].message.tool_calls), 2)

        # Round 2: even though tool_count is already 2, multi-round should still emit ANOTHER tool call
        r2 = asyncio.run(scripted_acompletion(
            messages=[
                {"role": "user", "content": "[parallel] [multi-round] calculate and lookup user bob"},
                {"role": "assistant", "content": None,
                 "tool_calls": [
                     {"id": "c1", "type": "function", "function": {"name": "calculate", "arguments": "{}"}},
                     {"id": "c2", "type": "function", "function": {"name": "lookup_user", "arguments": "{}"}},
                 ]},
                {"role": "tool", "tool_call_id": "c1", "content": "R1"},
                {"role": "tool", "tool_call_id": "c2", "content": "R2"},
            ],
        ))
        self.assertEqual(r2.choices[0].finish_reason, "tool_calls",
                         "Round 2 should still emit another tool call when [multi-round] is set")
        self.assertEqual(len(r2.choices[0].message.tool_calls), 1)

    def test_reasoning_on_final_message_reflects_tool_results(self):
        # After tool results, reasoning_content should mention "Reviewing N tool result(s)",
        # NOT "Planned tool(s): none" (which would contradict the trace).
        resp = asyncio.run(scripted_acompletion(
            messages=[
                {"role": "user", "content": "[reasoning] calculate the cost"},
                {"role": "assistant", "content": None,
                 "tool_calls": [
                     {"id": "c1", "type": "function",
                      "function": {"name": "calculate", "arguments": "{\"expression\": \"the cost\"}"}},
                 ]},
                {"role": "tool", "tool_call_id": "c1", "content": "RESULT"},
            ],
        ))
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        rc = resp.choices[0].message.reasoning_content
        self.assertIsNotNone(rc)
        self.assertIn("Reviewing", rc)
        self.assertNotIn("Planned tool(s): none", rc)


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

    def test_fourth_user_turn_emits_parallel_pair(self):
        # Four user messages → turn index 3 → parallel pair (calculate + get_current_time).
        msgs = []
        for i in range(4):
            msgs.append({"role": "user", "content": f"q{i}"})
            if i < 3:
                msgs.append({"role": "assistant", "content": f"a{i}"})
        with self._exhaustive_env():
            resp = asyncio.run(scripted_acompletion(messages=msgs))
        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        names = sorted(tc.function.name for tc in choice.message.tool_calls)
        self.assertEqual(names, sorted(["calculate", "get_current_time"]))
        self.assertIsNotNone(choice.message.reasoning_content)

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
        # Tighter: slot 0 would have given 4 tools; [error] should yield exactly
        # 1 with the heuristic default `search_documents`. Proves the override
        # was bypassed by the explicit marker, not just that the shapes differ.
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0].function.name, "search_documents")

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

    def test_heuristic_step_by_step_does_not_suppress_rotation(self):
        # Heuristic reasoning phrases like "step by step" set markers.reasoning
        # via the parser, but they must NOT set markers.bracketed, so the
        # exhaustive rotation should still fire on a fresh first turn.
        with self._exhaustive_env():
            resp = asyncio.run(scripted_acompletion(
                messages=[{"role": "user", "content": "explain step by step"}],
            ))
        names = sorted(tc.function.name for tc in resp.choices[0].message.tool_calls)
        # Slot 0 → all four tools in parallel.
        self.assertEqual(
            names,
            sorted(["search_documents", "calculate", "get_current_time", "lookup_user"]),
        )


if __name__ == "__main__":
    unittest.main()
