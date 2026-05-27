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

    def test_subsequent_user_after_tool_skips_new_search(self):
        # Thread already shows a completed search round; new user input
        # should be answered directly with no tool_call.
        resp = asyncio.run(scripted_acompletion(
            messages=[
                {"role": "user", "content": "turn 1"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
                {"role": "tool", "tool_call_id": "x", "name": "search_documents", "content": "RESULT"},
                {"role": "assistant", "content": "earlier scripted answer"},
                {"role": "user", "content": "turn 2"},
            ],
        ))
        choice = resp.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertEqual(choice.message.tool_calls, [])
        self.assertIn("turn 2", choice.message.content)

    def test_assistant_with_tool_calls_in_history_also_skips_new_search(self):
        # Even if there is no explicit `tool` message yet, an earlier
        # assistant message with tool_calls counts as prior tool activity.
        resp = asyncio.run(scripted_acompletion(
            messages=[
                {"role": "user", "content": "earlier user"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "y"}]},
                {"role": "user", "content": "new user"},
            ],
        ))
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual(resp.choices[0].message.tool_calls, [])

    def test_user_after_plain_assistant_still_emits_tool_call(self):
        # An earlier assistant message that did NOT call a tool is not
        # prior tool activity. Latest user input should still trigger a
        # new search.
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
    def test_chained_turn_skips_redundant_search(self):
        # Drive a fake "chain" at the Agent level: pass turn 1's
        # final_messages plus a new user turn as turn 2's input. The
        # scripted policy should answer directly without a second search.
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
            self.assertIn("Scripted answer", turn1.content)

            # Simulate the server's chain prefix-prepending behavior
            turn2_input = list(turn1.final_messages) + [{"role": "user", "content": "follow up"}]
            turn2 = asyncio.run(agent.chat(turn2_input))

            self.assertEqual(turn2.tool_messages, [])
            self.assertEqual(
                turn2.content,
                "Scripted direct answer (no new search) for: 'follow up'",
            )


class MarkerDispatchTests(unittest.TestCase):
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

    def test_markers_override_chained_turn_no_tool_call(self):
        # Chain: prior search activity present, but [parallel] marker forces tool calls.
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


if __name__ == "__main__":
    unittest.main()
