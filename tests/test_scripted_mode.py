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
        self.assertEqual(choice.message.tool_calls, [])

    def test_fallback_message_role_emits_generic_stop(self):
        resp = asyncio.run(scripted_acompletion(
            messages=[{"role": "assistant", "content": "previous answer"}],
        ))
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertEqual(resp.choices[0].message.content, "ok")


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

            self.assertTrue(result.content.startswith("Scripted answer based on search result:"))
            self.assertIn("hello world", result.content)
            roles = [m["role"] for m in result.tool_messages]
            self.assertEqual(roles, ["assistant", "tool"])
            assistant_msg = result.tool_messages[0]
            self.assertEqual(len(assistant_msg["tool_calls"]), 1)
            self.assertEqual(
                assistant_msg["tool_calls"][0]["function"]["name"],
                "search_documents",
            )


if __name__ == "__main__":
    unittest.main()
