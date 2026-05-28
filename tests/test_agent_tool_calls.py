import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from simple_chatbot.agent import SEARCH_TOOL, Agent
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
    def __init__(self, name: str | None, arguments) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id: str, name: str | None, arguments) -> None:
        self.id = id
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, content: str = "", tool_calls: list | None = None) -> None:
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
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ],
        }


class _Choice:
    def __init__(self, message: _Message, finish_reason: str) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, message: _Message, finish_reason: str) -> None:
        self.choices = [_Choice(message, finish_reason)]
        self.usage = None


def _config(tmp: str) -> SimpleChatbotConfig:
    return SimpleChatbotConfig(docs_dir=Path(tmp) / "docs")


class AgentToolCallTests(unittest.TestCase):
    def test_malformed_tool_arguments_do_not_crash_request(self):
        with TemporaryDirectory() as tmp:
            indexer = _FakeIndexer()
            agent = Agent(_config(tmp), indexer)
            responses = [
                _Response(
                    _Message(
                        tool_calls=[
                            _ToolCall("call_1", "search_documents", "{not-json"),
                        ]
                    ),
                    "tool_calls",
                ),
                _Response(_Message(content="Recovered answer."), "stop"),
            ]

            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                result = asyncio.run(agent.chat([{"role": "user", "content": "hello"}]))

            self.assertEqual(result.content, "Recovered answer.")
            self.assertEqual(indexer.search_calls, [])
            second_messages = completion.await_args_list[1].kwargs["messages"]
            self.assertIn("Tool error:", second_messages[-1]["content"])

    def test_unsupported_tool_name_does_not_call_index_search(self):
        with TemporaryDirectory() as tmp:
            indexer = _FakeIndexer()
            agent = Agent(_config(tmp), indexer)
            responses = [
                _Response(
                    _Message(
                        tool_calls=[
                            _ToolCall("call_1", "delete_documents", '{"query": "hello"}'),
                        ]
                    ),
                    "tool_calls",
                ),
                _Response(_Message(content="Final answer."), "stop"),
            ]

            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                result = asyncio.run(agent.chat([{"role": "user", "content": "hello"}]))

            self.assertEqual(result.content, "Final answer.")
            self.assertEqual(indexer.search_calls, [])
            second_messages = completion.await_args_list[1].kwargs["messages"]
            self.assertIn("unsupported tool", second_messages[-1]["content"])

    def test_chat_result_exposes_tool_schema_and_invocations(self):
        with TemporaryDirectory() as tmp:
            indexer = _FakeIndexer()
            agent = Agent(_config(tmp), indexer)
            responses = [
                _Response(
                    _Message(
                        tool_calls=[
                            _ToolCall("call_1", "search_documents", '{"query": "alpha"}'),
                        ]
                    ),
                    "tool_calls",
                ),
                _Response(_Message(content="Final answer."), "stop"),
            ]

            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                result = asyncio.run(agent.chat([{"role": "user", "content": "hello"}]))

            self.assertEqual(result.content, "Final answer.")
            self.assertEqual(result.tools, [SEARCH_TOOL])
            self.assertEqual(len(result.tool_messages), 2)

            assistant_msg, tool_msg = result.tool_messages
            self.assertEqual(assistant_msg["role"], "assistant")
            self.assertEqual(len(assistant_msg["tool_calls"]), 1)
            self.assertEqual(assistant_msg["tool_calls"][0]["id"], "call_1")
            self.assertEqual(
                assistant_msg["tool_calls"][0]["function"]["name"],
                "search_documents",
            )

            self.assertEqual(tool_msg["role"], "tool")
            self.assertEqual(tool_msg["tool_call_id"], "call_1")
            self.assertEqual(tool_msg["name"], "search_documents")
            self.assertIn("No relevant documents", tool_msg["content"])

    def test_chat_result_tools_present_when_no_tool_calls(self):
        with TemporaryDirectory() as tmp:
            indexer = _FakeIndexer()
            agent = Agent(_config(tmp), indexer)
            responses = [
                _Response(_Message(content="Direct answer."), "stop"),
            ]

            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                result = asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))

            self.assertEqual(result.content, "Direct answer.")
            self.assertEqual(result.tools, [SEARCH_TOOL])
            self.assertEqual(result.tool_messages, [])


class AgentAcompletionInjectionTests(unittest.TestCase):
    def test_injected_acompletion_is_used_instead_of_litellm(self):
        with TemporaryDirectory() as tmp:
            indexer = _FakeIndexer()
            calls = []

            async def fake_acompletion(**kwargs):
                calls.append(kwargs)
                return _Response(_Message(content="injected"), "stop")

            agent = Agent(_config(tmp), indexer, acompletion=fake_acompletion)

            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as real_litellm:
                result = asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))

            self.assertEqual(result.content, "injected")
            self.assertEqual(len(calls), 1)
            self.assertIn("messages", calls[0])
            real_litellm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
