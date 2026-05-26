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
