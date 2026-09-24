# tests/test_max_output_tokens.py
import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

import simple_chatbot.server as server
from simple_chatbot.agent import Agent
from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.scenario_registry import load_fixtures


class _FakeIndexer:
    def document_count(self) -> int:
        return 1

    async def search(self, query: str) -> list:
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
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ],
        }


class _Choice:
    def __init__(self, message, finish_reason) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, message, finish_reason) -> None:
        self.choices = [_Choice(message, finish_reason)]
        self.usage = None


def _capturing_acompletion(captured: list[dict], responses: list | None = None):
    """Fake litellm.acompletion recording every call's kwargs."""
    queue = list(responses or [])

    async def fake(**kwargs):
        captured.append(kwargs)
        if queue:
            return queue.pop(0)
        return _Response(_Message(content="done"), "stop")

    return fake


class _FakeConversationLogger:
    async def log(self, conversation_id, messages, response, chunks, misbehavior_injections=None) -> None:
        return None


class AgentMaxOutputTokensTests(unittest.TestCase):
    def test_max_output_tokens_forwarded_as_max_tokens(self):
        with TemporaryDirectory() as tmp:
            captured: list[dict] = []
            agent = Agent(
                SimpleChatbotConfig(docs_dir=Path(tmp) / "docs"),
                _FakeIndexer(),
                acompletion=_capturing_acompletion(captured),
            )
            asyncio.run(
                agent.chat(
                    [{"role": "user", "content": "hi"}], max_output_tokens=64
                )
            )
            self.assertEqual(captured[0]["max_tokens"], 64)

    def test_max_tokens_absent_when_not_requested(self):
        with TemporaryDirectory() as tmp:
            captured: list[dict] = []
            agent = Agent(
                SimpleChatbotConfig(docs_dir=Path(tmp) / "docs"),
                _FakeIndexer(),
                acompletion=_capturing_acompletion(captured),
            )
            asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))
            self.assertNotIn("max_tokens", captured[0])

    def test_forced_final_call_also_capped(self):
        # Exhaust the tool-round budget so the agent makes its forced final
        # call; that LLM call must carry the cap too.
        with TemporaryDirectory() as tmp:
            captured: list[dict] = []
            tool_round = _Response(
                _Message(
                    tool_calls=[_ToolCall("call_1", "search_documents", '{"query": "x"}')]
                ),
                "tool_calls",
            )
            agent = Agent(
                SimpleChatbotConfig(docs_dir=Path(tmp) / "docs", max_tool_rounds=1),
                _FakeIndexer(),
                acompletion=_capturing_acompletion(captured, responses=[tool_round]),
            )
            asyncio.run(
                agent.chat(
                    [{"role": "user", "content": "hi"}], max_output_tokens=32
                )
            )
            # round call + forced final call
            self.assertEqual(len(captured), 2)
            self.assertEqual(captured[-1]["max_tokens"], 32)


class ResponsesMaxOutputTokensTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = (
            server._config, server._agent, server._conversation_logger,
            getattr(server, "_response_store", None),
            getattr(server, "_scenario_registry", None),
            getattr(server, "_acompletion", None),
        )
        self.addCleanup(self._restore)

        self.captured: list[dict] = []
        server._config = SimpleChatbotConfig(
            docs_dir=Path(self.tmp.name) / "docs",
            conversation_log_dir=Path(self.tmp.name) / "conversations",
        )
        server._agent = Agent(
            server._config,
            _FakeIndexer(),
            acompletion=_capturing_acompletion(self.captured),
        )
        server._conversation_logger = _FakeConversationLogger()
        from simple_chatbot.responses import ResponseStore
        server._response_store = ResponseStore()
        server._scenario_registry = load_fixtures()
        self.client = TestClient(server.app)

    def _restore(self):
        (server._config, server._agent, server._conversation_logger,
         server._response_store, server._scenario_registry,
         server._acompletion) = self.old

    def test_agent_path_forwards_max_output_tokens(self):
        resp = self.client.post(
            "/v1/responses",
            json={"model": "not-a-fixture", "input": "hi", "max_output_tokens": 64},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(self.captured[0]["max_tokens"], 64)

    def test_agent_path_omits_max_tokens_when_not_requested(self):
        resp = self.client.post(
            "/v1/responses", json={"model": "not-a-fixture", "input": "hi"}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertNotIn("max_tokens", self.captured[0])

    def test_live_scenario_path_forwards_max_output_tokens(self):
        server._config.scenario_mode = "live"
        server._acompletion = _capturing_acompletion(self.captured)
        resp = self.client.post(
            "/v1/responses",
            json={"model": "cs-routing", "input": "refund", "max_output_tokens": 32},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(self.captured[0]["max_tokens"], 32)

    def test_live_scenario_path_omits_max_tokens_when_not_requested(self):
        server._config.scenario_mode = "live"
        server._acompletion = _capturing_acompletion(self.captured)
        resp = self.client.post(
            "/v1/responses", json={"model": "cs-routing", "input": "refund"}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertNotIn("max_tokens", self.captured[0])


if __name__ == "__main__":
    unittest.main()
