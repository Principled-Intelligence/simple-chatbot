import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

import simple_chatbot.server as server
from simple_chatbot.agent import ChatResult
from simple_chatbot.config import SimpleChatbotConfig


class _FakeAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages: list[dict], max_output_tokens: int | None = None) -> ChatResult:
        self.calls += 1
        return ChatResult(content="ok", retrieved_chunks=[])


class _FakeConversationLogger:
    async def log(
        self,
        conversation_id: str,
        messages: list[dict],
        response: str,
        chunks: list[dict],
        misbehavior_injections: list[dict] | None = None,
    ) -> None:
        return None


class ServerAuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old_config = server._config
        self.old_agent = server._agent
        self.old_logger = server._conversation_logger
        self.addCleanup(self._restore_server_globals)

        self.agent = _FakeAgent()
        server._config = SimpleChatbotConfig(
            docs_dir=Path(self.tmp.name) / "docs",
            conversation_log_dir=Path(self.tmp.name) / "conversations",
            api_key="secret",
        )
        server._agent = self.agent
        server._conversation_logger = _FakeConversationLogger()
        self.client = TestClient(server.app)

    def _restore_server_globals(self) -> None:
        server._config = self.old_config
        server._agent = self.old_agent
        server._conversation_logger = self.old_logger

    def test_chat_completion_requires_configured_api_key(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.agent.calls, 0)

    def test_chat_completion_accepts_bearer_api_key(self):
        response = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "ok")
        self.assertEqual(self.agent.calls, 1)

    def test_chat_completion_accepts_x_api_key(self):
        response = self.client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": "secret"},
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "ok")
        self.assertEqual(self.agent.calls, 1)

    def test_chat_completion_rejects_streaming_before_agent_call(self):
        response = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.agent.calls, 0)
        detail = response.json()["detail"]["error"]
        self.assertEqual(detail["type"], "invalid_request_error")
        self.assertEqual(detail["param"], "stream")


if __name__ == "__main__":
    unittest.main()
