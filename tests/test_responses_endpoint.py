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
