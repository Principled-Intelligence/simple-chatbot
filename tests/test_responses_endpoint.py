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
    async def log(self, conversation_id, messages, response, chunks, misbehavior_injections=None) -> None:
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

    def test_provider_failure_surfaces_as_structured_error(self):
        """A provider/agent failure must reach the client as a structured error
        envelope (so the OpenAI SDK / chat REPL can show the cause), not a bare
        unhandled 500 whose detail only lives in the server log."""
        import litellm

        class _RaisingAgent:
            async def chat(self, messages):
                raise litellm.exceptions.APIConnectionError(
                    message="Unable to convert openai tool calls to gemini tool calls",
                    llm_provider="vertex_ai",
                    model="gemini-2.5-pro",
                )

        server._agent = _RaisingAgent()
        client = TestClient(server.app, raise_server_exceptions=False)
        response = client.post("/v1/responses", json={"input": "hello"})

        self.assertEqual(response.status_code, 502, response.text)
        error = response.json()["detail"]["error"]
        self.assertEqual(error["type"], "upstream_provider_error")
        self.assertIn("APIConnectionError", error["message"])
        self.assertIn("gemini tool calls", error["message"])

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


class ResponsesChainTests(ResponsesEndpointTests):
    def test_chained_turn_prepends_prior_session_messages(self):
        first = self.client.post("/v1/responses", json={"input": "hi"})
        self.assertEqual(first.status_code, 200)
        first_body = first.json()
        first_id = first_body["id"]
        first_conv = first_body["conversation_id"]

        # Snapshot the messages the agent saw on turn 1
        turn1_messages = list(self.agent.calls[-1])

        second = self.client.post(
            "/v1/responses",
            json={"input": "follow-up", "previous_response_id": first_id},
        )
        self.assertEqual(second.status_code, 200, second.text)
        second_body = second.json()
        self.assertEqual(second_body["previous_response_id"], first_id)
        # Chained turn inherits the conversation_id
        self.assertEqual(second_body["conversation_id"], first_conv)

        # Turn 2's messages: prior session_messages + new user input
        turn2_messages = self.agent.calls[-1]
        # Should start with the prior turn's user + assistant
        self.assertEqual(turn2_messages[0], {"role": "user", "content": "hi"})
        self.assertEqual(turn2_messages[1], {"role": "assistant", "content": "ok"})
        # Then the new user input
        self.assertEqual(turn2_messages[-1], {"role": "user", "content": "follow-up"})

    def test_unknown_previous_response_id_returns_404(self):
        response = self.client.post(
            "/v1/responses",
            json={"input": "x", "previous_response_id": "resp_does_not_exist"},
        )
        self.assertEqual(response.status_code, 404)
        detail = response.json()["detail"]["error"]
        self.assertEqual(detail["param"], "previous_response_id")

    def test_previous_response_id_with_empty_input_returns_400(self):
        first = self.client.post("/v1/responses", json={"input": "hi"}).json()
        response = self.client.post(
            "/v1/responses",
            json={"input": "", "previous_response_id": first["id"]},
        )
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]["error"]
        self.assertEqual(detail["param"], "input")

    def test_three_turn_chain_preserves_full_history(self):
        first = self.client.post("/v1/responses", json={"input": "turn 1"}).json()
        second = self.client.post(
            "/v1/responses",
            json={"input": "turn 2", "previous_response_id": first["id"]},
        ).json()
        third = self.client.post(
            "/v1/responses",
            json={"input": "turn 3", "previous_response_id": second["id"]},
        )
        self.assertEqual(third.status_code, 200, third.text)
        third_body = third.json()
        # All three turns share the same conversation_id
        self.assertEqual(third_body["conversation_id"], first["conversation_id"])
        self.assertEqual(third_body["conversation_id"], second["conversation_id"])

        # Turn 3's agent input should contain all three prior turns in order.
        turn3_messages = self.agent.calls[-1]
        # First six entries: alternating user/assistant for turns 1 and 2,
        # then the new turn-3 user input.
        self.assertEqual(turn3_messages[0], {"role": "user", "content": "turn 1"})
        self.assertEqual(turn3_messages[1], {"role": "assistant", "content": "ok"})
        self.assertEqual(turn3_messages[2], {"role": "user", "content": "turn 2"})
        self.assertEqual(turn3_messages[3], {"role": "assistant", "content": "ok"})
        self.assertEqual(turn3_messages[-1], {"role": "user", "content": "turn 3"})


class ResponsesRetrieveTests(ResponsesEndpointTests):
    def test_get_returns_stored_response(self):
        created = self.client.post("/v1/responses", json={"input": "hi"}).json()
        got = self.client.get(f"/v1/responses/{created['id']}")
        self.assertEqual(got.status_code, 200)
        self.assertEqual(got.json(), created)

    def test_get_unknown_id_returns_404(self):
        got = self.client.get("/v1/responses/resp_unknown")
        self.assertEqual(got.status_code, 404)

    def test_get_respects_auth(self):
        server._config = SimpleChatbotConfig(
            docs_dir=Path(self.tmp.name) / "docs",
            conversation_log_dir=Path(self.tmp.name) / "conversations",
            api_key="secret",
        )
        created = self.client.post(
            "/v1/responses",
            json={"input": "hi"},
            headers={"Authorization": "Bearer secret"},
        ).json()

        no_auth = self.client.get(f"/v1/responses/{created['id']}")
        self.assertEqual(no_auth.status_code, 401)

        with_auth = self.client.get(
            f"/v1/responses/{created['id']}",
            headers={"Authorization": "Bearer secret"},
        )
        self.assertEqual(with_auth.status_code, 200)


if __name__ == "__main__":
    unittest.main()
