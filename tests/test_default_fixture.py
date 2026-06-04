# tests/test_default_fixture.py
"""Server-level default fixture: run a preselected fixture when the request
does not name one (Spectral cannot set the model)."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

import simple_chatbot.server as server
from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.scenario_registry import load_fixtures


class DefaultFixtureConfigTests(unittest.TestCase):
    def test_defaults_to_none(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(docs_dir=Path(tmp) / "docs")
            self.assertIsNone(config.default_fixture)

    def test_settable(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(
                docs_dir=Path(tmp) / "docs", default_fixture="parallel"
            )
            self.assertEqual(config.default_fixture, "parallel")


class _FakeConversationLogger:
    async def log(self, conversation_id, messages, response, chunks) -> None:
        return None


class _FakeAgent:
    async def chat(self, messages):
        from simple_chatbot.agent import ChatResult

        return ChatResult(
            content="agent-path",
            retrieved_chunks=[],
            final_messages=list(messages) + [{"role": "assistant", "content": "agent-path"}],
        )


class _ServerHarness(unittest.TestCase):
    """Mirrors tests/test_scenario_endpoint.py's in-process server harness."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = (
            server._config,
            server._agent,
            server._conversation_logger,
            getattr(server, "_response_store", None),
            getattr(server, "_scenario_registry", None),
            getattr(server, "_acompletion", None),
        )
        self.addCleanup(self._restore)

        server._config = SimpleChatbotConfig(
            docs_dir=Path(self.tmp.name) / "docs",
            conversation_log_dir=Path(self.tmp.name) / "conversations",
        )
        server._agent = _FakeAgent()
        server._conversation_logger = _FakeConversationLogger()
        from simple_chatbot.responses import ResponseStore

        server._response_store = ResponseStore()
        server._scenario_registry = load_fixtures()
        self.client = TestClient(server.app)

    def _restore(self):
        (
            server._config,
            server._agent,
            server._conversation_logger,
            server._response_store,
            server._scenario_registry,
            server._acompletion,
        ) = self.old


class DefaultFixtureEndpointTests(_ServerHarness):
    def _final_text(self, body):
        return body["output"][-1]["content"][0]["text"]

    def _call_names(self, body):
        return [it["name"] for it in body["output"] if it["type"] == "function_call"]

    def test_omitted_model_uses_default_fixture(self):
        server._config.default_fixture = "parallel"
        resp = self.client.post("/v1/responses", json={"input": "Lisbon this weekend?"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(self._call_names(body), ["get_weather", "get_events"])
        self.assertEqual(
            self._final_text(body),
            "Lisbon will be sunny, and there's a jazz festival this weekend.",
        )

    def test_unknown_model_uses_default_fixture(self):
        server._config.default_fixture = "parallel"
        resp = self.client.post(
            "/v1/responses", json={"model": "some-llm", "input": "Lisbon?"}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(self._call_names(resp.json()), ["get_weather", "get_events"])

    def test_explicit_fixture_overrides_default(self):
        server._config.default_fixture = "parallel"
        resp = self.client.post(
            "/v1/responses", json={"model": "cs-routing", "input": "refund"}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        # cs-routing ran (it has a route call); parallel has none
        self.assertIn("route", self._call_names(resp.json()))

    def test_no_default_unknown_model_still_falls_back_to_agent(self):
        # default_fixture is None: pre-existing behavior is unchanged
        resp = self.client.post(
            "/v1/responses", json={"model": "not-a-fixture", "input": "hi"}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(self._final_text(resp.json()), "agent-path")

    def test_default_fixture_resumes_active_agent_across_turns(self):
        server._config.default_fixture = "cs-routing"
        r1 = self.client.post("/v1/responses", json={"input": "refund"})
        self.assertEqual(r1.status_code, 200, r1.text)
        rid = r1.json()["id"]
        r2 = self.client.post(
            "/v1/responses",
            json={"input": "and the receipt?", "previous_response_id": rid},
        )
        self.assertEqual(r2.status_code, 200, r2.text)
        names = self._call_names(r2.json())
        self.assertNotIn("route", names)  # resumed in the routed sub-agent
        self.assertEqual(names[0], "lookup_invoice")


class DefaultFixtureInitValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = (
            server._config,
            server._agent,
            server._conversation_logger,
            getattr(server, "_response_store", None),
            getattr(server, "_scenario_registry", None),
            getattr(server, "_acompletion", None),
        )
        self.addCleanup(self._restore)

    def _restore(self):
        (
            server._config,
            server._agent,
            server._conversation_logger,
            server._response_store,
            server._scenario_registry,
            server._acompletion,
        ) = self.old

    def _config(self, **kw):
        return SimpleChatbotConfig(
            docs_dir=Path(self.tmp.name) / "docs",
            conversation_log_dir=Path(self.tmp.name) / "conversations",
            **kw,
        )

    def test_init_rejects_unknown_default_fixture(self):
        from simple_chatbot.scripted_indexer import ScriptedIndexer

        with self.assertRaises(ValueError):
            server.init(self._config(default_fixture="nope"), ScriptedIndexer(), tools=[])

    def test_init_accepts_known_default_fixture(self):
        from simple_chatbot.scripted_indexer import ScriptedIndexer

        server.init(self._config(default_fixture="parallel"), ScriptedIndexer(), tools=[])
        self.assertEqual(server._config.default_fixture, "parallel")


if __name__ == "__main__":
    unittest.main()
