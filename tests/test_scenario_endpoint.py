# tests/test_scenario_endpoint.py
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

import simple_chatbot.server as server
from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.scenario_registry import load_fixtures


class _FakeConversationLogger:
    async def log(self, conversation_id, messages, response, chunks, misbehavior_injections=None) -> None:
        return None


class _FakeAgent:
    async def chat(self, messages, max_output_tokens=None):
        from simple_chatbot.agent import ChatResult
        return ChatResult(content="agent-path", retrieved_chunks=[],
                          final_messages=list(messages) + [{"role": "assistant", "content": "agent-path"}])


class ScenarioEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = (
            server._config, server._agent, server._conversation_logger,
            getattr(server, "_response_store", None), getattr(server, "_scenario_registry", None),
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
        (server._config, server._agent, server._conversation_logger,
         server._response_store, server._scenario_registry,
         server._acompletion) = self.old

    def test_known_model_runs_scenario(self):
        resp = self.client.post("/v1/responses", json={"model": "cs-routing", "input": "refund"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        types = [it["type"] for it in body["output"]]
        # routing + sub-agent calls collapse into one output array
        self.assertIn("function_call", types)
        self.assertEqual(types[-1], "message")
        # catalog published in Response.tools
        names = {t["name"] for t in body["tools"]}
        self.assertIn("route", names)
        self.assertIn("lookup_invoice", names)

    def test_unknown_model_falls_back_to_agent(self):
        resp = self.client.post("/v1/responses", json={"model": "not-a-fixture", "input": "hi"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["output"][-1]["content"][0]["text"], "agent-path")

    def test_models_lists_fixtures(self):
        resp = self.client.get("/v1/models")
        ids = {m["id"] for m in resp.json()["data"]}
        self.assertIn("cs-routing", ids)
        self.assertIn("validity-probe", ids)


class ScenarioTraceShapeTests(ScenarioEndpointTests):
    def test_turn_window_has_route_then_subagent_calls(self):
        resp = self.client.post("/v1/responses", json={"model": "cs-routing", "input": "refund"})
        body = resp.json()
        # Emulate Spectral's per-turn windowing: one user message in => one turn;
        # the output array IS that turn's window. Extract ordered tool-call names.
        call_names = [
            it["name"] for it in body["output"] if it["type"] == "function_call"
        ]
        self.assertEqual(call_names[0], "route")  # routing decision is in-window
        self.assertIn("lookup_invoice", call_names)  # routed sub-agent's call too
        # function_call_output items are paired and present
        out_types = [it["type"] for it in body["output"]]
        self.assertIn("function_call_output", out_types)

    def test_catalog_resolves_calls_by_name(self):
        resp = self.client.post("/v1/responses", json={"model": "cs-routing", "input": "refund"})
        body = resp.json()
        catalog_names = {t["name"] for t in body["tools"]}
        call_names = {it["name"] for it in body["output"] if it["type"] == "function_call"}
        # every emitted call resolves to a catalog entry by name (validity precondition)
        self.assertTrue(call_names.issubset(catalog_names))


# These fakes mirror litellm's response object shape for the live path.
class _LiveFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _LiveToolCall:
    def __init__(self, name, arguments):
        self.function = _LiveFunction(name, arguments)


class _LiveMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _LiveChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _LiveResponse:
    def __init__(self, message, finish_reason):
        self.choices = [_LiveChoice(message, finish_reason)]
        self.usage = None


class ScenarioModeAndResumeTests(ScenarioEndpointTests):
    def test_deterministic_resume_stays_in_routed_agent(self):
        # turn 1: dispatcher routes to billing, billing answers
        r1 = self.client.post("/v1/responses", json={"model": "cs-routing", "input": "refund"})
        self.assertEqual(r1.status_code, 200, r1.text)
        rid = r1.json()["id"]
        # turn 2: chained — must resume in billing (no route call), per the requirement
        r2 = self.client.post(
            "/v1/responses",
            json={"model": "cs-routing", "input": "and the receipt?", "previous_response_id": rid},
        )
        self.assertEqual(r2.status_code, 200, r2.text)
        call_names = [it["name"] for it in r2.json()["output"] if it["type"] == "function_call"]
        self.assertNotIn("route", call_names)
        self.assertEqual(call_names[0], "lookup_invoice")

    def test_server_mode_override_selects_live_provider(self):
        # force the whole server live; inject a fake acompletion that finalizes immediately
        server._config.scenario_mode = "live"

        async def fake_acompletion(**kwargs):
            return _LiveResponse(_LiveMessage(content="live answer"), "stop")

        server._acompletion = fake_acompletion
        resp = self.client.post("/v1/responses", json={"model": "cs-routing", "input": "refund"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["output"][-1]["content"][0]["text"], "live answer")
        # catalog still published in live mode
        self.assertIn("route", {t["name"] for t in body["tools"]})
