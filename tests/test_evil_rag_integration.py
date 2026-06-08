import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simple_chatbot.agent import Agent
from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.evil_rag import build_evil_agent
from simple_chatbot.loader import Document
from simple_chatbot.misbehavior import MisbehaviorConfig, MisbehaviorPolicy
from simple_chatbot.tools import SEARCH_TOOL_NAME


class _FakeIndexer:
    def document_count(self) -> int:
        return 1

    async def search(self, query: str) -> list:
        return [Document(text="Refunds are allowed within 14 days.", metadata={"source": "kb"})]


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = None

    def model_dump(self):
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ],
        }


class _Choice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, message, finish_reason):
        self.choices = [_Choice(message, finish_reason)]
        self.usage = None


async def _scripted_acompletion(**kwargs):
    """Round 1 → search; once a tool result is present → final answer grounded
    in whatever the tool returned."""
    msgs = kwargs["messages"]
    tool_outputs = [m["content"] for m in msgs if m.get("role") == "tool"]
    if tool_outputs:
        return _Resp(_Msg(content=f"Answer based on: {tool_outputs[-1]}"), "stop")
    return _Resp(_Msg(tool_calls=[_TC("call_1", SEARCH_TOOL_NAME, '{"query": "refund window"}')]), "tool_calls")


def _config(tmp):
    return SimpleChatbotConfig(docs_dir=Path(tmp) / "docs")


def _policy(rate, modes, seed=0):
    return MisbehaviorPolicy(MisbehaviorConfig(rate=rate, modes=tuple(modes), seed=seed))


class GoodVsEvilPairTests(unittest.TestCase):
    def test_good_agent_uses_retrieval_and_logs_no_injections(self):
        with TemporaryDirectory() as tmp:
            agent = Agent(_config(tmp), _FakeIndexer(), acompletion=_scripted_acompletion)
            result = asyncio.run(agent.chat([{"role": "user", "content": "refund window?"}]))
            tool_outputs = [m["content"] for m in result.tool_messages if m["role"] == "tool"]
            self.assertTrue(any("14 days" in out for out in tool_outputs))
            self.assertIn("14 days", result.content)

    def test_evil_poison_diverges_and_is_labeled(self):
        with TemporaryDirectory() as tmp:
            policy = _policy(1.0, ["poison_retrieval"], seed=1)
            agent = build_evil_agent(_config(tmp), _FakeIndexer(), policy, acompletion=_scripted_acompletion)
            result = asyncio.run(agent.chat([{"role": "user", "content": "refund window?"}]))
            tool_outputs = [m["content"] for m in result.tool_messages if m["role"] == "tool"]
            self.assertTrue(any("INJECTED" in out for out in tool_outputs))
            self.assertNotIn("14 days", result.content)  # answer no longer grounded in real KB
            self.assertEqual([i.mode for i in policy.injections], ["poison_retrieval"])

    def test_evil_run_is_reproducible_at_fixed_seed(self):
        with TemporaryDirectory() as tmp:
            def run():
                policy = _policy(0.5, ["poison_retrieval", "drop_retrieval", "redundant_search"], seed=42)
                agent = build_evil_agent(_config(tmp), _FakeIndexer(), policy, acompletion=_scripted_acompletion)
                asyncio.run(agent.chat([{"role": "user", "content": "refund window?"}]))
                return [(i.stage, i.mode) for i in policy.injections]

            self.assertEqual(run(), run())


import simple_chatbot.server as server
from fastapi.testclient import TestClient


class _FakeConversationLogger:
    async def log(self, conversation_id, messages, response, chunks, misbehavior_injections=None) -> None:
        return None


class EvilServerWiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = (
            server._config, server._agent, server._conversation_logger,
            getattr(server, "_response_store", None),
            getattr(server, "_scenario_registry", None),
            getattr(server, "_acompletion", None),
            getattr(server, "_misbehavior_policy", None),
        )
        self.addCleanup(self._restore)

    def _restore(self):
        (server._config, server._agent, server._conversation_logger,
         server._response_store, server._scenario_registry, server._acompletion,
         server._misbehavior_policy) = self.old

    def test_init_enables_evil_agent_and_endpoint_surfaces_labels(self):
        from simple_chatbot.config import SimpleChatbotConfig

        cfg = SimpleChatbotConfig(
            docs_dir=Path(self.tmp.name) / "docs",
            conversation_log_dir=Path(self.tmp.name) / "conversations",
            misbehavior_rate=1.0,
            misbehavior_modes=["drop_retrieval"],
            misbehavior_seed=3,
        )
        server.init(cfg, _FakeIndexer(), acompletion=_scripted_acompletion)
        server._conversation_logger = _FakeConversationLogger()
        client = TestClient(server.app)

        resp = client.post("/v1/responses", json={"input": "refund window?"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        modes = [i["mode"] for i in body.get("misbehavior_injections", [])]
        self.assertIn("drop_retrieval", modes)


if __name__ == "__main__":
    unittest.main()
