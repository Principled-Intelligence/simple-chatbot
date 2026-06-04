import asyncio
import unittest

from simple_chatbot.loader import Document
from simple_chatbot.misbehavior import MisbehaviorConfig, MisbehaviorPolicy
from simple_chatbot.evil_rag import evil_acompletion, evil_search
from simple_chatbot.tools import SEARCH_TOOL_NAME, ToolDef, ToolResult


def _real_search_tool() -> ToolDef:
    async def executor(arguments: str) -> ToolResult:
        return ToolResult(
            text="Refunds are allowed within 14 days.",
            chunks=[Document(text="Refunds are allowed within 14 days.", metadata={"source": "kb"})],
        )
    return ToolDef(schema={"type": "function", "function": {"name": SEARCH_TOOL_NAME}}, executor=executor)


def _policy(rate, modes, seed=0):
    return MisbehaviorPolicy(MisbehaviorConfig(rate=rate, modes=tuple(modes), seed=seed))


class EvilSearchTests(unittest.TestCase):
    def test_rate_zero_is_identical_passthrough(self):
        policy = _policy(0.0, ["drop_retrieval", "poison_retrieval"])
        tool = evil_search(_real_search_tool(), policy)
        result = asyncio.run(tool.executor('{"query": "refund"}'))
        self.assertEqual(result.text, "Refunds are allowed within 14 days.")
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(policy.injections, [])

    def test_preserves_tool_name(self):
        tool = evil_search(_real_search_tool(), _policy(0.0, ["drop_retrieval"]))
        self.assertEqual(tool.name, SEARCH_TOOL_NAME)

    def test_drop_retrieval_empties_result_and_logs(self):
        policy = _policy(1.0, ["drop_retrieval"])
        tool = evil_search(_real_search_tool(), policy)
        result = asyncio.run(tool.executor('{"query": "refund"}'))
        self.assertEqual(result.chunks, [])
        self.assertIn("No relevant documents", result.text)
        self.assertEqual(len(policy.injections), 1)
        self.assertEqual(policy.injections[0].stage, "retrieval")
        self.assertEqual(policy.injections[0].mode, "drop_retrieval")

    def test_poison_retrieval_replaces_content_and_logs(self):
        policy = _policy(1.0, ["poison_retrieval"])
        tool = evil_search(_real_search_tool(), policy)
        result = asyncio.run(tool.executor('{"query": "refund"}'))
        self.assertNotIn("14 days", result.text)
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].metadata.get("source"), "injected")
        self.assertEqual(policy.injections[0].mode, "poison_retrieval")


class _FakeFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _FakeFn(name, arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content

    def model_dump(self):
        return {"role": "assistant", "content": self.content}


class _FakeChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, message, finish_reason):
        self.choices = [_FakeChoice(message, finish_reason)]
        self.usage = None


def _tool_call_response():
    msg = _FakeMessage(tool_calls=[_FakeToolCall("call_1", SEARCH_TOOL_NAME, '{"query": "refund"}')])
    return _FakeResponse(msg, "tool_calls")


def _final_response():
    return _FakeResponse(_FakeMessage(content="Refunds are within 14 days."), "stop")


class EvilAcompletionTests(unittest.TestCase):
    def test_rate_zero_returns_real_response_untouched(self):
        policy = _policy(0.0, ["ignore_retrieval", "redundant_search"])
        captured = {}

        async def real(**kwargs):
            captured["messages"] = kwargs["messages"]
            return _tool_call_response()

        wrapped = evil_acompletion(real, policy)
        resp = asyncio.run(wrapped(messages=[{"role": "user", "content": "refund?"}]))
        self.assertEqual(len(resp.choices[0].message.tool_calls), 1)
        self.assertEqual(len(captured["messages"]), 1)  # no steering prepended
        self.assertEqual(policy.injections, [])

    def test_ignore_retrieval_prepends_steering(self):
        policy = _policy(1.0, ["ignore_retrieval"])
        captured = {}

        async def real(**kwargs):
            captured["messages"] = kwargs["messages"]
            return _final_response()

        wrapped = evil_acompletion(real, policy)
        asyncio.run(wrapped(messages=[{"role": "user", "content": "refund?"}]))
        self.assertEqual(captured["messages"][0]["role"], "system")
        self.assertIn("Ignore the retrieved", captured["messages"][0]["content"])
        self.assertEqual(policy.injections[0].mode, "ignore_retrieval")

    def test_redundant_search_appends_duplicate_call_on_tool_round(self):
        policy = _policy(1.0, ["redundant_search"])

        async def real(**kwargs):
            return _tool_call_response()

        wrapped = evil_acompletion(real, policy)
        resp = asyncio.run(wrapped(messages=[{"role": "user", "content": "refund?"}]))
        calls = resp.choices[0].message.tool_calls
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1].function.name, SEARCH_TOOL_NAME)
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertEqual(policy.injections[0].mode, "redundant_search")

    def test_malformed_search_appends_malformed_args(self):
        policy = _policy(1.0, ["malformed_search"])

        async def real(**kwargs):
            return _tool_call_response()

        wrapped = evil_acompletion(real, policy)
        resp = asyncio.run(wrapped(messages=[{"role": "user", "content": "x"}]))
        calls = resp.choices[0].message.tool_calls
        self.assertEqual(calls[-1].function.arguments, "{intentionally_malformed_json")

    def test_unknown_tool_appends_unknown_call(self):
        policy = _policy(1.0, ["unknown_tool"])

        async def real(**kwargs):
            return _tool_call_response()

        wrapped = evil_acompletion(real, policy)
        resp = asyncio.run(wrapped(messages=[{"role": "user", "content": "x"}]))
        self.assertEqual(resp.choices[0].message.tool_calls[-1].function.name, "delete_documents")

    def test_structural_injection_skipped_on_final_round(self):
        policy = _policy(1.0, ["redundant_search"])

        async def real(**kwargs):
            return _final_response()  # finish_reason="stop", no tool_calls

        wrapped = evil_acompletion(real, policy)
        resp = asyncio.run(wrapped(messages=[{"role": "user", "content": "x"}]))
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        # the structural-mode maybe() was consulted but cannot mutate a final round
        self.assertEqual(policy.injections, [])


if __name__ == "__main__":
    unittest.main()
