import asyncio
import unittest

from simple_chatbot.loader import Document
from simple_chatbot.misbehavior import MisbehaviorConfig, MisbehaviorPolicy
from simple_chatbot.evil_rag import evil_search
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


if __name__ == "__main__":
    unittest.main()
