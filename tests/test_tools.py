import asyncio
import unittest

from simple_chatbot.loader import Document
from simple_chatbot.tools import (
    SEARCH_TOOL_NAME,
    SEARCH_TOOL_SCHEMA,
    ToolDef,
    ToolResult,
    make_search_tool,
)


class _FakeIndexer:
    def __init__(self, results=None):
        self._results = results or []

    async def search(self, query: str):
        self._last_query = query
        return self._results


class MakeSearchToolTests(unittest.TestCase):
    def test_search_tool_def_has_correct_name_and_schema(self):
        tool = make_search_tool(_FakeIndexer())
        self.assertEqual(tool.name, SEARCH_TOOL_NAME)
        self.assertIs(tool.schema, SEARCH_TOOL_SCHEMA)

    def test_executor_returns_results_with_chunks(self):
        docs = [Document(text="hello", metadata={"source": "a.md"})]
        tool = make_search_tool(_FakeIndexer(docs))
        result = asyncio.run(tool.executor('{"query": "hi"}'))
        self.assertIn("hello", result.text)
        self.assertIn("a.md", result.text)
        self.assertEqual(result.chunks, docs)

    def test_executor_returns_no_results_text_when_index_empty(self):
        tool = make_search_tool(_FakeIndexer())
        result = asyncio.run(tool.executor('{"query": "hi"}'))
        self.assertEqual(result.text, "No relevant documents found.")
        self.assertEqual(result.chunks, [])

    def test_executor_rejects_invalid_json(self):
        tool = make_search_tool(_FakeIndexer())
        result = asyncio.run(tool.executor('{not json'))
        self.assertIn("Tool error:", result.text)
        self.assertIn("invalid JSON", result.text)

    def test_executor_rejects_missing_query(self):
        tool = make_search_tool(_FakeIndexer())
        result = asyncio.run(tool.executor('{"unrelated": "x"}'))
        self.assertIn("Tool error:", result.text)
        self.assertIn("non-empty string 'query'", result.text)


class AgentRegistryTests(unittest.TestCase):
    def test_agent_default_registers_search_only(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from simple_chatbot.agent import Agent
        from simple_chatbot.config import SimpleChatbotConfig

        with TemporaryDirectory() as tmp:
            agent = Agent(
                SimpleChatbotConfig(docs_dir=Path(tmp) / "docs"),
                _FakeIndexer(),
            )
            self.assertEqual(list(agent._tool_by_name), [SEARCH_TOOL_NAME])

    def test_agent_accepts_custom_tools(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from simple_chatbot.agent import Agent
        from simple_chatbot.config import SimpleChatbotConfig

        async def custom_executor(args: str):
            return ToolResult(text="custom result")

        custom_tool = ToolDef(
            schema={
                "type": "function",
                "function": {
                    "name": "custom_tool",
                    "description": "test",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            executor=custom_executor,
        )

        with TemporaryDirectory() as tmp:
            agent = Agent(
                SimpleChatbotConfig(docs_dir=Path(tmp) / "docs"),
                _FakeIndexer(),
                tools=[custom_tool],
            )
            self.assertEqual(list(agent._tool_by_name), ["custom_tool"])


from simple_chatbot.tools import (
    CALCULATE_TOOL_NAME,
    GET_CURRENT_TIME_TOOL_NAME,
    LOOKUP_USER_TOOL_NAME,
    make_calculate_tool,
    make_get_current_time_tool,
    make_lookup_user_tool,
    scripted_tools,
)


class CalculateToolTests(unittest.TestCase):
    def test_echoes_expression_in_result(self):
        tool = make_calculate_tool()
        result = asyncio.run(tool.executor('{"expression": "2+2"}'))
        self.assertIn("'2+2'", result.text)
        self.assertIn("[scripted]", result.text)
        self.assertEqual(result.chunks, [])

    def test_rejects_missing_expression(self):
        tool = make_calculate_tool()
        result = asyncio.run(tool.executor('{}'))
        self.assertIn("Tool error:", result.text)
        self.assertIn("expression", result.text)

    def test_rejects_invalid_json(self):
        tool = make_calculate_tool()
        result = asyncio.run(tool.executor('{not json'))
        self.assertIn("Tool error:", result.text)


class GetCurrentTimeToolTests(unittest.TestCase):
    def test_no_tz_returns_canned_time(self):
        tool = make_get_current_time_tool()
        result = asyncio.run(tool.executor('{}'))
        self.assertIn("current time:", result.text)

    def test_with_tz_echoes_tz(self):
        tool = make_get_current_time_tool()
        result = asyncio.run(tool.executor('{"tz": "Europe/London"}'))
        self.assertIn("Europe/London", result.text)

    def test_rejects_non_string_tz(self):
        tool = make_get_current_time_tool()
        result = asyncio.run(tool.executor('{"tz": 123}'))
        self.assertIn("Tool error:", result.text)


class LookupUserToolTests(unittest.TestCase):
    def test_echoes_user_id_into_record(self):
        tool = make_lookup_user_tool()
        result = asyncio.run(tool.executor('{"user_id": "alice"}'))
        self.assertIn("alice", result.text)
        self.assertIn("@example.test", result.text)

    def test_rejects_missing_user_id(self):
        tool = make_lookup_user_tool()
        result = asyncio.run(tool.executor('{}'))
        self.assertIn("Tool error:", result.text)
        self.assertIn("user_id", result.text)


class ScriptedToolsTests(unittest.TestCase):
    def test_returns_four_tools_with_expected_names(self):
        # Use the existing _FakeIndexer at the top of the file.
        tools = scripted_tools(_FakeIndexer())
        names = [t.name for t in tools]
        self.assertEqual(
            names,
            [
                SEARCH_TOOL_NAME,
                CALCULATE_TOOL_NAME,
                GET_CURRENT_TIME_TOOL_NAME,
                LOOKUP_USER_TOOL_NAME,
            ],
        )


if __name__ == "__main__":
    unittest.main()
