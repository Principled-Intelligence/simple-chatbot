import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from simple_chatbot.agent import Agent
from simple_chatbot.config import SimpleChatbotConfig


class _FakeIndexer:
    def __init__(self) -> None:
        self.search_calls: list[str] = []

    def document_count(self) -> int:
        return 1

    async def search(self, query: str) -> list:
        self.search_calls.append(query)
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
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ],
        }


class _Choice:
    def __init__(self, message, finish_reason) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt, completion, total) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class _Response:
    def __init__(self, message, finish_reason, usage=None) -> None:
        self.choices = [_Choice(message, finish_reason)]
        self.usage = usage


def _config(tmp: str) -> SimpleChatbotConfig:
    return SimpleChatbotConfig(docs_dir=Path(tmp) / "docs")


class AgentUsageTests(unittest.TestCase):
    def test_usage_sums_across_all_litellm_calls(self):
        with TemporaryDirectory() as tmp:
            agent = Agent(_config(tmp), _FakeIndexer())
            responses = [
                _Response(
                    _Message(tool_calls=[_ToolCall("call_1", "search_documents", '{"query": "a"}')]),
                    "tool_calls",
                    usage=_Usage(10, 5, 15),
                ),
                _Response(_Message(content="done"), "stop", usage=_Usage(20, 7, 27)),
            ]
            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                result = asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))

            self.assertEqual(result.usage["prompt_tokens"], 30)
            self.assertEqual(result.usage["completion_tokens"], 12)
            self.assertEqual(result.usage["total_tokens"], 42)


class AgentFinalMessagesTests(unittest.TestCase):
    def test_final_messages_includes_final_assistant_when_no_tools(self):
        with TemporaryDirectory() as tmp:
            agent = Agent(_config(tmp), _FakeIndexer())
            responses = [_Response(_Message(content="hello"), "stop", usage=None)]
            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                result = asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))

            # Should contain the user input plus the final assistant message
            roles = [m["role"] for m in result.final_messages]
            self.assertEqual(roles[-1], "assistant")
            self.assertEqual(result.final_messages[-1]["content"], "hello")
            self.assertIn({"role": "user", "content": "hi"}, result.final_messages)

    def test_final_messages_includes_tool_loop_in_order(self):
        with TemporaryDirectory() as tmp:
            agent = Agent(_config(tmp), _FakeIndexer())
            responses = [
                _Response(
                    _Message(tool_calls=[_ToolCall("call_1", "search_documents", '{"query": "a"}')]),
                    "tool_calls",
                ),
                _Response(_Message(content="answered"), "stop"),
            ]
            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                result = asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))

            roles = [m["role"] for m in result.final_messages]
            # user, assistant(with tool_calls), tool, assistant(final)
            self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
            self.assertEqual(result.final_messages[-1]["content"], "answered")


class AgentSystemPromptTests(unittest.TestCase):
    def test_does_not_double_prepend_system_prompt(self):
        with TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            cfg = cfg.model_copy(update={"system_prompt": "You are a bot."})
            agent = Agent(cfg, _FakeIndexer())
            responses = [_Response(_Message(content="ok"), "stop")]

            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                # Caller already includes a system message — agent must not add another.
                asyncio.run(agent.chat([
                    {"role": "system", "content": "You are a bot."},
                    {"role": "user", "content": "hi"},
                ]))

            sent_messages = completion.await_args_list[0].kwargs["messages"]
            system_count = sum(1 for m in sent_messages if m.get("role") == "system")
            self.assertEqual(system_count, 1)

    def test_prepends_system_prompt_when_not_already_present(self):
        with TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            cfg = cfg.model_copy(update={"system_prompt": "You are a bot."})
            agent = Agent(cfg, _FakeIndexer())
            responses = [_Response(_Message(content="ok"), "stop")]

            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))

            sent_messages = completion.await_args_list[0].kwargs["messages"]
            self.assertEqual(sent_messages[0]["role"], "system")
            self.assertEqual(sent_messages[0]["content"], "You are a bot.")


class AgentReasoningCaptureTests(unittest.TestCase):
    def test_agent_captures_final_reasoning_content_when_present(self):
        with TemporaryDirectory() as tmp:
            agent = Agent(_config(tmp), _FakeIndexer())

            # Build a fake response where the message exposes reasoning_content
            class _MsgWithReasoning(_Message):
                def __init__(self, content="", reasoning_content=None):
                    super().__init__(content=content)
                    self.reasoning_content = reasoning_content

            responses = [
                _Response(_MsgWithReasoning(content="answer", reasoning_content="my reasoning"), "stop"),
            ]

            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                result = asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))

            self.assertEqual(result.final_reasoning_content, "my reasoning")

    def test_agent_captures_intermediate_reasoning_content_in_tool_messages(self):
        with TemporaryDirectory() as tmp:
            agent = Agent(_config(tmp), _FakeIndexer())

            class _MsgWithToolsAndReasoning(_Message):
                def __init__(self, tool_calls=None, reasoning_content=None):
                    super().__init__(tool_calls=tool_calls)
                    self.reasoning_content = reasoning_content

                def model_dump(self):
                    d = super().model_dump()
                    return d  # base model_dump doesn't include reasoning_content; agent must inject it

            responses = [
                _Response(
                    _MsgWithToolsAndReasoning(
                        tool_calls=[_ToolCall("call_1", "search_documents", '{"query": "x"}')],
                        reasoning_content="intermediate reasoning",
                    ),
                    "tool_calls",
                ),
                _Response(_Message(content="done"), "stop"),
            ]

            with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
                completion.side_effect = responses
                result = asyncio.run(agent.chat([{"role": "user", "content": "hi"}]))

            assistant_msg = result.tool_messages[0]
            self.assertEqual(assistant_msg.get("reasoning_content"), "intermediate reasoning")


if __name__ == "__main__":
    unittest.main()
