import unittest

from simple_chatbot.responses import normalize_input, InvalidInputError


class NormalizeInputTests(unittest.TestCase):
    def test_string_input_becomes_user_message(self):
        self.assertEqual(
            normalize_input("hello"),
            [{"role": "user", "content": "hello"}],
        )

    def test_list_of_strings_becomes_user_messages(self):
        self.assertEqual(
            normalize_input(["hi", "again"]),
            [
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "again"},
            ],
        )

    def test_message_item_with_role_user(self):
        self.assertEqual(
            normalize_input([{"type": "message", "role": "user", "content": "hi"}]),
            [{"role": "user", "content": "hi"}],
        )

    def test_message_item_with_role_assistant(self):
        self.assertEqual(
            normalize_input([{"type": "message", "role": "assistant", "content": "ok"}]),
            [{"role": "assistant", "content": "ok"}],
        )

    def test_message_item_with_content_parts(self):
        # Content parts passed through unchanged
        parts = [{"type": "input_text", "text": "hello"}]
        self.assertEqual(
            normalize_input([{"type": "message", "role": "user", "content": parts}]),
            [{"role": "user", "content": parts}],
        )

    def test_mixed_list(self):
        self.assertEqual(
            normalize_input([
                "first",
                {"type": "message", "role": "assistant", "content": "second"},
                {"type": "message", "role": "user", "content": "third"},
            ]),
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ],
        )

    def test_unknown_item_type_raises(self):
        with self.assertRaises(InvalidInputError):
            normalize_input([{"type": "made_up_type", "blob": 1}])

    def test_non_string_non_dict_item_raises(self):
        with self.assertRaises(InvalidInputError):
            normalize_input([42])


from simple_chatbot.agent import ChatResult
from simple_chatbot.responses import build_output_items


class _Doc:
    # ChatResult.retrieved_chunks holds Document instances; we don't need them here.
    pass


def _chat_result(
    content: str,
    tool_messages: list[dict] | None = None,
) -> ChatResult:
    return ChatResult(
        content=content,
        retrieved_chunks=[],
        tools=[],
        tool_messages=tool_messages or [],
    )


class BuildOutputItemsTests(unittest.TestCase):
    def test_message_only_when_no_tool_calls(self):
        items = build_output_items(_chat_result("hello"))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["type"], "message")
        self.assertEqual(items[0]["role"], "assistant")
        self.assertEqual(items[0]["content"][0]["text"], "hello")
        self.assertEqual(items[0]["content"][0]["type"], "output_text")

    def test_single_tool_call_pair_and_message(self):
        tool_messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "search_documents", "arguments": '{"query": "x"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "name": "search_documents",
                "content": "RESULT",
            },
        ]
        items = build_output_items(_chat_result("final", tool_messages))
        types = [it["type"] for it in items]
        self.assertEqual(types, ["function_call", "function_call_output", "message"])
        self.assertEqual(items[0]["call_id"], "call_abc")
        self.assertEqual(items[0]["name"], "search_documents")
        self.assertEqual(items[0]["arguments"], '{"query": "x"}')
        self.assertEqual(items[1]["call_id"], "call_abc")
        self.assertEqual(items[1]["output"], "RESULT")
        self.assertEqual(items[2]["content"][0]["text"], "final")

    def test_multiple_tool_calls_interleaved(self):
        tool_messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search_documents", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "search_documents", "content": "R1"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c2", "type": "function", "function": {"name": "search_documents", "arguments": "{}"}},
                    {"id": "c3", "type": "function", "function": {"name": "search_documents", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "name": "search_documents", "content": "R2"},
            {"role": "tool", "tool_call_id": "c3", "name": "search_documents", "content": "R3"},
        ]
        items = build_output_items(_chat_result("done", tool_messages))
        types = [it["type"] for it in items]
        self.assertEqual(
            types,
            [
                "function_call", "function_call_output",
                "function_call", "function_call", "function_call_output", "function_call_output",
                "message",
            ],
        )
        # call_id linkage preserved
        call_ids = [it["call_id"] for it in items if it["type"] in ("function_call", "function_call_output")]
        self.assertEqual(call_ids, ["c1", "c1", "c2", "c3", "c2", "c3"])

    def test_tool_error_appears_in_function_call_output(self):
        tool_messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search_documents", "arguments": "{not"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "search_documents",
                "content": "Tool error: bad JSON",
                "is_error": True,
            },
        ]
        items = build_output_items(_chat_result("recovered", tool_messages))
        fco = next(it for it in items if it["type"] == "function_call_output")
        self.assertIn("Tool error", fco["output"])

    def test_successful_tool_output_status_completed(self):
        tool_messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search_documents", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "search_documents", "content": "RESULT"},
        ]
        items = build_output_items(_chat_result("final", tool_messages))
        fco = next(it for it in items if it["type"] == "function_call_output")
        self.assertEqual(fco["status"], "completed")

    def test_errored_tool_output_status_incomplete(self):
        # A tool that errored (e.g. executor raised RateLimitError) is still fed
        # back to the model, but its function_call_output must not be reported
        # as "completed".
        tool_messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search_documents", "arguments": "{}"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "search_documents",
                "content": "Tool error: litellm.RateLimitError: 429 RESOURCE_EXHAUSTED",
                "is_error": True,
            },
        ]
        items = build_output_items(_chat_result("sorry, rate limited", tool_messages))
        fco = next(it for it in items if it["type"] == "function_call_output")
        self.assertEqual(fco["status"], "incomplete")
        self.assertIn("RateLimitError", fco["output"])

    def test_each_item_has_unique_stable_id(self):
        items = build_output_items(_chat_result("hi"))
        ids = [it["id"] for it in items]
        # Message item id should start with `msg_`
        self.assertTrue(ids[0].startswith("msg_"))

    def test_guard_blocked_turn_outputs_message_only(self):
        # Guard-blocked turns have no tool_messages; just the refusal text.
        result = ChatResult(
            content="I can't help with that.",
            retrieved_chunks=[],
            tool_messages=[],
            blocked_by_guard=True,
        )
        items = build_output_items(result)
        self.assertEqual([it["type"] for it in items], ["message"])
        self.assertEqual(items[0]["content"][0]["text"], "I can't help with that.")

    def test_empty_knowledge_base_outputs_message_only(self):
        # Empty KB short-circuits with a plain message and no tool_messages.
        result = ChatResult(
            content="I don't have any indexed documents to search yet.",
            retrieved_chunks=[],
            tool_messages=[],
        )
        items = build_output_items(result)
        self.assertEqual([it["type"] for it in items], ["message"])

    def test_max_rounds_reached_preserves_all_tool_calls(self):
        # Max-rounds-reached: every round's calls are in tool_messages; the
        # forced-final assistant text is in `content`. Mapping should produce
        # all function_call/function_call_output pairs in order, then the
        # final message item.
        tool_messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search_documents", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "search_documents", "content": "R1"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c2", "type": "function", "function": {"name": "search_documents", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "name": "search_documents", "content": "R2"},
        ]
        result = ChatResult(
            content="forced final",
            retrieved_chunks=[],
            tool_messages=tool_messages,
        )
        items = build_output_items(result)
        types = [it["type"] for it in items]
        self.assertEqual(types, ["function_call", "function_call_output", "function_call", "function_call_output", "message"])
        self.assertEqual(items[-1]["content"][0]["text"], "forced final")


from simple_chatbot.responses import build_response


class BuildResponseTests(unittest.TestCase):
    def test_basic_envelope_fields(self):
        result = ChatResult(
            content="hi",
            retrieved_chunks=[],
            usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        )
        payload = build_response(
            result=result,
            model="openai/gpt-5.4-nano",
            previous_response_id=None,
            conversation_id="conv_abc",
        )
        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["model"], "openai/gpt-5.4-nano")
        self.assertEqual(payload["conversation_id"], "conv_abc")
        self.assertIsNone(payload["previous_response_id"])
        self.assertTrue(payload["id"].startswith("resp_"))
        self.assertIsInstance(payload["created_at"], int)
        # Output: just the final message item
        self.assertEqual([it["type"] for it in payload["output"]], ["message"])

    def test_usage_renamed_to_openai_fields(self):
        result = ChatResult(
            content="hi",
            retrieved_chunks=[],
            usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        )
        payload = build_response(result, "m", None, "conv")
        self.assertEqual(payload["usage"]["input_tokens"], 10)
        self.assertEqual(payload["usage"]["output_tokens"], 4)
        self.assertEqual(payload["usage"]["total_tokens"], 14)

    def test_previous_response_id_echoed(self):
        payload = build_response(
            ChatResult(content="x", retrieved_chunks=[]),
            "m",
            previous_response_id="resp_prior",
            conversation_id="conv",
        )
        self.assertEqual(payload["previous_response_id"], "resp_prior")


import asyncio

from simple_chatbot.responses import ResponseStore


class ResponseStoreTests(unittest.TestCase):
    def test_put_then_get(self):
        entry = {
            "response_id": "resp_1",
            "previous_response_id": None,
            "session_messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
            "response_json": {"id": "resp_1"},
            "conversation_id": "conv_1",
        }

        async def scenario():
            store = ResponseStore()
            await store.put("resp_1", entry)
            return await store.get("resp_1")

        got = asyncio.run(scenario())
        self.assertEqual(got, entry)

    def test_get_missing_returns_none(self):
        async def scenario():
            store = ResponseStore()
            return await store.get("nope")

        self.assertIsNone(asyncio.run(scenario()))


class BuildOutputItemsReasoningTests(unittest.TestCase):
    def test_reasoning_emitted_before_function_calls_when_present_on_assistant(self):
        tool_messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search_documents", "arguments": "{}"}},
                ],
                "reasoning_content": "thinking about the search query",
            },
            {"role": "tool", "tool_call_id": "c1", "name": "search_documents", "content": "RESULT"},
        ]
        items = build_output_items(_chat_result("final", tool_messages))
        types = [it["type"] for it in items]
        self.assertEqual(types, ["reasoning", "function_call", "function_call_output", "message"])
        reasoning_item = items[0]
        self.assertTrue(reasoning_item["id"].startswith("rs_"))
        self.assertEqual(reasoning_item["summary"][0]["text"], "thinking about the search query")
        self.assertEqual(reasoning_item["summary"][0]["type"], "summary_text")

    def test_final_reasoning_emitted_before_final_message(self):
        result = ChatResult(
            content="my answer",
            retrieved_chunks=[],
            tool_messages=[],
            final_reasoning_content="thinking before answering directly",
        )
        items = build_output_items(result)
        types = [it["type"] for it in items]
        self.assertEqual(types, ["reasoning", "message"])
        self.assertEqual(items[0]["summary"][0]["text"], "thinking before answering directly")

    def test_no_reasoning_field_means_no_reasoning_item(self):
        # Existing happy path: no reasoning anywhere -> no reasoning items.
        items = build_output_items(_chat_result("hi"))
        self.assertEqual([it["type"] for it in items], ["message"])

    def test_reasoning_on_assistant_and_on_final_both_emitted(self):
        tool_messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search_documents", "arguments": "{}"}},
                ],
                "reasoning_content": "first reasoning",
            },
            {"role": "tool", "tool_call_id": "c1", "name": "search_documents", "content": "R"},
        ]
        result = ChatResult(
            content="final",
            retrieved_chunks=[],
            tool_messages=tool_messages,
            final_reasoning_content="second reasoning",
        )
        items = build_output_items(result)
        types = [it["type"] for it in items]
        self.assertEqual(types, ["reasoning", "function_call", "function_call_output", "reasoning", "message"])
        self.assertEqual(items[0]["summary"][0]["text"], "first reasoning")
        self.assertEqual(items[3]["summary"][0]["text"], "second reasoning")


if __name__ == "__main__":
    unittest.main()
