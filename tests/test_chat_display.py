import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock

from chat import (
    _conversation_log_filename,
    _format_tool_args,
    _format_tool_output,
    _render_meta,
)


class FormatToolArgsTests(unittest.TestCase):
    def test_pretty_prints_json(self):
        self.assertEqual(_format_tool_args('{"query": "index"}'), '{"query": "index"}')

    def test_non_json_passthrough(self):
        self.assertEqual(_format_tool_args("not-json"), "not-json")


class FormatToolOutputTests(unittest.TestCase):
    def test_splits_source_blocks(self):
        output = "[a.md]\nfirst chunk\n\n[b.md]\nsecond chunk"
        lines = _format_tool_output(output, preview_chars=200)
        self.assertEqual(len(lines), 2)
        self.assertIn("[a.md]", lines[0])
        self.assertIn("first chunk", lines[0])
        self.assertIn("[b.md]", lines[1])
        self.assertIn("second chunk", lines[1])

    def test_truncates_long_body(self):
        output = "[doc.md]\n" + ("x" * 500)
        lines = _format_tool_output(output, preview_chars=50)
        self.assertTrue(lines[0].endswith("..."))


class ConversationLogFilenameTests(unittest.TestCase):
    def test_stable_hash(self):
        self.assertEqual(
            _conversation_log_filename("team/session/123"),
            _conversation_log_filename("team/session/123"),
        )
        self.assertNotEqual(
            _conversation_log_filename("a"),
            _conversation_log_filename("b"),
        )


class RenderMetaTests(unittest.TestCase):
    def test_includes_conversation_id(self):
        response = MagicMock()
        response.id = "resp_abc"
        response.usage = MagicMock(input_tokens=10, output_tokens=20, total_tokens=30)
        buf = io.StringIO()
        with redirect_stdout(buf):
            _render_meta(response, "conv_123")
        out = buf.getvalue()
        self.assertIn("response_id=resp_abc", out)
        self.assertIn("conversation_id=conv_123", out)
        self.assertIn("tokens in=10", out)


if __name__ == "__main__":
    unittest.main()
