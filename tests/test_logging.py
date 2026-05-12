import contextlib
import io
import json
import unittest

from loguru import logger

from simple_chatbot._logging import setup_logging


class LoggingSetupTests(unittest.TestCase):
    def tearDown(self) -> None:
        logger.remove()

    def test_pretty_format_includes_bound_context(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            setup_logging(level="INFO", log_format="pretty")
            logger.bind(request_id="req123", conversation_id="conv456").info("hello")
            logger.complete()

        output = stderr.getvalue()
        self.assertIn("request_id=req123", output)
        self.assertIn("conversation_id=conv456", output)
        self.assertIn("hello", output)

    def test_json_format_serializes_bound_context(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            setup_logging(level="INFO", log_format="json")
            logger.bind(request_id="req123", conversation_id="conv456").info("hello")
            logger.complete()

        record = json.loads(stderr.getvalue())
        self.assertEqual(record["record"]["message"], "hello")
        self.assertEqual(record["record"]["extra"]["request_id"], "req123")
        self.assertEqual(record["record"]["extra"]["conversation_id"], "conv456")


if __name__ == "__main__":
    unittest.main()
