import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simple_chatbot.conversation_logger import ConversationLogger


class ConversationLoggerTests(unittest.TestCase):
    def test_client_conversation_id_cannot_escape_log_directory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "logs"
            outside_path = root / "escaped.jsonl"
            logger = ConversationLogger(log_dir)

            asyncio.run(
                logger.log(
                    conversation_id="../escaped",
                    messages=[{"role": "user", "content": "hello"}],
                    response="hi",
                    chunks=[],
                )
            )

            self.assertFalse(outside_path.exists())
            log_files = list(log_dir.glob("*.jsonl"))
            self.assertEqual(len(log_files), 1)
            self.assertEqual(log_files[0].parent, log_dir)

            record = json.loads(log_files[0].read_text(encoding="utf-8"))
            self.assertEqual(record["conversation_id"], "../escaped")

    def test_same_conversation_id_uses_same_hashed_log_file(self):
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            logger = ConversationLogger(log_dir)

            for content in ("first", "second"):
                asyncio.run(
                    logger.log(
                        conversation_id="team/session/123",
                        messages=[{"role": "user", "content": content}],
                        response="ok",
                        chunks=[],
                    )
                )

            log_files = list(log_dir.glob("*.jsonl"))
            self.assertEqual(len(log_files), 1)
            lines = log_files[0].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
