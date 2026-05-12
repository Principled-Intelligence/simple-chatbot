import unittest
from pathlib import Path

from pydantic import ValidationError

from simple_chatbot.config import SimpleChatbotConfig


class ConfigValidationTests(unittest.TestCase):
    def test_chunk_settings_accept_valid_boundaries(self):
        config = SimpleChatbotConfig(
            docs_dir=Path("docs"),
            chunk_size=1,
            chunk_overlap=0,
        )

        self.assertEqual(config.chunk_size, 1)
        self.assertEqual(config.chunk_overlap, 0)

    def test_chunk_size_must_be_positive(self):
        with self.assertRaisesRegex(ValidationError, "chunk_size must be greater than 0"):
            SimpleChatbotConfig(docs_dir=Path("docs"), chunk_size=0)

    def test_chunk_overlap_must_not_be_negative(self):
        with self.assertRaisesRegex(ValidationError, "chunk_overlap must be greater than or equal to 0"):
            SimpleChatbotConfig(docs_dir=Path("docs"), chunk_size=100, chunk_overlap=-1)

    def test_chunk_overlap_must_be_smaller_than_chunk_size(self):
        with self.assertRaisesRegex(ValidationError, "chunk_overlap must be smaller than chunk_size"):
            SimpleChatbotConfig(docs_dir=Path("docs"), chunk_size=100, chunk_overlap=100)


if __name__ == "__main__":
    unittest.main()
