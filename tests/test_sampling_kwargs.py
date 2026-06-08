# tests/test_sampling_kwargs.py
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simple_chatbot.agent import sampling_kwargs
from simple_chatbot.config import SimpleChatbotConfig


class SamplingKwargsTests(unittest.TestCase):
    def test_only_set_params_are_included(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(
                docs_dir=Path(tmp) / "docs", temperature=0.5, gen_top_k=40
            )
            kwargs = sampling_kwargs(config)
        # explicitly set params present under their wire names
        self.assertEqual(kwargs["temperature"], 0.5)
        self.assertEqual(kwargs["top_k"], 40)  # gen_top_k -> top_k
        # unset params absent (not None-valued)
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("presence_penalty", kwargs)

    def test_empty_when_nothing_set(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(docs_dir=Path(tmp) / "docs")
            self.assertEqual(sampling_kwargs(config), {})

    def test_reasoning_effort_forwarded_when_set(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(
                docs_dir=Path(tmp) / "docs", reasoning_effort="medium"
            )
            self.assertEqual(sampling_kwargs(config), {"reasoning_effort": "medium"})

    def test_reasoning_effort_normalized_to_lowercase(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(
                docs_dir=Path(tmp) / "docs", reasoning_effort="high"
            )
            self.assertEqual(config.reasoning_effort, "high")
