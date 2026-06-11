# tests/test_config_scenario_mode.py
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simple_chatbot.config import SimpleChatbotConfig


class ScenarioModeConfigTests(unittest.TestCase):
    def test_defaults_to_none(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(docs_dir=Path(tmp) / "docs")
            self.assertIsNone(config.scenario_mode)

    def test_accepts_live(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(
                docs_dir=Path(tmp) / "docs", scenario_mode="live"
            )
            self.assertEqual(config.scenario_mode, "live")
