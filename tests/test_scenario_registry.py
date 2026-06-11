# tests/test_scenario_registry.py
import unittest

from simple_chatbot.scenario import Scenario
from simple_chatbot.scenario_registry import load_fixtures


class RegistryTests(unittest.TestCase):
    def test_loads_known_fixtures_keyed_by_id(self):
        registry = load_fixtures()
        self.assertIn("cs-routing", registry)
        self.assertIn("validity-probe", registry)
        self.assertIsInstance(registry["cs-routing"], Scenario)

    def test_returns_empty_for_missing_package(self):
        self.assertEqual(load_fixtures("simple_chatbot.does_not_exist"), {})
