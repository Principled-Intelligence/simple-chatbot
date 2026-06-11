# tests/test_tool_catalog_cli.py
"""CLI `tools` command: print an agent's OpenAI Responses-API tool catalog."""

import json
import unittest

from typer.testing import CliRunner

from simple_chatbot.cli import cli

runner = CliRunner()


def _run(args, env=None):
    result = runner.invoke(cli, args, env=env)
    return result


class ToolCatalogCliTests(unittest.TestCase):
    def test_fixture_catalog_is_flat_openai_shape(self):
        result = _run(["tools", "cs-routing"])
        self.assertEqual(result.exit_code, 0, result.output)
        tools = json.loads(result.stdout)
        names = {t["name"] for t in tools}
        self.assertIn("lookup_invoice", names)
        self.assertIn("issue_refund", names)
        self.assertIn("route", names)  # generated handoff tool
        for t in tools:
            self.assertEqual(t["type"], "function")
            self.assertNotIn("function", t)  # flat, not nested chat-completions

    def test_live_rag_default_catalog(self):
        result = _run(["tools", "live-rag"])
        self.assertEqual(result.exit_code, 0, result.output)
        tools = json.loads(result.stdout)
        names = {t["name"] for t in tools}
        self.assertEqual(names, {"search_documents"})
        self.assertNotIn("function", tools[0])

    def test_no_arg_defaults_to_live_rag(self):
        result = _run(["tools"], env={"SIMPLE_CHATBOT_DEFAULT_FIXTURE": ""})
        self.assertEqual(result.exit_code, 0, result.output)
        tools = json.loads(result.stdout)
        self.assertEqual({t["name"] for t in tools}, {"search_documents"})

    def test_default_fixture_env_resolves_active_agent(self):
        result = _run(["tools"], env={"SIMPLE_CHATBOT_DEFAULT_FIXTURE": "cs-routing"})
        self.assertEqual(result.exit_code, 0, result.output)
        tools = json.loads(result.stdout)
        self.assertIn("lookup_invoice", {t["name"] for t in tools})

    def test_default_fixture_option_overrides_env(self):
        result = _run(
            ["tools", "--default-fixture", "cs-routing"],
            env={"SIMPLE_CHATBOT_DEFAULT_FIXTURE": ""},
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("lookup_invoice", {t["name"] for t in json.loads(result.stdout)})

    def test_explicit_agent_beats_default_fixture_env(self):
        result = _run(
            ["tools", "live-rag"], env={"SIMPLE_CHATBOT_DEFAULT_FIXTURE": "cs-routing"}
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            {t["name"] for t in json.loads(result.stdout)}, {"search_documents"}
        )

    def test_scripted_env_surfaces_full_toolset(self):
        result = _run(["tools", "live-rag"], env={"SIMPLE_CHATBOT_SCRIPTED_LLM": "1"})
        self.assertEqual(result.exit_code, 0, result.output)
        names = {t["name"] for t in json.loads(result.stdout)}
        self.assertEqual(
            names,
            {"search_documents", "calculate", "get_current_time", "lookup_user"},
        )

    def test_unknown_agent_errors_and_lists_available(self):
        result = _run(["tools", "does-not-exist"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("does-not-exist", result.output)
        self.assertIn("live-rag", result.output)


if __name__ == "__main__":
    unittest.main()
