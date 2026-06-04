import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import simple_chatbot.service_description as sd
from simple_chatbot.config import SimpleChatbotConfig


def _config(tmp, **kw):
    return SimpleChatbotConfig(docs_dir=Path(tmp) / "docs", **kw)


class ResolverUnitTests(unittest.TestCase):
    def setUp(self):
        self._orig_dir = sd._AISD_DIR
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        sd._AISD_DIR = Path(self._tmp.name)
        self.addCleanup(lambda: setattr(sd, "_AISD_DIR", self._orig_dir))

    def _write(self, key, text):
        (Path(self._tmp.name) / f"{key}.md").write_text(text, encoding="utf-8")

    def test_load_returns_text_for_existing_key(self):
        self._write("cs-routing", "# hello")
        self.assertEqual(sd.load_service_description("cs-routing"), "# hello")

    def test_load_returns_none_for_missing_key(self):
        self.assertIsNone(sd.load_service_description("nope"))

    def test_resolve_active_uses_default_fixture(self):
        self._write("cs-routing", "CS AISD body")
        with TemporaryDirectory() as tmp:
            label, text = sd.resolve_active(_config(tmp, default_fixture="cs-routing"))
        self.assertEqual(label, "cs-routing")
        self.assertEqual(text, "CS AISD body")

    def test_resolve_active_defaults_to_live_rag(self):
        self._write("live-rag", "RAG AISD body")
        with TemporaryDirectory() as tmp:
            label, text = sd.resolve_active(_config(tmp))
        self.assertEqual(label, "live-rag")
        self.assertEqual(text, "RAG AISD body")

    def test_resolve_active_fixture_fallback_to_scenario_description(self):
        # cs-routing.md absent from the temp dir → fall back to the real
        # scenario's one-line description, without raising.
        with TemporaryDirectory() as tmp:
            label, text = sd.resolve_active(_config(tmp, default_fixture="cs-routing"))
        self.assertEqual(label, "cs-routing")
        self.assertEqual(text, "CS dispatcher routing to billing / human escalation.")

    def test_resolve_active_live_rag_fallback_to_system_prompt(self):
        with TemporaryDirectory() as tmp:
            label, text = sd.resolve_active(_config(tmp, system_prompt="Custom scope."))
        self.assertEqual(label, "live-rag")
        self.assertEqual(text, "Custom scope.")

    def test_print_emits_delimited_block(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            sd.print_service_description("cs-routing", "BODY TEXT")
        out = buf.getvalue()
        self.assertIn("AI SERVICE DESCRIPTION — cs-routing", out)
        self.assertIn("BODY TEXT", out)
        self.assertIn("=" * 72, out)


from simple_chatbot.scenario_registry import load_fixtures
from simple_chatbot.service_description import LIVE_RAG_KEY


class ContentCoverageTests(unittest.TestCase):
    _REQUIRED_HEADERS = (
        "## Service overview",
        "## Surface & capabilities",
        "## Out of scope",
    )

    def _aisd_text(self, key):
        path = sd._AISD_DIR / f"{key}.md"
        self.assertTrue(path.is_file(), f"missing AISD file: {path}")
        return path.read_text(encoding="utf-8")

    def test_every_fixture_has_an_aisd(self):
        for fixture_id in load_fixtures():
            text = self._aisd_text(fixture_id)
            self.assertTrue(text.strip(), f"empty AISD for {fixture_id}")
            for header in self._REQUIRED_HEADERS:
                self.assertIn(header, text, f"{fixture_id} missing {header!r}")

    def test_live_rag_has_an_aisd(self):
        text = self._aisd_text(LIVE_RAG_KEY)
        self.assertTrue(text.strip())
        for header in self._REQUIRED_HEADERS:
            self.assertIn(header, text)


class RealFileResolutionTests(unittest.TestCase):
    # Uses the REAL sd._AISD_DIR (the authored files), not a monkeypatched dir.

    def test_resolve_active_returns_real_fixture_file(self):
        with TemporaryDirectory() as tmp:
            label, text = sd.resolve_active(_config(tmp, default_fixture="cs-routing"))
        self.assertEqual(label, "cs-routing")
        self.assertIn("## Surface & capabilities", text)
        self.assertIn("issue_refund", text)

    def test_resolve_active_returns_real_live_rag_file(self):
        with TemporaryDirectory() as tmp:
            label, text = sd.resolve_active(_config(tmp))
        self.assertEqual(label, "live-rag")
        self.assertIn("search_documents", text)

    def test_good_and_evil_resolve_to_identical_live_rag_aisd(self):
        with TemporaryDirectory() as tmp:
            good_label, good_text = sd.resolve_active(_config(tmp))
        with TemporaryDirectory() as tmp:
            evil_label, evil_text = sd.resolve_active(
                _config(tmp, misbehavior_rate=1.0, misbehavior_modes=["drop_retrieval"], misbehavior_seed=3)
            )
        self.assertEqual(good_label, evil_label)
        self.assertEqual(good_text, evil_text)


import os
from unittest.mock import patch

from typer.testing import CliRunner

from simple_chatbot.cli import cli


class CliStartupPrintTests(unittest.TestCase):
    def test_serve_prints_live_rag_aisd_at_startup(self):
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            env = {**os.environ, "SIMPLE_CHATBOT_SCRIPTED_LLM": "1"}
            # Patch uvicorn.run so serve returns instead of blocking on the server.
            with patch("simple_chatbot.cli.uvicorn.run", return_value=None):
                result = runner.invoke(
                    cli,
                    ["serve", "--docs-dir", str(Path(tmp) / "docs")],
                    env=env,
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("AI SERVICE DESCRIPTION — live-rag", result.output)
        self.assertIn("search_documents", result.output)

    def test_serve_prints_fixture_aisd_when_default_fixture_set(self):
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            env = {**os.environ, "SIMPLE_CHATBOT_SCRIPTED_LLM": "1"}
            with patch("simple_chatbot.cli.uvicorn.run", return_value=None):
                result = runner.invoke(
                    cli,
                    ["serve", "--docs-dir", str(Path(tmp) / "docs"), "--default-fixture", "cs-routing"],
                    env=env,
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("AI SERVICE DESCRIPTION — cs-routing", result.output)
        self.assertIn("issue_refund", result.output)


if __name__ == "__main__":
    unittest.main()
