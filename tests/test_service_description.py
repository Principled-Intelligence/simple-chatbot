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


if __name__ == "__main__":
    unittest.main()
