import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import typer
from pydantic import ValidationError

from simple_chatbot.agent import Agent
from simple_chatbot.cli import _parse_provider_routing
from simple_chatbot.config import SimpleChatbotConfig


class _FakeIndexer:
    def document_count(self) -> int:
        return 1

    async def search(self, query: str) -> list:
        return []


class _Message:
    content = "ok"
    tool_calls: list = []

    def model_dump(self) -> dict:
        return {"role": "assistant", "content": self.content}


class _Choice:
    message = _Message()
    finish_reason = "stop"


class _Response:
    choices = [_Choice()]
    usage = None


def _run(config: SimpleChatbotConfig, indexer: _FakeIndexer) -> dict:
    """Drive one agent turn and return the kwargs sent downstream."""
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _Response()

    agent = Agent(config, indexer, acompletion=fake_acompletion)
    asyncio.run(agent.chat([{"role": "user", "content": "ciao"}]))
    return captured


class ProviderRoutingConfigTests(unittest.TestCase):
    def test_off_by_default(self):
        config = SimpleChatbotConfig(docs_dir=Path("docs"))

        self.assertIsNone(config.provider_routing)

    def test_accepts_routing_object(self):
        config = SimpleChatbotConfig(
            docs_dir=Path("docs"), provider_routing={"quantizations": ["bf16"]}
        )

        self.assertEqual(config.provider_routing, {"quantizations": ["bf16"]})

    def test_rejects_empty_object(self):
        with self.assertRaisesRegex(ValidationError, "provider_routing must be a non-empty object"):
            SimpleChatbotConfig(docs_dir=Path("docs"), provider_routing={})


class ProviderRoutingParseTests(unittest.TestCase):
    def test_none_when_unset(self):
        self.assertIsNone(_parse_provider_routing(None))
        self.assertIsNone(_parse_provider_routing(""))

    def test_parses_json_object(self):
        self.assertEqual(
            _parse_provider_routing('{"only": ["akashml"]}'), {"only": ["akashml"]}
        )

    def test_rejects_malformed_json(self):
        # A silently-dropped pin would produce a plausible-looking run served by an
        # arbitrary backend, so bad input must fail loudly instead.
        with self.assertRaises(typer.BadParameter):
            _parse_provider_routing("not-json")

    def test_rejects_non_object_json(self):
        for raw in ("[1, 2]", '"akashml"', "{}"):
            with self.subTest(raw=raw), self.assertRaises(typer.BadParameter):
                _parse_provider_routing(raw)


class ProviderRoutingAgentTests(unittest.TestCase):
    def test_forwarded_as_extra_body_provider(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(
                docs_dir=Path(tmp) / "docs",
                provider_routing={"quantizations": ["bf16"]},
            )

            kwargs = _run(config, _FakeIndexer())

            self.assertEqual(
                kwargs.get("extra_body"), {"provider": {"quantizations": ["bf16"]}}
            )

    def test_absent_when_unset(self):
        with TemporaryDirectory() as tmp:
            config = SimpleChatbotConfig(docs_dir=Path(tmp) / "docs")

            kwargs = _run(config, _FakeIndexer())

            self.assertNotIn("extra_body", kwargs)


if __name__ == "__main__":
    unittest.main()
