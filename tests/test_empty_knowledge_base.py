import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from simple_chatbot.agent import EMPTY_KB_RESPONSE, Agent
from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.indexer import Indexer


def _empty_indexer() -> tuple[Indexer, TemporaryDirectory]:
    tmp = TemporaryDirectory()
    config = SimpleChatbotConfig(
        docs_dir=Path(tmp.name) / "docs",
        chroma_persist_dir=Path(tmp.name) / "chroma",
        collection_name="empty_kb_test",
    )
    config.docs_dir.mkdir()
    indexer = Indexer(config)
    return indexer, tmp


class EmptyKnowledgeBaseTests(unittest.TestCase):
    def test_empty_index_search_returns_no_results_without_embedding(self):
        indexer, tmp = _empty_indexer()
        self.addCleanup(tmp.cleanup)

        with patch("simple_chatbot.indexer.litellm.aembedding", new_callable=AsyncMock) as embed:
            results = asyncio.run(indexer.search("anything"))

        self.assertEqual(results, [])
        embed.assert_not_called()

    def test_agent_returns_empty_kb_fallback_without_calling_llm(self):
        indexer, tmp = _empty_indexer()
        self.addCleanup(tmp.cleanup)
        agent = Agent(indexer.config, indexer)

        with patch("simple_chatbot.agent.litellm.acompletion", new_callable=AsyncMock) as completion:
            result = asyncio.run(agent.chat([{"role": "user", "content": "What is in the docs?"}]))

        self.assertEqual(result.content, EMPTY_KB_RESPONSE)
        self.assertEqual(result.retrieved_chunks, [])
        completion.assert_not_called()


if __name__ == "__main__":
    unittest.main()
