import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.indexer import Indexer
from simple_chatbot.loader import Document


def _config(tmp: str) -> SimpleChatbotConfig:
    return SimpleChatbotConfig(
        docs_dir=Path(tmp) / "docs",
        chroma_persist_dir=Path(tmp) / "chroma",
        collection_name="reindex_test",
    )


class ReindexTests(unittest.TestCase):
    def test_force_reindex_removes_stale_chunks(self):
        with TemporaryDirectory() as tmp:
            config = _config(tmp)
            indexer = Indexer(config)

            with patch("simple_chatbot.indexer._embed_all_batches", new_callable=AsyncMock) as embed:
                embed.side_effect = [
                    [[[1.0, 0.0]]],
                    [[[0.0, 1.0]]],
                ]

                indexer.index([Document(text="old chunk", metadata={"source": "old.txt"})])
                indexer.index(
                    [Document(text="new chunk", metadata={"source": "new.txt"})],
                    force=True,
                )

            stored = indexer.collection.get(include=["documents", "metadatas"])
            self.assertEqual(stored["documents"], ["new chunk"])
            self.assertEqual(stored["metadatas"], [{"source": "new.txt"}])

    def test_force_reindex_with_no_documents_clears_collection(self):
        with TemporaryDirectory() as tmp:
            config = _config(tmp)
            indexer = Indexer(config)

            with patch("simple_chatbot.indexer._embed_all_batches", new_callable=AsyncMock) as embed:
                embed.return_value = [[[1.0, 0.0]]]

                indexer.index([Document(text="old chunk", metadata={"source": "old.txt"})])
                indexer.index([], force=True)

            self.assertEqual(indexer.document_count(), 0)
            self.assertEqual(embed.call_count, 1)

    def test_force_reindex_deduplicates_chunk_ids(self):
        with TemporaryDirectory() as tmp:
            config = _config(tmp)
            indexer = Indexer(config)

            with patch("simple_chatbot.indexer._embed_all_batches", new_callable=AsyncMock) as embed:
                embed.return_value = [[[1.0, 0.0]]]

                indexer.index(
                    [
                        Document(text="same chunk", metadata={"source": "a.txt"}),
                        Document(text="same chunk", metadata={"source": "b.txt"}),
                    ],
                    force=True,
                )

            self.assertEqual(indexer.document_count(), 1)

    def test_embedding_api_base_change_invalidates_existing_index(self):
        with TemporaryDirectory() as tmp:
            config = _config(tmp)
            first = Indexer(config)

            with patch("simple_chatbot.indexer._embed_all_batches", new_callable=AsyncMock) as embed:
                embed.return_value = [[[1.0, 0.0]]]
                first.index([Document(text="old chunk", metadata={"source": "old.txt"})])

            changed_config = _config(tmp)
            changed_config.embedding_api_base = "http://new-embedding-backend/v1"
            second = Indexer(changed_config)

            self.assertEqual(second.document_count(), 0)


if __name__ == "__main__":
    unittest.main()
