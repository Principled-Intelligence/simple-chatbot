import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import pydantic

from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.indexer import Indexer, _config_fingerprint, _similarity_from_distance
from simple_chatbot.loader import Document


def _config(tmp: str, **overrides) -> SimpleChatbotConfig:
    return SimpleChatbotConfig(
        docs_dir=Path(tmp) / "docs",
        chroma_persist_dir=Path(tmp) / "chroma",
        collection_name="threshold_test",
        **overrides,
    )


# Two orthogonal unit vectors: under cosine, a [1,0] query scores similarity 1.0
# against "near" (distance 0.0) and 0.0 against "far" (distance 1.0).
_NEAR = Document(text="near chunk", metadata={"source": "near.txt"})
_FAR = Document(text="far chunk", metadata={"source": "far.txt"})


def _index_two(indexer: Indexer) -> None:
    with patch("simple_chatbot.indexer._embed_all_batches", new_callable=AsyncMock) as embed:
        embed.return_value = [[[1.0, 0.0], [0.0, 1.0]]]
        indexer.index([_NEAR, _FAR])


def _search(indexer: Indexer, vector=(1.0, 0.0)) -> list[Document]:
    with patch("simple_chatbot.indexer._embed", new_callable=AsyncMock) as embed:
        embed.return_value = [list(vector)]
        return asyncio.run(indexer.search("anything"))


class SimilarityConversionTests(unittest.TestCase):
    def test_cosine_and_ip_invert_the_distance(self):
        self.assertAlmostEqual(_similarity_from_distance(0.25, "cosine"), 0.75)
        self.assertAlmostEqual(_similarity_from_distance(0.10, "ip"), 0.90)

    def test_l2_and_missing_distance_have_no_similarity(self):
        # Unbounded metric -> None, so a stray caller filters nothing rather than
        # comparing against a meaningless scale.
        self.assertIsNone(_similarity_from_distance(9.0, "l2"))
        self.assertIsNone(_similarity_from_distance(None, "cosine"))


class ConfigValidationTests(unittest.TestCase):
    def test_defaults_preserve_historical_behaviour(self):
        with TemporaryDirectory() as tmp:
            config = _config(tmp)
            self.assertEqual(config.distance_metric, "cosine")
            self.assertIsNone(config.similarity_threshold)

    def test_threshold_with_l2_is_rejected(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(pydantic.ValidationError) as ctx:
                _config(tmp, distance_metric="l2", similarity_threshold=0.3)
            self.assertIn("not supported with distance_metric='l2'", str(ctx.exception))

    def test_l2_without_a_threshold_is_allowed(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(_config(tmp, distance_metric="l2").distance_metric, "l2")

    def test_threshold_outside_unit_range_is_rejected(self):
        with TemporaryDirectory() as tmp:
            for bad in (1.5, -0.1):
                with self.assertRaises(pydantic.ValidationError):
                    _config(tmp, similarity_threshold=bad)


class ThresholdFilteringTests(unittest.TestCase):
    def test_no_threshold_returns_every_hit(self):
        with TemporaryDirectory() as tmp:
            indexer = Indexer(_config(tmp))
            _index_two(indexer)
            self.assertEqual(
                sorted(d.text for d in _search(indexer)), ["far chunk", "near chunk"]
            )

    def test_threshold_drops_dissimilar_chunks(self):
        with TemporaryDirectory() as tmp:
            indexer = Indexer(_config(tmp, similarity_threshold=0.5))
            _index_two(indexer)
            docs = _search(indexer)
            self.assertEqual([d.text for d in docs], ["near chunk"])

    def test_threshold_may_filter_everything(self):
        # A too-aggressive threshold must return an empty list rather than raising:
        # the agent then answers with no retrieved context.
        with TemporaryDirectory() as tmp:
            indexer = Indexer(_config(tmp, similarity_threshold=0.99))
            _index_two(indexer)
            self.assertEqual(_search(indexer, vector=(0.6, 0.8)), [])

    def test_threshold_at_the_boundary_keeps_the_chunk(self):
        # similarity == threshold is kept (only strictly-below is dropped).
        with TemporaryDirectory() as tmp:
            indexer = Indexer(_config(tmp, similarity_threshold=1.0))
            _index_two(indexer)
            self.assertEqual([d.text for d in _search(indexer)], ["near chunk"])


class DistanceMetricFingerprintTests(unittest.TestCase):
    def test_metric_is_part_of_the_fingerprint(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(_config_fingerprint(_config(tmp))["distance_metric"], "cosine")
            self.assertEqual(
                _config_fingerprint(_config(tmp, distance_metric="ip"))["distance_metric"], "ip"
            )

    def test_changing_the_metric_rebuilds_the_collection(self):
        # hnsw:space is fixed when the collection is created, so a metric change
        # MUST invalidate the index — otherwise queries silently keep using the
        # old space while the flag claims otherwise.
        with TemporaryDirectory() as tmp:
            first = Indexer(_config(tmp))
            _index_two(first)
            self.assertEqual(first.document_count(), 2)

            second = Indexer(_config(tmp, distance_metric="ip"))
            self.assertEqual(second.document_count(), 0)
            self.assertEqual(
                second.collection.metadata.get("hnsw:space"), "ip"
            )

    def test_unchanged_metric_reuses_the_collection(self):
        with TemporaryDirectory() as tmp:
            first = Indexer(_config(tmp))
            _index_two(first)
            self.assertEqual(Indexer(_config(tmp)).document_count(), 2)


if __name__ == "__main__":
    unittest.main()
