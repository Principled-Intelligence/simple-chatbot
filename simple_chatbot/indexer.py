import asyncio
import functools
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chromadb
import litellm
from loguru import logger

from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.loader import Document

_DEFAULT_EMBED_BATCH_SIZE = 500
# Vertex text-embeddings accept at most 250 instances per request.
_VERTEX_EMBED_BATCH_SIZE = 250
_MAX_CONCURRENT_EMBED_BATCHES = 10
_META_FILE = "simple_chatbot_index_meta.json"


def _embed_batch_size(config: SimpleChatbotConfig) -> int:
    if config.embedding_model.startswith("vertex_ai/"):
        return _VERTEX_EMBED_BATCH_SIZE
    return _DEFAULT_EMBED_BATCH_SIZE


def _doc_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def _embed(texts: list[str], config: SimpleChatbotConfig) -> list[list[float]]:
    logger.bind(
        text_count=len(texts),
        model=config.embedding_model,
    ).debug("Embedding texts")
    kwargs: dict = {"model": config.embedding_model, "input": texts}
    if config.embedding_api_base:
        kwargs["api_base"] = config.embedding_api_base
        logger.bind(api_base=config.embedding_api_base).debug(
            "Using custom embedding API base"
        )
    response = await litellm.aembedding(**kwargs)
    embeddings = [item["embedding"] for item in response.data]
    logger.bind(
        embedding_count=len(embeddings),
        dim=len(embeddings[0]) if embeddings else 0,
    ).debug("Received embeddings")
    return embeddings


async def _embed_all_batches(
    batches: list[list[Document]],
    config: SimpleChatbotConfig,
) -> list[list[list[float]]]:
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_EMBED_BATCHES)

    async def _bounded(batch: list[Document]) -> list[list[float]]:
        async with semaphore:
            return await _embed([d.text for d in batch], config)

    return await asyncio.gather(*(_bounded(b) for b in batches))


def _similarity_from_distance(distance: float | None, metric: str) -> float | None:
    """Convert a Chroma distance to a 0-1 similarity, or None when the metric has
    no bounded similarity.

    Chroma returns ``1 - cosine_similarity`` for "cosine" and ``1 - inner_product``
    for "ip", so both invert the same way (and coincide for L2-normalised vectors,
    which OpenAI embeddings are). "l2" is a squared euclidean distance, unbounded
    above, so no 0-1 similarity exists — config rejects pairing it with a
    threshold, and this returns None so a stray caller filters nothing.
    """
    if distance is None or metric == "l2":
        return None
    return 1.0 - distance


def _config_fingerprint(config: SimpleChatbotConfig) -> dict:
    return {
        "embedding_model": config.embedding_model,
        "embedding_api_base": config.embedding_api_base,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "collection_name": config.collection_name,
        # The metric is fixed when the collection is created ("hnsw:space"), and
        # get_or_create_collection silently ignores new metadata for a collection
        # that already exists. Without it in the fingerprint, changing the metric
        # would appear to work while queries kept using the old space.
        "distance_metric": config.distance_metric,
    }


class Indexer:
    def __init__(self, config: SimpleChatbotConfig) -> None:
        self.config = config
        persist_dir = Path(config.chroma_persist_dir)
        self._meta_path = persist_dir / _META_FILE

        logger.bind(
            persist_dir=str(persist_dir.resolve()),
            collection=config.collection_name,
        ).info("Initialising ChromaDB")
        self._client = chromadb.PersistentClient(path=str(persist_dir))

        # If embedding config changed since last run, the stored vectors are
        # stale — drop the collection so everything is re-embedded cleanly.
        if self._meta_path.exists():
            stored = json.loads(self._meta_path.read_text())
            current = _config_fingerprint(config)
            if stored != current:
                logger.bind(
                    old=stored,
                    new=current,
                ).warning("Embedding config changed since last run")
                try:
                    self._client.delete_collection(config.collection_name)
                    logger.bind(collection=config.collection_name).info(
                        "Deleted stale collection"
                    )
                except Exception as exc:
                    logger.bind(error=str(exc)).warning("Could not delete collection")
            else:
                logger.info("Embedding config unchanged — reusing existing index")
        else:
            logger.info("No index metadata found — fresh index will be created")

        self.collection = self._client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": config.distance_metric},
        )
        # Confine ALL Chroma access to ONE dedicated thread. The chromadb_rust_bindings
        # core (SQLite-backed) is thread-affine and SIGSEGVs when touched from multiple
        # threads at once — e.g. a search() running on an asyncio.to_thread worker while
        # another turn's document_count() runs collection.count() on the event loop.
        # A concurrency limit isn't enough (to_thread picks arbitrary threads); every
        # collection call must run on this single thread. Use _chroma_call/_achroma_call.
        self._chroma_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chroma")

        logger.bind(
            collection=config.collection_name,
            count=self.collection.count(),
        ).info("ChromaDB collection ready")

    def _chroma_call(self, fn, *args, **kwargs):
        """Run a Chroma collection op on the dedicated Chroma thread (sync caller)."""
        return self._chroma_pool.submit(functools.partial(fn, *args, **kwargs)).result()

    async def _achroma_call(self, fn, *args, **kwargs):
        """Run a Chroma collection op on the dedicated Chroma thread (async caller)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._chroma_pool, functools.partial(fn, *args, **kwargs)
        )

    def _save_meta(self) -> None:
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(
            json.dumps(_config_fingerprint(self.config), indent=2)
        )
        logger.bind(path=str(self._meta_path)).debug("Index metadata saved")

    def _reset_collection(self) -> None:
        self._client.delete_collection(self.config.collection_name)
        logger.bind(collection=self.config.collection_name).info(
            "Deleted collection for force re-index"
        )
        self.collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric},
        )

    def is_populated(self) -> bool:
        count = self._chroma_call(self.collection.count)
        logger.bind(count=count).debug("Collection count check")
        return count > 0

    def document_count(self) -> int:
        count = self._chroma_call(self.collection.count)
        logger.bind(count=count).debug("Collection count check")
        return count

    def index(self, docs: list[Document], force: bool = False) -> None:
        logger.bind(chunk_count=len(docs), force=force).info("Starting indexing")
        batch_size = _embed_batch_size(self.config)

        if force:
            self._reset_collection()
            if not docs:
                logger.warning("Force re-index requested with no documents")
                self._save_meta()
                return
            seen: dict[str, Document] = {}
            for d in docs:
                doc_id = _doc_id(d.text)
                if doc_id not in seen:
                    seen[doc_id] = d
            new_docs = list(seen.values())
            logger.bind(
                unique=len(new_docs),
                total=len(docs),
            ).info("Force re-index prepared")
        else:
            if not docs:
                logger.warning("No documents provided to index — nothing to do")
                return

            # Check which chunks are already stored so we can skip their embeddings.
            # Deduplicate by ID to avoid ChromaDB rejecting duplicate IDs in get/upsert.
            seen: dict[str, Document] = {}
            for d in docs:
                doc_id = _doc_id(d.text)
                if doc_id not in seen:
                    seen[doc_id] = d
            unique_ids = list(seen.keys())
            logger.bind(
                total=len(docs),
                unique=len(unique_ids),
            ).debug("Deduplicated document chunks")

            existing_ids: set[str] = set()
            for i in range(0, len(unique_ids), batch_size):
                batch_ids = unique_ids[i : i + batch_size]
                result = self.collection.get(ids=batch_ids, include=[])
                existing_ids.update(result["ids"])
            logger.bind(existing=len(existing_ids), unique=len(unique_ids)).debug(
                "Checked cached chunks"
            )

            new_docs = [d for doc_id, d in seen.items() if doc_id not in existing_ids]

            if not new_docs:
                logger.bind(chunk_count=len(docs)).info("All chunks already cached")
                self._save_meta()
                return

            cached = len(existing_ids)
            if cached:
                logger.bind(
                    cached=cached,
                    new=len(new_docs),
                ).info("Some chunks already cached")
            else:
                logger.bind(chunk_count=len(new_docs)).info("No cached chunks found")

        batches = [
            new_docs[start : start + batch_size]
            for start in range(0, len(new_docs), batch_size)
        ]
        total_batches = len(batches)
        logger.bind(
            chunk_count=len(new_docs),
            batch_count=total_batches,
            size=batch_size,
            concurrency=_MAX_CONCURRENT_EMBED_BATCHES,
        ).info("Embedding chunk batches")

        all_embeddings = asyncio.run(_embed_all_batches(batches, self.config))

        for batch_idx, (batch, embeddings) in enumerate(
            zip(batches, all_embeddings), start=1
        ):
            texts = [d.text for d in batch]
            ids = [_doc_id(t) for t in texts]
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=[d.metadata for d in batch],
            )
            indexed_so_far = min(batch_idx * batch_size, len(new_docs))
            logger.bind(
                done=indexed_so_far,
                total=len(new_docs),
                batch_index=batch_idx,
                batch_count=total_batches,
            ).info("Indexed chunk batch")

        self._save_meta()
        logger.bind(
            count=self.collection.count(),
        ).info("Indexing complete")

    async def search(self, query: str) -> list[Document]:
        count = await self._achroma_call(self.collection.count)
        n_results = min(self.config.top_k, count)
        logger.bind(
            query=query,
            top_k=self.config.top_k,
            count=count,
            n_results=n_results,
        ).debug("Search requested")

        if n_results <= 0:
            logger.warning(
                "Search requested but document index is empty or top_k is zero"
            )
            return []

        query_embedding = (await _embed([query], self.config))[0]
        results = await self._achroma_call(
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        distances = (results.get("distances") or [None])[0]
        threshold = self.config.similarity_threshold

        docs: list[Document] = []
        dropped = 0
        for i, (text, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            if threshold is not None and distances is not None:
                similarity = _similarity_from_distance(
                    distances[i], self.config.distance_metric
                )
                if similarity is not None and similarity < threshold:
                    dropped += 1
                    continue
            docs.append(Document(text=text, metadata=metadata))

        if threshold is not None:
            if distances is None:
                # Never silently drop everything because the backend withheld
                # distances — fall through unfiltered and say so.
                logger.bind(threshold=threshold).warning(
                    "similarity_threshold set but Chroma returned no distances — "
                    "returning unfiltered results"
                )
            else:
                logger.bind(
                    threshold=threshold,
                    metric=self.config.distance_metric,
                    kept=len(docs),
                    dropped=dropped,
                ).debug("Similarity threshold applied")
                if not docs:
                    logger.bind(
                        threshold=threshold, dropped=dropped, query=query
                    ).warning(
                        "similarity_threshold filtered out every retrieved chunk — "
                        "the model will answer with no retrieved context"
                    )

        logger.bind(
            result_count=len(docs),
            query=query,
        ).debug("Search completed")
        for i, doc in enumerate(docs):
            logger.bind(
                i=i,
                source=doc.metadata.get("source", "unknown"),
                chunk=doc.metadata.get("chunk"),
                preview=doc.text[:80],
            ).trace("Search result preview")

        return docs
