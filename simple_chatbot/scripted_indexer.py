from simple_chatbot.loader import Document


class ScriptedIndexer:
    """In-process indexer stub for offline testing.

    Returns a single canned Document on every search whose text echoes the
    query, so the round-trip is visible in the trace. Does not touch ChromaDB
    or any embedding API.
    """

    def document_count(self) -> int:
        return 1

    async def search(self, query: str) -> list[Document]:
        return [
            Document(
                text=f"Scripted search result for query: {query!r}",
                metadata={"source": "scripted", "chunk": 0},
            )
        ]
