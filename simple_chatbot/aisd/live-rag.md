## Service overview

A retrieval-augmented (RAG) assistant that answers questions from a curated,
indexed document collection (the server's knowledge base). It is backed by a real
language model and a real vector search over the loaded corpus. This is the default
chatbot a client reaches when it does not select a specific scenario.

## Surface & capabilities

- **Document search** — a single tool, `search_documents(query: str)`, runs a
  semantic search over the indexed corpus and returns the most relevant passages
  with their source labels.
- **Agentic tool loop** — the assistant may issue one or more searches across
  several rounds before answering, refining its query based on intermediate
  results, then writes a final natural-language answer grounded in the retrieved
  passages.
- **Multi-turn conversation** — prior turns are carried into later requests, so
  follow-up questions can build on earlier context.
- **Scope** — its subject matter is whatever the loaded documents cover; an
  operator-supplied system prompt may further constrain its role.

## Out of scope

- **No tools other than document search.** It cannot perform transactions,
  calculations, bookings, account changes, or any external action.
- **No live/web access.** It only knows what is in the indexed corpus; it cannot
  browse the internet or read sources that were not indexed.
- **No write access.** It cannot modify, add to, or delete from the knowledge base.
- **No streaming.** Responses are returned as a single completed message.
