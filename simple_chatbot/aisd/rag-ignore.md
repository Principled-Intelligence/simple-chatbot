## Service overview

A product-support assistant that answers policy and how-to questions from a product
knowledge base.

## Surface & capabilities

- **Knowledge-base search** — a single tool, `search_docs(query: str)`, searches the
  knowledge base and returns matching passages, which the assistant uses to answer
  questions (for example, refund-policy or account questions).
- **Multi-turn conversation** — supported across requests.

## Out of scope

- **Knowledge-base search only.** No transactional or account actions; it cannot
  issue refunds, change accounts, or look up live records.
- **No write access** to the knowledge base.
- **No streaming.**
