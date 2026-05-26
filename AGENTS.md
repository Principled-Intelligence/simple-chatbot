# AGENTS.md

## Project Overview

`simple-chatbot` is a small Python RAG service with an OpenAI-compatible HTTP API. It loads `.txt`, `.md`, `.pdf`, and `.docx` files, chunks them, stores embeddings in ChromaDB, and answers through a LiteLLM-backed agentic tool loop.

Primary modules:

- `simple_chatbot/cli.py`: Typer CLI and runtime configuration.
- `simple_chatbot/server.py`: FastAPI `/v1/models`, `/v1/chat/completions`, and `/v1/responses` endpoints.
- `simple_chatbot/agent.py`: Agentic chat loop and `search_documents` tool handling.
- `simple_chatbot/responses.py`: Responses API request parsing, output-item mapping, and in-memory response store.
- `simple_chatbot/indexer.py`: ChromaDB persistence, embedding, and search.
- `simple_chatbot/loader.py`: Document loading and chunking.
- `simple_chatbot/guard.py`: Optional ScopeGuard gate.
- `simple_chatbot/conversation_logger.py`: JSONL conversation logging.

## Setup Commands

- Install dependencies: `uv sync`
- Install editable package with pip: `pip install -e .`
- Install optional ScopeGuard vLLM dependencies: `uv sync --extra guard-vllm`
- Install optional ScopeGuard Hugging Face dependencies: `uv sync --extra guard-hf`

Use Python 3.12 or newer, as declared in `pyproject.toml`.

## Running Locally

- Start the API with indexed documents: `uv run simple-chatbot serve --docs-dir ./docs`
- Force a rebuild of the vector index: `uv run simple-chatbot serve --docs-dir ./docs --reindex`
- Use `./start.sh` as an example hosted OpenAI launcher.
- Use `vllm_serve.sh` as an example local vLLM model server launcher.

The default API listens on `http://127.0.0.1:8000`. If a local vLLM server already uses port `8000`, run the chatbot on another port such as `--port 15077`. If binding to a network interface such as `--host 0.0.0.0`, configure `SIMPLE_CHATBOT_API_KEY` or `--api-key` so chat completions require bearer auth or `X-API-Key`.

## Test Commands

- Run unit tests: `uv run python -m unittest discover -s tests`
- Run the empty knowledge base tests only: `uv run python -m unittest tests.test_empty_knowledge_base`
- Smoke-test a running server: `./smoke_test.sh`
- Smoke-test a custom server URL: `./smoke_test.sh http://localhost:8000`

Prefer unit tests for code changes that can be exercised without external model calls. Use the smoke test only when a server and model backend are intentionally running.

## Code Style

- Keep the codebase typed with modern Python syntax: built-in generics like `list[dict]`, `str | None`, and `Path` for filesystem paths.
- Follow the existing module style: small functions, explicit configuration objects, and straightforward control flow.
- Use `async` APIs for LiteLLM calls and request handling; avoid blocking network/model calls inside async paths unless they are wrapped appropriately.
- Use `loguru` for runtime logging and keep log messages actionable.
- Keep comments sparse and useful, especially around non-obvious persistence, indexing, or agent-loop behavior.
- Preserve OpenAI-compatible request and response shapes in `server.py` unless the task explicitly changes the API contract.

## Testing Guidance

- Add or update tests for behavioral changes, especially around empty indexes, indexing decisions, guard behavior, and API response shape.
- Mock LiteLLM calls in unit tests unless the test is explicitly an integration or smoke test.
- Use temporary directories for ChromaDB and document fixtures in tests; do not depend on the repo's local `.chroma`, `docs`, or `conversations` directories.
- When changing chunking, embedding fingerprints, or collection metadata, include a test or manual verification that existing indexes are invalidated or reused as intended.

## Security And Data Handling

- Do not commit API keys, provider tokens, conversation logs, local vector stores, or user documents.
- Treat `.chroma/`, `conversations/`, and ad hoc `docs/` directories as local runtime data.
- Keep ScopeGuard configuration explicit when enabled: `--guard-service-description` or `--system-prompt` must describe the assistant's allowed scope.
- Avoid logging raw secrets or full user documents. Prefer counts, paths, short previews, or metadata when debugging.
- Validate file paths loaded via `@/path/to/file.txt` style options before broadening that pattern.

## Dependency And Runtime Notes

- `litellm` model strings may refer to hosted providers, local OpenAI-compatible servers, Ollama, or vLLM. Keep provider-specific behavior isolated behind config options.
- ChromaDB persistence is controlled by `--chroma-persist-dir`, defaulting to `./.chroma`.
- Conversation logs are JSONL files under `--conversation-log-dir`, defaulting to `./conversations`.
- Supported document extensions are defined in `simple_chatbot/loader.py`; update loader tests and README docs when adding formats.

## Pull Request Checklist

- Run the relevant unit tests before finishing changes.
- Run `./smoke_test.sh` when touching server behavior or the OpenAI-compatible API surface and a backend is available.
- Update `README.md` when changing CLI flags, defaults, supported file types, setup steps, or public behavior.
- Keep changes focused; do not rewrite generated lockfiles or local runtime data unless the task requires it.
