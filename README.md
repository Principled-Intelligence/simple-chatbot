# simple-chatbot

> Spin up a configurable document chatbot with a local vector index and an
> OpenAI-compatible HTTP API.

[![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Built with uv](https://img.shields.io/badge/built%20with-uv-261230.svg)](https://docs.astral.sh/uv/)

Point `simple-chatbot` at a folder of `.txt`, `.md`, `.pdf`, or `.docx` files.
It chunks the documents, stores embeddings in a local ChromaDB collection, and
answers through a LiteLLM-backed agent that can search your documents when it
needs context.

This repo is intentionally small: it is meant to help developers get a useful
RAG chatbot running quickly, understand the moving parts, and customize the
models or retrieval settings without adopting a production platform.

| Good for | Not meant for |
| --- | --- |
| Local document Q&A prototypes | Production auth, billing, or tenant isolation |
| Testing OpenAI-compatible chat clients | Streaming chat completions |
| Trying hosted or local LiteLLM model backends | Managed document ingestion pipelines |
| Learning a compact agentic RAG flow | Large-scale observability or deployment automation |

## At a Glance

| Item | Default |
| --- | --- |
| Python | `>=3.12` |
| Server URL | `http://127.0.0.1:8000` |
| API routes | `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/responses`, `GET /v1/responses/{id}` |
| Chat model | `openai/gpt-5.4-nano` |
| Embedding model | `openai/text-embedding-3-small` |
| Documents | `.txt`, `.md`, `.pdf`, `.docx` |
| Vector store | `./.chroma` |
| Conversation logs | `./conversations` |

Streaming is not implemented yet. Requests with `stream=true` return HTTP 400.

## Quick Start

Requires Python `>=3.12`. The shortest path uses
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

<details>
<summary>Install with pip instead</summary>

```bash
pip install -e .
```

</details>

Create a tiny local knowledge base, or replace this with your own documents:

```bash
mkdir -p docs
echo "simple-chatbot indexes local documents and answers through an OpenAI-compatible API." > docs/intro.md
```

Set the provider key for the default OpenAI-backed chat and embedding models:

```bash
export OPENAI_API_KEY="sk-..."
```

Start the server:

```bash
uv run simple-chatbot serve --docs-dir ./docs
```

The first run chunks, embeds, and indexes everything under `./docs`. Later runs
reuse cached vectors unless the embedding configuration changes or you pass
`--reindex`.

Ask a question with `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple-chatbot",
    "messages": [{"role": "user", "content": "What does the intro document say?"}]
  }'
```

Or use the official OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="simple-chatbot",
    messages=[{"role": "user", "content": "What does the intro document say?"}],
)
print(resp.choices[0].message.content)
```

Prefer a local `.env` file? Copy the example, fill in values, then source it
before running the server:

```bash
cp .env.example .env
$EDITOR .env
set -a
source .env
set +a
```

The app does not load `.env` automatically.

## What You Get

- OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints for
  non-streaming clients.
- Agentic retrieval: the model decides when to call the `search_documents` tool
  and can search multiple times per turn.
- Chat completion responses expose the agent's tool schemas (`tools`) and the
  full intermediate tool-call / tool-result trace (`tool_messages`) as
  non-standard fields, so an external LLM-as-judge can inspect what the agent
  actually invoked.
- Persistent ChromaDB storage, with automatic rebuilds when the embedding or
  chunking fingerprint changes.
- Independent chat and embedding model selection through
  [LiteLLM](https://github.com/BerriAI/litellm).
- JSONL conversation logs containing messages, final answers, and retrieved
  chunks.
- Optional ScopeGuard pre-LLM classification for blocking out-of-scope or
  restricted requests.

## Common Configuration

Use the defaults first. Then reach for these flags when you need a specific
behavior:

| Task | Command |
| --- | --- |
| Rebuild the vector index | `uv run simple-chatbot serve --docs-dir ./docs --reindex` |
| Use a custom system prompt | `uv run simple-chatbot serve --docs-dir ./docs --system-prompt "Answer only from the indexed docs."` |
| Load a prompt from disk | `uv run simple-chatbot serve --docs-dir ./docs --system-prompt @./prompt.txt` |
| Change chunking | `uv run simple-chatbot serve --docs-dir ./docs --chunk-size 800 --chunk-overlap 100` |
| Run on another port | `uv run simple-chatbot serve --docs-dir ./docs --port 15077` |
| Require API auth | `uv run simple-chatbot serve --docs-dir ./docs --api-key "dev-secret"` |
| Run offline (no API keys) | `SIMPLE_CHATBOT_SCRIPTED_LLM=1 uv run simple-chatbot serve --docs-dir /tmp/empty` |

By default, the API binds to `127.0.0.1` and does not require a key. If you bind
to a network interface such as `--host 0.0.0.0`, set `--api-key` or
`SIMPLE_CHATBOT_API_KEY` so `/v1/chat/completions` requires either
`Authorization: Bearer <key>` or `X-API-Key: <key>`.

## Recipes

### Custom models with LiteLLM

Any chat or embedding model supported by LiteLLM can be used. Hosted providers,
OpenAI-compatible local servers, Ollama, and vLLM all fit behind the same flags:

```bash
uv run simple-chatbot serve \
  --docs-dir ./docs \
  --chat-model "openai/gpt-5.4-nano" \
  --embedding-model "openai/text-embedding-3-small"
```

Use `--chat-api-base` or `--embedding-api-base` when a backend exposes an
OpenAI-compatible endpoint at a custom URL.

<details>
<summary>Use a local vLLM backend</summary>

`vllm_serve.sh` shows an example of hosting
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) with vLLM on port `8000`:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B \
  --quantization fp8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Run `simple-chatbot` on a different port because vLLM is already using `8000`:

```bash
uv run simple-chatbot serve \
  --docs-dir ./docs \
  --chat-model "openai/Qwen/Qwen3-8B" \
  --chat-api-base "http://localhost:8000/v1" \
  --embedding-model "openai/text-embedding-3-small" \
  --port 15077
```

The `openai/` prefix tells LiteLLM to use the OpenAI-compatible protocol. vLLM
must be started with automatic tool choice and a tool-call parser so the
agent's `search_documents` calls work.

</details>

<details>
<summary>Enable ScopeGuard</summary>

ScopeGuard runs before the agentic RAG loop. You provide an AI service
description that says what the assistant may and may not do. Queries classified
into configured block classes receive a refusal instead of calling the chat
model.

Enable it with `--enable-guard` and set scope text with
`--guard-service-description`. If the description flag is omitted, ScopeGuard
falls back to `--system-prompt`.

Hosted API backend:

```bash
export ORBITALS_API_KEY="..."
uv run simple-chatbot serve --docs-dir ./docs \
  --enable-guard \
  --guard-backend api \
  --guard-api-key "$ORBITALS_API_KEY" \
  --guard-service-description "You answer only from indexed documents. Do not give medical or legal advice."
```

Self-hosted ScopeGuard HTTP server:

```bash
uv run simple-chatbot serve --docs-dir ./docs \
  --enable-guard \
  --guard-backend api \
  --guard-api-url "http://localhost:8000" \
  --system-prompt "You are a document Q&A assistant for internal docs only."
```

In-process vLLM backend:

```bash
uv sync --extra guard-vllm

uv run simple-chatbot serve --docs-dir ./docs \
  --enable-guard \
  --guard-backend vllm \
  --guard-model scope-guard-q \
  --guard-service-description @./scope_policy.txt
```

In-process Hugging Face backend:

```bash
uv sync --extra guard-hf
```

Block classes are configurable with `--guard-block-classes`. The default is
`Out of Scope,Restricted`.

</details>

<details>
<summary>Smoke-test a running server</summary>

The smoke test hits `GET /v1/models`, then `POST /v1/chat/completions`, and
asserts a non-empty assistant response.

```bash
./smoke_test.sh                              # defaults to http://localhost:15077
./smoke_test.sh http://localhost:8000        # custom base URL
./smoke_test.sh --message "hi"               # flags forwarded to smoke_test.py
./smoke_test.sh http://host:port --timeout 120
BASE_URL=http://host:port ./smoke_test.sh    # via env
```

You can also invoke the Python script directly. Its default base URL is the
server default, `http://localhost:8000`:

```bash
python smoke_test.py --base-url http://localhost:8000 --message "hi"
```

</details>

### Responses API

In addition to `/v1/chat/completions`, the server exposes
`POST /v1/responses` and `GET /v1/responses/{id}` implementing OpenAI's
Responses API wire format. Every tool call and tool result appears in the
response's `output` array, in the order it happened, so downstream evaluators
can inspect the full trace.

**Single turn:**

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input": "What does the doc say about onboarding?"}'
```

Response (abbreviated):

```json
{
  "id": "resp_...",
  "object": "response",
  "status": "completed",
  "output": [
    {"type": "function_call", "call_id": "call_1", "name": "search_documents", "arguments": "{\"query\": \"onboarding\"}"},
    {"type": "function_call_output", "call_id": "call_1", "output": "..."},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "...", "annotations": []}]}
  ],
  "usage": {"input_tokens": 123, "output_tokens": 45, "total_tokens": 168}
}
```

**Multi-turn:** Pass the prior response's `id` as `previous_response_id`. Only
new input items are needed; the server reconstructs prior context:

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input": "tell me more", "previous_response_id": "resp_..."}'
```

**Notes:**
- `tools` and `tool_choice` in the request are accepted but ignored — the
  server always exposes its built-in `search_documents` tool.
- Streaming (`stream: true`) is not supported.
- Response state is stored in-memory; `previous_response_id` chains do not
  survive a server restart.

**Offline testing:** Set `SIMPLE_CHATBOT_SCRIPTED_LLM=1` before starting the server
to skip every external call. A canned search tool returns a query-echoing fake
document and a canned LLM returns a final answer that includes the search result,
so a `POST /v1/responses` request produces a complete `[function_call,
function_call_output, message]` trace without touching the LLM or embedding APIs.
No `--docs-dir` content is read in this mode (the flag is still required for
CLI compatibility, but a non-existent or empty path works).

**Multi-tool evaluator harness:** When `SIMPLE_CHATBOT_SCRIPTED_LLM=1` is active,
the server exposes four tools to the model and selects them heuristically from
the user's input. Use this to produce varied trace shapes for downstream
evaluator testing.

| Tool                              | Triggered when the user input contains…           |
| --------------------------------- | ------------------------------------------------- |
| `search_documents` (default)      | Any input that doesn't match another tool         |
| `calculate(expression)`           | `calculate`, `compute`, `math`, or `<digit>±<digit>` |
| `get_current_time(tz?)`           | The word `time`; `tz` extracted from `in <name>`  |
| `lookup_user(user_id)`            | `lookup` or `user_id`                              |

Use explicit markers in the user input to force specific trace shapes:

| Marker          | Effect on the next turn                                                       |
| --------------- | ----------------------------------------------------------------------------- |
| `[parallel]`    | At least two tool calls in one assistant message → `[fc, fc, fco, fco, message]` |
| `[reasoning]`   | Adds a `reasoning` output item before tool calls / the final message          |
| `[multi-round]` | Two sequential tool rounds before answering → `[fc, fco, fc, fco, message]`   |
| `[error]`       | First tool call has malformed JSON arguments → exercises the agent's error path |

**Exhaustive showcase mode:** Set `EXHAUSTIVE_TOOL_USE=1` alongside
`SIMPLE_CHATBOT_SCRIPTED_LLM=1` to make every scripted user turn cycle through
a fixed 5-slot rotation that exercises every mocked tool and every trace
shape over a typical conversation. By construction, the first user turn fires
all four tools in parallel and subsequent turns rotate through multi-round,
errored, parallel-pair, and plain-single shapes:

| User turn (mod 5) | Shape          | Tools                                                          |
|-------------------|----------------|----------------------------------------------------------------|
| 0                 | All-parallel   | `search_documents`, `calculate`, `get_current_time`, `lookup_user` |
| 1                 | Multi-round    | round 1 `search_documents`, round 2 `calculate`                |
| 2                 | Errored call   | `lookup_user` with malformed JSON args                          |
| 3                 | Parallel pair  | `calculate`, `get_current_time`                                |
| 4                 | Plain single   | `search_documents`                                             |

Explicit bracketed markers (`[parallel]`, `[multi-round]`, `[error]`,
`[reasoning]`) in the user's text still take precedence over the rotation, so
a tester can force any specific shape on any individual turn.

Every user turn is routed through tool-selection heuristics — there is no
"chained-turn suppression" in the mock. Follow-up turns can pick different
tools than the prior turn, which is how you'd want a real agent to behave
across a multi-step conversation.

Examples:

```bash
# Parallel: two tool calls in one assistant message
curl -X POST http://localhost:15078/v1/responses -H 'Content-Type: application/json' \
  -d '{"input": "[parallel] calculate 2+2 and lookup user alice"}'

# Multi-round: two sequential tool rounds
curl -X POST http://localhost:15078/v1/responses -H 'Content-Type: application/json' \
  -d '{"input": "[multi-round] research deeply"}'

# Reasoning + tool call
curl -X POST http://localhost:15078/v1/responses -H 'Content-Type: application/json' \
  -d '{"input": "[reasoning] calculate the cost"}'
```

## How It Works

```text
Client
  |
  |  /v1/chat/completions
  v
FastAPI app
  |
  v
Agent loop (LiteLLM)
  |
  |  search_documents(query)
  v
Indexer + ChromaDB
```

1. `load_documents` walks `--docs-dir`, extracts text from supported files, and
   splits it into overlapping character chunks.
2. `Indexer` embeds chunks that are not already stored in ChromaDB and upserts
   them into the collection.
3. If the embedding fingerprint changes, the stale collection is dropped so the
   documents are re-embedded cleanly.
4. For each chat request, `Agent.chat` exposes one tool: `search_documents`.
   The model can call it, receive retrieved chunks as tool messages, and repeat
   until it produces a final answer or reaches `--max-tool-rounds`.
5. Each request is appended to a JSONL conversation log under
   `--conversation-log-dir`.

## Reference

<details>
<summary>Environment variables</summary>

Use `.env.example` as a template for local secrets and frequently used runtime
values. Local `.env` files are ignored by git.

| Variable | Used by | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | LiteLLM | Provider API key for OpenAI-backed chat and embedding models |
| `ORBITALS_API_KEY` | ScopeGuard | Hosted ScopeGuard API key when `--guard-api-url` is not set |
| `SIMPLE_CHATBOT_API_KEY` | simple-chatbot | API key required by `/v1/chat/completions` when `--api-key` is not used |
| `BASE_URL` | smoke test | Target server URL for `./smoke_test.sh` |

</details>

<details>
<summary>Full CLI options</summary>

```text
simple-chatbot serve --docs-dir PATH [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--docs-dir` | required | Directory of documents to index recursively |
| `--chat-model` | `openai/gpt-5.4-nano` | LiteLLM chat model string |
| `--embedding-model` | `openai/text-embedding-3-small` | LiteLLM embedding model string |
| `--chat-api-base` | `None` | Override API base for the chat model |
| `--embedding-api-base` | `None` | Override API base for the embedding model |
| `--chunk-size` | `500` | Characters per chunk; must be greater than `0` |
| `--chunk-overlap` | `50` | Overlap between chunks; must satisfy `0 <= overlap < chunk_size` |
| `--top-k` | `5` | Chunks returned per search |
| `--chroma-persist-dir` | `./.chroma` | Where ChromaDB stores vectors |
| `--collection-name` | `simple_chatbot` | ChromaDB collection name |
| `--host` | `127.0.0.1` | Server bind address |
| `--port` | `8000` | Server port |
| `--api-key` | `None` | Require this key on `/v1/chat/completions`; falls back to `SIMPLE_CHATBOT_API_KEY` |
| `--max-tool-rounds` | `5` | Max agentic loop iterations per request |
| `--system-prompt` | `None` | System prompt text, or `@/path/to/file.txt` to load from disk |
| `--conversation-log-dir` | `./conversations` | Directory for conversation JSONL logs |
| `--reindex` | off | Clear the collection and rebuild it from current documents |
| `--log-level` | `INFO` | `TRACE`, `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
| `--log-format` | `pretty` | Operational log format: `pretty` or `json` |
| `--enable-guard` | off | Enable ScopeGuard before the LLM |
| `--guard-backend` | `api` | `api`, `vllm`, or `huggingface` |
| `--guard-model` | `None` | Required for `vllm` or `huggingface` |
| `--guard-api-url` | `None` | Self-hosted ScopeGuard HTTP API base URL |
| `--guard-api-key` | `None` | ScopeGuard API key; falls back to `ORBITALS_API_KEY` only when `--guard-api-url` is unset |
| `--guard-service-description` | `None` | Scope policy text, or `@/path/to/file.txt`; else uses `--system-prompt` |
| `--guard-block-classes` | `Out of Scope,Restricted` | Comma-separated classes that trigger refusal |
| `--guard-refusal-message` | `I'm sorry, I can't help with that request.` | Assistant reply when a blocked class is detected |

</details>

<details>
<summary>Project layout</summary>

```text
simple_chatbot/
├── agent.py                 # Agentic chat loop (LiteLLM + tool calls)
├── guard.py                 # ScopeGuard pre-LLM gate
├── cli.py                   # simple-chatbot serve Typer CLI
├── config.py                # Pydantic config
├── conversation_logger.py   # JSONL conversation logs
├── indexer.py               # ChromaDB + LiteLLM embeddings
├── loader.py                # txt / md / pdf / docx loaders + chunking
├── server.py                # FastAPI OpenAI-compatible endpoints
└── _logging.py              # Loguru setup
main.py                      # python main.py entrypoint
start.sh                     # Example hosted OpenAI launcher
vllm_serve.sh                # Example local vLLM backend
```

</details>

## License

Apache 2.0. See [LICENSE](LICENSE).
