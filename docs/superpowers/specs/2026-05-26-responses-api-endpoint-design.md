# Responses API endpoint — design

**Date:** 2026-05-26
**Status:** Approved (pending spec review)
**Author:** Niccolò Campolungo / Claude

## Goal

Add a new HTTP endpoint that speaks OpenAI's Responses API wire format, alongside
the existing Chat Completions endpoint. The Responses endpoint exists to support a
downstream evaluation/guardrails integration that needs to inspect the complete
ordered trace of tool calls and tool results for each user turn.

The existing Chat Completions endpoint stays untouched for backward compatibility.

## Why this matters

The downstream evaluator scores tool selection, argument correctness, and
faithfulness of the final response to what the tools returned. If we collapse the
trace to "just the final assistant message", evaluation breaks. Every tool call and
its result must appear in the response body, in the order they happened, with stable
linkage between calls and results.

## Non-goals

- Streaming (`stream: true`). Rejected with 400, matching the existing endpoint.
- Sharing request-handling logic between the two endpoints. Duplication is fine.
- Honoring client-supplied `tools` / `tool_choice` (accepted but ignored — we expose
  only our built-in `search_documents`).
- Persistent storage of responses. In-memory only; lost on server restart.
- Observability / OpenTelemetry — the response body IS the trace.
- Image inputs, structured outputs, reasoning items, background mode.

## Approach

Build a thin Responses adapter on top of the existing `Agent`. The agent already
produces a `ChatResult` whose `tool_messages` field contains the full ordered trace
of assistant-with-tool-calls and tool-result messages (chat-completions-shaped), and
whose `content` field holds the final assistant text. A new module converts that
trace into Responses API output items using the `openai` SDK's typed objects.

Considered and rejected:
- A parallel `Agent.respond()` method that builds Responses output items directly —
  duplicates the loop for no real gain.
- Refactoring the agent to emit a neutral event stream both endpoints adapt — clean
  but overkill for one new endpoint.

## Module structure

- **New module:** `simple_chatbot/responses.py`
  - `ResponseStore`: in-memory dict keyed by `response_id`. Stores the full
    conversation state needed to extend a chain plus the rendered response JSON.
  - Input-item → chat-completions-messages conversion.
  - `ChatResult` + input messages → Responses API output items conversion. Uses
    `openai.types.responses` types (`Response`, `ResponseOutputMessage`,
    `ResponseFunctionToolCall`, `ResponseFunctionToolCallOutput`, `ResponseUsage`).
  - Responses-API-shaped tool schema derived from `SEARCH_TOOL` (flat
    `name`/`parameters`, no `function:` nesting).
- **`simple_chatbot/server.py`:** add `POST /v1/responses` and
  `GET /v1/responses/{response_id}`. Rename `_require_chat_auth` to `_require_auth`
  and reuse it.
- **`simple_chatbot/agent.py`:** additive changes only.
  - Skip the system-prompt prepend if `messages[0]` is already `role: "system"`.
  - Accumulate prompt/completion/total token counts across all LiteLLM calls and
    surface them on `ChatResult.usage`.
  - Expose `ChatResult.final_messages` — the full working list including the final
    assistant message appended.
- **`pyproject.toml`:** add `openai>=1.50.0` for typed schemas. We do not call
  OpenAI's HTTP client.

## Wire format

### Endpoint

`POST /v1/responses` — synchronous. Auth via `Authorization: Bearer <key>` or
`X-API-Key`, same logic as Chat Completions.

`GET /v1/responses/{response_id}` — returns the stored Response JSON. 404 if
missing. Useful for the evaluator to re-fetch a turn.

### Request body (fields honored)

| Field                  | Type                | Notes                                                          |
| ---------------------- | ------------------- | -------------------------------------------------------------- |
| `model`                | `str?`              | Echoed back; server uses its configured `chat_model`           |
| `input`                | `str \| list[item]` | Required. See "Input normalization" below.                     |
| `previous_response_id` | `str?`              | When set, prepend stored session messages from that response.  |
| `tools`                | list                | Accepted but ignored. Always exposes `search_documents`.       |
| `tool_choice`          | any                 | Accepted but ignored.                                          |
| `user`                 | `str?`              | Conversation-id role, same as Chat Completions.                |
| `stream`               | `bool`              | Rejected with 400 if true.                                     |

Other Responses API fields (e.g., `temperature`, `instructions`, `metadata`) are
accepted but ignored.

### Input normalization → chat-completions messages

- `str` → `{role: "user", content: <str>}`
- `list[...]` — walk in order:
  - bare `str` item → `{role: "user", content: ...}`
  - `{type: "message", role, content}` → corresponding chat-completions message.
    `content` may be a plain string or a content-parts list; pass through unchanged
    (LiteLLM handles content parts).
  - `{type: "function_call", ...}` and `{type: "function_call_output", ...}` — only
    expected when the evaluator is re-submitting a prior chain manually. Normally
    these come from `previous_response_id` lookup, not client input. Accepted in
    principle but discouraged.
  - Unknown item types → 400, `invalid_request_error`, `param: "input"`.

### Response body

Built from `openai.types.responses.Response` and dumped to JSON. Fields:

| Field                  | Value                                                    |
| ---------------------- | -------------------------------------------------------- |
| `id`                   | `resp_<uuid hex>`                                        |
| `object`               | `"response"`                                             |
| `created_at`           | epoch seconds (int)                                      |
| `status`               | `"completed"` on success                                 |
| `model`                | `_config.chat_model`                                     |
| `output`               | Ordered array — see "Output items mapping"               |
| `usage`                | Accumulated `input_tokens`, `output_tokens`, `total_tokens` |
| `previous_response_id` | Echoed back when supplied (else `null`)                  |
| `conversation_id`      | Our derivation, same as Chat Completions (non-standard)  |

Guard-blocked turns and empty-knowledge-base turns still return `status: "completed"`
with a single `message` item carrying the refusal/explanation text. They are valid
agent responses, not errors.

## Output items mapping — the critical part

The agent today exposes:

- `ChatResult.tool_messages` — list of dicts in chat-completions shape, in the order
  they happened:
  ```
  [
    {role: "assistant", content: "...", tool_calls: [{id, function: {name, arguments}}, ...]},
    {role: "tool", tool_call_id, name, content},
    {role: "tool", tool_call_id, name, content},
    {role: "assistant", content: "...", tool_calls: [...]},     // next round
    {role: "tool", ...},
    ...
  ]
  ```
- `ChatResult.content` — final assistant text (separately, because the agent loop
  returns it instead of appending it to `tool_messages`).

Mapping rule:

1. Walk `tool_messages` in order.
   - For each `role: "assistant"` entry that has `tool_calls`, emit one
     `ResponseFunctionToolCall` per tool call, preserving the array order. Fields:
     - `type: "function_call"`
     - `id`: `fc_<uuid hex>` (Responses item id, stable per call)
     - `call_id`: the LiteLLM `tool_calls[].id` — **pairs with the matching output**
     - `name`: `tool_calls[].function.name`
     - `arguments`: `tool_calls[].function.arguments` (JSON string per OpenAI spec)
     - `status`: `"completed"`
   - For each `role: "tool"` entry, emit one `ResponseFunctionToolCallOutput`:
     - `type: "function_call_output"`
     - `id`: `fco_<uuid hex>`
     - `call_id`: the same `tool_call_id` from the tool message — pairs with the
       matching `function_call`
     - `output`: the tool result string (search context or `"Tool error: ..."`)
2. Append a final `ResponseOutputMessage` for the assistant's text:
   - `type: "message"`
   - `id`: `msg_<uuid hex>`
   - `role: "assistant"`
   - `status: "completed"`
   - `content`: `[{type: "output_text", text: <ChatResult.content>, annotations: []}]`

### call_id linkage

The LiteLLM `tool_call.id` is already threaded through to the tool-result message's
`tool_call_id` in the existing agent code (`simple_chatbot/agent.py:208`). We reuse
that same string for both the `function_call.call_id` and the matching
`function_call_output.call_id`. That linkage is what the downstream evaluator parses
to align calls with their results.

### Relationship to vanilla OpenAI shape

OpenAI's vanilla Responses API for *function* tools returns only `function_call`
items in `output`, expecting the client to execute the tool and submit the result
as a `function_call_output` in the *next* request's `input`. A single Response from
OpenAI does not contain `function_call_output` items inline.

Our design inlines both `function_call` and `function_call_output` in the same
Response's `output`, because our agent executes the tool server-side. The total set
of items the evaluator sees is the same as it would see walking an OpenAI chain
(`previous_response_id` → `previous_response_id` → ...); we just package one user
turn as one Response object instead of multiple. The SDK types used are the same.

### Edge cases

- **Guard-blocked turn:** `tool_messages` is empty (the agent short-circuits before
  the loop); output is just `[message]`. Status `completed`.
- **Empty knowledge base:** same — single `message` item, status `completed`.
- **Max-rounds-reached (forced final):** `tool_messages` contains every round that
  did happen. The forced final assistant text from the no-tools follow-up call
  becomes the final `message` item.
- **Tool-error result** (malformed args, unsupported tool name): the
  `function_call_output.output` carries the `"Tool error: ..."` string — visible in
  the trace, not hidden.

## Multi-turn via `previous_response_id`

### What we store per response

`ResponseStore` keyed by `response_id`:

- `response_id`
- `previous_response_id` (or `None`)
- `session_messages: list[dict]` — the full working list the agent built up plus
  the final assistant message. Exactly the prefix needed for the next turn.
- `response_json: dict` — the rendered Response payload (so `GET` is a dict
  lookup, no re-derivation).
- `conversation_id: str` — for inheriting on chained turns.

### Turn lifecycle

1. Request arrives. If `previous_response_id` is set:
   - Look it up in the store. If missing, return 404 with a Responses-shaped error.
   - Start `messages = list(stored.session_messages)`.
   - Inherit `conversation_id` from the stored entry.
   - If the new request `input` is empty, return 400.
2. Normalize the request's `input` into chat-completions messages and append.
3. Call `agent.chat(messages)`. The agent's updated system-prompt logic detects the
   existing system message (if any) and does not double-prepend.
4. Build the Response:
   - Map `ChatResult` → output items (above).
   - Build `session_messages` for storage: `ChatResult.final_messages` from the
     agent (working list including final assistant content).
5. Store the entry and return the Response JSON.

### Agent changes (additive only)

- Skip system-prompt prepend if `messages[0]["role"] == "system"`.
- Accumulate `prompt_tokens` / `completion_tokens` / `total_tokens` across all
  LiteLLM calls in the loop (including the forced-final call). Surface as
  `ChatResult.usage` using LiteLLM's field names. The Responses adapter renames
  them to `input_tokens` / `output_tokens` / `total_tokens` when building the
  `ResponseUsage` payload — that's the OpenAI Responses naming convention.
- Expose `ChatResult.final_messages: list[dict]` — the working list the agent used,
  with the final `{role: "assistant", content: ...}` appended.

### Conversation id derivation

- When `previous_response_id` is set: reuse the stored conversation_id.
- Otherwise: same derivation as Chat Completions (`body.user` →
  `x-conversation-id` header → `derive_conversation_id(messages)`).

### Edge cases

- Unknown `previous_response_id` → 404, Responses-shaped error envelope.
- `previous_response_id` set + empty `input` → 400.
- No cap on chain depth or store size initially. Acceptable for evaluation runs
  inside a single server lifetime.

## Error handling

- All HTTP errors use the existing `_openai_error(message, type, param)` envelope.
  The Responses API uses the same `error` envelope shape as Chat Completions.
- Unknown `input` item types → 400, `invalid_request_error`, `param: "input"`.
- `stream: true` → 400, `invalid_request_error`, `param: "stream"`.
- Missing/invalid auth → 401, same as Chat Completions.
- `previous_response_id` unknown → 404, `invalid_request_error`,
  `param: "previous_response_id"`.
- Agent exceptions bubble up to the `_log_requests` middleware → 500.

## Testing

Unittest style, mocked LiteLLM. Add to `tests/`:

1. `test_responses_endpoint.py` (FastAPI `TestClient` against `/v1/responses`):
   - Single-turn, string `input`: output is `[message]`, status `completed`, usage
     summed across the single LiteLLM call.
   - Single-turn with one tool call: output is `[function_call,
     function_call_output, message]` in order; `call_id` matches across the pair.
   - Multi-tool-call single turn: 3 search calls → output is `[fc, fco, fc, fco,
     fc, fco, message]` reflecting the existing agent's interleaving of tool
     results after each round's calls.
   - List-form `input` with `{type: "message", role: "user", content: "..."}` works
     identically to bare string.
   - `previous_response_id` chain: second turn's LiteLLM call receives the prior
     turn's full session messages as the prefix; chained turns share the same
     `conversation_id`.
   - `previous_response_id` unknown → 404 with proper error envelope.
   - Streaming rejected with 400.
   - Auth: missing key when configured → 401; bearer + X-API-Key both accepted.
   - `GET /v1/responses/{id}` returns the stored response JSON; unknown id → 404.
2. `test_responses_mapping.py` (pure unit, no FastAPI):
   - `ChatResult` → output items conversion, including tool-error path
     (`"Tool error: ..."` reaches `function_call_output.output`).
   - Guard-blocked turn → output is `[message]`, status `completed`.
   - Empty knowledge base → output is `[message]`, status `completed`.
   - Max-rounds-reached → all tool calls appear in order; forced final message item
     appended.
3. Existing tests remain green. `test_agent_tool_calls.py` may need light updates
   to account for the new additive `ChatResult` fields (`usage`,
   `final_messages`).

## Docs / runtime updates

- `README.md`: brief section documenting the new endpoint, request/response
  example, `previous_response_id` behavior, what's ignored.
- `AGENTS.md`: add `responses.py` to the module list.
- `smoke_test.sh`: add a Responses API smoke check after the existing Chat
  Completions check.
