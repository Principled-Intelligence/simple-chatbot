import hmac
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from pydantic import BaseModel

from simple_chatbot.agent import Agent
from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.conversation_logger import ConversationLogger, derive_conversation_id
from simple_chatbot.guard import ScopeGuardGate
from simple_chatbot.indexer import Indexer
from simple_chatbot.responses import (
    InvalidInputError,
    ResponseStore,
    build_response,
    normalize_input,
)

app = FastAPI(title="simple-chatbot")

_config: SimpleChatbotConfig | None = None
_agent: Agent | None = None
_conversation_logger: ConversationLogger | None = None
_response_store: ResponseStore | None = None


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    with logger.contextualize(request_id=request_id):
        logger.bind(method=request.method, path=request.url.path).info("HTTP request started")
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.bind(
                method=request.method,
                path=request.url.path,
                duration_ms=round(elapsed_ms, 1),
            ).exception("HTTP request failed")
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.bind(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(elapsed_ms, 1),
        ).info("HTTP request completed")
        return response


def init(config: SimpleChatbotConfig, indexer: Indexer) -> None:
    global _config, _agent, _conversation_logger, _response_store
    _config = config
    gate: ScopeGuardGate | None = None
    if config.guard.enabled:
        gate = ScopeGuardGate(config.guard, config.system_prompt)
        logger.bind(
            backend=config.guard.backend,
            model=config.guard.model,
            api_url=config.guard.api_url,
            description_source=gate.describe_source(),
            block_classes=config.guard.block_classes,
        ).info("Scope guard enabled")
    _agent = Agent(config, indexer, gate=gate)
    _conversation_logger = ConversationLogger(config.conversation_log_dir)
    _response_store = ResponseStore()
    logger.bind(
        model=config.chat_model,
        top_k=config.top_k,
        max_tool_rounds=config.max_tool_rounds,
        guard_enabled=config.guard.enabled,
    ).info("Server initialised")


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    stream: bool = False
    # OpenAI-compatible optional user/session id. When provided by the
    # client, it's used as the conversation id; otherwise one is derived
    # from the first user message.
    user: str | None = None


class ResponsesRequest(BaseModel):
    model: str | None = None
    input: str | list = ""
    previous_response_id: str | None = None
    tools: list | None = None
    tool_choice: object = None
    user: str | None = None
    stream: bool = False
    # All other Responses API fields are accepted but ignored.
    model_config = {"extra": "allow"}


def _require_auth(request: Request) -> None:
    api_key = _config.api_key if _config else None
    if not api_key:
        return

    supplied_keys: list[str] = []
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        supplied_keys.append(token.strip())

    x_api_key = request.headers.get("x-api-key")
    if x_api_key:
        supplied_keys.append(x_api_key.strip())

    if any(hmac.compare_digest(key, api_key) for key in supplied_keys):
        return

    logger.bind(path=request.url.path).warning("Rejected unauthorized request")
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _openai_error(message: str, error_type: str, param: str | None = None) -> dict:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": None,
        }
    }


@app.get("/v1/models")
def list_models():
    model_id = _config.chat_model if _config else "unknown"
    logger.bind(model_id=model_id).debug("Returning model list")
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "simple-chatbot"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    _require_auth(request)

    if body.stream:
        logger.bind(stream=body.stream).warning("Rejected streaming chat completion request")
        raise HTTPException(
            status_code=400,
            detail=_openai_error(
                "stream=true is not supported by this server yet",
                "invalid_request_error",
                "stream",
            ),
        )

    conversation_id = (
        body.user
        or request.headers.get("x-conversation-id")
        or derive_conversation_id(body.messages)
    )

    with logger.contextualize(conversation_id=conversation_id):
        logger.bind(
            model=body.model,
            message_count=len(body.messages),
            stream=body.stream,
        ).info("Chat completion started")

        start = time.perf_counter()
        result = await _agent.chat(body.messages)
        elapsed_ms = (time.perf_counter() - start) * 1000

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        logger.bind(
            completion_id=completion_id,
            duration_ms=round(elapsed_ms, 1),
            response_chars=len(result.content),
            blocked_by_guard=result.blocked_by_guard,
        ).info("Chat completion finished")

        await _conversation_logger.log(
            conversation_id=conversation_id,
            messages=body.messages,
            response=result.content,
            chunks=[{"text": d.text, **d.metadata} for d in result.retrieved_chunks],
        )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _config.chat_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "conversation_id": conversation_id,
            "tools": result.tools,
            "tool_messages": result.tool_messages,
        }


@app.post("/v1/responses")
async def responses_create(request: Request, body: ResponsesRequest):
    _require_auth(request)

    if body.stream:
        logger.bind(stream=body.stream).warning("Rejected streaming response request")
        raise HTTPException(
            status_code=400,
            detail=_openai_error(
                "stream=true is not supported by this server yet",
                "invalid_request_error",
                "stream",
            ),
        )

    try:
        new_messages = normalize_input(body.input)
    except InvalidInputError as exc:
        raise HTTPException(
            status_code=400,
            detail=_openai_error(str(exc), "invalid_request_error", "input"),
        ) from exc

    # Single-turn path — chain handling lands in Task 10.
    messages = new_messages

    conversation_id = (
        body.user
        or request.headers.get("x-conversation-id")
        or derive_conversation_id(messages)
    )

    with logger.contextualize(conversation_id=conversation_id):
        logger.bind(
            model=body.model,
            message_count=len(messages),
            previous_response_id=body.previous_response_id,
        ).info("Response create started")

        start = time.perf_counter()
        result = await _agent.chat(messages)
        elapsed_ms = (time.perf_counter() - start) * 1000

        payload = build_response(
            result=result,
            model=_config.chat_model,
            previous_response_id=body.previous_response_id,
            conversation_id=conversation_id,
        )

        logger.bind(
            response_id=payload["id"],
            duration_ms=round(elapsed_ms, 1),
            output_item_count=len(payload["output"]),
        ).info("Response create finished")

        await _conversation_logger.log(
            conversation_id=conversation_id,
            messages=messages,
            response=result.content,
            chunks=[{"text": d.text, **d.metadata} for d in result.retrieved_chunks],
        )

        await _response_store.put(
            payload["id"],
            {
                "response_id": payload["id"],
                "previous_response_id": body.previous_response_id,
                "session_messages": list(result.final_messages),
                "response_json": payload,
                "conversation_id": conversation_id,
            },
        )

        return payload
