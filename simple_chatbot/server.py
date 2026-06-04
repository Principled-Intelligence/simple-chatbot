import hmac
import time
import uuid
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from pydantic import BaseModel

import litellm

from simple_chatbot.agent import Agent, sampling_kwargs
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
from simple_chatbot.scenario_orchestrator import ScenarioOrchestrator
from simple_chatbot.scenario_provider import DeterministicProvider, LiveProvider
from simple_chatbot.scenario_registry import load_fixtures
from simple_chatbot.scripted_indexer import ScriptedIndexer
from simple_chatbot.tools import ToolDef

app = FastAPI(title="simple-chatbot")

_config: SimpleChatbotConfig | None = None
_agent: Agent | None = None
_conversation_logger: ConversationLogger | None = None
_response_store: ResponseStore | None = None
_scenario_registry: dict | None = None
_acompletion: Callable[..., Awaitable[Any]] | None = None


def _require_config() -> SimpleChatbotConfig:
    assert _config is not None, "Server not initialised: call init() before serving requests"
    return _config


def _require_agent() -> Agent:
    assert _agent is not None, "Server not initialised: call init() before serving requests"
    return _agent


def _require_conversation_logger() -> ConversationLogger:
    assert _conversation_logger is not None, "Server not initialised: call init() before serving requests"
    return _conversation_logger


def _require_response_store() -> ResponseStore:
    assert _response_store is not None, "Server not initialised: call init() before serving requests"
    return _response_store


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


def init(
    config: SimpleChatbotConfig,
    indexer: Indexer | ScriptedIndexer,
    acompletion: Callable[..., Awaitable[Any]] | None = None,
    tools: list[ToolDef] | None = None,
) -> None:
    global _config, _agent, _conversation_logger, _response_store, _scenario_registry, _acompletion
    _config = config
    _acompletion = acompletion
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
    _agent = Agent(config, indexer, gate=gate, acompletion=acompletion, tools=tools)
    _conversation_logger = ConversationLogger(config.conversation_log_dir)
    _response_store = ResponseStore()
    _scenario_registry = load_fixtures()
    if config.default_fixture and config.default_fixture not in _scenario_registry:
        raise ValueError(
            f"default_fixture {config.default_fixture!r} is not a known fixture; "
            f"available: {sorted(_scenario_registry)}"
        )
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
    created = int(time.time())
    data = [{"id": model_id, "object": "model", "created": created, "owned_by": "simple-chatbot"}]
    for fixture_id in sorted(_scenario_registry or {}):
        data.append({"id": fixture_id, "object": "model", "created": created, "owned_by": "simple-chatbot-fixture"})
    logger.bind(model_id=model_id, fixture_count=len(_scenario_registry or {})).debug("Returning model list")
    return {"object": "list", "data": data}


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
        result = await _require_agent().chat(body.messages)
        elapsed_ms = (time.perf_counter() - start) * 1000

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        logger.bind(
            completion_id=completion_id,
            duration_ms=round(elapsed_ms, 1),
            response_chars=len(result.content),
            blocked_by_guard=result.blocked_by_guard,
        ).info("Chat completion finished")

        await _require_conversation_logger().log(
            conversation_id=conversation_id,
            messages=body.messages,
            response=result.content,
            chunks=[{"text": d.text, **d.metadata} for d in result.retrieved_chunks],
        )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _require_config().chat_model,
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

    prior_entry = None
    if body.previous_response_id:
        prior_entry = await _require_response_store().get(body.previous_response_id)
        if prior_entry is None:
            raise HTTPException(
                status_code=404,
                detail=_openai_error(
                    f"previous_response_id {body.previous_response_id!r} not found",
                    "invalid_request_error",
                    "previous_response_id",
                ),
            )
        raw_input = body.input
        input_is_empty = (
            (isinstance(raw_input, str) and not raw_input)
            or (isinstance(raw_input, list) and len(raw_input) == 0)
        )
        if input_is_empty:
            raise HTTPException(
                status_code=400,
                detail=_openai_error(
                    "input must contain at least one item when previous_response_id is set",
                    "invalid_request_error",
                    "input",
                ),
            )

    if prior_entry is not None:
        messages = list(prior_entry["session_messages"]) + new_messages
        conversation_id = prior_entry["conversation_id"]
    else:
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
        registry = _scenario_registry or {}
        config = _require_config()
        # Resolve which fixture (if any) handles this turn. An explicitly named
        # fixture always wins; otherwise fall back to the server's default
        # fixture so a client that can't set the model still gets one.
        effective_model = body.model
        scenario = registry.get(effective_model)
        if scenario is None and config.default_fixture:
            scenario = registry.get(config.default_fixture)
            if scenario is not None:
                effective_model = config.default_fixture
        if scenario is not None:
            resolved_mode = config.scenario_mode or scenario.mode
            if resolved_mode == "live":
                provider = LiveProvider(
                    acompletion=_acompletion or litellm.acompletion,
                    model=config.chat_model,
                    api_base=config.chat_api_base,
                    sampling_kwargs=sampling_kwargs(config),
                )
            else:
                provider = DeterministicProvider()
            orchestrator = ScenarioOrchestrator(scenario, provider)
            # Only resume the in-charge agent when the prior turn ran the SAME
            # scenario (selected by body.model). The orchestrator falls back to
            # the entry agent on an unknown name, so this is purely to avoid
            # carrying an agent across scenarios.
            start_agent = (
                prior_entry.get("active_agent")
                if prior_entry and prior_entry.get("model") == effective_model
                else None
            )
            result = await orchestrator.chat(messages, start_agent=start_agent)
        else:
            result = await _require_agent().chat(messages)
        elapsed_ms = (time.perf_counter() - start) * 1000

        payload = build_response(
            result=result,
            model=_require_config().chat_model,
            previous_response_id=body.previous_response_id,
            conversation_id=conversation_id,
        )

        logger.bind(
            response_id=payload["id"],
            duration_ms=round(elapsed_ms, 1),
            output_item_count=len(payload["output"]),
        ).info("Response create finished")

        await _require_conversation_logger().log(
            conversation_id=conversation_id,
            messages=messages,
            response=result.content,
            chunks=[{"text": d.text, **d.metadata} for d in result.retrieved_chunks],
        )

        await _require_response_store().put(
            payload["id"],
            {
                "response_id": payload["id"],
                "previous_response_id": body.previous_response_id,
                "session_messages": list(result.final_messages),
                "response_json": payload,
                "conversation_id": conversation_id,
                "model": effective_model,
                "active_agent": result.active_agent,
            },
        )

        return payload


@app.get("/v1/responses/{response_id}")
async def responses_retrieve(response_id: str, request: Request):
    _require_auth(request)
    entry = await _require_response_store().get(response_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=_openai_error(
                f"response {response_id!r} not found",
                "invalid_request_error",
                "response_id",
            ),
        )
    return entry["response_json"]
