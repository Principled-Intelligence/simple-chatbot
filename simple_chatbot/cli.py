import os
from pathlib import Path
from typing import Literal, Optional

import typer
import uvicorn
from loguru import logger

from simple_chatbot._logging import setup_logging
from simple_chatbot.config import GuardConfig, SimpleChatbotConfig
from simple_chatbot.indexer import Indexer
from simple_chatbot.loader import load_documents
from simple_chatbot.server import app, init

cli = typer.Typer(name="simple-chatbot", add_completion=False)


def _parse_guard_block_classes(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_text_or_file(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    if raw.startswith("@"):
        path = Path(raw[1:]).expanduser().resolve()
        return path.read_text(encoding="utf-8")
    return raw


def _resolve_guard_api_key(raw_api_key: Optional[str], api_url: Optional[str]) -> Optional[str]:
    if raw_api_key:
        return raw_api_key
    if api_url:
        return None
    return os.environ.get("ORBITALS_API_KEY")


@cli.callback()
def _callback():
    """simple-chatbot — simple agentic RAG bot with an OpenAI-compatible server."""


@cli.command()
def serve(
    docs_dir: Path = typer.Option(..., help="Directory containing documents to index"),
    embedding_model: str = typer.Option("openai/text-embedding-3-small", help="litellm embedding model string"),
    chat_model: str = typer.Option("openai/gpt-5.4-nano", help="litellm chat model string"),
    chat_api_base: Optional[str] = typer.Option(None, help="API base URL for the chat model (e.g. vLLM: http://localhost:8000/v1)"),
    embedding_api_base: Optional[str] = typer.Option(None, help="API base URL for the embedding model"),
    chunk_size: int = typer.Option(500, help="Characters per chunk"),
    chunk_overlap: int = typer.Option(50, help="Overlap between chunks in characters"),
    top_k: int = typer.Option(5, help="Number of documents returned per search"),
    chroma_persist_dir: Path = typer.Option(Path("./.chroma"), help="ChromaDB storage directory"),
    collection_name: str = typer.Option("simple_chatbot", help="ChromaDB collection name"),
    host: str = typer.Option("127.0.0.1", help="Server bind address"),
    port: int = typer.Option(8000, help="Server port"),
    api_key: Optional[str] = typer.Option(
        None,
        help=(
            "Require this API key on /v1/chat/completions via Authorization: Bearer "
            "or X-API-Key; falls back to SIMPLE_CHATBOT_API_KEY"
        ),
    ),
    default_fixture: Optional[str] = typer.Option(
        None,
        help=(
            "Fixture id to run on /v1/responses when the request names no known "
            "fixture (for clients that can't set the model); falls back to "
            "SIMPLE_CHATBOT_DEFAULT_FIXTURE. An explicitly named fixture still wins."
        ),
    ),
    scenario_mode: Optional[str] = typer.Option(
        None,
        help=(
            "Server-level override for how fixtures run: 'deterministic' (replay "
            "the authored script) or 'live' (model-driven via the configured chat "
            "model, ignoring the script). Unset honors each fixture's own mode; "
            "falls back to SIMPLE_CHATBOT_SCENARIO_MODE."
        ),
    ),
    misbehavior_rate: Optional[float] = typer.Option(
        None,
        help=(
            "Enable the EVIL RAG agent: probability (0.0–1.0) of injecting a "
            "misbehavior at each pipeline site. Unset disables it (production/good "
            "agent); falls back to SIMPLE_CHATBOT_MISBEHAVIOR_RATE."
        ),
    ),
    misbehavior_modes: Optional[str] = typer.Option(
        None,
        help=(
            "Comma-separated misbehavior modes: poison_retrieval, drop_retrieval, "
            "ignore_retrieval, redundant_search, malformed_search, unknown_tool, "
            "wrong_value. Falls back to SIMPLE_CHATBOT_MISBEHAVIOR_MODES."
        ),
    ),
    misbehavior_seed: int = typer.Option(
        0,
        help="Seed for reproducible misbehavior decisions; falls back to SIMPLE_CHATBOT_MISBEHAVIOR_SEED.",
    ),
    max_tool_rounds: int = typer.Option(5, help="Max agentic loop iterations per request"),
    system_prompt: Optional[str] = typer.Option(
        None,
        help="System prompt prepended to every conversation; use @/path/to/file.txt to load from disk",
    ),
    conversation_log_dir: Path = typer.Option(Path("./conversations"), help="Directory for conversation JSONL logs"),
    temperature: Optional[float] = typer.Option(None, help="Sampling temperature (0–2); None uses the model default"),
    top_p: Optional[float] = typer.Option(None, help="Nucleus sampling top-p; None uses the model default"),
    gen_top_k: Optional[int] = typer.Option(None, help="Top-k sampling for generation; None uses the model default"),
    min_p: Optional[float] = typer.Option(None, help="Min-p sampling threshold; None uses the model default"),
    presence_penalty: Optional[float] = typer.Option(None, help="Presence penalty (-2 to 2); None uses the model default"),
    frequency_penalty: Optional[float] = typer.Option(None, help="Frequency penalty (-2 to 2); None uses the model default"),
    repetition_penalty: Optional[float] = typer.Option(None, help="Repetition penalty (>0); None uses the model default"),
    reindex: bool = typer.Option(False, "--reindex", help="Force re-index even if collection already has documents"),
    log_level: str = typer.Option("INFO", help="Logging level (TRACE, DEBUG, INFO, WARNING, ERROR)"),
    log_format: Literal["pretty", "json"] = typer.Option("pretty", help="Operational log format"),
    enable_guard: bool = typer.Option(
        False,
        "--enable-guard/--no-enable-guard",
        help="Run orbitals ScopeGuard on each request before the LLM (requires scope description)",
    ),
    guard_backend: Literal["vllm", "huggingface", "api"] = typer.Option(
        "api",
        help="ScopeGuard backend: api (cloud or self-hosted HTTP), vllm, or huggingface",
    ),
    guard_model: Optional[str] = typer.Option(
        None,
        help="ScopeGuard model id (required for vllm/huggingface), e.g. scope-guard-q",
    ),
    guard_api_url: Optional[str] = typer.Option(
        None,
        help="Self-hosted ScopeGuard HTTP API base URL (api backend), e.g. http://localhost:8000",
    ),
    guard_api_key: Optional[str] = typer.Option(
        None,
        help="ScopeGuard API key; falls back to ORBITALS_API_KEY only when --guard-api-url is unset",
    ),
    guard_service_description: Optional[str] = typer.Option(
        None,
        help="AI service description for ScopeGuard; use @/path/to/file.txt to load from disk; else falls back to --system-prompt",
    ),
    guard_block_classes: str = typer.Option(
        "Out of Scope,Restricted",
        help="Comma-separated scope classes that block the request (see orbitals ScopeClass)",
    ),
    guard_refusal_message: str = typer.Option(
        "I'm sorry, I can't help with that request.",
        help="Assistant message when a blocked scope class is detected",
    ),
):
    """Start the simple-chatbot OpenAI-compatible server."""
    setup_logging(level=log_level.upper(), log_format=log_format)

    logger.info("simple-chatbot starting up")
    logger.bind(
        chat_model=chat_model,
        embedding_model=embedding_model,
        docs_dir=str(docs_dir),
        log_level=log_level.upper(),
        log_format=log_format,
    ).info("Runtime configuration")
    logger.bind(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        max_tool_rounds=max_tool_rounds,
        collection=collection_name,
        chroma_dir=str(chroma_persist_dir),
    ).info("Index configuration")
    if chat_api_base:
        logger.bind(api_base=chat_api_base).info("Chat API base override")
    if embedding_api_base:
        logger.bind(api_base=embedding_api_base).info("Embedding API base override")

    api_key_effective = api_key or os.environ.get("SIMPLE_CHATBOT_API_KEY")
    default_fixture_effective = default_fixture or os.environ.get("SIMPLE_CHATBOT_DEFAULT_FIXTURE")
    scenario_mode_effective = scenario_mode or os.environ.get("SIMPLE_CHATBOT_SCENARIO_MODE")
    misbehavior_rate_raw = misbehavior_rate
    if misbehavior_rate_raw is None:
        env_rate = os.environ.get("SIMPLE_CHATBOT_MISBEHAVIOR_RATE")
        misbehavior_rate_raw = float(env_rate) if env_rate else None
    misbehavior_modes_raw = misbehavior_modes or os.environ.get("SIMPLE_CHATBOT_MISBEHAVIOR_MODES")
    misbehavior_modes_list = (
        [m.strip() for m in misbehavior_modes_raw.split(",") if m.strip()]
        if misbehavior_modes_raw
        else []
    )
    env_seed = os.environ.get("SIMPLE_CHATBOT_MISBEHAVIOR_SEED")
    misbehavior_seed_effective = misbehavior_seed if misbehavior_seed else (int(env_seed) if env_seed else 0)
    guard_api_key_effective = _resolve_guard_api_key(guard_api_key, guard_api_url)
    guard_cfg = GuardConfig(
        enabled=enable_guard,
        backend=guard_backend,
        model=guard_model,
        api_url=guard_api_url,
        api_key=guard_api_key_effective,
        service_description=_load_text_or_file(guard_service_description),
        block_classes=_parse_guard_block_classes(guard_block_classes),
        refusal_message=guard_refusal_message,
    )

    config = SimpleChatbotConfig(
        docs_dir=docs_dir,
        embedding_model=embedding_model,
        chat_model=chat_model,
        chat_api_base=chat_api_base,
        embedding_api_base=embedding_api_base,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        chroma_persist_dir=chroma_persist_dir,
        collection_name=collection_name,
        host=host,
        port=port,
        api_key=api_key_effective,
        default_fixture=default_fixture_effective,
        scenario_mode=scenario_mode_effective,
        misbehavior_rate=misbehavior_rate_raw,
        misbehavior_modes=misbehavior_modes_list,
        misbehavior_seed=misbehavior_seed_effective,
        max_tool_rounds=max_tool_rounds,
        system_prompt=_load_text_or_file(system_prompt),
        conversation_log_dir=conversation_log_dir,
        temperature=temperature,
        top_p=top_p,
        gen_top_k=gen_top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        guard=guard_cfg,
    )

    scripted = os.environ.get("SIMPLE_CHATBOT_SCRIPTED_LLM", "").strip()
    scripted_enabled = scripted not in ("", "0", "false", "False", "no", "No")

    if scripted_enabled:
        logger.warning(
            "SIMPLE_CHATBOT_SCRIPTED_LLM is set: using scripted LLM and indexer "
            "(no embeddings, no LLM, no ChromaDB). Do not use in production."
        )
        from simple_chatbot.scripted_indexer import ScriptedIndexer
        from simple_chatbot.scripted_llm import acompletion as scripted_acompletion
        from simple_chatbot.tools import scripted_tools

        indexer = ScriptedIndexer()
        init(
            config,
            indexer,
            acompletion=scripted_acompletion,
            tools=scripted_tools(indexer),
        )
    else:
        logger.bind(docs_dir=str(docs_dir)).info("Loading documents")
        docs = load_documents(config)
        logger.bind(chunk_count=len(docs)).info("Loaded document chunks")

        indexer = Indexer(config)
        indexer.index(docs, force=reindex)

        init(config, indexer)

    logger.bind(host=host, port=port).info("Starting HTTP server")
    uvicorn.run(app, host=host, port=port, log_level="warning")
