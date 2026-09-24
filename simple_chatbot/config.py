from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, model_validator


class GuardConfig(BaseModel):
    enabled: bool = False
    backend: Literal["vllm", "huggingface", "api"] = "api"
    model: str | None = None  # e.g. "scope-guard-q" / "scope-guard-g"
    api_url: str | None = None  # self-hosted ScopeGuard HTTP API
    api_key: str | None = None  # explicit key, or env ORBITALS_API_KEY for cloud API
    service_description: str | None = None  # falls back to system_prompt when unset
    block_classes: list[str] = ["Out of Scope", "Restricted"]
    refusal_message: str = "I'm sorry, I can't help with that request."


class SimpleChatbotConfig(BaseModel):
    docs_dir: Path

    embedding_model: str = "openai/text-embedding-3-small"
    chat_model: str = "openai/gpt-5.4-nano"

    # API base overrides — required for vLLM and other self-hosted endpoints.
    # e.g. "http://localhost:8000/v1"
    chat_api_base: str | None = None
    embedding_api_base: str | None = None

    chunk_size: int = 500
    chunk_overlap: int = 50

    top_k: int = 5

    # Vector-space metric for the Chroma collection ("hnsw:space"). Baked into the
    # collection at creation time, so it is part of the index fingerprint —
    # changing it drops and rebuilds the collection (see Indexer._config_fingerprint).
    distance_metric: Literal["cosine", "l2", "ip"] = "cosine"

    # Minimum similarity a retrieved chunk must reach to be handed to the model.
    # None = off (return all top_k hits, the historical behaviour). Applied after
    # retrieval, so the effective result count is <= top_k. Only meaningful for
    # bounded metrics: see _validate_config for the l2 rejection.
    similarity_threshold: float | None = None

    chroma_persist_dir: Path = Path("./.chroma")
    collection_name: str = "simple_chatbot"

    conversation_log_dir: Path = Path("./conversations")

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None

    max_tool_rounds: int = 3
    system_prompt: str | None = None

    # Sampling parameters forwarded to litellm — all default None (use model defaults).
    temperature: float | None = None
    top_p: float | None = None
    gen_top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    # OpenAI reasoning models (gpt-5.x) reject function `tools` on /v1/chat/completions
    # unless reasoning is explicitly disabled; set "none" for those. None = model default.
    reasoning_effort: str | None = None

    # Route the downstream model call through the OpenAI Responses API
    # (litellm.aresponses) instead of chat completions. Needed for gpt-5.6 reasoning
    # models, which reject function tools + reasoning on /v1/chat/completions but accept
    # the combination on /v1/responses. Leaves reasoning_effort unset -> model default.
    use_responses_api: bool = False

    # Cap outbound calls to `chat_model` at this many requests per rolling minute,
    # shared across every concurrent request in the process. None = off (no
    # throttling, the historical behaviour) — enable it only for a model whose
    # provider enforces a low per-account ceiling, e.g. OpenRouter's 10 req/min
    # `new-account-rpm` cap on newly launched models, where exceeding it returns
    # 429s instead of just queueing. See simple_chatbot/ratelimit.py.
    rpm_limit: int | None = None

    # Provider-routing preferences forwarded verbatim to an OpenAI-compatible
    # gateway as the request body's `provider` object (via litellm `extra_body`).
    # Currently only OpenRouter reads it. None = off (the gateway's own default
    # routing). Set it when the *serving stack* must be pinned, not just the model:
    # an open-weight model on OpenRouter is offered by many hosts at different
    # quantizations, and default routing picks per request, so a benchmark number
    # would otherwise average over fp8 and bf16 backends that drift over time.
    # e.g. {"quantizations": ["bf16"]} or {"only": ["akashml"]}.
    provider_routing: dict[str, Any] | None = None

    guard: GuardConfig = GuardConfig()

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        if self.rpm_limit is not None and self.rpm_limit <= 0:
            raise ValueError(
                "rpm_limit must be greater than 0; omit it (None) to disable throttling."
            )

        if self.provider_routing is not None and not self.provider_routing:
            raise ValueError(
                "provider_routing must be a non-empty object; omit it (None) to use "
                "the gateway's default routing."
            )

        if self.similarity_threshold is not None:
            if not 0.0 <= self.similarity_threshold <= 1.0:
                raise ValueError("similarity_threshold must be between 0.0 and 1.0.")
            # Chroma's "l2" space returns a squared euclidean distance, which is
            # unbounded above — there is no [0,1] similarity to compare against.
            # Reject the combination rather than inventing a scale.
            if self.distance_metric == "l2":
                raise ValueError(
                    "similarity_threshold is not supported with distance_metric='l2' "
                    "(L2 distance is unbounded, so there is no 0-1 similarity). "
                    "Use distance_metric='cosine' (or 'ip') to filter by similarity."
                )

        if not self.guard.enabled:
            return self
        if not (self.guard.service_description or self.system_prompt):
            raise ValueError(
                "When guard is enabled, set --guard-service-description or --system-prompt "
                "so the assistant scope can be evaluated."
            )
        if self.guard.backend in ("vllm", "huggingface"):
            if not self.guard.model:
                raise ValueError(
                    f"When guard backend is {self.guard.backend!r}, --guard-model is required."
                )
        if self.guard.backend == "api":
            if not (self.guard.api_url or self.guard.api_key):
                raise ValueError(
                    "When guard backend is 'api', set --guard-api-url for self-hosted "
                    "or --guard-api-key / ORBITALS_API_KEY for the hosted cloud API."
                )
        return self
