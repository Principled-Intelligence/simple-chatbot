from pathlib import Path
from typing import Literal, Self

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

    chroma_persist_dir: Path = Path("./.chroma")
    collection_name: str = "simple_chatbot"

    conversation_log_dir: Path = Path("./conversations")

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None

    max_tool_rounds: int = 3
    system_prompt: str | None = None

    # Scenario engine: server-level override of a scenario's own `mode`.
    # None = honor each scenario's per-scenario default.
    scenario_mode: Literal["deterministic", "live"] | None = None

    # Scenario engine: fixture id to run when a /v1/responses request does not
    # name a known fixture (e.g. a client that can't set the model). None = no
    # fallback (unknown/absent model uses the RAG agent path, as before).
    default_fixture: str | None = None

    # Sampling parameters forwarded to litellm — all default None (use model defaults).
    temperature: float | None = None
    top_p: float | None = None
    gen_top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None

    guard: GuardConfig = GuardConfig()

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

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
