"""ScopeGuard integration: pre-LLM scope classification via orbitals."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger
from orbitals.scope_guard import ScopeClass, ScopeGuard

from simple_chatbot.config import GuardConfig


@dataclass
class GuardDecision:
    allowed: bool
    scope_class: str
    evidences: list[str]


class ScopeGuardGate:
    """Wraps `orbitals.scope_guard.ScopeGuard` and applies allow/block policy."""

    def __init__(self, cfg: GuardConfig, fallback_description: str | None) -> None:
        self._cfg = cfg
        self._description = (cfg.service_description or fallback_description or "").strip()
        if not self._description:
            raise ValueError("ScopeGuard requires a non-empty ai_service_description (guard or system prompt).")
        self._block = {c.strip() for c in cfg.block_classes if c.strip()}
        self._refusal = cfg.refusal_message
        kwargs: dict = {"backend": cfg.backend}
        if cfg.backend in ("vllm", "huggingface"):
            kwargs["model"] = cfg.model
        else:
            if cfg.api_url:
                kwargs["api_url"] = cfg.api_url
                if cfg.api_key:
                    kwargs["api_key"] = cfg.api_key
            elif cfg.api_key:
                kwargs["api_url"] = "https://api.orbitals.principled.app"
                kwargs["api_key"] = cfg.api_key
        self._sg = ScopeGuard(**kwargs)
        logger.bind(
            backend=cfg.backend,
            model=cfg.model,
            api_url=cfg.api_url,
        ).info("ScopeGuardGate initialised")

    @property
    def refusal_message(self) -> str:
        return self._refusal

    def describe_source(self) -> str:
        """Whether the service description came from guard-specific config or system prompt."""
        if self._cfg.service_description:
            return "guard-service-description"
        return "system-prompt"

    async def check(self, messages: list[dict]) -> GuardDecision:
        def _validate() -> GuardDecision:
            out = self._sg.validate(messages, ai_service_description=self._description)
            scope_value = (
                out.scope_class.value
                if isinstance(out.scope_class, ScopeClass)
                else str(out.scope_class)
            )
            evidences = list(out.evidences or [])
            allowed = scope_value not in self._block
            return GuardDecision(
                allowed=allowed,
                scope_class=scope_value,
                evidences=evidences,
            )

        return await asyncio.to_thread(_validate)

    def refusal_text(self, decision: GuardDecision) -> str:
        """User-facing refusal; policy text stays in logs only."""
        return self._refusal
