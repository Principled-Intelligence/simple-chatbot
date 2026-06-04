# simple_chatbot/misbehavior.py
"""Seeded, substrate-agnostic misbehavior policy for the evil agent.

A `MisbehaviorPolicy` is consulted at each injection site in the pipeline. Given
the site's `eligible_modes`, it intersects them with the enabled set, draws a
seeded coin against `rate`, and on a hit picks a mode, logs an `Injection`, and
returns it. Determinism depends only on the seed and the order of `maybe()`
calls — pure, with no I/O.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Mode vocabulary, grouped by the pipeline site that can inject them.
RETRIEVAL_MODES = ("poison_retrieval", "drop_retrieval")
DECISION_STEER_MODES = ("ignore_retrieval",)
DECISION_STRUCTURAL_MODES = ("redundant_search", "malformed_search", "unknown_tool")
ANSWER_STEER_MODES = ("wrong_value",)

KNOWN_MODES = frozenset(
    RETRIEVAL_MODES + DECISION_STEER_MODES + DECISION_STRUCTURAL_MODES + ANSWER_STEER_MODES
)


@dataclass(frozen=True)
class MisbehaviorConfig:
    """rate = p of firing at each eligible site; modes = enabled subset of
    KNOWN_MODES; seed = RNG seed for reproducible decisions."""

    rate: float
    modes: tuple[str, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"misbehavior rate must be in [0.0, 1.0], got {self.rate}")
        unknown = set(self.modes) - KNOWN_MODES
        if unknown:
            raise ValueError(
                f"unknown misbehavior modes: {sorted(unknown)}; "
                f"known: {sorted(KNOWN_MODES)}"
            )


@dataclass
class Injection:
    """A recorded misbehavior. `turn`/`round` are filled from ctx when the caller
    knows them (the acompletion wrapper does; the search wrapper leaves them None)."""

    stage: str
    mode: str
    turn: int | None = None
    round: int | None = None
    detail: dict = field(default_factory=dict)


class MisbehaviorPolicy:
    def __init__(self, config: MisbehaviorConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._log: list[Injection] = []

    def maybe(self, stage: str, eligible_modes, ctx: dict | None = None) -> Injection | None:
        ctx = ctx or {}
        enabled = [m for m in eligible_modes if m in self.config.modes]
        if not enabled:
            return None
        if self._rng.random() >= self.config.rate:
            return None
        mode = self._rng.choice(enabled)
        injection = Injection(
            stage=stage,
            mode=mode,
            turn=ctx.get("turn"),
            round=ctx.get("round"),
            detail=dict(ctx.get("detail", {})),
        )
        self._log.append(injection)
        return injection

    @property
    def injections(self) -> list[Injection]:
        return list(self._log)
