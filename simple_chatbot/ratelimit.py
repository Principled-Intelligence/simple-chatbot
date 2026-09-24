"""Process-wide outbound request throttle for the downstream model call.

Some providers cap requests *per account, per model* well below what a batch
workload naturally produces — e.g. OpenRouter applies a 10 req/min
``new-account-rpm`` ceiling to freshly launched models regardless of credit
balance. Exceeding it returns 429s, and because the eval harness holds a
per-conversation deadline a throttled retry storm turns into failed executions
rather than merely slower ones.

The limiter is **opt-in** (``SimpleChatbotConfig.rpm_limit`` unset -> no
throttling, no overhead) because the ceiling is provider- and model-specific;
every other model would only be slowed down by it.

Rolling window, not a fixed bucket: providers that publish an RPM figure
generally enforce it over a trailing 60s, so a fixed-bucket limiter would allow
a 2x burst across a boundary and still earn 429s.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

from loguru import logger

#: Trailing window the limit is enforced over. Providers quote RPM, so 60s.
WINDOW_SECONDS = 60.0

#: Floor on the re-check sleep so a herd of waiters cannot spin on the lock.
_MIN_SLEEP_SECONDS = 0.01


class AsyncRateLimiter:
    """Allow at most ``rpm`` acquisitions per trailing ``window_s`` seconds.

    Shared by every in-flight request in the process, which is what the provider
    counts: the chatbot serves one ``chat_model`` at a time, so process-wide and
    per-model are the same bound here.

    Fair-ish but not FIFO: waiters re-contend for the lock after sleeping, so
    ordering under contention is arbitrary. Throughput is what matters for a
    batch workload, and every waiter makes progress as slots free up.
    """

    def __init__(self, rpm: int, window_s: float = WINDOW_SECONDS) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be greater than 0")
        self._rpm = rpm
        self._window = window_s
        # Monotonic timestamps of the acquisitions still inside the window.
        self._hits: deque[float] = deque()
        self._lock = asyncio.Lock()

    @property
    def rpm(self) -> int:
        return self._rpm

    async def acquire(self) -> float:
        """Block until a slot is free. Returns seconds spent waiting (0 if none)."""
        waited = 0.0
        while True:
            async with self._lock:
                now = time.monotonic()
                # Drop acquisitions that have aged out of the trailing window.
                while self._hits and now - self._hits[0] >= self._window:
                    self._hits.popleft()
                if len(self._hits) < self._rpm:
                    self._hits.append(now)
                    return waited
                # Full: the oldest hit has to age out before a slot exists.
                sleep_for = max(self._window - (now - self._hits[0]), _MIN_SLEEP_SECONDS)
            # Sleep OUTSIDE the lock, otherwise no other waiter could ever
            # observe a freed slot and the limiter would serialize into a stall.
            await asyncio.sleep(sleep_for)
            waited += sleep_for

    async def __aenter__(self) -> float:
        return await self.acquire()

    async def __aexit__(self, *_exc: object) -> None:
        # Nothing to release: the window ages out on its own. Slots are consumed
        # at acquire time, so a long provider call does not hold one open.
        return None


def build_rate_limiter(rpm_limit: int | None) -> AsyncRateLimiter | None:
    """``None`` (the default) means no throttling — return no limiter at all so
    the call path stays exactly as it was for every unthrottled model."""
    if not rpm_limit:
        return None
    limiter = AsyncRateLimiter(rpm_limit)
    logger.bind(rpm_limit=rpm_limit, window_s=WINDOW_SECONDS).info(
        "Outbound model-call rate limiting enabled"
    )
    return limiter
