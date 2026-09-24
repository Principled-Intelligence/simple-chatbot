import asyncio
import unittest
from pathlib import Path

from pydantic import ValidationError

from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.ratelimit import AsyncRateLimiter, build_rate_limiter


class RateLimiterConfigTests(unittest.TestCase):
    def test_off_by_default(self):
        config = SimpleChatbotConfig(docs_dir=Path("docs"))

        self.assertIsNone(config.rpm_limit)
        self.assertIsNone(build_rate_limiter(config.rpm_limit))

    def test_enabled_when_set(self):
        config = SimpleChatbotConfig(docs_dir=Path("docs"), rpm_limit=10)

        limiter = build_rate_limiter(config.rpm_limit)
        self.assertIsNotNone(limiter)
        self.assertEqual(limiter.rpm, 10)

    def test_rpm_limit_must_be_positive(self):
        with self.assertRaisesRegex(ValidationError, "rpm_limit must be greater than 0"):
            SimpleChatbotConfig(docs_dir=Path("docs"), rpm_limit=0)


class AsyncRateLimiterTests(unittest.TestCase):
    def test_allows_burst_up_to_rpm_without_waiting(self):
        async def scenario():
            limiter = AsyncRateLimiter(rpm=5, window_s=60.0)
            return [await limiter.acquire() for _ in range(5)]

        waits = asyncio.run(scenario())

        self.assertEqual(waits, [0.0] * 5, "first `rpm` acquisitions must not block")

    def test_blocks_once_the_window_is_full(self):
        # Short window keeps the test fast; the boundary logic is window-relative.
        async def scenario():
            limiter = AsyncRateLimiter(rpm=2, window_s=0.2)
            for _ in range(2):
                await limiter.acquire()
            loop = asyncio.get_running_loop()
            started = loop.time()
            waited = await limiter.acquire()
            return waited, loop.time() - started

        waited, elapsed = asyncio.run(scenario())

        self.assertGreater(waited, 0.0, "the 3rd call in a 2-per-window limiter must wait")
        self.assertGreaterEqual(elapsed, 0.1)

    def test_concurrent_callers_are_capped_per_window(self):
        """The cap is what the provider counts, so it must hold across tasks —
        not just for sequential calls on one task."""

        async def scenario():
            limiter = AsyncRateLimiter(rpm=3, window_s=10.0)
            immediate = 0
            # 3 slots for 6 concurrent callers: exactly 3 may pass right away.
            results = await asyncio.gather(
                *(_try_acquire(limiter) for _ in range(6))
            )
            immediate = sum(1 for granted in results if granted)
            return immediate

        self.assertEqual(asyncio.run(scenario()), 3)

    def test_slot_frees_after_window_elapses(self):
        async def scenario():
            limiter = AsyncRateLimiter(rpm=1, window_s=0.1)
            await limiter.acquire()
            await asyncio.sleep(0.15)  # let the single hit age out
            return await limiter.acquire()

        self.assertEqual(asyncio.run(scenario()), 0.0)


async def _try_acquire(limiter: AsyncRateLimiter) -> bool:
    """True if the slot was granted without blocking, False if it had to wait."""
    try:
        await asyncio.wait_for(limiter.acquire(), timeout=0.05)
    except asyncio.TimeoutError:
        return False
    return True


if __name__ == "__main__":
    unittest.main()
