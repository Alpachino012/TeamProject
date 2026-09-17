"""In-process token-bucket limiter for outbound calls."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable


class AsyncTokenBucket:
    """Bound outbound request bursts while allowing a steady refill rate."""

    def __init__(
        self,
        capacity: int,
        refill_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1 or refill_per_second <= 0:
            raise ValueError("capacity and refill rate must be positive")
        self._capacity = float(capacity)
        self._refill_per_second = refill_per_second
        self._tokens = float(capacity)
        self._updated_at = clock()
        self._clock = clock
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until one request token is available, then consume it."""

        while True:
            wait_seconds = 0.0
            async with self._lock:
                now = self._clock()
                elapsed = max(0.0, now - self._updated_at)
                self._tokens = min(
                    self._capacity,
                    self._tokens + elapsed * self._refill_per_second,
                )
                self._updated_at = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_seconds = (1.0 - self._tokens) / self._refill_per_second
            await asyncio.sleep(wait_seconds)