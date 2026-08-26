"""SearchProvider protocol and shared utilities."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Protocol, runtime_checkable

from app.schemas.plan import SubQuery
from app.schemas.source import RawResult


@runtime_checkable
class SearchProvider(Protocol):
    """Every search provider implements this interface."""

    name: str

    async def search(self, q: SubQuery, *, limit: int = 10) -> list[RawResult]:
        """Execute a search query and return raw results."""
        ...

    def estimate_cost(self, q: SubQuery, *, limit: int = 10) -> Decimal:
        """Estimate the cost of this search in INR before executing."""
        ...


class AsyncRateLimiter:
    """Token-bucket rate limiter for providers with per-second caps
    (e.g. Semantic Scholar at 1 req/sec)."""

    def __init__(self, rate_per_sec: float):
        self._interval = 1.0 / rate_per_sec
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._last_call + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = asyncio.get_event_loop().time()
