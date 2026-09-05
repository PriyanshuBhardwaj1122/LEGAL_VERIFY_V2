"""Per-host politeness limits for outbound HTTP.

Every outbound request in the pipeline used one global concurrency cap,
which says nothing about how hard any single site is being hit. With
IndianKanoon the only case-law provider, most selected URLs share one
host — so a "safe" global limit of 6 still meant 6 simultaneous
connections to one server, with no spacing, from a self-identifying bot.
The resulting 429s were self-inflicted, and they cost the run exactly
the sources it most needed.

This bounds BOTH dimensions per host: how many requests may be in flight
at once, and how closely together they may start.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from app.core.config import get_settings
from app.providers.search.base import AsyncRateLimiter

# Hosts that need gentler treatment than the default. Keyed by hostname
# suffix so subdomains inherit. Values are (max in flight, requests/sec).
_HOST_OVERRIDES: dict[str, tuple[int, float]] = {
    "indiankanoon.org": (1, 1.0),
    "nclt.gov.in": (1, 1.0),
    "nclat.gov.in": (1, 1.0),
    "indiacode.nic.in": (1, 1.0),
}


def host_of(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


class HostLimiter:
    """Bounds concurrency and pace per host. One instance should be
    shared by every component making outbound requests in a run —
    separate instances give separate budgets, which defeats the point."""

    def __init__(self) -> None:
        settings = get_settings()
        self._default_concurrency = max(1, settings.per_host_fetch_concurrency)
        self._default_rate = max(0.1, settings.per_host_fetch_rate_per_sec)
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._limiters: dict[str, AsyncRateLimiter] = {}
        self._lock = asyncio.Lock()

    def _caps_for(self, host: str) -> tuple[int, float]:
        for suffix, caps in _HOST_OVERRIDES.items():
            if host == suffix or host.endswith("." + suffix):
                return caps
        return (self._default_concurrency, self._default_rate)

    async def _slots_for(self, host: str) -> tuple[asyncio.Semaphore, AsyncRateLimiter]:
        async with self._lock:
            if host not in self._semaphores:
                concurrency, rate = self._caps_for(host)
                self._semaphores[host] = asyncio.Semaphore(concurrency)
                self._limiters[host] = AsyncRateLimiter(rate)
            return self._semaphores[host], self._limiters[host]

    def slot(self, url: str) -> "_HostSlot":
        """`async with limiter.slot(url):` around one outbound request."""
        return _HostSlot(self, host_of(url))


class _HostSlot:
    def __init__(self, limiter: HostLimiter, host: str) -> None:
        self._limiter = limiter
        self._host = host
        self._sem: asyncio.Semaphore | None = None

    async def __aenter__(self) -> None:
        sem, rate = await self._limiter._slots_for(self._host)
        await sem.acquire()
        self._sem = sem
        # Pace inside the semaphore so waiting for a slot doesn't also
        # burn the interval — otherwise queued requests all fire at once
        # the moment a slot frees.
        await rate.acquire()

    async def __aexit__(self, *exc) -> None:
        if self._sem is not None:
            self._sem.release()
        return None


# Process-wide default. The pipeline is one run per process today, so a
# module-level instance is the shared budget every caller needs.
_DEFAULT_LIMITER: HostLimiter | None = None


def get_host_limiter() -> HostLimiter:
    global _DEFAULT_LIMITER
    if _DEFAULT_LIMITER is None:
        _DEFAULT_LIMITER = HostLimiter()
    return _DEFAULT_LIMITER
