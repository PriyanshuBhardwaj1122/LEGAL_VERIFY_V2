"""Registry of per-domain fetch strategies.

`get_fetch_strategy(url)` returns the first registered strategy that
claims the URL, or None when the ordinary HTTP fetch should handle it.
Register new jurisdictions here — the fetcher itself never needs to know
they exist.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.providers.extract.strategies.base import FetchStrategy
from app.providers.extract.strategies.indiankanoon import IndianKanoonDocStrategy

log = get_logger()

_STRATEGIES: list[FetchStrategy] | None = None


def _registry() -> list[FetchStrategy]:
    # Built lazily: strategies read settings at construction, and
    # importing this module must not require configuration to be loaded.
    global _STRATEGIES
    if _STRATEGIES is None:
        _STRATEGIES = [IndianKanoonDocStrategy()]
    return _STRATEGIES


def get_fetch_strategy(url: str) -> FetchStrategy | None:
    for strategy in _registry():
        try:
            if strategy.matches(url):
                return strategy
        except Exception as e:  # a broken strategy must not kill the fetch
            log.warning("fetch_strategy_match_failed", strategy=strategy.name, error=str(e))
    return None


__all__ = ["FetchStrategy", "get_fetch_strategy"]
