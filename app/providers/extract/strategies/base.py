"""Per-domain fetch strategies.

Most URLs are fetched by simply GETting them. A few sources have an
official API that returns the same document more reliably, more cheaply
in wall-clock terms, and without tripping the public site's rate limits.
Rather than special-casing those hosts inside `fetch_and_extract`, each
one is a strategy that declares which URLs it handles.

That keeps the core fetch path ignorant of any particular jurisdiction:
adding CourtListener (US) or BAILII (UK) later means adding a file and
registering it, not editing the fetcher.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol, runtime_checkable

from app.providers.extract.fetcher import FetchFailure, FetchResult


@runtime_checkable
class FetchStrategy(Protocol):
    """A way of retrieving one document."""

    name: str

    def matches(self, url: str) -> bool:
        """True when this strategy can handle the URL."""
        ...

    async def fetch(self, url: str) -> FetchResult | FetchFailure:
        """Retrieve and extract, or explain why it couldn't."""
        ...

    def estimate_cost(self, url: str) -> Decimal:
        """Metered cost of one call, for budget reservation."""
        ...
