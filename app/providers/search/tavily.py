"""Tavily search provider."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from decimal import Decimal

from tavily import AsyncTavilyClient

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.common import QueryIntent
from app.schemas.plan import SubQuery
from app.schemas.source import RawResult

from .url_utils import canonicalize_url

log = get_logger()


# Map QueryIntent to Tavily-specific params
_INTENT_CONFIG: dict[str, dict] = {
    QueryIntent.STATUTE_TEXT: {"search_depth": "basic", "topic": "general"},
    QueryIntent.CASE_LAW: {"search_depth": "advanced", "topic": "general"},
    QueryIntent.RECENT_DEVELOPMENT: {"search_depth": "advanced", "topic": "news"},
    QueryIntent.REGULATORY_ACTION: {"search_depth": "advanced", "topic": "general"},
    QueryIntent.ACADEMIC_COMMENTARY: {"search_depth": "basic", "topic": "general"},
    QueryIntent.COUNTER_VIEW: {"search_depth": "advanced", "topic": "general"},
    QueryIntent.COMPARATIVE: {"search_depth": "basic", "topic": "general"},
    QueryIntent.BACKGROUND: {"search_depth": "basic", "topic": "general"},
    QueryIntent.RELATED_PRECEDENT: {"search_depth": "advanced", "topic": "general"},
}

# Domain filters by intent
_DOMAIN_FILTERS: dict[str, list[str]] = {
    QueryIntent.STATUTE_TEXT: ["indiacode.nic.in", "egazette.gov.in"],
    QueryIntent.REGULATORY_ACTION: [
        "sebi.gov.in", "rbi.org.in", "mca.gov.in", "cci.gov.in",
        "irdai.gov.in", "trai.gov.in",
    ],
}


class TavilyProvider:
    name = "tavily"

    def __init__(self):
        settings = get_settings()
        self.client = AsyncTavilyClient(api_key=settings.tavily_api_key)
        self._cost_per_search = settings.cost_tavily_search

    def estimate_cost(self, q: SubQuery, *, limit: int = 10) -> Decimal:
        return self._cost_per_search

    async def search(self, q: SubQuery, *, limit: int = 10) -> list[RawResult]:
        intent_cfg = _INTENT_CONFIG.get(q.intent, {"search_depth": "basic", "topic": "general"})

        kwargs: dict = {
            "query": q.query_text,
            "max_results": min(limit, 20),
            "search_depth": intent_cfg["search_depth"],
            "topic": intent_cfg["topic"],
            "include_raw_content": True,
        }

        # Add domain filters
        domains = _DOMAIN_FILTERS.get(q.intent)
        if domains:
            kwargs["include_domains"] = domains

        # Date filtering
        if q.date_from:
            kwargs["start_date"] = q.date_from.isoformat()
        if q.date_to:
            kwargs["end_date"] = q.date_to.isoformat()

        # Country filter for India — Tavily expects the full lowercase
        # country name (e.g. "india"), not an ISO code. Sending "in"
        # fails every call with "Invalid country".
        if q.jurisdiction.value in ("IN", "IN_STATE"):
            kwargs["country"] = "india"

        try:
            response = await self.client.search(**kwargs)
        except Exception as e:
            log.error("tavily_search_error", error=str(e), query=q.query_text)
            raise

        results = []
        now = datetime.now(timezone.utc)

        for rank, item in enumerate(response.get("results", []), 1):
            url = item.get("url", "")
            url_canon = canonicalize_url(url)
            raw_id = hashlib.sha1(
                f"{q.query_id}:tavily:{url_canon}".encode()
            ).hexdigest()[:16]

            # Parse published date if available
            pub_date = None
            if item.get("published_date"):
                try:
                    pub_date = datetime.fromisoformat(
                        item["published_date"].replace("Z", "+00:00")
                    ).date()
                except (ValueError, TypeError):
                    pass

            results.append(
                RawResult(
                    raw_id=raw_id,
                    run_id=q.query_id[:16],  # placeholder, overwritten by caller
                    query_id=q.query_id,
                    provider="tavily",
                    url=url,
                    url_canonical=url_canon,
                    title=item.get("title"),
                    snippet=item.get("content", "")[:500],
                    published_at=pub_date,
                    provider_rank=rank,
                    provider_score=item.get("score"),
                    fetched_at=now,
                )
            )

        log.info(
            "tavily_search_ok",
            query_id=q.query_id,
            result_count=len(results),
            intent=q.intent,
        )
        return results