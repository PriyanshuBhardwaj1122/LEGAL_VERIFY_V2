"""Indian Kanoon search provider — real case-law search API.

This is the primary provider for QueryIntent.CASE_LAW and
QueryIntent.RELATED_PRECEDENT per the routing table in registry.py. It
was a stub returning [] until now, which is very likely why every real
run so far has come back with near-zero actual case law: general web
search (Tavily/Perplexity) surfaces commentary and circulars fine, but
doesn't reliably surface judgments the way a dedicated case-law index
does.

API docs: https://api.indiankanoon.org (token-based auth, POST /search/).
This provider only calls the search endpoint — /doc/ and /docmeta/
(full document fetch) are a fetch-layer concern, not search, and aren't
wired here.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timezone
from decimal import Decimal

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.plan import SubQuery
from app.schemas.source import RawResult

from .url_utils import canonicalize_url

log = get_logger()

_API_BASE = "https://api.indiankanoon.org"
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(s: str | None) -> str | None:
    if not s:
        return None
    return _TAG_RE.sub("", s).strip() or None


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except (ValueError, TypeError):
            continue
    return None


class IndianKanoonProvider:
    name = "indiankanoon"

    def __init__(self):
        settings = get_settings()
        self._token = settings.indiankanoon_api_token
        self._cost_per_search = settings.cost_indiankanoon_search
        self._client = httpx.AsyncClient(
            base_url=_API_BASE,
            headers={
                "Authorization": f"Token {self._token}",
                "Accept": "application/json",
            },
            timeout=20.0,
        )

    def estimate_cost(self, q: SubQuery, *, limit: int = 10) -> Decimal:
        return self._cost_per_search

    async def search(self, q: SubQuery, *, limit: int = 10) -> list[RawResult]:
        params = {"formInput": q.query_text, "pagenum": 0}

        try:
            response = await self._client.post("/search/", data=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.error(
                "indiankanoon_search_http_error",
                status=e.response.status_code,
                query=q.query_text,
            )
            raise
        except Exception as e:
            log.error("indiankanoon_search_error", error=str(e), query=q.query_text)
            raise

        try:
            payload = response.json()
        except ValueError as e:
            log.error("indiankanoon_search_bad_json", error=str(e), body_preview=response.text[:200])
            raise

        docs = payload.get("docs", [])
        results: list[RawResult] = []
        now = datetime.now(timezone.utc)

        for rank, doc in enumerate(docs[:limit], 1):
            tid = doc.get("tid")
            if tid is None:
                continue
            url = f"https://indiankanoon.org/doc/{tid}/"
            url_canon = canonicalize_url(url)
            raw_id = hashlib.sha1(
                f"{q.query_id}:indiankanoon:{url_canon}".encode()
            ).hexdigest()[:16]

            results.append(
                RawResult(
                    raw_id=raw_id,
                    run_id=q.query_id[:16],  # placeholder, overwritten by caller
                    query_id=q.query_id,
                    provider="indiankanoon",
                    url=url,
                    url_canonical=url_canon,
                    title=_strip_tags(doc.get("title")),
                    snippet=_strip_tags(doc.get("headline"))[:500] if doc.get("headline") else None,
                    published_at=_parse_date(doc.get("publishdate")),
                    provider_rank=rank,
                    provider_score=None,
                    venue=doc.get("docsource"),
                    fetched_at=now,
                )
            )

        log.info(
            "indiankanoon_search_ok",
            query_id=q.query_id,
            result_count=len(results),
            found=payload.get("found"),
        )
        return results
