"""SerpAPI (Google Search) provider — a third, independent general
search engine alongside Tavily and Perplexity. Registered into every
general-purpose routing tier (not the specialty CASE_LAW/STATUTE_TEXT
slots, which stay owned by IndianKanoon/indiacode) so a Tavily outage
or rate-limit doesn't stall a whole run on its own.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from decimal import Decimal

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.common import Jurisdiction
from app.schemas.plan import SubQuery
from app.schemas.source import RawResult

from .url_utils import canonicalize_url

log = get_logger()

_API_URL = "https://serpapi.com/search.json"

# Google country code ("gl") per jurisdiction — omitted entirely for
# jurisdictions with no natural mapping rather than guessing.
_GL_MAP: dict[str, str] = {
    Jurisdiction.IN: "in",
    Jurisdiction.IN_STATE: "in",
    Jurisdiction.UK: "uk",
    Jurisdiction.US: "us",
    Jurisdiction.SG: "sg",
}


class SerpApiProvider:
    name = "serpapi"

    def __init__(self):
        settings = get_settings()
        self._api_key = settings.serpapi_api_key
        self._cost_per_search = settings.cost_serpapi_search
        self._client = httpx.AsyncClient(timeout=20.0)

    def estimate_cost(self, q: SubQuery, *, limit: int = 10) -> Decimal:
        return self._cost_per_search

    async def search(self, q: SubQuery, *, limit: int = 10) -> list[RawResult]:
        params: dict = {
            "engine": "google",
            "q": q.query_text,
            "num": min(limit, 20),
            "api_key": self._api_key,
        }
        gl = _GL_MAP.get(q.jurisdiction.value)
        if gl:
            params["gl"] = gl
            params["hl"] = "en"

        try:
            response = await self._client.get(_API_URL, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.error("serpapi_search_http_error", status=e.response.status_code, query=q.query_text)
            raise
        except Exception as e:
            log.error("serpapi_search_error", error=str(e), query=q.query_text)
            raise

        payload = response.json()

        # SerpAPI can return HTTP 200 with an error body (bad key,
        # exhausted quota, etc.) — that's a terminal failure, not zero
        # results, so it must not be silently swallowed as "no matches".
        if "error" in payload:
            log.error("serpapi_search_api_error", error=payload["error"], query=q.query_text)
            raise RuntimeError(f"SerpAPI error: {payload['error']}")

        organic = payload.get("organic_results", [])
        results: list[RawResult] = []
        now = datetime.now(timezone.utc)

        for item in organic[:limit]:
            url = item.get("link")
            if not url:
                continue
            url_canon = canonicalize_url(url)
            raw_id = hashlib.sha1(
                f"{q.query_id}:serpapi:{url_canon}".encode()
            ).hexdigest()[:16]

            results.append(
                RawResult(
                    raw_id=raw_id,
                    run_id=q.query_id[:16],  # placeholder, overwritten by caller
                    query_id=q.query_id,
                    provider="serpapi",
                    url=url,
                    url_canonical=url_canon,
                    title=item.get("title"),
                    snippet=(item.get("snippet") or "")[:500] or None,
                    published_at=_parse_relative_or_iso_date(item.get("date")),
                    provider_rank=item.get("position"),
                    provider_score=None,
                    fetched_at=now,
                )
            )

        log.info(
            "serpapi_search_ok",
            query_id=q.query_id,
            result_count=len(results),
        )
        return results


def _parse_relative_or_iso_date(raw: str | None) -> date | None:
    """SerpAPI's per-result 'date' field is usually an ISO-ish string
    for news results and absent/relative ("3 days ago") for plain
    organic results. Only parse the unambiguous case; a relative or
    unparseable string safely becomes None rather than a guess."""
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except (ValueError, TypeError):
            continue
    return None
