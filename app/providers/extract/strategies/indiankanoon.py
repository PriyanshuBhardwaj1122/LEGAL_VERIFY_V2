"""Fetch judgments through IndianKanoon's authenticated /doc/ API.

The public indiankanoon.org pages rate-limit aggressively, and case law
is the source type this pipeline can least afford to lose — a run that
drops its judgments falls back to blog commentary. The API returns the
same documents without that limit, and returns them as structured JSON.

The response also carries `docsource` — the actual court name ("National
Company Law Appellate Tribunal"). Scraping the public page loses that,
which is why API-fetched judgments can be attributed to a real court
instead of guessing from the URL.

Costs a metered call per document (settings.cost_indiankanoon_doc),
which is already budgeted.
"""

from __future__ import annotations

import re
from decimal import Decimal

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.rate_limit import get_host_limiter
from app.providers.extract.fetcher import FetchFailure, FetchResult
from app.providers.extract.html import extract_html

log = get_logger()

_API_BASE = "https://api.indiankanoon.org"
_DOC_URL_RE = re.compile(r"^https?://(?:www\.)?indiankanoon\.org/doc/(?P<tid>\d+)/?", re.I)

# The API returns an HTML fragment, not a full page; trafilatura expects
# document structure and under-extracts on fragments, so wrap it.
_HTML_SHELL = "<html><body>{body}</body></html>"


class IndianKanoonDocStrategy:
    """Fetch https://indiankanoon.org/doc/<tid> via the authenticated API."""

    name = "indiankanoon_doc"

    def __init__(self) -> None:
        settings = get_settings()
        self._token = settings.indiankanoon_api_token
        self._cost = settings.cost_indiankanoon_doc
        self._timeout = float(settings.fetch_timeout_sec)

    def matches(self, url: str) -> bool:
        # Without a token the API call would just 401 — fall through to
        # the ordinary HTTP fetch rather than failing the source.
        return bool(self._token) and _DOC_URL_RE.match(url or "") is not None

    def estimate_cost(self, url: str) -> Decimal:
        return self._cost

    async def fetch(self, url: str) -> FetchResult | FetchFailure:
        m = _DOC_URL_RE.match(url)
        if m is None:
            return FetchFailure("not_applicable", detail="not an indiankanoon /doc/ URL")
        tid = m.group("tid")

        try:
            # Paced like any other host. The API tolerates more than the
            # public site, but it is metered and paid for — there is no
            # reason to burst against it.
            async with get_host_limiter().slot(_API_BASE):
                async with httpx.AsyncClient(
                    base_url=_API_BASE,
                    headers={
                        "Authorization": f"Token {self._token}",
                        "Accept": "application/json",
                    },
                    timeout=self._timeout,
                ) as client:
                    resp = await client.post(f"/doc/{tid}/")
                    resp.raise_for_status()
                    payload = resp.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            log.warning("indiankanoon_doc_http_error", tid=tid, status=status)
            return FetchFailure("http_error", status=status, detail=f"IK API HTTP {status}")
        except ValueError as e:
            log.warning("indiankanoon_doc_bad_json", tid=tid, error=str(e))
            return FetchFailure("bad_json", detail=str(e)[:200])
        except Exception as e:
            log.warning("indiankanoon_doc_error", tid=tid, error=str(e))
            return FetchFailure("network_error", retryable=True, detail=str(e)[:200])

        body = payload.get("doc") or ""
        if not body.strip():
            log.warning("indiankanoon_doc_empty", tid=tid)
            return FetchFailure("empty_document", detail="API returned no doc body")

        text, _method = extract_html(_HTML_SHELL.format(body=body), url)
        if len(text.strip()) < 50:
            return FetchFailure(
                "thin_extraction", detail=f"only {len(text.strip())} chars from IK doc {tid}"
            )

        log.info(
            "indiankanoon_doc_ok",
            tid=tid,
            chars=len(text),
            court=payload.get("docsource"),
        )
        return FetchResult(
            text=text,
            mime="text/html",
            extraction_method="provider_raw",
            # Structured API output beats scraping a rate-limited page.
            extraction_confidence=0.95,
            is_quotable=True,
        )
