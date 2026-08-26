"""httpx-based fetcher — download a URL and route to the right extractor.

Playwright escalation is deliberately out of scope for this pass (per
project decision — revisit once observability/budget work is settled).
Sites that require JS rendering will simply come back thin or empty;
that shows up as a low grounding pass rate and is a known limitation,
not a silent failure — it's logged at WARNING with the URL.
"""

from __future__ import annotations

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.providers.extract.html import extract_html
from app.providers.extract.pdf import extract_pdf
from app.providers.search.url_utils import is_blocked_domain

log = get_logger()

USER_AGENT = (
    "LegalResearchBot/0.1 (+https://github.com/your-org/legal-research; "
    "research tool, contact: it@lensvox.com)"
)


class FetchResult:
    __slots__ = ("text", "mime", "extraction_method", "extraction_confidence", "is_quotable")

    def __init__(
        self,
        text: str,
        mime: str,
        extraction_method: str,
        extraction_confidence: float,
        is_quotable: bool,
    ):
        self.text = text
        self.mime = mime
        self.extraction_method = extraction_method
        self.extraction_confidence = extraction_confidence
        self.is_quotable = is_quotable


async def fetch_and_extract(url: str) -> FetchResult | None:
    """Download a URL and extract text via the appropriate method.
    Returns None on hard failure (blocked domain, network error, too large)."""
    settings = get_settings()

    if is_blocked_domain(url):
        log.warning("fetch_blocked_domain", url=url)
        return None

    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=settings.fetch_timeout_sec,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        log.warning("fetch_http_error", url=url, status=e.response.status_code)
        return None
    except Exception as e:
        log.warning("fetch_network_error", url=url, error=str(e))
        return None

    content_type = resp.headers.get("content-type", "").lower()
    size_mb = len(resp.content) / (1024 * 1024)

    if size_mb > settings.max_pdf_size_mb:
        log.warning("fetch_too_large", url=url, size_mb=round(size_mb, 1))
        return None

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        text, confidence, is_quotable = extract_pdf(resp.content)
        if len(text.strip()) < 50:
            log.warning("fetch_pdf_empty", url=url)
            return None
        return FetchResult(text, "application/pdf", "pymupdf", confidence, is_quotable)

    if "text/html" in content_type or not content_type:
        html = resp.text
        text, method = extract_html(html, url)
        if len(text.strip()) < settings.min_text_length:
            log.warning(
                "fetch_thin_extraction",
                url=url,
                char_count=len(text.strip()),
                method=method,
            )
            if len(text.strip()) < 50:
                return None
        return FetchResult(text, "text/html", method, 0.85, True)

    log.warning("fetch_unsupported_content_type", url=url, content_type=content_type)
    return None
