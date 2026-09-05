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
from app.core.rate_limit import get_host_limiter, host_of
from app.core.retry import is_retryable, retry_async
from app.providers.extract.html import extract_html
from app.providers.extract.pdf import extract_pdf
from app.providers.search.url_utils import is_blocked_domain

log = get_logger()

USER_AGENT = (
    "LegalResearchBot/0.1 (+https://github.com/your-org/legal-research; "
    "research tool, contact: it@lensvox.com)"
)

# Some official sources refuse the honest bot UA outright. We keep
# identifying ourselves everywhere else; this list is the narrow set of
# primary-law sites that 403 a declared bot and serve the same public
# document to a browser. Verified per-domain, not assumed.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_BROWSER_UA_HOSTS = frozenset({"indiacode.nic.in", "upload.indiacode.nic.in"})

# Government sites that serve a valid certificate but omit the
# intermediate, so the chain can't be built locally. Relaxing
# verification is only ever attempted for these named hosts, and only
# after a chain error — never on a hostname mismatch, which is what an
# interception attempt looks like.
_TLS_CHAIN_EXEMPT_HOSTS = frozenset(
    {"egazette.gov.in", "consumeraffairs.gov.in", "indiacode.nic.in"}
)

# Distinguishes "server didn't send its intermediate" (a misconfiguration)
# from "this certificate is for someone else" (a red flag).
_CHAIN_ERROR_MARKERS = (
    "unable to get local issuer certificate",
    "self signed certificate in certificate chain",
)


def _ua_for(url: str) -> str:
    return _BROWSER_UA if host_of(url) in _BROWSER_UA_HOSTS else USER_AGENT


def _is_missing_chain_error(e: BaseException | None, depth: int = 0) -> bool:
    while e is not None and depth < 5:
        text = str(e).lower()
        if any(marker in text for marker in _CHAIN_ERROR_MARKERS):
            return True
        e = e.__cause__ or e.__context__
        depth += 1
    return False


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


class FetchFailure:
    """Why a fetch failed, and whether trying later could succeed.

    Without this the caller only saw `None` and recorded every failure as
    permanent — so a site that merely throttled us looked identical to one
    that no longer exists, and the repair loop had no way to prefer
    re-attempting the former.
    """

    __slots__ = ("reason", "status", "retryable", "detail")

    def __init__(
        self, reason: str, status: int | None = None, retryable: bool = False, detail: str = ""
    ):
        self.reason = reason
        self.status = status
        self.retryable = retryable
        self.detail = detail


async def fetch_and_extract(url: str) -> FetchResult | FetchFailure:
    """Download a URL and extract text via the appropriate method.
    Returns a FetchResult, or a FetchFailure describing why it failed and
    whether a later attempt could succeed.

    Retries 429/5xx/timeouts with backoff and honours Retry-After; a 403
    or 404 still fails on the first attempt, because retrying those just
    repeats the same refusal. Requests are paced per host so we don't
    cause the rate limiting we're retrying against.
    """
    settings = get_settings()

    if is_blocked_domain(url):
        log.warning("fetch_blocked_domain", url=url)
        return FetchFailure("blocked_domain", detail="domain is on the blocklist")

    # Some hosts have an official API that serves the same document more
    # reliably than scraping the rate-limited public page. Import here
    # rather than at module scope: strategies import FetchResult from
    # this module, so a top-level import would be circular.
    from app.providers.extract.strategies import get_fetch_strategy

    strategy = get_fetch_strategy(url)
    if strategy is not None:
        outcome = await strategy.fetch(url)
        if isinstance(outcome, FetchResult):
            return outcome
        # The API declined. Fall through to the ordinary fetch rather
        # than losing the source — the public page may still serve it.
        log.warning(
            "fetch_strategy_fallback",
            strategy=strategy.name,
            url=url,
            reason=outcome.reason,
        )

    limiter = get_host_limiter()

    async def _get(verify: bool = True) -> httpx.Response:
        async with limiter.slot(url):
            async with httpx.AsyncClient(
                headers={"User-Agent": _ua_for(url)},
                follow_redirects=True,
                timeout=settings.fetch_timeout_sec,
                verify=verify,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp

    async def _get_allowing_chain_exemption() -> httpx.Response:
        try:
            return await retry_async(_get, what=f"fetch {host_of(url) or url}")
        except Exception as e:
            # One narrowly-scoped second chance: a named government host
            # whose server omitted its intermediate certificate. Never for
            # any other host, and never for a hostname mismatch — that
            # failure is what interception looks like, and retrying it
            # unverified would defeat the point of checking at all.
            if not (
                host_of(url) in _TLS_CHAIN_EXEMPT_HOSTS and _is_missing_chain_error(e)
            ):
                raise
            log.warning(
                "fetch_tls_chain_exemption",
                url=url,
                host=host_of(url),
                detail="server omitted intermediate cert; retrying without verification",
            )
            return await retry_async(
                lambda: _get(verify=False), what=f"fetch(no-tls) {host_of(url)}"
            )

    try:
        resp = await _get_allowing_chain_exemption()
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        retryable = is_retryable(e)
        log.warning("fetch_http_error", url=url, status=status, retryable=retryable)
        return FetchFailure(
            "http_error", status=status, retryable=retryable, detail=f"HTTP {status}"
        )
    except Exception as e:
        retryable = is_retryable(e)
        log.warning("fetch_network_error", url=url, error=str(e), retryable=retryable)
        return FetchFailure("network_error", retryable=retryable, detail=str(e)[:200])

    content_type = resp.headers.get("content-type", "").lower()
    size_mb = len(resp.content) / (1024 * 1024)

    if size_mb > settings.max_pdf_size_mb:
        log.warning("fetch_too_large", url=url, size_mb=round(size_mb, 1))
        return FetchFailure("too_large", detail=f"{size_mb:.1f} MB")

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        text, confidence, is_quotable = extract_pdf(resp.content)
        if len(text.strip()) < 50:
            log.warning("fetch_pdf_empty", url=url)
            return FetchFailure("pdf_empty", detail="PDF yielded no extractable text")
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
                return FetchFailure(
                    "thin_extraction", detail=f"only {len(text.strip())} chars extracted"
                )
        return FetchResult(text, "text/html", method, 0.85, True)

    log.warning("fetch_unsupported_content_type", url=url, content_type=content_type)
    return FetchFailure("unsupported_content_type", detail=content_type)
