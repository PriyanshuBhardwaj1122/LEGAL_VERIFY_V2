"""Fetch-path resilience: retry classification, Retry-After, per-host limits.

This path had no test coverage at all, which is how a 429 (retryable)
came to be handled identically to a 403 (permanent) — silently dropping
court judgments from runs.

Run: PYTHONPATH=. python -m pytest tests/test_fetch_retry_synthetic.py -q
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from app.core.rate_limit import HostLimiter, host_of
from app.core.retry import (
    extract_status_code,
    is_retryable,
    retry_after_seconds,
    retry_async,
)


def _status_error(code: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.test/doc/1")
    response = httpx.Response(code, headers=headers or {}, request=request)
    return httpx.HTTPStatusError(f"HTTP {code}", request=request, response=response)


# ---------------------------------------------------------------------
# Classification — the bug that caused judgments to be dropped
# ---------------------------------------------------------------------

def test_429_is_retryable_403_is_not():
    assert is_retryable(_status_error(429)) is True
    assert is_retryable(_status_error(503)) is True
    assert is_retryable(_status_error(403)) is False
    assert is_retryable(_status_error(404)) is False


def test_timeouts_are_retryable():
    assert is_retryable(asyncio.TimeoutError()) is True


def test_status_code_extraction():
    assert extract_status_code(_status_error(429)) == 429
    assert extract_status_code(ValueError("no status here")) is None


# ---------------------------------------------------------------------
# Retry-After
# ---------------------------------------------------------------------

def test_retry_after_delta_seconds_is_honoured_and_capped():
    assert retry_after_seconds(_status_error(429, {"retry-after": "5"})) == 5.0
    # A server asking for an hour must not stall the run.
    assert retry_after_seconds(_status_error(429, {"retry-after": "3600"})) == 30.0


def test_retry_after_absent_or_junk_returns_none():
    assert retry_after_seconds(_status_error(429)) is None
    assert retry_after_seconds(_status_error(429, {"retry-after": "soon"})) is None


# ---------------------------------------------------------------------
# retry_async behaviour
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retries_then_succeeds():
    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _status_error(429, {"retry-after": "0"})
        return "ok"

    assert await retry_async(flaky, what="test", base_backoff=0.001) == "ok"
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_non_retryable_fails_on_first_attempt():
    attempts = {"n": 0}

    async def forbidden():
        attempts["n"] += 1
        raise _status_error(403)

    with pytest.raises(httpx.HTTPStatusError):
        await retry_async(forbidden, what="test", base_backoff=0.001)
    assert attempts["n"] == 1, "403 must not be retried"


@pytest.mark.asyncio
async def test_gives_up_after_max_retries():
    attempts = {"n": 0}

    async def always_429():
        attempts["n"] += 1
        raise _status_error(429, {"retry-after": "0"})

    with pytest.raises(httpx.HTTPStatusError):
        await retry_async(always_429, what="test", max_retries=3, base_backoff=0.001)
    assert attempts["n"] == 3


# ---------------------------------------------------------------------
# Per-host limiting — the actual root cause of the self-inflicted 429s
# ---------------------------------------------------------------------

def test_host_of_strips_www():
    assert host_of("https://www.indiankanoon.org/doc/1") == "indiankanoon.org"
    assert host_of("https://indiankanoon.org/doc/1") == "indiankanoon.org"
    assert host_of("not a url") == ""


@pytest.mark.asyncio
async def test_same_host_requests_are_serialised_not_parallel():
    """indiankanoon.org is capped at 1 in flight, so ten concurrent
    requests must not overlap — that overlap is what produced the 429s."""
    limiter = HostLimiter()
    in_flight = {"now": 0, "max": 0}

    async def one():
        async with limiter.slot("https://indiankanoon.org/doc/1"):
            in_flight["now"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["now"])
            await asyncio.sleep(0.01)
            in_flight["now"] -= 1

    # Four is enough to prove non-overlap; more just pays the 1/sec pace.
    await asyncio.gather(*[one() for _ in range(4)])
    assert in_flight["max"] == 1, f"expected serialised, saw {in_flight['max']} concurrent"


@pytest.mark.asyncio
async def test_different_hosts_are_not_blocked_by_each_other():
    limiter = HostLimiter()
    started = []

    async def one(host: str):
        async with limiter.slot(f"https://{host}/x"):
            started.append(host)
            await asyncio.sleep(0.01)

    t0 = time.monotonic()
    await asyncio.gather(one("a.example"), one("b.example"), one("c.example"))
    # Three different hosts should overlap, not serialise into 3x the sleep.
    assert time.monotonic() - t0 < 0.2
    assert len(started) == 3


def test_tls_failures_are_not_retried():
    """A hostname mismatch or expired cert is permanent — retrying it
    just triples the delay before recording the same failure."""
    import ssl as _ssl

    assert is_retryable(_ssl.SSLCertVerificationError("hostname mismatch")) is False

    # httpx wraps the ssl error, so the cause chain must be walked.
    wrapped = httpx.ConnectError("connection failed")
    wrapped.__cause__ = _ssl.SSLCertVerificationError(
        "certificate verify failed: Hostname mismatch"
    )
    assert is_retryable(wrapped) is False


# ---------------------------------------------------------------------
# Per-domain fetch strategies
# ---------------------------------------------------------------------

def test_indiankanoon_strategy_claims_only_doc_urls():
    from app.providers.extract.strategies import get_fetch_strategy

    claimed = [
        "https://indiankanoon.org/doc/102331880/",
        "https://www.indiankanoon.org/doc/7427609",
    ]
    for url in claimed:
        s = get_fetch_strategy(url)
        assert s is not None and s.name == "indiankanoon_doc", url

    # Search pages, other hosts and junk fall through to the plain fetch.
    for url in [
        "https://indiankanoon.org/search/?formInput=x",
        "https://indiacorplaw.in/2020/05/24/post",
        "not a url",
        "",
    ]:
        assert get_fetch_strategy(url) is None, url


@pytest.mark.asyncio
async def test_strategy_failure_falls_back_to_plain_fetch(monkeypatch):
    """An API outage must not lose the source — the public page is still
    worth trying."""
    from app.providers.extract import fetcher as fetcher_mod
    from app.providers.extract.fetcher import FetchFailure
    import app.providers.extract.strategies as strategies_mod

    class BrokenStrategy:
        name = "broken"

        def matches(self, url: str) -> bool:
            return True

        async def fetch(self, url: str):
            return FetchFailure("http_error", status=503, detail="API down")

        def estimate_cost(self, url: str):
            from decimal import Decimal

            return Decimal("0")

    monkeypatch.setattr(strategies_mod, "_STRATEGIES", [BrokenStrategy()])

    fell_through = {"yes": False}

    async def fake_retry(fn, **kwargs):
        fell_through["yes"] = True
        raise _status_error(404)

    monkeypatch.setattr(fetcher_mod, "retry_async", fake_retry)

    out = await fetcher_mod.fetch_and_extract("https://indiankanoon.org/doc/1/")
    assert fell_through["yes"], "should have attempted the plain fetch after strategy failure"
    assert isinstance(out, FetchFailure)


# ---------------------------------------------------------------------
# Per-domain access policy — narrow by design, and it must stay narrow
# ---------------------------------------------------------------------

def test_browser_ua_only_for_allowlisted_hosts():
    from app.providers.extract.fetcher import USER_AGENT, _ua_for

    # Sites that 403 the honest bot UA and serve the same public statute
    # to a browser.
    assert _ua_for("https://indiacode.nic.in/bitstream/1/a.pdf") != USER_AGENT
    # Everywhere else we keep identifying ourselves.
    for url in [
        "https://indiankanoon.org/doc/1",
        "https://indiacorplaw.in/post",
        "https://example.com/x",
    ]:
        assert _ua_for(url) == USER_AGENT, url


def test_tls_exemption_distinguishes_missing_chain_from_hostname_mismatch():
    """The exemption must cover server misconfiguration ONLY. A hostname
    mismatch is what interception looks like and must never qualify."""
    import ssl as _ssl
    from app.providers.extract.fetcher import _is_missing_chain_error

    missing_chain = _ssl.SSLCertVerificationError(
        "certificate verify failed: unable to get local issuer certificate"
    )
    assert _is_missing_chain_error(missing_chain) is True

    mismatch = _ssl.SSLCertVerificationError(
        "certificate verify failed: Hostname mismatch, certificate is not valid for 'evil.test'"
    )
    assert _is_missing_chain_error(mismatch) is False, (
        "hostname mismatch must NOT qualify for the TLS exemption"
    )

    expired = _ssl.SSLCertVerificationError("certificate verify failed: certificate has expired")
    assert _is_missing_chain_error(expired) is False


def test_tls_exemption_hosts_are_explicitly_listed():
    """Guard against the allowlist quietly widening."""
    from app.providers.extract.fetcher import _TLS_CHAIN_EXEMPT_HOSTS

    assert _TLS_CHAIN_EXEMPT_HOSTS == frozenset(
        {"egazette.gov.in", "consumeraffairs.gov.in", "indiacode.nic.in"}
    )
    for host in _TLS_CHAIN_EXEMPT_HOSTS:
        assert host.endswith((".gov.in", ".nic.in")), f"{host} is not an official domain"


# ---------------------------------------------------------------------
# Grounding must degrade, not crash
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_one_bad_candidate_does_not_destroy_the_batch():
    """A single candidate that fails schema validation used to raise out
    of grounding_node, discarding every other candidate in the loop —
    after search, fetch and extraction had all been paid for."""
    from datetime import date

    from app.graph.nodes.grounding import grounding_node
    from app.schemas.common import CourtLevel, Jurisdiction, SourceType
    from app.schemas.evidence import EvidenceCandidate
    from app.schemas.source import Source, SourceDocument, SourceScore

    doc_text = "The provision applies to every product seller. " + ("X" * 6000)
    src = Source(
        source_id="s1", run_id="r1", url_canonical="https://indiacode.nic.in/a.pdf",
        fetch_url=None, title="Act", source_type=SourceType.STATUTE,
        court_level=CourtLevel.NONE, issuing_body="India Code",
        jurisdiction=Jurisdiction.IN, decided_or_published_on=date(2019, 1, 1), citation=None,
        score=SourceScore(authority=0.9, relevance=0.9, recency=0.9, reliability=0.9,
                          composite=0.9, tier=1, reasons=[]),
        status="fetched", discovered_by=[], provenance_query_ids=[],
    )
    doc = SourceDocument(
        source_id="s1", content_hash="h", mime="application/pdf", char_count=len(doc_text),
        extraction_method="pymupdf", extraction_confidence=0.95, is_quotable=True, text=doc_text,
    )
    candidates = [
        # Over the verbatim_quote ceiling — must be dropped, not raised.
        EvidenceCandidate(source_id="s1", kind="statutory_text", statement="too long",
                          verbatim_quote="X" * 5000, supports_issues=["1"], llm_confidence=0.9),
        # Perfectly good — must survive.
        EvidenceCandidate(source_id="s1", kind="statutory_text", statement="fine",
                          verbatim_quote="The provision applies to every product seller.",
                          supports_issues=["1"], llm_confidence=0.9),
    ]

    out = await grounding_node(candidates, {"s1": src}, {"s1": doc}, "r1", Jurisdiction.IN, 0)
    kept = out["evidence"]
    assert len(kept) == 1, "the valid candidate must survive its malformed sibling"
    assert kept[0].statement == "fine"


def test_verbatim_quote_ceiling_fits_statutory_text():
    """Statutory sub-sections are quoted whole and run far longer than a
    pin-cited sentence from a judgment."""
    from app.schemas.evidence import Evidence

    ceiling = Evidence.model_fields["verbatim_quote"].metadata[0].max_length
    assert ceiling >= 4000, f"too tight for legislation: {ceiling}"
