"""Synthetic verification for the two new real search providers
(IndianKanoonProvider, SerpApiProvider) — mocked HTTP responses, no
real API calls / no cost. Covers: happy-path response parsing into
RawResult, HTML-tag stripping (Indian Kanoon's headline field), the
SerpAPI 200-with-error-body case (which must raise, not silently
return zero results), and registry wiring (both register only when
their key is present, and get slotted into the routing table).

Run: PYTHONPATH=. python tests/test_search_providers_synthetic.py
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.common import Jurisdiction, QueryIntent, SourceType
from app.schemas.plan import SubQuery

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _query(intent=QueryIntent.CASE_LAW) -> SubQuery:
    return SubQuery(
        query_id="q1",
        query_text="section 29A eligibility",
        intent=intent,
        rationale="test",
        target_source_types=[SourceType.JUDGMENT],
        jurisdiction=Jurisdiction.IN,
        providers=["indiankanoon"],
    )


async def test_indiankanoon_parses_docs_and_strips_html():
    print("IndianKanoonProvider.search() — parses docs, strips HTML from title/headline")
    from app.providers.search.indiankanoon import IndianKanoonProvider

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "docs": [
            {
                "tid": 12345,
                "title": "Swiss Ribbons Pvt <b>Ltd</b> vs Union Of India",
                "headline": "...eligibility under <b>Section 29A</b> of the Code...",
                "docsource": "Supreme Court of India",
                "publishdate": "2019-01-25",
            }
        ],
        "found": "1",
    }

    with patch("app.providers.search.indiankanoon.get_settings") as mock_settings:
        mock_settings.return_value.indiankanoon_api_token = "fake-token"
        mock_settings.return_value.cost_indiankanoon_search = 0.5
        provider = IndianKanoonProvider()
        provider._client.post = AsyncMock(return_value=fake_response)
        results = await provider.search(_query())

    check("one result parsed", len(results) == 1, results)
    r = results[0]
    check("title has no HTML tags", "<b>" not in r.title and "Ltd" in r.title, r.title)
    check("headline has no HTML tags", "<b>" not in r.snippet, r.snippet)
    check("url built from tid", r.url_canonical.endswith("/doc/12345"), r.url_canonical)
    check("published_at parsed", str(r.published_at) == "2019-01-25", r.published_at)
    check("venue carries docsource", r.venue == "Supreme Court of India", r.venue)


async def test_indiankanoon_http_error_raises():
    print("IndianKanoonProvider.search() — HTTP error propagates (so search_worker's retry logic sees it)")
    from app.providers.search.indiankanoon import IndianKanoonProvider
    import httpx

    fake_response = MagicMock()
    fake_response.status_code = 401
    fake_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("unauthorized", request=MagicMock(), response=fake_response)
    )

    with patch("app.providers.search.indiankanoon.get_settings") as mock_settings:
        mock_settings.return_value.indiankanoon_api_token = "bad-token"
        mock_settings.return_value.cost_indiankanoon_search = 0.5
        provider = IndianKanoonProvider()
        provider._client.post = AsyncMock(return_value=fake_response)
        raised = False
        try:
            await provider.search(_query())
        except httpx.HTTPStatusError:
            raised = True
    check("HTTPStatusError propagates rather than being swallowed", raised)


async def test_serpapi_parses_organic_results():
    print("SerpApiProvider.search() — parses organic_results into RawResult")
    from app.providers.search.serpapi import SerpApiProvider

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "organic_results": [
            {
                "position": 1,
                "title": "Section 29A of IBC — Explained",
                "link": "https://example-legal-blog.in/section-29a",
                "snippet": "An overview of eligibility criteria under Section 29A.",
            }
        ]
    }

    with patch("app.providers.search.serpapi.get_settings") as mock_settings:
        mock_settings.return_value.serpapi_api_key = "fake-key"
        mock_settings.return_value.cost_serpapi_search = 0.4
        provider = SerpApiProvider()
        provider._client.get = AsyncMock(return_value=fake_response)
        results = await provider.search(_query(intent=QueryIntent.BACKGROUND))

    check("one result parsed", len(results) == 1, results)
    check("title present", results[0].title == "Section 29A of IBC — Explained", results[0].title)
    check("provider tagged correctly", results[0].provider == "serpapi")


async def test_serpapi_200_with_error_body_raises():
    print("SerpApiProvider.search() — HTTP 200 with an 'error' body must raise, not return []")
    from app.providers.search.serpapi import SerpApiProvider

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()  # 200 OK at the transport level
    fake_response.json.return_value = {"error": "Invalid API key."}

    with patch("app.providers.search.serpapi.get_settings") as mock_settings:
        mock_settings.return_value.serpapi_api_key = "bad-key"
        mock_settings.return_value.cost_serpapi_search = 0.4
        provider = SerpApiProvider()
        provider._client.get = AsyncMock(return_value=fake_response)
        raised = False
        try:
            await provider.search(_query())
        except RuntimeError as e:
            raised = "Invalid API key" in str(e)
    check("RuntimeError raised with the API's error message", raised)


def test_registry_only_registers_providers_with_keys():
    print("ProviderRegistry — indiankanoon/serpapi register only when their key is set")
    from app.providers.search.registry import ProviderRegistry

    with patch("app.providers.search.registry.get_settings") as mock_settings:
        s = MagicMock()
        s.tavily_api_key = ""
        s.perplexity_api_key = ""
        s.indiankanoon_api_token = ""
        s.serpapi_api_key = ""
        mock_settings.return_value = s
        registry = ProviderRegistry()

    check("indiankanoon not registered without a token", "indiankanoon" not in registry._providers)
    check("serpapi not registered without a key", "serpapi" not in registry._providers)
    check("CASE_LAW resolves to nothing real without any keys", registry.resolve_providers(QueryIntent.CASE_LAW) == [])


def test_registry_routing_includes_new_providers_when_available():
    print("ProviderRegistry — routing table includes indiankanoon/serpapi once registered")
    from app.providers.search.registry import ProviderRegistry

    with patch("app.providers.search.registry.get_settings") as mock_settings, \
         patch("app.providers.search.registry.IndianKanoonProvider") as MockIK, \
         patch("app.providers.search.registry.SerpApiProvider") as MockSerp:
        s = MagicMock()
        s.tavily_api_key = ""
        s.perplexity_api_key = ""
        s.indiankanoon_api_token = "fake-token"
        s.serpapi_api_key = "fake-key"
        mock_settings.return_value = s
        MockIK.return_value.name = "indiankanoon"
        MockSerp.return_value.name = "serpapi"
        registry = ProviderRegistry()

    case_law = registry.resolve_providers(QueryIntent.CASE_LAW)
    check("indiankanoon is in CASE_LAW routing", "indiankanoon" in case_law, case_law)
    check("serpapi is in CASE_LAW routing (general fallback)", "serpapi" in case_law, case_law)
    background = registry.resolve_providers(QueryIntent.BACKGROUND)
    check("serpapi is in BACKGROUND routing", "serpapi" in background, background)


def main():
    asyncio.run(test_indiankanoon_parses_docs_and_strips_html())
    asyncio.run(test_indiankanoon_http_error_raises())
    asyncio.run(test_serpapi_parses_organic_results())
    asyncio.run(test_serpapi_200_with_error_body_raises())
    test_registry_only_registers_providers_with_keys()
    test_registry_routing_includes_new_providers_when_available()

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
