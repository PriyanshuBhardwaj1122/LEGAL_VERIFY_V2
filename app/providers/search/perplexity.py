"""Perplexity Sonar search provider (OpenAI-compatible API)."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from decimal import Decimal

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.common import QueryIntent
from app.schemas.plan import SubQuery
from app.schemas.source import RawResult

from .url_utils import canonicalize_url

log = get_logger()

# Perplexity recency filters
_RECENCY_MAP: dict[str, str] = {
    QueryIntent.RECENT_DEVELOPMENT: "month",
    QueryIntent.REGULATORY_ACTION: "month",
}

_DOMAIN_FILTERS: dict[str, list[str]] = {
    QueryIntent.STATUTE_TEXT: ["indiacode.nic.in", "egazette.gov.in"],
    QueryIntent.REGULATORY_ACTION: [
        "sebi.gov.in", "rbi.org.in", "mca.gov.in", "cci.gov.in",
    ],
    QueryIntent.CASE_LAW: [
        "indiankanoon.org", "sci.gov.in", "judgments.ecourts.gov.in",
    ],
}


class PerplexityProvider:
    name = "perplexity"

    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.perplexity_api_key,
            base_url="https://api.perplexity.ai",
        )
        self._cost_per_1k = settings.cost_perplexity_per_1k_tokens
        self._model = "sonar"

    def estimate_cost(self, q: SubQuery, *, limit: int = 10) -> Decimal:
        # Rough estimate: ~2k tokens per search call
        return self._cost_per_1k * 2

    async def search(self, q: SubQuery, *, limit: int = 10) -> list[RawResult]:
        system_msg = (
            "You are a legal research assistant. Return factual information "
            "with source URLs. Focus on Indian law unless asked for comparative material."
        )

        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": q.query_text},
            ],
        }

        # Add search-specific params via extra_body
        extra: dict = {}
        recency = _RECENCY_MAP.get(q.intent)
        if recency:
            extra["search_recency_filter"] = recency

        domains = _DOMAIN_FILTERS.get(q.intent)
        if domains:
            extra["search_domain_filter"] = domains

        if extra:
            kwargs["extra_body"] = extra

        start = time.monotonic()
        try:
            response = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            log.error("perplexity_search_error", error=str(e), query=q.query_text)
            raise
        latency = int((time.monotonic() - start) * 1000)

        results: list[RawResult] = []
        now = datetime.now(timezone.utc)

        # Extract citations from the response
        # Perplexity returns citations in the response object
        choice = response.choices[0] if response.choices else None
        if not choice:
            return results

        content = choice.message.content or ""

        # Citations are in response metadata
        citations: list[str] = []
        if hasattr(response, "citations") and response.citations:
            citations = response.citations
        # Also check model_extra for citations
        elif hasattr(response, "model_extra") and response.model_extra:
            citations = response.model_extra.get("citations", [])

        for rank, url in enumerate(citations, 1):
            url_canon = canonicalize_url(url)
            raw_id = hashlib.sha1(
                f"{q.query_id}:perplexity:{url_canon}".encode()
            ).hexdigest()[:16]

            results.append(
                RawResult(
                    raw_id=raw_id,
                    run_id=q.query_id[:16],  # placeholder
                    query_id=q.query_id,
                    provider="perplexity",
                    url=url,
                    url_canonical=url_canon,
                    title=None,  # Perplexity doesn't return per-citation titles
                    snippet=content[:500] if rank == 1 else None,
                    published_at=None,
                    provider_rank=rank,
                    provider_score=None,
                    fetched_at=now,
                )
            )

        log.info(
            "perplexity_search_ok",
            query_id=q.query_id,
            result_count=len(results),
            latency_ms=latency,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
        )
        return results

    def get_usage(self, response) -> dict:
        """Extract token usage for budget tracking."""
        if response.usage:
            return {
                "tokens_in": response.usage.prompt_tokens,
                "tokens_out": response.usage.completion_tokens,
            }
        return {}
