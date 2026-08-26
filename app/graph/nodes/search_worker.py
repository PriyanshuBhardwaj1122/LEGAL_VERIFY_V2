"""STEP 2b — search_worker node.

Executes exactly one (SubQuery, provider) pair. Wraps every provider call
with a budget reservation and retry-with-backoff. Never raises out of the
node — a failed or budget-skipped call degrades to an empty result plus a
NodeError, so the run continues rather than crashing.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from app.core.budget import BudgetGuard
from app.core.logging import ctx_node, get_logger
from app.providers.search.registry import get_registry
from app.schemas.evidence import NodeError
from app.schemas.plan import SubQuery
from app.schemas.source import RawResult

log = get_logger()

# Per-provider concurrency caps — semaphores, not the same thing as the
# Semantic Scholar rate limiter (that one is a hard 1 req/sec regardless
# of how many are "in flight").
_SEMAPHORES: dict[str, asyncio.Semaphore] = {}


def _get_semaphore(provider: str) -> asyncio.Semaphore:
    if provider not in _SEMAPHORES:
        caps = {
            "tavily": 8,
            "exa": 8,
            "perplexity": 4,
            "indiankanoon": 2,
            "indiacode": 2,
            "serpapi": 6,
            "semantic_scholar": 4,  # rate limiter enforces the real 1/sec cap
        }
        _SEMAPHORES[provider] = asyncio.Semaphore(caps.get(provider, 4))
    return _SEMAPHORES[provider]


MAX_RETRIES = 3
BASE_BACKOFF_SEC = 1.0


def _extract_status_code(e: Exception) -> int | None:
    """Best-effort status code extraction across httpx, openai, tavily,
    and other SDK exception shapes — they don't share a base class."""
    for attr in ("status_code", "status"):
        code = getattr(e, attr, None)
        if isinstance(code, int):
            return code
    response = getattr(e, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _is_retryable(e: Exception) -> bool:
    """429 and 5xx are retryable. Timeouts are retryable. Everything
    else (4xx auth/validation errors, unknown shapes) is not — retrying
    a malformed request just burns budget for the same failure."""
    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
        return True
    status = _extract_status_code(e)
    if status is None:
        # Unknown error shape — retry once defensively, callers still
        # bound this via MAX_RETRIES.
        return True
    return status == 429 or 500 <= status < 600


async def _search_with_retry(
    provider_impl,
    q: SubQuery,
    limit: int,
) -> list[RawResult]:
    """Retry on 429/5xx/timeout. 4xx (other than 429) is terminal."""
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            return await provider_impl.search(q, limit=limit)
        except Exception as e:
            last_exc = e
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                log.error(
                    "search_worker_terminal_error",
                    provider=provider_impl.name,
                    error=str(e),
                    status_code=_extract_status_code(e),
                    attempt=attempt + 1,
                )
                raise
            wait = BASE_BACKOFF_SEC * (2**attempt) + random.uniform(0, 0.5)
            log.warning(
                "search_worker_retry",
                provider=provider_impl.name,
                error=str(e),
                attempt=attempt + 1,
                wait_sec=round(wait, 2),
            )
            await asyncio.sleep(wait)

    if last_exc:
        raise last_exc
    return []


async def search_worker_node(
    payload: dict[str, Any],
    budget: BudgetGuard | None = None,
) -> dict[str, Any]:
    """Execute one (SubQuery, provider) search.

    payload keys: run_id, sub_query (SubQuery), provider (str)

    Returns a partial state update: {"raw_results": [...], "errors": [...]}
    Never raises — all failure modes degrade to empty results + NodeError.
    """
    ctx_node.set("search_worker")
    run_id: str = payload["run_id"]
    q: SubQuery = payload["sub_query"]
    provider_name: str = payload["provider"]

    registry = get_registry()
    provider_impl = registry.get_provider(provider_name)

    if provider_impl is None:
        log.warning("search_worker_no_provider", provider=provider_name, query_id=q.query_id)
        return {
            "raw_results": [],
            "errors": [
                NodeError(
                    node="search_worker",
                    kind="provider_error",
                    detail=f"Provider '{provider_name}' not registered",
                    query_id=q.query_id,
                    retryable=False,
                    occurred_at=datetime.now(timezone.utc),
                )
            ],
        }

    limit = 10
    estimated_cost = provider_impl.estimate_cost(q, limit=limit)

    # Budget check — skip gracefully rather than crash the run
    if budget is not None:
        allowed = await budget.reserve(provider_name, "search", estimated_cost)
        if not allowed:
            log.warning(
                "search_worker_budget_skipped",
                provider=provider_name,
                query_id=q.query_id,
                estimated_cost=float(estimated_cost),
            )
            return {
                "raw_results": [],
                "errors": [
                    NodeError(
                        node="search_worker",
                        kind="budget_skipped",
                        detail=f"Budget ceiling reached for {provider_name}",
                        query_id=q.query_id,
                        retryable=False,
                        occurred_at=datetime.now(timezone.utc),
                    )
                ],
            }

    semaphore = _get_semaphore(provider_name)
    errors: list[NodeError] = []
    results: list[RawResult] = []
    actual_cost = Decimal("0")

    async with semaphore:
        try:
            results = await _search_with_retry(provider_impl, q, limit)
            actual_cost = estimated_cost  # providers don't return exact spend; use estimate
        except Exception as e:
            log.error(
                "search_worker_failed",
                provider=provider_name,
                query_id=q.query_id,
                error=str(e),
            )
            errors.append(
                NodeError(
                    node="search_worker",
                    kind="provider_error",
                    detail=str(e),
                    query_id=q.query_id,
                    retryable=False,
                    occurred_at=datetime.now(timezone.utc),
                )
            )
            actual_cost = Decimal("0")  # failed call, don't charge

    # Fix run_id on results (providers stub it with a placeholder)
    for r in results:
        r.run_id = run_id

    if budget is not None:
        await budget.commit(
            provider=provider_name,
            operation="search",
            actual=actual_cost,
            node="search_worker",
            loop_index=0,
        )

    log.info(
        "search_worker_done",
        provider=provider_name,
        query_id=q.query_id,
        result_count=len(results),
        error_count=len(errors),
    )

    return {"raw_results": results, "errors": errors}
