"""STEP 2a/2b coordinator — fans out sub_queries x providers concurrently,
awaits every search_worker branch, and aggregates results.

In the fully compiled LangGraph (wired in a later milestone), fan-out is
driven by conditional edges returning Send objects and search_worker runs
as an independent graph node per branch. This coordinator does the same
fan-out-and-join directly, which keeps Step 2 independently testable
without requiring the whole StateGraph to exist yet.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.budget import BudgetGuard
from app.core.logging import ctx_node, get_logger
from app.graph.nodes.search_worker import search_worker_node
from app.graph.routing import fan_out_searches
from app.schemas.evidence import NodeError
from app.schemas.source import RawResult
from app.schemas.state import ResearchState

log = get_logger()


async def search_dispatch_node(
    state: ResearchState,
    budget: BudgetGuard | None = None,
) -> dict[str, Any]:
    """Fan out every (sub_query, provider) pair not already executed,
    run them concurrently, and merge the results back into state."""
    ctx_node.set("search_dispatch")

    branches = fan_out_searches(state)
    if not branches:
        log.info("search_dispatch_nothing_to_do")
        return {"raw_results": [], "errors": []}

    log.info("search_dispatch_start", branch_count=len(branches))

    tasks = [search_worker_node(payload, budget=budget) for payload in branches]
    branch_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_raw: list[RawResult] = []
    all_errors: list[NodeError] = []
    newly_executed: set[str] = set()

    for payload, result in zip(branches, branch_results):
        q = payload["sub_query"]
        provider = payload["provider"]
        key = f"{q.query_id}:{provider}"
        newly_executed.add(key)

        if isinstance(result, Exception):
            log.error("search_dispatch_branch_crashed", key=key, error=str(result))
            from datetime import datetime, timezone

            all_errors.append(
                NodeError(
                    node="search_dispatch",
                    kind="provider_error",
                    detail=str(result),
                    query_id=q.query_id,
                    retryable=False,
                    occurred_at=datetime.now(timezone.utc),
                )
            )
            continue

        all_raw.extend(result.get("raw_results", []))
        all_errors.extend(result.get("errors", []))

    log.info(
        "search_dispatch_done",
        branch_count=len(branches),
        raw_result_count=len(all_raw),
        error_count=len(all_errors),
    )

    return {
        "raw_results": all_raw,
        "errors": all_errors,
        "executed_query_ids": newly_executed,
    }