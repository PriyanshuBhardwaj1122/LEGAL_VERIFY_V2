"""STEP 4a/4b coordinator — fans out fetch_worker over every selected
source concurrently, capped by a global semaphore so we don't hammer
many domains (some government sites are slow/rate-sensitive) at once.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.logging import ctx_node, get_logger
from app.graph.nodes.fetch_worker import fetch_worker_node
from app.graph.routing import fan_out_fetches
from app.schemas.evidence import NodeError
from app.schemas.source import Source, SourceDocument
from app.schemas.state import ResearchState

log = get_logger()

GLOBAL_FETCH_CONCURRENCY = 6


async def fetch_dispatch_node(state: ResearchState) -> dict[str, Any]:
    """Fan out fetch_worker over every selected source, capped by a
    global semaphore. Returns updated sources (status flipped) plus the
    fetched documents for the caller to persist."""
    ctx_node.set("fetch_dispatch")

    branches = fan_out_fetches(state)
    if not branches:
        log.info("fetch_dispatch_nothing_to_do")
        return {"sources": [], "documents": [], "errors": []}

    log.info("fetch_dispatch_start", branch_count=len(branches))

    semaphore = asyncio.Semaphore(GLOBAL_FETCH_CONCURRENCY)

    async def _bounded(payload: dict) -> dict:
        async with semaphore:
            return await fetch_worker_node(payload)

    branch_results = await asyncio.gather(
        *[_bounded(payload) for payload in branches], return_exceptions=True
    )

    updated_sources: list[Source] = []
    documents: list[SourceDocument] = []
    errors: list[NodeError] = []

    for payload, result in zip(branches, branch_results):
        source: Source = payload["source"]

        if isinstance(result, Exception):
            log.error(
                "fetch_dispatch_branch_crashed", source_id=source.source_id, error=str(result)
            )
            source.status = "failed"
            updated_sources.append(source)
            from datetime import datetime, timezone

            errors.append(
                NodeError(
                    node="fetch_dispatch",
                    kind="fetch_failed",
                    detail=str(result),
                    source_id=source.source_id,
                    retryable=False,
                    occurred_at=datetime.now(timezone.utc),
                )
            )
            continue

        updated_sources.append(result["source"])
        if result.get("document") is not None:
            documents.append(result["document"])
        errors.extend(result.get("errors", []))

    fetched_count = sum(1 for s in updated_sources if s.status == "fetched")
    log.info(
        "fetch_dispatch_done",
        branch_count=len(branches),
        fetched_count=fetched_count,
        failed_count=len(branches) - fetched_count,
    )

    return {"sources": updated_sources, "documents": documents, "errors": errors}
