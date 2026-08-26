"""STEP 4b — fetch_worker node.

Downloads one selected source, extracts text via the escalation ladder
(provider raw content -> httpx+trafilatura/bs4 for HTML -> PyMuPDF for
PDF), normalizes it, and returns a SourceDocument. Never raises — a
failed fetch degrades to a NodeError and the source is marked "failed"
so downstream steps skip it instead of crashing the run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.logging import ctx_node, get_logger
from app.providers.extract.fetcher import fetch_and_extract
from app.providers.extract.normalize import content_hash, normalize_text
from app.schemas.evidence import NodeError
from app.schemas.source import Source, SourceDocument

log = get_logger()


async def fetch_worker_node(payload: dict[str, Any]) -> dict[str, Any]:
    """Fetch and extract text for one selected source.

    payload keys: run_id, source (Source)

    Returns a partial state update — in practice the caller is expected
    to persist the returned SourceDocument itself (state doesn't carry
    document text, per the "state carries IDs, Postgres carries text"
    design principle), so this returns both the updated Source (status
    flipped to "fetched"/"failed") and the SourceDocument for the caller
    to write.
    """
    ctx_node.set("fetch_worker")
    source: Source = payload["source"]
    run_id: str = payload["run_id"]

    url = source.fetch_url or source.url_canonical

    log.info("fetch_worker_start", source_id=source.source_id, url=url)

    result = await fetch_and_extract(url)

    if result is None:
        source.status = "failed"
        return {
            "source": source,
            "document": None,
            "errors": [
                NodeError(
                    node="fetch_worker",
                    kind="fetch_failed",
                    detail=f"Could not fetch or extract text from {url}",
                    source_id=source.source_id,
                    retryable=False,
                    occurred_at=datetime.now(timezone.utc),
                )
            ],
        }

    normalized = normalize_text(result.text)

    if len(normalized) < 50:
        source.status = "failed"
        return {
            "source": source,
            "document": None,
            "errors": [
                NodeError(
                    node="fetch_worker",
                    kind="fetch_failed",
                    detail=f"Extracted text too short after normalization ({len(normalized)} chars)",
                    source_id=source.source_id,
                    retryable=False,
                    occurred_at=datetime.now(timezone.utc),
                )
            ],
        }

    doc = SourceDocument(
        source_id=source.source_id,
        content_hash=content_hash(normalized),
        mime=result.mime,
        char_count=len(normalized),
        extraction_method=result.extraction_method,  # type: ignore[arg-type]
        extraction_confidence=result.extraction_confidence,
        is_quotable=result.is_quotable,
        language="en",
        text=normalized,
    )

    source.status = "fetched"

    log.info(
        "fetch_worker_done",
        source_id=source.source_id,
        char_count=doc.char_count,
        method=doc.extraction_method,
        is_quotable=doc.is_quotable,
        confidence=doc.extraction_confidence,
    )

    return {"source": source, "document": doc, "errors": []}
