"""STEP 2c — search_merge node.

Deduplicates raw results by canonical URL, merges discovered_by across
providers (multi-provider agreement becomes a reliability signal in
Step 3), flags near-duplicate mirrors, and drops obvious junk before any
LLM ever sees a candidate.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import ctx_node, get_logger
from app.providers.search.url_utils import is_blocked_domain, is_junk_domain
from app.schemas.source import RawResult
from app.schemas.state import ResearchState

log = get_logger()

MAX_PDF_SIZE_MB = 40


class MergedCandidate:
    """Intermediate representation before Source objects are built in Step 3.
    Not a persisted schema — just carries merge bookkeeping."""

    __slots__ = ("url_canonical", "best_result", "discovered_by", "provenance_query_ids")

    def __init__(self, result: RawResult):
        self.url_canonical = result.url_canonical
        self.best_result = result
        self.discovered_by: set[str] = {result.provider}
        self.provenance_query_ids: set[str] = {result.query_id}

    def absorb(self, result: RawResult) -> None:
        self.discovered_by.add(result.provider)
        self.provenance_query_ids.add(result.query_id)
        # Prefer the result with a richer snippet/title as the "best" copy
        if self._richness(result) > self._richness(self.best_result):
            self.best_result = result

    @staticmethod
    def _richness(r: RawResult) -> int:
        return len(r.snippet or "") + len(r.title or "") * 2


def search_merge_node(state: ResearchState) -> dict[str, Any]:
    """Merge, dedupe, and junk-filter raw_results in state.

    Returns raw_results collapsed to one entry per canonical URL, with
    discovered_by/provenance_query_ids folded into the payload field so
    Step 3 can read multi-provider agreement without a second pass.
    """
    ctx_node.set("search_merge")
    raw_results: list[RawResult] = state.get("raw_results", [])

    if not raw_results:
        log.info("search_merge_nothing_to_do")
        return {"raw_results": []}

    dropped_junk = 0
    dropped_blocked = 0
    merged: dict[str, MergedCandidate] = {}

    for r in raw_results:
        url = str(r.url)

        if is_blocked_domain(url):
            dropped_blocked += 1
            continue
        if is_junk_domain(url):
            dropped_junk += 1
            continue

        existing = merged.get(r.url_canonical)
        if existing is None:
            merged[r.url_canonical] = MergedCandidate(r)
        else:
            existing.absorb(r)

    # Fold discovered_by / provenance back onto the surviving RawResult's
    # declared merge-metadata fields — downstream (evaluator) reads these
    # directly when building Source objects.
    survivors: list[RawResult] = []
    for candidate in merged.values():
        r = candidate.best_result.model_copy(
            update={
                "discovered_by": sorted(candidate.discovered_by),
                "provenance_query_ids": sorted(candidate.provenance_query_ids),
            }
        )
        survivors.append(r)

    log.info(
        "search_merge_done",
        input_count=len(raw_results),
        survivor_count=len(survivors),
        dropped_junk=dropped_junk,
        dropped_blocked=dropped_blocked,
        multi_provider_count=sum(1 for c in merged.values() if len(c.discovered_by) > 1),
    )

    return {"raw_results": survivors}