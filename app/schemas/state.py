"""LangGraph state definition with idempotent reducers."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from .evidence import Evidence, EvidenceCandidate, NodeError, QuotaShortfall
from .gaps import GapReport
from .plan import ResearchPlan
from .request import ResearchRequest
from .source import RawResult, Source


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------

def upsert_by(key: str):
    """Last-write-wins merge keyed by an attribute. Makes parallel Send
    branches and retried nodes idempotent."""

    def _reducer(left: list | None, right: list | None) -> list:
        merged: dict[str, Any] = {}
        for x in left or []:
            k = getattr(x, key) if hasattr(x, key) else x.get(key)  # type: ignore[union-attr]
            merged[k] = x
        for x in right or []:
            k = getattr(x, key) if hasattr(x, key) else x.get(key)  # type: ignore[union-attr]
            merged[k] = x
        return list(merged.values())

    return _reducer


def union_set(left: set | None, right: set | None) -> set:
    return (left or set()) | (right or set())


def append(left: list | None, right: list | None) -> list:
    return (left or []) + (right or [])


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    # Immutable inputs
    run_id: str
    request: ResearchRequest

    # Step 1 — planner
    plan: ResearchPlan
    plan_history: Annotated[list[ResearchPlan], append]
    executed_query_ids: Annotated[set[str], union_set]

    # Step 2 — search
    raw_results: Annotated[list[RawResult], upsert_by("raw_id")]

    # Step 3 — evaluator
    sources: Annotated[list[Source], upsert_by("source_id")]
    quota_shortfalls: Annotated[list[QuotaShortfall], append]

    # Step 4 — fetch + extract + ground
    fetched_source_ids: Annotated[set[str], union_set]
    evidence_candidates: list[EvidenceCandidate]  # transient: overwrite reducer
    evidence: Annotated[list[Evidence], upsert_by("evidence_id")]

    # Step 5 — gap check
    gap_report: GapReport
    gap_history: Annotated[list[GapReport], append]

    # Control
    loop_index: int
    budget_spent_inr: float
    errors: Annotated[list[NodeError], append]
    halt_reason: str | None
