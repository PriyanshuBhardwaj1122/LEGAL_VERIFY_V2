"""Gap analysis models — Step 5 output."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from .plan import SubQuery


class GapKind(StrEnum):
    MISSING_STATUTORY_BASIS = "missing_statutory_basis"
    MISSING_PRIMARY_AUTHORITY = "missing_primary_authority"
    MISSING_APEX_RULING = "missing_apex_ruling"
    MISSING_COUNTER_VIEW = "missing_counter_view"
    STALE_LAW = "stale_law"
    JURISDICTION_MISMATCH = "jurisdiction_mismatch"
    THIN_COVERAGE = "thin_coverage"
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    NO_IMPLEMENTATION_DETAIL = "no_implementation_detail"


class Gap(BaseModel):
    gap_kind: GapKind
    issue: str | None = None
    detail: str
    severity: Literal["blocking", "important", "nice_to_have"]
    detected_by: Literal["rule", "llm"]
    suggested_queries: list[SubQuery] = []


class GapReport(BaseModel):
    run_id: str
    loop_index: int
    coverage: dict[str, int]
    tier_histogram: dict[int, int]
    gaps: list[Gap]
    verdict: Literal["complete", "needs_more_research", "exhausted"]
    rationale: str
