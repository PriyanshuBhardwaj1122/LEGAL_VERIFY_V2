"""Evidence models — the core contract consumed by Generation phase."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from .citation import LegalCitation
from .common import CourtLevel, Jurisdiction


class Evidence(BaseModel):
    evidence_id: str
    run_id: str
    source_id: str
    loop_index: int
    kind: Literal[
        "holding",
        "obiter",
        "statutory_text",
        "test_or_standard",
        "definition",
        "procedural_fact",
        "statistic",
        "policy_rationale",
        "commentary_opinion",
    ]
    statement: str = Field(max_length=500)
    # Statutory provisions are quoted whole and a single sub-section can
    # run well past a judgment's typical pin-cited sentence, so this is
    # sized for legislation rather than for case law.
    verbatim_quote: str | None = Field(default=None, max_length=4000)
    quote_start: int | None = None
    quote_end: int | None = None
    grounding: Literal["exact", "normalized", "fuzzy", "ungrounded"] = "ungrounded"
    pinpoint: str | None = None
    citation: LegalCitation | None = None
    supports_issues: list[str] = []
    jurisdiction: Jurisdiction
    court_level: CourtLevel = CourtLevel.NONE
    as_of: date | None = None
    binding_strength: Literal["binding", "persuasive", "informative", "opinion"]
    llm_confidence: float = Field(ge=0, le=1)
    authority_tier: int


class EvidenceCandidate(BaseModel):
    """What the extractor LLM emits. Narrower than Evidence:
    no offsets, no binding_strength, no tier — those are computed."""

    source_id: str
    kind: str
    statement: str = Field(max_length=500)
    verbatim_quote: str | None = None
    pinpoint: str | None = None
    citation_raw: str | None = None
    supports_issues: list[str] = []
    llm_confidence: float = Field(ge=0, le=1)


class NodeError(BaseModel):
    node: str
    kind: Literal[
        "provider_error",
        "budget_skipped",
        "fetch_failed",
        "captcha_blocked",
        "schema_violation",
        "extraction_failed",
        "timeout",
        "planner_fallback",
    ]
    detail: str
    source_id: str | None = None
    query_id: str | None = None
    retryable: bool = False
    occurred_at: datetime


class QuotaShortfall(BaseModel):
    """Emitted by Step 3 when a hard selection quota cannot be met."""

    requirement: Literal["statute_per_instrument", "min_tier12", "source_per_issue"]
    detail: str
    unmet_for: str | None = None
