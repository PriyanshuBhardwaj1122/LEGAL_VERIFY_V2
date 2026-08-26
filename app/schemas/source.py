"""Source models — raw results, scored sources, and document text."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, HttpUrl

from .citation import LegalCitation
from .common import CourtLevel, Jurisdiction, SourceType


class RawResult(BaseModel):
    raw_id: str
    run_id: str
    query_id: str
    provider: str
    url: HttpUrl
    url_canonical: str
    title: str | None = None
    snippet: str | None = None
    published_at: date | None = None
    provider_rank: int | None = None
    provider_score: float | None = None

    # Academic metadata (Semantic Scholar only)
    doi: str | None = None
    citation_count: int | None = None
    open_access_pdf: str | None = None

    # Merge-stage metadata (populated by search_merge, not persisted to
    # raw_result — used only in-memory to build Source.discovered_by /
    # provenance_query_ids in the evaluator).
    discovered_by: list[str] = []
    provenance_query_ids: list[str] = []
    venue: str | None = None

    fetched_at: datetime


class SourceScore(BaseModel):
    authority: float
    relevance: float
    recency: float
    reliability: float
    composite: float
    tier: int
    reasons: list[str]
    dropped_reason: str | None = None


class Source(BaseModel):
    source_id: str
    run_id: str
    url_canonical: str
    fetch_url: str | None = None
    title: str | None = None
    source_type: SourceType
    court_level: CourtLevel = CourtLevel.NONE
    issuing_body: str | None = None
    jurisdiction: Jurisdiction
    decided_or_published_on: date | None = None
    citation: LegalCitation | None = None
    score: SourceScore
    status: Literal["candidate", "selected", "fetched", "extracted", "failed", "dropped"]
    discovered_by: list[str] = []
    provenance_query_ids: list[str] = []


class SourceDocument(BaseModel):
    source_id: str
    content_hash: str
    mime: str
    char_count: int
    extraction_method: Literal["trafilatura", "bs4", "pymupdf", "ocr", "provider_raw"]
    extraction_confidence: float
    is_quotable: bool = True
    language: str = "en"
    text: str  # stored in Postgres, NOT in graph state