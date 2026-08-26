"""SQLAlchemy 2.0 ORM models matching §7 schema."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


def _utcnow():
    return datetime.now(timezone.utc)


class ResearchRun(Base):
    __tablename__ = "research_run"
    __table_args__ = {"schema": "research"}

    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    jurisdiction: Mapped[str] = mapped_column(Text, nullable=False)
    request: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="queued")
    loop_index: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    verdict: Mapped[str | None] = mapped_column(Text, nullable=True)
    budget_inr: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    spent_inr: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False, default=Decimal("0"))
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False, default="v0.1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ResearchPlanRow(Base):
    __tablename__ = "research_plan"
    __table_args__ = {"schema": "research"}

    plan_id: Mapped[str] = mapped_column(Text, primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    loop_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class SearchQuery(Base):
    __tablename__ = "search_query"
    __table_args__ = (
        Index("ix_search_query_run", "run_id"),
        {"schema": "research"},
    )

    query_id: Mapped[str] = mapped_column(Text, primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    provider: Mapped[str] = mapped_column(Text, primary_key=True)
    loop_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    result_count: Mapped[int] = mapped_column(Integer, default=0)


class RawResultRow(Base):
    __tablename__ = "raw_result"
    __table_args__ = (
        Index("ix_raw_result_run_url", "run_id", "url_canonical"),
        {"schema": "research"},
    )

    raw_id: Mapped[str] = mapped_column(Text, primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    query_id: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    url_canonical: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    provider_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class SourceRow(Base):
    __tablename__ = "source"
    __table_args__ = (
        Index("ix_source_run_status_tier", "run_id", "status", "tier"),
        {"schema": "research"},
    )

    source_id: Mapped[str] = mapped_column(Text, primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    url_canonical: Mapped[str] = mapped_column(Text, nullable=False)
    fetch_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    doi: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    court_level: Mapped[str] = mapped_column(Text, nullable=False, default="none")
    issuing_body: Mapped[str | None] = mapped_column(Text, nullable=True)
    jurisdiction: Mapped[str] = mapped_column(Text, nullable=False)
    published_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    citation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tier: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    score: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    discovered_by: Mapped[list] = mapped_column(ARRAY(Text), nullable=False, default=list)


class SourceDocumentRow(Base):
    __tablename__ = "source_document"
    __table_args__ = (
        Index("ix_source_document_url", "url_canonical", "fetched_at"),
        {"schema": "research"},
    )

    content_hash: Mapped[str] = mapped_column(Text, primary_key=True)
    url_canonical: Mapped[str] = mapped_column(Text, nullable=False)
    mime: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    extraction_method: Mapped[str] = mapped_column(Text, nullable=False)
    extraction_confidence: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False)
    is_quotable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    paragraph_map: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SourceDocumentLink(Base):
    __tablename__ = "source_document_link"
    __table_args__ = {"schema": "research"}

    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    source_id: Mapped[str] = mapped_column(Text, primary_key=True)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)


class EvidenceRow(Base):
    __tablename__ = "evidence"
    __table_args__ = (
        Index("ix_evidence_run_grounding", "run_id", "grounding", "authority_tier"),
        {"schema": "research"},
    )

    evidence_id: Mapped[str] = mapped_column(Text, primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    loop_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    verbatim_quote: Mapped[str | None] = mapped_column(Text, nullable=True)
    quote_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quote_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    grounding: Mapped[str] = mapped_column(Text, nullable=False)
    pinpoint: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    supports_issues: Mapped[list] = mapped_column(ARRAY(Text), nullable=False, default=list)
    jurisdiction: Mapped[str] = mapped_column(Text, nullable=False)
    court_level: Mapped[str] = mapped_column(Text, nullable=False)
    binding_strength: Mapped[str] = mapped_column(Text, nullable=False)
    authority_tier: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    llm_confidence: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class GapReportRow(Base):
    __tablename__ = "gap_report"
    __table_args__ = {"schema": "research"}

    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    loop_index: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)


class ArticleDraftRow(Base):
    """One persisted draft per run (latest overwrites — a run is
    redrafted, not versioned, at this stage of the pipeline)."""

    __tablename__ = "article_draft"
    __table_args__ = {"schema": "research"}

    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)  # ArticleDraft, model_dump
    verification: Mapped[dict] = mapped_column(JSONB, nullable=False)  # DraftVerificationReport
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ApiCallLog(Base):
    __tablename__ = "api_call_log"
    __table_args__ = (
        Index("ix_api_call_log_run_provider", "run_id", "provider"),
        {"schema": "research"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    node: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    operation: Mapped[str] = mapped_column(Text, nullable=False)
    loop_index: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    cost_inr: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False, default=Decimal("0"))
    units: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class RunMetric(Base):
    __tablename__ = "run_metric"
    __table_args__ = {"schema": "research"}

    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[float] = mapped_column(Numeric, nullable=False)


class NodeEvent(Base):
    __tablename__ = "node_event"
    __table_args__ = (
        Index("ix_node_event_run_started", "run_id", "started_at"),
        {"schema": "research"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    node: Mapped[str] = mapped_column(Text, nullable=False)
    loop_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    detail: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
