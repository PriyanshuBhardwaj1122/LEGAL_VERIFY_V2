"""Repository classes for research pipeline DB operations.

All writes use upsert semantics — idempotent by design.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    ApiCallLog,
    ArticleDraftRow,
    EvidenceRow,
    GapReportRow,
    NodeEvent,
    RawResultRow,
    ResearchPlanRow,
    ResearchRun,
    SearchQuery,
    SourceDocumentLink,
    SourceDocumentRow,
    SourceRow,
)
from app.schemas.article import ArticleDraft, DraftVerificationReport
from app.schemas.evidence import Evidence
from app.schemas.gaps import GapReport
from app.schemas.plan import ResearchPlan
from app.schemas.request import ResearchRequest
from app.schemas.source import RawResult, Source, SourceDocument


class ResearchRunRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_run(
        self,
        run_id: str,
        request: ResearchRequest,
        prompt_version: str = "v0.1",
    ) -> ResearchRun:
        row = ResearchRun(
            run_id=uuid.UUID(run_id),
            topic=request.topic,
            jurisdiction=request.jurisdiction.value,
            request=request.model_dump(mode="json"),
            status="queued",
            budget_inr=request.budget_inr,
            prompt_version=prompt_version,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )
        self.session.add(row)
        await self.session.commit()
        return row

    async def update_status(
        self,
        run_id: str,
        status: str,
        *,
        loop_index: int | None = None,
        verdict: str | None = None,
        spent_inr: Decimal | None = None,
    ) -> None:
        run = await self.session.get(ResearchRun, uuid.UUID(run_id))
        if run:
            run.status = status
            if loop_index is not None:
                run.loop_index = loop_index
            if verdict is not None:
                run.verdict = verdict
            if spent_inr is not None:
                run.spent_inr = spent_inr
            if status in ("complete", "exhausted", "failed"):
                run.completed_at = datetime.now(timezone.utc)
            await self.session.commit()


class PlanRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, plan: ResearchPlan) -> None:
        stmt = pg_insert(ResearchPlanRow).values(
            plan_id=plan.plan_id,
            run_id=uuid.UUID(plan.run_id),
            loop_index=plan.loop_index,
            payload=plan.model_dump(mode="json"),
        ).on_conflict_do_update(
            index_elements=["plan_id"],
            set_={"payload": plan.model_dump(mode="json")},
        )
        await self.session.execute(stmt)
        await self.session.commit()


class SearchQueryRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(
        self,
        run_id: str,
        query_id: str,
        provider: str,
        loop_index: int,
        payload: dict,
        status: str = "pending",
        result_count: int = 0,
    ) -> None:
        stmt = pg_insert(SearchQuery).values(
            query_id=query_id,
            run_id=uuid.UUID(run_id),
            provider=provider,
            loop_index=loop_index,
            payload=payload,
            status=status,
            result_count=result_count,
        ).on_conflict_do_update(
            index_elements=["query_id", "run_id", "provider"],
            set_={"status": status, "result_count": result_count},
        )
        await self.session.execute(stmt)
        await self.session.commit()


class RawResultRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def bulk_upsert(self, results: list[RawResult], run_id: str) -> None:
        if not results:
            return
        for r in results:
            stmt = pg_insert(RawResultRow).values(
                raw_id=r.raw_id,
                run_id=uuid.UUID(run_id),
                query_id=r.query_id,
                provider=r.provider,
                url_canonical=r.url_canonical,
                title=r.title,
                snippet=r.snippet,
                published_at=r.published_at,
                provider_rank=r.provider_rank,
                payload=None,
            ).on_conflict_do_nothing(index_elements=["raw_id"])
            await self.session.execute(stmt)
        await self.session.commit()


class SourceRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def bulk_upsert(self, sources: list[Source]) -> None:
        for s in sources:
            stmt = pg_insert(SourceRow).values(
                source_id=s.source_id,
                run_id=uuid.UUID(s.run_id),
                url_canonical=s.url_canonical,
                fetch_url=s.fetch_url,
                title=s.title,
                source_type=s.source_type.value,
                court_level=s.court_level.value,
                issuing_body=s.issuing_body,
                jurisdiction=s.jurisdiction.value,
                published_on=s.decided_or_published_on,
                citation=s.citation.model_dump(mode="json") if s.citation else None,
                tier=s.score.tier,
                score=s.score.model_dump(mode="json"),
                status=s.status,
                discovered_by=s.discovered_by,
            ).on_conflict_do_update(
                index_elements=["source_id", "run_id"],
                set_={
                    "status": s.status,
                    "score": s.score.model_dump(mode="json"),
                    "discovered_by": s.discovered_by,
                },
            )
            await self.session.execute(stmt)
        await self.session.commit()


class SourceDocumentRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, doc: SourceDocument, run_id: str) -> None:
        stmt = pg_insert(SourceDocumentRow).values(
            content_hash=doc.content_hash,
            url_canonical=doc.source_id,  # source_id as url_canonical for now
            mime=doc.mime,
            text=doc.text,
            char_count=doc.char_count,
            extraction_method=doc.extraction_method,
            extraction_confidence=doc.extraction_confidence,
            is_quotable=doc.is_quotable,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        ).on_conflict_do_nothing(index_elements=["content_hash"])
        await self.session.execute(stmt)

        # Link doc to this run+source
        link_stmt = pg_insert(SourceDocumentLink).values(
            run_id=uuid.UUID(run_id),
            source_id=doc.source_id,
            content_hash=doc.content_hash,
        ).on_conflict_do_nothing(index_elements=["run_id", "source_id"])
        await self.session.execute(link_stmt)
        await self.session.commit()


class EvidenceRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def bulk_upsert(self, evidence_list: list[Evidence]) -> None:
        for e in evidence_list:
            stmt = pg_insert(EvidenceRow).values(
                evidence_id=e.evidence_id,
                run_id=uuid.UUID(e.run_id),
                source_id=e.source_id,
                loop_index=e.loop_index,
                kind=e.kind,
                statement=e.statement,
                verbatim_quote=e.verbatim_quote,
                quote_start=e.quote_start,
                quote_end=e.quote_end,
                grounding=e.grounding,
                pinpoint=e.pinpoint,
                citation=e.citation.model_dump(mode="json") if e.citation else None,
                supports_issues=e.supports_issues,
                jurisdiction=e.jurisdiction.value,
                court_level=e.court_level.value,
                binding_strength=e.binding_strength,
                authority_tier=e.authority_tier,
                llm_confidence=e.llm_confidence,
            ).on_conflict_do_update(
                index_elements=["evidence_id"],
                set_={
                    "grounding": e.grounding,
                    "quote_start": e.quote_start,
                    "quote_end": e.quote_end,
                },
            )
            await self.session.execute(stmt)
        await self.session.commit()


class GapReportRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, report: GapReport) -> None:
        stmt = pg_insert(GapReportRow).values(
            run_id=uuid.UUID(report.run_id),
            loop_index=report.loop_index,
            payload=report.model_dump(mode="json"),
        ).on_conflict_do_update(
            index_elements=["run_id", "loop_index"],
            set_={"payload": report.model_dump(mode="json")},
        )
        await self.session.execute(stmt)
        await self.session.commit()


class ArticleDraftRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, draft: ArticleDraft, verification: DraftVerificationReport) -> None:
        stmt = pg_insert(ArticleDraftRow).values(
            run_id=uuid.UUID(draft.run_id),
            title=draft.title,
            payload=draft.model_dump(mode="json"),
            verification=verification.model_dump(mode="json"),
        ).on_conflict_do_update(
            index_elements=["run_id"],
            set_={
                "title": draft.title,
                "payload": draft.model_dump(mode="json"),
                "verification": verification.model_dump(mode="json"),
            },
        )
        await self.session.execute(stmt)
        await self.session.commit()


class NodeEventRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def record(
        self,
        run_id: str,
        node: str,
        loop_index: int,
        status: str,
        started_at: datetime,
        duration_ms: int,
        detail: dict | None = None,
    ) -> None:
        row = NodeEvent(
            run_id=uuid.UUID(run_id),
            node=node,
            loop_index=loop_index,
            status=status,
            started_at=started_at,
            duration_ms=duration_ms,
            detail=detail,
        )
        self.session.add(row)
        await self.session.commit()
