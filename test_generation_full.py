"""Generation phase (Steps 6-9) end-to-end verification — runs against
a run's already-persisted plan/sources/evidence from Postgres (i.e.
after test_m2_full.py and test_m3_full.py, and ideally test_gap_check_full.py,
have been run for that run_id).

Usage:
    python test_generation_full.py                # uses the most recent run
    python test_generation_full.py <run_id>        # uses a specific run

Requires the article_draft table. If it doesn't exist yet, create it
once with:
    PYTHONPATH=. python -c "
import asyncio
from app.db.session import get_engine
from app.db.base import Base
import app.db.models  # noqa: F401 -- registers all tables incl. ArticleDraftRow
async def main():
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
asyncio.run(main())
"

Note: like test_gap_check_full.py, this replays evidence reloaded from
Postgres rather than in-process state, so Evidence.as_of is always None
on reload (a pre-existing EvidenceRow gap, not introduced here) — this
doesn't affect generation, only gap_check's stale-law rule.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime

from sqlalchemy import select

from app.core.logging import get_logger
from app.db.models import EvidenceRow, ResearchPlanRow, ResearchRun, SourceRow
from app.db.repo.research import ArticleDraftRepo
from app.db.session import get_db_session
from app.graph.nodes.generation import generation_node
from app.schemas.citation import LegalCitation
from app.schemas.common import CourtLevel, Jurisdiction, SourceType
from app.schemas.evidence import Evidence
from app.schemas.handoff import EvidencePackage
from app.schemas.plan import ResearchPlan
from app.schemas.request import ResearchRequest
from app.schemas.source import Source, SourceScore

log = get_logger()


def _row_to_source(row: SourceRow) -> Source:
    citation = LegalCitation.model_validate(row.citation) if row.citation else None
    return Source(
        source_id=row.source_id,
        run_id=str(row.run_id),
        url_canonical=row.url_canonical,
        fetch_url=row.fetch_url,
        title=row.title,
        source_type=SourceType(row.source_type),
        court_level=CourtLevel(row.court_level),
        issuing_body=row.issuing_body,
        jurisdiction=Jurisdiction(row.jurisdiction),
        # SourceRow.published_on is a SQL Date column — always a plain
        # datetime.date at runtime, never datetime.datetime, regardless
        # of models.py's Mapped[datetime] hint. (Same fix as
        # test_m3_full.py / test_gap_check_full.py / test_repair_loop_full.py.)
        decided_or_published_on=(
            row.published_on.date() if isinstance(row.published_on, datetime) else row.published_on
        ),
        citation=citation,
        score=SourceScore.model_validate(row.score),
        status=row.status,
        discovered_by=row.discovered_by or [],
        provenance_query_ids=[],
    )


def _row_to_evidence(row: EvidenceRow) -> Evidence:
    citation = LegalCitation.model_validate(row.citation) if row.citation else None
    return Evidence(
        evidence_id=row.evidence_id,
        run_id=str(row.run_id),
        source_id=row.source_id,
        loop_index=row.loop_index,
        kind=row.kind,  # type: ignore[arg-type]
        statement=row.statement,
        verbatim_quote=row.verbatim_quote,
        quote_start=row.quote_start,
        quote_end=row.quote_end,
        grounding=row.grounding,  # type: ignore[arg-type]
        pinpoint=row.pinpoint,
        citation=citation,
        supports_issues=row.supports_issues or [],
        jurisdiction=Jurisdiction(row.jurisdiction),
        court_level=CourtLevel(row.court_level),
        as_of=None,
        binding_strength=row.binding_strength,  # type: ignore[arg-type]
        llm_confidence=float(row.llm_confidence),
        authority_tier=row.authority_tier,
    )


async def main():
    requested_run_id = sys.argv[1] if len(sys.argv) > 1 else None

    async with get_db_session() as session:
        if requested_run_id:
            run_id = requested_run_id
        else:
            result = await session.execute(
                select(ResearchRun).order_by(ResearchRun.created_at.desc()).limit(1)
            )
            run = result.scalar_one_or_none()
            if run is None:
                print("No research runs found in the database.")
                return
            run_id = str(run.run_id)

        run_row = await session.get(ResearchRun, run_id)
        if run_row is None:
            print(f"run_id={run_id} not found.")
            return

        print(f"Using run_id={run_id}  topic={run_row.topic[:60]!r}")

        plan_result = await session.execute(
            select(ResearchPlanRow)
            .where(ResearchPlanRow.run_id == run_id)
            .order_by(ResearchPlanRow.created_at.desc())
            .limit(1)
        )
        plan_row = plan_result.scalar_one_or_none()
        if plan_row is None:
            print("No plan persisted for this run — run test_m2_full.py first.")
            return
        plan = ResearchPlan.model_validate(plan_row.payload)

        evidence_rows = (
            await session.execute(select(EvidenceRow).where(EvidenceRow.run_id == run_id))
        ).scalars().all()
        evidence = [_row_to_evidence(r) for r in evidence_rows]

        source_rows = (
            await session.execute(select(SourceRow).where(SourceRow.run_id == run_id))
        ).scalars().all()
        sources = [_row_to_source(r) for r in source_rows]
        press_count = sum(
            1 for s in sources if s.source_type in (SourceType.LEGAL_NEWS, SourceType.FIRM_COMMENTARY)
        )

        print(f"Loaded: {len(sources)} sources ({press_count} press-tier), {len(evidence)} evidence items, "
              f"{len(plan.legal_issues)} legal issues\n")

        if not evidence:
            print("No evidence persisted for this run — run test_m3_full.py first.")
            return

        request = ResearchRequest.model_validate(run_row.request)

        package = EvidencePackage(
            run_id=run_id,
            topic=request.topic,
            jurisdiction=request.jurisdiction,
            as_of_date=request.as_of_date,
            legal_issues=plan.legal_issues,
            evidence=evidence,
            sources=sources,
            unresolved_gaps=[],
            coverage={},
            research_verdict="complete",
        )

        print("=== generation_node (outline -> draft -> assemble -> verify; real LLM calls) ===")
        result = await generation_node(package, request.article_config)
        draft = result["article_draft"]
        report = result["draft_verification"]

        # Write the file and print the full report BEFORE attempting to
        # persist — a DB error (e.g. the table doesn't exist yet) must
        # never hide what the run actually produced. Persistence is a
        # nice-to-have here, not the thing we're checking.
        out_path = f"article_draft_{run_id[:8]}.md"
        with open(out_path, "w") as f:
            f.write(draft.rendered_markdown)
        print(f"Wrote rendered article to {out_path}\n")

        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Title: {draft.title}")
        print(f"Sections: {len(draft.sections)}")
        print(f"Rendered length: {len(draft.rendered_markdown)} chars\n")

        print(f"Verification verdict: {report.verdict}")
        print(f"Rationale: {report.rationale}")
        print(f"Citations: {report.citation_count} total, {report.unique_evidence_cited} unique evidence items cited\n")

        if report.marker_issues:
            print(f"Marker issues ({len(report.marker_issues)}):")
            for mi in report.marker_issues:
                print(f"  [{mi.kind}] section={mi.section_id} evidence_id={mi.marker_evidence_id}: {mi.detail}")

        if report.quote_issues:
            print(f"\nQuote issues ({len(report.quote_issues)}):")
            for qi in report.quote_issues:
                print(f"  section={qi.section_id} evidence_id={qi.evidence_id}: {qi.detail}")
                print(f"    quoted: {qi.quoted_text[:150]!r}")

        if report.uncovered_issues:
            print(f"\nUncovered issues ({len(report.uncovered_issues)}):")
            for issue in report.uncovered_issues:
                print(f"  - {issue}")

        # Auto-create the article_draft table on first use (checkfirst=True
        # is a no-op if it already exists) — removes the manual setup
        # step that tripped up the first real run of this script.
        try:
            from app.db.base import Base
            from app.db.models import ArticleDraftRow
            from app.db.session import get_engine
            async with get_engine().begin() as conn:
                await conn.run_sync(
                    lambda sync_conn: Base.metadata.create_all(
                        sync_conn, tables=[ArticleDraftRow.__table__], checkfirst=True
                    )
                )
        except Exception:
            pass  # fall through to the real attempt below; its error is the useful one

        try:
            async with get_db_session() as write_session:
                await ArticleDraftRepo(write_session).upsert(draft, report)
            print("\nPersisted ArticleDraft + DraftVerificationReport to Postgres.")
        except Exception as e:
            print(f"\nWARNING: could not persist to Postgres ({type(e).__name__}: {e})")
            print("The article and verification report above are still valid — this only affects storage.")
            print("If the table doesn't exist yet, create it once with:")
            print("  PYTHONPATH=. python -c \"")
            print("import asyncio")
            print("from app.db.session import get_engine")
            print("from app.db.base import Base")
            print("import app.db.models")
            print("async def m():")
            print("    async with get_engine().begin() as c:")
            print("        await c.run_sync(Base.metadata.create_all)")
            print("asyncio.run(m())\"")

        print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())