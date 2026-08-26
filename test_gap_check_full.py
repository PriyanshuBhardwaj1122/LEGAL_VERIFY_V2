"""Step 5 (gap check) end-to-end verification — runs against a run's
already-persisted plan/sources/evidence from Postgres (i.e. after
test_m2_full.py and test_m3_full.py have both been run for that run_id).

Usage:
    python test_gap_check_full.py                # uses the most recent run
    python test_gap_check_full.py <run_id>        # uses a specific run

Note: quota_shortfalls from the M2 evaluator step are NOT persisted
anywhere in the DB (there's no table for them — they're only printed at
run time), so this script always passes quota_shortfalls=[] to
gap_check_node. That means the "promote an unmet selection-time quota to
a gap" rule path won't fire here even if the original M2 run reported
one — it's still covered by the synthetic unit test
(test_rule_quota_shortfall_promoted in tests/test_gap_check_synthetic.py).

Also note: Evidence.as_of isn't persisted by EvidenceRow either (a pre-
existing gap in the M0 schema, not introduced here), so the STALE_LAW
rule-based check will always read every evidence item's as_of as None
when evidence is reloaded from the DB like this — it'll only work
correctly on evidence that's still live in-process during a real
pipeline run, not on this kind of after-the-fact replay.
"""
from __future__ import annotations

import asyncio
import sys
from collections import Counter
from datetime import datetime

from sqlalchemy import select

from app.core.logging import get_logger
from app.db.models import EvidenceRow, ResearchPlanRow, ResearchRun, SourceRow
from app.db.repo.research import GapReportRepo
from app.db.session import get_db_session
from app.graph.nodes.gap_check import gap_check_node
from app.schemas.citation import LegalCitation
from app.schemas.common import CourtLevel, Jurisdiction, SourceType
from app.schemas.evidence import Evidence
from app.schemas.plan import ResearchPlan
from app.schemas.request import ArticleConfig, ResearchRequest
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
        # of models.py's Mapped[datetime] hint.
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
        as_of=None,  # not persisted — see module docstring
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

        source_rows = (
            await session.execute(select(SourceRow).where(SourceRow.run_id == run_id))
        ).scalars().all()
        sources = [_row_to_source(r) for r in source_rows]

        evidence_rows = (
            await session.execute(select(EvidenceRow).where(EvidenceRow.run_id == run_id))
        ).scalars().all()
        evidence = [_row_to_evidence(r) for r in evidence_rows]

        print(f"Loaded: {len(sources)} sources, {len(evidence)} evidence items, "
              f"{len(plan.legal_issues)} legal issues, {len(plan.key_instruments)} key instruments\n")

        if not evidence:
            print("No evidence persisted for this run — run test_m3_full.py first.")
            return

        # Reconstruct a minimal ResearchRequest for max_research_loops —
        # the persisted request JSON has everything gap_check needs.
        request = ResearchRequest.model_validate(run_row.request)

        state = {
            "run_id": run_id,
            "request": request,
            "plan": plan,
            "sources": sources,
            "evidence": evidence,
            "quota_shortfalls": [],
            "loop_index": run_row.loop_index or 0,
        }

        print("=== gap_check_node (rule-based + real LLM call) ===")
        result = await gap_check_node(state)
        gap_report = result["gap_report"]

        async with get_db_session() as write_session:
            await GapReportRepo(write_session).upsert(gap_report)
        print("Persisted GapReport to Postgres.\n")

        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Verdict: {gap_report.verdict}")
        print(f"Rationale: {gap_report.rationale}\n")

        print("Coverage (issue: evidence count):")
        for issue, count in gap_report.coverage.items():
            flag = " <-- THIN" if count < 2 else ""
            print(f"  [{count}] {issue}{flag}")

        print(f"\nTier histogram: {dict(sorted(gap_report.tier_histogram.items()))}")

        by_severity = Counter(g.severity for g in gap_report.gaps)
        by_detector = Counter(g.detected_by for g in gap_report.gaps)
        print(f"\nGaps found: {len(gap_report.gaps)}  (by severity: {dict(by_severity)}, "
              f"by detector: {dict(by_detector)})")

        for g in gap_report.gaps:
            issue_tag = f" [{g.issue}]" if g.issue else ""
            print(f"  ({g.severity}/{g.detected_by}) {g.gap_kind}{issue_tag}: {g.detail}")

        print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
