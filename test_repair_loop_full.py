"""Repair-loop verification — runs an actual loop_index=1 pass: repair
planner -> search -> merge -> evaluate -> fetch -> extract -> ground ->
gap_check again, against a run that already has a loop-0 GapReport with
verdict=needs_more_research.

Usage:
    python test_repair_loop_full.py                # most recent run
    python test_repair_loop_full.py <run_id>        # a specific run

IMPORTANT — this is a deliberately simplified stand-in for the real
LangGraph repair loop, not a full replay of it:

  - The real StateGraph accumulates raw_results/sources across loops via
    upsert_by reducers, so a loop-1 evaluator reconsiders the ENTIRE
    candidate pool ever found (including loop-0 candidates that didn't
    make the cut). This script does NOT reconstruct that: RawResultRow
    doesn't persist enough fields (url, discovered_by, provider_score,
    etc.) to losslessly rebuild loop-0's RawResult objects, so
    reconstructing them here would be a lossy approximation dressed up
    as a faithful one. Instead this script evaluates loop-1's NEW raw
    results on their own. That's a real behavioral difference from the
    eventual M4 graph, not a bug in either the pipeline or this script —
    the actual reducer-based accumulation is exactly the kind of thing
    that should come from wiring the real StateGraph in M4, not from a
    hand-rolled reconstruction in a throwaway verification script.

  - executed_query_ids is reconstructed from the SearchQuery table's
    run_id/query_id/provider columns — but that table was, until this
    script and the patched test_m2_full.py, never actually written to
    anywhere in the pipeline (SearchQueryRepo existed since M0 but
    nothing called it). So on a run whose loop-0 search happened before
    this fix, this will faithfully reconstruct... nothing, and the
    repair planner won't know what was already searched. Re-run
    test_m2_full.py fresh (it now persists SearchQuery rows) before
    relying on this. This script also persists its OWN loop's executed
    pairs at the end of its search step, so a hypothetical loop 2 run
    against the same run_id would see loop 1's queries correctly.

  - The repair-loop planner call can restate legal_issues with slightly
    different wording than loop 0's plan. gap_check's coverage counting
    is keyed by issue TEXT, so if that happens, loop-0 evidence's
    supports_issues (indexed against loop-0's issue list) won't line up
    against loop-1's issue list in the final coverage table. This is a
    pre-existing design property of the coverage/supports_issues scheme,
    not something introduced here — worth knowing when reading the
    final coverage numbers, not something this script tries to paper
    over.
"""
from __future__ import annotations

import asyncio
import sys
from collections import Counter
from datetime import datetime

from sqlalchemy import select

from app.core.budget import BudgetGuard
from app.core.logging import get_logger
from app.db.models import (
    EvidenceRow,
    GapReportRow,
    ResearchPlanRow,
    ResearchRun,
    SearchQuery,
    SourceRow,
)
from app.db.repo.research import (
    EvidenceRepo,
    GapReportRepo,
    PlanRepo,
    RawResultRepo,
    SearchQueryRepo,
    SourceDocumentRepo,
    SourceRepo,
)
from app.db.session import get_db_session
from app.graph.nodes.evaluator import evaluator_node
from app.graph.nodes.extractor import extractor_node
from app.graph.nodes.fetch_dispatch import fetch_dispatch_node
from app.graph.nodes.gap_check import gap_check_node
from app.graph.nodes.grounding import grounding_node
from app.graph.nodes.planner import planner_node
from app.graph.nodes.search_dispatch import search_dispatch_node
from app.graph.nodes.search_merge import search_merge_node
from app.schemas.citation import LegalCitation
from app.schemas.common import CourtLevel, Jurisdiction, SourceType
from app.schemas.evidence import Evidence
from app.schemas.gaps import GapReport
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
        as_of=None,  # not persisted — see test_gap_check_full.py's note
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
                print("No research runs found.")
                return
            run_id = str(run.run_id)

        run_row = await session.get(ResearchRun, run_id)
        if run_row is None:
            print(f"run_id={run_id} not found.")
            return
        request = ResearchRequest.model_validate(run_row.request)
        run_jurisdiction = Jurisdiction(run_row.jurisdiction)

        # Most recent GapReport for this run — must exist and must call
        # for more research, otherwise there's nothing to repair.
        gap_result = await session.execute(
            select(GapReportRow)
            .where(GapReportRow.run_id == run_id)
            .order_by(GapReportRow.loop_index.desc())
            .limit(1)
        )
        gap_row = gap_result.scalar_one_or_none()
        if gap_row is None:
            print("No GapReport persisted for this run — run test_gap_check_full.py first.")
            return
        prev_gap_report = GapReport.model_validate(gap_row.payload)
        prev_loop_index = gap_row.loop_index

        print(f"Using run_id={run_id}  topic={run_row.topic[:60]!r}")
        print(f"Previous verdict (loop {prev_loop_index}): {prev_gap_report.verdict}")
        if prev_gap_report.verdict != "needs_more_research":
            print("Verdict is not 'needs_more_research' — no repair loop to run. Exiting.")
            return

        new_loop_index = prev_loop_index + 1
        if new_loop_index > request.max_research_loops:
            print(
                f"loop {new_loop_index} would exceed max_research_loops="
                f"{request.max_research_loops} — the real pipeline would mark this "
                "'exhausted' rather than loop again. Exiting."
            )
            return

        # Reconstruct executed_query_ids from SearchQuery (see the module
        # docstring — this is only as complete as what test_m2_full.py
        # actually persisted for this run).
        sq_rows = (
            await session.execute(select(SearchQuery).where(SearchQuery.run_id == run_id))
        ).scalars().all()
        executed_query_ids = {f"{r.query_id}:{r.provider}" for r in sq_rows}
        print(f"Reconstructed {len(executed_query_ids)} executed (query_id:provider) pairs.\n")

        # Load the previous plan too — planner_node's repair path reads
        # state["plan"] to keep legal_issues stable across loops (see
        # planner.py). Without this, the repair planner has no way to
        # know what issue list evidence.supports_issues is indexed
        # against, and would silently emit a fresh, differently-worded
        # list that breaks that indexing for every prior loop's evidence.
        prev_plan_row = (
            await session.execute(
                select(ResearchPlanRow)
                .where(ResearchPlanRow.run_id == run_id)
                .order_by(ResearchPlanRow.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        prev_plan_for_state = ResearchPlan.model_validate(prev_plan_row.payload) if prev_plan_row else None

        state = {
            "run_id": run_id,
            "request": request,
            "loop_index": new_loop_index,
            "gap_report": prev_gap_report,
            "executed_query_ids": executed_query_ids,
            "plan": prev_plan_for_state,
        }

        # ---- STEP 1 (repair): planner --------------------------------
        print(f"=== planner_node (repair, loop {new_loop_index}) ===")
        planner_result = await planner_node(state)
        state.update(planner_result)
        plan: ResearchPlan = state["plan"]
        print(f"  -> {len(plan.sub_queries)} new sub-queries, {len(plan.legal_issues)} legal issues")
        if not plan.sub_queries:
            print("  No new queries generated by the repair planner — nothing to search. Exiting.")
            return

        async with get_db_session() as write_session:
            await PlanRepo(write_session).upsert(plan)

        # ---- STEP 2: search + merge -----------------------------------
        print("\n=== search_dispatch_node (repair queries only) ===")
        budget = BudgetGuard(run_id=run_id, ceiling_inr=request.budget_inr)
        search_result = await search_dispatch_node(state, budget=budget)
        state["raw_results"] = search_result.get("raw_results", [])
        print(f"  -> {len(state['raw_results'])} raw results, budget spent so far this loop: INR {budget.spent}")

        executed_pairs = search_result.get("executed_query_ids", set())
        if executed_pairs:
            async with get_db_session() as write_session:
                sq_repo = SearchQueryRepo(write_session)
                for pair in executed_pairs:
                    qid, provider = pair.split(":", 1)
                    result_count = sum(
                        1 for r in state["raw_results"] if r.query_id == qid and r.provider == provider
                    )
                    await sq_repo.upsert(
                        run_id=run_id,
                        query_id=qid,
                        provider=provider,
                        loop_index=new_loop_index,
                        payload={},
                        status="completed",
                        result_count=result_count,
                    )
            print(f"  -> {len(executed_pairs)} executed pairs persisted (for a future loop 2, if any)")

        print("\n=== search_merge_node ===")
        merge_result = search_merge_node(state)
        state["raw_results"] = merge_result["raw_results"]
        print(f"  -> {len(state['raw_results'])} unique candidates after merge")

        async with get_db_session() as write_session:
            await RawResultRepo(write_session).bulk_upsert(state["raw_results"], run_id)

        # ---- STEP 3: evaluate -------------------------------------------
        print("\n=== evaluator_node ===")
        eval_result = await evaluator_node(state)
        state["sources"] = eval_result["sources"]
        state["quota_shortfalls"] = eval_result["quota_shortfalls"]
        new_selected = [s for s in state["sources"] if s.status == "selected"]
        print(f"  -> {len(new_selected)} newly selected, {len(state['sources']) - len(new_selected)} dropped")

        async with get_db_session() as write_session:
            await SourceRepo(write_session).bulk_upsert(state["sources"])

        if not new_selected:
            print("\nNo new sources selected this loop — repair search found nothing usable. Exiting.")
            return

        # ---- STEP 4: fetch + extract + ground ----------------------------
        print("\n=== fetch_dispatch_node ===")
        fetch_result = await fetch_dispatch_node(state)
        updated_sources = fetch_result["sources"]
        documents = fetch_result["documents"]
        fetched = [s for s in updated_sources if s.status == "fetched"]
        print(f"  -> fetched={len(fetched)}/{len(new_selected)}")

        documents_by_source_id = {d.source_id: d for d in documents}
        async with get_db_session() as write_session:
            doc_repo = SourceDocumentRepo(write_session)
            for doc in documents:
                await doc_repo.upsert(doc, run_id)

        print("\n=== extractor_node ===")
        extract_result = await extractor_node(
            sources=updated_sources,
            documents_by_source_id=documents_by_source_id,
            legal_issues=plan.legal_issues,
        )
        new_candidates = extract_result["evidence_candidates"]
        print(f"  -> {len(new_candidates)} new evidence candidates")

        print("\n=== grounding_node ===")
        sources_by_id = {s.source_id: s for s in updated_sources}
        ground_result = await grounding_node(
            candidates=new_candidates,
            sources_by_id=sources_by_id,
            documents_by_source_id=documents_by_source_id,
            run_id=run_id,
            run_jurisdiction=run_jurisdiction,
            loop_index=new_loop_index,
        )
        new_evidence = ground_result["evidence"]
        pass_rate = round(len(new_evidence) / len(new_candidates), 3) if new_candidates else 0.0
        print(f"  -> {len(new_evidence)} new evidence grounded (pass rate {pass_rate})")

        async with get_db_session() as write_session:
            await EvidenceRepo(write_session).bulk_upsert(new_evidence)

        # ---- STEP 5: gap_check again, over ALL evidence so far -----------
        print(f"\n=== gap_check_node (loop {new_loop_index}, cumulative evidence) ===")
        all_evidence_rows = (
            await session.execute(select(EvidenceRow).where(EvidenceRow.run_id == run_id))
        ).scalars().all()
        all_evidence = [_row_to_evidence(r) for r in all_evidence_rows]

        all_source_rows = (
            await session.execute(select(SourceRow).where(SourceRow.run_id == run_id))
        ).scalars().all()
        all_sources = [_row_to_source(r) for r in all_source_rows]

        gap_state = {
            "run_id": run_id,
            "request": request,
            "plan": plan,
            "sources": all_sources,
            "evidence": all_evidence,
            "quota_shortfalls": state.get("quota_shortfalls", []),
            "loop_index": new_loop_index,
        }
        gap_result = await gap_check_node(gap_state)
        new_gap_report: GapReport = gap_result["gap_report"]

        async with get_db_session() as write_session:
            await GapReportRepo(write_session).upsert(new_gap_report)

        # ---- Summary --------------------------------------------------
        print("\n" + "=" * 60)
        print("REPAIR LOOP SUMMARY")
        print("=" * 60)
        print(f"Loop {prev_loop_index} verdict: {prev_gap_report.verdict}")
        print(f"Loop {new_loop_index} verdict: {new_gap_report.verdict}")
        print(f"Loop {new_loop_index} rationale: {new_gap_report.rationale}\n")

        print("Coverage comparison (issue: before -> after):")
        for issue in new_gap_report.coverage:
            before = prev_gap_report.coverage.get(issue, "?")
            after = new_gap_report.coverage[issue]
            print(f"  {issue[:70]}: {before} -> {after}")

        prev_blocking = {g.gap_kind for g in prev_gap_report.gaps if g.severity == "blocking"}
        new_blocking = {g.gap_kind for g in new_gap_report.gaps if g.severity == "blocking"}
        closed = prev_blocking - new_blocking
        still_open = prev_blocking & new_blocking
        newly_found = new_blocking - prev_blocking
        print(f"\nBlocking gaps closed this loop: {closed or 'none'}")
        print(f"Blocking gaps still open: {still_open or 'none'}")
        print(f"New blocking gaps surfaced: {newly_found or 'none'}")

        print(f"\nNew sources this loop: {len(new_selected)} selected, {len(fetched)} fetched")
        print(f"New evidence this loop: {len(new_evidence)}")
        print(f"Total evidence now: {len(all_evidence)}")

        print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
