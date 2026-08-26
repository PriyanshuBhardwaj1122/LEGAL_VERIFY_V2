"""M3 end-to-end verification — real fetch + real OpenAI extraction +
deterministic grounding, run against sources already SELECTED and
persisted by an M2 run.

Usage:
    python test_m3_full.py                # uses the most recent run
    python test_m3_full.py <run_id>        # uses a specific run

What it does:
  1. Loads the run's SELECTED sources from Postgres (from your M2 test).
  2. Runs fetch_dispatch_node — downloads + extracts text for each source,
     persists SourceDocuments via SourceDocumentRepo.
  3. Runs extractor_node — one OpenAI call per fetched source (or per
     chunk for long docs), pulling evidence candidates.
  4. Runs grounding_node — mechanically re-locates every verbatim_quote
     in the stored document text; drops anything that can't be found or
     that fails a §2.6 hard invariant.
  5. Persists surviving Evidence via EvidenceRepo.
  6. Prints a full summary: fetch success rate, candidates per source,
     grounding pass rate, grounding-tier breakdown, binding_strength
     breakdown, and a few example evidence items.

This is the real test — it hits your actual selected URLs (SC judgments,
bare acts, NCLT orders, gov PDFs) and your real OpenAI key, so results
depend on what those sites actually serve today.
"""
from __future__ import annotations

import asyncio
import sys
from collections import Counter
from datetime import datetime

from sqlalchemy import select

from app.core.logging import get_logger
from app.db.models import ResearchPlanRow, ResearchRun, SourceRow
from app.db.repo.research import EvidenceRepo, SourceDocumentRepo
from app.db.session import get_db_session
from app.graph.nodes.extractor import extractor_node
from app.graph.nodes.fetch_dispatch import fetch_dispatch_node
from app.graph.nodes.grounding import grounding_node
from app.schemas.citation import LegalCitation
from app.schemas.common import CourtLevel, Jurisdiction, SourceType
from app.schemas.source import Source, SourceScore
from app.schemas.state import ResearchState

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
        # SourceRow.published_on is a SQL Date column — asyncpg/SQLAlchemy
        # always hands that back as a plain datetime.date, never a
        # datetime.datetime, regardless of the (misleading) `Mapped[datetime]`
        # type hint in models.py. Calling .date() unconditionally only
        # "worked" before because no source had ever carried a real
        # published_on value; now that indiankanoon/serpapi supply real
        # dates, .date() on an already-a-date object raises AttributeError.
        decided_or_published_on=(
            row.published_on.date() if isinstance(row.published_on, datetime) else row.published_on
        ),
        citation=citation,
        score=SourceScore.model_validate(row.score),
        status="selected",
        discovered_by=row.discovered_by or [],
        provenance_query_ids=[],
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
                print("No research runs found in the database. Run an M2 test first.")
                return
            run_id = str(run.run_id)

        run_row = await session.get(ResearchRun, run_id)
        run_jurisdiction = Jurisdiction(run_row.jurisdiction) if run_row else Jurisdiction.IN

        print(f"Using run_id={run_id} (jurisdiction={run_jurisdiction.value})")

        result = await session.execute(
            select(SourceRow).where(
                SourceRow.run_id == run_id, SourceRow.status == "selected"
            )
        )
        rows = result.scalars().all()

        if not rows:
            print(f"No 'selected' sources found for run_id={run_id}. Run M2 first.")
            return

        sources = [_row_to_source(r) for r in rows]
        print(f"Loaded {len(sources)} selected sources.\n")

        # ---- Step 4a/4b: fetch + extract text -------------------------
        state = ResearchState(sources=sources, run_id=run_id)  # type: ignore[call-arg]
        print("=== fetch_dispatch_node ===")
        fetch_result = await fetch_dispatch_node(state)

        updated_sources: list[Source] = fetch_result["sources"]
        documents = fetch_result["documents"]
        fetch_errors = fetch_result["errors"]

        fetched = [s for s in updated_sources if s.status == "fetched"]
        failed = [s for s in updated_sources if s.status == "failed"]
        print(f"fetched={len(fetched)}  failed={len(failed)}  errors={len(fetch_errors)}")
        for s in failed:
            print(f"  FAILED: {s.title or s.url_canonical}")

        documents_by_source_id = {d.source_id: d for d in documents}

        doc_repo = SourceDocumentRepo(session)
        for doc in documents:
            await doc_repo.upsert(doc, run_id)
        print(f"Persisted {len(documents)} SourceDocuments.\n")

        # ---- Step 4c: extractor (real OpenAI calls) --------------------
        print("=== extractor_node (real LLM calls, one per source/chunk) ===")
        plan_result = await session.execute(
            select(ResearchPlanRow)
            .where(ResearchPlanRow.run_id == run_id)
            .order_by(ResearchPlanRow.created_at.desc())
            .limit(1)
        )
        plan_row = plan_result.scalar_one_or_none()
        if plan_row and plan_row.payload.get("legal_issues"):
            legal_issues = plan_row.payload["legal_issues"]
        else:
            legal_issues = [run_row.topic if run_row else "the researched legal issue"]
        extract_result = await extractor_node(
            sources=updated_sources,
            documents_by_source_id=documents_by_source_id,
            legal_issues=legal_issues,
        )
        candidates = extract_result["evidence_candidates"]
        print(f"candidates={len(candidates)}\n")

        by_source: Counter[str] = Counter(c.source_id for c in candidates)
        for sid, count in by_source.most_common():
            title = next((s.title for s in updated_sources if s.source_id == sid), sid)
            print(f"  {count:2d}  {title}")

        # ---- Step 4d: grounding (deterministic, no LLM) -----------------
        print("\n=== grounding_node ===")
        sources_by_id = {s.source_id: s for s in updated_sources}
        ground_result = await grounding_node(
            candidates=candidates,
            sources_by_id=sources_by_id,
            documents_by_source_id=documents_by_source_id,
            run_id=run_id,
            run_jurisdiction=run_jurisdiction,
            loop_index=0,
        )
        evidence = ground_result["evidence"]
        ground_errors = ground_result["errors"]

        evidence_repo = EvidenceRepo(session)
        await evidence_repo.bulk_upsert(evidence)

        # ---- Summary ------------------------------------------------
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Sources selected:   {len(sources)}")
        print(f"Sources fetched:    {len(fetched)} / {len(sources)}")
        print(f"Evidence candidates: {len(candidates)}")
        print(f"Evidence grounded:   {len(evidence)}")
        pass_rate = round(len(evidence) / len(candidates), 3) if candidates else 0.0
        print(f"Grounding pass rate: {pass_rate}")
        print(f"Grounding drops (ungrounded + invariant): {len(candidates) - len(evidence)}")

        kind_counts = Counter(e.kind for e in evidence)
        binding_counts = Counter(e.binding_strength for e in evidence)
        print(f"\nBy kind: {dict(kind_counts)}")
        print(f"By binding_strength: {dict(binding_counts)}")

        print("\nSample evidence (up to 5):")
        for e in evidence[:5]:
            print(f"  [{e.binding_strength}/{e.kind}] {e.statement}")
            print(f"     quote: {(e.verbatim_quote or '')[:140]}")
            print(f"     source_id={e.source_id}  pinpoint={e.pinpoint}")

        if ground_errors:
            print(f"\nDropped at grounding ({len(ground_errors)}):")
            for err in ground_errors[:10]:
                print(f"  - {err.detail}")

        print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
