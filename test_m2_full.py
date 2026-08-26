"""M2 end-to-end smoke test: Planner -> Search (Tavily+Perplexity) -> Merge -> Evaluate.

Run from the project root with the venv active and Docker Postgres up:
    python test_m2_full.py

This now PERSISTS every stage to Postgres (run, plan, raw_results,
selected sources) so downstream scripts (test_m3_full.py) can pick up
this run's selected sources by run_id.
"""

import asyncio
import uuid
from decimal import Decimal

from app.core.budget import BudgetGuard
from app.db.repo.research import (
    PlanRepo,
    RawResultRepo,
    ResearchRunRepo,
    SearchQueryRepo,
    SourceRepo,
)
from app.db.session import get_db_session
from app.graph.nodes.planner import planner_node
from app.graph.nodes.search_dispatch import search_dispatch_node
from app.graph.nodes.search_merge import search_merge_node
from app.graph.nodes.evaluator import evaluator_node
from app.schemas.request import ArticleConfig, ResearchRequest
from app.schemas.state import ResearchState


async def main():
    run_id = str(uuid.uuid4())
    print(f"=== Run ID: {run_id} ===\n")

    request = ResearchRequest(
        topic="Eligibility criteria under Section 29A of the Insolvency and Bankruptcy Code, 2016",
        practice_area="insolvency",
        article_config=ArticleConfig(article_type="explainer", target_words=2000),
        budget_inr=Decimal("50.00"),
    )

    state: ResearchState = {
        "run_id": run_id,
        "request": request,
        "loop_index": 0,
        "executed_query_ids": set(),
    }

    # ------------------------------------------------------------------
    # Persist the run row up front so everything below has a valid FK.
    # ------------------------------------------------------------------
    async with get_db_session() as session:
        await ResearchRunRepo(session).create_run(run_id, request)
    print(f"  -> run persisted to Postgres\n")

    # ------------------------------------------------------------------
    # STEP 1: Planner
    # ------------------------------------------------------------------
    print("[STEP 1] Running planner...")
    result = await planner_node(state)
    state.update(result)
    plan = state["plan"]
    print(f"  -> {len(plan.sub_queries)} sub-queries, {len(plan.legal_issues)} legal issues")

    async with get_db_session() as session:
        await PlanRepo(session).upsert(plan)
    print("  -> plan persisted to Postgres\n")

    # ------------------------------------------------------------------
    # STEP 2a/2b: Search dispatch (fan-out to Tavily/Perplexity)
    # ------------------------------------------------------------------
    print("[STEP 2] Running search dispatch (real Tavily + Perplexity calls)...")
    budget = BudgetGuard(run_id=run_id, ceiling_inr=request.budget_inr)
    result = await search_dispatch_node(state, budget=budget)
    state["raw_results"] = result.get("raw_results", [])
    state["executed_query_ids"] = state.get("executed_query_ids", set()) | result.get(
        "executed_query_ids", set()
    )
    print(f"  -> {len(state['raw_results'])} raw results")
    print(f"  -> budget spent so far: INR {budget.spent}\n")

    # Persist which (query_id, provider) pairs actually ran — this is
    # what a later repair loop reads back to avoid re-issuing an
    # identical search. Previously nothing ever wrote to this table.
    executed_pairs = result.get("executed_query_ids", set())
    if executed_pairs:
        async with get_db_session() as session:
            repo = SearchQueryRepo(session)
            for pair in executed_pairs:
                qid, provider = pair.split(":", 1)
                result_count = sum(1 for r in state["raw_results"] if r.query_id == qid and r.provider == provider)
                await repo.upsert(
                    run_id=run_id,
                    query_id=qid,
                    provider=provider,
                    loop_index=state.get("loop_index", 0),
                    payload={},
                    status="completed",
                    result_count=result_count,
                )
        print(f"  -> {len(executed_pairs)} executed (query_id, provider) pairs persisted to Postgres\n")

    if result.get("errors"):
        print(f"  Errors during search ({len(result['errors'])}):")
        for e in result["errors"][:5]:
            print(f"    [{e.kind}] {e.detail}")
        print()

    # ------------------------------------------------------------------
    # STEP 2c: Merge / dedupe
    # ------------------------------------------------------------------
    print("[STEP 2c] Merging and deduping...")
    result = search_merge_node(state)
    state["raw_results"] = result["raw_results"]
    print(f"  -> {len(state['raw_results'])} unique candidates after merge\n")

    async with get_db_session() as session:
        await RawResultRepo(session).bulk_upsert(state["raw_results"], run_id)
    print("  -> raw_results persisted to Postgres\n")

    for r in state["raw_results"][:10]:
        print(f"    [{','.join(r.discovered_by)}] {r.title or '(no title)'}")
        print(f"      {r.url_canonical}")

    # ------------------------------------------------------------------
    # STEP 3: Evaluate (deterministic + LLM relevance + selection)
    # ------------------------------------------------------------------
    print("\n[STEP 3] Evaluating and selecting sources...")
    result = await evaluator_node(state)
    state["sources"] = result["sources"]
    state["quota_shortfalls"] = result["quota_shortfalls"]

    selected = [s for s in state["sources"] if s.status == "selected"]
    dropped = [s for s in state["sources"] if s.status == "dropped"]

    print(f"  -> {len(selected)} selected, {len(dropped)} dropped\n")

    async with get_db_session() as session:
        await SourceRepo(session).bulk_upsert(state["sources"])
    print(f"  -> {len(state['sources'])} sources persisted to Postgres ({len(selected)} selected)\n")

    print("Selected sources (ranked):")
    for s in sorted(selected, key=lambda s: s.score.composite, reverse=True):
        print(
            f"  [tier{s.score.tier}] composite={s.score.composite:.2f} "
            f"authority={s.score.authority:.2f} relevance={s.score.relevance:.2f} "
            f"| {s.title or '(no title)'}"
        )
        print(f"      {s.url_canonical}")

    if state["quota_shortfalls"]:
        print(f"\nQuota shortfalls ({len(state['quota_shortfalls'])}):")
        for sf in state["quota_shortfalls"]:
            print(f"  [{sf.requirement}] {sf.detail}")

    print(f"\nTotal budget spent: INR {budget.spent} / {request.budget_inr}")
    print(f"\n=== SUCCESS: M2 pipeline (planner -> search -> merge -> evaluate) verified ===")
    print(f"=== run_id={run_id}  -- pass this to test_m3_full.py, or it'll pick it up automatically as the most recent run ===")


if __name__ == "__main__":
    asyncio.run(main())
