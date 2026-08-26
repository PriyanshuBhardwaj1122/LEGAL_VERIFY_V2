"""Conditional edge / fan-out routing functions for the research graph."""

from __future__ import annotations

from app.core.logging import get_logger
from app.schemas.state import ResearchState

log = get_logger()


def fan_out_searches(state: ResearchState) -> list[dict]:
    """STEP 2a — expand each sub_query x provider into an independent
    search_worker branch. Filters mechanically against executed_query_ids
    so a repair loop never re-runs a query the planner failed to exclude.

    Returns a list of payload dicts (LangGraph Send targets in the real
    graph; here we return plain payloads so this can be driven directly
    by an orchestrator without requiring the full compiled StateGraph).
    """
    plan = state["plan"]
    executed = state.get("executed_query_ids", set())
    run_id = state["run_id"]

    todo: list[dict] = []
    for q in plan.sub_queries:
        for provider in q.providers:
            key = f"{q.query_id}:{provider}"
            if key in executed:
                log.debug("fan_out_skip_executed", key=key)
                continue
            todo.append({"run_id": run_id, "sub_query": q, "provider": provider})

    log.info("fan_out_searches", branch_count=len(todo), query_count=len(plan.sub_queries))
    return todo


def fan_out_fetches(state: ResearchState) -> list[dict]:
    """STEP 4a — expand each selected source into an independent
    fetch_worker branch."""
    sources = [s for s in state.get("sources", []) if s.status == "selected"]
    run_id = state["run_id"]

    todo = [{"run_id": run_id, "source": s} for s in sources]
    log.info("fan_out_fetches", branch_count=len(todo))
    return todo


def route_after_gaps(state: ResearchState) -> str:
    """Router after gap_check: loop back to planner or end the run."""
    gap_report = state.get("gap_report")
    if gap_report and gap_report.verdict == "needs_more_research":
        return "planner"
    return "end"