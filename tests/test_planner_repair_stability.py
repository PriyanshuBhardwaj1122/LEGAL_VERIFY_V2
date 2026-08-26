"""Verifies planner_node's repair-loop legal_issues stability fix: a
repair-loop plan must preserve the previous plan's issue list verbatim,
in order, appending only genuinely new issues — never reordering,
rewording, or dropping existing ones. This is what keeps
Evidence.supports_issues (1-based indices) valid across loops; without
it, a repair loop silently orphans every earlier loop's evidence from
gap_check's coverage counting.

Run: PYTHONPATH=. python tests/test_planner_repair_stability.py
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.nodes.planner import LLMResearchPlan, LLMSubQuery, planner_node
from app.schemas.common import Jurisdiction, QueryIntent, SourceType
from app.schemas.gaps import Gap, GapKind, GapReport
from app.schemas.plan import ResearchPlan, SubQuery
from app.schemas.request import ArticleConfig, ResearchRequest

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _sub_query(text="q") -> LLMSubQuery:
    return LLMSubQuery(
        query_text=text,
        intent=QueryIntent.CASE_LAW,
        rationale="test",
        target_source_types=[SourceType.JUDGMENT],
    )


async def test_repair_preserves_issue_order_and_appends_new():
    print("planner_node() — repair loop preserves legal_issues order, appends new ones")

    request = ResearchRequest(
        topic="Eligibility criteria under Section 29A of the Insolvency and Bankruptcy Code, 2016",
        practice_area="insolvency",
        article_config=ArticleConfig(article_type="explainer", target_words=2000),
        max_research_loops=2,
    )

    prev_plan = ResearchPlan(
        plan_id="plan-0",
        run_id="run-1",
        loop_index=0,
        topic_restated="test topic",
        legal_issues=["Issue Alpha", "Issue Beta", "Issue Gamma"],
        key_instruments=[],
        sub_queries=[
            SubQuery(
                query_text=f"q{i}", intent=QueryIntent.CASE_LAW, rationale="r",
                target_source_types=[SourceType.JUDGMENT], jurisdiction=Jurisdiction.IN,
                providers=["tavily"],
            )
            for i in range(4)
        ],
        excluded_directions=[],
    )

    gap_report = GapReport(
        run_id="run-1",
        loop_index=0,
        coverage={"Issue Alpha": 5, "Issue Beta": 0, "Issue Gamma": 1},
        tier_histogram={},
        gaps=[Gap(gap_kind=GapKind.THIN_COVERAGE, issue="Issue Beta", detail="d", severity="blocking", detected_by="rule")],
        verdict="needs_more_research",
        rationale="test",
    )

    # The repair LLM call reorders + rewords the existing issues AND
    # adds one genuinely new one — exactly the failure mode being fixed.
    llm_plan_out = LLMResearchPlan(
        topic_restated="test topic restated",
        legal_issues=["Issue Beta (reworded)", "Issue Gamma", "Issue Alpha", "Issue Delta (new)"],
        key_instruments=[],
        sub_queries=[_sub_query("closing gap on Issue Beta"), _sub_query("q2"), _sub_query("q3")],
        excluded_directions=[],
    )

    state = {
        "run_id": "run-1",
        "request": request,
        "loop_index": 1,
        "gap_report": gap_report,
        "executed_query_ids": set(),
        "plan": prev_plan,
    }

    mock_registry = MagicMock()
    mock_registry.resolve_providers.return_value = ["tavily"]

    with patch("app.graph.nodes.planner.get_llm") as mock_get_llm, \
         patch("app.graph.nodes.planner.get_registry", return_value=mock_registry):
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=(llm_plan_out, {"input_tokens": 10, "output_tokens": 5}))
        mock_get_llm.return_value = mock_llm

        result = await planner_node(state)

    new_plan: ResearchPlan = result["plan"]

    check(
        "original 3 issues preserved, in original order",
        new_plan.legal_issues[:3] == ["Issue Alpha", "Issue Beta", "Issue Gamma"],
        new_plan.legal_issues,
    )
    check(
        "genuinely new issue appended at the end",
        "Issue Delta (new)" in new_plan.legal_issues[3:],
        new_plan.legal_issues,
    )
    check(
        "reworded duplicate ('Issue Beta (reworded)') is NOT a separate entry",
        "Issue Beta (reworded)" not in new_plan.legal_issues,
        new_plan.legal_issues,
    )
    check("total issue count is 4 (3 preserved + 1 new)", len(new_plan.legal_issues) == 4, new_plan.legal_issues)


async def test_loop0_uses_llm_issues_directly():
    print("planner_node() — loop 0 (no gap_report) uses the LLM's issues as-is")

    request = ResearchRequest(
        topic="Some topic that is long enough to pass validation checks",
        practice_area="general",
        article_config=ArticleConfig(article_type="explainer", target_words=1500),
    )

    llm_plan_out = LLMResearchPlan(
        topic_restated="restated",
        legal_issues=["Fresh Issue One", "Fresh Issue Two"],
        key_instruments=[],
        sub_queries=[_sub_query("a"), _sub_query("b"), _sub_query("c"), _sub_query("d")],
        excluded_directions=[],
    )

    state = {
        "run_id": "run-2",
        "request": request,
        "loop_index": 0,
        "executed_query_ids": set(),
    }

    mock_registry = MagicMock()
    mock_registry.resolve_providers.return_value = ["tavily"]

    with patch("app.graph.nodes.planner.get_llm") as mock_get_llm, \
         patch("app.graph.nodes.planner.get_registry", return_value=mock_registry):
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=(llm_plan_out, {"input_tokens": 10, "output_tokens": 5}))
        mock_get_llm.return_value = mock_llm

        result = await planner_node(state)

    new_plan: ResearchPlan = result["plan"]
    check(
        "loop 0 has no prior plan to preserve — uses LLM's issues directly",
        new_plan.legal_issues == ["Fresh Issue One", "Fresh Issue Two"],
        new_plan.legal_issues,
    )


def main():
    asyncio.run(test_repair_preserves_issue_order_and_appends_new())
    asyncio.run(test_loop0_uses_llm_issues_directly())

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
