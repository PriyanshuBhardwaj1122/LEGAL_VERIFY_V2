"""Synthetic-data tests for Step 5 (gap check) — no network/API calls
for the pure logic; the LLM detector is exercised with a mocked call.

Run: PYTHONPATH=. python tests/test_gap_check_synthetic.py
"""
from __future__ import annotations

import asyncio
import sys
from datetime import date
from unittest.mock import AsyncMock, patch

from app.domain.coverage import compute_coverage, compute_tier_histogram, detect_rule_based_gaps
from app.graph.nodes.gap_check import LLMGap, LLMGapBatch, _compute_verdict, gap_check_node
from app.schemas.common import CourtLevel, Jurisdiction, QueryIntent, SourceType
from app.schemas.evidence import Evidence, QuotaShortfall
from app.schemas.gaps import GapKind
from app.schemas.plan import ResearchPlan, SubQuery
from app.schemas.request import ArticleConfig, ResearchRequest
from app.schemas.source import Source, SourceScore

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


def _evidence(
    evidence_id: str,
    source_id: str,
    kind: str = "holding",
    tier: int = 1,
    supports_issues=None,
    binding_strength: str = "binding",
    as_of=None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        run_id="run-1",
        source_id=source_id,
        loop_index=0,
        kind=kind,  # type: ignore[arg-type]
        statement="A statement.",
        verbatim_quote="a quote",
        quote_start=0,
        quote_end=7,
        grounding="exact",
        pinpoint=None,
        citation=None,
        supports_issues=supports_issues or [],
        jurisdiction=Jurisdiction.IN,
        court_level=CourtLevel.NONE,
        as_of=as_of,
        binding_strength=binding_strength,  # type: ignore[arg-type]
        llm_confidence=0.9,
        authority_tier=tier,
    )


def _source(source_id: str, source_type=SourceType.JUDGMENT, title="Some Source", court_level=CourtLevel.NONE) -> Source:
    return Source(
        source_id=source_id,
        run_id="run-1",
        url_canonical=f"https://example.com/{source_id}",
        fetch_url=None,
        title=title,
        source_type=source_type,
        court_level=court_level,
        issuing_body=None,
        jurisdiction=Jurisdiction.IN,
        decided_or_published_on=None,
        citation=None,
        score=SourceScore(authority=0.5, relevance=0.5, recency=1.0, reliability=0.5, composite=0.5, tier=3, reasons=[], dropped_reason=None),
        status="fetched",
        discovered_by=["tavily"],
        provenance_query_ids=[],
    )


def _plan(legal_issues, key_instruments=None, intents=None) -> ResearchPlan:
    intents = intents or [QueryIntent.CASE_LAW, QueryIntent.STATUTE_TEXT]
    sub_queries = [
        SubQuery(
            query_text=f"query {i}",
            intent=intent,
            rationale="test",
            target_source_types=[SourceType.JUDGMENT],
            jurisdiction=Jurisdiction.IN,
            providers=["tavily"],
        )
        for i, intent in enumerate(intents)
    ]
    while len(sub_queries) < 4:
        sub_queries.append(
            SubQuery(
                query_text=f"filler {len(sub_queries)}",
                intent=QueryIntent.BACKGROUND,
                rationale="test",
                target_source_types=[SourceType.OTHER],
                jurisdiction=Jurisdiction.IN,
                providers=["tavily"],
            )
        )
    return ResearchPlan(
        plan_id="plan-1",
        run_id="run-1",
        loop_index=0,
        topic_restated="test topic",
        legal_issues=legal_issues,
        key_instruments=key_instruments or [],
        sub_queries=sub_queries,
        excluded_directions=[],
    )


def test_compute_coverage():
    print("compute_coverage()")
    plan_issues = ["Issue A", "Issue B", "Issue C"]
    evidence = [
        _evidence("e1", "s1", supports_issues=["1"]),
        _evidence("e2", "s1", supports_issues=["1", "2"]),
        _evidence("e3", "s1", supports_issues=[]),
    ]
    coverage = compute_coverage(evidence, plan_issues)
    check("issue A has 2 supporting items", coverage["Issue A"] == 2, coverage)
    check("issue B has 1 supporting item", coverage["Issue B"] == 1, coverage)
    check("issue C has 0 supporting items", coverage["Issue C"] == 0, coverage)


def test_compute_tier_histogram():
    print("compute_tier_histogram()")
    evidence = [_evidence("e1", "s1", tier=1), _evidence("e2", "s1", tier=1), _evidence("e3", "s1", tier=3)]
    hist = compute_tier_histogram(evidence)
    check("tier1 count is 2", hist[1] == 2, hist)
    check("tier3 count is 1", hist[3] == 1, hist)


def test_rule_missing_statutory_basis():
    print("detect_rule_based_gaps() — missing statutory basis")
    plan = _plan(["Issue A"], key_instruments=["Insolvency and Bankruptcy Code, 2016"])
    # No evidence at all references a STATUTE-typed source with a matching title.
    sources_by_id = {"s1": _source("s1", source_type=SourceType.JUDGMENT)}
    evidence = [_evidence("e1", "s1", kind="holding", supports_issues=["1"])]
    gaps = detect_rule_based_gaps(evidence, sources_by_id, plan, [])
    kinds = [g.gap_kind for g in gaps]
    check("flags MISSING_STATUTORY_BASIS", GapKind.MISSING_STATUTORY_BASIS in kinds, kinds)


def test_rule_statutory_basis_satisfied():
    print("detect_rule_based_gaps() — statutory basis present, no false positive")
    plan = _plan(["Issue A"], key_instruments=["Insolvency and Bankruptcy Code, 2016"])
    sources_by_id = {
        "s1": _source("s1", source_type=SourceType.STATUTE, title="Insolvency and Bankruptcy Code, 2016")
    }
    evidence = [_evidence("e1", "s1", kind="statutory_text", supports_issues=["1"])]
    gaps = detect_rule_based_gaps(evidence, sources_by_id, plan, [])
    kinds = [g.gap_kind for g in gaps]
    check("does NOT flag MISSING_STATUTORY_BASIS when statute evidence exists", GapKind.MISSING_STATUTORY_BASIS not in kinds, kinds)


def test_rule_missing_primary_authority():
    print("detect_rule_based_gaps() — missing primary authority")
    plan = _plan(["Issue A"])
    sources_by_id = {"s1": _source("s1", source_type=SourceType.FIRM_COMMENTARY)}
    evidence = [_evidence("e1", "s1", kind="commentary_opinion", tier=5, supports_issues=["1"], binding_strength="opinion")]
    gaps = detect_rule_based_gaps(evidence, sources_by_id, plan, [])
    kinds = [g.gap_kind for g in gaps]
    check("flags MISSING_PRIMARY_AUTHORITY when nothing is tier<=2", GapKind.MISSING_PRIMARY_AUTHORITY in kinds, kinds)
    blocking = [g for g in gaps if g.gap_kind == GapKind.MISSING_PRIMARY_AUTHORITY]
    check("MISSING_PRIMARY_AUTHORITY is blocking severity", blocking and blocking[0].severity == "blocking")


def test_rule_thin_coverage_zero_and_low():
    print("detect_rule_based_gaps() — thin coverage")
    plan = _plan(["Issue A", "Issue B"])
    sources_by_id = {"s1": _source("s1", source_type=SourceType.STATUTE, title="x")}
    evidence = [
        _evidence("e1", "s1", tier=1, supports_issues=["1"]),  # Issue A: 1 item -> important
        # Issue B: 0 items -> blocking
    ]
    gaps = detect_rule_based_gaps(evidence, sources_by_id, plan, [])
    thin = [g for g in gaps if g.gap_kind == GapKind.THIN_COVERAGE]
    by_issue = {g.issue: g.severity for g in thin}
    check("Issue A (1 item) flagged important", by_issue.get("Issue A") == "important", by_issue)
    check("Issue B (0 items) flagged blocking", by_issue.get("Issue B") == "blocking", by_issue)


def test_rule_quota_shortfall_promoted():
    print("detect_rule_based_gaps() — quota shortfalls promoted to gaps")
    plan = _plan(["Issue A"])
    shortfalls = [
        QuotaShortfall(requirement="min_tier12", detail="Only 1 tier-1/2 source found (need 3)"),
    ]
    gaps = detect_rule_based_gaps([], {}, plan, shortfalls)
    kinds = [g.gap_kind for g in gaps]
    check("promotes min_tier12 shortfall to MISSING_PRIMARY_AUTHORITY gap", GapKind.MISSING_PRIMARY_AUTHORITY in kinds, kinds)


def test_verdict_complete():
    print("_compute_verdict() — complete")
    verdict, rationale = _compute_verdict([], loop_index=0, max_research_loops=2)
    check("no gaps -> complete", verdict == "complete", verdict)


def test_verdict_needs_more_research():
    print("_compute_verdict() — needs_more_research")
    from app.schemas.gaps import Gap

    gaps = [Gap(gap_kind=GapKind.THIN_COVERAGE, issue="X", detail="d", severity="blocking", detected_by="rule")]
    verdict, rationale = _compute_verdict(gaps, loop_index=0, max_research_loops=2)
    check("blocking gap within loop budget -> needs_more_research", verdict == "needs_more_research", verdict)


def test_verdict_exhausted():
    print("_compute_verdict() — exhausted")
    from app.schemas.gaps import Gap

    gaps = [Gap(gap_kind=GapKind.THIN_COVERAGE, issue="X", detail="d", severity="blocking", detected_by="rule")]
    verdict, rationale = _compute_verdict(gaps, loop_index=2, max_research_loops=2)
    check("blocking gap at/past loop budget -> exhausted", verdict == "exhausted", verdict)


def test_verdict_important_only_is_complete():
    print("_compute_verdict() — important-only gaps still verdict complete")
    from app.schemas.gaps import Gap

    gaps = [Gap(gap_kind=GapKind.MISSING_APEX_RULING, issue=None, detail="d", severity="important", detected_by="rule")]
    verdict, rationale = _compute_verdict(gaps, loop_index=0, max_research_loops=2)
    check("only important/nice_to_have gaps -> still complete (non-blocking)", verdict == "complete", verdict)


async def test_gap_check_node_wiring():
    print("gap_check_node() — full wiring with mocked LLM")
    request = ResearchRequest(
        topic="Eligibility criteria under Section 29A of the Insolvency and Bankruptcy Code, 2016",
        practice_area="insolvency",
        article_config=ArticleConfig(article_type="explainer", target_words=2000),
        max_research_loops=2,
    )
    plan = _plan(["Issue A", "Issue B"], key_instruments=["Insolvency and Bankruptcy Code, 2016"])
    sources = [_source("s1", source_type=SourceType.STATUTE, title="Insolvency and Bankruptcy Code, 2016")]
    evidence = [_evidence("e1", "s1", kind="statutory_text", tier=1, supports_issues=["1"])]

    state = {
        "run_id": "run-1",
        "request": request,
        "plan": plan,
        "sources": sources,
        "evidence": evidence,
        "quota_shortfalls": [],
        "loop_index": 0,
    }

    mock_batch = LLMGapBatch(gaps=[])
    with patch("app.graph.nodes.gap_check.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=(mock_batch, {"input_tokens": 10, "output_tokens": 5}))
        mock_get_llm.return_value = mock_llm

        result = await gap_check_node(state)

    report = result["gap_report"]
    check("gap_report is returned", report is not None)
    check("gap_history contains this report", result["gap_history"] == [report])
    check("coverage computed for both issues", set(report.coverage.keys()) == {"Issue A", "Issue B"}, report.coverage)
    check("Issue B (uncovered) produces a blocking gap -> needs_more_research", report.verdict == "needs_more_research", report.verdict)


def main():
    test_compute_coverage()
    test_compute_tier_histogram()
    test_rule_missing_statutory_basis()
    test_rule_statutory_basis_satisfied()
    test_rule_missing_primary_authority()
    test_rule_thin_coverage_zero_and_low()
    test_rule_quota_shortfall_promoted()
    test_verdict_complete()
    test_verdict_needs_more_research()
    test_verdict_exhausted()
    test_verdict_important_only_is_complete()
    asyncio.run(test_gap_check_node_wiring())

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
