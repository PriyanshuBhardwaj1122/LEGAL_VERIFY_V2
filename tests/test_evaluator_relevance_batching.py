"""Verifies the Stage B relevance-scoring chunking fix in
app/graph/nodes/evaluator.py.

Real-run bug this locks in: with indiankanoon+serpapi both wired in,
search_merge routinely produces 150+ candidates now. The old
score_relevance_llm sent ALL of them in one LLM call; the output (one
item per candidate) blew past max_tokens, OpenAI truncated mid-JSON,
the parse failed, and every single candidate silently got the same
neutral relevance=0.4 fallback — which clears MIN_RELEVANCE_FOR_HARD_QUOTA
(0.35) for everyone at once, so the relevance floor stops doing
anything and authority alone decides selection again (the exact
wrong-statute contamination bug fixed earlier this session, reappearing
via a completely different mechanism).

Run: PYTHONPATH=. python tests/test_evaluator_relevance_batching.py
"""
from __future__ import annotations

import asyncio
import re
import sys
from unittest.mock import AsyncMock, patch

from app.graph.nodes.evaluator import (
    RELEVANCE_BATCH_SIZE,
    RelevanceBatch,
    RelevanceItem,
    score_relevance_llm,
)
from app.schemas.common import CourtLevel, Jurisdiction, QueryIntent, SourceType
from app.schemas.plan import ResearchPlan, SubQuery
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


def _plan() -> ResearchPlan:
    return ResearchPlan(
        plan_id="plan-1", run_id="run-1", loop_index=0,
        topic_restated="test topic", legal_issues=["issue one"], key_instruments=[],
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


def _candidates(n: int) -> list[Source]:
    out = []
    for i in range(n):
        sid = f"src-{i:03d}"
        out.append(
            Source(
                source_id=sid, run_id="run-1", url_canonical=f"https://example.com/{sid}",
                fetch_url=None, title=f"Source {i}", source_type=SourceType.JUDGMENT,
                court_level=CourtLevel.NONE, issuing_body=None, jurisdiction=Jurisdiction.IN,
                decided_or_published_on=None, citation=None,
                score=SourceScore(authority=0.5, relevance=0.0, recency=1.0, reliability=0.5, composite=0.0, tier=3, reasons=[]),
                status="candidate", discovered_by=["tavily"], provenance_query_ids=["q1"],
            )
        )
    return out


async def test_large_pool_splits_into_multiple_batches():
    print("score_relevance_llm() — a 60-candidate pool splits into ceil(60/25)=3 batches")
    candidates = _candidates(60)
    call_sizes: list[int] = []

    async def fake_generate(**kwargs):
        # Each candidate line looks like "src-000 | judgment | tier3 | ...".
        ids_in_batch = re.findall(r"src-\d{3}(?= \|)", kwargs["user"])
        call_sizes.append(len(ids_in_batch))
        batch = RelevanceBatch(scores=[
            RelevanceItem(source_id=sid, relevance=0.8, addresses_issues=["1"], one_line_reason="on point")
            for sid in ids_in_batch
        ])
        return batch, {"input_tokens": 10, "output_tokens": 10}

    with patch("app.graph.nodes.evaluator.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=fake_generate)
        mock_get_llm.return_value = mock_llm
        result = await score_relevance_llm(candidates, {}, _plan())

    check("3 LLM calls made for 60 candidates at batch size 25", mock_llm.generate.call_count == 3, mock_llm.generate.call_count)
    check("batch sizes are 25, 25, 10", sorted(call_sizes) == [10, 25, 25], call_sizes)
    check("every candidate got scored", len(result) == 60, len(result))
    check("real (non-fallback) relevance came through", all(v.relevance == 0.8 for v in result.values()), {k: v.relevance for k, v in list(result.items())[:3]})
    check("RELEVANCE_BATCH_SIZE is 25 as documented", RELEVANCE_BATCH_SIZE == 25)


async def test_one_batch_failure_does_not_poison_other_batches():
    print("score_relevance_llm() — a parse failure in one batch only neutral-defaults THAT batch")
    candidates = _candidates(50)  # -> 2 batches of 25
    call_count = 0

    async def fake_generate(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Unterminated string starting at: line 1 column 13186 (char 13185)")
        batch = RelevanceBatch(scores=[
            RelevanceItem(source_id=f"src-{j:03d}", relevance=0.9, addresses_issues=["1"], one_line_reason="on point")
            for j in range(25, 50)
        ])
        return batch, {"input_tokens": 10, "output_tokens": 10}

    with patch("app.graph.nodes.evaluator.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=fake_generate)
        mock_get_llm.return_value = mock_llm
        result = await score_relevance_llm(candidates, {}, _plan())

    check("all 50 candidates still get some score", len(result) == 50, len(result))
    first_batch_relevances = {result[f"src-{j:03d}"].relevance for j in range(0, 25)}
    second_batch_relevances = {result[f"src-{j:03d}"].relevance for j in range(25, 50)}
    check("failed batch (first 25) got the neutral 0.4 fallback", first_batch_relevances == {0.4}, first_batch_relevances)
    check("succeeding batch (last 25) kept its REAL scores, not neutral", second_batch_relevances == {0.9}, second_batch_relevances)
    check(
        "fallback is confined to ~half the run, not silently applied to everyone",
        second_batch_relevances != {0.4},
    )


async def test_empty_candidates_short_circuits():
    print("score_relevance_llm() — no candidates means no LLM call at all")
    with patch("app.graph.nodes.evaluator.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_get_llm.return_value = mock_llm
        result = await score_relevance_llm([], {}, _plan())
    check("empty result", result == {})
    check("get_llm never called", mock_llm.generate.call_count == 0)


def main():
    asyncio.run(test_large_pool_splits_into_multiple_batches())
    asyncio.run(test_one_batch_failure_does_not_poison_other_batches())
    asyncio.run(test_empty_candidates_short_circuits())

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
