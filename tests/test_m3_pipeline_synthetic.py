"""Synthetic end-to-end wiring test: extractor_node -> grounding_node,
with the LLM call mocked so no network/API keys are needed. Confirms
the data actually flows correctly across the node boundary (field
names, types, EvidenceCandidate -> Evidence construction).

Run: PYTHONPATH=. python tests/test_m3_pipeline_synthetic.py
"""
from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import AsyncMock, patch

from app.graph.nodes.extractor import extractor_node, LLMEvidenceBatch, LLMEvidenceCandidate
from app.graph.nodes.grounding import grounding_node
from app.schemas.common import CourtLevel, Jurisdiction, SourceType
from app.schemas.source import Source, SourceDocument, SourceScore


def make_source(source_id: str, tier: int, status: str = "fetched") -> Source:
    return Source(
        source_id=source_id,
        run_id="run-1",
        url_canonical="https://main.sci.gov.in/judgment/123.pdf",
        fetch_url="https://main.sci.gov.in/judgment/123.pdf",
        title="ABC Ltd v. XYZ Bank",
        source_type=SourceType.JUDGMENT,
        court_level=CourtLevel.SUPREME_COURT,
        issuing_body="Supreme Court of India",
        jurisdiction=Jurisdiction.IN,
        decided_or_published_on=date(2023, 5, 1),
        citation=None,
        score=SourceScore(
            authority=0.95, relevance=0.8, recency=0.5, reliability=0.9,
            composite=0.85, tier=tier, reasons=["tier1_court"], dropped_reason=None,
        ),
        status=status,
        discovered_by=["tavily"],
        provenance_query_ids=["q1"],
    )


DOC_TEXT = (
    "The Adjudicating Authority held that a related party under section "
    "5(24) of the Insolvency and Bankruptcy Code, 2016 shall not be "
    "eligible to submit a resolution plan under section 29A unless the "
    "disqualification is cured. This principle was affirmed in para 42."
)


async def main():
    source = make_source("src-1", tier=1)
    doc = SourceDocument(
        source_id="src-1",
        content_hash="deadbeef",
        mime="application/pdf",
        char_count=len(DOC_TEXT),
        extraction_method="pymupdf",
        extraction_confidence=0.95,
        is_quotable=True,
        language="en",
        text=DOC_TEXT,
    )

    # Mock the LLM to return one grounded candidate and one hallucinated
    # (non-existent) candidate, to check grounding actually drops the bad one.
    mock_batch = LLMEvidenceBatch(
        items=[
            LLMEvidenceCandidate(
                kind="holding",
                statement="A related party is ineligible under s.29A absent cure",
                verbatim_quote=(
                    "a related party under section 5(24) of the Insolvency "
                    "and Bankruptcy Code, 2016 shall not be eligible to "
                    "submit a resolution plan under section 29A"
                ),
                pinpoint="para 42",
                citation_raw=None,
                supports_issues=["issue-1"],
                llm_confidence=0.9,
            ),
            LLMEvidenceCandidate(
                kind="holding",
                statement="Fabricated proposition not present in the document",
                verbatim_quote="This exact sentence does not appear anywhere in the source text.",
                pinpoint=None,
                citation_raw=None,
                supports_issues=["issue-1"],
                llm_confidence=0.7,
            ),
        ]
    )

    with patch("app.graph.nodes.extractor.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=(mock_batch, {"input_tokens": 100, "output_tokens": 50}))
        mock_get_llm.return_value = mock_llm

        extract_result = await extractor_node(
            sources=[source],
            documents_by_source_id={"src-1": doc},
            legal_issues=["Is X a related party under s.29A?"],
        )

    candidates = extract_result["evidence_candidates"]
    assert len(candidates) == 2, f"expected 2 candidates, got {len(candidates)}"
    print(f"extractor_node -> {len(candidates)} candidates (mocked LLM)")

    ground_result = await grounding_node(
        candidates=candidates,
        sources_by_id={"src-1": source},
        documents_by_source_id={"src-1": doc},
        run_id="run-1",
        run_jurisdiction=Jurisdiction.IN,
        loop_index=0,
    )

    evidence = ground_result["evidence"]
    errors = ground_result["errors"]

    print(f"grounding_node -> {len(evidence)} evidence, {len(errors)} errors")

    assert len(evidence) == 1, f"expected exactly 1 grounded evidence item, got {len(evidence)}"
    ev = evidence[0]
    assert ev.grounding == "exact", f"expected exact grounding, got {ev.grounding}"
    assert ev.binding_strength == "binding", f"expected binding, got {ev.binding_strength}"
    assert ev.authority_tier == 1
    assert ev.verbatim_quote in DOC_TEXT, "resolved quote must be traceable to doc text"
    assert ev.evidence_id, "evidence_id must be set"

    assert len(errors) == 1, f"expected 1 NodeError for the hallucinated candidate, got {len(errors)}"
    assert errors[0].kind == "extraction_failed"

    print("\nALL PIPELINE WIRING CHECKS PASSED")
    print(f"  evidence[0].statement = {ev.statement!r}")
    print(f"  evidence[0].binding_strength = {ev.binding_strength}")
    print(f"  evidence[0].pinpoint = {ev.pinpoint}")
    print(f"  dropped: {errors[0].detail}")


if __name__ == "__main__":
    asyncio.run(main())
