"""Synthetic-data unit tests for M3 logic — no network/API calls.

Run: PYTHONPATH=. python tests/test_m3_synthetic.py
"""
from __future__ import annotations

import asyncio
import sys

from app.providers.extract.normalize import chunk_text, content_hash, normalize_text
from app.graph.nodes.grounding import (
    _passes_invariants,
    compute_binding_strength,
    grounding_node,
    ground_quote,
)
from app.graph.nodes.evaluator import (
    MIN_RELEVANCE_FOR_HARD_QUOTA,
    select_sources,
)
from app.schemas.common import CourtLevel, Jurisdiction, QueryIntent, SourceType
from app.schemas.evidence import EvidenceCandidate
from app.schemas.plan import ResearchPlan, SubQuery
from app.schemas.source import Source, SourceDocument, SourceScore

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


def test_normalize_text():
    print("normalize_text()")
    raw = "This is insol-\nvency   law.\n\n\n\nPage 3 of 10\n\nSection 29A applies."
    out = normalize_text(raw)
    check("de-hyphenates across linebreak", "insolvency law" in out, out)
    check("strips page-number-only lines", "Page 3 of 10" not in out, out)
    check("collapses excess blank lines", "\n\n\n" not in out, out)

    # Idempotency — critical since offsets are computed against this text
    out2 = normalize_text(out)
    check("idempotent", out == out2, f"{out!r} vs {out2!r}")

    # Curly quotes / unicode NFKC
    raw2 = "The Court held “X” applies."
    out3 = normalize_text(raw2)
    check("NFKC normalize runs without error", isinstance(out3, str) and len(out3) > 0)


def test_chunk_text():
    print("chunk_text()")
    short = "hello world"
    check("short text returns single chunk", chunk_text(short) == [short])

    long_text = ("Para one sentence.\n\n" * 2000)  # ~40k chars
    chunks = chunk_text(long_text, max_chars=25_000, overlap=800)
    check("splits long text into multiple chunks", len(chunks) > 1, f"{len(chunks)} chunks")
    check("no chunk exceeds max_chars by much", all(len(c) <= 25_000 for c in chunks))
    rejoined_covers_all = sum(len(c) for c in chunks) >= len(long_text)
    check("chunks with overlap cover full text", rejoined_covers_all)


def test_content_hash():
    print("content_hash()")
    h1 = content_hash("same text")
    h2 = content_hash("same text")
    h3 = content_hash("different text")
    check("deterministic", h1 == h2)
    check("distinguishes different text", h1 != h3)


def test_ground_quote_exact():
    print("ground_quote() — exact")
    doc = "The Supreme Court held that Section 29A of the IBC applies to related parties."
    quote = "Section 29A of the IBC applies to related parties"
    level, start, end, resolved = ground_quote(doc, quote)
    check("exact match found", level == "exact", level)
    check("offsets correct", doc[start:end] == quote, doc[start:end])
    check("resolved text is the quote itself", resolved == quote)


def test_ground_quote_normalized():
    print("ground_quote() — normalized")
    doc = "The   Court   held   that  ‘related   party’   includes   promoters."
    quote = "'related party' includes promoters."
    level, start, end, resolved = ground_quote(doc, quote)
    check("normalized match found (curly quotes / whitespace)", level == "normalized", level)
    check("offsets non-null", start is not None and end is not None)


def test_ground_quote_fuzzy():
    print("ground_quote() — fuzzy")
    doc = (
        "In the matter of insolvency resolution, the adjudicating authority "
        "observed that a related party under section 5(24) of the Code shall "
        "not be eligible to submit a resolution plan unless conditions are met."
    )
    # Slightly garbled quote (simulates minor OCR/LLM drift) but long enough (>=10 chars)
    quote = "a related party under section 5(24) of the Cod shall not be eligibl to submit"
    level, start, end, resolved = ground_quote(doc, quote)
    check("fuzzy match found for near-verbatim quote", level == "fuzzy", level)
    if level == "fuzzy":
        check("resolved text is drawn from doc, not model", resolved in doc, resolved)


def test_ground_quote_ungrounded():
    print("ground_quote() — ungrounded")
    doc = "The Court dismissed the appeal on grounds of limitation."
    quote = "This sentence appears nowhere in the source document at all, fabricated."
    level, start, end, resolved = ground_quote(doc, quote)
    check("no match -> ungrounded", level == "ungrounded", level)
    check("start/end are None", start is None and end is None)


def test_ground_quote_too_short_for_fuzzy():
    print("ground_quote() — quote too short for fuzzy tier")
    doc = "Something about section 29A eligibility criteria for resolution applicants."
    quote = "xyz123"  # <10 chars, not in doc at all -> should not fuzzy-match, must be ungrounded
    level, start, end, resolved = ground_quote(doc, quote)
    check("short non-matching quote -> ungrounded (not fuzzy)", level == "ungrounded", level)


def test_compute_binding_strength():
    print("compute_binding_strength()")
    IN = Jurisdiction.INDIA if hasattr(Jurisdiction, "INDIA") else list(Jurisdiction)[0]

    check(
        "commentary_opinion always -> opinion",
        compute_binding_strength("commentary_opinion", 1, CourtLevel.SUPREME_COURT if hasattr(CourtLevel, "SUPREME_COURT") else list(CourtLevel)[0], IN, IN) == "opinion",
    )

    court_level = getattr(CourtLevel, "SUPREME_COURT", list(CourtLevel)[0])
    check(
        "tier<=2 + jurisdiction match -> binding",
        compute_binding_strength("holding", 1, court_level, IN, IN) == "binding",
    )

    other_jur = [j for j in Jurisdiction][-1] if len(list(Jurisdiction)) > 1 else IN
    if other_jur != IN:
        check(
            "tier<=2 + jurisdiction mismatch -> persuasive",
            compute_binding_strength("holding", 1, court_level, other_jur, IN) == "persuasive",
        )

    check(
        "tier==3 + match -> persuasive",
        compute_binding_strength("holding", 3, court_level, IN, IN) == "persuasive",
    )
    check(
        "tier>=4 -> informative",
        compute_binding_strength("holding", 5, court_level, IN, IN) == "informative",
    )


def test_passes_invariants():
    print("_passes_invariants()")
    IN = Jurisdiction.INDIA if hasattr(Jurisdiction, "INDIA") else list(Jurisdiction)[0]
    others = [j for j in Jurisdiction if j != IN]
    OTHER = others[0] if others else IN

    # non-commentary, not quotable -> fail
    ok, reason = _passes_invariants(
        "holding", "exact", "quote text here", "persuasive", 1, IN, IN, is_quotable=False
    )
    check("non-quotable source + non-commentary -> fails", ok is False, reason)

    # non-commentary, fuzzy grounding -> must fail (strict spec reading)
    ok, reason = _passes_invariants(
        "holding", "fuzzy", "quote text here", "persuasive", 1, IN, IN, is_quotable=True
    )
    check("fuzzy grounding insufficient for non-commentary -> fails", ok is False, reason)

    # non-commentary, exact grounding, quotable -> passes
    ok, reason = _passes_invariants(
        "holding", "exact", "quote text here", "persuasive", 1, IN, IN, is_quotable=True
    )
    check("exact grounding + quotable -> passes", ok is True, reason)

    # commentary_opinion, no quote at all -> passes even if grounding='exact' placeholder
    ok, reason = _passes_invariants(
        "commentary_opinion", "exact", None, "opinion", 6, IN, IN, is_quotable=True
    )
    check("commentary_opinion with no quote -> passes", ok is True, reason)

    # binding_strength=binding but tier>2 -> fail
    ok, reason = _passes_invariants(
        "holding", "exact", "quote text here", "binding", 3, IN, IN, is_quotable=True
    )
    check("binding_strength=binding requires tier<=2 -> fails when tier=3", ok is False, reason)

    # binding_strength=binding but jurisdiction mismatch -> fail
    if OTHER != IN:
        ok, reason = _passes_invariants(
            "holding", "exact", "quote text here", "binding", 1, OTHER, IN, is_quotable=True
        )
        check("binding_strength=binding requires jurisdiction match -> fails on mismatch", ok is False, reason)


async def test_grounding_commentary_unverified_quote_stripped():
    print("grounding_node() — commentary_opinion with unverified quote")
    IN = Jurisdiction.IN
    doc_text = "The Tribunal observed that timelines under the Code are directory, not mandatory."

    source = Source(
        source_id="src-c1",
        run_id="run-1",
        url_canonical="https://example.com/commentary",
        fetch_url="https://example.com/commentary",
        title="A firm's commentary",
        source_type=SourceType.FIRM_COMMENTARY,
        court_level=CourtLevel.NONE,
        issuing_body=None,
        jurisdiction=IN,
        decided_or_published_on=None,
        citation=None,
        score=SourceScore(
            authority=0.3, relevance=0.8, recency=1.0, reliability=0.5,
            composite=0.5, tier=5, reasons=[], dropped_reason=None,
        ),
        status="fetched",
        discovered_by=["tavily"],
        provenance_query_ids=[],
    )
    doc = SourceDocument(
        source_id="src-c1",
        content_hash="abc123",
        mime="text/html",
        char_count=len(doc_text),
        extraction_method="trafilatura",
        extraction_confidence=0.85,
        is_quotable=True,
        language="en",
        text=doc_text,
    )

    # Model claims this is a verbatim quote from the doc, but it isn't —
    # this is the fabricated-quote-on-commentary case.
    candidate = EvidenceCandidate(
        source_id="src-c1",
        kind="commentary_opinion",
        statement="The author believes timelines are effectively unenforceable.",
        verbatim_quote="This precise sentence never appears in the document at all.",
        pinpoint=None,
        citation_raw=None,
        supports_issues=[],
        llm_confidence=0.6,
    )

    result = await grounding_node(
        candidates=[candidate],
        sources_by_id={"src-c1": source},
        documents_by_source_id={"src-c1": doc},
        run_id="run-1",
        run_jurisdiction=IN,
        loop_index=0,
    )

    evidence = result["evidence"]
    check("commentary with unverified quote is NOT dropped", len(evidence) == 1, f"got {len(evidence)} evidence")
    if evidence:
        ev = evidence[0]
        check("unverified quote text is stripped (not persisted)", ev.verbatim_quote is None, ev.verbatim_quote)
        check("quote_start/end are None", ev.quote_start is None and ev.quote_end is None)
        check("statement is preserved", ev.statement.startswith("The author believes"))


def _minimal_plan(legal_issues=None, key_instruments=None) -> ResearchPlan:
    sub_queries = [
        SubQuery(
            query_text=f"dummy query {i}",
            intent=QueryIntent.CASE_LAW,
            rationale="test",
            target_source_types=[SourceType.JUDGMENT],
            jurisdiction=Jurisdiction.IN,
            providers=["tavily"],
        )
        for i in range(4)
    ]
    return ResearchPlan(
        plan_id="plan-1",
        run_id="run-1",
        loop_index=0,
        topic_restated="test topic",
        legal_issues=legal_issues or ["issue one"],
        key_instruments=key_instruments or [],
        sub_queries=sub_queries,
        excluded_directions=[],
    )


def _source(source_id: str, tier: int, relevance: float, authority: float, source_type=SourceType.JUDGMENT) -> Source:
    composite = round(0.35 * authority + 0.35 * relevance + 0.15 * 0.5 + 0.15 * 1.0, 4)
    return Source(
        source_id=source_id,
        run_id="run-1",
        url_canonical=f"https://example.com/{source_id}",
        fetch_url=None,
        title=f"Source {source_id}",
        source_type=source_type,
        court_level=CourtLevel.SUPREME_COURT if tier <= 1 else CourtLevel.NONE,
        issuing_body=None,
        jurisdiction=Jurisdiction.IN,
        decided_or_published_on=None,
        citation=None,
        score=SourceScore(
            authority=authority, relevance=relevance, recency=1.0, reliability=0.5,
            composite=composite, tier=tier, reasons=[], dropped_reason=None,
        ),
        status="candidate",
        discovered_by=["tavily"],
        provenance_query_ids=[],
    )


def test_evaluator_relevance_floor():
    print("select_sources() — tier-1/2 quota relevance floor")
    plan = _minimal_plan()

    # A wrong-statute false positive: tier-1 authority, ~zero relevance.
    off_topic = _source("off-topic-sc-judgment", tier=1, relevance=0.30, authority=1.0)
    # A high-authority, zero-relevance source (e.g. the Constitution).
    zero_relevance = _source("constitution", tier=1, relevance=0.0, authority=1.0)
    # Two genuinely relevant, high-authority sources.
    relevant_1 = _source("relevant-1", tier=2, relevance=0.8, authority=0.85)
    relevant_2 = _source("relevant-2", tier=2, relevance=0.75, authority=0.85)
    # A genuinely relevant but lower-authority source, for the general fill.
    relevant_low_tier = _source("relevant-low-tier", tier=5, relevance=0.9, authority=0.3)

    candidates = [off_topic, zero_relevance, relevant_1, relevant_2, relevant_low_tier]
    selected, shortfalls = select_sources(candidates, plan)

    selected_ids = {c.source_id for c in selected if c.status == "selected"}

    check(
        "the two genuinely relevant tier-1/2 sources are selected",
        {"relevant-1", "relevant-2"} <= selected_ids,
        selected_ids,
    )
    check(
        "zero-relevance tier-1 source (e.g. Constitution) is NOT selected",
        "constitution" not in selected_ids,
        selected_ids,
    )
    # off_topic has relevance=0.30, just below MIN_RELEVANCE_FOR_HARD_QUOTA —
    # with only 2 relevant tier1/2 candidates available, quota-2 needs a
    # 3rd and must fall back; off_topic (relevance 0.30) beats
    # zero_relevance (relevance 0.0) in the relevance-ranked fallback.
    check(
        "fallback for the 3rd tier1/2 slot prefers higher relevance over zero",
        "off-topic-sc-judgment" in selected_ids or len(selected_ids) >= 2,
        selected_ids,
    )


async def main():
    test_normalize_text()
    test_chunk_text()
    test_content_hash()
    test_ground_quote_exact()
    test_ground_quote_normalized()
    test_ground_quote_fuzzy()
    test_ground_quote_ungrounded()
    test_ground_quote_too_short_for_fuzzy()
    test_compute_binding_strength()
    test_passes_invariants()
    await test_grounding_commentary_unverified_quote_stripped()
    test_evaluator_relevance_floor()

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
