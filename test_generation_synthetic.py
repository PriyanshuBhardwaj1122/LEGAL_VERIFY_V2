"""Synthetic verification for the Generation phase (Steps 6-9):
outline_node, draft_node, assemble_article, verify_draft.

Focus is Step 9 (mechanical draft verification) since that's the part
doing anti-hallucination work — everything else is scaffolding around
it. Covers: a clean draft passes; an out-of-scope marker (citing
evidence not offered to that section) fails; an unresolved marker
(citing an evidence_id that doesn't exist) fails; an altered quote
(text in quotes that doesn't match the evidence's verbatim_quote)
fails; an issue with zero cited evidence anywhere fails even if the
outline nominally assigned it to a section.

Run: PYTHONPATH=. python tests/test_generation_synthetic.py
"""
from __future__ import annotations

import asyncio
import sys
from datetime import date
from unittest.mock import AsyncMock, patch

from app.domain.citation_render import render_citation
from app.graph.nodes.generation import (
    LLMOutline,
    LLMSection,
    LLMSectionDraft,
    _domain_from_url,
    _draft_section,
    _evidence_line,
    assemble_article,
    draft_node,
    outline_node,
    verify_draft,
)
from app.schemas.article import ArticleOutline, ArticleSection, DraftedSection
from app.schemas.citation import LegalCitation
from app.schemas.common import CourtLevel, Jurisdiction, SourceType
from app.schemas.evidence import Evidence
from app.schemas.handoff import EvidencePackage
from app.schemas.request import ArticleConfig
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
    supports_issues: list[str],
    verbatim_quote: str | None = "The applicant must satisfy the eligibility criteria under section 29A.",
    kind: str = "holding",
    citation: LegalCitation | None = None,
) -> Evidence:
    # Real evidence normally carries a citation; defaulting to None made
    # every fixture look like an unattributable item, which the
    # unresolved-citation check in verify_draft correctly rejects.
    if citation is None:
        citation = LegalCitation(
            raw="AIR 2019 SC 123",
            kind="case",
            case_name="Swiss Ribbons v Union of India",
            reporter_citation="AIR 2019 SC 123",
            court="Supreme Court of India",
            is_parsed=True,
        )
    return Evidence(
        evidence_id=evidence_id,
        run_id="run-1",
        source_id=f"src-{evidence_id}",
        loop_index=0,
        kind=kind,  # type: ignore[arg-type]
        statement="Section 29A bars certain persons from submitting a resolution plan.",
        verbatim_quote=verbatim_quote,
        quote_start=0,
        quote_end=len(verbatim_quote) if verbatim_quote else None,
        grounding="exact" if verbatim_quote else "exact",
        pinpoint="para 12",
        citation=citation,
        supports_issues=supports_issues,
        jurisdiction=Jurisdiction.IN,
        court_level=CourtLevel.SUPREME_COURT,
        as_of=date(2021, 1, 1),
        binding_strength="binding",
        llm_confidence=0.9,
        authority_tier=1,
    )


def _source(
    source_id: str,
    source_type: SourceType = SourceType.JUDGMENT,
    issuing_body: str | None = None,
    url_canonical: str = "https://example.com/doc",
    decided_or_published_on: date | None = None,
) -> Source:
    return Source(
        source_id=source_id,
        run_id="run-1",
        url_canonical=url_canonical,
        fetch_url=url_canonical,
        title="Some title",
        source_type=source_type,
        court_level=CourtLevel.NONE,
        issuing_body=issuing_body,
        jurisdiction=Jurisdiction.IN,
        decided_or_published_on=decided_or_published_on,
        citation=None,
        score=SourceScore(authority=0.9, relevance=0.9, recency=0.9, reliability=0.9, composite=0.9, tier=1, reasons=[]),
        status="extracted",
        discovered_by=[],
        provenance_query_ids=[],
    )


def _package(evidence: list[Evidence], legal_issues: list[str]) -> EvidencePackage:
    return EvidencePackage(
        run_id="run-1",
        topic="Section 29A eligibility",
        jurisdiction=Jurisdiction.IN,
        as_of_date=date(2024, 1, 1),
        legal_issues=legal_issues,
        evidence=evidence,
        sources=[],
        unresolved_gaps=[],
        coverage={i: 1 for i in legal_issues},
        research_verdict="complete",
    )


# ---------------------------------------------------------------------
# render_citation
# ---------------------------------------------------------------------

def test_render_citation_case():
    print("render_citation() — case citation")
    c = LegalCitation(
        raw="raw", kind="case", case_name="Swiss Ribbons v. Union of India",
        neutral_citation="(2019) 4 SCC 17", court="Supreme Court of India",
        decided_on=date(2019, 1, 25), is_parsed=True,
    )
    rendered = render_citation(c)
    check("case name present", "Swiss Ribbons" in rendered, rendered)
    check("reporter present", "SCC" in rendered, rendered)
    check("court present", "Supreme Court" in rendered, rendered)


def test_render_citation_statute():
    print("render_citation() — statute citation")
    c = LegalCitation(raw="raw", kind="statute", act_name="Insolvency and Bankruptcy Code", act_year=2016, section="29A", is_parsed=True)
    rendered = render_citation(c)
    check("act name + year + section present", rendered == "Insolvency and Bankruptcy Code, 2016, s. 29A", rendered)


def test_render_citation_unparsed_falls_back_to_raw():
    print("render_citation() — unparsed falls back to raw")
    c = LegalCitation(raw="some raw citation string", kind="other", is_parsed=False)
    check("returns raw", render_citation(c) == "some raw citation string")


def test_render_citation_none():
    print("render_citation() — None citation")
    check("empty string for None", render_citation(None) == "")


# ---------------------------------------------------------------------
# verify_draft — the core anti-hallucination checks
# ---------------------------------------------------------------------

def test_verify_draft_clean_passes():
    print("verify_draft() — correctly cited, in-scope draft passes")
    ev = _evidence("ev1", ["1"])
    outline = ArticleOutline(
        run_id="run-1", title="T",
        sections=[ArticleSection(section_id="sec-0", title="Eligibility", target_words=300, issue_refs=["Issue A"])],
    )
    sections = [
        DraftedSection(
            section_id="sec-0", title="Eligibility",
            body='The Code provides that "The applicant must satisfy the eligibility criteria under section 29A." [[ev:ev1]]',
            evidence_ids_offered=["ev1"],
        )
    ]
    report = verify_draft(outline, sections, {"ev1": ev}, ["Issue A"])
    check("verdict passed", report.verdict == "passed", report.rationale)
    check("no marker issues", not report.marker_issues)
    check("no quote issues", not report.quote_issues)
    check("no uncovered issues", not report.uncovered_issues)


def test_verify_draft_out_of_scope_marker_fails():
    print("verify_draft() — citing evidence not offered to the section fails")
    ev1 = _evidence("ev1", ["1"])
    ev2 = _evidence("ev2", ["1"])
    outline = ArticleOutline(
        run_id="run-1", title="T",
        sections=[ArticleSection(section_id="sec-0", title="Eligibility", target_words=300, issue_refs=["Issue A"])],
    )
    sections = [
        DraftedSection(
            section_id="sec-0", title="Eligibility",
            body="The Code bars certain persons. [[ev:ev2]]",
            evidence_ids_offered=["ev1"],  # ev2 was NOT offered
        )
    ]
    report = verify_draft(outline, sections, {"ev1": ev1, "ev2": ev2}, ["Issue A"])
    check("verdict failed", report.verdict == "failed")
    check("one out_of_scope marker issue recorded", len(report.marker_issues) == 1 and report.marker_issues[0].kind == "out_of_scope", report.marker_issues)


def test_verify_draft_unresolved_marker_fails():
    print("verify_draft() — citing a nonexistent evidence_id fails")
    ev1 = _evidence("ev1", ["1"])
    outline = ArticleOutline(
        run_id="run-1", title="T",
        sections=[ArticleSection(section_id="sec-0", title="Eligibility", target_words=300, issue_refs=["Issue A"])],
    )
    sections = [
        DraftedSection(
            section_id="sec-0", title="Eligibility",
            body="The Code bars certain persons. [[ev:does-not-exist]]",
            evidence_ids_offered=["ev1"],
        )
    ]
    report = verify_draft(outline, sections, {"ev1": ev1}, ["Issue A"])
    check("verdict failed", report.verdict == "failed")
    check("one unresolved marker issue recorded", len(report.marker_issues) == 1 and report.marker_issues[0].kind == "unresolved", report.marker_issues)


def test_verify_draft_altered_quote_fails():
    print("verify_draft() — quoted text diverging from verbatim_quote fails")
    ev1 = _evidence("ev1", ["1"], verbatim_quote="The applicant must satisfy the eligibility criteria under section 29A.")
    outline = ArticleOutline(
        run_id="run-1", title="T",
        sections=[ArticleSection(section_id="sec-0", title="Eligibility", target_words=300, issue_refs=["Issue A"])],
    )
    sections = [
        DraftedSection(
            section_id="sec-0", title="Eligibility",
            # Model silently "improved" the quote — should be caught.
            body='The Code states "Applicants absolutely must meet every single eligibility requirement whatsoever under section 29A without exception." [[ev:ev1]]',
            evidence_ids_offered=["ev1"],
        )
    ]
    report = verify_draft(outline, sections, {"ev1": ev1}, ["Issue A"])
    check("verdict failed", report.verdict == "failed")
    check("one quote issue recorded", len(report.quote_issues) == 1, report.quote_issues)


def test_verify_draft_clean_substring_quote_passes():
    print("verify_draft() — quoting a clean substring of verbatim_quote (e.g. dropped a leading label) does NOT fail")
    # Real-world case: evidence's stored verbatim_quote is a circular's
    # full title including a leading label; the drafter quoted the
    # substantive part only, with a case change. This must NOT be
    # treated as an altered quote — it's a genuine excerpt.
    ev1 = _evidence("ev1", ["1"], verbatim_quote="Circular- Strengthening due diligence under Section 29A.")
    outline = ArticleOutline(
        run_id="run-1", title="T",
        sections=[ArticleSection(section_id="sec-0", title="Regulatory", target_words=300, issue_refs=["Issue A"])],
    )
    sections = [
        DraftedSection(
            section_id="sec-0", title="Regulatory",
            body='A circular aimed at "strengthening due diligence under Section 29A" [[ev:ev1]]',
            evidence_ids_offered=["ev1"],
        )
    ]
    report = verify_draft(outline, sections, {"ev1": ev1}, ["Issue A"])
    check("verdict passed (substring quote is not a divergence)", report.verdict == "passed", report.rationale)
    check("no quote issues", not report.quote_issues, report.quote_issues)


def test_verify_draft_quote_without_verbatim_fails():
    print("verify_draft() — quoting a commentary item with no verbatim_quote fails")
    ev1 = _evidence("ev1", ["1"], verbatim_quote=None, kind="commentary_opinion")
    outline = ArticleOutline(
        run_id="run-1", title="T",
        sections=[ArticleSection(section_id="sec-0", title="Eligibility", target_words=300, issue_refs=["Issue A"])],
    )
    sections = [
        DraftedSection(
            section_id="sec-0", title="Eligibility",
            body='Commentators argue "this is a fabricated quote that was never in the source." [[ev:ev1]]',
            evidence_ids_offered=["ev1"],
        )
    ]
    report = verify_draft(outline, sections, {"ev1": ev1}, ["Issue A"])
    check("verdict failed", report.verdict == "failed")
    check("one quote issue recorded (no verbatim_quote to check against)", len(report.quote_issues) == 1, report.quote_issues)


def test_verify_draft_uncovered_issue_fails():
    print("verify_draft() — an issue with zero cited evidence anywhere fails")
    ev1 = _evidence("ev1", ["1"])
    outline = ArticleOutline(
        run_id="run-1", title="T",
        sections=[ArticleSection(section_id="sec-0", title="Eligibility", target_words=300, issue_refs=["Issue A", "Issue B"])],
    )
    sections = [
        DraftedSection(
            section_id="sec-0", title="Eligibility",
            body='The Code provides that "The applicant must satisfy the eligibility criteria under section 29A." [[ev:ev1]]',
            evidence_ids_offered=["ev1"],
        )
    ]
    # ev1 only supports_issues=["1"] -> "Issue A"; "Issue B" never gets a marker.
    report = verify_draft(outline, sections, {"ev1": ev1}, ["Issue A", "Issue B"])
    check("verdict failed", report.verdict == "failed")
    check("Issue B recorded as uncovered", report.uncovered_issues == ["Issue B"], report.uncovered_issues)


# ---------------------------------------------------------------------
# assemble_article
# ---------------------------------------------------------------------

def test_assemble_article_resolves_markers():
    print("assemble_article() — resolves marker to rendered citation")
    c = LegalCitation(raw="raw", kind="statute", act_name="IBC", act_year=2016, section="29A", is_parsed=True)
    ev1 = _evidence("ev1", ["1"], citation=c)
    outline = ArticleOutline(run_id="run-1", title="Article Title", sections=[ArticleSection(section_id="sec-0", title="Eligibility", target_words=300, issue_refs=["Issue A"])])
    sections = [DraftedSection(section_id="sec-0", title="Eligibility", body="Text here. [[ev:ev1]]", evidence_ids_offered=["ev1"])]
    rendered = assemble_article(outline, sections, {"ev1": ev1})
    check("title present", "Article Title" in rendered)
    check("marker replaced, no raw marker left", "[[ev:" not in rendered, rendered)
    check("rendered citation content present", "IBC, 2016, s. 29A" in rendered, rendered)


def test_assemble_article_unresolved_marker_flagged():
    print("assemble_article() — unresolvable marker rendered as a visible flag, not silently dropped")
    outline = ArticleOutline(run_id="run-1", title="T", sections=[ArticleSection(section_id="sec-0", title="S", target_words=300, issue_refs=[])])
    sections = [DraftedSection(section_id="sec-0", title="S", body="Text. [[ev:missing]]", evidence_ids_offered=[])]
    rendered = assemble_article(outline, sections, {})
    check("visible unresolved marker in output", "[citation unresolved]" in rendered, rendered)


# ---------------------------------------------------------------------
# _domain_from_url / _evidence_line — press-marking / outlet attribution
# ---------------------------------------------------------------------

def test_domain_from_url_strips_www():
    print("_domain_from_url() — strips leading www.")
    check("www stripped", _domain_from_url("https://www.livelaw.in/some/article") == "livelaw.in")


def test_domain_from_url_no_www():
    print("_domain_from_url() — no www. present")
    check("bare domain returned", _domain_from_url("https://barandbench.com/x") == "barandbench.com")


def test_domain_from_url_malformed_falls_back():
    print("_domain_from_url() — malformed URL doesn't raise, falls back")
    result = _domain_from_url("not a url at all :::")
    check("no exception, some string returned", isinstance(result, str), result)


def test_evidence_line_press_source_marked_press():
    print("_evidence_line() — a LEGAL_NEWS source is marked PRESS with outlet+date")
    ev = _evidence("ev1", ["1"])
    src = _source(
        "src-ev1", source_type=SourceType.LEGAL_NEWS,
        issuing_body=None, url_canonical="https://www.livelaw.in/news/x",
        decided_or_published_on=date(2023, 6, 15),
    )
    line = _evidence_line(ev, src)
    check("marked PRESS", "PRESS" in line, line)
    check("outlet falls back to domain", "livelaw.in" in line, line)
    check("date present", "2023-06-15" in line, line)


def test_evidence_line_judgment_marked_authority():
    print("_evidence_line() — a JUDGMENT source is marked AUTHORITY, not PRESS")
    ev = _evidence("ev1", ["1"])
    src = _source("src-ev1", source_type=SourceType.JUDGMENT, issuing_body="Supreme Court of India")
    line = _evidence_line(ev, src)
    check("marked AUTHORITY", "AUTHORITY" in line, line)
    check("PRESS not present as tier label", not line.split("|")[3].strip() == "PRESS", line)
    check("issuing_body used as outlet", "Supreme Court of India" in line, line)


def test_evidence_line_no_source_defaults_gracefully():
    print("_evidence_line() — missing Source (None) doesn't raise, defaults sanely")
    ev = _evidence("ev1", ["1"])
    line = _evidence_line(ev, None)
    check("defaults to AUTHORITY (not press) when source unknown", "AUTHORITY" in line, line)
    check("unknown outlet placeholder present", "unknown outlet" in line, line)
    check("undated placeholder present", "undated" in line, line)


async def test_draft_section_press_marking_reaches_prompt():
    print("_draft_section() — press evidence's PRESS/outlet/date reaches the evidence_lines given to the LLM")
    ev = _evidence("ev1", ["1"])
    src = _source(
        "src-ev1", source_type=SourceType.FIRM_COMMENTARY,
        issuing_body="AZB & Partners", decided_or_published_on=date(2022, 3, 1),
    )
    section = ArticleSection(section_id="sec-0", title="Reception", target_words=300, issue_refs=["Issue A"])

    captured: dict = {}

    async def fake_generate(**kwargs):
        captured["user"] = kwargs["user"]
        captured["temperature"] = kwargs["temperature"]
        return LLMSectionDraft(body="drafted text [[ev:ev1]]"), {"input_tokens": 1, "output_tokens": 1}

    with patch("app.graph.nodes.generation.get_generation_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=fake_generate)
        mock_get_llm.return_value = mock_llm
        await _draft_section(section, [ev], {"src-ev1": src}, [], {})

    check("PRESS tag reached the prompt", "PRESS" in captured["user"], captured["user"])
    check("outlet reached the prompt", "AZB & Partners" in captured["user"], captured["user"])
    check("date reached the prompt", "2022-03-01" in captured["user"], captured["user"])
    check("temperature raised for drafting (voice/register)", captured["temperature"] == 0.65, captured["temperature"])


# ---------------------------------------------------------------------
# outline_node / draft_node — mocked LLM wiring
# ---------------------------------------------------------------------

async def test_outline_node_assigns_uncovered_issue_to_last_section():
    print("outline_node() — an issue the LLM forgot to assign gets attached to the last section, not lost")
    package = _package([], ["Issue A", "Issue B"])
    config = ArticleConfig(article_type="explainer", target_words=1500)

    llm_out = LLMOutline(
        title="Understanding Section 29A",
        sections=[
            LLMSection(title="Background", target_words=500, issue_refs=["Issue A"], role="convict", conclusion="Background conclusion."),
            LLMSection(title="Analysis", target_words=1000, issue_refs=[], role="convict", conclusion="Analysis conclusion."),  # forgot Issue B
        ],
    )

    with patch("app.graph.nodes.generation.get_generation_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=(llm_out, {"input_tokens": 1, "output_tokens": 1}))
        mock_get_llm.return_value = mock_llm
        result = await outline_node(package, config)

    outline: ArticleOutline = result["outline"]
    all_refs = {r for s in outline.sections for r in s.issue_refs}
    check("both issues covered across sections despite LLM omission", all_refs == {"Issue A", "Issue B"}, all_refs)


async def test_draft_node_scopes_evidence_per_section():
    print("draft_node() — each section only receives evidence for its own issue_refs")
    ev_a = _evidence("ev-a", ["1"])
    ev_b = _evidence("ev-b", ["2"])
    package = _package([ev_a, ev_b], ["Issue A", "Issue B"])
    outline = ArticleOutline(
        run_id="run-1", title="T",
        sections=[
            ArticleSection(section_id="sec-0", title="A", target_words=300, issue_refs=["Issue A"]),
            ArticleSection(section_id="sec-1", title="B", target_words=300, issue_refs=["Issue B"]),
        ],
    )

    async def fake_generate(**kwargs):
        return LLMSectionDraft(body="drafted text"), {"input_tokens": 1, "output_tokens": 1}

    with patch("app.graph.nodes.generation.get_generation_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=fake_generate)
        mock_get_llm.return_value = mock_llm
        result = await draft_node(outline, package)

    sections = result["drafted_sections"]
    sec_a = next(s for s in sections if s.section_id == "sec-0")
    sec_b = next(s for s in sections if s.section_id == "sec-1")
    check("section A only offered ev-a", sec_a.evidence_ids_offered == ["ev-a"], sec_a.evidence_ids_offered)
    check("section B only offered ev-b", sec_b.evidence_ids_offered == ["ev-b"], sec_b.evidence_ids_offered)


def main():
    test_render_citation_case()
    test_render_citation_statute()
    test_render_citation_unparsed_falls_back_to_raw()
    test_render_citation_none()

    test_verify_draft_clean_passes()
    test_verify_draft_out_of_scope_marker_fails()
    test_verify_draft_unresolved_marker_fails()
    test_verify_draft_altered_quote_fails()
    test_verify_draft_clean_substring_quote_passes()
    test_verify_draft_quote_without_verbatim_fails()
    test_verify_draft_uncovered_issue_fails()

    test_assemble_article_resolves_markers()
    test_assemble_article_unresolved_marker_flagged()

    test_domain_from_url_strips_www()
    test_domain_from_url_no_www()
    test_domain_from_url_malformed_falls_back()
    test_evidence_line_press_source_marked_press()
    test_evidence_line_judgment_marked_authority()
    test_evidence_line_no_source_defaults_gracefully()
    asyncio.run(test_draft_section_press_marking_reaches_prompt())

    asyncio.run(test_outline_node_assigns_uncovered_issue_to_last_section())
    asyncio.run(test_draft_node_scopes_evidence_per_section())

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()