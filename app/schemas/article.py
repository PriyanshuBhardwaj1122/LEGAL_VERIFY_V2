"""Generation phase schemas (Step 6+). Consumes EvidencePackage
(app/schemas/handoff.py) and produces a citation-bound article draft.

Design mirrors the Research phase: the LLM never emits citation text
directly. It writes prose with inline markers pointing at an
evidence_id it was actually handed — `[[ev:<evidence_id>]]` — and
deterministic code resolves those markers into real citation strings
and checks every marker against the evidence set it should be tracing
back to. A marker that can't be resolved, or resolves to evidence not
handed to that section, is a hard verification failure, not a warning
— same posture as grounding.py toward an unlocatable quote.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SectionRole = Literal[
    "intro", "concede", "convict", "refuse", "counterfactual", "case_against", "conclusion"
]


class ArticleThesis(BaseModel):
    """The article's specific, falsifiable claim — produced once, before
    the outline, and threaded into every downstream prompt so sections
    argue toward one thing instead of each finding its own hedge."""

    run_id: str
    thesis: str  # the specific claim, not a topic ("X is arbitrary because...", not "X is examined")
    falsifier: str  # what evidence would kill this thesis, stated concretely


class ArticleSection(BaseModel):
    section_id: str
    title: str
    target_words: int = Field(ge=50, le=4000)
    issue_refs: list[str] = []  # subset of EvidencePackage.legal_issues this section covers
    role: SectionRole = "convict"
    conclusion: str = ""  # the specific claim this section must land, not just its topic
    depends_on: list[str] = []  # section_ids of earlier sections this one builds on / must not re-explain


class ArticleOutline(BaseModel):
    run_id: str
    title: str
    sections: list[ArticleSection]


class CalcInput(BaseModel):
    evidence_id: str
    value: float
    label: str = ""  # what this number represents, e.g. "FY19 market size (Rs bn)"


class CalcClaim(BaseModel):
    """A figure the drafter derives from cited evidence rather than
    quoting directly — e.g. a CAGR from two market-size figures. The
    method is a fixed whitelist recomputed mechanically in verify_draft;
    an LLM never gets to assert arithmetic without it being checked,
    same posture as citations never being trusted to format themselves."""

    calc_id: str
    method: Literal["sum", "difference", "ratio", "delta_pct", "cagr"]
    inputs: list[CalcInput] = Field(min_length=1, max_length=4)
    periods: float | None = None  # required for cagr (number of periods between the two values)
    claimed_result: float
    unit: str = ""  # e.g. "%", "bn", "x" — appended when rendering


class DraftedSection(BaseModel):
    section_id: str
    title: str
    body: str  # markdown prose with inline [[ev:<evidence_id>]] and [[calc:<calc_id>]] markers
    evidence_ids_offered: list[str]  # what this section's draft call was allowed to cite
    calcs: list[CalcClaim] = []  # this section's own derived figures, scoped to it like evidence_ids_offered


class ArticleDraft(BaseModel):
    run_id: str
    title: str
    sections: list[DraftedSection]
    rendered_markdown: str  # sections assembled with markers resolved to real citations


class MarkerIssue(BaseModel):
    section_id: str
    marker_evidence_id: str
    kind: Literal["unresolved", "out_of_scope"]
    detail: str


class QuoteIssue(BaseModel):
    section_id: str
    evidence_id: str
    quoted_text: str
    detail: str


class CalcIssue(BaseModel):
    section_id: str
    calc_id: str
    detail: str


class DraftVerificationReport(BaseModel):
    run_id: str
    marker_issues: list[MarkerIssue] = []
    quote_issues: list[QuoteIssue] = []
    calc_issues: list[CalcIssue] = []
    uncovered_issues: list[str] = []  # legal_issues with zero cited evidence anywhere in the draft
    citation_count: int
    unique_evidence_cited: int
    # Markers that resolve to real, in-scope evidence but whose evidence
    # carries no renderable citation — they print as "[citation
    # unresolved]" to the reader. Valid plumbing, useless attribution.
    unresolved_citation_count: int = 0
    unresolved_citation_ratio: float = 0.0
    verdict: Literal["passed", "failed"]
    rationale: str
