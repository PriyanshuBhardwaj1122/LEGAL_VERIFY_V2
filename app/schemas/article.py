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


class ArticleSection(BaseModel):
    section_id: str
    title: str
    target_words: int = Field(ge=50, le=4000)
    issue_refs: list[str] = []  # subset of EvidencePackage.legal_issues this section covers


class ArticleOutline(BaseModel):
    run_id: str
    title: str
    sections: list[ArticleSection]


class DraftedSection(BaseModel):
    section_id: str
    title: str
    body: str  # markdown prose with inline [[ev:<evidence_id>]] markers
    evidence_ids_offered: list[str]  # what this section's draft call was allowed to cite


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


class DraftVerificationReport(BaseModel):
    run_id: str
    marker_issues: list[MarkerIssue] = []
    quote_issues: list[QuoteIssue] = []
    uncovered_issues: list[str] = []  # legal_issues with zero cited evidence anywhere in the draft
    citation_count: int
    unique_evidence_cited: int
    verdict: Literal["passed", "failed"]
    rationale: str
