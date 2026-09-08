"""STEP 4c — extractor node.

One LLM call per source (or per chunk for long documents), never one
call for all sources — scoping context to a single source is what stops
cross-source citation contamination. The model never emits offsets; it
cannot count characters reliably, and asking it to is a classic source
of silent corruption. Offsets are recovered deterministically in the
grounding step.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from app.core.budget import BudgetGuard
from app.core.config import get_settings
from app.core.logging import ctx_node, get_logger
from app.domain.passage_select import build_digests
from app.providers.extract.normalize import chunk_text
from app.providers.llm.base import get_llm
from app.schemas.evidence import EvidenceCandidate
from app.schemas.source import Source, SourceDocument

log = get_logger()

# How many items one chunk may yield. Kept small on purpose: a chunk is
# ~25k chars, and asking for more than a handful of genuinely citable
# propositions from one stretch of text invites padding.
MAX_ITEMS_PER_CHUNK = 3

# Per-source ceilings and how much of a document to read, keyed by the
# source's authority tier (app/domain/authority.py). Primary law earns
# the whole document; a blog gets its opening and nothing more.
#
# This replaces a single run-total cap of 8, which chunk 1 saturated on
# its own — so the loop broke and chunks 2..N were never sent to the
# LLM. On a 250k-char judgment that meant reading ~10% of the text, and
# the ~10% read was the cover page and counsel's submissions rather than
# the court's reasoning.
# tier: (LLM calls allowed for this source, max items to keep)
# Primary law earns three passes over a digest of the whole document;
# a blog gets one. Reading is not truncated by these — passage_select
# scans the full text and packs the best material into this many calls.
_TIER_BUDGET: dict[int, tuple[int, int]] = {
    1: (3, 16),
    2: (3, 16),
    3: (2, 12),
    4: (1, 8),
    5: (1, 8),
}
_DEFAULT_TIER_BUDGET = (1, 8)


def _budget_for(source: Source) -> tuple[int, int]:
    tier = getattr(getattr(source, "score", None), "tier", None)
    return _TIER_BUDGET.get(tier, _DEFAULT_TIER_BUDGET)

_SYSTEM = """You extract citable evidence from a single legal source. Your output feeds an automated verifier that will re-locate every quote you produce in this exact document. A quote that cannot be found is discarded and counts against you.

Rules, in order of importance:
1. Every item except commentary_opinion MUST include verbatim_quote copied character-for-character from the document. Do not fix typos, expand abbreviations, normalize spacing, or translate.
2. If the document does not support a proposition, do not emit it. Emitting nothing is a correct answer.
3. statement states ONE proposition in under 60 words, and states it SPECIFICALLY. Keep the detail that makes a proposition worth citing: party names, figures and sums, dates, the provision or test applied, and what turned on it. "The Court allowed the appeal" is nearly useless; "The Court allowed the appeal, holding the applicant ineligible because a related party's account had been an NPA for over a year before the plan was submitted" is citable. Do not summarise the document.
4. Classify honestly, and mind WHOSE words you are recording. A court's binding rule is `holding`. A passing remark is `obiter`. An author's view is `commentary_opinion`, even if you agree.
4a. A judgment recites each side's arguments before deciding. Text introduced by "it was argued", "counsel submitted", "according to the appellant/respondent", "they contended", "it was urged" is that party's ADVOCACY — it is NOT the court's holding, even when stated confidently and even when the court later agrees. Record such passages as `procedural_fact` if genuinely useful, otherwise skip them. The court's own reasoning usually arrives later in the document and is marked by "we are of the opinion", "we find", "in our view", "it is settled", "we therefore hold", "accordingly". Only that is `holding`.
5. pinpoint: give the paragraph number for judgments ("para 42") or the provision ("s. 11(6)", "s. 80-IA(4)(i)") for legislation, only if it appears in the text.
6. Do not carry over knowledge from outside this document. If you recognise the case and remember something not written here, that memory is not evidence.
7. At most 3 items from this passage. Choose the propositions an article would actually cite. You are seeing one part of a longer document, so do not try to cover the whole case here — take only what this passage genuinely establishes.
8. supports_issues: the legal issues below are numbered. For each item, list the number(s) (as strings, e.g. ["1"] or ["1","3"]) of every issue it helps resolve. Leave it empty only if the item genuinely doesn't bear on any listed issue — most items should map to at least one.
9. citation_raw: copy the source's own formal citation string verbatim when the document states one — a neutral citation ("2023 INSC 456"), reporter citation ("(2023) 5 SCC 1", "AIR 2019 SC 123"), SCC OnLine citation ("2021 SCC OnLine Del 456"), case/writ number ("Civil Appeal No. 1234 of 2020"), statute section ("Section 11 of the Arbitration and Conciliation Act, 1996"), or circular number ("SEBI/HO/CFD/CMD/CIR/P/2020/12") — whichever the document actually contains for this item. For a holding or obiter item this is the case's own citation, not the source's URL or page title. Leave it null only when the document genuinely states no formal citation for this proposition (a press article's own byline is not a citation). Never invent one and never supply one from memory rather than the document text."""

_USER_TEMPLATE = """Source: {title} | {source_type} | {issuing_body} | {date}

Legal issues this research must resolve (numbered — reference these numbers in supports_issues):
{issues}

--- DOCUMENT START ---
{document_text}
--- DOCUMENT END ---"""


class LLMEvidenceCandidate(BaseModel):
    kind: str
    statement: str = Field(max_length=500)
    verbatim_quote: str | None = None
    pinpoint: str | None = None
    citation_raw: str | None = None
    supports_issues: list[str] = []
    llm_confidence: float = Field(ge=0, le=1)


class LLMEvidenceBatch(BaseModel):
    items: list[LLMEvidenceCandidate] = Field(max_length=MAX_ITEMS_PER_CHUNK)


async def _record_llm_cost(
    budget: BudgetGuard | None, usage: dict[str, Any], source_id: str
) -> None:
    """Persist token usage for one extractor call.

    Extraction is the largest LLM consumer in the pipeline but was the
    only major node never wired to BudgetGuard, so `research.api_call_log`
    sat empty and no cost claim about this phase could be checked against
    reality. BudgetGuard.commit already writes the row — this just calls it.
    """
    if budget is None:
        return
    settings = get_settings()
    tokens_in = usage.get("input_tokens") or 0
    tokens_out = usage.get("output_tokens") or 0
    cost = (
        Decimal(tokens_in) / 1000 * settings.cost_openai_per_1k_input
        + Decimal(tokens_out) / 1000 * settings.cost_openai_per_1k_output
    )
    await budget.commit(
        provider=settings.llm_provider,
        operation="extract",
        actual=cost,
        node="extractor",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=usage.get("latency_ms"),
    )


async def _extract_from_chunk(
    chunk: str,
    source: Source,
    legal_issues: list[str],
    budget: BudgetGuard | None = None,
) -> list[EvidenceCandidate]:
    issues_rendered = "\n".join(f"  {i}. {issue}" for i, issue in enumerate(legal_issues, 1))
    date_str = source.decided_or_published_on.isoformat() if source.decided_or_published_on else "unknown"

    user = _USER_TEMPLATE.format(
        title=source.title or "(untitled)",
        source_type=source.source_type.value,
        issuing_body=source.issuing_body or "unknown",
        date=date_str,
        issues=issues_rendered,
        document_text=chunk,
    )

    llm = get_llm()
    try:
        result, usage = await llm.generate(
            system=_SYSTEM,
            user=user,
            output_schema=LLMEvidenceBatch,
            tool_name="emit_evidence",
            max_tokens=4096,
            temperature=0.0,
        )
    except Exception as e:
        log.warning(
            "extractor_llm_failed",
            source_id=source.source_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return []

    candidates = [
        EvidenceCandidate(
            source_id=source.source_id,
            kind=item.kind,
            statement=item.statement,
            verbatim_quote=item.verbatim_quote,
            pinpoint=item.pinpoint,
            citation_raw=item.citation_raw,
            supports_issues=item.supports_issues,
            llm_confidence=item.llm_confidence,
        )
        for item in result.items
    ]

    await _record_llm_cost(budget, usage, source.source_id)

    log.info(
        "extractor_chunk_done",
        source_id=source.source_id,
        candidate_count=len(candidates),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )
    return candidates


async def extract_evidence_for_source(
    source: Source,
    document: SourceDocument,
    legal_issues: list[str],
    budget: BudgetGuard | None = None,
) -> list[EvidenceCandidate]:
    """Extract evidence candidates from one source's document text.

    Long documents are chunked and EVERY chunk within the source's tier
    budget gets its own call — a judgment's ratio and disposition live
    near the end, so stopping early doesn't just read less, it reads the
    wrong part (cover matter and counsel's submissions) and mislabels it.
    """
    max_calls, max_items = _budget_for(source)
    chunks = chunk_text(document.text)

    if len(chunks) <= max_calls:
        # Short enough to read whole — no selection needed, and no
        # omission markers to confuse quoting.
        passes = chunks
        selection = "full"
    else:
        # Too long to send entirely: scan all of it, send the best parts.
        passes = build_digests(document.text, legal_issues, max_calls)
        selection = "digest"

    per_pass: list[list[EvidenceCandidate]] = []
    for chunk in passes:
        per_pass.append(await _extract_from_chunk(chunk, source, legal_issues, budget))

    total = sum(len(c) for c in per_pass)
    log.info(
        "extractor_source_done",
        source_id=source.source_id,
        doc_chars=len(document.text),
        chunks_if_read_whole=len(chunks),
        selection=selection,
        llm_calls=len(passes),
        chars_sent=sum(len(p) for p in passes),
        candidates=total,
    )

    if total <= max_items:
        return [c for pass_items in per_pass for c in pass_items]

    # Over budget: take round-robin across chunks rather than slicing the
    # flat list, which would hand every slot to the earliest chunks and
    # reintroduce the front-of-document bias this function exists to fix.
    trimmed: list[EvidenceCandidate] = []
    for i in range(max(len(c) for c in per_pass)):
        for pass_items in per_pass:
            if i < len(pass_items):
                trimmed.append(pass_items[i])
                if len(trimmed) == max_items:
                    return trimmed
    return trimmed


async def extractor_node(
    sources: list[Source],
    documents_by_source_id: dict[str, SourceDocument],
    legal_issues: list[str],
    budget: BudgetGuard | None = None,
) -> dict[str, Any]:
    """STEP 4c: extract evidence candidates from every fetched source.

    Takes explicit args rather than ResearchState because documents are
    not carried in graph state (state carries IDs, Postgres carries
    text) — the caller reads documents from the DB/dispatch result and
    passes them in directly.
    """
    ctx_node.set("extractor")

    fetched_sources = [s for s in sources if s.status == "fetched"]
    log.info("extractor_start", source_count=len(fetched_sources))

    all_candidates: list[EvidenceCandidate] = []
    for source in fetched_sources:
        doc = documents_by_source_id.get(source.source_id)
        if doc is None:
            log.warning("extractor_missing_document", source_id=source.source_id)
            continue
        candidates = await extract_evidence_for_source(source, doc, legal_issues, budget)
        all_candidates.extend(candidates)

    log.info("extractor_done", total_candidates=len(all_candidates))

    return {"evidence_candidates": all_candidates}
