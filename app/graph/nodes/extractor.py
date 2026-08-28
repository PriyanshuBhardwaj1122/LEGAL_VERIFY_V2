"""STEP 4c — extractor node.

One LLM call per source (or per chunk for long documents), never one
call for all sources — scoping context to a single source is what stops
cross-source citation contamination. The model never emits offsets; it
cannot count characters reliably, and asking it to is a classic source
of silent corruption. Offsets are recovered deterministically in the
grounding step.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import ctx_node, get_logger
from app.providers.extract.normalize import chunk_text
from app.providers.llm.base import get_llm
from app.schemas.evidence import EvidenceCandidate
from app.schemas.source import Source, SourceDocument

log = get_logger()

MAX_ITEMS_PER_SOURCE = 8

_SYSTEM = """You extract citable evidence from a single legal source. Your output feeds an automated verifier that will re-locate every quote you produce in this exact document. A quote that cannot be found is discarded and counts against you.

Rules, in order of importance:
1. Every item except commentary_opinion MUST include verbatim_quote copied character-for-character from the document. Do not fix typos, expand abbreviations, normalize spacing, or translate.
2. If the document does not support a proposition, do not emit it. Emitting nothing is a correct answer.
3. statement is a neutral paraphrase of ONE proposition, under 40 words. Not a summary of the document.
4. Classify honestly. A court's binding rule is `holding`. A passing remark is `obiter`. An author's view is `commentary_opinion`, even if you agree.
5. pinpoint: give the paragraph number for judgments ("para 42") or the provision ("s. 29A(3)(c)") for legislation, only if it appears in the text.
6. Do not carry over knowledge from outside this document. If you recognise the case and remember something not written here, that memory is not evidence.
7. At most 8 items. Choose the propositions an article would actually cite.
8. supports_issues: the legal issues below are numbered. For each item, list the number(s) (as strings, e.g. ["1"] or ["1","3"]) of every issue it helps resolve. Leave it empty only if the item genuinely doesn't bear on any listed issue — most items should map to at least one.
9. citation_raw: copy the source's own formal citation string verbatim when the document states one — a neutral citation ("2023 INSC 456"), reporter citation ("(2023) 5 SCC 1", "AIR 2019 SC 123"), SCC OnLine citation ("2021 SCC OnLine Del 456"), case/writ number ("Civil Appeal No. 1234 of 2020"), statute section ("Section 29A of the Insolvency and Bankruptcy Code, 2016"), or circular number ("SEBI/HO/CFD/CMD/CIR/P/2020/12") — whichever the document actually contains for this item. For a holding or obiter item this is the case's own citation, not the source's URL or page title. Leave it null only when the document genuinely states no formal citation for this proposition (a press article's own byline is not a citation). Never invent one and never supply one from memory rather than the document text."""

_USER_TEMPLATE = """Source: {title} | {source_type} | {issuing_body} | {date}

Legal issues this research must resolve (numbered — reference these numbers in supports_issues):
{issues}

--- DOCUMENT START ---
{document_text}
--- DOCUMENT END ---"""


class LLMEvidenceCandidate(BaseModel):
    kind: str
    statement: str = Field(max_length=400)
    verbatim_quote: str | None = None
    pinpoint: str | None = None
    citation_raw: str | None = None
    supports_issues: list[str] = []
    llm_confidence: float = Field(ge=0, le=1)


class LLMEvidenceBatch(BaseModel):
    items: list[LLMEvidenceCandidate] = Field(max_length=MAX_ITEMS_PER_SOURCE)


async def _extract_from_chunk(
    chunk: str,
    source: Source,
    legal_issues: list[str],
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
) -> list[EvidenceCandidate]:
    """Extract evidence candidates from one source's document text.
    Long documents are chunked; each chunk gets its own call, and results
    are pooled then capped at MAX_ITEMS_PER_SOURCE."""
    chunks = chunk_text(document.text)

    all_candidates: list[EvidenceCandidate] = []
    for chunk in chunks:
        candidates = await _extract_from_chunk(chunk, source, legal_issues)
        all_candidates.extend(candidates)
        if len(all_candidates) >= MAX_ITEMS_PER_SOURCE:
            break

    return all_candidates[:MAX_ITEMS_PER_SOURCE]


async def extractor_node(
    sources: list[Source],
    documents_by_source_id: dict[str, SourceDocument],
    legal_issues: list[str],
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
        candidates = await extract_evidence_for_source(source, doc, legal_issues)
        all_candidates.extend(candidates)

    log.info("extractor_done", total_candidates=len(all_candidates))

    return {"evidence_candidates": all_candidates}
