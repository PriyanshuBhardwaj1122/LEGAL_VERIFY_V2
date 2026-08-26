"""STEP 4d — grounding node. Deterministic quote validator, no LLM.

This is the single highest-leverage anti-hallucination control in the
pipeline: every extracted fact must carry a verbatim quote that this
code re-finds in the stored source text. If the quote is not found, the
evidence object is rejected — no LLM judgment involved.

Order: exact match -> normalized match -> fuzzy match -> drop.
Hard invariants (§2.6) are enforced after grounding and can still drop
an item even if a quote was located — grounding success alone is not
sufficient for a binding legal proposition.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from typing import Literal

from rapidfuzz import fuzz

from app.core.logging import ctx_node, get_logger
from app.domain.citation_parser import parse_citation
from app.schemas.common import CourtLevel, Jurisdiction
from app.schemas.evidence import Evidence, EvidenceCandidate, NodeError
from app.schemas.source import Source, SourceDocument

log = get_logger()

FUZZY_THRESHOLD = 92


# ---------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------

def _normalize_for_match(s: str) -> tuple[str, list[int]]:
    """Normalize whitespace and common quote-mark/punctuation variants
    for the 'normalized' matching tier. Returns (normalized_string,
    index_map) where index_map[i] is the offset in the ORIGINAL string
    that normalized character i corresponds to — this is what lets us
    map a match found in normalized space back to real offsets."""
    quote_map = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", " ": " ",
    }

    out_chars: list[str] = []
    index_map: list[int] = []
    prev_was_space = False

    for i, ch in enumerate(s):
        mapped = quote_map.get(ch, ch)
        is_space = mapped.isspace()
        if is_space:
            if prev_was_space:
                continue
            out_chars.append(" ")
            index_map.append(i)
            prev_was_space = True
        else:
            out_chars.append(mapped)
            index_map.append(i)
            prev_was_space = False

    return "".join(out_chars), index_map


def _exact_match(doc_text: str, quote: str) -> tuple[int, int] | None:
    idx = doc_text.find(quote)
    if idx == -1:
        return None
    return idx, idx + len(quote)


def _normalized_match(doc_text: str, quote: str) -> tuple[int, int] | None:
    norm_doc, doc_map = _normalize_for_match(doc_text)
    norm_quote, _ = _normalize_for_match(quote)

    if not norm_quote:
        return None

    idx = norm_doc.find(norm_quote)
    if idx == -1:
        return None

    end_norm = idx + len(norm_quote) - 1
    if end_norm >= len(doc_map):
        end_norm = len(doc_map) - 1

    start_orig = doc_map[idx]
    end_orig = doc_map[end_norm] + 1
    return start_orig, end_orig


def _fuzzy_match(doc_text: str, quote: str) -> tuple[int, int, str] | None:
    """Coarse-to-fine sliding window fuzzy search. Returns
    (start, end, actual_matched_text) — the caller rewrites
    verbatim_quote to actual_matched_text, never the model's version."""
    qlen = len(quote)
    if qlen < 10 or qlen > 5000:
        return None

    window = qlen
    # Bound total iterations on very long documents — step coarser
    # rather than scanning every character.
    step = max(1, qlen // 4)
    max_windows = 4000
    n_windows = max(1, (len(doc_text) - window) // step + 1)
    if n_windows > max_windows:
        step = max(step, (len(doc_text) - window) // max_windows)

    best_score = 0.0
    best_pos = -1

    pos = 0
    while pos + window <= len(doc_text):
        candidate = doc_text[pos : pos + window]
        score = fuzz.partial_ratio(quote, candidate)
        if score > best_score:
            best_score = score
            best_pos = pos
        pos += step

    if best_score < FUZZY_THRESHOLD or best_pos == -1:
        return None

    # Refine: search a small neighborhood around best_pos at finer
    # granularity to tighten the window boundaries.
    refine_start = max(0, best_pos - window // 2)
    refine_end = min(len(doc_text), best_pos + window + window // 2)
    best_refined_score = best_score
    best_refined_span = (best_pos, best_pos + window)

    p = refine_start
    while p + window <= refine_end:
        candidate = doc_text[p : p + window]
        score = fuzz.partial_ratio(quote, candidate)
        if score > best_refined_score:
            best_refined_score = score
            best_refined_span = (p, p + window)
        p += max(1, step // 4)

    start, end = best_refined_span
    end = min(end, len(doc_text))
    return start, end, doc_text[start:end]


def ground_quote(
    doc_text: str, quote: str
) -> tuple[Literal["exact", "normalized", "fuzzy", "ungrounded"], int | None, int | None, str]:
    """Returns (grounding_level, start, end, resolved_quote_text).
    resolved_quote_text is the model's quote for exact/normalized hits,
    or the document's actual text for a fuzzy hit — never trust the
    model's characters when it wasn't verified exactly."""
    match = _exact_match(doc_text, quote)
    if match:
        return "exact", match[0], match[1], quote

    match = _normalized_match(doc_text, quote)
    if match:
        return "normalized", match[0], match[1], quote

    fmatch = _fuzzy_match(doc_text, quote)
    if fmatch:
        start, end, actual_text = fmatch
        return "fuzzy", start, end, actual_text

    return "ungrounded", None, None, quote


# ---------------------------------------------------------------------
# Binding strength (deterministic, never LLM)
# ---------------------------------------------------------------------

def compute_binding_strength(
    kind: str,
    tier: int,
    court_level: CourtLevel,
    source_jurisdiction: Jurisdiction,
    run_jurisdiction: Jurisdiction,
) -> Literal["binding", "persuasive", "informative", "opinion"]:
    if kind == "commentary_opinion":
        return "opinion"

    jurisdiction_matches = source_jurisdiction == run_jurisdiction

    if tier <= 2 and jurisdiction_matches:
        return "binding"
    if tier <= 2 and not jurisdiction_matches:
        return "persuasive"
    if tier == 3:
        return "persuasive" if jurisdiction_matches else "informative"
    return "informative"


# ---------------------------------------------------------------------
# Invariant enforcement (§2.6) — any failure drops the item, no repair
# ---------------------------------------------------------------------

def _passes_invariants(
    kind: str,
    grounding: str,
    verbatim_quote: str | None,
    binding_strength: str,
    tier: int,
    source_jurisdiction: Jurisdiction,
    run_jurisdiction: Jurisdiction,
    is_quotable: bool,
) -> tuple[bool, str | None]:
    if not is_quotable and kind != "commentary_opinion":
        return False, "source document is not quotable (OCR/low-confidence) and kind != commentary_opinion"

    if kind != "commentary_opinion":
        if verbatim_quote is None or grounding not in ("exact", "normalized"):
            return False, f"non-commentary evidence requires exact/normalized grounding, got '{grounding}'"

    if binding_strength == "binding":
        if tier > 2:
            return False, f"binding_strength=binding requires authority_tier<=2, got tier={tier}"
        if source_jurisdiction != run_jurisdiction:
            return False, "binding_strength=binding requires jurisdiction to match the run"

    return True, None


# ---------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------

async def grounding_node(
    candidates: list[EvidenceCandidate],
    sources_by_id: dict[str, Source],
    documents_by_source_id: dict[str, SourceDocument],
    run_id: str,
    run_jurisdiction: Jurisdiction,
    loop_index: int = 0,
) -> dict:
    """STEP 4d: validate every EvidenceCandidate against its source
    document, compute offsets/binding_strength/citation, drop anything
    that fails grounding or an invariant."""
    ctx_node.set("grounding")

    evidence: list[Evidence] = []
    errors: list[NodeError] = []
    grounding_counts = {"exact": 0, "normalized": 0, "fuzzy": 0, "ungrounded": 0}
    invariant_drops = 0

    for cand in candidates:
        source = sources_by_id.get(cand.source_id)
        doc = documents_by_source_id.get(cand.source_id)

        if source is None or doc is None:
            log.warning("grounding_missing_source_or_doc", source_id=cand.source_id)
            continue

        if cand.kind == "commentary_opinion" or cand.verbatim_quote is None:
            grounding_level: Literal["exact", "normalized", "fuzzy", "ungrounded"] = "exact" if cand.verbatim_quote is None else "ungrounded"
            start, end, resolved_quote = None, None, cand.verbatim_quote
            if cand.verbatim_quote is not None:
                grounding_level, start, end, resolved_quote = ground_quote(doc.text, cand.verbatim_quote)
        else:
            grounding_level, start, end, resolved_quote = ground_quote(doc.text, cand.verbatim_quote)

        grounding_counts[grounding_level] = grounding_counts.get(grounding_level, 0) + 1

        # commentary_opinion never REQUIRES a quote, but if the model
        # supplied one and it can't be located in the document, the quote
        # is unverified and must not be persisted as if it were — strip
        # it rather than either fabricating a citation or dropping a
        # legitimate (quote-less) opinion statement outright.
        if cand.kind == "commentary_opinion" and grounding_level == "ungrounded":
            log.info(
                "grounding_commentary_quote_unverified_stripped",
                source_id=cand.source_id,
                statement=cand.statement[:100],
                quote_preview=(cand.verbatim_quote or "")[:100],
            )
            resolved_quote, start, end = None, None, None

        if grounding_level == "ungrounded" and cand.kind != "commentary_opinion":
            log.warning(
                "grounding_failed",
                source_id=cand.source_id,
                statement=cand.statement[:100],
                quote_preview=(cand.verbatim_quote or "")[:100],
            )
            errors.append(
                NodeError(
                    node="grounding",
                    kind="extraction_failed",
                    detail=f"Quote not found in source: {cand.statement[:80]}",
                    source_id=cand.source_id,
                    retryable=False,
                    occurred_at=datetime.now(timezone.utc),
                )
            )
            continue

        citation = parse_citation(cand.citation_raw) if cand.citation_raw else None
        binding_strength = compute_binding_strength(
            cand.kind, source.score.tier, source.court_level, source.jurisdiction, run_jurisdiction
        )

        ok, reason = _passes_invariants(
            cand.kind,
            grounding_level,
            resolved_quote,
            binding_strength,
            source.score.tier,
            source.jurisdiction,
            run_jurisdiction,
            doc.is_quotable,
        )
        if not ok:
            invariant_drops += 1
            log.warning(
                "grounding_invariant_failed",
                source_id=cand.source_id,
                kind=cand.kind,
                reason=reason,
            )
            continue

        evidence_id = hashlib.sha1(
            f"{cand.source_id}:{cand.statement}:{start}:{end}".encode()
        ).hexdigest()[:16]

        evidence.append(
            Evidence(
                evidence_id=evidence_id,
                run_id=run_id,
                source_id=cand.source_id,
                loop_index=loop_index,
                kind=cand.kind,  # type: ignore[arg-type]
                statement=cand.statement,
                verbatim_quote=resolved_quote,
                quote_start=start,
                quote_end=end,
                grounding=grounding_level,
                pinpoint=cand.pinpoint,
                citation=citation,
                supports_issues=cand.supports_issues,
                jurisdiction=source.jurisdiction,
                court_level=source.court_level,
                as_of=source.decided_or_published_on,
                binding_strength=binding_strength,
                llm_confidence=cand.llm_confidence,
                authority_tier=source.score.tier,
            )
        )

    total = len(candidates)
    grounded = total - grounding_counts.get("ungrounded", 0) - invariant_drops
    pass_rate = round(grounded / total, 3) if total else 0.0

    log.info(
        "grounding_done",
        total_candidates=total,
        evidence_count=len(evidence),
        grounding_counts=grounding_counts,
        invariant_drops=invariant_drops,
        pass_rate=pass_rate,
    )

    if total and pass_rate < 0.70:
        log.warning(
            "grounding_pass_rate_below_threshold",
            pass_rate=pass_rate,
            message="Grounding pass rate below 70% — extraction prompt may be drifting toward paraphrase",
        )

    return {
        "evidence": evidence,
        "errors": errors,
        "evidence_candidates": [],  # clear the transient buffer
    }
