"""STEP 5 — gap_check node (completeness check).

Two-stage design mirroring the evaluator: Stage A (app.domain.coverage)
is pure deterministic counting/matching and catches the gaps that don't
need judgment — zero coverage on an issue, no tier-1/2 evidence, a
missing statute. Stage B is one LLM call for the gaps that genuinely
need reading comprehension: unresolved conflicts between sources, a
counter-view that's present but too thin to matter, missing
implementation detail. The LLM never decides verdict or severity for
rule-detected gaps — only proposes NEW gaps rule-based detection can't
see, and even those get a severity we sanity-check.

verdict is fully deterministic given the gap list, loop_index, and the
request's max_research_loops — never LLM-decided, for the same reason
authority tiers aren't: the repair loop's termination must be provable,
not persuadable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import ctx_node, get_logger
from app.domain.coverage import compute_coverage, compute_tier_histogram, detect_rule_based_gaps
from app.providers.llm.base import get_llm
from app.schemas.evidence import Evidence, QuotaShortfall
from app.schemas.gaps import Gap, GapKind, GapReport
from app.schemas.plan import ResearchPlan
from app.schemas.request import ResearchRequest
from app.schemas.source import Source
from app.schemas.state import ResearchState

log = get_logger()

# Gaps below this severity never trigger another research loop on their
# own — "nice_to_have" gaps are surfaced in the report but don't cost a
# repair-loop round-trip.
BLOCKING_SEVERITIES = {"blocking"}


# ---------------------------------------------------------------------
# Stage B — LLM detector for gaps that need reading comprehension
# ---------------------------------------------------------------------

class LLMGap(BaseModel):
    gap_kind: str
    issue: str | None = None
    detail: str = Field(max_length=400)
    severity: str


class LLMGapBatch(BaseModel):
    gaps: list[LLMGap] = Field(max_length=10)


_GAP_SYSTEM = """You review a legal research evidence set for coverage gaps a mechanical count can't catch. You do not have access to full source text — only statements, kinds, and binding strength — so you cannot verify quotes; you are judging coverage and coherence, not correctness.

Look specifically for:
- unresolved_conflict: two or more evidence items that state incompatible positions with nothing in the set resolving which controls.
- missing_counter_view: a counter-view exists in form (a commentary_opinion item) but is too thin, off-point, or one-sided to actually represent the opposing position.
- no_implementation_detail: the issue is covered in principle (holdings/statute) but nothing addresses how it actually gets applied procedurally.
- thin_coverage: an issue that is technically non-zero but the evidence is too repetitive or shallow to support a real analysis.

Do not re-flag anything a mechanical count would already catch (zero evidence on an issue, zero tier-1/2 evidence, zero commentary at all) — assume those are handled separately. Only report gaps that require actually reading and comparing the statements."""

_GAP_USER_TEMPLATE = """Topic: {topic}

Legal issues:
{issues}

Evidence (id | kind | binding_strength | supports_issues | statement):
{evidence_lines}

Report at most 10 gaps. severity must be one of: blocking, important, nice_to_have. gap_kind must be one of: {gap_kinds}."""


async def _detect_llm_gaps(
    evidence: list[Evidence],
    plan: ResearchPlan,
) -> list[Gap]:
    if not evidence:
        return []

    issue_lines = "\n".join(f"  {i}. {issue}" for i, issue in enumerate(plan.legal_issues, 1))
    evidence_lines = "\n".join(
        f"{e.evidence_id} | {e.kind} | {e.binding_strength} | {e.supports_issues} | {e.statement}"
        for e in evidence
    )
    valid_kinds = [k.value for k in GapKind]

    user = _GAP_USER_TEMPLATE.format(
        topic=plan.topic_restated,
        issues=issue_lines,
        evidence_lines=evidence_lines,
        gap_kinds=", ".join(valid_kinds),
    )

    llm = get_llm()
    try:
        result, usage = await llm.generate(
            system=_GAP_SYSTEM,
            user=user,
            output_schema=LLMGapBatch,
            tool_name="emit_gaps",
            max_tokens=2048,
            temperature=0.0,
        )
    except Exception as e:
        log.warning("gap_check_llm_failed", error=str(e), error_type=type(e).__name__)
        return []

    gaps: list[Gap] = []
    for g in result.gaps:
        if g.gap_kind not in valid_kinds:
            log.warning("gap_check_llm_invalid_kind", gap_kind=g.gap_kind)
            continue
        if g.severity not in ("blocking", "important", "nice_to_have"):
            g.severity = "important"  # don't drop the finding over a formatting slip
        gaps.append(
            Gap(
                gap_kind=GapKind(g.gap_kind),
                issue=g.issue,
                detail=g.detail,
                severity=g.severity,  # type: ignore[arg-type]
                detected_by="llm",
            )
        )

    log.info("gap_check_llm_done", gap_count=len(gaps))
    return gaps


# ---------------------------------------------------------------------
# Verdict — deterministic
# ---------------------------------------------------------------------

def _compute_verdict(
    gaps: list[Gap], loop_index: int, max_research_loops: int
) -> tuple[str, str]:
    blocking = [g for g in gaps if g.severity in BLOCKING_SEVERITIES]
    important = [g for g in gaps if g.severity == "important"]

    if not blocking:
        rationale = (
            f"No blocking gaps. {len(important)} important gap(s) noted but not "
            "repair-worthy on their own."
            if important
            else "No blocking gaps found; coverage meets the minimum bar."
        )
        return "complete", rationale

    if loop_index >= max_research_loops:
        rationale = (
            f"{len(blocking)} blocking gap(s) remain but the run has used all "
            f"{max_research_loops} research loop(s) (currently at loop {loop_index}). "
            "Proceeding with the evidence gathered rather than looping indefinitely."
        )
        return "exhausted", rationale

    rationale = (
        f"{len(blocking)} blocking gap(s) found at loop {loop_index} "
        f"(of {max_research_loops} allowed) — repair loop triggered."
    )
    return "needs_more_research", rationale


# ---------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------

async def gap_check_node(state: ResearchState) -> dict[str, Any]:
    """STEP 5: assess coverage/quality of the evidence gathered so far
    and decide whether another research loop is needed."""
    ctx_node.set("gap_check")

    request: ResearchRequest = state["request"]
    plan: ResearchPlan = state["plan"]
    evidence: list[Evidence] = state.get("evidence", [])
    sources: list[Source] = state.get("sources", [])
    quota_shortfalls: list[QuotaShortfall] = state.get("quota_shortfalls", [])
    loop_index = state.get("loop_index", 0)

    sources_by_id = {s.source_id: s for s in sources}

    log.info("gap_check_start", loop_index=loop_index, evidence_count=len(evidence))

    coverage = compute_coverage(evidence, plan.legal_issues)
    tier_histogram = compute_tier_histogram(evidence)

    rule_gaps = detect_rule_based_gaps(evidence, sources_by_id, plan, quota_shortfalls)
    llm_gaps = await _detect_llm_gaps(evidence, plan)

    all_gaps = rule_gaps + llm_gaps

    verdict, rationale = _compute_verdict(all_gaps, loop_index, request.max_research_loops)

    gap_report = GapReport(
        run_id=state["run_id"],
        loop_index=loop_index,
        coverage=coverage,
        tier_histogram=tier_histogram,
        gaps=all_gaps,
        verdict=verdict,  # type: ignore[arg-type]
        rationale=rationale,
    )

    log.info(
        "gap_check_done",
        loop_index=loop_index,
        gap_count=len(all_gaps),
        blocking_count=sum(1 for g in all_gaps if g.severity == "blocking"),
        important_count=sum(1 for g in all_gaps if g.severity == "important"),
        verdict=verdict,
    )

    return {"gap_report": gap_report, "gap_history": [gap_report]}
