"""STEP 5a — deterministic coverage computation and rule-based gap
detectors. No LLM involved: every gap here follows mechanically from
counting/matching evidence against the plan. The LLM detector in
app.graph.nodes.gap_check handles subtler gaps (unresolved conflicts,
thin nuance) that counting can't catch.
"""

from __future__ import annotations

from app.schemas.common import CourtLevel, SourceType
from app.schemas.evidence import Evidence, QuotaShortfall
from app.schemas.gaps import Gap, GapKind
from app.schemas.plan import ResearchPlan
from app.schemas.source import Source

MIN_EVIDENCE_PER_ISSUE = 2
STALE_LAW_INTENTS = {"recent_development", "regulatory_action"}


def compute_coverage(evidence: list[Evidence], legal_issues: list[str]) -> dict[str, int]:
    """Coverage is keyed by issue TEXT (matching plan.legal_issues), not
    by numeric id — this is what planner._USER_REPAIR renders directly
    as 'issue: count'. Evidence.supports_issues is expected to carry the
    1-based numeric id string of the issue it addresses (see extractor's
    prompt), matching the same id convention evaluator.py's Stage B uses."""
    coverage: dict[str, int] = {}
    for i, issue in enumerate(legal_issues, 1):
        issue_id = str(i)
        coverage[issue] = sum(1 for e in evidence if issue_id in (e.supports_issues or []))
    return coverage


def compute_tier_histogram(evidence: list[Evidence]) -> dict[int, int]:
    hist: dict[int, int] = {}
    for e in evidence:
        hist[e.authority_tier] = hist.get(e.authority_tier, 0) + 1
    return hist


def _instrument_has_statutory_evidence(
    instrument: str, evidence: list[Evidence], sources_by_id: dict[str, Source]
) -> bool:
    instrument_lower = instrument.lower()
    for e in evidence:
        source = sources_by_id.get(e.source_id)
        if source is None:
            continue
        if source.source_type not in (SourceType.STATUTE, SourceType.SUBORDINATE_LEGISLATION):
            continue
        title = (source.title or "").lower()
        # Both conditions, not either: the evidence must actually BE
        # statutory text, and must come from a source about THIS
        # instrument. With `or`, a statute-typed source whose title
        # merely mentioned the instrument satisfied the quota using
        # commentary evidence — the quota is meant to prove we have the
        # provision's own words, so it has to check for them.
        if instrument_lower[:30] in title and e.kind == "statutory_text":
            return True
    return False


def detect_rule_based_gaps(
    evidence: list[Evidence],
    sources_by_id: dict[str, Source],
    plan: ResearchPlan,
    quota_shortfalls: list[QuotaShortfall],
) -> list[Gap]:
    gaps: list[Gap] = []

    # ---- Promote unresolved selection-time quota shortfalls -----------
    # If the evaluator already couldn't find a candidate for a hard quota,
    # that's still a gap at generation time — don't let the signal die
    # at Step 3 just because gap_check runs new checks of its own.
    for sf in quota_shortfalls:
        kind = {
            "statute_per_instrument": GapKind.MISSING_STATUTORY_BASIS,
            "min_tier12": GapKind.MISSING_PRIMARY_AUTHORITY,
            "source_per_issue": GapKind.THIN_COVERAGE,
        }.get(sf.requirement)
        if kind is None:
            continue
        gaps.append(
            Gap(
                gap_kind=kind,
                issue=sf.unmet_for if sf.requirement != "min_tier12" else None,
                detail=f"(from source selection) {sf.detail}",
                severity="blocking" if sf.requirement != "source_per_issue" else "important",
                detected_by="rule",
            )
        )

    # ---- Quota 1 equivalent at evidence level: statutory text ---------
    for instrument in plan.key_instruments:
        if not _instrument_has_statutory_evidence(instrument, evidence, sources_by_id):
            gaps.append(
                Gap(
                    gap_kind=GapKind.MISSING_STATUTORY_BASIS,
                    issue=None,
                    detail=f"No statutory-text evidence extracted for '{instrument}' "
                    "(a statute source may have been selected but yielded no grounded "
                    "statutory_text evidence, or fetch/extraction failed for it).",
                    severity="blocking",
                    detected_by="rule",
                )
            )

    # ---- Primary authority floor ---------------------------------------
    has_tier12_evidence = any(e.authority_tier <= 2 for e in evidence)
    if not has_tier12_evidence:
        gaps.append(
            Gap(
                gap_kind=GapKind.MISSING_PRIMARY_AUTHORITY,
                issue=None,
                detail="No evidence grounded from a tier-1/2 (statute/apex court/regulator) "
                "source — the article would have no binding legal basis.",
                severity="blocking",
                detected_by="rule",
            )
        )

    # ---- Apex ruling, only if case law was actually sought ------------
    sought_case_law = any(sq.intent.value == "case_law" for sq in plan.sub_queries)
    has_apex_ruling = any(
        sources_by_id.get(e.source_id) is not None
        and sources_by_id[e.source_id].court_level == CourtLevel.SUPREME_COURT
        for e in evidence
    )
    if sought_case_law and not has_apex_ruling:
        gaps.append(
            Gap(
                gap_kind=GapKind.MISSING_APEX_RULING,
                issue=None,
                detail="No Supreme Court of India ruling grounded as evidence, despite "
                "case law being sought — may genuinely not exist for this topic, but "
                "worth one more targeted pass before concluding that.",
                severity="important",
                detected_by="rule",
            )
        )

    # ---- Counter-view presence (coarse; LLM detector refines this) ----
    has_commentary = any(e.kind == "commentary_opinion" for e in evidence)
    if not has_commentary:
        gaps.append(
            Gap(
                gap_kind=GapKind.MISSING_COUNTER_VIEW,
                issue=None,
                detail="No commentary/opinion evidence at all — the article would present "
                "only primary sources with no critical or alternative viewpoint.",
                severity="important",
                detected_by="rule",
            )
        )

    # ---- Thin coverage per legal issue ---------------------------------
    coverage = compute_coverage(evidence, plan.legal_issues)
    for issue, count in coverage.items():
        if count == 0:
            gaps.append(
                Gap(
                    gap_kind=GapKind.THIN_COVERAGE,
                    issue=issue,
                    detail=f"Zero evidence items support this issue: '{issue}'",
                    severity="blocking",
                    detected_by="rule",
                )
            )
        elif count < MIN_EVIDENCE_PER_ISSUE:
            gaps.append(
                Gap(
                    gap_kind=GapKind.THIN_COVERAGE,
                    issue=issue,
                    detail=f"Only {count} evidence item(s) support '{issue}' "
                    f"(minimum {MIN_EVIDENCE_PER_ISSUE} recommended).",
                    severity="important",
                    detected_by="rule",
                )
            )

    # ---- Recent-development staleness ----------------------------------
    sought_recent = any(sq.intent.value in STALE_LAW_INTENTS for sq in plan.sub_queries)
    has_recent_evidence = any(
        e.as_of is not None
        for e in evidence
        if sources_by_id.get(e.source_id)
        and sources_by_id[e.source_id].source_type
        in (SourceType.REGULATOR_CIRCULAR, SourceType.GAZETTE_NOTIFICATION, SourceType.LEGAL_NEWS)
    )
    if sought_recent and not has_recent_evidence:
        gaps.append(
            Gap(
                gap_kind=GapKind.STALE_LAW,
                issue=None,
                detail="Recent-development queries were planned but no dated regulator "
                "circular, gazette notification, or legal-news evidence was grounded — "
                "the article may be missing the current state of the law.",
                severity="important",
                detected_by="rule",
            )
        )

    return gaps
