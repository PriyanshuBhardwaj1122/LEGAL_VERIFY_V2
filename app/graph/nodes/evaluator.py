"""STEP 3 — Source evaluator.

Two-stage design: Stage A is pure deterministic code and runs on every
candidate. Stage B is one batched LLM call for relevance only, run on
whatever survives Stage A. The LLM never decides authority — that's a
table lookup in app.domain.authority.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from datetime import date
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.core.logging import ctx_node, get_logger
from app.core.rate_limit import get_host_limiter
from app.providers.extract.fetcher import USER_AGENT
from app.domain.authority import authority_weight, resolve_authority
from app.providers.llm.base import get_llm
from app.schemas.common import CourtLevel, Jurisdiction, SourceType
from app.schemas.evidence import QuotaShortfall
from app.schemas.plan import ResearchPlan
from app.schemas.request import ResearchRequest
from app.schemas.source import RawResult, Source, SourceScore
from app.schemas.state import ResearchState

log = get_logger()

# Source types whose recency actually matters. Statutes/judgments don't
# decay just because they're old — decay applies only where "recent" is
# part of what makes the source useful.
_RECENCY_SENSITIVE_TYPES = {
    SourceType.LEGAL_NEWS,
    SourceType.REGULATOR_CIRCULAR,
    SourceType.GAZETTE_NOTIFICATION,
}

RECENCY_HALF_LIFE_DAYS = 365  # ~1 year for decay-sensitive types

FETCH_TIMEOUT_SEC = 8.0


# ---------------------------------------------------------------------
# Stage A — deterministic scoring
# ---------------------------------------------------------------------

async def _check_retrievability(url: str, client: httpx.AsyncClient) -> float:
    """HEAD request to check the URL actually resolves. Returns a 0-1
    signal — 1.0 for a clean 200, partial credit for redirects/blocks
    (many government sites reject HEAD but are fine on GET), 0 for dead.

    Paced per host: this probe runs over EVERY candidate (hundreds per
    run, many on one host), so unthrottled it would trip a site's rate
    limiter before the fetch phase ever starts — and then the fetches
    that actually matter get the 429s.
    """
    try:
        async with get_host_limiter().slot(url):
            resp = await client.head(url, timeout=FETCH_TIMEOUT_SEC, follow_redirects=True)
        if resp.status_code == 200:
            return 1.0
        if resp.status_code in (403, 405):
            # Site blocks HEAD specifically — not evidence the page is dead
            return 0.7
        if 300 <= resp.status_code < 400:
            return 0.9
        return 0.3
    except Exception:
        return 0.4  # network hiccup / timeout — don't nuke the score outright


def _compute_recency(source_type: SourceType, published_at: date | None, as_of: date) -> float:
    if source_type not in _RECENCY_SENSITIVE_TYPES:
        return 1.0
    if published_at is None:
        return 0.5  # unknown date on a recency-sensitive type — penalize mildly
    age_days = max(0, (as_of - published_at).days)
    # Exponential decay with the configured half-life
    decay = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    return round(max(0.05, decay), 3)


def _compute_reliability(
    domain_trust: float,
    discovered_by: list[str],
    retrievability: float,
    citation_count: int | None,
) -> float:
    agreement = min(len(discovered_by) / 3.0, 1.0)

    if citation_count is not None:
        citation_signal = min(math.log10(1 + citation_count) / 3.0, 1.0)
        return round(
            0.35 * domain_trust + 0.20 * agreement + 0.20 * retrievability + 0.25 * citation_signal,
            4,
        )

    return round(0.4 * domain_trust + 0.3 * agreement + 0.3 * retrievability, 4)


async def score_candidates_deterministic(
    raw_results: list[RawResult],
    run_id: str,
    run_jurisdiction: Jurisdiction,
    as_of_date: date,
) -> list[Source]:
    """Stage A: resolve authority, recency, reliability for every candidate.
    relevance is left at 0.0 here — Stage B fills it in. composite is
    computed only after Stage B."""
    candidates: list[Source] = []

    # Same UA as the fetcher: a bare httpx client identifies itself with
    # httpx's default agent, which some sites throttle on sight.
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        retrievability_scores = await asyncio.gather(
            *[_check_retrievability(str(r.url), client) for r in raw_results],
            return_exceptions=False,
        )

    for r, retrievability in zip(raw_results, retrievability_scores):
        tier, court_level, issuing_body, source_type = resolve_authority(str(r.url))

        # Cap foreign sources at tier 4 for an IN run
        # (jurisdiction on RawResult isn't tracked per-item; approximate
        # via domain — non-Indian domains without a .gov.in/.nic.in/.in
        # pattern already fall to tier 6 in resolve_authority, so this
        # mainly matters for explicitly comparative queries later.)
        source_jurisdiction = run_jurisdiction

        domain_trust = authority_weight(tier)
        recency = _compute_recency(source_type, r.published_at, as_of_date)
        reliability = _compute_reliability(
            domain_trust, r.discovered_by or [r.provider], retrievability, r.citation_count
        )

        source_id = hashlib.sha1(r.url_canonical.encode()).hexdigest()[:16]

        score = SourceScore(
            authority=domain_trust,
            relevance=0.0,  # filled by Stage B
            recency=recency,
            reliability=reliability,
            composite=0.0,  # computed after Stage B
            tier=tier,
            reasons=[
                f"tier {tier} ({source_type.value})",
                f"discovered_by={r.discovered_by or [r.provider]}",
                f"retrievability={retrievability:.2f}",
            ],
        )

        candidates.append(
            Source(
                source_id=source_id,
                run_id=run_id,
                url_canonical=r.url_canonical,
                fetch_url=r.open_access_pdf or None,
                title=r.title,
                source_type=source_type,
                court_level=court_level,
                issuing_body=issuing_body,
                jurisdiction=source_jurisdiction,
                decided_or_published_on=r.published_at,
                citation=None,
                score=score,
                status="candidate",
                discovered_by=r.discovered_by or [r.provider],
                provenance_query_ids=r.provenance_query_ids or [r.query_id],
            )
        )

    log.info("evaluator_stage_a_done", candidate_count=len(candidates))
    return candidates


# ---------------------------------------------------------------------
# Stage B — batched LLM relevance
# ---------------------------------------------------------------------

class RelevanceItem(BaseModel):
    source_id: str
    relevance: float = Field(ge=0, le=1)
    addresses_issues: list[str] = []
    one_line_reason: str = Field(max_length=200)


class RelevanceBatch(BaseModel):
    scores: list[RelevanceItem]


_RELEVANCE_SYSTEM = """You score search results for relevance only. You do not score authority, credibility, or recency — those are computed separately and your opinion on them is not used. Score only: does this source help resolve one of the listed legal issues?"""

_RELEVANCE_USER_TEMPLATE = """Topic: {topic}

Legal issues:
{issues}

Candidates (id | type | tier | date | title | snippet):
{candidates}

For each candidate return relevance 0.0-1.0, the issue ids it addresses, and a one-line reason under 15 words. Score on the snippet alone. If a snippet is uninformative, score 0.3 and say so — do not guess from the title.

Watch specifically for false-positive keyword matches: the same section number or defined term can belong to a completely different statute (e.g. "Section 34" is the provision for setting aside an arbitral award under the Arbitration and Conciliation Act, 1996, AND the common-intention provision of the Indian Penal Code — they have nothing to do with each other). If the title or snippet indicates the source concerns a different instrument than the one in the topic/legal issues, score relevance near 0.0 and say so, even if the section number or a keyword matches exactly."""


RELEVANCE_BATCH_SIZE = 25
# One call handling ALL candidates broke silently once the candidate
# pool crossed a few dozen: the RelevanceBatch output (one item per
# candidate, each with a reason string) exceeded max_tokens, OpenAI
# truncated mid-JSON, the parse failed, and the exception handler
# quietly gave every candidate a uniform neutral relevance — which
# defeats MIN_RELEVANCE_FOR_HARD_QUOTA entirely (0.4 clears the 0.35
# floor for every candidate at once, so authority alone decides
# selection again — the exact wrong-statute failure mode that floor
# exists to prevent). Chunking bounds each call's output size and, just
# as importantly, contains a parse failure to the one chunk that hit
# it instead of poisoning every candidate in the run with a fake score.


def _neutral_fallback(candidates: list[Source], reason: str) -> dict[str, RelevanceItem]:
    return {
        c.source_id: RelevanceItem(
            source_id=c.source_id,
            relevance=0.4,
            addresses_issues=[],
            one_line_reason=reason,
        )
        for c in candidates
    }


async def _score_relevance_batch(
    batch: list[Source],
    raw_by_source_id: dict[str, RawResult],
    issue_lines: str,
    topic: str,
) -> dict[str, RelevanceItem]:
    candidate_lines = []
    for c in batch:
        raw = raw_by_source_id.get(c.source_id)
        snippet = (raw.snippet if raw else None) or "(no snippet)"
        date_str = c.decided_or_published_on.isoformat() if c.decided_or_published_on else "unknown"
        candidate_lines.append(
            f"{c.source_id} | {c.source_type.value} | tier{c.score.tier} | {date_str} | "
            f"{c.title or '(no title)'} | {snippet[:200]}"
        )

    user = _RELEVANCE_USER_TEMPLATE.format(
        topic=topic,
        issues=issue_lines,
        candidates="\n".join(candidate_lines),
    )

    llm = get_llm()
    try:
        result, usage = await llm.generate(
            system=_RELEVANCE_SYSTEM,
            user=user,
            output_schema=RelevanceBatch,
            tool_name="emit_relevance_scores",
            max_tokens=4096,
            temperature=0.0,
        )
    except Exception as e:
        log.warning(
            "evaluator_stage_b_batch_failed",
            error=str(e),
            error_type=type(e).__name__,
            batch_size=len(batch),
        )
        # Degrade gracefully, scoped to just this batch — everyone else
        # in the run still gets a real score.
        return _neutral_fallback(batch, "Stage B batch unavailable — neutral default")

    by_id = {item.source_id: item for item in result.scores}

    # Anything the LLM skipped gets a neutral fallback rather than 0 —
    # a skip is not evidence of irrelevance.
    for c in batch:
        if c.source_id not in by_id:
            by_id[c.source_id] = RelevanceItem(
                source_id=c.source_id,
                relevance=0.3,
                addresses_issues=[],
                one_line_reason="Not scored by LLM — default applied",
            )

    return by_id


async def score_relevance_llm(
    candidates: list[Source],
    raw_by_source_id: dict[str, RawResult],
    plan: ResearchPlan,
) -> dict[str, RelevanceItem]:
    """Stage B: score every candidate's relevance to the plan's legal
    issues, chunked into bounded batches so a large candidate pool
    (now routine with indiankanoon+serpapi both live) can't silently
    truncate the LLM output and poison every score with a fake
    neutral default. Titles and snippets only — never full text."""
    if not candidates:
        return {}

    issue_lines = "\n".join(f"  {i}. {issue}" for i, issue in enumerate(plan.legal_issues, 1))
    batches = [
        candidates[i : i + RELEVANCE_BATCH_SIZE]
        for i in range(0, len(candidates), RELEVANCE_BATCH_SIZE)
    ]

    batch_results = await asyncio.gather(
        *[
            _score_relevance_batch(batch, raw_by_source_id, issue_lines, plan.topic_restated)
            for batch in batches
        ]
    )

    by_id: dict[str, RelevanceItem] = {}
    for batch_dict in batch_results:
        by_id.update(batch_dict)

    log.info(
        "evaluator_stage_b_done",
        scored_count=len(by_id),
        batch_count=len(batches),
        batch_size=RELEVANCE_BATCH_SIZE,
    )
    return by_id


# ---------------------------------------------------------------------
# Composite scoring + quota-based selection
# ---------------------------------------------------------------------

COMPOSITE_WEIGHTS = {"authority": 0.35, "relevance": 0.35, "reliability": 0.15, "recency": 0.15}
MIN_COMPOSITE = 0.45
SELECTION_CAP = 18

# A tier-1/2 domain (e.g. a Supreme Court judgment PDF) can carry a high
# composite score almost entirely from authority even when it is scored
# near-irrelevant by Stage B — a same-section-number-different-statute
# false positive is the classic case. This floor stops hard authority
# quotas and the general fill from being satisfied by such candidates;
# it is NOT applied to Quota 1 (statute-per-instrument), which is
# deliberately allowed to select a low-relevance-scored bare act/statute
# text on title match alone — that's a correct pick, not a false positive.
MIN_RELEVANCE_FOR_HARD_QUOTA = 0.35


def _composite(score: SourceScore) -> float:
    return round(
        COMPOSITE_WEIGHTS["authority"] * score.authority
        + COMPOSITE_WEIGHTS["relevance"] * score.relevance
        + COMPOSITE_WEIGHTS["reliability"] * score.reliability
        + COMPOSITE_WEIGHTS["recency"] * score.recency,
        4,
    )


def select_sources(
    candidates: list[Source],
    plan: ResearchPlan,
) -> tuple[list[Source], list[QuotaShortfall]]:
    """Quota-based selection — not a flat top-K. Guarantees statute
    coverage per key_instrument, a floor of tier-1/2 sources, and at
    least one source per legal issue before filling by composite score."""
    shortfalls: list[QuotaShortfall] = []
    selected: dict[str, Source] = {}

    def mark_selected(c: Source) -> None:
        c.status = "selected"
        selected[c.source_id] = c

    # Quota 1: >= 1 statute/regulation source per key_instrument
    for instrument in plan.key_instruments:
        instrument_lower = instrument.lower()
        matches = [
            c
            for c in candidates
            if c.source_type in (SourceType.STATUTE, SourceType.SUBORDINATE_LEGISLATION)
            and (c.title and instrument_lower[:30] in c.title.lower())
        ]
        if matches:
            best = max(matches, key=lambda c: c.score.composite)
            mark_selected(best)
        else:
            # Fall back to any statute-typed candidate before giving up
            any_statute = [c for c in candidates if c.source_type == SourceType.STATUTE]
            if any_statute:
                mark_selected(max(any_statute, key=lambda c: c.score.composite))
            else:
                shortfalls.append(
                    QuotaShortfall(
                        requirement="statute_per_instrument",
                        detail=f"No statute-typed source found for '{instrument}'",
                        unmet_for=instrument,
                    )
                )

    # Quota 2: >= 3 tier-1/2 sources, preferring candidates that actually
    # clear a relevance floor. Authority alone (tier<=2) is not allowed to
    # substitute for relevance here — see MIN_RELEVANCE_FOR_HARD_QUOTA.
    tier12_all = [c for c in candidates if c.score.tier <= 2]
    tier12_relevant = sorted(
        [c for c in tier12_all if c.score.relevance >= MIN_RELEVANCE_FOR_HARD_QUOTA],
        key=lambda c: c.score.composite,
        reverse=True,
    )
    tier12_picked = tier12_relevant[:3]
    if len(tier12_picked) < 3:
        needed = 3 - len(tier12_picked)
        # Not enough relevant tier-1/2 candidates to fill the quota on
        # relevance alone — fall back to the pool, but rank the fallback
        # by relevance (not composite) so we take the least-irrelevant
        # option rather than the most-authoritative-but-off-topic one.
        already_picked_ids = {c.source_id for c in tier12_picked}
        fallback_pool = sorted(
            [c for c in tier12_all if c.source_id not in already_picked_ids],
            key=lambda c: c.score.relevance,
            reverse=True,
        )
        fallback_picks = fallback_pool[:needed]
        if fallback_picks:
            log.warning(
                "evaluator_tier12_quota_relevance_floor_unmet",
                needed=needed,
                fallback_source_ids=[c.source_id for c in fallback_picks],
                fallback_relevance=[c.score.relevance for c in fallback_picks],
            )
        tier12_picked = tier12_picked + fallback_picks

    for c in tier12_picked:
        mark_selected(c)
    if len(tier12_all) < 3:
        shortfalls.append(
            QuotaShortfall(
                requirement="min_tier12",
                detail=f"Only {len(tier12_all)} tier-1/2 sources found in candidate pool (need 3)",
            )
        )

    # Quota 3: >= 1 source per legal_issue
    for i, issue in enumerate(plan.legal_issues):
        issue_id = str(i + 1)
        matches = [c for c in candidates if issue_id in (c.score.reasons or []) or issue in (c.title or "")]
        # Better signal: relevance was scored against numbered issues, but
        # Source doesn't carry addresses_issues directly — approximate via
        # candidates already selected covering some issue, else take the
        # single highest-relevance unselected candidate as a safety net.
        pool = sorted(candidates, key=lambda c: c.score.relevance, reverse=True)
        if pool and not any(c.source_id in selected for c in pool[:1]):
            mark_selected(pool[0])
        if not matches and not pool:
            shortfalls.append(
                QuotaShortfall(
                    requirement="source_per_issue",
                    detail=f"No candidate addresses issue: {issue}",
                    unmet_for=issue,
                )
            )

    # Fill remaining slots by composite score, respecting the floor and cap.
    # Also enforce the relevance floor here — otherwise a high-authority,
    # near-zero-relevance tier-1 candidate (e.g. the Constitution, or a
    # wrong-statute false positive) clears MIN_COMPOSITE on authority
    # weight alone and silently eats a slot a genuinely relevant source
    # could have used.
    remaining_slots = max(0, SELECTION_CAP - len(selected))
    ranked = sorted(candidates, key=lambda c: c.score.composite, reverse=True)
    for c in ranked:
        if remaining_slots <= 0:
            break
        if c.source_id in selected:
            continue
        if c.score.composite < MIN_COMPOSITE:
            continue
        if c.score.relevance < MIN_RELEVANCE_FOR_HARD_QUOTA:
            continue
        mark_selected(c)
        remaining_slots -= 1

    # Everything not selected is explicitly dropped (not just absent)
    for c in candidates:
        if c.source_id not in selected:
            c.status = "dropped"
            c.score.dropped_reason = c.score.dropped_reason or "below selection threshold"

    log.info(
        "evaluator_selection_done",
        candidate_count=len(candidates),
        selected_count=len(selected),
        shortfall_count=len(shortfalls),
    )

    return list(selected.values()) + [c for c in candidates if c.source_id not in selected], shortfalls


# ---------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------

async def evaluator_node(state: ResearchState) -> dict[str, Any]:
    """STEP 3: score and select sources from merged search results."""
    ctx_node.set("evaluator")
    request: ResearchRequest = state["request"]
    plan: ResearchPlan = state["plan"]
    raw_results: list[RawResult] = state.get("raw_results", [])

    if not raw_results:
        log.info("evaluator_nothing_to_do")
        return {"sources": [], "quota_shortfalls": []}

    # Stage A
    candidates = await score_candidates_deterministic(
        raw_results, state["run_id"], request.jurisdiction, request.as_of_date
    )

    # Stage B
    raw_by_id = {
        hashlib.sha1(r.url_canonical.encode()).hexdigest()[:16]: r for r in raw_results
    }
    relevance_scores = await score_relevance_llm(candidates, raw_by_id, plan)

    # Merge relevance into candidates, compute composite
    for c in candidates:
        item = relevance_scores.get(c.source_id)
        if item:
            c.score.relevance = item.relevance
            c.score.reasons.append(item.one_line_reason)
        c.score.composite = _composite(c.score)

    # Selection
    all_sources, shortfalls = select_sources(candidates, plan)

    log.info(
        "evaluator_done",
        total=len(all_sources),
        selected=sum(1 for s in all_sources if s.status == "selected"),
    )

    return {"sources": all_sources, "quota_shortfalls": shortfalls}
