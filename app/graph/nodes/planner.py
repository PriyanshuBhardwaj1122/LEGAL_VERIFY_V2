"""STEP 1 — Research planner node.

Turns a topic string into a typed, deduplicated, provider-routed search plan.
No API calls to search providers happen before this node has run.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from app.core.logging import ctx_node, get_logger
from app.providers.llm.base import get_llm
from app.providers.search.registry import get_registry
from app.schemas.common import Jurisdiction, QueryIntent, SourceType
from app.schemas.plan import ResearchPlan, SubQuery
from app.schemas.state import ResearchState

log = get_logger()

# How similar a repair-loop-proposed issue has to be to an existing one
# before it's treated as the same issue (reworded) rather than a
# genuinely new one. token_set_ratio is 0-100; 70 tolerates fairly loose
# rewording while still catching "Issue Beta" vs "Issue Beta (reworded)".
_ISSUE_DEDUP_THRESHOLD = 70

# -------------------------------------------------------------------
# LLM output schema (what the model emits via tool use)
# -------------------------------------------------------------------


class LLMSubQuery(BaseModel):
    query_text: str = Field(max_length=350)
    intent: QueryIntent
    rationale: str
    target_source_types: list[SourceType]
    min_authority_tier: int = Field(5, ge=1, le=6)
    jurisdiction: Jurisdiction = Jurisdiction.IN
    date_from: str | None = None  # ISO date string
    date_to: str | None = None
    priority: int = Field(3, ge=1, le=5)


class LLMResearchPlan(BaseModel):
    """Schema forced onto Claude via tool_choice."""

    topic_restated: str
    legal_issues: list[str] = Field(min_length=2, max_length=8)
    key_instruments: list[str] = []
    sub_queries: list[LLMSubQuery] = Field(min_length=3, max_length=12)
    excluded_directions: list[str] = []


# -------------------------------------------------------------------
# Prompt builders
# -------------------------------------------------------------------

_SYSTEM_LOOP0 = """You are a research director at an Indian law firm's knowledge team. You plan research for a scholarly, argumentative article, not a client explainer — the article this feeds will concede a position, convict on a narrow ground, and refuse the popular argument on its own side. Your job is to make sure the record contains, in their own words, every position the article will need to concede, convict, and refuse. You do not answer legal questions. You will be judged on coverage, not on eloquence.

Jurisdiction: India (Union). Comparative material from {comparative} is permitted only where Indian authority is absent or where the article explicitly compares regimes.

Indian research must be grounded in this authority hierarchy — what a source can be cited FOR:
  1. The bare text of the statute or regulation (India Code, Gazette)
  2. Supreme Court of India rulings interpreting it
  3. High Court rulings, appellate tribunal decisions (NCLAT/SAT/ITAT)
  4. Regulator instruments — SEBI/RBI/MCA/CCI circulars, notifications
  5. Law Commission and Parliamentary Committee reports
  6. Academic and professional commentary (never as the basis of a rule)

A separate axis matters as much as authority: distinctiveness — whether a source tells the article something it does not already have. A fifth Supreme Court paragraph restating a settled point is low-distinctiveness even at tier 2. A press report is always the lowest authority tier, and often the highest-distinctiveness source available, for exactly four things: whether the law is currently live (stayed, notified, deferred, challenged); what the prevailing or popular view actually is, stated by its own holders; a figure or statistic no primary source reports (enforcement numbers, uptake, litigation counts); and the gap between what the rule says and how it is actually applied on the ground. Plan queries for those four purposes explicitly — do not treat press as a fallback when primary sources run out; treat it as the only source type that can do those four jobs at all."""

_USER_LOOP0 = """Topic: {topic}
Practice area: {practice_area}
Article type: {article_type}, {target_words} words, audience: {audience}
Research as of: {as_of_date}

Produce a research plan:
- Restate the topic precisely as a legal question.
- Identify 3-6 legal issues the article must resolve. These are the coverage units the pipeline will check against, so make them discrete and checkable.
- Name every statute, regulation, or instrument you expect to be central, with the specific provisions if you know them.
- Write 8-12 search queries. Each query must state its intent, the source types it targets, and why it is needed. Queries must be phrased as a researcher would type them, not as questions to an assistant.
- Note any directions explicitly out of scope.

The record this plan builds must be able to support an argument, not just a summary. Cover, across your queries:
- The statutory basics: the bare provision, its stated purpose (Statement of Objects and Reasons, preamble, or regulator's rationale where one exists).
- Scholarly interpretation: academic and professional commentary on the provision's ambit, not just its existence.
- Judicial interpretation: how courts and tribunals have actually read the provision, at every level that has ruled on it.
- The strongest version of the orthodox or popular position — in its own words, from someone who holds it — even if you expect the eventual article to argue against it. An article cannot fairly refuse an argument it never let speak.
- The uncomfortable fact: search for what sits awkwardly with the conventional account — a historical origin, an unintended effect, a case the popular narrative doesn't mention.
- Currency and reception: is the instrument stayed, amended, under challenge, awaiting rules — and what press or professional commentary reports about how it is actually landing in practice.

Rules:
- At least one query per legal issue.
- At least one query targeting the primary statutory text.
- At least one query seeking the strongest statement of the prevailing or popular view, not merely "criticism" in general.
- At least one query aimed at an uncomfortable fact, an unintended consequence, or a gap between the rule's stated purpose and its operation.
- At least one query windowed to the last 18 months for recent developments and current status.
- Do not invent case names, citation numbers, or section numbers. If you are not certain a case exists, describe what you are looking for instead of naming it. Fabricated case names in a query poison the entire run."""

_SYSTEM_REPAIR = """You are a research director closing gaps in legal research for a scholarly, argumentative article. Produce only queries that address named gaps. Queries duplicating previous searches will be mechanically discarded.

Close gaps in this order of priority, regardless of the order they were reported in: first, anything about whether the law is currently live — an unresolved question of stay, amendment, challenge, or deferral outranks every other gap, because an article that analyses a stayed provision as settled law is wrong in a way no amount of other depth fixes. Second, any legal issue with thin or zero coverage. Third, missing depth on the prevailing/popular view or the uncomfortable fact the article needs for its concession and refusal — an article that only has evidence for its own conclusion cannot fairly state what it is arguing against.

You must still emit a legal_issues list (the schema requires it), but only ADD to the coverage table's existing issues if a gap genuinely names a new one not already listed — do not rename, merge, split, or reorder any issue already in the coverage table. Downstream evidence is indexed against that exact list; changing it silently orphans everything already gathered."""

_USER_REPAIR = """Previous research on this topic left specific gaps. Your only job is to close them, in priority order: currency/status gaps first, then thin-issue-coverage gaps, then missing prevailing-view or uncomfortable-fact evidence, then everything else.

Topic: {topic}
Practice area: {practice_area}

Gaps found:
{gap_report_rendered}

Already searched (do not repeat, do not rephrase trivially):
{executed_queries}

Coverage so far — issue: evidence count
{coverage_table}

Write at most 6 new queries. For each, name the gap_kind it closes. Prefer queries that reach a higher authority tier than what we already have: if an issue rests on commentary, target the judgment or the provision itself — unless the gap is specifically about currency or the prevailing view, in which case press and professional commentary are the right target, not a compromise."""


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _normalize_query(text: str) -> str:
    """Normalize query text for dedup: lowercase, collapse whitespace, strip punctuation."""
    text = unicodedata.normalize("NFKC", text.lower().strip())
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _make_query_id(query_text: str, intent: str) -> str:
    """Deterministic query_id = sha1(normalized query + intent)[:16]."""
    key = f"{_normalize_query(query_text)}:{intent}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _build_template_plan(request) -> LLMResearchPlan:
    """Deterministic fallback if LLM produces < 4 usable queries."""
    topic = request.topic
    practice_area = request.practice_area or "general"

    issues = [
        f"Primary statutory framework governing {topic}",
        f"Key judicial interpretation of {topic}",
        f"Recent regulatory developments on {topic}",
    ]

    queries = [
        LLMSubQuery(
            query_text=f"{topic} statute India bare act",
            intent=QueryIntent.STATUTE_TEXT,
            rationale="Find the primary statutory text",
            target_source_types=[SourceType.STATUTE],
            priority=1,
        ),
        LLMSubQuery(
            query_text=f"{topic} Supreme Court India judgment",
            intent=QueryIntent.CASE_LAW,
            rationale="Find apex court rulings",
            target_source_types=[SourceType.JUDGMENT],
            priority=1,
        ),
        LLMSubQuery(
            query_text=f"{topic} High Court India recent ruling",
            intent=QueryIntent.CASE_LAW,
            rationale="Find High Court decisions",
            target_source_types=[SourceType.JUDGMENT],
            priority=2,
        ),
        LLMSubQuery(
            query_text=f"{topic} India 2024 2025 amendment notification",
            intent=QueryIntent.RECENT_DEVELOPMENT,
            rationale="Find recent developments",
            target_source_types=[SourceType.GAZETTE_NOTIFICATION, SourceType.LEGAL_NEWS],
            date_from=(date.today() - timedelta(days=540)).isoformat(),
            priority=2,
        ),
        LLMSubQuery(
            query_text=f"{topic} criticism challenges India",
            intent=QueryIntent.COUNTER_VIEW,
            rationale="Find counter-views and criticisms",
            target_source_types=[SourceType.ACADEMIC, SourceType.FIRM_COMMENTARY],
            priority=3,
        ),
    ]

    return LLMResearchPlan(
        topic_restated=f"Legal analysis of {topic} under Indian law",
        legal_issues=issues,
        key_instruments=[],
        sub_queries=queries,
        excluded_directions=[],
    )


# -------------------------------------------------------------------
# Node
# -------------------------------------------------------------------

async def planner_node(state: ResearchState) -> dict[str, Any]:
    """Step 1: Generate a research plan from the topic."""
    ctx_node.set("planner")
    request = state["request"]
    loop_index = state.get("loop_index", 0)
    executed = state.get("executed_query_ids", set())
    gap_report = state.get("gap_report")

    log.info("planner_start", loop_index=loop_index, topic=request.topic)

    llm = get_llm()
    registry = get_registry()

    # Build prompt
    comparative = ", ".join(
        j.value for j in request.article_config.comparative_jurisdictions
    ) or "none"

    if loop_index == 0 or gap_report is None:
        system = _SYSTEM_LOOP0.format(comparative=comparative)
        user = _USER_LOOP0.format(
            topic=request.topic,
            practice_area=request.practice_area or "general",
            article_type=request.article_config.article_type,
            target_words=request.article_config.target_words,
            audience=request.article_config.audience,
            as_of_date=request.as_of_date.isoformat(),
        )
    else:
        # Repair loop
        gap_rendered = "\n".join(
            f"- [{g.gap_kind}] {g.detail} (severity: {g.severity})"
            for g in gap_report.gaps
        )
        exec_rendered = "\n".join(f"- {qid}" for qid in sorted(executed))
        coverage_rendered = "\n".join(
            f"- {issue}: {count} evidence items"
            for issue, count in gap_report.coverage.items()
        )
        system = _SYSTEM_REPAIR
        user = _USER_REPAIR.format(
            topic=request.topic,
            practice_area=request.practice_area or "general",
            gap_report_rendered=gap_rendered,
            executed_queries=exec_rendered,
            coverage_table=coverage_rendered,
        )

    # Call LLM with forced tool use. Retry once on a schema/validation
    # error before falling back to the template — these are usually a
    # one-off slip (e.g. the model puts a SourceType value where an
    # intent belongs) that a fresh call self-corrects, and giving up
    # immediately wastes the whole call for nothing.
    llm_plan = None
    usage: dict[str, Any] = {}
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            llm_plan, usage = await llm.generate(
                system=system,
                user=user,
                output_schema=LLMResearchPlan,
                tool_name="emit_research_plan",
                max_tokens=4096,
                temperature=0.0,
            )
            break
        except Exception as e:
            last_error = e
            log.warning(
                "planner_llm_attempt_failed",
                loop_index=loop_index,
                attempt=attempt + 1,
                error=str(e),
                error_type=type(e).__name__,
            )

    if llm_plan is None:
        log.warning(
            "planner_llm_failed_using_template",
            loop_index=loop_index,
            error=str(last_error),
            error_type=type(last_error).__name__ if last_error else None,
        )
        llm_plan = _build_template_plan(request)
        usage = {"input_tokens": 0, "output_tokens": 0, "model": "template", "latency_ms": 0}

    # -------------------------------------------------------------------
    # Deterministic post-processing
    # -------------------------------------------------------------------

    sub_queries: list[SubQuery] = []
    seen_ids: set[str] = set()

    for llm_sq in llm_plan.sub_queries:
        # Compute query_id deterministically
        qid = _make_query_id(llm_sq.query_text, llm_sq.intent)

        # Dedup within this plan and across previous loops
        if qid in seen_ids or qid in executed:
            log.debug("planner_dedup_query", query_id=qid, query=llm_sq.query_text)
            continue
        seen_ids.add(qid)

        # RELATED_PRECEDENT only on repair loops
        intent = llm_sq.intent
        if intent == QueryIntent.RELATED_PRECEDENT and loop_index == 0:
            intent = QueryIntent.CASE_LAW

        # Provider routing is code, not LLM
        providers = registry.resolve_providers(intent)

        sq = SubQuery(
            query_id=qid,
            query_text=llm_sq.query_text,
            intent=intent,
            rationale=llm_sq.rationale,
            target_source_types=llm_sq.target_source_types,
            min_authority_tier=llm_sq.min_authority_tier,
            jurisdiction=llm_sq.jurisdiction,
            date_from=_parse_date(llm_sq.date_from),
            date_to=_parse_date(llm_sq.date_to),
            providers=providers,
            priority=llm_sq.priority,
        )
        sub_queries.append(sq)

    # Inject mandatory baseline queries the model may have omitted
    as_of = request.as_of_date

    # Statute query for each key instrument
    for instrument in llm_plan.key_instruments:
        qid = _make_query_id(f"{instrument} bare act text", QueryIntent.STATUTE_TEXT)
        if qid not in seen_ids and qid not in executed:
            sub_queries.append(
                SubQuery(
                    query_id=qid,
                    query_text=f"{instrument} bare act text India",
                    intent=QueryIntent.STATUTE_TEXT,
                    rationale=f"Mandatory: primary text of {instrument}",
                    target_source_types=[SourceType.STATUTE],
                    jurisdiction=Jurisdiction.IN,
                    providers=registry.resolve_providers(QueryIntent.STATUTE_TEXT),
                    priority=1,
                )
            )
            seen_ids.add(qid)

    # Recent development query
    recent_qid = _make_query_id(
        f"{request.topic} recent development 18 months", QueryIntent.RECENT_DEVELOPMENT
    )
    has_recent = any(sq.intent == QueryIntent.RECENT_DEVELOPMENT for sq in sub_queries)
    if not has_recent and recent_qid not in executed:
        sub_queries.append(
            SubQuery(
                query_id=recent_qid,
                query_text=f"{request.topic} India recent amendment notification 2025 2026",
                intent=QueryIntent.RECENT_DEVELOPMENT,
                rationale="Mandatory: recent developments in the last 18 months",
                target_source_types=[SourceType.GAZETTE_NOTIFICATION, SourceType.LEGAL_NEWS],
                jurisdiction=Jurisdiction.IN,
                date_from=as_of - timedelta(days=540),
                providers=registry.resolve_providers(QueryIntent.RECENT_DEVELOPMENT),
                priority=2,
            )
        )

    # Clamp query count
    max_queries = 12 if loop_index == 0 else 6
    sub_queries = sorted(sub_queries, key=lambda q: q.priority)[:max_queries]

    # Fallback if too few queries survived
    if len(sub_queries) < 4 and loop_index == 0:
        log.warning(
            "planner_insufficient_queries_using_template",
            count=len(sub_queries),
        )
        template = _build_template_plan(request)
        for llm_sq in template.sub_queries:
            qid = _make_query_id(llm_sq.query_text, llm_sq.intent)
            if qid not in seen_ids and qid not in executed:
                providers = registry.resolve_providers(llm_sq.intent)
                sub_queries.append(
                    SubQuery(
                        query_id=qid,
                        query_text=llm_sq.query_text,
                        intent=llm_sq.intent,
                        rationale=llm_sq.rationale,
                        target_source_types=llm_sq.target_source_types,
                        jurisdiction=Jurisdiction.IN,
                        date_from=_parse_date(llm_sq.date_from),
                        date_to=_parse_date(llm_sq.date_to),
                        providers=providers,
                        priority=llm_sq.priority,
                    )
                )
                seen_ids.add(qid)

        sub_queries = sub_queries[:max_queries]

    # Build plan
    plan_id = hashlib.sha1(
        f"{state['run_id']}:{loop_index}".encode()
    ).hexdigest()[:16]

    # Evidence.supports_issues carries 1-based indices into whatever
    # legal_issues list was in effect when that evidence was extracted
    # (see extractor.py / domain/coverage.py). A repair-loop plan that
    # reorders or rewords issues silently breaks that indexing for every
    # evidence item gathered in earlier loops — gap_check would then be
    # counting old evidence against the wrong issue text. Preserve the
    # previous plan's issue list verbatim, in the same order, and only
    # append genuinely new issues the repair call proposed; never let a
    # repair loop replace the list wholesale.
    legal_issues = llm_plan.legal_issues
    if loop_index > 0 and gap_report is not None:
        prev_plan = state.get("plan")
        if prev_plan is not None and prev_plan.legal_issues:
            # Fuzzy, not exact, match: a repair call reliably rewords an
            # existing issue ("Issue Beta" -> "Issue Beta (reworded)")
            # rather than repeating it verbatim, so exact-text dedup
            # misses most real duplicates. token_set_ratio is
            # order/repetition-insensitive, which is what we want here.
            new_issues = [
                i
                for i in llm_plan.legal_issues
                if not any(
                    fuzz.token_set_ratio(i, existing) >= _ISSUE_DEDUP_THRESHOLD
                    for existing in prev_plan.legal_issues
                )
            ]
            legal_issues = list(prev_plan.legal_issues) + new_issues
            if new_issues:
                log.info(
                    "planner_repair_issues_appended",
                    new_issue_count=len(new_issues),
                    new_issues=new_issues,
                )

    plan = ResearchPlan(
        plan_id=plan_id,
        run_id=state["run_id"],
        loop_index=loop_index,
        topic_restated=llm_plan.topic_restated,
        legal_issues=legal_issues,
        key_instruments=llm_plan.key_instruments,
        sub_queries=sub_queries,
        excluded_directions=llm_plan.excluded_directions,
    )

    log.info(
        "planner_done",
        plan_id=plan_id,
        loop_index=loop_index,
        query_count=len(sub_queries),
        issue_count=len(plan.legal_issues),
        instrument_count=len(plan.key_instruments),
    )

    return {
        "plan": plan,
        "plan_history": [plan],
    }