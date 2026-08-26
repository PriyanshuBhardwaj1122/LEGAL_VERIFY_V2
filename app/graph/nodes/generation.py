"""STEP 6/7/8/9 — Generation phase: outline, per-section drafting,
citation resolution, and mechanical draft verification.

Same anti-hallucination posture as the Research phase: the drafting LLM
never emits citation text itself. It writes prose with inline markers
`[[ev:<evidence_id>]]` pointing at evidence it was actually handed for
that section; a marker resolving to evidence outside that section's
offered set, or to no evidence at all, is a hard failure recorded by
Step 9 — not silently dropped, not "fixed" by substituting something
else. Citation strings are rendered afterward, deterministically, from
each Evidence's structured LegalCitation (app/domain/citation_render.py)
— the model is never trusted to format a citation correctly.

Step 9 also re-checks any text the model puts in quotation marks
against the evidence's own (already Step-4d-grounded) verbatim_quote.
This does not re-touch the source document — that grounding already
happened once, mechanically, in Step 4d — it only catches the drafting
model silently altering a quote it was handed (dropping words, fixing
"errors", paraphrasing but keeping the quote marks).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from app.core.logging import ctx_node, get_logger
from app.domain.citation_render import render_citation
from app.providers.llm.base import get_llm
from app.schemas.article import (
    ArticleDraft,
    ArticleOutline,
    ArticleSection,
    DraftedSection,
    DraftVerificationReport,
    MarkerIssue,
    QuoteIssue,
)
from app.schemas.common import SourceType
from app.schemas.evidence import Evidence
from app.schemas.handoff import EvidencePackage
from app.schemas.request import ArticleConfig
from app.schemas.source import Source

# Source types that can never establish what the law IS — they may be
# cited for currency, the prevailing view, or reception, never for a
# rule. The drafting prompt depends on the evidence lines it's handed
# actually flagging these; see _evidence_line().
_PRESS_SOURCE_TYPES = {SourceType.LEGAL_NEWS, SourceType.FIRM_COMMENTARY}

log = get_logger()

MARKER_RE = re.compile(r"\[\[ev:([a-zA-Z0-9_-]+)\]\]")
QUOTED_BEFORE_MARKER_RE = re.compile(
    r"[“\"]([^”\"]{10,800})[”\"]\s*\[\[ev:([a-zA-Z0-9_-]+)\]\]"
)
QUOTE_MATCH_THRESHOLD = 92  # mirrors grounding.py's FUZZY_THRESHOLD

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def _normalize_quote_text(s: str) -> str:
    """Casefold, unify dash/quote variants, strip punctuation entirely,
    and collapse whitespace. Aggressive on purpose: this is only used to
    decide whether the drafter quoted a genuine (possibly partial)
    excerpt of the evidence's own verbatim_quote — not to relocate text
    in a raw document, where grounding.py's tighter normalization is
    used instead. A drafter is allowed to quote a clean substring (e.g.
    a circular's title minus a leading label) without being flagged;
    it is not allowed to reword, add, or drop substantive content."""
    s = s.casefold()
    s = s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", " ").replace("—", " ").replace("-", " ")
    s = _PUNCT_RE.sub("", s)
    return _SPACE_RE.sub(" ", s).strip()


# ---------------------------------------------------------------------
# Step 6 — outline (one LLM call)
# ---------------------------------------------------------------------

class LLMSection(BaseModel):
    title: str
    target_words: int = Field(ge=50, le=4000)
    issue_refs: list[str] = []


class LLMOutline(BaseModel):
    title: str
    sections: list[LLMSection] = Field(max_length=20)


_OUTLINE_SYSTEM = """You design the section structure for a scholarly legal article — a footnoted paper of the kind submitted to a law review, not a client explainer. You do not write prose here — only titles, target word counts, and which of the listed legal issues each section addresses.

Rules:
1. Every legal issue listed must be assigned to at least one section's issue_refs. Do not invent issues not in the list; copy the issue text exactly as given.
2. Section word targets should sum to roughly the article's target word count (some variance is fine).
3. Prefer one section per major legal issue unless issues are closely related enough to combine; do not create more than 8 sections for a short article or fewer than 3 for a long one.
4. issue_refs entries must be copied verbatim from the numbered list — do not paraphrase them.

THE OPENING

The first section is an introduction and must be planned as one. It does five things in order and nothing else:
  - Names the instrument and the fact that makes it live — enacted, notified, challenged, stayed — in flat sentences, with no throat-clearing and no scene-setting.
  - Frames the concept by contrast where the topic admits it: two to four competing models or jurisdictional approaches, a clause each, then the Indian position. This establishes command of the field faster than any amount of literature review.
  - States the uncomfortable fact — the historical, doctrinal or empirical thing that sits awkwardly with what everyone says about the topic.
  - States the thesis as concession plus conviction plus refusal: what the article grants, what it nonetheless concludes and on how narrow a ground, and which argument on its own side it rejects.
  - Maps the parts BY CONCLUSION, not by topic. Not "Part II examines the classification"; "Part II finds the classification arbitrary for four reasons and explains why curing them changes nothing."

There is no suspense in scholarship. Give away the ending.

ARCHITECTURE

The structure must be argumentative, not merely descriptive. A sequence of sections that each explain one issue is a summary, and summaries are not published. Build the sections so that the article moves through three positions:

  CONCEDE — an early Part establishing the strongest version of the position the article will ultimately rule against, or the historical or doctrinal fact that makes the criticised rule look defensible. This Part costs the article something. It is what buys the right to the Parts that follow. It should close by naming the conditions that made the criticised position defensible then, so that the article can later show those conditions have lapsed.

  CONVICT — the Parts carrying the legal issues, reaching a conclusion on a narrow and precisely specified ground rather than a broad one. Where an issue involves a classification, an exclusion or a threshold, its section should be structured to test the classification rather than to denounce it.

  REFUSE — a late Part showing that the argument commonly made in FAVOUR of the article's own conclusion is fallacious, incomplete, or aimed at the wrong target. This is the Part that distinguishes scholarship from advocacy, and it requires the prevailing view to be present in the evidence in its holders' own words.

Two further sections earn their place in any article long enough to hold them:

  THE COUNTERFACTUAL — a section that assumes every defect the article has identified is cured tomorrow, and asks whether the harm survives the cure. If the harm survives, that is the article's real thesis and the section belongs near the end, before the conclusion. If it does not survive, the article has a different thesis and the outline should be built around that instead.

  THE CASE AGAINST — a section stating the strongest objection to the whole argument, drawn from the counter-evidence in the record rather than invented, and conceding what it proves before showing what it does not prove.

Scaling. For an article under about 1,800 words, fold REFUSE, THE COUNTERFACTUAL and THE CASE AGAINST into a single closing analytical section before the conclusion. Between roughly 1,800 and 3,500 words, give THE CASE AGAINST its own section and fold the other two. Above roughly 3,500 words, give all three separate sections and organise the whole as numbered Parts with numbered subsections beneath them. Never drop all of them; an article with none of them has no argument, only a position.

The final section must be a conclusion that restates the popular argument fairly, states the narrower conclusion actually reached, and lists specific reforms or unresolved questions. It must not issue a call to action. The energy of a scholarly article is spent in proving; the ending is administrative.

CURRENCY

If the evidence indicates that the central instrument is stayed, under challenge, amended, awaiting rules, or otherwise unsettled, that fact belongs in the introduction and, where it bears on a specific issue, in that issue's section — not in a closing update paragraph. A reader who reaches Part IV before learning the provision is stayed has been misled by the structure.

TITLES

Section titles are flat, declarative and specific. They name what the section establishes, not what it discusses. "The permit that was hardest to obtain" is a title; "Analysis of the permit regime" is a label. Wordplay in a section title is almost always a mistake — the wit belongs in the prose, where it is earned, not in the furniture."""

_OUTLINE_USER_TEMPLATE = """Topic: {topic}
Article type: {article_type} | target words: {target_words} | audience: {audience}

Legal issues to cover (copy verbatim into issue_refs):
{issues}

Before choosing titles, settle privately on the shape of the argument: what the article concedes, what it convicts on and how narrowly, and which popular argument on its own side it refuses. Do not output that reasoning — let it determine the sections, their order and their word targets. A concession section that runs 80 words is not a concession; it is a formality. Give it real weight.

Test the outline before you emit it: could a reader who knows this practice area predict the conclusion from the section titles alone? If so, the article is a summary and the structure needs rebuilding around whatever in the record was genuinely unexpected."""


async def outline_node(package: EvidencePackage, article_config: ArticleConfig) -> dict[str, Any]:
    """STEP 6: produce the section plan for the article."""
    ctx_node.set("outline")

    issues_rendered = "\n".join(f"  - {issue}" for issue in package.legal_issues)
    user = _OUTLINE_USER_TEMPLATE.format(
        topic=package.topic,
        article_type=article_config.article_type,
        target_words=article_config.target_words,
        audience=article_config.audience,
        issues=issues_rendered,
    )

    llm = get_llm()
    result, usage = await llm.generate(
        system=_OUTLINE_SYSTEM,
        user=user,
        output_schema=LLMOutline,
        tool_name="emit_outline",
        max_tokens=2048,
        temperature=0.0,
    )

    valid_issues = set(package.legal_issues)
    sections: list[ArticleSection] = []
    covered: set[str] = set()
    for i, s in enumerate(result.sections):
        refs = [r for r in s.issue_refs if r in valid_issues]
        dropped = [r for r in s.issue_refs if r not in valid_issues]
        if dropped:
            log.warning("outline_issue_ref_not_in_list", section_title=s.title, dropped=dropped)
        covered |= set(refs)
        sections.append(
            ArticleSection(
                section_id=f"sec-{i}",
                title=s.title,
                target_words=s.target_words,
                issue_refs=refs,
            )
        )

    uncovered = [i for i in package.legal_issues if i not in covered]
    if uncovered:
        # Don't silently lose an issue — attach it to the last section
        # rather than leaving it with zero assigned coverage.
        log.warning("outline_issues_uncovered_by_llm", uncovered=uncovered)
        if sections:
            sections[-1].issue_refs = list(set(sections[-1].issue_refs) | set(uncovered))

    outline = ArticleOutline(run_id=package.run_id, title=result.title, sections=sections)
    log.info("outline_done", section_count=len(sections), title=outline.title)
    return {"outline": outline}


# ---------------------------------------------------------------------
# Step 7 — per-section drafting (one LLM call per section)
# ---------------------------------------------------------------------

class LLMSectionDraft(BaseModel):
    body: str = Field(max_length=8000)


_DRAFT_SYSTEM = """You draft one section of a scholarly legal article from a fixed set of evidence items. You may cite ONLY the evidence items listed below — never state a citation, case name, or section number from memory, even if you recognise it.

To cite an item, write your point and end it with the marker [[ev:<evidence_id>]] using the exact evidence_id given — do not write the citation text yourself; it will be rendered separately from the evidence's own record.

Rules:
1. If you quote an evidence item's exact words, wrap them in double quotes and place the marker immediately after the closing quote: "exact words" [[ev:abc123]]. Only do this if the evidence item has a verbatim_quote — do not put quote marks around a paraphrase.
2. Every substantive legal claim must carry at least one marker. A sentence with no marker should be transition/framing text only, not a legal proposition.
3. Never invent an evidence_id. Only use IDs from the list below.
4. Do not cite evidence outside the list below, even if it seems relevant — it was not selected for this section.
5. Write plain prose (markdown), no headers (the section title is added separately), target roughly {target_words} words.

WHAT IS AND IS NOT A CLAIM

Rules 2 and 4 govern propositions about the law and the world. They do not govern reasoning. An analogy, a hypothetical, a worked comparison, a restatement of an opponent's argument, or an inference drawn openly from evidence already cited is a construction, not a claim, and needs no marker — provided it introduces no new fact, case, provision or figure. If a device you want to use requires a fact you do not have in the list below, cut the device. Never manufacture the fact to save the sentence.

USING PRESS EVIDENCE

Items marked PRESS in the evidence list are not authority. Handle them under three rules.

  Attribute in text. Name the outlet and the date in the sentence: "the Economic and Political Weekly argued in December 2019 that ...". A press item cited like a judgment reads as an error even when the marker is correct.

  Never let press carry a rule. If an item reports what a court held or what a circular provides, and the judgment or circular is also in the list, cite the primary item for the proposition and the press item only for reception, timing or reaction. If the primary item is not in the list, do not state the rule at all — describe what was reported and attribute it.

  Press is the right source for currency and for the prevailing view, and should be used confidently for both. Where an item establishes that the instrument was stayed, notified, deferred or challenged, say so plainly and early; that fact conditions everything the section argues. Where an item states the position the article will concede to or refuse, quote it if a verbatim_quote exists — the opposing argument is always stronger in its own words than in your summary of it, and quoting it is what makes the refusal fair.

THE ANALYTICAL ARC FOR ISSUE SECTIONS

If this section's title names a legal issue rather than a structural role (introduction, concession, case against, conclusion), move through it in this order — as continuous prose, in the register described below, never as labelled sub-headers:

  1. Statutory rule. State the bare rule from the provision itself, drawn from statutory_text or definition evidence.
  2. Judicial interpretation. State how a court has actually read that rule, drawn from holding or obiter evidence — not what the statute could mean in the abstract, what a court said it means.
  3. Rationale. Say why the court or the instrument's own drafters reached that reading — drawn from policy_rationale evidence or the reasoning given in the holding/obiter itself, not from your own inference about why it might make sense.
  4. Competing interpretation. Give the strongest alternative reading in the record — a dissent, a different court, or commentary arguing the rule should be read otherwise. This is a real position from the evidence, not a hypothetical you construct to knock down.
  5. Factual distinction. Where the record shows the rule has landed differently depending on the facts — a tribunal split, a carve-out, a case distinguished on its facts — state what specifically distinguishes them. Where the evidence contains no such split, skip this beat rather than inventing one.
  6. Current position. State what the law appears to be now, on the evidence available — and if a press or currency item shows the position is unsettled, stayed, or under fresh challenge, that qualification belongs here.
  7. Critique. State what is unsatisfying, inconsistent, or unresolved about that current position, drawn from commentary_opinion evidence or from a tension the evidence itself exposes (e.g. a stated rationale the current position doesn't actually serve).
  8. Author's synthesis. Land your own conclusion on this narrow issue — this is where the section's real argumentative weight sits. It may draw only on points already established in beats 1-7; it needs no new marker of its own if it introduces no new fact, but it must not simply restate beat 6.

Not every issue's record will support all eight beats — evidence rarely arrives that complete. Skip a beat cleanly rather than manufacturing content for it, and never announce the skip ("there is no competing interpretation" is filler; just omit the beat). The arc is a discipline for ordering the section's thinking, not a checklist to visibly tick off — a reader should feel the section move from rule to interpretation to tension to conclusion, without seeing the joins.

ARGUMENTATIVE MOVES

The concession stack. Before reaching a strong conclusion, concede what is genuinely true against it, in sequence, and then turn: "True, X. True, Y. It is also true that Z. However, ..." Each concession must be a real point supported by evidence in the list, not a token. The rebuttal wins because the concessions were honest.

Steelman long, rebut short. Build the opposing argument at greater length than you dismantle it — roughly three sentences of construction for one of rebuttal. Length signals fairness; a short rebuttal signals that the flaw is obvious once seen. Never attack a version of the argument its holders would not recognise.

Isolated variable. To test a classification, hold everything constant and vary one term. Two parties identical in every respect but one, treated differently by the rule; then ask what the difference is doing. This is the most effective device available against an arbitrary distinction and it is unanswerable in a way that indignation is not. It requires no marker, because it invents no facts.

Self-defeating purpose. Do not argue that a provision is unfair. Argue, where the evidence supports it, that the provision defeats the object its own Statement of Objects and Reasons, preamble, or regulator's stated rationale declares. The state's own words become the standard it fails.

Reconstruct before deploying. When you rely on a judgment, set out what the case was actually about — the facts, what the losing side argued, why the court answered as it did — before you use the proposition it stands for. A phrase restored to its context often stops being usable by the other side, and you never have to accuse anyone of quoting it selectively.

Pair every figure. If the list contains a figure and its comparator, baseline or projection, use both in the same sentence or the next. A figure standing alone is decoration.

Calibrated hedging. Write "a compelling argument can be made", "on balance", "the better view", "may not have been accurate" wherever that is what the evidence supports. Hedge honestly and often, so that the sentences you do not hedge carry weight when they arrive.

REGISTER, IRONY AND ANALOGY

Default to flat prose. Subject, verb, object. Short Anglo-Saxon words where they exist. The evidence does the work; the prose is a clean pane of glass in front of it. Long sentence carrying the substance, short sentence landing it. Third person throughout; "this paper argues" and "this Part explains" are permitted, "we" and "our" are not.

Irony is dry, not loud. It is produced by understatement, by juxtaposition, by describing a thing precisely enough that its absurdity becomes visible without comment — never by sneering. "Tellingly, the exemption applied to nobody" is wit. "Astonishingly, the regulator seems to have forgotten how arithmetic works" is snark, and snark forfeits the authority the concession stack has just bought. Use at most one openly ironic construction per section, and never in the same paragraph as a quotation.

Let the evidence convict. Where a verbatim_quote is damaging, quote it, place the marker, and stop. Do not add "remarkably", "shockingly", "astonishingly", or any adjective characterising the quotation. The reader supplies the judgment, which is why the reader accepts it. An editorialising adverb next to a damning quote weakens both.

Analogies must be checkable and disposable. Draw them from ordinary life or from a different area of law, make them do one piece of work, and abandon them. Do not extend a metaphor across paragraphs, do not build a section around one, and never let an analogy carry a legal proposition — it may illuminate a proposition the evidence has already established, never replace it.

Diction budget. Allow yourself roughly one vivid or striking phrase per 400 words of this section, and no more. Scarcity is the entire mechanism: three arresting phrases in a paragraph cancel each other out and none lands. Everything else stays plain.

FORBIDDEN

- Exclamation marks; rhetorical questions used as filler rather than as an argumentative hinge; second person; "Let's"; "In today's fast-paced world"; "It is important to note that"; "In conclusion"
- Editorialising adjectives or adverbs adjacent to a quotation
- Jokes that carry a legal proposition, puns on party names, mockery of any identifiable person
- Any case name, provision, date, quotation or figure not present in the evidence list below
- Stating a legal rule on the authority of a press item, or letting "reportedly" do load-bearing work in a doctrinal claim
- Analysing a provision as though settled where the evidence shows it stayed, amended or under challenge
- Padding a section to its word target with restatement. If the evidence supports 300 words and the target is 500, write 300."""

_DRAFT_USER_TEMPLATE = """Section: {title}
Legal issues this section addresses: {issues}

Evidence available for this section (id | kind | binding_strength | AUTHORITY/PRESS | outlet | date | statement | verbatim_quote):
{evidence_lines}

Read the section title as an instruction about posture. If it is the introduction, name the instrument and what has happened to it, frame the concept by contrast, state the uncomfortable fact, state the thesis as concession plus conviction plus refusal, and map the Parts by their conclusions rather than their topics. If it addresses a legal issue — the CONVICT parts — work the ANALYTICAL ARC FOR ISSUE SECTIONS above: statutory rule, judicial interpretation, rationale, competing interpretation, factual distinction, current position, critique, author's synthesis, in that order, skipping any beat the evidence doesn't support. If it concedes ground, concede it fully and without hedging the concession itself — a grudging concession is worse than none. If it states the case against the article, state that case as its own holders would state it, using their own words where a verbatim_quote exists, and concede what it proves before you show what it does not prove. If it is the conclusion, restate the popular argument fairly, state the narrower conclusion actually reached, list the specific reforms or open questions, and stop — no call to action, no rising cadence.

Before drafting, scan the evidence for anything bearing on the present status of the instrument — stay, amendment, challenge, deferral, supersession, rules not yet made. If such an item exists and this section analyses that instrument, state its status before you analyse it, not after."""


def _domain_from_url(url: str) -> str:
    """Best-effort outlet name when Source.issuing_body isn't populated
    (true for most press-tier domains in app/domain/authority.py's
    tier table — livelaw.in, barandbench.com, etc. all map to
    issuing_body=None today). A bare domain is a truthful, if inelegant,
    attribution — better than the drafter having nothing to attribute
    a press item to at all."""
    try:
        from urllib.parse import urlparse

        host = urlparse(url).hostname or ""
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return "(unknown outlet)"


def _evidence_line(e: Evidence, source: Source | None) -> str:
    is_press = bool(source and source.source_type in _PRESS_SOURCE_TYPES)
    tier_label = "PRESS" if is_press else "AUTHORITY"
    outlet = "(unknown outlet)"
    if source is not None:
        outlet = source.issuing_body or (
            _domain_from_url(source.url_canonical) if source.url_canonical else "(unknown outlet)"
        )
    date_str = "(undated)"
    if source is not None and source.decided_or_published_on is not None:
        date_str = source.decided_or_published_on.isoformat()

    quote = e.verbatim_quote if e.verbatim_quote else "(no quote)"
    return (
        f"{e.evidence_id} | {e.kind} | {e.binding_strength} | {tier_label} | "
        f"{outlet} | {date_str} | {e.statement} | {quote}"
    )


async def _draft_section(
    section: ArticleSection,
    evidence_for_section: list[Evidence],
    sources_by_id: dict[str, Source],
) -> DraftedSection:
    evidence_lines = "\n".join(
        _evidence_line(e, sources_by_id.get(e.source_id)) for e in evidence_for_section
    )
    if not evidence_lines:
        evidence_lines = "(no evidence available — write only framing text, no legal claims)"

    system = _DRAFT_SYSTEM.format(target_words=section.target_words)
    user = _DRAFT_USER_TEMPLATE.format(
        title=section.title,
        issues="; ".join(section.issue_refs) or "(general)",
        evidence_lines=evidence_lines,
    )

    llm = get_llm()
    try:
        result, usage = await llm.generate(
            system=system,
            user=user,
            output_schema=LLMSectionDraft,
            tool_name="emit_section_draft",
            max_tokens=4096,
            # Higher than the rest of the pipeline's calls (which stay
            # near 0.0 for structural/factual determinism) — voice,
            # irony and register variation need room the rest of the
            # pipeline deliberately doesn't give the model. Citation
            # correctness is enforced mechanically by Step 9 regardless
            # of temperature, so this doesn't trade away grounding.
            temperature=0.65,
        )
        body = result.body
    except Exception as e:
        log.warning("draft_section_llm_failed", section_id=section.section_id, error=str(e))
        body = ""

    log.info("draft_section_done", section_id=section.section_id, body_chars=len(body))
    return DraftedSection(
        section_id=section.section_id,
        title=section.title,
        body=body,
        evidence_ids_offered=[e.evidence_id for e in evidence_for_section],
    )


async def draft_node(outline: ArticleOutline, package: EvidencePackage) -> dict[str, Any]:
    """STEP 7: draft every section. Each section only ever sees the
    evidence bound to its own issue_refs — this is what makes an
    'out of scope' marker in Step 9 meaningful rather than vacuous."""
    ctx_node.set("draft")

    sources_by_id: dict[str, Source] = {s.source_id: s for s in package.sources}
    evidence_by_issue: dict[str, list[Evidence]] = {issue: [] for issue in package.legal_issues}
    for e in package.evidence:
        for ref in e.supports_issues:
            # supports_issues holds 1-based index strings into
            # package.legal_issues, same convention as coverage.py.
            try:
                idx = int(ref) - 1
            except ValueError:
                continue
            if 0 <= idx < len(package.legal_issues):
                evidence_by_issue[package.legal_issues[idx]].append(e)

    sections: list[DraftedSection] = []
    for section in outline.sections:
        seen: set[str] = set()
        evidence_for_section: list[Evidence] = []
        for issue in section.issue_refs:
            for e in evidence_by_issue.get(issue, []):
                if e.evidence_id not in seen:
                    seen.add(e.evidence_id)
                    evidence_for_section.append(e)
        drafted = await _draft_section(section, evidence_for_section, sources_by_id)
        sections.append(drafted)

    log.info("draft_node_done", section_count=len(sections))
    return {"drafted_sections": sections}


# ---------------------------------------------------------------------
# Step 8 — citation resolution / assembly (deterministic)
# ---------------------------------------------------------------------

def assemble_article(
    outline: ArticleOutline,
    sections: list[DraftedSection],
    evidence_by_id: dict[str, Evidence],
) -> str:
    parts = [f"# {outline.title}\n"]
    for s in sections:
        parts.append(f"## {s.title}\n")
        body = s.body

        def _replace(m: re.Match) -> str:
            ev_id = m.group(1)
            ev = evidence_by_id.get(ev_id)
            if ev is None:
                return " [citation unresolved]"
            cite = render_citation(ev.citation) or ev.source_id
            pin = f", {ev.pinpoint}" if ev.pinpoint else ""
            return f" ({cite}{pin})"

        parts.append(MARKER_RE.sub(_replace, body) + "\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------
# Step 9 — mechanical draft verification (no LLM)
# ---------------------------------------------------------------------

def verify_draft(
    outline: ArticleOutline,
    sections: list[DraftedSection],
    evidence_by_id: dict[str, Evidence],
    legal_issues: list[str],
) -> DraftVerificationReport:
    marker_issues: list[MarkerIssue] = []
    quote_issues: list[QuoteIssue] = []
    all_marker_ids: list[str] = []
    covered_issues: set[str] = set()

    section_by_id = {s.section_id: s for s in outline.sections}

    for s in sections:
        offered = set(s.evidence_ids_offered)
        outline_sec = section_by_id.get(s.section_id)
        section_issues = set(outline_sec.issue_refs) if outline_sec else set()

        markers_in_body = MARKER_RE.findall(s.body)
        for ev_id in markers_in_body:
            all_marker_ids.append(ev_id)
            if ev_id not in evidence_by_id:
                marker_issues.append(
                    MarkerIssue(
                        section_id=s.section_id,
                        marker_evidence_id=ev_id,
                        kind="unresolved",
                        detail="marker does not resolve to any known evidence_id",
                    )
                )
                continue
            if ev_id not in offered:
                marker_issues.append(
                    MarkerIssue(
                        section_id=s.section_id,
                        marker_evidence_id=ev_id,
                        kind="out_of_scope",
                        detail="evidence was not offered to this section's drafting call",
                    )
                )
                continue
            # Valid, in-scope citation — credit its issues as covered.
            ev_issue_texts = {
                legal_issues[int(r) - 1]
                for r in evidence_by_id[ev_id].supports_issues
                if r.isdigit() and 0 < int(r) <= len(legal_issues)
            }
            covered_issues |= section_issues & ev_issue_texts

        for quoted_text, ev_id in QUOTED_BEFORE_MARKER_RE.findall(s.body):
            ev = evidence_by_id.get(ev_id)
            if ev is None:
                continue  # already recorded as unresolved above
            if not ev.verbatim_quote:
                quote_issues.append(
                    QuoteIssue(
                        section_id=s.section_id,
                        evidence_id=ev_id,
                        quoted_text=quoted_text[:200],
                        detail="draft quotes this evidence but it has no verbatim_quote on record",
                    )
                )
                continue
            norm_quoted = _normalize_quote_text(quoted_text)
            norm_verbatim = _normalize_quote_text(ev.verbatim_quote)

            # A clean substring match (drafter quoted a genuine excerpt,
            # e.g. dropped a leading "Circular- " label or trailing
            # punctuation) is not a divergence — only score it if it
            # ISN'T a substring either way.
            if norm_quoted and (norm_quoted in norm_verbatim or norm_verbatim in norm_quoted):
                continue

            score = fuzz.partial_ratio(norm_quoted, norm_verbatim)
            if score < QUOTE_MATCH_THRESHOLD:
                quote_issues.append(
                    QuoteIssue(
                        section_id=s.section_id,
                        evidence_id=ev_id,
                        quoted_text=quoted_text[:200],
                        detail=f"quoted text diverges from evidence's verbatim_quote (similarity={score:.0f})",
                    )
                )

    uncovered_issues = [i for i in legal_issues if i not in covered_issues]

    unique_evidence_cited = len({m for m in all_marker_ids if m in evidence_by_id})

    if marker_issues or quote_issues:
        verdict: str = "failed"
        rationale = (
            f"{len(marker_issues)} marker issue(s), {len(quote_issues)} quote issue(s) — "
            "draft cites evidence it wasn't given, or alters a quote it was given."
        )
    elif uncovered_issues:
        verdict = "failed"
        rationale = f"{len(uncovered_issues)} legal issue(s) have zero cited evidence anywhere in the draft."
    else:
        verdict = "passed"
        rationale = (
            f"All {len(all_marker_ids)} citation marker(s) resolve to in-scope evidence, "
            f"all quoted text matches its source evidence, and every legal issue is cited."
        )

    return DraftVerificationReport(
        run_id=outline.run_id,
        marker_issues=marker_issues,
        quote_issues=quote_issues,
        uncovered_issues=uncovered_issues,
        citation_count=len(all_marker_ids),
        unique_evidence_cited=unique_evidence_cited,
        verdict=verdict,  # type: ignore[arg-type]
        rationale=rationale,
    )


# ---------------------------------------------------------------------
# Node entry point — ties Steps 6-9 together
# ---------------------------------------------------------------------

async def generation_node(package: EvidencePackage, article_config: ArticleConfig) -> dict[str, Any]:
    ctx_node.set("generation")

    outline_result = await outline_node(package, article_config)
    outline: ArticleOutline = outline_result["outline"]

    draft_result = await draft_node(outline, package)
    sections: list[DraftedSection] = draft_result["drafted_sections"]

    evidence_by_id = {e.evidence_id: e for e in package.evidence}
    rendered = assemble_article(outline, sections, evidence_by_id)

    report = verify_draft(outline, sections, evidence_by_id, package.legal_issues)

    draft = ArticleDraft(
        run_id=package.run_id,
        title=outline.title,
        sections=sections,
        rendered_markdown=rendered,
    )

    log.info(
        "generation_node_done",
        verdict=report.verdict,
        citation_count=report.citation_count,
        marker_issue_count=len(report.marker_issues),
        quote_issue_count=len(report.quote_issues),
        uncovered_issue_count=len(report.uncovered_issues),
    )

    return {"outline": outline, "article_draft": draft, "draft_verification": report}