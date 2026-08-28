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
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from app.core.logging import ctx_node, get_logger
from app.domain.citation_render import render_citation
from app.providers.llm.base import get_generation_llm
from app.schemas.article import (
    ArticleDraft,
    ArticleOutline,
    ArticleSection,
    ArticleThesis,
    CalcClaim,
    CalcIssue,
    DraftedSection,
    DraftVerificationReport,
    MarkerIssue,
    QuoteIssue,
    SectionRole,
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
# Catches every "ev:<id>" token regardless of bracketing — including
# malformed multi-citation attempts like "[[ev:id1], [ev:id2]]" that
# don't match MARKER_RE's strict [[ev:<id>]] shape and would otherwise
# leak the raw evidence_id straight into the rendered article,
# unresolved and unflagged. Superset of MARKER_RE's matches.
_EV_TOKEN_RE = re.compile(r"ev:([a-zA-Z0-9_-]+)")
# Any leftover bracket/comma debris around a stray ev: token, once the
# well-formed markers have already been substituted out.
_STRAY_MARKER_DEBRIS_RE = re.compile(r"\[*ev:[a-zA-Z0-9_-]+\]*(?:,\s*\[*ev:[a-zA-Z0-9_-]+\]*)*")

CALC_MARKER_RE = re.compile(r"\[\[calc:([a-zA-Z0-9_-]+)\]\]")
_CALC_TOLERANCE_REL = 0.01  # 1% relative tolerance
_CALC_TOLERANCE_ABS = 0.05  # floor, for near-zero results where relative tolerance is meaningless


def _compute_calc(claim: CalcClaim) -> float | None:
    """Recompute a CalcClaim's result from its inputs via a fixed
    whitelist of methods — never eval, never trust the LLM's own
    arithmetic. Returns None if the method's input shape is wrong
    (caller records that as a CalcIssue, same as any other malformed
    marker)."""
    vals = [i.value for i in claim.inputs]
    try:
        if claim.method == "sum":
            return sum(vals)
        if claim.method == "difference" and len(vals) == 2:
            return vals[0] - vals[1]
        if claim.method == "ratio" and len(vals) == 2 and vals[1] != 0:
            return vals[0] / vals[1]
        if claim.method == "delta_pct" and len(vals) == 2 and vals[0] != 0:
            return (vals[1] - vals[0]) / vals[0] * 100
        if claim.method == "cagr" and len(vals) == 2 and vals[0] > 0 and claim.periods and claim.periods > 0:
            return ((vals[1] / vals[0]) ** (1 / claim.periods) - 1) * 100
    except (ZeroDivisionError, ValueError, OverflowError):
        return None
    return None


def _calc_matches(computed: float, claimed: float) -> bool:
    tolerance = max(_CALC_TOLERANCE_REL * abs(computed), _CALC_TOLERANCE_ABS)
    return abs(computed - claimed) <= tolerance
QUOTE_MATCH_THRESHOLD = 92  # mirrors grounding.py's FUZZY_THRESHOLD

# The fixed calibrated-hedging phrases _DRAFT_SYSTEM instructs every
# section to use. Tracked across sections (see _draft_section's
# hedge_counts_so_far) so a later section can be told a phrase is
# already spent, instead of every independent call reaching for the
# same three phrases.
_HEDGE_PHRASES = [
    "a compelling argument can be made",
    "on balance",
    "the better view",
    "may not have been accurate",
]

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
# Step 5b — thesis (one LLM call, runs before the outline)
# ---------------------------------------------------------------------
#
# Runs before outline_node and is threaded into both the outline and
# every draft call. Without this, each section independently invents
# its own hedge ("may require refinement") because nothing tells it
# what the article as a whole is trying to prove — that's what produced
# the unfalsifiable, symmetric-hedging conclusion this exists to fix.

class LLMThesis(BaseModel):
    # Generous ceilings, not stylistic limits — the prompt asks for one
    # or two sentences; a model that states the claim more precisely at
    # greater length shouldn't fail the call over it.
    thesis: str = Field(min_length=1, max_length=1500)
    falsifier: str = Field(min_length=1, max_length=1500)


_THESIS_SYSTEM = """You state the one specific, falsifiable claim this article will argue — not its topic, not a balanced summary of the area.

A thesis is falsifiable: a specific reader could point to a specific fact and say "no, because X." "Section 29A is necessary but may require refinement" is not a thesis — it is compatible with every possible finding, so nothing could ever falsify it. "The classification in Section 29A(c) is arbitrary because it turns on a fact — a related party's own default — that the applicant cannot control or discover" is a thesis: a single counter-fact (the applicant could in fact discover it) would kill it.

Two rules:
1. thesis: one or two sentences. State the specific conclusion, not the area of law and not a call for "balance" or "refinement." If your thesis could be published about a different statute by swapping the noun, it is not specific enough.
2. falsifier: one or two sentences stating concretely what evidence, if it existed in the record, would defeat this thesis. If you cannot state a falsifier, the thesis is not yet specific enough — narrow it until you can.

Base the thesis on the legal issues and evidence actually available below — not on what would make the best story. If the evidence doesn't support a strong claim, the thesis should be the narrowest claim it does support, stated precisely, not a vague one stated broadly."""

_THESIS_USER_TEMPLATE = """Topic: {topic}

Legal issues in the record:
{issues}

Evidence available (kind | statement):
{evidence_lines}"""


async def thesis_node(package: EvidencePackage) -> dict[str, Any]:
    """STEP 5b: produce the article's specific, falsifiable thesis."""
    ctx_node.set("thesis")

    issues_rendered = "\n".join(f"  - {issue}" for issue in package.legal_issues)
    evidence_lines = "\n".join(f"  - {e.kind} | {e.statement}" for e in package.evidence[:60])
    user = _THESIS_USER_TEMPLATE.format(
        topic=package.topic, issues=issues_rendered, evidence_lines=evidence_lines or "(none)"
    )

    llm = get_generation_llm()
    result, usage = await llm.generate(
        system=_THESIS_SYSTEM,
        user=user,
        output_schema=LLMThesis,
        tool_name="emit_thesis",
        # Budgets across this phase leave headroom for models that think
        # before answering — reasoning tokens share max_tokens with the
        # response, and a budget sized for a non-thinking model truncates
        # the tool call mid-JSON. This is a ceiling, not a target.
        max_tokens=4096,
        temperature=0.0,
    )

    thesis = ArticleThesis(run_id=package.run_id, thesis=result.thesis, falsifier=result.falsifier)
    log.info("thesis_done", thesis=thesis.thesis)
    return {"thesis": thesis}


# ---------------------------------------------------------------------
# Step 6 — outline (one LLM call)
# ---------------------------------------------------------------------

class LLMSection(BaseModel):
    title: str
    target_words: int = Field(ge=50, le=4000)
    issue_refs: list[str] = []
    role: SectionRole
    # Generous ceiling rather than a stylistic limit — the prompt asks
    # for one or two sentences, and a model that writes a longer, more
    # specific claim shouldn't fail the whole outline over it.
    conclusion: str = Field(min_length=1, max_length=1200)
    depends_on: list[str] = []  # exact titles of earlier sections this one builds on


class LLMOutline(BaseModel):
    title: str
    sections: list[LLMSection] = Field(max_length=20)


_OUTLINE_SYSTEM = """You design the section structure for a scholarly legal article — a footnoted paper of the kind submitted to a law review, not a client explainer. You do not write prose here — only titles, target word counts, and which of the listed legal issues each section addresses.

Rules:
1. Every legal issue listed must be assigned to at least one section's issue_refs. Do not invent issues not in the list; copy the issue text exactly as given.
2. Section word targets should sum to roughly the article's target word count (some variance is fine).
3. Prefer one section per major legal issue unless issues are closely related enough to combine; do not create more than 8 sections for a short article or fewer than 3 for a long one.
4. issue_refs entries must be copied verbatim from the numbered list — do not paraphrase them.
5. Every section carries a `role` (one of: intro, concede, convict, refuse, counterfactual, case_against, conclusion), matching the structural roles defined below. The first section is always role=intro and the last is always role=conclusion. Most sections carrying legal issues are role=convict.
6. Every section carries a `conclusion`: one or two sentences stating the specific, falsifiable claim that section must land — not its topic. "Examines the classification under Section 29A" is a topic; "The classification is arbitrary because it turns on a fact the applicant cannot control" is a conclusion.
7. Every section carries `depends_on`: the exact titles (copied verbatim, as you write them) of earlier sections this one builds on and must not re-explain from scratch. Leave empty if the section is self-contained. A section may only depend on sections that come before it.

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

    REFUSE is not a second CASE_AGAINST. CASE_AGAINST attacks the article's thesis; REFUSE attacks a popular argument that supports the article's own thesis but supports it badly. Get the direction right: if the article's conclusion is "Section 29A's breadth is unjustified," a REFUSE section does NOT restate criticisms of Section 29A's breadth — that is CASE_AGAINST territory, or just more CONVICT. REFUSE instead takes a commonly-heard argument for narrowing/striking 29A and shows it's the wrong reason to reach that conclusion (e.g. "critics say 29A is unconstitutional; it isn't, and resting the case on that claim is a mistake that lets the real objection go unmade"). A REFUSE section titled "The Case Against X" or "Criticisms of X" is almost always this error — wrong direction, restating the article's own side instead of correcting a bad argument for it.

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
{thesis_context}
Legal issues to cover (copy verbatim into issue_refs):
{issues}

Before choosing titles, settle privately on the shape of the argument: what the article concedes, what it convicts on and how narrowly, and which popular argument on its own side it refuses. Do not output that reasoning — let it determine the sections, their order and their word targets. A concession section that runs 80 words is not a concession; it is a formality. Give it real weight. Every section's `conclusion` field must be a specific claim that serves the thesis above — never restate the thesis wholesale, and never a topic sentence with no claim in it.

Test the outline before you emit it: could a reader who knows this practice area predict the conclusion from the section titles alone? If so, the article is a summary and the structure needs rebuilding around whatever in the record was genuinely unexpected."""


async def outline_node(
    package: EvidencePackage, article_config: ArticleConfig, thesis: ArticleThesis | None = None
) -> dict[str, Any]:
    """STEP 6: produce the section plan for the article."""
    ctx_node.set("outline")

    issues_rendered = "\n".join(f"  - {issue}" for issue in package.legal_issues)
    thesis_context = (
        f"Article thesis (every section's conclusion must serve this): {thesis.thesis}\n"
        f"What would falsify it: {thesis.falsifier}\n"
        if thesis is not None
        else ""
    )
    user = _OUTLINE_USER_TEMPLATE.format(
        topic=package.topic,
        article_type=article_config.article_type,
        target_words=article_config.target_words,
        audience=article_config.audience,
        thesis_context=thesis_context,
        issues=issues_rendered,
    )

    llm = get_generation_llm()
    result, usage = await llm.generate(
        system=_OUTLINE_SYSTEM,
        user=user,
        output_schema=LLMOutline,
        tool_name="emit_outline",
        max_tokens=8192,
        temperature=0.0,
    )

    valid_issues = set(package.legal_issues)
    sections: list[ArticleSection] = []
    covered: set[str] = set()
    title_to_id: dict[str, str] = {}
    for i, s in enumerate(result.sections):
        refs = [r for r in s.issue_refs if r in valid_issues]
        dropped = [r for r in s.issue_refs if r not in valid_issues]
        if dropped:
            log.warning("outline_issue_ref_not_in_list", section_title=s.title, dropped=dropped)
        covered |= set(refs)

        # depends_on must resolve to an earlier section's exact title —
        # a forward or unrecognised reference is dropped rather than
        # trusted, same posture as issue_refs above.
        deps = [title_to_id[t] for t in s.depends_on if t in title_to_id]
        bad_deps = [t for t in s.depends_on if t not in title_to_id]
        if bad_deps:
            log.warning("outline_depends_on_unresolved", section_title=s.title, dropped=bad_deps)

        section_id = f"sec-{i}"
        title_to_id[s.title] = section_id
        sections.append(
            ArticleSection(
                section_id=section_id,
                title=s.title,
                target_words=s.target_words,
                issue_refs=refs,
                role=s.role,
                conclusion=s.conclusion,
                depends_on=deps,
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
    # ~8000 chars is only ~1300 words; section targets can exceed that,
    # and overshooting the cap fails the call rather than trimming.
    body: str = Field(max_length=24000)
    calcs: list[CalcClaim] = Field(default_factory=list, max_length=6)


_DRAFT_SYSTEM = """You draft one section of a scholarly legal article from a fixed set of evidence items. You may cite ONLY the evidence items listed below — never state a citation, case name, or section number from memory, even if you recognise it.

To cite an item, write your point and end it with the marker [[ev:<evidence_id>]] using the exact evidence_id given — do not write the citation text yourself; it will be rendered separately from the evidence's own record.

Rules:
1. If you quote an evidence item's exact words, wrap them in double quotes and place the marker immediately after the closing quote: "exact words" [[ev:abc123]]. Only do this if the evidence item has a verbatim_quote — do not put quote marks around a paraphrase.
2. Every substantive legal claim must carry at least one marker. A sentence with no marker should be transition/framing text only, not a legal proposition.
3. Never invent an evidence_id. Only use IDs from the list below.
4. Do not cite evidence outside the list below, even if it seems relevant — it was not selected for this section.
4a. One marker per citation, always. To cite two items for the same point, write two complete markers back to back — [[ev:abc123]] [[ev:def456]] — never combine IDs inside a single bracket like [[ev:abc123], [ev:def456]] or [[ev:abc123, ev:def456]]. A malformed marker cannot be resolved to a citation.
5. Write plain prose (markdown), no headers (the section title is added separately), target roughly {target_words} words.

WHAT IS AND IS NOT A CLAIM

Rules 2 and 4 govern propositions about the law and the world. They do not govern reasoning. An analogy, a hypothetical, a worked comparison, a restatement of an opponent's argument, or an inference drawn openly from evidence already cited is a construction, not a claim, and needs no marker — provided it introduces no new fact, case, provision or figure. If a device you want to use requires a fact you do not have in the list below, cut the device. Never manufacture the fact to save the sentence.

COMPUTATION

When the evidence list contains two or more numeric figures, you may derive a new figure from them — a percentage change, a ratio, a sum, a compound growth rate — instead of only quoting figures as given. This is a construction, like an analogy, provided every number you use is a real figure from the evidence list and the arithmetic is emitted as a calc entry the code will independently recompute. A figure standing alone is decoration; a figure derived and shown working is analysis.

To do this: state the derived figure in your prose and place [[calc:<calc_id>]] immediately after it — never write a derived number in prose without this marker, and never emit a calc entry your prose doesn't cite. Then add a matching entry to `calcs`: the same calc_id, one method (sum, difference, ratio, delta_pct, cagr), each input as {{evidence_id, value, label}} drawn only from the evidence list's own figures, `periods` if the method is cagr (the count of periods between the two values — computable from two already-cited dates, not a new fact), and your claimed_result. The arithmetic will be recomputed from your own inputs; a wrong result is a hard verification failure, not a style note, so check it before you emit it.

This adds calculation on top of real numbers — it never substitutes for having them. Do not invent a figure to make a device work.

USING PRESS EVIDENCE

Items marked PRESS in the evidence list are not authority. Handle them under three rules.

  Attribute in text. Name the outlet and the date in the sentence: "the Economic and Political Weekly argued in December 2019 that ...". A press item cited like a judgment reads as an error even when the marker is correct.

  Never let press carry a rule. If an item reports what a court held or what a circular provides, and the judgment or circular is also in the list, cite the primary item for the proposition and the press item only for reception, timing or reaction. If the primary item is not in the list, do not state the rule at all — describe what was reported and attribute it.

  Press is the right source for currency and for the prevailing view, and should be used confidently for both. Where an item establishes that the instrument was stayed, notified, deferred or challenged, say so plainly and early; that fact conditions everything the section argues. Where an item states the position the article will concede to or refuse, quote it if a verbatim_quote exists — the opposing argument is always stronger in its own words than in your summary of it, and quoting it is what makes the refusal fair.

THE ANALYTICAL ARC FOR CONVICT SECTIONS

If this section's role (given in the user message) is convict, move through it in this order — as continuous prose, in the register described below, never as labelled sub-headers:

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

# Per-role drafting instructions, selected in code by ArticleSection.role
# rather than pattern-matched from the title text. Each block replaces
# the single compressed paragraph this template used to give every
# role — the full CONCEDE/CONVICT/REFUSE/COUNTERFACTUAL/CASE_AGAINST
# architecture from _OUTLINE_SYSTEM now actually reaches the drafter.
_ROLE_INSTRUCTIONS: dict[str, str] = {
    "intro": (
        "This is the introduction. Do five things, in order, and nothing else: "
        "name the instrument and the fact that makes it live — enacted, notified, challenged, "
        "stayed — in flat sentences, no throat-clearing; frame the concept by contrast where "
        "the topic admits it (two to four competing models or jurisdictional approaches, a "
        "clause each, then the Indian position); state the uncomfortable fact that sits "
        "awkwardly with what everyone says about the topic; state the thesis as concession "
        "plus conviction plus refusal — what the article grants, what it nonetheless "
        "concludes and on how narrow a ground, and which argument on its own side it rejects; "
        "map the sections that follow by their conclusions, not their topics. There is no "
        "suspense in scholarship — give away the ending."
    ),
    "concede": (
        "This section concedes ground to the position the article will ultimately argue "
        "against, or the historical/doctrinal fact that makes the criticised rule look "
        "defensible. Concede it fully and without hedging the concession itself — a "
        "grudging concession is worse than none. Close by naming the conditions that made "
        "the criticised position defensible then, so a later section can show those "
        "conditions have lapsed."
    ),
    "convict": (
        "This section carries a legal issue toward the required conclusion above. Work the "
        "ANALYTICAL ARC FOR CONVICT SECTIONS defined in the system prompt: statutory rule, "
        "judicial interpretation, rationale, competing interpretation, factual distinction, "
        "current position, critique, author's synthesis — in that order, skipping any beat "
        "the evidence doesn't support. Land on the required conclusion in the final beat; "
        "it needs no new marker if it introduces no new fact."
    ),
    "refuse": (
        "This section shows that a popular argument made IN FAVOUR of the article's own "
        "conclusion is fallacious, incomplete, or aimed at the wrong target — it is not a "
        "second attack on the topic and must not restate criticism already made elsewhere "
        "in the article. State the popular argument fairly and, where a verbatim_quote "
        "exists, in its own holders' words — then show specifically why it fails, and say "
        "what the article's conclusion should rest on instead."
    ),
    "counterfactual": (
        "Assume every defect the article has already identified is cured tomorrow. Ask "
        "directly whether the harm survives that cure. If it does, say so and explain why "
        "the cure doesn't reach the real mechanism. If it does not survive, say that "
        "plainly too — do not force the thesis to survive a test it fails."
    ),
    "case_against": (
        "State the strongest objection to the article's whole argument, drawn only from "
        "the counter-evidence actually offered below — never invented. State it as its own "
        "holders would state it. Concede what it proves before showing what it does not "
        "prove. This section should read as a fair fight, not a formality."
    ),
    "conclusion": (
        "Restate the popular argument fairly, state the narrower conclusion actually "
        "reached, and list specific reforms or unresolved questions. No call to action, no "
        "rising cadence, no restating the full case — this section is administrative, not "
        "persuasive."
    ),
}

_DRAFT_USER_TEMPLATE = """Section: {title}
Role: {role}
This section's required conclusion — land on this, precisely: {conclusion}
{thesis_context}Legal issues this section addresses: {issues}
{established_context}
Evidence available for this section (id | kind | binding_strength | AUTHORITY/PRESS | outlet | date | statement | verbatim_quote):
{evidence_lines}

{role_instructions}

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


def _established_context(
    established_facts: list[str], hedge_counts: dict[str, int], is_first_section: bool
) -> str:
    """Render the cross-section ledger into the compact block injected
    into each draft call after the first. Deterministic — built from
    what previous sections actually cited/used, not an LLM summary."""
    lines: list[str] = []
    if not is_first_section:
        # Fires unconditionally past the first section, independent of
        # established_facts (which only tracks cited evidence) — the
        # topic/instrument's general definition and purpose is always
        # already covered by the introduction, and every later section
        # was still re-opening with it (uncited framing prose the
        # evidence-based ledger above can't see).
        lines.append(
            "The instrument/topic has already been named and defined in the introduction — "
            "do not restate what it is, when it was enacted, or its general purpose. Open "
            "directly with this section's own specific point."
        )
    if established_facts:
        lines.append("Already established earlier in the article — do not re-explain these facts:")
        lines.extend(f"  - {fact}" for fact in established_facts)
    spent = [f'"{phrase}" (used {n}x)' for phrase, n in hedge_counts.items() if n > 0]
    if spent:
        lines.append(
            "Hedge phrases already used elsewhere in the article — reach for a different "
            "one, don't repeat: " + ", ".join(spent)
        )
    return "\n".join(lines)


async def _draft_section(
    section: ArticleSection,
    evidence_for_section: list[Evidence],
    sources_by_id: dict[str, Source],
    established_facts: list[str],
    hedge_counts: dict[str, int],
    is_first_section: bool = False,
    thesis: ArticleThesis | None = None,
) -> DraftedSection:
    evidence_lines = "\n".join(
        _evidence_line(e, sources_by_id.get(e.source_id)) for e in evidence_for_section
    )
    if not evidence_lines:
        evidence_lines = "(no evidence available — write only framing text, no legal claims)"

    thesis_context = (
        f"Article thesis (this section's conclusion must serve it, not restate it): {thesis.thesis}\n"
        if thesis is not None
        else ""
    )

    system = _DRAFT_SYSTEM.format(target_words=section.target_words)
    user = _DRAFT_USER_TEMPLATE.format(
        title=section.title,
        role=section.role,
        conclusion=section.conclusion or "(none specified — use sound judgment)",
        thesis_context=thesis_context,
        issues="; ".join(section.issue_refs) or "(general)",
        established_context=_established_context(established_facts, hedge_counts, is_first_section),
        evidence_lines=evidence_lines,
        role_instructions=_ROLE_INSTRUCTIONS.get(section.role, _ROLE_INSTRUCTIONS["convict"]),
    )

    llm = get_generation_llm()
    try:
        result, usage = await llm.generate(
            system=system,
            user=user,
            output_schema=LLMSectionDraft,
            tool_name="emit_section_draft",
            max_tokens=12000,
            # Higher than the rest of the pipeline's calls (which stay
            # near 0.0 for structural/factual determinism) — voice,
            # irony and register variation need room the rest of the
            # pipeline deliberately doesn't give the model. Citation
            # correctness is enforced mechanically by Step 9 regardless
            # of temperature, so this doesn't trade away grounding.
            temperature=0.65,
        )
        body = result.body
        calcs = result.calcs
    except Exception as e:
        log.warning("draft_section_llm_failed", section_id=section.section_id, error=str(e))
        body = ""
        calcs = []

    log.info("draft_section_done", section_id=section.section_id, body_chars=len(body), calc_count=len(calcs))
    return DraftedSection(
        section_id=section.section_id,
        title=section.title,
        body=body,
        evidence_ids_offered=[e.evidence_id for e in evidence_for_section],
        calcs=calcs,
    )


async def draft_node(
    outline: ArticleOutline, package: EvidencePackage, thesis: ArticleThesis | None = None
) -> dict[str, Any]:
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

    evidence_by_id: dict[str, Evidence] = {e.evidence_id: e for e in package.evidence}

    # Cross-section ledger: sections are drafted sequentially (not
    # gathered concurrently), so each call can be told what earlier
    # sections already established — killing the "29A re-defined from
    # scratch in 6 of 7 sections" failure mode without an extra LLM call.
    established_facts: list[str] = []
    established_ids: set[str] = set()
    hedge_counts: dict[str, int] = dict.fromkeys(_HEDGE_PHRASES, 0)

    sections: list[DraftedSection] = []
    for i, section in enumerate(outline.sections):
        seen: set[str] = set()
        evidence_for_section: list[Evidence] = []
        for issue in section.issue_refs:
            for e in evidence_by_issue.get(issue, []):
                if e.evidence_id not in seen:
                    seen.add(e.evidence_id)
                    evidence_for_section.append(e)
        drafted = await _draft_section(
            section, evidence_for_section, sources_by_id, established_facts, hedge_counts,
            is_first_section=(i == 0), thesis=thesis,
        )
        sections.append(drafted)

        for ev_id in MARKER_RE.findall(drafted.body):
            if ev_id not in established_ids and ev_id in evidence_by_id:
                established_ids.add(ev_id)
                established_facts.append(evidence_by_id[ev_id].statement)
        body_lower = drafted.body.lower()
        for phrase in _HEDGE_PHRASES:
            hedge_counts[phrase] += body_lower.count(phrase)

    log.info("draft_node_done", section_count=len(sections))
    return {"drafted_sections": sections}


# ---------------------------------------------------------------------
# Step 8 — citation resolution / assembly (deterministic)
# ---------------------------------------------------------------------

def _render_section_body(
    body: str, evidence_by_id: dict[str, Evidence], calc_by_id: dict[str, CalcClaim]
) -> str:
    """Resolve every [[ev:<id>]] and [[calc:<id>]] marker in one
    section's raw drafted body into its final reader-facing text.
    Extracted from assemble_article so voice_node can run the exact
    same resolution ahead of its own pass — the voice pass only ever
    sees fully-resolved text, never a raw marker."""

    def _replace(m: re.Match) -> str:
        ev_id = m.group(1)
        ev = evidence_by_id.get(ev_id)
        if ev is None:
            return " [citation unresolved]"
        cite = render_citation(ev.citation) or "[citation unresolved]"
        pin = f", {ev.pinpoint}" if ev.pinpoint else ""
        return f" ({cite}{pin})"

    def _replace_calc(m: re.Match) -> str:
        calc = calc_by_id.get(m.group(1))
        if calc is None:
            return "[calc unresolved]"
        formatted = f"{calc.claimed_result:.2f}".rstrip("0").rstrip(".")
        return f"{formatted}{calc.unit}"

    body = CALC_MARKER_RE.sub(_replace_calc, body)
    body = MARKER_RE.sub(_replace, body)
    # Defensive cleanup: any ev: token that survived the strict
    # substitution above is a malformed marker (e.g. a combined
    # "[[ev:id1], [ev:id2]]") — never let a raw evidence_id hash
    # reach the reader just because the LLM mangled the bracket
    # syntax. verify_draft's marker check flags the same thing.
    return _STRAY_MARKER_DEBRIS_RE.sub("[citation unresolved]", body)


def assemble_article(
    outline: ArticleOutline,
    sections: list[DraftedSection],
    evidence_by_id: dict[str, Evidence],
    voiced_bodies: dict[str, str] | None = None,
) -> str:
    """STEP 8. `voiced_bodies` (section_id -> already fully-resolved,
    voice-pass-adjusted text), when given, is used verbatim instead of
    re-resolving that section's markers — Step 10 already did that and
    verified the rewrite preserved every citation/quote byte-for-byte."""
    parts = [f"# {outline.title}\n"]
    for s in sections:
        parts.append(f"## {s.title}\n")
        if voiced_bodies is not None and s.section_id in voiced_bodies:
            body = voiced_bodies[s.section_id]
        else:
            calc_by_id = {c.calc_id: c for c in s.calcs}
            body = _render_section_body(s.body, evidence_by_id, calc_by_id)
        parts.append(body + "\n")
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
    calc_issues: list[CalcIssue] = []
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

        # Any "ev:<id>" token in the body that ISN'T one of the
        # well-formed [[ev:<id>]] markers just processed above is a
        # malformed marker the LLM mangled (e.g. combining two
        # citations into one bracket) — assemble_article scrubs the
        # raw id from the rendered text regardless, but that must not
        # be a silent "passed" verdict.
        malformed_ids = list((Counter(_EV_TOKEN_RE.findall(s.body)) - Counter(markers_in_body)).elements())
        for ev_id in malformed_ids:
            marker_issues.append(
                MarkerIssue(
                    section_id=s.section_id,
                    marker_evidence_id=ev_id,
                    kind="unresolved",
                    detail="marker is malformed (not a clean [[ev:<id>]] token) — would leak raw evidence_id if not scrubbed",
                )
            )

        # Calc claims: recompute every [[calc:<id>]] actually cited in
        # prose from its own emitted inputs via the whitelisted method —
        # never trust the drafter's arithmetic, same posture as never
        # trusting its citation formatting.
        calc_by_id = {c.calc_id: c for c in s.calcs}
        for calc_id in set(CALC_MARKER_RE.findall(s.body)):
            calc = calc_by_id.get(calc_id)
            if calc is None:
                calc_issues.append(
                    CalcIssue(
                        section_id=s.section_id,
                        calc_id=calc_id,
                        detail="marker does not resolve to any calc entry this section emitted",
                    )
                )
                continue
            out_of_scope = [i.evidence_id for i in calc.inputs if i.evidence_id not in offered]
            if out_of_scope:
                calc_issues.append(
                    CalcIssue(
                        section_id=s.section_id,
                        calc_id=calc_id,
                        detail=f"input evidence not offered to this section: {out_of_scope}",
                    )
                )
                continue
            computed = _compute_calc(calc)
            if computed is None:
                calc_issues.append(
                    CalcIssue(
                        section_id=s.section_id,
                        calc_id=calc_id,
                        detail=f"method '{calc.method}' has the wrong number/shape of inputs to compute",
                    )
                )
            elif not _calc_matches(computed, calc.claimed_result):
                calc_issues.append(
                    CalcIssue(
                        section_id=s.section_id,
                        calc_id=calc_id,
                        detail=f"claimed_result={calc.claimed_result} does not match recomputed value={computed:.4f}",
                    )
                )

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

    if marker_issues or quote_issues or calc_issues:
        verdict: str = "failed"
        rationale = (
            f"{len(marker_issues)} marker issue(s), {len(quote_issues)} quote issue(s), "
            f"{len(calc_issues)} calc issue(s) — draft cites evidence it wasn't given, alters "
            "a quote it was given, or its arithmetic doesn't recompute."
        )
    elif uncovered_issues:
        verdict = "failed"
        rationale = f"{len(uncovered_issues)} legal issue(s) have zero cited evidence anywhere in the draft."
    else:
        verdict = "passed"
        rationale = (
            f"All {len(all_marker_ids)} citation marker(s) resolve to in-scope evidence, "
            f"all quoted text matches its source evidence, all calc claims recompute, and "
            f"every legal issue is cited."
        )

    return DraftVerificationReport(
        run_id=outline.run_id,
        marker_issues=marker_issues,
        quote_issues=quote_issues,
        calc_issues=calc_issues,
        uncovered_issues=uncovered_issues,
        citation_count=len(all_marker_ids),
        unique_evidence_cited=unique_evidence_cited,
        verdict=verdict,  # type: ignore[arg-type]
        rationale=rationale,
    )


# ---------------------------------------------------------------------
# Step 10 — voice pass (one LLM call per section, opportunistic)
# ---------------------------------------------------------------------
#
# Runs last, on already fully-resolved text (citations and calc results
# rendered — no [[ev:]]/[[calc:]] markers left to hallucinate). Its only
# job is rhythm: break up runs of same-length sentences and cut the
# mechanical "thereby ensuring..." participial tail. Every parenthetical
# citation and every quoted span must survive byte-for-byte; if the
# model touches even one, that section's rewrite is discarded and the
# original resolved text is kept. This pass is opportunistic, never
# load-bearing — a rejected rewrite is not a failure, just a no-op.

_PAREN_SPAN_RE = re.compile(r"\([^()]*\)")
_QUOTE_SPAN_RE = re.compile(r"[“\"]([^”\"]{3,})[”\"]")


class LLMVoicePass(BaseModel):
    # Matches LLMSectionDraft — the voice pass returns roughly the same
    # length it was given, so it needs the same headroom.
    body: str = Field(max_length=24000)


_VOICE_SYSTEM = """You are a copy-editor with an extremely limited mandate: improve the sentence-level rhythm of one already-complete, fully-cited section of a scholarly legal article. The facts, citations, quotations, and argument are already correct and final. You may not change any of them — only how the existing sentences are built.

Do exactly these three things, nothing else:
1. Vary sentence length. A section where every sentence runs 20-35 words reads as mechanical. Break some up. Land a point in three or four words if that is the strongest way to state it. Some sentences should be long and carry substance; the next should be short and land it.
2. Cut the participial tail. "...thereby ensuring the integrity of the process." / "...underscoring its importance." / "...highlighting the need for reform." / "...emphasizing its role in..." / "...reflecting legislative intent." — this dangling "-ing" clause bolted onto an already-finished sentence is the single most mechanical tic in this prose. Find every one. Either cut it (if it adds nothing) or split it into its own short, direct sentence.
3. Don't repeat the same transition word to open more than one sentence in the section (e.g. "However," starting two sentences). Vary it, or delete the transition if the sentences already flow without it.

Do not do anything else. Do not add adjectives, adverbs, jokes, or new sentences of substance. Do not reorder paragraphs. Do not shorten the section meaningfully.

ABSOLUTE — must survive byte-for-byte, unchanged, in the same order, in the output:
- Every parenthetical citation exactly as given, e.g. "(Swiss Ribbons Pvt. Ltd. vs Union Of India, 2019, para 4)" — same case name, same punctuation, same pinpoint. Do not reformat, abbreviate, merge, reorder, or drop a single one.
- Every quoted string in double quotes, character for character. Do not paraphrase, correct, or trim a quotation.
- Every case name, statute section, date, number, and the section's conclusion, unchanged.

If improving a sentence's rhythm would require touching a citation, a quotation, or a fact, leave that sentence exactly as it is instead. An unchanged sentence is always the safe choice; a rewritten one that drops a citation is not acceptable under any circumstance."""

_VOICE_USER_TEMPLATE = """Section text (already fully cited — rewrite only for the rhythm/tics above, nothing else):
{body}"""


def _voice_pass_is_safe(original: str, rewritten: str) -> bool:
    if Counter(_PAREN_SPAN_RE.findall(original)) != Counter(_PAREN_SPAN_RE.findall(rewritten)):
        return False
    return Counter(_QUOTE_SPAN_RE.findall(original)) == Counter(_QUOTE_SPAN_RE.findall(rewritten))


async def _voice_pass_section(rendered_body: str, section_id: str) -> str:
    if not rendered_body.strip():
        return rendered_body

    llm = get_generation_llm()
    try:
        result, usage = await llm.generate(
            system=_VOICE_SYSTEM,
            user=_VOICE_USER_TEMPLATE.format(body=rendered_body),
            output_schema=LLMVoicePass,
            tool_name="emit_voice_pass",
            max_tokens=12000,
            temperature=0.5,
        )
    except Exception as e:
        log.warning("voice_pass_llm_failed", section_id=section_id, error=str(e))
        return rendered_body

    if _voice_pass_is_safe(rendered_body, result.body):
        return result.body
    log.warning("voice_pass_rejected_unsafe_edit", section_id=section_id)
    return rendered_body


async def voice_node(
    sections: list[DraftedSection], evidence_by_id: dict[str, Evidence]
) -> dict[str, Any]:
    """STEP 10: final per-section rhythm pass. Returns section_id ->
    final text, ready to hand to assemble_article's voiced_bodies."""
    ctx_node.set("voice")

    voiced_bodies: dict[str, str] = {}
    for s in sections:
        calc_by_id = {c.calc_id: c for c in s.calcs}
        resolved = _render_section_body(s.body, evidence_by_id, calc_by_id)
        voiced_bodies[s.section_id] = await _voice_pass_section(resolved, s.section_id)

    log.info("voice_node_done", section_count=len(voiced_bodies))
    return {"voiced_bodies": voiced_bodies}


# ---------------------------------------------------------------------
# Node entry point — ties Steps 5b-10 together
# ---------------------------------------------------------------------

async def generation_node(package: EvidencePackage, article_config: ArticleConfig) -> dict[str, Any]:
    ctx_node.set("generation")

    thesis_result = await thesis_node(package)
    thesis: ArticleThesis = thesis_result["thesis"]

    outline_result = await outline_node(package, article_config, thesis)
    outline: ArticleOutline = outline_result["outline"]

    draft_result = await draft_node(outline, package, thesis)
    sections: list[DraftedSection] = draft_result["drafted_sections"]

    evidence_by_id = {e.evidence_id: e for e in package.evidence}

    # Verify BEFORE the voice pass — verify_draft checks the raw
    # [[ev:]]/[[calc:]] markers in each section's drafted body, which
    # the voice pass never touches (it only rewrites already-resolved
    # text). Running verification first also means a failed draft still
    # gets a rendered article for inspection, same as before Step 10 existed.
    report = verify_draft(outline, sections, evidence_by_id, package.legal_issues)

    voice_result = await voice_node(sections, evidence_by_id)
    voiced_bodies: dict[str, str] = voice_result["voiced_bodies"]

    rendered = assemble_article(outline, sections, evidence_by_id, voiced_bodies)

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
        calc_issue_count=len(report.calc_issues),
        uncovered_issue_count=len(report.uncovered_issues),
    )

    return {
        "thesis": thesis,
        "outline": outline,
        "article_draft": draft,
        "draft_verification": report,
    }