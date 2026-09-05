"""Choose which parts of a long document are worth sending to the LLM.

The extractor used to read a document's first 25,000 characters and stop.
On a 250,000-char judgment that is the cover sheet and the recital of
counsel's submissions — the court's own reasoning and its final order sit
tens of thousands of characters further down and were never read at all.

Reading everything fixes that but costs ~7x more. This module is the
middle path: score every passage in the whole document, then spend the
budget on the best ones. The document is scanned in full; only the
selected passages are paid for.

Scoring is jurisdiction-agnostic by construction:
  * relevance comes from the caller's own legal issues, which derive from
    whatever topic the user asked about;
  * discourse signals come from app/domain/legal_discourse.py, which
    encodes common-law phrasing rather than any one country's rules.

Nothing here touches stored quote offsets: grounding always re-locates
quotes against the FULL document text, so a selection artefact can only
cause a quote to fail grounding, never to be recorded at a wrong offset.
"""

from __future__ import annotations

import math
import re

from app.domain.legal_discourse import counts

# Roughly a long paragraph. Small enough to score meaningfully, large
# enough that a holding is rarely split across two.
PASSAGE_CHARS = 2_000

OMISSION_MARKER = "\n\n[... omitted ...]\n\n"

_WORD_RE = re.compile(r"[a-z]{3,}")

# Words too common in legal prose to discriminate between passages.
_STOPWORDS = frozenset(
    """the and that for with this from would could shall may not are was were
    has have had been being which who whom whose any all such other than then
    upon under over into out its his her their our your there here when where
    what while about after before during between within without also more most
    said case court order section act code law legal rule",""".split()
)


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


def split_passages(text: str, size: int = PASSAGE_CHARS) -> list[tuple[int, str]]:
    """Split into (offset, passage) on sentence/line boundaries where
    possible. Offsets are into the original text so callers can reason
    about document position."""
    out: list[tuple[int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window_floor = start + size // 2
            cut = -1
            for sep in ("\n", ". "):
                found = text.rfind(sep, window_floor, end)
                if found > cut:
                    cut = found + len(sep)
            if cut > window_floor:
                end = cut
        out.append((start, text[start:end]))
        start = end
    return out


def _issue_idf(passages: list[tuple[int, str]], issue_terms: set[str]) -> dict[str, float]:
    """Down-weight issue terms that appear everywhere in this document —
    on a Section 29A judgment, "29a" is in every passage and carries no
    information about which passage is best."""
    df: dict[str, int] = {}
    for _, p in passages:
        seen = set(_tokens(p)) & issue_terms
        for t in seen:
            df[t] = df.get(t, 0) + 1
    n = max(1, len(passages))
    return {t: math.log(1 + n / (1 + df.get(t, 0))) for t in issue_terms}


def score_passages(
    text: str, legal_issues: list[str], size: int = PASSAGE_CHARS
) -> list[tuple[int, str, float]]:
    """Return (offset, passage, score) for every passage in the document."""
    passages = split_passages(text, size)
    issue_terms = set(_tokens(" ".join(legal_issues)))
    idf = _issue_idf(passages, issue_terms)
    n = max(1, len(passages))

    scored: list[tuple[int, str, float]] = []
    for idx, (offset, p) in enumerate(passages):
        toks = _tokens(p)
        denom = math.sqrt(len(toks)) or 1.0
        relevance = sum(idf.get(t, 0.0) for t in toks if t in issue_terms) / denom

        c = counts(p)
        # Court reasoning and separate opinions are what we want; a
        # party's submissions and the counsel list are what we don't.
        discourse = (
            1.6 * c["court_voice"]
            + 1.2 * c["dissent"]
            + 0.5 * c["instrument"]
            - 1.4 * c["attribution"]
            - 1.0 * c["front_matter"]
        )

        # Judgments put the disposition at the very end; the opening
        # carries the cause title and the case's own citation.
        pos = idx / n
        positional = 1.2 if pos > 0.85 else (0.4 if pos < 0.05 else 0.0)

        scored.append((offset, p, 3.0 * relevance + discourse + positional))
    return scored


def build_digests(
    text: str,
    legal_issues: list[str],
    n_digests: int,
    max_chars: int = 25_000,
    passage_size: int = PASSAGE_CHARS,
) -> list[str]:
    """Pack the highest-scoring passages of `text` into at most
    `n_digests` LLM-sized payloads.

    The opening passage and the last two are always included regardless
    of score: the opening carries the cause title and formal citation the
    extractor needs for `citation_raw`, and the final passages carry the
    operative order, which is the single most citable line in a judgment
    and sits where nothing else looks.

    Passages are emitted in document order within each digest, separated
    by an explicit omission marker so the model can see that material was
    skipped and must not quote across the gap.
    """
    scored = score_passages(text, legal_issues, passage_size)
    if not scored:
        return []

    budget = n_digests * max_chars
    forced = {0, len(scored) - 1, max(0, len(scored) - 2)}

    chosen: set[int] = set()
    used = 0
    for i in sorted(forced):
        if used + len(scored[i][1]) <= budget:
            chosen.add(i)
            used += len(scored[i][1])

    for i, (_, passage, _score) in sorted(
        enumerate(scored), key=lambda kv: kv[1][2], reverse=True
    ):
        if i in chosen:
            continue
        if used + len(passage) > budget:
            continue
        chosen.add(i)
        used += len(passage)

    # Greedy packing is not perfectly efficient, so a selection that fits
    # the raw character budget can still need one more bin than allowed.
    # Shed the WEAKEST passages until it fits — never the tail. Truncating
    # the digest list instead would discard the final passages, which is
    # where a judgment's operative order lives and which `forced` exists
    # specifically to protect.
    while True:
        digests = _pack(scored, chosen, max_chars)
        if len(digests) <= n_digests:
            return digests
        droppable = [i for i in chosen if i not in forced]
        if not droppable:
            # Only forced passages left and still over budget: keep the
            # first n_digests bins rather than loop forever.
            return digests[:n_digests]
        chosen.discard(min(droppable, key=lambda i: scored[i][2]))


def _pack(
    scored: list[tuple[int, str, float]], chosen: set[int], max_chars: int
) -> list[str]:
    """Pack the chosen passages into digests of at most `max_chars`,
    preserving document order and marking skipped material."""
    digests: list[str] = []
    current: list[str] = []
    current_len = 0
    for i in sorted(chosen):
        passage = scored[i][1]
        if current and current_len + len(passage) > max_chars:
            digests.append(OMISSION_MARKER.join(current))
            current, current_len = [], 0
        current.append(passage)
        current_len += len(passage)
    if current:
        digests.append(OMISSION_MARKER.join(current))
    return digests
