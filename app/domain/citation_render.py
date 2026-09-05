"""Pure, deterministic rendering of a LegalCitation into a human-readable
Indian-legal-style citation string. No LLM involved — this is the same
"deterministic code decides the final form" posture as coverage.py and
grounding.py's binding-strength computation. If the structured fields
needed for a clean render aren't present (unparsed citation), we fall
back to the raw string rather than fabricating structure.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from app.schemas.citation import LegalCitation
from app.schemas.common import SourceType
from app.schemas.source import Source

# IndianKanoon and similar indexes title judgments as a cause title with
# a date tail — "X vs Y on 4 October, 2018". The tail is redundant once
# the date is rendered separately, so it's stripped.
_TITLE_DATE_TAIL_RE = re.compile(r"\s+on\s+\d{1,2}\s+\w+,?\s+\d{4}\s*$", re.IGNORECASE)

# Indexes append an ellipsis where they truncated a long cause title.
# Printed in an article it reads as a defect, because it is one.
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\s*\.\s*\.|…)\s*$")

# A judgment often carries every parallel citation at once —
# "1994 AIR 988 1993 SCR (3) 128 1993 SCC (3) 499 JT 1993 (3) 15 ...".
# Lawyers cite ONE reporter, by preference the most authoritative.
# Each pattern below captures a single complete citation.
_REPORTER_PATTERNS = [
    # Supreme Court Cases is the preferred Indian reporter, and appears
    # in two word orders depending on the source that scraped it.
    re.compile(r"\(\d{4}\)\s*\d+\s*SCC\s*\d+"),              # (1993) 3 SCC 499
    re.compile(r"\d{4}\s*SCC\s*\(\d+\)\s*\d+"),              # 1993 SCC (3) 499
    re.compile(r"\d{4}\s*\(\d+\)\s*S\.?\s?C\.?\s?C\.?\s*\d+"),  # 1983 (1) S.C.C. 71
    re.compile(r"\d{4}\s*SCC\s*OnLine\s*\w+\s*\d+"),
    re.compile(r"\d{4}\s*INSC\s*\d+"),                        # 2023 INSC 709
    re.compile(r"AIR\s*\d{4}\s*[A-Z]{2,4}\s*\d+"),            # AIR 1994 SC 988
    re.compile(r"\d{4}\s*AIR\s*\d+"),                         # 1994 AIR 988
]

# Roughly "this looks like several citations jammed together".
_MULTI_CITATION_HINT = re.compile(r"(SCC|AIR|SCR|SCALE|JT)\b.*\b(SCC|AIR|SCR|SCALE|JT)\b")


def tidy_citation_string(raw: str) -> str:
    """Make a stored citation string fit to print.

    Two defects show up constantly in scraped legal sources: a dump of
    every parallel citation for one judgment, and a trailing ellipsis
    from a truncated title. Both are instantly recognisable as
    machine-assembled when they reach the page.
    """
    s = (raw or "").strip()
    if not s:
        return ""

    if _MULTI_CITATION_HINT.search(s):
        # Prefer the first pattern that matches, in authority order.
        for pattern in _REPORTER_PATTERNS:
            m = pattern.search(s)
            if m:
                return m.group(0).strip()

    s = _ELLIPSIS_RE.sub("", s)
    # Cause titles from indexes use "vs"; law reports use "v".
    s = re.sub(r"\s+vs\.?\s+", " v ", s, flags=re.IGNORECASE)
    return s.strip(" ,;")

_PRIMARY_SOURCE_TYPES = {
    SourceType.JUDGMENT,
    SourceType.TRIBUNAL_ORDER,
}
_INSTRUMENT_SOURCE_TYPES = {
    SourceType.STATUTE,
    SourceType.SUBORDINATE_LEGISLATION,
    SourceType.GAZETTE_NOTIFICATION,
    SourceType.REGULATOR_CIRCULAR,
    SourceType.BILL_OR_DRAFT,
}


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def render_source_attribution(source: Source | None) -> str:
    """Best available attribution built from a Source's own metadata,
    for evidence that carries no parseable citation of its own.

    This is deliberately NOT dressed up to look like a formal citation —
    a commentary piece renders as a quoted title plus outlet so a reader
    can tell at a glance that it isn't authority. Returns "" when there
    is genuinely nothing to say, so the caller still flags it rather
    than printing an empty parenthesis.
    """
    if source is None:
        return ""

    title = (source.title or "").strip()
    outlet = (source.issuing_body or "").strip() or _domain_of(source.url_canonical or "")
    year = source.decided_or_published_on.year if source.decided_or_published_on else None

    if source.source_type in _PRIMARY_SOURCE_TYPES:
        # Cause title carries the identity; court and date qualify it.
        name = tidy_citation_string(_TITLE_DATE_TAIL_RE.sub("", title))
        if not name:
            return ""
        qualifiers = [q for q in (outlet or None, str(year) if year else None) if q]
        return f"{name} ({', '.join(qualifiers)})" if qualifiers else name

    if source.source_type in _INSTRUMENT_SOURCE_TYPES:
        # The title of a statute/circular page is usually the instrument.
        if not title:
            return ""
        return f"{title} ({outlet}, {year})" if outlet and year else title

    # Commentary, news, academic, other — quote the title and name the
    # outlet, so it never reads as authority.
    title = _ELLIPSIS_RE.sub("", title).strip(" ,;")
    if title.isupper() and len(title) > 12:
        # Scraped headings are often shouted; printing them verbatim
        # makes the page look scraped.
        title = title.title()
    if title and outlet:
        return f'"{title}", {outlet}' + (f", {year}" if year else "")
    if title:
        return f'"{title}"' + (f" ({year})" if year else "")
    if outlet:
        return f"{outlet}" + (f", {year}" if year else "")
    return ""


def render_evidence_citation(
    citation: LegalCitation | None, source: Source | None = None
) -> str:
    """The citation for a piece of evidence: its own parsed/raw citation
    when it has one, otherwise an attribution built from the source it
    came from. Returns "" only when neither yields anything."""
    return render_citation(citation) or render_source_attribution(source)


def render_citation(citation: LegalCitation | None) -> str:
    if citation is None:
        return ""

    if not citation.is_parsed:
        return tidy_citation_string(citation.raw)

    if citation.kind == "case":
        parts = []
        if citation.case_name:
            parts.append(citation.case_name)
        reporter = citation.neutral_citation or citation.reporter_citation or citation.scc_online
        if reporter:
            parts.append(reporter)
        if citation.court:
            parts.append(f"({citation.court})")
        if citation.decided_on:
            parts.append(str(citation.decided_on.year))
        return ", ".join(p for p in parts if p) or citation.raw

    if citation.kind in ("statute", "regulation"):
        parts = []
        if citation.act_name:
            act = citation.act_name
            if citation.act_year:
                act += f", {citation.act_year}"
            parts.append(act)
        if citation.section:
            parts.append(f"s. {citation.section}")
        return ", ".join(p for p in parts if p) or citation.raw

    if citation.kind == "circular":
        parts = []
        if citation.circular_number:
            parts.append(f"Circular No. {citation.circular_number}")
        if citation.issued_on:
            parts.append(f"dated {citation.issued_on.isoformat()}")
        return ", ".join(p for p in parts if p) or citation.raw

    if citation.kind == "academic":
        parts = []
        if citation.case_name:  # reused as title if present
            parts.append(citation.case_name)
        if citation.doi:
            parts.append(f"DOI: {citation.doi}")
        return ", ".join(p for p in parts if p) or citation.raw

    return tidy_citation_string(citation.raw)
