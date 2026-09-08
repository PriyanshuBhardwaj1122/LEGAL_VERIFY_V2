"""Indian citation regex suite.

Parses heterogeneous Indian legal citation formats into structured
LegalCitation objects. Sets is_parsed=True only when a named group set
is complete — downstream must treat is_parsed=False as "string only".
"""

from __future__ import annotations

import re
from datetime import date

from app.schemas.citation import LegalCitation

# -------------------------------------------------------------------
# Compiled patterns — order matters: try most specific first
# -------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # Neutral SC citation: "2023 INSC 456"
    (
        "neutral_sc",
        re.compile(r"(?P<year>\d{4})\s+INSC\s+(?P<num>\d+)"),
        "case",
    ),
    # Neutral HC citation: "2024:DHC:1234"
    (
        "neutral_hc",
        re.compile(r"(?P<year>\d{4}):(?P<court>[A-Z]{2,6}):(?P<num>\d+)"),
        "case",
    ),
    # SCC: "(2023) 5 SCC 1"
    (
        "scc",
        re.compile(r"\((?P<year>\d{4})\)\s+(?P<vol>\d+)\s+SCC\s+(?P<page>\d+)"),
        "case",
    ),
    # AIR: "AIR 2019 SC 123"
    (
        "air",
        re.compile(r"AIR\s+(?P<year>\d{4})\s+(?P<court>SC|[A-Z][a-z]+)\s+(?P<page>\d+)"),
        "case",
    ),
    # SCC OnLine: "2021 SCC OnLine Del 456"
    (
        "scc_online",
        re.compile(r"(?P<year>\d{4})\s+SCC\s+OnLine\s+(?P<court>[A-Za-z]+)\s+(?P<num>\d+)"),
        "case",
    ),
    # Case number: "Civil Appeal No. 1234 of 2020"
    (
        "case_number",
        re.compile(
            r"(?P<type>Civil|Criminal|Special\s+Leave|Writ)\s+"
            r"(?P<noun>Appeal|Petition|Application)\s+No\.?\s*(?P<num>\d+)\s+of\s+(?P<year>\d{4})"
        ),
        "case",
    ),
    # Writ petition: "W.P.(C) No. 123 of 2024"
    (
        "writ_petition",
        re.compile(
            r"W\.?P\.?\s*\((?P<type>C|Crl)\)\s*No\.?\s*(?P<num>\d+)\s*of\s*(?P<year>\d{4})"
        ),
        "case",
    ),
    # Statute section: "Section 29A(3)(c) of the Insolvency and Bankruptcy Code, 2016"
    (
        "statute_section",
        re.compile(
            r"[Ss]ection\s+(?P<sec>\d+[A-Z]{0,2}(?:\(\w+\))*)\s+of\s+(?:the\s+)?"
            r"(?P<act>[A-Z][\w\s,\.]+?,\s*\d{4})"
        ),
        "statute",
    ),
    # SEBI circular: "SEBI/HO/CFD/CMD/CIR/P/2020/12"
    (
        "sebi_circular",
        re.compile(r"SEBI/HO/[A-Z0-9/\-]+/(?P<year>\d{4})/(?P<num>\d+)"),
        "circular",
    ),
    # RBI circular: "RBI/2023-24/45"
    (
        "rbi_circular",
        re.compile(r"RBI/(?P<yr>\d{4}-\d{2})/(?P<num>\d+)"),
        "circular",
    ),
]


def parse_citation(raw: str) -> LegalCitation:
    """Parse a raw citation string into a structured LegalCitation.

    Tries each pattern in order; returns the first match with is_parsed=True.
    Falls back to an unparsed citation with kind='other'.
    """
    raw_stripped = raw.strip()

    for name, pattern, kind in _PATTERNS:
        m = pattern.search(raw_stripped)
        if not m:
            continue

        groups = m.groupdict()

        if kind == "case":
            return _build_case_citation(raw_stripped, name, groups)
        elif kind == "statute":
            return _build_statute_citation(raw_stripped, groups)
        elif kind == "circular":
            return _build_circular_citation(raw_stripped, name, groups)

    # No pattern matched
    return LegalCitation(raw=raw_stripped, kind="other", is_parsed=False)


def _build_case_citation(raw: str, pattern_name: str, g: dict) -> LegalCitation:
    """Build a case citation from a regex match."""
    cit = LegalCitation(raw=raw, kind="case", is_parsed=True)

    year = g.get("year")
    court = g.get("court")

    if pattern_name == "neutral_sc":
        cit.neutral_citation = f"{year} INSC {g['num']}"
        cit.court = "Supreme Court of India"
    elif pattern_name == "neutral_hc":
        cit.neutral_citation = f"{year}:{court}:{g['num']}"
        cit.court = _expand_hc_code(court) if court else None
    elif pattern_name == "scc":
        cit.reporter_citation = f"({year}) {g['vol']} SCC {g['page']}"
        cit.court = "Supreme Court of India"  # SCC is almost always SC
    elif pattern_name == "air":
        cit.reporter_citation = f"AIR {year} {court} {g['page']}"
        cit.court = _expand_air_court(court) if court else None
    elif pattern_name == "scc_online":
        cit.scc_online = f"{year} SCC OnLine {court} {g['num']}"
        cit.court = _expand_hc_code(court) if court else None
    elif pattern_name == "case_number":
        # Use the captured noun — hardcoding "Appeal" rendered a
        # Special Leave Petition as a "Special Leave Appeal", i.e. a
        # citation that is confidently wrong. Worse than unresolved.
        cit.case_number = f"{g['type']} {g.get('noun') or 'Appeal'} No. {g['num']} of {year}"
    elif pattern_name == "writ_petition":
        cit.case_number = f"W.P.({g['type']}) No. {g['num']} of {year}"

    return cit


def _build_statute_citation(raw: str, g: dict) -> LegalCitation:
    act_raw = g.get("act", "").strip().rstrip(",. ")
    year = None
    # Extract the year from the act name AND remove it, so act_name is
    # the bare title. render_citation re-appends act_year itself, so
    # leaving it here produced "Insolvency and Bankruptcy Code, 2016,
    # 2016, s. 29A" — which shipped in a real article.
    year_match = re.search(r"(\d{4})\s*$", act_raw)
    if year_match:
        year = int(year_match.group(1))
        act_raw = act_raw[: year_match.start()].strip().rstrip(",. ")

    return LegalCitation(
        raw=raw,
        kind="statute",
        act_name=act_raw,
        act_year=year,
        section=g.get("sec"),
        is_parsed=True,
    )


def _build_circular_citation(raw: str, pattern_name: str, g: dict) -> LegalCitation:
    match_str = raw  # Use the full matched string
    if pattern_name == "sebi_circular":
        return LegalCitation(
            raw=raw,
            kind="circular",
            circular_number=match_str,
            is_parsed=True,
        )
    elif pattern_name == "rbi_circular":
        return LegalCitation(
            raw=raw,
            kind="circular",
            circular_number=f"RBI/{g['yr']}/{g['num']}",
            is_parsed=True,
        )
    return LegalCitation(raw=raw, kind="circular", is_parsed=False)


# -------------------------------------------------------------------
# Court code expansion helpers
# -------------------------------------------------------------------

_HC_CODES: dict[str, str] = {
    "DHC": "Delhi High Court",
    "BHC": "Bombay High Court",
    "MHC": "Madras High Court",
    "Cal": "Calcutta High Court",
    "KHC": "Karnataka High Court",
    "AHC": "Allahabad High Court",
    "APHC": "Andhra Pradesh High Court",
    "TSHC": "Telangana High Court",
    "KER": "Kerala High Court",
    "GHC": "Gujarat High Court",
    "PHC": "Punjab and Haryana High Court",
    "Del": "Delhi High Court",
    "Bom": "Bombay High Court",
    "Mad": "Madras High Court",
    "Kar": "Karnataka High Court",
    "All": "Allahabad High Court",
    "Ker": "Kerala High Court",
    "Guj": "Gujarat High Court",
    "SC": "Supreme Court of India",
}


def _expand_hc_code(code: str) -> str | None:
    return _HC_CODES.get(code)


def _expand_air_court(code: str) -> str | None:
    if code == "SC":
        return "Supreme Court of India"
    return _HC_CODES.get(code)
