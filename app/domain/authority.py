"""Indian source authority tier table and resolver.

Authority is a table lookup, not a judgment call. The LLM never decides
whether the Supreme Court outranks a law-firm blog.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from app.schemas.common import CourtLevel, Jurisdiction, SourceType


# -----------------------------------------------------------------------
# Tier table: domain patterns → (tier, court_level, issuing_body, source_type)
# Checked in order; first match wins.
# -----------------------------------------------------------------------

_DOMAIN_RULES: list[tuple[str, int, CourtLevel, str | None, SourceType | None]] = [
    # Tier 1 — Apex primary law
    ("sci.gov.in", 1, CourtLevel.SUPREME_COURT, "Supreme Court of India", SourceType.JUDGMENT),
    ("digiscr.sci.gov.in", 1, CourtLevel.SUPREME_COURT, "Supreme Court of India", SourceType.JUDGMENT),
    ("indiacode.nic.in", 1, CourtLevel.NONE, None, SourceType.STATUTE),
    ("egazette.gov.in", 1, CourtLevel.NONE, None, SourceType.GAZETTE_NOTIFICATION),

    # Tier 2 — Binding/authoritative primary
    ("judgments.ecourts.gov.in", 2, CourtLevel.HIGH_COURT, None, SourceType.JUDGMENT),
    ("hcservices.ecourts.gov.in", 2, CourtLevel.HIGH_COURT, None, SourceType.JUDGMENT),
    ("sebi.gov.in", 2, CourtLevel.NONE, "SEBI", SourceType.REGULATOR_CIRCULAR),
    ("rbi.org.in", 2, CourtLevel.NONE, "RBI", SourceType.REGULATOR_CIRCULAR),
    ("mca.gov.in", 2, CourtLevel.NONE, "MCA", SourceType.REGULATOR_CIRCULAR),
    ("cci.gov.in", 2, CourtLevel.NONE, "CCI", SourceType.REGULATOR_CIRCULAR),
    ("irdai.gov.in", 2, CourtLevel.NONE, "IRDAI", SourceType.REGULATOR_CIRCULAR),
    ("trai.gov.in", 2, CourtLevel.NONE, "TRAI", SourceType.REGULATOR_CIRCULAR),
    ("incometaxindia.gov.in", 2, CourtLevel.NONE, "CBDT", SourceType.REGULATOR_CIRCULAR),
    ("cbic.gov.in", 2, CourtLevel.NONE, "CBIC", SourceType.REGULATOR_CIRCULAR),
    ("ibbi.gov.in", 2, CourtLevel.NONE, "IBBI", SourceType.REGULATOR_CIRCULAR),

    # Tier 3 — First-instance tribunal & official secondary
    ("nclat.nic.in", 3, CourtLevel.TRIBUNAL_APPELLATE, "NCLAT", SourceType.TRIBUNAL_ORDER),
    ("nclt.gov.in", 3, CourtLevel.TRIBUNAL, "NCLT", SourceType.TRIBUNAL_ORDER),
    ("greentribunal.gov.in", 3, CourtLevel.TRIBUNAL, "NGT", SourceType.TRIBUNAL_ORDER),
    ("lawcommissionofindia.nic.in", 3, CourtLevel.NONE, "Law Commission of India", SourceType.LAW_COMMISSION_REPORT),
    ("prsindia.org", 4, CourtLevel.NONE, "PRS Legislative Research", SourceType.COMMITTEE_REPORT),

    # Tier 4 — Scholarly
    ("ssrn.com", 4, CourtLevel.NONE, None, SourceType.ACADEMIC),
    ("papers.ssrn.com", 4, CourtLevel.NONE, None, SourceType.ACADEMIC),
    ("academic.oup.com", 4, CourtLevel.NONE, None, SourceType.ACADEMIC),
    ("link.springer.com", 4, CourtLevel.NONE, None, SourceType.ACADEMIC),
    ("jstor.org", 4, CourtLevel.NONE, None, SourceType.ACADEMIC),
    ("semanticscholar.org", 4, CourtLevel.NONE, None, SourceType.ACADEMIC),

    # Tier 5 — Professional commentary
    ("indiankanoon.org", 5, CourtLevel.NONE, None, None),  # IK hosts judgments (tier varies)
    ("barandbench.com", 5, CourtLevel.NONE, None, SourceType.LEGAL_NEWS),
    ("livelaw.in", 5, CourtLevel.NONE, None, SourceType.LEGAL_NEWS),
    ("scconline.com", 5, CourtLevel.NONE, None, SourceType.FIRM_COMMENTARY),
    ("indiacorplaw.in", 5, CourtLevel.NONE, None, SourceType.FIRM_COMMENTARY),
    ("vinodkothari.com", 5, CourtLevel.NONE, None, SourceType.FIRM_COMMENTARY),
    ("mondaq.com", 5, CourtLevel.NONE, None, SourceType.FIRM_COMMENTARY),
    ("nishithdesai.com", 5, CourtLevel.NONE, None, SourceType.FIRM_COMMENTARY),
    ("cyrilshroff.com", 5, CourtLevel.NONE, None, SourceType.FIRM_COMMENTARY),
    ("khaitanco.com", 5, CourtLevel.NONE, None, SourceType.FIRM_COMMENTARY),
    ("trilegal.com", 5, CourtLevel.NONE, None, SourceType.FIRM_COMMENTARY),
    ("azbpartners.com", 5, CourtLevel.NONE, None, SourceType.FIRM_COMMENTARY),
]

# Indian Kanoon special handling: the URL path tells you if it's a judgment
_IK_JUDGMENT_PATTERN = re.compile(r"/doc/\d+")


def resolve_authority(
    url: str,
    source_type_hint: SourceType | None = None,
) -> tuple[int, CourtLevel, str | None, SourceType]:
    """Resolve a URL to its authority tier and metadata.

    Returns (tier, court_level, issuing_body, source_type).
    Tier 6 = unknown/junk.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or ""

    # Check against domain rules
    for domain, tier, court, body, stype in _DOMAIN_RULES:
        if host == domain or host.endswith(f".{domain}"):
            # Special case: Indian Kanoon hosts actual judgments
            if domain == "indiankanoon.org" and _IK_JUDGMENT_PATTERN.search(path):
                # IK judgment docs — could be SC, HC, or tribunal
                # Without parsing the doc, default to tier 2 (HC level)
                return (2, CourtLevel.HIGH_COURT, None, SourceType.JUDGMENT)

            resolved_type = stype or source_type_hint or SourceType.OTHER
            return (tier, court, body, resolved_type)

    # HC sites follow patterns like bombay.hc.in, delhihighcourt.nic.in
    if "highcourt" in host or "hc.in" in host or host.endswith("hcourt.gov.in"):
        return (2, CourtLevel.HIGH_COURT, None, SourceType.JUDGMENT)

    # Government domains default to tier 3
    if host.endswith(".gov.in") or host.endswith(".nic.in"):
        return (3, CourtLevel.NONE, None, source_type_hint or SourceType.OTHER)

    # Academic publisher patterns
    if any(kw in host for kw in ["journal", "nlsiu", "nujs", "nluj", "nlu"]):
        return (4, CourtLevel.NONE, None, SourceType.ACADEMIC)

    # Default: tier 6 (junk) — caller decides whether to keep or drop
    return (6, CourtLevel.NONE, None, source_type_hint or SourceType.OTHER)


def authority_weight(tier: int) -> float:
    """Convert tier (1-6) to a 0-1 authority score."""
    weights = {1: 1.00, 2: 0.85, 3: 0.65, 4: 0.50, 5: 0.30, 6: 0.00}
    return weights.get(tier, 0.0)


def cap_foreign_tier(tier: int, jurisdiction: Jurisdiction, run_jurisdiction: Jurisdiction) -> int:
    """Foreign sources in an IN run are capped at tier 4."""
    if run_jurisdiction in (Jurisdiction.IN, Jurisdiction.IN_STATE):
        if jurisdiction not in (Jurisdiction.IN, Jurisdiction.IN_STATE):
            return max(tier, 4)
    return tier