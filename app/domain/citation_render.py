"""Pure, deterministic rendering of a LegalCitation into a human-readable
Indian-legal-style citation string. No LLM involved — this is the same
"deterministic code decides the final form" posture as coverage.py and
grounding.py's binding-strength computation. If the structured fields
needed for a clean render aren't present (unparsed citation), we fall
back to the raw string rather than fabricating structure.
"""

from __future__ import annotations

from app.schemas.citation import LegalCitation


def render_citation(citation: LegalCitation | None) -> str:
    if citation is None:
        return ""

    if not citation.is_parsed:
        return citation.raw

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

    return citation.raw
