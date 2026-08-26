"""Legal citation model — explicit structure rather than a free string."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class LegalCitation(BaseModel):
    raw: str

    kind: str  # case | statute | regulation | circular | gazette | academic | other

    # Case fields
    case_name: str | None = None
    neutral_citation: str | None = None
    reporter_citation: str | None = None
    scc_online: str | None = None
    court: str | None = None
    decided_on: date | None = None
    case_number: str | None = None

    # Statute / subordinate legislation
    act_name: str | None = None
    act_year: int | None = None
    section: str | None = None

    # Regulator circular
    circular_number: str | None = None
    issued_on: date | None = None

    # Academic
    doi: str | None = None

    is_parsed: bool = False
