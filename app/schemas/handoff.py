"""Handoff contract — what the Generation phase consumes."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel

from .common import Jurisdiction
from .evidence import Evidence
from .gaps import Gap
from .source import Source


class EvidencePackage(BaseModel):
    run_id: str
    topic: str
    jurisdiction: Jurisdiction
    as_of_date: date
    legal_issues: list[str]
    evidence: list[Evidence]
    sources: list[Source]
    unresolved_gaps: list[Gap]
    coverage: dict[str, int]
    research_verdict: Literal["complete", "exhausted"]
