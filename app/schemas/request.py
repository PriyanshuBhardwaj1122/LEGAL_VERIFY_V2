"""Inbound request models."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from .common import Jurisdiction


class ArticleConfig(BaseModel):
    article_type: Literal["blog", "explainer", "client_alert", "white_paper", "journal_note"]
    target_words: int = Field(1500, ge=400, le=12000)
    audience: Literal["practitioner", "student", "general", "policy"] = "practitioner"
    comparative_jurisdictions: list[Jurisdiction] = []


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=8, max_length=400)
    jurisdiction: Jurisdiction = Jurisdiction.IN
    state: str | None = None
    practice_area: str | None = None
    as_of_date: date = Field(default_factory=date.today)
    article_config: ArticleConfig
    max_research_loops: int = Field(2, ge=0, le=3)
    budget_inr: Decimal = Decimal("120.00")
    allow_paid_sources: bool = True
