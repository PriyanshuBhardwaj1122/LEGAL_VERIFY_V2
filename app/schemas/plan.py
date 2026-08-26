"""Research plan — Step 1 output."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from .common import Jurisdiction, QueryIntent, SourceType


class SubQuery(BaseModel):
    query_id: str = ""  # computed post-LLM: sha1(normalized query + provider + intent)[:16]
    query_text: str = Field(max_length=350)
    intent: QueryIntent
    rationale: str
    target_source_types: list[SourceType]
    min_authority_tier: int = 5
    jurisdiction: Jurisdiction
    date_from: date | None = None
    date_to: date | None = None
    providers: list[str] = []  # resolved by registry, not the LLM
    priority: int = Field(3, ge=1, le=5)


class ResearchPlan(BaseModel):
    plan_id: str = ""
    run_id: str = ""
    loop_index: int = 0
    topic_restated: str
    legal_issues: list[str]
    key_instruments: list[str]
    sub_queries: list[SubQuery] = Field(min_length=4, max_length=12)
    excluded_directions: list[str] = []
