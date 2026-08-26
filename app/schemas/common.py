"""Shared enums and small types used across all schema modules."""

from __future__ import annotations

from enum import StrEnum


class Jurisdiction(StrEnum):
    IN = "IN"
    IN_STATE = "IN_STATE"
    UK = "UK"
    US = "US"
    EU = "EU"
    SG = "SG"
    INTL = "INTL"


class SourceType(StrEnum):
    JUDGMENT = "judgment"
    STATUTE = "statute"
    SUBORDINATE_LEGISLATION = "subordinate_legislation"
    GAZETTE_NOTIFICATION = "gazette_notification"
    REGULATOR_CIRCULAR = "regulator_circular"
    TRIBUNAL_ORDER = "tribunal_order"
    BILL_OR_DRAFT = "bill_or_draft"
    COMMITTEE_REPORT = "committee_report"
    LAW_COMMISSION_REPORT = "law_commission_report"
    ACADEMIC = "academic"
    LEGAL_NEWS = "legal_news"
    FIRM_COMMENTARY = "firm_commentary"
    OTHER = "other"


class CourtLevel(StrEnum):
    SUPREME_COURT = "supreme_court"
    HIGH_COURT = "high_court"
    TRIBUNAL_APPELLATE = "tribunal_appellate"
    TRIBUNAL = "tribunal"
    DISTRICT = "district"
    FOREIGN = "foreign"
    NONE = "none"


class QueryIntent(StrEnum):
    STATUTE_TEXT = "statute_text"
    CASE_LAW = "case_law"
    RELATED_PRECEDENT = "related_precedent"
    RECENT_DEVELOPMENT = "recent_development"
    REGULATORY_ACTION = "regulatory_action"
    ACADEMIC_COMMENTARY = "academic_commentary"
    COUNTER_VIEW = "counter_view"
    COMPARATIVE = "comparative"
    BACKGROUND = "background"
