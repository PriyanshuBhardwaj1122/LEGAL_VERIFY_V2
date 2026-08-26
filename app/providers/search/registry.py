"""Provider registry — maps QueryIntent to ordered provider list.

Provider routing is deterministic code, never LLM-decided.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.common import QueryIntent

from .base import SearchProvider
from .indiankanoon import IndianKanoonProvider
from .perplexity import PerplexityProvider
from .serpapi import SerpApiProvider
from .stubs import (
    ExaStubProvider,
    IndiaCodeStubProvider,
    SemanticScholarStubProvider,
)
from .tavily import TavilyProvider

log = get_logger()

# Provider routing table from §5.1
# Order matters: first = preferred, rest = fallback
_ROUTING_TABLE: dict[QueryIntent, list[str]] = {
    QueryIntent.STATUTE_TEXT: ["indiacode", "tavily", "serpapi"],
    QueryIntent.CASE_LAW: ["indiankanoon", "tavily", "serpapi"],
    QueryIntent.RELATED_PRECEDENT: ["exa", "indiankanoon", "serpapi"],
    QueryIntent.RECENT_DEVELOPMENT: ["perplexity", "tavily", "serpapi"],
    QueryIntent.REGULATORY_ACTION: ["tavily", "serpapi", "exa"],
    QueryIntent.ACADEMIC_COMMENTARY: ["semantic_scholar", "exa", "tavily", "serpapi"],
    QueryIntent.COUNTER_VIEW: ["exa", "perplexity", "tavily", "serpapi"],
    QueryIntent.COMPARATIVE: ["exa", "tavily", "perplexity", "serpapi"],
    QueryIntent.BACKGROUND: ["tavily", "serpapi"],
}


class ProviderRegistry:
    """Singleton registry of all search providers."""

    def __init__(self):
        settings = get_settings()
        self._providers: dict[str, SearchProvider] = {}

        # Always register providers that have keys
        if settings.tavily_api_key:
            self._providers["tavily"] = TavilyProvider()
        if settings.perplexity_api_key:
            self._providers["perplexity"] = PerplexityProvider()
        if settings.indiankanoon_api_token:
            self._providers["indiankanoon"] = IndianKanoonProvider()
        if settings.serpapi_api_key:
            self._providers["serpapi"] = SerpApiProvider()

        # Stubs for providers without keys yet
        if "exa" not in self._providers:
            self._providers["exa"] = ExaStubProvider()
        if "semantic_scholar" not in self._providers:
            self._providers["semantic_scholar"] = SemanticScholarStubProvider()
        if "indiacode" not in self._providers:
            self._providers["indiacode"] = IndiaCodeStubProvider()

        log.info(
            "provider_registry_init",
            available=[
                n for n, p in self._providers.items()
                if not type(p).__name__.endswith("StubProvider")
            ],
            stubbed=[
                n for n, p in self._providers.items()
                if type(p).__name__.endswith("StubProvider")
            ],
        )

    def get_provider(self, name: str) -> SearchProvider | None:
        return self._providers.get(name)

    def resolve_providers(self, intent: QueryIntent) -> list[str]:
        """Return the ordered list of provider names for a given intent,
        filtered to only those that are actually available (not stubbed)."""
        candidates = _ROUTING_TABLE.get(intent, ["tavily"])
        available = []
        for name in candidates:
            provider = self._providers.get(name)
            if provider and not type(provider).__name__.endswith("StubProvider"):
                available.append(name)

        # If nothing is available for this intent, fall back to whatever we have
        if not available:
            available = [
                n for n, p in self._providers.items()
                if not type(p).__name__.endswith("StubProvider")
            ]

        return available

    def all_available_names(self) -> list[str]:
        return [
            n for n, p in self._providers.items()
            if not type(p).__name__.endswith("StubProvider")
        ]


_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
