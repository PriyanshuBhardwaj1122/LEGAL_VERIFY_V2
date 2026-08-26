"""Typed exception hierarchy."""

from __future__ import annotations


class ResearchError(Exception):
    """Base for all pipeline errors."""


class BudgetExhaustedError(ResearchError):
    """Hard budget ceiling reached — run should degrade, not crash."""

    def __init__(self, provider: str, requested: float, remaining: float):
        self.provider = provider
        self.requested = requested
        self.remaining = remaining
        super().__init__(
            f"Budget exhausted: {provider} requested {requested:.2f} INR, "
            f"only {remaining:.2f} remaining"
        )


class ProviderError(ResearchError):
    """A search or legal-source provider returned a non-retryable error."""

    def __init__(self, provider: str, detail: str, status_code: int | None = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider}: {detail} (HTTP {status_code})")


class ProviderRateLimitError(ProviderError):
    """429 — retryable."""

    def __init__(self, provider: str, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(provider, "rate limited", 429)


class FetchError(ResearchError):
    """Failed to download or extract text from a source URL."""

    def __init__(self, url: str, detail: str):
        self.url = url
        super().__init__(f"Fetch failed for {url}: {detail}")


class SchemaViolationError(ResearchError):
    """LLM output did not conform to the forced tool schema after retries."""

    def __init__(self, node: str, detail: str):
        self.node = node
        super().__init__(f"Schema violation in {node}: {detail}")


class PlannerFallbackError(ResearchError):
    """Planner produced fewer than 4 usable queries — falling back to template."""
