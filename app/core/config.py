"""Application configuration via pydantic-settings."""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://legal:legal_dev@localhost:5432/legal_research"
    database_url_sync: str = "postgresql+psycopg://legal:legal_dev@localhost:5432/legal_research"
    checkpointer_dsn: str = (
        "postgresql://legal:legal_dev@localhost:5432/legal_research"
        "?options=-csearch_path%3Dlanggraph"
    )

    # LLM — provider selection
    llm_provider: str = "openai"  # "openai" or "anthropic" — used by the research phase
    # Generation phase (thesis/outline/draft/voice) provider override.
    # Empty string falls back to llm_provider — set this independently
    # to run research on one provider and generation (the reader-facing
    # prose) on another, e.g. GENERATION_LLM_PROVIDER=anthropic while
    # LLM_PROVIDER stays "openai".
    generation_llm_provider: str = ""

    # One model per provider — used for every call made through that
    # provider's client, in whichever phase (research or generation)
    # selected it. There is no per-node (planner/extractor/draft/voice)
    # model override anywhere in the codebase today.
    anthropic_api_key: str = ""
    # Required only for identity-linked API keys, which the API rejects
    # without an `anthropic-workspace-id` header. Leave empty for a
    # normal workspace-scoped key.
    anthropic_workspace_id: str = ""
    claude_model: str = "claude-sonnet-5"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Search providers
    tavily_api_key: str = ""
    perplexity_api_key: str = ""
    exa_api_key: str = ""
    semantic_scholar_api_key: str = ""
    indiankanoon_api_token: str = ""
    serpapi_api_key: str = ""

    # Provider concurrency
    tavily_concurrency: int = 8
    exa_concurrency: int = 8
    perplexity_concurrency: int = 4
    indiankanoon_concurrency: int = 2
    serpapi_concurrency: int = 6
    government_concurrency: int = 2
    semantic_scholar_rate_per_sec: float = 1.0

    # Budget
    default_budget_inr: Decimal = Decimal("120.00")

    # Fetch
    fetch_timeout_sec: int = 30
    max_pdf_size_mb: int = 40
    min_text_length: int = 400  # below this, escalate extraction method

    # Logging
    log_level: str = "INFO"
    debug_payloads: bool = False

    # INR cost per unit for each provider (for budget tracking)
    cost_tavily_search: Decimal = Decimal("0.50")
    cost_perplexity_per_1k_tokens: Decimal = Decimal("0.15")
    cost_indiankanoon_search: Decimal = Decimal("0.50")
    cost_indiankanoon_doc: Decimal = Decimal("0.20")
    cost_indiankanoon_docmeta: Decimal = Decimal("0.02")
    cost_serpapi_search: Decimal = Decimal("0.40")
    cost_anthropic_per_1k_input: Decimal = Decimal("0.25")
    cost_anthropic_per_1k_output: Decimal = Decimal("1.25")
    cost_openai_per_1k_input: Decimal = Decimal("0.20")
    cost_openai_per_1k_output: Decimal = Decimal("0.80")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
