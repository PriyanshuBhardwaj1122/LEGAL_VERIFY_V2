"""StructuredLLM protocol — all LLM calls go through this."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class StructuredLLM(Protocol):
    """Protocol for making structured LLM calls with tool-use forcing."""

    async def generate(
        self,
        *,
        system: str,
        user: str,
        output_schema: type[T],
        tool_name: str,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[T, dict[str, Any]]:
        """Call the LLM with a forced tool-use schema.

        Returns:
            tuple of (parsed output, usage dict with keys:
                input_tokens, output_tokens, model, latency_ms)
        """
        ...


def _llm_for_provider(provider: str) -> "StructuredLLM":
    if provider == "anthropic":
        from app.providers.llm.claude import ClaudeLLM

        return ClaudeLLM()

    # default: openai
    from app.providers.llm.openai_llm import OpenAILLM

    return OpenAILLM()


def get_llm() -> "StructuredLLM":
    """Factory: returns the configured LLM provider (openai | anthropic)
    for the research phase (planner/search/evaluator/extractor/gap_check).

    Nodes should call this instead of importing a concrete provider class
    directly, so switching providers is a one-line config change.
    """
    from app.core.config import get_settings

    settings = get_settings()
    return _llm_for_provider(settings.llm_provider)


def get_generation_llm() -> "StructuredLLM":
    """Factory: returns the LLM provider for the Generation phase
    (thesis/outline/draft/voice) specifically. Falls back to
    settings.llm_provider when settings.generation_llm_provider is unset,
    so research and generation can run on different providers — e.g.
    the cheaper/proven provider for structured extraction, a different
    one for the reader-facing prose where voice actually matters."""
    from app.core.config import get_settings

    settings = get_settings()
    return _llm_for_provider(settings.generation_llm_provider or settings.llm_provider)