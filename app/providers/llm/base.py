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


def get_llm() -> "StructuredLLM":
    """Factory: returns the configured LLM provider (openai | anthropic).

    Nodes should call this instead of importing a concrete provider class
    directly, so switching providers is a one-line config change.
    """
    from app.core.config import get_settings

    settings = get_settings()

    if settings.llm_provider == "anthropic":
        from app.providers.llm.claude import ClaudeLLM

        return ClaudeLLM()

    # default: openai
    from app.providers.llm.openai_llm import OpenAILLM

    return OpenAILLM()