"""Claude (Anthropic) LLM provider with tool-use structured output."""

from __future__ import annotations

import json
import time
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger()
T = TypeVar("T", bound=BaseModel)


def _pydantic_to_tool_schema(schema_cls: type[BaseModel], tool_name: str) -> dict:
    """Convert a Pydantic model to an Anthropic tool definition."""
    json_schema = schema_cls.model_json_schema()

    # Remove $defs and resolve refs inline isn't needed for Anthropic —
    # they handle $defs natively. But we do need to strip 'title' from
    # the top level as Anthropic doesn't want it in input_schema.
    json_schema.pop("title", None)

    return {
        "name": tool_name,
        "description": f"Emit structured output as {schema_cls.__name__}",
        "input_schema": json_schema,
    }


class ClaudeLLM:
    """Anthropic Claude client with forced tool-use for structured output."""

    def __init__(self):
        settings = get_settings()
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.default_model = settings.claude_planner_model

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
        """Call Claude with forced tool use, return parsed Pydantic model + usage."""
        model = model or self.default_model
        tool_def = _pydantic_to_tool_schema(output_schema, tool_name)

        start = time.monotonic()
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[tool_def],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except anthropic.APIError as e:
            log.error("claude_api_error", error=str(e), model=model)
            raise

        latency_ms = int((time.monotonic() - start) * 1000)

        # Extract the tool use block
        tool_block = None
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                tool_block = block
                break

        if tool_block is None:
            raise ValueError(
                f"Claude did not return a tool_use block for {tool_name}. "
                f"Got: {[b.type for b in response.content]}"
            )

        # Parse and validate through Pydantic
        parsed = output_schema.model_validate(tool_block.input)

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": model,
            "latency_ms": latency_ms,
        }

        log.debug(
            "claude_call",
            tool=tool_name,
            model=model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            latency_ms=latency_ms,
        )

        return parsed, usage
