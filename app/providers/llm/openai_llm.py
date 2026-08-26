"""OpenAI LLM provider with forced function-calling structured output.

Implements the same StructuredLLM protocol as ClaudeLLM so nodes can
swap providers without changing call sites.
"""

from __future__ import annotations

import json
import time
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger()
T = TypeVar("T", bound=BaseModel)


def _strictify(node: Any) -> None:
    """Recursively rewrite a JSON schema node in place to satisfy OpenAI's
    strict structured-output requirements:
      - every object gets additionalProperties=False
      - every object's `required` list includes ALL of its properties,
        even ones with a Pydantic default — strict mode has no concept
        of "optional", only "nullable"; the model must always emit a
        key, using null for fields it would otherwise have omitted.
      - the `default` keyword is stripped everywhere — it's not part of
        the strict-mode-supported keyword set and OpenAI rejects it.

    Without this, `additionalProperties`/`required` are advisory (as is
    an enum's `$ref`) and the model can emit anything, including a value
    from a totally different enum that happens to share a $defs entry —
    which is exactly how a SourceType value ("committee_report") ended
    up in a QueryIntent-typed field before this was added.
    """
    if isinstance(node, dict):
        node.pop("default", None)

        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())

        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for item in node:
            _strictify(item)


def _pydantic_to_function_schema(schema_cls: type[BaseModel], tool_name: str) -> dict:
    """Convert a Pydantic model to a strict OpenAI function-tool definition."""
    json_schema = schema_cls.model_json_schema()
    json_schema.pop("title", None)
    _strictify(json_schema)

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": f"Emit structured output as {schema_cls.__name__}",
            "parameters": json_schema,
            "strict": True,
        },
    }


class OpenAILLM:
    """OpenAI client with forced function-calling for structured output."""

    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.default_model = settings.openai_planner_model

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
        """Call OpenAI with forced function-calling, return parsed Pydantic model + usage."""
        model = model or self.default_model
        tool_def = _pydantic_to_function_schema(output_schema, tool_name)

        start = time.monotonic()
        try:
            response = await self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tools=[tool_def],
                tool_choice={"type": "function", "function": {"name": tool_name}},
            )
        except Exception as e:
            log.error("openai_api_error", error=str(e), model=model)
            raise

        latency_ms = int((time.monotonic() - start) * 1000)

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        if not tool_calls:
            raise ValueError(
                f"OpenAI did not return a tool call for {tool_name}. "
                f"finish_reason={choice.finish_reason}"
            )

        call = tool_calls[0]
        if call.function.name != tool_name:
            raise ValueError(
                f"OpenAI returned tool call for '{call.function.name}', expected '{tool_name}'"
            )

        raw_args = json.loads(call.function.arguments)
        parsed = output_schema.model_validate(raw_args)

        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "model": model,
            "latency_ms": latency_ms,
        }

        log.debug(
            "openai_call",
            tool=tool_name,
            model=model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            latency_ms=latency_ms,
        )

        return parsed, usage
