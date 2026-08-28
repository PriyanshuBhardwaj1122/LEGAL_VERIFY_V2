"""Claude (Anthropic) LLM provider with tool-use structured output."""

from __future__ import annotations

import json
import time
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger()
T = TypeVar("T", bound=BaseModel)

# Repair round-trips allowed when the model returns a tool call that
# doesn't validate. One is enough in practice — handed the specific
# error, the model almost always corrects on the next try; a second
# failure usually means the schema itself is wrong.
_MAX_SCHEMA_REPAIR_ATTEMPTS = 1

# Sampling parameters (temperature/top_p/top_k) were removed on the
# Claude 5 generation and on Opus 4.7/4.8 — sending temperature to one
# of these returns a 400 ("`temperature` is deprecated for this model").
# They are still accepted on 4.6 and older, so match on the families
# that reject it rather than assuming either way for unknown ids.
_NO_TEMPERATURE_PREFIXES = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
)


def _supports_temperature(model: str) -> bool:
    return not model.startswith(_NO_TEMPERATURE_PREFIXES)


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
        # An identity-linked API key is rejected without this header;
        # a normal workspace-scoped key doesn't need it, so only send
        # it when it's actually configured.
        default_headers = (
            {"anthropic-workspace-id": settings.anthropic_workspace_id}
            if settings.anthropic_workspace_id
            else None
        )
        self.client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            default_headers=default_headers,
        )
        self.default_model = settings.claude_model

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
        """Call Claude with forced tool use, return parsed Pydantic model + usage.

        Unlike OpenAI's strict structured-output mode (see openai_llm's
        _strictify), Anthropic treats a tool schema's `required` list as
        advisory — the model can and occasionally does omit a required
        field. That surfaces as a Pydantic ValidationError on an
        otherwise-fine response, so one repair round-trip is made:
        the invalid call and the validation error are handed back so the
        model can re-emit. Cheaper than losing a run that already paid
        for the whole research phase.
        """
        model = model or self.default_model
        tool_def = _pydantic_to_tool_schema(output_schema, tool_name)

        # Callers still pass temperature (the OpenAI provider honours it,
        # and the drafting node deliberately raises it for voice
        # variation); on models where sampling was removed we drop it
        # rather than 400, and those models pick their own sampling.
        extra: dict[str, Any] = (
            {"temperature": temperature} if _supports_temperature(model) else {}
        )

        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        total_input = 0
        total_output = 0
        start = time.monotonic()

        for attempt in range(_MAX_SCHEMA_REPAIR_ATTEMPTS + 1):
            is_last = attempt == _MAX_SCHEMA_REPAIR_ATTEMPTS
            try:
                response = await self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                    tools=[tool_def],
                    tool_choice={"type": "tool", "name": tool_name},
                    **extra,
                )
            except anthropic.APIError as e:
                log.error("claude_api_error", error=str(e), model=model)
                raise

            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens

            # Thinking is on by default on current Claude models and its
            # tokens come out of max_tokens, so a budget tuned for a
            # non-thinking model can truncate the tool input mid-JSON.
            # Retrying won't help — the budget itself is wrong — so fail
            # immediately and name the real cause.
            if response.stop_reason == "max_tokens":
                log.error(
                    "claude_truncated_at_max_tokens",
                    tool=tool_name,
                    model=model,
                    max_tokens=max_tokens,
                    output_tokens=response.usage.output_tokens,
                )
                raise ValueError(
                    f"Claude hit max_tokens ({max_tokens}) before finishing the "
                    f"{tool_name} tool call, so its output is incomplete. Raise "
                    f"max_tokens for this call — on models with thinking enabled "
                    f"the reasoning tokens share this budget with the response."
                )

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

            try:
                parsed = output_schema.model_validate(tool_block.input)
                break
            except ValidationError as e:
                if is_last:
                    log.error(
                        "claude_schema_repair_exhausted",
                        tool=tool_name,
                        model=model,
                        attempts=attempt + 1,
                        error=str(e),
                    )
                    raise
                log.warning(
                    "claude_schema_invalid_retrying",
                    tool=tool_name,
                    model=model,
                    attempt=attempt + 1,
                    error=str(e),
                )
                # Hand the model its own invalid call plus the specific
                # validation failure. response.content is appended whole
                # so any thinking blocks are echoed back unchanged, as
                # the API requires when continuing on the same model.
                messages.append({"role": "assistant", "content": response.content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_block.id,
                                "is_error": True,
                                "content": (
                                    f"That tool call failed schema validation:\n{e}\n\n"
                                    "Call the tool again with the SAME content, corrected — "
                                    "every required field present, and every field within its "
                                    "documented limits. Do not drop or shorten the substance."
                                ),
                            }
                        ],
                    }
                )

        latency_ms = int((time.monotonic() - start) * 1000)

        usage = {
            "input_tokens": total_input,
            "output_tokens": total_output,
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
