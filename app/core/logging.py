"""Structured logging with run_id in contextvars."""

from __future__ import annotations

import contextvars
import logging
import sys

import structlog

# Context variables bound per-run so every log line carries them.
ctx_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="")
ctx_node: contextvars.ContextVar[str] = contextvars.ContextVar("node", default="")


def _add_context(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict,
) -> dict:
    run_id = ctx_run_id.get("")
    if run_id:
        event_dict["run_id"] = run_id
    node = ctx_node.get("")
    if node:
        event_dict["node"] = node
    return event_dict


def setup_logging(log_level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_context,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(**kwargs) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(**kwargs)
