"""Shared retry-with-backoff for outbound calls.

This logic lived inside search_worker and was never available to the
fetch path, so a 429 on a court website — the most retryable failure
there is — was treated exactly like a permanent 403: one attempt, log,
give up. Judgments were being dropped from runs for a condition that
resolves by waiting a second.

Honours Retry-After when the server sends it, capped, so a hostile or
mistaken header can't stall a run.
"""

from __future__ import annotations

import asyncio
import random
import ssl
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Awaitable, Callable, TypeVar

from app.core.logging import get_logger

log = get_logger()
T = TypeVar("T")

MAX_RETRIES = 3
BASE_BACKOFF_SEC = 1.0
# A server may ask us to wait minutes. Honour the signal, not the number.
MAX_RETRY_AFTER_SEC = 30.0


def extract_status_code(e: Exception) -> int | None:
    """Best-effort status code extraction across httpx, openai, tavily,
    and other SDK exception shapes — they don't share a base class."""
    for attr in ("status_code", "status"):
        code = getattr(e, attr, None)
        if isinstance(code, int):
            return code
    response = getattr(e, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _is_tls_failure(e: BaseException | None, depth: int = 0) -> bool:
    """A certificate that doesn't verify won't verify a second later.
    Walks __cause__/__context__ because httpx wraps the ssl error."""
    while e is not None and depth < 5:
        if isinstance(e, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in str(e):
            return True
        e = e.__cause__ or e.__context__
        depth += 1
    return False


def is_retryable(e: Exception) -> bool:
    """429 and 5xx are retryable. Timeouts are retryable. Everything
    else (4xx auth/validation errors, unknown shapes) is not — retrying
    a malformed request just burns budget for the same failure."""
    if _is_tls_failure(e):
        # A hostname mismatch or expired cert is a property of the site,
        # not a transient condition — retrying just triples the delay
        # before recording the same failure.
        return False
    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
        return True
    status = extract_status_code(e)
    if status is None:
        # Unknown error shape — retry once defensively, callers still
        # bound this via MAX_RETRIES.
        return True
    return status == 429 or 500 <= status < 600


def retry_after_seconds(e: Exception) -> float | None:
    """Parse a Retry-After header off an exception's response, in either
    delta-seconds or HTTP-date form. Returns None when absent/unusable."""
    response = getattr(e, "response", None)
    if response is None:
        return None
    raw = None
    try:
        raw = response.headers.get("retry-after")
    except Exception:
        return None
    if not raw:
        return None

    raw = raw.strip()
    if raw.isdigit():
        return min(float(raw), MAX_RETRY_AFTER_SEC)
    try:
        when = parsedate_to_datetime(raw)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        delta = (when - datetime.now(timezone.utc)).total_seconds()
        return min(max(delta, 0.0), MAX_RETRY_AFTER_SEC) if delta > 0 else None
    except Exception:
        return None


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    what: str,
    max_retries: int = MAX_RETRIES,
    base_backoff: float = BASE_BACKOFF_SEC,
) -> T:
    """Await `fn()`, retrying retryable failures with jittered exponential
    backoff. `what` is a short label used only for logging."""
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception as e:
            last_exc = e
            if not is_retryable(e) or attempt == max_retries - 1:
                log.warning(
                    "retry_terminal",
                    what=what,
                    error=str(e),
                    status_code=extract_status_code(e),
                    attempt=attempt + 1,
                )
                raise

            server_wait = retry_after_seconds(e)
            wait = (
                server_wait
                if server_wait is not None
                else base_backoff * (2**attempt) + random.uniform(0, 0.5)
            )
            log.warning(
                "retry_backoff",
                what=what,
                status_code=extract_status_code(e),
                attempt=attempt + 1,
                wait_sec=round(wait, 2),
                honored_retry_after=server_wait is not None,
            )
            await asyncio.sleep(wait)

    if last_exc:
        raise last_exc
    raise RuntimeError(f"retry_async({what}) exhausted with no result")
