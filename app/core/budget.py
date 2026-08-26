"""BudgetGuard — reserve-before-call cost control.

Every outbound call must reserve before executing and commit after.
If the reservation would cross the ceiling, the call is skipped and the
node degrades to a partial result rather than crashing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from app.core.logging import get_logger

log = get_logger()


class BudgetGuard:
    def __init__(self, run_id: str, ceiling_inr: Decimal, db_session_factory=None):
        self.run_id = run_id
        self.ceiling_inr = ceiling_inr
        self._spent = Decimal("0")
        self._reserved = Decimal("0")
        self._lock = asyncio.Lock()
        self._db_session_factory = db_session_factory

    @property
    def spent(self) -> Decimal:
        return self._spent

    @property
    def remaining(self) -> Decimal:
        return max(Decimal("0"), self.ceiling_inr - self._spent - self._reserved)

    async def reserve(self, provider: str, operation: str, estimated: Decimal) -> bool:
        """Return False when this call would cross the ceiling.
        The caller should degrade gracefully."""
        async with self._lock:
            if self._spent + self._reserved + estimated > self.ceiling_inr:
                log.warning(
                    "budget_refused",
                    provider=provider,
                    operation=operation,
                    estimated=float(estimated),
                    remaining=float(self.remaining),
                )
                return False
            self._reserved += estimated
            return True

    async def commit(
        self,
        *,
        provider: str,
        operation: str,
        actual: Decimal,
        node: str = "",
        loop_index: int = 0,
        units: float | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        latency_ms: int | None = None,
        http_status: int | None = None,
        error: str | None = None,
    ) -> None:
        """Record actual cost and release reservation."""
        async with self._lock:
            self._reserved = max(Decimal("0"), self._reserved - actual)
            self._spent += actual

        # Persist to api_call_log if we have a DB session factory
        if self._db_session_factory is not None:
            try:
                from app.db.models import ApiCallLog

                async with self._db_session_factory() as session:
                    row = ApiCallLog(
                        run_id=self.run_id,
                        node=node or "unknown",
                        provider=provider,
                        operation=operation,
                        loop_index=loop_index,
                        cost_inr=actual,
                        units=units,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        latency_ms=latency_ms,
                        http_status=http_status,
                        error=error,
                        created_at=datetime.now(timezone.utc),
                    )
                    session.add(row)
                    await session.commit()
            except Exception:
                log.exception("budget_commit_db_error", provider=provider)

        log.debug(
            "budget_committed",
            provider=provider,
            operation=operation,
            actual=float(actual),
            total_spent=float(self._spent),
            remaining=float(self.remaining),
        )

    async def release_reservation(self, estimated: Decimal) -> None:
        """Release a reservation that was never used (e.g. call was skipped)."""
        async with self._lock:
            self._reserved = max(Decimal("0"), self._reserved - estimated)
