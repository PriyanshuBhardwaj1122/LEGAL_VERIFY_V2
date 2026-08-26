"""Stub search providers — satisfy the protocol, return empty results.

Wire in real implementations when API keys become available.
"""

from __future__ import annotations

from decimal import Decimal

from app.core.logging import get_logger
from app.schemas.plan import SubQuery
from app.schemas.source import RawResult

log = get_logger()


class ExaStubProvider:
    name = "exa"

    def estimate_cost(self, q: SubQuery, *, limit: int = 10) -> Decimal:
        return Decimal("0")

    async def search(self, q: SubQuery, *, limit: int = 10) -> list[RawResult]:
        log.debug("exa_stub_called", query_id=q.query_id)
        return []


class SemanticScholarStubProvider:
    name = "semantic_scholar"

    def estimate_cost(self, q: SubQuery, *, limit: int = 10) -> Decimal:
        return Decimal("0")

    async def search(self, q: SubQuery, *, limit: int = 10) -> list[RawResult]:
        log.debug("semantic_scholar_stub_called", query_id=q.query_id)
        return []


class IndiaCodeStubProvider:
    name = "indiacode"

    def estimate_cost(self, q: SubQuery, *, limit: int = 10) -> Decimal:
        return Decimal("0")

    async def search(self, q: SubQuery, *, limit: int = 10) -> list[RawResult]:
        log.debug("indiacode_stub_called", query_id=q.query_id)
        return []
