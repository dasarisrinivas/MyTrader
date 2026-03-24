"""Tests for GoldContractFactory.

All tests run *offline* (no IB connection required).
"""
from __future__ import annotations

import pytest
from ib_insync import Future

from shree.config.gold import GoldStrategyConfig
from shree.execution.gold.contract import GoldContractFactory


def _mgc_cfg() -> GoldStrategyConfig:
    return GoldStrategyConfig(enabled=True, symbol="MGC", allow_gc=False)


def _gc_cfg() -> GoldStrategyConfig:
    return GoldStrategyConfig(enabled=True, symbol="GC", allow_gc=True)


class TestBuildUnqualified:
    def test_mgc_returns_future(self) -> None:
        factory = GoldContractFactory(_mgc_cfg())
        contract = factory.build_unqualified()
        assert isinstance(contract, Future)
        assert contract.symbol == "MGC"
        assert contract.exchange == "COMEX"
        assert contract.currency == "USD"

    def test_gc_allowed_when_flag_set(self) -> None:
        factory = GoldContractFactory(_gc_cfg())
        contract = factory.build_unqualified()
        assert contract.symbol == "GC"

    def test_gc_blocked_without_allow_flag(self) -> None:
        cfg = GoldStrategyConfig(enabled=True, symbol="GC", allow_gc=False)
        factory = GoldContractFactory(cfg)
        with pytest.raises(ValueError, match="allow_gc"):
            factory.build_unqualified()

    def test_no_expiry_in_unqualified(self) -> None:
        """Leave expiry blank so IB selects front-month automatically."""
        factory = GoldContractFactory(_mgc_cfg())
        contract = factory.build_unqualified()
        assert not contract.lastTradeDateOrContractMonth   # Empty string or None is fine

    def test_get_qualified_requires_ib(self) -> None:
        factory = GoldContractFactory(_mgc_cfg(), ib=None)
        with pytest.raises(RuntimeError, match="IB instance required"):
            import asyncio
            asyncio.run(factory.get_qualified_contract())

    def test_invalidate_cache(self) -> None:
        factory = GoldContractFactory(_mgc_cfg())
        # Can call invalidate without a cached contract
        factory.invalidate_cache()
        assert factory._cached_contract is None
