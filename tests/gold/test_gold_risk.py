"""Tests for GoldRiskManager — position sizing and daily guardrails."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from shree.config.gold import GoldRiskConfig
from shree.execution.gold.risk import DailyState, GoldRiskManager
from shree.risk.trade_math import get_contract_spec


def _mgc_spec():
    return get_contract_spec("MGC")


def _mgr(
    max_risk: float = 100.0,
    daily_limit: float = 300.0,
    max_trades: int = 5,
    max_consec: int = 3,
    cooldown_min: int = 30,
    hard_cap: int = 2,
) -> GoldRiskManager:
    cfg = GoldRiskConfig(
        max_risk_per_trade_usd=max_risk,
        daily_loss_limit_usd=daily_limit,
        max_trades_per_day=max_trades,
        max_consecutive_losses=max_consec,
        post_loss_cooldown_minutes=cooldown_min,
        max_contracts_hard_cap=hard_cap,
    )
    return GoldRiskManager(cfg, _mgc_spec())


_NOW = datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc)


class TestSizePosition:
    def test_basic_sizing(self) -> None:
        """$100 risk / ($10 risk/contract with 1-pt stop) = 10 contracts → capped at hard_cap=2."""
        mgr = _mgr(max_risk=100.0, hard_cap=2)
        # MGC: point_value=10, stop=1pt → risk/contract=$10
        result = mgr.size_position(1.0, DailyState(), now_utc=_NOW)
        assert result.approved
        assert result.contracts == 2   # Capped by hard_cap

    def test_minimum_one_contract(self) -> None:
        """Even if risk math gives 0.2 contracts, minimum is 1 — when within daily limit."""
        mgr = _mgr(max_risk=5.0, hard_cap=10, daily_limit=500.0)
        # MGC stop=1pt → risk/contract=1×10=$10; max_risk=5 → floor(5/10)=0 → min 1
        result = mgr.size_position(1.0, DailyState(), now_utc=_NOW)
        assert result.approved
        assert result.contracts == 1

    def test_daily_loss_limit_blocks_entry(self) -> None:
        mgr = _mgr(daily_limit=300.0)
        daily = DailyState(realized_pnl=-300.0)
        result = mgr.size_position(2.0, daily, now_utc=_NOW)
        assert not result.approved
        assert "daily loss limit" in result.reason.lower()

    def test_max_trades_blocks_entry(self) -> None:
        mgr = _mgr(max_trades=5)
        daily = DailyState(trades_today=5)
        result = mgr.size_position(2.0, daily, now_utc=_NOW)
        assert not result.approved
        assert "max trades" in result.reason.lower()

    def test_cooldown_blocks_entry(self) -> None:
        future_cooldown = _NOW + timedelta(minutes=15)
        mgr = _mgr()
        daily = DailyState(cooldown_until=future_cooldown)
        result = mgr.size_position(2.0, daily, now_utc=_NOW)
        assert not result.approved
        assert "cooldown" in result.reason.lower()

    def test_expired_cooldown_does_not_block(self) -> None:
        past_cooldown = _NOW - timedelta(minutes=5)
        mgr = _mgr()
        daily = DailyState(cooldown_until=past_cooldown)
        result = mgr.size_position(2.0, daily, now_utc=_NOW)
        assert result.approved

    def test_zero_stop_distance_blocked(self) -> None:
        mgr = _mgr()
        result = mgr.size_position(0.0, DailyState(), now_utc=_NOW)
        assert not result.approved

    def test_total_risk_does_not_exceed_daily_limit(self) -> None:
        """If remaining daily budget < risk for 1 contract, we reject."""
        mgr = _mgr(max_risk=1000.0, daily_limit=300.0, hard_cap=10)
        # Already lost $295 of $300 limit; remaining = $5
        # MGC stop=10pt → risk/contract=$100 → $100 > $5 → reject
        daily = DailyState(realized_pnl=-295.0)
        result = mgr.size_position(10.0, daily, now_utc=_NOW)
        assert not result.approved


class TestCooldown:
    def test_no_cooldown_below_threshold(self) -> None:
        mgr = _mgr(max_consec=3, cooldown_min=30)
        assert mgr.compute_cooldown_until(2, now_utc=_NOW) is None

    def test_cooldown_at_threshold(self) -> None:
        mgr = _mgr(max_consec=3, cooldown_min=30)
        result = mgr.compute_cooldown_until(3, now_utc=_NOW)
        assert result is not None
        expected = _NOW + timedelta(minutes=30)
        assert abs((result - expected).total_seconds()) < 2

    def test_cooldown_beyond_threshold(self) -> None:
        mgr = _mgr(max_consec=3, cooldown_min=30)
        result = mgr.compute_cooldown_until(5, now_utc=_NOW)
        assert result is not None


class TestStopDistance:
    def test_buy_stop_distance(self) -> None:
        dist = GoldRiskManager.compute_stop_distance(2350.0, 2340.0, "BUY")
        assert abs(dist - 10.0) < 1e-9

    def test_sell_stop_distance(self) -> None:
        dist = GoldRiskManager.compute_stop_distance(2350.0, 2360.0, "SELL")
        assert abs(dist - 10.0) < 1e-9

    def test_negative_distance_clamped(self) -> None:
        # Stop on wrong side — distance cannot be negative
        dist = GoldRiskManager.compute_stop_distance(2350.0, 2360.0, "BUY")
        assert dist == 0.0
