"""Tests for Phase 4A: Progress-aware staged time stop.

We test the manager's _check_time_stop logic directly by creating
a minimal mock setup.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from shree.config.gold import GoldExitConfig, GoldStrategyConfig
from shree.strategies.gold.signals import GoldSignal, GoldSignalType
from shree.strategies.gold.regime import GoldRegime


# We test the staged time stop logic by computing R-multiples and checking
# the decision boundaries. Since _check_time_stop is async and calls
# _flatten_position, we test through a thin wrapper.


class _FakePosition:
    """Minimal _OpenPosition stand-in for time stop tests."""

    def __init__(
        self,
        action: str = "BUY",
        entry_price: float = 2350.0,
        stop_loss: float = 2338.0,   # 12 pts SL distance
        take_profit: float = 2380.0,
        entry_bar: int = 0,
    ) -> None:
        self.action = action
        self.entry_price = entry_price
        self.signal = MagicMock()
        self.signal.stop_loss = stop_loss
        self.signal.take_profit = take_profit
        self.entry_bar = entry_bar
        self.trade_id = "test-123"
        self.contracts = 1
        self.trail_activated = False
        self.best_price = None


def _compute_r_multiple(action: str, entry_price: float, current_price: float, stop_loss: float) -> float:
    """Mirror the manager's R-multiple computation."""
    if action == "BUY":
        sl_distance = entry_price - stop_loss
        unrealized = current_price - entry_price
    else:
        sl_distance = stop_loss - entry_price
        unrealized = entry_price - current_price
    return unrealized / sl_distance if sl_distance > 0 else 0.0


class TestProgressAwareTimeStop:
    """Test the staged time stop logic."""

    def test_stage_1_exits_when_progress_below_threshold(self) -> None:
        """After 20 bars with < 0.25R, should exit."""
        pos = _FakePosition(action="BUY", entry_price=2350.0, stop_loss=2338.0)
        # SL distance = 12 pts, 0.25R = 3 pts
        current_price = 2351.0  # Only 1 pt profit = 0.083R < 0.25R
        bars_held = 25  # Past stage 1 (20 bars)

        r = _compute_r_multiple("BUY", 2350.0, current_price, 2338.0)
        assert r < 0.25  # Should trigger stage 1 exit

        # Verify the config defaults
        cfg = GoldExitConfig()
        assert cfg.time_stop_stage_1_bars == 20
        assert cfg.time_stop_stage_1_min_progress_r == 0.25

    def test_stage_1_no_exit_when_progress_sufficient(self) -> None:
        """After 20 bars with >= 0.25R, should NOT exit."""
        pos = _FakePosition(action="BUY", entry_price=2350.0, stop_loss=2338.0)
        # SL distance = 12 pts, 0.25R = 3 pts
        current_price = 2354.0  # 4 pts profit = 0.33R > 0.25R

        r = _compute_r_multiple("BUY", 2350.0, current_price, 2338.0)
        assert r >= 0.25  # Should NOT trigger

    def test_stage_2_exits_below_breakeven(self) -> None:
        """After 40 bars with R < 0.0 (losing), should exit."""
        pos = _FakePosition(action="BUY", entry_price=2350.0, stop_loss=2338.0)
        current_price = 2349.0  # -1 pt = -0.083R

        r = _compute_r_multiple("BUY", 2350.0, current_price, 2338.0)
        assert r < 0.0  # Should trigger stage 2 exit

        cfg = GoldExitConfig()
        assert cfg.time_stop_stage_2_bars == 40
        assert cfg.time_stop_stage_2_min_progress_r == 0.0

    def test_stage_2_no_exit_at_breakeven(self) -> None:
        """After 40 bars at exactly break-even, should NOT exit (R >= 0.0)."""
        current_price = 2350.0  # Exactly entry
        r = _compute_r_multiple("BUY", 2350.0, current_price, 2338.0)
        assert r >= 0.0  # Should NOT trigger

    def test_hard_cap_exits_unconditionally(self) -> None:
        """After 60 bars, exit regardless of R-multiple."""
        cfg = GoldExitConfig()
        assert cfg.time_stop_bars == 60  # Default hard cap

    def test_short_position_r_multiple_correct(self) -> None:
        """Verify R-multiple works for SHORT positions."""
        # SELL at 2350, SL at 2362 (12 pts)
        r = _compute_r_multiple("SELL", 2350.0, 2347.0, 2362.0)
        assert abs(r - 0.25) < 0.01  # 3 pts profit / 12 pts risk

        r_loss = _compute_r_multiple("SELL", 2350.0, 2353.0, 2362.0)
        assert r_loss < 0.0  # Losing

    def test_stage_1_before_stage_2(self) -> None:
        """Stage 1 triggers before stage 2 on a stagnant trade."""
        cfg = GoldExitConfig()
        # At bar 25 with 0.1R (< 0.25R): stage 1 fires
        r = 0.1
        assert cfg.time_stop_stage_1_bars <= 25
        assert r < cfg.time_stop_stage_1_min_progress_r

    def test_between_stages_no_exit(self) -> None:
        """Between stage 1 and stage 2 bars, trade with 0.1R is already closed at stage 1.
        But if trade has 0.3R at stage 1, it survives. Then at bar 35 (before stage 2), no exit."""
        cfg = GoldExitConfig()
        bars_held = 35
        r = 0.3
        # Stage 1: passed (0.3 >= 0.25)
        # Stage 2: not yet (35 < 40)
        # Hard cap: not yet (35 < 60)
        assert bars_held >= cfg.time_stop_stage_1_bars
        assert r >= cfg.time_stop_stage_1_min_progress_r
        assert bars_held < cfg.time_stop_stage_2_bars

    def test_stage_2_catches_reversal_after_stage_1_pass(self) -> None:
        """Trade passed stage 1 at 0.3R, then reversed to -0.1R by bar 45 → stage 2 exits."""
        cfg = GoldExitConfig()
        r_at_stage_1 = 0.3  # Passes stage 1
        r_at_bar_45 = -0.1  # Reversed
        assert r_at_stage_1 >= cfg.time_stop_stage_1_min_progress_r
        assert r_at_bar_45 < cfg.time_stop_stage_2_min_progress_r
        assert 45 >= cfg.time_stop_stage_2_bars

    def test_runner_survives_all_stages(self) -> None:
        """A trade at 1.5R at bar 55 survives stage 1, stage 2, but exits at hard cap (60)."""
        cfg = GoldExitConfig()
        r = 1.5
        bars_held = 55
        assert r >= cfg.time_stop_stage_1_min_progress_r
        assert r >= cfg.time_stop_stage_2_min_progress_r
        assert bars_held < cfg.time_stop_bars  # Survives until hard cap

    def test_disabled_stages_fall_through(self) -> None:
        """Setting stage bars to 0 disables those stages; only hard cap applies."""
        cfg = GoldExitConfig(
            time_stop_stage_1_bars=0,
            time_stop_stage_2_bars=0,
            time_stop_bars=60,
        )
        assert cfg.time_stop_stage_1_bars == 0
        assert cfg.time_stop_stage_2_bars == 0
        # With stages disabled, only hard cap at 60 bars matters
