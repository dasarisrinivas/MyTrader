"""Tests for EMA21 pullback overnight cap + R:R slippage buffer (MAR 16 2026).

Fix #1 — A/D per-session overnight cap:
  Signal A (EMA21_PB_LONG) and D (EMA21_PB_SHORT) previously had no daily counter,
  unlike every other signal (B, E, F, G). Root cause of the Mar 16 01:30 duplicate trade.
  ft_ema21_pb_max_overnight: 1  (max per direction outside RTH)
  ft_ema21_pb_max_rth:       3  (generous cap for multiple valid RTH pullbacks)

Fix #2 — Slippage buffer in overnight R:R gate:
  Market orders fill above/below close; gate computed from close overstates true R:R.
  ft_entry_slippage_pts: 0.5  (added to SL, subtracted from TP before comparing to floor)
  Effect: (12.0-0.5)/(7.2+0.5) = 11.5/7.7 = 1.49 — just below 1.5 floor.

Tests:
  Fix #1 — Overnight cap:
    [1]  First overnight A-signal fires (count=0, max=1)
    [2]  Second overnight A-signal blocked (count=1, max=1)
    [3]  First overnight D-signal fires (count=0, max=1)
    [4]  Second overnight D-signal blocked (count=1, max=1)
    [5]  RTH A-signal fires twice (max=3 — not blocked on 1st or 2nd)
    [6]  RTH A-signal blocked on 4th (count=3, max=3)
    [7]  Counter resets at session boundary (new day)
    [8]  Counter persists same session (survives bot restart simulation)
    [9]  rollback_counter undoes EMA21_PB_LONG increment
    [10] rollback_counter undoes EMA21_PB_SHORT increment

  Fix #2 — Slippage buffer:
    [11] Overnight trade passes R:R gate when slippage=0 (no buffer)
    [12] Overnight trade blocked when slippage=0.5 makes R:R < floor
    [13] RTH trade is NOT affected by slippage buffer (RTH-exempt gate)
    [14] Slippage buffer of 0.0 disabled — gate uses raw R:R from close
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, time, date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from shree.strategies.es_fifteen_min import EsFifteenMinStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> MagicMock:
    defaults: dict[str, Any] = {
        "ft_fixed_sl_points": 6.0,
        "ft_fixed_tp_points": 8.0,
        "ft_fixed_sl_points_ema9": 8.0,
        "ft_fixed_tp_points_ema9": 10.0,
        "ft_fixed_sl_points_trend": 8.0,
        "ft_fixed_tp_points_trend": 12.0,
        "ft_pb_stop_mult": 1.5,
        "ft_pb_target_mult": 2.5,
        "ft_short_pb_stop_mult": 1.5,
        "ft_short_pb_target_mult": 1.0,
        "ft_or_target_r": 1.5,
        "ft_short_or_target_r": 1.0,
        "ft_adx_min": 18.0,
        "ft_adx_max": 45.0,
        "ft_ema_touch_pct": 0.0015,
        "ft_ema_touch_atr_mult": 0.75,
        "ft_atr_very_low_threshold": 8.0,
        "ft_atr_high_threshold": 13.0,
        "ft_atr_extreme_threshold": 20.0,
        "ft_proximity_enabled": False,
        "ft_proximity_gap_mult": 0.3,
        "ft_proximity_size_mult": 0.7,
        "ft_proximity_sl_mult": 0.8,
        "ft_proximity_tp_mult": 0.8,
        "ft_proximity_max_per_day": 2,
        "ft_or_minutes": 30,
        "ft_or_break_max_per_day": 2,
        "ft_or_break_sl_atr_mult": 0.75,
        "ft_or_break_sl_floor_pts": 6.0,
        "ft_or_break_sl_ceiling_pts": 12.0,
        "ft_or_break_rr_ratio": 1.33,
        "ft_or_break_short_rsi_min": 40.0,
        "ft_or_break_long_rsi_max": 60.0,
        "ft_ema9_pb_enabled": False,
        "ft_ema9_pb_stop_mult": 1.2,
        "ft_ema9_pb_target_mult": 1.5,
        "ft_ema9_touch_pct": 0.0015,
        "ft_ema9_sl_atr_mult": 1.0,
        "ft_ema9_sl_floor_pts": 8.0,
        "ft_ema9_sl_ceiling_pts": 20.0,
        "ft_ema9_rr_ratio": 1.25,
        "ft_trend_cont_enabled": False,  # disabled — not relevant for A/D cap tests
        "ft_trend_cont_stop_mult": 1.0,
        "ft_trend_cont_target_mult": 2.0,
        "ft_trend_cont_adx_min": 25.0,
        "ft_trend_cont_ema9_pct": 0.003,
        "ft_trend_cont_max_ext_pts": 30.0,
        "ft_trend_cont_gap_adx_min": 25.0,
        "ft_trend_cont_max_per_day": 5,
        "ft_trend_sl_atr_mult": 1.0,
        "ft_trend_sl_floor_pts": 6.0,
        "ft_trend_sl_ceiling_pts": 20.0,
        "ft_trend_rr_ratio": 1.25,
        "ft_trend_exhaustion_atr_multiple": 0.0,
        "ft_london_enabled": False,
        "ft_london_start_hour": 2,
        "ft_london_start_minute": 0,
        "ft_london_end_hour": 5,
        "ft_london_end_minute": 0,
        "ft_london_adx_min": 15.0,
        "ft_london_sl_points": 5.0,
        "ft_london_sl_atr_cap": 1.0,
        "ft_london_tp_points": 8.0,
        "ft_london_max_per_day": 1,
        "ft_shorts_enabled": True,
        "ft_overnight_sl_mult": 1.2,
        "ft_overnight_tp_mult": 1.5,
        "ft_overnight_min_rr": 1.5,
        "ft_overnight_rsi_extreme_block": 0.0,   # disabled — not testing these guards here
        "ft_overnight_macd_divergence_threshold": 0.0,
        # Fix #1 fields
        "ft_ema21_pb_max_overnight": 1,
        "ft_ema21_pb_max_rth": 3,
        # Fix — ATR-adaptive SL for A/D (replaces fixed 6pt)
        "ft_ema21_sl_atr_mult": 1.0,
        "ft_ema21_sl_floor_pts": 6.0,
        "ft_ema21_sl_ceiling_pts": 15.0,
        "ft_ema21_rr_ratio": 1.33,
        # Fix — post-exhaustion cooldown (disabled for cap tests)
        "ft_exhaustion_cooldown_bars": 0,
        # Fix — Signal C MACD floor
        "ft_ema9_pb_macd_min": 0.3,
        # Fix #2 field — disabled by default so Fix #1 tests are not contaminated;
        # Fix #2 tests override this explicitly.
        "ft_entry_slippage_pts": 0.0,
        # Wide entry/RTH windows so all hours qualify
        "ft_entry_start_hour": 0,
        "ft_entry_start_minute": 0,
        "ft_entry_end_hour": 23,
        "ft_entry_end_minute": 59,
        "rth_start_hour": 0,
        "rth_start_minute": 0,
        "rth_end_hour": 23,
        "rth_end_minute": 59,
    }
    defaults.update(overrides)
    cfg = MagicMock()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    cfg.configure_mock(**{f"ft_{k}": None for k in []})
    return cfg


def _make_strategy(**cfg_overrides) -> EsFifteenMinStrategy:
    cfg = _make_config(**cfg_overrides)
    # Patch _load_counters during __init__ so the live data/signal_counters.json
    # never bleeds into tests (it's called inside __init__ before we can redirect).
    with patch.object(EsFifteenMinStrategy, '_load_counters', lambda self: None):
        strat = EsFifteenMinStrategy(cfg)
    # Redirect counter file so any saves/loads during the test go to /tmp
    strat._COUNTER_FILE = Path("/tmp/shreebot_test_counters_UNUSED.json")
    # Explicitly zero A/D counters (all others are irrelevant for these tests)
    strat._ema21_pb_long_count = 0
    strat._ema21_pb_short_count = 0
    # Align session date to the test dates (2026-03-16) so _reset_session
    # is NOT triggered inside generate() and doesn't zero manually-set counters.
    from datetime import date as _date
    strat._session_date = _date(2026, 3, 16)
    strat._or_computed = True
    strat._or_high = 6820.0
    strat._or_low = 6812.0
    return strat


def _overnight_ts() -> pd.Timestamp:
    """22:00 ET — clearly overnight (outside 9:30-16:00 ET)."""
    return pd.Timestamp("2026-03-16 22:00:00", tz="America/New_York")


def _rth_ts() -> pd.Timestamp:
    """11:00 ET — core RTH."""
    return pd.Timestamp("2026-03-16 11:00:00", tz="America/New_York")


def _make_index(last_ts: pd.Timestamp, rows: int) -> pd.DatetimeIndex:
    start = last_ts - pd.Timedelta(minutes=15 * (rows - 1))
    return pd.date_range(start=start, periods=rows, freq="15min", tz=last_ts.tzinfo)


# ---------------------------------------------------------------------------
# A-signal (EMA21_PB_LONG) DataFrame builder
# Conditions: EMA21 > EMA50, low <= EMA21 touch band, close > EMA21,
#             close > open (bullish), ADX in [18, 45]
# ---------------------------------------------------------------------------

def _a_signal_df(
    last_ts: pd.Timestamp | None = None,
    rows: int = 65,
    atr: float = 6.4,
    adx: float = 23.0,
    rsi: float = 54.0,
    macd: float = 0.25,
) -> pd.DataFrame:
    if last_ts is None:
        last_ts = _overnight_ts()
    close = 6666.5
    ema21 = 6663.8
    ema50 = ema21 - 5.0   # EMA21 > EMA50 (uptrend)
    ema9  = ema21 + 3.0
    # Bar low touches EMA21 band: use ema21 itself (exactly on band)
    low   = ema21
    open_ = close - 2.0   # bullish (open < close)
    high  = close + 1.0

    closes = [close] * rows
    opens  = [open_] * rows
    lows   = [low]   * rows
    highs  = [high]  * rows

    idx = _make_index(last_ts, rows)
    return pd.DataFrame({
        "close": closes, "open": opens, "high": highs, "low": lows,
        "volume": [1000] * rows,
        "EMA_9":  [ema9]  * rows,
        "EMA_21": [ema21] * rows,
        "EMA_50": [ema50] * rows,
        "RSI_14": [rsi]   * rows,
        "ADX_14": [adx]   * rows,
        "ATR_14": [atr]   * rows,
        "MACDhist_12_26_9": [macd] * rows,
    }, index=idx)


# ---------------------------------------------------------------------------
# D-signal (EMA21_PB_SHORT) DataFrame builder
# Conditions: EMA21 < EMA50, high >= EMA21 touch band, close < EMA21,
#             close < open (bearish), ADX in [18, 45]
# ---------------------------------------------------------------------------

def _d_signal_df(
    last_ts: pd.Timestamp | None = None,
    rows: int = 65,
    atr: float = 6.4,
    adx: float = 23.0,
    rsi: float = 46.0,
    macd: float = -0.25,
) -> pd.DataFrame:
    if last_ts is None:
        last_ts = _overnight_ts()
    close = 6656.5
    ema21 = 6659.8
    ema50 = ema21 + 5.0   # EMA21 < EMA50 (downtrend)
    ema9  = ema21 - 3.0
    # Bar high touches EMA21 band (from below)
    high  = ema21
    open_ = close + 2.0   # bearish (open > close)
    low   = close - 1.0

    closes = [close] * rows
    opens  = [open_] * rows
    lows   = [low]   * rows
    highs  = [high]  * rows

    idx = _make_index(last_ts, rows)
    return pd.DataFrame({
        "close": closes, "open": opens, "high": highs, "low": lows,
        "volume": [1000] * rows,
        "EMA_9":  [ema9]  * rows,
        "EMA_21": [ema21] * rows,
        "EMA_50": [ema50] * rows,
        "RSI_14": [rsi]   * rows,
        "ADX_14": [adx]   * rows,
        "ATR_14": [atr]   * rows,
        "MACDhist_12_26_9": [macd] * rows,
    }, index=idx)


def _call_generate(strat: EsFifteenMinStrategy, df: pd.DataFrame):
    return strat.generate(df)


# ===========================================================================
# Fix #1 — Overnight cap tests
# ===========================================================================

class TestEma21OvernightCap:

    def test_01_first_overnight_a_signal_fires(self):
        """First overnight A-signal fires when count=0."""
        strat = _make_strategy(ft_ema21_pb_max_overnight=1)
        assert strat._ema21_pb_long_count == 0
        sig = _call_generate(strat, _a_signal_df(last_ts=_overnight_ts()))
        assert sig.action == "BUY", f"Expected BUY, got {sig.action} (meta={sig.metadata})"
        assert strat._ema21_pb_long_count == 1

    def test_02_second_overnight_a_signal_blocked(self):
        """Second overnight A-signal is blocked (count=1 >= max=1)."""
        strat = _make_strategy(ft_ema21_pb_max_overnight=1)
        strat._ema21_pb_long_count = 1   # simulate first already fired
        sig = _call_generate(strat, _a_signal_df(last_ts=_overnight_ts()))
        assert sig.action == "HOLD", f"Expected HOLD (cap), got {sig.action}"

    def test_03_first_overnight_d_signal_fires(self):
        """First overnight D-signal fires when count=0."""
        strat = _make_strategy(ft_ema21_pb_max_overnight=1)
        assert strat._ema21_pb_short_count == 0
        sig = _call_generate(strat, _d_signal_df(last_ts=_overnight_ts()))
        assert sig.action == "SELL", f"Expected SELL, got {sig.action} (meta={sig.metadata})"
        assert strat._ema21_pb_short_count == 1

    def test_04_second_overnight_d_signal_blocked(self):
        """Second overnight D-signal is blocked (count=1 >= max=1)."""
        strat = _make_strategy(ft_ema21_pb_max_overnight=1)
        strat._ema21_pb_short_count = 1
        sig = _call_generate(strat, _d_signal_df(last_ts=_overnight_ts()))
        assert sig.action == "HOLD", f"Expected HOLD (cap), got {sig.action}"

    def test_05_rth_a_signal_fires_twice(self):
        """RTH A-signal fires on first and second call (max_rth=3, not blocked)."""
        strat = _make_strategy(ft_ema21_pb_max_rth=3)
        # First
        sig1 = _call_generate(strat, _a_signal_df(last_ts=_rth_ts()))
        assert sig1.action == "BUY", f"1st RTH A should fire, got {sig1.action}"
        assert strat._ema21_pb_long_count == 1
        # Second (advance timestamp by 15min)
        ts2 = _rth_ts() + pd.Timedelta(minutes=15)
        sig2 = _call_generate(strat, _a_signal_df(last_ts=ts2))
        assert sig2.action == "BUY", f"2nd RTH A should fire, got {sig2.action}"
        assert strat._ema21_pb_long_count == 2

    def test_06_rth_a_signal_blocked_on_fourth(self):
        """RTH A-signal is blocked when count=3 >= max_rth=3."""
        strat = _make_strategy(ft_ema21_pb_max_rth=3)
        strat._ema21_pb_long_count = 3
        sig = _call_generate(strat, _a_signal_df(last_ts=_rth_ts()))
        assert sig.action == "HOLD", f"4th RTH A should be blocked, got {sig.action}"

    def test_07_counter_resets_on_new_session(self):
        """Counter resets to 0 when session date changes."""
        strat = _make_strategy(ft_ema21_pb_max_overnight=1)
        strat._ema21_pb_long_count = 1
        strat._ema21_pb_short_count = 1
        # Trigger a session reset by calling _reset_session with a new date
        from datetime import date as _date
        strat._reset_session(_date(2026, 3, 17))
        assert strat._ema21_pb_long_count == 0
        assert strat._ema21_pb_short_count == 0

    def test_08_counter_persists_same_session(self):
        """Counter survives a simulated bot restart within same CME session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            counter_path = Path(tmpdir) / "signal_counters.json"

            strat = _make_strategy(ft_ema21_pb_max_overnight=1)
            strat._COUNTER_FILE = counter_path
            strat._ema21_pb_long_count = 0   # ensure clean start

            # Simulate first signal fired
            _call_generate(strat, _a_signal_df(last_ts=_overnight_ts()))
            assert strat._ema21_pb_long_count == 1
            assert counter_path.exists(), "Counter file should have been written"

            # Create fresh strategy pointing at same counter file
            strat2 = _make_strategy(ft_ema21_pb_max_overnight=1)
            strat2._COUNTER_FILE = counter_path
            strat2._ema21_pb_long_count = 0   # start fresh before load
            with patch.object(strat2, '_get_cme_session_date',
                              return_value=strat._get_cme_session_date()):
                strat2._load_counters()
            assert strat2._ema21_pb_long_count == 1, (
                "Counter should have loaded from disk as 1"
            )

    def test_09_rollback_counter_ema21_pb_long(self):
        """rollback_counter decrements _ema21_pb_long_count."""
        strat = _make_strategy()
        strat._ema21_pb_long_count = 2
        strat.rollback_counter("EMA21_PB_LONG | ADX=23")
        assert strat._ema21_pb_long_count == 1

    def test_10_rollback_counter_ema21_pb_short(self):
        """rollback_counter decrements _ema21_pb_short_count."""
        strat = _make_strategy()
        strat._ema21_pb_short_count = 1
        strat.rollback_counter("EMA21_PB_SHORT | ADX=23")
        assert strat._ema21_pb_short_count == 0

    def test_10b_rollback_counter_does_not_go_negative(self):
        """rollback_counter on zero count is a no-op (no underflow)."""
        strat = _make_strategy()
        strat._ema21_pb_long_count = 0
        strat.rollback_counter("EMA21_PB_LONG | ADX=23")
        assert strat._ema21_pb_long_count == 0


# ===========================================================================
# Fix #2 — Slippage buffer in R:R gate tests
# ===========================================================================

class TestSlippageRRGate:
    """
    Overnight R:R after scaling: TP=8×1.5=12.0, SL=6×1.2=7.2 → R:R=1.67 from close.
    With slippage=0.5: (12.0-0.5)/(7.2+0.5) = 11.5/7.7 = 1.49 < 1.5 floor → HOLD.
    Without slippage (=0.0): 12.0/7.2 = 1.67 >= 1.5 floor → passes.
    """

    def test_11_overnight_passes_without_slippage_buffer(self):
        """With slippage=0.0, overnight A-signal at R:R=1.67 passes gate."""
        strat = _make_strategy(ft_entry_slippage_pts=0.0)
        sig = _call_generate(strat, _a_signal_df(last_ts=_overnight_ts()))
        assert sig.action == "BUY", (
            f"Expected BUY (R:R=1.67 >= 1.5 floor with no slippage), got {sig.action}"
        )

    def test_12_overnight_blocked_with_slippage_buffer(self):
        """With slippage=0.5, overnight A-signal at effective R:R=1.49 blocked by gate."""
        strat = _make_strategy(ft_entry_slippage_pts=0.5)
        sig = _call_generate(strat, _a_signal_df(last_ts=_overnight_ts()))
        assert sig.action == "HOLD", (
            f"Expected HOLD (R:R=1.49 < 1.5 floor with slippage=0.5), got {sig.action} "
            f"meta={sig.metadata}"
        )
        assert "OVERNIGHT_RR" in sig.metadata.get("reason", ""), (
            "HOLD reason should reference OVERNIGHT_RR gate"
        )

    def test_13_rth_not_affected_by_slippage_buffer(self):
        """RTH A-signal is not subject to the overnight R:R gate regardless of slippage."""
        strat = _make_strategy(ft_entry_slippage_pts=0.5)
        sig = _call_generate(strat, _a_signal_df(last_ts=_rth_ts()))
        assert sig.action == "BUY", (
            f"RTH signal should not be blocked by overnight R:R gate, got {sig.action}"
        )

    def test_14_slippage_zero_disables_buffer(self):
        """ft_entry_slippage_pts=0 means gate uses raw R:R — 1.67 passes 1.5 floor."""
        strat = _make_strategy(ft_entry_slippage_pts=0.0, ft_overnight_min_rr=1.5)
        sig = _call_generate(strat, _a_signal_df(last_ts=_overnight_ts()))
        assert sig.action == "BUY", (
            f"With slippage=0, R:R=1.67 should pass 1.5 floor, got {sig.action}"
        )
