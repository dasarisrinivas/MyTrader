"""Tests for MAR 16 2026 fixes: ATR-adaptive SL, exhaustion cooldown, MACD floor, MACD divergence.

Fix 1 — ATR-adaptive SL for Signal A/D:
  SL = clamp(ATR × 1.0, 6pt floor, 15pt ceiling), TP = SL × 1.33 (ticked to 0.25pt)
  Replaces fixed 6pt/8pt SL/TP. At ATR=10: SL=10pt, TP=13.25pt.

Fix 2 — Post-exhaustion cooldown:
  When session move > N×ATR from OR midpoint, block ALL same-direction
  signals for 4 bars (60 min). Previously only blocked TREND_CONT.

Fix 3 — Signal C MACD floor:
  Raised from `macd_hist > 0` to `macd_hist >= 0.3`. Near-zero MACD
  has no edge for shallow EMA9 pullbacks.

Fix 5 — MACD divergence filter for Signal A/D:
  Block A when MACD_H < -1.0 (bearish momentum opposes long pullback).
  Block D when MACD_H > +1.0 (bullish momentum opposes short pullback).
  Set to 0 to disable.

Tests:
  Fix 1 — ATR-adaptive SL (A/D):
    [1]  Signal A: ATR=10 → SL=10pt, TP=13.25pt (not old fixed 6/8)
    [2]  Signal A: ATR=5 → SL=6pt (floor), TP=8pt (floor × 1.33 ticked)
    [3]  Signal A: ATR=20 → SL=15pt (ceiling), TP=20pt (ceiling × 1.33 ticked)
    [4]  Signal D: ATR=12 → SL=12pt, TP=16pt (mirrored short side)
    [5]  Signal D: ATR=4 → SL=6pt (floor), TP=8pt

  Fix 2 — Post-exhaustion cooldown:
    [6]  No exhaustion → A-signal fires normally
    [7]  Long exhaustion active → BUY signals blocked, SELL signals pass
    [8]  Short exhaustion active → SELL signals blocked, BUY signals pass
    [9]  Cooldown decrements each bar and expires
    [10] Cooldown=0 disables exhaustion blocking entirely
    [11] Exhaustion detection triggers at > N×ATR from OR midpoint

  Fix 3 — Signal C MACD floor:
    [12] MACD=0.5 >= 0.3 → Signal C fires
    [13] MACD=0.2 < 0.3 → Signal C blocked
    [14] MACD=0.0 < 0.3 → Signal C blocked (was allowed under old > 0)
    [15] Custom threshold=0.0 → reverts to old behaviour (any positive MACD)

  Fix 5 — MACD divergence filter (A/D):
    [16] Signal A: MACD=-1.40 < -1.0 → blocked
    [17] Signal A: MACD=-0.5 > -1.0 → fires
    [18] Signal A: MACD=+2.0 → fires
    [19] Signal A: MACD=-1.0 (boundary) → fires (strict <)
    [20] Signal D: MACD=+1.50 > +1.0 → blocked
    [21] Signal D: MACD=+0.5 < +1.0 → fires
    [22] Signal D: MACD=+1.0 (boundary) → fires (strict >)
    [23] threshold=0 disables A filter
    [24] threshold=0 disables D filter
"""
from __future__ import annotations

from datetime import date, time
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
        "ft_or_break_max_chase_atr": 1.0,
        "ft_or_break_short_rsi_min": 40.0,
        "ft_or_break_long_rsi_max": 60.0,
        "ft_ema9_pb_enabled": True,   # enabled for Signal C MACD tests
        "ft_ema9_pb_stop_mult": 1.2,
        "ft_ema9_pb_target_mult": 1.5,
        "ft_ema9_touch_pct": 0.0015,
        "ft_ema9_pb_macd_min": 0.3,   # Fix 3: MACD floor
        "ft_ema9_sl_atr_mult": 1.0,
        "ft_ema9_sl_floor_pts": 8.0,
        "ft_ema9_sl_ceiling_pts": 20.0,
        "ft_ema9_rr_ratio": 1.25,
        "ft_trend_cont_enabled": False,
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
        "ft_trend_exhaustion_atr_multiple": 8.0,
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
        "ft_overnight_rsi_extreme_block": 0.0,
        "ft_overnight_macd_divergence_threshold": 0.0,
        # Fix 1: ATR-adaptive SL for A/D
        "ft_ema21_sl_atr_mult": 1.0,
        "ft_ema21_sl_floor_pts": 6.0,
        "ft_ema21_sl_ceiling_pts": 15.0,
        "ft_ema21_rr_ratio": 1.33,
        # Fix 5: MACD divergence filter for A/D (default=1.0)
        "ft_ema21_macd_divergence_block": 1.0,
        # Fix 1 prereq: A/D caps
        "ft_ema21_pb_max_overnight": 10,  # high cap so not hit in tests
        "ft_ema21_pb_max_rth": 10,
        "ft_entry_slippage_pts": 0.0,
        # Fix 2: Exhaustion cooldown (disabled by default for non-cooldown tests)
        "ft_exhaustion_cooldown_bars": 0,
        # MAY 12 2026: new fields — disable in existing tests to preserve behaviour
        "ft_monday_block_enabled": False,        # test session_date is 2026-03-16 (Monday)
        "ft_late_afternoon_block_hour_utc": 20,  # 8 PM UTC default — don't block 11 AM ET tests
        "ft_ema21_sl_use_structural": False,     # ATR SL tests validate ATR behaviour
        "ft_ema21_sl_buffer_pts": 0.5,
        "ft_htf_filter_enabled": False,          # HTF filter not under test here
        "ft_htf_filter_mode": "block_counter",
        # Wide entry/RTH windows
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
    return cfg


def _make_strategy(**cfg_overrides) -> EsFifteenMinStrategy:
    cfg = _make_config(**cfg_overrides)
    with patch.object(EsFifteenMinStrategy, '_load_counters', lambda self: None):
        strat = EsFifteenMinStrategy(cfg)
    strat._COUNTER_FILE = Path("/tmp/shreebot_test_mar16_UNUSED.json")
    strat._ema21_pb_long_count = 0
    strat._ema21_pb_short_count = 0
    strat._session_date = date(2026, 3, 16)
    strat._or_computed = True
    strat._or_high = 6820.0
    strat._or_low = 6812.0
    return strat


def _rth_ts() -> pd.Timestamp:
    """11:00 ET — core RTH."""
    return pd.Timestamp("2026-03-16 11:00:00", tz="America/New_York")


def _make_index(last_ts: pd.Timestamp, rows: int = 65) -> pd.DatetimeIndex:
    start = last_ts - pd.Timedelta(minutes=15 * (rows - 1))
    return pd.date_range(start=start, periods=rows, freq="15min", tz=last_ts.tzinfo)


# ---------------------------------------------------------------------------
# DataFrame builders for each signal type
# ---------------------------------------------------------------------------

def _a_signal_df(
    close: float = 6850.0,
    ema21: float = 6848.0,
    ema50: float = 6840.0,
    atr: float = 10.0,
    adx: float = 25.0,
    rsi: float = 50.0,
    macd_hist: float = 0.5,
    rows: int = 65,
) -> pd.DataFrame:
    """Signal A: uptrend, bar low touches EMA21, bullish close."""
    idx = _make_index(_rth_ts(), rows)
    # Touch band at low vol: ema21 * (1 + 0.0015)
    touch_threshold = ema21 * 1.0015
    data = {
        "open":  [close - 2.0] * rows,
        "high":  [close + 2.0] * rows,
        "low":   [ema21 - 0.5] * rows,   # touches EMA21
        "close": [close] * rows,
        "volume": [100] * rows,
        "EMA_9":  [close + 1.0] * rows,
        "EMA_21": [ema21] * rows,
        "EMA_50": [ema50] * rows,
        "ATR_14": [atr] * rows,
        "ADX_14": [adx] * rows,
        "RSI_14": [rsi] * rows,
        "MACDhist_12_26_9": [macd_hist] * rows,
    }
    return pd.DataFrame(data, index=idx)


def _d_signal_df(
    close: float = 6830.0,
    ema21: float = 6832.0,
    ema50: float = 6840.0,
    atr: float = 10.0,
    adx: float = 25.0,
    rsi: float = 50.0,
    macd_hist: float = -0.5,
    rows: int = 65,
) -> pd.DataFrame:
    """Signal D: downtrend, bar high touches EMA21, bearish close."""
    idx = _make_index(_rth_ts(), rows)
    data = {
        "open":  [close + 2.0] * rows,
        "high":  [ema21 + 0.5] * rows,   # touches EMA21
        "low":   [close - 2.0] * rows,
        "close": [close] * rows,
        "volume": [100] * rows,
        "EMA_9":  [close - 1.0] * rows,
        "EMA_21": [ema21] * rows,
        "EMA_50": [ema50] * rows,
        "ATR_14": [atr] * rows,
        "ADX_14": [adx] * rows,
        "RSI_14": [rsi] * rows,
        "MACDhist_12_26_9": [macd_hist] * rows,
    }
    return pd.DataFrame(data, index=idx)


def _c_signal_df(
    close: float = 6860.0,
    ema9: float = 6858.0,
    ema21: float = 6840.0,
    ema50: float = 6830.0,
    atr: float = 10.0,
    adx: float = 28.0,
    rsi: float = 55.0,
    macd_hist: float = 0.5,
    rows: int = 65,
) -> pd.DataFrame:
    """Signal C: strong uptrend, bar low touches EMA9, bullish close.
    EMA9 > EMA21 > EMA50, ADX 22-35, RSI 40-70.
    Bar low near EMA9 but NOT near EMA21 (must not trigger Signal A).
    """
    idx = _make_index(_rth_ts(), rows)
    # Low touches EMA9 (~6858) but stays well above EMA21 touch band (~6840*1.0015=6849.6)
    data = {
        "open":  [close - 2.0] * rows,
        "high":  [close + 2.0] * rows,
        "low":   [ema9 - 0.3] * rows,    # 6857.7 — touches EMA9, above EMA21 band
        "close": [close] * rows,
        "volume": [100] * rows,
        "EMA_9":  [ema9] * rows,
        "EMA_21": [ema21] * rows,
        "EMA_50": [ema50] * rows,
        "ATR_14": [atr] * rows,
        "ADX_14": [adx] * rows,
        "RSI_14": [rsi] * rows,
        "MACDhist_12_26_9": [macd_hist] * rows,
    }
    return pd.DataFrame(data, index=idx)


# ===========================================================================
#  Fix 1: ATR-adaptive SL for Signal A/D
# ===========================================================================

class TestAtrAdaptiveSL:

    # -- Signal A (long) --

    def test_01_signal_a_atr10_adaptive_sl(self):
        """ATR=10: SL=10pt, TP=10×1.33=13.25pt (ticked to 0.25)."""
        strat = _make_strategy()
        df = _a_signal_df(close=6850.0, atr=10.0)
        sig = strat.generate(df)
        assert sig.action == "BUY"
        sl = sig.metadata["stop_loss"]
        tp = sig.metadata["take_profit"]
        assert abs(sl - (6850.0 - 10.0)) < 0.01, f"SL should be 6840.0, got {sl}"
        assert abs(tp - (6850.0 + 13.25)) < 0.01, f"TP should be 6863.25, got {tp}"

    def test_02_signal_a_low_atr_floor(self):
        """ATR=5: SL=6pt (floor), TP=6×1.33=7.98→8.0 (ticked)."""
        strat = _make_strategy()
        df = _a_signal_df(close=6850.0, atr=5.0)
        sig = strat.generate(df)
        assert sig.action == "BUY"
        sl = sig.metadata["stop_loss"]
        tp = sig.metadata["take_profit"]
        assert abs(sl - (6850.0 - 6.0)) < 0.01, f"SL should be 6844.0, got {sl}"
        expected_tp = round(6.0 * 1.33 * 4) / 4  # 7.98 → 8.0
        assert abs(tp - (6850.0 + expected_tp)) < 0.01, f"TP should be {6850.0 + expected_tp}, got {tp}"

    def test_03_signal_a_high_atr_ceiling(self):
        """ATR=20: SL=15pt (ceiling), TP=15×1.33=19.95→20.0 (ticked)."""
        strat = _make_strategy()
        df = _a_signal_df(close=6850.0, atr=20.0)
        sig = strat.generate(df)
        assert sig.action == "BUY"
        sl = sig.metadata["stop_loss"]
        tp = sig.metadata["take_profit"]
        assert abs(sl - (6850.0 - 15.0)) < 0.01, f"SL should be 6835.0, got {sl}"
        expected_tp = round(15.0 * 1.33 * 4) / 4  # 19.95 → 20.0
        assert abs(tp - (6850.0 + expected_tp)) < 0.01, f"TP should be {6850.0 + expected_tp}, got {tp}"

    # -- Signal D (short) --

    def test_04_signal_d_atr12_adaptive_sl(self):
        """ATR=12: SL=12pt, TP=12×1.33=15.96→16.0 (ticked). Short side."""
        strat = _make_strategy()
        df = _d_signal_df(close=6830.0, atr=12.0)
        sig = strat.generate(df)
        assert sig.action == "SELL"
        sl = sig.metadata["stop_loss"]
        tp = sig.metadata["take_profit"]
        assert abs(sl - (6830.0 + 12.0)) < 0.01, f"SL should be 6842.0, got {sl}"
        expected_tp = round(12.0 * 1.33 * 4) / 4  # 15.96 → 16.0
        assert abs(tp - (6830.0 - expected_tp)) < 0.01, f"TP should be {6830.0 - expected_tp}, got {tp}"

    def test_05_signal_d_low_atr_floor(self):
        """ATR=4: SL=6pt (floor), TP=8pt (floor × 1.33 ticked). Short side."""
        strat = _make_strategy()
        df = _d_signal_df(close=6830.0, atr=4.0)
        sig = strat.generate(df)
        assert sig.action == "SELL"
        sl = sig.metadata["stop_loss"]
        tp = sig.metadata["take_profit"]
        assert abs(sl - (6830.0 + 6.0)) < 0.01, f"SL should be 6836.0, got {sl}"
        expected_tp = round(6.0 * 1.33 * 4) / 4
        assert abs(tp - (6830.0 - expected_tp)) < 0.01, f"TP should be {6830.0 - expected_tp}, got {tp}"


# ===========================================================================
#  Fix 2: Post-exhaustion cooldown
# ===========================================================================

class TestExhaustionCooldown:

    def test_06_no_exhaustion_signal_fires(self):
        """Without exhaustion, A-signal fires normally."""
        strat = _make_strategy(ft_exhaustion_cooldown_bars=4)
        df = _a_signal_df(close=6850.0, atr=10.0)
        sig = strat.generate(df)
        assert sig.action == "BUY"

    def test_07_long_exhaustion_blocks_buy(self):
        """Long exhaustion active → BUY blocked."""
        strat = _make_strategy(ft_exhaustion_cooldown_bars=4)
        # Manually set long cooldown active
        strat._exhaustion_long_bars_left = 3
        df = _a_signal_df(close=6850.0, atr=10.0)
        sig = strat.generate(df)
        assert sig.action == "HOLD", f"Expected HOLD, got {sig.action}"

    def test_08_short_exhaustion_blocks_sell(self):
        """Short exhaustion active → SELL blocked, BUY still works."""
        strat = _make_strategy(ft_exhaustion_cooldown_bars=4)
        # Set short cooldown active but not long
        strat._exhaustion_short_bars_left = 3
        # A-signal (BUY) should still fire
        df = _a_signal_df(close=6850.0, atr=10.0)
        sig = strat.generate(df)
        assert sig.action == "BUY", "Short exhaustion should not block BUY"

    def test_09_cooldown_decrements_and_expires(self):
        """Cooldown decrements each bar and eventually expires."""
        strat = _make_strategy(ft_exhaustion_cooldown_bars=4)
        strat._exhaustion_long_bars_left = 1  # last bar of cooldown
        df = _a_signal_df(close=6850.0, atr=10.0)
        # On this call, decrement → 0, so BUY should NOT be blocked
        sig = strat.generate(df)
        assert sig.action == "BUY", "Cooldown should have expired (1→0)"
        assert strat._exhaustion_long_bars_left == 0

    def test_10_cooldown_zero_disables(self):
        """ft_exhaustion_cooldown_bars=0 disables exhaustion blocking entirely."""
        strat = _make_strategy(
            ft_exhaustion_cooldown_bars=0,
            ft_trend_exhaustion_atr_multiple=8.0,
        )
        # Set OR so close is far from midpoint — would trigger exhaustion if enabled
        strat._or_high = 6816.0
        strat._or_low = 6812.0
        # OR midpoint = 6814, close = 6850 → move = 36pts, at ATR=10 = 3.6×ATR
        # Not > 8× so won't trigger anyway. Let's use a more extreme case.
        strat._or_high = 6780.0
        strat._or_low = 6770.0
        # midpoint = 6775, close=6850, move=75pts, at ATR=10 = 7.5×ATR — still <8
        # Make it > 8×ATR:
        strat._or_high = 6770.0
        strat._or_low = 6760.0
        # midpoint = 6765, close=6850, move=85pts, at ATR=10 = 8.5×ATR > 8 — would trigger
        df = _a_signal_df(close=6850.0, atr=10.0)
        sig = strat.generate(df)
        # With cooldown=0, exhaustion detection is skipped
        assert sig.action == "BUY", "Cooldown=0 should not block any signals"
        assert strat._exhaustion_long_bars_left == 0

    def test_11_exhaustion_detection_sets_cooldown(self):
        """Session move > 8×ATR from OR midpoint sets cooldown."""
        strat = _make_strategy(ft_exhaustion_cooldown_bars=4,
                               ft_trend_exhaustion_atr_multiple=8.0)
        # OR midpoint = (6770+6760)/2 = 6765
        # close=6850, move=85pts, at ATR=10 = 8.5× > 8.0 → exhaustion
        strat._or_high = 6770.0
        strat._or_low = 6760.0
        df = _a_signal_df(close=6850.0, atr=10.0)
        sig = strat.generate(df)
        # Long exhaustion should be set, blocking BUY
        assert sig.action == "HOLD", f"Expected HOLD (exhaustion), got {sig.action}"
        assert strat._exhaustion_long_bars_left > 0, "Long cooldown should be active"


# ===========================================================================
#  Fix 3: Signal C MACD floor
# ===========================================================================

class TestSignalCMacdFloor:

    def test_12_macd_above_floor_fires(self):
        """MACD=0.5 >= 0.3 → Signal C fires."""
        strat = _make_strategy(ft_ema9_pb_macd_min=0.3)
        df = _c_signal_df(macd_hist=0.5)
        sig = strat.generate(df)
        assert sig.action == "BUY", f"Signal C should fire with MACD=0.5, got {sig.action}"
        assert "EMA9_PB" in sig.metadata.get("reason", "")

    def test_13_macd_below_floor_blocked(self):
        """MACD=0.2 < 0.3 → Signal C blocked."""
        strat = _make_strategy(ft_ema9_pb_macd_min=0.3)
        df = _c_signal_df(macd_hist=0.2)
        sig = strat.generate(df)
        # C is blocked, no other signal should fire (A might fire if conditions met)
        # Make sure it's not EMA9_PB
        reason = sig.metadata.get("reason", "")
        assert "EMA9_PB" not in reason, f"Signal C should NOT fire with MACD=0.2, got {reason}"

    def test_14_macd_zero_blocked(self):
        """MACD=0.0 < 0.3 → Signal C blocked (was allowed under old > 0)."""
        strat = _make_strategy(ft_ema9_pb_macd_min=0.3)
        df = _c_signal_df(macd_hist=0.0)
        sig = strat.generate(df)
        reason = sig.metadata.get("reason", "")
        assert "EMA9_PB" not in reason, f"Signal C should NOT fire with MACD=0.0, got {reason}"

    def test_15_macd_floor_zero_reverts(self):
        """ft_ema9_pb_macd_min=0.0 → old behaviour, any positive MACD passes."""
        strat = _make_strategy(ft_ema9_pb_macd_min=0.0)
        df = _c_signal_df(macd_hist=0.05)  # tiny positive
        sig = strat.generate(df)
        assert sig.action == "BUY", f"Signal C should fire with MACD=0.05 at floor=0.0, got {sig.action}"
        assert "EMA9_PB" in sig.metadata.get("reason", "")


# ===========================================================================
#  Fix 5: MACD divergence filter for Signal A/D
#
#  Block Signal A (BUY) when MACD_H < -threshold (bearish momentum opposes
#  the long pullback). Block Signal D (SELL) when MACD_H > +threshold
#  (bullish momentum opposes the short pullback).
#  Default threshold = 1.0. Set to 0 to disable.
#
#  Root cause: Mar 16 12:30 trade — Signal A fired BUY with MACD_H = -1.40.
#  Price was in a momentum slide, not a healthy pullback. Lost -$57.50.
# ===========================================================================

class TestMacdDivergenceAD:
    """MACD divergence filter for Signal A/D (Fix #5)."""

    # ── Signal A (Long) ──────────────────────────────────────────────

    def test_16_a_blocked_when_macd_deeply_negative(self):
        """Signal A: MACD=-1.40 < -1.0 threshold → BUY blocked."""
        strat = _make_strategy(ft_ema21_macd_divergence_block=1.0)
        df = _a_signal_df(macd_hist=-1.40)
        sig = strat.generate(df)
        reason = sig.metadata.get("reason", "")
        assert "EMA21_PB_LONG" not in reason, \
            f"Signal A should be BLOCKED with MACD=-1.40, got {reason}"

    def test_17_a_fires_when_macd_mildly_negative(self):
        """Signal A: MACD=-0.5 > -1.0 → BUY fires (mild negative OK)."""
        strat = _make_strategy(ft_ema21_macd_divergence_block=1.0)
        df = _a_signal_df(macd_hist=-0.5)
        sig = strat.generate(df)
        assert sig.action == "BUY", f"Signal A should fire with MACD=-0.5, got {sig.action}"
        assert "EMA21_PB_LONG" in sig.metadata.get("reason", "")

    def test_18_a_fires_when_macd_positive(self):
        """Signal A: MACD=+2.0 → BUY fires (momentum aligned)."""
        strat = _make_strategy(ft_ema21_macd_divergence_block=1.0)
        df = _a_signal_df(macd_hist=2.0)
        sig = strat.generate(df)
        assert sig.action == "BUY", f"Signal A should fire with MACD=+2.0, got {sig.action}"
        assert "EMA21_PB_LONG" in sig.metadata.get("reason", "")

    def test_19_a_boundary_exactly_at_threshold(self):
        """Signal A: MACD=-1.0 is NOT < -1.0 → BUY fires (strict inequality)."""
        strat = _make_strategy(ft_ema21_macd_divergence_block=1.0)
        df = _a_signal_df(macd_hist=-1.0)
        sig = strat.generate(df)
        assert sig.action == "BUY", f"Signal A should fire at MACD exactly -1.0, got {sig.action}"
        assert "EMA21_PB_LONG" in sig.metadata.get("reason", "")

    # ── Signal D (Short) ─────────────────────────────────────────────

    def test_20_d_blocked_when_macd_deeply_positive(self):
        """Signal D: MACD=+1.50 > +1.0 threshold → SELL blocked."""
        strat = _make_strategy(ft_ema21_macd_divergence_block=1.0)
        df = _d_signal_df(macd_hist=1.50)
        sig = strat.generate(df)
        reason = sig.metadata.get("reason", "")
        assert "EMA21_PB_SHORT" not in reason, \
            f"Signal D should be BLOCKED with MACD=+1.50, got {reason}"

    def test_21_d_fires_when_macd_mildly_positive(self):
        """Signal D: MACD=+0.5 < +1.0 → SELL fires (mild positive OK)."""
        strat = _make_strategy(ft_ema21_macd_divergence_block=1.0)
        df = _d_signal_df(macd_hist=0.5)
        sig = strat.generate(df)
        assert sig.action == "SELL", f"Signal D should fire with MACD=+0.5, got {sig.action}"
        assert "EMA21_PB_SHORT" in sig.metadata.get("reason", "")

    def test_22_d_boundary_exactly_at_threshold(self):
        """Signal D: MACD=+1.0 is NOT > +1.0 → SELL fires (strict inequality)."""
        strat = _make_strategy(ft_ema21_macd_divergence_block=1.0)
        df = _d_signal_df(macd_hist=1.0)
        sig = strat.generate(df)
        assert sig.action == "SELL", f"Signal D should fire at MACD exactly +1.0, got {sig.action}"
        assert "EMA21_PB_SHORT" in sig.metadata.get("reason", "")

    # ── Disable (threshold=0) ────────────────────────────────────────

    def test_23_threshold_zero_disables_a(self):
        """ft_ema21_macd_divergence_block=0 → A fires even with deeply negative MACD."""
        strat = _make_strategy(ft_ema21_macd_divergence_block=0.0)
        df = _a_signal_df(macd_hist=-5.0)  # extremely negative
        sig = strat.generate(df)
        assert sig.action == "BUY", f"Signal A should fire with MACD=-5.0 when filter disabled, got {sig.action}"
        assert "EMA21_PB_LONG" in sig.metadata.get("reason", "")

    def test_24_threshold_zero_disables_d(self):
        """ft_ema21_macd_divergence_block=0 → D fires even with deeply positive MACD."""
        strat = _make_strategy(ft_ema21_macd_divergence_block=0.0)
        df = _d_signal_df(macd_hist=5.0)  # extremely positive
        sig = strat.generate(df)
        assert sig.action == "SELL", f"Signal D should fire with MACD=+5.0 when filter disabled, got {sig.action}"
        assert "EMA21_PB_SHORT" in sig.metadata.get("reason", "")

    # ── MAR 19 2026 regression (threshold lowered 1.0 → 0.5) ─────────

    def test_25_d_blocked_mar19_loser_macd_095(self):
        """MAR 19 2026 regression: 11:00 RTH short MACD_H=+0.95 hit SL at threshold=1.0.
        At threshold=0.5, MACD=+0.95 > +0.5 → SELL must be blocked."""
        strat = _make_strategy(ft_ema21_macd_divergence_block=0.5)
        df = _d_signal_df(macd_hist=0.95)
        sig = strat.generate(df)
        reason = sig.metadata.get("reason", "")
        assert "EMA21_PB_SHORT" not in reason, \
            f"Signal D should be BLOCKED with MACD=+0.95 at threshold=0.5, got {reason}"

    def test_26_d_fires_mar19_winner_macd_neg147(self):
        """MAR 19 2026 regression: 09:00 RTH short MACD_H=-1.47 won — must still fire."""
        strat = _make_strategy(ft_ema21_macd_divergence_block=0.5)
        df = _d_signal_df(macd_hist=-1.47)
        sig = strat.generate(df)
        assert sig.action == "SELL", f"Signal D should fire with MACD=-1.47, got {sig.action}"
        assert "EMA21_PB_SHORT" in sig.metadata.get("reason", "")
