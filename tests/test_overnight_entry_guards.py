"""Tests for overnight entry-quality guards (MAR 15 2026).

Two guards added to es_fifteen_min.py to block signal patterns with no
overnight edge (outside core RTH 9:30–16:00 ET):

Guard 1 — RSI extreme (ft_overnight_rsi_extreme_block, default 35):
  Block TREND_CONT_SHORT overnight when RSI < threshold (oversold exhaustion).
  Block TREND_CONT_LONG  overnight when RSI > (100-threshold) (overbought exhaustion).
  Evidence: 2 overnight TREND_CONT losses at RSI=30 and RSI=32 → −$110.

Guard 2 — MACD divergence (ft_overnight_macd_divergence_threshold, default 0.5):
  Block D/D-prime/E short overnight when MACD histogram > threshold.
  Block A/C       long  overnight when MACD histogram < -threshold.
  Evidence: 2 overnight D/E short losses with MACD=+1.03 and +1.63 → −$78.
  (MACD was removed from D/E globally in Mar 2026; this restores it only overnight.)

Tests:
  RSI extreme guard:
    [1]  TREND_CONT_SHORT overnight RSI=30 → HOLD (guard fires)
    [2]  TREND_CONT_SHORT overnight RSI=32 → HOLD (guard fires)
    [3]  TREND_CONT_SHORT overnight RSI=38 → signal passes (above threshold)
    [4]  TREND_CONT_SHORT RTH       RSI=30 → signal passes (RTH exempt)
    [5]  TREND_CONT_LONG  overnight RSI=71 → HOLD (guard fires)
    [6]  TREND_CONT_LONG  overnight RSI=68 → signal passes
    [7]  TREND_CONT_LONG  RTH       RSI=72 → signal passes (RTH exempt)
    [8]  Guard disabled (threshold=0): overnight RSI=28 → signal passes

  MACD divergence guard:
    [9]  D-short overnight MACD=+1.03 → HOLD (guard fires)
    [10] D-short overnight MACD=+1.63 → HOLD (guard fires)
    [11] D-short overnight MACD=+0.3  → signal passes (below threshold)
    [12] D-short overnight MACD=-1.0  → signal passes (aligned)
    [13] D-short RTH       MACD=+2.0  → signal passes (RTH exempt)
    [14] E-short overnight MACD=+0.8  → HOLD (guard fires)
    [15] A-long  overnight MACD=-1.0  → HOLD (guard fires)
    [16] A-long  overnight MACD=-0.3  → signal passes (below threshold)
    [17] A-long  RTH       MACD=-2.0  → signal passes (RTH exempt)
    [18] Guard disabled (threshold=0): overnight MACD=+5.0 → signal passes
"""
from __future__ import annotations

import sys
from datetime import datetime, time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from shree.strategies.es_fifteen_min import EsFifteenMinStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> MagicMock:
    """Build a mock config.  All strategy parameters needed by __init__."""
    defaults: dict[str, Any] = {
        # Core stop/target points
        "ft_fixed_sl_points": 6.0,
        "ft_fixed_tp_points": 8.0,
        "ft_fixed_sl_points_ema9": 8.0,
        "ft_fixed_tp_points_ema9": 10.0,
        "ft_fixed_sl_points_trend": 8.0,
        "ft_fixed_tp_points_trend": 12.0,
        # Pullback / proximity
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
        # EMA9 pullback (Signal C)
        "ft_ema9_pb_enabled": False,
        "ft_ema9_pb_stop_mult": 1.2,
        "ft_ema9_pb_target_mult": 1.5,
        "ft_ema9_touch_pct": 0.0015,
        "ft_ema9_sl_atr_mult": 1.0,
        "ft_ema9_sl_floor_pts": 8.0,
        "ft_ema9_sl_ceiling_pts": 20.0,
        "ft_ema9_rr_ratio": 1.25,
        # OR break ATR-adaptive stops
        "ft_or_break_sl_atr_mult": 0.75,
        "ft_or_break_sl_floor_pts": 6.0,
        "ft_or_break_sl_ceiling_pts": 12.0,
        "ft_or_break_rr_ratio": 1.33,
        # Trend continuation (Signal F)
        "ft_trend_cont_enabled": True,
        "ft_trend_cont_stop_mult": 1.0,
        "ft_trend_cont_target_mult": 2.0,
        "ft_trend_cont_adx_min": 25.0,
        "ft_trend_cont_ema9_pct": 0.003,
        "ft_trend_cont_max_ext_pts": 30.0,
        "ft_trend_cont_gap_adx_min": 25.0,
        "ft_trend_cont_max_per_day": 5,   # high limit so tests aren't blocked by counter
        "ft_trend_sl_atr_mult": 1.0,
        "ft_trend_sl_floor_pts": 6.0,
        "ft_trend_sl_ceiling_pts": 20.0,
        "ft_trend_rr_ratio": 1.25,
        "ft_trend_exhaustion_atr_multiple": 0.0,  # disabled — don't block by exhaustion in these tests
        # London (Signal G) — disabled
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
        # Shorts enabled
        "ft_shorts_enabled": True,
        # Overnight scaling
        "ft_overnight_sl_mult": 1.2,
        "ft_overnight_tp_mult": 1.5,
        "ft_overnight_min_rr": 1.5,
        # Overnight entry-quality guards (MAR 15 2026)
        "ft_overnight_rsi_extreme_block": 35.0,  # MAR 15: raised from 30 → 35
        "ft_overnight_macd_divergence_threshold": 0.5,
        # 24h RTH / entry windows
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
    strat = EsFifteenMinStrategy(cfg)
    # Pre-seed an OR so OR signals can evaluate
    strat._or_computed = True
    strat._or_high = 6820.0
    strat._or_low = 6812.0
    return strat


def _overnight_ts() -> pd.Timestamp:
    """Return a timestamp clearly in OVERNIGHT (22:00 ET = well outside 9:30-16:00 ET)."""
    return pd.Timestamp("2026-03-12 22:00:00", tz="America/New_York")


def _rth_ts() -> pd.Timestamp:
    """Return a timestamp in core RTH (11:00 ET)."""
    return pd.Timestamp("2026-03-12 11:00:00", tz="America/New_York")


def _make_index(last_ts: pd.Timestamp, rows: int) -> pd.DatetimeIndex:
    """Build a DatetimeIndex of `rows` 15-min bars ending at last_ts."""
    start = last_ts - pd.Timedelta(minutes=15 * (rows - 1))
    return pd.date_range(start=start, periods=rows, freq="15min", tz=last_ts.tzinfo)


# ---------------------------------------------------------------------------
# DataFrame builders for specific signal conditions
# ---------------------------------------------------------------------------

def _trend_cont_short_df(
    close: float = 6720.0,
    rsi: float = 32.0,
    macd: float = -2.0,
    adx: float = 30.0,
    atr: float = 10.0,
    rows: int = 5,
    last_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame that would trigger TREND_CONT_SHORT:
      - EMA9 < EMA21 < EMA50 (full bearish stack)
      - close < EMA9
      - bearish close (close < open)
      - descending closes
      - ADX >= 25, MACD < 0, RSI in [22,55]
    """
    if last_ts is None:
        last_ts = _overnight_ts()
    ema9  = close + 5.0   # price below EMA9
    ema21 = ema9  + 8.0
    ema50 = ema21 + 8.0
    closes = [close + (rows - 1 - i) * 2 for i in range(rows)]  # descending
    opens  = [c + 3.0 for c in closes]  # bearish bars (open > close)
    highs  = [o + 1.0 for o in opens]
    lows   = [c - 1.0 for c in closes]

    idx = _make_index(last_ts, rows)
    df = pd.DataFrame({
        "close": closes,
        "open": opens,
        "high": highs,
        "low": lows,
        "volume": [1000] * rows,
        "EMA_9":  [ema9]  * rows,
        "EMA_21": [ema21] * rows,
        "EMA_50": [ema50] * rows,
        "RSI_14": [rsi]   * rows,
        "ADX_14": [adx]   * rows,
        "ATR_14": [atr]   * rows,
        "MACDhist_12_26_9": [macd] * rows,
    }, index=idx)
    return df


def _trend_cont_long_df(
    close: float = 6900.0,
    rsi: float = 71.0,
    macd: float = 3.0,
    adx: float = 30.0,
    atr: float = 10.0,
    rows: int = 5,
    last_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame that would trigger TREND_CONT_LONG:
      - EMA9 > EMA21 > EMA50 (full bullish stack)
      - close > EMA9
      - bullish close (close > open)
      - ascending closes
      - ADX >= 25, MACD > 0, RSI in [45,78]
    """
    if last_ts is None:
        last_ts = _overnight_ts()
    ema9  = close - 5.0
    ema21 = ema9  - 8.0
    ema50 = ema21 - 8.0
    closes = [close - (rows - 1 - i) * 2 for i in range(rows)]  # ascending
    opens  = [c - 3.0 for c in closes]  # bullish bars (close > open)
    highs  = [c + 1.0 for c in closes]
    lows   = [o - 1.0 for o in opens]

    idx = _make_index(last_ts, rows)
    df = pd.DataFrame({
        "close": closes,
        "open": opens,
        "high": highs,
        "low": lows,
        "volume": [1000] * rows,
        "EMA_9":  [ema9]  * rows,
        "EMA_21": [ema21] * rows,
        "EMA_50": [ema50] * rows,
        "RSI_14": [rsi]   * rows,
        "ADX_14": [adx]   * rows,
        "ATR_14": [atr]   * rows,
        "MACDhist_12_26_9": [macd] * rows,
    }, index=idx)
    return df


def _d_signal_short_df(
    close: float = 6720.0,
    macd: float = 1.2,
    rsi: float = 45.0,
    adx: float = 28.0,
    atr: float = 7.0,
    rows: int = 5,
    last_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame that would trigger Signal D (EMA21 pullback SHORT):
      - EMA21 < EMA50 (downtrend)
      - close touches or bounces off EMA21 from above
      - bearish bar (close < open)
      - ADX in [18,45]
    """
    if last_ts is None:
        last_ts = _overnight_ts()
    ema21 = close + 2.0   # price pulled back to EMA21
    ema50 = ema21 + 15.0  # EMA21 < EMA50
    ema9  = close - 3.0   # EMA9 < close (typical downtrend)
    highs = [close + 3.0] * rows
    opens = [close + 2.0] * rows  # bearish (open > close)
    lows  = [close - 2.0] * rows

    idx = _make_index(last_ts, rows)
    df = pd.DataFrame({
        "close": [close] * rows,
        "open": opens,
        "high": highs,
        "low": lows,
        "volume": [1000] * rows,
        "EMA_9":  [ema9]  * rows,
        "EMA_21": [ema21] * rows,
        "EMA_50": [ema50] * rows,
        "RSI_14": [rsi]   * rows,
        "ADX_14": [adx]   * rows,
        "ATR_14": [atr]   * rows,
        "MACDhist_12_26_9": [macd] * rows,
    }, index=idx)
    return df


def _e_signal_short_df(
    close: float = 6710.0,
    or_low: float = 6712.5,
    prev_close: float = 6713.0,
    macd: float = 1.2,
    rsi: float = 38.0,
    adx: float = 28.0,
    atr: float = 7.0,
    rows: int = 5,
    last_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame that triggers Signal E (OR breakdown SHORT):
      - close < or_low (crossed below)
      - prev_close >= or_low (just broke)
      - EMA9 < EMA21 (downtrend)
    """
    if last_ts is None:
        last_ts = _overnight_ts()
    ema9  = close - 2.0
    ema21 = close + 5.0
    ema50 = ema21 + 10.0
    highs = [close + 2.0] * rows
    opens = [close + 1.0] * rows
    lows  = [close - 2.0] * rows
    closes = [prev_close] * (rows - 1) + [close]

    idx = _make_index(last_ts, rows)
    df = pd.DataFrame({
        "close": closes,
        "open": opens,
        "high": highs,
        "low": lows,
        "volume": [1000] * rows,
        "EMA_9":  [ema9]  * rows,
        "EMA_21": [ema21] * rows,
        "EMA_50": [ema50] * rows,
        "RSI_14": [rsi]   * rows,
        "ADX_14": [adx]   * rows,
        "ATR_14": [atr]   * rows,
        "MACDhist_12_26_9": [macd] * rows,
    }, index=idx)
    return df


def _a_signal_long_df(
    close: float = 6820.0,
    macd: float = -1.2,
    rsi: float = 52.0,
    adx: float = 28.0,
    atr: float = 7.0,
    rows: int = 5,
    last_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame that would trigger Signal A (EMA21 pullback LONG):
      - EMA21 > EMA50 (uptrend)
      - close touches EMA21 from above
      - bullish bar (close > open)
      - ADX in [18,45]
    """
    if last_ts is None:
        last_ts = _overnight_ts()
    ema21 = close - 2.0   # price pulled back to EMA21
    ema50 = ema21 - 15.0  # EMA21 > EMA50
    ema9  = close + 3.0
    lows  = [close - 3.0] * rows  # low touches EMA21 band
    opens = [close - 2.0] * rows  # bullish (close > open)
    highs = [close + 2.0] * rows

    idx = _make_index(last_ts, rows)
    df = pd.DataFrame({
        "close": [close] * rows,
        "open": opens,
        "high": highs,
        "low": lows,
        "volume": [1000] * rows,
        "EMA_9":  [ema9]  * rows,
        "EMA_21": [ema21] * rows,
        "EMA_50": [ema50] * rows,
        "RSI_14": [rsi]   * rows,
        "ADX_14": [adx]   * rows,
        "ATR_14": [atr]   * rows,
        "MACDhist_12_26_9": [macd] * rows,
    }, index=idx)
    return df


def _call_generate(strat: EsFifteenMinStrategy, df: pd.DataFrame,
                   prev_close: float | None = None):
    """Call strategy.generate() and return the Signal.
    The bar timestamp is taken from df.index[-1] by the strategy.
    """
    if prev_close is not None:
        strat._prev_close = prev_close
    return strat.generate(df)


# ===========================================================================
# Guard 1: RSI extreme — TREND_CONT overnight
# ===========================================================================

class TestRsiExtremeGuard:

    # [1] RSI=30 on TREND_CONT_SHORT overnight → HOLD
    def test_trend_cont_short_rsi_30_overnight_blocked(self):
        strat = _make_strategy()
        df = _trend_cont_short_df(rsi=30.0, macd=-4.44, atr=10.65, last_ts=_overnight_ts())
        sig = _call_generate(strat, df)
        assert sig.action == "HOLD", (
            f"Expected HOLD (RSI=30 overnight), got {sig.action} | {sig.metadata}"
        )
        reason = sig.metadata.get("reason", "")
        assert "ON_RSI_EXTREME" in reason or sig.confidence == 0.0

    # [2] RSI=32 on TREND_CONT_SHORT overnight → HOLD
    def test_trend_cont_short_rsi_32_overnight_blocked(self):
        strat = _make_strategy()
        df = _trend_cont_short_df(rsi=32.0, macd=-0.38, atr=6.82, last_ts=_overnight_ts())
        sig = _call_generate(strat, df)
        assert sig.action == "HOLD", (
            f"Expected HOLD (RSI=32 overnight), got {sig.action} | {sig.metadata}"
        )

    # [3] RSI=38 on TREND_CONT_SHORT overnight → signal passes (above threshold=35)
    def test_trend_cont_short_rsi_38_overnight_passes(self):
        strat = _make_strategy()
        df = _trend_cont_short_df(rsi=38.0, macd=-2.0, atr=10.0, last_ts=_overnight_ts())
        sig = _call_generate(strat, df)
        # RSI=38 > threshold=35, guard does NOT block
        assert sig.action in ("SELL", "HOLD"), "signal_f_short should be unaffected by RSI guard"
        # If it returns HOLD it must NOT be due to RSI guard
        if sig.action == "HOLD":
            reason = sig.metadata.get("reason", "")
            assert "ON_RSI_EXTREME" not in reason, f"RSI=38 should not trigger RSI extreme guard"

    # [4] RSI=30 on TREND_CONT_SHORT during RTH → signal passes (RTH exempt)
    def test_trend_cont_short_rsi_30_rth_not_blocked(self):
        strat = _make_strategy()
        df = _trend_cont_short_df(rsi=30.0, macd=-4.44, atr=10.65, last_ts=_rth_ts())
        sig = _call_generate(strat, df)
        # Guard only applies outside core RTH
        if sig.action == "HOLD":
            reason = sig.metadata.get("reason", "")
            assert "ON_RSI_EXTREME" not in reason, (
                "RSI extreme guard must NOT fire during core RTH"
            )

    # [5] RSI=71 on TREND_CONT_LONG overnight → HOLD
    def test_trend_cont_long_rsi_71_overnight_blocked(self):
        strat = _make_strategy()
        df = _trend_cont_long_df(rsi=71.0, macd=3.0, adx=30.0, atr=10.0, last_ts=_overnight_ts())
        sig = _call_generate(strat, df)
        assert sig.action == "HOLD", (
            f"Expected HOLD (RSI=71 > 70 overnight), got {sig.action} | {sig.metadata}"
        )

    # [6] RSI=68 on TREND_CONT_LONG overnight → signal passes
    def test_trend_cont_long_rsi_68_overnight_passes(self):
        strat = _make_strategy()
        df = _trend_cont_long_df(rsi=68.0, macd=3.0, adx=30.0, atr=10.0, last_ts=_overnight_ts())
        sig = _call_generate(strat, df)
        if sig.action == "HOLD":
            reason = sig.metadata.get("reason", "")
            assert "ON_RSI_EXTREME" not in reason

    # [7] RSI=72 on TREND_CONT_LONG during RTH → signal passes (RTH exempt)
    def test_trend_cont_long_rsi_72_rth_not_blocked(self):
        strat = _make_strategy()
        df = _trend_cont_long_df(rsi=72.0, macd=3.0, adx=30.0, atr=10.0, last_ts=_rth_ts())
        sig = _call_generate(strat, df)
        if sig.action == "HOLD":
            reason = sig.metadata.get("reason", "")
            assert "ON_RSI_EXTREME" not in reason

    # [8] Guard disabled (threshold=0): RSI=28 overnight → signal passes
    def test_rsi_extreme_guard_disabled(self):
        strat = _make_strategy(ft_overnight_rsi_extreme_block=0.0)
        df = _trend_cont_short_df(rsi=28.0, macd=-3.0, atr=10.0, last_ts=_overnight_ts())
        sig = _call_generate(strat, df)
        if sig.action == "HOLD":
            reason = sig.metadata.get("reason", "")
            assert "ON_RSI_EXTREME" not in reason, "Guard disabled, must not fire"


# ===========================================================================
# Guard 2: MACD divergence — D/E short and A/C long overnight
# ===========================================================================

class TestMacdDivergenceGuard:

    # [9] D-short overnight MACD=+1.03 → HOLD
    def test_signal_d_short_macd_positive_overnight_blocked(self):
        strat = _make_strategy()
        df = _d_signal_short_df(macd=1.03, adx=28.0, atr=7.0, last_ts=_overnight_ts())
        sig = _call_generate(strat, df)
        assert sig.action == "HOLD", (
            f"Expected HOLD (D-short MACD=+1.03 overnight), got {sig.action} | {sig.metadata}"
        )

    # [10] D-short overnight MACD=+1.63 → HOLD
    def test_signal_d_short_macd_1_63_overnight_blocked(self):
        strat = _make_strategy()
        df = _d_signal_short_df(macd=1.63, adx=27.5, atr=6.0, last_ts=_overnight_ts())
        sig = _call_generate(strat, df)
        assert sig.action == "HOLD", (
            f"Expected HOLD (D-short MACD=+1.63 overnight), got {sig.action} | {sig.metadata}"
        )

    # [11] D-short overnight MACD=+0.3 → signal passes (below threshold 0.5)
    def test_signal_d_short_macd_0_3_overnight_passes(self):
        strat = _make_strategy()
        df = _d_signal_short_df(macd=0.3, adx=28.0, atr=7.0, last_ts=_overnight_ts())
        sig = _call_generate(strat, df)
        if sig.action == "HOLD":
            reason = sig.metadata.get("reason", "")
            assert "ON_MACD_DIVERGE" not in reason, "MACD=+0.3 is below threshold=0.5"

    # [12] D-short overnight MACD=-1.0 → signal passes (momentum aligned)
    def test_signal_d_short_macd_negative_overnight_passes(self):
        strat = _make_strategy()
        df = _d_signal_short_df(macd=-1.0, adx=28.0, atr=7.0, last_ts=_overnight_ts())
        sig = _call_generate(strat, df)
        # MACD negative on short = aligned, guard must not fire
        if sig.action == "HOLD":
            reason = sig.metadata.get("reason", "")
            assert "ON_MACD_DIVERGE" not in reason

    # [13] D-short RTH MACD=+2.0 → signal passes (RTH exempt)
    def test_signal_d_short_macd_positive_rth_not_blocked(self):
        strat = _make_strategy()
        df = _d_signal_short_df(macd=2.0, adx=28.0, atr=7.0, last_ts=_rth_ts())
        sig = _call_generate(strat, df)
        if sig.action == "HOLD":
            reason = sig.metadata.get("reason", "")
            assert "ON_MACD_DIVERGE" not in reason, "MACD divergence guard must NOT fire during RTH"

    # [14] E-short overnight MACD=+0.8 → HOLD
    def test_signal_e_short_macd_positive_overnight_blocked(self):
        strat = _make_strategy()
        strat._or_high = 6717.25
        strat._or_low  = 6712.5
        df = _e_signal_short_df(
            close=6710.0, or_low=6712.5, prev_close=6713.0,
            macd=0.8, adx=28.0, atr=7.0,
            last_ts=_overnight_ts(),
        )
        strat._prev_close = 6713.0
        sig = _call_generate(strat, df)
        assert sig.action == "HOLD", (
            f"Expected HOLD (E-short MACD=+0.8 overnight), got {sig.action} | {sig.metadata}"
        )

    # [15] A-long overnight MACD=-1.0 → HOLD
    def test_signal_a_long_macd_negative_overnight_blocked(self):
        strat = _make_strategy()
        df = _a_signal_long_df(macd=-1.0, adx=28.0, atr=7.0, rsi=52.0, last_ts=_overnight_ts())
        sig = _call_generate(strat, df)
        assert sig.action == "HOLD", (
            f"Expected HOLD (A-long MACD=-1.0 overnight), got {sig.action} | {sig.metadata}"
        )

    # [16] A-long overnight MACD=-0.3 → signal passes (below threshold)
    def test_signal_a_long_macd_small_negative_passes(self):
        strat = _make_strategy()
        df = _a_signal_long_df(macd=-0.3, adx=28.0, atr=7.0, rsi=52.0, last_ts=_overnight_ts())
        sig = _call_generate(strat, df)
        if sig.action == "HOLD":
            reason = sig.metadata.get("reason", "")
            assert "ON_MACD_DIVERGE" not in reason, "MACD=-0.3 is within threshold=0.5"

    # [17] A-long RTH MACD=-2.0 → signal passes (RTH exempt)
    def test_signal_a_long_macd_negative_rth_not_blocked(self):
        strat = _make_strategy()
        df = _a_signal_long_df(macd=-2.0, adx=28.0, atr=7.0, rsi=52.0, last_ts=_rth_ts())
        sig = _call_generate(strat, df)
        if sig.action == "HOLD":
            reason = sig.metadata.get("reason", "")
            assert "ON_MACD_DIVERGE" not in reason, "MACD divergence guard must NOT fire during RTH"

    # [18] Guard disabled (threshold=0): D-short overnight MACD=+5.0 → signal passes
    def test_macd_divergence_guard_disabled(self):
        strat = _make_strategy(ft_overnight_macd_divergence_threshold=0.0)
        df = _d_signal_short_df(macd=5.0, adx=28.0, atr=7.0, last_ts=_overnight_ts())
        sig = _call_generate(strat, df)
        if sig.action == "HOLD":
            reason = sig.metadata.get("reason", "")
            assert "ON_MACD_DIVERGE" not in reason, "Guard disabled, must not fire"

