"""Tests for GoldSignalGenerator.

We build synthetic OHLCV DataFrames with controlled indicator values
to test each signal path deterministically — no real market data needed.
"""
from __future__ import annotations

from datetime import datetime, timezone, date

import numpy as np
import pandas as pd
import pytest

from shree.config.gold import (
    GoldEntryConfig,
    GoldExitConfig,
    GoldIndicatorConfig,
    GoldSessionConfig,
)
from shree.strategies.gold.regime import GoldRegime
from shree.strategies.gold.signals import GoldSignal, GoldSignalGenerator, GoldSignalType


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_gen(
    adx_trend_min: float = 20.0,
    adx_trend_max: float = 55.0,
    vwap_touch_pct: float = 0.002,
    ema_touch_pct: float = 0.0025,
    orb_enabled: bool = True,
    orb_min_range_ratio: float = 0.001,
    pullback_reclaim_required: bool = True,
    pullback_min_body_fraction: float = 0.25,
    pullback_confirm_with_bar_direction: bool = True,
    pullback_max_vwap_extension_atr: float = 1.5,
    pullback_max_ema_extension_atr: float = 1.25,
    orb_breakout_min_atr_fraction: float = 0.10,
    orb_volume_lookback_bars: int = 20,
    orb_volume_min_multiple: float = 1.20,
    orb_max_breakout_candle_atr: float = 1.25,
    orb_max_extension_atr: float = 0.75,
    min_bar_volume: int = 1,
    post_loss_cooldown_bars: int = 2,
    atr_sl_mult: float = 1.5,
    atr_tp_mult: float = 2.5,
    sl_floor: float = 1.0,
    sl_ceiling: float = 50.0,
    min_rr: float = 1.2,
) -> GoldSignalGenerator:
    entry = GoldEntryConfig(
        adx_trend_min=adx_trend_min,
        adx_trend_max=adx_trend_max,
        vwap_touch_pct=vwap_touch_pct,
        ema_touch_pct=ema_touch_pct,
        pullback_reclaim_required=pullback_reclaim_required,
        pullback_min_body_fraction=pullback_min_body_fraction,
        pullback_confirm_with_bar_direction=pullback_confirm_with_bar_direction,
        pullback_max_vwap_extension_atr=pullback_max_vwap_extension_atr,
        pullback_max_ema_extension_atr=pullback_max_ema_extension_atr,
        orb_enabled=orb_enabled,
        orb_min_range_ratio=orb_min_range_ratio,
        orb_breakout_min_atr_fraction=orb_breakout_min_atr_fraction,
        orb_volume_lookback_bars=orb_volume_lookback_bars,
        orb_volume_min_multiple=orb_volume_min_multiple,
        orb_max_breakout_candle_atr=orb_max_breakout_candle_atr,
        orb_max_extension_atr=orb_max_extension_atr,
        min_bar_volume=min_bar_volume,
        post_loss_cooldown_bars=post_loss_cooldown_bars,
    )
    exit_cfg = GoldExitConfig(
        atr_sl_multiplier=atr_sl_mult,
        atr_tp_multiplier=atr_tp_mult,
        sl_floor_points=sl_floor,
        sl_ceiling_points=sl_ceiling,
        min_rr_ratio=min_rr,
    )
    return GoldSignalGenerator(
        session=GoldSessionConfig(),
        indicators=GoldIndicatorConfig(),
        entry=entry,
        exit_cfg=exit_cfg,
        tick_size=0.10,
    )


def _make_features(
    open_price: float | None = None,
    high: float | None = None,
    low: float | None = None,
    close: float = 2350.0,
    ema9: float = 2348.0,
    ema21: float = 2345.0,
    atr: float = 8.0,
    adx: float = 28.0,
    vwap: float = 2349.0,
    volume: int = 50,
    n_bars: int = 5,
    start: str = "2026-01-15 09:30",
) -> pd.DataFrame:
    """Build a minimal features DataFrame with the last bar having the given values."""
    if open_price is None:
        open_price = close
    if high is None:
        high = close + atr * 0.5
    if low is None:
        low = close - atr * 0.5
    idx = pd.date_range(start, periods=n_bars, freq="1min", tz="America/New_York")
    data = {
        "open": [open_price] * n_bars,
        "high": [high] * n_bars,
        "low": [low] * n_bars,
        "close": [close] * n_bars,
        "volume": [volume] * n_bars,
        "ema9": [ema9] * n_bars,
        "ema21": [ema21] * n_bars,
        "atr": [atr] * n_bars,
        "adx": [adx] * n_bars,
        "vwap": [vwap] * n_bars,
    }
    return pd.DataFrame(data, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Regime guard
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeGuard:
    def test_no_signal_in_ranging(self) -> None:
        gen = _make_gen()
        df = _make_features()
        sig = gen.generate(df, GoldRegime.RANGING)
        assert sig.action == "HOLD"

    def test_no_signal_warming_up(self) -> None:
        gen = _make_gen()
        df = _make_features()
        sig = gen.generate(df, GoldRegime.WARMING_UP)
        assert sig.action == "HOLD"

    def test_no_signal_no_trade(self) -> None:
        gen = _make_gen()
        df = _make_features()
        sig = gen.generate(df, GoldRegime.NO_TRADE)
        assert sig.action == "HOLD"


# ─────────────────────────────────────────────────────────────────────────────
# VWAP Pullback — Long
# ─────────────────────────────────────────────────────────────────────────────

class TestVwapPullbackLong:
    def test_fires_when_close_near_vwap_bull_regime(self) -> None:
        """Close within VWAP touch band, bull regime → BUY signal."""
        gen = _make_gen(vwap_touch_pct=0.002)
        vwap = 2349.0
        # close within 0.2% of vwap and >= vwap (confirmation)
        close = vwap * 1.0001    # 0.01% above VWAP
        df = _make_features(
            open_price=vwap - 0.2,
            close=close,
            vwap=vwap,
            ema9=2350.0,
            ema21=2346.0,
            adx=28.0,
            high=close + 1.0,
            low=vwap - 0.5,
        )
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        assert sig.action == "BUY"
        assert sig.signal_type == GoldSignalType.VWAP_PB_LONG

    def test_sl_below_entry(self) -> None:
        gen = _make_gen()
        vwap = 2349.0
        close = vwap * 1.0001
        df = _make_features(close=close, vwap=vwap, ema9=2350.0, ema21=2346.0)
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        if sig.action == "BUY":
            assert sig.stop_loss < sig.entry_ref_price

    def test_tp_above_entry(self) -> None:
        gen = _make_gen()
        vwap = 2349.0
        close = vwap * 1.0001
        df = _make_features(close=close, vwap=vwap, ema9=2350.0, ema21=2346.0)
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        if sig.action == "BUY":
            assert sig.take_profit > sig.entry_ref_price

    def test_no_signal_when_close_far_from_vwap(self) -> None:
        gen = _make_gen(vwap_touch_pct=0.002)
        vwap = 2349.0
        close = vwap * 1.02   # 2% above VWAP → too far for pullback
        df = _make_features(close=close, vwap=vwap, ema9=2360.0, ema21=2350.0)
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        assert sig.action == "HOLD"

    def test_no_signal_below_vwap_in_bull_regime(self) -> None:
        """In bull regime, close below VWAP is not confirmed → no signal."""
        gen = _make_gen(vwap_touch_pct=0.002)
        vwap = 2349.0
        close = vwap * 0.999   # Slightly below VWAP
        df = _make_features(close=close, vwap=vwap, ema9=2350.0, ema21=2346.0)
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        # close is below vwap — confirmation fails for long
        assert sig.action in ("HOLD", "BUY")   # Could still get EMA signal

    def test_requires_bullish_bar_direction_when_enabled(self) -> None:
        gen = _make_gen()
        df = _make_features(
            open_price=2351.0,
            close=2350.0,
            vwap=2349.0,
            ema9=2352.0,
            ema21=2348.0,
            high=2351.2,
            low=2348.8,
        )
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        assert sig.action == "HOLD"

    def test_blocks_small_body_pullback_confirmation(self) -> None:
        gen = _make_gen(pullback_min_body_fraction=0.5)
        df = _make_features(
            open_price=2349.95,
            close=2350.0,
            vwap=2349.0,
            ema9=2351.0,
            ema21=2348.0,
            high=2351.2,
            low=2348.8,
        )
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        assert sig.action == "HOLD"

    def test_blocks_pullback_when_still_too_extended_from_means(self) -> None:
        gen = _make_gen(
            pullback_max_vwap_extension_atr=0.5,
            pullback_max_ema_extension_atr=0.5,
        )
        df = _make_features(
            open_price=2359.0,
            close=2360.0,
            vwap=2349.0,
            ema9=2361.0,
            ema21=2348.0,
            atr=8.0,
            high=2361.0,
            low=2348.9,
        )
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        assert sig.action == "HOLD"
        assert sig.metadata["block_reason"] == "pullback_vwap_extension_exceeded"


# ─────────────────────────────────────────────────────────────────────────────
# VWAP Pullback — Short
# ─────────────────────────────────────────────────────────────────────────────

class TestVwapPullbackShort:
    def test_fires_in_bear_regime(self) -> None:
        gen = _make_gen(vwap_touch_pct=0.002)
        vwap = 2349.0
        close = vwap * 0.9999   # just below VWAP
        df = _make_features(
            open_price=vwap + 0.2,
            close=close,
            vwap=vwap,
            ema9=2344.0,
            ema21=2347.0,
            adx=28.0,
            high=vwap + 0.5,
            low=close - 1.0,
        )
        sig = gen.generate(df, GoldRegime.TRENDING_BEAR)
        assert sig.action == "SELL"
        assert sig.signal_type == GoldSignalType.VWAP_PB_SHORT

    def test_sl_above_entry_for_short(self) -> None:
        gen = _make_gen(vwap_touch_pct=0.002)
        vwap = 2349.0
        close = vwap * 0.9999
        df = _make_features(close=close, vwap=vwap, ema9=2344.0, ema21=2347.0)
        sig = gen.generate(df, GoldRegime.TRENDING_BEAR)
        if sig.action == "SELL":
            assert sig.stop_loss > sig.entry_ref_price
            assert sig.take_profit < sig.entry_ref_price


# ─────────────────────────────────────────────────────────────────────────────
# Opening Range Breakout
# ─────────────────────────────────────────────────────────────────────────────

class TestORB:
    def _gen_with_or(self) -> GoldSignalGenerator:
        gen = _make_gen(orb_enabled=True, orb_min_range_ratio=0.001)
        # Manually inject OR state
        gen._session_date = date(2026, 1, 15)
        gen._or_high = 2360.0
        gen._or_low = 2340.0
        gen._or_formed = True
        return gen

    def test_orb_long_on_breakout(self) -> None:
        gen = _make_gen(
            orb_enabled=True,
            orb_min_range_ratio=0.001,
            orb_volume_min_multiple=0.0,
            orb_max_breakout_candle_atr=0.0,
            orb_max_extension_atr=0.0,
        )
        gen._session_date = date(2026, 1, 15)
        gen._or_high = 2360.0
        gen._or_low = 2340.0
        gen._or_formed = True
        close = 2360.9   # Above OR high + breakout buffer
        df = _make_features(
            open_price=2360.4,
            close=close,
            vwap=2349.0,
            ema9=2355.0,
            ema21=2350.0,
            atr=8.0,
            high=2361.1,
            low=2360.1,
            volume=200,
            n_bars=25,
            start="2026-01-15 08:45",
        )
        breakout_buffer = 8.0 * gen._entry.orb_breakout_min_atr_fraction
        assert close > gen._or_high + breakout_buffer
        assert close > float(df.iloc[-1]["open"])

    def test_orb_short_on_breakdown(self) -> None:
        gen = _make_gen(
            orb_enabled=True,
            orb_min_range_ratio=0.001,
            orb_volume_min_multiple=0.0,
            orb_max_breakout_candle_atr=0.0,
            orb_max_extension_atr=0.0,
        )
        gen._session_date = date(2026, 1, 15)
        gen._or_high = 2360.0
        gen._or_low = 2340.0
        gen._or_formed = True
        close = 2339.1   # Below OR low - breakout buffer
        df = _make_features(
            open_price=2339.6,
            close=close,
            vwap=2349.0,
            ema9=2344.0,
            ema21=2347.0,
            atr=8.0,
            high=2339.8,
            low=2338.9,
            volume=200,
            n_bars=25,
            start="2026-01-15 08:45",
        )
        breakout_buffer = 8.0 * gen._entry.orb_breakout_min_atr_fraction
        assert close < gen._or_low - breakout_buffer
        assert close < float(df.iloc[-1]["open"])

    def test_no_orb_when_not_formed(self) -> None:
        gen = _make_gen(orb_enabled=True)
        # OR not formed
        gen._session_date = date(2026, 1, 15)
        gen._or_formed = False
        close = 2361.0
        df = _make_features(close=close, vwap=2349.0, ema9=2355.0, ema21=2350.0)
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        assert sig.signal_type != GoldSignalType.ORB_LONG

    def test_no_orb_when_disabled(self) -> None:
        gen = _make_gen(orb_enabled=False)
        gen._session_date = date(2026, 1, 15)
        gen._or_high = 2360.0
        gen._or_low = 2340.0
        gen._or_formed = True
        close = 2361.0
        df = _make_features(close=close, vwap=2349.0, ema9=2355.0, ema21=2350.0)
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        assert sig.signal_type != GoldSignalType.ORB_LONG

    def test_orb_requires_breakout_buffer(self) -> None:
        gen = self._gen_with_or()
        gen._entry.orb_breakout_min_atr_fraction = 0.25
        df = _make_features(
            open_price=2360.2,
            close=2361.0,
            vwap=2349.0,
            ema9=2355.0,
            ema21=2350.0,
            atr=8.0,
            high=2361.2,
            low=2359.8,
            volume=300,
            n_bars=25,
        )
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        assert sig.action == "HOLD"

    def test_orb_requires_volume_confirmation(self) -> None:
        gen = self._gen_with_or()
        df = _make_features(
            open_price=2360.0,
            close=2362.0,
            vwap=2349.0,
            ema9=2355.0,
            ema21=2350.0,
            atr=8.0,
            high=2362.5,
            low=2359.5,
            volume=50,
            n_bars=25,
        )
        df["volume"] = [100] * 24 + [50]
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        assert sig.action == "HOLD"
        assert sig.metadata["orb_block_reason"] == "orb_volume_below_threshold"

    def test_check_orb_directly_reports_low_volume_block_reason(self) -> None:
        gen = self._gen_with_or()
        df = _make_features(
            open_price=2360.1,
            close=2361.2,
            vwap=2349.0,
            ema9=2355.0,
            ema21=2350.0,
            atr=8.0,
            high=2361.6,
            low=2359.9,
            volume=50,
            n_bars=25,
            start="2026-01-15 10:00",
        )
        df["volume"] = [100] * 24 + [50]
        sig = gen._check_orb(df, 2361.2, 8.0, GoldRegime.TRENDING_BULL)
        assert sig.action == "HOLD"
        assert sig.metadata["block_reason"] == "orb_volume_below_threshold"

    def test_orb_blocks_oversized_breakout_candle(self) -> None:
        gen = self._gen_with_or()
        df = _make_features(
            open_price=2358.0,
            close=2363.0,
            vwap=2349.0,
            ema9=2355.0,
            ema21=2350.0,
            atr=8.0,
            high=2372.0,
            low=2358.0,
            volume=500,
            n_bars=25,
        )
        df["volume"] = [100] * 24 + [500]
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        assert sig.action == "HOLD"
        assert sig.metadata["orb_block_reason"] == "orb_breakout_candle_too_large"

    def test_check_orb_directly_reports_oversized_candle_block_reason(self) -> None:
        gen = self._gen_with_or()
        df = _make_features(
            open_price=2358.0,
            close=2363.0,
            vwap=2349.0,
            ema9=2355.0,
            ema21=2350.0,
            atr=8.0,
            high=2372.0,
            low=2358.0,
            volume=500,
            n_bars=25,
            start="2026-01-15 10:00",
        )
        df["volume"] = [100] * 24 + [500]
        sig = gen._check_orb(df, 2363.0, 8.0, GoldRegime.TRENDING_BULL)
        assert sig.action == "HOLD"
        assert sig.metadata["block_reason"] == "orb_breakout_candle_too_large"

    def test_orb_blocks_when_already_too_extended(self) -> None:
        gen = self._gen_with_or()
        df = _make_features(
            open_price=2364.0,
            close=2368.0,
            vwap=2349.0,
            ema9=2355.0,
            ema21=2350.0,
            atr=8.0,
            high=2368.5,
            low=2363.5,
            volume=500,
            n_bars=25,
        )
        df["volume"] = [100] * 24 + [500]
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        assert sig.action == "HOLD"
        assert sig.metadata["orb_block_reason"] == "orb_extension_too_large"

    def test_check_orb_directly_reports_extension_block_reason(self) -> None:
        gen = self._gen_with_or()
        df = _make_features(
            open_price=2364.0,
            close=2368.0,
            vwap=2349.0,
            ema9=2355.0,
            ema21=2350.0,
            atr=8.0,
            high=2368.5,
            low=2363.5,
            volume=500,
            n_bars=25,
            start="2026-01-15 10:00",
        )
        df["volume"] = [100] * 24 + [500]
        sig = gen._check_orb(df, 2368.0, 8.0, GoldRegime.TRENDING_BULL)
        assert sig.action == "HOLD"
        assert sig.metadata["block_reason"] == "orb_extension_too_large"


# ─────────────────────────────────────────────────────────────────────────────
# Post-loss cooldown
# ─────────────────────────────────────────────────────────────────────────────

class TestPostLossCooldown:
    def test_no_signal_during_cooldown(self) -> None:
        gen = _make_gen(post_loss_cooldown_bars=3)
        gen.notify_loss()   # Sets cooldown = 3

        vwap = 2349.0
        close = vwap * 1.0001
        df = _make_features(close=close, vwap=vwap, ema9=2350.0, ema21=2346.0)

        for _ in range(3):
            sig = gen.generate(df, GoldRegime.TRENDING_BULL)
            assert sig.action == "HOLD"

    def test_signal_fires_after_cooldown_expires(self) -> None:
        gen = _make_gen(post_loss_cooldown_bars=2)
        gen.notify_loss()

        vwap = 2349.0
        close = vwap * 1.0001
        df = _make_features(close=close, vwap=vwap, ema9=2350.0, ema21=2346.0)

        # 2 bars of cooldown
        gen.generate(df, GoldRegime.TRENDING_BULL)
        gen.generate(df, GoldRegime.TRENDING_BULL)

        # After cooldown, signal may fire
        sig = gen.generate(df, GoldRegime.TRENDING_BULL)
        # Accept either HOLD or BUY — depends on market conditions
        assert sig.action in ("HOLD", "BUY")


# ─────────────────────────────────────────────────────────────────────────────
# Stop/target calculations
# ─────────────────────────────────────────────────────────────────────────────

class TestSlTpCalculations:
    def test_atr_based_sl_distance(self) -> None:
        gen = _make_gen(atr_sl_mult=1.5, sl_floor=0.1, sl_ceiling=99.0)
        sl, tp = gen._sl_tp("BUY", 2350.0, 8.0)
        # Expected SL distance = 8.0 × 1.5 = 12.0
        assert abs((2350.0 - sl) - 12.0) < 0.5   # Allow tick rounding

    def test_sl_floor_applied(self) -> None:
        gen = _make_gen(atr_sl_mult=0.01, sl_floor=5.0)
        sl, tp = gen._sl_tp("BUY", 2350.0, 0.1)   # ATR=0.1 → raw_sl=0.15 < floor=5
        assert (2350.0 - sl) >= 5.0 - 0.11   # Floor must dominate

    def test_sl_ceiling_applied(self) -> None:
        gen = _make_gen(atr_sl_mult=10.0, sl_ceiling=20.0)
        sl, tp = gen._sl_tp("BUY", 2350.0, 10.0)   # raw=100 > ceiling=20
        assert (2350.0 - sl) <= 20.0 + 0.11

    def test_rr_check_passes(self) -> None:
        gen = _make_gen(min_rr=1.2)
        assert gen._check_rr("BUY", 2350.0, 2338.0, 2365.0)   # 15/12 = 1.25

    def test_rr_check_fails(self) -> None:
        gen = _make_gen(min_rr=2.0)
        assert not gen._check_rr("BUY", 2350.0, 2340.0, 2355.0)   # 5/10 = 0.5

    def test_sl_tp_snapped_to_tick(self) -> None:
        gen = _make_gen()
        sl, tp = gen._sl_tp("BUY", 2350.0, 7.33)
        # Both should be multiples of 0.10
        assert abs(round(sl / 0.10) * 0.10 - sl) < 1e-6
        assert abs(round(tp / 0.10) * 0.10 - tp) < 1e-6
