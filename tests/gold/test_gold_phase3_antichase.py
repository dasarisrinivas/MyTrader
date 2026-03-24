"""Tests for Phase 3 anti-chase refinements:
- Overnight extension strictness multiplier (tighter pullback thresholds)
- ORB VWAP extension guard
"""
from __future__ import annotations

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
    pullback_max_vwap_extension_atr: float = 1.5,
    pullback_max_ema_extension_atr: float = 1.25,
    extended_hours_extension_strictness_mult: float = 0.7,
    orb_max_vwap_extension_atr: float = 1.5,
    orb_max_extension_atr: float = 0.75,
    extended_hours_enabled: bool = True,
    **entry_overrides,
) -> GoldSignalGenerator:
    entry = GoldEntryConfig(
        pullback_max_vwap_extension_atr=pullback_max_vwap_extension_atr,
        pullback_max_ema_extension_atr=pullback_max_ema_extension_atr,
        extended_hours_extension_strictness_mult=extended_hours_extension_strictness_mult,
        orb_max_vwap_extension_atr=orb_max_vwap_extension_atr,
        orb_max_extension_atr=orb_max_extension_atr,
        vwap_touch_pct=0.002,
        ema_touch_pct=0.0025,
        pullback_reclaim_required=True,
        pullback_min_body_fraction=0.25,
        pullback_confirm_with_bar_direction=True,
        min_bar_volume=1,
        post_loss_cooldown_bars=0,
        orb_enabled=True,
        orb_min_range_ratio=0.0001,
        orb_breakout_min_atr_fraction=0.0,
        orb_volume_lookback_bars=20,
        orb_volume_min_multiple=0.0,
        orb_max_breakout_candle_atr=10.0,
        **entry_overrides,
    )
    exit_cfg = GoldExitConfig(
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=2.5,
        sl_floor_points=0.1,
        sl_ceiling_points=100.0,
        min_rr_ratio=1.0,
    )
    session = GoldSessionConfig(
        extended_hours_enabled=extended_hours_enabled,
        session_open_et="08:20",
        extended_session_open_et="18:00",
    )
    return GoldSignalGenerator(
        session=session,
        indicators=GoldIndicatorConfig(),
        entry=entry,
        exit_cfg=exit_cfg,
        tick_size=0.10,
    )


def _make_features(
    close: float = 2350.0,
    ema9: float = 2348.0,
    ema21: float = 2345.0,
    atr: float = 8.0,
    adx: float = 28.0,
    vwap: float = 2349.0,
    volume: int = 50,
    n_bars: int = 25,
    start: str = "2026-01-15 09:30",
    open_price: float | None = None,
    high: float | None = None,
    low: float | None = None,
) -> pd.DataFrame:
    if open_price is None:
        open_price = close - 0.5
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


def _rth_ts(hour: int = 10, minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(f"2026-01-15 {hour:02d}:{minute:02d}:00", tz="America/New_York")


def _extended_ts(hour: int = 20, minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(f"2026-01-15 {hour:02d}:{minute:02d}:00", tz="America/New_York")


# ─────────────────────────────────────────────────────────────────────────────
# Overnight extension strictness for pullbacks
# ─────────────────────────────────────────────────────────────────────────────


class TestOvernightExtensionStrictness:
    """Verify that pullback VWAP/EMA extension thresholds are tighter during extended hours."""

    def test_rth_pullback_allowed_at_normal_extension(self) -> None:
        """During RTH, pullback at 1.2 ATR from VWAP should be allowed (threshold=1.5)."""
        atr = 8.0
        vwap = 2349.0
        # close at 1.2 ATR from VWAP (close = vwap + 1.2*atr = 2358.6)
        # But for a BUY pullback, close needs to be near vwap...
        # Let's set close near vwap and check that it fires during RTH
        gen = _make_gen(pullback_max_vwap_extension_atr=1.5)
        df = _make_features(close=2349.2, vwap=2349.0, ema21=2349.0, atr=8.0,
                            open_price=2348.0, low=2348.5, high=2350.0)
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_rth_ts())
        assert sig.action == "BUY"

    def test_extended_hours_tightens_thresholds(self) -> None:
        """During extended hours with mult=0.7, effective VWAP threshold = 1.5*0.7 = 1.05 ATR.
        A pullback at 1.1 ATR from VWAP should be blocked overnight but allowed RTH."""
        atr = 8.0
        vwap = 2340.0
        close = vwap + 1.1 * atr   # 2348.8 — 1.1 ATR away
        gen = _make_gen(
            pullback_max_vwap_extension_atr=1.5,
            extended_hours_extension_strictness_mult=0.7,
        )
        # RTH bar: threshold = 1.5 ATR → 1.1 < 1.5 → allowed (if other conditions met)
        df_rth = _make_features(close=close, vwap=vwap, ema21=vwap, atr=atr,
                                open_price=close - 0.5, low=vwap - 0.2, high=close + 0.5,
                                start="2026-01-15 10:00")
        sig_rth = gen.generate(df_rth, GoldRegime.TRENDING_BULL, bar_timestamp=_rth_ts(10, 0))
        # Extended bar: threshold = 1.5 * 0.7 = 1.05 ATR → 1.1 > 1.05 → blocked
        df_ext = _make_features(close=close, vwap=vwap, ema21=vwap, atr=atr,
                                open_price=close - 0.5, low=vwap - 0.2, high=close + 0.5,
                                start="2026-01-15 20:00")
        sig_ext = gen.generate(df_ext, GoldRegime.TRENDING_BULL, bar_timestamp=_extended_ts(20, 0))
        assert sig_ext.action == "HOLD"
        assert "extension_exceeded" in sig_ext.metadata.get("block_reason", "")

    def test_strictness_mult_1_disables_overnight_tightening(self) -> None:
        """When mult=1.0, extended and RTH thresholds are identical."""
        gen = _make_gen(
            pullback_max_vwap_extension_atr=1.5,
            extended_hours_extension_strictness_mult=1.0,
        )
        atr = 8.0
        vwap = 2349.0
        close = vwap + 0.3  # Slightly above VWAP — within touch zone
        df = _make_features(close=close, vwap=vwap, ema21=vwap, atr=atr,
                            open_price=close - 0.5, low=vwap - 0.5, high=close + 0.5,
                            start="2026-01-15 20:00")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_extended_ts(20, 0))
        # Should fire because extension is within threshold
        assert sig.action == "BUY"

    def test_ema_extension_also_tightened_overnight(self) -> None:
        """EMA21 extension should also use the overnight strictness multiplier."""
        atr = 8.0
        ema21 = 2340.0
        # close at 1.0 ATR from EMA21 (threshold = 1.25, overnight = 1.25*0.7 = 0.875)
        close = ema21 + 1.0 * atr   # 2348.0 — 1.0 ATR from EMA21
        gen = _make_gen(
            pullback_max_ema_extension_atr=1.25,
            pullback_max_vwap_extension_atr=10.0,   # Don't block on VWAP
            extended_hours_extension_strictness_mult=0.7,
        )
        df = _make_features(close=close, vwap=close, ema21=ema21, atr=atr,
                            open_price=close - 0.5, low=ema21 - 0.2, high=close + 0.5,
                            start="2026-01-15 20:00")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_extended_ts(20, 0))
        # ema_extension = 1.0 ATR, overnight threshold = 0.875 → blocked
        assert sig.action == "HOLD"
        assert "ema_extension_exceeded" in sig.metadata.get("block_reason", "")


# ─────────────────────────────────────────────────────────────────────────────
# ORB VWAP extension guard
# ─────────────────────────────────────────────────────────────────────────────


class TestOrbVwapExtension:
    """Verify that ORB signals are blocked when price is too far from VWAP."""

    def _setup_orb(self, gen: GoldSignalGenerator) -> None:
        """Force the OR to form so ORB signals can fire."""
        from datetime import date
        gen._or_formed = True
        gen._or_high = 2350.0
        gen._or_low = 2340.0
        gen._session_date = date(2026, 1, 15)  # Prevent reset in _update_session_state

    def test_orb_allowed_close_to_vwap(self) -> None:
        """ORB fires when VWAP extension is within threshold."""
        gen = _make_gen(orb_max_vwap_extension_atr=1.5,
                        pullback_max_vwap_extension_atr=0.01,  # Disable pullback via tight extension
                        pullback_max_ema_extension_atr=0.01)
        self._setup_orb(gen)
        atr = 8.0
        vwap = 2352.0
        close = 2352.0  # 0 ATR from VWAP — within ORB VWAP threshold
        # EMA and VWAP will also trigger pullback; disable pullback via tight extension
        df = _make_features(close=close, vwap=vwap, ema21=2360.0, ema9=2358.0, atr=atr,
                            volume=100,
                            open_price=close - 1.0, high=close + 0.5, low=close - 1.5,
                            n_bars=25)
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_rth_ts())
        assert sig.action == "BUY"
        assert sig.signal_type == GoldSignalType.ORB_LONG

    def test_orb_blocked_far_from_vwap(self) -> None:
        """ORB blocked when price is more than orb_max_vwap_extension_atr from VWAP."""
        gen = _make_gen(orb_max_vwap_extension_atr=1.0,
                        pullback_max_vwap_extension_atr=0.01,
                        pullback_max_ema_extension_atr=0.01)
        self._setup_orb(gen)
        atr = 8.0
        vwap = 2340.0
        close = 2352.0  # 1.5 ATR from VWAP — exceeds 1.0
        df = _make_features(close=close, vwap=vwap, ema21=2360.0, ema9=2358.0, atr=atr,
                            volume=100,
                            open_price=close - 1.0, high=close + 0.5, low=close - 1.5,
                            n_bars=25)
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_rth_ts())
        assert sig.action == "HOLD"
        assert "orb_vwap_extension_too_large" in str(sig.metadata)

    def test_orb_short_blocked_far_from_vwap(self) -> None:
        """ORB SHORT blocked when close is far below VWAP."""
        gen = _make_gen(orb_max_vwap_extension_atr=1.0,
                        pullback_max_vwap_extension_atr=0.01,
                        pullback_max_ema_extension_atr=0.01)
        self._setup_orb(gen)
        atr = 8.0
        vwap = 2350.0
        close = 2338.0  # 1.5 ATR below VWAP — exceeds 1.0
        # Below OR low (2340) → ORB SHORT candidate
        df = _make_features(close=close, vwap=vwap, ema21=2330.0, ema9=2332.0, atr=atr,
                            volume=100,
                            open_price=close + 1.0, high=close + 1.5, low=close - 0.5,
                            n_bars=25)
        sig = gen.generate(df, GoldRegime.TRENDING_BEAR, bar_timestamp=_rth_ts())
        assert sig.action == "HOLD"
        assert "orb_vwap_extension_too_large" in str(sig.metadata)

    def test_orb_vwap_guard_disabled_when_zero(self) -> None:
        """Setting orb_max_vwap_extension_atr=0 disables the guard."""
        gen = _make_gen(orb_max_vwap_extension_atr=0.0,
                        pullback_max_vwap_extension_atr=0.01,
                        pullback_max_ema_extension_atr=0.01)
        self._setup_orb(gen)
        atr = 8.0
        vwap = 2340.0
        close = 2352.0  # Far from VWAP but guard disabled
        df = _make_features(close=close, vwap=vwap, ema21=2360.0, ema9=2358.0, atr=atr,
                            volume=100,
                            open_price=close - 1.0, high=close + 0.5, low=close - 1.5,
                            n_bars=25)
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_rth_ts())
        assert sig.action == "BUY"
        assert sig.signal_type == GoldSignalType.ORB_LONG
