"""Tests for Phase 5 — Session-aware specialization.

Covers:
- GoldSessionBucket enum
- GoldSessionBucketConfig defaults
- _classify_session_bucket() classifier correctness
- Per-bucket ORB enable/disable
- Per-bucket pullback enable/disable
- Per-bucket confidence offset
- Per-bucket volume minimum multiplier
- Per-bucket ADX minimum multiplier
- Per-bucket ATR ratio minimum multiplier
- Per-bucket extension strictness multiplier
- UNKNOWN bucket blocks all signals
- Fallback when bar_ts is None
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shree.config.gold import (
    GoldEntryConfig,
    GoldExitConfig,
    GoldIndicatorConfig,
    GoldSessionBucket,
    GoldSessionBucketConfig,
    GoldSessionConfig,
    _default_session_buckets,
)
from shree.strategies.gold.regime import GoldRegime
from shree.strategies.gold.signals import GoldSignal, GoldSignalGenerator, GoldSignalType


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_gen(
    extended_hours_enabled: bool = True,
    orb_enabled: bool = True,
    min_bar_volume: int = 5,
    adx_trend_min: float = 20.0,
    atr_min_ratio: float = 0.0002,
    session_overrides: dict | None = None,
    bucket_overrides: dict[str, GoldSessionBucketConfig] | None = None,
    **entry_overrides,
) -> GoldSignalGenerator:
    """Build a GoldSignalGenerator with sensible test defaults."""
    entry = GoldEntryConfig(
        adx_trend_min=adx_trend_min,
        adx_trend_max=55.0,
        vwap_touch_pct=0.002,
        ema_touch_pct=0.0025,
        pullback_reclaim_required=True,
        pullback_min_body_fraction=0.25,
        pullback_confirm_with_bar_direction=True,
        pullback_max_vwap_extension_atr=entry_overrides.pop("pullback_max_vwap_extension_atr", 1.5),
        pullback_max_ema_extension_atr=entry_overrides.pop("pullback_max_ema_extension_atr", 1.25),
        orb_enabled=orb_enabled,
        orb_min_range_ratio=0.0001,
        orb_breakout_min_atr_fraction=0.0,
        orb_volume_lookback_bars=20,
        orb_volume_min_multiple=0.0,
        orb_max_breakout_candle_atr=10.0,
        orb_max_extension_atr=10.0,
        orb_max_vwap_extension_atr=10.0,
        min_bar_volume=min_bar_volume,
        post_loss_cooldown_bars=0,
        atr_min_ratio=atr_min_ratio,
        **entry_overrides,
    )
    exit_cfg = GoldExitConfig(
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=2.5,
        sl_floor_points=0.1,
        sl_ceiling_points=100.0,
        min_rr_ratio=1.0,
    )
    session_kwargs = dict(
        extended_hours_enabled=extended_hours_enabled,
        session_open_et="08:20",
        session_close_et="13:30",
        extended_session_open_et="18:00",
        opening_range_minutes=15,
    )
    if session_overrides:
        session_kwargs.update(session_overrides)
    session = GoldSessionConfig(**session_kwargs)
    if bucket_overrides:
        session.buckets.update(bucket_overrides)
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
    """Build a synthetic OHLCV+indicator DataFrame."""
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


def _bullish_features(
    start: str = "2026-01-15 09:30",
    atr: float = 8.0,
    adx: float = 28.0,
    volume: int = 50,
    n_bars: int = 25,
) -> pd.DataFrame:
    """Build features that produce a valid VWAP pullback LONG signal."""
    vwap = 2349.0
    ema21 = 2349.0
    ema9 = 2351.0
    close = vwap + 0.3       # Just above VWAP — in touch zone
    open_price = close - 1.0  # Bullish candle (close > open)
    low = vwap - 0.5          # Probed below VWAP (touched + reclaimed)
    high = close + 1.0
    return _make_features(
        close=close, ema9=ema9, ema21=ema21, atr=atr, adx=adx, vwap=vwap,
        volume=volume, n_bars=n_bars, start=start,
        open_price=open_price, low=low, high=high,
    )


def _ts(hour: int, minute: int = 0, day: int = 15) -> pd.Timestamp:
    """Create an ET timestamp for testing."""
    return pd.Timestamp(f"2026-01-{day:02d} {hour:02d}:{minute:02d}:00", tz="America/New_York")


def _set_or(gen: GoldSignalGenerator, or_high: float, or_low: float) -> None:
    """Force-set the opening range on the generator (for ORB tests)."""
    gen._or_high = or_high
    gen._or_low = or_low
    gen._or_formed = True
    gen._session_date = pd.Timestamp("2026-01-15", tz="America/New_York").date()


# ─────────────────────────────────────────────────────────────────────────────
# Test: Session bucket classification
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifySessionBucket:
    """Verify _classify_session_bucket maps bar timestamps to the correct bucket."""

    def test_overnight_evening(self) -> None:
        """20:00 ET → OVERNIGHT (Asia session evening side)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(20, 0)) == GoldSessionBucket.OVERNIGHT

    def test_overnight_late_night(self) -> None:
        """01:30 ET → OVERNIGHT (Asia session past midnight)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(1, 30)) == GoldSessionBucket.OVERNIGHT

    def test_overnight_boundary_at_1800(self) -> None:
        """18:00 ET exactly → OVERNIGHT (boundary inclusive)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(18, 0)) == GoldSessionBucket.OVERNIGHT

    def test_pre_comex_at_0300(self) -> None:
        """03:00 ET → PRE_COMEX (London session start)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(3, 0)) == GoldSessionBucket.PRE_COMEX

    def test_pre_comex_at_0700(self) -> None:
        """07:00 ET → PRE_COMEX (London mid-session)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(7, 0)) == GoldSessionBucket.PRE_COMEX

    def test_pre_comex_at_0819(self) -> None:
        """08:19 ET → PRE_COMEX (one minute before COMEX open)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(8, 19)) == GoldSessionBucket.PRE_COMEX

    def test_comex_open_at_0820(self) -> None:
        """08:20 ET → COMEX_OPEN (boundary inclusive)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(8, 20)) == GoldSessionBucket.COMEX_OPEN

    def test_comex_open_mid_morning(self) -> None:
        """09:45 ET → COMEX_OPEN."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(9, 45)) == GoldSessionBucket.COMEX_OPEN

    def test_comex_open_at_1029(self) -> None:
        """10:29 ET → COMEX_OPEN (last minute before midday)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(10, 29)) == GoldSessionBucket.COMEX_OPEN

    def test_midday_at_1030(self) -> None:
        """10:30 ET → MIDDAY (boundary inclusive)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(10, 30)) == GoldSessionBucket.MIDDAY

    def test_midday_at_1145(self) -> None:
        """11:45 ET → MIDDAY."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(11, 45)) == GoldSessionBucket.MIDDAY

    def test_pre_close_at_1200(self) -> None:
        """12:00 ET → PRE_CLOSE (boundary inclusive)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(12, 0)) == GoldSessionBucket.PRE_CLOSE

    def test_pre_close_at_1320(self) -> None:
        """13:20 ET → PRE_CLOSE (near session end)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(13, 20)) == GoldSessionBucket.PRE_CLOSE

    def test_unknown_at_1330(self) -> None:
        """13:30 ET → UNKNOWN (post-session gap)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(13, 30)) == GoldSessionBucket.UNKNOWN

    def test_unknown_at_1500(self) -> None:
        """15:00 ET → UNKNOWN (well after close, before overnight)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(_ts(15, 0)) == GoldSessionBucket.UNKNOWN

    def test_none_timestamp_defaults_to_comex_open(self) -> None:
        """None bar_ts → COMEX_OPEN (most permissive fallback)."""
        gen = _make_gen()
        assert gen._classify_session_bucket(None) == GoldSessionBucket.COMEX_OPEN

    def test_custom_bucket_boundaries(self) -> None:
        """Custom bucket boundaries should override defaults."""
        gen = _make_gen(session_overrides={
            "bucket_midday_start_et": "11:00",  # Later midday
        })
        # 10:45 ET should now be COMEX_OPEN instead of MIDDAY
        assert gen._classify_session_bucket(_ts(10, 45)) == GoldSessionBucket.COMEX_OPEN
        # 11:00 ET should now be MIDDAY
        assert gen._classify_session_bucket(_ts(11, 0)) == GoldSessionBucket.MIDDAY


# ─────────────────────────────────────────────────────────────────────────────
# Test: Default bucket configs
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultBucketConfigs:
    """Verify the defaults in _default_session_buckets are sensible."""

    def test_overnight_orb_disabled(self) -> None:
        defaults = _default_session_buckets()
        assert defaults[GoldSessionBucket.OVERNIGHT.value].orb_enabled is False

    def test_pre_comex_orb_disabled(self) -> None:
        defaults = _default_session_buckets()
        assert defaults[GoldSessionBucket.PRE_COMEX.value].orb_enabled is False

    def test_comex_open_orb_enabled(self) -> None:
        defaults = _default_session_buckets()
        assert defaults[GoldSessionBucket.COMEX_OPEN.value].orb_enabled is True

    def test_comex_open_is_neutral(self) -> None:
        """COMEX_OPEN bucket should have all neutral multipliers."""
        cfg = _default_session_buckets()[GoldSessionBucket.COMEX_OPEN.value]
        assert cfg.confidence_offset == 0.0
        assert cfg.adx_min_mult == 1.0
        assert cfg.atr_min_ratio_mult == 1.0
        assert cfg.volume_min_mult == 1.0
        assert cfg.extension_strictness_mult == 1.0

    def test_overnight_has_negative_confidence_offset(self) -> None:
        cfg = _default_session_buckets()[GoldSessionBucket.OVERNIGHT.value]
        assert cfg.confidence_offset < 0.0

    def test_pre_close_orb_disabled(self) -> None:
        defaults = _default_session_buckets()
        assert defaults[GoldSessionBucket.PRE_CLOSE.value].orb_enabled is False

    def test_midday_has_negative_confidence_offset(self) -> None:
        cfg = _default_session_buckets()[GoldSessionBucket.MIDDAY.value]
        assert cfg.confidence_offset < 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: ORB enable/disable per session bucket
# ─────────────────────────────────────────────────────────────────────────────

class TestORBSessionRouting:
    """Verify that ORB is only active in buckets where orb_enabled=True."""

    def _bullish_orb_features(self, start: str) -> pd.DataFrame:
        """Features for a bullish ORB breakout."""
        or_high = 2355.0
        close = or_high + 2.0  # Above OR high → breakout
        return _make_features(
            close=close, open_price=close - 1.5, high=close + 0.5, low=close - 2.0,
            ema9=close - 1.0, ema21=close - 3.0, atr=8.0, adx=28.0,
            vwap=close - 1.0, volume=50, n_bars=25, start=start,
        )

    def test_orb_fires_during_comex_open(self) -> None:
        gen = _make_gen()
        _set_or(gen, or_high=2355.0, or_low=2340.0)
        df = self._bullish_orb_features("2026-01-15 09:00")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(9, 0))
        assert sig.action == "BUY"
        assert sig.signal_type == GoldSignalType.ORB_LONG

    def test_orb_blocked_during_overnight(self) -> None:
        gen = _make_gen()
        _set_or(gen, or_high=2355.0, or_low=2340.0)
        df = self._bullish_orb_features("2026-01-15 22:00")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(22, 0))
        # ORB should be disabled in overnight bucket
        assert sig.signal_type != GoldSignalType.ORB_LONG

    def test_orb_blocked_during_pre_comex(self) -> None:
        gen = _make_gen()
        _set_or(gen, or_high=2355.0, or_low=2340.0)
        df = self._bullish_orb_features("2026-01-15 05:00")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(5, 0))
        assert sig.signal_type != GoldSignalType.ORB_LONG

    def test_orb_blocked_during_pre_close(self) -> None:
        gen = _make_gen()
        _set_or(gen, or_high=2355.0, or_low=2340.0)
        df = self._bullish_orb_features("2026-01-15 12:30")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(12, 30))
        assert sig.signal_type != GoldSignalType.ORB_LONG

    def test_orb_allowed_during_midday(self) -> None:
        """Default midday config still allows ORB."""
        gen = _make_gen()
        _set_or(gen, or_high=2355.0, or_low=2340.0)
        df = self._bullish_orb_features("2026-01-15 11:00")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(11, 0))
        assert sig.action == "BUY"
        assert sig.signal_type == GoldSignalType.ORB_LONG

    def test_orb_override_enable_in_overnight(self) -> None:
        """If bucket config overrides orb_enabled=True, ORB should work overnight."""
        gen = _make_gen(bucket_overrides={
            GoldSessionBucket.OVERNIGHT.value: GoldSessionBucketConfig(
                orb_enabled=True,
                pullback_enabled=True,
                confidence_offset=0.0,
            ),
        })
        _set_or(gen, or_high=2355.0, or_low=2340.0)
        df = self._bullish_orb_features("2026-01-15 22:00")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(22, 0))
        assert sig.action == "BUY"
        assert sig.signal_type == GoldSignalType.ORB_LONG


# ─────────────────────────────────────────────────────────────────────────────
# Test: Pullback enable/disable per session bucket
# ─────────────────────────────────────────────────────────────────────────────

class TestPullbackSessionRouting:
    """Verify pullback signals honor per-bucket pullback_enabled flag."""

    def test_pullback_fires_during_comex_open(self) -> None:
        gen = _make_gen()
        df = _bullish_features(start="2026-01-15 09:30")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(9, 30))
        assert sig.action == "BUY"
        assert "PB" in sig.signal_type.value

    def test_pullback_fires_during_pre_comex(self) -> None:
        """London session — pullbacks should be allowed by default."""
        gen = _make_gen()
        df = _bullish_features(start="2026-01-15 05:00")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(5, 0))
        assert sig.action == "BUY"

    def test_pullback_fires_during_overnight(self) -> None:
        """Overnight pullbacks allowed by default (but tighter thresholds)."""
        gen = _make_gen()
        df = _bullish_features(start="2026-01-15 22:00")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(22, 0))
        assert sig.action == "BUY"

    def test_pullback_blocked_when_disabled_in_bucket(self) -> None:
        """Override pullback_enabled=False for PRE_CLOSE → should block."""
        gen = _make_gen(bucket_overrides={
            GoldSessionBucket.PRE_CLOSE.value: GoldSessionBucketConfig(
                orb_enabled=False,
                pullback_enabled=False,
                confidence_offset=-0.15,
            ),
        })
        df = _bullish_features(start="2026-01-15 12:30")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(12, 30))
        assert sig.action == "HOLD"
        assert "pullback_disabled_in_bucket" in sig.metadata.get("block_reason", "")


# ─────────────────────────────────────────────────────────────────────────────
# Test: UNKNOWN bucket blocks all signals
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownBucket:
    """UNKNOWN bucket (13:30–18:00 ET) should produce HOLD regardless."""

    def test_unknown_bucket_blocks_pullback(self) -> None:
        gen = _make_gen()
        df = _bullish_features(start="2026-01-15 14:00")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(14, 0))
        assert sig.action == "HOLD"
        assert sig.metadata.get("block_reason") == "session_bucket_unknown"

    def test_unknown_bucket_blocks_at_1500(self) -> None:
        gen = _make_gen()
        df = _bullish_features(start="2026-01-15 15:00")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(15, 0))
        assert sig.action == "HOLD"
        assert sig.metadata.get("session_bucket") == GoldSessionBucket.UNKNOWN.value


# ─────────────────────────────────────────────────────────────────────────────
# Test: Confidence offset per bucket
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceOffset:
    """Verify per-bucket confidence offset is applied to signals."""

    def test_comex_open_no_offset(self) -> None:
        """COMEX_OPEN has offset=0.0 → confidence unchanged."""
        gen = _make_gen()
        df = _bullish_features(start="2026-01-15 09:30")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(9, 30))
        assert sig.is_actionable
        # Base pullback confidence is >= 0.50 (see _pullback_confidence)
        assert sig.confidence >= 0.50

    def test_overnight_reduces_confidence(self) -> None:
        """OVERNIGHT has offset=-0.10 → confidence reduced."""
        gen = _make_gen()
        # Same features at COMEX_OPEN vs OVERNIGHT
        df_rth = _bullish_features(start="2026-01-15 09:30")
        sig_rth = gen.generate(df_rth, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(9, 30))

        gen2 = _make_gen()
        df_on = _bullish_features(start="2026-01-15 22:00")
        sig_on = gen2.generate(df_on, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(22, 0))

        assert sig_rth.is_actionable and sig_on.is_actionable
        assert sig_on.confidence < sig_rth.confidence

    def test_confidence_floor_at_zero(self) -> None:
        """Confidence offset should never take confidence below 0.0."""
        gen = _make_gen(bucket_overrides={
            GoldSessionBucket.COMEX_OPEN.value: GoldSessionBucketConfig(
                confidence_offset=-999.0,  # Extreme negative
            ),
        })
        df = _bullish_features(start="2026-01-15 09:30")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(9, 30))
        if sig.is_actionable:
            assert sig.confidence >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Volume minimum multiplier per bucket
# ─────────────────────────────────────────────────────────────────────────────

class TestVolumeMinMultiplier:
    """Verify per-bucket volume_min_mult adjusts the volume floor."""

    def test_comex_open_base_volume(self) -> None:
        """COMEX_OPEN: volume_min_mult=1.0 → base min_bar_volume applies."""
        gen = _make_gen(min_bar_volume=5)
        df = _bullish_features(start="2026-01-15 09:30")
        # Volume=50 in features, well above min_bar_volume=5
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(9, 30))
        assert sig.is_actionable

    def test_overnight_higher_volume_floor(self) -> None:
        """OVERNIGHT: volume_min_mult=2.0 → effective floor=10 for base=5."""
        gen = _make_gen(min_bar_volume=5)
        # Low volume: 8 → passes base (5) but fails overnight (5*2=10)
        df = _bullish_features(start="2026-01-15 22:00")
        # Override volume to 8
        df["volume"] = 8
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(22, 0))
        assert sig.action == "HOLD"
        assert sig.metadata.get("block_reason") == "bar_volume_below_minimum"

    def test_overnight_passes_with_sufficient_volume(self) -> None:
        """OVERNIGHT with volume=50 → passes the doubled floor easily."""
        gen = _make_gen(min_bar_volume=5)
        df = _bullish_features(start="2026-01-15 22:00")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(22, 0))
        assert sig.is_actionable


# ─────────────────────────────────────────────────────────────────────────────
# Test: ADX minimum multiplier per bucket
# ─────────────────────────────────────────────────────────────────────────────

class TestADXMinMultiplier:
    """Verify per-bucket adx_min_mult raises the ADX floor."""

    def test_comex_open_base_adx(self) -> None:
        """COMEX_OPEN: adx_min_mult=1.0 → base adx_trend_min=20 applies."""
        gen = _make_gen(adx_trend_min=20.0)
        df = _bullish_features(start="2026-01-15 09:30")
        # adx=28 in features → passes 20
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(9, 30))
        assert sig.is_actionable

    def test_overnight_higher_adx_floor(self) -> None:
        """OVERNIGHT: adx_min_mult=1.25 → effective floor=25 for base=20.
        ADX=22 should fail overnight but pass COMEX_OPEN."""
        gen = _make_gen(adx_trend_min=20.0)
        df = _bullish_features(start="2026-01-15 22:00")
        df["adx"] = 22.0  # Passes 20 but fails 25 (20*1.25)
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(22, 0))
        assert sig.action == "HOLD"
        assert sig.metadata.get("block_reason") == "adx_below_bucket_minimum"

    def test_overnight_passes_with_high_adx(self) -> None:
        """OVERNIGHT with ADX=30 → passes the 25 floor."""
        gen = _make_gen(adx_trend_min=20.0)
        df = _bullish_features(start="2026-01-15 22:00")
        df["adx"] = 30.0
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(22, 0))
        assert sig.is_actionable


# ─────────────────────────────────────────────────────────────────────────────
# Test: ATR ratio minimum multiplier per bucket
# ─────────────────────────────────────────────────────────────────────────────

class TestATRRatioMinMultiplier:
    """Verify per-bucket atr_min_ratio_mult adjusts the ATR/price floor."""

    def test_overnight_rejects_low_atr(self) -> None:
        """OVERNIGHT: atr_min_ratio_mult=1.5 → effective floor=0.0003.
        ATR=0.5 with close=2350 → ratio=0.000213 < 0.0003 → blocked."""
        gen = _make_gen(atr_min_ratio=0.0002)
        df = _bullish_features(start="2026-01-15 22:00")
        df["atr"] = 0.5  # atr/close = 0.5/2349.3 ≈ 0.000213
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(22, 0))
        assert sig.action == "HOLD"
        assert sig.metadata.get("block_reason") == "atr_ratio_below_bucket_minimum"

    def test_comex_open_allows_low_atr(self) -> None:
        """COMEX_OPEN: atr_min_ratio_mult=1.0 → floor stays at 0.0002.
        ATR=0.5 with close=2350 → ratio=0.000213 > 0.0002 → allowed."""
        gen = _make_gen(atr_min_ratio=0.0002)
        df = _bullish_features(start="2026-01-15 09:30")
        df["atr"] = 0.5
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(9, 30))
        # May still HOLD for other reasons (e.g. SL too small), but not for ATR ratio
        if sig.action == "HOLD":
            assert sig.metadata.get("block_reason") != "atr_ratio_below_bucket_minimum"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Extension strictness multiplier per bucket
# ─────────────────────────────────────────────────────────────────────────────

class TestExtensionStrictnessBucket:
    """Verify that bucket_cfg.extension_strictness_mult is used in pullback checks."""

    def test_comex_open_normal_extension(self) -> None:
        """COMEX_OPEN: extension_strictness_mult=1.0 → normal thresholds."""
        gen = _make_gen()
        df = _bullish_features(start="2026-01-15 09:30")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(9, 30))
        assert sig.is_actionable

    def test_overnight_tighter_extension(self) -> None:
        """OVERNIGHT has extension_strictness_mult=0.6 → 40% tighter.
        A pullback at ~0.8 ATR from VWAP should be blocked overnight but allowed in COMEX_OPEN.

        Extension guard: abs(close - vwap) / atr > pullback_max_vwap_extension_atr * ext_mult
        - COMEX_OPEN: threshold = 1.5 * 1.0 = 1.5 → 0.8 < 1.5 → pass
        - OVERNIGHT:  threshold = 1.5 * 0.6 = 0.9 → 0.8 < 0.9 → pass extension...

        We need to pick a distance that passes COMEX but fails OVERNIGHT.
        Extension: 1.5*0.6 = 0.9 threshold vs 1.5*1.0 = 1.5 threshold.
        Use distance = 1.0 ATR → passes COMEX (1.0 < 1.5) but fails OVERNIGHT (1.0 > 0.9).

        But we also need close to be in the VWAP touch zone for the pullback to fire.
        Touch zone = vwap ± vwap*touch_pct. For VWAP=2350, touch_pct=0.002, band=4.7.
        So close must be between ~2345.3 and 2354.7.

        Problem: if close is 1.0 ATR (=8pts) from VWAP, it's outside the touch zone.
        Solution: use EMA21 pullback instead (ema_touch_pct=0.0025, wider zone).
        Or, test via the EMA extension guard with ema21 set = vwap so both checks see the same distance.

        Actually, the extension guard checks VWAP extension AND EMA extension independently.
        Let's place close in the VWAP touch zone but make the EMA extension fail.
        """
        gen_rth = _make_gen(
            pullback_max_ema_extension_atr=1.2,  # Tighter EMA extension
        )
        gen_on = _make_gen(
            pullback_max_ema_extension_atr=1.2,
        )

        atr = 8.0
        vwap = 2349.0
        ema21 = 2342.0  # EMA21 far from close
        close = vwap + 0.3  # In VWAP touch zone
        # EMA extension = |2349.3 - 2342| / 8 = 0.9125 ATR
        # COMEX: threshold = 1.2 * 1.0 = 1.2 → 0.9125 < 1.2 → pass
        # OVERNIGHT: threshold = 1.2 * 0.6 = 0.72 → 0.9125 > 0.72 → blocked

        df_rth = _make_features(
            close=close, vwap=vwap, ema21=ema21, ema9=close + 1.0, atr=atr,
            open_price=close - 1.0, low=vwap - 0.5, high=close + 1.0,
            start="2026-01-15 09:30",
        )
        sig_rth = gen_rth.generate(df_rth, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(9, 30))

        df_on = _make_features(
            close=close, vwap=vwap, ema21=ema21, ema9=close + 1.0, atr=atr,
            open_price=close - 1.0, low=vwap - 0.5, high=close + 1.0,
            start="2026-01-15 22:00",
        )
        sig_on = gen_on.generate(df_on, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(22, 0))

        assert sig_rth.is_actionable  # COMEX_OPEN allows it
        assert sig_on.action == "HOLD"  # OVERNIGHT blocks it
        assert "extension_exceeded" in sig_on.metadata.get("block_reason", "")


# ─────────────────────────────────────────────────────────────────────────────
# Test: Session bucket metadata in signals
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionBucketMetadata:
    """Verify that session_bucket is included in signal metadata."""

    def test_actionable_signal_has_bucket(self) -> None:
        gen = _make_gen()
        df = _bullish_features(start="2026-01-15 09:30")
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(9, 30))
        assert sig.is_actionable
        assert sig.metadata.get("session_bucket") == GoldSessionBucket.COMEX_OPEN.value

    def test_hold_signal_has_bucket(self) -> None:
        gen = _make_gen()
        # Use a RANGING regime → HOLD (regime_not_tradeable)
        df = _bullish_features(start="2026-01-15 09:30")
        sig = gen.generate(df, GoldRegime.RANGING, bar_timestamp=_ts(9, 30))
        # regime_not_tradeable doesn't have bucket (gated before classification matters)
        # But UNKNOWN bucket does
        df2 = _bullish_features(start="2026-01-15 14:00")
        sig2 = gen.generate(df2, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(14, 0))
        assert sig2.metadata.get("session_bucket") == GoldSessionBucket.UNKNOWN.value

    def test_overnight_hold_has_bucket(self) -> None:
        """When overnight volume blocks a signal, bucket should be in metadata."""
        gen = _make_gen(min_bar_volume=5)
        df = _bullish_features(start="2026-01-15 22:00")
        df["volume"] = 3  # Below floor (5 * 2.0 = 10)
        sig = gen.generate(df, GoldRegime.TRENDING_BULL, bar_timestamp=_ts(22, 0))
        assert sig.action == "HOLD"
        assert sig.metadata.get("session_bucket") == GoldSessionBucket.OVERNIGHT.value
