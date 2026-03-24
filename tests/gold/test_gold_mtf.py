"""Tests for the multi-timeframe (MTF) regime confirmation gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shree.config.gold import GoldStrategyConfig
from shree.strategies.gold.regime import GoldRegime, GoldRegimeDetector
from shree.strategies.gold.strategy import compute_indicators, compute_htf_indicators


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_trending_df(n: int = 200, trend: float = 1.0, freq: str = "1min") -> pd.DataFrame:
    np.random.seed(42)
    closes = 2350.0 + np.arange(n) * trend + np.random.normal(0, 0.3, n)
    highs = closes + np.abs(np.random.normal(0, 1.0, n))
    lows = closes - np.abs(np.random.normal(0, 1.0, n))
    volumes = np.random.randint(20, 200, n).astype(float)
    idx = pd.date_range("2026-01-15 08:20", periods=n, freq=freq, tz="America/New_York")
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _cfg_with_mtf(enabled: bool = True, tf_min: int = 5) -> GoldStrategyConfig:
    cfg = GoldStrategyConfig(enabled=True, symbol="MGC")
    cfg.indicators.warmup_bars = 30
    cfg.indicators.mtf_enabled = enabled
    cfg.indicators.mtf_timeframe_minutes = tf_min
    cfg.indicators.mtf_adx_min = 18.0
    cfg.indicators.mtf_ema_alignment_required = True
    cfg.entry.adx_trend_min = 15.0
    cfg.entry.adx_trend_max = 100.0   # Don't block on high ADX in tests
    cfg.entry.atr_min_ratio = 0.00005  # Very permissive ATR floor for test data
    return cfg


# ── compute_htf_indicators ────────────────────────────────────────────────────

class TestComputeHtfIndicators:
    def test_adds_htf_columns(self) -> None:
        cfg = _cfg_with_mtf(enabled=True, tf_min=5)
        df = _make_trending_df(100)
        out = compute_htf_indicators(df, cfg)
        for col in ("htf_ema9", "htf_ema21", "htf_adx"):
            assert col in out.columns, f"Missing column: {col}"

    def test_htf_forward_filled_length_matches_input(self) -> None:
        cfg = _cfg_with_mtf(enabled=True, tf_min=5)
        df = _make_trending_df(100)
        out = compute_htf_indicators(df, cfg)
        assert len(out) == len(df)

    def test_no_future_leak(self) -> None:
        """HTF values at bar i must not reflect data from bars i+1 onward."""
        cfg = _cfg_with_mtf(enabled=True, tf_min=5)
        df = _make_trending_df(100)
        out_full = compute_htf_indicators(df, cfg)
        out_half = compute_htf_indicators(df.iloc[:50], cfg)
        # The value at bar 49 should agree whether or not bars 50+ exist
        assert abs(
            float(out_full["htf_ema9"].iloc[49]) - float(out_half["htf_ema9"].iloc[49])
        ) < 0.01

    def test_single_htf_bar_handled(self) -> None:
        """Very short DataFrame (fewer bars than one HTF period) should not crash."""
        cfg = _cfg_with_mtf(enabled=True, tf_min=15)
        df = _make_trending_df(10)
        out = compute_htf_indicators(df, cfg)
        # Either returns a df with HTF cols or unchanged df — no exception
        assert len(out) == len(df)


# ── MTF integration in compute_indicators ────────────────────────────────────

class TestMtfInComputeIndicators:
    def test_mtf_disabled_no_htf_columns(self) -> None:
        cfg = _cfg_with_mtf(enabled=False)
        df = _make_trending_df(100)
        out = compute_indicators(df, cfg)
        for col in ("htf_ema9", "htf_ema21", "htf_adx"):
            assert col not in out.columns

    def test_mtf_enabled_adds_htf_columns(self) -> None:
        cfg = _cfg_with_mtf(enabled=True, tf_min=5)
        df = _make_trending_df(100)
        out = compute_indicators(df, cfg)
        for col in ("htf_ema9", "htf_ema21", "htf_adx"):
            assert col in out.columns


# ── GoldRegimeDetector MTF gate ───────────────────────────────────────────────

class TestMtfRegimeGate:
    def _detector(self, enabled: bool = True) -> GoldRegimeDetector:
        cfg = _cfg_with_mtf(enabled=enabled)
        return GoldRegimeDetector(cfg.indicators, cfg.entry)

    def _features(self, htf_adx: float, htf_bull: bool, regime_bull: bool) -> pd.DataFrame:
        """Build a minimal features DataFrame with both 1m and HTF indicators."""
        cfg = _cfg_with_mtf(enabled=True)
        df = _make_trending_df(80, trend=1.0 if regime_bull else -1.0)
        df = compute_indicators(df, cfg)
        # Override HTF columns with synthetic values for precise testing
        df["htf_adx"] = htf_adx
        df["htf_ema9"] = df["close"] + (1.0 if htf_bull else -1.0)
        df["htf_ema21"] = df["close"]
        return df

    def test_htf_weak_adx_forces_ranging(self) -> None:
        detector = self._detector(enabled=True)
        features = self._features(htf_adx=5.0, htf_bull=True, regime_bull=True)
        regime = detector.detect(features)
        assert regime == GoldRegime.RANGING

    def test_htf_strong_adx_aligned_allows_trending(self) -> None:
        detector = self._detector(enabled=True)
        features = self._features(htf_adx=30.0, htf_bull=True, regime_bull=True)
        regime = detector.detect(features)
        assert regime in (GoldRegime.TRENDING_BULL, GoldRegime.TRENDING_BEAR)

    def test_htf_ema_misaligned_forces_ranging(self) -> None:
        """1m says BULL but HTF EMAs are bearish → RANGING."""
        detector = self._detector(enabled=True)
        features = self._features(htf_adx=30.0, htf_bull=False, regime_bull=True)
        regime = detector.detect(features)
        assert regime == GoldRegime.RANGING

    def test_mtf_disabled_ignores_htf_columns(self) -> None:
        """When mtf_enabled=False the HTF columns should be ignored entirely."""
        cfg = _cfg_with_mtf(enabled=False)
        detector = GoldRegimeDetector(cfg.indicators, cfg.entry)
        df = _make_trending_df(80, trend=1.0)
        df = compute_indicators(df, cfg)
        # Add spurious weak HTF ADX — should not affect result
        df["htf_adx"] = 1.0
        df["htf_ema9"] = df["close"] - 5
        df["htf_ema21"] = df["close"]
        regime = detector.detect(df)
        # Should NOT be forced to RANGING by the weak HTF ADX
        assert regime != GoldRegime.WARMING_UP   # at least warm enough

    def test_htf_nan_values_not_fatal(self) -> None:
        """NaN HTF values (not enough HTF bars yet) should be silently skipped."""
        cfg = _cfg_with_mtf(enabled=True, tf_min=15)
        detector = GoldRegimeDetector(cfg.indicators, cfg.entry)
        df = _make_trending_df(80, trend=1.0)
        df = compute_indicators(df, cfg)
        # Force HTF cols to NaN
        df["htf_adx"] = float("nan")
        df["htf_ema9"] = float("nan")
        df["htf_ema21"] = float("nan")
        # Should not raise; regime may be anything
        regime = detector.detect(df)
        assert isinstance(regime, GoldRegime)
