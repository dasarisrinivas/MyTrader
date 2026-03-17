"""Tests for ATR-adaptive SL/TP on Signal F (TREND_CONT) — FEB 24 2026.

Validates that TREND_CONT_LONG and TREND_CONT_SHORT use ATR-adaptive
stop-loss and take-profit instead of fixed 8pt/12pt values.

Root cause: 252-trade backtest showed fixed 8/12 SL/TP yields:
  - ATR <7:  TP=2.1×ATR (unreachable) → 49% WR, −$72
  - ATR 7-10: TP=1.4×ATR → 57% WR, +$247  (sweet spot: SL≈1×ATR)
  - ATR 15-25: SL=0.5×ATR (noise band) → 31% WR, −$317

Fix: SL = clamp(ATR × 1.0, 6, 20), TP = SL × 1.25
Same ATR-adaptive pattern already proven on Signal C (EMA9_PB).
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from types import SimpleNamespace

from shree.strategies.es_fifteen_min import EsFifteenMinStrategy
from shree.config.strategy import OneMinuteStrategyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> OneMinuteStrategyConfig:
    """Create a minimal config with trend continuation enabled."""
    cfg = OneMinuteStrategyConfig()
    cfg.ft_trend_cont_enabled = True
    cfg.ft_shorts_enabled = True
    cfg.ft_ema9_pb_enabled = False  # Isolate Signal F
    # ATR-adaptive params (defaults match strategy)
    cfg.ft_trend_sl_atr_mult = overrides.get('sl_mult', 1.0)
    cfg.ft_trend_sl_floor_pts = overrides.get('sl_floor', 6.0)
    cfg.ft_trend_sl_ceiling_pts = overrides.get('sl_ceiling', 20.0)
    cfg.ft_trend_rr_ratio = overrides.get('rr_ratio', 1.25)
    # Entry window: allow all hours
    cfg.ft_entry_start_hour = 0
    cfg.ft_entry_end_hour = 23
    return cfg


def _make_features(
    n: int = 60,
    close: float = 6000.0,
    ema9: float = 5995.0,
    ema21: float = 5940.0,  # Far enough below close that EMA21 touch-band never reaches our low=close-2
    ema50: float = 5900.0,  # Must be < ema21 for uptrend stack
    atr: float = 10.0,
    adx: float = 25.0,
    rsi: float = 60.0,
    macd_hist: float = 3.0,
    trend: str = "up",
) -> pd.DataFrame:
    """Build a feature DataFrame that triggers TREND_CONT signal.

    The last few bars are constructed to satisfy all Signal F conditions:
      - EMA stack: EMA9 > EMA21 > EMA50 (long) or reversed (short)
      - Close > EMA9 (long) or Close < EMA9 (short)
      - Bullish bar (close > open for long)
      - ADX above threshold
      - MACD_H > 0 (long) or < 0 (short)
      - RSI in valid range
      - Ascending closes (long) — last 2 bars rising
    """
    # Build basic OHLCV bars
    # Start at 23:00 CST Feb 23 so bars 0-1 (= 00:00-00:15 ET Feb 24) land in the
    # OR window [00:00, 00:30) ET and bar 59 lands at 13:45 CST = 14:45 ET (RTH).
    dates = pd.date_range("2026-02-23 23:00", periods=n, freq="15min", tz="America/Chicago")
    
    if trend == "up":
        # Gentle uptrend: each bar close slightly higher
        closes = [close - (n - i) * 0.5 for i in range(n)]
        closes[-1] = close  # Last bar at target close
        closes[-2] = close - 1.0  # Ascending closes requirement
    else:
        # Downtrend for short signals
        closes = [close + (n - i) * 0.5 for i in range(n)]
        closes[-1] = close
        closes[-2] = close + 1.0  # Descending closes requirement
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": [c - 1.5 if trend == "up" else c + 1.5 for c in closes],
        "high": [c + 2.0 for c in closes],
        "low": [c - 2.0 for c in closes],
        "close": closes,
        "volume": [1000] * n,
        "EMA_9": [ema9] * n,
        "EMA_21": [ema21] * n,
        "EMA_50": [ema50] * n,
        "ATR_14": [atr] * n,
        "ADX_14": [adx] * n,
        "RSI_14": [rsi] * n,
        "MACDhist_12_26_9": [macd_hist if trend == "up" else -macd_hist] * n,
        "Stoch_K": [50.0] * n,
    })
    df.set_index("timestamp", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Tests: ATR-adaptive SL/TP for TREND_CONT_LONG
# ---------------------------------------------------------------------------

class TestTrendContLongATRAdaptive:
    """Signal F long: SL/TP should scale with ATR."""

    def test_low_atr_uses_floor(self):
        """ATR=4 → SL should clamp to floor (6pts), not 4pts."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)
        
        features = _make_features(atr=4.0, close=6000.0, ema9=5998.0, adx=25)
        signal = strat.generate(features)
        
        assert signal.action == "BUY"
        meta = signal.metadata
        assert meta["stop_loss"] == pytest.approx(6000.0 - 6.0, abs=0.5)  # floor=6
        assert meta["take_profit"] == pytest.approx(6000.0 + 7.5, abs=0.5)  # 6 × 1.25

    def test_normal_atr_scales_with_atr(self):
        """ATR=10 → SL=10pts (1.0×ATR), TP=12.5pts (10×1.25)."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)
        
        features = _make_features(atr=10.0, close=6000.0, ema9=5997.0, adx=25)
        signal = strat.generate(features)
        
        assert signal.action == "BUY"
        meta = signal.metadata
        assert meta["stop_loss"] == pytest.approx(6000.0 - 10.0, abs=0.5)
        assert meta["take_profit"] == pytest.approx(6000.0 + 12.5, abs=0.5)

    def test_today_atr_8_7(self):
        """ATR=8.7 (today's actual value) → SL=8.7pts, TP=10.9pts.
        
        Old fixed system: SL=8, TP=12. New: SL=8.7, TP=10.9.
        Key improvement: TP drops from 12→10.9 (reachable), SL slightly wider.
        """
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)
        
        features = _make_features(atr=8.7, close=6902.5, ema9=6900.0, adx=25)
        signal = strat.generate(features)
        
        assert signal.action == "BUY"
        meta = signal.metadata
        expected_sl_pts = 8.7  # ATR × 1.0
        expected_tp_pts = 8.7 * 1.25  # 10.875
        assert meta["stop_loss"] == pytest.approx(6902.5 - expected_sl_pts, abs=0.5)
        assert meta["take_profit"] == pytest.approx(6902.5 + expected_tp_pts, abs=0.5)

    def test_high_atr_uses_ceiling(self):
        """ATR=25 → SL should clamp to ceiling (20pts), not 25pts."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)
        
        features = _make_features(atr=25.0, close=6000.0, ema9=5995.0, adx=30)
        signal = strat.generate(features)
        
        assert signal.action == "BUY"
        meta = signal.metadata
        assert meta["stop_loss"] == pytest.approx(6000.0 - 20.0, abs=0.5)  # ceiling=20
        assert meta["take_profit"] == pytest.approx(6000.0 + 25.0, abs=0.5)  # 20 × 1.25

    def test_rr_ratio_always_1_25(self):
        """R:R should always be 1.25:1 regardless of ATR level."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)
        
        for atr_val in [5.0, 8.7, 10.0, 15.0, 22.0]:
            strat._trend_cont_long_count = 0  # Reset daily counter
            strat._session_date = None
            features = _make_features(atr=atr_val, close=6000.0, ema9=5997.0, adx=25)
            signal = strat.generate(features)
            
            if signal.action == "BUY":
                meta = signal.metadata
                sl_dist = 6000.0 - meta["stop_loss"]
                tp_dist = meta["take_profit"] - 6000.0
                rr = tp_dist / sl_dist if sl_dist > 0 else 0
                assert rr == pytest.approx(1.25, abs=0.01), (
                    f"ATR={atr_val}: R:R={rr:.3f} (expected 1.25)"
                )

    def test_reason_includes_sl_tp_pts(self):
        """Signal reason string should include SL/TP point values for logging."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)
        
        features = _make_features(atr=10.0, close=6000.0, ema9=5997.0, adx=25)
        signal = strat.generate(features)
        
        assert signal.action == "BUY"
        reason = signal.metadata.get("reason", "")
        assert "SL=" in reason
        assert "TP=" in reason
        assert "pts" in reason

    def test_custom_config_overrides(self):
        """Config params should override default ATR-adaptive values."""
        cfg = _make_config(sl_mult=1.5, sl_floor=8.0, sl_ceiling=25.0, rr_ratio=1.5)
        strat = EsFifteenMinStrategy(cfg)
        
        features = _make_features(atr=10.0, close=6000.0, ema9=5997.0, adx=25)
        signal = strat.generate(features)
        
        assert signal.action == "BUY"
        meta = signal.metadata
        # SL = min(25, max(8, 10 × 1.5)) = min(25, max(8, 15)) = 15
        assert meta["stop_loss"] == pytest.approx(6000.0 - 15.0, abs=0.5)
        # TP = 15 × 1.5 = 22.5
        assert meta["take_profit"] == pytest.approx(6000.0 + 22.5, abs=0.5)


# ---------------------------------------------------------------------------
# Tests: ATR-adaptive SL/TP for TREND_CONT_SHORT
# ---------------------------------------------------------------------------

class TestTrendContShortATRAdaptive:
    """Signal F short: SL/TP should scale with ATR (inverted direction)."""

    def test_normal_atr_short(self):
        """ATR=10 short → SL=close+10, TP=close-12.5."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)
        
        # TREND_CONT_SHORT needs: EMA9 < EMA21 < EMA50, close < EMA9
        # Price must be well BELOW EMA9 (not near EMA21 — that triggers Signal D)
        close = 5950.0
        features = _make_features(
            atr=10.0, close=close, ema9=5955.0,
            ema21=5970.0, ema50=5990.0, adx=25,
            rsi=35.0, macd_hist=3.0, trend="down",
        )
        signal = strat.generate(features)
        
        assert signal.action == "SELL", f"Expected SELL, got {signal.action}: {signal.metadata}"
        meta = signal.metadata
        assert "TREND_CONT_SHORT" in meta.get("reason", "")
        assert meta["stop_loss"] == pytest.approx(close + 10.0, abs=0.5)
        assert meta["take_profit"] == pytest.approx(close - 12.5, abs=0.5)

    def test_low_atr_short_uses_floor(self):
        """ATR=4 short → SL clamps to floor (6pts)."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)
        
        close = 5950.0
        features = _make_features(
            atr=4.0, close=close, ema9=5952.0,
            ema21=5970.0, ema50=5990.0, adx=25,
            rsi=35.0, macd_hist=3.0, trend="down",
        )
        signal = strat.generate(features)
        
        assert signal.action == "SELL", f"Expected SELL, got {signal.action}: {signal.metadata}"
        meta = signal.metadata
        assert meta["stop_loss"] == pytest.approx(close + 6.0, abs=0.5)  # floor
        assert meta["take_profit"] == pytest.approx(close - 7.5, abs=0.5)  # 6×1.25

    def test_short_reason_includes_pts(self):
        """Short signal reason should include SL/TP point values."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)
        
        close = 5950.0
        features = _make_features(
            atr=12.0, close=close, ema9=5955.0,
            ema21=5970.0, ema50=5990.0, adx=25,
            rsi=35.0, macd_hist=3.0, trend="down",
        )
        signal = strat.generate(features)
        
        assert signal.action == "SELL", f"Expected SELL, got {signal.action}: {signal.metadata}"
        reason = signal.metadata.get("reason", "")
        assert "TREND_CONT_SHORT" in reason
        assert "SL=" in reason
        assert "TP=" in reason


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestTrendContEdgeCases:
    """Edge cases for ATR-adaptive SL/TP."""

    def test_atr_exactly_at_floor(self):
        """ATR=6.0 (exactly floor) → SL=6pts."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)
        
        features = _make_features(atr=6.0, close=6000.0, ema9=5998.0, adx=25)
        signal = strat.generate(features)
        
        if signal.action == "BUY":
            meta = signal.metadata
            sl_dist = 6000.0 - meta["stop_loss"]
            assert sl_dist == pytest.approx(6.0, abs=0.1)

    def test_atr_exactly_at_ceiling(self):
        """ATR=20.0 (exactly ceiling) → SL=20pts."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)
        
        features = _make_features(atr=20.0, close=6000.0, ema9=5995.0, adx=30)
        signal = strat.generate(features)
        
        if signal.action == "BUY":
            meta = signal.metadata
            sl_dist = 6000.0 - meta["stop_loss"]
            assert sl_dist == pytest.approx(20.0, abs=0.1)

    def test_very_high_atr_43(self):
        """ATR=43 (max seen in backtest) → SL clamped to 20pts ceiling."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)
        
        features = _make_features(atr=43.0, close=6000.0, ema9=5990.0, adx=35)
        signal = strat.generate(features)
        
        if signal.action == "BUY":
            meta = signal.metadata
            sl_dist = 6000.0 - meta["stop_loss"]
            assert sl_dist == pytest.approx(20.0, abs=0.1)  # ceiling, not 43
            tp_dist = meta["take_profit"] - 6000.0
            assert tp_dist == pytest.approx(25.0, abs=0.1)  # 20 × 1.25
