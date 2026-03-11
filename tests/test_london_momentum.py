"""Tests for Signal G — London Momentum Breakout.

MAR 9 2026: Signal G fires during London session (2-5 AM CST) when
EMA9 crosses EMA21, capturing the first directional impulse from
European liquidity.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock
from shree.strategies.es_fifteen_min import EsFifteenMinStrategy


def _make_config(**overrides):
    """Create a mock config with London-specific defaults."""
    cfg = MagicMock()
    defaults = {
        "ft_pb_stop_mult": 1.5,
        "ft_pb_target_mult": 2.5,
        "ft_or_target_r": 1.5,
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
        "ft_ema9_pb_enabled": False,
        "ft_ema9_pb_stop_mult": 1.2,
        "ft_ema9_pb_target_mult": 1.5,
        "ft_ema9_touch_pct": 0.0015,
        "ft_shorts_enabled": True,
        "ft_short_pb_stop_mult": 1.5,
        "ft_short_pb_target_mult": 1.0,
        "ft_short_or_target_r": 1.0,
        "ft_or_break_max_per_day": 2,
        "ft_fixed_tp_points": 8.0,
        "ft_fixed_tp_points_ema9": 10.0,
        "ft_fixed_tp_points_trend": 12.0,
        "ft_fixed_sl_points": 6.0,
        "ft_fixed_sl_points_ema9": 8.0,
        "ft_fixed_sl_points_trend": 8.0,
        "ft_ema9_sl_atr_mult": 1.0,
        "ft_ema9_sl_floor_pts": 8.0,
        "ft_ema9_sl_ceiling_pts": 20.0,
        "ft_ema9_rr_ratio": 1.25,
        "ft_trend_sl_atr_mult": 1.0,
        "ft_trend_sl_floor_pts": 6.0,
        "ft_trend_sl_ceiling_pts": 20.0,
        "ft_trend_rr_ratio": 1.25,
        "ft_trend_cont_enabled": False,
        "ft_trend_cont_stop_mult": 1.0,
        "ft_trend_cont_target_mult": 2.0,
        "ft_trend_cont_adx_min": 25.0,
        "ft_trend_cont_ema9_pct": 0.003,
        "ft_trend_cont_max_ext_pts": 30.0,
        "ft_trend_cont_gap_adx_min": 25.0,
        "ft_trend_cont_max_per_day": 2,
        # London defaults
        "ft_london_enabled": True,
        "ft_london_start_hour": 2,
        "ft_london_start_minute": 0,
        "ft_london_end_hour": 5,
        "ft_london_end_minute": 0,
        "ft_london_adx_min": 15.0,
        "ft_london_sl_points": 5.0,
        "ft_london_sl_atr_cap": 1.0,
        "ft_london_tp_points": 8.0,
        "ft_london_max_per_day": 1,
        # Overnight SL/TP scaling
        "ft_overnight_sl_mult": 1.2,
        "ft_overnight_tp_mult": 0.85,
        # Entry/RTH windows (24h)
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

    def getattr_side_effect(name, default=None):
        return defaults.get(name, default)

    cfg.__class__ = type("OneMinuteStrategyConfig", (), {})
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_df(bars, tz="US/Central"):
    """Create a 15m bar DataFrame with indicators.

    Each bar dict: {time, open, high, low, close, ema9, ema21, ema50, atr, adx, rsi, macd_hist}
    """
    rows = []
    timestamps = []
    for bar in bars:
        rows.append({
            "open": bar.get("open", bar["close"]),
            "high": bar.get("high", bar["close"] + 1),
            "low": bar.get("low", bar["close"] - 1),
            "close": bar["close"],
            "volume": bar.get("volume", 1000),
            "EMA_9": bar.get("ema9", bar["close"]),
            "EMA_21": bar.get("ema21", bar["close"]),
            "EMA_50": bar.get("ema50", bar["close"] - 10),
            "ATR_14": bar.get("atr", 3.0),
            "ADX_14": bar.get("adx", 16.0),
            "RSI_14": bar.get("rsi", 50.0),
            "MACDhist_12_26_9": bar.get("macd_hist", 0.5),
        })
        timestamps.append(pd.Timestamp(bar["time"], tz=tz))
    index = pd.DatetimeIndex(timestamps)
    return pd.DataFrame(rows, index=index)


def _pad_warmup(bars, n=60):
    """Pad bars with n warmup rows at the beginning so len >= 60."""
    if len(bars) >= n:
        return bars
    first = bars[0]
    warmup = []
    # Create warmup bars with same indicators but earlier times
    base_time = pd.Timestamp(first["time"], tz="US/Central")
    for i in range(n - len(bars)):
        t = base_time - pd.Timedelta(minutes=15 * (n - len(bars) - i))
        warmup_bar = {
            "time": str(t),
            "close": first.get("close", 6800),
            "open": first.get("close", 6800),
            "high": first.get("close", 6800) + 1,
            "low": first.get("close", 6800) - 1,
            "ema9": first.get("ema21", 6800),  # No cross in warmup
            "ema21": first.get("ema21", 6800),
            "ema50": first.get("ema50", 6790),
            "atr": first.get("atr", 3.0),
            "adx": first.get("adx", 16.0),
        }
        warmup.append(warmup_bar)
    return warmup + bars


class TestLondonMomentumSignalG:
    """Tests for Signal G — London Momentum Breakout."""

    def test_bullish_cross_fires_during_london(self):
        """Signal G fires BUY when EMA9 crosses above EMA21 during London."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)

        # Previous bar: EMA9 < EMA21 (no cross yet)
        # Current bar:  EMA9 > EMA21 (bullish cross!)
        bars = _pad_warmup([
            {
                "time": "2026-03-10 03:00",
                "close": 6800, "open": 6798, "high": 6802, "low": 6797,
                "ema9": 6799.0, "ema21": 6800.5, "ema50": 6790.0,
                "atr": 3.0, "adx": 16.0,
            },
            {
                "time": "2026-03-10 03:15",
                "close": 6805, "open": 6800, "high": 6806, "low": 6799,
                "ema9": 6803.0, "ema21": 6801.0, "ema50": 6790.0,
                "atr": 3.5, "adx": 17.0,
            },
        ])
        df = _make_df(bars)
        signal = strat.generate(df)
        assert signal.action == "BUY"
        assert "LONDON_MOMENTUM_LONG" in signal.metadata.get("reason", "")

    def test_bearish_cross_fires_during_london(self):
        """Signal G fires SELL when EMA9 crosses below EMA21 during London."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)

        # Previous bar: EMA9 > EMA21
        # Current bar:  EMA9 < EMA21 (bearish cross)
        bars = _pad_warmup([
            {
                "time": "2026-03-10 03:00",
                "close": 6805, "open": 6808, "high": 6809, "low": 6804,
                "ema9": 6806.0, "ema21": 6804.5, "ema50": 6810.0,
                "atr": 3.0, "adx": 16.0,
            },
            {
                "time": "2026-03-10 03:15",
                "close": 6798, "open": 6805, "high": 6806, "low": 6797,
                "ema9": 6800.0, "ema21": 6803.0, "ema50": 6810.0,
                "atr": 3.5, "adx": 17.0,
            },
        ])
        df = _make_df(bars)
        signal = strat.generate(df)
        assert signal.action == "SELL"
        assert "LONDON_MOMENTUM_SHORT" in signal.metadata.get("reason", "")

    def test_no_fire_outside_london_hours(self):
        """Signal G should NOT fire outside the London window."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)

        # Same crossover conditions but at 7 AM CST (outside 2-5 AM)
        bars = _pad_warmup([
            {
                "time": "2026-03-10 07:00",
                "close": 6800, "open": 6798, "high": 6802, "low": 6797,
                "ema9": 6799.0, "ema21": 6800.5, "ema50": 6790.0,
                "atr": 3.0, "adx": 16.0,
            },
            {
                "time": "2026-03-10 07:15",
                "close": 6805, "open": 6800, "high": 6806, "low": 6799,
                "ema9": 6803.0, "ema21": 6801.0, "ema50": 6790.0,
                "atr": 3.5, "adx": 17.0,
            },
        ])
        df = _make_df(bars)
        signal = strat.generate(df)
        # Should be HOLD (outside London, but may fire as another signal type
        # if conditions happen to match — key is LONDON_MOMENTUM shouldn't be in reason)
        if signal.action != "HOLD":
            assert "LONDON_MOMENTUM" not in signal.metadata.get("reason", "")

    def test_no_fire_when_disabled(self):
        """Signal G should NOT fire when ft_london_enabled=False."""
        cfg = _make_config(ft_london_enabled=False)
        strat = EsFifteenMinStrategy(cfg)

        bars = _pad_warmup([
            {
                "time": "2026-03-10 03:00",
                "close": 6800, "open": 6798, "high": 6802, "low": 6797,
                "ema9": 6799.0, "ema21": 6800.5, "ema50": 6790.0,
                "atr": 3.0, "adx": 16.0,
            },
            {
                "time": "2026-03-10 03:15",
                "close": 6805, "open": 6800, "high": 6806, "low": 6799,
                "ema9": 6803.0, "ema21": 6801.0, "ema50": 6790.0,
                "atr": 3.5, "adx": 17.0,
            },
        ])
        df = _make_df(bars)
        signal = strat.generate(df)
        assert "LONDON_MOMENTUM" not in signal.metadata.get("reason", "")

    def test_no_fire_when_no_cross(self):
        """Signal G should NOT fire when EMA9 and EMA21 don't cross."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)

        # Both bars: EMA9 > EMA21 (no cross, already above)
        bars = _pad_warmup([
            {
                "time": "2026-03-10 03:00",
                "close": 6805, "open": 6803, "high": 6806, "low": 6802,
                "ema9": 6804.0, "ema21": 6802.0, "ema50": 6790.0,
                "atr": 3.0, "adx": 16.0,
            },
            {
                "time": "2026-03-10 03:15",
                "close": 6807, "open": 6805, "high": 6808, "low": 6804,
                "ema9": 6806.0, "ema21": 6803.0, "ema50": 6790.0,
                "atr": 3.5, "adx": 17.0,
            },
        ])
        df = _make_df(bars)
        signal = strat.generate(df)
        assert "LONDON_MOMENTUM" not in signal.metadata.get("reason", "")

    def test_max_per_day_limit(self):
        """Signal G should not fire more than ft_london_max_per_day times."""
        cfg = _make_config(ft_london_max_per_day=1)
        strat = EsFifteenMinStrategy(cfg)

        # First cross — should fire
        bars1 = _pad_warmup([
            {
                "time": "2026-03-10 03:00",
                "close": 6800, "open": 6798, "high": 6802, "low": 6797,
                "ema9": 6799.0, "ema21": 6800.5, "ema50": 6790.0,
                "atr": 3.0, "adx": 16.0,
            },
            {
                "time": "2026-03-10 03:15",
                "close": 6805, "open": 6800, "high": 6806, "low": 6799,
                "ema9": 6803.0, "ema21": 6801.0, "ema50": 6790.0,
                "atr": 3.5, "adx": 17.0,
            },
        ])
        df1 = _make_df(bars1)
        signal1 = strat.generate(df1)
        assert signal1.action == "BUY"
        assert "LONDON_MOMENTUM_LONG" in signal1.metadata.get("reason", "")

        # Second cross — should NOT fire (maxed out)
        bars2 = _pad_warmup([
            {
                "time": "2026-03-10 04:00",
                "close": 6810, "open": 6812, "high": 6813, "low": 6809,
                "ema9": 6811.0, "ema21": 6812.0, "ema50": 6790.0,
                "atr": 3.0, "adx": 17.0,
            },
            {
                "time": "2026-03-10 04:15",
                "close": 6815, "open": 6810, "high": 6816, "low": 6809,
                "ema9": 6814.0, "ema21": 6811.0, "ema50": 6790.0,
                "atr": 3.5, "adx": 18.0,
            },
        ])
        df2 = _make_df(bars2)
        signal2 = strat.generate(df2)
        assert "LONDON_MOMENTUM" not in signal2.metadata.get("reason", "")

    def test_adx_too_low_blocks(self):
        """Signal G should NOT fire when ADX < ft_london_adx_min."""
        cfg = _make_config(ft_london_adx_min=15.0)
        strat = EsFifteenMinStrategy(cfg)

        # ADX = 12 (below 15)
        bars = _pad_warmup([
            {
                "time": "2026-03-10 03:00",
                "close": 6800, "open": 6798, "high": 6802, "low": 6797,
                "ema9": 6799.0, "ema21": 6800.5, "ema50": 6790.0,
                "atr": 3.0, "adx": 12.0,
            },
            {
                "time": "2026-03-10 03:15",
                "close": 6805, "open": 6800, "high": 6806, "low": 6799,
                "ema9": 6803.0, "ema21": 6801.0, "ema50": 6790.0,
                "atr": 3.5, "adx": 13.0,
            },
        ])
        df = _make_df(bars)
        signal = strat.generate(df)
        assert "LONDON_MOMENTUM" not in signal.metadata.get("reason", "")

    def test_sl_tp_values(self):
        """Signal G should use structural SL and data-driven TP."""
        cfg = _make_config(ft_london_sl_points=5.0, ft_london_tp_points=8.0)
        strat = EsFifteenMinStrategy(cfg)

        bars = _pad_warmup([
            {
                "time": "2026-03-10 03:00",
                "close": 6800, "open": 6798, "high": 6802, "low": 6797,
                "ema9": 6799.0, "ema21": 6800.5, "ema50": 6790.0,
                "atr": 3.0, "adx": 16.0,
            },
            {
                "time": "2026-03-10 03:15",
                "close": 6805, "open": 6800, "high": 6806, "low": 6799,
                "ema9": 6803.0, "ema21": 6801.0, "ema50": 6790.0,
                "atr": 3.5, "adx": 17.0,
            },
        ])
        df = _make_df(bars)
        signal = strat.generate(df)
        assert signal.action == "BUY"
        # Structural SL: max(5.0, (6805-6801)+0.5=4.5) = 5.0 (floor wins)
        #   capped at ATR×1.0 = 3.5 → 3.5, re-floored to 5.0
        # SL = 6805 - 5.0 = 6800.0
        # TP = 6805 + 8 = 6813.0
        assert abs(signal.metadata["stop_loss"] - 6800.0) < 0.01
        assert abs(signal.metadata["take_profit"] - 6813.0) < 0.01

    def test_sl_structural_wider_gap(self):
        """Structural SL should widen when EMA21 gap is large and ATR allows."""
        cfg = _make_config(ft_london_sl_points=5.0, ft_london_sl_atr_cap=1.0)
        strat = EsFifteenMinStrategy(cfg)

        # EMA21 gap = 8pts, ATR = 10 → structural = max(5, 8+0.5)=8.5,
        # cap at 10×1.0=10 → stays 8.5
        bars = _pad_warmup([
            {
                "time": "2026-03-10 03:00",
                "close": 6800, "open": 6798, "high": 6802, "low": 6797,
                "ema9": 6799.0, "ema21": 6800.5, "ema50": 6790.0,
                "atr": 8.0, "adx": 16.0,
            },
            {
                "time": "2026-03-10 03:15",
                "close": 6810, "open": 6800, "high": 6811, "low": 6799,
                "ema9": 6808.0, "ema21": 6802.0, "ema50": 6790.0,
                "atr": 10.0, "adx": 17.0,
            },
        ])
        df = _make_df(bars)
        signal = strat.generate(df)
        assert signal.action == "BUY"
        # SL = max(4, (6810-6802)+0.5=8.5) = 8.5, cap=min(8.5, 10)=8.5
        # SL price = 6810 - 8.5 = 6801.5
        assert abs(signal.metadata["stop_loss"] - 6801.5) < 0.01

    def test_sl_atr_cap_limits_wide_gap(self):
        """ATR cap should prevent SL from being too wide on big-body bars."""
        cfg = _make_config(ft_london_sl_points=5.0, ft_london_sl_atr_cap=1.0)
        strat = EsFifteenMinStrategy(cfg)

        # EMA21 gap = 12pts but ATR only 5 → structural=12.5, cap at 5 → floor=5
        bars = _pad_warmup([
            {
                "time": "2026-03-10 03:00",
                "close": 6800, "open": 6798, "high": 6802, "low": 6797,
                "ema9": 6799.0, "ema21": 6800.5, "ema50": 6790.0,
                "atr": 4.0, "adx": 16.0,
            },
            {
                "time": "2026-03-10 03:15",
                "close": 6815, "open": 6800, "high": 6816, "low": 6799,
                "ema9": 6812.0, "ema21": 6803.0, "ema50": 6790.0,
                "atr": 5.0, "adx": 17.0,
            },
        ])
        df = _make_df(bars)
        signal = strat.generate(df)
        assert signal.action == "BUY"
        # SL = max(5, (6815-6803)+0.5=12.5) = 12.5
        #   cap at ATR×1.0 = 5.0 → 5.0
        #   re-floor: max(5.0, 5.0) = 5.0
        # SL price = 6815 - 5.0 = 6810.0
        assert abs(signal.metadata["stop_loss"] - 6810.0) < 0.01

    def test_session_type_is_london(self):
        """Signal G metadata should indicate LONDON session."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)

        bars = _pad_warmup([
            {
                "time": "2026-03-10 03:00",
                "close": 6800, "open": 6798, "high": 6802, "low": 6797,
                "ema9": 6799.0, "ema21": 6800.5, "ema50": 6790.0,
                "atr": 3.0, "adx": 16.0,
            },
            {
                "time": "2026-03-10 03:15",
                "close": 6805, "open": 6800, "high": 6806, "low": 6799,
                "ema9": 6803.0, "ema21": 6801.0, "ema50": 6790.0,
                "atr": 3.5, "adx": 17.0,
            },
        ])
        df = _make_df(bars)
        signal = strat.generate(df)
        assert signal.metadata.get("session_type") == "LONDON"
        assert signal.metadata.get("entry_type") == "london_momentum"

    def test_bearish_bar_blocks_long_cross(self):
        """Bullish cross but bearish candle should NOT fire BUY."""
        cfg = _make_config()
        strat = EsFifteenMinStrategy(cfg)

        # Cross happens but candle is bearish (close < open)
        bars = _pad_warmup([
            {
                "time": "2026-03-10 03:00",
                "close": 6800, "open": 6798, "high": 6802, "low": 6797,
                "ema9": 6799.0, "ema21": 6800.5, "ema50": 6790.0,
                "atr": 3.0, "adx": 16.0,
            },
            {
                "time": "2026-03-10 03:15",
                "close": 6802, "open": 6806, "high": 6807, "low": 6801,
                "ema9": 6803.0, "ema21": 6801.0, "ema50": 6790.0,
                "atr": 3.5, "adx": 17.0,
            },
        ])
        df = _make_df(bars)
        signal = strat.generate(df)
        assert "LONDON_MOMENTUM_LONG" not in signal.metadata.get("reason", "")

    def test_shorts_disabled_blocks_bearish_cross(self):
        """Bearish London cross should NOT fire when shorts disabled."""
        cfg = _make_config(ft_shorts_enabled=False)
        strat = EsFifteenMinStrategy(cfg)

        bars = _pad_warmup([
            {
                "time": "2026-03-10 03:00",
                "close": 6805, "open": 6808, "high": 6809, "low": 6804,
                "ema9": 6806.0, "ema21": 6804.5, "ema50": 6810.0,
                "atr": 3.0, "adx": 16.0,
            },
            {
                "time": "2026-03-10 03:15",
                "close": 6798, "open": 6805, "high": 6806, "low": 6797,
                "ema9": 6800.0, "ema21": 6803.0, "ema50": 6810.0,
                "atr": 3.5, "adx": 17.0,
            },
        ])
        df = _make_df(bars)
        signal = strat.generate(df)
        assert "LONDON_MOMENTUM_SHORT" not in signal.metadata.get("reason", "")

    def test_daily_reset_clears_counter(self):
        """London counter should reset on new trading day."""
        cfg = _make_config(ft_london_max_per_day=1)
        strat = EsFifteenMinStrategy(cfg)

        # Fire on day 1
        bars1 = _pad_warmup([
            {
                "time": "2026-03-10 03:00",
                "close": 6800, "open": 6798, "high": 6802, "low": 6797,
                "ema9": 6799.0, "ema21": 6800.5, "ema50": 6790.0,
                "atr": 3.0, "adx": 16.0,
            },
            {
                "time": "2026-03-10 03:15",
                "close": 6805, "open": 6800, "high": 6806, "low": 6799,
                "ema9": 6803.0, "ema21": 6801.0, "ema50": 6790.0,
                "atr": 3.5, "adx": 17.0,
            },
        ])
        df1 = _make_df(bars1)
        signal1 = strat.generate(df1)
        assert signal1.action == "BUY"

        # Fire on day 2 — should work (counter reset)
        bars2 = _pad_warmup([
            {
                "time": "2026-03-11 03:00",
                "close": 6820, "open": 6818, "high": 6822, "low": 6817,
                "ema9": 6819.0, "ema21": 6820.5, "ema50": 6810.0,
                "atr": 3.0, "adx": 16.0,
            },
            {
                "time": "2026-03-11 03:15",
                "close": 6825, "open": 6820, "high": 6826, "low": 6819,
                "ema9": 6823.0, "ema21": 6821.0, "ema50": 6810.0,
                "atr": 3.5, "adx": 17.0,
            },
        ])
        df2 = _make_df(bars2)
        signal2 = strat.generate(df2)
        assert signal2.action == "BUY"
        assert "LONDON_MOMENTUM_LONG" in signal2.metadata.get("reason", "")
