"""Tests for extended/overnight hours support in the Gold strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shree.config.gold import GoldStrategyConfig
from shree.strategies.gold.signals import GoldSignalGenerator


def _cfg(extended: bool = True) -> GoldStrategyConfig:
    cfg = GoldStrategyConfig(enabled=True, symbol="MGC")
    cfg.indicators.warmup_bars = 30
    cfg.session.extended_hours_enabled = extended
    cfg.session.session_open_et = "08:20"
    cfg.session.session_close_et = "13:30"
    cfg.session.extended_session_open_et = "18:00"
    cfg.exit.atr_sl_multiplier = 1.5
    cfg.exit.atr_tp_multiplier = 2.5
    cfg.exit.extended_atr_sl_multiplier = 2.5
    cfg.exit.extended_atr_tp_multiplier = 4.0
    cfg.exit.sl_floor_points = 0.1
    cfg.exit.sl_ceiling_points = 100.0
    return cfg


def _gen(cfg: GoldStrategyConfig) -> GoldSignalGenerator:
    return GoldSignalGenerator(
        session=cfg.session,
        indicators=cfg.indicators,
        entry=cfg.entry,
        exit_cfg=cfg.exit,
        tick_size=0.10,
    )


def _ts(hour: int, minute: int) -> pd.Timestamp:
    return pd.Timestamp(f"2026-01-15 {hour:02d}:{minute:02d}:00",
                        tz="America/New_York")


class TestIsExtendedHours:
    """Unit-test the _is_extended_hours helper directly."""

    def test_rth_bar_not_extended(self) -> None:
        gen = _gen(_cfg(extended=True))
        assert not gen._is_extended_hours(_ts(9, 30))
        assert not gen._is_extended_hours(_ts(12, 0))

    def test_pre_open_is_extended(self) -> None:
        gen = _gen(_cfg(extended=True))
        assert gen._is_extended_hours(_ts(7, 0))
        assert gen._is_extended_hours(_ts(8, 0))

    def test_after_close_is_extended(self) -> None:
        gen = _gen(_cfg(extended=True))
        assert gen._is_extended_hours(_ts(18, 30))
        assert gen._is_extended_hours(_ts(23, 0))

    def test_extended_disabled_always_false(self) -> None:
        gen = _gen(_cfg(extended=False))
        assert not gen._is_extended_hours(_ts(7, 0))
        assert not gen._is_extended_hours(_ts(20, 0))

    def test_none_timestamp_not_extended(self) -> None:
        gen = _gen(_cfg(extended=True))
        assert not gen._is_extended_hours(None)


class TestSlTpMultipliers:
    """Verify that _sl_tp uses wider multipliers during extended hours."""

    def test_rth_uses_rth_multipliers(self) -> None:
        cfg = _cfg(extended=True)
        gen = _gen(cfg)
        sl_rth, tp_rth = gen._sl_tp("BUY", 2350.0, atr=10.0, bar_ts=_ts(10, 0))
        # RTH: sl_distance = 10 × 1.5 = 15 pts → SL ≈ 2335
        assert abs(2350.0 - sl_rth - 15.0) < 0.2

    def test_extended_uses_wider_multipliers(self) -> None:
        cfg = _cfg(extended=True)
        gen = _gen(cfg)
        sl_ext, tp_ext = gen._sl_tp("BUY", 2350.0, atr=10.0, bar_ts=_ts(20, 0))
        # Extended: sl_distance = 10 × 2.5 = 25 pts → SL ≈ 2325
        assert abs(2350.0 - sl_ext - 25.0) < 0.2

    def test_extended_tp_wider_than_rth(self) -> None:
        cfg = _cfg(extended=True)
        gen = _gen(cfg)
        _, tp_rth = gen._sl_tp("BUY", 2350.0, atr=10.0, bar_ts=_ts(10, 0))
        _, tp_ext = gen._sl_tp("BUY", 2350.0, atr=10.0, bar_ts=_ts(20, 0))
        assert tp_ext > tp_rth   # extended TP must be farther out

    def test_short_extended_hours_sl_above_entry(self) -> None:
        cfg = _cfg(extended=True)
        gen = _gen(cfg)
        sl, _ = gen._sl_tp("SELL", 2350.0, atr=10.0, bar_ts=_ts(22, 0))
        assert sl > 2350.0    # For SELL, SL is above entry

    def test_no_bar_ts_falls_back_to_rth(self) -> None:
        cfg = _cfg(extended=True)
        gen = _gen(cfg)
        sl_no_ts, _ = gen._sl_tp("BUY", 2350.0, atr=10.0, bar_ts=None)
        sl_rth, _ = gen._sl_tp("BUY", 2350.0, atr=10.0, bar_ts=_ts(10, 0))
        assert abs(sl_no_ts - sl_rth) < 0.01


class TestSessionGate:
    """GoldIntradayStrategy should gate signals to the configured session."""

    def _make_df(self, n: int, start_hour: int = 8, start_min: int = 20) -> pd.DataFrame:
        np.random.seed(77)
        closes = 2350.0 + np.arange(n) * 0.5 + np.random.normal(0, 0.3, n)
        highs = closes + np.abs(np.random.normal(0, 1.0, n))
        lows = closes - np.abs(np.random.normal(0, 1.0, n))
        idx = pd.date_range(
            f"2026-01-15 {start_hour:02d}:{start_min:02d}",
            periods=n, freq="1min", tz="America/New_York",
        )
        return pd.DataFrame(
            {"open": closes, "high": highs, "low": lows, "close": closes,
             "volume": np.ones(n) * 50},
            index=idx,
        )

    def test_outside_session_returns_hold(self) -> None:
        from shree.strategies.gold.regime import GoldRegime
        from shree.strategies.gold.strategy import GoldIntradayStrategy
        cfg = _cfg(extended=False)
        cfg.indicators.warmup_bars = 10
        strategy = GoldIntradayStrategy(cfg)
        # Build a window ending at 02:00 AM — clearly outside session
        df = self._make_df(50, start_hour=1, start_min=0)
        signal = strategy.generate_gold(df)
        assert signal.action == "HOLD"
        assert signal.regime in (GoldRegime.NO_TRADE, GoldRegime.WARMING_UP)
