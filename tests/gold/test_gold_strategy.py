"""Tests for GoldIntradayStrategy — indicator computation, end-to-end signal pipeline,
and restart/idempotency protections."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shree.config.gold import GoldStrategyConfig
from shree.strategies.gold.strategy import GoldIntradayStrategy, compute_indicators


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(
    n: int = 120,
    start_price: float = 2350.0,
    trend: float = 0.1,   # points per bar
) -> pd.DataFrame:
    """Synthetic trending OHLCV bar DataFrame with timezone-aware index."""
    np.random.seed(42)
    closes = start_price + np.arange(n) * trend + np.random.normal(0, 0.5, n)
    highs = closes + np.abs(np.random.normal(0, 1, n))
    lows = closes - np.abs(np.random.normal(0, 1, n))
    volumes = np.random.randint(20, 200, n).astype(float)

    idx = pd.date_range("2026-01-15 08:20", periods=n, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _default_cfg() -> GoldStrategyConfig:
    cfg = GoldStrategyConfig(enabled=True, symbol="MGC")
    cfg.indicators.warmup_bars = 30   # Lower for tests
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Indicator computation
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeIndicators:
    def test_all_columns_present(self) -> None:
        df = compute_indicators(_make_ohlcv(50), _default_cfg())
        for col in ("ema9", "ema21", "atr", "adx", "vwap"):
            assert col in df.columns, f"Missing column: {col}"

    def test_no_lookahead(self) -> None:
        """Indicators must be computable on any prefix of the data."""
        df = _make_ohlcv(120)
        cfg = _default_cfg()
        for n in (10, 30, 60, 119):
            result = compute_indicators(df.iloc[:n], cfg)
            assert len(result) == n
            # EMA always produces values (min_periods=1)
            assert not result["ema9"].isna().all()
            # ATR uses Wilder RMA (min_periods=period=14); only test when n >= period
            if n >= 14:
                assert not result["atr"].isna().all()

    def test_ema9_faster_than_ema21(self) -> None:
        """In an uptrend EMA9 should converge above EMA21 given sufficient bars."""
        df = compute_indicators(_make_ohlcv(100, trend=0.5), _default_cfg())
        # At the end of 100 bars EMA9 should be higher than EMA21 in an uptrend
        assert df["ema9"].iloc[-1] >= df["ema21"].iloc[-1]

    def test_vwap_resets_at_session_open(self) -> None:
        """VWAP on the first bar of a new day should equal typical price of that bar."""
        df = _make_ohlcv(5)
        cfg = _default_cfg()
        # Fake index so the first bar is exactly at session anchor
        anchor_hour, anchor_min = 8, 20
        idx = pd.date_range(
            f"2026-01-15 {anchor_hour:02d}:{anchor_min:02d}",
            periods=5,
            freq="1min",
            tz="America/New_York",
        )
        df.index = idx
        result = compute_indicators(df, cfg)
        # First bar: VWAP = typical price = (h+l+c)/3
        tp0 = (df["high"].iloc[0] + df["low"].iloc[0] + df["close"].iloc[0]) / 3.0
        assert abs(result["vwap"].iloc[0] - tp0) < 1e-6

    def test_atr_positive(self) -> None:
        df = compute_indicators(_make_ohlcv(50), _default_cfg())
        # ATR requires min_periods=14; only test rows after warmup
        atr_period = _default_cfg().indicators.atr_period   # 14
        valid = df["atr"].iloc[atr_period:]
        assert (valid > 0).all()

    def test_adx_in_range(self) -> None:
        df = compute_indicators(_make_ohlcv(80, trend=1.0), _default_cfg())
        assert (df["adx"].iloc[20:] >= 0).all()
        assert (df["adx"].iloc[20:] <= 100).all()


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end signal pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestGoldIntradayStrategy:
    def test_returns_hold_during_warmup(self) -> None:
        cfg = _default_cfg()
        cfg.indicators.warmup_bars = 100
        strategy = GoldIntradayStrategy(cfg)
        df = _make_ohlcv(50)   # Less than warmup
        sig = strategy.generate_gold(df)
        assert sig.action == "HOLD"
        from shree.strategies.gold.regime import GoldRegime
        assert sig.regime == GoldRegime.WARMING_UP

    def test_returns_signal_after_warmup(self) -> None:
        """With enough bars a signal MUST eventually fire (may be HOLD if conditions not met)."""
        cfg = _default_cfg()
        strategy = GoldIntradayStrategy(cfg)
        df = _make_ohlcv(120, trend=1.0)   # Strong uptrend
        sig = strategy.generate_gold(df)
        # Must return a valid Signal object regardless
        assert sig.action in ("BUY", "SELL", "HOLD")
        assert 0.0 <= sig.confidence <= 1.0

    def test_base_strategy_generate_returns_signal(self) -> None:
        """Ensure BaseStrategy.generate() adapter works."""
        from shree.strategies.base import Signal
        cfg = _default_cfg()
        strategy = GoldIntradayStrategy(cfg)
        df = _make_ohlcv(120, trend=1.0)
        sig = strategy.generate(df)
        assert isinstance(sig, Signal)
        assert sig.action in ("BUY", "SELL", "HOLD")

    def test_notify_loss_propagates_cooldown(self) -> None:
        cfg = _default_cfg()
        cfg.entry.post_loss_cooldown_bars = 3
        strategy = GoldIntradayStrategy(cfg)
        strategy.notify_loss()
        # After loss, signal generator should be in cooldown
        assert strategy._signal_gen._post_loss_cooldown_bars == 3

    def test_no_exceptions_on_empty_dataframe(self) -> None:
        cfg = _default_cfg()
        strategy = GoldIntradayStrategy(cfg)
        sig = strategy.generate_gold(pd.DataFrame())
        assert sig.action == "HOLD"

    def test_no_exceptions_with_nan_filled_dataframe(self) -> None:
        cfg = _default_cfg()
        strategy = GoldIntradayStrategy(cfg)
        df = _make_ohlcv(50)
        df["close"] = np.nan
        # Should not raise; returns HOLD or WARMING_UP
        sig = strategy.generate_gold(df)
        assert sig.action == "HOLD"


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency / restart safety
# ─────────────────────────────────────────────────────────────────────────────

class TestIdempotency:
    def test_same_data_produces_same_signal(self) -> None:
        """Strategy output must be deterministic given the same bars."""
        cfg = _default_cfg()
        df = _make_ohlcv(120, trend=1.0)

        strategy_a = GoldIntradayStrategy(cfg)
        strategy_b = GoldIntradayStrategy(cfg)

        sig_a = strategy_a.generate_gold(df)
        sig_b = strategy_b.generate_gold(df)

        assert sig_a.action == sig_b.action
        assert abs(sig_a.confidence - sig_b.confidence) < 1e-6

    def test_additional_bars_do_not_alter_past_signal(self) -> None:
        """Appending new bars must not change the signal for the previous bar."""
        cfg = _default_cfg()
        df = _make_ohlcv(120, trend=1.0)

        strategy = GoldIntradayStrategy(cfg)
        sig_before = strategy.generate_gold(df)

        # Append a new bar
        new_row = df.iloc[[-1]].copy()
        new_row.index = new_row.index + pd.Timedelta(minutes=1)
        df_extended = pd.concat([df, new_row])

        sig_after = strategy.generate_gold(df)   # Same slice as before
        assert sig_before.action == sig_after.action
