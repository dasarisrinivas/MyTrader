import numpy as np
import pandas as pd

from mytrader.config import OneMinuteStrategyConfig
from mytrader.strategies.mes_one_minute import MesOneMinuteTrendStrategy


def _base_df(atr_value: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 08:30", periods=130, freq="T", tz="America/Chicago")
    close = pd.Series(
        np.linspace(4800, 4800 + 0.5 * (len(idx) - 1), len(idx)),
        index=idx,
        dtype=float,
    )
    atr_series = np.linspace(atr_value, atr_value * 1.05, len(idx))
    df = pd.DataFrame(
        {
            "open": close - 0.25,
            "high": close + 0.25,
            "low": close - 0.25,
            "close": close,
            "volume": 1000,
            "ATR_14": atr_series,
            "ADX_14": 25.0,
            "EMA_9": close + 0.1,
            "EMA_21": close - 0.1,
            "RSI_14": 50.0,
            "SESSION_VWAP": close - 1.0,
        },
        index=idx,
    )
    df["PDH"] = df["high"].shift(1).bfill()
    df["PDL"] = df["low"].shift(1).bfill()
    return df


def test_trend_label_uptrend():
    strategy = MesOneMinuteTrendStrategy(
        OneMinuteStrategyConfig(warmup_bars=100, atr_percentile_low=0.0, atr_percentile_high=1.0)
    )
    df = _base_df()
    signal = strategy.generate(df)
    assert signal.metadata["trend_label"] == "UPTREND"
    assert signal.action == "BUY"
    assert signal.metadata["stop_loss"] < signal.metadata["take_profit"]


def test_atr_based_brackets():
    strategy = MesOneMinuteTrendStrategy(
        OneMinuteStrategyConfig(warmup_bars=50, atr_percentile_low=0.0, atr_percentile_high=1.0)
    )
    decision = strategy._enter_with_brackets(
        direction="BUY",
        close=4500.0,
        atr=2.0,
        reason="TEST",
        base_conf=0.7,
        extra_meta={},
    )
    expected_stop = 4500.0 - (2.0 * strategy.config.stop_atr_multiplier)
    expected_tp = 4500.0 + (
        2.0 * strategy.config.stop_atr_multiplier * strategy.config.take_profit_multiple
    )
    assert abs(decision.stop_loss - expected_stop) < 1e-6
    assert abs(decision.take_profit - expected_tp) < 1e-6


def test_atr_extremes_block():
    strategy = MesOneMinuteTrendStrategy(OneMinuteStrategyConfig(warmup_bars=100))
    df = _base_df()
    # Override ATR series to force high-percentile block
    df["ATR_14"] = pd.Series(
        list(np.linspace(1, 10, len(df) - 1)) + [10.5],
        index=df.index,
    )
    signal = strategy.generate(df)
    assert signal.action == "HOLD"
    assert "ATR_TOO_HIGH" in signal.metadata.get("reason", "") or signal.metadata.get("reason") == "ATR_TOO_HIGH"


def test_tiny_candle_filter_blocks():
    strategy = MesOneMinuteTrendStrategy(
        OneMinuteStrategyConfig(warmup_bars=50, atr_percentile_low=0.0, atr_percentile_high=1.0)
    )
    df = _base_df(atr_value=2.5)
    df["high"] = df["close"] + 0.01
    df["low"] = df["close"] - 0.01
    signal = strategy.generate(df)
    assert signal.action == "HOLD"
    assert "TINY_CANDLE" in signal.metadata.get("reason", "") or signal.metadata.get("reason") == "TINY_CANDLE"


def test_pullback_confirmation_rsi():
    strategy = MesOneMinuteTrendStrategy(
        OneMinuteStrategyConfig(warmup_bars=50, atr_percentile_low=0.0, atr_percentile_high=1.0)
    )
    df = _base_df()
    df["RSI_14"] = pd.Series([45, 46, 47, 48, 50, 52, 54, 55, 54, 53] * 13, index=df.index)
    ok, reason = strategy._pullback_confirmation(df)
    assert ok
    assert "RSI_PULLBACK" in reason
