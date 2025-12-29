import pandas as pd

from mytrader.hybrid.rsi_trend_filter import RSITrendPullbackFilter


def test_pullback_reentry_allows_long_when_trend_up():
    data = pd.DataFrame(
        [
            {"close": 100, "EMA_50": 99.5, "RSI_14": 28, "ATR_14": 5},
            {"close": 101, "EMA_50": 99.6, "RSI_14": 42, "ATR_14": 5},
        ],
        index=pd.date_range("2024-01-01", periods=2, freq="5min"),
    )
    flt = RSITrendPullbackFilter()
    result = flt.evaluate(data)
    assert result.trend == "UPTREND"
    assert result.pullback_reentry
    assert result.allow_long


def test_pullback_filter_blocks_when_atr_out_of_range():
    data = pd.DataFrame(
        [
            {"close": 100, "EMA_50": 101, "RSI_14": 75, "ATR_14": 0.01},
            {"close": 99, "EMA_50": 100.5, "RSI_14": 65, "ATR_14": 0.01},
        ],
        index=pd.date_range("2024-01-01", periods=2, freq="5min"),
    )
    flt = RSITrendPullbackFilter({"atr_min_threshold": 0.5})
    result = flt.evaluate(data)
    assert not result.allow_short
    assert not result.atr_ok
