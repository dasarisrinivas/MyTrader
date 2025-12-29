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


def test_downtrend_pullback_allows_short_on_reentry():
    data = pd.DataFrame(
        [
            {"close": 100, "EMA_50": 101.5, "RSI_14": 72, "ATR_14": 2.0},
            {"close": 99, "EMA_50": 101.2, "RSI_14": 58, "ATR_14": 2.0},
        ],
        index=pd.date_range("2024-01-01", periods=2, freq="5min"),
    )
    flt = RSITrendPullbackFilter()
    result = flt.evaluate(data)
    assert result.trend == "DOWNTREND"
    assert result.pullback_reentry
    assert result.allow_short


def test_no_trade_band_blocks_entries():
    ema_val = 100.0
    data = pd.DataFrame(
        [
            {"close": ema_val * 1.0001, "EMA_50": ema_val, "RSI_14": 35, "ATR_14": 2.0},
            {"close": ema_val * 1.00005, "EMA_50": ema_val, "RSI_14": 45, "ATR_14": 2.0},
        ],
        index=pd.date_range("2024-01-01", periods=2, freq="5min"),
    )
    flt = RSITrendPullbackFilter({"ema_no_trade_band_pct": 0.0005})
    result = flt.evaluate(data)
    assert result.in_no_trade_band
    assert not result.allow_long and not result.allow_short


def test_overnight_adjusts_thresholds_and_slackens_long_entry():
    data = pd.DataFrame(
        [
            {"close": 101, "EMA_50": 100, "RSI_14": 27, "ATR_14": 2.0},
            {"close": 102, "EMA_50": 100, "RSI_14": 37, "ATR_14": 2.0},
        ],
        index=pd.date_range("2024-01-01 20:00", periods=2, freq="5min"),
    )
    flt = RSITrendPullbackFilter({"overnight_rsi_buffer": 5.0})
    result = flt.evaluate(data, now=data.index[-1].to_pydatetime())
    assert result.session == "OVERNIGHT"
    assert result.pullback_reentry
    assert result.allow_long


def test_rsi_equals_thresholds_is_handled():
    data = pd.DataFrame(
        [
            {"close": 100, "EMA_50": 99, "RSI_14": 30, "ATR_14": 2.0},
            {"close": 101, "EMA_50": 99, "RSI_14": 40, "ATR_14": 2.0},
        ],
        index=pd.date_range("2024-01-01", periods=2, freq="5min"),
    )
    flt = RSITrendPullbackFilter()
    result = flt.evaluate(data)
    assert result.pullback_reentry
    assert result.allow_long
