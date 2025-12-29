import pandas as pd

from mytrader.features.indicators import rsi_series


def test_rsi_handles_short_history_gracefully():
    close = pd.Series([100.0, 101.0])
    rsi = rsi_series(close, period=14)
    assert len(rsi) == len(close)
    assert not rsi.isna().any()
    assert all(0 <= v <= 100 for v in rsi)


def test_rsi_trends_higher_on_uptrend():
    close = pd.Series(range(1, 30))
    rsi = rsi_series(close, period=14)
    assert rsi.iloc[-1] > 60


def test_rsi_trends_lower_on_downtrend():
    close = pd.Series(range(30, 1, -1))
    rsi = rsi_series(close, period=14)
    assert rsi.iloc[-1] < 40
