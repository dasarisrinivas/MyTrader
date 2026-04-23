"""Unit tests for the rules_v2 structure utilities."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shree.spy_options.rules_v2 import structure as s  # noqa: E402


def _bars(values):
    t = datetime(2026, 4, 21, 12, 0)
    out = []
    for v in values:
        out.append({
            "date": t, "open": v, "high": v + 0.3, "low": v - 0.3,
            "close": v, "volume": 100000,
        })
        t += timedelta(minutes=5)
    return out


def test_sma_none_when_short():
    assert s.sma([1.0, 2.0], period=5) is None


def test_sma_basic():
    assert s.sma([1.0, 2.0, 3.0, 4.0, 5.0], period=5) == 3.0


def test_ema_basic_monotone():
    vals = list(range(1, 21))
    e = s.ema(vals, 5)
    assert e, "EMA should produce values"
    assert e[-1] > e[0]   # rising


def test_atr_none_for_short_bars():
    assert s.atr(_bars([710.0]), 14) is None


def test_atr_positive_on_chop():
    bars = _bars([710.0 + (i % 3) * 0.5 for i in range(25)])
    a = s.atr(bars, 14)
    assert a is not None and a > 0


def test_session_vwap_equals_average_when_equal_vol():
    bars = _bars([710.0, 711.0, 712.0])
    # typ = close (since h=c+0.3, l=c-0.3 ⇒ (h+l+c)/3 = c)
    v = s.session_vwap(bars)
    assert v is not None
    assert round(v, 2) == round((710 + 711 + 712) / 3, 2)


def test_rsi_trend_up_yields_high_rsi():
    closes = list(range(1, 30))
    r = s.rsi(closes, period=14)
    assert r is not None and r > 60


def test_rsi_trend_down_yields_low_rsi():
    closes = list(range(30, 1, -1))
    r = s.rsi(closes, period=14)
    assert r is not None and r < 40


def test_detect_pivots_finds_local_high():
    bars = _bars([700, 702, 705, 708, 710, 708, 705, 702, 700])
    pivots = s.detect_pivots(bars, lookback=3)
    highs = [p for p in pivots if p.kind == "HIGH"]
    assert highs, "should detect a HIGH pivot at index 4"
    assert highs[0].idx == 4


def test_bearish_rejection_candle():
    bar = {"open": 705, "high": 706, "low": 704.8, "close": 704.9}
    # Upper wick = 706 - max(705,704.9) = 1.0; range = 1.2 ⇒ ratio ~0.83
    assert s.is_bearish_rejection(bar, wick_ratio=0.5)


def test_bullish_rejection_candle():
    bar = {"open": 700, "high": 700.1, "low": 698.5, "close": 700}
    # Lower wick = min(700,700)-698.5 = 1.5; range = 1.6 ⇒ ratio ~0.94
    assert s.is_bullish_rejection(bar, wick_ratio=0.5)


def test_expansion_bar():
    assert s.expansion_bar({"high": 10, "low": 8}, atr_val=1.0, multiple=1.5)
    assert not s.expansion_bar({"high": 10, "low": 9.5}, atr_val=1.0, multiple=1.5)
