"""Unit tests for RegimeV2Detector.

Verifies the classifier short-circuits correctly on each leg of the
decision tree: insufficient data → TRANSITION, flat chop → RANGE_BOUND,
shallow decline → TREND_DOWN, shallow rally → TREND_UP.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shree.spy_options.rules_v2.config import RegimeV2Config  # noqa: E402
from shree.spy_options.rules_v2.regime import (  # noqa: E402
    RANGE_BOUND,
    TRANSITION,
    TREND_DOWN,
    TREND_UP,
    RegimeV2Detector,
)


ET = ZoneInfo("America/New_York")


def _bar(dt, o, h, l, c, v=100000):
    return {"date": dt, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _session(start_price: float, deltas: list[float], v: int = 100000):
    """Build a session of 5m bars, each advancing by the given delta.

    The bar's high/low wrap the close with a fixed half-range of 0.15.
    """
    bars = []
    t = datetime(2026, 4, 21, 9, 30, tzinfo=ET)
    price = start_price
    for d in deltas:
        o = price
        c = price + d
        h = max(o, c) + 0.15
        l = min(o, c) - 0.15
        bars.append(_bar(t, o, h, l, c, v))
        t += timedelta(minutes=5)
        price = c
    return bars


def _zigzag_trend(start_price: float, direction: str = "down", v: int = 100000):
    """30-bar zig-zag tape with visible swings + expanding ATR + real drift.

    Structure: five 6-bar cycles, each of 4 trend-direction bars followed by
    2 counter-trend bars. Trend-leg magnitude grows each cycle (0.50 → 1.60)
    while counter-leg magnitude grows more slowly (0.50 → 0.90), so:

      • Net drift over 30 bars ≈ 13 pts in ``direction`` (≈1.8%) — enough
        that cumulative session-VWAP slope clears the 0.00010 threshold.
      • Recent bars have bigger TRs than older bars — ATR(5)/ATR(20) > 1.1.
      • Constant 0.30 half-range prevents tiny floating-point biases from
        making every subsequent bar a hair lower than its predecessor, so
        LOW pivots can actually form at each cycle's turning point.
      • The 4:2 trend:counter ratio is asymmetric enough that the counter-
        bounce end (cycle bar 5) sits clearly above the next cycle's first
        3 bars' lows, yielding HIGH pivots; and the trend leg end (cycle
        bar 3) sits below the preceding counter-bounce bars and the
        following 3 counter bars, yielding LOW pivots.
    """
    sign = -1.0 if direction == "down" else 1.0
    mags_trend = [0.50, 0.75, 1.00, 1.30, 1.60]
    mags_counter = [0.50, 0.60, 0.70, 0.80, 0.90]
    half = 0.30
    bars = []
    t = datetime(2026, 4, 21, 9, 30, tzinfo=ET)
    price = start_price
    for cyc in range(5):
        mt = mags_trend[cyc]
        mc = mags_counter[cyc]
        # 4 trend bars
        for _ in range(4):
            o, c = price, price + sign * mt
            bars.append(_bar(t, o, max(o, c) + half, min(o, c) - half, c, v))
            t += timedelta(minutes=5)
            price = c
        # 2 counter bars
        for _ in range(2):
            o, c = price, price + (-sign) * mc
            bars.append(_bar(t, o, max(o, c) + half, min(o, c) - half, c, v))
            t += timedelta(minutes=5)
            price = c
    return bars


def test_insufficient_bars_returns_transition():
    det = RegimeV2Detector()
    bars = _session(710.0, [0.1] * 5)  # only 5 bars; min_bars=22
    ctx = det.classify(bars, bars[-1]["close"])
    assert ctx.regime == TRANSITION
    assert "insufficient" in ctx.reasons[0]


def test_flat_chop_classifies_range_bound():
    det = RegimeV2Detector()
    # 40 bars oscillating ±0.10 around 710
    deltas = [0.10 if i % 2 == 0 else -0.10 for i in range(40)]
    bars = _session(710.0, deltas)
    ctx = det.classify(bars, bars[-1]["close"])
    # Slope should be ~0, atr ratio low ⇒ RANGE_BOUND or TRANSITION (not TREND)
    assert ctx.regime in (RANGE_BOUND, TRANSITION)
    assert ctx.regime != TREND_UP and ctx.regime != TREND_DOWN


def test_shallow_decline_classifies_trend_down():
    det = RegimeV2Detector()
    # A pure monotonic staircase produces NO swing pivots (every new low is
    # the new lowest), so detect_pivots finds nothing. Use a zig-zag tape
    # with expanding ranges so LH+LL pivots form and ATR expands.
    bars = _zigzag_trend(710.0, "down")
    ctx = det.classify(bars, bars[-1]["close"])
    assert ctx.regime == TREND_DOWN, f"got {ctx.regime}; reasons={ctx.reasons}"
    assert ctx.vwap_slope < 0
    assert ctx.spy_vs_vwap < 0


def test_shallow_rally_classifies_trend_up():
    det = RegimeV2Detector()
    bars = _zigzag_trend(700.0, "up")
    ctx = det.classify(bars, bars[-1]["close"])
    assert ctx.regime == TREND_UP, f"got {ctx.regime}; reasons={ctx.reasons}"
    assert ctx.vwap_slope > 0
    assert ctx.spy_vs_vwap > 0


def test_custom_config_threshold_applies():
    # With a very strict slope threshold, a shallow trend should NOT classify.
    strict = RegimeV2Config(trend_vwap_slope=0.005)
    det = RegimeV2Detector(strict)
    bars = _zigzag_trend(710.0, "down")
    ctx = det.classify(bars, bars[-1]["close"])
    assert ctx.regime != TREND_DOWN
