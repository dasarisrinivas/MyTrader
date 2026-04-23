"""Unit tests for PcRatioAlignmentGate."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shree.spy_options.rules_v2.pc_ratio_alignment import PcRatioAlignmentGate  # noqa: E402
from shree.spy_options.rules_v2.regime import (  # noqa: E402
    RANGE_BOUND,
    TRANSITION,
    TREND_DOWN,
    TREND_UP,
    RegimeV2Context,
)


def _ctx(regime: str) -> RegimeV2Context:
    return RegimeV2Context(
        regime=regime, vwap=710.0, vwap_slope=0.0, atr_ratio=1.0,
        pivots_recent=0, has_hhhl=False, has_lhll=False,
        vwap_crosses_30m=0, spy_vs_vwap=0.0, timestamp=datetime.utcnow(),
    )


def _bars_with_expansion(direction: str):
    """Build 30 bars with one strong expansion bar at the end."""
    out = []
    t = datetime(2026, 4, 21, 12, 0)
    price = 710.0
    for _ in range(29):
        out.append({
            "date": t, "open": price, "high": price + 0.3, "low": price - 0.3,
            "close": price + 0.05, "volume": 100000,
        })
        t += timedelta(minutes=5)
        price += 0.05
    # Expansion bar — 2× normal range, closes in trend direction
    if direction == "P":
        out.append({
            "date": t, "open": price,
            "high": price + 0.1, "low": price - 1.5,
            "close": price - 1.3, "volume": 200000,
        })
    else:
        out.append({
            "date": t, "open": price,
            "high": price + 1.5, "low": price - 0.1,
            "close": price + 1.3, "volume": 200000,
        })
    return out


def test_blocks_in_range_bound():
    gate = PcRatioAlignmentGate()
    r = gate.check("P", _ctx(RANGE_BOUND), _bars_with_expansion("P"))
    assert not r.allowed
    assert "RANGE_BOUND" in r.reason


def test_blocks_in_transition():
    gate = PcRatioAlignmentGate()
    r = gate.check("P", _ctx(TRANSITION), _bars_with_expansion("P"))
    assert not r.allowed
    assert "TRANSITION" in r.reason


def test_blocks_bearish_pc_in_trend_up():
    gate = PcRatioAlignmentGate()
    r = gate.check("P", _ctx(TREND_UP), _bars_with_expansion("P"))
    assert not r.allowed
    assert "not aligned" in r.reason


def test_allows_bearish_pc_in_trend_down():
    gate = PcRatioAlignmentGate()
    r = gate.check("P", _ctx(TREND_DOWN), _bars_with_expansion("P"))
    assert r.allowed, r.reason
