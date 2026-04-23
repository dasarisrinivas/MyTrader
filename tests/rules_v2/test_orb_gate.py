"""Unit tests for OrbGate.

Covers:
- Hard time gate: 11:10 ET → blocked (the Apr 21 11:11 carryover signal)
- In-window + no confirmation → blocked
- In-window + confirmation + volume + VWAP distance → allowed
- Direction inconsistency with VWAP → blocked
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shree.spy_options.rules_v2.orb_gate import OrbGate  # noqa: E402


ET = ZoneInfo("America/New_York")


def _bar(dt, o, h, l, c, v=100000):
    return {"date": dt, "open": o, "high": h, "low": l, "close": c, "volume": v}


def test_blocks_outside_time_window():
    """11:10 ET is outside the 09:45–11:00 ET window — matches Apr 21 carryover."""
    gate = OrbGate()
    now = datetime(2026, 4, 21, 11, 10, tzinfo=ET)
    bars = [_bar(now, 710.0, 711.0, 709.0, 710.5)]
    result = gate.check(
        bars=bars, direction="P",
        orb_high=710.87, orb_low=708.75,
        rsi_value=45.0, vwap_value=710.3, spy_price=710.25, now=now,
    )
    assert not result.allowed
    assert "time-gate" in result.reason


def test_blocks_without_confirmation():
    gate = OrbGate()
    now = datetime(2026, 4, 21, 10, 20, tzinfo=ET)
    # bullish setup but close is below ORB high → not confirmed
    bars = [
        _bar(now - timedelta(minutes=5), 710.5, 710.9, 710.2, 710.6, 150000),
        _bar(now, 710.6, 710.8, 710.3, 710.5, 180000),
    ]
    result = gate.check(
        bars=bars, direction="C",
        orb_high=711.0, orb_low=709.0,
        rsi_value=55.0, vwap_value=710.0, spy_price=710.5, now=now,
    )
    assert not result.allowed
    assert "not held" in result.reason


def test_allows_confirmed_breakout_in_window():
    gate = OrbGate()
    now = datetime(2026, 4, 21, 10, 15, tzinfo=ET)
    # 20 prior bars for avg volume baseline, then 2 confirmation bars
    bars = []
    t = now - timedelta(minutes=5 * 22)
    for _ in range(20):
        bars.append(_bar(t, 710.5, 710.9, 710.1, 710.6, 100000))
        t += timedelta(minutes=5)
    # Two closes above ORB high, strong volume
    bars.append(_bar(t, 710.6, 711.4, 710.5, 711.3, 180000))
    t += timedelta(minutes=5)
    bars.append(_bar(t, 711.3, 711.8, 711.1, 711.6, 220000))

    result = gate.check(
        bars=bars, direction="C",
        orb_high=711.0, orb_low=709.0,
        rsi_value=55.0, vwap_value=710.0, spy_price=711.6, now=now,
    )
    assert result.allowed, result.reason


def test_blocks_vwap_direction_inconsistency():
    """Bullish ORB with price below VWAP is a classic fakeout — must block."""
    gate = OrbGate()
    now = datetime(2026, 4, 21, 10, 15, tzinfo=ET)
    bars = []
    t = now - timedelta(minutes=5 * 22)
    for _ in range(20):
        bars.append(_bar(t, 710.5, 710.9, 710.1, 710.6, 100000))
        t += timedelta(minutes=5)
    bars.append(_bar(t, 710.6, 711.4, 710.5, 711.3, 180000))
    t += timedelta(minutes=5)
    bars.append(_bar(t, 711.3, 711.8, 711.1, 711.6, 220000))

    result = gate.check(
        bars=bars, direction="C",
        orb_high=711.0, orb_low=709.0,
        rsi_value=55.0,
        # VWAP clearly above price so distance > min_vwap_distance_pct (0.0015)
        # and the direction-inconsistency check (not the hugging check) is the blocker.
        vwap_value=713.5,
        spy_price=711.6, now=now,
    )
    assert not result.allowed
    assert "below VWAP" in result.reason
