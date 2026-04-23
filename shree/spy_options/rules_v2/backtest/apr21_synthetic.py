"""Deterministic Apr-21-2026 synthetic tape.

Rebuilds the shape of the real Apr 21 day as described in the postmortem:

  • 09:30–10:40 ET: SPY hugs 710.45 inside the 708.75 / 710.87 ORB.
    Very tight range (< 0.10%), chop. — matches "morning chop-blocked".
  • 10:40–12:00 ET: Slow drift higher inside the range.
  • 12:07 ET: SPY rolls over. Start of the trend-down leg.
  • 12:07–14:44 ET: Series of shallow down-legs with VWAP-reverting
    pullbacks. Total decline ~710 → 703 (≈ 1%). VIX creeps 19.0 → 19.6.
  • 14:44–15:30 ET: Bounce off 703 back to ~705 (mean reversion).
  • 15:30–16:00 ET: Drift sideways. Close near 704.

Legacy "signal candidates" mirror the nine signals fired by the real bot
so we can replay them through the v2 filter and verify blocking/allowing.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _bar(dt: datetime, o: float, h: float, l: float, c: float, v: int = 120000) -> Dict:
    return {"date": dt, "open": o, "high": h, "low": l, "close": c, "volume": v}


def apr21_bars() -> List[Dict]:
    """Produce a 5m bar series from 09:30 to 16:00 ET on 2026-04-21.

    Prices are intentionally *coarse* — the goal is to exercise the regime
    classifier, ORB gate, continuation detector, and throttle, not to fit
    the real tick-by-tick tape.
    """
    bars: List[Dict] = []
    t = datetime(2026, 4, 21, 9, 30, tzinfo=ET)

    # ── Morning chop inside ORB (09:30–10:40 ET) ───────────────────────────
    #   ORB high = 710.87, low = 708.75. Keep tight oscillation.
    chop = [
        (710.45, 710.90, 709.80, 710.20),   # 09:30
        (710.20, 710.60, 709.50, 709.90),   # 09:35
        (709.90, 710.30, 708.80, 710.10),   # 09:40  ORB-low tag
        (710.10, 710.70, 709.80, 710.50),   # 09:45
        (710.50, 710.87, 710.10, 710.30),   # 09:50
        (710.30, 710.60, 709.90, 710.10),   # 09:55
        (710.10, 710.50, 709.70, 710.25),   # 10:00
        (710.25, 710.80, 710.05, 710.65),   # 10:05
        (710.65, 710.87, 710.30, 710.40),   # 10:10
        (710.40, 710.60, 710.10, 710.30),   # 10:15
        (710.30, 710.55, 709.95, 710.15),   # 10:20
        (710.15, 710.45, 709.80, 710.00),   # 10:25
        (710.00, 710.40, 709.60, 710.20),   # 10:30
        (710.20, 710.55, 709.95, 710.35),   # 10:35
    ]
    for o, h, l, c in chop:
        bars.append(_bar(t, o, h, l, c, v=90000))
        t += timedelta(minutes=5)

    # ── Mid-morning drift 10:40–12:00 ET, still range-bound ────────────────
    drift = [
        (710.35, 710.70, 710.15, 710.55),
        (710.55, 710.85, 710.35, 710.70),
        (710.70, 710.87, 710.45, 710.60),
        (710.60, 710.80, 710.30, 710.45),
        (710.45, 710.70, 710.20, 710.40),
        (710.40, 710.60, 710.10, 710.30),
        (710.30, 710.55, 710.00, 710.25),
        (710.25, 710.50, 710.00, 710.20),
        (710.20, 710.45, 709.90, 710.10),
        (710.10, 710.35, 709.85, 710.05),
        (710.05, 710.30, 709.75, 709.90),
        (709.90, 710.15, 709.60, 709.70),
        (709.70, 709.95, 709.40, 709.55),
        (709.55, 709.80, 709.25, 709.45),
        (709.45, 709.65, 709.10, 709.25),
        (709.25, 709.45, 709.00, 709.10),
    ]
    for o, h, l, c in drift:
        bars.append(_bar(t, o, h, l, c, v=100000))
        t += timedelta(minutes=5)

    # ── Trend-down leg 12:00 CST-ish → 14:44 ET (actually 13:07–15:44 ET) ──
    # Staircase down with two VWAP-reverting pullbacks.
    trend = [
        # Initial breakdown
        (709.10, 709.20, 708.60, 708.70),     # 12:00 ET  — break ORB low
        (708.70, 708.85, 708.20, 708.35),
        (708.35, 708.60, 707.80, 708.00),
        (708.00, 708.20, 707.40, 707.60),
        # Pullback #1 toward VWAP (rejection short)
        (707.60, 708.10, 707.40, 707.95),
        (707.95, 708.25, 707.70, 707.80),      # 12:20 ET — upper wick forms
        (707.80, 707.95, 707.20, 707.30),     # rejection candle
        # Continuation leg
        (707.30, 707.45, 706.60, 706.75),
        (706.75, 706.90, 706.20, 706.35),
        (706.35, 706.55, 705.80, 705.95),
        # Pullback #2 (weaker)
        (705.95, 706.45, 705.80, 706.20),
        (706.20, 706.50, 706.00, 706.15),
        (706.15, 706.30, 705.60, 705.70),     # rejection
        # Final push down
        (705.70, 705.85, 705.10, 705.25),
        (705.25, 705.40, 704.60, 704.80),
        (704.80, 704.95, 704.10, 704.30),
        (704.30, 704.50, 703.60, 703.80),
        (703.80, 704.00, 703.20, 703.40),
        (703.40, 703.60, 702.90, 703.10),      # 13:25 ET area — bottom
    ]
    for o, h, l, c in trend:
        bars.append(_bar(t, o, h, l, c, v=140000))
        t += timedelta(minutes=5)

    # ── Afternoon bounce 14:44–15:30 ET (the retracement that scratched
    # the real afternoon put entries) ───────────────────────────────────────
    bounce = [
        (703.10, 703.80, 702.95, 703.65),
        (703.65, 704.20, 703.50, 704.05),
        (704.05, 704.60, 703.90, 704.45),
        (704.45, 705.00, 704.25, 704.85),
        (704.85, 705.30, 704.60, 705.05),
        (705.05, 705.40, 704.85, 705.20),
        (705.20, 705.50, 704.90, 705.10),
    ]
    for o, h, l, c in bounce:
        bars.append(_bar(t, o, h, l, c, v=110000))
        t += timedelta(minutes=5)

    # ── Close drift 15:30–16:00 ET ─────────────────────────────────────────
    close_drift = [
        (705.10, 705.30, 704.75, 704.90),
        (704.90, 705.15, 704.50, 704.70),
        (704.70, 704.95, 704.35, 704.50),
        (704.50, 704.80, 704.20, 704.40),
        (704.40, 704.65, 704.10, 704.25),
        (704.25, 704.50, 704.00, 704.10),
    ]
    for o, h, l, c in close_drift:
        bars.append(_bar(t, o, h, l, c, v=95000))
        t += timedelta(minutes=5)

    return bars


def apr21_legacy_signal_candidates() -> List[Dict]:
    """The nine signals that the legacy engine actually fired on Apr 21.

    Times are expressed in ET (the postmortem lists them in CST; +1h).
    """
    return [
        # 10:11 CST = 11:11 ET (carry-over ORB_BREAKOUT)
        {
            "ts": datetime(2026, 4, 21, 11, 11, tzinfo=ET),
            "signal_type": "ORB_BREAKOUT",
            "direction": "P",
            "strike": 708.0,
            "confidence": 0.78,
        },
        # 15:08 CST = 16:08 ET — after-hours in postmortem? Postmortem CST
        # numbers are a bit noisy; assume afternoon ET.
        {
            "ts": datetime(2026, 4, 21, 14, 8, tzinfo=ET),
            "signal_type": "PC_RATIO_EXTREME",
            "direction": "P",
            "strike": 708.0,
            "confidence": 0.879,
        },
        {
            "ts": datetime(2026, 4, 21, 14, 8, tzinfo=ET),
            "signal_type": "ORB_BREAKOUT",
            "direction": "P",
            "strike": 708.0,
            "confidence": 0.82,
        },
        {
            "ts": datetime(2026, 4, 21, 14, 25, tzinfo=ET),
            "signal_type": "ORB_BREAKOUT",
            "direction": "P",
            "strike": 707.0,
            "confidence": 0.82,
        },
        {
            "ts": datetime(2026, 4, 21, 15, 43, tzinfo=ET),
            "signal_type": "PC_RATIO_EXTREME",
            "direction": "P",
            "strike": 706.0,
            "confidence": 0.87,
        },
        {
            "ts": datetime(2026, 4, 21, 16, 0, tzinfo=ET),
            "signal_type": "ORB_BREAKOUT",
            "direction": "P",
            "strike": 706.0,
            "confidence": 0.78,
        },
        {
            "ts": datetime(2026, 4, 21, 16, 5, tzinfo=ET),
            "signal_type": "ORB_BREAKOUT",
            "direction": "P",
            "strike": 705.0,
            "confidence": 0.80,
        },
    ]
