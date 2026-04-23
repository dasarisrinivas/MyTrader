"""Price-structure utilities: pivots, VWAP slope, ATR ratios, RSI, crosses.

These are intentionally *independent* of the legacy indicator helpers in
``regime_detector.py``. Keeping a thin set of pure functions here lets the
rules_v2 layer be unit-tested in isolation (and swapped / mocked easily).

Contract: every function takes a list of bar dicts whose newest element is
last. Each bar is ``{"date": datetime, "open": float, "high": float, "low":
float, "close": float, "volume": int}``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from ...utils.timezone_utils import now_cst

Bar = Dict  # typed alias for clarity


# ─── Indicators ────────────────────────────────────────────────────────────


def sma(values: Sequence[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema(values: Sequence[float], period: int) -> List[float]:
    """Return the full EMA series; [] if insufficient data."""
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    out: List[float] = [sum(values[:period]) / period]
    for v in values[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def atr(bars: Sequence[Bar], period: int) -> Optional[float]:
    """Wilder ATR of the given period; None if insufficient bars."""
    if len(bars) < period + 1:
        return None
    trs: List[float] = []
    for i in range(1, len(bars)):
        h = bars[i]["high"]
        l = bars[i]["low"]
        pc = bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else None
    # seed with simple average, then Wilder smooth
    atr_val = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return atr_val


def session_vwap(bars: Sequence[Bar]) -> Optional[float]:
    """Cumulative session VWAP from the supplied bars."""
    pv = 0.0
    vol = 0
    for b in bars:
        typ = (b["high"] + b["low"] + b["close"]) / 3.0
        v = b["volume"]
        pv += typ * v
        vol += v
    return pv / vol if vol > 0 else None


def vwap_series(bars: Sequence[Bar]) -> List[float]:
    """Cumulative session VWAP value at every bar."""
    out: List[float] = []
    pv = 0.0
    vol = 0
    for b in bars:
        typ = (b["high"] + b["low"] + b["close"]) / 3.0
        v = b["volume"]
        pv += typ * v
        vol += v
        out.append(pv / vol if vol > 0 else b["close"])
    return out


def vwap_slope(bars: Sequence[Bar], lookback: int = 5) -> float:
    """Slope of VWAP over `lookback` bars, as fraction-of-price per bar.

    0.0003 ≈ 0.03%/bar. Returns 0.0 if insufficient data.
    """
    vs = vwap_series(bars)
    if len(vs) < lookback + 1:
        return 0.0
    delta = vs[-1] - vs[-lookback - 1]
    base = vs[-1] if vs[-1] > 0 else 1.0
    return (delta / lookback) / base


def rsi(values: Sequence[float], period: int = 14) -> Optional[float]:
    """Standard Wilder RSI on the given series; None if insufficient data."""
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        if change > 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ─── Pivots ────────────────────────────────────────────────────────────────


@dataclass
class Pivot:
    """Swing pivot detected on closed bars."""

    idx: int          # bar index in source list
    ts: datetime
    price: float
    kind: str         # "HIGH" or "LOW"


def detect_pivots(bars: Sequence[Bar], lookback: int = 3) -> List[Pivot]:
    """Return swing pivots with `lookback` bars on each side.

    A HIGH pivot at bar i satisfies high[i] > high[j] for every j in
    [i-lookback, i+lookback] with j != i. Likewise for LOW pivots.
    Pivots within `lookback` bars of the edges are skipped (not yet confirmed).
    """
    pivots: List[Pivot] = []
    n = len(bars)
    for i in range(lookback, n - lookback):
        hi = bars[i]["high"]
        lo = bars[i]["low"]
        window = range(i - lookback, i + lookback + 1)
        if all(hi >= bars[j]["high"] for j in window if j != i) and any(
            hi > bars[j]["high"] for j in window if j != i
        ):
            pivots.append(
                Pivot(idx=i, ts=bars[i].get("date", now_cst()), price=hi, kind="HIGH")
            )
        if all(lo <= bars[j]["low"] for j in window if j != i) and any(
            lo < bars[j]["low"] for j in window if j != i
        ):
            pivots.append(
                Pivot(idx=i, ts=bars[i].get("date", now_cst()), price=lo, kind="LOW")
            )
    return pivots


def has_higher_highs_and_lows(pivots: Sequence[Pivot], min_count: int = 2) -> bool:
    """True when the last `min_count` HIGHs rise AND last `min_count` LOWs rise."""
    highs = [p for p in pivots if p.kind == "HIGH"]
    lows = [p for p in pivots if p.kind == "LOW"]
    if len(highs) < min_count or len(lows) < min_count:
        return False
    hh = all(
        highs[-i].price > highs[-i - 1].price for i in range(1, min_count)
    )
    hl = all(
        lows[-i].price > lows[-i - 1].price for i in range(1, min_count)
    )
    return hh and hl


def has_lower_highs_and_lows(pivots: Sequence[Pivot], min_count: int = 2) -> bool:
    """True when the last `min_count` HIGHs fall AND last `min_count` LOWs fall."""
    highs = [p for p in pivots if p.kind == "HIGH"]
    lows = [p for p in pivots if p.kind == "LOW"]
    if len(highs) < min_count or len(lows) < min_count:
        return False
    lh = all(
        highs[-i].price < highs[-i - 1].price for i in range(1, min_count)
    )
    ll = all(
        lows[-i].price < lows[-i - 1].price for i in range(1, min_count)
    )
    return lh and ll


def count_vwap_crosses(bars: Sequence[Bar], lookback_bars: int) -> int:
    """Count closes on the opposite side of VWAP compared to prior close."""
    if len(bars) < lookback_bars + 1:
        return 0
    vs = vwap_series(bars)
    recent = bars[-lookback_bars - 1:]
    recent_vs = vs[-lookback_bars - 1:]
    crosses = 0
    for i in range(1, len(recent)):
        prev_side = 1 if recent[i - 1]["close"] >= recent_vs[i - 1] else -1
        cur_side = 1 if recent[i]["close"] >= recent_vs[i] else -1
        if prev_side != cur_side:
            crosses += 1
    return crosses


# ─── Rejection candles ─────────────────────────────────────────────────────


def is_bullish_rejection(bar: Bar, wick_ratio: float) -> bool:
    """Large lower wick + close in upper half of range."""
    hi, lo, op, cl = bar["high"], bar["low"], bar["open"], bar["close"]
    rng = hi - lo
    if rng <= 0:
        return False
    body_lo = min(op, cl)
    lower_wick = body_lo - lo
    return lower_wick / rng >= wick_ratio and cl >= (lo + hi) / 2


def is_bearish_rejection(bar: Bar, wick_ratio: float) -> bool:
    """Large upper wick + close in lower half of range."""
    hi, lo, op, cl = bar["high"], bar["low"], bar["open"], bar["close"]
    rng = hi - lo
    if rng <= 0:
        return False
    body_hi = max(op, cl)
    upper_wick = hi - body_hi
    return upper_wick / rng >= wick_ratio and cl <= (lo + hi) / 2


def is_bullish_engulfing(prev: Bar, cur: Bar) -> bool:
    return (
        cur["close"] > cur["open"]
        and prev["close"] < prev["open"]
        and cur["close"] >= prev["open"]
        and cur["open"] <= prev["close"]
    )


def is_bearish_engulfing(prev: Bar, cur: Bar) -> bool:
    return (
        cur["close"] < cur["open"]
        and prev["close"] > prev["open"]
        and cur["close"] <= prev["open"]
        and cur["open"] >= prev["close"]
    )


# ─── Expansion / parabolic detection ───────────────────────────────────────


def expansion_bar(bar: Bar, atr_val: float, multiple: float) -> bool:
    """True if the bar's range is at least `multiple` × ATR."""
    if atr_val <= 0:
        return False
    return (bar["high"] - bar["low"]) >= multiple * atr_val


def consecutive_expansion_bars(
    bars: Sequence[Bar], atr_val: float, multiple: float
) -> int:
    """Count trailing consecutive expansion bars in the same direction."""
    if atr_val <= 0 or not bars:
        return 0
    count = 0
    for bar in reversed(bars):
        if expansion_bar(bar, atr_val, multiple):
            count += 1
        else:
            break
    return count
