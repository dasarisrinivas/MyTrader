"""SPY market regime classifier.

Classifies the current market into one of six regimes using 5-minute
OHLCV bars fetched from IB Gateway. Pure computation — no IB calls here.

Regimes:
    NEWS_DRIVEN   — ATR blow-up vs recent average (earnings, FOMC, CPI)
    HIGH_VOL      — VIX elevated or ATR expanded
    LOW_VOL       — VIX suppressed and ATR compressed
    TREND_UP      — EMA9 > EMA21, positive slope, price above VWAP
    TREND_DOWN    — EMA9 < EMA21, negative slope, price below VWAP
    RANGE_BOUND   — default when no other regime is confirmed
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class RegimeContext:
    """Output of RegimeDetector.classify()."""

    regime: str           # see module docstring for valid values
    ema9: float           # last-bar EMA-9 of close
    ema21: float          # last-bar EMA-21 of close
    atr14: float          # last-bar ATR-14
    vwap: float           # session VWAP at last bar (reset each day)
    spy_vs_vwap: float    # spy_price - vwap (positive = above VWAP)
    ema_slope: float      # (ema9[-1] - ema9[-5]) / 5 — directional momentum
    timestamp: datetime


def _ema(values: List[float], period: int) -> List[float]:
    """Compute EMA series using standard multiplier 2/(period+1)."""
    if not values or len(values) < period:
        return [values[-1]] if values else [0.0]
    k = 2.0 / (period + 1)
    result = [sum(values[:period]) / period]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _atr14(bars: List[Dict]) -> List[float]:
    """Compute ATR-14 series (Wilder smoothing)."""
    if len(bars) < 2:
        return [0.0]

    trs = []
    for i in range(1, len(bars)):
        h = bars[i]["high"]
        l = bars[i]["low"]
        pc = bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))

    if not trs:
        return [0.0]

    period = 14
    if len(trs) < period:
        return [sum(trs) / len(trs)] * len(trs)

    # Wilder smoothing (same as standard ATR)
    result = [sum(trs[:period]) / period]
    for tr in trs[period:]:
        result.append((result[-1] * (period - 1) + tr) / period)
    return result


def _vwap(bars: List[Dict]) -> float:
    """Compute session VWAP from a list of bars."""
    total_pv = 0.0
    total_vol = 0
    for b in bars:
        typical = (b["high"] + b["low"] + b["close"]) / 3.0
        vol = b["volume"]
        total_pv += typical * vol
        total_vol += vol
    return total_pv / total_vol if total_vol > 0 else 0.0


class RegimeDetector:
    """Stateless SPY regime classifier.

    Call classify() once per poll — it is pure computation and holds no state.
    """

    def classify(
        self,
        bars: List[Dict],
        spy_price: float,
        vix: Optional[float],
        vix_high: float = 26.0,
        vix_low: float = 12.0,
    ) -> RegimeContext:
        """Classify current market regime from 5-minute bars.

        Args:
            bars: List of dicts with keys date/open/high/low/close/volume.
                  Newest bar is last. Must contain at least 22 bars.
            spy_price: Current SPY last price.
            vix: Current VIX level (None if unavailable).
            vix_high: VIX level above which HIGH_VOL triggers.
            vix_low: VIX level below which LOW_VOL triggers.

        Returns:
            RegimeContext with regime classification and indicator values.
        """
        _FALLBACK = RegimeContext(
            regime="RANGE_BOUND",
            ema9=spy_price, ema21=spy_price, atr14=0.0,
            vwap=spy_price, spy_vs_vwap=0.0, ema_slope=0.0,
            timestamp=datetime.utcnow(),
        )

        if len(bars) < 22:
            return _FALLBACK

        closes  = [b["close"] for b in bars]
        ema9s   = _ema(closes, 9)
        ema21s  = _ema(closes, 21)
        atr14s  = _atr14(bars)
        vwap    = _vwap(bars)

        ema9_last  = ema9s[-1]
        ema21_last = ema21s[-1]
        atr_last   = atr14s[-1]

        # EMA slope: change over last 5 bars of the EMA-9 series
        if len(ema9s) >= 5:
            ema_slope = (ema9s[-1] - ema9s[-5]) / 5.0
        else:
            ema_slope = 0.0

        spy_vs_vwap = spy_price - vwap

        # 20-bar median ATR for regime comparison (exclude last bar)
        lookback_atrs = atr14s[max(0, len(atr14s) - 21):-1]
        median_atr = sorted(lookback_atrs)[len(lookback_atrs) // 2] if lookback_atrs else atr_last

        # ── Priority 1: NEWS_DRIVEN ───────────────────────────────────────────
        if median_atr > 0 and atr_last > median_atr * 2.5:
            regime = "NEWS_DRIVEN"

        # ── Priority 2: HIGH_VOL ─────────────────────────────────────────────
        elif (vix is not None and vix > vix_high) or (median_atr > 0 and atr_last > median_atr * 2.0):
            regime = "HIGH_VOL"

        # ── Priority 3: LOW_VOL ──────────────────────────────────────────────
        elif (vix is not None and vix < vix_low) and (median_atr > 0 and atr_last < median_atr * 0.5):
            regime = "LOW_VOL"

        # ── Priority 4: TREND_UP ─────────────────────────────────────────────
        elif ema9_last > ema21_last and ema_slope > 0 and spy_vs_vwap > 0:
            regime = "TREND_UP"

        # ── Priority 5: TREND_DOWN ───────────────────────────────────────────
        elif ema9_last < ema21_last and ema_slope < 0 and spy_vs_vwap < 0:
            regime = "TREND_DOWN"

        # ── Default: RANGE_BOUND ─────────────────────────────────────────────
        else:
            regime = "RANGE_BOUND"

        return RegimeContext(
            regime=regime,
            ema9=ema9_last,
            ema21=ema21_last,
            atr14=atr_last,
            vwap=vwap,
            spy_vs_vwap=spy_vs_vwap,
            ema_slope=ema_slope,
            timestamp=datetime.utcnow(),
        )
