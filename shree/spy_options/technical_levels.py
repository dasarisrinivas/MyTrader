"""SPY Options — intraday technical levels tracker.

Computes session-scoped indicators that every veteran SPY options day trader
watches:

  1. Opening Range Breakout (ORB)       — first-30-min range; breakout/breakdown bias
  2. VWAP standard-deviation bands      — ±1σ / ±2σ volume-weighted extension zones
  3. Daily pivot points (Floor-Trader)  — PP, R1, R2, S1, S2 from prior session H/L/C
  4. Expected Daily Range exhaustion    — VIX-implied 1-σ daily range; how much is used
  5. RSI (5-min) with divergence        — overbought/oversold + price vs RSI direction

Additionally exports:
  compute_max_pain()  — strike that minimises total option value paid to buyers
                        (market-maker pinning target; critical for 0DTE)

State is scoped to the trading session.  Call reset() at each daily open.

Design notes:
  - All computation is pure Python + math; zero external network calls.
  - The tracker is injected into manager._poll() alongside the regime detector.
  - Results are merged into ExternalContext so DynamicConfidence can read them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TechnicalLevels:
    """Point-in-time snapshot of intraday technical levels for SPY options."""

    # ── Opening Range Breakout ────────────────────────────────────────────────
    # The 30-min ORB is the single most-watched intraday reference for SPY.
    # Institutional desks and systematic traders all use it as the first
    # directional filter after the open.
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_established: bool = False       # True once the 10:00 ET build window closes
    orb_width_pct: float = 0.0          # range width as % of SPY price
    orb_status: str = "BUILDING"        # BUILDING | INSIDE | ABOVE_ORB | BELOW_ORB
    orb_breakout_confirmed: bool = False  # closed outside ORB for ≥1 bar

    # ── VWAP standard-deviation bands ────────────────────────────────────────
    # Classic mean-reversion / extension zones.
    # +2σ and −2σ are the preferred fade levels; rare to see SPY spend time there.
    vwap_1sd_upper: Optional[float] = None
    vwap_1sd_lower: Optional[float] = None
    vwap_2sd_upper: Optional[float] = None
    vwap_2sd_lower: Optional[float] = None
    # Which band is the current price in?
    vwap_band_position: str = "INSIDE_1SD"
    # ABOVE_2SD | ABOVE_1SD | INSIDE_1SD | BELOW_1SD | BELOW_2SD

    # ── Daily pivot points (Floor-Trader formula) ─────────────────────────────
    # Computed from the prior session's high, low, and close.
    # SPY exhibits measurable mean-reversion at PP, R1, R2, S1, S2 intraday.
    pivot_pp: Optional[float] = None
    pivot_r1: Optional[float] = None
    pivot_r2: Optional[float] = None
    pivot_s1: Optional[float] = None
    pivot_s2: Optional[float] = None
    near_pivot: bool = False
    pivot_nearest: str = ""             # "PP" | "R1" | "R2" | "S1" | "S2"
    pivot_bias: str = "NEUTRAL"         # AT_RESISTANCE | AT_SUPPORT | AT_PIVOT | NEUTRAL

    # ── Expected Daily Range exhaustion ──────────────────────────────────────
    # VIX implies a 1-σ expected daily range for SPY.
    # When 85%+ of that range is consumed, fade signals become more likely than
    # continuation signals — especially relevant for 0DTE afternoon entries.
    edr_points: float = 0.0             # VIX / √252 / 100 × spy_price (in SPY $)
    edr_used_pct: float = 0.0           # how much of the expected range is already consumed
    edr_exhausted: bool = False         # True when edr_used_pct ≥ 85%

    # ── RSI (5-min) ───────────────────────────────────────────────────────────
    # Simple overbought/oversold flag plus price-vs-RSI divergence detection.
    # Divergence is a reliable early reversal warning at VWAP band extremes.
    rsi_5m: float = 50.0
    rsi_overbought: bool = False        # RSI ≥ 70
    rsi_oversold: bool = False          # RSI ≤ 30
    rsi_divergence: str = "NONE"        # BULLISH_DIV | BEARISH_DIV | NONE


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_max_pain(
    strikes: List[float],
    call_oi: Dict[float, int],
    put_oi: Dict[float, int],
) -> Optional[float]:
    """Return the strike where total option-buyer losses are maximised.

    For each candidate closing price K, total option premium at risk:
        call_pain(K) = Σ_S>K  call_OI[S] × (S − K)
        put_pain(K)  = Σ_S<K  put_OI[S]  × (K − S)

    The strike that minimises this sum is where market makers profit most
    — the 'max pain' (or 'pin risk') target.  Especially relevant for 0DTE
    after noon ET when pin pressure strengthens.

    Args:
        strikes:   All candidate strikes from the options chain.
        call_oi:   Dict of strike → call open interest.
        put_oi:    Dict of strike → put open interest.

    Returns:
        Max-pain strike, or None if inputs are insufficient.
    """
    if not strikes or not (call_oi or put_oi):
        return None

    min_pain: float = float("inf")
    result: Optional[float] = None

    for k in strikes:
        call_pain = sum(
            max(s - k, 0.0) * oi
            for s, oi in call_oi.items()
        )
        put_pain = sum(
            max(k - s, 0.0) * oi
            for s, oi in put_oi.items()
        )
        total = call_pain + put_pain
        if total < min_pain:
            min_pain = total
            result = k

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal math helpers (module-private)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi_series(closes: List[float], period: int = 14) -> List[float]:
    """Wilder-smoothed RSI series of the same length as *closes*."""
    n = len(closes)
    if n < period + 1:
        return [50.0] * n

    result: List[float] = [50.0] * period
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for i in range(period, n):
        diff = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(diff, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-diff, 0.0)) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        result.append(100.0 - 100.0 / (1.0 + rs))
    return result


def _detect_divergence(
    closes: List[float],
    rsi: List[float],
    lookback: int = 8,
) -> str:
    """Compare price direction with RSI direction over the last *lookback* bars.

    Bearish divergence: price made a higher high but RSI did not follow.
      — Only flagged when RSI is still elevated (> 55) suggesting tired momentum.
    Bullish divergence: price made a lower low but RSI is recovering.
      — Only flagged when RSI is still depressed (< 45) suggesting selling exhaustion.

    A stricter filter (requiring both endpoints above/below thresholds) reduces
    false positives in choppy RANGE_BOUND conditions.
    """
    if len(closes) < lookback + 1 or len(rsi) < lookback + 1:
        return "NONE"

    price_rising = closes[-1] > closes[-lookback]
    rsi_rising   = rsi[-1] > rsi[-lookback]
    rsi_now      = rsi[-1]

    # Bearish: price up, RSI flat/down, RSI was elevated both then and now
    if price_rising and not rsi_rising and rsi_now > 55 and rsi[-lookback] > 60:
        return "BEARISH_DIV"

    # Bullish: price down, RSI flat/up, RSI was depressed both then and now
    if not price_rising and rsi_rising and rsi_now < 45 and rsi[-lookback] < 40:
        return "BULLISH_DIV"

    return "NONE"


def _vwap_bands(
    bars: List[Dict],
) -> Tuple[float, float, float, float, float]:
    """Compute session VWAP and volume-weighted ±1σ / ±2σ bands.

    Uses the volume-weighted variance formula:
        Var = E[x²] − E[x]²   where x is the typical price and E is volume-weighted.

    Returns:
        (vwap, upper_1sd, lower_1sd, upper_2sd, lower_2sd)
        Returns all zeros when fewer than 5 bars are available.
    """
    if len(bars) < 5:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    cum_pv   = 0.0
    cum_pv2  = 0.0
    cum_vol  = 0.0

    for b in bars:
        typical = (b["high"] + b["low"] + b["close"]) / 3.0
        vol = max(float(b.get("volume", 1)), 1.0)
        cum_pv   += typical * vol
        cum_pv2  += typical * typical * vol
        cum_vol  += vol

    if cum_vol <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    vwap     = cum_pv / cum_vol
    variance = (cum_pv2 / cum_vol) - vwap * vwap
    sd       = math.sqrt(max(variance, 0.0))

    return (
        vwap,
        vwap + sd,
        vwap - sd,
        vwap + 2.0 * sd,
        vwap - 2.0 * sd,
    )


def _floor_pivots(
    prev_high: float,
    prev_low: float,
    prev_close: float,
) -> Tuple[float, float, float, float, float]:
    """Classic Floor-Trader pivot formula.

    Returns:
        (PP, R1, R2, S1, S2)
    """
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2.0 * pp - prev_low
    r2 = pp + (prev_high - prev_low)
    s1 = 2.0 * pp - prev_high
    s2 = pp - (prev_high - prev_low)
    return pp, r1, r2, s1, s2


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class TechnicalLevelsTracker:
    """Session-scoped tracker for intraday SPY options technical levels.

    Usage in manager::

        tracker = TechnicalLevelsTracker()
        ...
        # In _daily_reset_if_needed():
        tracker.reset()

        # In _poll():
        tech = tracker.update(bars_5m, spy_price, vix)
        # Merge tech into ext_ctx fields before passing to SignalContext.

    The tracker holds minimal session state:
      - ORB high/low (accumulated during the 9:30–10:00 ET build window)
      - Previous-day H/L/C (derived from prior-day bars the first time they appear)
    """

    # Build window: first 30 minutes of RTH
    _ORB_CLOSE_TIME = time(10, 0)

    def __init__(self) -> None:
        self._orb_high: Optional[float] = None
        self._orb_low:  Optional[float] = None
        self._orb_established: bool = False
        self._orb_confirmed_bars: int = 0

        # Previous session data — persists across the day boundary
        self._prev_high:  Optional[float] = None
        self._prev_low:   Optional[float] = None
        self._prev_close: Optional[float] = None

    def reset(self) -> None:
        """Reset intraday state.  Call once per trading day (at daily open).

        Preserves previous-day H/L/C so pivot points are available immediately
        on the new day without waiting for prior-day bars to re-appear.
        """
        self._orb_high = None
        self._orb_low  = None
        self._orb_established = False
        self._orb_confirmed_bars = 0
        # _prev_* intentionally NOT reset — carry forward for pivots

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        bars: List[Dict],
        spy_price: float,
        vix: Optional[float],
    ) -> TechnicalLevels:
        """Compute and return a fresh TechnicalLevels snapshot.

        Args:
            bars:       5-min OHLCV bars (newest last).  Each bar must be a dict
                        with keys: date (datetime/str), open, high, low, close, volume.
            spy_price:  Current SPY last price.
            vix:        Current VIX level (used for EDR calculation).

        Returns:
            Populated TechnicalLevels.  Fields whose data is insufficient will
            hold their zero/None defaults — callers should tolerate this.
        """
        levels = TechnicalLevels()
        if not bars or spy_price <= 0:
            return levels

        today_bars, prev_bars = self._split_days(bars)

        # Update previous-day anchor whenever we see prior-session bars
        if prev_bars:
            self._prev_high  = max(b["high"]  for b in prev_bars)
            self._prev_low   = min(b["low"]   for b in prev_bars)
            self._prev_close = prev_bars[-1]["close"]

        # Fallback: if IB only returned today's bars, use all bars as "today"
        if not today_bars:
            today_bars = bars

        # ── ORB ───────────────────────────────────────────────────────────────
        self._update_orb(today_bars)
        levels.orb_high = self._orb_high
        levels.orb_low  = self._orb_low
        levels.orb_established       = self._orb_established
        levels.orb_breakout_confirmed = self._orb_confirmed_bars >= 1
        if self._orb_high and self._orb_low:
            levels.orb_width_pct = (self._orb_high - self._orb_low) / spy_price * 100.0
        levels.orb_status = self._orb_position(spy_price)

        # ── VWAP bands ────────────────────────────────────────────────────────
        vwap, u1, l1, u2, l2 = _vwap_bands(today_bars)
        if vwap > 0:
            levels.vwap_1sd_upper = round(u1, 2)
            levels.vwap_1sd_lower = round(l1, 2)
            levels.vwap_2sd_upper = round(u2, 2)
            levels.vwap_2sd_lower = round(l2, 2)
            if spy_price >= u2:
                levels.vwap_band_position = "ABOVE_2SD"
            elif spy_price >= u1:
                levels.vwap_band_position = "ABOVE_1SD"
            elif spy_price <= l2:
                levels.vwap_band_position = "BELOW_2SD"
            elif spy_price <= l1:
                levels.vwap_band_position = "BELOW_1SD"
            else:
                levels.vwap_band_position = "INSIDE_1SD"

        # ── Pivot points ──────────────────────────────────────────────────────
        if self._prev_high and self._prev_low and self._prev_close:
            pp, r1, r2, s1, s2 = _floor_pivots(
                self._prev_high, self._prev_low, self._prev_close
            )
            levels.pivot_pp = round(pp, 2)
            levels.pivot_r1 = round(r1, 2)
            levels.pivot_r2 = round(r2, 2)
            levels.pivot_s1 = round(s1, 2)
            levels.pivot_s2 = round(s2, 2)

            # Identify which pivot the current price is nearest and whether
            # it is acting as resistance or support
            named = {"PP": pp, "R1": r1, "R2": r2, "S1": s1, "S2": s2}
            nearest = min(named, key=lambda k: abs(named[k] - spy_price))
            dist = abs(named[nearest] - spy_price)
            threshold = max(spy_price * 0.003, 1.50)  # 0.3% or $1.50, whichever is larger
            if dist <= threshold:
                levels.near_pivot = True
                levels.pivot_nearest = nearest
                levels.pivot_bias = (
                    "AT_RESISTANCE" if nearest in ("R1", "R2")
                    else "AT_SUPPORT"  if nearest in ("S1", "S2")
                    else "AT_PIVOT"
                )

        # ── Expected Daily Range exhaustion ───────────────────────────────────
        if vix and vix > 0:
            # 1-σ expected intraday move: VIX / √252 / 100 × price
            edr = (vix / math.sqrt(252)) / 100.0 * spy_price
            levels.edr_points = round(edr, 2)
            if edr > 0 and today_bars:
                day_open  = today_bars[0]["open"]
                day_high  = max(b["high"] for b in today_bars)
                day_low   = min(b["low"]  for b in today_bars)
                consumed  = max(day_high - day_open, day_open - day_low, 0.0)
                levels.edr_used_pct  = round(min(consumed / edr * 100.0, 200.0), 1)
                levels.edr_exhausted = levels.edr_used_pct >= 85.0

        # ── RSI (5-min) ───────────────────────────────────────────────────────
        closes = [b["close"] for b in today_bars]
        if len(closes) >= 15:
            rsi_vals = _rsi_series(closes, period=14)
            levels.rsi_5m         = round(rsi_vals[-1], 1)
            levels.rsi_overbought = rsi_vals[-1] >= 70.0
            levels.rsi_oversold   = rsi_vals[-1] <= 30.0
            levels.rsi_divergence = _detect_divergence(closes, rsi_vals, lookback=8)

        return levels

    # ── Internals ─────────────────────────────────────────────────────────────

    def _split_days(
        self, bars: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Partition bars into today vs previous day(s) using the 'date' field."""
        if not bars:
            return [], []

        def _to_date(b: Dict):
            d = b.get("date")
            if isinstance(d, datetime):
                return d.astimezone(ET).date() if d.tzinfo else d.date()
            if isinstance(d, str):
                try:
                    return datetime.fromisoformat(d).date()
                except ValueError:
                    pass
            return None   # unrecognised format

        try:
            dates = [_to_date(b) for b in bars]
            valid_dates = [d for d in dates if d is not None]
            if not valid_dates:
                return bars, []
            latest = max(valid_dates)
            today = [b for b, d in zip(bars, dates) if d == latest]
            prev  = [b for b, d in zip(bars, dates) if d is not None and d < latest]
            return today, prev
        except Exception:
            return bars, []

    def _update_orb(self, today_bars: List[Dict]) -> None:
        """Update ORB high/low from bars that fall within the 9:30–10:00 ET window."""
        orb_bars: List[Dict] = []
        rth_open  = time(9, 30)
        orb_close = self._ORB_CLOSE_TIME

        for b in today_bars:
            bar_time: Optional[time] = None
            d = b.get("date")
            if isinstance(d, datetime):
                dt_et = d.astimezone(ET) if d.tzinfo else d
                bar_time = dt_et.time()
            if bar_time is None:
                continue
            if rth_open <= bar_time < orb_close:
                orb_bars.append(b)

        if orb_bars:
            self._orb_high = max(b["high"] for b in orb_bars)
            self._orb_low  = min(b["low"]  for b in orb_bars)

        # ORB is "established" after the build window has passed for the day
        now_et = datetime.now(ET).time()
        if now_et >= orb_close and self._orb_high is not None:
            self._orb_established = True

        # Count consecutive recent bars that closed outside the ORB
        if self._orb_established and self._orb_high and self._orb_low:
            self._orb_confirmed_bars = 0
            for b in reversed(today_bars[-3:]):
                c = b["close"]
                if c > self._orb_high or c < self._orb_low:
                    self._orb_confirmed_bars += 1
                else:
                    break  # chain broken — reset counter

    def _orb_position(self, spy_price: float) -> str:
        """Return price position relative to the Opening Range."""
        if not self._orb_established or self._orb_high is None:
            return "BUILDING"
        if spy_price > self._orb_high:
            return "ABOVE_ORB"
        if spy_price < self._orb_low:
            return "BELOW_ORB"
        return "INSIDE"
