"""
DynamicConfidence — time-of-day, DTE, and regime-aware confidence adjuster.

The base confidence from the signal engine (0.0–1.0) is adjusted by a set of
additive modifiers that reflect the current market environment.

Adjustment tiers:
  Weak alignment     ±0.03  (3%)
  Moderate alignment ±0.05  (5%)
  Strong alignment   ±0.10  (10%)
  Exceptional        ±0.15  (15%)
  Severe penalty     -0.10 to -0.20  (event risk, conflicting signals)

Time-of-day buckets (ET):
  OPEN       09:30–10:30   Breakout/momentum bias; flow and regime matter most
  MIDDAY     10:30–14:00   Chop zone; reduce confidence for all signals
  PRE_POWER  14:00–15:00   Building momentum; increasing reliability
  POWER_HOUR 15:00–16:00   Directional moves; strong flow alignment required for 0DTE
  CLOSE      16:00+        RTH over

DTE rules:
  0DTE:  require strong confirmation (flow ≥ ±25, regime aligned)
         reduce holding time signal; strong stop-loss recommendation
  1DTE:  standard rules
  2-7DTE: wider tolerance; flow alignment matters more; macro signal weighted higher
  8+DTE: standard longer-dated rules; event proximity matters more

Macro environment:
  DXY↑ + TNX↑ + VIX↑ → bearish headwind; penalise call signals
  DXY↓ + TNX↓ + VIX↓ → bullish tailwind; boost call signals
  Strong breadth (VIX falling, advance/decline positive) → boost directional

Conflict detection:
  If flow_score and IB_sentiment are opposite signs (>±20 each)
  AND macro headwind opposes signal direction:
    → reduce confidence by 0.08 ("conflicting signals")
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

from .external.composite import ExternalContext

ET = ZoneInfo("America/New_York")


# ── Time-of-day buckets ────────────────────────────────────────────────────────

class _TOD:
    OPEN        = "OPEN"        # 09:30–10:30
    MIDDAY      = "MIDDAY"      # 10:30–14:00
    PRE_POWER   = "PRE_POWER"   # 14:00–15:00
    POWER_HOUR  = "POWER_HOUR"  # 15:00–16:00
    CLOSE       = "CLOSE"       # ≥16:00


def _tod_bucket(now_et: Optional[datetime] = None) -> str:
    if now_et is None:
        now_et = datetime.now(ET)
    t = now_et.time()
    if time(9, 30) <= t < time(10, 30):
        return _TOD.OPEN
    if time(10, 30) <= t < time(14, 0):
        return _TOD.MIDDAY
    if time(14, 0) <= t < time(15, 0):
        return _TOD.PRE_POWER
    if time(15, 0) <= t < time(16, 0):
        return _TOD.POWER_HOUR
    return _TOD.CLOSE


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ConfidenceAdjustment:
    final: float            # adjusted confidence, clamped 0.0–1.0
    delta: float            # net adjustment applied
    time_bucket: str
    dte_rule: str
    macro_factor: str
    flow_factor: str
    conflict_detected: bool
    breakdown: dict         # component → delta for logging


# ── Engine ────────────────────────────────────────────────────────────────────

class DynamicConfidence:
    """
    Adjusts a base confidence score using contextual factors.

    Usage:
        engine = DynamicConfidence(cfg)
        adj = engine.adjust(base=0.74, right="C", dte=0, ext_ctx=ctx)
        final_confidence = adj.final
    """

    def __init__(
        self,
        # Time-of-day modifiers
        open_multiplier: float = 1.10,          # opening hour boosts signal reliability
        midday_penalty: float = -0.05,          # chop zone
        power_hour_multiplier: float = 1.05,    # directional moves solidify
        # DTE thresholds
        dte_0_min_flow_score: float = 25.0,     # 0DTE requires flow ≥ ±25
        dte_0_confidence_boost: float = 0.00,   # no extra boost just from 0DTE
        dte_0_unconfirmed_penalty: float = -0.08,  # penalty if flow doesn't confirm
        # Macro thresholds
        macro_strong_headwind_threshold: float = -0.40,
        macro_strong_tailwind_threshold: float = 0.30,
        # Flow thresholds
        flow_strong: float = 40.0,
        flow_moderate: float = 20.0,
        flow_weak: float = 10.0,
        # Conflict threshold
        conflict_flow_threshold: float = 20.0,
        conflict_sentiment_threshold: float = 20.0,
    ):
        self._open_mult = open_multiplier
        self._midday_pen = midday_penalty
        self._power_mult = power_hour_multiplier
        self._dte0_min_flow = dte_0_min_flow_score
        self._dte0_unconf_pen = dte_0_unconfirmed_penalty
        self._macro_headwind_thr = macro_strong_headwind_threshold
        self._macro_tailwind_thr = macro_strong_tailwind_threshold
        self._flow_strong = flow_strong
        self._flow_moderate = flow_moderate
        self._flow_weak = flow_weak
        self._conflict_flow_thr = conflict_flow_threshold
        self._conflict_sent_thr = conflict_sentiment_threshold

    def adjust(
        self,
        base: float,
        right: str,                    # "C" | "P" | "BOTH"
        dte: int,
        ext_ctx: Optional[ExternalContext],
        ib_sentiment_score: float = 0.0,  # IB sentiment -100 to +100
        vix: Optional[float] = None,
        now_et: Optional[datetime] = None,
    ) -> ConfidenceAdjustment:
        """
        Args:
            base:               Signal engine confidence (0.0–1.0)
            right:              "C" (call) | "P" (put) | "BOTH"
            dte:                Days to expiry of the option
            ext_ctx:            ExternalContext from composite aggregator
            ib_sentiment_score: IB-based sentiment (-100 to +100)
            vix:                Current VIX level
            now_et:             Override for testing; defaults to datetime.now(ET)

        Returns:
            ConfidenceAdjustment with final confidence and breakdown.
        """
        breakdown: dict = {}
        total_delta = 0.0
        now = now_et or datetime.now(ET)
        tod = _tod_bucket(now)

        flow_score   = ext_ctx.composite_score * 100.0 if ext_ctx else 0.0  # -100..+100
        macro_hw     = ext_ctx.macro_headwind if ext_ctx else 0.0            # -1..+1
        event_risk   = ext_ctx.event_risk if ext_ctx else False
        event_mins   = ext_ctx.event_minutes if ext_ctx else 999.0

        # ── 1. Time-of-day ─────────────────────────────────────────────────
        tod_delta = 0.0
        if tod == _TOD.MIDDAY:
            tod_delta = self._midday_pen          # -0.05 chop zone
        elif tod == _TOD.OPEN:
            tod_delta = (base * self._open_mult) - base   # 10% multiplicative boost
        elif tod == _TOD.POWER_HOUR:
            tod_delta = (base * self._power_mult) - base  # 5% boost
        total_delta += tod_delta
        breakdown["time_of_day"] = round(tod_delta, 4)

        # ── 2. DTE rules ────────────────────────────────────────────────────
        dte_delta = 0.0
        dte_rule = "STANDARD"
        if dte == 0:
            dte_rule = "0DTE"
            # 0DTE requires strong external flow confirmation
            directional_flow = abs(flow_score)
            if directional_flow < self._dte0_min_flow:
                dte_delta = self._dte0_unconf_pen   # -0.08 unconfirmed 0DTE
                dte_rule = "0DTE_UNCONFIRMED"
            else:
                dte_delta = 0.02  # small boost for confirmed 0DTE
                dte_rule = "0DTE_CONFIRMED"
        elif dte <= 7:
            dte_rule = "SWING"
            # Swing: give small positive adjustment for confirmed flow
            if abs(flow_score) >= self._flow_moderate:
                dte_delta = 0.02
        total_delta += dte_delta
        breakdown["dte"] = round(dte_delta, 4)

        # ── 3. External flow alignment ──────────────────────────────────────
        flow_delta = 0.0
        flow_factor = "NEUTRAL"
        if ext_ctx is not None and ext_ctx.sources_available >= 1:
            # Directional alignment: is flow pointing the same way as the signal?
            if right == "C":
                aligned = flow_score > 0
                flow_abs = flow_score
            elif right == "P":
                aligned = flow_score < 0
                flow_abs = -flow_score
            else:
                aligned = True   # BOTH (straddle)
                flow_abs = abs(flow_score)

            if aligned:
                if flow_abs >= self._flow_strong:
                    flow_delta = 0.10
                    flow_factor = "STRONG_ALIGNMENT"
                elif flow_abs >= self._flow_moderate:
                    flow_delta = 0.05
                    flow_factor = "MODERATE_ALIGNMENT"
                elif flow_abs >= self._flow_weak:
                    flow_delta = 0.03
                    flow_factor = "WEAK_ALIGNMENT"
            else:
                # Flow conflicts with signal direction
                if flow_abs >= self._flow_strong:
                    flow_delta = -0.10
                    flow_factor = "STRONG_CONFLICT"
                elif flow_abs >= self._flow_moderate:
                    flow_delta = -0.05
                    flow_factor = "MODERATE_CONFLICT"
                elif flow_abs >= self._flow_weak:
                    flow_delta = -0.03
                    flow_factor = "WEAK_CONFLICT"

        total_delta += flow_delta
        breakdown["flow_alignment"] = round(flow_delta, 4)

        # ── 4. Macro environment ─────────────────────────────────────────────
        macro_delta = 0.0
        macro_factor = "NEUTRAL"
        if macro_hw <= self._macro_headwind_thr:  # strong headwind (< -0.40)
            if right == "C":
                macro_delta = -0.08
                macro_factor = "MACRO_HEADWIND_BEARISH"
            elif right == "P":
                macro_delta = 0.05
                macro_factor = "MACRO_HEADWIND_BULLISH_FOR_PUTS"
        elif macro_hw >= self._macro_tailwind_thr:  # tailwind (> 0.30)
            if right == "C":
                macro_delta = 0.05
                macro_factor = "MACRO_TAILWIND_BULLISH"
            elif right == "P":
                macro_delta = -0.05
                macro_factor = "MACRO_TAILWIND_BEARISH_FOR_PUTS"

        # VIX regime
        if vix is not None:
            if vix > 30 and right == "C":
                macro_delta -= 0.05   # high VIX is dangerous for long calls
                macro_factor += "+HIGH_VIX"
            elif vix < 14 and right == "P":
                macro_delta -= 0.03   # low VIX, puts are expensive for no reason
                macro_factor += "+LOW_VIX"

        total_delta += macro_delta
        breakdown["macro"] = round(macro_delta, 4)

        # ── 5. Event risk ────────────────────────────────────────────────────
        event_delta = 0.0
        if event_risk and event_mins < 30:
            # Severe penalty handled by signal_engine already (-30%)
            # Add small additional penalty here for directional signals
            if right in ("C", "P"):
                event_delta = -0.05
            breakdown["event_risk"] = round(event_delta, 4)
        total_delta += event_delta

        # ── 6. Conflict detection ─────────────────────────────────────────────
        conflict = False
        conflict_delta = 0.0
        if ext_ctx is not None:
            flow_directional = flow_score  # positive = bullish
            sent_directional = ib_sentiment_score  # -100..+100
            macro_directional = macro_hw * 100  # -100..+100

            # Call signal: all 3 should be positive; penalise if majority oppose
            if right == "C":
                opposing = sum([
                    flow_directional < -self._conflict_flow_thr,
                    sent_directional < -self._conflict_sent_thr,
                    macro_directional < -20,
                ])
            elif right == "P":
                opposing = sum([
                    flow_directional > self._conflict_flow_thr,
                    sent_directional > self._conflict_sent_thr,
                    macro_directional > 20,
                ])
            else:
                opposing = 0

            if opposing >= 2:
                conflict = True
                conflict_delta = -0.08
                total_delta += conflict_delta
                breakdown["conflict"] = round(conflict_delta, 4)

        # ── Finalise ─────────────────────────────────────────────────────────
        final = max(0.0, min(1.0, base + total_delta))
        breakdown["base"] = round(base, 4)
        breakdown["total_delta"] = round(total_delta, 4)

        return ConfidenceAdjustment(
            final=round(final, 4),
            delta=round(total_delta, 4),
            time_bucket=tod,
            dte_rule=dte_rule,
            macro_factor=macro_factor,
            flow_factor=flow_factor,
            conflict_detected=conflict,
            breakdown=breakdown,
        )
