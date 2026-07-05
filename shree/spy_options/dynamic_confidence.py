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

from shree.utils.logger import logger
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
        regime: str = "RANGE_BOUND",      # current market regime
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

        flow_score   = ext_ctx.flow_score if ext_ctx else 0.0               # -100..+100
        macro_hw     = ext_ctx.macro_headwind if ext_ctx else 0.0            # -1..+1
        event_risk   = ext_ctx.event_risk if ext_ctx else False
        event_mins   = ext_ctx.event_minutes if ext_ctx else 999.0

        # ── 1. Time-of-day ─────────────────────────────────────────────────
        tod_delta = 0.0
        if tod == _TOD.MIDDAY:
            tod_delta = self._midday_pen          # -0.05 chop zone
        elif tod == _TOD.OPEN:
            # Before 10:15 ET early-session options flow is often noisy (hedging,
            # positioning, gap-fills).  Only apply the full boost if there is
            # directional confirmation from ORB or breadth.  Without confirmation,
            # apply a mild penalty to avoid overstating conviction on raw flow.
            pre_10_15 = now.time() < time(10, 15)
            if pre_10_15:
                orb_confirmed = (
                    getattr(ext_ctx, "orb_breakout_confirmed", False)
                    if ext_ctx else False
                )
                breadth = getattr(ext_ctx, "breadth_ratio", 0.5) if ext_ctx else 0.5
                breadth_aligned = (
                    (right == "C" and breadth >= 0.65)
                    or (right == "P" and breadth <= 0.35)
                    or right == "BOTH"
                )
                if orb_confirmed or breadth_aligned:
                    tod_delta = (base * 1.05) - base   # confirmed: 5% boost (not 10%)
                else:
                    tod_delta = -0.04                  # unconfirmed: early-session caution
            else:
                # 10:15–10:30 ET: ORB is established but lock not yet active.
                # Apply boost ONLY if context confirms direction; otherwise neutral.
                # Previously this was a blind +10% which fired calls during declining
                # BELOW-VWAP conditions (root cause of MAY 27 2026 false positive).
                orb_confirmed = (
                    getattr(ext_ctx, "orb_breakout_confirmed", False)
                    if ext_ctx else False
                )
                orb_status = getattr(ext_ctx, "orb_status", "INSIDE") if ext_ctx else "INSIDE"
                vwap_pos = getattr(ext_ctx, "vwap_band_position", "INSIDE_1SD") if ext_ctx else "INSIDE_1SD"
                breadth = getattr(ext_ctx, "breadth_ratio", 0.5) if ext_ctx else 0.5

                # Direction-aligned confirmation required
                orb_aligned = (
                    (right == "C" and orb_confirmed and orb_status == "ABOVE_ORB") or
                    (right == "P" and orb_confirmed and orb_status == "BELOW_ORB")
                )
                vwap_aligned = (
                    (right == "C" and vwap_pos in ("ABOVE_1SD", "ABOVE_2SD")) or
                    (right == "P" and vwap_pos in ("BELOW_1SD", "BELOW_2SD")) or
                    right == "BOTH"
                )
                breadth_aligned = (
                    (right == "C" and breadth >= 0.60) or
                    (right == "P" and breadth <= 0.40) or
                    right == "BOTH"
                )
                confirms = sum([orb_aligned, vwap_aligned, breadth_aligned])
                if confirms >= 2:
                    tod_delta = (base * 1.05) - base   # well-confirmed: 5% boost
                elif confirms == 1:
                    tod_delta = 0.0                    # weak confirmation: neutral
                else:
                    tod_delta = -0.04                  # no confirmation: caution
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
            elif right == "P":
                aligned = flow_score < 0
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

            # Call signal: all sources should be positive; penalise if majority oppose
            if right == "C":
                opposing = sum([
                    flow_directional < -self._conflict_flow_thr,
                    sent_directional < -self._conflict_sent_thr,
                    macro_directional < -20,
                    regime == "TREND_DOWN",  # regime opposes call signal
                ])
            elif right == "P":
                opposing = sum([
                    flow_directional > self._conflict_flow_thr,
                    sent_directional > self._conflict_sent_thr,
                    macro_directional > 20,
                    regime == "TREND_UP",    # regime opposes put signal
                ])
            else:
                opposing = 0

            if opposing >= 2:
                conflict = True
                conflict_delta = -0.08
                total_delta += conflict_delta
                breakdown["conflict"] = round(conflict_delta, 4)

        # ── 7. Market breadth ────────────────────────────────────────────────
        if ext_ctx is not None and hasattr(ext_ctx, "breadth_ratio"):
            breadth = ext_ctx.breadth_ratio         # 0.0–1.0
            breadth_delta = 0.0
            if breadth >= 0.70 and right in ("C", "BOTH"):
                breadth_delta = 0.03   # strong breadth boosts calls
            elif breadth <= 0.30 and right in ("P", "BOTH"):
                breadth_delta = 0.03   # weak breadth boosts puts
            elif breadth <= 0.30 and right == "C":
                breadth_delta = -0.05  # weak breadth penalises calls
            elif breadth >= 0.70 and right == "P":
                breadth_delta = -0.05  # strong breadth penalises puts
            if breadth_delta != 0:
                total_delta += breadth_delta
                breakdown["breadth"] = round(breadth_delta, 4)

        # ── 8. Sector alignment ──────────────────────────────────────────────
        if ext_ctx is not None and hasattr(ext_ctx, "sector_label"):
            sec_delta = 0.0
            sector_label = ext_ctx.sector_label
            qqq_pct = getattr(ext_ctx, "qqq_vs_spy_pct", 0.0)
            if sector_label in ("BULL_SWEEP", "BULL_LEANING") and right == "C":
                sec_delta = 0.03
            elif sector_label in ("BEAR_SWEEP", "BEAR_LEANING") and right == "P":
                sec_delta = 0.03
            elif sector_label in ("BULL_SWEEP", "BULL_LEANING") and right == "P":
                sec_delta = -0.03
            elif sector_label in ("BEAR_SWEEP", "BEAR_LEANING") and right == "C":
                sec_delta = -0.03
            # Additional boost if QQQ is leading (tech leadership = momentum)
            if qqq_pct > 0.20 and right == "C" and sec_delta > 0:
                sec_delta += 0.02
            # Rotation detection: if QQQ/IWM are strong but signal is bearish,
            # the SPY weakness is likely sector rotation, not broad selling.
            # Similarly, if QQQ/IWM are weak but signal is bullish, SPY strength
            # may be defensive rotation, not a genuine rally.
            iwm_pct = getattr(ext_ctx, "iwm_vs_spy_pct", 0.0)
            if right == "P" and qqq_pct > 0.15:
                sec_delta -= 0.08  # QQQ green while SPY red → rotation, not sell-off
            elif right == "C" and qqq_pct < -0.15:
                sec_delta -= 0.06  # QQQ lagging while SPY up → narrowing rally
            # IWM divergence as secondary confirmation
            if right == "P" and iwm_pct > 0.20:
                sec_delta -= 0.04  # small caps strong → risk-on, not sell-off
            elif right == "C" and iwm_pct < -0.20:
                sec_delta -= 0.03  # small caps weak → risk-off headwind
            if sec_delta != 0:
                total_delta += sec_delta
                breakdown["sector"] = round(sec_delta, 4)

        # ── 9. Gamma wall proximity ──────────────────────────────────────────
        if ext_ctx is not None and hasattr(ext_ctx, "at_call_wall"):
            wall_delta = 0.0
            if ext_ctx.at_call_wall and right == "C":
                # Price at call wall = strong resistance → penalise call signals
                wall_delta = -0.05
            elif ext_ctx.at_call_wall and right == "P":
                # At call wall = natural support → slight boost for puts
                wall_delta = 0.03
            elif ext_ctx.at_put_wall and right == "P":
                # Price at put wall = strong support → penalise put signals
                wall_delta = -0.05
            elif ext_ctx.at_put_wall and right == "C":
                # Put wall = floor → boost calls
                wall_delta = 0.03
            if wall_delta != 0:
                total_delta += wall_delta
                breakdown["gamma_wall"] = round(wall_delta, 4)

        # ── 10. Volatility term structure ────────────────────────────────────
        if ext_ctx is not None and hasattr(ext_ctx, "vol_structure"):
            vs = ext_ctx.vol_structure
            vs_delta = 0.0
            vvix_elev = getattr(ext_ctx, "vvix_elevated", False)
            if vs in ("BACKWARDATION", "STEEP_BACKWARDATION") and right == "C":
                vs_delta = -0.05   # backwardation = fear, penalise calls
            elif vs == "STEEP_BACKWARDATION" and right == "P":
                vs_delta = 0.03    # panic → put premium elevated
            elif vs in ("CONTANGO", "STEEP_CONTANGO") and right == "C":
                vs_delta = 0.02    # calm regime, calls are cheaper
            if vvix_elev:
                # VVIX elevated → vol-of-vol is high; reduce conviction on any directional
                vs_delta -= 0.03
            if vs_delta != 0:
                total_delta += vs_delta
                breakdown["vol_structure"] = round(vs_delta, 4)

        # ── 11. Overnight context ─────────────────────────────────────────────
        if ext_ctx is not None and hasattr(ext_ctx, "above_overnight_high"):
            on_delta = 0.0
            if ext_ctx.above_overnight_high and right == "C":
                on_delta = 0.03   # breaking above overnight high = bullish momentum
            elif ext_ctx.below_overnight_low and right == "P":
                on_delta = 0.03   # breaking below overnight low = bearish momentum
            elif ext_ctx.above_overnight_high and right == "P":
                on_delta = -0.02  # puts are swimming against the current
            elif ext_ctx.below_overnight_low and right == "C":
                on_delta = -0.02  # calls against the current
            if on_delta != 0:
                total_delta += on_delta
                breakdown["overnight"] = round(on_delta, 4)

        # ── 12. OPEX gamma environment ───────────────────────────────────────
        if ext_ctx is not None and hasattr(ext_ctx, "gamma_environment"):
            ge = ext_ctx.gamma_environment
            opex_delta = 0.0
            if ge == "PINNING" and right in ("C", "P"):
                # Pinning environment → directional moves less likely
                opex_delta = -0.03
            elif ge == "EXPANSIVE" and right in ("C", "P"):
                # Expansive (short gamma) → moves accelerate → boost
                opex_delta = 0.02
            if opex_delta != 0:
                total_delta += opex_delta
                breakdown["opex_gamma_env"] = round(opex_delta, 4)

        # ── 13. Opening Range Breakout ────────────────────────────────────────
        # The 30-min ORB is the most reliable intraday directional filter for SPY.
        # A confirmed breakout above ORB strongly favours calls; below favours puts.
        # Signals fired while price is INSIDE the ORB (range-bound) are penalised.
        if ext_ctx is not None and hasattr(ext_ctx, "orb_established"):
            orb_delta = 0.0
            orb_status = getattr(ext_ctx, "orb_status", "BUILDING")
            orb_confirmed = getattr(ext_ctx, "orb_breakout_confirmed", False)

            if orb_status == "ABOVE_ORB" and orb_confirmed:
                if right == "C":
                    orb_delta = 0.05   # confirmed ORB breakout → strong call bias
                elif right == "P":
                    orb_delta = -0.05  # fading ORB breakout = lower probability
            elif orb_status == "BELOW_ORB" and orb_confirmed:
                if right == "P":
                    orb_delta = 0.05   # confirmed ORB breakdown → strong put bias
                elif right == "C":
                    orb_delta = -0.05  # fading ORB breakdown = lower probability
            elif orb_status == "INSIDE" and getattr(ext_ctx, "orb_established", False):
                # Price stuck inside ORB after 10:00 ET → range-bound; avoid direction
                if right in ("C", "P"):
                    orb_delta = -0.03
            # BUILDING: ORB not yet established — no adjustment

            if orb_delta != 0:
                total_delta += orb_delta
                breakdown["orb"] = round(orb_delta, 4)

        # ── 14. VWAP band exhaustion ──────────────────────────────────────────
        # Backtest (60 sessions, Apr–Jul 2026): the fade edge is ASYMMETRIC.
        #   BELOW −2σ → fading (buying calls) won 61% of the time  → boost kept
        #   ABOVE +2σ → fading (buying puts)  won only 43%         → upside
        #   extension tends to CONTINUE; put boost removed, call penalty softened.
        if ext_ctx is not None and hasattr(ext_ctx, "vwap_band_position"):
            band_pos = ext_ctx.vwap_band_position
            band_delta = 0.0

            if band_pos == "ABOVE_2SD":
                if right == "P":
                    band_delta = 0.0    # no fade edge above +2σ (43% backtest win)
                elif right == "C":
                    band_delta = -0.03  # stretched, but continuation is common
            elif band_pos == "BELOW_2SD":
                if right == "C":
                    band_delta = 0.05   # fade validated: 61% win rate
                elif right == "P":
                    band_delta = -0.06
            elif band_pos == "ABOVE_1SD":
                if right == "P":
                    band_delta = 0.0    # consistent with no-fade-above finding
                elif right == "C":
                    band_delta = -0.02
            elif band_pos == "BELOW_1SD":
                if right == "C":
                    band_delta = 0.02
                elif right == "P":
                    band_delta = -0.02

            if band_delta != 0:
                total_delta += band_delta
                breakdown["vwap_band"] = round(band_delta, 4)

        # ── 15. Expected Daily Range (EDR) exhaustion ─────────────────────────
        # Backtest (60 sessions): the exhaustion effect is TIME-DEPENDENT.
        #   85% crossed before 13:00 ET → continuation only 25% (penalty right)
        #   85% crossed after 13:00 ET  → continuation 67% — trend days run
        #   into the close, so the penalty is HALVED in the afternoon buckets.
        if ext_ctx is not None and hasattr(ext_ctx, "edr_used_pct"):
            edr_used = ext_ctx.edr_used_pct
            edr_delta = 0.0
            if edr_used >= 120:
                # Extreme extension — heavy penalty for continuation
                if right in ("C", "P"):
                    edr_delta = -0.08
            elif edr_used >= 85:
                # Exhausted — moderate penalty for directional
                if right in ("C", "P"):
                    edr_delta = -0.05
            elif edr_used >= 60:
                # Getting stretched — mild caution
                if right in ("C", "P"):
                    edr_delta = -0.02

            # Afternoon trend days tend to extend, not revert — soften penalty
            if edr_delta != 0 and tod in (_TOD.PRE_POWER, _TOD.POWER_HOUR):
                edr_delta = round(edr_delta / 2, 4)

            if edr_delta != 0:
                total_delta += edr_delta
                breakdown["edr_exhaustion"] = round(edr_delta, 4)

        # ── 16. RSI overbought/oversold ───────────────────────────────────────
        # Backtest (60 sessions, 134 divergence events): RSI divergence showed
        # NO predictive edge — trading in the divergence direction won only
        # 41-44% at 30/60-min horizons.  Divergence adjustments are therefore
        # NEUTRALIZED (0.0).  Divergence is still computed and reported on the
        # signal for context, but it no longer moves confidence.
        # Mild overbought/oversold adjustments are retained (untested but small).
        if ext_ctx is not None and hasattr(ext_ctx, "rsi_divergence"):
            rsi_ob  = getattr(ext_ctx, "rsi_overbought", False)
            rsi_os  = getattr(ext_ctx, "rsi_oversold", False)
            rsi_delta = 0.0

            if rsi_ob and right == "P":
                rsi_delta = 0.02   # overbought + put = slight boost
            elif rsi_ob and right == "C":
                rsi_delta = -0.02  # overbought momentum already baked in
            elif rsi_os and right == "C":
                rsi_delta = 0.02
            elif rsi_os and right == "P":
                rsi_delta = -0.02

            if rsi_delta != 0:
                total_delta += rsi_delta
                breakdown["rsi"] = round(rsi_delta, 4)

        # ── 17. Pivot point proximity ─────────────────────────────────────────
        # Floor-Trader pivot levels (PP, R1, R2, S1, S2) are key intraday support
        # and resistance. SPY exhibits measurable mean-reversion at these levels.
        if ext_ctx is not None and getattr(ext_ctx, "near_pivot", False):
            pivot_bias  = getattr(ext_ctx, "pivot_bias", "NEUTRAL")
            pivot_delta = 0.0

            if pivot_bias == "AT_RESISTANCE":
                if right == "P":
                    pivot_delta = 0.03   # natural ceiling → puts have edge
                elif right == "C":
                    pivot_delta = -0.04  # buying into resistance is low-probability
            elif pivot_bias == "AT_SUPPORT":
                if right == "C":
                    pivot_delta = 0.03   # natural floor → calls have edge
                elif right == "P":
                    pivot_delta = -0.04  # shorting at support is low-probability
            elif pivot_bias == "AT_PIVOT":
                # At PP exactly: direction could go either way — reduce conviction
                if right in ("C", "P"):
                    pivot_delta = -0.02

            if pivot_delta != 0:
                total_delta += pivot_delta
                breakdown["pivot"] = round(pivot_delta, 4)

        # ── 18. Max pain proximity ────────────────────────────────────────────
        # When SPY is near the max pain strike (options market-maker pinning target),
        # strong directional moves become less likely — especially 0DTE after noon.
        # Straddle/neutral signals are unaffected.
        if ext_ctx is not None and getattr(ext_ctx, "near_max_pain", False):
            mp_delta = 0.0
            if dte == 0 and right in ("C", "P"):
                mp_delta = -0.06   # 0DTE near max pain = strong pin risk
            elif dte <= 2 and right in ("C", "P"):
                mp_delta = -0.03   # short-dated options still affected
            if mp_delta != 0:
                total_delta += mp_delta
                breakdown["max_pain"] = round(mp_delta, 4)

        # ── 19. Dark pool bias ────────────────────────────────────────────────
        # flow_dark_pool is already collected from ExternalFlowConfirmation but
        # was never wired into confidence adjustments.  Institutional dark pool
        # activity is a strong directional confirmer.
        if ext_ctx is not None and hasattr(ext_ctx, "flow_dark_pool"):
            dp_delta = 0.0
            dp_bias = ext_ctx.flow_dark_pool
            if dp_bias == "DISTRIBUTION" and right == "P":
                dp_delta = 0.03    # institutional selling → confirms put thesis
            elif dp_bias == "DISTRIBUTION" and right == "C":
                dp_delta = -0.03   # selling pressure opposes call thesis
            elif dp_bias == "ACCUMULATION" and right == "C":
                dp_delta = 0.03    # institutional buying → confirms call thesis
            elif dp_bias == "ACCUMULATION" and right == "P":
                dp_delta = -0.03   # buying pressure opposes put thesis
            if dp_delta != 0:
                total_delta += dp_delta
                breakdown["dark_pool"] = round(dp_delta, 4)

        # ── 20. Real order flow: tape + L2 depth (JUL 5 2026) ─────────────────
        # Real data from IB: tick-by-tick prints classified against the NBBO
        # (tape_score) and aggregated SMART book imbalance (depth_imbalance).
        # UNBACKTESTED (no historical tick data) — weights deliberately small
        # (max combined ±8%) pending live calibration.  Conflict asymmetry:
        # opposing tape is penalised slightly harder than aligned tape is
        # rewarded, because fighting real-time aggression is the costlier error.
        if ext_ctx is not None and getattr(ext_ctx, "tape_available", False):
            tape = ext_ctx.tape_score          # -100..+100
            tape_delta = 0.0
            if right == "C":
                aligned_score = tape
            elif right == "P":
                aligned_score = -tape
            else:
                aligned_score = 0.0            # straddle: direction-agnostic

            if aligned_score >= 60:
                tape_delta = 0.05              # strong real aggression with us
            elif aligned_score >= 30:
                tape_delta = 0.03
            elif aligned_score <= -60:
                tape_delta = -0.06             # strong real aggression against us
            elif aligned_score <= -30:
                tape_delta = -0.04

            # Block-print bias adds ±2% when it agrees/disagrees with the signal
            large = getattr(ext_ctx, "tape_large_bias", "NEUTRAL")
            if large == "BUY" and right == "C":
                tape_delta += 0.02
            elif large == "SELL" and right == "P":
                tape_delta += 0.02
            elif large == "BUY" and right == "P":
                tape_delta -= 0.02
            elif large == "SELL" and right == "C":
                tape_delta -= 0.02

            if tape_delta != 0:
                total_delta += tape_delta
                breakdown["tape"] = round(tape_delta, 4)

        if ext_ctx is not None and getattr(ext_ctx, "depth_available", False):
            imb = ext_ctx.depth_imbalance      # -1..+1 (positive = bid support)
            depth_delta = 0.0
            if right == "C":
                aligned_imb = imb
            elif right == "P":
                aligned_imb = -imb
            else:
                aligned_imb = 0.0

            if aligned_imb >= 0.30:
                depth_delta = 0.02             # book stacked in our favour
            elif aligned_imb <= -0.30:
                depth_delta = -0.03            # book stacked against us

            if depth_delta != 0:
                total_delta += depth_delta
                breakdown["depth"] = round(depth_delta, 4)

        # ── Finalise ─────────────────────────────────────────────────────────
        # Hard cap at 0.95 — no signal should ever reach 100% confidence.
        # This preserves uncertainty and prevents over-conviction from
        # stacking additive adjustments on an already-high base.
        final = max(0.0, min(0.95, base + total_delta))
        breakdown["base"] = round(base, 4)
        breakdown["total_delta"] = round(total_delta, 4)

        # Log breakdown for debugging (only when delta is significant)
        if abs(total_delta) >= 0.03:
            active = {k: v for k, v in breakdown.items()
                      if k not in ("base", "total_delta") and v != 0}
            logger.debug(
                "DynConf {}: base={:.0%} delta={:+.0%} final={:.0%} | {}",
                right, base, total_delta, final,
                " ".join(f"{k}={v:+.0%}" for k, v in active.items()) or "no adj",
            )

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
