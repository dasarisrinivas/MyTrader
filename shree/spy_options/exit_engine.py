"""Exit Engine v2 — multi-stage, confirmation-based exit decisions (JUL 14 2026).

Design: docs/SPY_EXIT_ENGINE_DESIGN.md.

Replaces the legacy 7-trigger "any one reason → full close" exit logic with:
  • a grace period (soft exits suppressed right after entry; hard exits live),
  • an Exit Confidence score (binary factors × weights, 0–100) with an
    adaptive threshold and two-evaluation confirmation + hysteresis,
  • per-closed-bar confirmation counters (no single-candle exits),
  • a profit ladder (partial at +1R → stop to breakeven → trail),
  • a strict priority ordering (catastrophic > time ceiling > confirmed
    confidence > profit protection; informational factors never exit alone).

PURITY CONTRACT: this module imports nothing from the trading stack and does
no I/O. `ExitEngine.evaluate(state, snap)` is deterministic: identical
(state, snap) → identical decision. All bar-based counters advance only when
`snap.last_bar_ts` changes (i.e. on a newly CLOSED 5-minute bar — the bar
feed already excludes the forming bar). The manager adapter owns building
`ExitSnapshot` and executing decisions; the executor owns order mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..config.spy_options import SpyOptionsExitEngineConfig

# Actions
HOLD = "HOLD"
FULL_EXIT = "FULL_EXIT"
PARTIAL_EXIT = "PARTIAL_EXIT"        # carries fraction + new_stop_r (breakeven)
TRAIL_STOP = "TRAIL_STOP"            # carries new_stop_r

# Stages (derived, for logging/telemetry)
STAGE_GRACE = "GRACE"
STAGE_MANAGE = "MANAGE"
STAGE_BE_LOCK = "BE_LOCK"
STAGE_EXIT_PENDING = "EXIT_PENDING"

_BULL_BANDS = ("ABOVE_1SD", "ABOVE_2SD")
_BEAR_BANDS = ("BELOW_1SD", "BELOW_2SD")


@dataclass
class ExitSnapshot:
    """Frozen inputs for one evaluation. Scalars only — trivially testable.

    The adapter (manager) computes everything here from data it already has
    per poll; the engine never fetches anything.
    """

    direction: str                    # "BULLISH" | "BEARISH"
    spy_price: float
    entry_spy: float

    # Latest CLOSED 5m bar (forming bar excluded upstream).
    last_bar_ts: str                  # ISO ts of the latest closed bar
    last_close: float
    last_bar_range: float             # high − low of that bar

    # Indicators at/for the latest closed bar.
    vwap: float
    ema9: float
    ema21: float
    ema9_slope: float                 # >0 rising
    rsi_5m: float
    atr14: float
    vwap_band: str                    # ABOVE_2SD…BELOW_2SD
    regime: str                       # TREND_UP / TREND_DOWN / …

    # Live context.
    minutes_held: float
    max_hold_min: float
    dte: int
    is_late_0dte: bool                # 0DTE and past 14:00 ET (adapter computes)
    vix: Optional[float]
    tape_score: Optional[float]       # RealFlow tape; None when unavailable
    delta_now: Optional[float]
    unrealized_r: Optional[float]     # premium R vs the premium stop distance
    premium_loss_pct_est: float       # ≥0; adverse premium move estimate, %
    spy_adverse_pct: float            # ≥0; SPY move against the thesis, %

    # Entry anchors.
    entry_vwap_band: str
    entry_regime: str
    entry_tier: str                   # MEDIUM/HIGH/EXTREME
    entry_confidence: float           # 0..1
    entry_rsi: Optional[float]
    entry_delta: Optional[float]
    iv_stop_pct: float                # premium stop % (IV-adjusted)


@dataclass
class PositionExitState:
    """Persistent per-position state. Mutated only inside evaluate()."""

    # Confirmation counters (advance once per newly closed bar).
    vwap_cross_closes: int = 0
    band_lost_closes: int = 0
    ema9_adverse_closes: int = 0
    ema21_adverse_closes: int = 0
    adverse_close_streak: int = 0
    bars_since_extreme: int = 0
    # Poll-cadence counters.
    regime_flip_evals: int = 0
    tape_against_polls: int = 0
    # Bar bookkeeping.
    last_bar_ts: str = ""
    last_seen_close: Optional[float] = None
    favorable_extreme: Optional[float] = None
    atr_spike_bar_ts: str = ""
    # Ladder / confirmation state.
    hwm_r: float = 0.0
    partial_taken: bool = False
    trailed_to_r: float = -999.0
    pending: bool = False

    def stage(self, in_grace: bool) -> str:
        if in_grace:
            return STAGE_GRACE
        if self.pending:
            return STAGE_EXIT_PENDING
        if self.partial_taken:
            return STAGE_BE_LOCK
        return STAGE_MANAGE


@dataclass
class ExitDecision:
    action: str
    score: int
    threshold: int
    factors: List[Tuple[str, int]] = field(default_factory=list)
    reason: str = ""
    fraction: float = 0.0             # PARTIAL_EXIT only
    new_stop_r: Optional[float] = None  # PARTIAL_EXIT (0.0=BE) / TRAIL_STOP
    stage: str = STAGE_MANAGE

    def factors_str(self) -> str:
        return "+".join(f"{n}({w})" for n, w in self.factors) or "none"


class ExitEngine:
    """Pure decision engine. One instance per bot; state lives per position."""

    def __init__(self, cfg: SpyOptionsExitEngineConfig):
        self._cfg = cfg

    # ── public API ──────────────────────────────────────────────────────────

    def evaluate(self, state: PositionExitState, snap: ExitSnapshot) -> ExitDecision:
        cfg = self._cfg
        bull = snap.direction == "BULLISH"

        # P2 — catastrophic: grace does NOT shield these.
        if snap.spy_adverse_pct >= cfg.catastrophic_adverse_spy_pct:
            return self._exit(state, snap, 100, [],
                              f"catastrophic: SPY {snap.spy_adverse_pct:.2f}% adverse")
        if (snap.iv_stop_pct > 0 and snap.premium_loss_pct_est
                >= snap.iv_stop_pct + cfg.catastrophic_premium_buffer_pct):
            return self._exit(state, snap, 100, [],
                              f"catastrophic: est premium −{snap.premium_loss_pct_est:.0f}%"
                              f" ≥ stop {snap.iv_stop_pct:.0f}%+{cfg.catastrophic_premium_buffer_pct:.0f}")

        # P3 — time ceiling (the 0DTE 15:50 flatten lives in the executor).
        if snap.minutes_held >= snap.max_hold_min:
            return self._exit(state, snap, 100, [],
                              f"max hold {snap.max_hold_min:.0f}min reached")

        # Advance confirmation counters (also during grace — evidence accrues,
        # action is what grace suppresses).
        self._update_counters(state, snap, bull)

        grace_min = cfg.grace_min_0dte if snap.dte == 0 else cfg.grace_min_swing
        in_grace = snap.minutes_held < grace_min

        score, factors = self._score(state, snap, bull)
        thr = self._threshold(state, snap)

        if in_grace:
            return ExitDecision(HOLD, score, thr, factors, "grace",
                                stage=state.stage(True))

        # P5 — profit ladder (winners are managed before soft-exit logic).
        r = snap.unrealized_r
        if r is not None:
            state.hwm_r = max(state.hwm_r, r)
            if r >= cfg.partial_at_r and not state.partial_taken:
                state.partial_taken = True
                state.trailed_to_r = 0.0
                return ExitDecision(
                    PARTIAL_EXIT, score, thr, factors,
                    f"profit ladder: +{r:.2f}R ≥ {cfg.partial_at_r:.1f}R — "
                    f"scale out {cfg.partial_fraction:.0%}, stop → breakeven",
                    fraction=cfg.partial_fraction, new_stop_r=0.0,
                    stage=STAGE_BE_LOCK,
                )
            if state.partial_taken and state.hwm_r - r >= cfg.trail_giveback_r:
                lock_r = round(state.hwm_r * cfg.trail_lock_frac, 2)
                if lock_r > state.trailed_to_r + 0.1:
                    state.trailed_to_r = lock_r
                    return ExitDecision(
                        TRAIL_STOP, score, thr, factors,
                        f"trail: HWM {state.hwm_r:.2f}R, gave back "
                        f"{state.hwm_r - r:.2f}R — lock +{lock_r:.2f}R",
                        new_stop_r=lock_r, stage=STAGE_BE_LOCK,
                    )

        # P4 — exit confidence with confirmation + hysteresis.
        if score >= thr + cfg.one_shot_margin:
            return self._exit(state, snap, score, factors,
                              f"exit confidence {score} ≥ {thr}+{cfg.one_shot_margin} (strong)")
        if score >= thr:
            if state.pending:
                return self._exit(state, snap, score, factors,
                                  f"exit confidence {score} ≥ {thr} (2nd consecutive eval)")
            state.pending = True
            return ExitDecision(HOLD, score, thr, factors,
                                "pending confirmation", stage=STAGE_EXIT_PENDING)
        if state.pending and score < thr - cfg.deescalate_margin:
            state.pending = False          # hysteresis: de-escalate
        return ExitDecision(HOLD, score, thr, factors, "hold",
                            stage=state.stage(False))

    # ── internals ────────────────────────────────────────────────────────────

    def _exit(self, state: PositionExitState, snap: ExitSnapshot,
              score: int, factors: List[Tuple[str, int]], reason: str) -> ExitDecision:
        state.pending = False
        return ExitDecision(FULL_EXIT, score, self._threshold(state, snap),
                            factors, reason, stage=state.stage(False))

    def _update_counters(self, state: PositionExitState, snap: ExitSnapshot,
                         bull: bool) -> None:
        """Advance per-closed-bar counters exactly once per new bar, plus the
        poll-cadence counters every evaluation."""
        new_bar = snap.last_bar_ts != state.last_bar_ts
        if new_bar:
            state.last_bar_ts = snap.last_bar_ts
            c = snap.last_close

            adverse_vwap = c < snap.vwap if bull else c > snap.vwap
            state.vwap_cross_closes = state.vwap_cross_closes + 1 if adverse_vwap else 0

            side = _BULL_BANDS if bull else _BEAR_BANDS
            band_lost = (snap.entry_vwap_band in side) and (snap.vwap_band not in side)
            state.band_lost_closes = state.band_lost_closes + 1 if band_lost else 0

            adverse_e9 = c < snap.ema9 if bull else c > snap.ema9
            state.ema9_adverse_closes = state.ema9_adverse_closes + 1 if adverse_e9 else 0

            adverse_e21 = c < snap.ema21 if bull else c > snap.ema21
            state.ema21_adverse_closes = state.ema21_adverse_closes + 1 if adverse_e21 else 0

            adverse_close = False
            if state.last_seen_close is not None:
                adverse_close = c < state.last_seen_close if bull else c > state.last_seen_close
                state.adverse_close_streak = (
                    state.adverse_close_streak + 1 if adverse_close else 0
                )
            state.last_seen_close = c

            if state.favorable_extreme is None:
                state.favorable_extreme = c
                state.bars_since_extreme = 0
            else:
                new_extreme = c > state.favorable_extreme if bull else c < state.favorable_extreme
                if new_extreme:
                    state.favorable_extreme = c
                    state.bars_since_extreme = 0
                else:
                    state.bars_since_extreme += 1

            # Wide adverse bar: range blew past 1.5×ATR AND the bar closed
            # against the thesis. Transient — active only while this bar is
            # the latest closed bar.
            if (snap.atr14 > 0 and snap.last_bar_range > 1.5 * snap.atr14
                    and adverse_close):
                state.atr_spike_bar_ts = snap.last_bar_ts

        # Poll-cadence counters (every evaluation).
        opp = "TREND_DOWN" if bull else "TREND_UP"
        slope_against = snap.ema9_slope < 0 if bull else snap.ema9_slope > 0
        if snap.regime == opp and slope_against:
            state.regime_flip_evals += 1
        else:
            state.regime_flip_evals = 0

        if snap.tape_score is not None:
            against = snap.tape_score < 0 if bull else snap.tape_score > 0
            state.tape_against_polls = state.tape_against_polls + 1 if against else 0

    def _score(self, state: PositionExitState, snap: ExitSnapshot,
               bull: bool) -> Tuple[int, List[Tuple[str, int]]]:
        cfg = self._cfg
        f: List[Tuple[str, int]] = []

        if state.vwap_cross_closes >= cfg.vwap_cross_closes:
            f.append(("vwap_full_reversion", cfg.w_vwap_full_reversion))
        slope_against = snap.ema9_slope < 0 if bull else snap.ema9_slope > 0
        if state.band_lost_closes >= cfg.vwap_band_decay_closes and slope_against:
            f.append(("vwap_band_decay", cfg.w_vwap_band_decay))
        if state.ema9_adverse_closes >= cfg.ema9_cross_closes:
            f.append(("ema9_cross", cfg.w_ema9_cross))
        if state.ema21_adverse_closes >= 1:
            f.append(("ema21_break", cfg.w_ema21_break))
        if state.regime_flip_evals >= cfg.regime_confirm_evals:
            f.append(("regime_flip_confirmed", cfg.w_regime_flip))
        if state.bars_since_extreme >= 4 and state.adverse_close_streak >= 2:
            f.append(("momentum_stall", cfg.w_momentum_stall))
        if self._rsi_reversal(snap, bull):
            f.append(("rsi_reversal", cfg.w_rsi_reversal))
        if state.tape_against_polls >= 2:
            f.append(("tape_flip", cfg.w_tape_flip))
        if state.atr_spike_bar_ts and state.atr_spike_bar_ts == snap.last_bar_ts:
            f.append(("atr_adverse_expansion", cfg.w_atr_adverse_expansion))
        if (snap.delta_now is not None and snap.entry_delta
                and abs(snap.delta_now) < 0.7 * abs(snap.entry_delta)):
            f.append(("delta_decay", cfg.w_delta_decay))

        return min(100, sum(w for _, w in f)), f

    @staticmethod
    def _rsi_reversal(snap: ExitSnapshot, bull: bool) -> bool:
        """RSI crossed 50 against the thesis, from an entry-side extreme.
        Without a recorded entry RSI, require a deeper cross (45/55) so a
        50.1→49.9 wiggle can't score."""
        if snap.entry_rsi is not None:
            if bull:
                return snap.entry_rsi >= 60 and snap.rsi_5m < 50
            return snap.entry_rsi <= 40 and snap.rsi_5m > 50
        return snap.rsi_5m < 45 if bull else snap.rsi_5m > 55

    def _threshold(self, state: PositionExitState, snap: ExitSnapshot) -> int:
        cfg = self._cfg
        thr = cfg.base_threshold
        if snap.entry_tier == "EXTREME" or snap.entry_confidence >= 0.85:
            thr += 10
        structure_intact = (
            snap.regime == snap.entry_regime
            and (snap.ema9_slope > 0 if snap.direction == "BULLISH" else snap.ema9_slope < 0)
        )
        if structure_intact:
            thr += 5
        if snap.unrealized_r is not None and snap.unrealized_r >= 0.5:
            thr += 5
        if snap.is_late_0dte:
            thr -= 10
        if snap.vix is not None and snap.vix >= 24.0:
            thr -= 5
        if snap.max_hold_min > 0 and snap.minutes_held >= 0.75 * snap.max_hold_min:
            thr -= 5
        return max(40, min(80, thr))
