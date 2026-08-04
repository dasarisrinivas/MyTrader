"""Rules-v2 configuration schema.

A regime-first, structure-based rules layer that sits *between* the legacy
``SignalEngine.evaluate()`` output and ``SpyOptionsManager._dispatch_signals``.
All behaviour is gated by ``enabled`` — when ``False`` the legacy pipeline is
entirely untouched.

All numeric thresholds expressed as percentages are in *decimal* form
(0.0025 == 0.25%). Times are expressed in US/Eastern hours:minutes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RegimeV2Config:
    """Thresholds for the v2 regime classifier."""

    # Minimum continuous minutes above/below VWAP for a TREND regime
    trend_min_vwap_minutes: int = 15

    # VWAP 5-bar slope threshold (decimal of price per bar).
    # Calibrated against the Apr 21 2026 tape: a 1% decline spread over ~90
    # minutes produces a *session* VWAP slope around -0.00014 to -0.00018/bar
    # (the session VWAP is slow-moving because it integrates the whole morning's
    # chop). Setting the threshold to 0.00010 catches shallow-grind trends that
    # are nonetheless directional, while chop days (slope < 0.00008) still
    # classify as RANGE_BOUND.
    trend_vwap_slope: float = 0.00010

    # ATR expansion ratio: ATR(5) / ATR(20) must exceed this for TREND.
    # Lowered from 1.10 → 1.05 so we catch slow-but-real expansions (Apr 21
    # reached ~1.11 only at the very bottom; 1.05 triggers mid-trend).
    trend_atr_expansion: float = 1.05

    # Pivots required in last 30 min for HH/HL (or LH/LL) trend structure.
    # A pure staircase (e.g. Apr 21 13:00–13:30 ET) produces one confirmed
    # LOW pivot but rarely two + two matching HIGHs without a pullback; one
    # LH/LL pair is sufficient evidence of trend structure.
    trend_pivot_count: int = 1

    # VWAP slope below this → range-eligible. Must be strictly less than
    # trend_vwap_slope to avoid an ambiguous overlap zone.
    range_slope_max: float = 0.00008

    # VWAP crosses in last 30 min ≥ this → range-eligible
    range_vwap_crosses: int = 3

    # ATR contraction ratio: ATR(5) / ATR(20) below this → range-eligible
    range_atr_contraction: float = 0.90

    # Minimum bars required to classify; fewer ⇒ TRANSITION
    min_bars: int = 22

    # ── Grind-trend path (JUL 7 2026) ──────────────────────────────────────
    # The strict trend gates above require BOTH a VWAP-slope threshold AND ATR
    # expansion. That excludes low-volatility directional grinds — steady drift
    # on CONTRACTING ATR with a flat anchored-VWAP slope (e.g. SPY 750→746 on
    # Jul 7). Those are real trends for pullback continuation. This path calls a
    # trend from structure + EMA cross + a decisive VWAP side, without requiring
    # ATR expansion or the VWAP-slope threshold. Guarded three ways so chop
    # can't qualify: price decisively beyond VWAP (band), EMA9/EMA21 separated
    # by a real margin, and EMA9 sloping the trend way over 5 bars (the
    # responsive directional check the anchored-VWAP slope misses).
    grind_vwap_band_pct: float = 0.0015   # price ≥ 0.15% beyond VWAP
    grind_ema_sep_min: float = 0.0003     # EMA9/EMA21 gap ≥ 0.03% of price
    # A real grind stays on ONE side of VWAP; chop whips across it. Veto the
    # grind path when price crossed VWAP more than this in the last 30 min.
    grind_max_vwap_crosses: int = 1


@dataclass
class OrbGateConfig:
    """Time-window + confirmation gates for ORB_BREAKOUT."""

    # Hard time window (US/Eastern) where ORB_BREAKOUT is even considered.
    # Outside this window, every ORB signal is blocked regardless of confidence.
    # Default: 09:45–11:00 ET (15 min after ORB forms, to 90 min after open).
    window_start_et: str = "09:45"
    window_end_et: str = "11:00"

    # Number of consecutive 5-min bars that must close beyond ORB before entry
    confirmation_bars: int = 2

    # Volume on breakout bar as multiple of 20-bar average
    volume_multiplier: float = 1.5

    # Minimum VWAP distance at entry (decimal). ORB hugging VWAP is fakeout-prone.
    min_vwap_distance_pct: float = 0.0015

    # RSI extreme opposite to trade direction blocks entry
    rsi_exhaustion_upper: float = 70.0
    rsi_exhaustion_lower: float = 30.0


@dataclass
class ContinuationConfig:
    """TREND_CONTINUATION signal generator settings."""

    # Signal is only eligible when regime ∈ {TREND_UP, TREND_DOWN}
    # and the clock is past this ET time (after the ORB window closes)
    min_time_et: str = "10:30"

    # Max time ET — last-chance continuation entries
    max_time_et: str = "15:30"

    # Pullback anchor: which line price must pull back to before a rejection
    # counts as a continuation entry.
    #   "vwap"     — session VWAP (works on choppy days where trend reverts)
    #   "ema9"     — 9-period EMA on closes (closest anchor; matches bull-flag /
    #                bear-flag pullbacks on strong intraday trends)
    #   "ema21"    — 21-period EMA on closes (slower anchor for medium trends)
    #   "nearest"  — whichever of VWAP / EMA9 / EMA21 is closest to current
    #                price on the pullback side (most forgiving). This is the
    #                recommended default: the pullback hits whichever anchor
    #                is closest first.
    pullback_anchor: str = "nearest"

    # Pullback band around the anchor line (decimal): entry only triggers if
    # the pullback bar's range crossed anchor ± band. 0.0010 = 0.10% of SPY
    # (≈ $0.71 at SPY=710) — wide enough to catch a bar whose wick reaches
    # toward the anchor without demanding a perfect touch.
    pullback_band_pct: float = 0.0010

    # Rejection candle: wick on the trend side ≥ this fraction of total bar range
    rejection_wick_ratio: float = 0.50

    # Alternative: engulfing bar counts as a rejection
    allow_engulfing: bool = True

    # Invalidation: 5m close beyond the pullback anchor by more than this
    # (decimal) kills the setup (trend-break).
    invalidation_vwap_pct: float = 0.0020

    # Default confidence for a clean continuation setup (0.80 → HIGH tier)
    base_confidence: float = 0.82


@dataclass
class PcRatioAlignmentConfig:
    """Alignment gate for PC_RATIO_EXTREME (legacy signal, re-scoped)."""

    # If True, PC_RATIO_EXTREME signals are suppressed when direction disagrees
    # with the prevailing regime. (Bearish PC extreme ⇒ must have TREND_DOWN.)
    require_trend_alignment: bool = True

    # Also require an expansion bar in trend direction within last N minutes
    require_expansion: bool = True
    expansion_lookback_min: int = 15
    expansion_atr_multiple: float = 1.5

    # Fully suppressed in RANGE_BOUND regime
    suppress_in_range: bool = True


@dataclass
class ThrottleConfig:
    """Structure-based throttle replacing the legacy time-window directional cap."""

    # Zone lock: block re-entry within ±this fraction of prior entry price
    zone_lock_pct: float = 0.0010

    # Max entries per structural leg (leg = run between opposing pivots)
    max_entries_per_leg: int = 3

    # New pivot unlocks fresh entry. Swing lookback for pivot detection (bars).
    pivot_lookback_bars: int = 3

    # Any regime flip to TRANSITION or opposite trend flushes the leg counter
    reset_on_regime_flip: bool = True


@dataclass
class EntryGateConfig:
    """Pre-entry filters preventing exhausted / parabolic entries."""

    # Block entry when price is this far extended from VWAP in trade direction
    max_vwap_extension_pct: float = 0.0025

    # Block entry during the N-th consecutive expansion bar (parabolic)
    parabolic_bar_count: int = 3

    # RSI extremes opposite to trade direction
    rsi_upper: float = 70.0
    rsi_lower: float = 30.0

    # Minimum signal confidence to pass the gate.
    #
    # AUG 4 2026 — DISABLED (defect D1, 2026-08-04 forensic audit Phase 2).
    # This floor evaluates `sig.confidence`, the SAME value the engine-level
    # confidence gate uses (manager.py passes `confidence=sig.confidence` into
    # RulesV2Engine.filter). When the engine gate was demoted to a passive
    # metric on 2026-08-03, this second, independent gate survived — so
    # confidence was still blocking trades and the confidence experiment was
    # not running end-to-end.
    #
    # Evidence (2026-08-04 session): 91 of 125 `rules_v2:entry_gate`
    # rejections (72.8%) were this floor. Rejected confidences ran
    # min 0.09 / median 0.43 / max 0.70 — and the confidence score itself has
    # AUC(conf -> win) = 0.4882, i.e. no rank information.
    #
    # Production must have exactly ONE confidence decision. The engine gate is
    # the designated owner and it is passive, so this floor is off.
    # Revert = set confidence_floor_enabled back to True.
    confidence_floor_enabled: bool = False
    min_confidence: float = 0.70   # retained as the reference value only


@dataclass
class ExitRulesConfig:
    """Scratch-resistant exit logic: min-hold + structural stops + scale-out."""

    # Minimum hold in minutes OR 1R move, whichever first, before scratch allowed.
    # Prevents first-tick noise exits that produced 5/6 scratches on Apr 21.
    min_hold_minutes: int = 7

    # Hold until at least 1R before scratch permitted (R = adverse distance to
    # the structural stop at entry, expressed in SPY points).
    min_hold_r_multiple: float = 1.0

    # Scale first tranche at this R multiple
    scale_r_multiple: float = 1.5
    scale_fraction: float = 0.50

    # Trail remaining on 5-min swing pivots (lookback for pivot)
    trail_pivot_lookback: int = 3

    # Hard flat time ET
    hard_flat_et: str = "15:45"


@dataclass
class StrikeSelectorConfig:
    """Delta-based strike selection per signal type."""

    # (min_delta, max_delta) windows for absolute delta
    trend_continuation: Tuple[float, float] = (0.40, 0.45)
    orb_breakout: Tuple[float, float] = (0.30, 0.40)
    pc_ratio_extreme: Tuple[float, float] = (0.35, 0.45)
    call_sweep: Tuple[float, float] = (0.30, 0.50)
    put_sweep: Tuple[float, float] = (0.30, 0.50)

    # Floor: reject strikes with |delta| below this unless VIX + regime override
    min_delta_floor: float = 0.25
    min_delta_floor_vix_override: float = 25.0
    min_delta_floor_ivr_override: float = 40.0

    # Prefer 0DTE when IVR < this, else 1DTE
    prefer_0dte_ivr_max: float = 40.0


@dataclass
class ExpectedMoveGateConfig:
    """Reject debit signals where IV-implied expected move does not
    justify the leg's own debit.

    EM = spy_price * leg_iv * sqrt(max(dte,1) / 365)
    Reject if EM < leg_mid * em_multiplier.

    Default multiplier 1.2 = 20% cushion above breakeven. Conservative;
    tune downward if block rate is too aggressive.
    """

    em_multiplier: float = 1.2


@dataclass
class TimeOfDayConfig:
    """Session-phase behaviour toggles (all ET)."""

    observation_end: str = "09:45"     # 09:30–09:45 no signals
    orb_window_end: str = "11:00"      # 09:45–11:00 ORB primary
    midday_start: str = "11:30"
    midday_end: str = "13:30"          # 11:30–13:30 midday — tightened thresholds
    afternoon_end: str = "15:00"       # 13:30–15:00 continuation primary
    close_end: str = "15:30"           # 15:00–15:30 last entries, tighter stops

    # Midday modifiers
    midday_confidence_multiplier: float = 2.0   # Effective: 2× base threshold
    midday_size_fraction: float = 0.50


@dataclass
class RulesV2Config:
    """Top-level feature flag + sub-configs for the rules_v2 layer."""

    # Master feature flag — OFF by default preserves legacy behaviour.
    enabled: bool = False

    # Per-rule toggles for selective rollout
    regime_v2_enabled: bool = True
    orb_gate_enabled: bool = True
    continuation_enabled: bool = True
    pc_ratio_alignment_enabled: bool = True
    throttle_structure_enabled: bool = True
    entry_gate_enabled: bool = True
    exit_rules_enabled: bool = True
    strike_selector_enabled: bool = True
    time_of_day_enabled: bool = True
    # NEW: expected-move gate. Off by default — flip on in config.yaml after
    # paper-soak validates block rate is in the expected 5–15% band.
    expected_move_gate_enabled: bool = False

    regime: RegimeV2Config = field(default_factory=RegimeV2Config)
    orb_gate: OrbGateConfig = field(default_factory=OrbGateConfig)
    continuation: ContinuationConfig = field(default_factory=ContinuationConfig)
    pc_ratio_alignment: PcRatioAlignmentConfig = field(default_factory=PcRatioAlignmentConfig)
    throttle: ThrottleConfig = field(default_factory=ThrottleConfig)
    entry_gate: EntryGateConfig = field(default_factory=EntryGateConfig)
    exit_rules: ExitRulesConfig = field(default_factory=ExitRulesConfig)
    strike_selector: StrikeSelectorConfig = field(default_factory=StrikeSelectorConfig)
    time_of_day: TimeOfDayConfig = field(default_factory=TimeOfDayConfig)
    expected_move_gate: ExpectedMoveGateConfig = field(
        default_factory=ExpectedMoveGateConfig
    )
