"""Enhanced SPY options signal engine with weighted confidence scoring.

═══════════════════════════════════════════════════════════════════════════════
ADVISORY-ONLY — NO ORDERS ARE EVER PLACED.
═══════════════════════════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNAL TYPES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIMARY (directional, actionable):
    CALL_SWEEP       — Large call volume spike with bullish bid pressure
    PUT_SWEEP        — Large put volume spike with bearish pressure
    BULL_CALL_SPREAD — Low IV rank + call sweep → debit spread preferred
    BEAR_PUT_SPREAD  — Low IV rank + put sweep  → debit spread preferred
    ORB_BREAKOUT     — Confirmed 30-min Opening Range Breakout (strongest filter)
    LONG_STRADDLE    — Both call AND put volume spike → volatility play

INFORMATIONAL (environment alerts, use as modifiers):
    HIGH_IV_ALERT    — Elevated VIX / high IV rank → premium-selling environment
                       Note: Not a directional trade — signals credit structure preference.
    PC_RATIO_EXTREME — Chain-level P/C at extreme → secondary confirmer only.
                       Never override primary price/flow signals with this alone.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNAL PRIORITY HIERARCHY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tier 1 (highest authority — can block a signal outright):
  • Options flow score  (aggregate_score ≥ ±30 = strong directional conviction)
  • ORB breakout        (confirmed above/below 30-min range = momentum structure)

Tier 2 (strong modifier — can heavily adjust confidence):
  • Market regime       (TREND_UP / TREND_DOWN / RANGE_BOUND)
  • VWAP band position  (above/below +2σ = stretched; inside = ambiguous)

Tier 3 (secondary filter — fine-tunes but cannot override Tier 1):
  • RSI divergence
  • Pivot proximity
  • EDR exhaustion

Rule: If 2 Tier-1 factors oppose a signal direction → confidence reduced by −15%.
      Sentiment (StockTwits, Reddit, News) is NEVER Tier 1 — it only adjusts
      within ±5% and cannot override price action or flow.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL CONFIDENCE FORMULA (single authoritative equation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  final_confidence = min(0.95, max(0.0,
      base_confidence                  # 10-component weighted model (0.0–1.0)
      + signal_specific_adjustments    # stale-decay, move-exhaustion, RSI-div,
                                       # DTE-adj, vol-gate, macro-vel, TICK,
                                       # price-structure, trap-detect, sentiment,
                                       # external-composite, VWAP, regime,
                                       # multi-factor (flow/breadth/GEX/DP/QQQ/IWM)
      + dynamic_confidence_delta       # 19-factor DynamicConfidence adjuster
      + priority_conflict_adjustment   # Tier-1 priority hierarchy (±0.0–0.15)
  ))

  Then quality gate (hard blocks — no confidence override):
    • Flow opposes direction strongly (score ≤ −30 for calls, ≥ +30 for puts)
    • 0DTE AND near max pain
    • Directional AND inside ORB after 10:30 ET
    • SPY intraday range < 0.20% (chop day)

  Then TOD ceiling (time-of-day hard cap):
    OPEN    0.82 | PRIME  0.95 | LUNCH 0.80 | AFTERNOON 0.88 | CLOSE 0.93

  Confidence tiers (from SpyOptionsSignalConfig):
    MEDIUM  ≥ confidence_tier_high    (default 0.70)
    HIGH    ≥ confidence_tier_high    (default 0.70)
    EXTREME ≥ confidence_tier_extreme (default 0.80)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE CONFIDENCE COMPONENTS (10 weighted factors)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    25% volume spike strength  (spike_mult vs rolling avg)
    15% bid/ask imbalance      (directional order pressure)
    10% delta quality          (ideal: calls 0.30–0.60, puts −0.60 to −0.30)
    10% gamma quality          (ideal ATM range: 0.005–0.08)
    10% theta penalty          (rapid decay hurts short-dated longs)
    10% IV regime alignment    (low IV → debit spreads; high IV → credit spreads)
    10% sentiment alignment    (IB order-flow sentiment score)
     5% open interest strength (≥10k OI = full score)
     5% flow score             (repeat sweep bonus within 15-min window)
    ±N% external composite     (composite_confidence_boost from config, default ±5%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHEN NOT TO TRADE (hard blocks applied before dispatch)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • First 5 min after open (09:30–09:35 ET) — MM positioning noise
    • High-impact economic event within ±30 min — event_risk flag active
    • SPY in tight intraday range (< 0.20% high-to-low) — chop, no edge
    • 0DTE near max pain — strong gamma pin risk
    • Directional signal inside ORB after 10:30 ET — range-bound, no trend

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE TRADE WALKTHROUGH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  9:52 AM ET (OPEN bucket):
    Volume spike: 7× rolling avg on 545C  →  vol_score = 0.30
    Bid/ask imbalance: 4.2× threshold     →  imb_score = 0.15
    Delta = 0.44 (ideal call range)       →  delta_score = 0.10
    Gamma = 0.025 (in range)              →  gamma_score = 0.10
    Theta = −0.05 (mild)                  →  theta_score = 0.10
    IV rank = 38% (neutral)               →  iv_score = 0.062
    Sentiment = +45 (bullish)             →  sent_score = 0.072
    OI = 8,500                            →  oi_score = 0.043
    Flow score = 0.62 (repeat sweeps)     →  flow_score = 0.031
    External composite = +0.35 (bullish)  →  ext_adj   = +0.018
    ────────────────────────────────────────
    base_confidence                       =   0.726

    Signal-specific adjustments:
      ORB confirmed ABOVE_ORB             →  +0.03
      VWAP: ABOVE_1SD (slight caution)    →  −0.02
      Regime: TREND_UP                    →  +0.05
    ────────────────────────────────────────
    after signal adjustments              =   0.786

    DynamicConfidence (OPEN bucket):
      time_of_day (pre-10:15, confirmed)  →  +0.037
      flow_alignment (STRONG +63)         →  +0.10
      macro (mild tailwind)               →  +0.05
      ORB confirmed above                 →  +0.05
    ────────────────────────────────────────
    final_confidence                      =   0.84 → HIGH tier → DISPATCHED ✓
    Signal: CALL_SWEEP 545C exp APR26 confidence=84%
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import os
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from ..config.spy_options import SpyOptionsSignalConfig
from ..utils.logger import logger
from .chain_builder import ChainSnapshot, OptionQuote, VolumeTracker
from .dynamic_confidence import DynamicConfidence, ConfidenceAdjustment
from .external import ExternalContext
from .regime_detector import RegimeContext
from .sentiment_engine import SentimentContext
from .sweep_tracker import SweepTracker


class SignalType(str, Enum):
    CALL_SWEEP         = "CALL_SWEEP"
    PUT_SWEEP          = "PUT_SWEEP"
    BULL_CALL_SPREAD   = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD    = "BEAR_PUT_SPREAD"
    LONG_STRADDLE      = "LONG_STRADDLE"
    HIGH_IV_ALERT      = "HIGH_IV_ALERT"
    PC_RATIO_EXTREME   = "PC_RATIO_EXTREME"
    PC_AFTERNOON_FLOW  = "PC_AFTERNOON_FLOW"   # pilot 2026-07-19: ≥14:00 ET extreme-P/C put flow-follow
    VWAP_REVERSION     = "VWAP_REVERSION"      # shadow-incubating 2026-07-19: 2SD+RSI-extreme fade (range engine)
    ORB_BREAKOUT       = "ORB_BREAKOUT"   # Opening Range Breakout — confirmed directional move
    TREND_CONTINUATION = "TREND_CONTINUATION"  # Pullback-to-anchor rejection in TREND_UP/DOWN (rules_v2)


@dataclass
class SignalContext:
    """Per-poll context passed from manager to signal engine."""

    regime: RegimeContext
    sentiment: SentimentContext
    iv_rank: float                              # 0-100 based on VIX 52w range
    vix: Optional[float]
    spy_price: float
    external: Optional[ExternalContext] = None  # External signals (news/macro/social)

    # True session high/low from today's 5-min bars (JUL 2 2026). The engine's
    # internal intraday tracker only sees prices AFTER process start, so after
    # a mid-session restart it under-states the day range (a -0.5% trend day
    # was blocked as "0.12% chop" on Jul 2 after 8 restarts). Bars are fetched
    # fresh each poll with 1-day duration, so these survive restarts.
    day_high: Optional[float] = None
    day_low: Optional[float] = None


@dataclass
class SpySignal:
    """A single SPY options signal ready for delivery and analytics."""

    signal_type: SignalType
    strike: float
    expiry: str
    right: str              # "C", "P", or "BOTH"
    confidence: float       # 0.0 – 1.0
    spy_price: float
    vix: Optional[float]
    volume: int
    volume_spike_mult: float
    bid_size: int
    ask_size: int
    reasoning: List[str] = field(default_factory=list)
    suggested_trade: str = ""
    expiry_date: str = ""   # YYYYMMDD (e.g. "20260417") — actual contract expiration

    # Rich fields (populated by engine)
    confidence_tier: str = "MEDIUM"  # "MEDIUM" | "HIGH" | "EXTREME"
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    impl_vol: float = 0.0
    open_interest: int = 0
    spread_pct: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    iv_rank: float = 0.0
    regime: str = "RANGE_BOUND"
    sentiment_score: float = 0.0
    sentiment_label: str = "NEUTRAL"
    flow_score: float = 0.0

    # DTE (days to expiry) — populated from expiry string
    dte: int = 0
    # DTE counted in TRADING sessions (weekends/NYSE holidays excluded).
    # Set by the manager before execution; the executor's max_dte gate prefers
    # this so a Thu→Mon contract (4 calendar / 2 trading days) isn't rejected
    # as "too far out" while a Tue→Fri (3/3) passes. None → calendar fallback.
    trading_dte: Optional[int] = None

    # External signal context (news/macro/social/flow)
    external_composite: float = 0.0   # -1.0 to +1.0
    event_risk: bool = False
    event_minutes: float = 999.0
    next_event_title: str = ""
    news_score: float = 0.0
    retail_score: float = 0.0
    macro_headwind: float = 0.0
    macro_label: str = "NEUTRAL"
    tnx_trend: str = "FLAT"
    dxy_trend: str = "FLAT"
    equity_pc: Optional[float] = None

    # Flow confirmation fields
    flow_confirmation_score: float = 0.0  # -100..+100
    dark_pool_bias: str = "NEUTRAL"
    gex_bias: str = "NEUTRAL"
    intraday_pc_ratio: Optional[float] = None

    # Cross-asset confirmation (live QQQ/IWM via IB)
    cross_asset_divergence: str = "NONE"  # BEARISH_NONCONFIRM / BULLISH_NONCONFIRM / NONE
    cross_asset_bias: str = "NEUTRAL"     # RISK_ON / RISK_OFF / MIXED / NEUTRAL
    qqq_rs: float = 0.0                   # QQQ vs SPY intraday, pct points

    # Dynamic confidence adjustment
    dynamic_confidence_delta: float = 0.0
    confidence_time_bucket: str = "MIDDAY"
    confidence_dte_rule: str = "STANDARD"
    conflict_detected: bool = False

    # ── Edge Reality fields (populated by manager._apply_edge_reality) ────
    # All zero/empty by default → safe for code paths that bypass the manager
    # (tests, backtests). Telegram formatter and analytics_db both treat
    # zero/empty as "not computed" and skip rendering / persisting.
    historical_wr: int = 0           # base (regime-neutral) pattern WR
    regime_wr: int = 0               # regime-adjusted WR for the current regime
    breakeven_wr: float = 0.0        # min WR after transaction costs
    edge_margin: float = 0.0         # regime_wr − breakeven_wr (signed)
    edge_color: str = ""             # "green" | "amber" | "red"
    round_trip_cost_pct: float = 0.0
    iv_adjusted_stop_pct: float = 0.0
    hourly_theta_dollars: float = 0.0
    gamma_accel_mult: float = 1.0
    effective_gamma: float = 0.0
    gamma_warning_active: bool = False
    skew_warning: str = ""
    target_pct_assumed: float = 0.0  # target used in breakeven calc
    # Source attribution — see edge_reality.WR_SOURCE_* / RT_SOURCE_* constants
    wr_source: str = ""              # "doc_prior" | "empirical"
    rt_source: str = ""              # "live_quote" | "default"

    # ── Structure tagging (MAY 19 2026) ───────────────────────────────────
    # Tells the Trading Manager how to size *actual* per-trade risk.
    # Without these the TM falls back to the legacy naked-option heuristic
    # (50% × long_mid × 100), which over-estimates risk for defined-risk
    # spreads and triggers spurious REJECTs in SUSPECT mode.
    #
    #   "LONG"             — naked long call/put (current behaviour)
    #   "BULL_CALL_SPREAD" — long lower call + short higher call (debit)
    #   "BEAR_PUT_SPREAD"  — long higher put + short lower put (debit)
    #   "LONG_STRADDLE"    — long ATM call + long ATM put
    #   ""                 — unspecified → TM treats as "LONG" for back-compat
    structure: str = ""
    short_strike: float = 0.0        # second leg of a spread; 0.0 for naked
    short_bid: float = 0.0
    short_ask: float = 0.0

    # Structural stop in SPY (underlying) terms — populated for entries that
    # carry a price-structure invalidation level (TREND_CONTINUATION: the far
    # side of the rejection bar). The executor derives the option premium
    # stop/target from this via delta, instead of a fixed premium %.
    structural_stop: float = 0.0     # SPY level; 0.0 = none (use fixed premium bracket)

    # Full ExternalContext snapshot at dispatch, for the trade research log
    # (features not already on this signal: VWAP bands, tape, depth, breadth,
    # sector, vol structure, gamma env, RSI, EDR, ORB, pivots, …). Not used for
    # trading — captured so closed trades carry the complete feature vector.
    research_ctx: dict = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        """Dedup key excludes expiry — same signal across APR/MAY is one signal.

        APR 10 2026: Previously included expiry, causing ORB_BREAKOUT and
        PC_RATIO signals to emit APR + MAY duplicates (22 dupes in 3 days).
        """
        return f"{self.signal_type}:{self.strike:.0f}:{self.right}"


def log_blocked_signal(sig: "SpySignal", gate: str, reason: str) -> None:
    """Opportunity-cost ledger: one JSONL line per pre-dispatch kill.

    Strategy-selection audit 2026-07-17: ~2,500 gate kills vs ~40 dispatches
    in 2 weeks, and the only family allowed to trade (TREND_CONTINUATION) has
    the WORST shadow win-rate of all families (33% vs ORB 75%). Statistical
    strategy selection is impossible without recording what the filters
    rejected — this file is the forward-looking counterfactual record.
    Best-effort: never raises, never blocks the signal path.
    """
    try:
        import json
        import os
        rec = {
            "ts": datetime.utcnow().isoformat(),
            "gate": gate,
            "reason": (reason or "")[:160],
            "signal_type": getattr(sig.signal_type, "value", str(sig.signal_type)),
            "right": sig.right,
            "strike": sig.strike,
            "expiry": sig.expiry,
            # JUL 27 2026 (audit fix): `expiry` is a MONTH CODE ("AUG26"), which
            # does not identify a contract — blocked signals were therefore
            # UNREPLAYABLE (0/508 on 2026-07-27) and gate effectiveness could
            # never be measured. Record the exact expiration date too, matching
            # the dispatched-signal column `spy_signals.expiry_date`.
            "expiry_date": getattr(sig, "expiry_date", "") or "",
            "confidence": round(float(sig.confidence or 0.0), 4),
            "tier": getattr(sig, "confidence_tier", ""),
            "regime": getattr(sig, "regime", ""),
            "spy_price": sig.spy_price,
            "dte": getattr(sig, "dte", None),
        }
        os.makedirs("logs", exist_ok=True)
        with open("logs/blocked_signals.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _tier(confidence: float, high: float = 0.80, extreme: float = 0.90) -> str:
    """Return the confidence tier label for a given confidence score.

    Thresholds default to the dataclass values (0.80 HIGH / 0.90 EXTREME) so
    the function can still be called without a config reference.  The
    ``SignalEngine`` instance method ``_tier_cfg`` calls this with the actual
    config thresholds so runtime signals respect user-configured values.
    """
    if confidence >= extreme:
        return "EXTREME"
    if confidence >= high:
        return "HIGH"
    return "MEDIUM"


class SignalEngine:
    """Evaluates ChainSnapshot and emits SpySignal objects."""

    def __init__(
        self,
        cfg: SpyOptionsSignalConfig,
        tracker: VolumeTracker,
        composite_confidence_boost: float = 0.05,
    ) -> None:
        self._cfg = cfg
        self._tracker = tracker
        self._dyn = DynamicConfidence()
        # Maximum ±adjustment from external composite score (read from config,
        # default 0.05 matches prior hardcoded value).
        self._composite_boost = composite_confidence_boost
        # Stale-flow decay tracking for PC_RATIO_EXTREME signals.
        # Key: direction ("BEARISH" | "BULLISH"), Value: (first_seen_mono, spy_price_at_trigger)
        self._pc_ratio_first_seen: Dict[str, Tuple[float, float]] = {}
        # Intraday move exhaustion tracking: record day's high/low for
        # detecting when the move has already happened.
        self._intraday_high: Optional[float] = None
        self._intraday_low: Optional[float] = None
        self._intraday_date: Optional[str] = None

    def _tier_cfg(self, confidence: float) -> str:
        """Confidence tier using thresholds from SpyOptionsSignalConfig.

        Uses ``cfg.confidence_tier_high`` and ``cfg.confidence_tier_extreme``
        so the tier boundaries are driven by config rather than hardcoded
        constants.  This replaces module-level ``_tier()`` for all internal
        signal construction.
        """
        return _tier(
            confidence,
            high=self._cfg.confidence_tier_high,
            extreme=self._cfg.confidence_tier_extreme,
        )

    # ── Quality gate (hard blocks — confidence cannot override) ───────────────

    def _quality_gate(
        self,
        sig: "SpySignal",
        ext: Optional["ExternalContext"],
    ) -> Tuple[bool, List[str]]:
        """Hard trade-quality filter applied after confidence scoring.

        Returns (passes, [rejection_reasons]).  A signal that fails ANY
        check is dropped regardless of how high its confidence is.

        Checks (in priority order):
          1. Flow strongly opposes direction (Tier-1 block)
          2. 0DTE AND near max pain (gamma-pin risk)
          3. 0DTE missing required ORB/flow confirmation
          4. Directional inside ORB after 10:30 ET (range-bound)
          5. SPY tight intraday range < 0.20% (chop day — no edge)
        """
        fails: List[str] = []
        is_directional = sig.signal_type in {
            SignalType.CALL_SWEEP, SignalType.PUT_SWEEP,
            SignalType.BULL_CALL_SPREAD, SignalType.BEAR_PUT_SPREAD,
            SignalType.PC_RATIO_EXTREME, SignalType.ORB_BREAKOUT,
        }

        # ── 1. Flow strongly opposes direction (Tier-1 hard block) ────────
        if ext is not None and is_directional:
            fs = ext.flow_score  # -100..+100
            if sig.right == "C" and fs <= -30:
                fails.append(
                    f"Quality gate: flow strongly BEARISH ({fs:.0f}) opposes CALL — blocked"
                )
            elif sig.right == "P" and fs >= 30:
                fails.append(
                    f"Quality gate: flow strongly BULLISH ({fs:.0f}) opposes PUT — blocked"
                )

        # ── 2. 0DTE near max pain — HARD BLOCK REMOVED (JUL 6 2026) ─────────
        # Live evidence (Jul 6): on a TREND_UP day SPY walked from $749→$752
        # with max pain tracking price the whole way, so this distance-based
        # hard block vetoed 215 call signals across a clean 3.5-hour trend.
        # Pin risk is now handled by the −6% confidence penalty in
        # DynamicConfidence block 18 (discourages pin trades without a hard
        # veto), so genuine trends near a walking pin can still trade.

        # ── 3. 0DTE confirmation — ADAPTIVE (JUL 7 2026) ───────────────────
        # OLD behaviour: hard-require |external flow| ≥ 25 OR an aligned ORB.
        # This blocked today's clean morning short (TREND_DOWN + RISK_OFF + QQQ
        # leading down, but slow composite flow only −12 to −21). The slow
        # external composite was vetoing a trade every FAST signal confirmed.
        #
        # NEW behaviour: the flow requirement scales down with trend conviction
        # from the fast signals (regime, cross-asset, tape). Strong multi-factor
        # agreement waives the flow bar entirely; weak/mixed keeps the full ±25.
        # Real-time tape ABSORPTION against the signal (buying into a downtrend)
        # subtracts conviction — so a good morning breakdown passes while a late
        # afternoon chase into absorption still gets held to the full bar.
        # Flow still VETOES strong opposition via check #1 above.
        if sig.dte == 0 and is_directional and ext is not None:
            orb_confirmed = getattr(ext, "orb_breakout_confirmed", False)
            orb_status = getattr(ext, "orb_status", "INSIDE")
            orb_aligned = orb_confirmed and (
                (sig.right == "C" and orb_status == "ABOVE_ORB") or
                (sig.right == "P" and orb_status == "BELOW_ORB")
            )

            required_flow, conviction = self._adaptive_flow_requirement(sig, ext)
            flow_ok = abs(ext.flow_score) >= required_flow

            if not flow_ok and not orb_aligned:
                fails.append(
                    f"Quality gate: 0DTE flow {ext.flow_score:+.0f} < adaptive req "
                    f"±{required_flow:.0f} (conviction={conviction:+d}) and no aligned ORB — blocked"
                )
            elif required_flow < 25 and not orb_aligned:
                # Passed only because fast-signal conviction lowered the bar — log it.
                logger.info(
                    "Adaptive flow PASS: {} {}{} flow={:+.0f} req±{:.0f} conviction={:+d} "
                    "(regime/cross-asset/tape confirm)",
                    sig.signal_type.value, sig.strike, sig.right,
                    ext.flow_score, required_flow, conviction,
                )

        # ── 4. Directional inside ORB after 10:30 ET ──────────────────────
        # NOTE: This was previously a hard block but caused 91/106 quality-gate
        # blocks (85% of all blocks) — SPY spends most of normal trading days
        # inside the ORB, effectively shutting down the bot for the majority of
        # the session in low-IV environments.  Converted to a -0.10 confidence
        # penalty applied later in the pipeline (see _orb_inside_penalty below).
        # A signal with genuine strength can still exceed the threshold after this
        # penalty; a marginal signal will be filtered at the threshold check.

        # ── 5. Tight intraday range < 0.20% (chop day — no directional edge) ─
        # Requires a REAL measured range (high > low). A single-point seed
        # (first poll, no bars yet) carries no range information — skipping
        # beats false-blocking everything at 0.00% (JUL 2 2026).
        if (
            is_directional
            and self._intraday_high and self._intraday_low
            and self._intraday_low > 0
            and self._intraday_high > self._intraday_low
        ):
            range_pct = (self._intraday_high - self._intraday_low) / self._intraday_low * 100
            if range_pct < 0.20:
                fails.append(
                    f"Quality gate: SPY intraday range only {range_pct:.2f}% "
                    f"(${self._intraday_low:.2f}–${self._intraday_high:.2f}) — chop day, blocked"
                )

        return len(fails) == 0, fails

    @staticmethod
    def _adaptive_flow_requirement(
        sig: "SpySignal", ext: "ExternalContext"
    ) -> Tuple[float, int]:
        """Scale the 0DTE flow-confirmation bar by fast-signal trend conviction.

        Conviction counts how many *fast* signals confirm the trade direction:
          +1 regime aligned (TREND_DOWN for puts / TREND_UP for calls)
          +1 cross-asset risk bias aligned (RISK_OFF puts / RISK_ON calls)
          +1 QQQ leading the move (trend + relative strength ≥ 0.15pp)
          +1 tape aggression WITH the direction (real-time buyers/sellers)
          −1 tape ABSORBING against the direction (buying into a downtrend →
             late-move exhaustion; the tell that separates a chase from an entry)
          −2 cross-asset non-confirmation against the direction (SPY made an
             extreme QQQ didn't confirm)

        Required |flow|:
          conviction ≥ 3 → 0   (fast signals unanimous — waive the slow composite)
          conviction = 2 → 10  (mostly aligned — light confirmation)
          conviction ≤ 1 → 25  (weak/mixed — full confirmation, as before)

        Returns (required_flow, conviction).
        """
        right = sig.right
        regime = getattr(sig, "regime", "RANGE_BOUND")
        bias = getattr(ext, "cross_asset_bias", "NEUTRAL")
        qqq_trend = getattr(ext, "qqq_trend", "FLAT")
        qqq_rs = getattr(ext, "qqq_rs", 0.0)
        tape_ok = getattr(ext, "tape_available", False)
        tape = getattr(ext, "tape_score", 0.0)
        divergence = getattr(ext, "cross_asset_divergence", "NONE")

        conviction = 0
        if (right == "P" and regime == "TREND_DOWN") or (right == "C" and regime == "TREND_UP"):
            conviction += 1
        if (right == "P" and bias == "RISK_OFF") or (right == "C" and bias == "RISK_ON"):
            conviction += 1
        if (right == "P" and qqq_trend == "DOWN" and qqq_rs <= -0.15) or \
           (right == "C" and qqq_trend == "UP" and qqq_rs >= 0.15):
            conviction += 1
        if tape_ok:
            tape_with = (right == "P" and tape <= -20) or (right == "C" and tape >= 20)
            tape_against = (right == "P" and tape >= 20) or (right == "C" and tape <= -20)
            if tape_with:
                conviction += 1
            elif tape_against:
                conviction -= 1   # absorption against us — hold the full bar
        if (divergence == "BEARISH_NONCONFIRM" and right == "C") or \
           (divergence == "BULLISH_NONCONFIRM" and right == "P"):
            conviction -= 2

        if conviction >= 3:
            required = 0.0
        elif conviction == 2:
            required = 10.0
        else:
            required = 25.0
        return required, conviction

    # ── ORB-inside confidence penalty (soft, replaces former hard block) ─────

    @staticmethod
    def _orb_inside_penalty(
        sig: "SpySignal",
        ext: Optional["ExternalContext"],
    ) -> Tuple[float, str]:
        """Return a confidence penalty when price is inside the ORB after 10:30 ET.

        Replaces the former hard quality-gate block which killed 91/106 signals
        (85% of all blocks) in low-volatility environments.  Being inside the ORB
        is a genuine risk factor — directional signals there are lower probability
        — but not a categorical veto.  A strong signal (high flow, clear breadth,
        9x+ spike) can legitimately fire inside the ORB if confidence is high
        enough to absorb this -0.10 penalty and still clear the threshold.

        Returns (penalty, note_str).  penalty is 0.0 or negative.
        """
        is_directional = sig.signal_type in {
            SignalType.CALL_SWEEP, SignalType.PUT_SWEEP,
            SignalType.BULL_CALL_SPREAD, SignalType.BEAR_PUT_SPREAD,
            SignalType.PC_RATIO_EXTREME, SignalType.ORB_BREAKOUT,
        }
        if not is_directional or ext is None:
            return 0.0, ""

        _ET = ZoneInfo("America/New_York")
        now_et = _dt.datetime.now(_ET)
        after_orb_lock = now_et.time() >= _dt.time(10, 30)
        orb_established = getattr(ext, "orb_established", False)
        orb_status = getattr(ext, "orb_status", "BUILDING")

        if after_orb_lock and orb_established and orb_status == "INSIDE":
            return -0.10, "ORB inside after 10:30 ET — range-bound penalty (−10%)"

        return 0.0, ""

    # ── Priority conflict check (Tier-1 / 2 / 3 hierarchy) ───────────────────

    def _priority_conflict_check(
        self,
        sig: "SpySignal",
        ext: Optional["ExternalContext"],
        regime: str,
    ) -> Tuple[float, List[str]]:
        """Apply Tier-1/2/3 priority hierarchy and return a confidence delta.

        Tier 1 (flow + ORB) has authority over all other factors.  If 2 Tier-1
        signals oppose a trade direction the confidence is reduced heavily — the
        signal can still survive but it must already be extremely high.

        Returns (delta, [notes_for_reasoning]).
        """
        if ext is None:
            return 0.0, []

        direction: str
        if sig.right == "C":
            direction = "BULLISH"
        elif sig.right == "P":
            direction = "BEARISH"
        else:
            return 0.0, []  # straddle — no directional priority

        notes: List[str] = []
        tier1_confirms = 0
        tier1_conflicts = 0

        # ── Tier 1a: Options flow ─────────────────────────────────────────
        fs = ext.flow_score
        if direction == "BULLISH":
            if fs >= 30:
                tier1_confirms += 1
            elif fs <= -30:
                tier1_conflicts += 1
                notes.append(f"Tier-1 conflict: flow BEARISH ({fs:.0f}) vs CALL")
        else:  # BEARISH
            if fs <= -30:
                tier1_confirms += 1
            elif fs >= 30:
                tier1_conflicts += 1
                notes.append(f"Tier-1 conflict: flow BULLISH ({fs:.0f}) vs PUT")

        # ── Tier 1b: ORB breakout ─────────────────────────────────────────
        orb_confirmed = getattr(ext, "orb_breakout_confirmed", False)
        if orb_confirmed:
            orb_status = getattr(ext, "orb_status", "INSIDE")
            if direction == "BULLISH":
                if orb_status == "ABOVE_ORB":
                    tier1_confirms += 1
                elif orb_status == "BELOW_ORB":
                    tier1_conflicts += 1
                    notes.append("Tier-1 conflict: ORB breakdown vs CALL")
            else:
                if orb_status == "BELOW_ORB":
                    tier1_confirms += 1
                elif orb_status == "ABOVE_ORB":
                    tier1_conflicts += 1
                    notes.append("Tier-1 conflict: ORB breakout vs PUT")

        # ── Apply Tier-1 priority ruling ─────────────────────────────────
        delta = 0.0
        if tier1_conflicts >= 2:
            delta = -0.15
            notes.append("Priority ruling: 2 Tier-1 factors oppose direction → −15%")
        elif tier1_conflicts == 1 and tier1_confirms == 0:
            delta = -0.08
            notes.append("Priority ruling: 1 Tier-1 factor opposes with no Tier-1 support → −8%")
        elif tier1_confirms >= 2:
            delta = 0.05
            notes.append("Priority ruling: 2 Tier-1 factors confirm direction → +5%")
        elif tier1_confirms == 1 and tier1_conflicts == 0:
            delta = 0.02
            notes.append("Priority ruling: 1 Tier-1 factor confirms direction → +2%")

        # ── Tier 2: Regime conflict (caps, not full blocks) ───────────────
        if tier1_conflicts == 0:  # only apply if Tier-1 already didn't penalise
            if direction == "BULLISH" and regime == "TREND_DOWN":
                delta -= 0.05
                notes.append("Tier-2: regime TREND_DOWN conflicts with CALL → −5%")
            elif direction == "BEARISH" and regime == "TREND_UP":
                delta -= 0.05
                notes.append("Tier-2: regime TREND_UP conflicts with PUT → −5%")

        return round(delta, 4), notes

    def _stale_flow_decay(self, direction: str, spy_price: float) -> float:
        """Return a negative confidence adjustment if extreme P/C flow has been
        present for multiple cycles but price hasn't followed through.

        SPY options flow loses value quickly intraday — if unusual put/call
        activity occurred 30-45+ minutes ago and price hasn't moved >0.15%
        in the signal direction, the flow is likely hedging or noise.

        Returns 0.0 if signal is fresh, negative value if stale.
        """
        now = _time.monotonic()
        entry = self._pc_ratio_first_seen.get(direction)
        if entry is None:
            # First time seeing this direction — record and return no decay
            self._pc_ratio_first_seen[direction] = (now, spy_price)
            return 0.0

        first_seen, trigger_price = entry
        elapsed_min = (now - first_seen) / 60.0

        if elapsed_min < 15:
            return 0.0  # too early to apply decay

        # Check if price followed through (>0.15% in signal direction)
        pct_move = (spy_price - trigger_price) / trigger_price * 100
        if direction == "BEARISH" and pct_move <= -0.15:
            # Price confirmed the bearish thesis — reset tracker, no decay
            self._pc_ratio_first_seen[direction] = (now, spy_price)
            return 0.0
        if direction == "BULLISH" and pct_move >= 0.15:
            self._pc_ratio_first_seen[direction] = (now, spy_price)
            return 0.0

        # Price did NOT follow through — apply nonlinear decay.
        # Once a flow signal is 60+ min old without confirmation, it is
        # often actively wrong rather than merely stale.
        if elapsed_min >= 60:
            return -0.10   # 60+ min: likely wrong, not just stale
        if elapsed_min >= 45:
            return -0.07   # 45-60 min: heavy penalty
        if elapsed_min >= 30:
            return -0.04   # 30-45 min: moderate penalty
        return -0.02       # 15-30 min: mild penalty

    def _sync_intraday_range(self, context: "SignalContext") -> None:
        """Merge bar-derived session high/low into the intraday tracker.

        The tracker previously seeded from the first spy_price seen after
        process start, so any mid-session restart erased the morning range —
        breaking BOTH the chop-day gate (range under-stated → real trend days
        blocked as chop) and the move-exhaustion penalty (drop-from-high
        under-stated → late chasers not penalized). Bars carry the full
        session, so take the max/min of both sources.
        """
        today_str = _dt.date.today().isoformat()
        if self._intraday_date != today_str:
            self._intraday_high = None
            self._intraday_low = None
            self._intraday_date = today_str

        px = context.spy_price
        hi_candidates = [v for v in (self._intraday_high, context.day_high, px) if v]
        lo_candidates = [v for v in (self._intraday_low, context.day_low, px) if v]
        if hi_candidates:
            self._intraday_high = max(hi_candidates)
        if lo_candidates:
            self._intraday_low = min(lo_candidates)

    def _intraday_move_penalty(self, direction: str, spy_price: float) -> Tuple[float, str]:
        """Penalize continuation signals when the intraday move is already exhausted.

        If SPY has already moved significantly from today's high (for PUTs) or
        today's low (for CALLs), the probability of profitable continuation
        drops sharply.  This was the #1 root cause of high-confidence PUT
        signals losing money: signals fired AFTER SPY had already dropped 0.3%+.

        Returns (penalty, note_str).  penalty is 0.0 or negative.
        """
        today_str = _dt.date.today().isoformat()
        if self._intraday_date != today_str:
            # New day — reset
            self._intraday_high = spy_price
            self._intraday_low = spy_price
            self._intraday_date = today_str
        else:
            if spy_price > (self._intraday_high or spy_price):
                self._intraday_high = spy_price
            if spy_price < (self._intraday_low or spy_price):
                self._intraday_low = spy_price

        if direction == "BEARISH":
            # For puts: how far has SPY already dropped from today's high?
            day_high = self._intraday_high or spy_price
            if day_high <= 0:
                return 0.0, ""
            drop_pct = (day_high - spy_price) / day_high * 100
            if drop_pct >= 0.50:
                return -0.12, f"Intraday move exhaustion: SPY already -{drop_pct:.2f}% from day high"
            if drop_pct >= 0.30:
                return -0.08, f"Intraday move extended: SPY already -{drop_pct:.2f}% from day high"
            if drop_pct >= 0.20:
                return -0.04, f"Intraday move in progress: SPY -{drop_pct:.2f}% from day high"
        else:
            # For calls: how far has SPY already rallied from today's low?
            day_low = self._intraday_low or spy_price
            if day_low <= 0:
                return 0.0, ""
            rally_pct = (spy_price - day_low) / day_low * 100
            if rally_pct >= 0.50:
                return -0.12, f"Intraday move exhaustion: SPY already +{rally_pct:.2f}% from day low"
            if rally_pct >= 0.30:
                return -0.08, f"Intraday move extended: SPY already +{rally_pct:.2f}% from day low"
            if rally_pct >= 0.20:
                return -0.04, f"Intraday move in progress: SPY +{rally_pct:.2f}% from day low"

        return 0.0, ""

    @staticmethod
    def _rsi_divergence_penalty(direction: str, ext: Optional["ExternalContext"]) -> Tuple[float, str]:
        """Hard penalty when RSI divergence contradicts the signal direction.

        This is the #2 root cause: RSI at 30 with BULLISH_DIV while PUT signals
        fire at 88% confidence.  The dynamic_confidence adjuster only applied
        -0.05 which was insufficient to prevent dispatch.

        Returns (penalty, note_str).  penalty is 0.0 or negative.
        """
        if ext is None:
            return 0.0, ""
        rsi_div = getattr(ext, "rsi_divergence", "NONE")
        rsi_5m = getattr(ext, "rsi_5m", 50.0)
        rsi_os = getattr(ext, "rsi_oversold", False)
        rsi_ob = getattr(ext, "rsi_overbought", False)

        if direction == "BEARISH":
            # BULLISH_DIV + oversold RSI → high reversal probability, penalize puts
            if rsi_div == "BULLISH_DIV":
                if rsi_5m <= 30:
                    return -0.12, f"RSI divergence block: BULLISH_DIV at RSI={rsi_5m:.0f} (strong reversal signal)"
                if rsi_5m <= 35:
                    return -0.08, f"RSI divergence penalty: BULLISH_DIV at RSI={rsi_5m:.0f}"
                return -0.05, f"RSI divergence: BULLISH_DIV at RSI={rsi_5m:.0f}"
            # No divergence but oversold → mild penalty (bounce risk)
            if rsi_os and rsi_5m <= 30:
                return -0.04, f"RSI oversold ({rsi_5m:.0f}) — bounce risk"
        else:
            # BEARISH_DIV + overbought RSI → high reversal probability, penalize calls
            if rsi_div == "BEARISH_DIV":
                if rsi_5m >= 70:
                    return -0.12, f"RSI divergence block: BEARISH_DIV at RSI={rsi_5m:.0f} (strong reversal signal)"
                if rsi_5m >= 65:
                    return -0.08, f"RSI divergence penalty: BEARISH_DIV at RSI={rsi_5m:.0f}"
                return -0.05, f"RSI divergence: BEARISH_DIV at RSI={rsi_5m:.0f}"
            # No divergence but overbought → mild penalty (pullback risk)
            if rsi_ob and rsi_5m >= 70:
                return -0.04, f"RSI overbought ({rsi_5m:.0f}) — pullback risk"

        return 0.0, ""

    @staticmethod
    def _tod_ceiling() -> Tuple[float, str]:
        """Return (max_confidence, bucket_label) based on time of day (ET).

        PC_RATIO_EXTREME signals are unreliable at certain times of day:
        - Pre-open / first 30 min: market-maker positioning skews flow
        - Lunch hour: low volume → noisy ratios
        - Power hour: closing hedges mimic directional flow

        Buckets (Eastern Time):
            09:30-10:00 → 0.82  (OPEN — market-maker noise)
            10:00-11:30 → 0.95  (PRIME — highest accuracy)
            11:30-13:30 → 0.80  (LUNCH — thin volume)
            13:30-15:00 → 0.88  (AFTERNOON — moderate)
            15:00-16:00 → 0.93  (CLOSE — hedging noise, still decent)
            else        → 0.75  (OFF-HOURS — pre/post market)
        """
        _ET = ZoneInfo("America/New_York")
        now_et = _dt.datetime.now(_ET).time()
        h, m = now_et.hour, now_et.minute
        mins = h * 60 + m

        if mins < 570:        # before 09:30
            return 0.75, "PRE_MARKET"
        if mins < 600:        # 09:30-10:00
            return 0.82, "OPEN"
        if mins < 690:        # 10:00-11:30
            return 0.95, "PRIME"
        if mins < 810:        # 11:30-13:30
            return 0.80, "LUNCH"
        if mins < 900:        # 13:30-15:00
            return 0.88, "AFTERNOON"
        if mins < 960:        # 15:00-16:00
            return 0.93, "CLOSE"
        return 0.75, "POST_MARKET"   # after 16:00

    def evaluate(
        self,
        chain: ChainSnapshot,
        context: SignalContext,
        sweep_tracker: SweepTracker,
    ) -> List[SpySignal]:
        """Run all signal rules against the chain snapshot.

        Side channel: ``self.last_blocked`` — (sig, gate) tuples for signals
        killed by the quality gate / confidence threshold this cycle, so the
        manager can register them in the shadow book (strategy audit
        2026-07-17: blocked signals get simulated outcomes too, or per-family
        expectancy is unmeasurable).

        Args:
            chain:         Current option chain data for one expiry.
            context:       Per-poll regime, sentiment, IV rank, VIX, SPY price.
            sweep_tracker: Shared repeat-sweep detector.

        Returns:
            List of signals passing minimum confidence threshold.
        """
        c = self._cfg
        signals: List[SpySignal] = []
        self.last_blocked: List[tuple] = []   # (sig, gate) — shadow-book feed

        # Sync the intraday high/low tracker with the TRUE session range from
        # bars before any gate reads it (restart-proofing — see SignalContext).
        self._sync_intraday_range(context)

        iv_low = context.iv_rank < 30          # Low IV → debit spreads cheap
        iv_high = context.iv_rank > 70         # High IV → premium selling

        # ── Rule 1: Chain-level P/C ratio ─────────────────────────────────────
        signals.extend(self._pc_ratio_signal(chain, context, c))

        # ── Rules 2-5: Per-strike volume spike signals ─────────────────────────
        spiked_call_strikes: Set[float] = set()
        spiked_put_strikes: Set[float] = set()

        for quote, right in (
            [(q, "C") for q in chain.calls] + [(q, "P") for q in chain.puts]
        ):
            if quote.volume < c.min_volume_for_signal:
                self._tracker.update(quote.conid, quote.volume)
                continue

            increment = self._tracker.update(quote.conid, quote.volume)
            avg = self._tracker.rolling_avg(quote.conid)
            spike_mult = (increment / avg) if avg > 1 else 0.0
            is_spike = (
                spike_mult >= c.volume_spike_mult
                and increment >= c.sweep_poll_volume_threshold
            )
            if not is_spike:
                continue

            # Record in sweep tracker before calculating flow score
            sweep_tracker.record(quote.strike, right, chain.expiry_month)
            fscore = sweep_tracker.flow_score(quote.strike, right, chain.expiry_month)

            if right == "C":
                spiked_call_strikes.add(quote.strike)
                signals.extend(
                    self._call_spike_signals(quote, chain.expiry_month, spike_mult, fscore, iv_low, context, c)
                )
            else:
                spiked_put_strikes.add(quote.strike)
                signals.extend(
                    self._put_spike_signals(quote, chain.expiry_month, spike_mult, fscore, iv_low, context, c)
                )

        # ── Rule 6: Straddle ──────────────────────────────────────────────────
        signals.extend(
            self._straddle_signals(spiked_call_strikes, spiked_put_strikes, chain, context, c)
        )

        # ── Rule 7: High IV alert ─────────────────────────────────────────────
        if iv_high or (context.vix is not None and context.vix > c.vix_high):
            signals.extend(self._high_iv_signal(chain, context, c))

        # ── Rule 9: PC_AFTERNOON_FLOW pilot (2026-07-19) ──────────────────────
        # Log-mined backtest (9 sessions, 2,331 poll prints, ±0.5% barrier walk):
        # ≥14:00 ET chain-P/C extremes resolved 7W/1L/2S for PUTS across 6
        # distinct days (+0.34% avg favorable); MIDDAY same trigger = no edge
        # (rejected). Flow-follow into the close. 1x/day; executor caps size
        # at 1 contract and auto-kills the family on rolling negative EV.
        signals.extend(self._pc_afternoon_flow(chain, context, c))

        # ── Rule 10: VWAP_REVERSION range engine — SHADOW-INCUBATING ─────────
        # Deliberately sub-threshold confidence: never dispatches, always lands
        # in the shadow book with simulated exits. Promotion = scorecard math.
        signals.extend(self._vwap_reversion(chain, context, c))

        # ── Rule 8: Opening Range Breakout ────────────────────────────────────
        # Fires once the 30-min ORB is established and price has confirmed a
        # break above (call) or below (put) the range for ≥1 bar.
        # Only fires on the first expiry evaluated to avoid duplicates.
        ext = context.external
        if ext is not None and getattr(ext, "orb_established", False):
            orb_status    = getattr(ext, "orb_status", "INSIDE")
            orb_confirmed = getattr(ext, "orb_breakout_confirmed", False)
            if orb_confirmed and orb_status in ("ABOVE_ORB", "BELOW_ORB"):
                signals.extend(self._orb_breakout_signal(chain, context, c))

        # ── Intraday move exhaustion — sweep signals ───────────────────────────
        # CALL_SWEEP / PUT_SWEEP signals skip the PC_RATIO confidence pipeline,
        # so _intraday_move_penalty was never applied to them.  Root cause of
        # Signal 124 (Apr 15): 700C CALL_SWEEP entered at SPY 700.03 (the intraday
        # high) after a +0.72% rally — no exhaustion check blocked it.  Apply the
        # same penalty here, before DynamicConfidence, so exhausted moves reduce
        # confidence below the dispatch threshold.
        for sig in signals:
            if sig.signal_type in {
                SignalType.CALL_SWEEP, SignalType.PUT_SWEEP,
                SignalType.BULL_CALL_SPREAD, SignalType.BEAR_PUT_SPREAD,
            }:
                _ex_direction = "BULLISH" if sig.right == "C" else "BEARISH"
                _ex_adj, _ex_note = self._intraday_move_penalty(
                    _ex_direction, context.spy_price
                )
                if _ex_adj < 0:
                    _pre_conf = sig.confidence
                    sig.confidence = max(0.0, sig.confidence + _ex_adj)
                    sig.confidence_tier = self._tier_cfg(sig.confidence)
                    if _ex_note:
                        sig.reasoning.append(f"⚠ {_ex_note}")
                        logger.info(
                            "Move-exhaustion penalty on {}: {} conf {:.0f}%→{:.0f}%",
                            sig.signal_type.value, _ex_note,
                            _pre_conf * 100, sig.confidence * 100,
                        )

        # ── Event risk modifier ────────────────────────────────────────────────
        ext = context.external
        if ext is not None and ext.event_risk:
            for sig in signals:
                directional = sig.signal_type in {
                    SignalType.CALL_SWEEP, SignalType.PUT_SWEEP,
                    SignalType.BULL_CALL_SPREAD, SignalType.BEAR_PUT_SPREAD,
                    SignalType.PC_RATIO_EXTREME, SignalType.ORB_BREAKOUT,
                }
                if directional:
                    # 30% confidence penalty within event window
                    sig.confidence = max(0.0, sig.confidence * 0.70)
                    sig.confidence_tier = self._tier_cfg(sig.confidence)
                    sig.reasoning.append(
                        f"⚠ Event risk: {ext.next_event_title} in {ext.event_minutes:.0f} min — confidence penalised"
                    )
                elif sig.signal_type == SignalType.LONG_STRADDLE:
                    # Boost straddle during event window (big move expected)
                    sig.confidence = min(1.0, sig.confidence * 1.10)
                    sig.confidence_tier = self._tier_cfg(sig.confidence)
                    sig.reasoning.append(
                        f"📅 Event catalyst: {ext.next_event_title} in {ext.event_minutes:.0f} min — straddle boosted"
                    )

        # Populate expiry_date and DTE on all signals from chain metadata
        for sig in signals:
            sig.expiry_date = chain.expiry_date
            if chain.expiry_date:
                try:
                    exp_dt = _dt.datetime.strptime(chain.expiry_date, "%Y%m%d").date()
                    sig.dte = max(0, (exp_dt - _dt.date.today()).days)
                except (ValueError, Exception):
                    pass

        # ── Dynamic confidence adjustment ─────────────────────────────────────
        # Applied after event-risk modifier; updates confidence, tier, and stores
        # the adjustment breakdown on the signal for logging / analytics.
        for sig in signals:
            adj = self._dyn.adjust(
                base=sig.confidence,
                right=sig.right,
                dte=sig.dte,
                ext_ctx=context.external,
                ib_sentiment_score=context.sentiment.score,
                vix=context.vix,
                regime=context.regime.regime,
            )
            sig.confidence = adj.final
            sig.confidence_tier = self._tier_cfg(adj.final)
            sig.dynamic_confidence_delta = adj.delta

            # ── TEMP DIAGNOSTIC: ORB confidence decomposition ──────────────────
            # Flow-causality verification (Phase 2). Emits the exact reason each
            # ORB candidate passes/fails the threshold: base, flow contribution,
            # and each continuation-vs-fade penalty. Gated by env so it can be
            # silenced; REMOVE this block once causality is established.
            if (
                sig.signal_type == SignalType.ORB_BREAKOUT
                and os.getenv("SPY_ORB_DIAG", "1") == "1"
            ):
                bd = adj.breakdown
                _flow = bd.get("flow_alignment", 0.0) + bd.get("dte", 0.0)
                _vwap = bd.get("vwap_band", 0.0)
                _rsi = bd.get("rsi", 0.0)
                _edr = bd.get("edr_exhaustion", 0.0)
                _orb = bd.get("orb", 0.0)
                _other = adj.delta - (_flow + _vwap + _rsi + _edr + _orb)
                logger.info(
                    "ORB-DIAG {} {:.0f}{} | base={:.0f}% flow={:+.0f} orb={:+.0f} "
                    "vwap={:+.0f} rsi={:+.0f} edr={:+.0f} other={:+.0f} "
                    "→ final={:.0f}% need={:.0f}% {} | flow_score={:+.0f}",
                    sig.expiry, sig.strike, sig.right,
                    bd.get("base", 0.0) * 100,
                    _flow * 100, _orb * 100, _vwap * 100, _rsi * 100,
                    _edr * 100, _other * 100,
                    adj.final * 100, c.min_confidence * 100,
                    "PASS" if adj.final >= c.min_confidence else "FAIL",
                    getattr(context.external, "flow_score", 0.0)
                    if context.external else 0.0,
                )

            sig.confidence_time_bucket = adj.time_bucket
            sig.confidence_dte_rule = adj.dte_rule
            sig.conflict_detected = adj.conflict_detected
            if adj.conflict_detected:
                sig.reasoning.append(
                    f"⚡ Conflict: flow + sentiment + macro oppose signal direction "
                    f"({adj.flow_factor}) — confidence adjusted {adj.delta:+.0%}"
                )
            elif abs(adj.delta) >= 0.05:
                sig.reasoning.append(
                    f"📊 Dynamic adj: {adj.delta:+.0%} "
                    f"[{adj.time_bucket} | {adj.dte_rule} | {adj.flow_factor}]"
                )

        # ── Opening period caution (all directional signals) ──────────────
        # First 30 min of RTH (9:30-10:00 ET) is dominated by market-maker
        # positioning and overnight order flow unwinding.  Directional signals
        # in this window have historically been the most unreliable — e.g.
        # Apr 9 fired 2 PUT signals at 0.95 at open, then SPY rallied +6pt.
        # Cap ALL directional signals at 0.82 during the opening period.
        _ET = ZoneInfo("America/New_York")
        _now_et = _dt.datetime.now(_ET)
        _now_mins = _now_et.hour * 60 + _now_et.minute
        _OPEN_START, _OPEN_END = 570, 600   # 09:30 - 10:00 ET
        if _OPEN_START <= _now_mins < _OPEN_END:
            _OPEN_CAP = 0.82
            for sig in signals:
                if sig.right in ("C", "P") and sig.confidence > _OPEN_CAP:
                    sig.reasoning.append(
                        f"⏰ Opening period cap: {sig.confidence:.2f}→{_OPEN_CAP:.2f} "
                        f"(first 30 min — MM positioning noise)"
                    )
                    sig.confidence = _OPEN_CAP
                    sig.confidence_tier = self._tier_cfg(sig.confidence)

        # ── Priority conflict check (Tier-1/2/3 hierarchy) ────────────────
        # Applied AFTER dynamic confidence; this adjusts for Tier-1 conflicts
        # (flow + ORB) that the additive-adjustment model can under-penalise.
        ext = context.external
        for sig in signals:
            delta, priority_notes = self._priority_conflict_check(
                sig, ext, context.regime.regime,
            )
            if delta != 0.0:
                sig.confidence = min(0.95, max(0.0, sig.confidence + delta))
                sig.confidence_tier = self._tier_cfg(sig.confidence)
                for note in priority_notes:
                    sig.reasoning.append(f"🔺 {note}")

        # ── PC_RATIO_EXTREME PUT: hard block in confirmed bullish environment ───
        # Signal 115 (Apr 15): PC_RATIO fired a PUT at 09:58 ET when SPY regime was
        # TREND_UP and price was well above VWAP.  The priority-conflict penalty
        # (−5% regime conflict) was insufficient to block it — SPY then rallied
        # +4.8 pts and crushed the put.
        #
        # Block PC_RATIO_EXTREME PUT outright when ≥2 of these structural bullish
        # flags are simultaneously true:
        #   • Regime = TREND_UP
        #   • SPY above VWAP +1SD or +2SD
        #   • SPY above prior-day high (overnight high broken = trend up)
        #   • Confirmed ORB breakout to the upside
        #
        # Requiring ≥2 flags avoids over-blocking (a single TREND_UP tag in a
        # choppy pre-ORB environment should still allow PC_RATIO to fire).
        ext = context.external
        if ext is not None:
            _vwap_pos  = getattr(ext, "vwap_band_position", "INSIDE")
            _above_pdh = getattr(ext, "above_overnight_high", False)
            _orb_bull  = (
                getattr(ext, "orb_breakout_confirmed", False)
                and getattr(ext, "orb_status", "INSIDE") == "ABOVE_ORB"
            )
            _regime_up = context.regime.regime == "TREND_UP"
            _vwap_bull = _vwap_pos in ("ABOVE_1SD", "ABOVE_2SD")
            _bull_flags = sum([_regime_up, _vwap_bull, _above_pdh, _orb_bull])
            # Default OFF: duplicates rules_v2 pc_ratio_alignment with a
            # DIFFERENT regime classifier (audit CRIT — signal survived only
            # when classifiers disagreed). rules_v2 is the single authority.
            if c.pc_structural_block_enabled and _bull_flags >= 2:
                _pc_put_blocked = [
                    s for s in signals
                    if s.signal_type == SignalType.PC_RATIO_EXTREME and s.right == "P"
                ]
                for _blk in _pc_put_blocked:
                    signals.remove(_blk)
                    logger.info(
                        "PC_RATIO PUT hard-blocked: {} bullish structural flags "
                        "(regime={}, vwap={}, above_pdh={}, orb_bull={})",
                        _bull_flags, context.regime.regime,
                        _vwap_pos, _above_pdh, _orb_bull,
                    )

            # ── Symmetric: block PC_RATIO CALL in confirmed bearish environment ──
            _regime_dn = context.regime.regime == "TREND_DOWN"
            _vwap_bear = _vwap_pos in ("BELOW_1SD", "BELOW_2SD")
            _below_pdl = getattr(ext, "below_overnight_low", False)
            _orb_bear  = (
                getattr(ext, "orb_breakout_confirmed", False)
                and getattr(ext, "orb_status", "INSIDE") == "BELOW_ORB"
            )
            _bear_flags = sum([_regime_dn, _vwap_bear, _below_pdl, _orb_bear])
            if c.pc_structural_block_enabled and _bear_flags >= 2:
                _pc_call_blocked = [
                    s for s in signals
                    if s.signal_type == SignalType.PC_RATIO_EXTREME and s.right == "C"
                ]
                for _blk in _pc_call_blocked:
                    signals.remove(_blk)
                    logger.info(
                        "PC_RATIO CALL hard-blocked: {} bearish structural flags "
                        "(regime={}, vwap={}, below_pdl={}, orb_bear={})",
                        _bear_flags, context.regime.regime,
                        _vwap_pos, _below_pdl, _orb_bear,
                    )

        # ── ORB-inside soft penalty ───────────────────────────────────────
        # Applied BEFORE the quality gate so the penalised confidence is what
        # gets checked at the threshold.  Strong signals can survive; marginal
        # signals will be filtered.
        for sig in signals:
            orb_pen, orb_pen_note = self._orb_inside_penalty(sig, context.external)
            if orb_pen < 0:
                _pre = sig.confidence
                sig.confidence = max(0.0, sig.confidence + orb_pen)
                sig.confidence_tier = self._tier_cfg(sig.confidence)
                sig.reasoning.append(f"🔶 {orb_pen_note}")
                logger.info(
                    "ORB-inside penalty: {} {} {}{} conf {:.0f}%→{:.0f}%",
                    sig.signal_type.value, sig.expiry, sig.strike, sig.right,
                    _pre * 100, sig.confidence * 100,
                )

        # ── Quality gate (hard blocks — confidence cannot override) ──────
        # Any signal that fails the quality gate is dropped here and counted
        # separately from confidence-threshold rejections.
        quality_passed: List[SpySignal] = []
        quality_blocked: List[SpySignal] = []
        for sig in signals:
            passes, fail_reasons = self._quality_gate(sig, context.external)
            if passes:
                quality_passed.append(sig)
            else:
                quality_blocked.append(sig)
                self.last_blocked.append((sig, "quality_gate"))
                log_blocked_signal(sig, "quality_gate", "; ".join(fail_reasons))
                for reason in fail_reasons:
                    logger.info(
                        "Quality gate BLOCKED: {} {} {}{} conf={:.0f}% — {}",
                        sig.signal_type.value, sig.expiry, sig.strike, sig.right,
                        sig.confidence * 100, reason,
                    )
        if quality_blocked:
            logger.info(
                "SignalEngine {}: {} signals quality-blocked (hard rules)",
                chain.expiry_month, len(quality_blocked),
            )
        signals = quality_passed

        filtered = [s for s in signals if s.confidence >= c.min_confidence]
        rejected = [s for s in signals if s.confidence < c.min_confidence]
        if rejected:
            for s in rejected:
                self.last_blocked.append((s, "confidence_threshold"))
                log_blocked_signal(
                    s, "confidence_threshold",
                    f"conf {s.confidence:.2f} < min {c.min_confidence:.2f}",
                )
                logger.info(
                    "Signal below threshold: {} {} {}{} conf={:.1f}% (need {:.0f}%)",
                    s.signal_type.value, s.expiry, s.strike, s.right,
                    s.confidence * 100, c.min_confidence * 100,
                )

        # ── Cross-signal directional conflict filter ──────────────────────
        # If both CALL and PUT directional signals passed the threshold in the
        # same cycle, the signals are contradictory.  Keep only the direction
        # with the highest single-signal confidence.
        if filtered:
            calls = [s for s in filtered if s.right == "C"]
            puts  = [s for s in filtered if s.right == "P"]
            both  = [s for s in filtered if s.right == "BOTH"]
            if calls and puts:
                best_call = max(s.confidence for s in calls)
                best_put  = max(s.confidence for s in puts)
                if best_call >= best_put:
                    logger.warning(
                        "Directional conflict: {} CALL + {} PUT signals — "
                        "keeping CALL (best={:.0f}% vs PUT best={:.0f}%)",
                        len(calls), len(puts), best_call * 100, best_put * 100,
                    )
                    filtered = calls + both
                else:
                    logger.warning(
                        "Directional conflict: {} CALL + {} PUT signals — "
                        "keeping PUT (best={:.0f}% vs CALL best={:.0f}%)",
                        len(calls), len(puts), best_put * 100, best_call * 100,
                    )
                    filtered = puts + both

        if filtered:
            logger.info(
                "SignalEngine {}: {} signals → {} passed (threshold={:.0f}%)",
                chain.expiry_month, len(signals), len(filtered),
                c.min_confidence * 100,
            )
        return filtered

    # ── Weighted confidence model ─────────────────────────────────────────────

    def _weighted_confidence(
        self,
        quote: OptionQuote,
        right: str,
        spike_mult: float,
        flow_score: float,
        signal_type: SignalType,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> float:
        """Compute weighted confidence score (0.0–1.0) for a spike-based signal."""

        # 25% — volume spike strength (normalized against threshold)
        # Denominator is 10.0: full score requires a 14× spike (rare — extreme events
        # only). Typical 6-8× sweeps score 0.05–0.10 of the 0.25 max, keeping the
        # base honest. High-conviction setups reach threshold via DynConf context
        # boosts (ORB confirmed + above VWAP + regime), not inflated base alone.
        vol_score = min(1.0, max(0.0, (spike_mult - c.volume_spike_mult) / 10.0))
        w_vol = 0.25 * vol_score

        # 15% — bid/ask imbalance
        if right == "C":
            ratio = quote.bid_ask_ratio
        else:
            ratio = quote.ask_bid_ratio
        threshold = c.bid_ask_imbalance_threshold
        imb_score = min(1.0, max(0.0, (ratio - 1.0) / max(1.0, threshold - 1.0)))
        w_imb = 0.15 * imb_score

        # 10% — delta quality
        d = quote.delta
        if d == 0.0:
            delta_score = 0.5  # unknown
        elif right == "C":
            delta_score = 1.0 if 0.30 <= d <= 0.60 else 0.3
        else:
            delta_score = 1.0 if -0.60 <= d <= -0.30 else 0.3
        w_delta = 0.10 * delta_score

        # 10% — gamma quality (ideal ATM range for SPY: 0.005–0.08)
        g = quote.gamma
        if g == 0.0:
            gamma_score = 0.5
        else:
            gamma_score = 1.0 if 0.005 <= g <= 0.08 else 0.3
        w_gamma = 0.10 * gamma_score

        # 10% — theta penalty (theta is negative; rapid decay hurts long premium)
        t = quote.theta
        if t == 0.0:
            theta_score = 0.5
        elif t < -0.15:    # very high decay
            theta_score = 0.2
        elif t < -0.08:
            theta_score = 0.6
        else:
            theta_score = 1.0
        w_theta = 0.10 * theta_score

        # 10% — IV regime alignment
        # Credit signals (HIGH_IV_ALERT, BEAR_PUT_SPREAD, BULL_CALL_SPREAD) → high rank better
        # Directional debit signals (SWEEP) → low rank better
        credit_types = {SignalType.HIGH_IV_ALERT, SignalType.BEAR_PUT_SPREAD, SignalType.BULL_CALL_SPREAD}
        if signal_type in credit_types:
            iv_score = context.iv_rank / 100.0
        else:
            iv_score = 1.0 - (context.iv_rank / 100.0)
        w_iv = 0.10 * iv_score

        # 10% — sentiment alignment
        sent = context.sentiment.score / 100.0  # -1 to +1
        if right == "C":
            sent_score = (sent + 1.0) / 2.0        # +1 bullish → 1.0
        elif right == "P":
            sent_score = (-sent + 1.0) / 2.0       # -1 bearish → 1.0
        else:
            sent_score = 0.5                        # direction-agnostic
        w_sent = 0.10 * sent_score

        # 5% — open interest strength (10k+ = max)
        oi_score = min(1.0, quote.open_interest / 10_000.0)
        w_oi = 0.05 * oi_score

        # 5% — flow score (repeat sweep confirmation)
        w_flow = 0.05 * flow_score

        base = w_vol + w_imb + w_delta + w_gamma + w_theta + w_iv + w_sent + w_oi + w_flow

        # External composite adjustment (±composite_confidence_boost from config)
        ext = context.external
        if ext is not None:
            boost_cap = self._composite_boost   # max ±N% from external (config-driven)
            # Directional alignment: composite positive boosts calls, negative boosts puts
            if right == "C":
                ext_adj = ext.composite_score * boost_cap
            elif right == "P":
                ext_adj = -ext.composite_score * boost_cap
            else:
                ext_adj = abs(ext.composite_score) * boost_cap * 0.5  # small boost for straddle
            base += ext_adj

        return min(1.0, max(0.0, base))

    def _enrich(self, sig: SpySignal, quote: OptionQuote, context: SignalContext) -> SpySignal:
        """Copy Greek/liquidity/context fields from quote and context into signal."""
        sig.delta = quote.delta
        sig.gamma = quote.gamma
        sig.theta = quote.theta
        sig.vega = quote.vega
        sig.impl_vol = quote.impl_vol
        sig.open_interest = quote.open_interest
        sig.spread_pct = quote.spread_pct
        sig.bid = quote.bid
        sig.ask = quote.ask
        sig.iv_rank = context.iv_rank
        sig.regime = context.regime.regime
        sig.sentiment_score = context.sentiment.score
        sig.sentiment_label = context.sentiment.label
        sig.confidence_tier = self._tier_cfg(sig.confidence)
        # External fields
        if context.external is not None:
            ext = context.external
            sig.external_composite = ext.composite_score
            sig.event_risk = ext.event_risk
            sig.event_minutes = ext.event_minutes
            sig.next_event_title = ext.next_event_title
            sig.news_score = ext.news_score
            sig.retail_score = ext.retail_score
            sig.macro_headwind = ext.macro_headwind
            sig.macro_label = ext.macro_label
            sig.tnx_trend = ext.tnx_trend
            sig.dxy_trend = ext.dxy_trend
            sig.equity_pc = ext.equity_pc
            # Flow confirmation
            sig.flow_confirmation_score = ext.flow_score
            sig.dark_pool_bias = ext.flow_dark_pool
            sig.gex_bias = ext.flow_gex_bias
            sig.intraday_pc_ratio = ext.flow_pc_ratio
            # Cross-asset confirmation
            sig.cross_asset_divergence = getattr(ext, "cross_asset_divergence", "NONE")
            sig.cross_asset_bias = getattr(ext, "cross_asset_bias", "NEUTRAL")
            sig.qqq_rs = getattr(ext, "qqq_rs", 0.0)
        return sig

    # ── Rule implementations ──────────────────────────────────────────────────

    def _pc_ratio_signal(
        self,
        chain: ChainSnapshot,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        signals = []
        pc = chain.put_call_ratio
        if pc is None:
            return signals

        atm = chain.atm_strike(context.spy_price)

        if (
            pc > c.pc_ratio_bearish
            and chain.total_put_volume >= c.min_volume_for_signal
            and chain.total_call_volume >= c.pc_ratio_min_denom_volume
        ):
            atm_put = chain.put_at(atm)
            # Confidence: base capped at 0.76 — extreme P/C alone does NOT warrant
            # EXTREME tier.  High P/C can reflect hedging, dealer positioning, or
            # protection buying rather than directional bearish conviction.
            # Scale: P/C 1.8→0.68, 3.0→0.73, 5.0→0.76 (capped)
            base = min(0.68 + (pc - c.pc_ratio_bearish) * 0.04, 0.76)

            # ── Stale flow decay ───────────────────────────────────────────
            # If extreme P/C has persisted for 15-45+ min without price
            # follow-through, the flow is likely hedging noise.
            stale_decay = self._stale_flow_decay("BEARISH", context.spy_price)

            # ── Intraday move exhaustion ───────────────────────────────────
            # If SPY has already dropped ≥0.20% from today's high, the put
            # thesis may be exhausted — most of the move has already happened.
            # This was the #1 root cause of high-confidence signals losing.
            intraday_move_adj, _intraday_note = self._intraday_move_penalty(
                "BEARISH", context.spy_price,
            )

            # ── RSI divergence penalty ─────────────────────────────────────
            # RSI BULLISH_DIV at oversold levels = strong reversal signal.
            # Put signals issued during bullish RSI divergence are very
            # likely to lose money.  #2 root cause of high-conf losses.
            rsi_adj, _rsi_note = self._rsi_divergence_penalty(
                "BEARISH", context.external,
            )

            # ── 0DTE hedging discount ──────────────────────────────────────
            # On OPEX day (DTE=0), put volume is heavily inflated by
            # expiry-day hedging and gamma-related repositioning.  This makes
            # P/C ratio unreliable as a directional indicator.  Multi-week
            # accumulation (DTE>=8) is more credible.
            chain_dte = -1
            if chain.expiry_date:
                try:
                    exp_d = _dt.datetime.strptime(chain.expiry_date, "%Y%m%d").date()
                    chain_dte = max(0, (exp_d - _dt.date.today()).days)
                except (ValueError, Exception):
                    pass
            dte_adj = 0.0
            if chain_dte == 0:
                dte_adj = -0.06   # 0DTE: high P/C is mostly hedging noise
            elif chain_dte == 1:
                dte_adj = -0.03   # 1DTE: still noisy
            elif chain_dte >= 8:
                dte_adj = 0.03    # 8+ DTE: multi-week accumulation is directional

            # ── VIX term structure gate ────────────────────────────────────
            # Steep contango (VIX/VXV < 0.85) + extreme P/C = complacent
            # protection buying, not genuine fear.  Cap base lower.
            # Backwardation (VIX/VXV > 1.05) + extreme P/C = real fear,
            # allow a small boost.
            vol_gate_adj = 0.0
            ext = context.external
            if ext is not None:
                vs = ext.vol_structure
                if vs in ("CONTANGO", "STEEP_CONTANGO"):
                    vol_gate_adj = -0.04  # calm vol → puts are hedging
                elif vs == "STEEP_BACKWARDATION":
                    vol_gate_adj = 0.04   # panic → put flow is genuine
                elif vs == "BACKWARDATION":
                    vol_gate_adj = 0.02   # mild fear → slight credibility boost

            # ── TNX / DXY velocity ─────────────────────────────────────
            # Rising yields + strong dollar = bearish headwind for equities
            # → confirms put flow.  Falling yields + weak dollar = risk-on
            # → puts are likely hedging, not directional.
            macro_vel_adj = 0.0
            if ext is not None:
                _tnx = ext.tnx_trend
                _dxy = ext.dxy_trend
                if _tnx == "RISING" and _dxy == "RISING":
                    macro_vel_adj = 0.04   # double tightening confirms bearish
                elif _tnx == "RISING" or _dxy == "RISING":
                    macro_vel_adj = 0.02   # single tightening → mild confirmation
                elif _tnx == "FALLING" and _dxy == "FALLING":
                    macro_vel_adj = -0.04  # double easing opposes bearish put signal
                elif _tnx == "FALLING" or _dxy == "FALLING":
                    macro_vel_adj = -0.02  # single easing → mild headwind for puts

            # ── NYSE TICK ──────────────────────────────────────────────────
            # Extreme negative TICK readings confirm institutional selling;
            # strong positive TICK opposes bearish put thesis.
            tick_adj = 0.0
            if ext is not None:
                tick_val = getattr(ext, "tick_value", None)
                if tick_val is not None:
                    if tick_val <= -500:
                        tick_adj = 0.03   # strong selling confirms puts
                    elif tick_val <= -200:
                        tick_adj = 0.01
                    elif tick_val >= 500:
                        tick_adj = -0.03  # strong buying opposes puts
                    elif tick_val >= 200:
                        tick_adj = -0.01

            # ── Price structure ────────────────────────────────────────────
            # Price action at key levels is the strongest confirmation an
            # experienced discretionary trader uses.  These levels are
            # already tracked in ExternalContext.
            price_struct_adj = 0.0
            _price_struct_parts: list = []
            spy = context.spy_price
            if ext is not None:
                # Prior-day level breaks
                if ext.below_overnight_low:
                    price_struct_adj += 0.03
                    _price_struct_parts.append("below prior-day low")
                elif ext.above_overnight_high:
                    price_struct_adj -= 0.03   # above PDH opposes bearish
                    _price_struct_parts.append("above prior-day high (opposes)")

                # Gap direction: gap-down confirms bearish, gap-up opposes
                if ext.gap_pct <= -0.30:
                    price_struct_adj += 0.04   # meaningful gap-down
                    _price_struct_parts.append(f"gap-down {ext.gap_pct:+.2f}%")
                elif ext.gap_pct >= 0.30:
                    price_struct_adj -= 0.04   # gap-up opposes bearish
                    _price_struct_parts.append(f"gap-up {ext.gap_pct:+.2f}% (opposes)")

                # Opening range breakout (after 10:00 ET)
                if ext.orb_established:
                    if ext.orb_status == "BELOW_ORB" and ext.orb_breakout_confirmed:
                        price_struct_adj += 0.03
                        _price_struct_parts.append("confirmed ORB breakdown")
                    elif ext.orb_status == "ABOVE_ORB" and ext.orb_breakout_confirmed:
                        price_struct_adj -= 0.03  # bullish ORB opposes puts
                        _price_struct_parts.append("ORB breakout (opposes)")
                    elif ext.orb_status == "INSIDE":
                        price_struct_adj -= 0.02  # range-bound → uncertain
                        _price_struct_parts.append("inside ORB (uncertain)")

                # Gamma wall failure: price at/near put wall = dealers hedging
                if ext.at_put_wall:
                    price_struct_adj += 0.02
                    _price_struct_parts.append("at put wall (dealer hedging)")
                elif ext.at_call_wall:
                    price_struct_adj -= 0.02  # at call wall = strong resistance, puts may fail
                    _price_struct_parts.append("at call wall (opposes)")

            # ── Trap detector (bear trap) ──────────────────────────────────
            # Price breaking below key support BUT internals diverge upward
            # signals a bear trap — put buyers will get squeezed.
            # Count bearish-surface + bullish-internals divergences.
            trap_adj = 0.0
            _trap_parts: list = []
            if ext is not None:
                _trap_flags = 0
                # Surface looks bearish (price broke below support)
                _surface_bearish = (
                    ext.below_overnight_low
                    or (ext.orb_established and ext.orb_status == "BELOW_ORB"
                        and ext.orb_breakout_confirmed)
                )
                if _surface_bearish:
                    # Check for internals diverging bullish (trap signals)
                    if ext.breadth_ratio >= 0.55:
                        _trap_flags += 1
                        _trap_parts.append("breadth improving")
                    _tick_v = getattr(ext, "tick_value", None)
                    if _tick_v is not None and _tick_v >= 200:
                        _trap_flags += 1
                        _trap_parts.append("TICK positive")
                    if getattr(ext, "vix_intraday", "FLAT") == "FALLING":
                        _trap_flags += 1
                        _trap_parts.append("VIX fading")
                    _qqq_p = getattr(ext, "qqq_vs_spy_pct", 0.0)
                    if _qqq_p > 0.10:
                        _trap_flags += 1
                        _trap_parts.append("QQQ leading higher")

                    # 2+ divergences = likely bear trap
                    if _trap_flags >= 3:
                        trap_adj = -0.06
                    elif _trap_flags >= 2:
                        trap_adj = -0.04

            # Sentiment adjustment: bearish sentiment boosts, bullish PENALISES.
            # Bullish composite + extreme put flow = high probability of hedging.
            sent_norm = context.sentiment.score / 100.0  # -1 to +1
            if sent_norm < 0:
                sent_adj = abs(sent_norm) * 0.08   # bearish aligns → boost
            else:
                sent_adj = -sent_norm * 0.12       # bullish opposes → stronger penalty

            # External composite alignment: bullish composite with extreme put flow
            # strongly suggests hedging activity rather than directional conviction.
            ext_adj = 0.0
            if ext is not None:
                if ext.composite_score > 0.30:
                    ext_adj = -0.06  # bullish macro/flow composite → reduce put conviction
                elif ext.composite_score < -0.30:
                    ext_adj = 0.03   # bearish composite confirms put signal

            # VWAP position: above VWAP → put flow more likely hedging/protection;
            # below VWAP → BUT at extreme extension (BELOW_2SD), the move is
            # likely exhausted and a mean-reversion bounce is imminent.
            # Fix: BELOW_2SD was previously +0.05 which BOOSTED chasing puts
            # into an already-stretched decline — root cause #4.
            vwap_adj = 0.0
            if ext is not None:
                bp = ext.vwap_band_position
                if bp in ("ABOVE_1SD", "ABOVE_2SD"):
                    vwap_adj = -0.07  # SPY above VWAP: puts likely protection, not directional
                elif bp == "INSIDE_1SD":
                    vwap_adj = -0.03  # ambiguous zone: slight caution on put conviction
                elif bp == "BELOW_1SD":
                    vwap_adj = 0.03   # below VWAP: bearish flow more credible
                elif bp == "BELOW_2SD":
                    vwap_adj = -0.05  # STRETCHED: chasing a -2σ decline is dangerous

            # Regime alignment: TREND_UP directly opposes bearish put signal
            regime = context.regime.regime
            regime_adj = 0.0
            if regime == "TREND_UP":
                regime_adj = -0.12  # strong penalty: trend opposes put signal
            elif regime == "TREND_DOWN":
                regime_adj = 0.05   # aligned: trend supports put signal

            # ── Weighted multi-factor confirmation ──────────────────────
            # P/C ratio alone caps at 0.76.  The remaining ~19% to reach
            # the 0.90 dispatch threshold MUST come from confirming sources.
            # Weights: Breadth=2, GEX=2, VWAP=2, Flow=1, DarkPool=1, QQQ=1, IWM=1
            # 5+ weighted points → EXTREME tier quality.
            confirm_pts = 0   # weighted confirmation points (out of 10 possible)
            flow_adj = 0.0
            breadth_adj = 0.0
            gex_adj = 0.0
            dp_adj = 0.0
            qqq_adj = 0.0
            iwm_adj = 0.0

            if ext is not None:
                # Flow confirmation (weight=1): aggregate flow score should be bearish
                if ext.flow_score < -30:
                    flow_adj = 0.04
                    confirm_pts += 1
                elif ext.flow_score < -10:
                    flow_adj = 0.02

                # Breadth (weight=2): weak internals confirm bearish thesis
                if ext.breadth_ratio <= 0.30:
                    breadth_adj = 0.05
                    confirm_pts += 2
                elif ext.breadth_ratio <= 0.40:
                    breadth_adj = 0.03
                    confirm_pts += 1

                # GEX (weight=2): dealer gamma exposure supports downside
                if ext.flow_gex_bias == "SUPPORTIVE_DOWNSIDE":
                    gex_adj = 0.05
                    confirm_pts += 2
                elif ext.flow_gex_bias != "NEUTRAL":
                    pass  # SUPPORTIVE_UPSIDE = no penalty, just no bonus

                # Dark pool (weight=1): distribution bias confirms institutional selling
                if ext.flow_dark_pool == "DISTRIBUTION":
                    dp_adj = 0.03
                    confirm_pts += 1

                # Relative strength: QQQ (weight=1) + IWM (weight=1)
                qqq_pct = getattr(ext, "qqq_vs_spy_pct", 0.0)
                if qqq_pct < -0.10:
                    qqq_adj = 0.03
                    confirm_pts += 1
                elif qqq_pct > 0.20:
                    qqq_adj = -0.04  # QQQ strong while SPY weak = rotation, not sell-off

                iwm_pct = getattr(ext, "iwm_vs_spy_pct", 0.0)
                if iwm_pct < -0.15:
                    iwm_adj = 0.02
                    confirm_pts += 1
                elif iwm_pct > 0.20:
                    iwm_adj = -0.02  # IWM strong = risk-on rotation

                # VWAP alignment also earns weighted confirmation points (weight=2)
                # (vwap_adj amount already set above; just add confirmation credit)
                if vwap_adj > 0:
                    confirm_pts += 2   # below VWAP = strong bearish confirmation
                elif vwap_adj >= -0.03:
                    confirm_pts += 1   # inside 1SD = partial

            multi_adj = flow_adj + breadth_adj + gex_adj + dp_adj + qqq_adj + iwm_adj

            conf = max(0.0, min(0.95, base + stale_decay + intraday_move_adj + rsi_adj + dte_adj + vol_gate_adj + macro_vel_adj + tick_adj + price_struct_adj + trap_adj + sent_adj + ext_adj + vwap_adj + regime_adj + multi_adj))

            # ── Time-of-day confidence ceiling ─────────────────────────────
            # Always apply during market hours (9:30–16:00 ET).  Outside
            # market hours the aggressive POST_MARKET cap (0.75) only
            # applies when we have live external data to confirm the
            # session context; without ext we can't be sure the signal
            # is truly post-market vs. a unit test running after 4 PM.
            _tod_note = ""
            _tod_cap, _tod_bucket = self._tod_ceiling()
            _in_rth = 570 <= (_dt.datetime.now(ZoneInfo("America/New_York")).hour * 60
                              + _dt.datetime.now(ZoneInfo("America/New_York")).minute) < 960
            if _in_rth or ext is not None:
                if conf > _tod_cap:
                    _tod_note = f"TOD ceiling ({_tod_bucket}): {conf:.2f}→{_tod_cap:.2f}"
                    conf = _tod_cap

            _comp_note = f" | Composite: {ext.composite_score:+.2f}" if ext else ""
            _gex_note  = f" | GEX: {ext.flow_gex_bias}" if ext else ""
            _vwap_note = f"VWAP position: {ext.vwap_band_position}" if ext else ""
            _dte_note = f"Chain DTE: {chain_dte}" if chain_dte >= 0 else ""
            _macro_vel_note = f"TNX {ext.tnx_trend} / DXY {ext.dxy_trend}" if ext else ""
            _price_note = f"Price structure: {', '.join(_price_struct_parts)}" if _price_struct_parts else ""
            _stale_note = f"Stale flow decay: {stale_decay:+.2f}" if stale_decay < 0 else ""
            _trap_note = f"Bear trap detected ({trap_adj:+.2f}): {', '.join(_trap_parts)}" if trap_adj < 0 else ""
            _confirm_note = f"Weighted confirmations: {confirm_pts}/10 (breadth×2/GEX×2/VWAP×2/flow/dark-pool/QQQ/IWM)"
            sig = SpySignal(
                signal_type=SignalType.PC_RATIO_EXTREME,
                strike=atm, expiry=chain.expiry_month, right="P",
                confidence=conf, spy_price=context.spy_price, vix=context.vix,
                volume=chain.total_put_volume, volume_spike_mult=pc,
                bid_size=atm_put.bid_size if atm_put else 0,
                ask_size=atm_put.ask_size if atm_put else 0,
                reasoning=[
                    r for r in [
                        f"P/C ratio = {pc:.2f} (>{c.pc_ratio_bearish}) — may reflect hedging/protection",
                        f"Put vol: {chain.total_put_volume:,} vs call vol: {chain.total_call_volume:,}",
                        f"Sentiment: {context.sentiment.label} ({context.sentiment.score:+.0f}){_comp_note}",
                        f"Regime: {context.regime.regime}{_gex_note}",
                        _vwap_note,
                        _dte_note,
                        _macro_vel_note,
                        _price_note,
                        _stale_note,
                        _intraday_note,
                        _rsi_note,
                        _trap_note,
                        _tod_note,
                        _confirm_note,
                    ] if r
                ],
                suggested_trade=(
                    f"Cautious bearish below {atm:.0f}. "
                    f"Bear Put Spread preferred (not naked puts) exp {chain.expiry_month}. "
                    f"Exit if SPY reclaims VWAP."
                ),
                # Suggestion text explicitly recommends a defined-risk spread —
                # tag it so the TM sizes the bear put spread, not the naked put.
                structure="BEAR_PUT_SPREAD",
                short_strike=float(atm - 5),
            )
            if atm_put:
                self._enrich(sig, atm_put, context)
            else:
                sig.confidence_tier = self._tier_cfg(conf)
                sig.iv_rank = context.iv_rank
                sig.regime = context.regime.regime
                sig.sentiment_score = context.sentiment.score
                sig.sentiment_label = context.sentiment.label
            signals.append(sig)

        elif (
            pc < c.pc_ratio_bullish
            and chain.total_call_volume >= c.min_volume_for_signal
            and chain.total_put_volume >= c.pc_ratio_min_denom_volume
        ):
            # Base capped at 0.76 — extreme low P/C alone does not warrant EXTREME tier.
            # Scale: P/C 0.5→0.68, 0.3→0.70, 0.1→0.72 (capped at 0.76)
            base = min(0.68 + (c.pc_ratio_bullish - pc) * 0.08, 0.76)

            # ── Stale flow decay ───────────────────────────────────────────
            stale_decay = self._stale_flow_decay("BULLISH", context.spy_price)

            # ── Intraday move exhaustion (symmetric to bearish) ────────────
            intraday_move_adj, _intraday_note = self._intraday_move_penalty(
                "BULLISH", context.spy_price,
            )

            # ── RSI divergence penalty (symmetric to bearish) ──────────────
            rsi_adj, _rsi_note = self._rsi_divergence_penalty(
                "BULLISH", context.external,
            )

            # ── 0DTE hedging discount (symmetric to bearish) ───────────────
            chain_dte = -1
            if chain.expiry_date:
                try:
                    exp_d = _dt.datetime.strptime(chain.expiry_date, "%Y%m%d").date()
                    chain_dte = max(0, (exp_d - _dt.date.today()).days)
                except (ValueError, Exception):
                    pass
            dte_adj = 0.0
            if chain_dte == 0:
                dte_adj = -0.06   # 0DTE: extreme low P/C may be call hedging
            elif chain_dte == 1:
                dte_adj = -0.03
            elif chain_dte >= 8:
                dte_adj = 0.03    # multi-week accumulation is directional

            # ── VIX term structure gate (symmetric to bearish) ─────────────
            vol_gate_adj = 0.0
            ext = context.external
            if ext is not None:
                vs = ext.vol_structure
                if vs in ("CONTANGO", "STEEP_CONTANGO"):
                    # Calm vol favours call-buying → slight boost
                    vol_gate_adj = 0.02
                elif vs in ("BACKWARDATION", "STEEP_BACKWARDATION"):
                    # Fear environment → calls are fighting the current
                    vol_gate_adj = -0.04

            # ── TNX / DXY velocity (asymmetric to bearish) ────────────────
            # Falling yields + weak dollar = risk-on → confirms call flow.
            # Rising yields + strong dollar = tightening → opposes calls.
            macro_vel_adj = 0.0
            if ext is not None:
                _tnx = ext.tnx_trend
                _dxy = ext.dxy_trend
                if _tnx == "FALLING" and _dxy == "FALLING":
                    macro_vel_adj = 0.04   # double easing confirms bullish
                elif _tnx == "FALLING" or _dxy == "FALLING":
                    macro_vel_adj = 0.02   # single easing → mild confirmation
                elif _tnx == "RISING" and _dxy == "RISING":
                    macro_vel_adj = -0.04  # double tightening opposes call signal
                elif _tnx == "RISING" or _dxy == "RISING":
                    macro_vel_adj = -0.02  # single tightening → mild headwind for calls

            # ── NYSE TICK (asymmetric to bearish) ──────────────────────────
            # Strong positive TICK confirms institutional buying;
            # extreme negative TICK opposes bullish call thesis.
            tick_adj = 0.0
            if ext is not None:
                tick_val = getattr(ext, "tick_value", None)
                if tick_val is not None:
                    if tick_val >= 500:
                        tick_adj = 0.03   # strong buying confirms calls
                    elif tick_val >= 200:
                        tick_adj = 0.01
                    elif tick_val <= -500:
                        tick_adj = -0.03  # strong selling opposes calls
                    elif tick_val <= -200:
                        tick_adj = -0.01

            # ── Price structure (asymmetric to bearish) ────────────────────
            price_struct_adj = 0.0
            _price_struct_parts: list = []
            spy = context.spy_price
            if ext is not None:
                # Prior-day level breaks
                if ext.above_overnight_high:
                    price_struct_adj += 0.03
                    _price_struct_parts.append("above prior-day high")
                elif ext.below_overnight_low:
                    price_struct_adj -= 0.03   # below PDL opposes bullish
                    _price_struct_parts.append("below prior-day low (opposes)")

                # Gap direction: gap-up confirms bullish, gap-down opposes
                if ext.gap_pct >= 0.30:
                    price_struct_adj += 0.04   # meaningful gap-up
                    _price_struct_parts.append(f"gap-up {ext.gap_pct:+.2f}%")
                elif ext.gap_pct <= -0.30:
                    price_struct_adj -= 0.04   # gap-down opposes bullish
                    _price_struct_parts.append(f"gap-down {ext.gap_pct:+.2f}% (opposes)")

                # Opening range breakout
                if ext.orb_established:
                    if ext.orb_status == "ABOVE_ORB" and ext.orb_breakout_confirmed:
                        price_struct_adj += 0.03
                        _price_struct_parts.append("confirmed ORB breakout")
                    elif ext.orb_status == "BELOW_ORB" and ext.orb_breakout_confirmed:
                        price_struct_adj -= 0.03  # bearish ORB opposes calls
                        _price_struct_parts.append("ORB breakdown (opposes)")
                    elif ext.orb_status == "INSIDE":
                        price_struct_adj -= 0.02  # range-bound → uncertain
                        _price_struct_parts.append("inside ORB (uncertain)")

                # Gamma wall: price at call wall = ceiling; at put wall = floor
                if ext.at_call_wall:
                    price_struct_adj -= 0.02  # call wall = resistance overhead
                    _price_struct_parts.append("at call wall (resistance)")
                elif ext.at_put_wall:
                    price_struct_adj += 0.02  # at put wall = dealer support
                    _price_struct_parts.append("at put wall (support)")

            # ── Trap detector (bull trap) ──────────────────────────────────
            # Price breaking above key resistance BUT internals diverge
            # downward signals a bull trap — call buyers will get crushed.
            # Count bullish-surface + bearish-internals divergences.
            trap_adj = 0.0
            _trap_parts: list = []
            if ext is not None:
                _trap_flags = 0
                # Surface looks bullish (price broke above resistance)
                _surface_bullish = (
                    ext.above_overnight_high
                    or (ext.orb_established and ext.orb_status == "ABOVE_ORB"
                        and ext.orb_breakout_confirmed)
                )
                if _surface_bullish:
                    # Check for internals diverging bearish (trap signals)
                    if ext.breadth_ratio <= 0.45:
                        _trap_flags += 1
                        _trap_parts.append("breadth weakening")
                    _tick_v = getattr(ext, "tick_value", None)
                    if _tick_v is not None and _tick_v <= -200:
                        _trap_flags += 1
                        _trap_parts.append("TICK fading")
                    if getattr(ext, "vix_intraday", "FLAT") == "RISING":
                        _trap_flags += 1
                        _trap_parts.append("VIX rising")
                    _qqq_p = getattr(ext, "qqq_vs_spy_pct", 0.0)
                    if _qqq_p < -0.10:
                        _trap_flags += 1
                        _trap_parts.append("QQQ diverging lower")

                    # 2+ divergences = likely bull trap
                    if _trap_flags >= 3:
                        trap_adj = -0.06
                    elif _trap_flags >= 2:
                        trap_adj = -0.04

            # Sentiment adjustment: bullish sentiment boosts, bearish PENALISES.
            sent_norm = context.sentiment.score / 100.0  # -1 to +1
            if sent_norm > 0:
                sent_adj = sent_norm * 0.08        # bullish aligns → boost
            else:
                sent_adj = sent_norm * 0.12        # bearish opposes → stronger penalty

            # External composite: bearish composite opposing extreme call flow = hedging risk
            ext_adj = 0.0
            if ext is not None:
                if ext.composite_score < -0.30:
                    ext_adj = -0.06  # bearish composite → reduce call conviction
                elif ext.composite_score > 0.30:
                    ext_adj = 0.03   # bullish composite confirms call signal

            # VWAP position: above VWAP → calls credible (with trend);
            # BUT at extreme extension (ABOVE_2SD), the rally is likely
            # exhausted and mean-reversion is imminent.
            # Fix: ABOVE_2SD was previously +0.05 which BOOSTED chasing calls
            # into an already-stretched rally — mirror of bearish fix.
            vwap_adj = 0.0
            if ext is not None:
                bp = ext.vwap_band_position
                if bp in ("BELOW_1SD", "BELOW_2SD"):
                    vwap_adj = -0.07  # SPY below VWAP: call flow may be short-covering
                elif bp == "INSIDE_1SD":
                    vwap_adj = -0.03  # ambiguous zone: slight caution
                elif bp == "ABOVE_1SD":
                    vwap_adj = 0.03   # above VWAP: bullish call flow more credible
                elif bp == "ABOVE_2SD":
                    vwap_adj = -0.05  # STRETCHED: chasing a +2σ rally is dangerous

            # Regime alignment: TREND_DOWN opposes bullish call signal
            regime = context.regime.regime
            regime_adj = 0.0
            if regime == "TREND_DOWN":
                regime_adj = -0.12  # strong penalty: trend opposes call signal
            elif regime == "TREND_UP":
                regime_adj = 0.05   # aligned: trend supports call signal

            # ── Weighted multi-factor confirmation (asymmetric to bearish) ─
            # Weights: Breadth=2, GEX=2, VWAP=2, Flow=1, DarkPool=1, QQQ=1, IWM=1
            confirm_pts = 0
            flow_adj = 0.0
            breadth_adj = 0.0
            gex_adj = 0.0
            dp_adj = 0.0
            qqq_adj = 0.0
            iwm_adj = 0.0

            if ext is not None:
                # Flow confirmation (weight=1): aggregate flow score should be bullish
                if ext.flow_score > 30:
                    flow_adj = 0.04
                    confirm_pts += 1
                elif ext.flow_score > 10:
                    flow_adj = 0.02

                # Breadth (weight=2): strong internals confirm bullish thesis
                if ext.breadth_ratio >= 0.70:
                    breadth_adj = 0.05
                    confirm_pts += 2
                elif ext.breadth_ratio >= 0.60:
                    breadth_adj = 0.03
                    confirm_pts += 1

                # GEX (weight=2): dealer gamma exposure supports upside
                if ext.flow_gex_bias == "SUPPORTIVE_UPSIDE":
                    gex_adj = 0.05
                    confirm_pts += 2

                # Dark pool (weight=1): accumulation bias confirms institutional buying
                if ext.flow_dark_pool == "ACCUMULATION":
                    dp_adj = 0.03
                    confirm_pts += 1

                # Relative strength: QQQ (weight=1) + IWM (weight=1)
                qqq_pct = getattr(ext, "qqq_vs_spy_pct", 0.0)
                if qqq_pct > 0.10:
                    qqq_adj = 0.03
                    confirm_pts += 1
                elif qqq_pct < -0.20:
                    qqq_adj = -0.04  # QQQ weak while SPY strong = rotation, not rally

                iwm_pct = getattr(ext, "iwm_vs_spy_pct", 0.0)
                if iwm_pct > 0.15:
                    iwm_adj = 0.02
                    confirm_pts += 1
                elif iwm_pct < -0.20:
                    iwm_adj = -0.02  # IWM weak = risk-off undercurrent

                # VWAP alignment earns weighted confirmation points (weight=2)
                if vwap_adj > 0:
                    confirm_pts += 2   # above VWAP = strong bullish confirmation
                elif vwap_adj >= -0.03:
                    confirm_pts += 1   # inside 1SD = partial

            multi_adj = flow_adj + breadth_adj + gex_adj + dp_adj + qqq_adj + iwm_adj

            conf = max(0.0, min(0.95, base + stale_decay + intraday_move_adj + rsi_adj + dte_adj + vol_gate_adj + macro_vel_adj + tick_adj + price_struct_adj + trap_adj + sent_adj + ext_adj + vwap_adj + regime_adj + multi_adj))

            # ── Time-of-day confidence ceiling ─────────────────────────────
            # Always apply during market hours (9:30–16:00 ET).  Outside
            # market hours the aggressive POST_MARKET cap (0.75) only
            # applies when we have live external data to confirm the
            # session context; without ext we can't be sure the signal
            # is truly post-market vs. a unit test running after 4 PM.
            _tod_note = ""
            _tod_cap, _tod_bucket = self._tod_ceiling()
            _in_rth = 570 <= (_dt.datetime.now(ZoneInfo("America/New_York")).hour * 60
                              + _dt.datetime.now(ZoneInfo("America/New_York")).minute) < 960
            if _in_rth or ext is not None:
                if conf > _tod_cap:
                    _tod_note = f"TOD ceiling ({_tod_bucket}): {conf:.2f}→{_tod_cap:.2f}"
                    conf = _tod_cap

            _comp_note = f" | Composite: {ext.composite_score:+.2f}" if ext else ""
            _gex_note  = f" | GEX: {ext.flow_gex_bias}" if ext else ""
            _vwap_note = f"VWAP position: {ext.vwap_band_position}" if ext else ""
            _dte_note = f"Chain DTE: {chain_dte}" if chain_dte >= 0 else ""
            _macro_vel_note = f"TNX {ext.tnx_trend} / DXY {ext.dxy_trend}" if ext else ""
            _price_note = f"Price structure: {', '.join(_price_struct_parts)}" if _price_struct_parts else ""
            _stale_note = f"Stale flow decay: {stale_decay:+.2f}" if stale_decay < 0 else ""
            _trap_note = f"Bull trap detected ({trap_adj:+.2f}): {', '.join(_trap_parts)}" if trap_adj < 0 else ""
            _confirm_note = f"Weighted confirmations: {confirm_pts}/10 (breadth×2/GEX×2/VWAP×2/flow/dark-pool/QQQ/IWM)"
            _atm_call = chain.call_at(atm)
            sig = SpySignal(
                signal_type=SignalType.PC_RATIO_EXTREME,
                strike=atm, expiry=chain.expiry_month, right="C",
                confidence=conf, spy_price=context.spy_price, vix=context.vix,
                volume=chain.total_call_volume, volume_spike_mult=1.0 / pc if pc > 0 else 0.0,
                bid_size=_atm_call.bid_size if _atm_call else 0,
                ask_size=_atm_call.ask_size if _atm_call else 0,
                reasoning=[
                    r for r in [
                        f"P/C ratio = {pc:.2f} (<{c.pc_ratio_bullish}) — broad call activity",
                        f"Call vol: {chain.total_call_volume:,} vs put vol: {chain.total_put_volume:,}",
                        f"Sentiment: {context.sentiment.label} ({context.sentiment.score:+.0f}){_comp_note}",
                        f"Regime: {context.regime.regime}{_gex_note}",
                        _vwap_note,
                        _dte_note,
                        _macro_vel_note,
                        _price_note,
                        _stale_note,
                        _intraday_note,
                        _rsi_note,
                        _trap_note,
                        _tod_note,
                        _confirm_note,
                    ] if r
                ],
                # Suggestion text explicitly recommends a defined-risk spread.
                structure="BULL_CALL_SPREAD",
                short_strike=float(atm + 5),
                suggested_trade=(
                    f"Broad call interest near {atm:.0f} exp {chain.expiry_month}. "
                    "Bull Call Spread preferred if VWAP holds. "
                    "Watch for short-covering vs genuine breakout."
                ),
            )
            sig.confidence_tier = self._tier_cfg(conf)
            sig.iv_rank = context.iv_rank
            sig.regime = context.regime.regime
            sig.sentiment_score = context.sentiment.score
            sig.sentiment_label = context.sentiment.label
            # Enrich with the ATM call's live quote + Greeks so the signal is
            # tradeable. The PUT branch does this; the CALL branch omitted it,
            # leaving bid/ask/delta = 0 → every bullish PC_RATIO signal was
            # untradeable (dropped by the executor's no-live-quote / zero-delta
            # gates). (audit 2026-07-13, P1-8)
            if _atm_call:
                self._enrich(sig, _atm_call, context)
            signals.append(sig)

        return signals

    def _call_spike_signals(
        self,
        quote: OptionQuote,
        expiry: str,
        spike_mult: float,
        flow_score: float,
        iv_low: bool,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        signals: List[SpySignal] = []
        conf = self._weighted_confidence(quote, "C", spike_mult, flow_score, SignalType.CALL_SWEEP, context, c)

        reasoning = [
            f"Call volume spike: {quote.volume:,} contracts at {quote.strike:.0f}C",
            f"Spike: {spike_mult:.1f}× rolling avg",
        ]
        if quote.bid_ask_ratio >= c.bid_ask_imbalance_threshold:
            reasoning.append(f"Bid/Ask ratio {quote.bid_ask_ratio:.1f}× — aggressive buyer at ask")
        if iv_low:
            reasoning.append(f"Low IV rank ({context.iv_rank:.0f}) — debit strategies are cheap")
        if flow_score > 0:
            reasoning.append(f"Repeat sweep detected (flow score: {flow_score:.1f})")
        reasoning.append(f"Regime: {context.regime.regime}  Sentiment: {context.sentiment.label} ({context.sentiment.score:+.0f})")
        if quote.delta != 0.0:
            reasoning.append(f"Delta {quote.delta:+.3f}  Gamma {quote.gamma:.4f}  IV {quote.impl_vol:.1%}")

        wing = quote.strike + 5
        suggested = f"Call Sweep: {quote.strike:.0f}C exp {expiry}"
        if iv_low:
            suggested += f"\nBull Call Spread: Buy {quote.strike:.0f}C / Sell {wing:.0f}C exp {expiry}"
        suggested += f"\nRisk: Exit if SPY loses VWAP (${context.regime.vwap:.2f})"

        sig = SpySignal(
            signal_type=SignalType.CALL_SWEEP,
            strike=quote.strike, expiry=expiry, right="C",
            confidence=conf, spy_price=context.spy_price, vix=context.vix,
            volume=quote.volume, volume_spike_mult=spike_mult,
            bid_size=quote.bid_size, ask_size=quote.ask_size,
            reasoning=reasoning, suggested_trade=suggested,
            flow_score=flow_score,
            structure="LONG",
        )
        self._enrich(sig, quote, context)
        signals.append(sig)

        # BULL_CALL_SPREAD — low IV rank bonus
        if iv_low and quote.volume >= c.min_volume_for_signal * 2:
            conf2 = self._weighted_confidence(quote, "C", spike_mult, flow_score, SignalType.BULL_CALL_SPREAD, context, c)
            sig2 = SpySignal(
                signal_type=SignalType.BULL_CALL_SPREAD,
                strike=quote.strike, expiry=expiry, right="C",
                confidence=conf2, spy_price=context.spy_price, vix=context.vix,
                volume=quote.volume, volume_spike_mult=spike_mult,
                bid_size=quote.bid_size, ask_size=quote.ask_size,
                reasoning=[
                    f"Low IV rank ({context.iv_rank:.0f}) makes debit spreads cost-effective",
                    f"Call spike {spike_mult:.1f}× at {quote.strike:.0f}C",
                    "Capped-risk: buy lower call, sell higher call",
                ],
                suggested_trade=(
                    f"Buy {quote.strike:.0f}C / Sell {wing:.0f}C exp {expiry}\n"
                    f"Max profit if SPY closes above {wing:.0f} at expiry"
                ),
                flow_score=flow_score,
                structure="BULL_CALL_SPREAD",
                short_strike=float(wing),
            )
            self._enrich(sig2, quote, context)
            signals.append(sig2)

        return signals

    def _put_spike_signals(
        self,
        quote: OptionQuote,
        expiry: str,
        spike_mult: float,
        flow_score: float,
        iv_low: bool,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        signals: List[SpySignal] = []
        conf = self._weighted_confidence(quote, "P", spike_mult, flow_score, SignalType.PUT_SWEEP, context, c)

        reasoning = [
            f"Put volume spike: {quote.volume:,} contracts at {quote.strike:.0f}P",
            f"Spike: {spike_mult:.1f}× rolling avg",
        ]
        if quote.ask_bid_ratio >= c.bid_ask_imbalance_threshold:
            reasoning.append(f"Ask/Bid ratio {quote.ask_bid_ratio:.1f}× — aggressive put buyer")
        if iv_low:
            reasoning.append(f"Low IV rank ({context.iv_rank:.0f}) — puts are cheap")
        if flow_score > 0:
            reasoning.append(f"Repeat sweep detected (flow score: {flow_score:.1f})")
        reasoning.append(f"Regime: {context.regime.regime}  Sentiment: {context.sentiment.label} ({context.sentiment.score:+.0f})")
        if quote.delta != 0.0:
            reasoning.append(f"Delta {quote.delta:+.3f}  Gamma {quote.gamma:.4f}  IV {quote.impl_vol:.1%}")

        wing = quote.strike - 5
        suggested = f"Put Sweep: {quote.strike:.0f}P exp {expiry}"
        if iv_low:
            suggested += f"\nBear Put Spread: Buy {quote.strike:.0f}P / Sell {wing:.0f}P exp {expiry}"
        suggested += f"\nRisk: Exit if SPY reclaims VWAP (${context.regime.vwap:.2f}) or VIX spikes"

        sig = SpySignal(
            signal_type=SignalType.PUT_SWEEP,
            strike=quote.strike, expiry=expiry, right="P",
            confidence=conf, spy_price=context.spy_price, vix=context.vix,
            volume=quote.volume, volume_spike_mult=spike_mult,
            bid_size=quote.bid_size, ask_size=quote.ask_size,
            reasoning=reasoning, suggested_trade=suggested,
            flow_score=flow_score,
            structure="LONG",
        )
        self._enrich(sig, quote, context)
        signals.append(sig)

        # BEAR_PUT_SPREAD — low IV rank bonus
        if iv_low and quote.volume >= c.min_volume_for_signal * 2:
            conf2 = self._weighted_confidence(quote, "P", spike_mult, flow_score, SignalType.BEAR_PUT_SPREAD, context, c)
            sig2 = SpySignal(
                signal_type=SignalType.BEAR_PUT_SPREAD,
                strike=quote.strike, expiry=expiry, right="P",
                confidence=conf2, spy_price=context.spy_price, vix=context.vix,
                volume=quote.volume, volume_spike_mult=spike_mult,
                bid_size=quote.bid_size, ask_size=quote.ask_size,
                reasoning=[
                    f"Low IV rank ({context.iv_rank:.0f}) makes debit spreads cost-effective",
                    f"Put spike {spike_mult:.1f}× at {quote.strike:.0f}P",
                    "Capped-risk: buy higher put, sell lower put",
                ],
                suggested_trade=(
                    f"Buy {quote.strike:.0f}P / Sell {wing:.0f}P exp {expiry}\n"
                    f"Max profit if SPY closes below {wing:.0f} at expiry"
                ),
                flow_score=flow_score,
                structure="BEAR_PUT_SPREAD",
                short_strike=float(wing),
            )
            self._enrich(sig2, quote, context)
            signals.append(sig2)

        return signals

    def _straddle_signals(
        self,
        spiked_calls: Set[float],
        spiked_puts: Set[float],
        chain: ChainSnapshot,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        signals: List[SpySignal] = []
        both: Set[float] = spiked_calls & spiked_puts

        # Adjacent: call and put spikes within ±5 strike
        if not both:
            for cs in spiked_calls:
                for ps in spiked_puts:
                    if abs(cs - ps) <= 5:
                        both.add(round((cs + ps) / 2 / 5) * 5)

        # ── Regime / IV gate (May 2026 review fix) ────────────────────────
        # Long straddles need realised-vol expansion: that means a catalyst
        # OR a regime that produces large directional moves. RANGE_BOUND /
        # TRANSITION / LOW_VOL with IVR < 30 is exactly the wrong setup —
        # premium is cheap because the market is *not* expecting a move,
        # and selling vol (not buying it) is the profitable side. Skip
        # generation entirely unless a high-impact catalyst is imminent.
        block_flag = getattr(c, "block_long_straddle_in_range_low_iv", True)
        ext = context.external
        event_imminent = bool(
            ext is not None
            and getattr(ext, "event_risk", False)
            and getattr(ext, "event_minutes", 999.0) <= 60.0
        )
        if block_flag and both:
            quiet_regimes = {"RANGE_BOUND", "TRANSITION", "LOW_VOL"}
            if (
                context.regime.regime in quiet_regimes
                and context.iv_rank < 30
                and not event_imminent
            ):
                logger.info(
                    "Long-straddle gate: skipping {} candidate strike(s) — "
                    "regime={}, IVR={:.0f}, no imminent catalyst → "
                    "compressed-vol environment is unfavourable for long vol",
                    len(both),
                    context.regime.regime, context.iv_rank,
                )
                return []

        # ── Directional-regime gate (May 14 2026 review fix) ──────────────
        # A LONG_STRADDLE is a *vol-expansion* play, not a directional play.
        # If sentiment is strongly directional (|score| > 50) AND the regime
        # is itself directional (TREND_UP / TREND_DOWN), a two-sided volume
        # spike is much more likely to be directional flow + MM hedging than
        # genuine long-vol conviction. The signal would already be served
        # better as a directional sweep. Skip generation unless an imminent
        # catalyst justifies a vol play despite the directional tape.
        directional_regimes = {"TREND_UP", "TREND_DOWN"}
        sentiment_score = float(getattr(context.sentiment, "score", 0.0) or 0.0)
        if (
            both
            and context.regime.regime in directional_regimes
            and abs(sentiment_score) > 50.0
            and not event_imminent
        ):
            logger.info(
                "Long-straddle gate: skipping {} candidate strike(s) — "
                "regime={} with directional sentiment={:+.0f} → two-sided "
                "flow is likely directional, not long-vol conviction",
                len(both),
                context.regime.regime,
                sentiment_score,
            )
            return []

        for strike in both:
            atm = chain.atm_strike(context.spy_price)
            # Straddle confidence (May 14 2026 review fix):
            # A LONG_STRADDLE is direction-agnostic — strongly directional
            # sentiment is *evidence against* the long-vol thesis, so it must
            # actually drag confidence DOWN (the old formula had a 0.78 floor
            # that cleared the manager's 0.50 gate even at score=±100). The
            # new formula: base 0.65, +0.20 for full neutrality, +0.05 if an
            # event catalyst is imminent (justifying a vol play), capped at
            # 0.90. With score=±100 and no catalyst, conf=0.65 — still above
            # the floor but no longer EXTREME-tier, and the gating layer can
            # then react to the self-flagged ambiguity.
            neutrality = 1.0 - abs(sentiment_score) / 100.0
            conf = 0.65 + neutrality * 0.20
            if event_imminent:
                conf += 0.05
            conf = min(conf, 0.90)
            # Look up ATM call + put quotes to compute combined straddle mid/bid/ask
            _atm_call = next((q for q in chain.calls if q.strike == atm), None)
            _atm_put  = next((q for q in chain.puts  if q.strike == atm), None)
            _straddle_bid = (
                (_atm_call.bid + _atm_put.bid)
                if _atm_call and _atm_put and _atm_call.bid > 0 and _atm_put.bid > 0
                else 0.0
            )
            _straddle_ask = (
                (_atm_call.ask + _atm_put.ask)
                if _atm_call and _atm_put and _atm_call.ask > 0 and _atm_put.ask > 0
                else 0.0
            )
            sig = SpySignal(
                signal_type=SignalType.LONG_STRADDLE,
                strike=strike, expiry=chain.expiry_month, right="BOTH",
                confidence=conf, spy_price=context.spy_price, vix=context.vix,
                volume=chain.total_call_volume + chain.total_put_volume,
                volume_spike_mult=c.straddle_spike_mult,
                bid=_straddle_bid, ask=_straddle_ask,
                bid_size=0, ask_size=0,
                # bid/ask already carry the *combined* premium of both legs;
                # short_strike==strike marks "second leg is the put at same strike"
                structure="LONG_STRADDLE",
                short_strike=float(strike),
                reasoning=[
                    "Both call AND put volume spiking simultaneously",
                    # Honest disclosure (May 2026 review fix): without
                    # bid/ask aggressor data we cannot tell directional
                    # conviction from MM hedging or short-straddle opening.
                    # The "[AMBIGUOUS]" prefix is machine-readable — the
                    # trading manager downgrades these in rules.evaluate_spy.
                    "[AMBIGUOUS] Direction ambiguous — could be MM hedging "
                    "or short-straddle opening rather than long-vol conviction",
                    "[AMBIGUOUS] Profits require realised-vol expansion "
                    "(catalyst or breakout); not sufficient on flow alone",
                    f"Total flow: {chain.total_call_volume + chain.total_put_volume:,} contracts",
                    f"Regime: {context.regime.regime}  IV rank: {context.iv_rank:.0f}",
                    f"Sentiment: {context.sentiment.label} ({sentiment_score:+.0f}) — "
                    f"high |score| weakens the long-vol thesis",
                ],
                suggested_trade=(
                    f"Long Straddle: Buy {atm:.0f}C + Buy {atm:.0f}P exp {chain.expiry_month}\n"
                    f"Profit if SPY moves more than combined premium in either direction\n"
                    f"VWAP: ${context.regime.vwap:.2f}"
                ),
            )
            sig.confidence_tier = self._tier_cfg(conf)
            sig.iv_rank = context.iv_rank
            sig.regime = context.regime.regime
            sig.sentiment_score = context.sentiment.score
            sig.sentiment_label = context.sentiment.label
            signals.append(sig)
        return signals

    def _vwap_reversion(
        self,
        chain: ChainSnapshot,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        """Range engine, SHADOW-INCUBATING: fade a 2SD VWAP stretch confirmed
        by RSI extreme, morning window only. Confidence is deliberately below
        min_confidence so this NEVER dispatches — every occurrence lands in the
        shadow book with simulated exits. Promotion to live goes through the
        scorecard (Wilson-lo > breakeven WR at n≥30), not through opinion.
        Backtest basis (n=4, 3W/1L) is directional evidence only."""
        if not getattr(c, "vrev_enabled", False):
            return []
        ext = context.external
        if ext is None:
            return []
        band = getattr(ext, "vwap_band_position", "INSIDE_1SD")
        rsi = float(getattr(ext, "rsi_5m", 50.0) or 50.0)
        if band == "BELOW_2SD" and rsi <= c.vrev_rsi_low:
            right, why = "C", f"BELOW_2SD + RSI {rsi:.0f} ≤ {c.vrev_rsi_low} — fade down-stretch"
        elif band == "ABOVE_2SD" and rsi >= c.vrev_rsi_high:
            right, why = "P", f"ABOVE_2SD + RSI {rsi:.0f} ≥ {c.vrev_rsi_high} — fade up-stretch"
        else:
            return []
        now = _dt.datetime.now(ZoneInfo("America/New_York"))
        h0, m0 = map(int, c.vrev_window_start_et.split(":"))
        h1, m1 = map(int, c.vrev_window_end_et.split(":"))
        if not (_dt.time(h0, m0) <= now.time() <= _dt.time(h1, m1)):
            return []
        last = getattr(self, "_vrev_last_fire", {})
        prev = last.get(right)
        if prev and (now - prev) < _dt.timedelta(minutes=c.vrev_cooldown_min):
            return []
        atm = chain.atm_strike(context.spy_price)
        q = chain.call_at(atm) if right == "C" else chain.put_at(atm)
        sig = SpySignal(
            signal_type=SignalType.VWAP_REVERSION,
            strike=atm, expiry=chain.expiry_month, right=right,
            confidence=float(c.vrev_confidence),
            spy_price=context.spy_price, vix=context.vix,
            volume=(chain.total_call_volume if right == "C" else chain.total_put_volume),
            volume_spike_mult=0.0,
            bid_size=q.bid_size if q else 0, ask_size=q.ask_size if q else 0,
            reasoning=[
                f"VWAP_REVERSION (shadow-incubating): {why}",
                "Log-mined n=4 (3W/1L, +0.15%) — below promotion bar; this "
                "family is measured in the shadow book only",
            ],
            suggested_trade="(shadow-only — not dispatched)",
        )
        sig.confidence_tier = self._tier_cfg(sig.confidence)
        sig.iv_rank = context.iv_rank
        sig.regime = context.regime.regime
        sig.sentiment_score = context.sentiment.score
        sig.sentiment_label = context.sentiment.label
        if q is not None:
            self._enrich(sig, q, context)
        last[right] = now
        self._vrev_last_fire = last
        return [sig]

    def _pc_afternoon_flow(
        self,
        chain: ChainSnapshot,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        """Pilot: extreme afternoon put-flow → BUY ATM PUT (see Rule 9 note).

        Fires at most once per day, only in the configured ET window, only on
        a 1–3 DTE chain (0DTE afternoon theta is the executor-measured trap;
        the near-dated-but-not-expiring chain keeps delta capture with
        survivable decay). Confidence is the backtest-derived constant — it
        represents measured WR, not the additive heuristic stack.
        """
        if not getattr(c, "pcaf_enabled", False):
            return []
        today = _dt.datetime.now(ZoneInfo("America/New_York"))
        if getattr(self, "_pcaf_fired_date", "") == today.strftime("%Y-%m-%d"):
            return []
        h0, m0 = map(int, c.pcaf_window_start_et.split(":"))
        h1, m1 = map(int, c.pcaf_window_end_et.split(":"))
        t = today.time()
        if not (_dt.time(h0, m0) <= t <= _dt.time(h1, m1)):
            return []
        # 1–3 DTE chain only (skip the 0DTE chain when this evaluates on it).
        chain_dte = -1
        if chain.expiry_date:
            try:
                exp_d = _dt.datetime.strptime(chain.expiry_date, "%Y%m%d").date()
                chain_dte = (exp_d - today.date()).days
            except ValueError:
                pass
        if not (1 <= chain_dte <= 3):
            return []
        pc = chain.put_call_ratio
        if pc < c.pcaf_min_pc:
            return []
        if chain.total_put_volume < c.min_volume_for_signal:
            return []
        # Regime guard: edge proven only for VIX 15.9–22.0 (stress 2026-07-19,
        # 24 events). Above the guard = unobserved territory, not proven-bad.
        if context.vix and context.vix > getattr(c, "pcaf_max_vix", 25.0):
            logger.info(
                "PC_AFTERNOON_FLOW suppressed: VIX {:.1f} > {:.0f} (outside "
                "proven regime)", context.vix, c.pcaf_max_vix,
            )
            return []
        atm = chain.atm_strike(context.spy_price)
        q = chain.put_at(atm)
        if q is None or not (q.bid > 0 and q.ask > 0):
            return []
        sig = SpySignal(
            signal_type=SignalType.PC_AFTERNOON_FLOW,
            strike=atm, expiry=chain.expiry_month, right="P",
            confidence=float(c.pcaf_confidence),
            spy_price=context.spy_price, vix=context.vix,
            volume=chain.total_put_volume, volume_spike_mult=pc,
            bid_size=q.bid_size, ask_size=q.ask_size,
            reasoning=[
                f"AFTERNOON FLOW pilot: chain P/C {pc:.2f} ≥ {c.pcaf_min_pc:.1f} "
                f"at {today.strftime('%H:%M')} ET — extreme put flow into close",
                f"Put vol {chain.total_put_volume:,} vs call vol "
                f"{chain.total_call_volume:,}",
                "Backtest (9 sessions, log-mined): ≥14:00 ET events 7W/1L/2S "
                "puts, +0.34% avg favorable — MIDDAY same trigger no edge",
                "Pilot discipline: 1 contract, 1/day, rolling auto-kill",
            ],
            suggested_trade=(
                f"BUY {atm:.0f}P {chain.expiry_month} ({chain_dte} DTE) — "
                "afternoon flow-follow, bracket-managed"
            ),
        )
        sig.confidence_tier = self._tier_cfg(sig.confidence)
        sig.iv_rank = context.iv_rank
        sig.regime = context.regime.regime
        sig.sentiment_score = context.sentiment.score
        sig.sentiment_label = context.sentiment.label
        self._enrich(sig, q, context)
        self._pcaf_fired_date = today.strftime("%Y-%m-%d")
        logger.info(
            "PC_AFTERNOON_FLOW pilot fired: {}P {} P/C={:.2f} dte={} conf={:.0%}",
            atm, chain.expiry_month, pc, chain_dte, sig.confidence,
        )
        return [sig]

    def _orb_breakout_signal(
        self,
        chain: ChainSnapshot,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        """Generate an ORB_BREAKOUT signal on confirmed 30-min range break.

        Confidence model for ORB:
          - Base is 0.72 (reasonable prior — ORB breakouts are high-probability
            but not infallible; ~60-65% historical follow-through on SPY)
          - Boosted by: regime alignment, sentiment, flow confirmation
          - Penalised by: EDR exhaustion, max pain proximity
        The dynamic confidence engine will then apply ORB-specific modifiers
        on top of these via adjustment block 13.

        Guards (APR 9 2026):
          1. Extension guard — block when price > 0.5% beyond ORB boundary
          2. Time decay — penalise ORB signals >90min after ORB established
          3. RSI overbought/oversold — penalise stretched momentum
        """
        ext = context.external
        if ext is None:
            return []

        orb_status = getattr(ext, "orb_status", "INSIDE")
        orb_high   = getattr(ext, "orb_high", None)
        orb_low    = getattr(ext, "orb_low", None)
        orb_width  = getattr(ext, "orb_width_pct", 0.0)

        if orb_status == "ABOVE_ORB":
            right = "C"
        elif orb_status == "BELOW_ORB":
            right = "P"
        else:
            return []

        # ── Guard 1: Extension guard ─────────────────────────────────────────
        # If price has extended too far beyond the ORB boundary, the breakout
        # trade is stale — we'd be chasing, not trading the breakout.
        # Today's loss: SPY was +0.76% ($5.13) above ORB high at 12:39 CST.
        spy = context.spy_price
        if right == "C" and orb_high is not None:
            extension_pct = (spy - orb_high) / orb_high * 100
            if extension_pct > 0.50:
                logger.info(
                    "ORB extension block: SPY ${:.2f} is {:.2f}% above ORB high ${:.2f} "
                    "(max 0.50%) — suppressing stale breakout",
                    spy, extension_pct, orb_high,
                )
                return []
        elif right == "P" and orb_low is not None:
            extension_pct = (orb_low - spy) / orb_low * 100
            if extension_pct > 0.50:
                logger.info(
                    "ORB extension block: SPY ${:.2f} is {:.2f}% below ORB low ${:.2f} "
                    "(max 0.50%) — suppressing stale breakout",
                    spy, extension_pct, orb_low,
                )
                return []

        # ── Guard 2: ORB time gate ───────────────────────────────────────────
        # Backtest (60 sessions, Apr–Jul 2026, SPY 5-min): breakouts entered
        # 10:00–11:00 ET won 60% first-touch; entries 11:00–13:00 won only 25%.
        # The old graduated decay (−3% at 11:30+) was far too soft for a setup
        # that loses 3 out of 4 times — hard gate at 11:30 ET instead.
        _ET = ZoneInfo("America/New_York")
        now_et = _dt.datetime.now(_ET)
        # Default OFF: this duplicated rules_v2 orb_gate's window with a
        # DIFFERENT cutoff (11:30 here vs the v2 window) — ORB paid two time
        # gates in two files. rules_v2 orb_gate is the single authority.
        if c.orb_engine_time_gate_enabled and now_et.time() >= _dt.time(11, 30):
            logger.info(
                "ORB time gate: {} ET is past 11:30 — late breakouts won only "
                "25% in backtest, suppressing",
                now_et.strftime("%H:%M"),
            )
            return []
        # Mild staleness penalty within the allowed window (ORB set at 10:00 ET)
        orb_established_mins = (now_et.hour * 60 + now_et.minute) - 600
        orb_time_penalty = -0.03 if orb_established_mins > 60 else 0.0

        atm = chain.atm_strike(context.spy_price)
        atm_quote = chain.call_at(atm) if right == "C" else chain.put_at(atm)

        # Base confidence: ORB breakouts on SPY have strong follow-through stats.
        # Sentiment, flow, regime alignment are handled by DynamicConfidence engine
        # (blocks 3, 4, 13) to avoid double-counting — only ORB-specific width
        # adjustment is applied here.
        base_conf = 0.72

        # Narrow ORB = more reliable breakout (backtest: <0.20% width → 80%
        # first-touch win, 100% direction-correct at close, +0.31% avg).
        # Wide ORB (>0.60%) popped briefly but closed direction-wrong 100% of
        # the time (−0.84% avg) — a reversal trap, so the signal is suppressed.
        if orb_width > 0.60:
            logger.info(
                "ORB width gate: {:.2f}% > 0.60% — wide-range breakouts closed "
                "direction-wrong 100% in backtest, suppressing",
                orb_width,
            )
            return []
        if orb_width < 0.20:
            base_conf += 0.05

        # Apply ORB time decay
        base_conf += orb_time_penalty

        # ── Guard 3: RSI overbought/oversold penalty ─────────────────────────
        # Buying calls when RSI > 65 (or puts when RSI < 35) means chasing
        # stretched momentum. Today's loss had RSI 69.3 with no penalty.
        rsi = getattr(ext, "rsi_5m", 50.0)
        rsi_penalty = 0.0
        if right == "C" and rsi > 65:
            rsi_penalty = -0.03 if rsi < 70 else -0.06  # -3% mild, -6% strong
        elif right == "P" and rsi < 35:
            rsi_penalty = -0.03 if rsi > 30 else -0.06
        base_conf += rsi_penalty

        conf = max(0.0, min(0.95, base_conf))

        # Build reasoning
        orb_ref = f"${orb_high:.2f}" if orb_high else "n/a"
        orb_ref_low = f"${orb_low:.2f}" if orb_low else "n/a"
        direction_word = "ABOVE" if right == "C" else "BELOW"
        wall = orb_ref if right == "C" else orb_ref_low

        reasoning = [
            f"30-min ORB breakout: SPY closed {direction_word} {wall}",
            f"ORB range: {orb_ref_low} – {orb_ref}  ({orb_width:.2f}% width)",
            f"Regime: {context.regime.regime}  Sentiment: {context.sentiment.label} ({context.sentiment.score:+.0f})",
        ]
        if orb_time_penalty < 0:
            reasoning.append(
                f"⏱ ORB age: {orb_established_mins:.0f}min since 10:00 ET "
                f"({orb_time_penalty:+.0%} decay)"
            )
        if getattr(ext, "rsi_5m", 50.0) != 50.0:
            rsi_tag = (
                " [OVERBOUGHT]" if getattr(ext, "rsi_overbought", False) else
                " [OVERSOLD]"   if getattr(ext, "rsi_oversold",   False) else ""
            )
            if rsi_penalty < 0:
                rsi_tag += f" ({rsi_penalty:+.0%} penalty)"
            reasoning.append(f"RSI (5m): {ext.rsi_5m:.1f}{rsi_tag}")
        if getattr(ext, "near_pivot", False):
            reasoning.append(
                f"Near pivot {ext.pivot_nearest} ({ext.pivot_bias.replace('_', ' ')})"
            )

        wing = (atm + 5) if right == "C" else (atm - 5)
        suggested = (
            f"ORB {'Call' if right == 'C' else 'Put'} Sweep: {atm:.0f}{right} exp {chain.expiry_month}\n"
            f"{'Bull' if right == 'C' else 'Bear'} {'Call' if right == 'C' else 'Put'} Spread: "
            f"Buy {atm:.0f}{right} / Sell {wing:.0f}{right} exp {chain.expiry_month}\n"
            f"Risk: Exit if SPY {'falls back below' if right == 'C' else 'reclaims'} "
            f"{wall}"
        )

        sig = SpySignal(
            signal_type=SignalType.ORB_BREAKOUT,
            strike=atm, expiry=chain.expiry_month, right=right,
            confidence=conf, spy_price=context.spy_price, vix=context.vix,
            volume=chain.total_call_volume if right == "C" else chain.total_put_volume,
            volume_spike_mult=0.0,
            bid_size=atm_quote.bid_size if atm_quote else 0,
            ask_size=atm_quote.ask_size if atm_quote else 0,
            reasoning=reasoning,
            suggested_trade=suggested,
            structure="LONG",
        )

        if atm_quote:
            self._enrich(sig, atm_quote, context)
        else:
            sig.confidence_tier = self._tier_cfg(conf)
            sig.iv_rank = context.iv_rank
            sig.regime = context.regime.regime
            sig.sentiment_score = context.sentiment.score
            sig.sentiment_label = context.sentiment.label

        return [sig]

    def _high_iv_signal(
        self,
        chain: ChainSnapshot,
        context: SignalContext,
        c: SpyOptionsSignalConfig,
    ) -> List[SpySignal]:
        if (chain.total_call_volume + chain.total_put_volume) < c.min_volume_for_signal:
            return []
        atm = chain.atm_strike(context.spy_price)
        # Higher IV rank = stronger signal for premium selling
        conf = min(0.65 + context.iv_rank / 100.0 * 0.20, 0.85)
        sig = SpySignal(
            signal_type=SignalType.HIGH_IV_ALERT,
            strike=atm, expiry=chain.expiry_month, right="BOTH",
            confidence=conf, spy_price=context.spy_price, vix=context.vix,
            volume=chain.total_call_volume + chain.total_put_volume,
            volume_spike_mult=0.0, bid_size=0, ask_size=0,
            # HIGH_IV_ALERT is informational — no concrete long/short legs;
            # leave structure="" so the TM doesn't try to size risk on it.
            reasoning=[
                f"VIX={context.vix:.1f}" if context.vix else "VIX elevated",
                f"IV rank: {context.iv_rank:.0f}/100 — options are expensive",
                "Premium selling strategies (Iron Condor, credit spreads) favoured",
                f"Regime: {context.regime.regime}",
            ],
            suggested_trade=(
                f"Iron Condor or credit spread near {atm:.0f} exp {chain.expiry_month}\n"
                "Sell OTM call + OTM put. Profit if SPY stays range-bound.\n"
                f"VWAP anchor: ${context.regime.vwap:.2f}"
            ),
        )
        sig.confidence_tier = self._tier_cfg(conf)
        sig.iv_rank = context.iv_rank
        sig.regime = context.regime.regime
        sig.sentiment_score = context.sentiment.score
        sig.sentiment_label = context.sentiment.label
        return [sig]
