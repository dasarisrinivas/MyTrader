"""Top-level rules_v2 engine.

A thin orchestrator that wires:

    RegimeV2Detector → regime
    ContinuationDetector → new signal candidates
    OrbGate, PcRatioAlignmentGate, EntryGate → filters
    StructureThrottle → per-session throttle state
    StrikeSelector → delta-based strike choice

The caller (``manager.py``) holds one ``RulesV2Engine`` instance per session.

Public API
----------

``filter(sig, ctx) -> FilterDecision``
    Decide whether ``sig`` may be dispatched. Stateless with respect to
    signal content — only the throttle keeps state across calls, and only
    when ``commit()`` is called.

``commit(sig, ctx)``
    Record the allowed entry into the throttle state.

``generate_additional(bars, regime, ctx) -> List[Candidate]``
    Produce NEW signals that the legacy engine cannot produce today —
    specifically, TREND_CONTINUATION.

All three are intentionally small, easy to wrap with a config feature flag
(the manager only calls them when ``cfg.rules_v2.enabled`` is True).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

try:                                               # pragma: no cover
    from ..utils.logger import logger
except Exception:                                  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

from .config import RulesV2Config
from .continuation import ContinuationCandidate, ContinuationDetector
from .entry_gate import EntryGate, EntryGateResult
from .orb_gate import OrbGate, OrbGateResult
from .pc_ratio_alignment import PcRatioAlignmentGate, PcRatioGateResult
from .regime import RegimeV2Context, RegimeV2Detector, TRANSITION
from .strike_selector import StrikePick, select_strike
from .throttle import StructureThrottle, ThrottleResult


# ─── Public result types ───────────────────────────────────────────────────


@dataclass
class FilterDecision:
    """Verdict for a single candidate signal."""

    allowed: bool
    reason: str
    rule: str                  # "orb_gate" | "pc_ratio" | "entry_gate" | "throttle" | "time_of_day" | "pass"
    leg_id: int = -1
    augmented_strike: Optional[StrikePick] = None


@dataclass
class EngineInputs:
    """Per-poll inputs supplied by the manager."""

    bars: List[Dict]
    spy_price: float
    vix: Optional[float]
    iv_rank: Optional[float]
    orb_high: Optional[float]
    orb_low: Optional[float]
    rsi_5m: Optional[float]
    vwap: Optional[float]
    option_quotes: Optional[List] = None      # for strike selection
    now: Optional[datetime] = None


# ─── Engine ────────────────────────────────────────────────────────────────


class RulesV2Engine:
    def __init__(self, cfg: Optional[RulesV2Config] = None) -> None:
        self._cfg = cfg or RulesV2Config()
        self._regime_det = RegimeV2Detector(self._cfg.regime)
        self._orb_gate = OrbGate(self._cfg.orb_gate)
        self._pc_gate = PcRatioAlignmentGate(self._cfg.pc_ratio_alignment)
        self._entry_gate = EntryGate(self._cfg.entry_gate)
        self._continuation = ContinuationDetector(self._cfg.continuation)
        self._throttle = StructureThrottle(self._cfg.throttle)
        self._last_regime: Optional[RegimeV2Context] = None

    # ─── Session lifecycle ─────────────────────────────────────────────────

    def begin_session(self) -> None:
        """Call once per trading day to flush throttle + pivot state."""
        self._throttle.reset()
        self._last_regime = None

    # ─── Regime ────────────────────────────────────────────────────────────

    def classify_regime(self, bars: List[Dict], spy_price: float) -> RegimeV2Context:
        if not self._cfg.regime_v2_enabled:
            # Neutral passthrough regime so downstream code has something to read
            return RegimeV2Context(
                regime=TRANSITION,
                vwap=spy_price,
                vwap_slope=0.0,
                atr_ratio=1.0,
                pivots_recent=0,
                has_hhhl=False,
                has_lhll=False,
                vwap_crosses_30m=0,
                spy_vs_vwap=0.0,
                timestamp=datetime.utcnow(),
                reasons=["regime_v2 disabled"],
            )
        ctx = self._regime_det.classify(bars, spy_price)
        self._last_regime = ctx
        return ctx

    # ─── New signal generation ─────────────────────────────────────────────

    def generate_additional(
        self,
        bars: List[Dict],
        regime: RegimeV2Context,
        spy_price: float,
        now: Optional[datetime] = None,
    ) -> List[ContinuationCandidate]:
        if not self._cfg.continuation_enabled:
            return []
        cand = self._continuation.detect(bars, regime, spy_price, now=now)
        return [cand] if cand is not None else []

    # ─── Main filter ───────────────────────────────────────────────────────

    def filter(
        self,
        signal_type: str,
        direction: str,                  # "C" or "P"
        price: float,
        confidence: float,
        regime: RegimeV2Context,
        inputs: EngineInputs,
    ) -> FilterDecision:
        """Decide whether this candidate may be dispatched.

        Order matters: cheap checks first (time-of-day, ORB window),
        then alignment, entry quality, throttle.
        """
        cfg = self._cfg

        # ── ORB time + confirmation gate ────────────────────────────────
        if cfg.orb_gate_enabled and signal_type == "ORB_BREAKOUT":
            orb_res = self._orb_gate.check(
                bars=inputs.bars,
                direction=direction,
                orb_high=inputs.orb_high,
                orb_low=inputs.orb_low,
                rsi_value=inputs.rsi_5m,
                vwap_value=inputs.vwap if inputs.vwap is not None else regime.vwap,
                spy_price=price,
                now=inputs.now,
            )
            if not orb_res.allowed:
                return FilterDecision(
                    allowed=False, reason=orb_res.reason, rule="orb_gate"
                )

        # ── PC_RATIO alignment gate ─────────────────────────────────────
        if cfg.pc_ratio_alignment_enabled and signal_type == "PC_RATIO_EXTREME":
            pc_res = self._pc_gate.check(
                direction=direction,
                regime=regime,
                bars=inputs.bars,
                now=inputs.now,
            )
            if not pc_res.allowed:
                return FilterDecision(
                    allowed=False, reason=pc_res.reason, rule="pc_ratio"
                )

        # ── Entry-quality gate (applies to all directional signals) ─────
        if cfg.entry_gate_enabled and direction in ("C", "P"):
            eg = self._entry_gate.check(
                direction=direction,
                bars=inputs.bars,
                spy_price=price,
                confidence=confidence,
                regime=regime,
                signal_type=signal_type,
            )
            if not eg.allowed:
                return FilterDecision(
                    allowed=False, reason=eg.reason, rule="entry_gate"
                )

        # ── Structure-based throttle (last, stateful) ───────────────────
        if cfg.throttle_structure_enabled and direction in ("C", "P"):
            tr = self._throttle.check_allow(
                direction=direction,
                price=price,
                regime_ctx=regime,
                now=inputs.now,
            )
            if not tr.allowed:
                return FilterDecision(
                    allowed=False, reason=tr.reason, rule="throttle", leg_id=tr.leg_id
                )
            return FilterDecision(
                allowed=True,
                reason=f"pass: {tr.reason}",
                rule="pass",
                leg_id=tr.leg_id,
            )

        return FilterDecision(allowed=True, reason="pass (no throttle)", rule="pass")

    # ─── Strike selection ──────────────────────────────────────────────────

    def pick_strike(
        self,
        signal_type: str,
        right: str,
        inputs: EngineInputs,
    ) -> Optional[StrikePick]:
        if not self._cfg.strike_selector_enabled or not inputs.option_quotes:
            return None
        ivr = inputs.iv_rank
        prefer_0dte = ivr is not None and ivr < self._cfg.strike_selector.prefer_0dte_ivr_max
        return select_strike(
            quotes=inputs.option_quotes,
            signal_type=signal_type,
            right=right,
            cfg=self._cfg.strike_selector,
            vix=inputs.vix,
            ivr=ivr,
            prefer_0dte=prefer_0dte,
        )

    # ─── Commit (post-dispatch) ────────────────────────────────────────────

    def commit(
        self, direction: str, price: float, now: Optional[datetime] = None
    ) -> None:
        """Record a successful dispatch into the throttle leg state."""
        if direction in ("C", "P"):
            self._throttle.record_entry(direction, price, now=now)
