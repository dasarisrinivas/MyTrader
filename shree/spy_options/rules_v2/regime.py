"""Regime classifier v2 — trend/range/transition with VWAP + structure gates.

Replaces the legacy ``RegimeDetector`` output (string regime) with a richer
``RegimeV2Context`` that exposes the individual votes (slope, structure, ATR)
so downstream logic can make finer decisions.

The legacy ``RegimeContext`` is still populated and passed through for
backwards compatibility — v2 is additive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from ...utils.timezone_utils import now_cst
from . import structure as _s
from .config import RegimeV2Config


# ─── Regime constants ──────────────────────────────────────────────────────

TREND_UP = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
RANGE_BOUND = "RANGE_BOUND"
TRANSITION = "TRANSITION"

VALID_REGIMES_V2 = {TREND_UP, TREND_DOWN, RANGE_BOUND, TRANSITION}


@dataclass
class RegimeV2Context:
    """Extended regime output used by rules_v2 layer."""

    regime: str
    vwap: float
    vwap_slope: float        # fraction-of-price per bar
    atr_ratio: float         # ATR(5) / ATR(20)
    pivots_recent: int
    has_hhhl: bool
    has_lhll: bool
    vwap_crosses_30m: int
    spy_vs_vwap: float
    timestamp: datetime
    reasons: List[str] = field(default_factory=list)


class RegimeV2Detector:
    """Stateless v2 regime classifier.

    Contract:
      - bars: list of 5m bar dicts, newest last
      - spy_price: current last price (used for vwap distance)
      - cfg: RegimeV2Config thresholds

    Decision tree (short-circuits):
      1. If bars < cfg.min_bars → TRANSITION
      2. Compute VWAP slope, ATR ratio, pivots, VWAP crosses
      3. TREND_UP  if slope>τ, HH+HL, ATR expanding, price>VWAP
      4. TREND_DOWN mirror
      5. RANGE_BOUND if |slope|<τ_range AND (crosses≥k OR ATR contracting)
      6. Else TRANSITION
    """

    def __init__(self, cfg: Optional[RegimeV2Config] = None) -> None:
        self._cfg = cfg or RegimeV2Config()

    def classify(self, bars: List[dict], spy_price: float) -> RegimeV2Context:
        cfg = self._cfg
        now = now_cst()

        if len(bars) < cfg.min_bars:
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
                timestamp=now,
                reasons=[f"insufficient bars ({len(bars)} < {cfg.min_bars})"],
            )

        vwap = _s.session_vwap(bars) or spy_price
        slope = _s.vwap_slope(bars, lookback=5)

        atr5 = _s.atr(bars, 5) or 0.0
        atr20 = _s.atr(bars, 20) or atr5
        atr_ratio = (atr5 / atr20) if atr20 > 0 else 1.0

        pivots = _s.detect_pivots(bars, lookback=3)
        pivots_recent = sum(1 for p in pivots if p.idx >= len(bars) - 10)
        has_hhhl = _s.has_higher_highs_and_lows(pivots, min_count=cfg.trend_pivot_count)
        has_lhll = _s.has_lower_highs_and_lows(pivots, min_count=cfg.trend_pivot_count)

        # 5m bars * 6 = 30 min of lookback for cross count
        crosses = _s.count_vwap_crosses(bars, lookback_bars=6)

        spy_vs_vwap = spy_price - vwap

        reasons: List[str] = []

        # ── TREND_UP ────────────────────────────────────────────────────────
        if (
            slope > cfg.trend_vwap_slope
            and has_hhhl
            and atr_ratio >= cfg.trend_atr_expansion
            and spy_vs_vwap > 0
        ):
            reasons.append(
                f"slope={slope:.4f}>τ, HH/HL, atr_ratio={atr_ratio:.2f}>{cfg.trend_atr_expansion}"
            )
            return RegimeV2Context(
                regime=TREND_UP,
                vwap=vwap,
                vwap_slope=slope,
                atr_ratio=atr_ratio,
                pivots_recent=pivots_recent,
                has_hhhl=has_hhhl,
                has_lhll=has_lhll,
                vwap_crosses_30m=crosses,
                spy_vs_vwap=spy_vs_vwap,
                timestamp=now,
                reasons=reasons,
            )

        # ── TREND_DOWN ──────────────────────────────────────────────────────
        if (
            slope < -cfg.trend_vwap_slope
            and has_lhll
            and atr_ratio >= cfg.trend_atr_expansion
            and spy_vs_vwap < 0
        ):
            reasons.append(
                f"slope={slope:.4f}<-τ, LH/LL, atr_ratio={atr_ratio:.2f}>{cfg.trend_atr_expansion}"
            )
            return RegimeV2Context(
                regime=TREND_DOWN,
                vwap=vwap,
                vwap_slope=slope,
                atr_ratio=atr_ratio,
                pivots_recent=pivots_recent,
                has_hhhl=has_hhhl,
                has_lhll=has_lhll,
                vwap_crosses_30m=crosses,
                spy_vs_vwap=spy_vs_vwap,
                timestamp=now,
                reasons=reasons,
            )

        # ── RANGE_BOUND ─────────────────────────────────────────────────────
        range_votes = 0
        if abs(slope) < cfg.range_slope_max:
            range_votes += 1
            reasons.append(f"|slope|={abs(slope):.4f}<{cfg.range_slope_max}")
        if crosses >= cfg.range_vwap_crosses:
            range_votes += 1
            reasons.append(f"crosses={crosses}>={cfg.range_vwap_crosses}")
        if atr_ratio < cfg.range_atr_contraction:
            range_votes += 1
            reasons.append(f"atr_ratio={atr_ratio:.2f}<{cfg.range_atr_contraction}")

        if range_votes >= 2:
            return RegimeV2Context(
                regime=RANGE_BOUND,
                vwap=vwap,
                vwap_slope=slope,
                atr_ratio=atr_ratio,
                pivots_recent=pivots_recent,
                has_hhhl=has_hhhl,
                has_lhll=has_lhll,
                vwap_crosses_30m=crosses,
                spy_vs_vwap=spy_vs_vwap,
                timestamp=now,
                reasons=reasons,
            )

        # ── TRANSITION ──────────────────────────────────────────────────────
        reasons.append(
            f"transition: slope={slope:.4f} atr_ratio={atr_ratio:.2f} "
            f"crosses={crosses} hhhl={has_hhhl} lhll={has_lhll}"
        )
        return RegimeV2Context(
            regime=TRANSITION,
            vwap=vwap,
            vwap_slope=slope,
            atr_ratio=atr_ratio,
            pivots_recent=pivots_recent,
            has_hhhl=has_hhhl,
            has_lhll=has_lhll,
            vwap_crosses_30m=crosses,
            spy_vs_vwap=spy_vs_vwap,
            timestamp=now,
            reasons=reasons,
        )
