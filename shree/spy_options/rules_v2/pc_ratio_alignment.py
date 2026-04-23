"""PC_RATIO_EXTREME alignment gate.

Root cause #3 on Apr 21: PC_RATIO_EXTREME fired at 15:08 and 16:43 *against*
trend exhaustion — contrarian entries at local swing extremes. That's a
textbook way to scratch a directionally-correct regime thesis.

This gate keeps the existing PC_RATIO_EXTREME signal but only lets it
through when:

  • direction agrees with current regime (bearish PC + TREND_DOWN, etc.)
  • an expansion bar occurred in the trend direction within lookback
  • regime is not RANGE_BOUND (contrarian PCs in chop = noise)

Everything else is dropped with an explicit reason for analytics.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from . import structure as _s
from .config import PcRatioAlignmentConfig
from .regime import RANGE_BOUND, TREND_DOWN, TREND_UP, TRANSITION, RegimeV2Context


@dataclass
class PcRatioGateResult:
    allowed: bool
    reason: str


class PcRatioAlignmentGate:
    def __init__(self, cfg: Optional[PcRatioAlignmentConfig] = None) -> None:
        self._cfg = cfg or PcRatioAlignmentConfig()

    def check(
        self,
        direction: str,                 # "C" (bullish PC) or "P" (bearish PC)
        regime: RegimeV2Context,
        bars: List[Dict],
        now: Optional[datetime] = None,
    ) -> PcRatioGateResult:
        cfg = self._cfg

        if cfg.suppress_in_range and regime.regime == RANGE_BOUND:
            return PcRatioGateResult(
                allowed=False,
                reason=f"PC_RATIO suppressed in RANGE_BOUND regime",
            )
        if regime.regime == TRANSITION:
            return PcRatioGateResult(
                allowed=False,
                reason="PC_RATIO suppressed in TRANSITION regime",
            )

        if cfg.require_trend_alignment:
            if direction == "P" and regime.regime != TREND_DOWN:
                return PcRatioGateResult(
                    allowed=False,
                    reason=f"PC_RATIO bearish not aligned with regime={regime.regime}",
                )
            if direction == "C" and regime.regime != TREND_UP:
                return PcRatioGateResult(
                    allowed=False,
                    reason=f"PC_RATIO bullish not aligned with regime={regime.regime}",
                )

        if cfg.require_expansion:
            # Expansion lookback in 5m bars
            lookback_bars = max(1, cfg.expansion_lookback_min // 5)
            recent = bars[-lookback_bars:]
            atr_val = _s.atr(bars, 14) or 0.0
            if atr_val <= 0:
                return PcRatioGateResult(
                    allowed=False, reason="no ATR baseline for expansion check"
                )

            if direction == "P":
                expansion_in_dir = any(
                    _s.expansion_bar(b, atr_val, cfg.expansion_atr_multiple)
                    and b["close"] < b["open"]
                    for b in recent
                )
            else:
                expansion_in_dir = any(
                    _s.expansion_bar(b, atr_val, cfg.expansion_atr_multiple)
                    and b["close"] > b["open"]
                    for b in recent
                )
            if not expansion_in_dir:
                return PcRatioGateResult(
                    allowed=False,
                    reason=(
                        f"PC_RATIO {direction}: no expansion bar "
                        f">{cfg.expansion_atr_multiple}×ATR in last "
                        f"{cfg.expansion_lookback_min}min"
                    ),
                )

        return PcRatioGateResult(
            allowed=True,
            reason=f"PC_RATIO {direction} aligned with {regime.regime}",
        )
