"""Expected-move gate.

Rejects directional debit signals where the IV-implied expected move does
not comfortably exceed the leg's own debit. Rationale: a long C/P needs
the underlying to traverse one debit's worth of distance just to break
even at expiry — if EM doesn't justify that, the option is priced richer
than the move it implicitly forecasts.

Inputs come from ``SpySignal`` already (no new market-data plumbing):
    impl_vol  -> per-leg IV from IB greeks
    bid, ask  -> leg quotes
    dte       -> days to expiry
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .config import ExpectedMoveGateConfig


@dataclass
class ExpectedMoveGateResult:
    allowed: bool
    reason: str
    expected_move: Optional[float] = None
    leg_mid: Optional[float] = None


class ExpectedMoveGate:
    """Gate a long-debit signal on EM vs. leg debit.

    Skips (returns allowed=True) when inputs are missing — the gate is
    additive and must never block on plumbing gaps.
    """

    def __init__(self, cfg: Optional[ExpectedMoveGateConfig] = None) -> None:
        self._cfg = cfg or ExpectedMoveGateConfig()

    def check(
        self,
        spy_price: float,
        leg_iv: Optional[float],
        leg_mid: Optional[float],
        dte: int,
        direction: str,
    ) -> ExpectedMoveGateResult:
        # ── Skip-on-missing-data ────────────────────────────────────────
        if leg_iv is None or leg_iv <= 0:
            return ExpectedMoveGateResult(True, "skip: no leg IV")
        if leg_mid is None or leg_mid <= 0:
            return ExpectedMoveGateResult(True, "skip: no leg mid")
        if direction == "BOTH":
            # BOTH (straddle) handled separately — Phase 1 is single-leg only.
            return ExpectedMoveGateResult(True, "skip: BOTH not handled in Phase 1")
        if spy_price <= 0:
            return ExpectedMoveGateResult(True, "skip: no spy price")

        # ── Compute EM and threshold ────────────────────────────────────
        days = max(dte, 1) / 365.0
        em = spy_price * leg_iv * math.sqrt(days)
        threshold = leg_mid * self._cfg.em_multiplier

        if em < threshold:
            return ExpectedMoveGateResult(
                allowed=False,
                reason=(
                    f"EM ${em:.2f} < {self._cfg.em_multiplier:.2f}x leg mid "
                    f"${leg_mid:.2f} (need >= ${threshold:.2f}); "
                    f"IV={leg_iv:.3f} dte={dte}"
                ),
                expected_move=em,
                leg_mid=leg_mid,
            )

        return ExpectedMoveGateResult(
            allowed=True,
            reason=f"EM ${em:.2f} >= ${threshold:.2f}",
            expected_move=em,
            leg_mid=leg_mid,
        )
