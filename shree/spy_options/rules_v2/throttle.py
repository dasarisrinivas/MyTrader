"""Structure-based throttle — replacement for the legacy 90-min directional cap.

Root cause #3 on Apr 21: the 90-minute rolling counter blocked the highest-
quality signals (12:07 PM onward, when the trend was actually developing)
and then unblocked *after* the trend was exhausted, letting afternoon ORB
entries through at the worst possible moment.

This module's primitives:

  • Session leg     — a directional run between two opposing pivots.
                      A new HH in TREND_UP starts a new leg. A new LL in
                      TREND_DOWN starts a new leg.
  • Zone lock       — no new entry within ±zone_lock_pct of any prior entry
                      that is still in the current leg.
  • Leg cap         — max N entries per leg.
  • Regime reset    — any regime flip to TRANSITION or opposite trend flushes
                      the leg state.

State per session:
    prior_entries: List[Entry]      # all entries this leg
    current_leg_id: int             # increments on new pivot
    anchor_pivot_price: Optional[float]
    last_regime: str

All times are timezone-aware; the caller may pass `now`, otherwise current
Central time is used via ``now_cst()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from ...utils.timezone_utils import now_cst
from .config import ThrottleConfig
from .regime import RegimeV2Context, TRANSITION, TREND_DOWN, TREND_UP


@dataclass
class _Entry:
    leg_id: int
    direction: str
    price: float
    ts: datetime


@dataclass
class ThrottleResult:
    allowed: bool
    reason: str
    leg_id: int


class StructureThrottle:
    """Per-instance (per-session) state. Create one at session-start.

    The caller flows the latest RegimeV2Context + entry candidate through
    ``check_allow``; if allowed, call ``record_entry`` after the signal is
    dispatched.
    """

    def __init__(self, cfg: Optional[ThrottleConfig] = None) -> None:
        self._cfg = cfg or ThrottleConfig()
        self._entries: List[_Entry] = []
        self._leg_id: int = 0
        self._anchor_pivot_price: Optional[float] = None
        self._last_regime: Optional[str] = None
        self._last_pivot_price: Optional[float] = None

    # ─── State helpers ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Flush all state (call at session start or midnight rollover)."""
        self._entries.clear()
        self._leg_id = 0
        self._anchor_pivot_price = None
        self._last_regime = None
        self._last_pivot_price = None

    def _flush_leg(self, reason: str = "regime flip") -> None:
        self._entries.clear()
        self._leg_id += 1
        self._anchor_pivot_price = None

    def _update_regime(self, regime_ctx: RegimeV2Context) -> None:
        """Detect regime flips that should flush the leg."""
        cur = regime_ctx.regime
        prev = self._last_regime
        if prev is None:
            self._last_regime = cur
            return

        if cur == prev:
            return

        if self._cfg.reset_on_regime_flip:
            # Flush on any transition-in or opposite-trend
            if cur == TRANSITION or (
                (prev == TREND_UP and cur == TREND_DOWN)
                or (prev == TREND_DOWN and cur == TREND_UP)
            ):
                self._flush_leg("regime flip")

        self._last_regime = cur

    def _update_pivot(self, regime_ctx: RegimeV2Context, latest_price: float) -> None:
        """Detect a new structural pivot in the current regime's direction."""
        # Simple proxy for "new pivot": in TREND_UP a higher last-bar high
        # beyond the anchor is a new pivot; in TREND_DOWN a lower low.
        if regime_ctx.regime == TREND_UP:
            if self._anchor_pivot_price is None:
                self._anchor_pivot_price = latest_price
            elif latest_price > self._anchor_pivot_price:
                self._leg_id += 1
                self._anchor_pivot_price = latest_price
                self._last_pivot_price = latest_price
                self._entries.clear()
        elif regime_ctx.regime == TREND_DOWN:
            if self._anchor_pivot_price is None:
                self._anchor_pivot_price = latest_price
            elif latest_price < self._anchor_pivot_price:
                self._leg_id += 1
                self._anchor_pivot_price = latest_price
                self._last_pivot_price = latest_price
                self._entries.clear()

    # ─── Public API ────────────────────────────────────────────────────────

    def check_allow(
        self,
        direction: str,
        price: float,
        regime_ctx: RegimeV2Context,
        now: Optional[datetime] = None,
    ) -> ThrottleResult:
        """Return a ThrottleResult; does NOT mutate state.

        Call ``record_entry`` after dispatch if allowed == True.
        """
        self._update_regime(regime_ctx)
        self._update_pivot(regime_ctx, price)

        cfg = self._cfg

        # Zone lock — any prior entry within ±zone_lock_pct blocks a new one.
        for e in self._entries:
            if abs(price - e.price) / max(price, 1e-9) < cfg.zone_lock_pct:
                return ThrottleResult(
                    allowed=False,
                    reason=(
                        f"zone-lock: prior entry at {e.price:.2f} within "
                        f"±{cfg.zone_lock_pct*100:.2f}% of {price:.2f}"
                    ),
                    leg_id=self._leg_id,
                )

        # Per-leg cap
        leg_entries = [e for e in self._entries if e.leg_id == self._leg_id]
        if len(leg_entries) >= cfg.max_entries_per_leg:
            return ThrottleResult(
                allowed=False,
                reason=(
                    f"leg cap: {len(leg_entries)}/{cfg.max_entries_per_leg} "
                    f"entries already taken in leg#{self._leg_id}"
                ),
                leg_id=self._leg_id,
            )

        return ThrottleResult(
            allowed=True,
            reason=f"new zone (leg#{self._leg_id}, {len(leg_entries)} entries prior)",
            leg_id=self._leg_id,
        )

    def record_entry(
        self, direction: str, price: float, now: Optional[datetime] = None
    ) -> None:
        """Commit an entry to the current leg. Must be called after dispatch."""
        self._entries.append(
            _Entry(
                leg_id=self._leg_id,
                direction=direction,
                price=price,
                ts=now or now_cst(),
            )
        )
