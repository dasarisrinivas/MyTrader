"""Dynamic structural support floor — auto-computed from market data.

Instead of hardcoding a price level that goes stale (e.g. 6860 when market
is at 6821), this module derives the support floor from *live* structural
levels the bot already knows:

  1. **Previous Day Low (PDL)** — loaded from IBKR daily bars at startup
  2. **Weekly Low** — lowest low across the daily bars window
  3. **Opening Range Low (OR Low)** — computed by the 15m strategy

The floor is the *lowest* of these levels minus a configurable buffer
(default 5 pts).  This means the floor automatically adjusts each day
as PDL/weekly-low/OR change.

Usage:
  floor = DynamicSupportFloor(buffer_points=5.0)
  floor.update_from_historical_context(manager._historical_context)
  floor.update_or_levels(or_high, or_low)
  level = floor.get_floor()  # float or None

The buffer prevents the floor from sitting exactly at a structural level
(which gets hit by normal noise); instead it sits *below*, acting as a
true "last line of defense".

If all inputs are missing/zero the floor returns None (disabled), which
is the correct behaviour — no data means don't apply an arbitrary block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from ..utils.logger import logger
from ..utils.timezone_utils import now_cst


@dataclass
class DynamicSupportFloorConfig:
    """Configuration knobs for the dynamic floor."""

    # Points below the lowest structural level
    buffer_points: float = 5.0
    # Include each source in the floor calculation?
    use_pdl: bool = True
    use_weekly_low: bool = True
    use_or_low: bool = True
    # Minimum number of valid levels required to compute a floor.
    # Set to 1 = any single level is enough; set to 2 = need at least two.
    min_sources: int = 1
    # Logging
    log_updates: bool = True


class DynamicSupportFloor:
    """Auto-computes structural support floor from available market data.

    Thread-safe — all state mutations are simple attribute assignments
    and the floor is always computed on read.
    """

    def __init__(self, config: Optional[DynamicSupportFloorConfig] = None):
        self.config = config or DynamicSupportFloorConfig()
        # Raw levels (0.0 = not yet set)
        self._pdl: float = 0.0
        self._weekly_low: float = 0.0
        self._or_low: float = 0.0
        # Metadata
        self._last_update_ts: Optional[datetime] = None
        self._last_floor: Optional[float] = None

    # ------------------------------------------------------------------
    # Updaters — called when new data arrives
    # ------------------------------------------------------------------

    def update_from_historical_context(self, context: Dict) -> None:
        """Pull PDL and weekly low from LiveTradingManager._historical_context.

        Expected shape::

            {
                'previous_day': {'high': ..., 'low': ..., 'close': ...},
                'weekly': {'high': ..., 'low': ...},
            }
        """
        if not context:
            return

        prev_day = context.get("previous_day", {})
        pdl = prev_day.get("low", 0.0)
        if pdl and pdl > 0:
            self._pdl = float(pdl)

        weekly = context.get("weekly", {})
        wl = weekly.get("low", 0.0)
        if wl and wl > 0:
            self._weekly_low = float(wl)

        self._recompute("historical_context")

    def update_or_levels(self, or_high: float, or_low: float) -> None:
        """Called when the 15m strategy finishes computing the opening range."""
        if or_low and or_low > 0:
            self._or_low = float(or_low)
            self._recompute("or_levels")

    def update_pdl(self, pdl: float) -> None:
        """Direct setter for previous day low (e.g. from a daily bar close)."""
        if pdl and pdl > 0:
            self._pdl = float(pdl)
            self._recompute("pdl_direct")

    def update_weekly_low(self, weekly_low: float) -> None:
        """Direct setter for weekly low."""
        if weekly_low and weekly_low > 0:
            self._weekly_low = float(weekly_low)
            self._recompute("weekly_low_direct")

    def reset(self) -> None:
        """Reset all levels (e.g. on daily reset)."""
        self._pdl = 0.0
        self._weekly_low = 0.0
        self._or_low = 0.0
        self._last_floor = None
        self._last_update_ts = None

    # ------------------------------------------------------------------
    # Reader — used by order_coordinator and exit_manager
    # ------------------------------------------------------------------

    def get_floor(self) -> Optional[float]:
        """Return the current dynamic support floor, or None if insufficient data."""
        return self._last_floor

    def get_diagnostics(self) -> Dict:
        """Return full state for logging / Telegram / Prometheus."""
        return {
            "pdl": self._pdl,
            "weekly_low": self._weekly_low,
            "or_low": self._or_low,
            "buffer_points": self.config.buffer_points,
            "floor": self._last_floor,
            "last_update": self._last_update_ts.isoformat() if self._last_update_ts else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recompute(self, source: str) -> None:
        """Recompute the floor from all available levels."""
        candidates = []
        if self.config.use_pdl and self._pdl > 0:
            candidates.append(("PDL", self._pdl))
        if self.config.use_weekly_low and self._weekly_low > 0:
            candidates.append(("WL", self._weekly_low))
        if self.config.use_or_low and self._or_low > 0:
            candidates.append(("OR_L", self._or_low))

        if len(candidates) < self.config.min_sources:
            self._last_floor = None
            return

        # Floor = lowest level minus buffer
        lowest_label, lowest_val = min(candidates, key=lambda x: x[1])
        new_floor = round(lowest_val - self.config.buffer_points, 2)

        old_floor = self._last_floor
        self._last_floor = new_floor
        self._last_update_ts = now_cst()

        if self.config.log_updates and new_floor != old_floor:
            sources_str = ", ".join(f"{lbl}={val:.2f}" for lbl, val in candidates)
            logger.info(
                f"🛡️ Dynamic support floor updated: {new_floor:.2f} "
                f"(lowest={lowest_label}={lowest_val:.2f} - {self.config.buffer_points}pt buffer) "
                f"[sources: {sources_str}] trigger={source}"
            )
