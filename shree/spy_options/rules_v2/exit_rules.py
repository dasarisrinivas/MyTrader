"""Scratch-resistant exit logic.

Root cause #6 on Apr 21: five of six closed trades scratched between −1.4%
and −8.6%. The legacy exit monitor in ``manager._check_exit_conditions``
uses six absolute triggers (adverse 0.5%, regime flip, time stops, etc.),
but nothing forces a *minimum hold* — so the first adverse tick ends
directionally-correct trades.

This module provides pure decision functions callable from the manager's
exit loop. It *complements* the existing logic by answering:
  1. ``can_scratch_now(...)``    → may the current unfavourable price
                                    condition trigger a scratch?
  2. ``take_scale_out(...)``     → should the first 50% be taken now?
  3. ``trail_stop_price(...)``   → where should the trailing stop sit?

If ``can_scratch_now`` returns False, the existing exit triggers must
wait — the rules_v2 hook in the manager suppresses the scratch alert.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .config import ExitRulesConfig


@dataclass
class ScratchDecision:
    allowed: bool
    reason: str


def can_scratch_now(
    direction: str,                  # "C" (bullish) or "P" (bearish)
    entry_price: float,
    current_price: float,
    entry_ts: datetime,
    now: datetime,
    stop_distance: float,            # distance from entry to structural stop (SPY pts)
    cfg: ExitRulesConfig,
) -> ScratchDecision:
    """True once minimum-hold *and* 1R thresholds have both been satisfied.

    The 1R test compares *adverse excursion* against stop_distance — if the
    trade has not even moved 1R adverse, noise is still dominant and a
    scratch here is almost always the wrong move.
    """
    age = now - entry_ts
    if age < timedelta(minutes=cfg.min_hold_minutes):
        return ScratchDecision(
            allowed=False,
            reason=(
                f"min-hold not met: {age.total_seconds()/60:.1f}min "
                f"< {cfg.min_hold_minutes}min"
            ),
        )

    adverse = (
        entry_price - current_price if direction == "C" else current_price - entry_price
    )
    r_multiple = adverse / stop_distance if stop_distance > 0 else 0.0
    if r_multiple < cfg.min_hold_r_multiple:
        return ScratchDecision(
            allowed=False,
            reason=(
                f"min-R not met: adverse={adverse:+.2f} "
                f"= {r_multiple:.2f}R < {cfg.min_hold_r_multiple}R"
            ),
        )

    return ScratchDecision(
        allowed=True,
        reason=f"min-hold+R met (age={age.total_seconds()/60:.1f}min, {r_multiple:.2f}R)",
    )


def take_scale_out(
    direction: str,
    entry_price: float,
    current_price: float,
    stop_distance: float,
    cfg: ExitRulesConfig,
    already_scaled: bool,
) -> bool:
    """Trigger first scale-out once price reaches ``scale_r_multiple`` × R."""
    if already_scaled or stop_distance <= 0:
        return False
    favorable = (
        current_price - entry_price if direction == "C" else entry_price - current_price
    )
    return favorable >= cfg.scale_r_multiple * stop_distance


def trail_stop_price(
    direction: str,
    bars: List[Dict],
    current_stop: float,
    cfg: ExitRulesConfig,
) -> float:
    """Ratcheted trailing stop using the most recent swing pivot.

    For a long (``C``) we sit below the most-recent swing-low; for a short
    (``P``) above the most-recent swing-high. The stop never *retreats* —
    only ratchets in the trade's favour.
    """
    lookback = cfg.trail_pivot_lookback
    if len(bars) < 2 * lookback + 1:
        return current_stop

    recent = bars[-(2 * lookback + 1):]
    if direction == "C":
        swing_low = min(b["low"] for b in recent[:-lookback])
        new_stop = swing_low - 0.01
        return max(current_stop, new_stop)
    else:
        swing_high = max(b["high"] for b in recent[:-lookback])
        new_stop = swing_high + 0.01
        # For a short, lower is better — don't move the stop *up* if the
        # fresh swing-high is above it
        return min(current_stop, new_stop) if current_stop > 0 else new_stop
