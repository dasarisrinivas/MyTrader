"""ATR helpers for deriving protective stop/target offsets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ATRProtectiveOffsets:
    stop_offset: float
    target_offset: float
    fallback_used: bool = False
    reason: str = ""


def _min_ticks_for_volatility(volatility: Optional[str], scalper: bool) -> float:
    """Map qualitative volatility to minimum tick distances."""
    bucket = (volatility or "MEDIUM").upper()
    mapping = {"LOW": 6.0, "MEDIUM": 8.0, "HIGH": 12.0}
    base = mapping.get(bucket, 8.0)
    if scalper:
        return max(4.0, base * 0.5)
    return base


def compute_protective_offsets(
    atr_value: Optional[float],
    tick_size: float,
    scalper: bool = False,
    volatility: Optional[str] = None,
) -> ATRProtectiveOffsets:
    """
    Convert ATR to protective offsets while guaranteeing non-zero results.

    Ensures offsets remain at least one volatility-aware tick distance even
    when ATR is unavailable.
    """
    min_ticks = _min_ticks_for_volatility(volatility, scalper)
    min_distance = max(tick_size * min_ticks, tick_size)
    fallback_reason = ""
    fallback_used = False

    atr_input = atr_value or 0.0
    atr_threshold = tick_size * 0.25

    if atr_input <= atr_threshold:
        fallback_used = True
        fallback_reason = "ATR unavailable" if atr_input == 0 else "ATR below threshold"
        stop_offset = min_distance
        reward_mult = 1.25 if scalper else 2.0
        target_offset = min_distance * reward_mult
    else:
        stop_mult = 0.75 if scalper else 1.5
        target_mult = 1.0 if scalper else 2.0
        stop_offset = atr_input * stop_mult
        target_offset = atr_input * target_mult

        if stop_offset < min_distance:
            fallback_used = True
            fallback_reason = "ATR distance below min tick distance"
            stop_offset = min_distance
        if target_offset <= stop_offset:
            target_offset = stop_offset + tick_size * max(1.0, min_ticks * 0.25)

    return ATRProtectiveOffsets(
        stop_offset=stop_offset,
        target_offset=target_offset,
        fallback_used=fallback_used,
        reason=fallback_reason,
    )
