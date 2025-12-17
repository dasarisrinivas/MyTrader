"""Helpers for validating and normalizing protective levels."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .atr_module import compute_protective_offsets

TRADE_ACTIONS = {"BUY", "SELL", "SCALP_BUY", "SCALP_SELL"}
SCALP_ACTIONS = {"SCALP_BUY", "SCALP_SELL"}
SELL_ACTIONS = {"SELL", "SCALP_SELL"}


@dataclass
class ProtectionComputation:
    """Encapsulates normalized protection distances and absolute prices."""

    stop_offset: float
    target_offset: float
    stop_price: float
    target_price: float
    source: str
    fallback_reason: str = ""


def _sanitize_offset(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric) or numeric <= 0:
        return 0.0
    return numeric


def calculate_protection(
    action: str,
    entry_price: float,
    stop_points: Optional[float],
    target_points: Optional[float],
    atr_value: Optional[float],
    tick_size: float,
    volatility: Optional[str] = None,
) -> ProtectionComputation:
    """Normalize offsets and convert them into absolute prices."""
    action_upper = (action or "HOLD").upper()
    stop_offset = _sanitize_offset(stop_points)
    target_offset = _sanitize_offset(target_points)
    source = "pipeline"
    fallback_reason = ""

    if action_upper not in TRADE_ACTIONS:
        return ProtectionComputation(
            stop_offset=stop_offset,
            target_offset=target_offset,
            stop_price=entry_price,
            target_price=entry_price,
            source="inactive",
        )

    if stop_offset <= 0 or target_offset <= 0:
        offsets = compute_protective_offsets(
            atr_value=atr_value,
            tick_size=tick_size,
            scalper=action_upper in SCALP_ACTIONS,
            volatility=volatility,
        )
        stop_offset = offsets.stop_offset
        target_offset = offsets.target_offset
        source = "fallback_atr"
        fallback_reason = offsets.reason or "invalid_pipeline_offsets"

    if action_upper in SELL_ACTIONS:
        stop_price = entry_price + stop_offset
        target_price = entry_price - target_offset
    else:
        stop_price = entry_price - stop_offset
        target_price = entry_price + target_offset

    return ProtectionComputation(
        stop_offset=stop_offset,
        target_offset=target_offset,
        stop_price=stop_price,
        target_price=target_price,
        source=source,
        fallback_reason=fallback_reason,
    )


__all__ = ["calculate_protection", "ProtectionComputation"]

