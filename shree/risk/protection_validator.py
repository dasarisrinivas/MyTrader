"""Helpers for validating and normalizing protective levels."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .atr_module import compute_protective_offsets
from .trade_math import enforce_min_take_profit, get_contract_spec
from .trade_math import TradingMode

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
    trading_mode: TradingMode = "paper",
    symbol: str = "MES",
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
            current_price=entry_price,
        )
        stop_offset = offsets.stop_offset
        target_offset = offsets.target_offset
        source = "fallback_atr"
        fallback_reason = offsets.reason or "invalid_pipeline_offsets"

    # Scalp normalization: for SCALP_* actions, keep brackets tight by default.
    # This applies even when the pipeline provided offsets (source='pipeline'), so the
    # Hyper/LLM layer can't accidentally suggest swing-style 4/8 point brackets for MES scalps.
    if action_upper in SCALP_ACTIONS:
        # Hard minimum stop offset (avoid 1-2 tick noise stops). Use 4 ticks by default.
        min_stop_offset = max(tick_size * 4, tick_size)
        stop_offset = max(stop_offset, min_stop_offset)

        # For scalps, target can be tighter than stop, but should at least be a viable distance.
        min_target_offset = max(tick_size * 4, tick_size)
        target_offset = max(target_offset, min_target_offset)

    if action_upper in SELL_ACTIONS:
        stop_price = entry_price + stop_offset
        target_price = entry_price - target_offset
    else:
        stop_price = entry_price - stop_offset
        target_price = entry_price + target_offset

    # Live-mode enforcement: ensure TP is not too small to be worth placing in real trading.
    # We keep this local and dependency-light by inferring MES/ES spec via known defaults.
    try:
        spec = get_contract_spec(symbol)
        ok, min_points = enforce_min_take_profit(
            entry_price=entry_price,
            take_profit=target_price,
            spec=spec,
            mode=trading_mode,
            action=action_upper,
        )
        if not ok:
            if action_upper in SELL_ACTIONS:
                target_price = entry_price - float(min_points)
                target_offset = float(min_points)
            else:
                target_price = entry_price + float(min_points)
                target_offset = float(min_points)
    except Exception:
        # If config is missing (or running in paper), skip enforcement.
        pass

    return ProtectionComputation(
        stop_offset=stop_offset,
        target_offset=target_offset,
        stop_price=stop_price,
        target_price=target_price,
        source=source,
        fallback_reason=fallback_reason,
    )


__all__ = ["calculate_protection", "ProtectionComputation"]
