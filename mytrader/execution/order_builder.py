"""Utilities for validating and constructing bracket orders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BracketValidation:
    """Result of validating a proposed stop-loss / take-profit bracket."""

    stop_loss: Optional[float]
    take_profit: Optional[float]
    valid: bool
    reason: Optional[str] = None
    adjusted_fields: Tuple[str, ...] = ()


def validate_bracket_prices(
    action: str,
    entry_price: Optional[float],
    stop_loss: Optional[float],
    take_profit: Optional[float],
    tick_size: float,
    min_distance_ticks: int = 1,
) -> BracketValidation:
    """
    Ensure protective orders are on the correct side of entry and respect tick spacing.

    Args:
        action: Executed action ("BUY"/"SELL").
        entry_price: Price used to determine bracket orientation.
        stop_loss: Proposed stop price.
        take_profit: Proposed limit target.
        tick_size: Minimum tick size for the instrument.
        min_distance_ticks: Minimum distance in ticks (default 4 = 1 point for ES).

    Returns:
        BracketValidation containing potentially adjusted prices.
    """
    if entry_price is None or entry_price <= 0:
        return BracketValidation(stop_loss, take_profit, False, "Missing entry price for bracket validation")

    normalized_action = action.upper()
    if normalized_action not in ("BUY", "SELL"):
        raise ValueError(f"Unsupported action for bracket validation: {action}")

    # Disallow zero/negative protective levels to avoid instant fills.
    for label, value in (("stop_loss", stop_loss), ("take_profit", take_profit)):
        if value is not None and value <= 0:
            return BracketValidation(stop_loss, take_profit, False, f"{label} must be positive")

    is_buy = normalized_action == "BUY"
    validations = []

    if stop_loss is not None:
        if is_buy and stop_loss >= entry_price:
            return BracketValidation(stop_loss, take_profit, False, "Stop-loss must be below entry for BUY")
        if not is_buy and stop_loss <= entry_price:
            return BracketValidation(stop_loss, take_profit, False, "Stop-loss must be above entry for SELL")
        validations.append("stop_loss")

    if take_profit is not None:
        if is_buy and take_profit <= entry_price:
            return BracketValidation(stop_loss, take_profit, False, "Take-profit must be above entry for BUY")
        if not is_buy and take_profit >= entry_price:
            return BracketValidation(stop_loss, take_profit, False, "Take-profit must be below entry for SELL")
        validations.append("take_profit")

    adjusted: list[str] = []
    # Enforce minimum distance in ticks to prevent immediate fills
    min_distance = max(tick_size * min_distance_ticks, tick_size, 1e-6)

    def _clamp(value: float, prefer_above: bool) -> float:
        distance = abs(value - entry_price)
        if distance < min_distance:
            adjusted.append("take_profit" if prefer_above else "stop_loss")
            return entry_price + min_distance if prefer_above else entry_price - min_distance
        return value

    if take_profit is not None:
        take_profit = _clamp(take_profit, prefer_above=is_buy)
    if stop_loss is not None:
        stop_loss = _clamp(stop_loss, prefer_above=not is_buy)

    return BracketValidation(
        stop_loss=stop_loss,
        take_profit=take_profit,
        valid=True,
        adjusted_fields=tuple(adjusted),
    )


def format_bracket_snapshot(
    entry_price: Optional[float],
    stop_loss: Optional[float],
    take_profit: Optional[float],
    fallback_used: Optional[bool] = None,
) -> str:
    """Human-readable snapshot for telemetry logging."""
    def _fmt(value: Optional[float]) -> str:
        return f"{value:.2f}" if value is not None else "NA"

    fallback = "yes" if fallback_used else "no"
    return f"entry={_fmt(entry_price)} SL={_fmt(stop_loss)} TP={_fmt(take_profit)} fallback={fallback}"
