"""Centralized futures trade math utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

from ..config import TradingConfig

TradingMode = Literal["paper", "live"]


@dataclass(frozen=True)
class ContractSpec:
    """Describes tick/point math and safety rails for a futures contract."""

    root_symbol: str
    point_value: float  # Dollars per 1.0 point move
    tick_size: float
    min_take_profit_points_live: float
    live_commission_per_side: Optional[float] = None
    paper_commission_per_side: float = 0.0


_KNOWN_SPECS: Dict[str, ContractSpec] = {
    "MES": ContractSpec(
        root_symbol="MES",
        point_value=5.0,
        tick_size=0.25,
        min_take_profit_points_live=1.25,
        live_commission_per_side=1.0,
    ),
    "ES": ContractSpec(
        root_symbol="ES",
        point_value=50.0,
        tick_size=0.25,
        min_take_profit_points_live=0.25,
        live_commission_per_side=2.25,
    ),
}


@dataclass(frozen=True)
class ExpectedOutcome:
    """Represents projected profit metrics for a potential trade."""

    reward_points: float
    gross_pnl: float
    commission: float
    net_pnl: float


def normalize_symbol(symbol: str | None) -> str:
    """Reduce contract codes like MESH6 → MES for lookup."""
    if not symbol:
        return ""
    symbol = symbol.upper()
    for known in _KNOWN_SPECS:
        if symbol.startswith(known):
            return known
    return symbol


def get_contract_spec(symbol: str, config: TradingConfig | None = None) -> ContractSpec:
    """Return the contract specification for a symbol, falling back to config."""
    normalized = normalize_symbol(symbol)
    if normalized in _KNOWN_SPECS:
        return _KNOWN_SPECS[normalized]

    if not config:
        raise ValueError(f"Missing trading config for unknown symbol '{symbol}'")

    point_value = getattr(config, "contract_multiplier", 50.0)
    tick_size = getattr(config, "tick_size", 0.25)
    min_tp_points = max(
        tick_size * max(1, getattr(config, "min_stop_distance_ticks", 4)),
        tick_size,
    )
    live_commission = getattr(config, "commission_per_contract", None)
    return ContractSpec(
        root_symbol=normalized,
        point_value=point_value,
        tick_size=tick_size,
        min_take_profit_points_live=min_tp_points,
        live_commission_per_side=live_commission if live_commission and live_commission > 0 else None,
    )


def get_commission_per_side(
    spec: ContractSpec,
    mode: TradingMode,
    override: Optional[float] = None,
) -> float:
    """Return commission per side for the contract, raising in live mode if missing."""
    if mode == "paper":
        return 0.0
    if override and override > 0:
        return override
    if spec.live_commission_per_side is None:
        raise RuntimeError(
            f"No commission configured for {spec.root_symbol} in live mode"
        )
    return spec.live_commission_per_side


def estimate_commission(
    quantity: float,
    spec: ContractSpec,
    mode: TradingMode,
    sides: int = 1,
    override: Optional[float] = None,
) -> float:
    """Estimate total commissions for a fill."""
    per_side = get_commission_per_side(spec, mode, override)
    if mode == "paper":
        return 0.0
    return float(quantity) * per_side * max(1, sides)


def calculate_realized_pnl(
    entry_price: float,
    exit_price: float,
    position_qty: float,
    spec: ContractSpec,
) -> tuple[float, float]:
    """
    Calculate realized gross PnL and points for a filled quantity.

    Args:
        entry_price: Average entry price.
        exit_price: Fill price for the exit.
        position_qty: Positive for long contracts being closed, negative for short.
        spec: Contract specification describing point value.

    Returns:
        (gross_pnl_dollars, points_moved_signed)
    """
    if position_qty == 0:
        return 0.0, 0.0
    direction = 1.0 if position_qty > 0 else -1.0
    points = (exit_price - entry_price) * direction
    gross = points * spec.point_value * abs(position_qty)
    return gross, points


def compute_risk_reward(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    action: str,
) -> tuple[float, float, float]:
    """Return (risk_points, reward_points, ratio) for an order."""
    normalized = action.upper()
    if normalized == "BUY":
        risk_points = entry_price - stop_loss
        reward_points = take_profit - entry_price
    else:
        risk_points = stop_loss - entry_price
        reward_points = entry_price - take_profit
    risk_points = max(risk_points, 1e-9)
    reward_points = max(reward_points, 0.0)
    ratio = reward_points / risk_points if risk_points else 0.0
    return risk_points, reward_points, ratio


def enforce_min_take_profit(
    entry_price: float,
    take_profit: float,
    spec: ContractSpec,
    mode: TradingMode,
    action: str,
) -> tuple[bool, float]:
    """Ensure live trades honour the per-symbol minimum viable take-profit."""
    if mode == "paper":
        return True, spec.min_take_profit_points_live
    if spec.min_take_profit_points_live <= 0:
        return True, 0.0
    normalized = action.upper()
    points = abs(take_profit - entry_price)
    if points + 1e-9 < spec.min_take_profit_points_live:
        return False, spec.min_take_profit_points_live
    return True, spec.min_take_profit_points_live


def expected_target_outcome(
    entry_price: float,
    take_profit: float,
    quantity: int,
    spec: ContractSpec,
    mode: TradingMode,
    commission_override: Optional[float] = None,
) -> ExpectedOutcome:
    """Project gross and net payout for a planned target."""
    reward_points = abs(take_profit - entry_price)
    gross = reward_points * spec.point_value * max(1, quantity)
    commission = 0.0
    if mode == "live":
        commission = get_commission_per_side(spec, mode, commission_override) * 2 * quantity
    return ExpectedOutcome(
        reward_points=reward_points,
        gross_pnl=gross,
        commission=commission,
        net_pnl=gross - commission,
    )

