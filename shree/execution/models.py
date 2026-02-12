"""Data models for the execution layer — shared across executor modules.

Contains the lightweight value objects used to communicate order results,
position snapshots, and fill records.  No business logic lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from ib_insync import Trade


@dataclass
class OrderResult:
    """Outcome of a submitted IB order."""

    trade: Trade
    status: str
    message: Optional[str] = None
    fill_price: Optional[float] = None
    filled_quantity: int = 0


@dataclass
class PositionInfo:
    """Snapshot of a live position with risk metadata."""

    symbol: str
    quantity: int
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_atr_multiplier: Optional[float] = None
    trailing_percent: Optional[float] = None
    atr_value: Optional[float] = None
    entry_metadata: Optional[Dict] = None


@dataclass
class CloseFill:
    """Represents the result of closing part of a position."""

    contracts: float
    entry_price: float
    exit_price: float
    direction: int  # +1 for closing long, -1 for closing short
    gross_pnl: float
    points: float
