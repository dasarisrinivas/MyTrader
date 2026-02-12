"""Reusable safety guard helpers for execution logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class WaitDecisionContext:
    """Encapsulates AWS wait advisory details."""

    decision: str = "GO"
    advisory_only: bool = False
    confidence: float = 0.0
    size_multiplier: Optional[float] = None


def should_block_on_wait(
    wait_context: WaitDecisionContext,
    block_on_wait: bool,
    override_confidence: float,
    signal_confidence: float,
) -> bool:
    """Return True if the trade must be blocked due to an AWS WAIT decision."""
    if wait_context.decision.upper() != "WAIT":
        return False
    if not block_on_wait:
        return False
    if wait_context.advisory_only:
        return signal_confidence < override_confidence
    return True


def compute_trade_risk_dollars(
    entry_price: float,
    stop_loss: float,
    contract_multiplier: float,
) -> float:
    """Estimate dollar risk for a single contract."""
    return abs(entry_price - stop_loss) * contract_multiplier
