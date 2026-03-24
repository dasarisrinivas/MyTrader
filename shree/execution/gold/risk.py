"""Gold risk management — position sizing and daily guardrails.

Responsibilities:
    - Compute contract quantity from dollar risk and stop distance
    - Enforce daily loss limit, max trades per day, consecutive-loss cooldown
    - Enforce hard contract cap (independent of sizing math)

Deliberately *stateless* with respect to positions — the manager passes
current state as arguments so this class remains independently testable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

from ...config.gold import GoldRiskConfig
from ...risk.trade_math import ContractSpec
from ...utils.logger import logger


@dataclass
class SizingResult:
    """Output of ``GoldRiskManager.size_position``."""

    approved: bool
    contracts: int
    risk_per_contract: float       # $ at risk per contract
    total_risk_usd: float          # contracts × risk_per_contract
    reason: str = ""               # Non-empty when approved=False


@dataclass
class DailyState:
    """Snapshot of today's trading state, passed by the manager."""

    realized_pnl: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    cooldown_until: Optional[datetime] = None   # UTC datetime


class GoldRiskManager:
    """Evaluate proposed trades against risk guardrails.

    All guardrails are *stateless* relative to this object — the daily state
    is passed in on each call so the manager owns persistence.
    """

    def __init__(
        self,
        risk_cfg: GoldRiskConfig,
        spec: ContractSpec,
        effective_max_risk: Optional[float] = None,
    ) -> None:
        self._cfg = risk_cfg
        self._spec = spec
        # Allow callers to override max_risk (e.g. GC risk scaling via gc_adjusted_risk_usd)
        self._effective_max_risk = effective_max_risk if effective_max_risk is not None else risk_cfg.max_risk_per_trade_usd

    # ── Public API ────────────────────────────────────────────────────────────

    def size_position(
        self,
        stop_distance_points: float,
        daily: DailyState,
        now_utc: Optional[datetime] = None,
    ) -> SizingResult:
        """Determine how many contracts to trade.

        Args:
            stop_distance_points: Absolute distance from entry to stop loss
                                  in price points (always positive).
            daily:                Current trading-day state.
            now_utc:              Override for unit testing.

        Returns:
            SizingResult with ``approved=True`` and ``contracts`` if safe.
        """
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        # ── Hard gate 1: daily loss limit ─────────────────────────────────────
        if daily.realized_pnl <= -abs(self._cfg.daily_loss_limit_usd):
            return SizingResult(
                approved=False,
                contracts=0,
                risk_per_contract=0.0,
                total_risk_usd=0.0,
                reason=(
                    f"Daily loss limit hit: realized_pnl={daily.realized_pnl:.2f} "
                    f"limit={self._cfg.daily_loss_limit_usd:.2f}"
                ),
            )

        # ── Hard gate 2: max trades per day ───────────────────────────────────
        if daily.trades_today >= self._cfg.max_trades_per_day:
            return SizingResult(
                approved=False,
                contracts=0,
                risk_per_contract=0.0,
                total_risk_usd=0.0,
                reason=(
                    f"Max trades per day reached: {daily.trades_today}"
                    f"/{self._cfg.max_trades_per_day}"
                ),
            )

        # ── Hard gate 3: consecutive-loss cooldown ────────────────────────────
        if daily.cooldown_until is not None:
            if daily.cooldown_until.tzinfo is None:
                daily.cooldown_until = daily.cooldown_until.replace(tzinfo=timezone.utc)
            if now_utc < daily.cooldown_until:
                remaining = (daily.cooldown_until - now_utc).total_seconds() / 60
                return SizingResult(
                    approved=False,
                    contracts=0,
                    risk_per_contract=0.0,
                    total_risk_usd=0.0,
                    reason=f"Consecutive-loss cooldown active: {remaining:.0f} min remaining",
                )

        # ── Size calculation ──────────────────────────────────────────────────
        if stop_distance_points <= 0:
            return SizingResult(
                approved=False,
                contracts=0,
                risk_per_contract=0.0,
                total_risk_usd=0.0,
                reason=f"Invalid stop distance: {stop_distance_points}",
            )

        risk_per_contract = stop_distance_points * self._spec.point_value
        if risk_per_contract <= 0:
            return SizingResult(
                approved=False,
                contracts=0,
                risk_per_contract=0.0,
                total_risk_usd=0.0,
                reason="risk_per_contract ≤ 0 (bad point_value or stop_distance)",
            )

        # Floor: at least 1 contract; cap by hard cap
        raw_contracts = self._effective_max_risk / risk_per_contract
        contracts = max(1, math.floor(raw_contracts))
        contracts = min(contracts, self._cfg.max_contracts_hard_cap)

        total_risk = contracts * risk_per_contract

        # ── Sanity: total risk must not exceed daily limit remainder ──────────
        remaining_daily = abs(self._cfg.daily_loss_limit_usd) - abs(daily.realized_pnl)
        if total_risk > remaining_daily:
            # Scale down to fit
            contracts = max(1, math.floor(remaining_daily / risk_per_contract))
            total_risk = contracts * risk_per_contract
            if total_risk > remaining_daily:
                return SizingResult(
                    approved=False,
                    contracts=0,
                    risk_per_contract=risk_per_contract,
                    total_risk_usd=total_risk,
                    reason=(
                        f"Even 1 contract (${risk_per_contract:.2f} risk) would "
                        f"exceed daily limit remainder (${remaining_daily:.2f})"
                    ),
                )

        logger.debug(
            "GoldRiskManager: sized %d contract(s), risk/contract=%.2f, total_risk=%.2f",
            contracts,
            risk_per_contract,
            total_risk,
        )
        return SizingResult(
            approved=True,
            contracts=contracts,
            risk_per_contract=risk_per_contract,
            total_risk_usd=total_risk,
        )

    def compute_cooldown_until(
        self,
        consecutive_losses: int,
        now_utc: Optional[datetime] = None,
    ) -> Optional[datetime]:
        """Return a UTC cooldown expiry if the loss threshold is breached, else None."""
        if consecutive_losses < self._cfg.max_consecutive_losses:
            return None
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        cooldown = now_utc + timedelta(minutes=self._cfg.post_loss_cooldown_minutes)
        logger.warning(
            "GoldRiskManager: %d consecutive losses — cooldown until %s UTC",
            consecutive_losses,
            cooldown.strftime("%H:%M:%S"),
        )
        return cooldown

    @staticmethod
    def compute_stop_distance(entry: float, stop_loss: float, action: str) -> float:
        """Return the absolute stop distance in points (always positive)."""
        if action.upper() == "BUY":
            return max(0.0, entry - stop_loss)
        return max(0.0, stop_loss - entry)
