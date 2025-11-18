"""Risk management module."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from ..config import TradingConfig


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    timestamp: datetime


class RiskManager:
    """Advanced risk management with Kelly Criterion and ATR-based stops."""
    
    def __init__(
        self, 
        config: TradingConfig,
        position_sizing_method: Literal["fixed_fraction", "kelly"] = "fixed_fraction"
    ) -> None:
        self.config = config
        self.position_sizing_method = position_sizing_method
        self.daily_loss = 0.0
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_wins = 0.0
        self.total_losses = 0.0
        self.portfolio_heat = 0.0  # Current risk exposure as % of capital

    def reset(self) -> None:
        """Reset daily counters."""
        self.daily_loss = 0.0
        self.trade_count = 0

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.reset()
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_wins = 0.0
        self.total_losses = 0.0
        self.portfolio_heat = 0.0

    def update_pnl(self, realized_pnl: float) -> None:
        """Update PnL and win/loss statistics."""
        self.daily_loss -= realized_pnl
        
        if realized_pnl > 0:
            self.winning_trades += 1
            self.total_wins += realized_pnl
        elif realized_pnl < 0:
            self.losing_trades += 1
            self.total_losses += abs(realized_pnl)

    def can_trade(self, proposed_qty: int) -> bool:
        """Check if we can place a new trade given risk limits."""
        if self.trade_count >= self.config.max_daily_trades:
            return False
        if abs(proposed_qty) > self.config.max_position_size:
            return False
        if self.daily_loss >= self.config.max_daily_loss:
            return False
        return True

    def position_size(
        self, 
        account_value: float, 
        confidence: float,
        win_rate: float | None = None,
        avg_win: float | None = None,
        avg_loss: float | None = None
    ) -> int:
        """Calculate position size using selected method."""
        if self.position_sizing_method == "kelly":
            return self._kelly_criterion_size(
                account_value, confidence, win_rate, avg_win, avg_loss
            )
        else:
            return self._fixed_fraction_size(account_value, confidence)

    def _fixed_fraction_size(self, account_value: float, confidence: float) -> int:
        """
        Fixed fractional position sizing based on risk percentage.
        
        This is the RECOMMENDED method for position sizing (not Kelly).
        Risks a fixed percentage of account per trade (e.g., 0.5% or 1%).
        """
        # Get risk percentage from config (default 0.5%)
        risk_pct = getattr(self.config, 'risk_per_trade_pct', 0.005)
        
        # Calculate dollar risk
        dollar_risk = account_value * risk_pct
        
        # Calculate contracts based on stop distance
        tick_value = self.config.tick_value
        stop_ticks = max(1.0, self.config.stop_loss_ticks)
        dollar_risk_per_contract = tick_value * stop_ticks
        
        contracts = int(dollar_risk / dollar_risk_per_contract)
        
        # Apply confidence scaling (optional - can be removed for pure fixed fractional)
        # contracts = int(contracts * min(1.0, confidence))
        
        return max(1, min(self.config.max_position_size, contracts))

    def _kelly_criterion_size(
        self,
        account_value: float,
        confidence: float,
        win_rate: float | None = None,
        avg_win: float | None = None,
        avg_loss: float | None = None
    ) -> int:
        """Kelly Criterion position sizing."""
        # Use historical stats if available
        if win_rate is None and self.winning_trades + self.losing_trades > 10:
            win_rate = self.winning_trades / (self.winning_trades + self.losing_trades)
        if avg_win is None and self.winning_trades > 0:
            avg_win = self.total_wins / self.winning_trades
        if avg_loss is None and self.losing_trades > 0:
            avg_loss = self.total_losses / self.losing_trades

        # Default assumptions if no history
        if win_rate is None:
            win_rate = 0.5
        if avg_win is None:
            avg_win = self.config.take_profit_ticks * self.config.tick_value
        if avg_loss is None:
            avg_loss = self.config.stop_loss_ticks * self.config.tick_value

        # Kelly formula: f = (p*b - q) / b
        # where p = win rate, q = 1-p, b = avg_win/avg_loss
        if avg_loss == 0:
            kelly_fraction = 0.01
        else:
            b = avg_win / avg_loss
            kelly_fraction = (win_rate * b - (1 - win_rate)) / b
        
        # Apply half-Kelly for safety and scale by confidence
        kelly_fraction = max(0, min(0.25, kelly_fraction * 0.5 * confidence))
        
        # Convert to contracts
        dollar_risk = account_value * kelly_fraction
        tick_value = self.config.tick_value
        stop_ticks = max(1.0, self.config.stop_loss_ticks)
        contracts = int(dollar_risk / (tick_value * stop_ticks))
        
        return max(1, min(self.config.max_position_size, contracts))

    def calculate_atr_stop(
        self, 
        current_price: float, 
        atr: float, 
        direction: Literal["long", "short"],
        atr_multiplier: float = 2.0
    ) -> float:
        """Calculate ATR-based stop loss."""
        if direction == "long":
            stop_price = current_price - (atr * atr_multiplier)
        else:  # short
            stop_price = current_price + (atr * atr_multiplier)
        return stop_price

    def calculate_dynamic_stops(
        self,
        entry_price: float,
        current_atr: float,
        direction: Literal["long", "short"],
        atr_multiplier: float = 2.0,
        risk_reward_ratio: float = 2.0
    ) -> tuple[float, float]:
        """Calculate dynamic stop loss and take profit based on ATR."""
        stop_distance = current_atr * atr_multiplier
        target_distance = stop_distance * risk_reward_ratio
        
        if direction == "long":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + target_distance
        else:  # short
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - target_distance
            
        return stop_loss, take_profit

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        direction: Literal["long", "short"],
        trail_percent: float = 0.5,
        current_atr: float | None = None,
        atr_multiplier: float = 1.5
    ) -> float:
        """
        Calculate trailing stop loss.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            direction: Trade direction (long/short)
            trail_percent: Percent of profit to trail (0.5 = 50%)
            current_atr: Current ATR for dynamic trailing
            atr_multiplier: Multiplier for ATR-based trailing
            
        Returns:
            Trailing stop price
        """
        if current_atr and current_atr > 0:
            # ATR-based trailing stop
            trail_distance = current_atr * atr_multiplier
            
            if direction == "long":
                # Trail stop up as price moves higher
                return max(
                    entry_price - trail_distance,
                    current_price - trail_distance
                )
            else:  # short
                # Trail stop down as price moves lower
                return min(
                    entry_price + trail_distance,
                    current_price + trail_distance
                )
        else:
            # Percent-based trailing stop
            if direction == "long":
                profit = current_price - entry_price
                if profit > 0:
                    return entry_price + (profit * trail_percent)
                return entry_price - abs(entry_price * 0.01)  # Initial stop 1%
            else:  # short
                profit = entry_price - current_price
                if profit > 0:
                    return entry_price - (profit * trail_percent)
                return entry_price + abs(entry_price * 0.01)  # Initial stop 1%

    def update_portfolio_heat(self, position_value: float, account_value: float) -> None:
        """Update current portfolio heat (risk exposure)."""
        self.portfolio_heat = (position_value / account_value) * 100 if account_value > 0 else 0

    def register_trade(self) -> None:
        """Register a new trade."""
        self.trade_count += 1

    def get_statistics(self) -> dict:
        """Get current risk statistics."""
        total_trades = self.winning_trades + self.losing_trades
        win_rate = self.winning_trades / total_trades if total_trades > 0 else 0
        avg_win = self.total_wins / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = self.total_losses / self.losing_trades if self.losing_trades > 0 else 0
        profit_factor = self.total_wins / self.total_losses if self.total_losses > 0 else float('inf')
        
        return {
            "total_trades": total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "daily_loss": self.daily_loss,
            "daily_trade_count": self.trade_count,
            "portfolio_heat": self.portfolio_heat
        }
