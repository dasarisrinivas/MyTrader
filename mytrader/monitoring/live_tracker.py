"""Real-time performance tracking for live trading."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List

import numpy as np
import pandas as pd

from ..utils.logger import logger


@dataclass
class TradeRecord:
    timestamp: datetime
    action: str
    price: float
    quantity: int
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class PerformanceSnapshot:
    timestamp: datetime
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float  # Sum of realized + unrealized
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    daily_pnl: float
    portfolio_heat: float
    trade_count: int  # Alias for total_trades


class LivePerformanceTracker:
    """Track and monitor trading performance in real-time."""
    
    def __init__(
        self,
        initial_capital: float,
        max_history: int = 10000,
        risk_free_rate: float = 0.02
    ) -> None:
        self.initial_capital = initial_capital
        self.max_history = max_history
        self.risk_free_rate = risk_free_rate
        
        # Equity tracking
        self.equity_curve: Deque[tuple[datetime, float]] = deque(maxlen=max_history)
        self.equity_curve.append((datetime.utcnow(), initial_capital))
        
        # Trade tracking
        self.trades: List[TradeRecord] = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_realized_pnl = 0.0
        
        # Position tracking
        self.current_position = 0
        self.position_entry_price = 0.0
        self.unrealized_pnl = 0.0
        
        # Daily tracking
        self.daily_start_equity = initial_capital
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.utcnow().date()
        
        # Drawdown tracking
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        
        # Performance metrics cache
        self._metrics_cache: Dict[str, float] = {}
        self._cache_timestamp = datetime.utcnow()

    def update_equity(self, current_price: float, realized_pnl: float = 0.0) -> None:
        """Update equity based on current position and price."""
        now = datetime.utcnow()
        
        # Reset daily tracking if new day
        if now.date() != self.last_reset_date:
            self.daily_start_equity = self.get_current_equity()
            self.daily_pnl = 0.0
            self.last_reset_date = now.date()
        
        # Calculate unrealized PnL
        if self.current_position != 0:
            self.unrealized_pnl = (current_price - self.position_entry_price) * self.current_position
        else:
            self.unrealized_pnl = 0.0
        
        # Update realized PnL
        if realized_pnl != 0.0:
            self.total_realized_pnl += realized_pnl
            self.daily_pnl += realized_pnl
            
            # Track win/loss
            if realized_pnl > 0:
                self.winning_trades += 1
            elif realized_pnl < 0:
                self.losing_trades += 1
        
        # Calculate total equity
        total_equity = self.initial_capital + self.total_realized_pnl + self.unrealized_pnl
        
        # Update equity curve
        self.equity_curve.append((now, total_equity))
        
        # Update peak and drawdown
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
        
        current_drawdown = (total_equity - self.peak_equity) / self.peak_equity
        if current_drawdown < self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Invalidate metrics cache
        self._cache_timestamp = now
        self._metrics_cache = {}

    def record_trade(
        self,
        action: str,
        price: float,
        quantity: int,
        realized_pnl: float = 0.0
    ) -> None:
        """Record a trade execution."""
        trade = TradeRecord(
            timestamp=datetime.utcnow(),
            action=action,
            price=price,
            quantity=quantity,
            realized_pnl=realized_pnl,
            unrealized_pnl=self.unrealized_pnl
        )
        
        self.trades.append(trade)
        
        # Update position
        if action == "BUY":
            if self.current_position <= 0:
                self.position_entry_price = price
            self.current_position += quantity
        elif action == "SELL":
            if self.current_position >= 0:
                self.position_entry_price = price
            self.current_position -= quantity
        
        logger.info("Recorded trade: %s %d @ %.2f, position=%d", 
                   action, quantity, price, self.current_position)

    def get_current_equity(self) -> float:
        """Get current total equity."""
        if self.equity_curve:
            return self.equity_curve[-1][1]
        return self.initial_capital

    def get_sharpe_ratio(self, window: int = 252) -> float:
        """Calculate rolling Sharpe ratio."""
        if len(self.equity_curve) < 2:
            return 0.0
        
        # Get equity series
        equity_df = pd.DataFrame(list(self.equity_curve), columns=["timestamp", "equity"])
        equity_df.set_index("timestamp", inplace=True)
        
        # Calculate returns
        returns = equity_df["equity"].pct_change().dropna()
        
        if len(returns) < 2:
            return 0.0
        
        # Use last 'window' periods
        recent_returns = returns.tail(window)
        
        mean_return = recent_returns.mean()
        std_return = recent_returns.std()
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe
        sharpe = (mean_return - self.risk_free_rate / 252) / std_return * np.sqrt(252)
        return float(sharpe)

    def get_sortino_ratio(self, window: int = 252) -> float:
        """Calculate rolling Sortino ratio."""
        if len(self.equity_curve) < 2:
            return 0.0
        
        equity_df = pd.DataFrame(list(self.equity_curve), columns=["timestamp", "equity"])
        equity_df.set_index("timestamp", inplace=True)
        returns = equity_df["equity"].pct_change().dropna()
        
        if len(returns) < 2:
            return 0.0
        
        recent_returns = returns.tail(window)
        downside_returns = recent_returns[recent_returns < 0]
        
        mean_return = recent_returns.mean()
        downside_std = downside_returns.std() if len(downside_returns) > 0 else recent_returns.std()
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return - self.risk_free_rate / 252) / downside_std * np.sqrt(252)
        return float(sortino)

    def get_win_rate(self) -> float:
        """Calculate win rate."""
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total if total > 0 else 0.0

    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak."""
        current_equity = self.get_current_equity()
        if self.peak_equity == 0:
            return 0.0
        return (current_equity - self.peak_equity) / self.peak_equity

    def get_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot."""
        total_pnl = self.total_realized_pnl + self.unrealized_pnl
        total_trades_count = len(self.trades)
        
        return PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            equity=self.get_current_equity(),
            realized_pnl=self.total_realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            total_pnl=total_pnl,
            total_trades=total_trades_count,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
            win_rate=self.get_win_rate(),
            sharpe_ratio=self.get_sharpe_ratio(),
            sortino_ratio=self.get_sortino_ratio(),
            max_drawdown=self.max_drawdown,
            current_drawdown=self.get_current_drawdown(),
            daily_pnl=self.daily_pnl,
            portfolio_heat=self._calculate_portfolio_heat(),
            trade_count=total_trades_count
        )

    def _calculate_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (risk exposure)."""
        if self.current_position == 0:
            return 0.0
        
        current_equity = self.get_current_equity()
        if current_equity == 0:
            return 0.0
        
        # Assuming each contract is worth position_entry_price
        position_value = abs(self.current_position * self.position_entry_price)
        return (position_value / current_equity) * 100

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity"])
        
        df = pd.DataFrame(list(self.equity_curve), columns=["timestamp", "equity"])
        df.set_index("timestamp", inplace=True)
        return df

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame(columns=["timestamp", "action", "price", "quantity", "realized_pnl"])
        
        trades_data = [
            {
                "timestamp": t.timestamp,
                "action": t.action,
                "price": t.price,
                "quantity": t.quantity,
                "realized_pnl": t.realized_pnl
            }
            for t in self.trades
        ]
        
        df = pd.DataFrame(trades_data)
        df.set_index("timestamp", inplace=True)
        return df

    def export_snapshot(self, filepath: str) -> None:
        """Export current performance snapshot to JSON."""
        import json
        
        snapshot = self.get_snapshot()
        data = {
            "timestamp": snapshot.timestamp.isoformat(),
            "equity": snapshot.equity,
            "realized_pnl": snapshot.realized_pnl,
            "unrealized_pnl": snapshot.unrealized_pnl,
            "total_trades": snapshot.total_trades,
            "winning_trades": snapshot.winning_trades,
            "losing_trades": snapshot.losing_trades,
            "win_rate": snapshot.win_rate,
            "sharpe_ratio": snapshot.sharpe_ratio,
            "sortino_ratio": snapshot.sortino_ratio,
            "max_drawdown": snapshot.max_drawdown,
            "current_drawdown": snapshot.current_drawdown,
            "daily_pnl": snapshot.daily_pnl,
            "portfolio_heat": snapshot.portfolio_heat
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info("Performance snapshot exported to %s", filepath)

    def log_status(self) -> None:
        """Log current performance status."""
        snapshot = self.get_snapshot()
        logger.info(
            "Performance: Equity=%.2f PnL=%.2f(%.2f%%) Trades=%d WinRate=%.1f%% Sharpe=%.2f DD=%.2f%% Heat=%.1f%%",
            snapshot.equity,
            snapshot.realized_pnl + snapshot.unrealized_pnl,
            ((snapshot.equity / self.initial_capital - 1) * 100),
            snapshot.total_trades,
            snapshot.win_rate * 100,
            snapshot.sharpe_ratio,
            snapshot.current_drawdown * 100,
            snapshot.portfolio_heat
        )
