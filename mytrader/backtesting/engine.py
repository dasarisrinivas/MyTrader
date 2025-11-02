"""Simple vectorized backtesting engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from ..config import BacktestConfig, TradingConfig
from ..features.feature_engineer import engineer_features
from ..risk.manager import RiskManager
from ..strategies.base import BaseStrategy
from ..strategies.engine import StrategyEngine
from ..utils.logger import logger


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Dict[str, float]]
    metrics: Dict[str, float]


class BacktestingEngine:
    def __init__(self, strategies: Iterable[BaseStrategy], trading_config: TradingConfig, backtest_config: BacktestConfig) -> None:
        self.strategy_engine = StrategyEngine(strategies)
        self.trading_config = trading_config
        self.backtest_config = backtest_config
        self.risk = RiskManager(trading_config)

    def run(self, data: pd.DataFrame) -> BacktestResult:
        price_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in price_cols if col not in data.columns]
        if missing:
            raise ValueError(f"backtest data missing columns: {missing}")

        sentiment_cols = [col for col in data.columns if "sentiment" in col]
        sentiment_df = data[sentiment_cols] if sentiment_cols else None
        features = engineer_features(data[price_cols], sentiment_df)
        if features.empty:
            raise ValueError("engineered feature set is empty")

        capital = self.backtest_config.initial_capital
        equity_curve: List[tuple[pd.Timestamp, float]] = []
        trades: List[Dict[str, float]] = []
        position = 0
        entry_price = 0.0
        self.risk.reset()

        returns = features["close"].pct_change().fillna(0)

        for idx, row in features.iterrows():
            price = float(row["close"])
            history = features.loc[:idx]
            history_returns = returns.loc[:idx]
            signal = self.strategy_engine.evaluate(history, history_returns)

            exit_price = None
            exit_reason = ""
            if position > 0:
                stop_price = entry_price - self.trading_config.stop_loss_ticks * self.trading_config.tick_size
                take_price = entry_price + self.trading_config.take_profit_ticks * self.trading_config.tick_size
                if price <= stop_price:
                    exit_price = price - self.backtest_config.slippage
                    exit_reason = "stop"
                elif price >= take_price:
                    exit_price = price - self.backtest_config.slippage
                    exit_reason = "target"
                elif signal.action == "SELL":
                    exit_price = price - self.backtest_config.slippage
                    exit_reason = "signal"
            elif position < 0:
                stop_price = entry_price + self.trading_config.stop_loss_ticks * self.trading_config.tick_size
                take_price = entry_price - self.trading_config.take_profit_ticks * self.trading_config.tick_size
                if price >= stop_price:
                    exit_price = price + self.backtest_config.slippage
                    exit_reason = "stop"
                elif price <= take_price:
                    exit_price = price + self.backtest_config.slippage
                    exit_reason = "target"
                elif signal.action == "BUY":
                    exit_price = price + self.backtest_config.slippage
                    exit_reason = "signal"

            if exit_price is not None and position != 0:
                realized = (exit_price - entry_price) * position * self.trading_config.contract_multiplier
                capital += realized
                commission = abs(position) * self.trading_config.commission_per_contract
                capital -= commission
                self.risk.update_pnl(realized)
                self.risk.register_trade()
                trades.append({
                    "timestamp": idx.isoformat(),
                    "action": "EXIT",
                    "reason": exit_reason,
                    "price": float(exit_price),
                    "qty": int(position),
                    "realized": float(realized),
                })
                position = 0
                entry_price = 0.0

            if position == 0 and signal.action in {"BUY", "SELL"}:
                qty = self.risk.position_size(capital, signal.confidence)
                if qty > 0 and self.risk.can_trade(qty):
                    fill_price = price + self.backtest_config.slippage if signal.action == "BUY" else price - self.backtest_config.slippage
                    direction = 1 if signal.action == "BUY" else -1
                    position = direction * qty
                    entry_price = fill_price
                    capital -= qty * self.trading_config.commission_per_contract
                    self.risk.register_trade()
                    trades.append({
                        "timestamp": idx.isoformat(),
                        "action": signal.action,
                        "price": float(fill_price),
                        "qty": int(direction * qty),
                        "signal_confidence": float(signal.confidence),
                    })

            pnl = (price - entry_price) * position * self.trading_config.contract_multiplier if position != 0 else 0.0
            equity = capital + pnl
            equity_curve.append((idx, equity))

        if position != 0:
            final_price = float(features.iloc[-1]["close"])
            exit_price = final_price - self.backtest_config.slippage if position > 0 else final_price + self.backtest_config.slippage
            realized = (exit_price - entry_price) * position * self.trading_config.contract_multiplier
            capital += realized
            commission = abs(position) * self.trading_config.commission_per_contract
            capital -= commission
            self.risk.update_pnl(realized)
            self.risk.register_trade()
            trades.append({
                "timestamp": features.index[-1].isoformat(),
                "action": "FORCED_EXIT",
                "price": float(exit_price),
                "qty": int(position),
                "realized": float(realized),
            })
            position = 0

        if equity_curve:
            ts, vals = zip(*equity_curve)
            curve = pd.Series(vals, index=pd.Index(ts, name="timestamp"))
        else:
            curve = pd.Series(dtype=float)
        metrics = self._compute_metrics(curve, trades)
        logger.info("Backtest completed: {}", metrics)
        return BacktestResult(equity_curve=curve, trades=trades, metrics=metrics)

    def _compute_metrics(self, curve: pd.Series, trades: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute comprehensive performance metrics."""
        from .performance import summarize_performance
        return summarize_performance(curve, trades)
