"""Simple vectorized backtesting engine with dynamic risk controls."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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
        equity_curve: List[Tuple[pd.Timestamp, float]] = []
        trades: List[Dict[str, float]] = []

        position = 0
        entry_price = 0.0
        current_stop_price: float | None = None
        current_target_price: float | None = None
        trailing_atr_multiplier: float | None = None
        trailing_percent: float | None = None
        active_trade_metadata: Dict[str, float] | None = None

        self.risk.reset()
        returns = features["close"].pct_change().fillna(0)

        max_abs_position = 0

        for idx, row in features.iterrows():
            price = float(row["close"])
            history = features.loc[:idx]
            history_returns = returns.loc[:idx]
            signal = self.strategy_engine.evaluate(history, history_returns)

            exit_price = None
            exit_reason = ""

            # Update trailing logic for open positions
            if position != 0:
                direction = 1 if position > 0 else -1
                atr_value = self._safe_float(row.get("ATR_14"))
                if not atr_value and active_trade_metadata:
                    atr_value = self._safe_float(active_trade_metadata.get("atr_value"))

                if trailing_atr_multiplier and atr_value > 0:
                    trail_distance = atr_value * trailing_atr_multiplier
                    if direction > 0:
                        new_stop = price - trail_distance
                        if current_stop_price is None or new_stop > current_stop_price:
                            current_stop_price = new_stop
                    else:
                        new_stop = price + trail_distance
                        if current_stop_price is None or new_stop < current_stop_price:
                            current_stop_price = new_stop

                if trailing_percent and trailing_percent > 0:
                    profit = (price - entry_price) * direction
                    if profit > 0:
                        trail_price = entry_price + direction * profit * (1 - trailing_percent)
                        if direction > 0:
                            current_stop_price = max(current_stop_price or trail_price, trail_price)
                        else:
                            current_stop_price = min(current_stop_price or trail_price, trail_price)

            if position > 0:
                stop_price = current_stop_price if current_stop_price is not None else entry_price - self.trading_config.stop_loss_ticks * self.trading_config.tick_size
                take_price = current_target_price if current_target_price is not None else entry_price + self.trading_config.take_profit_ticks * self.trading_config.tick_size
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
                stop_price = current_stop_price if current_stop_price is not None else entry_price + self.trading_config.stop_loss_ticks * self.trading_config.tick_size
                take_price = current_target_price if current_target_price is not None else entry_price - self.trading_config.take_profit_ticks * self.trading_config.tick_size
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

                exit_record = {
                    "timestamp": idx.isoformat(),
                    "action": "EXIT",
                    "reason": exit_reason,
                    "price": float(exit_price),
                    "qty": int(position),
                    "realized": float(realized),
                }
                if current_stop_price is not None:
                    exit_record["stop_price"] = float(current_stop_price)
                if current_target_price is not None:
                    exit_record["target_price"] = float(current_target_price)
                if active_trade_metadata:
                    exit_record["entry_metadata"] = active_trade_metadata
                trades.append(exit_record)

                position = 0
                entry_price = 0.0
                current_stop_price = None
                current_target_price = None
                trailing_atr_multiplier = None
                trailing_percent = None
                active_trade_metadata = None

            if position == 0 and signal.action in {"BUY", "SELL"}:
                qty = self.risk.position_size(capital, signal.confidence)
                metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
                scaler = self._safe_float(metadata.get("position_scaler")) or 1.0
                if scaler > 0:
                    qty = max(1, int(round(qty * scaler)))
                
                # Enforce hard cap
                hard_cap = self.trading_config.max_contracts_limit
                qty = min(qty, hard_cap)
                
                # Also respect max_position_size if it's lower (though they should be synced)
                qty = min(qty, self.trading_config.max_position_size)

                if qty > 0 and self.risk.can_trade(qty):
                    fill_price = price + self.backtest_config.slippage if signal.action == "BUY" else price - self.backtest_config.slippage
                    direction = 1 if signal.action == "BUY" else -1
                    position = direction * qty
                    entry_price = fill_price
                    capital -= qty * self.trading_config.commission_per_contract
                    self.risk.register_trade()

                    current_stop_price, current_target_price = self._compute_trade_levels(fill_price, direction, metadata, row)
                    trailing_atr_multiplier = self._safe_float(metadata.get("trailing_atr_multiplier")) or None
                    trailing_percent = self._safe_float(metadata.get("trailing_percent")) or None
                    active_trade_metadata = dict(metadata) if metadata else None

                    entry_record = {
                        "timestamp": idx.isoformat(),
                        "action": signal.action,
                        "price": float(fill_price),
                        "qty": int(direction * qty),
                        "signal_confidence": float(signal.confidence),
                    }
                    if current_stop_price is not None:
                        entry_record["stop_price"] = float(current_stop_price)
                    if current_target_price is not None:
                        entry_record["target_price"] = float(current_target_price)
                    if metadata:
                        entry_record["metadata"] = metadata
                    trades.append(entry_record)
            
            # Track max concurrent contracts
            max_abs_position = max(max_abs_position, abs(position))

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

            forced_record = {
                "timestamp": features.index[-1].isoformat(),
                "action": "FORCED_EXIT",
                "price": float(exit_price),
                "qty": int(position),
                "realized": float(realized),
            }
            if current_stop_price is not None:
                forced_record["stop_price"] = float(current_stop_price)
            if current_target_price is not None:
                forced_record["target_price"] = float(current_target_price)
            if active_trade_metadata:
                forced_record["entry_metadata"] = active_trade_metadata
            trades.append(forced_record)

        if equity_curve:
            ts, vals = zip(*equity_curve)
            curve = pd.Series(vals, index=pd.Index(ts, name="timestamp"))
        else:
            curve = pd.Series(dtype=float)

        metrics = self._compute_metrics(curve, trades)
        metrics["max_concurrent_contracts"] = float(max_abs_position)
        logger.info("Backtest completed: {}", metrics)
        return BacktestResult(equity_curve=curve, trades=trades, metrics=metrics)

    def _compute_metrics(self, curve: pd.Series, trades: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute comprehensive performance metrics."""
        from .performance import summarize_performance

        return summarize_performance(curve, trades)

    def _compute_trade_levels(
        self,
        fill_price: float,
        direction: int,
        metadata: Dict[str, float],
        row: pd.Series,
    ) -> tuple[float | None, float | None]:
        """Determine trade-specific stop and target prices."""
        raw_stop = metadata.get("stop_loss_price")
        stop_price = float(raw_stop) if isinstance(raw_stop, (int, float)) else None

        raw_target = metadata.get("take_profit_price")
        target_price = float(raw_target) if isinstance(raw_target, (int, float)) else None

        atr_value = self._safe_float(metadata.get("atr_value"))
        if atr_value <= 0:
            atr_value = self._safe_float(row.get("ATR_14"))

        tick_size = self.trading_config.tick_size
        default_stop_offset = self.trading_config.stop_loss_ticks * tick_size
        default_target_offset = self.trading_config.take_profit_ticks * tick_size

        if stop_price is None:
            atr_multiplier = self._safe_float(metadata.get("atr_stop_multiplier"))
            if atr_multiplier > 0 and atr_value > 0:
                stop_offset = atr_value * atr_multiplier
            else:
                stop_offset = default_stop_offset

            if stop_offset <= 0:
                stop_offset = default_stop_offset

            stop_price = fill_price - stop_offset if direction > 0 else fill_price + stop_offset

        if target_price is None:
            risk_reward = self._safe_float(metadata.get("risk_reward"))
            if risk_reward <= 0:
                if default_stop_offset > 0:
                    risk_reward = self.trading_config.take_profit_ticks / max(1e-6, self.trading_config.stop_loss_ticks)
                else:
                    risk_reward = 2.0

            stop_distance = abs(fill_price - stop_price)
            if stop_distance <= 0:
                stop_distance = default_stop_offset

            target_offset = stop_distance * risk_reward if risk_reward > 0 else default_target_offset
            target_price = fill_price + target_offset if direction > 0 else fill_price - target_offset

        return stop_price, target_price

    @staticmethod
    def _safe_float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
