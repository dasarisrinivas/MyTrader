"""Live Trading Manager with WebSocket broadcasting."""
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from ib_insync import IB
from ..config import Settings
from ..utils.logger import logger
from .ib_executor import TradeExecutor
from ..monitoring.live_tracker import LivePerformanceTracker
from ..strategies.engine import StrategyEngine
from ..features.feature_engineer import engineer_features
from ..risk.manager import RiskManager
from ..optimization.optimizer import ParameterOptimizer


@dataclass
class TradingStatus:
    """Current trading status."""
    is_running: bool = False
    session_start: Optional[str] = None
    bars_collected: int = 0
    min_bars_needed: int = 50
    current_price: Optional[float] = None
    last_signal: Optional[str] = None
    signal_confidence: Optional[float] = None
    active_orders: int = 0
    current_position: int = 0
    unrealized_pnl: float = 0.0
    message: str = ""


class LiveTradingManager:
    """Manages live trading session with WebSocket broadcasting."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ib: Optional[IB] = None
        self.executor: Optional[TradeExecutor] = None
        self.tracker: Optional[LivePerformanceTracker] = None
        self.engine: Optional[StrategyEngine] = None
        self.risk: Optional[RiskManager] = None
        
        self.status = TradingStatus()
        self.price_history: List[Dict] = []
        self.running = False
        self.stop_requested = False
        
        # Callbacks for WebSocket broadcasting
        self.on_status_update: Optional[callable] = None
        self.on_signal_generated: Optional[callable] = None
        self.on_order_update: Optional[callable] = None
        self.on_trade_executed: Optional[callable] = None
        self.on_error: Optional[callable] = None
    
    async def initialize(self):
        """Initialize trading components."""
        try:
            logger.info("Initializing trading components...")
            
            # Initialize other components first (non-IB)
            self.tracker = LivePerformanceTracker(
                initial_capital=self.settings.trading.initial_capital,
                risk_free_rate=self.settings.backtest.risk_free_rate
            )
            
            self.engine = StrategyEngine(
                self.settings.strategies,
                self.settings.trading,
                self.settings.backtest
            )
            
            self.risk = RiskManager(self.settings.trading, position_sizing_method="kelly")
            
            # Initialize IB connection in a thread pool to avoid event loop conflicts
            loop = asyncio.get_event_loop()
            
            def connect_ib():
                """Connect to IB in a separate thread with its own event loop."""
                import nest_asyncio
                nest_asyncio.apply()
                
                ib = IB()
                executor = TradeExecutor(
                    ib,
                    self.settings.trading,
                    self.settings.data.ibkr_symbol,
                    self.settings.data.ibkr_exchange
                )
                
                # Run connection in this thread's event loop
                loop_local = asyncio.new_event_loop()
                asyncio.set_event_loop(loop_local)
                
                try:
                    loop_local.run_until_complete(executor.connect(
                        self.settings.data.ibkr_host,
                        self.settings.data.ibkr_port,
                        client_id=3
                    ))
                    return ib, executor
                finally:
                    # Don't close the loop, we need it running
                    pass
            
            # Run IB connection in thread pool
            with ThreadPoolExecutor(max_workers=1) as executor_pool:
                future = loop.run_in_executor(executor_pool, connect_ib)
                self.ib, self.executor = await future
            
            logger.info("✅ Connected to IBKR")
            
            self.status.is_running = True
            self.status.session_start = datetime.now(timezone.utc).isoformat()
            self.status.message = "Initialized successfully"
            
            await self._broadcast_status()
            
            logger.info("✅ Live trading manager initialized")
            return True
            
        except Exception as e:
            self.status.message = f"Initialization failed: {str(e)}"
            await self._broadcast_error(str(e))
            logger.error(f"Failed to initialize: {e}")
            return False
    
    async def start(self):
        """Start the live trading loop."""
        if not await self.initialize():
            return
        
        self.running = True
        self.stop_requested = False
        
        poll_interval = 5  # seconds
        
        try:
            while self.running and not self.stop_requested:
                try:
                    # Get current price
                    current_price = await self.executor.get_current_price()
                    if not current_price:
                        self.status.message = "Waiting for price data..."
                        await self._broadcast_status()
                        await asyncio.sleep(poll_interval)
                        continue
                    
                    self.status.current_price = current_price
                    
                    # Add to price history
                    price_bar = {
                        'timestamp': datetime.now(timezone.utc),
                        'open': current_price,
                        'high': current_price,
                        'low': current_price,
                        'close': current_price,
                        'volume': 0
                    }
                    self.price_history.append(price_bar)
                    
                    # Keep only recent history
                    if len(self.price_history) > 500:
                        self.price_history = self.price_history[-500:]
                    
                    self.status.bars_collected = len(self.price_history)
                    
                    # Check if we have enough bars
                    if len(self.price_history) < self.status.min_bars_needed:
                        self.status.message = f"Collecting data: {len(self.price_history)}/{self.status.min_bars_needed} bars"
                        await self._broadcast_status()
                        await asyncio.sleep(poll_interval)
                        continue
                    
                    # Process trading logic
                    await self._process_trading_cycle(current_price)
                    
                    await asyncio.sleep(poll_interval)
                    
                except Exception as cycle_error:
                    logger.error(f"Error in trading cycle: {cycle_error}")
                    await self._broadcast_error(str(cycle_error))
                    await asyncio.sleep(poll_interval)
        
        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}")
            await self._broadcast_error(str(e))
        finally:
            await self.stop()
    
    async def _process_trading_cycle(self, current_price: float):
        """Process one trading cycle."""
        import pandas as pd
        
        # Convert to DataFrame and engineer features
        df = pd.DataFrame(self.price_history)
        df.set_index('timestamp', inplace=True)
        
        features = engineer_features(df[['open', 'high', 'low', 'close', 'volume']], None)
        if features.empty:
            self.status.message = "Feature engineering returned empty"
            await self._broadcast_status()
            return
        
        returns = features["close"].pct_change().dropna()
        
        # Generate signal
        signal = self.engine.evaluate(features, returns)
        self.status.last_signal = signal.action
        self.status.signal_confidence = signal.confidence
        
        # Broadcast signal
        await self._broadcast_signal(signal, current_price)
        
        # Get current position
        current_position = await self.executor.get_current_position()
        self.status.current_position = current_position.quantity if current_position else 0
        self.status.active_orders = self.executor.get_active_order_count()
        
        if current_position:
            self.status.unrealized_pnl = await self.executor.get_unrealized_pnl()
            self.tracker.update_equity(current_price, realized_pnl=0.0)
            
            # Update trailing stops
            atr_val = float(features.iloc[-1].get("ATR_14", 0.0))
            await self.executor.update_trailing_stops(current_price, atr_val)
        
        await self._broadcast_status()
        
        # Execute trading logic
        if signal.action == "HOLD":
            return
        
        # Don't open new position if we already have one
        if current_position and current_position.quantity != 0:
            return
        
        # Place order
        await self._place_order(signal, current_price, features)
    
    async def _place_order(self, signal, current_price: float, features):
        """Place an order based on signal."""
        # Position sizing
        risk_stats = self.risk.get_statistics()
        qty = self.risk.position_size(
            self.settings.trading.initial_capital,
            signal.confidence,
            win_rate=risk_stats.get("win_rate"),
            avg_win=risk_stats.get("avg_win"),
            avg_loss=risk_stats.get("avg_loss")
        )
        
        metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
        scaler = float(metadata.get("position_scaler", 1.0))
        if scaler > 0:
            qty = max(1, int(round(qty * scaler)))
        
        qty = min(qty, self.settings.trading.max_position_size)
        
        if not self.risk.can_trade(qty):
            await self._broadcast_error("Risk limits exceeded")
            return
        
        # Calculate stop/target
        direction = 1 if signal.action == "BUY" else -1
        row = features.iloc[-1]
        
        atr = float(metadata.get("atr_value", 0.0))
        if atr <= 0:
            atr = float(row.get("ATR_14", 0.0))
        
        tick_size = self.settings.trading.tick_size
        default_stop_offset = self.settings.trading.stop_loss_ticks * tick_size
        default_target_offset = self.settings.trading.take_profit_ticks * tick_size
        
        stop_offset = default_stop_offset
        stop_loss = current_price - stop_offset if direction > 0 else current_price + stop_offset
        take_profit = current_price + default_target_offset if direction > 0 else current_price - default_target_offset
        
        # Broadcast order intent
        await self._broadcast_order_update({
            "status": "placing",
            "action": signal.action,
            "quantity": qty,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        })
        
        # Place order
        result = await self.executor.place_order(
            action=signal.action,
            quantity=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata
        )
        
        # Broadcast result
        await self._broadcast_order_update({
            "status": result.status,
            "action": signal.action,
            "quantity": qty,
            "fill_price": result.fill_price,
            "filled_quantity": result.filled_quantity,
            "order_id": result.trade.order.orderId if result.trade else None
        })
        
        if result.status not in {"Cancelled", "Inactive"}:
            self.risk.register_trade()
            if result.fill_price:
                self.tracker.record_trade(
                    action=signal.action,
                    price=result.fill_price,
                    quantity=qty
                )
    
    async def stop(self):
        """Stop the trading session."""
        self.running = False
        self.stop_requested = True
        self.status.is_running = False
        self.status.message = "Stopped"
        
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
        
        await self._broadcast_status()
        logger.info("Trading session stopped")
    
    # Broadcasting methods
    async def _broadcast_status(self):
        """Broadcast status update."""
        if self.on_status_update:
            await self.on_status_update(asdict(self.status))
    
    async def _broadcast_signal(self, signal, price: float):
        """Broadcast signal generated."""
        if self.on_signal_generated:
            await self.on_signal_generated({
                "action": signal.action,
                "confidence": signal.confidence,
                "price": price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": signal.metadata if isinstance(signal.metadata, dict) else {}
            })
    
    async def _broadcast_order_update(self, order_data: Dict):
        """Broadcast order update."""
        if self.on_order_update:
            order_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            await self.on_order_update(order_data)
    
    async def _broadcast_error(self, error_msg: str):
        """Broadcast error."""
        if self.on_error:
            await self.on_error({
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    def get_performance_snapshot(self) -> Dict:
        """Get current performance snapshot."""
        if not self.tracker:
            return {}
        
        snapshot = self.tracker.get_snapshot()
        return {
            "total_pnl": snapshot.total_pnl,
            "total_return": (snapshot.equity / self.tracker.initial_capital - 1) * 100,
            "sharpe_ratio": snapshot.sharpe_ratio,
            "max_drawdown": snapshot.max_drawdown,
            "win_rate": snapshot.win_rate,
            "total_trades": snapshot.trade_count,
            "winning_trades": snapshot.winning_trades,
            "losing_trades": snapshot.losing_trades,
        }
