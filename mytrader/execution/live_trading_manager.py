"""Live Trading Manager with WebSocket broadcasting."""
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
import uuid

from ib_insync import IB
from ..config import Settings
from ..utils.logger import logger
from ..utils.telegram_notifier import TelegramNotifier
from .ib_executor import TradeExecutor
from ..monitoring.live_tracker import LivePerformanceTracker
from ..strategies.engine import StrategyEngine
from ..features.feature_engineer import engineer_features
from ..risk.manager import RiskManager
from ..optimization.optimizer import ParameterOptimizer
from ..llm.rag_storage import RAGStorage, TradeRecord as RAGTradeRecord
from ..strategies.momentum_reversal import MomentumReversalStrategy
from ..strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy


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
        self.rag_storage: Optional[RAGStorage] = None
        self.telegram: Optional[TelegramNotifier] = None
        
        self.status = TradingStatus()
        self.price_history: List[Dict] = []
        self.running = False
        self.stop_requested = False
        
        # Trade context tracking
        self.current_trade_id: Optional[str] = None
        self.current_trade_entry_time: Optional[str] = None
        self.current_trade_entry_price: Optional[float] = None
        self.current_trade_features: Optional[Dict] = None
        self.current_trade_buckets: Optional[Dict] = None
        self.current_trade_rationale: Optional[Dict] = None
        
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
                [RsiMacdSentimentStrategy(), MomentumReversalStrategy()]
            )
            
            self.risk = RiskManager(self.settings.trading, position_sizing_method="kelly")
            
            # Initialize Telegram notifier if configured
            if hasattr(self.settings, 'telegram') and self.settings.telegram.enabled:
                self.telegram = TelegramNotifier(
                    bot_token=self.settings.telegram.bot_token,
                    chat_id=self.settings.telegram.chat_id,
                    enabled=True
                )
                logger.info("✅ Telegram notifications initialized")
            else:
                self.telegram = None
                logger.info("ℹ️  Telegram notifications disabled")
            
            # Initialize IB connection directly (ib_insync is already async-friendly)
            self.ib = IB()
            self.executor = TradeExecutor(
                self.ib,
                self.settings.trading,
                self.settings.data.ibkr_symbol,
                self.settings.data.ibkr_exchange,
                telegram_notifier=self.telegram
            )
            
            # Connect to IB
            await self.executor.connect(
                self.settings.data.ibkr_host,
                self.settings.data.ibkr_port,
                client_id=3
            )
            
            # Initialize RAG Storage
            try:
                self.rag_storage = RAGStorage()
                logger.info("✅ RAG Storage initialized")
            except Exception as e:
                logger.error(f"Failed to initialize RAG Storage: {e}")
                # Continue without RAG if it fails, but log error
            
            # Subscribe to execution events for RAG updates
            if self.executor and self.executor.ib:
                self.executor.ib.execDetailsEvent += self._on_execution_details
            
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
    
    def _on_execution_details(self, trade, fill):
        """Handle execution details to track trade exits for RAG."""
        try:
            if not self.rag_storage or not self.current_trade_id:
                return

            # Check if this execution generated realized PnL (indicates exit/partial exit)
            realized_pnl = 0.0
            if hasattr(fill, 'commissionReport') and hasattr(fill.commissionReport, 'realizedPNL'):
                # realizedPNL can be extremely small float for 0, check for non-zero
                if abs(fill.commissionReport.realizedPNL) > 0.001:
                    realized_pnl = fill.commissionReport.realizedPNL
            
            # If we have realized PnL, update the RAG record
            if realized_pnl != 0:
                logger.info(f"RAG: Detected trade exit with PnL: {realized_pnl}")
                
                # Calculate hold time
                hold_seconds = 0
                if self.current_trade_entry_time:
                    try:
                        entry_time = datetime.fromisoformat(self.current_trade_entry_time)
                        exit_time = datetime.now(timezone.utc)
                        hold_seconds = int((exit_time - entry_time).total_seconds())
                    except Exception as e:
                        logger.error(f"Error calculating hold time: {e}")

                # Create updated record
                # Note: We recreate the record with available info. 
                # Ideally we would fetch the existing one, but for now we rely on the ID to update.
                # We need to preserve the entry details.
                
                record = RAGTradeRecord(
                    uuid=self.current_trade_id,
                    timestamp_utc=self.current_trade_entry_time or datetime.now(timezone.utc).isoformat(),
                    contract_month=self.settings.data.ibkr_symbol,
                    entry_price=self.current_trade_entry_price or 0.0,
                    entry_qty=0, # Not updating entry qty
                    exit_price=fill.execution.price,
                    exit_qty=fill.execution.shares,
                    pnl=realized_pnl,
                    fees=fill.commissionReport.commission if hasattr(fill, 'commissionReport') else 0.0,
                    hold_seconds=hold_seconds,
                    decision_features=self.current_trade_features or {},
                    decision_rationale=self.current_trade_rationale or {}
                )
                
                self.rag_storage.save_trade(record, self.current_trade_buckets)
                logger.info(f"Updated RAG record {self.current_trade_id} with exit info")
                
                # If position is closed, clear current trade context
                # We can't easily know if it's fully closed from just this fill without checking position
                # But for simplicity, if we realize PnL, we assume the trade cycle is completing.
                # A better check would be to see if self.executor.get_current_position() is 0.
                # But that is async and we are in a sync callback? No, ib_insync callbacks are sync?
                # We'll clear it for now. If we re-enter, a new ID is generated.
                self.current_trade_id = None
                self.current_trade_entry_time = None
                self.current_trade_entry_price = None
                
        except Exception as e:
            logger.error(f"Error in _on_execution_details: {e}")

    async def start(self):
        """Start the live trading loop."""
        if not await self.initialize():
            return
        
        logger.info("🔄 Starting trading loop...")
        self.running = True
        self.stop_requested = False
        
        poll_interval = 5  # seconds
        
        try:
            while self.running and not self.stop_requested:
                try:
                    # Get current price
                    logger.debug("Fetching current price...")
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
                        logger.info(f"📊 Collecting data: {len(self.price_history)}/{self.status.min_bars_needed} bars")
                        await self._broadcast_status()
                        await asyncio.sleep(poll_interval)
                        continue
                    
                    # On first cycle after warmup, verify position state
                    if self.status.bars_collected == self.status.min_bars_needed and not hasattr(self, '_position_verified'):
                        logger.info("🔍 Warmup complete. Verifying existing positions before trading...")
                        existing_position = await self.executor.get_current_position()
                        if existing_position and existing_position.quantity != 0:
                            logger.warning(f"⚠️  EXISTING POSITION DETECTED: {existing_position.quantity} contracts @ {existing_position.avg_cost:.2f}")
                            logger.warning(f"⚠️  Bot will manage this position. Use opposite signals to exit.")
                        else:
                            logger.info("✅ No existing positions. Ready to trade fresh.")
                        self._position_verified = True
                    
                    # Process trading logic
                    logger.debug("Processing trading cycle...")
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
        
        # --- RAG Integration ---
        rag_adjustment = 0.0
        rag_rationale = {}
        
        if self.rag_storage:
            try:
                # Determine buckets
                # Volatility bucket
                volatility = features.iloc[-1].get('volatility_5m', 0.0)
                if volatility > 0.002:
                    vol_bucket = "HIGH"
                elif volatility < 0.0005:
                    vol_bucket = "LOW"
                else:
                    vol_bucket = "MEDIUM"
                
                # Time of day bucket
                hour = datetime.now(timezone.utc).hour
                if 13 <= hour < 16: # 9:30 AM - 12:00 PM ET approx
                    time_bucket = "MORNING"
                elif 16 <= hour < 19: # 12:00 PM - 3:00 PM ET approx
                    time_bucket = "MIDDAY"
                else:
                    time_bucket = "CLOSE"
                
                buckets = {
                    'volatility': vol_bucket,
                    'time_of_day': time_bucket,
                    'signal_type': signal.action
                }
                
                # Retrieve stats
                stats = self.rag_storage.get_bucket_stats(buckets)
                similar_trades = self.rag_storage.retrieve_similar_trades(buckets, limit=5)
                
                rag_rationale = {
                    'buckets': buckets,
                    'stats': stats,
                    'similar_trades_count': len(similar_trades)
                }
                
                # Adjust confidence based on historical win rate
                win_rate = stats.get('win_rate', 0.0)
                count = stats.get('count', 0)
                
                if count >= 5:
                    if win_rate > 0.6:
                        rag_adjustment = 0.1
                        rag_rationale['adjustment'] = "Positive history"
                    elif win_rate < 0.4:
                        rag_adjustment = -0.2
                        rag_rationale['adjustment'] = "Negative history"
                
                # Store for order placement
                self.current_trade_buckets = buckets
                self.current_trade_rationale = rag_rationale
                
                # Save market snapshot
                self.save_snapshot(features.iloc[-1], buckets)
                
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}")
        
        # Apply RAG adjustment
        original_confidence = signal.confidence
        signal.confidence = max(0.0, min(1.0, signal.confidence + rag_adjustment))

        # DEBUG: Log every signal generated
        logger.info(f"📊 Signal: action={signal.action}, confidence={signal.confidence:.3f} (original={original_confidence:.3f}, rag_adj={rag_adjustment:+.3f})")
        
        if rag_adjustment != 0:
            logger.info(f"RAG adjusted confidence: {original_confidence:.2f} -> {signal.confidence:.2f} ({rag_rationale.get('adjustment')})")
        
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
            logger.info(f"  ↳ Signal is HOLD, skipping order placement")
            return
        
        # CRITICAL: Check for active orders FIRST - don't place ANY order (entry or exit) if we have pending orders
        active_orders = self.executor.get_active_order_count(sync=True)
        if active_orders > 0:
            logger.info(f"  ↳ {active_orders} active orders pending, waiting for completion before placing new orders")
            return
        
        # Check if we should exit existing position
        if current_position and current_position.quantity != 0:
            # Exit logic: opposite signal closes position
            if (current_position.quantity > 0 and signal.action == "SELL") or \
               (current_position.quantity < 0 and signal.action == "BUY"):
                logger.info(f"  ↳ EXIT SIGNAL: Position={current_position.quantity}, Signal={signal.action}, closing position")
                # Place exit order (flatten position)
                exit_qty = abs(current_position.quantity)
                await self._place_exit_order(signal.action, exit_qty, current_price)
                return
            else:
                logger.info(f"  ↳ Position already open (qty={current_position.quantity}), signal={signal.action}, no action (same direction)")
                return
        
        # Place order
        logger.info(f"  ↳ Attempting to place order: {signal.action}")
        await self._place_order(signal, current_price, features)
    
    async def _place_exit_order(self, action: str, quantity: int, current_price: float):
        """Place a market order to exit existing position."""
        logger.info(f"🚪 Placing EXIT order: {action} {quantity} contracts @ market price ~{current_price:.2f}")
        
        try:
            # Place market order to flatten position
            order_id = await self.executor.place_order(
                action=action,  # BUY to close SHORT, SELL to close LONG
                quantity=quantity,
                limit_price=current_price,
                stop_loss=None,  # No stops for exit orders
                take_profit=None
            )
            
            logger.info(f"✅ Exit order placed: ID={order_id}")
            
            # Broadcast exit
            await self._broadcast_order_update({
                "type": "EXIT",
                "action": action,
                "quantity": quantity,
                "price": current_price,
                "order_id": order_id
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to place exit order: {e}")
            await self._broadcast_error(f"Exit order failed: {e}")
    
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
                
                # --- RAG Persistence (Entry) ---
                if self.rag_storage:
                    try:
                        trade_uuid = str(uuid.uuid4())
                        self.current_trade_id = trade_uuid
                        entry_time = datetime.now(timezone.utc).isoformat()
                        
                        # Store context for exit handler
                        self.current_trade_entry_time = entry_time
                        self.current_trade_entry_price = result.fill_price
                        self.current_trade_features = {
                            'confidence': signal.confidence,
                            'atr': atr,
                            'volatility': row.get('volatility_5m', 0.0)
                        }
                        
                        # Create trade record
                        record = RAGTradeRecord(
                            uuid=trade_uuid,
                            timestamp_utc=entry_time,
                            contract_month=self.settings.data.ibkr_symbol, # Approximation
                            entry_price=result.fill_price,
                            entry_qty=qty,
                            exit_price=None,
                            exit_qty=None,
                            pnl=None,
                            fees=0.0, # To be updated
                            hold_seconds=None,
                            decision_features=self.current_trade_features,
                            decision_rationale=self.current_trade_rationale or {}
                        )
                        
                        self.rag_storage.save_trade(record, self.current_trade_buckets)
                        logger.info(f"Saved trade entry to RAG: {trade_uuid}")
                        
                    except Exception as e:
                        logger.error(f"Failed to save trade to RAG: {e}")
    
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

    def save_snapshot(self, row, buckets: Dict):
        """Save market snapshot to RAG storage."""
        if not self.rag_storage:
            return
            
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Construct data package for RAGStorage
            data = {
                "ohlcv": {
                    "open": float(row.get("open", 0.0)),
                    "high": float(row.get("high", 0.0)),
                    "low": float(row.get("low", 0.0)),
                    "close": float(row["close"]),
                    "volume": int(row.get("volume", 0))
                },
                "vwap": float(row.get("vwap", 0.0)), # Assuming vwap is in row if available
                "volatility": float(row.get("volatility_5m", 0.0)),
                "indicators": {
                    "atr_14": float(row.get("ATR_14", 0.0)),
                    "rsi_14": float(row.get("RSI_14", 0.0)),
                    "macd": float(row.get("MACD", 0.0)),
                    "bb_upper": float(row.get("BB_upper", 0.0)),
                    "bb_lower": float(row.get("BB_lower", 0.0)),
                    "sma_20": float(row.get("SMA_20", 0.0)),
                    "sma_50": float(row.get("SMA_50", 0.0)),
                    "buckets": buckets # Store buckets in indicators for now if useful
                }
            }
            
            self.rag_storage.save_snapshot(timestamp, data)
                
        except Exception as e:
            logger.error(f"Failed to save market snapshot: {e}")
