"""Live Trading Manager with WebSocket broadcasting.

ENHANCED VERSION with:
- Trade cooldown period (prevents over-trading)
- Candle close validation
- Higher-timeframe level filters (PDH/PDL, WH/WL)
- Trend confirmation (EMA)
- Simulation mode (dry run)
- Enhanced confidence scoring
- Hybrid RAG+LLM Pipeline (3-layer decision system)
- CST timestamps throughout (Central Standard Time)
"""
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import json
import uuid

from ib_insync import IB
from ..config import Settings
from ..utils.logger import logger
from ..utils.telegram_notifier import TelegramNotifier
from ..utils.timezone_utils import now_cst, format_cst, today_cst, CST, utc_to_cst
from .ib_executor import TradeExecutor
from ..monitoring.live_tracker import LivePerformanceTracker
from ..strategies.engine import StrategyEngine
from ..features.feature_engineer import engineer_features
from ..risk.manager import RiskManager
from ..optimization.optimizer import ParameterOptimizer
from ..llm.rag_storage import RAGStorage, TradeRecord as RAGTradeRecord
from ..strategies.momentum_reversal import MomentumReversalStrategy
from ..strategies.momentum_reversal import MomentumReversalStrategy
from ..strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from ..strategies.market_regime import detect_market_regime, get_regime_parameters, MarketRegime
from ..strategies.trading_filters import TradingFilters, calculate_enhanced_confidence, PriceLevels

# NEW: Hybrid RAG+LLM Pipeline imports
try:
    from ..rag.pipeline_integration import HybridPipelineIntegration, create_hybrid_integration
    HYBRID_PIPELINE_AVAILABLE = True
except ImportError:
    HYBRID_PIPELINE_AVAILABLE = False
    logger.warning("Hybrid RAG pipeline not available - using legacy signal generation")


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
    # NEW: Cooldown and filter status
    cooldown_remaining_seconds: int = 0
    simulation_mode: bool = False
    filters_applied: List[str] = field(default_factory=list)
    # NEW: Hybrid pipeline status
    hybrid_pipeline_enabled: bool = False
    hybrid_market_trend: str = ""
    hybrid_volatility_regime: str = ""


class LiveTradingManager:
    """Manages live trading session with WebSocket broadcasting.
    
    ENHANCED with:
    - Trade cooldown period (configurable, default 5 minutes)
    - Candle close validation (wait for candle to close before entry)
    - Higher-timeframe levels (PDH/PDL, WH/WL, PWH/PWL)
    - Trend confirmation filters
    - Simulation mode for testing without real orders
    """
    
    # === CONFIGURATION CONSTANTS ===
    DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes between trades
    MIN_CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence to place trade
    POLL_INTERVAL_SECONDS = 5  # How often to poll price (when NOT waiting for candle)
    CANDLE_PERIOD_SECONDS = 60  # 1-minute candles - only evaluate at candle close
    
    def __init__(self, settings: Settings, simulation_mode: bool = False):
        self.settings = settings
        self.simulation_mode = simulation_mode  # NEW: Dry run mode
        self.ib: Optional[IB] = None
        self.executor: Optional[TradeExecutor] = None
        self.tracker: Optional[LivePerformanceTracker] = None
        self.engine: Optional[StrategyEngine] = None
        self.risk: Optional[RiskManager] = None
        self.rag_storage: Optional[RAGStorage] = None
        self.telegram: Optional[TelegramNotifier] = None
        
        # NEW: Trading filters for multi-timeframe analysis
        self.trading_filters: Optional[TradingFilters] = None
        
        # NEW: Hybrid RAG+LLM Pipeline
        self.hybrid_pipeline: Optional[HybridPipelineIntegration] = None
        self._use_hybrid_pipeline: bool = False
        
        self.status = TradingStatus()
        self.status.simulation_mode = simulation_mode
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
        
        # NEW: Cooldown tracking
        self._last_trade_time: Optional[datetime] = None
        self._cooldown_seconds = getattr(
            settings.trading, 'trade_cooldown_minutes', 5
        ) * 60  # Convert minutes to seconds
        
        # NEW: Candle tracking for proper candle-close validation
        self._last_candle_processed: Optional[datetime] = None
        self._waiting_for_candle_close: bool = True
        
        # Callbacks for WebSocket broadcasting
        self.on_status_update: Optional[callable] = None
        self.on_signal_generated: Optional[callable] = None
        self.on_order_update: Optional[callable] = None
        self.on_trade_executed: Optional[callable] = None
        self.on_error: Optional[callable] = None
        
        if simulation_mode:
            logger.warning("🔶 SIMULATION MODE ENABLED - Orders will NOT be sent to IBKR")
    
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
                client_id=11  # Use unique client_id to avoid conflicts with stale connections
            )
            # Force reconciliation of active orders after IB connection
            await self.force_order_reconciliation()
            # Initialize RAG Storage
            try:
                self.rag_storage = RAGStorage()
                logger.info("✅ RAG Storage initialized")
            except Exception as e:
                logger.error(f"Failed to initialize RAG Storage: {e}")
                # Continue without RAG if it fails, but log error
            
            # NEW: Initialize Hybrid RAG+LLM Pipeline
            if HYBRID_PIPELINE_AVAILABLE:
                try:
                    # Check if hybrid is enabled in config
                    hybrid_enabled = getattr(self.settings, 'hybrid', None)
                    if hybrid_enabled and getattr(hybrid_enabled, 'enabled', False):
                        self.hybrid_pipeline = create_hybrid_integration(
                            settings=self.settings,
                            llm_client=None,  # Will use Bedrock if available
                        )
                        self._use_hybrid_pipeline = True
                        self.status.hybrid_pipeline_enabled = True
                        logger.info("✅ Hybrid RAG+LLM Pipeline initialized (3-layer decision system)")
                    else:
                        logger.info("ℹ️  Hybrid pipeline disabled in config")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to initialize Hybrid Pipeline: {e}")
                    self._use_hybrid_pipeline = False
            
            # Subscribe to execution events for RAG updates
            if self.executor and self.executor.ib:
                self.executor.ib.execDetailsEvent += self._on_execution_details
            
            logger.info("✅ Connected to IBKR")
            
            self.status.is_running = True
            self.status.session_start = now_cst().isoformat()
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
                        exit_time = now_cst()
                        hold_seconds = int((exit_time - entry_time).total_seconds())
                    except Exception as e:
                        logger.error(f"Error calculating hold time: {e}")

                # Create updated record
                # Note: We recreate the record with available info. 
                # Ideally we would fetch the existing one, but for now we rely on the ID to update.
                # We need to preserve the entry details.
                
                record = RAGTradeRecord(
                    uuid=self.current_trade_id,
                    timestamp_utc=self.current_trade_entry_time or now_cst().isoformat(),
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
                        'timestamp': now_cst(),
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
        """Process one trading cycle.
        
        ENHANCED with:
        - Cooldown period between trades
        - Candle close validation
        - Higher-timeframe level filters (PDH/PDL, WH/WL)
        - Trend confirmation (EMA 9>20)
        - Enhanced confidence scoring
        """
        import pandas as pd
        
        # === NEW: Check cooldown period ===
        if self._last_trade_time:
            elapsed = (now_cst() - self._last_trade_time).total_seconds()
            cooldown_remaining = self._cooldown_seconds - elapsed
            if cooldown_remaining > 0:
                self.status.cooldown_remaining_seconds = int(cooldown_remaining)
                logger.debug(f"⏳ Cooldown active: {cooldown_remaining:.0f}s remaining")
                await self._broadcast_status()
                return
            else:
                self.status.cooldown_remaining_seconds = 0
        
        # === NEW: Candle close validation ===
        # Only evaluate signals at the start of a new candle (every CANDLE_PERIOD_SECONDS)
        now = now_cst()
        current_candle_start = now.replace(second=0, microsecond=0)
        
        if self._last_candle_processed == current_candle_start:
            # Already processed this candle, skip until next one
            logger.debug(f"⏳ Waiting for next candle close (current minute already processed)")
            return
        
        # Mark this candle as processed
        self._last_candle_processed = current_candle_start
        logger.info(f"🕐 New candle close at {current_candle_start.strftime('%H:%M:%S')} CST")
        
        # Convert to DataFrame and engineer features
        df = pd.DataFrame(self.price_history)
        df.set_index('timestamp', inplace=True)
        
        features = engineer_features(df[['open', 'high', 'low', 'close', 'volume']], None)
        if features.empty:
            self.status.message = "Feature engineering returned empty"
            await self._broadcast_status()
            return
        
        returns = features["close"].pct_change().dropna()
        
        # === NEW: Use Hybrid RAG+LLM Pipeline if available ===
        if self._use_hybrid_pipeline and self.hybrid_pipeline:
            try:
                hybrid_signal, pipeline_result = await self.hybrid_pipeline.process(features, current_price)
                
                # Update status with hybrid pipeline info
                if pipeline_result:
                    self.status.hybrid_market_trend = pipeline_result.rule_engine.market_trend
                    self.status.hybrid_volatility_regime = pipeline_result.rule_engine.volatility_regime
                    self.status.filters_applied = pipeline_result.rule_engine.filters_passed
                
                # Log hybrid pipeline decision
                logger.info(
                    f"🤖 Hybrid Pipeline: {hybrid_signal.action} "
                    f"(conf={hybrid_signal.confidence:.2f}, "
                    f"trend={self.status.hybrid_market_trend}, "
                    f"vol={self.status.hybrid_volatility_regime})"
                )
                
                # Use hybrid signal instead of legacy signal
                signal = hybrid_signal
                
                # Store pipeline result for trade logging
                self._current_pipeline_result = pipeline_result
                
                # Skip legacy RAG and filter processing since hybrid handles it
                await self._process_hybrid_signal(signal, pipeline_result, current_price, features)
                return
                
            except Exception as e:
                logger.error(f"Hybrid pipeline error, falling back to legacy: {e}")
                # Fall through to legacy signal generation
        
        # Generate signal (legacy path)
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
                
                # Time of day bucket (CST hours)
                hour = now_cst().hour
                if 8 <= hour < 11:  # 8:00 AM - 11:00 AM CST (morning session)
                    time_bucket = "MORNING"
                elif 11 <= hour < 14:  # 11:00 AM - 2:00 PM CST (midday)
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
        
        # === NEW: Apply Trading Filters ===
        filters_passed = True
        filters_applied = []
        enhanced_conf = signal.confidence
        
        if self.trading_filters is None:
            # Initialize trading filters on first use (lazy initialization)
            try:
                self.trading_filters = TradingFilters()
                logger.info("✅ Trading filters initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize trading filters: {e}")
        
        if self.trading_filters and signal.action != "HOLD":
            try:
                # Update levels with current price data
                # Update levels using historical data (lazy: only update once per day)
                if self.trading_filters._levels is None or len(features) > 200:
                    self.trading_filters.set_historical_data(features)
                
                # Evaluate filters
                filter_result = self.trading_filters.evaluate(
                    current_price=current_price,
                    proposed_action=signal.action,
                    features=features,
                )
                
                filters_passed = filter_result.can_trade
                filters_applied = filter_result.reasons
                
                # Calculate enhanced confidence using the enhanced function
                enhanced_conf, conf_reasons = calculate_enhanced_confidence(
                    base_confidence=signal.confidence,
                    features=features,
                    action=signal.action,
                    price_levels=filter_result.levels,
                    current_price=current_price
                )
                
                # Add confidence adjustment from filter result
                enhanced_conf = min(1.0, max(0.0, enhanced_conf + filter_result.confidence_adjustment))
                
                # Log filter results
                if not filters_passed:
                    logger.warning(f"🚫 Signal BLOCKED by filters: {filter_result.reasons}")
                else:
                    logger.info(f"✅ Filters PASSED: {filters_applied}")
                    if conf_reasons:
                        logger.info(f"   Confidence factors: {conf_reasons}")
                    logger.info(f"   Enhanced confidence: {signal.confidence:.3f} -> {enhanced_conf:.3f}")
                
                # Update status with filter info
                self.status.filters_applied = filters_applied
                
            except Exception as e:
                logger.error(f"Filter evaluation error: {e}")
                # Continue without filters on error
                filters_passed = True
        
        # Apply enhanced confidence
        signal.confidence = enhanced_conf
        
        # === NEW: Minimum confidence threshold check ===
        if signal.confidence < self.MIN_CONFIDENCE_THRESHOLD and signal.action != "HOLD":
            logger.info(f"🔽 Signal confidence {signal.confidence:.3f} below threshold {self.MIN_CONFIDENCE_THRESHOLD:.3f}, converting to HOLD")
            signal.action = "HOLD"
        
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
        
        # === NEW: Check if filters blocked the trade ===
        if not filters_passed:
            logger.info(f"  ↳ Trade blocked by filters, skipping order placement")
            return
        
        # CRITICAL: Check for active orders FIRST - don't place ANY order (entry or exit) if we have pending orders
        active_orders = self.executor.get_active_order_count(sync=True)
        if active_orders > 0:
            logger.info(f"  ↳ {active_orders} active orders pending, waiting for completion before placing new orders")
            return
        
        # Check if we should exit existing position
        if current_position and current_position.quantity != 0:
            # Exit logic: opposite signal closes position
            is_buy_signal = signal.action in ["BUY", "SCALP_BUY"]
            is_sell_signal = signal.action in ["SELL", "SCALP_SELL"]
            if (current_position.quantity > 0 and is_sell_signal) or \
               (current_position.quantity < 0 and is_buy_signal):
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
    
    async def _process_hybrid_signal(
        self,
        signal,
        pipeline_result,
        current_price: float,
        features,
    ):
        """Process a signal from the hybrid RAG+LLM pipeline.
        
        This is the new 3-layer decision path:
        1. Rule Engine has already evaluated filters
        2. RAG has retrieved similar trades and docs
        3. LLM has made final decision with reasoning
        
        Args:
            signal: HybridSignal from pipeline
            pipeline_result: HybridPipelineResult with full context
            current_price: Current price
            features: Features DataFrame
        """
        # Update status
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
        
        # Check if pipeline blocked the trade
        if signal.action == "HOLD":
            if pipeline_result and pipeline_result.rule_engine.filters_blocked:
                logger.info(f"  ↳ Trade blocked by hybrid filters: {pipeline_result.rule_engine.filters_blocked}")
            else:
                logger.info(f"  ↳ Signal is HOLD, skipping order placement")
            return
        
        # CRITICAL: Check for active orders FIRST
        active_orders = self.executor.get_active_order_count(sync=True)
        if active_orders > 0:
            logger.info(f"  ↳ {active_orders} active orders pending, waiting for completion")
            return
        
        # Check if we should exit existing position
        if current_position and current_position.quantity != 0:
            is_buy_signal = signal.action in ["BUY", "SCALP_BUY"]
            is_sell_signal = signal.action in ["SELL", "SCALP_SELL"]
            if (current_position.quantity > 0 and is_sell_signal) or \
               (current_position.quantity < 0 and is_buy_signal):
                logger.info(f"  ↳ HYBRID EXIT: Position={current_position.quantity}, Signal={signal.action}")
                exit_qty = abs(current_position.quantity)
                await self._place_exit_order(signal.action, exit_qty, current_price)
                
                # Log trade exit through hybrid pipeline
                if self.hybrid_pipeline and hasattr(self, '_current_pipeline_result'):
                    self.hybrid_pipeline.log_trade_exit(
                        exit_price=current_price,
                        exit_reason="SIGNAL_EXIT",
                    )
                return
            else:
                logger.info(f"  ↳ Position open (qty={current_position.quantity}), same direction, no action")
                return
        
        # Place order using hybrid pipeline's risk parameters
        logger.info(f"  ↳ Placing HYBRID order: {signal.action}")
        await self._place_hybrid_order(signal, pipeline_result, current_price, features)
    
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
    
    async def _place_hybrid_order(self, signal, pipeline_result, current_price: float, features):
        """Place an order using hybrid pipeline's risk parameters.
        
        Uses stop loss and take profit from the pipeline result,
        which incorporates LLM suggestions and ATR-based calculations.
        """
        try:
            logger.info(f"🤖 HYBRID: Placing {signal.action} order")
            
            # Position sizing
            risk_stats = self.risk.get_statistics()
            qty = self.risk.position_size(
                self.settings.trading.initial_capital,
                signal.confidence,
                win_rate=risk_stats.get("win_rate"),
                avg_win=risk_stats.get("avg_win"),
                avg_loss=risk_stats.get("avg_loss")
            )
            
            # Apply position size factor from hybrid pipeline
            if pipeline_result:
                position_factor = pipeline_result.position_size
                qty = max(1, int(round(qty * position_factor)))
            
            qty = min(qty, self.settings.trading.max_position_size)
            
            if not self.risk.can_trade(qty):
                await self._broadcast_error("Risk limits exceeded")
                return
            
            # Get risk parameters from hybrid pipeline
            # Handle both normal and scalp signals
            is_buy = signal.action in ["BUY", "SCALP_BUY"]
            is_sell = signal.action in ["SELL", "SCALP_SELL"]
            direction = 1 if is_buy else -1
            is_scalp = signal.action in ["SCALP_BUY", "SCALP_SELL"]
            row = features.iloc[-1]
            
            # Use pipeline's calculated stop/target
            if pipeline_result and pipeline_result.stop_loss > 0:
                stop_offset = pipeline_result.stop_loss
                target_offset = pipeline_result.take_profit
                # Tighter stops for scalps
                if is_scalp:
                    stop_offset = stop_offset * 0.6  # 60% of normal stop
                    target_offset = target_offset * 0.5  # 50% of normal target
                    logger.info(f"🎯 Using SCALP risk params: SL={stop_offset:.2f}, TP={target_offset:.2f}")
                else:
                    logger.info(f"🎯 Using HYBRID risk params: SL={stop_offset:.2f}, TP={target_offset:.2f}")
            else:
                # Fallback to ATR-based
                atr = float(row.get("ATR_14", 0.0))
                if is_scalp:
                    stop_offset = atr * 0.75  # Tighter for scalps
                    target_offset = atr * 1.0
                    logger.info(f"🎯 Using ATR SCALP fallback: SL={stop_offset:.2f}, TP={target_offset:.2f}")
                else:
                    stop_offset = atr * 1.5
                    target_offset = atr * 2.0
                    logger.info(f"🎯 Using ATR fallback: SL={stop_offset:.2f}, TP={target_offset:.2f}")
            
            stop_loss = current_price - stop_offset if direction > 0 else current_price + stop_offset
            take_profit = current_price + target_offset if direction > 0 else current_price - target_offset
            
            # Build metadata from hybrid pipeline
            metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
            metadata["hybrid_pipeline"] = True
            metadata["hybrid_reasoning"] = metadata.get("hybrid_reasoning", "")
            metadata["market_trend"] = self.status.hybrid_market_trend
            metadata["volatility_regime"] = self.status.hybrid_volatility_regime
            
            # Prepare market data for trade logging
            market_data = {
                "close": float(row.get("close", current_price)),
                "rsi": float(row.get("RSI_14", 50)),
                "macd_hist": float(row.get("MACD", 0)),
                "atr": float(row.get("ATR_14", 0)),
                "ema_9": float(row.get("EMA_9", current_price)),
                "ema_20": float(row.get("EMA_20", current_price)),
                "pdh": float(row.get("PDH", 0)),
                "pdl": float(row.get("PDL", 0)),
            }
            
            # Broadcast order intent
            await self._broadcast_order_update({
                "status": "placing",
                "action": signal.action,
                "quantity": qty,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "hybrid_pipeline": True,
                "market_trend": self.status.hybrid_market_trend,
            })
            
            # === Simulation mode check ===
            if self.simulation_mode:
                logger.warning(f"🔶 SIMULATION: Would place HYBRID {signal.action} order for {qty} contracts @ {current_price:.2f}")
                logger.warning(f"   SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
                self._last_trade_time = now_cst()
                
                # Log simulated trade entry
                if self.hybrid_pipeline:
                    self.hybrid_pipeline.log_trade_entry(
                        action=signal.action,
                        entry_price=current_price,
                        quantity=qty,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        market_data=market_data,
                        pipeline_result=pipeline_result,
                    )
                
                await self._broadcast_order_update({
                    "status": "SIMULATED",
                    "action": signal.action,
                    "quantity": qty,
                    "fill_price": current_price,
                    "order_id": f"SIM-HYBRID-{now_cst().strftime('%H%M%S')}"
                })
                return
            
            # Place real order
            result = await self.executor.place_order(
                action=signal.action,
                quantity=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
                rationale=metadata,
                features=market_data,
                market_regime=self.status.hybrid_market_trend
            )
            
            # Broadcast result
            await self._broadcast_order_update({
                "status": result.status,
                "action": signal.action,
                "quantity": qty,
                "fill_price": result.fill_price,
                "filled_quantity": result.filled_quantity,
                "order_id": result.trade.order.orderId if result.trade else None,
                "hybrid_pipeline": True,
            })
            
            if result.status not in {"Cancelled", "Inactive"}:
                # Update cooldown
                self._last_trade_time = now_cst()
                if self.hybrid_pipeline:
                    self.hybrid_pipeline.record_trade_for_cooldown()
                logger.info(f"⏱️ HYBRID trade placed - cooldown activated")
                
                self.risk.register_trade()
                
                if result.fill_price:
                    self.tracker.record_trade(
                        action=signal.action,
                        price=result.fill_price,
                        quantity=qty
                    )
                    
                    # Log trade entry through hybrid pipeline
                    if self.hybrid_pipeline:
                        self.hybrid_pipeline.log_trade_entry(
                            action=signal.action,
                            entry_price=result.fill_price,
                            quantity=qty,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            market_data=market_data,
                            pipeline_result=pipeline_result,
                        )
                        logger.info(f"✅ Trade logged to Hybrid RAG system")
        
        except Exception as e:
            logger.error(f"❌ CRITICAL: _place_hybrid_order failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            await self._broadcast_error(f"Hybrid order placement failed: {e}")
    
    async def _place_order(self, signal, current_price: float, features):
        """Place an order based on signal."""
        try:
            logger.info(f"🛠️ DEBUG: Entered _place_order for {signal.action}")
            
            # Position sizing
            risk_stats = self.risk.get_statistics()
            logger.info(f"🛠️ DEBUG: Got risk_stats: {risk_stats}")
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
            
            # Capture features for logging
            self.current_trade_features = {
                'confidence': signal.confidence,
                'atr': float(row.get("ATR_14", 0.0)),
                'volatility': float(row.get('volatility_5m', 0.0)),
                'rsi': float(row.get("RSI_14", 0.0)),
                'macd': float(row.get("MACD", 0.0)),
                'close': float(row["close"]),
                'volume': int(row.get("volume", 0))
            }
            
            atr = float(metadata.get("atr_value", 0.0))
            if atr <= 0:
                atr = float(row.get("ATR_14", 0.0))
            
            # Detect market regime for dynamic risk parameters
            regime, regime_conf = detect_market_regime(features)
            regime_params = get_regime_parameters(regime)
            
            logger.info(f"📊 Market Regime: {regime.value} (conf={regime_conf:.2f}) - Using dynamic stops")
            
            # Get multipliers from regime params
            atr_mult_sl = regime_params.get("atr_multiplier_sl", 2.0)
            atr_mult_tp = regime_params.get("atr_multiplier_tp", 4.0)
            
            # Calculate dynamic distances
            if atr > 0:
                stop_offset = atr * atr_mult_sl
                target_offset = atr * atr_mult_tp
            else:
                # Fallback to fixed ticks if ATR is invalid
                tick_size = self.settings.trading.tick_size
                stop_offset = self.settings.trading.stop_loss_ticks * tick_size
                target_offset = self.settings.trading.take_profit_ticks * tick_size
                logger.warning("⚠️ ATR is 0 or invalid, using fixed ticks fallback")
                
            stop_loss = current_price - stop_offset if direction > 0 else current_price + stop_offset
            take_profit = current_price + target_offset if direction > 0 else current_price - target_offset
            
            logger.info(f"🎯 Dynamic Risk: ATR={atr:.2f}, SL={atr_mult_sl}x ({stop_offset:.2f}), TP={atr_mult_tp}x ({target_offset:.2f})")
            
            # Broadcast order intent
            await self._broadcast_order_update({
                "status": "placing",
                "action": signal.action,
                "quantity": qty,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "regime": regime.value
            })
            
            # === NEW: Simulation mode check ===
            if self.simulation_mode:
                logger.warning(f"🔶 SIMULATION: Would place {signal.action} order for {qty} contracts @ {current_price:.2f}")
                logger.warning(f"   SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
                
                # Update cooldown even in simulation to test timing
                self._last_trade_time = datetime.now(timezone.utc)
                
                await self._broadcast_order_update({
                    "status": "SIMULATED",
                    "action": signal.action,
                    "quantity": qty,
                    "fill_price": current_price,
                    "filled_quantity": qty,
                    "order_id": f"SIM-{datetime.now().strftime('%H%M%S')}"
                })
                return
            
            # Place order
            result = await self.executor.place_order(
                action=signal.action,
                quantity=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
                rationale=self.current_trade_rationale,
                features=self.current_trade_features,
                market_regime=regime.value
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
                # === NEW: Update cooldown after successful trade ===
                self._last_trade_time = datetime.now(timezone.utc)
                logger.info(f"⏱️ Trade placed - cooldown activated for {self._cooldown_seconds}s")
                
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
        
        except Exception as e:
            logger.error(f"❌ CRITICAL: _place_order failed with exception: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            await self._broadcast_error(f"Order placement failed: {e}")
    
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
    
    async def force_order_reconciliation(self):
        """Force reconciliation of active orders with IBKR."""
        if self.executor:
            logger.info("Forcing reconciliation of active orders with IBKR...")
            await self.executor._reconcile_orders()
            logger.info("Order reconciliation complete.")
    
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
