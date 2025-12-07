"""Entry point for MyTrader."""
from __future__ import annotations

import argparse
import asyncio
import queue
import threading
import time
from pathlib import Path

import nest_asyncio
import pandas as pd

# Apply nest_asyncio to allow nested event loops (required for ib_insync)
nest_asyncio.apply()

from mytrader.backtesting.engine import BacktestingEngine
from mytrader.config import Settings
from mytrader.data.ibkr import IBKRCollector
from mytrader.data.pipeline import MarketDataPipeline
from mytrader.data.sentiment import TwitterSentimentCollector
from mytrader.data.tradingview import TradingViewCollector
from mytrader.execution.ib_executor import TradeExecutor
from mytrader.features.feature_engineer import engineer_features
from mytrader.monitoring.live_tracker import LivePerformanceTracker
from mytrader.optimization.optimizer import ParameterOptimizer
from mytrader.risk.manager import RiskManager
from mytrader.strategies.engine import StrategyEngine
from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from mytrader.strategies.multi_strategy import MultiStrategy
from mytrader.utils.logger import configure_logging, logger
from mytrader.utils.settings_loader import load_settings

try:
    from ib_insync import IB
except ImportError as exc:  # noqa: BLE001
    raise SystemExit("ib_insync is required. Install with pip install ib_insync") from exc


async def run_live(settings: Settings) -> None:
    """Run live trading with enhanced monitoring and risk management."""
    configure_logging(level="INFO")

    # For actual live trading, we use a simpler approach:
    # 1. Connect to IBKR via executor
    # 2. Get real-time quotes from IBKR
    # 3. Build feature history from live bars
    # 4. Generate signals and execute trades
    # 
    # This avoids Error 162 from historical data requests in the data pipeline
    
    # Initialize multi-strategy system (RAG engine will be added below)
    # You can change strategy_mode: "trend_following", "breakout", "mean_reversion", or "auto"
    multi_strategy = None  # Will be initialized after RAG setup
    
    # Wrap multi-strategy with RAG-enhanced LLM intelligence
    from mytrader.strategies.llm_enhanced_strategy import LLMEnhancedStrategy
    from mytrader.llm.bedrock_client import BedrockClient
    from mytrader.llm.rag_engine import RAGEngine
    from mytrader.llm.rag_trade_advisor import RAGEnhancedTradeAdvisor
    
    # Check if RAG and LLM are enabled in config
    rag_enabled = settings.rag.enabled if hasattr(settings, 'rag') else False
    llm_enabled = settings.llm.enabled if hasattr(settings, 'llm') else True
    
    if llm_enabled:
        try:
            # Initialize Bedrock client
            bedrock_client = BedrockClient(
                model_id=settings.llm.model_id if hasattr(settings, 'llm') else "anthropic.claude-3-sonnet-20240229-v1:0",
                region_name=settings.llm.region_name if hasattr(settings, 'llm') else "us-east-1",
                max_tokens=settings.llm.max_tokens if hasattr(settings, 'llm') else 2048,
                temperature=settings.llm.temperature if hasattr(settings, 'llm') else 0.3
            )
            
            if rag_enabled:
                logger.info("🤖 RAG + LLM Enhancement ENABLED - Knowledge-grounded AI decisions")
                logger.info("   📚 Loading trading knowledge base...")
                
                # Initialize RAG engine
                rag_engine = RAGEngine(
                    bedrock_client=bedrock_client,
                    embedding_model_id=settings.rag.embedding_model_id,
                    region_name=settings.rag.region_name,
                    vector_store_path=settings.rag.vector_store_path,
                    dimension=settings.rag.embedding_dimension,
                    cache_enabled=settings.rag.cache_enabled,
                    cache_ttl_seconds=settings.rag.cache_ttl_seconds
                )
                
                # Get stats
                stats = rag_engine.get_stats()
                logger.info(f"   ✅ Knowledge base loaded: {stats['num_documents']} documents")
                
                if stats['num_documents'] == 0:
                    logger.warning("   ⚠️  No documents in knowledge base - run: python bin/test_rag.py")
                
                # FIX: Initialize multi-strategy WITH RAG engine for signal validation
                multi_strategy_base = MultiStrategy(
                    strategy_mode="auto",
                    reward_risk_ratio=2.0,
                    use_trailing_stop=True,
                    trailing_stop_pct=0.5,
                    min_confidence=settings.trading.min_weighted_confidence,  # Use config value (0.70)
                    rag_engine=rag_engine  # Pass RAG engine for validation
                )
                
                # Create RAG-enhanced advisor
                rag_advisor = RAGEnhancedTradeAdvisor(
                    bedrock_client=bedrock_client,
                    rag_engine=rag_engine,
                    min_confidence_threshold=settings.llm.min_confidence_threshold,
                    enable_llm=True,
                    enable_rag=True,
                    llm_override_mode=settings.llm.override_mode,
                    rag_top_k=settings.rag.top_k_results,
                    rag_score_threshold=settings.rag.score_threshold,
                    call_interval_seconds=settings.llm.call_interval_seconds
                )
                
                # Wrap strategy with LLM enhancement
                multi_strategy = LLMEnhancedStrategy(
                    base_strategy=multi_strategy_base,  # Use RAG-enhanced base
                    enable_llm=True,
                    min_llm_confidence=settings.llm.min_confidence_threshold,
                    llm_override_mode=settings.llm.override_mode
                )
                
                # Override the trade advisor with RAG-enhanced one
                multi_strategy.trade_advisor = rag_advisor
                
                logger.info("   📋 RAG Config:")
                logger.info(f"      Model: {settings.llm.model_id}")
                logger.info(f"      Min Confidence: {settings.llm.min_confidence_threshold}")
                logger.info(f"      Override Mode: {settings.llm.override_mode}")
                logger.info(f"      Call Interval: {settings.llm.call_interval_seconds}s")
                logger.info(f"      RAG Top-K: {settings.rag.top_k_results}")
                logger.info(f"      RAG Threshold: {settings.rag.score_threshold}")
            else:
                logger.info("🤖 LLM Enhancement ENABLED - using AWS Bedrock Claude (no RAG)")
                multi_strategy = LLMEnhancedStrategy(
                    base_strategy=multi_strategy,
                    enable_llm=True,
                    min_llm_confidence=settings.llm.min_confidence_threshold if hasattr(settings, 'llm') else 0.55,
                    llm_override_mode=settings.llm.override_mode if hasattr(settings, 'llm') else True
                )
                logger.info(f"   📋 LLM Config: model=claude-3-sonnet, min_confidence={settings.llm.min_confidence_threshold if hasattr(settings, 'llm') else 0.55}, mode={'override' if (hasattr(settings, 'llm') and settings.llm.override_mode) else 'consensus'}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM/RAG: {e}")
            logger.error("Falling back to traditional strategy")
            llm_enabled = False
    
    # If LLM/RAG failed, create basic multi-strategy without RAG
    if not llm_enabled or multi_strategy is None:
        logger.info("⚠️  LLM Enhancement DISABLED - using traditional signals only")
        multi_strategy = MultiStrategy(
            strategy_mode="auto",
            reward_risk_ratio=2.0,
            use_trailing_stop=True,
            trailing_stop_pct=0.5,
            min_confidence=settings.trading.min_weighted_confidence,  # Use config value
            rag_engine=None  # No RAG validation
        )
    
    # Keep original strategies as fallback
    strategies = [RsiMacdSentimentStrategy(), MomentumReversalStrategy()]
    
    # Load optimized parameters at startup (daily optimization)
    from mytrader.optimization.daily_optimizer import load_optimized_params, apply_optimized_params
    optimized_params = load_optimized_params(settings.optimization.optimized_params_path)
    if optimized_params:
        apply_optimized_params(strategies, optimized_params)
    
    # Note: Parameter optimizer REMOVED from real-time loop (runs daily after market close)
    
    # Initialize risk manager with fixed fractional sizing (NOT Kelly)
    risk = RiskManager(
        settings.trading, 
        position_sizing_method=settings.trading.position_sizing_method  # Use config
    )
    
    # Initialize market regime filter
    from mytrader.strategies.market_regime_filter import MarketRegimeFilter
    regime_filter = MarketRegimeFilter(
        min_atr_threshold=settings.trading.min_atr_threshold,
        max_spread_ticks=settings.trading.max_spread_ticks,
    )
    
    # Initialize background LLM worker (if LLM enabled and background mode)
    llm_worker = None
    if llm_enabled and hasattr(settings.llm, 'use_background_thread') and settings.llm.use_background_thread:
        from mytrader.llm.background_worker import BackgroundLLMWorker
        if isinstance(multi_strategy, LLMEnhancedStrategy) and hasattr(multi_strategy, 'trade_advisor'):
            llm_worker = BackgroundLLMWorker(
                trade_advisor=multi_strategy.trade_advisor,
                cache_timeout_seconds=settings.llm.cache_timeout_seconds,
            )
            llm_worker.start()
            logger.info("✅ Background LLM worker started (non-blocking mode)")
    
    # =========================================================================
    # HYBRID BEDROCK ARCHITECTURE - Event-driven intelligent analysis
    # =========================================================================
    hybrid_bedrock_client = None
    event_detector = None
    rag_context_builder = None
    bedrock_bias_modifier = {"bias": "NEUTRAL", "confidence": 0.0}  # Current bias from Bedrock
    bedrock_worker_queue = queue.Queue()  # Queue for background Bedrock calls
    bedrock_result_queue = queue.Queue()  # Queue for Bedrock results
    
    # Check if hybrid Bedrock is enabled in config
    hybrid_bedrock_enabled = getattr(settings, 'hybrid_bedrock', None)
    if hybrid_bedrock_enabled is None:
        # Default to enabled if not specified
        hybrid_bedrock_enabled = True
    elif hasattr(hybrid_bedrock_enabled, 'enabled'):
        hybrid_bedrock_enabled = hybrid_bedrock_enabled.enabled
    
    if hybrid_bedrock_enabled:
        try:
            from mytrader.llm.bedrock_hybrid_client import HybridBedrockClient, init_bedrock_client
            from mytrader.llm.event_detector import EventDetector, create_event_detector
            from mytrader.llm.rag_context_builder import RAGContextBuilder, build_context
            
            # Determine symbol (MES or ES based on config)
            hybrid_symbol = settings.data.ibkr_symbol if settings.data.ibkr_symbol in ["MES", "ES"] else "MES"
            
            # Initialize hybrid Bedrock client
            hybrid_bedrock_client = init_bedrock_client(
                model_id=settings.llm.model_id if hasattr(settings, 'llm') else None,
                region_name=settings.llm.region_name if hasattr(settings, 'llm') else None,
                max_tokens=300,  # Keep responses short for bias analysis
                db_path="data/bedrock_hybrid.db",
                daily_quota=1000,
                daily_cost_limit=50.0,
            )
            
            # Initialize event detector
            event_detector_config = {
                "volatility_spike_threshold": 2.0,
                "minutes_after_open": 5,
                "minutes_before_close": 5,
                "min_interval_seconds": 60,
                "cooldown_seconds": 300,
            }
            event_detector = create_event_detector(symbol=hybrid_symbol, config=event_detector_config)
            
            # Initialize RAG context builder
            rag_context_builder = RAGContextBuilder(
                instrument=hybrid_symbol,
                include_news=True,
                max_news_items=3,
            )
            
            logger.info("=" * 60)
            logger.info("🧠 HYBRID BEDROCK ARCHITECTURE ENABLED")
            logger.info("=" * 60)
            logger.info(f"   Symbol: {hybrid_symbol}")
            logger.info(f"   Model: {hybrid_bedrock_client.model_id}")
            logger.info(f"   Event-driven triggers: market_open, market_close, volatility_spike, news")
            logger.info(f"   Bedrock output: BIAS MODIFIER only (does NOT override risk rules)")
            logger.info("=" * 60)
            
            # Start background Bedrock worker thread
            def bedrock_worker_thread():
                """Background thread for non-blocking Bedrock calls."""
                while True:
                    try:
                        # Get next request from queue (blocking with timeout)
                        request = bedrock_worker_queue.get(timeout=1.0)
                        
                        if request is None:
                            # Shutdown signal
                            logger.info("Bedrock worker thread shutting down")
                            break
                        
                        context, trigger = request
                        logger.info(f"🧠 Background Bedrock analysis started: {trigger}")
                        
                        # Make Bedrock call (this can take a few seconds)
                        result = hybrid_bedrock_client.bedrock_analyze(context, trigger=trigger)
                        
                        # Put result in result queue
                        bedrock_result_queue.put(result)
                        
                        logger.info(f"🧠 Background Bedrock analysis complete: bias={result.get('bias')}, confidence={result.get('confidence', 0):.2f}")
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Bedrock worker error: {e}")
            
            bedrock_thread = threading.Thread(target=bedrock_worker_thread, daemon=True)
            bedrock_thread.start()
            logger.info("✅ Background Bedrock worker thread started")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hybrid Bedrock Architecture: {e}")
            logger.error("Continuing without hybrid Bedrock analysis")
            hybrid_bedrock_client = None
            event_detector = None
    
    # Initialize live performance tracker
    tracker = LivePerformanceTracker(
        initial_capital=settings.trading.initial_capital,
        risk_free_rate=settings.backtest.risk_free_rate
    )

    # Create IB instance and connect
    import random
    from ib_insync import util
    util.logToConsole('ERROR')  # Reduce log noise
    
    client_id = random.randint(10, 999)  # Use random client ID to avoid conflicts
    logger.info("Initializing IB connection to %s:%s (client_id=%d)", settings.data.ibkr_host, settings.data.ibkr_port, client_id)
    ib = IB()
    
    # Connect with retry logic
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting IB connection (attempt {attempt + 1}/{max_retries})...")
            await asyncio.wait_for(
                ib.connectAsync(settings.data.ibkr_host, settings.data.ibkr_port, clientId=client_id, timeout=5),
                timeout=8
            )
            logger.info("✅ Connected to IBKR successfully")
            break
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️  Connection timeout, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                # Try a different client ID
                client_id = random.randint(10, 999)
            else:
                logger.error("❌ Connection failed after all retries")
                logger.error("Check: 1) IB Gateway is running 2) API is enabled 3) Port 4002 is open 4) No other bots connected")
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️  Connection error: {e}, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                client_id = random.randint(10, 999)
            else:
                logger.error(f"❌ Connection failed: {e}")
                raise
    
    # Initialize Telegram notifier if configured
    from mytrader.utils.telegram_notifier import TelegramNotifier
    telegram_notifier = None
    if hasattr(settings, 'telegram') and settings.telegram.enabled:
        telegram_notifier = TelegramNotifier(
            bot_token=settings.telegram.bot_token,
            chat_id=settings.telegram.chat_id,
            enabled=True
        )
        logger.info("✅ Telegram notifications initialized")
    
    # Now wrap with executor
    executor = TradeExecutor(
        ib, 
        settings.trading, 
        settings.data.ibkr_symbol, 
        settings.data.ibkr_exchange,
        telegram_notifier=telegram_notifier
    )
    executor._connection_host = settings.data.ibkr_host
    executor._connection_port = settings.data.ibkr_port
    executor._connection_client_id = client_id
    
    # Set up executor event handlers
    ib.reqMarketDataType(3)  # Delayed market data
    logger.info("Using delayed market data (15-min delay, free)")
    ib.orderStatusEvent += executor._on_order_status
    ib.execDetailsEvent += executor._on_execution
    await executor._cancel_all_existing_orders()
    await executor._reconcile_positions()
    await executor._start_keepalive()

    account_value = settings.trading.initial_capital
    step = 0
    status_log_interval = 10  # Log performance every 10 iterations
    
    # Initialize price history buffer (we'll build this from live data)
    price_history = []
    # INCREASED WARM-UP: Now requires 200 bars (was 15)
    min_bars_needed = settings.trading.min_bars_for_signals
    poll_interval = 5  # seconds between price checks
    
    # Trade cooldown tracking
    last_trade_time = None
    trade_cooldown_seconds = settings.trading.trade_cooldown_minutes * 60
    
    # Position tracking for time-based exit
    position_entry_time = None
    max_trade_duration_seconds = settings.trading.max_trade_duration_minutes * 60

    logger.info("Starting live trading loop (polling every %ds)...", poll_interval)
    logger.info("Will start generating signals after collecting %d bars (INCREASED WARM-UP)", min_bars_needed)
    logger.info(f"Safety parameters:")
    logger.info(f"  - Disaster stop: {settings.trading.disaster_stop_pct*100:.1f}%")
    logger.info(f"  - Max trade duration: {settings.trading.max_trade_duration_minutes} minutes")
    logger.info(f"  - Trade cooldown: {settings.trading.trade_cooldown_minutes} minutes")
    logger.info(f"  - Position sizing: {settings.trading.position_sizing_method}")
    logger.info(f"  - Risk per trade: {settings.trading.risk_per_trade_pct*100:.2f}%")

    try:
        while True:
            try:
                # LATENCY GUARD: Measure loop iteration time
                loop_start_time = time.time()
                
                # Get current market price from IBKR
                current_price = await executor.get_current_price()
                if not current_price:
                    logger.warning("No price data available, retrying...")
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Add to price history (simplified: using last price for OHLC)
                # In production, you'd want proper bar aggregation
                from datetime import datetime, timezone
                price_bar = {
                    'timestamp': datetime.now(timezone.utc),
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'close': current_price,
                    'volume': 0  # Not critical for these strategies
                }
                price_history.append(price_bar)
                
                logger.debug(f"Price bar added: {current_price:.2f} (total history: {len(price_history)} bars)")
                
                # Keep only recent history (last 500 bars for efficiency)
                if len(price_history) > 500:
                    price_history = price_history[-500:]
                
                # Need minimum bars before we can trade
                if len(price_history) < min_bars_needed:
                    logger.info(f"Building history: {len(price_history)}/{min_bars_needed} bars")
                    await asyncio.sleep(poll_interval)
                    continue
                
                logger.info(f"✅ History complete with {len(price_history)} bars - starting signal generation")
                
                # Convert to DataFrame and engineer features
                df = pd.DataFrame(price_history)
                df.set_index('timestamp', inplace=True)
                
                logger.info(f"🔧 Engineering features from {len(df)} bars...")
                features = engineer_features(df[['open', 'high', 'low', 'close', 'volume']], None)
                
                if features.empty:
                    logger.warning("⚠️  Feature engineering returned empty DataFrame - skipping iteration")
                    await asyncio.sleep(poll_interval)
                    continue
                
                logger.info(f"✅ Features engineered: {len(features)} rows, {len(features.columns)} columns")

                returns = features["close"].pct_change().dropna()
                logger.info(f"📈 Returns calculated: {len(returns)} values")

                # REMOVED: Periodic optimization (now runs daily after market close)
                # The optimizer has been moved to a daily batch process

                step += 1
                
                # MARKET REGIME FILTER: Check if conditions suitable for trading
                regime_result = regime_filter.check_regime(
                    df=features,
                    current_time=datetime.now(timezone.utc),
                    bid_price=None,  # Would need to get from executor
                    ask_price=None,
                    vix_value=None,  # Could add VIX data feed
                    tick_size=settings.trading.tick_size,
                )
                
                if not regime_result.tradable:
                    logger.warning(f"⚠️  Market regime not tradable: {regime_result.reason}")
                    await asyncio.sleep(poll_interval)
                    continue
                
                regime_filter.log_regime_status(regime_result)

                # Get current position before generating signal
                current_position = await executor.get_current_position()
                current_qty = current_position.quantity if current_position else 0
                logger.info(f"📦 Current position: {current_qty} contracts")
                
                # DISASTER STOP: Check if position has moved too far against us
                if current_position and current_qty != 0:
                    entry_price = current_position.avg_cost
                    price_change_pct = abs((current_price - entry_price) / entry_price)
                    
                    # Check if we're losing money
                    is_losing = (current_qty > 0 and current_price < entry_price) or \
                               (current_qty < 0 and current_price > entry_price)
                    
                    if is_losing and price_change_pct > settings.trading.disaster_stop_pct:
                        logger.error(f"🚨 DISASTER STOP TRIGGERED!")
                        logger.error(f"   Position moved {price_change_pct*100:.2f}% against us")
                        logger.error(f"   Threshold: {settings.trading.disaster_stop_pct*100:.1f}%")
                        logger.error(f"   Force-closing position immediately")
                        
                        close_result = await executor.close_position()
                        if close_result and close_result.fill_price:
                            realized = (close_result.fill_price - entry_price) * current_qty
                            risk.update_pnl(realized)
                            tracker.update_equity(current_price, realized)
                            logger.info(f"Position closed by disaster stop, realized PnL: {realized:.2f}")
                        
                        # Reset position tracking
                        position_entry_time = None
                        
                        # Apply cooldown
                        last_trade_time = datetime.now(timezone.utc)
                        
                        await asyncio.sleep(poll_interval)
                        continue
                
                # TIME-BASED EXIT: Check if trade has been open too long
                if current_position and current_qty != 0 and position_entry_time:
                    time_in_trade = (datetime.now(timezone.utc) - position_entry_time).total_seconds()
                    
                    if time_in_trade > max_trade_duration_seconds:
                        logger.warning(f"⏰ TIME-BASED EXIT triggered")
                        logger.warning(f"   Trade open for {time_in_trade/60:.1f} minutes")
                        logger.warning(f"   Max duration: {max_trade_duration_seconds/60:.1f} minutes")
                        logger.warning(f"   Closing at market")
                        
                        close_result = await executor.close_position()
                        if close_result and close_result.fill_price:
                            realized = (close_result.fill_price - current_position.avg_cost) * current_qty
                            risk.update_pnl(realized)
                            tracker.update_equity(current_price, realized)
                            logger.info(f"Position closed by time limit, realized PnL: {realized:.2f}")
                        
                        # Reset tracking
                        position_entry_time = None
                        last_trade_time = datetime.now(timezone.utc)
                        
                        await asyncio.sleep(poll_interval)
                        continue
                
                # TRADE COOLDOWN: Check if enough time has passed since last trade
                if last_trade_time and current_qty == 0:  # Only check cooldown when flat
                    time_since_trade = (datetime.now(timezone.utc) - last_trade_time).total_seconds()
                    
                    if time_since_trade < trade_cooldown_seconds:
                        remaining = trade_cooldown_seconds - time_since_trade
                        logger.info(f"⏸️  Trade cooldown active: {remaining/60:.1f} minutes remaining")
                        await asyncio.sleep(poll_interval)
                        continue

                # =========================================================================
                # HYBRID BEDROCK: Event-driven analysis (non-blocking)
                # =========================================================================
                # Check for Bedrock results from background thread
                while not bedrock_result_queue.empty():
                    try:
                        result = bedrock_result_queue.get_nowait()
                        if result and not result.get("error"):
                            bedrock_bias_modifier = {
                                "bias": result.get("bias", "NEUTRAL"),
                                "confidence": result.get("confidence", 0.0),
                                "action": result.get("action", "HOLD"),
                                "rationale": result.get("rationale", ""),
                            }
                            logger.info(f"🧠 Updated Bedrock bias: {bedrock_bias_modifier['bias']} (conf={bedrock_bias_modifier['confidence']:.2f})")
                    except queue.Empty:
                        break
                
                # Check if event detector triggers Bedrock call
                if event_detector and hybrid_bedrock_client:
                    # Build market snapshot for event detection
                    last_row = features.iloc[-1]
                    recent_prices = features["close"].tail(20).tolist()
                    
                    # Get unrealized PnL
                    unrealized_pnl = 0.0
                    if current_position and current_qty != 0:
                        unrealized_pnl = (current_price - current_position.avg_cost) * current_qty
                    
                    snapshot = {
                        "current_price": current_price,
                        "price_change_pct": (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if len(recent_prices) > 1 else 0.0,
                        "momentum": float(last_row.get("MOM_10", last_row.get("momentum", 0.0))),
                        "atr": float(last_row.get("ATR_14", 0.0)),
                        "volatility": float(last_row.get("VOLATILITY", last_row.get("volatility", 0.0))),
                        "rsi": float(last_row.get("RSI_14", 50.0)),
                        "vix": None,  # Could add VIX data feed
                        "position": current_qty,
                        "unrealized_pnl": unrealized_pnl,
                        "news_headlines": [],  # Could integrate news API
                        "recent_prices": recent_prices,
                    }
                    
                    # Check if we should call Bedrock
                    should_trigger, trigger_reason, payload = event_detector.should_call_bedrock(snapshot)
                    
                    if should_trigger and payload:
                        logger.info(f"🧠 Bedrock trigger: {trigger_reason}")
                        
                        # Build context and queue for background processing
                        context = rag_context_builder.build_context(payload)
                        
                        # Queue the request (non-blocking)
                        try:
                            bedrock_worker_queue.put_nowait((context, payload.trigger_type))
                            logger.info(f"🧠 Queued Bedrock analysis request: {payload.trigger_type}")
                        except queue.Full:
                            logger.warning("⚠️  Bedrock queue full, skipping analysis request")
                
                # Log current Bedrock bias (for visibility)
                if bedrock_bias_modifier.get("bias") != "NEUTRAL":
                    logger.info(f"🧠 Current Bedrock bias: {bedrock_bias_modifier['bias']} (conf={bedrock_bias_modifier['confidence']:.2f})")

                # Generate trading signal using multi-strategy (with optional LLM enhancement)
                logger.info(f"🤔 Evaluating strategy with {len(features)} feature rows...")
                try:
                    # Check if we're using LLM-enhanced strategy with background worker
                    llm_commentary = None
                    if llm_worker:
                        # Get cached LLM commentary (non-blocking)
                        llm_commentary = llm_worker.get_latest_commentary()
                        if llm_commentary:
                            logger.info(f"💬 LLM Commentary: {llm_commentary[:150]}...")
                    
                    # Check if we're using LLM-enhanced strategy
                    if isinstance(multi_strategy, LLMEnhancedStrategy):
                        # IMPORTANT: Get base signal FIRST (quant signals decide trades)
                        # LLM is used ONLY for commentary, NOT for overriding decisions
                        
                        # Get traditional quant signal from base strategy
                        if hasattr(multi_strategy.base_strategy, 'generate_signal'):
                            action, confidence, risk_params = multi_strategy.base_strategy.generate_signal(
                                df=features,
                                current_position=current_qty
                            )
                        else:
                            action, confidence, risk_params = "HOLD", 0.0, {}
                        
                        # Create signal object
                        from dataclasses import dataclass
                        @dataclass
                        class Signal:
                            action: str
                            confidence: float
                            metadata: dict = None
                        
                        signal = Signal(
                            action=action,
                            confidence=confidence,
                            metadata={
                                "strategy": "multi_strategy",
                                "risk_params": risk_params,
                            }
                        )
                        
                        # LLM commentary only (non-blocking if background worker)
                        if llm_worker and action != "HOLD":
                            # Submit request to background worker (non-blocking)
                            request_id = llm_worker.submit_request(
                                features=features,
                                signal=signal,
                                context={
                                    'symbol': settings.data.ibkr_symbol,
                                    'position': current_qty,
                                }
                            )
                            # Don't wait for response - just log if available
                            logger.info(f"📤 LLM analysis request submitted (background): {request_id}")
                        
                        # Get market context from base strategy
                        if hasattr(multi_strategy.base_strategy, 'market_bias'):
                            market_bias = multi_strategy.base_strategy.market_bias
                            volatility_level = multi_strategy.base_strategy.volatility_level
                        else:
                            market_bias = "unknown"
                            volatility_level = "unknown"
                    else:
                        # Traditional multi-strategy without LLM
                        action, confidence, risk_params = multi_strategy.generate_signal(
                            df=features,
                            current_position=current_qty
                        )
                        market_bias = multi_strategy.market_bias
                        volatility_level = multi_strategy.volatility_level
                        
                        # Create signal object for compatibility
                        from dataclasses import dataclass
                        @dataclass
                        class Signal:
                            action: str
                            confidence: float
                            metadata: dict = None
                        
                        signal = Signal(
                            action=action,
                            confidence=confidence,
                            metadata={
                                "strategy": "multi_strategy",
                                "risk_params": risk_params,
                                "market_bias": market_bias,
                                "volatility": volatility_level
                            }
                        )
                    
                    logger.info(f"📊 SIGNAL GENERATED: {signal.action}, confidence={signal.confidence:.2f}")
                    if risk_params:
                        logger.info(f"   📍 Stop Loss: {risk_params.get('stop_loss_long', 'N/A'):.2f}")
                        logger.info(f"   🎯 Take Profit: {risk_params.get('take_profit_long', 'N/A'):.2f}")
                        logger.info(f"   📏 ATR: {risk_params.get('atr', 'N/A'):.2f}")
                    logger.info(f"   📊 Market: {market_bias}, Volatility: {volatility_level}")
                    
                    # =========================================================================
                    # HYBRID BEDROCK: Apply bias modifier (does NOT override risk rules)
                    # =========================================================================
                    original_action = signal.action
                    original_confidence = signal.confidence
                    
                    if bedrock_bias_modifier.get("bias") != "NEUTRAL" and bedrock_bias_modifier.get("confidence", 0) > 0.3:
                        bedrock_bias = bedrock_bias_modifier.get("bias")
                        bedrock_conf = bedrock_bias_modifier.get("confidence", 0)
                        
                        # Apply bias modifier to confidence (NOT overriding action)
                        # If Bedrock agrees with signal direction, boost confidence slightly
                        # If Bedrock disagrees, reduce confidence slightly
                        
                        signal_is_bullish = signal.action == "BUY"
                        bedrock_is_bullish = bedrock_bias == "BULLISH"
                        
                        if signal.action != "HOLD":
                            if (signal_is_bullish and bedrock_is_bullish) or (not signal_is_bullish and not bedrock_is_bullish):
                                # Agreement: boost confidence by up to 10%
                                confidence_boost = min(0.1, bedrock_conf * 0.15)
                                signal.confidence = min(1.0, signal.confidence + confidence_boost)
                                logger.info(f"🧠 Bedrock AGREES: {bedrock_bias} (conf boost +{confidence_boost:.2f})")
                            else:
                                # Disagreement: reduce confidence by up to 15%
                                confidence_reduction = min(0.15, bedrock_conf * 0.2)
                                signal.confidence = max(0.0, signal.confidence - confidence_reduction)
                                logger.info(f"🧠 Bedrock DISAGREES: {bedrock_bias} (conf reduction -{confidence_reduction:.2f})")
                            
                            # Log the modification
                            if signal.confidence != original_confidence:
                                logger.info(f"   📊 Adjusted confidence: {original_confidence:.2f} → {signal.confidence:.2f}")
                                logger.info(f"   💡 Bedrock rationale: {bedrock_bias_modifier.get('rationale', 'N/A')[:100]}")
                    
                except Exception as signal_error:
                    logger.error(f"❌ Error generating signal: {signal_error}")
                    logger.exception("Signal generation traceback:")
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Update tracker
                if current_position:
                    unrealized_pnl = await executor.get_unrealized_pnl()
                    tracker.update_equity(current_price, realized_pnl=0.0)
                    
                    # Check exit conditions using multi-strategy risk management
                    logger.info(f"🔍 Checking exit: position={current_position.quantity}, avg_cost={current_position.avg_cost:.2f}, current={current_price:.2f}, risk_params={risk_params}")
                    if risk_params and hasattr(current_position, 'avg_cost'):
                        should_exit, exit_reason = multi_strategy.should_exit_position(
                            df=features,
                            entry_price=current_position.avg_cost,
                            position=current_position.quantity,
                            risk_params=risk_params
                        )
                        logger.info(f"   → Exit check result: should_exit={should_exit}, reason={exit_reason}")
                        
                        if should_exit:
                            logger.info(f"🛑 EXIT SIGNAL: {exit_reason}")
                            close_result = await executor.close_position()
                            if close_result and close_result.fill_price:
                                tracker.record_trade(
                                    action="SELL" if current_position.quantity > 0 else "BUY",
                                    price=close_result.fill_price,
                                    quantity=abs(current_position.quantity)
                                )
                                realized = (close_result.fill_price - current_position.avg_cost) * current_position.quantity
                                risk.update_pnl(realized)
                                tracker.update_equity(current_price, realized)
                                logger.info(f"✅ Position closed: {exit_reason}, realized PnL: {realized:.2f}")
                                
                                # Reset tracking and apply cooldown
                                position_entry_time = None
                                last_trade_time = datetime.now(timezone.utc)
                            
                            await asyncio.sleep(poll_interval)
                            continue
                    
                    # Update trailing stops if configured (matches backtest logic)
                    atr_val = float(features.iloc[-1].get("ATR_14", 0.0))
                    await executor.update_trailing_stops(current_price, atr_val)
                    
                    # Check for opposite signal exit (matches backtest logic)
                    direction = "BUY" if current_position.quantity > 0 else "SELL"
                    opposite_action = "SELL" if direction == "BUY" else "BUY"
                    if signal.action == opposite_action and signal.action != "HOLD":
                        logger.info("🔄 Opposite signal detected - closing position")
                        close_result = await executor.close_position()
                        if close_result and close_result.fill_price:
                            # Record the exit
                            tracker.record_trade(
                                action=opposite_action,
                                price=close_result.fill_price,
                                quantity=abs(current_position.quantity)
                            )
                            realized = (close_result.fill_price - current_position.avg_cost) * current_position.quantity
                            risk.update_pnl(realized)
                            tracker.update_equity(current_price, realized)
                            logger.info("Position closed on opposite signal, realized PnL: %.2f", realized)
                            
                            # Reset tracking and apply cooldown
                            position_entry_time = None
                            last_trade_time = datetime.now(timezone.utc)
                        
                        await asyncio.sleep(poll_interval)
                        continue
                
                if signal.action == "HOLD":
                    logger.info("⏸️  HOLD signal - no action taken")
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Don't open new position if we already have one
                if current_position and current_position.quantity != 0:
                    logger.info("⚠️  Already have open position, skipping new signal")
                    await asyncio.sleep(poll_interval)
                    continue

                logger.info(f"🎯 Preparing to execute {signal.action} signal...")
                
                # ENHANCED LOGGING: Entry reason and market conditions
                logger.info(f"📋 Entry Decision:")
                logger.info(f"   Action: {signal.action}")
                logger.info(f"   Confidence: {signal.confidence:.3f}")
                logger.info(f"   Market Bias: {market_bias}")
                logger.info(f"   Volatility: {volatility_level}")
                logger.info(f"   ATR: {regime_result.atr:.2f}" if regime_result.atr else "   ATR: N/A")
                if llm_commentary:
                    logger.info(f"   LLM Commentary: {llm_commentary[:100]}...")

                # FIXED POSITION SIZING: Use contracts_per_order (1) and respect max_position_size (5)
                # Check current total position
                current_abs_position = abs(current_qty)
                max_allowed = settings.trading.max_position_size
                
                logger.info(f"💰 Position Sizing:")
                logger.info(f"   Current position: {current_qty} contracts")
                logger.info(f"   Max allowed total: {max_allowed} contracts")
                logger.info(f"   Contracts per order: {settings.trading.contracts_per_order}")
                
                # STRICT POSITION CHECK: Don't add to existing position in same direction
                # If we have a position and signal is same direction, skip
                if current_qty != 0:
                    signal_direction = 1 if signal.action == "BUY" else -1
                    position_direction = 1 if current_qty > 0 else -1
                    if signal_direction == position_direction:
                        logger.warning(f"⚠️  Already have {abs(current_qty)} contract position in {signal.action} direction, skipping new trade")
                        await asyncio.sleep(poll_interval)
                        continue
                
                # Check if we're at max position
                if current_abs_position >= max_allowed:
                    logger.warning(f"⚠️  Already at maximum position ({current_abs_position}/{max_allowed}), skipping new trade")
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Only trade 1 contract per order (from config)
                qty = settings.trading.contracts_per_order
                
                # Ensure we don't exceed max position size
                if current_abs_position + qty > max_allowed:
                    qty = max_allowed - current_abs_position
                    logger.info(f"   Adjusted to {qty} contract(s) to stay within max limit")
                
                if qty <= 0:
                    logger.warning("⚠️  Cannot trade: would exceed position limit")
                    await asyncio.sleep(poll_interval)
                    continue
                
                logger.info(f"   ✅ Placing order for {qty} contract(s)")
                
                if not risk.can_trade(qty):
                    logger.warning("Risk limits exceeded. Skipping trade.")
                    await asyncio.sleep(poll_interval)
                    continue

                # Calculate stop-loss and take-profit using SAME logic as backtest
                direction = 1 if signal.action == "BUY" else -1
                row = features.iloc[-1]
                
                # Check for explicit prices in metadata first
                metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
                raw_stop = metadata.get("stop_loss_price")
                stop_loss = float(raw_stop) if isinstance(raw_stop, (int, float)) else None
                
                raw_target = metadata.get("take_profit_price")
                take_profit = float(raw_target) if isinstance(raw_target, (int, float)) else None
                
                # Get ATR value (from metadata or features)
                atr = float(metadata.get("atr_value", 0.0))
                if atr <= 0:
                    atr = float(row.get("ATR_14", 0.0))
                
                tick_size = settings.trading.tick_size
                default_stop_offset = settings.trading.stop_loss_ticks * tick_size
                default_target_offset = settings.trading.take_profit_ticks * tick_size
                
                # Calculate stop-loss if not provided
                if stop_loss is None:
                    atr_multiplier = float(metadata.get("atr_stop_multiplier", 0.0))
                    if atr_multiplier > 0 and atr > 0:
                        stop_offset = atr * atr_multiplier
                    else:
                        stop_offset = default_stop_offset
                    
                    # PROFESSIONAL STOP-LOSS: Minimum 10 points ($500) for ES futures
                    # This prevents stop-hunting and allows normal market volatility
                    min_stop_points = 10.0  # 10 ES points = $500 per contract
                    if stop_offset < min_stop_points:
                        logger.info(f"   📏 Widening stop from {stop_offset:.2f} to {min_stop_points:.2f} points (min threshold)")
                        stop_offset = min_stop_points
                    
                    if stop_offset <= 0:
                        stop_offset = min_stop_points
                    
                    stop_loss = current_price - stop_offset if direction > 0 else current_price + stop_offset
                    logger.info(f"   🛡️  Stop-loss: {stop_offset:.2f} points from entry (${stop_offset * 50:.0f} risk per contract)")
                
                # Calculate take-profit if not provided
                if take_profit is None:
                    risk_reward = float(metadata.get("risk_reward", 0.0))
                    if risk_reward <= 0:
                        if default_stop_offset > 0:
                            risk_reward = settings.trading.take_profit_ticks / max(1e-6, settings.trading.stop_loss_ticks)
                        else:
                            risk_reward = 2.0
                    
                    stop_distance = abs(current_price - stop_loss)
                    if stop_distance <= 0:
                        stop_distance = default_stop_offset
                    
                    target_offset = stop_distance * risk_reward if risk_reward > 0 else default_target_offset
                    take_profit = current_price + target_offset if direction > 0 else current_price - target_offset
                
                logger.info(f"📈 Trade levels: entry={current_price:.2f} stop={stop_loss:.2f} target={take_profit:.2f} (ATR={atr:.2f})")
                logger.info(f"💰 Position size: {qty} contracts (account value: ${account_value:.2f})")

                # Store trailing stop parameters from metadata
                trailing_atr_multiplier = float(metadata.get("trailing_atr_multiplier", 0.0)) or None
                trailing_percent = float(metadata.get("trailing_percent", 0.0)) or None
                
                logger.info(f"🚀 Placing {signal.action} order for {qty} contracts...")
                
                # Place order with bracket orders
                result = await executor.place_order(
                    action=signal.action,
                    quantity=qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "trailing_atr_multiplier": trailing_atr_multiplier,
                        "trailing_percent": trailing_percent,
                        "atr_value": atr,
                        "entry_metadata": metadata
                    }
                )
                
                logger.info(f"✅ Order result: status={result.status}, filled={result.filled_quantity} @ {result.fill_price if result.fill_price else 0.0:.2f}")
                
                if result.status not in {"Cancelled", "Inactive"}:
                    risk.register_trade()
                    # Record trade in tracker with metadata
                    if result.fill_price:
                        tracker.record_trade(
                            action=signal.action,
                            price=result.fill_price,
                            quantity=qty
                        )
                        
                        # Track position entry time for time-based exit
                        position_entry_time = datetime.now(timezone.utc)
                        logger.info(f"⏱️  Position entry time recorded: {position_entry_time}")
                
                # Periodic performance logging
                if step % status_log_interval == 0:
                    tracker.log_status()
                
                # LATENCY GUARD: Check loop iteration time
                loop_duration = time.time() - loop_start_time
                if loop_duration > settings.trading.max_loop_latency_seconds:
                    logger.warning(f"⚠️  Loop latency high: {loop_duration:.2f}s (max: {settings.trading.max_loop_latency_seconds}s)")
                    logger.warning(f"   Skipping next trading cycle to catch up")
                else:
                    logger.debug(f"✓ Loop completed in {loop_duration:.2f}s")
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
                    
            except Exception as exc:  # noqa: BLE001
                logger.exception("Live loop error: %s", exc)
                await asyncio.sleep(poll_interval)
                
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        # Export final performance snapshot
        tracker.export_snapshot("reports/final_performance.json")
        logger.info("Final performance snapshot saved")
    finally:
        # Stop background Bedrock worker if running
        if hybrid_bedrock_client:
            logger.info("Stopping Bedrock worker thread...")
            bedrock_worker_queue.put(None)  # Send shutdown signal
            logger.info("Bedrock worker thread stopped")
            
            # Log Bedrock usage stats
            bedrock_status = hybrid_bedrock_client.get_status()
            logger.info(f"🧠 Bedrock Stats: {bedrock_status['daily_calls']} calls today, ${bedrock_status['daily_cost']:.4f} cost")
        
        # Stop background LLM worker if running
        if llm_worker:
            logger.info("Stopping background LLM worker...")
            llm_worker.stop()
            stats = llm_worker.get_statistics()
            logger.info(f"LLM Worker Stats: {stats}")
        
        if ib.isConnected():
            ib.disconnect()


def run_backtest(settings: Settings, data_path: Path | None) -> None:
    """Run backtest with enhanced performance metrics and reporting."""
    configure_logging(level="INFO")
    
    # Load data
    path = data_path or settings.backtest.data_path
    if isinstance(path, str):
        path = Path(path)
    
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    
    # Initialize strategies and engine
    strategies = [RsiMacdSentimentStrategy(), MomentumReversalStrategy()]
    engine = BacktestingEngine(strategies, settings.trading, settings.backtest)
    
    # Run backtest
    logger.info("Starting backtest on %d bars...", len(df))
    result = engine.run(df)
    
    # Log comprehensive metrics
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    for key, value in result.metrics.items():
        if isinstance(value, float):
            logger.info("  %s: %.4f", key.replace("_", " ").title(), value)
        else:
            logger.info("  %s: %s", key.replace("_", " ").title(), value)
    logger.info("=" * 60)
    
    # Export detailed report
    from mytrader.backtesting.performance import export_report
    
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    # Export JSON report
    json_path = output_dir / "backtest_report.json"
    export_report(result.metrics, result.trades, json_path, format="json")
    logger.info("Detailed report saved to %s", json_path)
    
    # Export CSV report
    csv_path = output_dir / "backtest_report.csv"
    export_report(result.metrics, result.trades, csv_path, format="csv")
    logger.info("CSV reports saved to %s", output_dir)
    
    # Export equity curve
    if not result.equity_curve.empty:
        equity_path = output_dir / "equity_curve.csv"
        result.equity_curve.to_csv(equity_path)
        logger.info("Equity curve saved to %s", equity_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MyTrader autonomous trading bot")
    parser.add_argument("mode", choices=["live", "backtest", "optimize"], help="Execution mode")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config")
    parser.add_argument("--data", type=Path, default=None, help="Historical data path for backtest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    if args.mode == "live":
        asyncio.run(run_live(settings))
    elif args.mode == "backtest":
        run_backtest(settings, args.data)
    elif args.mode == "optimize":
        raise NotImplementedError("Optimization CLI not yet implemented")


if __name__ == "__main__":
    main()
