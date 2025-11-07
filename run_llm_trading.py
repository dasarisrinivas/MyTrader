#!/usr/bin/env python3
"""
Simple script to run live trading with LLM enhancement enabled.
This avoids the async event loop issues in the dashboard.
"""
import sys
import logging
from mytrader.utils.settings_loader import load_settings
from mytrader.strategies.multi_strategy import MultiStrategy
from mytrader.strategies.llm_enhanced_strategy import LLMEnhancedStrategy
from mytrader.execution.ib_executor import TradeExecutor
from ib_insync import IB
import asyncio

# Setup logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/llm_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def main():
    # Load config
    logger.info("📖 Loading configuration from config.yaml...")
    settings = load_settings('config.yaml')
    
    # Create base strategy
    logger.info("🎯 Initializing MultiStrategy...")
    base_strategy = MultiStrategy(strategy_mode='auto', reward_risk_ratio=2.0)
    
    # Wrap with LLM if enabled
    if settings.llm.enabled:
        logger.info("🤖 LLM Enhancement ENABLED - wrapping with LLMEnhancedStrategy")
        logger.info(f"   Model: {settings.llm.model_id}")
        logger.info(f"   Min Confidence: {settings.llm.min_confidence_threshold}")
        logger.info(f"   Override Mode: {settings.llm.override_mode}")
        
        strategy = LLMEnhancedStrategy(
            base_strategy=base_strategy,
            enable_llm=True,
            min_llm_confidence=settings.llm.min_confidence_threshold,
            llm_override_mode=settings.llm.override_mode
        )
        logger.info("✅ LLM Enhancement initialized successfully")
    else:
        logger.info("⚠️  LLM Enhancement DISABLED - using traditional strategy only")
        strategy = base_strategy
    
    # Connect to IB
    logger.info(f"🔌 Connecting to Interactive Brokers at {settings.data.ibkr_host}:{settings.data.ibkr_port}...")
    ib = IB()
    executor = TradeExecutor(
        ib, 
        settings.trading, 
        settings.data.ibkr_symbol, 
        settings.data.ibkr_exchange
    )
    
    try:
        await executor.connect(settings.data.ibkr_host, settings.data.ibkr_port, client_id=5, timeout=120)
        logger.info("✅ Connected to Interactive Brokers")
        
        # Simple trading loop - just demonstrate the LLM is working
        logger.info("🚀 Starting trading loop - will generate signals every 60 seconds")
        logger.info("   Watch for '🤖 LLM Recommendation' messages in the logs")
        logger.info("   Press Ctrl+C to stop")
        
        from mytrader.features.feature_engineer import engineer_features
        from datetime import datetime, timezone
        import pandas as pd
        
        price_history = []
        min_bars_needed = 50
        poll_interval = 5  # Check price every 5 seconds (takes ~4 minutes to collect 50 bars)
        
        while True:
            try:
                # Get current market price
                logger.info("📊 Fetching current market price...")
                current_price = await executor.get_current_price()
                
                if current_price:
                    # Add to price history
                    price_bar = {
                        'timestamp': datetime.now(timezone.utc),
                        'open': current_price,
                        'high': current_price,
                        'low': current_price,
                        'close': current_price,
                        'volume': 0
                    }
                    price_history.append(price_bar)
                    
                    # Keep only recent history
                    if len(price_history) > 500:
                        price_history = price_history[-500:]
                    
                    logger.info(f"   Price: ${current_price:.2f} (history: {len(price_history)} bars)")
                    
                    # Need minimum bars before trading
                    if len(price_history) >= min_bars_needed:
                        # Convert to DataFrame
                        df = pd.DataFrame(price_history)
                        df.set_index('timestamp', inplace=True)
                        
                        # Engineer features
                        features = engineer_features(df[['open', 'high', 'low', 'close', 'volume']], None)
                        logger.info(f"   Engineered {len(features)} feature rows")
                        
                        # Generate signal - this is where LLM gets called!
                        logger.info("🤔 Generating trading signal with LLM...")
                        result = strategy.generate(features)
                        
                        # Handle both tuple and Signal object returns
                        if isinstance(result, tuple):
                            signal, metadata = result
                        else:
                            signal = result
                            metadata = {}
                        
                        logger.info(f"📊 SIGNAL: {signal.action}, Confidence: {signal.confidence:.2f}")
                        if metadata:
                            logger.info(f"   💡 LLM Analysis: {metadata}")
                    else:
                        logger.info(f"   Building history: {len(price_history)}/{min_bars_needed} bars")
                else:
                    logger.warning("⚠️  No price data received")
                
                logger.info(f"💤 Sleeping {poll_interval} seconds before next check...")
                await asyncio.sleep(poll_interval)
                
            except KeyboardInterrupt:
                logger.info("👋 Stopping trading...")
                break
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(10)
        
    finally:
        logger.info("🔌 Disconnecting from Interactive Brokers...")
        ib.disconnect()
        logger.info("✅ Stopped")

if __name__ == "__main__":
    asyncio.run(main())
