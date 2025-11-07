#!/usr/bin/env python3
"""Quick test to verify live trading can start without event loop errors."""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mytrader.config import Settings
from mytrader.utils.settings_loader import load_settings
from mytrader.utils.logger import configure_logging, logger

configure_logging(level="INFO")

async def test_startup():
    """Test that we can initialize components without event loop conflicts."""
    try:
        settings = load_settings(None)
        logger.info("✅ Settings loaded successfully")
        
        # Import and test IB components
        from ib_insync import IB
        from mytrader.data.ibkr import IBKRCollector
        from mytrader.execution.ib_executor import TradeExecutor
        
        logger.info("✅ Imports successful")
        
        # Test IBKRCollector initialization (doesn't connect yet)
        ib_collector = IBKRCollector(
            host=settings.data.ibkr_host,
            port=settings.data.ibkr_port,
            client_id=1,
            symbol=settings.data.ibkr_symbol,
            exchange=settings.data.ibkr_exchange,
            currency=settings.data.ibkr_currency,
            max_retries=3,
            base_delay=1.0
        )
        logger.info("✅ IBKRCollector initialized (client_id=1)")
        
        # Test TradeExecutor initialization
        ib = IB()
        executor = TradeExecutor(ib, settings.trading, settings.data.ibkr_symbol)
        logger.info("✅ TradeExecutor initialized")
        
        # Test connection with different client IDs
        logger.info("Testing connection to IB Gateway at %s:%s...", 
                   settings.data.ibkr_host, settings.data.ibkr_port)
        
        try:
            # Connect data collector (client_id=1)
            await ib_collector._connect()
            logger.info("✅ Data collector connected successfully (client_id=1)")
            
            # Connect executor (client_id=2)
            await executor.connect(settings.data.ibkr_host, settings.data.ibkr_port, client_id=2)
            logger.info("✅ Trade executor connected successfully (client_id=2)")
            
            # Check accounts
            accounts = executor.ib.managedAccounts()
            logger.info("✅ Connected accounts: %s", accounts)
            
            logger.info("\n" + "="*60)
            logger.info("🎉 SUCCESS! Both connections working with separate client IDs")
            logger.info("="*60)
            
            # Cleanup
            executor.ib.disconnect()
            ib_collector.ib.disconnect()
            logger.info("✅ Disconnected cleanly")
            
            return True
            
        except Exception as e:
            logger.error("❌ Connection test failed: %s", e)
            logger.error("Make sure IB Gateway is running on port %s", settings.data.ibkr_port)
            return False
            
    except Exception as e:
        logger.error("❌ Startup test failed: %s", e, exc_info=True)
        return False

def main():
    """Run the async test."""
    try:
        result = asyncio.run(test_startup())
        sys.exit(0 if result else 1)
    except RuntimeError as e:
        if "already running" in str(e):
            logger.error("❌ Event loop conflict detected!")
            logger.error("This means there's still a synchronous IB connection being made")
            sys.exit(2)
        raise

if __name__ == "__main__":
    main()
