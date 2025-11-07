#!/usr/bin/env python3
"""Test IBKR streaming data collection."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mytrader.data.ibkr import IBKRCollector
from mytrader.utils.logger import configure_logging, logger
from mytrader.utils.settings_loader import load_settings

configure_logging(level="INFO")

async def test_stream():
    """Test streaming data from IBKR."""
    settings = load_settings(None)
    
    collector = IBKRCollector(
        host=settings.data.ibkr_host,
        port=settings.data.ibkr_port,
        client_id=1,
        symbol=settings.data.ibkr_symbol,
        exchange=settings.data.ibkr_exchange,
        currency=settings.data.ibkr_currency,
        max_retries=3,
        base_delay=1.0
    )
    
    logger.info("Starting IBKR stream test...")
    logger.info("Will collect up to 5 bars then exit")
    
    try:
        count = 0
        async for bar in collector.stream():
            count += 1
            logger.info("Bar %d: %s", count, bar)
            if count >= 5:
                logger.info("✅ Successfully received 5 bars, test passed!")
                break
    except Exception as e:
        logger.error("❌ Stream test failed: %s", e, exc_info=True)
        return False
    
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(test_stream())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(0)
