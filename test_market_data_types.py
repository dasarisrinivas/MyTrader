#!/usr/bin/env python3
"""Test what market data types are available."""
import asyncio
from ib_insync import IB, Future
from mytrader.utils.settings_loader import load_settings
from mytrader.utils.logger import configure_logging, logger

configure_logging(level="INFO")

async def test_market_data_types():
    """Test different market data types."""
    settings = load_settings(None)
    
    ib = IB()
    await ib.connectAsync(settings.data.ibkr_host, settings.data.ibkr_port, clientId=99)
    
    # Find front month contract
    contract = Future(symbol="ES", exchange="CME")
    details = await ib.reqContractDetailsAsync(contract)
    front_month = details[0].contract
    logger.info("Testing with contract: %s", front_month.localSymbol)
    
    # Test different market data types
    for data_type, name in [(1, "Live"), (2, "Frozen"), (3, "Delayed"), (4, "Delayed-Frozen")]:
        logger.info("\n--- Testing %s (type %d) ---", name, data_type)
        try:
            ib.reqMarketDataType(data_type)
            logger.info("Set market data type to %s", name)
            
            # Try to get a tick
            ticker = ib.reqMktData(front_month, "", False, False)
            await ib.sleep(3)
            
            if ticker.last or ticker.close or ticker.bid or ticker.ask:
                logger.info("✅ %s data WORKS! Last: %s, Bid: %s, Ask: %s", 
                           name, ticker.last, ticker.bid, ticker.ask)
            else:
                logger.warning("❌ %s data: No prices received", name)
            
            ib.cancelMktData(front_month)
            await ib.sleep(1)
            
        except Exception as e:
            logger.error("❌ %s data failed: %s", name, e)
    
    # Test real-time bars with different data types
    logger.info("\n--- Testing Real-Time Bars ---")
    for data_type, name in [(3, "Delayed"), (4, "Delayed-Frozen")]:
        try:
            logger.info("Testing real-time bars with %s (type %d)", name, data_type)
            ib.reqMarketDataType(data_type)
            bars = ib.reqRealTimeBars(front_month, barSize=5, whatToShow="TRADES", useRTH=False)
            await ib.sleep(10)  # Wait 10 seconds for a bar
            if bars:
                logger.info("✅ Real-time bars with %s: SUCCESS", name)
            else:
                logger.warning("❌ Real-time bars with %s: No bars", name)
            ib.cancelRealTimeBars(bars)
        except Exception as e:
            logger.error("❌ Real-time bars with %s failed: %s", name, e)
    
    # Test historical data
    logger.info("\n--- Testing Historical Data ---")
    try:
        bars = await ib.reqHistoricalDataAsync(
            front_month,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="5 mins",
            whatToShow="TRADES",
            useRTH=False
        )
        if bars:
            logger.info("✅ Historical data: SUCCESS (%d bars)", len(bars))
            logger.info("Latest bar: %s", bars[-1])
        else:
            logger.warning("❌ Historical data: No bars")
    except Exception as e:
        logger.error("❌ Historical data failed: %s", e)
    
    ib.disconnect()
    logger.info("\nTest complete!")

if __name__ == "__main__":
    asyncio.run(test_market_data_types())
