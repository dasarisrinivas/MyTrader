#!/usr/bin/env python3
"""Discover available ES contract months from IBKR."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ib_insync import IB, Future
from mytrader.utils.logger import configure_logging, logger
from mytrader.utils.settings_loader import load_settings

configure_logging(level="INFO")

async def discover_contracts():
    """Search for available ES futures contracts."""
    settings = load_settings(None)
    
    ib = IB()
    await ib.connectAsync(settings.data.ibkr_host, settings.data.ibkr_port, clientId=99)
    
    logger.info("Connected to IBKR, searching for ES futures contracts...")
    
    # Try different variations
    variations = [
        ("ES", "GLOBEX"),
        ("MES", "GLOBEX"),  # Micro E-mini
        ("ES", "CME"),
    ]
    
    for symbol, exchange in variations:
        logger.info(f"\nSearching for {symbol} on {exchange}...")
        try:
            contract = Future(symbol=symbol, exchange=exchange)
            details = await ib.reqContractDetailsAsync(contract)
            
            if details:
                logger.info(f"✅ Found {len(details)} contracts for {symbol}:")
                for i, detail in enumerate(details[:5], 1):  # Show first 5
                    c = detail.contract
                    logger.info(f"  {i}. {c.localSymbol} - Expiry: {c.lastTradeDateOrContractMonth}")
                if len(details) > 5:
                    logger.info(f"  ... and {len(details) - 5} more")
            else:
                logger.warning(f"❌ No contracts found for {symbol}")
        except Exception as e:
            logger.error(f"Error searching {symbol}: {e}")
    
    ib.disconnect()
    logger.info("\nDone!")

if __name__ == "__main__":
    try:
        asyncio.run(discover_contracts())
    except KeyboardInterrupt:
        logger.info("Interrupted")
