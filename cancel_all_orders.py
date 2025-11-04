#!/usr/bin/env python3
"""Cancel all open orders in IBKR."""
import asyncio
from ib_insync import IB
from mytrader.config import Settings
from mytrader.utils.logger import configure_logging, logger

async def cancel_all_orders():
    """Cancel all open orders."""
    configure_logging(level="INFO")
    settings = Settings()
    
    ib = IB()
    logger.info(f"Connecting to IBKR at {settings.data.ibkr_host}:{settings.data.ibkr_port}...")
    
    try:
        await ib.connectAsync(
            settings.data.ibkr_host,
            settings.data.ibkr_port,
            clientId=999,  # Use unique client ID
            timeout=30
        )
        logger.info("✅ Connected to IBKR")
        
        # Get all open trades
        open_trades = ib.openTrades()
        
        if not open_trades:
            logger.info("No open orders to cancel")
            return
        
        logger.info(f"Found {len(open_trades)} open orders")
        
        # Cancel all orders
        canceled = 0
        for trade in open_trades:
            logger.info(f"Canceling order {trade.order.orderId}: {trade.order.action} {trade.order.totalQuantity} {trade.contract.symbol}")
            ib.cancelOrder(trade.order)
            canceled += 1
        
        # Wait for cancellations to process
        await ib.sleep(3)
        
        logger.info(f"✅ Successfully canceled {canceled} orders")
        
        # Check remaining orders
        remaining = ib.openTrades()
        if remaining:
            logger.warning(f"⚠️  {len(remaining)} orders still pending:")
            for trade in remaining:
                logger.warning(f"  - Order {trade.order.orderId}: {trade.orderStatus.status}")
        else:
            logger.info("✅ All orders canceled successfully")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
    finally:
        if ib.isConnected():
            ib.disconnect()
            logger.info("Disconnected from IBKR")

if __name__ == "__main__":
    asyncio.run(cancel_all_orders())
