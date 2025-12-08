#!/usr/bin/env python3
"""Cancel all open orders in IBKR - improved version with better waiting logic."""
import asyncio
import argparse
from ib_insync import IB
from mytrader.config import Settings
from mytrader.utils.logger import configure_logging, logger

async def cancel_all_orders(symbol: str = None, use_global_cancel: bool = False):
    """Cancel all open orders, optionally filtering by symbol.
    
    Args:
        symbol: If specified, only cancel orders for this symbol
        use_global_cancel: If True, use reqGlobalCancel() as last resort
    """
    configure_logging(level="INFO")
    settings = Settings()
    
    ib = IB()
    logger.info(f"Connecting to IBKR at {settings.data.ibkr_host}:{settings.data.ibkr_port}...")
    
    try:
        await ib.connectAsync(
            settings.data.ibkr_host,
            settings.data.ibkr_port,
            clientId=999,  # Use unique client ID to not conflict with bot
            timeout=30
        )
        logger.info("✅ Connected to IBKR")
        
        # Get all open trades
        open_trades = ib.openTrades()
        
        if not open_trades:
            logger.info("No open orders to cancel")
            return
        
        # Filter by symbol if specified
        if symbol:
            trades_to_cancel = [t for t in open_trades if t.contract.symbol == symbol]
            logger.info(f"Found {len(trades_to_cancel)} open orders for {symbol} (out of {len(open_trades)} total)")
        else:
            trades_to_cancel = open_trades
            logger.info(f"Found {len(trades_to_cancel)} open orders")
        
        if not trades_to_cancel:
            logger.info("No orders to cancel")
            return
        
        # Print order details
        for trade in trades_to_cancel:
            logger.info(f"  📋 Order {trade.order.orderId}: {trade.order.action} {trade.order.totalQuantity} {trade.contract.symbol} @ {getattr(trade.order, 'lmtPrice', 'MKT')} (status: {trade.orderStatus.status})")
        
        # Cancel orders
        logger.info(f"🔄 Canceling {len(trades_to_cancel)} orders...")
        for trade in trades_to_cancel:
            ib.cancelOrder(trade.order)
        
        # Wait for cancellations with polling
        max_wait = 15
        poll_interval = 1.0
        elapsed = 0.0
        
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            
            # Check remaining active orders
            current_trades = ib.openTrades()
            if symbol:
                still_active = [t for t in current_trades 
                               if t.contract.symbol == symbol 
                               and t.orderStatus.status in ['PreSubmitted', 'Submitted', 'PendingSubmit', 'PendingCancel']]
            else:
                still_active = [t for t in current_trades 
                               if t.orderStatus.status in ['PreSubmitted', 'Submitted', 'PendingSubmit', 'PendingCancel']]
            
            if not still_active:
                logger.info(f"✅ All orders canceled after {elapsed:.1f}s")
                break
            
            logger.info(f"   Waiting... {len(still_active)} orders still active ({elapsed:.1f}s)")
        
        # Final check
        remaining = ib.openTrades()
        if symbol:
            remaining = [t for t in remaining 
                        if t.contract.symbol == symbol 
                        and t.orderStatus.status in ['PreSubmitted', 'Submitted', 'PendingSubmit']]
        else:
            remaining = [t for t in remaining if t.orderStatus.status in ['PreSubmitted', 'Submitted', 'PendingSubmit']]
        
        if remaining:
            logger.warning(f"⚠️  {len(remaining)} orders still pending after {max_wait}s:")
            for trade in remaining:
                logger.warning(f"  - Order {trade.order.orderId}: {trade.contract.symbol} {trade.orderStatus.status}")
            
            if use_global_cancel:
                logger.warning("🚨 Using global cancel as last resort...")
                ib.reqGlobalCancel()
                await asyncio.sleep(3)
                
                # Check again
                final_remaining = [t for t in ib.openTrades() if t.orderStatus.status in ['PreSubmitted', 'Submitted', 'PendingSubmit']]
                if final_remaining:
                    logger.error(f"❌ {len(final_remaining)} orders STILL pending after global cancel")
                else:
                    logger.info("✅ Global cancel succeeded")
        else:
            logger.info("✅ All orders canceled successfully")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if ib.isConnected():
            ib.disconnect()
            logger.info("Disconnected from IBKR")
            logger.info("Disconnected from IBKR")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cancel open orders in IBKR")
    parser.add_argument("--symbol", "-s", type=str, help="Only cancel orders for this symbol (e.g., ES, MES)")
    parser.add_argument("--global-cancel", "-g", action="store_true", help="Use global cancel as last resort")
    args = parser.parse_args()
    
    asyncio.run(cancel_all_orders(symbol=args.symbol, use_global_cancel=args.global_cancel))
