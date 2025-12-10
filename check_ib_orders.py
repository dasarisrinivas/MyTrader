#!/usr/bin/env python3
"""Quick script to check actual IB order state."""

from ib_insync import IB, Contract, Future
import asyncio

async def main():
    ib = IB()
    
    print("Connecting to IB Gateway...")
    await ib.connectAsync('127.0.0.1', 4002, clientId=999)  # Use different clientId
    print(f"Connected: {ib.isConnected()}")
    
    # Get all open trades
    print("\n=== All Open Trades ===")
    open_trades = ib.openTrades()
    print(f"Total open trades: {len(open_trades)}")
    
    for trade in open_trades:
        print(f"  Order {trade.order.orderId}: {trade.contract.symbol} {trade.order.action} "
              f"{trade.order.totalQuantity} @ {getattr(trade.order, 'lmtPrice', 'MKT')} "
              f"- Status: {trade.orderStatus.status}")
    
    # Get ES-specific trades
    print("\n=== ES Orders ===")
    es_trades = [t for t in open_trades if t.contract.symbol == 'ES']
    print(f"ES open trades: {len(es_trades)}")
    
    for trade in es_trades:
        status = trade.orderStatus
        print(f"  Order {trade.order.orderId}:")
        print(f"    Contract: {trade.contract.localSymbol}")
        print(f"    Action: {trade.order.action} {trade.order.totalQuantity}")
        print(f"    Type: {trade.order.orderType}")
        print(f"    Status: {status.status}")
        print(f"    Filled: {status.filled}/{trade.order.totalQuantity}")
    
    # Get all orders
    print("\n=== All Orders ===")
    all_orders = ib.orders()
    print(f"Total orders: {len(all_orders)}")
    
    for order in all_orders:
        print(f"  Order {order.orderId}: {order.action} {order.totalQuantity} - {order.orderType}")
    
    # Get positions
    print("\n=== Positions ===")
    positions = ib.positions()
    for pos in positions:
        if pos.contract.symbol == 'ES':
            print(f"  ES Position: {pos.position} contracts")
    
    ib.disconnect()
    print("\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
