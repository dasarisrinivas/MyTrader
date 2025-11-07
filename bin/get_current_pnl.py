#!/usr/bin/env python3
"""Quick script to get current P&L from IBKR."""
import asyncio
from ib_insync import IB, util
from mytrader.utils.settings_loader import load_settings

async def get_pnl():
    """Connect to IBKR and get current P&L."""
    settings = load_settings("config.yaml")
    
    ib = IB()
    try:
        print(f"Connecting to IBKR at {settings.data.ibkr_host}:{settings.data.ibkr_port}...")
        await ib.connectAsync(
            settings.data.ibkr_host,
            settings.data.ibkr_port,
            clientId=10,  # Different client ID to avoid conflicts
            timeout=30
        )
        print("✅ Connected!")
        
        # Get account summary
        account_summary = ib.accountSummary()
        print("\n" + "="*60)
        print("ACCOUNT SUMMARY")
        print("="*60)
        
        for item in account_summary:
            if 'PnL' in item.tag or 'Net' in item.tag or 'Unrealized' in item.tag or 'Realized' in item.tag:
                print(f"{item.tag:30} {item.value:>15} {item.currency}")
        
        # Get account values
        account_values = ib.accountValues()
        print("\n" + "="*60)
        print("DETAILED P&L")
        print("="*60)
        
        pnl_items = [v for v in account_values if 'PnL' in v.tag or 'pnl' in v.tag.lower()]
        for item in pnl_items:
            print(f"{item.tag:30} {item.value:>15} {item.currency}")
        
        # Get positions
        positions = ib.positions()
        print("\n" + "="*60)
        print("CURRENT POSITIONS")
        print("="*60)
        
        if not positions:
            print("No open positions")
        else:
            for pos in positions:
                print(f"\nSymbol: {pos.contract.symbol}")
                print(f"  Exchange: {pos.contract.exchange}")
                print(f"  Position: {pos.position} contracts")
                print(f"  Avg Cost: ${pos.avgCost:.2f}")
                print(f"  Market Price: ${pos.marketPrice:.2f}")
                print(f"  Market Value: ${pos.marketValue:.2f}")
                print(f"  Unrealized P&L: ${pos.unrealizedPNL:.2f}")
                print(f"  Realized P&L: ${pos.realizedPNL:.2f}")
        
        # Get today's executions
        executions = ib.executions()
        print("\n" + "="*60)
        print("TODAY'S EXECUTIONS")
        print("="*60)
        
        if not executions:
            print("No executions today")
        else:
            print(f"Total executions: {len(executions)}\n")
            for i, exec_detail in enumerate(executions[-10:], 1):  # Last 10
                e = exec_detail.execution
                print(f"{i}. {e.time} - {e.side} {e.shares} {exec_detail.contract.symbol} @ ${e.price:.2f}")
                if hasattr(exec_detail, 'commissionReport') and exec_detail.commissionReport:
                    cr = exec_detail.commissionReport
                    if hasattr(cr, 'realizedPNL') and cr.realizedPNL:
                        print(f"   Realized P&L: ${cr.realizedPNL:.2f}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\n✅ Disconnected")

if __name__ == "__main__":
    asyncio.run(get_pnl())
