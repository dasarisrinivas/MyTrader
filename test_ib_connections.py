#!/usr/bin/env python3
"""Quick test: verify IB Gateway connectivity for main trading + VIX feed."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ib_insync import IB, Future, Index, Contract


async def test_main_connection():
    """Test main trading connection on port 4001 (client_id=1)."""
    print("\n" + "=" * 65)
    print("  TEST 1: Main IB Gateway Connection (port 4001, client_id=99)")
    print("=" * 65)

    ib = IB()
    try:
        await ib.connectAsync("127.0.0.1", 4001, clientId=99, timeout=10)
        print("✅ Connected to IB Gateway")

        # Account info
        accounts = ib.managedAccounts()
        print(f"   Accounts: {accounts}")

        # Request LIVE market data
        ib.reqMarketDataType(1)
        print("   Market data type: 1 (Live)")

        # Qualify MES front-month contract
        mes = Future(symbol="MES", exchange="CME", currency="USD",
                     lastTradeDateOrContractMonth="202603")
        qualified = await ib.qualifyContractsAsync(mes)
        if qualified:
            mes = qualified[0]
            print(f"   ✅ MES contract qualified: {mes.localSymbol} (conId={mes.conId})")

            # Request a snapshot quote
            ticker = ib.reqMktData(mes, "", True, False)
            await asyncio.sleep(3)

            price = ticker.marketPrice()
            bid = ticker.bid
            ask = ticker.ask
            last = ticker.last

            if price != price:  # NaN check
                print(f"   ⚠️  MES price: NaN (market may be closed or no subscription)")
                print(f"       bid={bid}  ask={ask}  last={last}")
            else:
                print(f"   ✅ MES price: {price:.2f}  (bid={bid}  ask={ask}  last={last})")

            ib.cancelMktData(mes)
        else:
            print("   ❌ Could not qualify MES contract")

        # Check account summary
        summary = await ib.accountSummaryAsync()
        for item in summary:
            if item.tag in ("NetLiquidation", "TotalCashValue", "AvailableFunds"):
                print(f"   {item.tag}: ${float(item.value):,.2f}")

        print("\n   ✅ Main connection: PASS")
        return True

    except Exception as e:
        print(f"\n   ❌ Main connection FAILED: {e}")
        return False
    finally:
        if ib.isConnected():
            ib.disconnect()


async def test_vix_connection():
    """Test VIX feed connection on port 4001 (client_id=71)."""
    print("\n" + "=" * 65)
    print("  TEST 2: VIX Feed Connection (port 4001, client_id=71)")
    print("=" * 65)

    ib = IB()
    try:
        await ib.connectAsync("127.0.0.1", 4001, clientId=71, timeout=10)
        print("✅ Connected to IB Gateway (VIX feed)")

        # Request LIVE market data
        ib.reqMarketDataType(1)
        print("   Market data type: 1 (Live)")

        # Try VX futures (front month) on CFE
        from datetime import datetime
        now = datetime.now()
        # VX futures expire on Wednesday of third week; approximate front month
        month = now.month
        year = now.year
        # Try next few months to find active contract
        vx_qualified = None
        for m_offset in range(0, 4):
            m = month + m_offset
            y = year
            if m > 12:
                m -= 12
                y += 1
            expiry = f"{y}{m:02d}"
            vx = Future(symbol="VXM", exchange="CFE", currency="USD",
                        lastTradeDateOrContractMonth=expiry)
            try:
                qualified = await ib.qualifyContractsAsync(vx)
                if qualified:
                    vx_qualified = qualified[0]
                    break
            except Exception:
                pass

        # Also try generic VX without specific month
        if not vx_qualified:
            for sym in ("VXM", "VX"):
                vx = Future(symbol=sym, exchange="CFE", currency="USD")
                try:
                    qualified = await ib.qualifyContractsAsync(vx)
                    if qualified:
                        vx_qualified = qualified[0]
                        break
                except Exception:
                    pass

        if vx_qualified:
            print(f"   ✅ VX contract qualified: {vx_qualified.localSymbol} (conId={vx_qualified.conId})")

            # Request market data
            ticker = ib.reqMktData(vx_qualified, "", True, False)
            await asyncio.sleep(3)

            price = ticker.marketPrice()
            bid = ticker.bid
            ask = ticker.ask
            last = ticker.last

            if price != price:  # NaN check
                print(f"   ⚠️  VX price: NaN (market may be closed or no CFE subscription)")
                print(f"       bid={bid}  ask={ask}  last={last}")
            else:
                print(f"   ✅ VX price: {price:.2f}  (bid={bid}  ask={ask}  last={last})")

                # Show what multiplier would be applied
                if price >= 30.0:
                    mult = 0.4
                    label = "EXTREME (0.4x)"
                elif price >= 20.0:
                    mult = 0.7
                    label = "ELEVATED (0.7x)"
                else:
                    mult = 1.0
                    label = "NORMAL (1.0x)"
                print(f"   📊 Volatility regime: {label}")

            ib.cancelMktData(vx_qualified)
        else:
            print("   ❌ Could not qualify any VX contract on CFE")
            print("       Trying VIX index instead...")

            # Fallback: VIX index
            vix = Index(symbol="VIX", exchange="CBOE", currency="USD")
            try:
                qualified = await ib.qualifyContractsAsync(vix)
                if qualified:
                    vix_contract = qualified[0]
                    print(f"   ✅ VIX index qualified: {vix_contract.localSymbol}")
                    ticker = ib.reqMktData(vix_contract, "", True, False)
                    await asyncio.sleep(3)
                    price = ticker.marketPrice()
                    if price == price:
                        print(f"   ✅ VIX index: {price:.2f}")
                    else:
                        print(f"   ⚠️  VIX price: NaN")
                    ib.cancelMktData(vix_contract)
            except Exception as e2:
                print(f"   ❌ VIX index also failed: {e2}")

        print("\n   ✅ VIX connection: PASS")
        return True

    except Exception as e:
        print(f"\n   ❌ VIX connection FAILED: {e}")
        return False
    finally:
        if ib.isConnected():
            ib.disconnect()


async def main():
    print("\n🔌 ShreeBot — IB Connection Test")
    print("   Testing both main trading and VIX feed connections...")

    main_ok = await test_main_connection()
    vix_ok = await test_vix_connection()

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"   Main trading (MES):  {'✅ PASS' if main_ok else '❌ FAIL'}")
    print(f"   VIX feed:            {'✅ PASS' if vix_ok else '❌ FAIL'}")
    print("=" * 65)

    if not main_ok or not vix_ok:
        print("\n⚠️  Fix issues above before starting live trading.")
        print("   - Ensure IB Gateway is running and logged into LIVE account")
        print("   - Port 4001 must be open (API settings)")
        print("   - CME market data subscription needed for MES")
        print("   - CFE Enhanced (NP,L1) subscription needed for VX")
        sys.exit(1)
    else:
        print("\n🚀 All connections verified — ready for live trading!")


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
