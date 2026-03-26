#!/usr/bin/env python3
"""
Place a single MGC (Micro Gold) BUY bracket order on the PAPER account.

  Entry : Market order (BUY 1 MGC)
  TP    : Limit SELL at entry + tp_points
  SL    : Stop  SELL at entry - sl_points

Usage:
  python3 scripts/paper_gold_bracket_buy.py                     # defaults: TP=$5, SL=$3
  python3 scripts/paper_gold_bracket_buy.py --tp 8 --sl 4       # custom TP/SL in dollars
  python3 scripts/paper_gold_bracket_buy.py --tp-pts 5.0 --sl-pts 3.0  # in points
  python3 scripts/paper_gold_bracket_buy.py --dry-run            # show order details, don't submit

All orders: outsideRth=True, tif=GTC, transmitted as a bracket group.
Connects to IB Gateway PAPER port 4002 (configurable with --port).

MGC: $10/point, tick_size=0.10
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone

from ib_insync import IB, Future, LimitOrder, MarketOrder, StopOrder


# ── Defaults ─────────────────────────────────────────────────────────
PAPER_PORT = 4002
PAPER_HOST = "127.0.0.1"
CLIENT_ID = 99          # unique — avoids clashing with bot (clientId=3)
SYMBOL = "MGC"
EXCHANGE = "COMEX"
CURRENCY = "USD"
POINT_VALUE = 10.0      # MGC = $10 per point
TICK_SIZE = 0.10


def round_tick(price: float) -> float:
    """Round to nearest valid MGC tick (0.10)."""
    return round(round(price / TICK_SIZE) * TICK_SIZE, 2)


async def main(args: argparse.Namespace) -> None:
    ib = IB()

    # ── Connect ──────────────────────────────────────────────────────
    print(f"🔌 Connecting to IB Gateway at {args.host}:{args.port} (clientId={args.client_id}) …")
    try:
        await ib.connectAsync(args.host, args.port, clientId=args.client_id, timeout=15)
    except Exception as exc:
        print(f"❌ Connection failed: {exc}")
        sys.exit(1)
    print(f"✅ Connected — account: {ib.managedAccounts()}")

    # ── Qualify contract (pick front month) ──────────────────────────
    #   IB returns multiple months for MGC — we must pick the nearest
    #   expiry that is still tradeable (i.e. expiry > today).
    unqual = Future(symbol=SYMBOL, exchange=EXCHANGE, currency=CURRENCY)
    details_list = await ib.reqContractDetailsAsync(unqual)
    if not details_list:
        print(f"❌ No contract details for {SYMBOL} on {EXCHANGE}")
        ib.disconnect()
        sys.exit(1)

    # Sort by expiry, pick the nearest future expiry
    now_utc = datetime.now(timezone.utc)
    candidates = []
    for d in details_list:
        exp = d.contract.lastTradeDateOrContractMonth
        exp_dt = datetime.strptime(exp, "%Y%m%d").replace(tzinfo=timezone.utc)
        if exp_dt > now_utc:
            candidates.append((exp_dt, d.contract))
    candidates.sort(key=lambda x: x[0])

    if not candidates:
        print(f"❌ No active MGC contracts found")
        ib.disconnect()
        sys.exit(1)

    contract = candidates[0][1]
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        print(f"❌ Could not qualify front-month contract")
        ib.disconnect()
        sys.exit(1)
    contract = qualified[0]
    print(f"📋 Contract: {contract.localSymbol}  conId={contract.conId}  expiry={contract.lastTradeDateOrContractMonth}")

    # ── Get current price ────────────────────────────────────────────
    print("📈 Requesting market data …")
    ib.reqMktData(contract, genericTickList="", snapshot=False, regulatorySnapshot=False)
    await asyncio.sleep(2)  # let ticks arrive

    ticker = ib.ticker(contract)
    mid = None
    if ticker and ticker.last and ticker.last > 0:
        mid = ticker.last
    elif ticker and ticker.bid and ticker.ask and ticker.bid > 0:
        mid = round_tick((ticker.bid + ticker.ask) / 2)

    if mid is None:
        print("⚠️  No live price available — using delayed/close data")
        bars = await ib.reqHistoricalDataAsync(
            contract, endDateTime="", durationStr="60 S",
            barSizeSetting="1 min", whatToShow="TRADES", useRTH=False,
        )
        if bars:
            mid = bars[-1].close
        else:
            print("❌ Cannot determine price. Aborting.")
            ib.disconnect()
            sys.exit(1)

    print(f"💰 Reference price: {mid:.2f}")

    # ── Compute TP / SL ─────────────────────────────────────────────
    if args.tp_pts is not None:
        tp_pts = args.tp_pts
    else:
        tp_pts = args.tp / POINT_VALUE   # dollars → points

    if args.sl_pts is not None:
        sl_pts = args.sl_pts
    else:
        sl_pts = args.sl / POINT_VALUE   # dollars → points

    tp_price = round_tick(mid + tp_pts)
    sl_price = round_tick(mid - sl_pts)
    qty = args.qty

    print()
    print("┌──────────────────────────────────────────┐")
    print("│         📋 BRACKET ORDER PREVIEW         │")
    print("├──────────────────────────────────────────┤")
    print(f"│  Symbol     : {contract.localSymbol:<26}│")
    print(f"│  Action     : BUY {qty} @ MARKET{' ' * 18}│")
    print(f"│  Ref Price  : {mid:<26.2f}│")
    print(f"│  Take Profit: {tp_price:<11.2f} (+{tp_pts:.2f} pts = +${tp_pts * POINT_VALUE:.0f}){' ' * max(0, 5 - len(f'+${tp_pts * POINT_VALUE:.0f}'))}│")
    print(f"│  Stop Loss  : {sl_price:<11.2f} (-{sl_pts:.2f} pts = -${sl_pts * POINT_VALUE:.0f}){' ' * max(0, 5 - len(f'-${sl_pts * POINT_VALUE:.0f}'))}│")
    print(f"│  R:R        : 1:{tp_pts/sl_pts:.2f}{' ' * 24}│")
    print("└──────────────────────────────────────────┘")
    print()

    if args.dry_run:
        print("🏁 --dry-run: not submitting. Exiting.")
        ib.disconnect()
        return

    # ── Build bracket orders ─────────────────────────────────────────
    entry_order = MarketOrder("BUY", qty)
    entry_order.outsideRth = True
    entry_order.tif = "GTC"
    entry_order.transmit = False            # don't transmit until children are ready

    tp_order = LimitOrder("SELL", qty, tp_price)
    tp_order.outsideRth = True
    tp_order.tif = "GTC"
    tp_order.transmit = False

    sl_order = StopOrder("SELL", qty, sl_price)
    sl_order.outsideRth = True
    sl_order.tif = "GTC"
    sl_order.transmit = True                # transmit entire bracket on last child

    # ── Submit ───────────────────────────────────────────────────────
    print("🚀 Submitting bracket order …")
    entry_trade = ib.placeOrder(contract, entry_order)

    # Link children to parent
    tp_order.parentId = entry_trade.order.orderId
    sl_order.parentId = entry_trade.order.orderId

    tp_trade = ib.placeOrder(contract, tp_order)
    sl_trade = ib.placeOrder(contract, sl_order)

    print(f"   Entry orderId : {entry_trade.order.orderId}")
    print(f"   TP    orderId : {tp_trade.order.orderId}")
    print(f"   SL    orderId : {sl_trade.order.orderId}")

    # ── Wait for fill ────────────────────────────────────────────────
    print("\n⏳ Waiting for entry fill (up to 30s) …")
    for _ in range(60):
        await asyncio.sleep(0.5)
        if entry_trade.orderStatus.status == "Filled":
            fill_price = entry_trade.orderStatus.avgFillPrice
            print(f"✅ FILLED at {fill_price:.2f}")
            print(f"   TP target : {tp_price:.2f}  (+${(tp_price - fill_price) * POINT_VALUE:.0f})")
            print(f"   SL target : {sl_price:.2f}  (-${(fill_price - sl_price) * POINT_VALUE:.0f})")
            break
        if entry_trade.orderStatus.status in ("Cancelled", "ApiCancelled", "Inactive"):
            print(f"❌ Order {entry_trade.orderStatus.status}: {entry_trade.log}")
            break
    else:
        print(f"⚠️  Not yet filled — status: {entry_trade.orderStatus.status}")
        print("   The bracket is live; TP/SL will work even after this script exits.")

    print("\n📊 Open orders:")
    for trade in ib.openTrades():
        o = trade.order
        s = trade.orderStatus
        print(f"   {o.orderId:>6}  {o.action:<4} {o.totalQuantity}x  "
              f"type={o.orderType:<6}  lmt={o.lmtPrice or '-':<10}  aux={o.auxPrice or '-':<10}  "
              f"status={s.status}")

    print("\n🏁 Done. Bracket is live on IB — TP/SL will manage the exit.")
    ib.disconnect()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Place a paper MGC BUY bracket order (entry + TP + SL)",
    )
    p.add_argument("--host", default=PAPER_HOST, help=f"IB Gateway host (default {PAPER_HOST})")
    p.add_argument("--port", type=int, default=PAPER_PORT, help=f"IB Gateway port (default {PAPER_PORT})")
    p.add_argument("--client-id", type=int, default=CLIENT_ID, help=f"IB client ID (default {CLIENT_ID})")
    p.add_argument("--qty", type=int, default=1, help="Number of contracts (default 1)")
    p.add_argument("--tp", type=float, default=50.0, help="Take profit in dollars (default $50 = 5.0 pts)")
    p.add_argument("--sl", type=float, default=30.0, help="Stop loss in dollars (default $30 = 3.0 pts)")
    p.add_argument("--tp-pts", type=float, default=None, help="Take profit in points (overrides --tp)")
    p.add_argument("--sl-pts", type=float, default=None, help="Stop loss in points (overrides --sl)")
    p.add_argument("--dry-run", action="store_true", help="Preview order without submitting")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
