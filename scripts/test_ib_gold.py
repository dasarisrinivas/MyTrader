#!/usr/bin/env python3
"""IB Gateway / TWS diagnostic script for Gold futures (MGC / GC).

Tests every layer the gold bot depends on:
  1. TCP connectivity to IB Gateway / TWS
  2. IB API login + managed accounts
  3. COMEX futures contract qualification (front-month selection)
  4. Market data subscription (snapshot quote)
  5. Historical bar request (1-min bars, last 30 min)
  6. Real-time (5-sec) bar subscription
  7. Account summary — available funds & margin
  8. Order permission check (whatIf order)

Usage:
    python3 scripts/test_ib_gold.py                     # paper (port 4002, cid 99)
    python3 scripts/test_ib_gold.py --port 4001          # live gateway
    python3 scripts/test_ib_gold.py --symbol GC           # full-size gold
    python3 scripts/test_ib_gold.py --timeout 20          # longer timeout per test
"""
from __future__ import annotations

import argparse
import asyncio
import socket
import sys
import time
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Attempt ib_insync import early so we can give a clear error
# ---------------------------------------------------------------------------
try:
    from ib_insync import IB, Future, MarketOrder, util
except ImportError:
    print(
        "❌ FAIL — ib_insync is not installed.\n"
        "   pip install ib_insync\n"
    )
    sys.exit(1)


# ── Helpers ────────────────────────────────────────────────────────────────────

_PASS = "\033[92m✅ PASS\033[0m"
_FAIL = "\033[91m❌ FAIL\033[0m"
_WARN = "\033[93m⚠️  WARN\033[0m"
_INFO = "\033[94mℹ️  INFO\033[0m"

_results: list[tuple[str, str, str]] = []  # (test_name, status, detail)


def _record(name: str, passed: bool, detail: str = "", warn: bool = False) -> bool:
    status = _PASS if passed else (_WARN if warn else _FAIL)
    _results.append((name, status, detail))
    tag = status
    print(f"  {tag}  {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")
    return passed


# ── Individual tests ──────────────────────────────────────────────────────────

def test_tcp_connect(host: str, port: int, timeout: float) -> bool:
    """1. Raw TCP socket connect to IB Gateway / TWS port."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return _record("TCP connect", True, f"{host}:{port} reachable")
    except OSError as exc:
        return _record(
            "TCP connect",
            False,
            f"{host}:{port} unreachable — {exc}\n"
            "→ Is IB Gateway / TWS running?\n"
            "→ Paper: port 4002, Live: port 4001\n"
            "→ Check API Settings → Socket port & 'Allow connections from localhost'",
        )


async def test_ib_login(ib: IB, host: str, port: int, client_id: int, timeout: float) -> bool:
    """2. IB API login and managed accounts."""
    try:
        await asyncio.wait_for(
            ib.connectAsync(host, port, clientId=client_id, readonly=True),
            timeout=timeout,
        )
        accounts = ib.managedAccounts()
        if not accounts:
            return _record("IB login", False, "Connected but no managed accounts returned")
        return _record(
            "IB login",
            True,
            f"Connected — accounts: {', '.join(accounts)}",
        )
    except asyncio.TimeoutError:
        return _record(
            "IB login",
            False,
            f"Timed out after {timeout}s.\n"
            "→ Check 'Enable ActiveX and Socket Clients' in API settings.\n"
            "→ Make sure no other process is using clientId={client_id}.",
        )
    except Exception as exc:
        return _record("IB login", False, str(exc))


async def test_contract_qualify(ib: IB, symbol: str, exchange: str, timeout: float) -> Optional[Future]:
    """3. Contract qualification — same method as the gold bot."""
    contract = Future(symbol=symbol, exchange=exchange, currency="USD")
    try:
        details = await asyncio.wait_for(
            ib.reqContractDetailsAsync(contract),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        _record(
            "Contract qualify",
            False,
            f"reqContractDetails timed out for {symbol}/{exchange}.\n"
            "→ COMEX market data subscription may be missing.\n"
            "→ TWS: Account → Settings → Market Data Subscriptions → add 'COMEX Different Fees'.",
        )
        return None
    except Exception as exc:
        _record("Contract qualify", False, str(exc))
        return None

    if not details:
        _record(
            "Contract qualify",
            False,
            f"IB returned 0 contract details for {symbol}/{exchange}/USD.\n\n"
            "Most likely causes:\n"
            "  1. No COMEX market data subscription on this account.\n"
            "     → TWS: Account → Settings → Market Data Subscriptions\n"
            "     → Add: 'COMEX Different Fees' (covers MGC + GC)\n"
            "  2. Paper account inherits subscriptions from linked live account.\n"
            "     If the live account doesn't have COMEX, paper won't either.\n"
            "  3. Symbol/exchange mismatch (e.g. NYMEX vs COMEX).\n"
            "  4. Weekend / holiday — exchange closed, no data available.\n",
        )
        return None

    # Filter expired, pick front-month
    now_utc = datetime.now(timezone.utc)

    def _expiry(d) -> Optional[datetime]:
        raw = getattr(d.contract, "lastTradeDateOrContractMonth", "") or ""
        try:
            if len(raw) >= 8:
                return datetime.strptime(raw[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
        except (ValueError, OverflowError):
            pass
        return None

    valid = [d for d in details if (_expiry(d) or now_utc) >= now_utc]
    if not valid:
        valid = details

    valid.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
    front = valid[0].contract

    _record(
        "Contract qualify",
        True,
        f"Front-month: conId={front.conId} localSymbol={front.localSymbol} "
        f"expiry={front.lastTradeDateOrContractMonth}\n"
        f"Total contract months found: {len(details)} (valid/active: {len(valid)})",
    )
    return front


async def test_market_data_snapshot(ib: IB, contract: Future, timeout: float) -> bool:
    """4. Snapshot quote — tests market data subscription."""
    try:
        ticker = ib.reqMktData(contract, genericTickList="", snapshot=True, regulatorySnapshot=False)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(0.3)
            if ticker.last is not None or ticker.close is not None or ticker.bid is not None:
                break

        last = ticker.last if ticker.last is not None else ticker.close
        bid = ticker.bid
        ask = ticker.ask

        if last is None and bid is None:
            _record(
                "Market data snapshot",
                False,
                "Snapshot returned no prices (last=None, bid=None, ask=None).\n"
                "→ Market data subscription for COMEX may be missing or delayed.\n"
                "→ If weekend/holiday, prices may not be available — try during RTH.\n"
                "→ TWS: Account → Settings → Market Data Subscriptions → COMEX.",
            )
            ib.cancelMktData(contract)
            return False

        detail = f"last={last}  bid={bid}  ask={ask}  volume={ticker.volume}"
        ib.cancelMktData(contract)
        return _record("Market data snapshot", True, detail)

    except Exception as exc:
        _record("Market data snapshot", False, str(exc))
        return False


async def test_historical_bars(ib: IB, contract: Future, timeout: float) -> bool:
    """5. Historical 1-min bars (last 30 min) — tests historical data permission."""
    try:
        bars = await asyncio.wait_for(
            ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr="1800 S",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=2,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        _record(
            "Historical bars",
            False,
            "reqHistoricalData timed out.\n"
            "→ Historical data permission may not be enabled for COMEX.\n"
            "→ Some paper accounts require a separate historical data subscription.",
        )
        return False
    except Exception as exc:
        _record("Historical bars", False, str(exc))
        return False

    if not bars:
        _record(
            "Historical bars",
            False,
            "0 bars returned for last 30 min.\n"
            "→ If market is closed (weekend/holiday), this is expected.\n"
            "→ Otherwise, check COMEX historical data permissions.",
            warn=True,
        )
        return False

    first_ts = bars[0].date
    last_ts = bars[-1].date
    last_close = bars[-1].close
    _record(
        "Historical bars",
        True,
        f"{len(bars)} bars returned  |  range: {first_ts} → {last_ts}  |  last close={last_close}",
    )
    return True


async def test_realtime_bars(ib: IB, contract: Future, timeout: float) -> bool:
    """6. Real-time 5-sec bars — tests streaming data capability."""
    received: list = []

    def _on_bar(bars, has_new_bar):
        if has_new_bar and bars:
            received.append(bars[-1])

    try:
        rt_bars = ib.reqRealTimeBars(contract, barSize=5, whatToShow="TRADES", useRTH=False)
        rt_bars.updateEvent += _on_bar

        wait_secs = min(timeout, 15.0)  # Wait up to 15s for a bar
        deadline = time.monotonic() + wait_secs
        while time.monotonic() < deadline and not received:
            await asyncio.sleep(0.5)

        ib.cancelRealTimeBars(rt_bars)

        if not received:
            _record(
                "Realtime bars",
                False,
                f"No 5-sec bars received in {wait_secs:.0f}s.\n"
                "→ Real-time data requires an active market data subscription.\n"
                "→ If market is closed, no bars will arrive — try during trading hours.\n"
                "→ Check TWS log for 'No market data permissions' errors.",
                warn=True,
            )
            return False

        bar = received[0]
        _record(
            "Realtime bars",
            True,
            f"Received {len(received)} bar(s)  |  first: time={bar.time} close={bar.close} vol={bar.volume}",
        )
        return True
    except Exception as exc:
        _record("Realtime bars", False, str(exc))
        return False


async def test_account_summary(ib: IB, timeout: float) -> bool:
    """7. Account summary — available funds and margin."""
    try:
        summary = await asyncio.wait_for(
            ib.accountSummaryAsync(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        _record("Account summary", False, "accountSummary timed out")
        return False
    except Exception as exc:
        _record("Account summary", False, str(exc))
        return False

    if not summary:
        _record("Account summary", False, "Empty summary returned")
        return False

    # Extract key values
    values = {}
    for item in summary:
        if item.tag in (
            "AvailableFunds",
            "BuyingPower",
            "NetLiquidation",
            "InitMarginReq",
            "MaintMarginReq",
            "TotalCashValue",
        ):
            values[item.tag] = item.value

    detail_lines = [f"{k}: {v}" for k, v in sorted(values.items())]
    avail = float(values.get("AvailableFunds", 0))

    if avail < 500:
        _record(
            "Account summary",
            False,
            "AvailableFunds < $500 — insufficient for gold futures.\n"
            + "\n".join(detail_lines),
            warn=True,
        )
        return False

    _record("Account summary", True, "\n".join(detail_lines))
    return True


async def test_order_permission(ib: IB, contract: Future, timeout: float) -> bool:
    """8. WhatIf order — tests if the account can place gold futures orders."""
    try:
        # Build a tiny whatIf order — never actually submitted
        order = MarketOrder("BUY", 1)
        order.whatIf = True

        trade = ib.placeOrder(contract, order)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(0.3)
            if trade.orderStatus.status in ("PreSubmitted", "Submitted", "Filled", "Inactive"):
                break
            # whatIf orders return immediately with commission/margin info
            if trade.log and any("commission" in str(e).lower() or "margin" in str(e).lower() for e in trade.log):
                break
            # Check if whatIf result is populated
            if hasattr(trade, "orderState") and trade.orderStatus.status:
                break

        # Cancel just in case
        try:
            ib.cancelOrder(order)
        except Exception:
            pass

        init_margin = getattr(trade.order, "whatIfInitMargin", None) or "N/A"
        maint_margin = getattr(trade.order, "whatIfMaintMargin", None) or "N/A"
        commission = getattr(trade.order, "whatIfCommission", None) or "N/A"

        status = trade.orderStatus.status or "unknown"

        if status == "Inactive":
            _record(
                "Order permission (whatIf)",
                False,
                f"Order status: Inactive — account may lack futures trading permission.\n"
                "→ Check TWS: Account → Permissions → Futures must be enabled.\n"
                "→ For COMEX specifically, Gold futures may need separate permission.",
            )
            return False

        detail = (
            f"Status: {status}\n"
            f"Init margin: {init_margin}\n"
            f"Maint margin: {maint_margin}\n"
            f"Commission: {commission}"
        )
        _record("Order permission (whatIf)", True, detail)
        return True

    except Exception as exc:
        msg = str(exc)
        if "No trading permissions" in msg or "not allowed" in msg.lower():
            _record(
                "Order permission (whatIf)",
                False,
                f"{msg}\n"
                "→ Enable futures trading: TWS → Account → Settings → Trading Permissions → Futures.\n"
                "→ COMEX Gold may need explicit approval.",
            )
        else:
            _record("Order permission (whatIf)", False, msg)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_all(args: argparse.Namespace) -> int:
    """Run all diagnostic tests in sequence, return exit code."""
    host = args.host
    port = args.port
    cid = args.client_id
    symbol = args.symbol
    exchange = args.exchange
    timeout = args.timeout

    print()
    print("━" * 64)
    print(f"  IB Gold Futures Diagnostic — {symbol}/{exchange}")
    print(f"  Gateway: {host}:{port}  clientId={cid}  timeout={timeout}s")
    print("━" * 64)
    print()

    # 1. TCP
    if not test_tcp_connect(host, port, timeout):
        _print_summary()
        return 1

    # 2. IB login
    ib = IB()
    if not await test_ib_login(ib, host, port, cid, timeout):
        _print_summary()
        return 1

    # 3. Contract
    contract = await test_contract_qualify(ib, symbol, exchange, timeout)
    if contract is None:
        ib.disconnect()
        _print_summary()
        return 1

    # 4–8 can run even if some fail
    await test_market_data_snapshot(ib, contract, timeout)
    await test_historical_bars(ib, contract, timeout)
    await test_realtime_bars(ib, contract, timeout)
    await test_account_summary(ib, timeout)
    await test_order_permission(ib, contract, timeout)

    ib.disconnect()
    _print_summary()

    failures = sum(1 for _, s, _ in _results if _FAIL in s)
    return 1 if failures > 0 else 0


def _print_summary() -> None:
    """Print a summary table at the end."""
    print()
    print("━" * 64)
    print("  SUMMARY")
    print("━" * 64)
    passes = sum(1 for _, s, _ in _results if _PASS in s)
    warns = sum(1 for _, s, _ in _results if _WARN in s)
    fails = sum(1 for _, s, _ in _results if _FAIL in s)
    total = len(_results)

    for name, status, _ in _results:
        print(f"  {status}  {name}")

    print()
    print(f"  Total: {total}  |  Pass: {passes}  |  Warn: {warns}  |  Fail: {fails}")

    if fails == 0 and warns == 0:
        print("\n  🎉 All checks passed — IB Gateway is fully configured for gold futures.\n")
    elif fails == 0:
        print(
            "\n  ⚠️  Some warnings (likely market closed). Core connectivity is OK.\n"
            "  Re-run during COMEX trading hours (Sun 6 PM – Fri 5 PM ET) for full validation.\n"
        )
    else:
        print(
            "\n  ❌ One or more critical checks failed.\n"
            "  Fix the issues above and re-run this script.\n"
        )
    print("━" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IB Gateway diagnostic for Gold futures (MGC/GC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/test_ib_gold.py                  # paper, MGC\n"
            "  python3 scripts/test_ib_gold.py --port 4001      # live gateway\n"
            "  python3 scripts/test_ib_gold.py --symbol GC      # full-size gold\n"
            "  python3 scripts/test_ib_gold.py --timeout 20     # slower network\n"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="IB Gateway host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=4002, help="IB Gateway port (default: 4002 paper)")
    parser.add_argument("--client-id", type=int, default=99, help="IB client ID (default: 99 — avoids conflicts)")
    parser.add_argument("--symbol", default="MGC", help="Gold symbol: MGC (micro) or GC (full)")
    parser.add_argument("--exchange", default="COMEX", help="Exchange (default: COMEX)")
    parser.add_argument("--timeout", type=float, default=15.0, help="Timeout per test in seconds (default: 15)")
    args = parser.parse_args()

    exit_code = asyncio.run(run_all(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
