#!/usr/bin/env python3
"""Flatten any open SPY OPTION positions at market, and cancel resting SPY
option orders. Run AFTER the bot is stopped (so nothing races), by the 14:00 CT
stop job — ensures no continuation position carries overnight (its bracket
stays at IB when the bot stops, but a multi-DTE trade that doesn't hit TP/SL
by the close would otherwise hold overnight).

STRICTLY scoped to secType == 'OPT' and symbol == 'SPY' — never touches MES,
Gold, or any other position in the account. Reports what it did to Telegram.
"""
from __future__ import annotations

import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _exec_conn():
    """(host, port, account) from config.yaml spy_options.execution."""
    host, port, account = "127.0.0.1", 4001, ""
    try:
        import yaml
        with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        ex = ((data.get("spy_options") or {}).get("execution")) or {}
        host = ex.get("ibkr_host", host)
        port = int(ex.get("ibkr_port", port))
        account = ex.get("account", account) or ""
    except Exception:
        pass
    return host, port, account


def _notify(text: str):
    try:
        import daily_summary as ds
        tok, chat = ds.load_telegram()
        if tok:
            ds.send(tok, chat, text)
    except Exception as exc:
        print(f"[flatten] notify failed: {exc}")


async def flatten() -> str:
    from ib_insync import IB, MarketOrder

    host, port, account = _exec_conn()
    ib = IB()
    try:
        await ib.connectAsync(host, port, clientId=8, timeout=20)
    except Exception as exc:
        msg = f"⚠️ <b>Flatten FAILED</b> — could not connect to IB {host}:{port} ({exc})"
        print(msg)
        _notify(msg)
        return msg

    lines = []
    try:
        # 1) Cancel resting SPY option orders (the bot's brackets after shutdown).
        cancelled = 0
        for tr in ib.openTrades():
            c = tr.contract
            if getattr(c, "secType", "") == "OPT" and getattr(c, "symbol", "") == "SPY":
                if tr.orderStatus.status not in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
                    try:
                        ib.cancelOrder(tr.order)
                        cancelled += 1
                    except Exception:
                        pass
        if cancelled:
            await asyncio.sleep(2)

        # 2) Market-close any open SPY option position.
        closed = []
        for pos in ib.positions():
            c = pos.contract
            if getattr(c, "secType", "") == "OPT" and getattr(c, "symbol", "") == "SPY" and pos.position != 0:
                qty = abs(int(pos.position))
                action = "SELL" if pos.position > 0 else "BUY"
                order = MarketOrder(action, qty)
                if account:
                    order.account = account
                ib.placeOrder(c, order)
                closed.append(f"{action} {qty}× {c.localSymbol}")

        if closed:
            await asyncio.sleep(6)  # let market fills complete
            lines.append(f"🔻 Flattened {len(closed)} SPY option position(s):")
            lines += [f"  • {x}" for x in closed]
        else:
            lines.append("✅ No open SPY option positions to flatten.")
        if cancelled:
            lines.append(f"🧹 Cancelled {cancelled} resting SPY option order(s).")
    finally:
        ib.disconnect()

    msg = "<b>SPY Options — EOD flatten</b>\n" + "\n".join(lines)
    print(msg)
    # Only ping Telegram if something was actually done.
    if closed or cancelled:
        _notify(msg)
    return msg


if __name__ == "__main__":
    asyncio.run(flatten())
