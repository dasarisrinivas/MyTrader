#!/usr/bin/env python3
"""Send a Telegram summary of the SPY options bot's trading day.

Reads today's EXEC activity from logs/spy_options.log (the authoritative
source for orders / fills / closes / cancels — analytics fill columns can be
incomplete) plus any recorded P&L, formats a summary, and sends it via the
same Telegram bot the SPY options bot uses. Called by stop_trading_day.sh
before the bot is stopped. Safe to run standalone: python3 scripts/daily_summary.py

Optional: --message "custom text"  sends an arbitrary alert instead of the summary.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import urllib.parse
import urllib.request
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs", "spy_options.log")
DB = os.path.join(ROOT, "data", "spy_options_signals.db")
CFG = os.path.join(ROOT, "config.yaml")


def load_telegram():
    try:
        import yaml
    except ImportError:
        return None, None
    try:
        with open(CFG, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return None, None
    for section in ((data.get("spy_options") or {}).get("telegram"), data.get("telegram")):
        if section and section.get("enabled") and section.get("bot_token"):
            return section["bot_token"], str(section["chat_id"])
    return None, None


def send(token, chat, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = urllib.parse.urlencode(
        {"chat_id": chat, "text": text, "parse_mode": "HTML"}
    ).encode()
    req = urllib.request.Request(url, data=body)
    with urllib.request.urlopen(req, timeout=20) as resp:
        resp.read()


_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2})")


def parse_day(day: str):
    orders, fills, closes, cancels, signals = [], [], [], [], 0
    if not os.path.exists(LOG):
        return orders, fills, closes, cancels, signals
    with open(LOG, errors="ignore") as fh:
        for line in fh:
            if day not in line:
                continue
            if "EXEC ORDER: BUY" in line:
                orders.append(line.strip())
            elif "EXEC FILL" in line:
                fills.append(line.strip())
            elif "EXEC CLOSE" in line:
                closes.append(line.strip())
            elif "EXEC CANCEL" in line:
                cancels.append(line.strip())
            elif "Sending signal:" in line:
                signals += 1
    return orders, fills, closes, cancels, signals


def _hhmm(line: str) -> str:
    m = _TS.search(line)
    return m.group(1)[11:] if m else "--:--"


def build_summary(day: str) -> str:
    orders, fills, closes, cancels, signals = parse_day(day)
    L = [f"📊 <b>SPY Options — Daily Summary</b>", f"🗓 {day}", ""]
    L.append(f"🔔 Signals dispatched: <b>{signals}</b>")
    L.append(f"🟢 Orders placed: <b>{len(orders)}</b>   ✅ Fills: <b>{len(fills)}</b>"
             f"   🚫 Unfilled/cancel: <b>{len(cancels)}</b>")

    # Closes with P&L (new close logs carry "P&L $X")
    total = 0.0
    wins = losses = scratch = counted = 0
    if closes:
        L.append("")
        L.append("<b>Closed trades:</b>")
        for c in closes:
            sm = re.search(r"SPY\s+(\S+)", c)
            sym = sm.group(1) if sm else "?"
            m = re.search(r"P&L \$([+-]?\d+(?:\.\d+)?)", c)
            if m:
                pnl = float(m.group(1))
                total += pnl
                counted += 1
                wins += pnl > 0
                losses += pnl < 0
                scratch += pnl == 0
                emoji = "🟩" if pnl > 0 else "🟥" if pnl < 0 else "⬜"
                reason = c.split("—", 1)[-1].strip()[:48] if "—" in c else ""
                L.append(f"  {emoji} {_hhmm(c)} {sym}  <b>${pnl:+.0f}</b>  {reason}")
            else:
                # a close line without P&L (bracket child / unconfirmed)
                reason = c.split("—", 1)[-1].strip()[:48] if "—" in c else ""
                L.append(f"  ▫️ {_hhmm(c)} {sym}  (P&L in bracket/unconfirmed) {reason}")

    if counted:
        wr = wins / counted * 100
        L += ["", f"<b>Realized P&L: ${total:+.0f}</b>  "
                   f"({wins}W / {losses}L / {scratch} scratch, win {wr:.0f}%)"]
    elif orders:
        L += ["", "No closes with recorded P&L (positions may rest at IB brackets)."]
    else:
        L += ["", "No trades today."]

    L += ["", f"🕐 {datetime.now().strftime('%H:%M %Z')}  ·  bot stopping for the day"]
    return "\n".join(L)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--message":
        msg = sys.argv[2]
    else:
        msg = build_summary(datetime.now().strftime("%Y-%m-%d"))
    print(msg)
    token, chat = load_telegram()
    if token:
        try:
            send(token, chat, msg)
            print("[daily_summary] sent to Telegram")
        except Exception as exc:
            print(f"[daily_summary] Telegram send failed: {exc}")
    else:
        print("[daily_summary] no Telegram config — printed only")


if __name__ == "__main__":
    main()
