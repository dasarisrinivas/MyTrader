#!/usr/bin/env python3
"""
Daily trade journal extractor — pulls key stats from live_trading.log
and stores them in data/trade_journal.db (SQLite).

Usage:
    python3 scripts/daily_journal.py                    # ingest today
    python3 scripts/daily_journal.py 2026-03-05         # ingest specific date
    python3 scripts/daily_journal.py 2026-03-06 --week  # ingest last 7 calendar days
    python3 scripts/daily_journal.py --report            # print weekly report from DB
    python3 scripts/daily_journal.py --report 2026-03-03 2026-03-06  # report for date range
    python3 scripts/daily_journal.py --blocked           # list all blocked signals
    python3 scripts/daily_journal.py --misses            # list all near-misses
    python3 scripts/daily_journal.py --observations      # list all observations
    python3 scripts/daily_journal.py --gates             # check decision gate metrics
"""

import os
import re
import sys
import json
from datetime import datetime, timedelta
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shree.monitoring.trade_journal_db import (
    get_connection, init_db,
    insert_daily_summary, insert_signal, insert_blocked_signal,
    insert_near_miss, insert_trade, insert_observation, insert_gate_metric,
    get_weekly_summary, get_blocked_by_reason, get_near_miss_summary,
    get_observations_by_category,
)

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "live_trading.log")


def parse_signal_meta(meta: str) -> dict:
    """Parse signal metadata string into structured dict."""
    result = {"signal_type": "", "adx": None, "rsi": None, "atr": None, "macd_h": None}
    parts = [p.strip() for p in meta.split("|")]
    if parts:
        result["signal_type"] = parts[0].strip()
    for part in parts[1:]:
        part = part.strip()
        if part.startswith("ADX="):
            try: result["adx"] = float(part.split("=")[1])
            except: pass
        elif part.startswith("RSI="):
            try: result["rsi"] = float(part.split("=")[1])
            except: pass
        elif part.startswith("ATR="):
            try: result["atr"] = float(part.split("=")[1])
            except: pass
        elif part.startswith("MACD_H="):
            try: result["macd_h"] = float(part.split("=")[1])
            except: pass
    return result


def parse_sl_tp_from_generate(lines: list, target_time_prefix: str) -> dict:
    """Find SL/TP from the generate log line matching a time prefix (HH:MM)."""
    result = {"entry_price": None, "stop_loss": None, "take_profit": None}
    for line in lines:
        if "generate:" in line and target_time_prefix in line:
            m = re.search(r"close=([\d.]+)", line)
            if m:
                result["entry_price"] = float(m.group(1))
            m_sl = re.search(r"SL=([\d.]+)", line)
            if m_sl:
                result["stop_loss"] = float(m_sl.group(1))
            m_tp = re.search(r"TP=([\d.]+)", line)
            if m_tp:
                result["take_profit"] = float(m_tp.group(1))
            break
    return result


def compute_hypothetical_outcome(lines: list, direction: str, entry_price: float,
                                  stop_loss: float, take_profit: float,
                                  signal_time: str) -> dict:
    """
    Scan heartbeat prices AFTER the signal time to determine if a hypothetical
    trade would have hit TP or SL first.

    Returns dict with hypo_outcome, hypo_pnl, hypo_notes.
    """
    result = {"hypo_outcome": None, "hypo_pnl": None, "hypo_notes": None}

    if not entry_price or not stop_loss or not take_profit:
        result["hypo_outcome"] = "NO_DATA"
        result["hypo_notes"] = "Missing entry/SL/TP — cannot compute"
        return result

    # Parse subsequent prices from heartbeat and bar data
    subsequent_prices = []
    past_signal_time = False

    for line in lines:
        # Find lines after the signal time
        time_m = re.search(r"(\d{2}:\d{2}:\d{2}) CST", line)
        if time_m:
            line_time = time_m.group(1)
            if line_time >= signal_time:
                past_signal_time = True

        if not past_signal_time:
            continue

        # Extract price from heartbeat lines
        price_m = re.search(r"price=([\d.]+)", line)
        if price_m and price_m.group(1) != "None":
            subsequent_prices.append(float(price_m.group(1)))

        # Also extract high/low from bar data for more accuracy
        high_m = re.search(r"high=([\d.]+)", line)
        low_m = re.search(r"low=([\d.]+)", line)
        if high_m:
            subsequent_prices.append(float(high_m.group(1)))
        if low_m:
            subsequent_prices.append(float(low_m.group(1)))

    if not subsequent_prices:
        result["hypo_outcome"] = "NO_DATA"
        result["hypo_notes"] = "No price data after signal time"
        return result

    # Simulate: check each subsequent price to see if SL or TP hit first
    is_long = direction == "BUY"

    for price in subsequent_prices:
        if is_long:
            if price <= stop_loss:
                pnl = (stop_loss - entry_price) * 5.0  # $5/point for MES
                result["hypo_outcome"] = "SL_HIT"
                result["hypo_pnl"] = round(pnl, 2)
                result["hypo_notes"] = f"Price hit SL {stop_loss:.2f} (entry {entry_price:.2f})"
                return result
            elif price >= take_profit:
                pnl = (take_profit - entry_price) * 5.0
                result["hypo_outcome"] = "TP_HIT"
                result["hypo_pnl"] = round(pnl, 2)
                result["hypo_notes"] = f"Price hit TP {take_profit:.2f} (entry {entry_price:.2f})"
                return result
        else:  # SHORT
            if price >= stop_loss:
                pnl = (entry_price - stop_loss) * 5.0  # negative
                result["hypo_outcome"] = "SL_HIT"
                result["hypo_pnl"] = round(pnl, 2)
                result["hypo_notes"] = f"Price hit SL {stop_loss:.2f} (entry {entry_price:.2f})"
                return result
            elif price <= take_profit:
                pnl = (entry_price - take_profit) * 5.0
                result["hypo_outcome"] = "TP_HIT"
                result["hypo_pnl"] = round(pnl, 2)
                result["hypo_notes"] = f"Price hit TP {take_profit:.2f} (entry {entry_price:.2f})"
                return result

    # Neither hit — check distance to each
    last_price = subsequent_prices[-1]
    if is_long:
        dist_to_tp = take_profit - last_price
        dist_to_sl = last_price - stop_loss
    else:
        dist_to_tp = last_price - take_profit
        dist_to_sl = stop_loss - last_price

    unrealized_pnl = ((last_price - entry_price) if is_long else (entry_price - last_price)) * 5.0
    result["hypo_outcome"] = "EXPIRED"
    result["hypo_pnl"] = round(unrealized_pnl, 2)
    result["hypo_notes"] = (f"Neither SL nor TP hit by EOD. Last price {last_price:.2f}, "
                            f"unrealized ${unrealized_pnl:.2f}")
    return result


def ingest_date(log_path: str, target_date: str, conn):
    """Parse log for a date and insert all data into the DB."""

    lines = []
    with open(log_path, "r") as f:
        for line in f:
            if target_date in line:
                lines.append(line.strip())

    if not lines:
        print(f"  ⚠️  No log entries for {target_date}")
        return

    # Clear existing data for this date (re-ingest is idempotent)
    for table in ["signals", "blocked_signals", "near_misses"]:
        conn.execute(f"DELETE FROM {table} WHERE date = ?", (target_date,))

    # ── Heartbeat prices ──
    prices = []
    for line in lines:
        m = re.search(r"price=([\d.]+)", line)
        if m and m.group(1) != "None":
            time_m = re.search(r"(\d{2}:\d{2}:\d{2}) CST", line)
            prices.append({
                "time": time_m.group(1) if time_m else "?",
                "price": float(m.group(1)),
            })

    # ── VX level ──
    vx_level = None
    for line in lines:
        m = re.search(r"VX=([\d.]+)", line)
        if m:
            vx_level = float(m.group(1))

    # ── Opening Range ──
    or_high, or_low = None, None
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    or_file = os.path.join(data_dir, f"or_{target_date}.json")
    if os.path.exists(or_file):
        try:
            with open(or_file) as f:
                or_data = json.load(f)
                or_high = or_data.get("or_high")
                or_low = or_data.get("or_low")
        except Exception:
            pass

    # ── Support floor ──
    support_floor = None
    for line in lines:
        m = re.search(r"Dynamic support floor updated: ([\d.]+)", line)
        if m:
            support_floor = float(m.group(1))

    # ── Signals generated ──
    signals = []
    for line in lines:
        if "Strategy Signal: BUY" in line or "Strategy Signal: SELL" in line:
            m = re.search(r"Strategy Signal: (BUY|SELL) \(conf=([\d.]+), meta=(.*?)\)", line)
            if m:
                time_m = re.search(r"(\d{2}:\d{2}:\d{2}) CST", line)
                time_str = time_m.group(1) if time_m else "?"
                meta = parse_signal_meta(m.group(3))
                sl_tp = parse_sl_tp_from_generate(lines, time_str[:5])
                sig_data = {
                    "date": target_date,
                    "time": time_str,
                    "signal_type": meta["signal_type"],
                    "direction": m.group(1),
                    "confidence": float(m.group(2)),
                    "adx": meta["adx"],
                    "rsi": meta["rsi"],
                    "atr": meta["atr"],
                    "macd_h": meta["macd_h"],
                    "entry_price": sl_tp["entry_price"],
                    "stop_loss": sl_tp["stop_loss"],
                    "take_profit": sl_tp["take_profit"],
                    "outcome": None,  # filled in after block/trade parsing
                }
                signals.append(sig_data)
                insert_signal(conn, sig_data)

    # ── CHOP blocks ──
    chop_blocks = []
    for line in lines:
        if "CHOP Block-All Guard Activated" in line:
            m = re.search(r"blocking (BUY|SELL) \w+ signal \((.*?)\).*Was conf=([\d.]+)", line)
            time_m = re.search(r"(\d{2}:\d{2}:\d{2}) CST", line)
            if m:
                meta = parse_signal_meta(m.group(2))
                direction = m.group(1)
                signal_time = time_m.group(1) if time_m else "?"
                sl_tp = parse_sl_tp_from_generate(lines, signal_time[:5] if signal_time != "?" else "")

                # Compute hypothetical outcome
                hypo = compute_hypothetical_outcome(
                    lines, direction,
                    sl_tp["entry_price"], sl_tp["stop_loss"], sl_tp["take_profit"],
                    signal_time
                )

                block_data = {
                    "date": target_date,
                    "time": signal_time,
                    "signal_type": meta["signal_type"],
                    "direction": direction,
                    "block_reason": "CHOP_GUARD",
                    "confidence_at_block": float(m.group(3)),
                    "adx": meta["adx"],
                    "rsi": meta["rsi"],
                    "atr": meta["atr"],
                    "entry_price": sl_tp["entry_price"],
                    "stop_loss": sl_tp["stop_loss"],
                    "take_profit": sl_tp["take_profit"],
                    "hypo_outcome": hypo["hypo_outcome"],
                    "hypo_pnl": hypo["hypo_pnl"],
                    "hypo_notes": hypo["hypo_notes"],
                }
                chop_blocks.append(block_data)
                insert_blocked_signal(conn, block_data)

    # ── Confidence blocks ──
    conf_block_count = 0
    for line in lines:
        if "BLOCKED" in line and "confidence" in line.lower():
            time_m = re.search(r"(\d{2}:\d{2}:\d{2}) CST", line)
            m = re.search(r"confidence ([\d.]+) < threshold ([\d.]+)", line)
            if m:
                conf_block_count += 1
                insert_blocked_signal(conn, {
                    "date": target_date,
                    "time": time_m.group(1) if time_m else "?",
                    "signal_type": "UNKNOWN",
                    "direction": "UNKNOWN",
                    "block_reason": "CONFIDENCE",
                    "confidence_at_block": float(m.group(1)),
                })

    # ── Touch-band near-misses ──
    d_near_misses = []
    for line in lines:
        if "NO_SIGNAL diag:" in line:
            # D short pullback near-miss
            m = re.search(r"D:high\(([\d.]+)\)<touch\(([\d.]+)\)", line)
            if m:
                high_val = float(m.group(1))
                touch_val = float(m.group(2))
                gap = touch_val - high_val
                time_m = re.search(r"(\d{2}:\d{2}:\d{2}) CST", line)
                miss = {
                    "date": target_date,
                    "time": time_m.group(1) if time_m else "?",
                    "signal_type": "D_SHORT_PB",
                    "miss_reason": "touch_band",
                    "miss_detail": f"high={high_val:.1f} needed={touch_val:.1f}",
                    "gap_pts": round(gap, 2),
                }
                d_near_misses.append(miss)
                insert_near_miss(conn, miss)

            # A long pullback near-miss
            m = re.search(r"A:low\(([\d.]+)\)>touch\(([\d.]+)\)", line)
            if m:
                low_val = float(m.group(1))
                touch_val = float(m.group(2))
                gap = low_val - touch_val
                if gap <= 10.0:  # only log reasonably close misses
                    time_m = re.search(r"(\d{2}:\d{2}:\d{2}) CST", line)
                    insert_near_miss(conn, {
                        "date": target_date,
                        "time": time_m.group(1) if time_m else "?",
                        "signal_type": "A_LONG_PB",
                        "miss_reason": "touch_band",
                        "miss_detail": f"low={low_val:.1f} needed={touch_val:.1f}",
                        "gap_pts": round(gap, 2),
                    })

            # E gap-through miss
            m = re.search(r"E:no_cross\(c=([\d.]+),prev=([\d.]+),OR_L=([\d.]+)\)", line)
            if m:
                close_val = float(m.group(1))
                prev_val = float(m.group(2))
                or_l = float(m.group(3))
                # Both below OR_L = gap-through
                if close_val < or_l and prev_val < or_l:
                    time_m = re.search(r"(\d{2}:\d{2}:\d{2}) CST", line)
                    insert_near_miss(conn, {
                        "date": target_date,
                        "time": time_m.group(1) if time_m else "?",
                        "signal_type": "E_OR_BREAKDOWN",
                        "miss_reason": "gap_through",
                        "miss_detail": f"c={close_val:.1f} prev={prev_val:.1f} OR_L={or_l:.1f} (both below)",
                        "gap_pts": 0.0,
                    })

    # ── Trades (from Realized PnL) ──
    trade_count = 0
    gross_pnl = 0.0
    net_pnl = 0.0
    for line in lines:
        m = re.search(r"Realized PnL gross=([-\d.]+) net=([-\d.]+)", line)
        if m:
            trade_count += 1
            gross_pnl += float(m.group(1))
            net_pnl += float(m.group(2))

    # ── Resolve signal outcomes (EXECUTED vs BLOCKED) ──
    # Build a set of blocked signal times for matching
    blocked_times = set()
    for b in chop_blocks:
        blocked_times.add(b["time"])

    conf_blocked_times = set()
    for line in lines:
        if "BLOCKED" in line and "confidence" in line.lower():
            time_m = re.search(r"(\d{2}:\d{2}:\d{2}) CST", line)
            if time_m:
                conf_blocked_times.add(time_m.group(1))

    # Check for order execution events (executed signals)
    executed_times = set()
    for line in lines:
        if any(kw in line for kw in ["order_placed", "Placing bracket order", "Placing HYBRID order", "Placing BUY order", "Placing SELL order", "HYBRID: Placing"]):
            time_m = re.search(r"(\d{2}:\d{2}:\d{2}) CST", line)
            if time_m:
                executed_times.add(time_m.group(1))

    for sig in signals:
        sig_time = sig["time"]
        # Match signal to nearest blocked/executed event within ~2 min window
        if sig_time in blocked_times or any(abs(int(sig_time[:2])*60 + int(sig_time[3:5]) - int(bt[:2])*60 - int(bt[3:5])) <= 2 for bt in blocked_times):
            sig["outcome"] = "CHOP_BLOCKED"
        elif sig_time in conf_blocked_times or any(abs(int(sig_time[:2])*60 + int(sig_time[3:5]) - int(ct[:2])*60 - int(ct[3:5])) <= 2 for ct in conf_blocked_times):
            sig["outcome"] = "CONF_BLOCKED"
        elif executed_times or trade_count > 0:
            # If there are executed trades and this signal isn't blocked, likely executed
            if any(abs(int(sig_time[:2])*60 + int(sig_time[3:5]) - int(et[:2])*60 - int(et[3:5])) <= 5 for et in executed_times):
                sig["outcome"] = "EXECUTED"
            else:
                sig["outcome"] = "UNKNOWN"
        else:
            sig["outcome"] = "UNKNOWN"

    # Update signal outcomes in DB
    for sig in signals:
        if sig["outcome"]:
            conn.execute("""
                UPDATE signals SET outcome = ?
                WHERE date = ? AND time = ? AND signal_type = ?
            """, (sig["outcome"], sig["date"], sig["time"], sig["signal_type"]))

    # ── Insert daily summary ──
    summary = {
        "date": target_date,
        "open_price": prices[0]["price"] if prices else None,
        "close_price": prices[-1]["price"] if prices else None,
        "high_price": max(p["price"] for p in prices) if prices else None,
        "low_price": min(p["price"] for p in prices) if prices else None,
        "range_pts": round(max(p["price"] for p in prices) - min(p["price"] for p in prices), 2) if prices else None,
        "vx_level": vx_level,
        "or_high": or_high,
        "or_low": or_low,
        "support_floor": support_floor,
        "total_trades": trade_count,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "signals_generated": len(signals),
        "chop_blocks": len(chop_blocks),
        "confidence_blocks": conf_block_count,
        "d_near_misses": len(d_near_misses),
        "d_near_misses_5pt": len([d for d in d_near_misses if d["gap_pts"] <= 5.0]),
    }
    insert_daily_summary(conn, summary)

    conn.commit()
    print(f"  ✅ {target_date}: {len(signals)} signals, {len(chop_blocks)} CHOP blocks, "
          f"{len(d_near_misses)} D near-misses, {trade_count} trades, P&L=${net_pnl:.2f}")


# ── Report / Query Functions ──

def print_report(conn, start_date: str, end_date: str):
    """Print a formatted report from the DB."""
    print(f"\n{'='*70}")
    print(f"  TRADE JOURNAL REPORT: {start_date} → {end_date}")
    print(f"{'='*70}\n")

    # Weekly summary
    ws = get_weekly_summary(conn, start_date, end_date)
    if ws:
        print(f"📊 SUMMARY ({ws['trading_days']} trading days)")
        print(f"   Total trades:        {ws['total_trades'] or 0}")
        print(f"   Wins / Losses:       {ws['total_wins'] or 0}W / {ws['total_losses'] or 0}L")
        print(f"   Net P&L:             ${ws['total_pnl'] or 0:.2f}")
        print(f"   Trades/day:          {ws['trades_per_day'] or 0:.2f}")
        print(f"   Signals generated:   {ws['total_signals'] or 0}")
        print(f"   CHOP blocks:         {ws['total_chop_blocks'] or 0}")
        print(f"   D near-misses:       {ws['total_d_near_misses'] or 0}")
        print()

    # Daily breakdown
    rows = conn.execute("""
        SELECT * FROM daily_summary WHERE date BETWEEN ? AND ? ORDER BY date
    """, (start_date, end_date)).fetchall()
    if rows:
        print(f"📅 DAILY BREAKDOWN")
        print(f"   {'Date':<12} {'Price':>10} {'Range':>8} {'Trades':>7} {'Signals':>8} {'CHOP':>6} {'D Miss':>7} {'P&L':>10}")
        print(f"   {'─'*12} {'─'*10} {'─'*8} {'─'*7} {'─'*8} {'─'*6} {'─'*7} {'─'*10}")
        for r in rows:
            price_str = f"{r['close_price']:.0f}" if r['close_price'] else "—"
            range_str = f"{r['range_pts']:.0f}pt" if r['range_pts'] else "—"
            pnl_str = f"${r['net_pnl']:.2f}" if r['net_pnl'] else "$0.00"
            print(f"   {r['date']:<12} {price_str:>10} {range_str:>8} "
                  f"{r['total_trades'] or 0:>7} {r['signals_generated'] or 0:>8} "
                  f"{r['chop_blocks'] or 0:>6} {r['d_near_misses'] or 0:>7} {pnl_str:>10}")
        print()

    # Blocked signals by reason
    blocked = get_blocked_by_reason(conn, start_date, end_date)
    if blocked:
        print(f"🚫 BLOCKED SIGNALS BY REASON")
        for b in blocked:
            won = b['would_have_won'] or 0
            lost = b['would_have_lost'] or 0
            print(f"   {b['block_reason']:<20} {b['count']:>3} blocks | "
                  f"Would-have-won: {won} | Would-have-lost: {lost}")
        print()

    # Near-miss summary
    nm = get_near_miss_summary(conn, start_date, end_date)
    if nm and nm.get('total'):
        print(f"📏 NEAR-MISS SUMMARY")
        print(f"   Total:       {nm['total']}")
        print(f"   Within 1pt:  {nm['within_1pt'] or 0}")
        print(f"   Within 3pt:  {nm['within_3pt'] or 0}")
        print(f"   Within 5pt:  {nm['within_5pt'] or 0}")
        if nm['closest_miss'] is not None:
            print(f"   Closest:     {nm['closest_miss']:.1f}pts")
        if nm['avg_gap'] is not None:
            print(f"   Avg gap:     {nm['avg_gap']:.1f}pts")
        print()

    # Decision gate check
    print(f"🚦 DECISION GATE CHECK")
    total_nm = nm.get('total', 0) or 0
    gate1 = "✅ MET" if total_nm >= 10 else f"❌ NOT MET ({total_nm}/10)"
    print(f"   Mar 10 — Touch-band near-misses ≥ 10:  {gate1}")
    tpd = ws.get('trades_per_day', 0) or 0
    gate2 = "✅ MET (need action)" if tpd < 2 else f"❌ Adequate ({tpd:.1f}/day)"
    print(f"   Mar 17 — Trades/day < 2:               {gate2}")
    print()


def print_blocked(conn):
    """Print all blocked signals."""
    rows = conn.execute("""
        SELECT * FROM blocked_signals ORDER BY date, time
    """).fetchall()
    print(f"\n🚫 ALL BLOCKED SIGNALS ({len(rows)} total)\n")
    print(f"   {'Date':<12} {'Time':<10} {'Dir':<5} {'Signal':<25} {'Reason':<15} {'Conf':>6} {'ADX':>5} {'Hypo':>8}")
    print(f"   {'─'*12} {'─'*10} {'─'*5} {'─'*25} {'─'*15} {'─'*6} {'─'*5} {'─'*8}")
    for r in rows:
        hypo = r['hypo_outcome'] or "—"
        adx = f"{r['adx']:.0f}" if r['adx'] else "—"
        print(f"   {r['date']:<12} {r['time']:<10} {r['direction']:<5} {r['signal_type']:<25} "
              f"{r['block_reason']:<15} {r['confidence_at_block'] or 0:>6.3f} {adx:>5} {hypo:>8}")
    print()


def print_misses(conn):
    """Print all near-misses."""
    rows = conn.execute("""
        SELECT * FROM near_misses ORDER BY gap_pts ASC
    """).fetchall()
    print(f"\n📏 ALL NEAR-MISSES ({len(rows)} total, sorted by gap)\n")
    print(f"   {'Date':<12} {'Time':<10} {'Signal':<15} {'Detail':<40} {'Gap':>8}")
    print(f"   {'─'*12} {'─'*10} {'─'*15} {'─'*40} {'─'*8}")
    for r in rows:
        print(f"   {r['date']:<12} {r['time']:<10} {r['signal_type']:<15} "
              f"{r['miss_detail'] or '—':<40} {r['gap_pts']:>7.1f}pt")
    print()


def print_observations(conn):
    """Print all observations."""
    rows = conn.execute("SELECT * FROM observations ORDER BY date DESC, id DESC").fetchall()
    print(f"\n📝 OBSERVATIONS ({len(rows)} total)\n")
    for r in rows:
        sev_icon = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🔴", "ACTION_NEEDED": "🟡"}.get(r['severity'], "•")
        print(f"   {sev_icon} [{r['date']}] [{r['category']}] {r['title']}")
        if r['detail']:
            print(f"      {r['detail'][:120]}")
        if r['recommended_fix']:
            print(f"      → Fix: {r['recommended_fix']}")
        print()


def print_gates(conn):
    """Print decision gate metrics."""
    rows = conn.execute("SELECT * FROM gate_metrics ORDER BY gate_date, metric_name").fetchall()
    print(f"\n🚦 DECISION GATE METRICS\n")
    if not rows:
        print("   No gate metrics recorded yet. Run --report to see computed metrics.")
        return
    for r in rows:
        met = "✅" if r['gate_met'] else "❌"
        print(f"   {met} [{r['gate_date']}] {r['metric_name']}: "
              f"{r['metric_value']:.1f} (threshold: {r['threshold']:.1f})")
        if r['notes']:
            print(f"      {r['notes']}")
    print()


def main():
    init_db()
    conn = get_connection()

    args = sys.argv[1:]

    # Report / query modes
    if "--report" in args:
        args_clean = [a for a in args if a != "--report"]
        if len(args_clean) >= 2:
            start_date, end_date = args_clean[0], args_clean[1]
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        print_report(conn, start_date, end_date)
        conn.close()
        return

    if "--blocked" in args:
        print_blocked(conn)
        conn.close()
        return

    if "--misses" in args:
        print_misses(conn)
        conn.close()
        return

    if "--observations" in args:
        print_observations(conn)
        conn.close()
        return

    if "--gates" in args:
        print_gates(conn)
        conn.close()
        return

    # ── Ingest mode ──
    target_date = None
    for a in args:
        if not a.startswith("--"):
            target_date = a
            break
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")

    is_week = "--week" in args

    print(f"\n📥 Ingesting trade journal data from logs...")

    if is_week:
        base = datetime.strptime(target_date, "%Y-%m-%d")
        for i in range(6, -1, -1):
            d = base - timedelta(days=i)
            if d.weekday() < 5:  # Mon-Fri
                ingest_date(LOG_PATH, d.strftime("%Y-%m-%d"), conn)
    else:
        ingest_date(LOG_PATH, target_date, conn)

    conn.close()
    print(f"\n✅ Done. Query with:")
    print(f"   python3 scripts/daily_journal.py --report")
    print(f"   python3 scripts/daily_journal.py --blocked")
    print(f"   python3 scripts/daily_journal.py --misses")


if __name__ == "__main__":
    main()
