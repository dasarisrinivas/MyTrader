"""Replay MES shadow signals against real IB 1-min bars and report P&L.

Shadow-only audit tool: reads logs/mes_signals.jsonl, replays every actionable
signal with realistic execution, prints per-trade + daily + aggregate stats.
Places no orders; uses IB read-only for historical bars.

Usage:
    python3 tools/shadow_replay.py                    # last full Mon-Fri week
    python3 tools/shadow_replay.py 2026-07-27 2026-07-31
    python3 tools/shadow_replay.py --bars /tmp/mes_1m.parquet 2026-07-27 2026-07-31

Execution assumptions (documented, not tuned):
    entry    = next 1-min bar OPEN after the signal timestamp
    slippage = 1 tick (0.25 pt) adverse each side
    fees     = $0.85/side ($1.70 round turn) IBKR MES all-in
    exits    = stop -> target -> EOD flatten at RTH close (20:00 UTC)
    intrabar = if a bar touches both stop and target, STOP is assumed first
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SLIP = 0.25
FEE_SIDE = 0.85
FEE_RT = FEE_SIDE * 2
POINT_VALUE = 5.0
SIGNAL_LOG = Path("logs/mes_signals.jsonl")
IB_CLIENT_ID = 17  # spare id — never collides with the live bot (cid=1)


def last_full_week() -> tuple[str, str]:
    """Most recent completed Monday-Friday."""
    today = date.today()
    monday_this_week = today - timedelta(days=today.weekday())
    monday = monday_this_week - timedelta(days=7)
    return monday.isoformat(), (monday + timedelta(days=4)).isoformat()


def load_signals(start: str, end: str) -> list[dict]:
    if not SIGNAL_LOG.exists():
        sys.exit(f"missing {SIGNAL_LOG}")
    out = []
    for line in SIGNAL_LOG.open():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("signal") in ("BUY", "SELL") and start <= d["ts"][:10] <= end:
            out.append(d)
    return out


def fetch_bars(days: int = 10) -> pd.DataFrame:
    from ib_insync import IB, ContFuture

    ib = IB()
    ib.connect("127.0.0.1", 4001, clientId=IB_CLIENT_ID, timeout=30)
    ib.RequestTimeout = 180
    try:
        contract = ib.qualifyContracts(ContFuture("MES", "CME", currency="USD"))[0]
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr=f"{days} D",
            barSizeSetting="1 min", whatToShow="TRADES", useRTH=False, formatDate=2,
        )
    finally:
        ib.disconnect()
    if not bars:
        sys.exit("IB returned no bars — is the gateway up on 4001?")
    df = pd.DataFrame(
        {"open": [b.open for b in bars], "high": [b.high for b in bars],
         "low": [b.low for b in bars], "close": [b.close for b in bars]},
        index=pd.DatetimeIndex([pd.Timestamp(b.date) for b in bars]),
    )
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[~df.index.duplicated()].sort_index()


def replay(signals: list[dict], bars: pd.DataFrame) -> list[dict]:
    trades = []
    for n, sig in enumerate(signals, 1):
        ts = pd.Timestamp(sig["ts"]).tz_convert("UTC")
        stop, target = float(sig["stop"]), float(sig["target"])
        is_long = sig["signal"] == "BUY"
        fwd = bars[bars.index > ts]
        if fwd.empty:
            continue
        entry = float(fwd.iloc[0]["open"]) + (SLIP if is_long else -SLIP)
        fill_ts = fwd.index[0]
        session_end = ts.normalize() + pd.Timedelta(hours=20)  # RTH close
        path = fwd[fwd.index <= session_end]
        if path.empty:
            path = fwd.iloc[:1]

        exit_px = exit_ts = reason = None
        for bar_ts, bar in path.iterrows():
            if is_long:
                if bar["low"] <= stop:
                    exit_px, exit_ts, reason = stop - SLIP, bar_ts, "STOP"
                    break
                if bar["high"] >= target:
                    exit_px, exit_ts, reason = target - SLIP, bar_ts, "TARGET"
                    break
            else:
                if bar["high"] >= stop:
                    exit_px, exit_ts, reason = stop + SLIP, bar_ts, "STOP"
                    break
                if bar["low"] <= target:
                    exit_px, exit_ts, reason = target + SLIP, bar_ts, "TARGET"
                    break
        if exit_px is None:
            exit_px = float(path.iloc[-1]["close"]) + (-SLIP if is_long else SLIP)
            exit_ts, reason = path.index[-1], "EOD_FLAT"

        pts = (exit_px - entry) if is_long else (entry - exit_px)
        risk = abs(entry - stop)
        trades.append(dict(
            n=n, sig_ts=ts, fill_ts=fill_ts, exit_ts=exit_ts,
            direction=sig["signal"], setup=sig["supporting_evidence"].split(" |")[0],
            entry=entry, stop=stop, target=target, exit=exit_px, pts=pts,
            gross=pts * POINT_VALUE, fees=FEE_RT, net=pts * POINT_VALUE - FEE_RT,
            R=(pts / risk) if risk > 0 else 0.0, reason=reason,
            hold_min=(exit_ts - fill_ts).total_seconds() / 60,
            day=ts.tz_convert("US/Central").strftime("%Y-%m-%d"),
        ))
    return trades


def summarize(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"{label}: no trades")
        return
    net = np.array([t["net"] for t in rows])
    wins = [t for t in rows if t["net"] > 0]
    losses = [t for t in rows if t["net"] <= 0]
    gp = sum(t["net"] for t in wins)
    gl = -sum(t["net"] for t in losses)
    eq = np.cumsum(net)
    dd = (eq - np.maximum.accumulate(eq)).min()
    t_stat = net.mean() / (net.std(ddof=1) / math.sqrt(len(net))) if len(net) > 1 else 0.0
    pf = gp / gl if gl > 0 else float("inf")
    print(f"{label:34} n={len(rows):>3}  net=${net.sum():>+9.2f}  EV=${net.mean():>+7.2f}  "
          f"WR={100*len(wins)/len(rows):>5.1f}%  PF={pf:>5.2f}  maxDD=${dd:>8.2f}  t={t_stat:>+5.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay MES shadow signals (no orders)")
    ap.add_argument("start", nargs="?"), ap.add_argument("end", nargs="?")
    ap.add_argument("--bars", help="parquet of 1-min bars (skip IB fetch)")
    args = ap.parse_args()
    start, end = (args.start, args.end) if args.start and args.end else last_full_week()

    signals = load_signals(start, end)
    print(f"Window {start} .. {end} — actionable signals: {len(signals)}")
    if not signals:
        print("No BUY/SELL signals in window. Nothing to replay.")
        return

    bars = pd.read_parquet(args.bars) if args.bars else fetch_bars()
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    trades = replay(signals, bars.sort_index())
    if not trades:
        print("Signals found but no forward bars to replay them against.")
        return

    print("\n" + "=" * 104)
    print(f"{'#':>2} {'signal (CT)':16} {'dir':4} {'setup':17} {'entry':>8} {'exit':>8} "
          f"{'pts':>7} {'net':>8} {'R':>6} {'hold':>6} {'exit':9}")
    for t in trades:
        ct = t["sig_ts"].tz_convert("US/Central").strftime("%m-%d %H:%M")
        print(f"{t['n']:>2} {ct:16} {t['direction']:4} {t['setup']:17} {t['entry']:>8.2f} "
              f"{t['exit']:>8.2f} {t['pts']:>+7.2f} {t['net']:>+8.2f} {t['R']:>+6.2f} "
              f"{t['hold_min']:>5.0f}m {t['reason']:9}")

    print("\nDAILY")
    for day in sorted({t["day"] for t in trades}):
        dt = [t for t in trades if t["day"] == day]
        w = len([t for t in dt if t["net"] > 0])
        print(f"  {day}  trades={len(dt):>2}  net=${sum(t['net'] for t in dt):>+8.2f}  "
              f"wins={w}/{len(dt)}")

    print("\nSTRATEGY")
    for s in sorted({t["setup"] for t in trades}):
        summarize([t for t in trades if t["setup"] == s], f"  {s}")

    print("\nAGGREGATE")
    summarize(trades, "A) every shadow signal")
    sequential, busy_until = [], None
    for t in sorted(trades, key=lambda x: x["fill_ts"]):
        if busy_until is not None and t["fill_ts"] < busy_until:
            continue
        sequential.append(t)
        busy_until = t["exit_ts"]
    summarize(sequential, "B) 1-position-at-a-time")

    spans = [(t["fill_ts"], t["exit_ts"]) for t in trades]
    edges = sorted({x for pair in spans for x in pair})
    max_conc = max(sum(1 for a, b in spans if a <= e < b) for e in edges)
    print(f"\n   Peak concurrent positions if all signals taken: {max_conc} contract(s)")
    print("   B) is the honest number for a 1-contract account.")
    print("\n   Reminder: shadow only — no orders placed. Judge at 100+ trades, not one week.")


if __name__ == "__main__":
    main()
