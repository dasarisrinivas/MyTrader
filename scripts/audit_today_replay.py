#!/usr/bin/env python3
"""Missed-opportunity audit: replay TODAY's session through the FROZEN engine.

Answers: was zero trades correct? Real option dollar P&L for every dispatched
signal AND every blocked signal, plus MFE/MAE, gate effectiveness, confidence
calibration, strategy dashboard.

Engine is frozen (v2) — used unchanged. Two labeled assumptions where production
gives no exit:
  A1 open signals (no exit_at) -> 90-min horizon (production time_stop default)
  A2 blocked signals (never dispatched, no exit) -> same 90-min horizon
Both are clearly separated from production-exit replays in the output.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))
from shree.flow_research.thetadata import ThetaClient
from shree.research.replay_engine import QuoteCache, ExecutionModel, _prevailing

ET = ZoneInfo("America/New_York")
DB = "/Users/svss/Documents/code/ShreeBot/data/spy_options_signals.db"
BLOCKED = "/Users/svss/Documents/code/ShreeBot/logs/blocked_signals.jsonl"
TODAY = "2026-07-27"
HORIZON_MIN = 90
COMM_RT = 1.30
PER_GATE_CAP = 45


def et(ts):
    s = ts.split(".")[0].rstrip("Z")
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).astimezone(ET)


def key(dt):
    return dt.replace(tzinfo=None).isoformat(timespec="milliseconds")


def replay(cache, expiry, strike, right, entry_dt, exit_dt):
    """Return (entry_ask, exit_bid, net$, ret, mfe$, mae$) or None."""
    try:
        quotes, keys = cache.get(expiry, float(strike), right,
                                 entry_dt.strftime("%Y%m%d"))
    except Exception:
        return None
    if not quotes:
        return None
    _, ea = _prevailing(quotes, keys, key(entry_dt))
    _, _xa = None, None
    xb, _ = None, None
    xb_val = _prevailing(quotes, keys, key(exit_dt))[0]
    if not ea or not xb_val:
        return None
    net = (xb_val - ea) * 100.0 - COMM_RT
    ret = net / (ea * 100.0)
    # MFE/MAE on the bid path between entry and exit
    lo_k, hi_k = key(entry_dt), key(exit_dt)
    best = worst = None
    for ts, b, a in quotes:
        if ts < lo_k or ts > hi_k:
            continue
        try:
            bid = float(b)
        except (TypeError, ValueError):
            continue
        if bid <= 0:
            continue
        pnl = (bid - ea) * 100.0 - COMM_RT
        best = pnl if best is None else max(best, pnl)
        worst = pnl if worst is None else min(worst, pnl)
    return ea, xb_val, net, ret, (best if best is not None else net), \
        (worst if worst is not None else net)


def main():
    client = ThetaClient()
    cache = QuoteCache(client)

    # ---------- dispatched signals ----------
    c = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    sigs = c.execute(
        """SELECT id, sent_at, exit_at, signal_type, strike, right, expiry_date,
                  dte, confidence, confidence_tier, outcome, pnl_pct
           FROM spy_signals WHERE id>=887 ORDER BY id""").fetchall()
    c.close()
    sigs = [s for s in sigs if et(s["sent_at"]).date().isoformat() == TODAY]

    rows = []
    for s in sigs:
        e = et(s["sent_at"])
        if s["exit_at"]:
            x = et(s["exit_at"]); basis = "prod_exit"
        else:
            x = e + timedelta(minutes=HORIZON_MIN); basis = "A1_90min"
        if x.date() != e.date():
            x = e.replace(hour=15, minute=55, second=0); basis = "A1_eod"
        r = replay(cache, s["expiry_date"], s["strike"], s["right"], e, x)
        if not r:
            continue
        ea, xb, net, ret, mfe, mae = r
        rows.append({"id": s["id"], "fam": s["signal_type"], "conf": s["confidence"],
                     "net": net, "ret": ret, "mfe": mfe, "mae": mae,
                     "basis": basis, "dte": s["dte"]})

    n = len(rows)
    prof = sum(1 for r in rows if r["net"] > 1)
    lose = sum(1 for r in rows if r["net"] < -1)
    flat = n - prof - lose
    tot = sum(r["net"] for r in rows)
    print(f"=== #1 MISSED-OPPORTUNITY: {len(sigs)} dispatched, {n} replayable ===")
    print(f"  profitable={prof}  losing={lose}  breakeven={flat}")
    print(f"  EV if EVERY dispatched signal traded (1 contract): ${tot/n:+.2f}/trade, "
          f"total ${tot:+.0f}")
    print(f"  EV after today's gates (0 trades executed):        $0.00")
    print(f"  => gates {'SAVED' if tot < 0 else 'COST'} ${abs(tot):.0f}")
    print(f"  mean MFE=${sum(r['mfe'] for r in rows)/n:+.0f}  "
          f"mean MAE=${sum(r['mae'] for r in rows)/n:+.0f}")

    # ---------- #9 dashboard ----------
    print("\n=== #9 STRATEGY DASHBOARD (dispatched, replay EV) ===")
    print(f"{'strategy':<20}{'sigs':>6}{'replayed':>10}{'EV$':>9}{'win%':>7}{'total$':>9}")
    byf = defaultdict(list)
    for r in rows:
        byf[r["fam"]].append(r)
    for fam in sorted(byf, key=lambda f: -len(byf[f])):
        g = byf[fam]
        cnt = sum(1 for s in sigs if s["signal_type"] == fam)
        w = sum(1 for r in g if r["net"] > 0)
        print(f"{fam:<20}{cnt:>6}{len(g):>10}{sum(r['net'] for r in g)/len(g):>9.2f}"
              f"{100*w/len(g):>7.0f}{sum(r['net'] for r in g):>9.0f}")

    # ---------- #3 confidence calibration ----------
    print("\n=== #3 CONFIDENCE CALIBRATION (replay EV by confidence) ===")
    buckets = [("LOW <0.3", 0, .3), ("MED 0.3-0.6", .3, .6), ("HIGH >=0.6", .6, 2)]
    print(f"{'bucket':<14}{'n':>5}{'EV$':>9}{'win%':>7}")
    for lab, lo, hi in buckets:
        g = [r for r in rows if lo <= (r["conf"] or 0) < hi]
        if not g:
            continue
        w = sum(1 for r in g if r["net"] > 0)
        print(f"{lab:<14}{len(g):>5}{sum(r['net'] for r in g)/len(g):>9.2f}{100*w/len(g):>7.0f}")

    # ---------- #2 gate effectiveness ----------
    print(f"\n=== #2 GATE EFFECTIVENESS (blocked signals, A2 {HORIZON_MIN}min horizon) ===")
    bl = []
    for line in open(BLOCKED):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        ts = r.get("ts") or ""
        if not ts:
            continue
        try:
            d = et(ts)
        except Exception:
            continue
        if d.date().isoformat() == TODAY and r.get("strike") and r.get("expiry"):
            bl.append((d, r))
    bygate = defaultdict(list)
    for d, r in bl:
        bygate[r.get("gate")].append((d, r))
    print(f"{'gate':<26}{'blocked':>8}{'replayed':>9}{'EV$':>8}{'winners':>8}{'losers':>7}{'net_saved$':>11}")
    for gate, items in sorted(bygate.items(), key=lambda kv: -len(kv[1])):
        sub = items[:: max(1, len(items) // PER_GATE_CAP)][:PER_GATE_CAP]
        res = []
        for d, r in sub:
            exp = str(r.get("expiry", "")).replace("-", "")
            rr = replay(cache, exp, r["strike"], r["right"], d,
                        d + timedelta(minutes=HORIZON_MIN))
            if rr:
                res.append(rr[2])
        if not res:
            print(f"{gate:<26}{len(items):>8}{0:>9}      n/a")
            continue
        w = sum(1 for x in res if x > 0)
        ev = sum(res) / len(res)
        print(f"{gate:<26}{len(items):>8}{len(res):>9}{ev:>8.2f}{w:>8}{len(res)-w:>7}"
              f"{-ev*len(items):>11.0f}")
    print("\n(net_saved$ = -EV x blocked count; positive => gate SAVED money)")


if __name__ == "__main__":
    main()
