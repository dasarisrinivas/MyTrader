#!/usr/bin/env python3
"""NEW ALPHA #1 — defined-risk SHORT PREMIUM landscape (research, not optimization).

Constructs SYNTHETIC historical SPY credit spreads (no such signals exist) and
replays them with the validated v3.0 instrument. Maps the landscape across
structure / moneyness / width / DTE / entry time / exit rule BEFORE any tuning.

Structures (defined risk, fixed max loss):
  bull_put  : SELL put  K       + BUY put  K-width      (credit)
  bear_call : SELL call K       + BUY call K+width      (credit)

Conservative fills (entry long=ask/short=bid, exit long=bid/short=ask),
$2.60 RT fees, no assignment modeling.

KNOWN LIMITATION: American-style SPY options — early assignment of short legs is
NOT modeled. (Material for short legs; see report footer.)
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from shree.flow_research.thetadata import ThetaClient
from shree.research.replay_engine import QuoteCache
from shree.research.replay_engine_v3 import (
    synthetic_vertical, replay_multileg, provenance, V3_LIMITATION,
)

ET = ZoneInfo("America/New_York")
DB = "/Users/svss/Documents/code/ShreeBot/data/spy_options_signals.db"
RNG = np.random.default_rng(20260727)

ENTRY_TIMES = ["10:00", "12:00", "14:00"]
OFFSETS = [0.003, 0.007]      # short-strike distance from spot (delta proxy)
WIDTHS = [2, 5]
DTES = [0, 1]
EXITS = [("90m", 90), ("EOD", None)]
TYPES = ["bull_put", "bear_call"]


def et(t):
    s = t.split(".")[0].rstrip("Z")
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).astimezone(ET)


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) < 5:
        return float("nan"), float("nan")
    m = RNG.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=20)
    a = ap.parse_args()

    c = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    rows = c.execute("SELECT sent_at, spy_price FROM spy_signals "
                     "WHERE spy_price IS NOT NULL ORDER BY sent_at").fetchall()
    c.close()
    spot = defaultdict(list)
    for r in rows:
        d = et(r["sent_at"])
        spot[d.date()].append((d, float(r["spy_price"])))
    sessions = sorted(spot)[-a.sessions:]

    client = ThetaClient()
    cache = QuoteCache(client)
    exps = sorted(client.list_expirations("SPY"))

    def spot_at(sess, when):
        arr = spot[sess]
        ts = [x[0] for x in arr]
        i = bisect_right(ts, when) - 1
        return arr[i][1] if i >= 0 else None

    def expiry_for(sess, dte):
        fut = [e for e in exps if e >= sess]
        return fut[dte] if len(fut) > dte else None

    grid = defaultdict(list)   # config -> [(net, spy_ret, credit, maxloss)]
    n_try = n_ok = 0
    for sess in sessions:
        arr = spot[sess]
        if len(arr) < 2:
            continue
        spy_ret = (arr[-1][1] - arr[0][1]) / arr[0][1]
        for hhmm in ENTRY_TIMES:
            h, m = map(int, hhmm.split(":"))
            entry = datetime.combine(sess, datetime(2000, 1, 1, h, m).time(),
                                     tzinfo=ET)
            s0 = spot_at(sess, entry)
            if not s0:
                continue
            for dte in DTES:
                exp = expiry_for(sess, dte)
                if exp is None:
                    continue
                eymd = exp.strftime("%Y%m%d")
                for typ in TYPES:
                    for off in OFFSETS:
                        for w in WIDTHS:
                            if typ == "bull_put":
                                ks = round(s0 * (1 - off))
                                st = synthetic_vertical("P", ks - w, ks, eymd)
                            else:
                                ks = round(s0 * (1 + off))
                                st = synthetic_vertical("C", ks + w, ks, eymd)
                            for elabel, mins in EXITS:
                                x = (entry + timedelta(minutes=mins)) if mins \
                                    else entry.replace(hour=15, minute=55)
                                if x.time() > datetime(2000, 1, 1, 15, 55).time():
                                    x = entry.replace(hour=15, minute=55)
                                if x <= entry:
                                    continue
                                n_try += 1
                                r = replay_multileg(cache, st, entry, x)
                                if not r.ok or r.entry_net is None:
                                    continue
                                if r.entry_net >= 0:
                                    continue   # not a credit — skip
                                n_ok += 1
                                cfg = (typ, f"{off:.3f}", w, dte, hhmm, elabel)
                                grid[cfg].append(
                                    (r.net_dollar, spy_ret, -r.entry_net * 100,
                                     r.max_loss))
    print(f"synthetic credit spreads attempted={n_try} priced={n_ok} "
          f"sessions={len(sessions)}")
    print(f"configs mapped: {len(grid)}\n")

    # ── landscape ───────────────────────────────────────────────────────────
    out = []
    for cfg, vals in grid.items():
        nets = [v[0] for v in vals]
        rets = [v[1] for v in vals]
        creds = [v[2] for v in vals]
        n = len(nets)
        if n < 5:
            continue
        wins = [x for x in nets if x > 0]
        losses = [x for x in nets if x <= 0]
        ev = sum(nets) / n
        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None
        lo, hi = boot_ci(nets)
        cum = 0.0; peak = 0.0; mdd = 0.0
        for x in nets:
            cum += x; peak = max(peak, cum); mdd = min(mdd, cum - peak)
        corr = alpha = float("nan")
        if n >= 5 and np.std(rets) > 0 and np.std(nets) > 0:
            corr = float(np.corrcoef(rets, nets)[0, 1])
            slope, _ = np.polyfit(np.array(rets), np.array(nets), 1)
            alpha = float(np.mean(np.array(nets) - slope * np.array(rets)))
        theta_capture = float(np.mean([nets[i] / creds[i] for i in range(n)
                                       if creds[i] > 0])) if creds else float("nan")
        rr = (np.mean(wins) / abs(np.mean(losses))) if wins and losses else None
        out.append({"cfg": cfg, "n": n, "ev": ev, "win": len(wins) / n,
                    "mdd": mdd, "pf": pf, "rr": rr, "theta_cap": theta_capture,
                    "beta": corr, "alpha": alpha, "ci": (lo, hi)})

    out.sort(key=lambda r: -r["ev"])
    print(f"{'type':<10}{'off':>6}{'w':>3}{'dte':>4}{'entry':>7}{'exit':>5}"
          f"{'n':>5}{'EV$':>8}{'win%':>6}{'PF':>6}{'thetaC':>8}{'beta':>7}"
          f"{'alpha$':>8}{'CI95':>18}")
    for r in out:
        t, off, w, dte, hhmm, ex = r["cfg"]
        print(f"{t:<10}{off:>6}{w:>3}{dte:>4}{hhmm:>7}{ex:>5}{r['n']:>5}"
              f"{r['ev']:>8.2f}{100*r['win']:>6.0f}"
              f"{(r['pf'] if r['pf'] else 0):>6.2f}{r['theta_cap']:>8.2f}"
              f"{(r['beta'] if r['beta']==r['beta'] else 0):>7.2f}"
              f"{(r['alpha'] if r['alpha']==r['alpha'] else 0):>8.2f}"
              f"  [{r['ci'][0]:.1f},{r['ci'][1]:.1f}]")

    pos = [r for r in out if r["ev"] > 0]
    ci_pos = [r for r in pos if r["ci"][0] > 0]
    regime_ind = [r for r in ci_pos if abs(r["beta"]) < 0.5 and r["alpha"] > 0]
    print(f"\nconfigs with EV>0: {len(pos)}/{len(out)}")
    print(f"  of those, 95% CI excludes 0: {len(ci_pos)}")
    print(f"  of those, |beta|<0.5 AND alpha>0: {len(regime_ind)}")
    print(f"\nPROVENANCE: {json.dumps(provenance(), indent=None)}")
    print(f"LIMITATION: {V3_LIMITATION}")
    Path("data").mkdir(exist_ok=True)
    with open("data/short_premium_landscape.json", "w") as f:
        json.dump({"provenance": provenance(), "limitation": V3_LIMITATION,
                   "configs": [{**r, "cfg": list(r["cfg"])} for r in out]},
                  f, indent=2, default=float)


if __name__ == "__main__":
    main()
