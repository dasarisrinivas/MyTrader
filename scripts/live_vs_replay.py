#!/usr/bin/env python3
"""Live-vs-replay divergence monitor — the operational-validation loop.

Reads logs/execution_quality.jsonl (real execution telemetry) and, for every
LIVE fill, replays the SAME contract at the SAME timestamps through the FROZEN
research engine. Reports divergence so a micro-live pilot is scientifically
useful rather than just "did it lose money".

Compares:
    entry fill px   vs replay entry (NBBO ask at submit)
    exit  fill px   vs replay exit  (NBBO bid at exit)
    realized $      vs replay $
    slippage vs the submit-time NBBO mid
    submit -> fill latency

DIVERGENCE = live materially worse (or better) than the replay model. If this
is large, the replay model is not predictive of live execution and every
research conclusion built on it needs re-examination BEFORE size increases.

Read-only. No trading, no production changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))
from shree.flow_research.thetadata import ThetaClient
from shree.research.replay_engine import QuoteCache, _prevailing

ET = ZoneInfo("America/New_York")
TELEM = "/Users/svss/Documents/code/ShreeBot/logs/execution_quality.jsonl"
# tolerances: flag when live deviates from replay by more than this
TOL_PRICE = 0.05      # $/share on a leg
TOL_PNL_PCT = 0.15    # 15% of replay P&L


def _load(path):
    ev = defaultdict(list)
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev[r.get("key")].append(r)
    except FileNotFoundError:
        return None
    return ev


def _et(iso):
    d = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(ET)


def _k(d):
    return d.replace(tzinfo=None).isoformat(timespec="milliseconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", default=TELEM)
    a = ap.parse_args()

    ev = _load(a.telemetry)
    if ev is None:
        print(f"no telemetry yet at {a.telemetry}")
        print("Expected — the pilot has not fired. CALL_SWEEP reaches HIGH/EXTREME")
        print("~2% of the time; this file appears on the first live fill.")
        return
    keys = [k for k, rows in ev.items()
            if any(r["event"] == "FILL" for r in rows)]
    if not keys:
        n = sum(len(v) for v in ev.values())
        print(f"telemetry has {n} events but ZERO fills — nothing to compare yet.")
        return

    client = ThetaClient()
    cache = QuoteCache(client)
    print(f"live fills to verify: {len(keys)}\n")
    print(f"{'key':<24}{'live_in':>9}{'rep_in':>8}{'d_in':>7}"
          f"{'live_out':>10}{'rep_out':>9}{'d_out':>8}{'live$':>9}{'rep$':>9}{'flag':>6}")
    agg = []
    for key in keys:
        rows = ev[key]
        sub = next((r for r in rows if r["event"] == "SUBMIT"), None)
        fil = next((r for r in rows if r["event"] == "FILL"), None)
        ex = next((r for r in rows if r["event"] == "EXIT_FILL"), None)
        if not sub or not fil:
            continue
        exp = str(sub.get("expiry") or "")
        strike = sub.get("strike")
        right = sub.get("right")
        if not exp or strike is None or not right:
            continue
        t_sub = _et(sub["ts_utc"])
        t_fill = _et(fil["ts_utc"])
        t_exit = _et(ex["ts_utc"]) if ex else None
        try:
            q, kk = cache.get(exp, float(strike), right[:1], t_sub.strftime("%Y%m%d"))
        except Exception as e:
            print(f"{key:<24} replay pull failed: {e}")
            continue
        if not q:
            print(f"{key:<24} no replay quotes")
            continue
        _, rep_in = _prevailing(q, kk, _k(t_sub))
        rep_out = _prevailing(q, kk, _k(t_exit))[0] if t_exit else None
        live_in = fil.get("avg_fill_price")
        live_out = ex.get("exit_price") if ex else None
        qty = fil.get("qty_filled") or 1
        live_pnl = ex.get("gross_usd") if ex else None
        rep_pnl = (None if (rep_in is None or rep_out is None)
                   else round((rep_out - rep_in) * 100.0 * qty - 1.30, 2))
        d_in = (None if (live_in is None or rep_in is None) else round(live_in - rep_in, 3))
        d_out = (None if (live_out is None or rep_out is None) else round(live_out - rep_out, 3))
        flag = ""
        if d_in is not None and abs(d_in) > TOL_PRICE:
            flag = "IN"
        if d_out is not None and abs(d_out) > TOL_PRICE:
            flag += "/OUT"
        if (live_pnl is not None and rep_pnl not in (None, 0)
                and abs(live_pnl - rep_pnl) > abs(rep_pnl) * TOL_PNL_PCT):
            flag += "/PNL"
        print(f"{key[:24]:<24}{(live_in or 0):>9.2f}{(rep_in or 0):>8.2f}"
              f"{(d_in if d_in is not None else 0):>7.3f}"
              f"{(live_out or 0):>10.2f}{(rep_out or 0):>9.2f}"
              f"{(d_out if d_out is not None else 0):>8.3f}"
              f"{(live_pnl or 0):>9.2f}{(rep_pnl or 0):>9.2f}{flag or 'ok':>6}")
        agg.append({"d_in": d_in, "d_out": d_out, "live": live_pnl, "rep": rep_pnl,
                    "slip_mid": fil.get("slippage_vs_mid"),
                    "lat": fil.get("latency_submit_to_fill_ms"),
                    "partial": fil.get("partial")})

    if not agg:
        return
    import statistics as st
    di = [x["d_in"] for x in agg if x["d_in"] is not None]
    sm = [x["slip_mid"] for x in agg if x["slip_mid"] is not None]
    lat = [x["lat"] for x in agg if x["lat"] is not None]
    lp = [x["live"] for x in agg if x["live"] is not None]
    rp = [x["rep"] for x in agg if x["rep"] is not None]
    print("\n=== DIVERGENCE SUMMARY ===")
    if di:
        print(f"  entry px live-vs-replay: median {st.median(di):+.3f}  "
              f"max |{max(abs(x) for x in di):.3f}|")
    if sm:
        print(f"  slippage vs submit-mid : median ${st.median(sm):+.3f}/share")
    if lat:
        print(f"  submit->fill latency   : median {st.median(lat):.0f} ms  "
              f"max {max(lat):.0f} ms")
    print(f"  partial fills          : {sum(1 for x in agg if x['partial'])}/{len(agg)}")
    if lp and rp and len(lp) == len(rp):
        print(f"  realized live P&L      : ${sum(lp):,.2f}")
        print(f"  replay-model P&L       : ${sum(rp):,.2f}")
        print(f"  DIVERGENCE             : ${sum(lp)-sum(rp):+,.2f}")
    print("\n  If divergence is large, the replay model does NOT predict live")
    print("  execution — investigate BEFORE any size increase.")


if __name__ == "__main__":
    main()
