#!/usr/bin/env python3
"""NEW ALPHA #2 — holding-horizon research (EXPLORATORY, not a promotion run).

Every prior evaluation held positions intraday (production exit, or a 90-min
stop). Holding period has NEVER been varied. This keeps the SIGNAL fixed and
varies ONLY the exit horizon:

    30m · 90m · EOD · +1 session · +3 sessions · +5 sessions

Question: does any holding horizon change the sign of expectancy?

Uses the FROZEN replay engine's components (QuoteCache, _prevailing) unchanged —
the engine itself is not modified. Sampling is permitted because this is
exploratory (constitution G9: sampled runs can never promote).

Expiry handling: if the horizon lands past expiration, the option is settled at
INTRINSIC value using the SPY close on the expiry date (ThetaData stock EOD),
not marked to a quote that no longer exists.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))
from shree.flow_research.thetadata import ThetaClient
from shree.research.replay_engine import QuoteCache, _prevailing

ET = ZoneInfo("America/New_York")
DB = "/Users/svss/Documents/code/ShreeBot/data/spy_options_signals.db"
COMM_RT = 1.30
HORIZONS = [("30m", 30), ("90m", 90), ("EOD", None),
            ("+1sess", 1), ("+3sess", 3), ("+5sess", 5)]


def et(t):
    s = t.split(".")[0].rstrip("Z")
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).astimezone(ET)


def key(d):
    return d.replace(tzinfo=None).isoformat(timespec="milliseconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--days", type=int, default=120)
    a = ap.parse_args()

    c = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    sigs = c.execute(
        """SELECT id, sent_at, signal_type, strike, right, expiry_date, dte
           FROM spy_signals WHERE expiry_date IS NOT NULL AND expiry_date!=''
           ORDER BY sent_at""").fetchall()
    c.close()
    cut = (datetime.now(ET) - timedelta(days=a.days)).date()
    sigs = [s for s in sigs if et(s["sent_at"]).date() >= cut]
    if len(sigs) > a.limit:
        sigs = sigs[:: max(1, len(sigs) // a.limit)][:a.limit]

    client = ThetaClient()
    cache = QuoteCache(client)
    # trading calendar from listed expirations (daily SPY expiries ~ trading days)
    cal = sorted({d for d in client.list_expirations("SPY")})
    spy_close = {}

    def close_on(d):
        if d not in spy_close:
            try:
                m = client.stock_eod_close("SPY", d, d)
                spy_close[d] = m.get(d)
            except Exception:
                spy_close[d] = None
        return spy_close[d]

    def next_sessions(d, k):
        fut = [x for x in cal if x > d]
        return fut[k - 1] if len(fut) >= k else None

    res = defaultdict(lambda: defaultdict(list))   # horizon -> family -> [net]
    expired_ct = defaultdict(int)
    for s in sigs:
        e = et(s["sent_at"])
        exp_d = datetime.strptime(s["expiry_date"], "%Y%m%d").date()
        try:
            q, k = cache.get(s["expiry_date"], float(s["strike"]), s["right"],
                             e.strftime("%Y%m%d"))
        except Exception:
            continue
        if not q:
            continue
        ea = _prevailing(q, k, key(e))[1]
        if not ea:
            continue
        for label, h in HORIZONS:
            if label in ("30m", "90m"):
                x = e + timedelta(minutes=h)
                if x.time() > datetime(2000, 1, 1, 15, 55).time():
                    x = e.replace(hour=15, minute=55, second=0)
                xd = x.date()
            elif label == "EOD":
                x = e.replace(hour=15, minute=55, second=0)
                xd = x.date()
            else:
                xd = next_sessions(e.date(), h)
                if xd is None:
                    continue
                x = datetime.combine(xd, datetime(2000, 1, 1, 15, 55).time(),
                                     tzinfo=ET)
            # settle at intrinsic if horizon lands past expiration
            if xd > exp_d:
                sc = close_on(exp_d)
                if sc is None:
                    continue
                intr = max(0.0, sc - float(s["strike"])) if s["right"] == "C" \
                    else max(0.0, float(s["strike"]) - sc)
                net = (intr - ea) * 100 - (COMM_RT / 2)   # no exit commission
                expired_ct[label] += 1
            else:
                try:
                    q2, k2 = cache.get(s["expiry_date"], float(s["strike"]),
                                       s["right"], xd.strftime("%Y%m%d"))
                except Exception:
                    continue
                if not q2:
                    continue
                xb = _prevailing(q2, k2, key(x))[0]
                if not xb:
                    continue
                net = (xb - ea) * 100 - COMM_RT
            res[label][s["signal_type"]].append(net)

    print(f"HOLDING-HORIZON RESEARCH (EXPLORATORY — cannot promote, G9)")
    print(f"signals sampled: {len(sigs)}  (signal fixed; only exit horizon varies)\n")
    fams = sorted({f for h in res for f in res[h]})
    print(f"{'horizon':<9}{'n':>6}{'EV$':>9}{'win%':>7}  " +
          "".join(f"{f[:11]:>12}" for f in fams))
    for label, _ in HORIZONS:
        if label not in res:
            continue
        allv = [v for f in res[label] for v in res[label][f]]
        if not allv:
            continue
        w = sum(1 for x in allv if x > 0)
        row = f"{label:<9}{len(allv):>6}{sum(allv)/len(allv):>9.2f}{100*w/len(allv):>7.0f}  "
        for f in fams:
            v = res[label].get(f, [])
            row += f"{(sum(v)/len(v) if v else 0):>12.1f}"
        print(row)
    print("\nexpired-at-intrinsic counts:", dict(expired_ct))
    print("\nREAD: if every horizon is negative, holding period is NOT the lever.")


if __name__ == "__main__":
    main()
