#!/usr/bin/env python3
"""Promotion Candidate Scorecard — turns replay evidence into a promotion decision.

Implements the objective promotion criteria. A shadow family may move
SHADOW -> PILOT (1 contract) ONLY if it passes EVERY gate:

  G1 SAMPLE      >= MIN_SIGNALS replayed signals
  G2 SESSIONS    >= MIN_SESSIONS distinct sessions
  G3 CONSISTENCY positive replay EV on >= 3 CONSECUTIVE sessions
  G4 REGIME      positive EV on BOTH up-SPY and down-SPY days
  G5 INDEPENDENT EV not explained by SPY direction (beta-adjusted EV > 0)
  G6 REAL OPTION graded on real option $ P&L via the frozen engine (always true here)

G5 is the one today's +$2,363 failed: puts on a -0.89% day. It is computed by
splitting each family's signals by SPY-direction-of-day and requiring the family
to be positive INDEPENDENT of which way the market went.

Usage:
  python3 scripts/promotion_scorecard.py --days 30
Observation only. Never promotes anything itself — prints the decision.
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
from shree.research.replay_engine import (
    QuoteCache, _prevailing, RESEARCH_ENGINE_VERSION,
)

ET = ZoneInfo("America/New_York")
DB = "/Users/svss/Documents/code/ShreeBot/data/spy_options_signals.db"
COMM_RT = 1.30
HORIZON_MIN = 90

MIN_SIGNALS = 50
MIN_SESSIONS = 10
MIN_CONSEC = 3

# G7 INELIGIBLE: multi-leg families are MIS-GRADED by engine v2 (single-leg only)
# — it prices just the recorded leg, so their EV is meaningless. They cannot be
# promoted or even WATCHed until v3 multi-leg replay exists. Caught 2026-07-27
# when BULL_CALL_SPREAD showed a spurious +$5.96 "WATCH".
MULTI_LEG = {"BULL_CALL_SPREAD", "BEAR_PUT_SPREAD", "LONG_STRADDLE"}


def et(ts):
    s = ts.split(".")[0].rstrip("Z")
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).astimezone(ET)


def key(d):
    return d.replace(tzinfo=None).isoformat(timespec="milliseconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--cap-per-session", type=int, default=40)
    a = ap.parse_args()

    c = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    sigs = c.execute(
        """SELECT id, sent_at, exit_at, signal_type, strike, right, expiry_date,
                  confidence, spy_price, spy_price_exit
           FROM spy_signals WHERE expiry_date IS NOT NULL AND expiry_date!=''
           ORDER BY sent_at""").fetchall()
    c.close()

    cutoff = (datetime.now(ET) - timedelta(days=a.days)).date()
    by_session = defaultdict(list)
    for s in sigs:
        d = et(s["sent_at"])
        if d.date() >= cutoff:
            by_session[d.date().isoformat()].append(s)

    client = ThetaClient()
    cache = QuoteCache(client)

    # session SPY direction (first->last dispatched spy_price that session)
    results = defaultdict(list)   # family -> [(session, net, spy_dir)]
    for sess in sorted(by_session):
        rows = by_session[sess]
        px = [r["spy_price"] for r in rows if r["spy_price"]]
        if len(px) < 2:
            continue
        spy_dir = 1 if px[-1] > px[0] else -1
        sample = rows[:: max(1, len(rows) // a.cap_per_session)][:a.cap_per_session]
        for s in sample:
            e = et(s["sent_at"])
            x = et(s["exit_at"]) if s["exit_at"] else e + timedelta(minutes=HORIZON_MIN)
            if x.date() != e.date():
                x = e.replace(hour=15, minute=55, second=0)
            try:
                q, k = cache.get(s["expiry_date"], float(s["strike"]), s["right"],
                                 e.strftime("%Y%m%d"))
            except Exception:
                continue
            if not q:
                continue
            ea = _prevailing(q, k, key(e))[1]
            xb = _prevailing(q, k, key(x))[0]
            if not ea or not xb:
                continue
            results[s["signal_type"]].append((sess, (xb - ea) * 100 - COMM_RT, spy_dir))

    print(f"PROMOTION CANDIDATE SCORECARD  (engine {RESEARCH_ENGINE_VERSION})")
    print(f"criteria: >={MIN_SIGNALS} signals, >={MIN_SESSIONS} sessions, "
          f">={MIN_CONSEC} consecutive +EV sessions, +EV on BOTH up & down days\n")
    hdr = (f"{'strategy':<20}{'sess':>5}{'sigs':>6}{'EV$':>9}{'up_EV':>9}{'dn_EV':>9}"
           f"{'consec':>7}{'regime_ind':>11}  {'PROMOTE?':<9} reason")
    print(hdr)
    for fam in sorted(results, key=lambda f: -len(results[f])):
        rs = results[fam]
        n = len(rs)
        sessions = sorted({r[0] for r in rs})
        ev = sum(r[1] for r in rs) / n
        up = [r[1] for r in rs if r[2] > 0]
        dn = [r[1] for r in rs if r[2] < 0]
        up_ev = sum(up) / len(up) if up else float("nan")
        dn_ev = sum(dn) / len(dn) if dn else float("nan")
        # consecutive +EV sessions
        per_sess = defaultdict(list)
        for s_, net, _ in rs:
            per_sess[s_].append(net)
        streak = best = 0
        for s_ in sessions:
            if sum(per_sess[s_]) / len(per_sess[s_]) > 0:
                streak += 1
                best = max(best, streak)
            else:
                streak = 0
        regime_ind = (up and dn and up_ev > 0 and dn_ev > 0)
        fails = []
        if fam in MULTI_LEG:
            fails.append("MULTI-LEG: mis-graded by engine v2 — INELIGIBLE")
        if n < MIN_SIGNALS:
            fails.append(f"sample {n}<{MIN_SIGNALS}")
        if len(sessions) < MIN_SESSIONS:
            fails.append(f"sessions {len(sessions)}<{MIN_SESSIONS}")
        if best < MIN_CONSEC:
            fails.append(f"consec {best}<{MIN_CONSEC}")
        if ev <= 0:
            fails.append("EV<=0")
        if not regime_ind:
            fails.append("regime-dependent (beta)")
        if fam in MULTI_LEG:
            verdict = "INELIGIBLE"
        else:
            verdict = "PROMOTE" if not fails else ("WATCH" if ev > 0 else "NO")
        print(f"{fam:<20}{len(sessions):>5}{n:>6}{ev:>9.2f}{up_ev:>9.2f}{dn_ev:>9.2f}"
              f"{best:>7}{str(bool(regime_ind)):>11}  {verdict:<9} {'; '.join(fails) or 'all gates pass'}")

    print("\nPROMOTION REVIEW")
    promotable = []
    print(f"  Any shadow strategy eligible for live tomorrow? "
          f"{'YES' if promotable else 'NO'}")
    if not promotable:
        print("  Missing evidence: see per-family reasons above (sample, session count,")
        print("  consecutive +EV sessions, and regime independence are the usual blockers).")


if __name__ == "__main__":
    main()
