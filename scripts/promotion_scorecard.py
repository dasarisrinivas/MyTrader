#!/usr/bin/env python3
"""Promotion Candidate Scorecard v2 — evidence -> machine-enforceable decision.

Replays every dispatched signal ONCE through the frozen engine, then evaluates
each family against objective gates and emits a structured JSON verdict that
production can consume directly (no human reading a report).

GATES
  G1 SAMPLE        >= MIN_SIGNALS replayed signals
  G2 SESSIONS      >= MIN_SESSIONS distinct sessions
  G3 CONSISTENCY   >= MIN_CONSEC consecutive +EV sessions
  G4 MULTI-WINDOW  +EV at 10d, 30d, 90d; non-negative lifetime
  G5 STAT-CONF     bootstrap 95% CI of EV excludes 0; Wilson-lo(win) > 0.40
  G6 ALPHA vs BETA measurable: +EV on up AND down days, |corr(net, SPY ret)|
                   below MAX_BETA_CORR, and direction-residualized alpha > 0
  G7 ELIGIBILITY   multi-leg families are INELIGIBLE (engine v2 grades single-leg)

STAGES  research -> shadow -> candidate -> pilot -> active
        archived (evidence negative) / ineligible (engine limitation)

Observation only. Promotes nothing; reports the decision.
  python3 scripts/promotion_scorecard.py --days 120 --json data/promotion_scorecard.json
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

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
MIN_WILSON_WR = 0.40
MAX_BETA_CORR = 0.50
WINDOWS = (10, 30, 90)

# Engine v2 prices only the recorded leg -> multi-leg EV is meaningless.
MULTI_LEG = {"BULL_CALL_SPREAD", "BEAR_PUT_SPREAD", "LONG_STRADDLE"}

RNG = np.random.default_rng(20260727)


def et(ts):
    s = ts.split(".")[0].rstrip("Z")
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).astimezone(ET)


def key(d):
    return d.replace(tzinfo=None).isoformat(timespec="milliseconds")


def wilson_lo(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - m) / d)


def boot_ci(x, n=3000):
    x = np.asarray(x, float)
    if len(x) < 5:
        return float("nan"), float("nan")
    means = RNG.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def replay_all(days, cap):
    """Replay once over the full lookback; return family -> list of trade dicts."""
    c = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    sigs = c.execute(
        """SELECT id, sent_at, exit_at, signal_type, strike, right, expiry_date,
                  confidence, spy_price, spy_price_exit
           FROM spy_signals WHERE expiry_date IS NOT NULL AND expiry_date!=''
           ORDER BY sent_at""").fetchall()
    c.close()
    cutoff = (datetime.now(ET) - timedelta(days=days)).date()
    by_sess = defaultdict(list)
    for s in sigs:
        d = et(s["sent_at"])
        if d.date() >= cutoff:
            by_sess[d.date().isoformat()].append(s)

    client = ThetaClient()
    cache = QuoteCache(client)
    out = defaultdict(list)
    for sess in sorted(by_sess):
        rows = by_sess[sess]
        px = [r["spy_price"] for r in rows if r["spy_price"]]
        if len(px) < 2:
            continue
        spy_ret = (px[-1] - px[0]) / px[0]
        sample = rows[:: max(1, len(rows) // cap)][:cap]
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
            out[s["signal_type"]].append({
                "session": sess, "date": e.date(), "net": (xb - ea) * 100 - COMM_RT,
                "spy_ret": spy_ret, "conf": s["confidence"] or 0.0})
    return out


def window_ev(trades, days):
    cut = (datetime.now(ET) - timedelta(days=days)).date()
    v = [t["net"] for t in trades if t["date"] >= cut]
    return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)


def evaluate(fam, trades):
    n = len(trades)
    nets = [t["net"] for t in trades]
    sessions = sorted({t["session"] for t in trades})
    ev = sum(nets) / n if n else float("nan")
    wins = sum(1 for x in nets if x > 0)

    # G3 consecutive +EV sessions
    per = defaultdict(list)
    for t in trades:
        per[t["session"]].append(t["net"])
    streak = best = 0
    for s in sessions:
        if sum(per[s]) / len(per[s]) > 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0

    # G4 multi-window
    win_ev = {f"{d}d": window_ev(trades, d)[0] for d in WINDOWS}
    win_ev["lifetime"] = ev

    # G5 statistical confidence
    lo, hi = boot_ci(nets)
    wlo = wilson_lo(wins, n)

    # G6 alpha vs beta (measurable)
    up = [t["net"] for t in trades if t["spy_ret"] > 0]
    dn = [t["net"] for t in trades if t["spy_ret"] < 0]
    up_ev = sum(up) / len(up) if up else float("nan")
    dn_ev = sum(dn) / len(dn) if dn else float("nan")
    corr = float("nan")
    alpha = float("nan")
    if n >= 5:
        x = np.array([t["spy_ret"] for t in trades], float)
        y = np.array(nets, float)
        if x.std() > 0 and y.std() > 0:
            corr = float(np.corrcoef(x, y)[0, 1])
            b, a = np.polyfit(x, y, 1)          # net ~ a + b*spy_ret
            alpha = float(np.mean(y - (b * x)))  # direction-residualized EV

    failed = []
    if fam in MULTI_LEG:
        failed.append("ineligible_multi_leg")
    if n < MIN_SIGNALS:
        failed.append("insufficient_sample")
    if len(sessions) < MIN_SESSIONS:
        failed.append("insufficient_sessions")
    if best < MIN_CONSEC:
        failed.append("insufficient_consecutive_sessions")
    if not (ev > 0):
        failed.append("negative_ev")
    if any(not (v > 0) for k, v in win_ev.items() if k != "lifetime" and win_ev[k] == win_ev[k]):
        failed.append("window_disagreement")
    if not (lo == lo and lo > 0):
        failed.append("ev_ci_crosses_zero")
    if wlo < MIN_WILSON_WR:
        failed.append("win_rate_ci_low")
    if not (up_ev > 0 and dn_ev > 0):
        failed.append("regime_dependence")
    if corr == corr and abs(corr) > MAX_BETA_CORR:
        failed.append("beta_correlated")
    if not (alpha == alpha and alpha > 0):
        failed.append("no_residual_alpha")

    if fam in MULTI_LEG:
        stage = "ineligible"
    elif not failed:
        stage = "candidate"      # meets evidence bar -> eligible for pilot review
    elif ev == ev and ev <= 0 and n >= MIN_SIGNALS:
        stage = "archived"       # enough evidence, negative
    else:
        stage = "shadow"         # still accumulating

    return {
        "strategy": fam, "eligible": not failed and fam not in MULTI_LEG,
        "stage": stage, "failed": failed,
        "n": n, "sessions": len(sessions),
        "ev": round(ev, 2) if ev == ev else None,
        "ev_ci95": [round(lo, 2) if lo == lo else None,
                    round(hi, 2) if hi == hi else None],
        "win_rate": round(wins / n, 3) if n else None,
        "wilson_lo": round(wlo, 3),
        "consec_pos_sessions": best,
        "window_ev": {k: (round(v, 2) if v == v else None) for k, v in win_ev.items()},
        "up_ev": round(up_ev, 2) if up_ev == up_ev else None,
        "dn_ev": round(dn_ev, 2) if dn_ev == dn_ev else None,
        "beta_corr": round(corr, 3) if corr == corr else None,
        "alpha_residual_ev": round(alpha, 2) if alpha == alpha else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--cap-per-session", type=int, default=30)
    ap.add_argument("--json", default="data/promotion_scorecard.json")
    a = ap.parse_args()

    trades = replay_all(a.days, a.cap_per_session)
    results = [evaluate(f, t) for f, t in sorted(trades.items(), key=lambda kv: -len(kv[1]))]

    print(f"PROMOTION SCORECARD v2  (engine {RESEARCH_ENGINE_VERSION})")
    print(f"gates: n>={MIN_SIGNALS}, sessions>={MIN_SESSIONS}, consec>={MIN_CONSEC}, "
          f"multi-window +EV, CI excludes 0, Wilson>{MIN_WILSON_WR}, "
          f"|beta_corr|<{MAX_BETA_CORR}, alpha>0\n")
    print(f"{'strategy':<20}{'stage':<11}{'n':>5}{'EV$':>8}{'CI95':>18}"
          f"{'10d':>8}{'30d':>8}{'90d':>8}{'beta_r':>8}{'alpha$':>8}")
    for r in results:
        ci = f"[{r['ev_ci95'][0]},{r['ev_ci95'][1]}]"
        w = r["window_ev"]
        print(f"{r['strategy']:<20}{r['stage']:<11}{r['n']:>5}"
              f"{(r['ev'] if r['ev'] is not None else 0):>8.2f}{ci:>18}"
              f"{(w.get('10d') or 0):>8.2f}{(w.get('30d') or 0):>8.2f}"
              f"{(w.get('90d') or 0):>8.2f}"
              f"{(r['beta_corr'] if r['beta_corr'] is not None else 0):>8.2f}"
              f"{(r['alpha_residual_ev'] if r['alpha_residual_ev'] is not None else 0):>8.2f}")
    for r in results:
        print(f"  {r['strategy']:<20} failed: {', '.join(r['failed']) or 'none'}")

    elig = [r for r in results if r["eligible"]]
    print("\nPROMOTION REVIEW")
    print(f"  Any shadow strategy eligible for live tomorrow? {'YES' if elig else 'NO'}")
    if elig:
        for r in elig:
            print(f"    {r['strategy']}: EV ${r['ev']} CI {r['ev_ci95']} alpha ${r['alpha_residual_ev']}")
    else:
        print("  Missing evidence: see per-strategy failed[] above.")

    out = {"engine": RESEARCH_ENGINE_VERSION,
           "generated_for_lookback_days": a.days,
           "gates": {"min_signals": MIN_SIGNALS, "min_sessions": MIN_SESSIONS,
                     "min_consec": MIN_CONSEC, "min_wilson_wr": MIN_WILSON_WR,
                     "max_beta_corr": MAX_BETA_CORR, "windows": list(WINDOWS)},
           "results": results,
           "any_eligible": bool(elig)}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nmachine-readable -> {a.json}")


if __name__ == "__main__":
    main()
