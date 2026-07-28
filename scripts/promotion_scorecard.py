#!/usr/bin/env python3
"""Promotion Candidate Scorecard v3 — deterministic, versioned, reproducible.

Enforces the frozen constitution in shree/research/promotion_constitution.py.
Emits a provenance block (engine version + file hashes + dataset hash + framework
version) so any verdict is reproducible years later.

INVARIANT (G8/G9): promotion decisions require 100% replay completeness.
Sampling (--cap-per-session) is EXPLORATORY ONLY and automatically marks the run
non-binding — every family is forced ineligible in that mode.

  python3 scripts/promotion_scorecard.py --days 120            # binding (no cap)
  python3 scripts/promotion_scorecard.py --days 120 --cap 25   # exploratory only
"""
from __future__ import annotations

import argparse
import hashlib
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
from shree.research import promotion_constitution as K

ET = ZoneInfo("America/New_York")
DB = "/Users/svss/Documents/code/ShreeBot/data/spy_options_signals.db"
COMM_RT = 1.30
HORIZON_MIN = 90
RNG = np.random.default_rng(20260727)


def et(ts):
    s = ts.split(".")[0].rstrip("Z")
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).astimezone(ET)


def key(d):
    return d.replace(tzinfo=None).isoformat(timespec="milliseconds")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


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
    m = RNG.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def classify_session(spy_ret, vix):
    regs = []
    if spy_ret > K.BULL_BEAR_THRESHOLD:
        regs.append("bull")
    elif spy_ret < -K.BULL_BEAR_THRESHOLD:
        regs.append("bear")
    else:
        regs.append("range")
    if vix is not None:
        regs.append("high_vix" if vix >= K.HIGH_VIX_THRESHOLD else "low_vix")
    return regs


def replay_all(days, cap):
    c = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    sigs = c.execute(
        """SELECT id, sent_at, exit_at, signal_type, strike, right, expiry_date,
                  confidence, spy_price, vix, volume, open_interest
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
    attempted = replayed = 0
    ids = []
    unreplayable = []   # v2.1: every failure itemized, none silently dropped
    for sess in sorted(by_sess):
        rows = by_sess[sess]
        px = [r["spy_price"] for r in rows if r["spy_price"]]
        # v2.1 DENOMINATOR FIX: a session with <2 price points is NOT skipped —
        # its signals are still counted and replayed. spy_ret is unavailable, so
        # those trades are excluded from beta/regime math only (spy_ret=None).
        if len(px) >= 2:
            spy_ret = (px[-1] - px[0]) / px[0]
            vixv = [r["vix"] for r in rows if r["vix"]]
            vix = sum(vixv) / len(vixv) if vixv else None
            regimes = classify_session(spy_ret, vix)
        else:
            spy_ret, regimes = None, []
        sample = rows if not cap else rows[:: max(1, len(rows) // cap)][:cap]
        for s in sample:
            attempted += 1
            e = et(s["sent_at"])
            x = et(s["exit_at"]) if s["exit_at"] else e + timedelta(minutes=HORIZON_MIN)
            if x.date() != e.date():
                x = e.replace(hour=15, minute=55, second=0)
            reason = None
            q = k = None
            try:
                q, k = cache.get(s["expiry_date"], float(s["strike"]), s["right"],
                                 e.strftime("%Y%m%d"))
            except Exception as exc:
                reason = f"vendor_error:{type(exc).__name__}"
            if reason is None and not q:
                reason = "no_quote_rows"
            ea = xb = None
            if reason is None:
                ea = _prevailing(q, k, key(e))[1]
                xb = _prevailing(q, k, key(x))[0]
                if not ea:
                    reason = "no_nbbo_at_entry"
                elif not xb:
                    reason = "no_nbbo_at_exit"
            if reason:
                unreplayable.append({
                    "id": s["id"], "session": sess, "signal_type": s["signal_type"],
                    "contract": f"{s['expiry_date']} {s['strike']}{s['right']}",
                    "volume": s["volume"], "open_interest": s["open_interest"],
                    "reason": reason})
                continue
            replayed += 1
            ids.append(s["id"])
            out[s["signal_type"]].append({
                "session": sess, "date": e.date(), "net": (xb - ea) * 100 - COMM_RT,
                "spy_ret": spy_ret, "regimes": regimes,
                "hold": (x - e).total_seconds() / 60.0})
    dataset_hash = hashlib.sha256(
        ",".join(str(i) for i in sorted(ids)).encode()).hexdigest()[:16]
    return out, attempted, replayed, dataset_hash, unreplayable


def window_ev(trades, days):
    cut = (datetime.now(ET) - timedelta(days=days)).date()
    v = [t["net"] for t in trades if t["date"] >= cut]
    return sum(v) / len(v) if v else float("nan")


def evaluate(fam, trades, completeness, binding):
    n = len(trades)
    nets = [t["net"] for t in trades]
    sessions = sorted({t["session"] for t in trades})
    ev = sum(nets) / n if n else float("nan")
    wins = sum(1 for x in nets if x > 0)

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

    wev = {f"{d}d": window_ev(trades, d) for d in K.G5_WINDOWS}
    wev["lifetime"] = ev
    lo, hi = boot_ci(nets)
    wlo = wilson_lo(wins, n)

    # v2.1: trades whose session lacked a computable SPY return (spy_ret None)
    # are counted in n/EV but excluded from beta/regime math.
    beta_t = [t for t in trades if t["spy_ret"] is not None]
    up = [t["net"] for t in beta_t if t["spy_ret"] > K.BULL_BEAR_THRESHOLD]
    dn = [t["net"] for t in beta_t if t["spy_ret"] < -K.BULL_BEAR_THRESHOLD]
    fl = [t["net"] for t in beta_t if abs(t["spy_ret"]) <= K.BULL_BEAR_THRESHOLD]
    f = lambda v: (sum(v) / len(v)) if v else float("nan")
    up_ev, dn_ev, fl_ev = f(up), f(dn), f(fl)

    corr = slope = alpha = alpha_sharpe = float("nan")
    if len(beta_t) >= 5:
        x = np.array([t["spy_ret"] for t in beta_t], float)
        y = np.array([t["net"] for t in beta_t], float)
        if x.std() > 0 and y.std() > 0:
            corr = float(np.corrcoef(x, y)[0, 1])
            slope, icept = np.polyfit(x, y, 1)
            slope = float(slope)
            resid = y - (slope * x)
            alpha = float(resid.mean())
            if resid.std() > 0:
                alpha_sharpe = float(resid.mean() / resid.std())

    covered = set()
    for t in trades:
        covered.update(t["regimes"])
    missing_regimes = [r for r in K.REQUIRED_REGIMES if r not in covered]

    failed = []
    if fam in K.G10_MULTI_LEG:
        failed.append("G10_ineligible_multi_leg")
    if not binding:
        failed.append("G9_non_deterministic_sampled_run")
    if completeness < K.G8_MIN_REPLAY_COMPLETENESS:
        failed.append("G8_incomplete_replay")
    if n < K.G1_MIN_SIGNALS:
        failed.append("G1_insufficient_sample")
    if not (ev > K.G2_MIN_EV):
        failed.append("G2_negative_ev")
    if not (lo == lo and lo > K.G3_CI_LOWER_ABOVE):
        failed.append("G3_ev_ci_crosses_zero")
    if wlo <= K.G4_MIN_WILSON_WR:
        failed.append("G4_win_rate_ci_low")
    if any(not (v > 0) for k_, v in wev.items() if k_ != "lifetime" and v == v):
        failed.append("G5_window_disagreement")
    if corr == corr and abs(corr) >= K.G6_MAX_BETA_CORR:
        failed.append("G6_beta_correlated")
    if slope == slope and abs(slope) >= K.G6_MAX_BETA_SLOPE:
        failed.append("G6_beta_slope")
    if not (alpha == alpha and alpha > K.G7_MIN_RESIDUAL_ALPHA):
        failed.append("G7_no_residual_alpha")
    if len(sessions) < K.MIN_COMPLETE_SESSIONS:
        failed.append("COVERAGE_insufficient_sessions")
    if missing_regimes:
        failed.append("COVERAGE_missing_regimes:" + "|".join(missing_regimes))

    if fam in K.G10_MULTI_LEG:
        stage = "ineligible"
    elif not failed:
        stage = "candidate"
    elif ev == ev and ev <= 0 and n >= K.G1_MIN_SIGNALS:
        stage = "archived"
    else:
        stage = "shadow"

    return {
        "strategy": fam, "eligible": (not failed), "stage": stage, "failed": failed,
        "promotion_metrics": {
            "n": n, "sessions": len(sessions),
            "ev": round(ev, 2) if ev == ev else None,
            "ev_ci95": [round(lo, 2) if lo == lo else None,
                        round(hi, 2) if hi == hi else None],
            "wilson_lo": round(wlo, 3),
            "window_ev": {k_: (round(v, 2) if v == v else None) for k_, v in wev.items()},
            "up_ev": round(up_ev, 2) if up_ev == up_ev else None,
            "dn_ev": round(dn_ev, 2) if dn_ev == dn_ev else None,
            "beta_corr": round(corr, 3) if corr == corr else None,
            "beta_slope": round(slope, 1) if slope == slope else None,
            "alpha_residual_ev": round(alpha, 2) if alpha == alpha else None,
            "replay_completeness": round(completeness, 4),
            "regime_coverage": sorted(covered),
        },
        "exploratory_metrics": {
            "flat_ev": round(fl_ev, 2) if fl_ev == fl_ev else None,
            "alpha_residual_sharpe": round(alpha_sharpe, 3) if alpha_sharpe == alpha_sharpe else None,
            "consec_pos_sessions": best,
            "win_rate": round(wins / n, 3) if n else None,
            "avg_hold_min": round(sum(t["hold"] for t in trades) / n, 1) if n else None,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--cap", type=int, default=None,
                    help="EXPLORATORY ONLY — sampling voids binding promotion")
    ap.add_argument("--json", default="data/promotion_scorecard.json")
    a = ap.parse_args()

    binding = a.cap is None
    trades, attempted, replayed, dhash, unreplayable = replay_all(a.days, a.cap)
    completeness = (replayed / attempted) if attempted else 0.0

    results = [evaluate(f, t, completeness, binding)
               for f, t in sorted(trades.items(), key=lambda kv: -len(kv[1]))]

    prov = {
        "replay_engine_version": RESEARCH_ENGINE_VERSION,
        "replay_engine_hash": sha(Path(__file__).parent.parent /
                                  "shree/research/replay_engine.py"),
        "promotion_framework_version": K.PROMOTION_FRAMEWORK_VERSION,
        "constitution_hash": sha(Path(__file__).parent.parent /
                                 "shree/research/promotion_constitution.py"),
        "scorecard_hash": sha(__file__),
        "dataset_hash": dhash,
        "lookback_days": a.days,
        "signals_attempted": attempted, "signals_replayed": replayed,
        "replay_completeness": round(completeness, 4),
        "binding": binding,
    }

    print(K.gate_summary())
    print(f"\nPROVENANCE  engine={prov['replay_engine_version']}"
          f"({prov['replay_engine_hash']}) framework=v{prov['promotion_framework_version']}"
          f"({prov['constitution_hash']}) dataset={dhash}")
    print(f"  replay completeness {completeness:.1%} "
          f"({replayed}/{attempted})  BINDING={binding}")
    if not binding:
        print("  ** SAMPLED RUN — EXPLORATORY ONLY, cannot promote anything (G9) **")

    print(f"\n{'strategy':<20}{'stage':<11}{'n':>5}{'sess':>5}{'EV$':>8}"
          f"{'CI95':>18}{'beta_r':>8}{'slope':>9}{'alpha$':>8}")
    for r in results:
        m = r["promotion_metrics"]
        ci = f"[{m['ev_ci95'][0]},{m['ev_ci95'][1]}]"
        print(f"{r['strategy']:<20}{r['stage']:<11}{m['n']:>5}{m['sessions']:>5}"
              f"{(m['ev'] or 0):>8.2f}{ci:>18}{(m['beta_corr'] or 0):>8.2f}"
              f"{(m['beta_slope'] or 0):>9.0f}{(m['alpha_residual_ev'] or 0):>8.2f}")

    print("\nFAILED GATES (promotion):")
    for r in results:
        print(f"  {r['strategy']:<20} {', '.join(r['failed']) or 'NONE — all gates pass'}")

    print("\nEXPLORATORY (interesting, NOT actionable):")
    for r in results:
        e = r["exploratory_metrics"]
        print(f"  {r['strategy']:<20} alpha_sharpe={e['alpha_residual_sharpe']} "
              f"flat_ev={e['flat_ev']} consec={e['consec_pos_sessions']} "
              f"wr={e['win_rate']}")

    # v2.1 MANDATORY ITEMIZATION — every unreplayable signal, with bias note
    print(f"\nUNREPLAYABLE SIGNALS (v2.1 mandatory itemization): {len(unreplayable)}")
    for u in unreplayable:
        print(f"  id={u['id']} {u['session']} {u['signal_type']} {u['contract']} "
              f"vol={u['volume']} oi={u['open_interest']} reason={u['reason']}")
    if unreplayable:
        print("  BIAS: unreplayable contracts skew illiquid/never-traded; excluding")
        print("  them biases EV UPWARD.")

    elig = [r for r in results if r["eligible"]]
    print("\nPROMOTION REVIEW")
    print(f"  Any strategy eligible for live tomorrow? {'YES' if elig else 'NO'}")
    if not elig:
        print("  All strategies fail >=1 gate. See FAILED GATES above.")

    out = {"provenance": prov, "constitution": {
        "version": K.PROMOTION_FRAMEWORK_VERSION, "effective": K.EFFECTIVE_DATE,
        "min_signals": K.G1_MIN_SIGNALS, "min_sessions": K.MIN_COMPLETE_SESSIONS,
        "required_regimes": list(K.REQUIRED_REGIMES),
        "max_beta_corr": K.G6_MAX_BETA_CORR,
    }, "results": results, "any_eligible": bool(elig),
        "unreplayable_signals": unreplayable}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nmachine-readable -> {a.json}")


if __name__ == "__main__":
    main()
