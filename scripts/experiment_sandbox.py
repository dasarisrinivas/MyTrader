#!/usr/bin/env python3
"""
Safe experiment sandbox — MES Phase 1 (ISOLATION ONLY).

Simulates the REJECTED experimental filters (ADX≥22, RTH-only, OR_BREAK removal,
combined) as post-hoc scenarios on a COPY of a backtest trades CSV.

Isolation guarantees:
  • Copy-on-read: the trades CSV is loaded read-only into memory; nothing is
    written back. No production DB, learning store, or live state is touched.
  • No import of any execution/live module — pure pandas analysis.
  • Output goes to stdout (and an optional --out file in /tmp), never to
    production stores.

This lets you re-examine the rejected scenarios any time WITHOUT risk to the
locked Phase 1 production baseline.

Usage:
  python3 scripts/experiment_sandbox.py <trades.csv>
"""
import sys, os, ast
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import pandas as pd
import numpy as np


def _pf(s):
    gp = s[s > 0].sum(); gl = -s[s <= 0].sum()
    return round(gp / gl, 2) if gl > 0 else float("inf")


def _metrics(s):
    n = len(s)
    eq = s.cumsum(); dd = round((eq - eq.cummax()).min(), 1) if n else 0.0
    sh = round(s.mean() / s.std() * np.sqrt(n), 2) if n > 1 and s.std() > 0 else 0.0
    return dict(n=n, pnl=round(s.sum(), 1), pf=_pf(s),
                wr=round(100 * (s > 0).mean()) if n else 0, dd=dd, sharpe=sh)


def _bucketize_session(ts):
    try:
        from shree.trading_manager.learning import bucketize_time
        return bucketize_time(ts)
    except Exception:
        return "NA"


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    # copy-on-read: load a private in-memory copy
    d = pd.read_csv(sys.argv[1]).copy()
    d["m"] = d["entry_metadata"].apply(lambda r: ast.literal_eval(r) if isinstance(r, str) else {})
    d["setup"] = d["m"].apply(lambda m: (m.get("signal_reason", "") or "").split("|")[0].strip())
    d["adx"] = d["m"].apply(lambda m: float(m.get("adx_value", 20) or 20))
    d["tod"] = d["entry_time"].apply(_bucketize_session)

    scenarios = {
        "BASELINE (production)": d,
        "ADX>=22": d[d["adx"] >= 22],
        "RTH_OPEN/MID only": d[d["tod"].isin(["RTH_OPEN", "RTH_MID"])],
        "remove OR_BREAK": d[d["setup"] != "OR_BREAK_LONG"],
        "COMBINED (ADX>=22 + RTH + no ORB)": d[(d["adx"] >= 22)
                                               & (d["tod"].isin(["RTH_OPEN", "RTH_MID"]))
                                               & (d["setup"] != "OR_BREAK_LONG")],
    }
    base = _metrics(d["realized_pnl"])
    print("=" * 78)
    print("EXPERIMENT SANDBOX (isolated, copy-on-read) —", sys.argv[1].split("/")[-1])
    print("=" * 78)
    print(f"{'scenario':36}{'n':>4}{'pnl':>9}{'pf':>6}{'wr':>5}{'dd':>9}{'sharpe':>7}{'Δpnl':>8}")
    for name, s in scenarios.items():
        mm = _metrics(s["realized_pnl"])
        dpnl = mm["pnl"] - base["pnl"]
        print(f"{name:36}{mm['n']:>4}{mm['pnl']:>9}{mm['pf']:>6}{mm['wr']:>5}{mm['dd']:>9}{mm['sharpe']:>7}{dpnl:>+8.0f}")
    # deployment-rule check (informational only — does not change anything)
    W = (d["realized_pnl"] > 0).sum()
    comb = scenarios["COMBINED (ADX>=22 + RTH + no ORB)"]
    Wk = (comb["realized_pnl"] > 0).sum()
    print("\nDeployment-rule check (informational): COMBINED gate destroys "
          f"{W - Wk}/{W} profitable trades = {100*(W-Wk)/W:.0f}%  "
          f"({'PASS' if (W-Wk)/W <= 0.30 else 'FAIL'} vs 30% limit)")
    print("\n(sandbox only — production runtime and state untouched)")


if __name__ == "__main__":
    main()
