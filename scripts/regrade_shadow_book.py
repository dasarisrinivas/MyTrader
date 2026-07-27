#!/usr/bin/env python3
"""Re-grade the ENTIRE shadow book on real option P&L (canonical engine v2).

Priority-1 migration: replace SPY-barrier ground truth with real historical
option replay for every family. Side-by-side vs the legacy barrier so we can see
exactly where and why conclusions change. Observation only.

  python3 scripts/regrade_shadow_book.py                 # production contract, all families
  python3 scripts/regrade_shadow_book.py --multi CALL_SWEEP   # multi-contract for one family
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shree.flow_research.thetadata import ThetaClient
from shree.research.replay_engine import (
    RESEARCH_ENGINE_VERSION, ExecutionModel, ContractChoice,
    QuoteCache, replay_signal, scorecard,
)

DB = "/Users/svss/Documents/code/ShreeBot/data/spy_options_signals.db"
FAMILIES = ["CALL_SWEEP", "PUT_SWEEP", "TREND_CONTINUATION", "ORB_BREAKOUT",
            "PC_RATIO_EXTREME", "LONG_STRADDLE", "BULL_CALL_SPREAD",
            "BEAR_PUT_SPREAD"]
MULTI = [ContractChoice.PRODUCTION, ContractChoice.ATM_0DTE,
         ContractChoice.ATM_1DTE, ContractChoice.DELTA_40, ContractChoice.DELTA_25]


def load(fam, limit):
    c = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    rows = c.execute(
        """SELECT id, sent_at, exit_at, strike, right, expiry_date, dte,
                  spy_price, spy_price_exit, pnl_pct
           FROM spy_signals WHERE signal_type=? AND pnl_pct IS NOT NULL
           AND exit_at IS NOT NULL AND strike IS NOT NULL AND expiry_date IS NOT NULL
           ORDER BY sent_at""", (fam,)).fetchall()
    c.close()
    if limit and len(rows) > limit:
        rows = rows[:: max(1, len(rows) // limit)][:limit]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60, help="per-family cap")
    ap.add_argument("--multi", default=None, help="family for multi-contract run")
    ap.add_argument("--base-url", default="http://127.0.0.1:25503")
    ap.add_argument("--out", default="data/regrade_v2.json")
    a = ap.parse_args()
    client = ThetaClient(a.base_url)
    cache = QuoteCache(client)
    ex = ExecutionModel()
    report = {"engine": RESEARCH_ENGINE_VERSION, "families": {}}

    if a.multi:
        rows = load(a.multi, a.limit)
        print(f"MULTI-CONTRACT: {a.multi} n={len(rows)}  (signal fixed, contract varies)")
        print(f"{'contract':<16}{'opt_WR':>8}{'exp_%':>9}{'exp_$':>9}{'PF':>7}{'tailp95':>9}")
        for ch in MULTI:
            res = [replay_signal(cache, client, r, ch, ex) for r in rows]
            sc = scorecard(res)
            print(f"{ch.value:<16}{sc.get('option_win_rate',0):>8.2f}"
                  f"{sc.get('expectancy_pct',0):>9.3f}{sc.get('expectancy_dollar',0):>9.2f}"
                  f"{(sc.get('profit_factor') or 0):>7.2f}{sc.get('tail_loss_p95_pct',0):>9.2f}")
            report["families"].setdefault(a.multi, {})[ch.value] = sc
    else:
        print(f"RE-GRADE (production contract) — engine {RESEARCH_ENGINE_VERSION}")
        print(f"{'family':<20}{'n':>4}{'barrier_WR':>11}{'barrier_E%':>11}"
              f"{'REAL_WR':>9}{'REAL_E%':>9}{'REAL_E$':>9}{'PF':>6}{'maxDD$':>9}")
        for fam in FAMILIES:
            rows = load(fam, a.limit)
            if not rows:
                continue
            res = [replay_signal(cache, client, r, ContractChoice.PRODUCTION, ex)
                   for r in rows]
            sc = scorecard(res)
            report["families"][fam] = sc
            if sc.get("n_replayed", 0) == 0:
                print(f"{fam:<20}{len(rows):>4}  (no replayable NBBO)")
                continue
            print(f"{fam:<20}{sc['n_replayed']:>4}"
                  f"{sc['legacy_barrier_win_rate']:>11.2f}{sc['legacy_barrier_expectancy_pct']:>11.3f}"
                  f"{sc['option_win_rate']:>9.2f}{sc['expectancy_pct']:>9.3f}"
                  f"{sc['expectancy_dollar']:>9.2f}{(sc.get('profit_factor') or 0):>6.2f}"
                  f"{sc['max_drawdown_dollar']:>9.0f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
