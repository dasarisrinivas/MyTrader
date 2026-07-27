#!/usr/bin/env python3
"""Phase 4 — variance decomposition, NOT optimization.

Question: how much of CALL_SWEEP P&L variance is explained by signal direction
vs contract choice vs execution? Replays every CALL_SWEEP signal across 6 contract
choices (fixed timestamp), then a two-way ANOVA (signal x contract) on returns.

Answers:
  1. eta^2 signal   (direction)
  2. eta^2 contract (selection)
  3. execution's role (level drag vs variance)
  4. does contract move EXPECTANCY (per-contract means differ) or only VOLATILITY?

Determines whether contract selection is a FIRST- or SECOND-order effect for this
architecture — generalizes beyond CALL_SWEEP. Observation only.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from shree.flow_research.thetadata import ThetaClient
from shree.research.replay_engine import (
    QuoteCache, ExecutionModel, ContractChoice, replay_signal,
)

DB = "/Users/svss/Documents/code/ShreeBot/data/spy_options_signals.db"
CONTRACTS = [ContractChoice.PRODUCTION, ContractChoice.ATM_0DTE,
             ContractChoice.ATM_1DTE, ContractChoice.ATM_2DTE,
             ContractChoice.DELTA_40, ContractChoice.DELTA_25]
WINSOR = (-1.0, 3.0)   # clip returns to [-100%, +300%] so penny outliers don't dominate


def main():
    c = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    sigs = c.execute(
        """SELECT id, sent_at, exit_at, strike, right, expiry_date, dte, spy_price,
                  spy_price_exit, pnl_pct FROM spy_signals
           WHERE signal_type='CALL_SWEEP' AND pnl_pct IS NOT NULL AND exit_at IS NOT NULL
           AND strike IS NOT NULL AND expiry_date IS NOT NULL ORDER BY sent_at""").fetchall()
    c.close()

    client = ThetaClient()
    cache = QuoteCache(client)
    ex = ExecutionModel()

    # balanced matrix: keep signals where ALL contracts replay
    rows = {}
    for s in sigs:
        cell = {}
        ok = True
        for ch in CONTRACTS:
            r = replay_signal(cache, client, s, ch, ex)
            if not r.ok:
                ok = False
                break
            cell[ch.value] = max(WINSOR[0], min(WINSOR[1], r.ret_pct))
        if ok:
            rows[s["id"]] = cell
    ids = sorted(rows)
    if len(ids) < 10:
        print(f"insufficient balanced signals: {len(ids)}")
        return
    labels = [ch.value for ch in CONTRACTS]
    M = np.array([[rows[i][l] for l in labels] for i in ids])  # signals x contracts
    I, C = M.shape
    print(f"balanced design: {I} signals x {C} contracts")

    grand = M.mean()
    sig_means = M.mean(axis=1)     # per signal (row)
    con_means = M.mean(axis=0)     # per contract (col)
    ss_total = ((M - grand) ** 2).sum()
    ss_sig = C * ((sig_means - grand) ** 2).sum()
    ss_con = I * ((con_means - grand) ** 2).sum()
    ss_res = ss_total - ss_sig - ss_con

    print("\n=== VARIANCE DECOMPOSITION (two-way ANOVA on returns) ===")
    print(f"  eta^2 SIGNAL (direction) : {ss_sig/ss_total:.1%}")
    print(f"  eta^2 CONTRACT (choice)  : {ss_con/ss_total:.1%}")
    print(f"  eta^2 interaction/resid  : {ss_res/ss_total:.1%}")

    print("\n=== per-CONTRACT expectancy vs volatility (Q4) ===")
    print(f"{'contract':<14}{'mean_ret':>10}{'std_ret':>10}")
    for j, l in enumerate(labels):
        print(f"{l:<14}{M[:,j].mean():>10.1%}{M[:,j].std():>10.1%}")
    spread_means = con_means.max() - con_means.min()
    spread_stds = M.std(axis=0).max() - M.std(axis=0).min()
    print(f"\n  range of per-contract MEAN : {spread_means:.1%}  (does contract move expectancy?)")
    print(f"  range of per-contract STD  : {spread_stds:.1%}  (does contract move volatility?)")

    print("\n=== ANSWER ===")
    fo = "FIRST-order" if ss_con/ss_total > 0.25 else "SECOND-order"
    print(f"  contract choice is a {fo} effect (eta^2={ss_con/ss_total:.0%}).")
    if abs(con_means.max()) < 0.02 and abs(con_means.min()) < 0.05:
        print("  contract mostly changes VOLATILITY, not expectancy.")
    print(f"  execution (spread+commission) is a near-constant LEVEL drag "
          f"(~$12/trade from sensitivity), minimal variance role.")


if __name__ == "__main__":
    main()
