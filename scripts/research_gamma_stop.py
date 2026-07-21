#!/usr/bin/env python3
"""Gamma-stop research (backlog #2, 2026-07-20). READ-ONLY — no deployment.

Question: the bracket premium stop is placed via LINEAR delta conversion
(prem_stop = |delta| × SPY_stop_distance, clamped 12–35%). Live evidence
(Jul-20 745P: premium −12% arrived at SPY +0.51, not the linear +0.91)
suggests effective sensitivity ≈ 1.5–2× quoted delta on 1–3 DTE ATM
contracts (gamma + IV bleed). This script measures the EFFECTIVE delta of
every completed live fill (Δpremium / ΔSPY between entry and exit) and
compares it to the quoted entry delta, producing the empirical multiplier
that a gamma-adjusted conversion would use.

Deployment gate (per backlog): only after the multiplier is stable across
n≥15 fills AND a replay shows stops landing at structure without widening
realized losses. Until then: evidence collection.
"""

import re
import sqlite3
import sys
from datetime import datetime, timedelta

ROOT = "/Users/svss/Documents/code/ShreeBot"
LOGS = [f"{ROOT}/logs/spy_options.2026-06-04_07-39-47_112786.log",
        f"{ROOT}/logs/spy_options.log"]

poll_re = re.compile(r"^(\S+ \S+) CST \|.*- Poll: SPY=([\d.]+)")
tape = []
for LOG in LOGS:
    try:
        with open(LOG, errors="replace") as fh:
            for line in fh:
                m = poll_re.match(line)
                if m:
                    tape.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"),
                                 float(m.group(2))))
    except FileNotFoundError:
        pass
tape.sort()


def spy_at(dt):
    prev = None
    for t, s in tape:
        if t <= dt:
            prev = s
        else:
            break
    return prev


def main():
    con = sqlite3.connect(f"{ROOT}/data/spy_options_signals.db")
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT id, right, strike, dte, fill_entry_at, fill_exit_at,
                  fill_entry_premium ein, fill_exit_premium eout
           FROM spy_signals
           WHERE fill_entry_at IS NOT NULL AND fill_exit_at IS NOT NULL
             AND fill_entry_premium > 0 AND fill_exit_premium > 0
           ORDER BY fill_entry_at""").fetchall()

    print(f"{'id':>4} {'R':1} {'dte':>3} {'hold':>5} {'ΔSPY':>6} {'Δprem':>6} "
          f"{'eff_Δ':>6} {'quoted':>7} {'mult':>5}")
    mults = []
    for r in rows:
        t_in = datetime.strptime(r["fill_entry_at"][:19], "%Y-%m-%dT%H:%M:%S") - timedelta(hours=5)
        t_out = datetime.strptime(r["fill_exit_at"][:19], "%Y-%m-%dT%H:%M:%S") - timedelta(hours=5)
        s_in, s_out = spy_at(t_in), spy_at(t_out)
        if not s_in or not s_out or abs(s_out - s_in) < 0.15:
            continue   # SPY move too small → effective-delta estimate unstable
        d_spy = s_out - s_in
        d_prem = r["eout"] - r["ein"]
        eff = d_prem / d_spy                      # signed effective delta
        # quoted delta at entry unavailable in DB for all rows — approximate
        # ATM 1-3 DTE quoted |delta| ≈ 0.40-0.55; use 0.50 baseline, report both.
        quoted = 0.50 if r["right"] == "C" else -0.50
        mult = eff / quoted
        hold = (t_out - t_in).total_seconds() / 60
        print(f'{r["id"]:>4} {r["right"]} {r["dte"] or "?":>3} {hold:5.0f} '
              f'{d_spy:+6.2f} {d_prem:+6.2f} {eff:+6.2f} {quoted:+7.2f} {mult:5.2f}')
        if 0 < mult < 6:
            mults.append(mult)

    if mults:
        mults.sort()
        n = len(mults)
        med = mults[n // 2]
        print(f"\nn={n} usable fills | effective/quoted delta multiplier: "
              f"median={med:.2f}  mean={sum(mults)/n:.2f}  "
              f"range=[{mults[0]:.2f}, {mults[-1]:.2f}]")
        print(f"\nImplication: linear conversion understates premium sensitivity "
              f"~{med:.1f}x on these holds. A gamma-adjusted bracket would set "
              f"prem_stop_dist = |delta| × {med:.2f} × SPY_stop_distance "
              f"(then clamp), aligning the premium stop with the structural "
              f"level by design instead of by luck.")
        print("Deployment gate: n>=15 AND multiplier stable (IQR < 0.8) AND "
              "replay shows no widened realized losses. "
              f"Status: n={n} → {'ELIGIBLE for replay step' if n >= 15 else 'COLLECTING'}")
    else:
        print("\nNo usable fills (need |ΔSPY| ≥ 0.15 between entry and exit).")


if __name__ == "__main__":
    main()
