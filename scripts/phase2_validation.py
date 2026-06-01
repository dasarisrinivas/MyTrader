#!/usr/bin/env python3
"""
Phase 2 validation report — conviction sizing.

Produces the MANDATORY dual output (never mixed):
  A. RAW STRATEGY (truth layer)   : realized_pnl, qty=1, no scaling
  B. SIZING APPLIED (economic)    : realized_pnl * size_multiplier

Plus: catastrophic audit, sizing distribution, and (if a Phase-1 trades CSV is
given) universe-integrity / path-divergence check.

Usage:
  python3 scripts/phase2_validation.py <phase2_trades.csv> [<phase1_trades.csv>]
"""
import sys, ast, math
import pandas as pd


def _meta(df):
    def g(m, k, d):
        try:
            return ast.literal_eval(m).get(k, d)
        except Exception:
            return d
    df["size_multiplier"] = df["entry_metadata"].apply(lambda m: float(g(m, "size_multiplier", 1.0)))
    df["size_tier"] = df["entry_metadata"].apply(lambda m: g(m, "size_tier", "NEUTRAL"))
    df["catastrophic"] = df["entry_metadata"].apply(lambda m: bool(g(m, "catastrophic_flag", False)))
    return df


def _metrics(pnl: pd.Series):
    n = len(pnl)
    wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
    gp = wins.sum(); gl = -losses.sum()
    pf = (gp / gl) if gl > 0 else float("inf")
    return dict(n=n, pnl=round(pnl.sum(), 2), wr=round(100 * len(wins) / n, 1) if n else 0,
                pf=round(pf, 2), avg_win=round(wins.mean(), 2) if len(wins) else 0,
                avg_loss=round(losses.mean(), 2) if len(losses) else 0)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    p2 = _meta(pd.read_csv(sys.argv[1]))

    raw = _metrics(p2["realized_pnl"])
    weighted = _metrics(p2["realized_pnl"] * p2["size_multiplier"])

    print("=" * 60)
    print("PHASE 2 VALIDATION —", sys.argv[1].split("/")[-1])
    print("=" * 60)
    print("\n--- A. RAW STRATEGY (truth layer, qty=1, no scaling) ---")
    for k, v in raw.items(): print(f"   {k:9}: {v}")
    print("\n--- B. SIZING APPLIED (economic, pnl × size_mult) ---")
    for k, v in weighted.items(): print(f"   {k:9}: {v}")
    print(f"\n   economic Δ vs raw P&L: {weighted['pnl'] - raw['pnl']:+.2f}")

    print("\n--- CATASTROPHIC AUDIT (always evaluated) ---")
    cat = p2[p2["catastrophic"]]
    z = p2[p2["size_multiplier"] == 0.0]
    print(f"   catastrophic_flag=True : {len(cat)}")
    print(f"   size_mult == 0.0       : {len(z)}")
    print(f"   raw P&L of those trades: {cat['realized_pnl'].sum():.2f}")

    print("\n--- SIZING DISTRIBUTION ---")
    dist = p2.groupby("size_tier").agg(
        n=("realized_pnl", "size"),
        mult=("size_multiplier", "first"),
        raw_pnl=("realized_pnl", "sum"),
    ).sort_values("mult")
    for tier, r in dist.iterrows():
        print(f"   {tier:12} x{r['mult']:.2f}  n={int(r['n']):>3}  raw_pnl={r['raw_pnl']:+.1f}")
    print(f"   mean size_mult: {p2['size_multiplier'].mean():.3f}  "
          f"min: {p2['size_multiplier'].min():.2f}  max: {p2['size_multiplier'].max():.2f}")

    # ── Universe integrity vs Phase 1 (optional) ──
    if len(sys.argv) >= 3:
        p1 = pd.read_csv(sys.argv[2])
        s1, s2 = set(p1["entry_time"]), set(p2["entry_time"])
        both = s1 & s2
        print("\n--- UNIVERSE INTEGRITY vs Phase 1 ---")
        print(f"   Phase1 trades: {len(s1)}   Phase2 trades: {len(s2)}")
        print(f"   shared entry_time: {len(both)}")
        print(f"   only Phase1: {len(s1 - s2)}   only Phase2: {len(s2 - s1)}")
        ov = 100 * len(both) / len(s1 | s2) if (s1 | s2) else 0
        diff = abs(len(s2) - len(s1)) / len(s1) * 100 if len(s1) else 0
        print(f"   overlap: {ov:.1f}%   trade-count diff: {diff:.1f}%")
        # first positional divergence
        fd = None
        for i in range(min(len(p1), len(p2))):
            if p1.iloc[i]["entry_time"] != p2.iloc[i]["entry_time"]:
                fd = i; break
        print(f"   first positional divergence: {'none' if fd is None else f'index {fd}'}")
        print("\n   ACCEPTANCE:")
        print(f"   {'PASS' if diff <= 1 else 'FAIL'}  trade-count diff ≤ 1%  ({diff:.1f}%)")
        print(f"   {'PASS' if ov > 95 else 'FAIL'}  universe overlap > 95%  ({ov:.1f}%)")
        print(f"   {'PASS' if fd is None else 'FAIL'}  zero path-divergence")
        print(f"   {'PASS' if weighted['pf'] >= raw['pf'] else 'CHECK'}  weighted PF ≥ raw PF "
              f"({weighted['pf']} vs {raw['pf']})")


if __name__ == "__main__":
    main()
