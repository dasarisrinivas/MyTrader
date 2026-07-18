#!/usr/bin/env python3
"""Per-family strategy scorecard — LIVE / SHADOW-dispatched / SHADOW-blocked.

Strategy audit 2026-07-17. Three evidence tiers, NEVER mixed:
  LIVE     — real executor fills (fill_pnl_usd), the only dollars that exist.
  SHADOW-D — dispatched signals' simulated exits (blocked_gate IS NULL).
  SHADOW-B — blocked signals' simulated exits (blocked_gate set) — the
             counterfactual: "if the gate hadn't fired, would it have paid?"

Usage: python3 scripts/strategy_scorecard.py [--days N] (default 30)
"""

import argparse
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "spy_options_signals.db")
LEDGER = os.path.join(ROOT, "logs", "blocked_signals.jsonl")


def table(title, rows, hdr):
    print(f"\n── {title} " + "─" * max(1, 66 - len(title)))
    if not rows:
        print("  (no data)")
        return
    print("  " + " ".join(f"{h:>10}" for h in hdr))
    for r in rows:
        print("  " + " ".join(f"{str(v):>10}" for v in r))


def wilson_low(w, n, z=1.96):
    """Wilson lower bound on win rate — promotion metric (anti-overfit)."""
    if n == 0:
        return 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    e = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return round((c - e) / d * 100, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    since = (datetime.utcnow() - timedelta(days=args.days)).isoformat()

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row

    # ── LIVE: real fills only ────────────────────────────────────────────
    rows = []
    for r in con.execute(
        """SELECT signal_type st, COUNT(*) n,
                  SUM(fill_pnl_usd > 0) w,
                  SUM(fill_pnl_usd IS NOT NULL AND fill_exit_at IS NOT NULL) dec,
                  ROUND(SUM(fill_pnl_usd), 0) pnl,
                  ROUND(AVG(fill_pnl_usd), 0) avg
           FROM spy_signals
           WHERE fill_entry_at >= ? AND fill_entry_at IS NOT NULL
           GROUP BY st ORDER BY n DESC""", (since,)):
        rows.append((r["st"][:10], r["n"], r["dec"] or 0, r["w"] or 0,
                     r["pnl"] or 0, r["avg"] or 0,
                     wilson_low(r["w"] or 0, r["dec"] or 0)))
    table(f"LIVE fills (last {args.days}d)", rows,
          ["family", "fills", "decided", "wins", "pnl$", "avg$", "wilsonLo%"])

    # ── SHADOW-dispatched ────────────────────────────────────────────────
    for label, cond in [
        ("SHADOW-DISPATCHED (sim exits)", "blocked_gate IS NULL"),
        ("SHADOW-BLOCKED (counterfactual)", "blocked_gate IS NOT NULL"),
    ]:
        rows = []
        for r in con.execute(
            f"""SELECT signal_type st, COUNT(*) n,
                       SUM(outcome='win') w, SUM(outcome='loss') l,
                       ROUND(AVG(CASE WHEN outcome IN ('win','loss')
                                 THEN pnl_pct END) * 100, 1) avgpct
                FROM spy_signals
                WHERE sent_at >= ? AND {cond} AND fill_entry_at IS NULL
                GROUP BY st ORDER BY n DESC""", (since,)):
            d = (r["w"] or 0) + (r["l"] or 0)
            wr = f"{100 * (r['w'] or 0) / d:.0f}%" if d else "—"
            rows.append((r["st"][:10], r["n"], d, wr, r["avgpct"],
                         wilson_low(r["w"] or 0, d)))
        table(f"{label} (last {args.days}d)", rows,
              ["family", "signals", "decided", "WR", "avgPnl%", "wilsonLo%"])

    # ── Blocked-by-gate histogram (counterfactual detail) ────────────────
    rows = []
    for r in con.execute(
        """SELECT blocked_gate g, signal_type st, COUNT(*) n,
                  SUM(outcome='win') w, SUM(outcome='loss') l
           FROM spy_signals
           WHERE blocked_gate IS NOT NULL AND sent_at >= ?
           GROUP BY g, st ORDER BY n DESC LIMIT 15""", (since,)):
        d = (r["w"] or 0) + (r["l"] or 0)
        wr = f"{100 * (r['w'] or 0) / d:.0f}%" if d else "—"
        rows.append((r["g"][:22], r["st"][:10], r["n"], d, wr))
    table("BLOCKED by gate × family", rows,
          ["gate", "family", "n", "decided", "WR"])

    # ── Kill-ledger histogram (all kills incl. pre-shadow) ───────────────
    if os.path.exists(LEDGER):
        cnt = Counter()
        with open(LEDGER, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    cnt[(rec.get("gate", "?"), rec.get("signal_type", "?"))] += 1
                except Exception:
                    continue
        rows = [(g[:22], s[:10], n) for (g, s), n in cnt.most_common(12)]
        table("Kill ledger (blocked_signals.jsonl, all-time)", rows,
              ["gate", "family", "kills"])

    print("\nPromotion rule (router design): a family×regime cell is eligible "
          "for live trading only when SHADOW-BLOCKED+DISPATCHED decided n≥30 "
          "AND Wilson lower bound of WR clears breakeven WR for the bracket "
          "geometry. Demotion: rolling-10 live EV < 0. Never mix tiers.")


if __name__ == "__main__":
    main()
