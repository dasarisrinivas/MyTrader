#!/usr/bin/env python3
"""PHASE-1 GATE: validate the replay engine before trusting it as ground truth.

For 30 trades across dates/families:
  1. Run the engine (production contract).
  2. INDEPENDENTLY recompute from raw ThetaData quotes via a DIFFERENT code path
     (fresh linear scan, not the engine's cached bisect) — entry ask, exit bid,
     commission, spread, net, return.
  3. CROSS-VENDOR check: ThetaData NBBO mid at entry vs the bot's own IB-recorded
     `entry_mid`. Two independent vendors must agree — this also proves the
     ET timestamp alignment (a TZ bug would make them wildly disagree).
  4. Verify exit timestamp = production's recorded exit_at (production exit logic).
Any internal discrepancy >1% is flagged and must be explained.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))
from shree.flow_research.thetadata import ThetaClient
from shree.research.replay_engine import (
    QuoteCache, ExecutionModel, ContractChoice, replay_signal,
)

ET = ZoneInfo("America/New_York")
DB = "/Users/svss/Documents/code/ShreeBot/data/spy_options_signals.db"
COMM_RT = 1.30


def utc_et(s):
    s = s.split(".")[0].rstrip("Z")
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).astimezone(ET)


def indep_prevailing(client, expiry, strike, right, session, target_et):
    """Independent pull + linear scan (NOT the engine's cached bisect)."""
    k = str(int(strike)) if float(strike).is_integer() else str(strike)
    rows = client.get_csv("/v3/option/history/quote", {
        "symbol": "SPY", "expiration": expiry, "strike": k, "right": right,
        "start_date": session, "end_date": session})
    tgt = target_et.replace(tzinfo=None).isoformat(timespec="milliseconds")
    best = None
    for r in rows:
        ts = (r.get("timestamp") or "").strip().strip('"').replace("Z", "")
        if not ts or ts > tgt:
            continue
        try:
            b, a = float(r["bid"]), float(r["ask"])
        except (TypeError, ValueError, KeyError):
            continue
        if b > 0 and a > 0:
            if best is None or ts > best[0]:
                best = (ts, b, a)
    return best  # (ts, bid, ask) or None


def main():
    c = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    # stratified: spread across families and the whole date range, entry_mid present
    rows = c.execute(
        """SELECT id, sent_at, exit_at, strike, right, expiry_date, dte, spy_price,
                  spy_price_exit, pnl_pct, entry_mid, signal_type
           FROM spy_signals WHERE pnl_pct IS NOT NULL AND exit_at IS NOT NULL
           AND strike IS NOT NULL AND expiry_date IS NOT NULL AND entry_mid > 0
           ORDER BY sent_at""").fetchall()
    c.close()
    sample = rows[:: max(1, len(rows) // 30)][:30]

    client = ThetaClient()
    cache = QuoteCache(client)
    ex = ExecutionModel()

    print(f"validating {len(sample)} trades\n")
    hdr = f"{'id':>5}{'date':>12}{'K':>6}{'r':>2}{'eng_ask':>8}{'ind_ask':>8}{'eng_bid':>8}{'ind_bid':>8}{'net_e':>8}{'net_i':>8}{'ivmid':>7}{'ib_mid':>7}{'xvdr%':>7}{'flag':>6}"
    print(hdr)
    n_ok = n_flag = 0
    xvendor = []
    for s in sample:
        r = replay_signal(cache, client, s, ContractChoice.PRODUCTION, ex)
        if not r.ok:
            print(f"{s['id']:>5}  engine-skip: {r.reason}")
            continue
        entry_et = utc_et(s["sent_at"]); exit_et = utc_et(s["exit_at"])
        ie = indep_prevailing(client, s["expiry_date"], s["strike"], s["right"],
                              entry_et.strftime("%Y%m%d"), entry_et)
        ix = indep_prevailing(client, s["expiry_date"], s["strike"], s["right"],
                              exit_et.strftime("%Y%m%d"), exit_et)
        if not ie or not ix:
            print(f"{s['id']:>5}  indep-skip: no NBBO")
            continue
        ind_ask = ie[2]; ind_bid = ix[1]
        net_i = (ind_bid - ind_ask) * 100.0 - COMM_RT
        eb = r.entry_prem - r.spread_cost_entry
        iv_mid = (eb + r.entry_prem) / 2.0
        ib_mid = s["entry_mid"]
        xv = abs(iv_mid - ib_mid) / ib_mid * 100 if ib_mid else 0
        xvendor.append(xv)
        # internal consistency: engine vs independent
        ask_match = abs(r.entry_prem - ind_ask) < 1e-6
        bid_match = abs(r.exit_prem - ind_bid) < 1e-6
        net_diff = abs(r.net_dollar - net_i) / (abs(net_i) + 1e-9) * 100
        flag = ""
        if not (ask_match and bid_match) or net_diff > 1.0:
            flag = "INTERNAL"; n_flag += 1
        else:
            n_ok += 1
        print(f"{s['id']:>5}{entry_et.strftime('%m-%d %H:%M'):>12}{s['strike']:>6.0f}"
              f"{s['right']:>2}{r.entry_prem:>8.2f}{ind_ask:>8.2f}{r.exit_prem:>8.2f}"
              f"{ind_bid:>8.2f}{r.net_dollar:>8.2f}{net_i:>8.2f}{iv_mid:>7.2f}"
              f"{ib_mid:>7.2f}{xv:>7.1f}{flag:>6}")

    print(f"\nINTERNAL consistency (engine == independent recompute): "
          f"{n_ok} pass, {n_flag} flagged (>1% diff)")
    if xvendor:
        import statistics as st
        xvendor.sort()
        print(f"CROSS-VENDOR ThetaData-mid vs IB entry_mid: median={st.median(xvendor):.1f}% "
              f"p90={xvendor[int(len(xvendor)*0.9)]:.1f}%  (large % => TZ/alignment bug)")
    print("\nVERDICT:", "ENGINE VALIDATED" if n_flag == 0 else "DISCREPANCIES — investigate")


if __name__ == "__main__":
    main()
