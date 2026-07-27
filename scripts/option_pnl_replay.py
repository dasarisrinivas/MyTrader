#!/usr/bin/env python3
"""Real-option-P&L replay (Outcome B) — validation-pipeline upgrade.

For each shadow/dispatched signal, in addition to the existing SPY-barrier grade
(Outcome A = pnl_pct), replay the ACTUAL option premium from ThetaData NBBO with
REALISTIC fills, then decompose:

    Direction   Option P&L   Interpretation
    correct     positive     TRUE EDGE
    correct     negative     execution / contract problem
    wrong       positive     option mechanics dominated
    wrong       negative     signal failure

Realistic fills (per the caution — do NOT build another optimistic sim):
  * BUY at the prevailing NBBO ASK at entry timestamp (marketable)
  * SELL at the prevailing NBBO BID at the recorded exit timestamp (production's
    own exit time = production exit logic)
  * IBKR options commission $0.65/contract/side ($1.30 round trip)
  * bid/ask crossing IS the spread cost; no free mid fills

Read-only. Observation only. No production changes.
"""
from __future__ import annotations

import argparse
import bisect
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))
from shree.flow_research.thetadata import ThetaClient  # noqa: E402

ET = ZoneInfo("America/New_York")
DB = "/Users/svss/Documents/code/ShreeBot/data/spy_options_signals.db"
COMMISSION_RT = 1.30  # $/contract round trip (IBKR ~0.65/side)


def _utc_to_et(s: str) -> datetime:
    s = s.split(".")[0].rstrip("Z")
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).astimezone(ET)


def _et_key(dt: datetime) -> str:
    return dt.replace(tzinfo=None).isoformat(timespec="milliseconds")


def prevailing_quote(quotes, ts_keys, ts_et_iso):
    """Last NBBO row with timestamp <= ts. Returns (bid, ask) or (None,None)."""
    i = bisect.bisect_right(ts_keys, ts_et_iso) - 1
    while i >= 0:
        b = quotes[i].get("bid"); a = quotes[i].get("ask")
        try:
            b = float(b); a = float(a)
        except (TypeError, ValueError):
            i -= 1; continue
        if b > 0 and a > 0:
            return b, a
        i -= 1
    return None, None


def load_quotes(client, expiry_ymd, strike, right, session_ymd):
    rows = client.get_csv("/v3/option/history/quote", {
        "symbol": "SPY", "expiration": expiry_ymd,
        "strike": str(int(strike)) if float(strike).is_integer() else str(strike),
        "right": right, "start_date": session_ymd, "end_date": session_ymd})
    # ThetaData quote timestamps are ET; key by ET iso for bisect
    out = []
    for r in rows:
        ts = (r.get("timestamp") or "").strip().strip('"')
        if not ts:
            continue
        out.append({"ts": ts.replace("Z", ""), "bid": r.get("bid"),
                    "ask": r.get("ask")})
    out.sort(key=lambda x: x["ts"])
    return out, [x["ts"] for x in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-type", default="CALL_SWEEP")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--base-url", default="http://127.0.0.1:25503")
    a = ap.parse_args()

    c = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    sigs = c.execute(
        """SELECT id, sent_at, exit_at, strike, right, expiry_date, dte,
                  spy_price, spy_price_exit, pnl_pct
           FROM spy_signals WHERE signal_type=? AND pnl_pct IS NOT NULL
           AND exit_at IS NOT NULL AND strike IS NOT NULL
           AND expiry_date IS NOT NULL ORDER BY sent_at""",
        (a.signal_type,)).fetchall()
    c.close()
    sample = sigs[:: max(1, len(sigs) // a.limit)][:a.limit]
    client = ThetaClient(a.base_url)

    cells = {"TRUE_EDGE": 0, "EXEC_CONTRACT": 0, "MECH_DOMINATED": 0,
             "SIGNAL_FAIL": 0}
    realB, barrierA, agree = [], [], 0
    done = 0
    for s in sample:
        entry = _utc_to_et(s["sent_at"]); exit_ = _utc_to_et(s["exit_at"])
        if entry.date() != exit_.date():
            continue  # keep POC intraday same-day
        try:
            q, keys = load_quotes(client, s["expiry_date"], s["strike"],
                                  s["right"], entry.strftime("%Y%m%d"))
        except Exception as e:
            print(f"  id={s['id']} pull err {e}"); continue
        if not q:
            continue
        eb, ea = prevailing_quote(q, keys, _et_key(entry))
        xb, xa = prevailing_quote(q, keys, _et_key(exit_))
        if not ea or not xb:
            continue
        # realistic: buy at ask, sell at bid, minus commission
        net = (xb - ea) * 100.0 - COMMISSION_RT
        retB = net / (ea * 100.0)
        outA_win = s["pnl_pct"] > 0
        outB_win = net > 0
        dir_ok = s["spy_price_exit"] > s["spy_price"]  # calls predict up
        realB.append(retB); barrierA.append(s["pnl_pct"])
        if outA_win == outB_win:
            agree += 1
        if dir_ok and outB_win:
            cells["TRUE_EDGE"] += 1
        elif dir_ok and not outB_win:
            cells["EXEC_CONTRACT"] += 1
        elif not dir_ok and outB_win:
            cells["MECH_DOMINATED"] += 1
        else:
            cells["SIGNAL_FAIL"] += 1
        done += 1
        print(f"  id={s['id']} {entry.strftime('%m-%d %H:%M')} K={s['strike']:.0f} "
              f"dte={s['dte']} entry_ask={ea:.2f} exit_bid={xb:.2f} "
              f"realB={retB:+.1%} barrierA={s['pnl_pct']:+.1%} "
              f"dir={'UP' if dir_ok else 'DN'}")

    n = done
    print(f"\n=== REAL-OPTION-P&L REPLAY ({a.signal_type}, n={n}) ===")
    if n:
        import statistics as st
        print(f"Outcome A (SPY barrier)  WR={sum(1 for x in barrierA if x>0)/n:.0%} "
              f"mean={st.mean(barrierA):+.1%}")
        print(f"Outcome B (REAL option)  WR={sum(1 for x in realB if x>0)/n:.0%} "
              f"mean={st.mean(realB):+.1%}")
        print(f"A vs B agree on win/loss: {agree}/{n} ({agree/n:.0%})")
        print("\n4-way decomposition (direction x real option P&L):")
        for k, v in cells.items():
            print(f"  {k:<16} {v:>3} ({v*100//max(n,1)}%)")


if __name__ == "__main__":
    main()
