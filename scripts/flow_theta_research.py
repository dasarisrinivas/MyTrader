#!/usr/bin/env python3
"""ThetaData research pull → snapshot, one pass per session (observation only).

For each trading session it: pulls the near-ATM 0..dte_max option tape from
ThetaData (Standard), classifies (Lee-Ready on the paired NBBO), then computes a
flow snapshot at every signal time AND at fixed intervals, and writes ONLY the
small shadow_flow rows. Raw ticks live in memory for one session then are freed —
no multi-hundred-million-row tick DB.

Read-only against production (spy_signals opened immutable). Writes only the
flow-research DB. Imports nothing from executor/manager/governor. No trading.

Resumable: sessions already present in shadow_flow are skipped.

Usage:
  python3 scripts/flow_theta_research.py \
    --flow-db data/flow_research.db \
    --spy-db /Users/svss/Documents/code/ShreeBot/data/spy_options_signals.db \
    --start 2026-04-01 --end 2026-07-23 --dte-max 0 --strikes 2
"""
from __future__ import annotations

import argparse
import bisect
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from shree.flow_research.thetadata import ThetaClient, ThetaDataSource
from shree.flow_research.classify import classify_all, mark_blocks, mark_sweeps
from shree.flow_research.features import compute_features
from shree.flow_research.schema import open_db, insert_snapshot
from shree.flow_research.snapshotter import _to_et_iso

ET = ZoneInfo("America/New_York")
WINDOW_S = 1800
INTERVALS = ("09:45", "10:15", "10:45", "11:15", "11:45", "12:15", "12:45",
             "13:15", "13:45", "14:15", "14:45", "15:15", "15:45")


def _load_signal_times(spy_db: str, start: date, end: date,
                       signal_dte_max: int) -> Dict[date, List[Tuple[int, str]]]:
    """Return {ET session date: [(signal_id, ts_et_iso), ...]} for signals whose
    ET session is in [start, end] and dte <= signal_dte_max.

    NOTE: the FLOW universe (what we measure) is 0..dte_max near-ATM; the SIGNALS
    we attach it to are ALL signals (any DTE) by default — the pre-registration
    tests flow against the full signal set, and 0DTE-only signals collapse into
    an 8-session July cluster with no out-of-sample period. Attaching 0DTE flow
    to every signal restores the Apr-Jul temporal spread + the OOS split."""
    conn = sqlite3.connect(f"file:{spy_db}?mode=ro&immutable=1", uri=True)
    rows = conn.execute(
        "SELECT id, sent_at, dte FROM spy_signals WHERE dte IS NOT NULL "
        "AND dte >= 0 AND dte <= ?", (signal_dte_max,)).fetchall()
    conn.close()
    out: Dict[date, List[Tuple[int, str]]] = {}
    for sid, sent_at, _dte in rows:
        et = _to_et_iso(sent_at)
        if not et:
            continue
        d = date.fromisoformat(et[:10])
        if start <= d <= end:
            out.setdefault(d, []).append((int(sid), et))
    return out


def _session_already_done(conn, session: date) -> bool:
    r = conn.execute("SELECT 1 FROM shadow_flow WHERE session_date=? LIMIT 1",
                     (session.isoformat(),)).fetchone()
    return r is not None


def _window_slice(prints, ts_keys, end_et: str):
    start_et = (datetime.fromisoformat(end_et) -
                timedelta(seconds=WINDOW_S)).isoformat()
    lo = bisect.bisect_right(ts_keys, start_et)
    hi = bisect.bisect_right(ts_keys, end_et)
    return prints[lo:hi]


def run(flow_db: str, spy_db: str, start: date, end: date, dte_max: int,
        strikes: int, symbol: str, base_url: str, limit: Optional[int],
        signal_dte_max: int) -> None:
    client = ThetaClient(base_url)
    src = ThetaDataSource(client, start, end, symbol=symbol, dte_max=dte_max,
                          strikes_around_atm=strikes)
    conn = open_db(flow_db)

    sig_by_day = _load_signal_times(spy_db, start, end, signal_dte_max)
    exps = [e for e in client.list_expirations(symbol) if start <= e <= end]
    # process the sessions where signals actually fired (bounds the pull; each
    # gets snapshots at every signal + fixed intervals)
    sessions = sorted(sig_by_day)
    closes = client.stock_eod_close(symbol, start, end)
    print(f"[theta] {len(sessions)} signal-sessions, "
          f"{sum(len(v) for v in sig_by_day.values())} signals "
          f"(flow universe 0..{dte_max} DTE, signals dte<= {signal_dte_max})")

    done = 0
    for S in sessions:
        if limit is not None and done >= limit:
            break
        if _session_already_done(conn, S):
            print(f"[theta] {S} already in shadow_flow — skip")
            continue
        atm = closes.get(S)
        targets = [e for e in exps if e >= S][: dte_max + 1]
        t0 = time.time()
        prints = []
        for exp in targets:
            for strike in src._near_atm(exp, atm):
                for right in ("C", "P"):
                    try:
                        rows = client.option_trade_quote(symbol, exp, strike,
                                                          right, S, S)
                    except Exception as exc:
                        print(f"[theta]   {S} {exp} {strike}{right} ERR {exc}")
                        continue
                    dte = (exp - S).days
                    for r in rows:
                        p = src._row_to_print(r, exp, strike, right, S, atm, dte)
                        if p is not None:
                            prints.append(p)
        if not prints:
            print(f"[theta] {S}: no prints")
            continue
        classify_all(prints)
        mark_blocks(prints)
        mark_sweeps(prints)
        prints.sort(key=lambda p: p.ts_et)
        ts_keys = [p.ts_et for p in prints]

        n_sig = n_int = 0
        for sid, et in sig_by_day.get(S, []):
            w = _window_slice(prints, ts_keys, et)
            snap = compute_features(w, snapshot_kind="SIGNAL",
                                    session_date=S.isoformat(), ts_et=et,
                                    window_s=WINDOW_S, signal_id=sid)
            insert_snapshot(conn, snap)
            n_sig += 1
        for hhmm in INTERVALS:
            et = f"{S.isoformat()}T{hhmm}:00"
            w = _window_slice(prints, ts_keys, et)
            if not w:
                continue
            snap = compute_features(w, snapshot_kind="INTERVAL",
                                    session_date=S.isoformat(), ts_et=et,
                                    window_s=WINDOW_S)
            insert_snapshot(conn, snap)
            n_int += 1
        print(f"[theta] {S}: {len(prints):>7} ticks, SIGNAL={n_sig} INTERVAL={n_int} "
              f"({time.time()-t0:.0f}s)")
        del prints, ts_keys
        done += 1
    conn.close()
    print("[theta] done")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--flow-db", default="data/flow_research.db")
    p.add_argument("--spy-db", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--dte-max", type=int, default=0)
    p.add_argument("--strikes", type=int, default=2)
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--base-url", default="http://127.0.0.1:25503")
    p.add_argument("--limit", type=int, default=None,
                   help="max sessions to process this run (for a bounded test)")
    p.add_argument("--signal-dte-max", type=int, default=99,
                   help="attach flow to signals with dte <= this (default all)")
    a = p.parse_args(argv)
    run(a.flow_db, a.spy_db, date.fromisoformat(a.start),
        date.fromisoformat(a.end), a.dte_max, a.strikes, a.symbol,
        a.base_url, a.limit, a.signal_dte_max)


if __name__ == "__main__":
    main()
