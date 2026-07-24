#!/usr/bin/env python3
"""Forward-return test — the pre-registered UNBIASED flow test.

Does flow (measured at fixed interval times, unconditional on signals) predict
the UNDERLYING's forward move? Uses the 598 INTERVAL snapshots already in
shadow_flow + SPY 1-min bars pulled FREE from IB (Options-Standard sub does not
include ThetaData stock data). Read-only; no trading.

Runs quintile lift + shuffle for each flow measure vs forward return at
+5/+15/+30 min, plus an OOS split (train Apr-May / test Jun-Jul).
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from ib_insync import IB, Stock

sys.path.insert(0, str(Path(__file__).parent.parent))
from shree.flow_research import validate as V  # noqa: E402

ET = ZoneInfo("America/New_York")
FLOW = "data/flow_research_theta.db"
HORIZONS = (5, 15, 30)


def pull_spy_minute(sessions, client_id=151):
    ib = IB()
    ib.connect("127.0.0.1", 4001, clientId=client_id, timeout=20, readonly=True)
    spy = Stock("SPY", "SMART", "USD")
    ib.qualifyContracts(spy)
    price = {}  # "YYYY-MM-DDTHH:MM" -> close
    try:
        for s in sessions:
            end = f"{s.replace('-','')} 16:00:00 US/Eastern"
            bars = ib.reqHistoricalData(spy, end, "1 D", "1 min", "TRADES",
                                        useRTH=True, formatDate=1)
            for b in bars:
                dt = b.date if isinstance(b.date, datetime) else None
                if dt is None:
                    continue
                key = dt.strftime("%Y-%m-%dT%H:%M")
                price[key] = float(b.close)
            print(f"  {s}: {len(bars)} bars")
    finally:
        ib.disconnect()
    return price


def fwd_return(price, ts_et, horizon_min):
    base = datetime.fromisoformat(ts_et).replace(second=0, microsecond=0)
    k0 = base.strftime("%Y-%m-%dT%H:%M")
    k1 = (base + timedelta(minutes=horizon_min)).strftime("%Y-%m-%dT%H:%M")
    p0, p1 = price.get(k0), price.get(k1)
    if p0 and p1 and p0 > 0:
        return (p1 - p0) / p0
    return None


def main():
    conn = sqlite3.connect(FLOW)
    iv = pd.read_sql_query(
        "SELECT * FROM shadow_flow WHERE snapshot_kind='INTERVAL'", conn)
    conn.close()
    sessions = sorted(iv["session_date"].unique())
    print(f"interval snapshots: {len(iv)} over {len(sessions)} sessions")
    price = pull_spy_minute(sessions)
    print(f"minute bars loaded: {len(price)}")

    for h in HORIZONS:
        iv[f"fwd_{h}"] = iv["ts_et"].apply(lambda t: fwd_return(price, t, h))

    for h in HORIZONS:
        col = f"fwd_{h}"
        d = iv[iv[col].notna()].copy()
        print(f"\n===== forward return +{h}min  (n={len(d)}) =====")
        print(f"{'measure':<22}{'top-bot':>11}{'monotone':>10}{'shuffle_p':>11}")
        ps = []
        for m in V.FLOW_MEASURES:
            if m not in d or d[m].nunique() < 5:
                continue
            L = V.quintile_lift(d, m, outcome=col)
            S = V.shuffle_control(d, m, outcome=col, n_shuffles=800)
            ps.append((m, S.get("p_value", 1.0)))
            print(f"{m:<22}{L.get('top_minus_bottom','-'):>11}"
                  f"{str(L.get('monotone')):>10}{S.get('p_value','-'):>11}")
        ps_sorted = sorted(ps, key=lambda x: x[1])
        n = len(ps_sorted)
        hits = [m for i, (m, p) in enumerate(ps_sorted) if p <= (i + 1) / n * 0.10]
        print(f"  BH-FDR q=0.10: {'NONE pass' if not hits else hits}")

    # OOS split on the primary measure dw_flow at +15min
    d = iv[iv["fwd_15"].notna()].copy()
    tr = d[d["session_date"] <= "2026-05-31"]
    te = d[d["session_date"] >= "2026-06-01"]
    print(f"\n===== OOS (dw_flow, +15min): train={len(tr)} test={len(te)} =====")
    for name, sub in (("train Apr-May", tr), ("test Jun-Jul", te)):
        if len(sub) >= 25 and sub["dw_flow"].nunique() >= 5:
            L = V.quintile_lift(sub, "dw_flow", outcome="fwd_15")
            S = V.shuffle_control(sub, "dw_flow", outcome="fwd_15", n_shuffles=800)
            print(f"  {name}: top-bot={L.get('top_minus_bottom')} "
                  f"shuffle_p={S.get('p_value')}")
        else:
            print(f"  {name}: insufficient (n={len(sub)})")


if __name__ == "__main__":
    main()
