"""RESEARCH ONLY — measure what the overnight allowlist suppresses.

Runs the FROZEN production strategy (shree.strategies.es_fifteen_min) twice over
identical bars, changing exactly ONE thing:

    A) PRODUCTION  ft_overnight_allowed_signals = ["EMA9_PB_LONG"]
    B) FULL_ETH    ft_overnight_allowed_signals = []      # empty = no restriction

No strategy code, indicator, threshold, exit, confidence or ranking logic is
modified. "shadow_full_eth" needs no new flag — the empty list already disables
the gate (es_fifteen_min.py:1275 `if _is_overnight_pb and self._overnight_allowed_signals`).

Every run is fully sandboxed: counters, opening-range JSON and the shadow
decision log are redirected into a temp dir so production state is untouched.

Usage:
    python3 tools/eth_suppression_research.py --bars /tmp/mes_15m_eth_full.parquet
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics as st
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SLIP = 0.25          # 1 tick adverse per side
FEE_RT = 1.70        # IBKR MES all-in round turn
PV = 5.0             # $ per point
MAX_HOLD_BARS = 8    # engine's own ft_max_hold_bars
WARMUP = 60
ET = "US/Eastern"


def is_core_rth(ts: pd.Timestamp) -> bool:
    e = ts.tz_convert(ET)
    t = e.time()
    return (t >= pd.Timestamp("09:30").time()) and (t < pd.Timestamp("16:00").time())


def run_variant(bars: pd.DataFrame, allowlist: list[str], label: str) -> list[dict]:
    """Walk bars through a fresh frozen-strategy instance in a sandbox."""
    from shree.strategies.es_fifteen_min import EsFifteenMinStrategy
    from shree.utils.settings_loader import load_settings

    sandbox = tempfile.mkdtemp(prefix=f"ethres_{label}_")
    (Path(sandbox) / "data").mkdir(parents=True, exist_ok=True)
    (Path(sandbox) / "logs").mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(load_settings(str(REPO / "config.yaml")).one_minute)
    cfg.ft_overnight_allowed_signals = list(allowlist)          # THE ONLY CHANGE
    cfg.ft_counter_file = str(Path(sandbox) / "counters.json")  # isolate caps

    cwd = os.getcwd()
    os.chdir(sandbox)  # or_YYYY-MM-DD.json writes land here
    EsFifteenMinStrategy._SHADOW_LOG_PATH = Path(sandbox) / "logs" / "decisions.jsonl"
    try:
        strat = EsFifteenMinStrategy(cfg)
        out = []
        for i in range(WARMUP, len(bars)):
            window = bars.iloc[max(0, i - 400):i + 1]
            try:
                sig = strat.generate(window)
            except Exception:
                continue
            if sig.action in ("BUY", "SELL"):
                meta = sig.metadata or {}
                row = bars.iloc[i]
                out.append(dict(
                    bar_i=i, ts=bars.index[i], action=sig.action,
                    conf=float(sig.confidence),
                    entry=float(row["close"]),
                    stop=meta.get("stop_loss"), target=meta.get("take_profit"),
                    family=str(meta.get("reason", "")).split("|")[0].strip(),
                    core_rth=is_core_rth(bars.index[i]),
                ))
    finally:
        os.chdir(cwd)
        EsFifteenMinStrategy._SHADOW_LOG_PATH = None
    return out


def score(sigs: list[dict], bars: pd.DataFrame) -> list[dict]:
    """Realistic replay: next-bar open fill, 1 tick slip/side, fees, stop->target->time."""
    hi, lo, op, cl = bars["high"].values, bars["low"].values, bars["open"].values, bars["close"].values
    trades = []
    for s in sigs:
        i = s["bar_i"]
        if s["stop"] is None or s["target"] is None or i + 1 >= len(bars):
            continue
        stop, tgt = float(s["stop"]), float(s["target"])
        long = s["action"] == "BUY"
        entry = op[i + 1] + (SLIP if long else -SLIP)
        exit_px = why = None
        end = min(i + 1 + MAX_HOLD_BARS, len(bars) - 1)
        for j in range(i + 1, end + 1):
            if long:
                if lo[j] <= stop:
                    exit_px, why = stop - SLIP, "STOP"; break
                if hi[j] >= tgt:
                    exit_px, why = tgt - SLIP, "TARGET"; break
            else:
                if hi[j] >= stop:
                    exit_px, why = stop + SLIP, "STOP"; break
                if lo[j] <= tgt:
                    exit_px, why = tgt + SLIP, "TARGET"; break
        if exit_px is None:
            exit_px, why = cl[end] + (-SLIP if long else SLIP), "TIME_STOP"
        pts = (exit_px - entry) if long else (entry - exit_px)
        risk = abs(entry - stop)
        trades.append({**s, "exit": exit_px, "pts": pts, "net": pts * PV - FEE_RT,
                       "R": pts / risk if risk > 0 else 0.0, "why": why,
                       "hour_et": s["ts"].tz_convert(ET).hour})
    return trades


def stats(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    net = np.array([t["net"] for t in rows])
    w = [t for t in rows if t["net"] > 0]
    l = [t for t in rows if t["net"] <= 0]
    gp, gl = sum(t["net"] for t in w), -sum(t["net"] for t in l)
    eq = np.cumsum(net)
    return dict(n=len(rows), net=net.sum(), ev=net.mean(),
                wr=100 * len(w) / len(rows), pf=(gp / gl) if gl > 0 else float("inf"),
                dd=(eq - np.maximum.accumulate(eq)).min(),
                t=(net.mean() / (net.std(ddof=1) / math.sqrt(len(net)))) if len(net) > 1 else 0.0)


def line(label: str, s: dict | None, width: int = 30) -> str:
    if not s:
        return f"{label:<{width}} (none)"
    return (f"{label:<{width}} n={s['n']:>4}  net=${s['net']:>+9.2f}  EV=${s['ev']:>+7.2f}  "
            f"WR={s['wr']:>5.1f}%  PF={s['pf']:>5.2f}  DD=${s['dd']:>+9.2f}  t={s['t']:>+5.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", required=True)
    args = ap.parse_args()

    bars = pd.read_parquet(args.bars)
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.sort_index()
    if "volume" not in bars.columns:
        bars["volume"] = 1.0
    et = bars.index.tz_convert(ET)
    n_rth = sum(is_core_rth(t) for t in bars.index)
    print(f"BARS {len(bars)}  {bars.index[0]} -> {bars.index[-1]}")
    print(f"  RTH={n_rth}  ETH={len(bars)-n_rth}  sessions={len(set(et.date))}\n")

    prod = score(run_variant(bars, ["EMA9_PB_LONG"], "prod"), bars)
    full = score(run_variant(bars, [], "fulleth"), bars)

    print("=" * 100)
    print("A) PRODUCTION (allowlist=['EMA9_PB_LONG'])  vs  B) FULL_ETH (allowlist=[])")
    print("=" * 100)
    print(line("A) production  ALL", stats(prod)))
    print(line("B) full_eth    ALL", stats(full)))
    print()
    print(line("A) production  RTH", stats([t for t in prod if t["core_rth"]])))
    print(line("B) full_eth    RTH", stats([t for t in full if t["core_rth"]])))
    print(line("A) production  ETH", stats([t for t in prod if not t["core_rth"]])))
    print(line("B) full_eth    ETH", stats([t for t in full if not t["core_rth"]])))

    key = lambda t: (t["bar_i"], t["action"], t["family"])
    extra = [t for t in full if key(t) not in {key(x) for x in prod}]
    print("\n" + "=" * 100)
    print(f"INCREMENTAL SIGNALS UNLOCKED BY FULL_ETH: {len(extra)}")
    print("=" * 100)
    s = stats(extra)
    print(line("  incremental only", s))
    if s:
        print(f"  winners={len([t for t in extra if t['net']>0])}  "
              f"losers={len([t for t in extra if t['net']<=0])}")
        print("\n  by family:")
        for f in sorted({t["family"] for t in extra}):
            print(line(f"    {f}", stats([t for t in extra if t["family"] == f]), 30))
        print("\n  by ET hour:")
        for h in sorted({t["hour_et"] for t in extra}):
            print(line(f"    {h:02d}:00 ET", stats([t for t in extra if t["hour_et"] == h]), 30))

    print("\n" + "=" * 100)
    print("FAMILY BREAKDOWN — full_eth, RTH vs ETH")
    print("=" * 100)
    for f in sorted({t["family"] for t in full}):
        print(line(f"  {f} [RTH]", stats([t for t in full if t["family"] == f and t["core_rth"]])))
        print(line(f"  {f} [ETH]", stats([t for t in full if t["family"] == f and not t["core_rth"]])))

    print("\n" + "=" * 100)
    print("HOURLY (full_eth, ET)")
    print("=" * 100)
    for h in range(24):
        hr = [t for t in full if t["hour_et"] == h]
        if hr:
            print(line(f"  {h:02d}:00 ET", stats(hr)))

    print("\n" + "=" * 100)
    print("FEATURE DISTRIBUTION — RTH vs ETH (all bars, not just signals)")
    print("=" * 100)
    c = bars["close"]
    tr = pd.concat([bars["high"] - bars["low"],
                    (bars["high"] - c.shift()).abs(),
                    (bars["low"] - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    rng = (bars["high"] - bars["low"])
    mask = np.array([is_core_rth(t) for t in bars.index])
    for nm, ser in [("ATR(14)", atr), ("bar range", rng), ("volume", bars["volume"]),
                    ("|log ret| bp", (np.log(c / c.shift()).abs() * 1e4))]:
        r_, e_ = ser[mask].dropna(), ser[~mask].dropna()
        if len(r_) and len(e_):
            print(f"  {nm:14} RTH median={r_.median():>10.2f}   ETH median={e_.median():>10.2f}   "
                  f"ETH/RTH={e_.median()/r_.median() if r_.median() else float('nan'):>5.2f}x")

    out = Path("/tmp/eth_research_trades.json")
    out.write_text(json.dumps(
        [{**t, "ts": str(t["ts"])} for t in full], indent=1, default=str))
    print(f"\nfull_eth trades -> {out}")


if __name__ == "__main__":
    main()
