#!/usr/bin/env python3
"""
Extended ATR-Adaptive Stop Backtest — Signal C
===============================================

Runs the A-Modified stop (SL=min(20, max(8, ATR×1.0)), TP=SL×1.25) across
the full 1-year ES 15m dataset (~25K bars, Dec 2024 – Jan 2026).

Reports:
  - Overall performance: WR, PF, expectancy, drawdown
  - Performance by ATR bucket: ATR<10, 10-18, >18
  - Monthly breakdown
  - Equity curve stats

Usage:
    python3 tools/backtest_atr_stops.py
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shree.features.feature_engineer import _ema, _atr, _adx, _rsi

POINT_VALUE = 5.0
SLIPPAGE_PTS = 0.50
COMMISSION_RT = 2.48
TIMEOUT_BARS = 20


# ── Data loading ─────────────────────────────────────────────────────

def load_15m_data() -> pd.DataFrame:
    """Load 15m parquet data, add indicators."""
    parquet_path = ROOT / "data" / "ib" / "ES_15m_1y.parquet"
    if not parquet_path.exists():
        print(f"❌ {parquet_path} not found")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    print(f"  Loaded: {len(df)} bars, {df.index[0]} → {df.index[-1]}")

    c, h, lo = df["close"], df["high"], df["low"]
    df["EMA_9"] = _ema(c, 9)
    df["EMA_21"] = _ema(c, 21)
    df["EMA_50"] = _ema(c, 50)
    df["RSI_14"] = _rsi(c, 14)
    df["ATR_14"] = _atr(h, lo, c, 14)
    df["ADX_14"] = _adx(h, lo, c, 14)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACDhist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    for col in ["EMA_9", "EMA_21", "EMA_50", "RSI_14", "ATR_14", "ADX_14", "MACDhist"]:
        df[col] = df[col].ffill().bfill()

    return df


# ── Signal C detector ────────────────────────────────────────────────

@dataclass
class SignalCHit:
    bar_time: Any
    bar_idx: int
    close: float
    atr: float
    adx: float
    rsi: float
    month: str  # YYYY-MM for monthly breakdown


def detect_signal_c(df: pd.DataFrame) -> List[SignalCHit]:
    """Detect all Signal C (EMA9 Pullback Long) fires."""
    hits = []
    ema_touch_pct = 0.0015
    ema21_touch_pct = 0.0015

    for idx in range(60, len(df)):
        row = df.iloc[idx]
        ts = df.index[idx]

        # Convert to ET
        try:
            if hasattr(ts, 'tz') and ts.tz is not None:
                et = ts.tz_convert("US/Eastern")
            else:
                et = ts
        except Exception:
            continue

        et_t = et.time()
        # RTH only, skip OR (9:30-10:00)
        or_end = time(10, 0)
        if not (or_end <= et_t < time(16, 0)):
            continue

        close = float(row["close"])
        open_p = float(row["open"])
        low = float(row["low"])
        ema9 = float(row["EMA_9"])
        ema21 = float(row["EMA_21"])
        ema50 = float(row["EMA_50"])
        atr = float(row["ATR_14"])
        adx = float(row["ADX_14"])
        rsi = float(row["RSI_14"])
        macd = float(row["MACDhist"])

        if np.isnan(atr) or atr <= 0:
            continue

        # Signal C conditions
        if not (ema9 > ema21 > ema50): continue
        if low > ema9 * (1 + ema_touch_pct): continue
        if close <= ema9: continue
        if close <= open_p: continue
        if adx < 22 or adx > 35: continue
        if rsi > 70 or rsi < 40: continue
        if low <= ema21 * (1 + ema21_touch_pct): continue
        if macd <= 0: continue

        month = et.strftime("%Y-%m")
        hits.append(SignalCHit(bar_time=et, bar_idx=idx, close=close,
                                atr=atr, adx=adx, rsi=rsi, month=month))

    return hits


# ── Stop computation ─────────────────────────────────────────────────

def compute_stop_old(atr: float):
    """Old fixed 8pt stop."""
    return 8.0, 10.0

def compute_stop_new(atr: float, mult=1.0, floor=8.0, ceiling=20.0, rr=1.25):
    """A-Modified: SL = min(ceiling, max(floor, ATR × mult)), TP = SL × rr."""
    sl = min(ceiling, max(floor, atr * mult))
    tp = sl * rr
    return sl, tp


# ── Trade simulator ──────────────────────────────────────────────────

@dataclass
class Trade:
    hit: SignalCHit
    sl_pts: float
    tp_pts: float
    exit_reason: str
    pnl: float
    bars_held: int


def simulate(hits: List[SignalCHit], df: pd.DataFrame, stop_fn) -> List[Trade]:
    trades = []
    for hit in hits:
        sl_pts, tp_pts = stop_fn(hit.atr)
        sl_price = hit.close - sl_pts
        tp_price = hit.close + tp_pts
        entry = hit.close + SLIPPAGE_PTS

        exit_price = entry
        exit_reason = "TIMEOUT"
        bars_held = 0

        for j in range(hit.bar_idx + 1, min(hit.bar_idx + 1 + TIMEOUT_BARS, len(df))):
            bar = df.iloc[j]
            bars_held += 1
            if float(bar["low"]) <= sl_price:
                exit_price = sl_price - SLIPPAGE_PTS
                exit_reason = "STOP_LOSS"
                break
            if float(bar["high"]) >= tp_price:
                exit_price = tp_price - SLIPPAGE_PTS
                exit_reason = "PROFIT_TARGET"
                break

        pnl = (exit_price - entry) * POINT_VALUE - COMMISSION_RT
        trades.append(Trade(hit=hit, sl_pts=sl_pts, tp_pts=tp_pts,
                           exit_reason=exit_reason, pnl=pnl, bars_held=bars_held))
    return trades


# ── Analysis ─────────────────────────────────────────────────────────

def analyze_trades(trades: List[Trade], label: str) -> Dict:
    if not trades:
        return {"label": label, "n": 0}

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    total_pnl = sum(t.pnl for t in trades)
    wr = len(wins) / len(trades) * 100
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    expectancy = total_pnl / len(trades)

    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    equity = []
    running = 0.0
    for t in trades:
        running += t.pnl
        equity.append(running)
    peak, max_dd = 0.0, 0.0
    for e in equity:
        if e > peak: peak = e
        dd = peak - e
        if dd > max_dd: max_dd = dd

    # Avg R-multiple: pnl / risk_per_trade
    r_multiples = []
    for t in trades:
        risk = t.sl_pts * POINT_VALUE
        if risk > 0:
            r_multiples.append(t.pnl / risk)
    avg_r = np.mean(r_multiples) if r_multiples else 0

    return {
        "label": label,
        "n": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "wr": wr,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "total_pnl": total_pnl,
        "pf": pf,
        "max_dd": max_dd,
        "avg_r": avg_r,
        "avg_sl": np.mean([t.sl_pts for t in trades]),
        "avg_tp": np.mean([t.tp_pts for t in trades]),
    }


def print_results(old: Dict, new: Dict):
    print(f"\n{'='*75}")
    print(f"  EXTENDED BACKTEST — Signal C: Fixed 8pt vs ATR×1.0 Dynamic")
    print(f"{'='*75}")

    hdr = f"  {'Metric':<22} {'CURRENT (8pt)':>18} {'A-MODIFIED (ATR×1)':>20} {'Delta':>12}"
    print(hdr)
    print(f"  {'─'*22} {'─'*18} {'─'*20} {'─'*12}")

    metrics = [
        ("Trades", "n", "d", ""),
        ("Wins", "wins", "d", ""),
        ("Losses", "losses", "d", ""),
        ("Win Rate", "wr", ".1f", "%"),
        ("Avg Win", "avg_win", ".2f", "$"),
        ("Avg Loss", "avg_loss", ".2f", "$"),
        ("Expectancy/Trade", "expectancy", ".2f", "$"),
        ("Total P&L", "total_pnl", ".2f", "$"),
        ("Profit Factor", "pf", ".2f", ""),
        ("Max Drawdown", "max_dd", ".2f", "$"),
        ("Avg R-Multiple", "avg_r", ".3f", "R"),
        ("Avg SL (pts)", "avg_sl", ".1f", ""),
        ("Avg TP (pts)", "avg_tp", ".1f", ""),
    ]

    for name, key, fmt, sfx in metrics:
        ov = old.get(key, 0)
        nv = new.get(key, 0)
        delta = nv - ov

        if fmt == "d":
            row = f"  {name:<22} {int(ov):>18} {int(nv):>20} {int(delta):>+12}"
        elif sfx == "$":
            o_str = f"${ov:{fmt}}"
            n_str = f"${nv:{fmt}}"
            d_str = f"${delta:+{fmt}}"
            row = f"  {name:<22} {o_str:>18} {n_str:>20} {d_str:>12}"
        elif sfx == "%":
            o_str = f"{ov:{fmt}}%"
            n_str = f"{nv:{fmt}}%"
            d_str = f"{delta:+{fmt}}pp"
            row = f"  {name:<22} {o_str:>18} {n_str:>20} {d_str:>12}"
        else:
            o_str = f"{ov:{fmt}}"
            n_str = f"{nv:{fmt}}"
            d_str = f"{delta:+{fmt}}"
            row = f"  {name:<22} {o_str:>18} {n_str:>20} {d_str:>12}"
        print(row)


def print_atr_buckets(old_trades: List[Trade], new_trades: List[Trade]):
    print(f"\n{'='*75}")
    print(f"  PERFORMANCE BY ATR BUCKET")
    print(f"{'='*75}")

    buckets = [
        ("ATR < 10", 0, 10),
        ("ATR 10–14", 10, 14),
        ("ATR 14–18", 14, 18),
        ("ATR > 18", 18, 999),
    ]

    print(f"  {'Bucket':<12} {'N':>4}  {'Old WR':>8} {'New WR':>8}  {'Old Exp':>10} {'New Exp':>10}  {'Old PF':>7} {'New PF':>7}")
    print(f"  {'─'*12} {'─'*4}  {'─'*8} {'─'*8}  {'─'*10} {'─'*10}  {'─'*7} {'─'*7}")

    for label, lo, hi in buckets:
        ob = [t for t in old_trades if lo <= t.hit.atr < hi]
        nb = [t for t in new_trades if lo <= t.hit.atr < hi]
        n = len(ob)

        o_wr = (sum(1 for t in ob if t.pnl > 0) / len(ob) * 100) if ob else 0
        n_wr = (sum(1 for t in nb if t.pnl > 0) / len(nb) * 100) if nb else 0
        o_exp = (sum(t.pnl for t in ob) / len(ob)) if ob else 0
        n_exp = (sum(t.pnl for t in nb) / len(nb)) if nb else 0
        o_gw = sum(t.pnl for t in ob if t.pnl > 0)
        o_gl = abs(sum(t.pnl for t in ob if t.pnl <= 0))
        n_gw = sum(t.pnl for t in nb if t.pnl > 0)
        n_gl = abs(sum(t.pnl for t in nb if t.pnl <= 0))
        o_pf = o_gw / o_gl if o_gl > 0 else float('inf')
        n_pf = n_gw / n_gl if n_gl > 0 else float('inf')

        marker = "⬆" if n_exp > o_exp else "⬇" if n_exp < o_exp else "="
        print(f"  {label:<12} {n:>4}  {o_wr:>6.1f}% {n_wr:>6.1f}%  ${o_exp:>+8.2f} ${n_exp:>+8.2f}  {o_pf:>6.2f} {n_pf:>6.2f}  {marker}")


def print_monthly(old_trades: List[Trade], new_trades: List[Trade]):
    print(f"\n{'='*75}")
    print(f"  MONTHLY BREAKDOWN")
    print(f"{'='*75}")

    months = sorted(set(t.hit.month for t in old_trades + new_trades))
    print(f"  {'Month':<10} {'N':>4}  {'Old WR':>8} {'New WR':>8}  {'Old PnL':>10} {'New PnL':>10}  {'Δ PnL':>10}")
    print(f"  {'─'*10} {'─'*4}  {'─'*8} {'─'*8}  {'─'*10} {'─'*10}  {'─'*10}")

    for m in months:
        ob = [t for t in old_trades if t.hit.month == m]
        nb = [t for t in new_trades if t.hit.month == m]
        n = len(ob)
        if n == 0:
            continue

        o_wr = (sum(1 for t in ob if t.pnl > 0) / len(ob) * 100) if ob else 0
        n_wr = (sum(1 for t in nb if t.pnl > 0) / len(nb) * 100) if nb else 0
        o_pnl = sum(t.pnl for t in ob)
        n_pnl = sum(t.pnl for t in nb)
        delta = n_pnl - o_pnl

        print(f"  {m:<10} {n:>4}  {o_wr:>6.1f}% {n_wr:>6.1f}%  ${o_pnl:>+8.2f} ${n_pnl:>+8.2f}  ${delta:>+8.2f}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  EXTENDED BACKTEST: ATR-Adaptive Stops for Signal C")
    print("  A-Modified: SL = min(20, max(8, ATR × 1.0)), TP = SL × 1.25")
    print("="*60)

    print("\nLoading 15m data ...")
    df = load_15m_data()

    # Count trading days
    et_dates = set()
    for ts in df.index:
        try:
            et = ts.tz_convert("US/Eastern") if hasattr(ts, 'tz') and ts.tz else ts
            et_dates.add(et.date())
        except:
            pass
    total_days = len(et_dates)
    print(f"  Trading days: {total_days}")

    print("\nDetecting Signal C fires ...")
    hits = detect_signal_c(df)
    print(f"  Found {len(hits)} Signal C fires across {total_days} days")
    print(f"  Average {len(hits)/total_days:.2f} Signal C/day")

    if not hits:
        print("❌ No Signal C hits found.")
        return

    # ATR distribution
    atrs = [h.atr for h in hits]
    print(f"\n  ATR distribution:")
    print(f"    Mean: {np.mean(atrs):.2f}  Median: {np.median(atrs):.2f}  Min: {np.min(atrs):.2f}  Max: {np.max(atrs):.2f}")
    above_12 = sum(1 for a in atrs if a > 12) / len(atrs) * 100
    above_18 = sum(1 for a in atrs if a > 18) / len(atrs) * 100
    print(f"    ATR > 12: {above_12:.1f}%  ATR > 18: {above_18:.1f}%")

    # Run both strategies
    print("\n  Running CURRENT (fixed 8pt) ...")
    old_trades = simulate(hits, df, compute_stop_old)
    old_result = analyze_trades(old_trades, "CURRENT (8pt)")

    print("  Running A-MODIFIED (ATR×1.0) ...")
    new_trades = simulate(hits, df, compute_stop_new)
    new_result = analyze_trades(new_trades, "A-MODIFIED (ATR×1.0)")

    # Print results
    print_results(old_result, new_result)
    print_atr_buckets(old_trades, new_trades)
    print_monthly(old_trades, new_trades)

    # Summary
    print(f"\n{'='*75}")
    print(f"  VERDICT")
    print(f"{'='*75}")
    wr_delta = new_result["wr"] - old_result["wr"]
    exp_delta = new_result["expectancy"] - old_result["expectancy"]
    pf_delta = new_result["pf"] - old_result["pf"]
    dd_delta = new_result["max_dd"] - old_result["max_dd"]

    print(f"  Sample size: {old_result['n']} trades over {total_days} days ({'✅ sufficient' if old_result['n'] >= 30 else '⚠ small N, treat as directional'})")
    print(f"  Win Rate:    {wr_delta:+.1f}pp  {'✅ improved' if wr_delta > 0 else '⚠ degraded' if wr_delta < 0 else '= neutral'}")
    print(f"  Expectancy:  ${exp_delta:+.2f}/trade  {'✅ improved' if exp_delta > 0 else '⚠ degraded' if exp_delta < 0 else '= neutral'}")
    print(f"  Profit Factor: {pf_delta:+.2f}  {'✅ improved' if pf_delta > 0 else '⚠ degraded' if pf_delta < 0 else '= neutral'}")
    print(f"  Max Drawdown: ${dd_delta:+.2f}  {'⚠ worse' if dd_delta > 0 else '✅ better' if dd_delta < 0 else '= neutral'}")
    print(f"  Avg R-Multiple: {new_result['avg_r']:+.3f}R (old: {old_result['avg_r']:+.3f}R)")
    print()


if __name__ == "__main__":
    main()
