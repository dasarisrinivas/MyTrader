#!/usr/bin/env python3
"""
ATR-Adaptive Stop Simulation — Signal C Analysis
==================================================

Replays Oct 2025 historical data to compare 4 stop strategies for Signal C
(EMA9 Pullback Long):

  1. CURRENT   — Fixed 8pt SL / 10pt TP
  2. OPTION A  — Dynamic: SL = max(8, ATR × 0.75),  TP = SL × 1.25 (preserve R:R)
  3. OPTION B  — Volatility tiering: ATR≤12 → 8pt, 12<ATR≤18 → 11pt, ATR>18 → skip
  4. ATR GUARD — Skip when ATR > 12 (agent suggestion)

Reports:
  - ATR distribution for Signal C trades
  - Win rate, expectancy, trades/day for each option
  - Side-by-side comparison table

Usage:
    python3 tools/simulate_atr_stops.py
"""

from __future__ import annotations
import sys, os
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import pandas as pd
import numpy as np

# ── Add project root to path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shree.features.feature_engineer import _ema, _atr, _adx, _rsi

# ── MES constants ────────────────────────────────────────────────────
POINT_VALUE = 5.0       # $5 per point
SLIPPAGE_PTS = 0.50     # 2 ticks
COMMISSION_RT = 2.48    # Round-trip commission
TIMEOUT_BARS = 20       # 5 hours @ 15m


# ── Data loading ─────────────────────────────────────────────────────

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Load 1m CSV, resample to 15m, add indicators."""
    # Try 'timestamp' column first, fall back to 'date'
    raw = pd.read_csv(csv_path)
    date_col = "timestamp" if "timestamp" in raw.columns else "date"
    df = pd.read_csv(csv_path, parse_dates=[date_col], index_col=date_col)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Resample 1m → 15m
    ohlcv = df[["open", "high", "low", "close", "volume"]].resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()

    # Add indicators
    c, h, lo = ohlcv["close"], ohlcv["high"], ohlcv["low"]
    ohlcv["EMA_9"] = _ema(c, 9)
    ohlcv["EMA_21"] = _ema(c, 21)
    ohlcv["EMA_50"] = _ema(c, 50)
    ohlcv["RSI_14"] = _rsi(c, 14)
    ohlcv["ATR_14"] = _atr(h, lo, c, 14)
    ohlcv["ADX_14"] = _adx(h, lo, c, 14)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    ohlcv["MACDhist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    # Forward-fill NaN
    for col in ["EMA_9", "EMA_21", "EMA_50", "RSI_14", "ATR_14", "ADX_14", "MACDhist"]:
        ohlcv[col] = ohlcv[col].ffill().bfill()

    return ohlcv


# ── Signal C detector ────────────────────────────────────────────────

@dataclass
class SignalCHit:
    """A raw Signal C detection (before SL/TP applied)."""
    bar_time: Any
    bar_idx: int
    close: float
    atr: float
    adx: float
    rsi: float


def detect_all_signal_c(df: pd.DataFrame) -> List[SignalCHit]:
    """Walk through 15m bars and detect every Signal C (EMA9 Pullback Long) firing."""
    hits: List[SignalCHit] = []
    ema_touch_pct = 0.0015      # Current config value
    ema21_touch_pct = 0.0015    # Signal A overlap exclusion
    adx_min, adx_max = 22, 35
    rsi_lo, rsi_hi = 40, 70

    session_date = None

    for idx in range(60, len(df)):
        row = df.iloc[idx]
        ts = df.index[idx]

        # Convert to ET
        try:
            et = ts.tz_convert("US/Eastern") if ts.tz else ts
        except Exception:
            continue

        et_date = et.date()
        et_t = et.time()

        # RTH only (9:30–16:00 ET), skip OR (first 30m)
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

        # ── Signal C conditions (exact mirror of es_fifteen_min.py) ──
        # 1. Strong uptrend
        if not (ema9 > ema21 > ema50):
            continue
        # 2. Low touches EMA9
        touch_thresh = ema9 * (1 + ema_touch_pct)
        if low > touch_thresh:
            continue
        # 3. Close above EMA9
        if close <= ema9:
            continue
        # 4. Bullish bar
        if close <= open_p:
            continue
        # 5. ADX filter
        if adx < adx_min or adx > adx_max:
            continue
        # 6. RSI filter
        if rsi > rsi_hi or rsi < rsi_lo:
            continue
        # 7. No overlap with Signal A
        ema21_touch = ema21 * (1 + ema21_touch_pct)
        if low <= ema21_touch:
            continue
        # 8. MACD histogram positive
        if macd <= 0:
            continue

        hits.append(SignalCHit(
            bar_time=et, bar_idx=idx, close=close,
            atr=atr, adx=adx, rsi=rsi,
        ))

    return hits


# ── Stop strategy definitions ────────────────────────────────────────

@dataclass
class StopResult:
    sl_points: float
    tp_points: float
    skipped: bool = False


def stop_current(atr: float) -> StopResult:
    """Option 0: Current fixed 8pt SL / 10pt TP."""
    return StopResult(sl_points=8.0, tp_points=10.0)


def stop_dynamic(atr: float) -> StopResult:
    """Option A: Dynamic — SL = max(8, ATR × 0.75), TP scales to preserve ~1.25 R:R."""
    sl = max(8.0, atr * 0.75)
    tp = sl * 1.25  # Preserve 1.25:1 reward:risk ratio
    return StopResult(sl_points=sl, tp_points=tp)


def stop_tiered(atr: float) -> StopResult:
    """Option B: Volatility tiering — ATR≤12→8pt, 12<ATR≤18→11pt, ATR>18→skip."""
    if atr <= 12:
        return StopResult(sl_points=8.0, tp_points=10.0)
    elif atr <= 18:
        return StopResult(sl_points=11.0, tp_points=13.75)  # R:R 1.25
    else:
        return StopResult(sl_points=0, tp_points=0, skipped=True)


def stop_atr_guard(atr: float) -> StopResult:
    """Agent suggestion: Skip when ATR > 12 (fixed stop below that)."""
    if atr > 12:
        return StopResult(sl_points=0, tp_points=0, skipped=True)
    return StopResult(sl_points=8.0, tp_points=10.0)


STRATEGIES = {
    "CURRENT (Fixed 8pt)":           stop_current,
    "OPTION A (Dynamic ATR×0.75)":   stop_dynamic,
    "OPTION B (Vol Tiering)":        stop_tiered,
    "ATR GUARD (Skip ATR>12)":       stop_atr_guard,
}


# ── Trade simulator ──────────────────────────────────────────────────

@dataclass
class SimTrade:
    hit: SignalCHit
    sl_points: float
    tp_points: float
    exit_price: float
    exit_reason: str      # PROFIT_TARGET / STOP_LOSS / TIMEOUT / SKIPPED
    pnl: float
    bars_held: int
    atr_at_entry: float


def simulate_strategy(
    hits: List[SignalCHit],
    df: pd.DataFrame,
    stop_fn,
) -> List[SimTrade]:
    """Walk-forward SL/TP resolution for each Signal C hit under a given stop strategy."""
    trades: List[SimTrade] = []

    for hit in hits:
        sr = stop_fn(hit.atr)

        if sr.skipped:
            trades.append(SimTrade(
                hit=hit, sl_points=0, tp_points=0,
                exit_price=hit.close, exit_reason="SKIPPED",
                pnl=0.0, bars_held=0, atr_at_entry=hit.atr,
            ))
            continue

        sl_price = hit.close - sr.sl_points
        tp_price = hit.close + sr.tp_points
        entry = hit.close + SLIPPAGE_PTS  # Slippage on entry (buy higher)

        exit_price = entry
        exit_reason = "TIMEOUT"
        bars_held = 0

        for j in range(hit.bar_idx + 1, min(hit.bar_idx + 1 + TIMEOUT_BARS, len(df))):
            bar = df.iloc[j]
            bars_held += 1
            bar_low = float(bar["low"])
            bar_high = float(bar["high"])

            # Check SL first (worst-case)
            if bar_low <= sl_price:
                exit_price = sl_price - SLIPPAGE_PTS
                exit_reason = "STOP_LOSS"
                break
            if bar_high >= tp_price:
                exit_price = tp_price - SLIPPAGE_PTS  # Conservative fill
                exit_reason = "PROFIT_TARGET"
                break

        pnl = (exit_price - entry) * POINT_VALUE - COMMISSION_RT

        trades.append(SimTrade(
            hit=hit, sl_points=sr.sl_points, tp_points=sr.tp_points,
            exit_price=exit_price, exit_reason=exit_reason,
            pnl=pnl, bars_held=bars_held, atr_at_entry=hit.atr,
        ))

    return trades


# ── Analysis ─────────────────────────────────────────────────────────

def analyze(trades: List[SimTrade], label: str, total_days: int) -> Dict:
    executed = [t for t in trades if t.exit_reason != "SKIPPED"]
    skipped = [t for t in trades if t.exit_reason == "SKIPPED"]
    wins = [t for t in executed if t.pnl > 0]
    losses = [t for t in executed if t.pnl <= 0]

    total_pnl = sum(t.pnl for t in executed)
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    win_rate = len(wins) / len(executed) * 100 if executed else 0

    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Expectancy per trade
    expectancy = total_pnl / len(executed) if executed else 0

    # Win rate by exit type
    sl_count = len([t for t in executed if t.exit_reason == "STOP_LOSS"])
    tp_count = len([t for t in executed if t.exit_reason == "PROFIT_TARGET"])
    to_count = len([t for t in executed if t.exit_reason == "TIMEOUT"])

    # Average SL/TP points
    avg_sl = np.mean([t.sl_points for t in executed]) if executed else 0
    avg_tp = np.mean([t.tp_points for t in executed]) if executed else 0

    # Max drawdown
    equity = []
    running = 0.0
    for t in executed:
        running += t.pnl
        equity.append(running)
    peak, max_dd = 0.0, 0.0
    for e in equity:
        if e > peak: peak = e
        dd = peak - e
        if dd > max_dd: max_dd = dd

    return {
        "label": label,
        "total_signals": len(trades),
        "skipped": len(skipped),
        "executed": len(executed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "total_pnl": total_pnl,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "trades_per_day": len(executed) / total_days if total_days > 0 else 0,
        "sl_exits": sl_count,
        "tp_exits": tp_count,
        "timeout_exits": to_count,
        "avg_sl_pts": avg_sl,
        "avg_tp_pts": avg_tp,
    }


# ── Printing ─────────────────────────────────────────────────────────

def print_atr_distribution(hits: List[SignalCHit]):
    """Print ATR distribution analysis for Signal C trades."""
    if not hits:
        print("  ❌ No Signal C hits found.")
        return

    atrs = [h.atr for h in hits]
    print(f"\n{'='*70}")
    print(f"  ATR DISTRIBUTION — Signal C (EMA9 Pullback Long)")
    print(f"{'='*70}")
    print(f"  Total Signal C fires:  {len(hits)}")
    print(f"  ATR mean:              {np.mean(atrs):.2f}")
    print(f"  ATR median:            {np.median(atrs):.2f}")
    print(f"  ATR min:               {np.min(atrs):.2f}")
    print(f"  ATR max:               {np.max(atrs):.2f}")
    print(f"  ATR std:               {np.std(atrs):.2f}")

    # Histogram buckets
    buckets = [
        ("ATR ≤  6", 0, 6),
        ("ATR  6–8", 6, 8),
        ("ATR  8–10", 8, 10),
        ("ATR 10–12", 10, 12),
        ("ATR 12–14", 12, 14),
        ("ATR 14–16", 14, 16),
        ("ATR 16–18", 16, 18),
        ("ATR 18–20", 18, 20),
        ("ATR > 20", 20, 999),
    ]

    print(f"\n  {'Bucket':<14} {'Count':>6} {'%':>7}  Bar")
    print(f"  {'─'*14} {'─'*6} {'─'*7}  {'─'*30}")
    for label, lo, hi in buckets:
        count = sum(1 for a in atrs if lo < a <= hi) if lo > 0 else sum(1 for a in atrs if a <= hi)
        pct = count / len(atrs) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<14} {count:>6} {pct:>6.1f}%  {bar}")

    # Key thresholds
    above_8 = sum(1 for a in atrs if a > 8) / len(atrs) * 100
    above_10 = sum(1 for a in atrs if a > 10) / len(atrs) * 100
    above_12 = sum(1 for a in atrs if a > 12) / len(atrs) * 100
    above_14 = sum(1 for a in atrs if a > 14) / len(atrs) * 100
    above_18 = sum(1 for a in atrs if a > 18) / len(atrs) * 100

    print(f"\n  Key thresholds:")
    print(f"  ATR >  8:  {above_8:5.1f}%  ({sum(1 for a in atrs if a >  8)}/{len(atrs)} trades)")
    print(f"  ATR > 10:  {above_10:5.1f}%  ({sum(1 for a in atrs if a > 10)}/{len(atrs)} trades)")
    print(f"  ATR > 12:  {above_12:5.1f}%  ({sum(1 for a in atrs if a > 12)}/{len(atrs)} trades)  ← agent guard threshold")
    print(f"  ATR > 14:  {above_14:5.1f}%  ({sum(1 for a in atrs if a > 14)}/{len(atrs)} trades)  ← today's loss ATR")
    print(f"  ATR > 18:  {above_18:5.1f}%  ({sum(1 for a in atrs if a > 18)}/{len(atrs)} trades)")

    # Stop/ATR ratio analysis
    print(f"\n  Stop/ATR ratio with 8pt fixed stop:")
    for label, lo, hi in [("ATR ≤ 8", 0, 8), ("ATR 8-12", 8, 12), ("ATR 12-16", 12, 16), ("ATR > 16", 16, 999)]:
        bucket_atrs = [a for a in atrs if (lo < a <= hi) if lo > 0] if lo > 0 else [a for a in atrs if a <= hi]
        if bucket_atrs:
            avg_ratio = np.mean([8.0 / a for a in bucket_atrs])
            print(f"  {label:<12}  avg ratio = {avg_ratio:.2f}×  {'✓ safe' if avg_ratio >= 1.0 else '⚠ noise zone' if avg_ratio >= 0.7 else '✗ dangerous'}")


def print_comparison(results: List[Dict]):
    """Print side-by-side comparison of all strategies."""
    print(f"\n{'='*90}")
    print(f"  STRATEGY COMPARISON — Signal C Stop Variants")
    print(f"{'='*90}")

    # Header
    labels = [r["label"] for r in results]
    hdr = f"  {'Metric':<22}"
    for lbl in labels:
        hdr += f" {lbl:>15}"
    print(hdr)
    print(f"  {'─'*22}" + "─" * (16 * len(labels)))

    metrics = [
        ("Total Signals", "total_signals", "d"),
        ("Skipped", "skipped", "d"),
        ("Executed", "executed", "d"),
        ("Wins", "wins", "d"),
        ("Losses", "losses", "d"),
        ("Win Rate", "win_rate", ".1f", "%"),
        ("Avg Win", "avg_win", ".2f", "$"),
        ("Avg Loss", "avg_loss", ".2f", "$"),
        ("Expectancy/Trade", "expectancy", ".2f", "$"),
        ("Total P&L", "total_pnl", ".2f", "$"),
        ("Profit Factor", "profit_factor", ".2f", ""),
        ("Max Drawdown", "max_drawdown", ".2f", "$"),
        ("Trades/Day", "trades_per_day", ".2f", ""),
        ("SL Exits", "sl_exits", "d"),
        ("TP Exits", "tp_exits", "d"),
        ("Timeout Exits", "timeout_exits", "d"),
        ("Avg SL (pts)", "avg_sl_pts", ".1f", ""),
        ("Avg TP (pts)", "avg_tp_pts", ".1f", ""),
    ]

    for m in metrics:
        name, key, fmt = m[0], m[1], m[2]
        suffix = m[3] if len(m) > 3 else ""
        row = f"  {name:<22}"
        for r in results:
            val = r[key]
            if fmt == "d":
                row += f" {int(val):>15}"
            elif suffix == "$":
                row += f" {'$' + format(val, fmt):>15}"
            elif suffix == "%":
                row += f" {format(val, fmt) + '%':>15}"
            else:
                row += f" {format(val, fmt):>15}"
        print(row)

    # ── Highlight best ──
    print(f"\n  {'─'*70}")
    best_wr = max(results, key=lambda r: r["win_rate"] if r["executed"] > 0 else 0)
    best_exp = max(results, key=lambda r: r["expectancy"])
    best_pnl = max(results, key=lambda r: r["total_pnl"])
    best_pf = max(results, key=lambda r: r["profit_factor"] if r["profit_factor"] != float('inf') else 0)

    print(f"  🏆 Best Win Rate:    {best_wr['label']}  ({best_wr['win_rate']:.1f}%)")
    print(f"  🏆 Best Expectancy:  {best_exp['label']}  (${best_exp['expectancy']:.2f}/trade)")
    print(f"  🏆 Best Total P&L:   {best_pnl['label']}  (${best_pnl['total_pnl']:.2f})")
    print(f"  🏆 Best Profit Factor: {best_pf['label']}  ({best_pf['profit_factor']:.2f})")


def print_per_trade_detail(trades: List[SimTrade], label: str, max_show: int = 10):
    """Show individual trade detail for inspection."""
    executed = [t for t in trades if t.exit_reason != "SKIPPED"]
    if not executed:
        print(f"\n  [{label}] No executed trades.")
        return

    print(f"\n  [{label}] — Sample Trades (first {min(max_show, len(executed))})")
    print(f"  {'Time':<22} {'Entry':>8} {'SL':>8} {'TP':>8} {'ATR':>6} {'Exit':>10} {'P&L':>8} {'Bars':>5}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*10} {'─'*8} {'─'*5}")

    for t in executed[:max_show]:
        sl_price = t.hit.close - t.sl_points
        tp_price = t.hit.close + t.tp_points
        t_str = str(t.hit.bar_time)[:19]
        print(f"  {t_str:<22} {t.hit.close:>8.2f} {sl_price:>8.2f} {tp_price:>8.2f} {t.atr_at_entry:>6.1f} {t.exit_reason:>10} {t.pnl:>+8.2f} {t.bars_held:>5}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    csv_path = ROOT / "data" / "es_october_2025.csv"
    if not csv_path.exists():
        # Try alternative
        csv_path = ROOT / "data" / "es_historical.csv"
    if not csv_path.exists():
        print("❌ No historical data found in data/")
        sys.exit(1)

    print(f"Loading data from {csv_path.name} ...")
    df = load_and_prepare(str(csv_path))
    print(f"  15m bars: {len(df)}")

    # Count trading days
    et_dates = set()
    for ts in df.index:
        try:
            et = ts.tz_convert("US/Eastern") if ts.tz else ts
            et_dates.add(et.date())
        except:
            pass
    total_days = len(et_dates)
    print(f"  Trading days: {total_days}")

    # ── Detect all Signal C hits ──
    print("\nDetecting Signal C (EMA9 Pullback Long) fires ...")
    hits = detect_all_signal_c(df)
    print(f"  Found {len(hits)} Signal C fires")

    if not hits:
        print("❌ No Signal C hits in this dataset. Cannot compare.")
        sys.exit(0)

    # ── ATR Distribution ──
    print_atr_distribution(hits)

    # ── Run all 4 stop strategies ──
    all_results: List[Dict] = []
    all_trades: Dict[str, List[SimTrade]] = {}

    for label, stop_fn in STRATEGIES.items():
        print(f"\n  Simulating: {label} ...")
        trades = simulate_strategy(hits, df, stop_fn)
        result = analyze(trades, label, total_days)
        all_results.append(result)
        all_trades[label] = trades

    # ── Comparison table ──
    print_comparison(all_results)

    # ── Per-trade detail for each strategy ──
    for label, trades in all_trades.items():
        print_per_trade_detail(trades, label, max_show=8)

    # ── Win rate by ATR bucket for current vs dynamic ──
    print(f"\n{'='*70}")
    print(f"  WIN RATE BY ATR BUCKET — Current vs Option A (Dynamic)")
    print(f"{'='*70}")

    current_trades = all_trades["CURRENT (Fixed 8pt)"]
    dynamic_trades = all_trades["OPTION A (Dynamic ATR×0.75)"]

    atr_buckets = [
        ("ATR ≤  8", 0, 8),
        ("ATR  8–12", 8, 12),
        ("ATR 12–16", 12, 16),
        ("ATR > 16", 16, 999),
    ]

    print(f"  {'Bucket':<14} {'Current WR':>12} {'Dynamic WR':>12} {'Current PnL':>13} {'Dynamic PnL':>13} {'N':>4}")
    print(f"  {'─'*14} {'─'*12} {'─'*12} {'─'*13} {'─'*13} {'─'*4}")

    for bucket_label, lo_atr, hi_atr in atr_buckets:
        c_bucket = [t for t in current_trades
                     if t.exit_reason != "SKIPPED" and lo_atr < t.atr_at_entry <= hi_atr]
        d_bucket = [t for t in dynamic_trades
                     if t.exit_reason != "SKIPPED" and lo_atr < t.atr_at_entry <= hi_atr]

        c_wr = (sum(1 for t in c_bucket if t.pnl > 0) / len(c_bucket) * 100) if c_bucket else 0
        d_wr = (sum(1 for t in d_bucket if t.pnl > 0) / len(d_bucket) * 100) if d_bucket else 0
        c_pnl = sum(t.pnl for t in c_bucket)
        d_pnl = sum(t.pnl for t in d_bucket)
        n = len(c_bucket)

        delta_wr = d_wr - c_wr
        delta_marker = "⬆" if delta_wr > 0 else "⬇" if delta_wr < 0 else "="

        print(f"  {bucket_label:<14} {c_wr:>10.1f}% {d_wr:>10.1f}% {c_pnl:>+12.2f}$ {d_pnl:>+12.2f}$ {n:>4}  {delta_marker}")

    # ── Recommendation ──
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"{'='*70}")

    # Find best by expectancy among the realistic options (exclude agent guard if it kills too many)
    current = next(r for r in all_results if "CURRENT" in r["label"])
    opt_a = next(r for r in all_results if "OPTION A" in r["label"])
    opt_b = next(r for r in all_results if "OPTION B" in r["label"])
    guard = next(r for r in all_results if "ATR GUARD" in r["label"])

    print(f"\n  Current fixed 8pt stop:")
    print(f"    WR={current['win_rate']:.1f}%  Exp=${current['expectancy']:.2f}  PF={current['profit_factor']:.2f}  Trades/day={current['trades_per_day']:.2f}")

    print(f"\n  Option A — Dynamic max(8, ATR×0.75):")
    exp_delta = opt_a['expectancy'] - current['expectancy']
    freq_delta = opt_a['trades_per_day'] - current['trades_per_day']
    print(f"    WR={opt_a['win_rate']:.1f}%  Exp=${opt_a['expectancy']:.2f} ({'+' if exp_delta >= 0 else ''}{exp_delta:.2f})  PF={opt_a['profit_factor']:.2f}  Trades/day={opt_a['trades_per_day']:.2f} ({'+' if freq_delta >= 0 else ''}{freq_delta:.2f})")

    print(f"\n  Option B — Volatility tiering:")
    exp_delta_b = opt_b['expectancy'] - current['expectancy']
    freq_delta_b = opt_b['trades_per_day'] - current['trades_per_day']
    print(f"    WR={opt_b['win_rate']:.1f}%  Exp=${opt_b['expectancy']:.2f} ({'+' if exp_delta_b >= 0 else ''}{exp_delta_b:.2f})  PF={opt_b['profit_factor']:.2f}  Trades/day={opt_b['trades_per_day']:.2f} ({'+' if freq_delta_b >= 0 else ''}{freq_delta_b:.2f})")

    print(f"\n  ATR Guard (skip ATR>12):")
    exp_delta_g = guard['expectancy'] - current['expectancy']
    freq_delta_g = guard['trades_per_day'] - current['trades_per_day']
    print(f"    WR={guard['win_rate']:.1f}%  Exp=${guard['expectancy']:.2f} ({'+' if exp_delta_g >= 0 else ''}{exp_delta_g:.2f})  PF={guard['profit_factor']:.2f}  Trades/day={guard['trades_per_day']:.2f} ({'+' if freq_delta_g >= 0 else ''}{freq_delta_g:.2f})")

    # Automatic recommendation
    candidates = [
        ("OPTION A", opt_a),
        ("OPTION B", opt_b),
        ("ATR GUARD", guard),
    ]
    # Score: expectancy improvement + frequency preservation
    best_name, best = max(candidates, key=lambda x: x[1]["expectancy"] * (1 + x[1]["trades_per_day"]))
    print(f"\n  → Recommended: {best_name}")
    print(f"    Rationale: Best balance of expectancy (${best['expectancy']:.2f}/trade) × frequency ({best['trades_per_day']:.2f}/day)")

    # ── Sensitivity sweep: ATR multipliers for dynamic stop ──
    print(f"\n{'='*70}")
    print(f"  SENSITIVITY SWEEP — Dynamic Stop ATR Multipliers")
    print(f"{'='*70}")
    print(f"  {'Multiplier':<12} {'SL Formula':<24} {'WR':>7} {'Exp/Trade':>11} {'PF':>6} {'PnL':>10} {'N':>4}")
    print(f"  {'─'*12} {'─'*24} {'─'*7} {'─'*11} {'─'*6} {'─'*10} {'─'*4}")

    for mult in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00, 1.10, 1.20]:
        def _make_fn(m):
            def fn(atr):
                sl = max(8.0, atr * m)
                tp = sl * 1.25
                return StopResult(sl_points=sl, tp_points=tp)
            return fn
        trades = simulate_strategy(hits, df, _make_fn(mult))
        r = analyze(trades, f"ATR×{mult}", total_days)
        print(f"  ATR × {mult:<5.2f}  SL=max(8, ATR×{mult:.2f})      {r['win_rate']:>5.1f}%  ${r['expectancy']:>+9.2f}  {r['profit_factor']:>5.2f}  ${r['total_pnl']:>+8.2f}  {r['executed']:>4}")

    # ── Sensitivity sweep: Tiering thresholds ──
    print(f"\n{'='*70}")
    print(f"  SENSITIVITY SWEEP — Tiering Thresholds")
    print(f"{'='*70}")
    print(f"  {'Config':<30} {'WR':>7} {'Exp/Trade':>11} {'PF':>6} {'PnL':>10} {'N':>4} {'Skip':>5}")
    print(f"  {'─'*30} {'─'*7} {'─'*11} {'─'*6} {'─'*10} {'─'*4} {'─'*5}")

    tier_configs = [
        ("≤12→8, 12-16→10, >16→skip", 12, 16, 8, 10, 10, 12.5),
        ("≤12→8, 12-18→11, >18→skip", 12, 18, 8, 10, 11, 13.75),
        ("≤14→8, 14-20→11, >20→skip", 14, 20, 8, 10, 11, 13.75),
        ("≤10→8, 10-16→10, >16→skip", 10, 16, 8, 10, 10, 12.5),
        ("≤12→8, 12-18→12, >18→skip", 12, 18, 8, 10, 12, 15),
        ("≤14→8, >14→12, no skip",     14, 999, 8, 10, 12, 15),
        ("≤16→8, >16→12, no skip",     16, 999, 8, 10, 12, 15),
    ]

    for label, t1, t2, sl1, tp1, sl2, tp2 in tier_configs:
        def _make_tier(t1, t2, sl1, tp1, sl2, tp2):
            def fn(atr):
                if atr <= t1:
                    return StopResult(sl_points=sl1, tp_points=tp1)
                elif atr <= t2:
                    return StopResult(sl_points=sl2, tp_points=tp2)
                else:
                    return StopResult(sl_points=0, tp_points=0, skipped=True)
            return fn
        trades = simulate_strategy(hits, df, _make_tier(t1, t2, sl1, tp1, sl2, tp2))
        r = analyze(trades, label, total_days)
        sk = r['skipped']
        print(f"  {label:<30} {r['win_rate']:>5.1f}%  ${r['expectancy']:>+9.2f}  {r['profit_factor']:>5.2f}  ${r['total_pnl']:>+8.2f}  {r['executed']:>4}  {sk:>4}")

    print()


if __name__ == "__main__":
    main()
