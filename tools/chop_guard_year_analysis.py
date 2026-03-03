#!/usr/bin/env python3
"""
CHOP Guard — 1-Year Direction Asymmetry Analysis
=================================================

Expands the 14-trade live analysis (Feb 25 – Mar 2 2026) to the full
1-year 15m dataset (Dec 2024 – Jan 2026, 25,646 bars).

Methodology:
1. Load 15m OHLCV data
2. Run feature engineering (EMA9/21/50, RSI, ADX, MACD, ATR)
3. Run EsFifteenMinStrategy.generate() on every bar to get signals A–F
4. Compute hybrid pipeline trend label (CHOP/UPTREND/etc.) at each bar
5. Identify signals that WOULD have been CHOP-blocked
6. Simulate each blocked trade using forward bars (SL/TP hit check)
7. Report LONG vs SHORT outcomes, monthly breakdown, statistical tests

Data source: data/ib/ES_15m_1y.parquet (25,646 bars)
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shree.features.feature_engineer import add_technical_indicators
from shree.config import OneMinuteStrategyConfig
from shree.strategies.es_fifteen_min import EsFifteenMinStrategy
from shree.strategies.base import Signal


# ──────────────────────────────────────────────────────────────
# 1. Hybrid trend label (replicated from hybrid_rag_pipeline.py
#    lines 350–473, without needing IBKR connection)
# ──────────────────────────────────────────────────────────────

def compute_hybrid_trend(row: pd.Series, chop_ema_spread_min_pct: float = 0.0005) -> Tuple[str, float]:
    """
    Replicate the hybrid pipeline's trend detection logic.
    Returns (trend_label, trend_score).
    """
    price = row["close"]
    ema_9 = row["EMA_9"]
    ema_20 = row["EMA_20"]
    ema_50 = row["EMA_50"]
    rsi = row["RSI_14"]
    macd_hist = row["MACDhist_12_26_9"]
    adx = row["ADX_14"]

    # EMA spread check → CHOP_RANGE
    ema_spread = abs(ema_9 - ema_50) / ema_50 if ema_50 > 0 else 0
    if ema_spread < chop_ema_spread_min_pct:
        return "CHOP_RANGE", 0.0

    is_trending = adx > 20

    # EMA relationships
    ema_diff_pct = (ema_9 - ema_20) / ema_20 * 100 if ema_20 > 0 else 0
    price_vs_ema20_pct = (price - ema_20) / ema_20 * 100 if ema_20 > 0 else 0
    price_vs_ema50_pct = (price - ema_50) / ema_50 * 100 if ema_50 > 0 else 0

    macd_bullish = macd_hist > 0
    macd_bearish = macd_hist < 0
    rsi_extreme_bullish = rsi > 60
    rsi_extreme_bearish = rsi < 40
    rsi_bullish = rsi > 50
    rsi_bearish = rsi < 50

    # Multi-factor trend scoring
    trend_score = 0.0

    # Factor 1: EMA alignment (40%)
    if price > ema_9 > ema_20 > ema_50:
        trend_score += 40
    elif price < ema_9 < ema_20 < ema_50:
        trend_score -= 40
    elif price > ema_20 and ema_diff_pct > 0:
        trend_score += 20
    elif price < ema_20 and ema_diff_pct < 0:
        trend_score -= 20

    # Factor 2: EMA_50 anchor (20%)
    if price_vs_ema50_pct > 0.1:
        trend_score += 20
    elif price_vs_ema50_pct < -0.1:
        trend_score -= 20

    # Factor 3: MACD momentum (15%)
    if macd_bullish:
        trend_score += 15
    elif macd_bearish:
        trend_score -= 15

    # Factor 4: RSI position (10%)
    if rsi_extreme_bullish:
        trend_score += 10
    elif rsi_extreme_bearish:
        trend_score -= 10
    elif rsi_bullish:
        trend_score += 5
    elif rsi_bearish:
        trend_score -= 5

    # Factor 5 & 6: HTF trend and sentiment (15%)
    # Not available offline — skip (conservative: makes more things CHOP)

    # Determine trend label
    if trend_score >= 60 and is_trending:
        return "UPTREND", trend_score
    elif trend_score <= -60 and is_trending:
        return "DOWNTREND", trend_score
    elif trend_score >= 30 and is_trending:
        return "MICRO_UP", trend_score
    elif trend_score <= -30 and is_trending:
        return "MICRO_DOWN", trend_score
    elif trend_score >= 10 and is_trending:
        return "WEAK_UP", trend_score
    elif trend_score <= -10 and is_trending:
        return "WEAK_DOWN", trend_score
    elif abs(ema_diff_pct) < 0.02 and abs(price_vs_ema20_pct) < 0.05:
        return "RANGE", trend_score
    else:
        return "CHOP", trend_score


# ──────────────────────────────────────────────────────────────
# 2. Trade simulation
# ──────────────────────────────────────────────────────────────

@dataclass
class SimTrade:
    """A simulated trade from a CHOP-blocked signal."""
    bar_idx: int
    timestamp: str
    signal_type: str       # e.g. "EMA21_PB_LONG", "TREND_CONT_SHORT"
    action: str            # BUY or SELL
    direction: str         # LONG or SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    adx: float
    rsi: float
    trend_score: float
    trend_label: str
    ema_spread: float
    # Outcome (filled by simulation)
    outcome: str = ""       # TP_HIT, SL_HIT, TIMEOUT
    exit_price: float = 0.0
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    bars_held: int = 0
    month: str = ""


def simulate_trade(trade: SimTrade, df: pd.DataFrame, max_hold_bars: int = 6) -> SimTrade:
    """Walk forward through bars to determine outcome."""
    start_idx = trade.bar_idx + 1  # Next bar after signal
    end_idx = min(start_idx + max_hold_bars, len(df))

    for i in range(start_idx, end_idx):
        bar = df.iloc[i]
        trade.bars_held = i - trade.bar_idx

        if trade.direction == "LONG":
            # Check SL first (worst case), then TP
            if bar["low"] <= trade.stop_loss:
                trade.outcome = "SL_HIT"
                trade.exit_price = trade.stop_loss
                break
            if bar["high"] >= trade.take_profit:
                trade.outcome = "TP_HIT"
                trade.exit_price = trade.take_profit
                break
        else:  # SHORT
            if bar["high"] >= trade.stop_loss:
                trade.outcome = "SL_HIT"
                trade.exit_price = trade.stop_loss
                break
            if bar["low"] <= trade.take_profit:
                trade.outcome = "TP_HIT"
                trade.exit_price = trade.take_profit
                break

    if not trade.outcome:
        trade.outcome = "TIMEOUT"
        if start_idx < len(df):
            last_idx = min(end_idx - 1, len(df) - 1)
            trade.exit_price = df.iloc[last_idx]["close"]
        else:
            trade.exit_price = trade.entry_price
        trade.bars_held = max_hold_bars

    # Calculate P&L
    if trade.direction == "LONG":
        trade.pnl_points = trade.exit_price - trade.entry_price
    else:
        trade.pnl_points = trade.entry_price - trade.exit_price
    trade.pnl_dollars = trade.pnl_points * 5.0  # MES = $5/pt

    return trade


# ──────────────────────────────────────────────────────────────
# 3. Main analysis
# ──────────────────────────────────────────────────────────────

def run_analysis():
    data_path = Path("data/ib/ES_15m_1y.parquet")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        sys.exit(1)

    print("=" * 70)
    print("CHOP GUARD — 1-YEAR DIRECTION ASYMMETRY ANALYSIS")
    print("=" * 70)

    # Load data
    df = pd.read_parquet(data_path)
    print(f"\nData: {len(df)} bars, {df.index[0]} → {df.index[-1]}")

    # Add technical indicators
    df = add_technical_indicators(df)
    print(f"Indicators computed: {[c for c in df.columns if c.startswith(('EMA_', 'RSI_', 'ADX_', 'MACD', 'ATR'))]}")

    # Compute hybrid trend label for every bar
    trends = []
    scores = []
    ema_spreads = []
    for i in range(len(df)):
        row = df.iloc[i]
        # Skip warmup period (need 50 bars for EMA_50)
        if i < 60:
            trends.append("WARMUP")
            scores.append(0.0)
            ema_spreads.append(0.0)
            continue
        trend, score = compute_hybrid_trend(row)
        trends.append(trend)
        scores.append(score)
        ema_9 = row["EMA_9"]
        ema_50 = row["EMA_50"]
        ema_spreads.append(abs(ema_9 - ema_50) / ema_50 if ema_50 > 0 else 0)

    df["hybrid_trend"] = trends
    df["trend_score"] = scores
    df["ema_spread"] = ema_spreads

    # Trend distribution
    trend_counts = df[df["hybrid_trend"] != "WARMUP"]["hybrid_trend"].value_counts()
    total_bars = len(df[df["hybrid_trend"] != "WARMUP"])
    print(f"\n--- Trend Distribution ({total_bars} bars) ---")
    for t, c in trend_counts.items():
        print(f"  {t:15s}: {c:5d} bars ({c/total_bars*100:.1f}%)")
    chop_bars = trend_counts.get("CHOP", 0) + trend_counts.get("CHOP_RANGE", 0) + trend_counts.get("RANGE", 0)
    print(f"  {'CHOP+RANGE':15s}: {chop_bars:5d} bars ({chop_bars/total_bars*100:.1f}%)")

    # Run strategy on every bar to find signals
    config = OneMinuteStrategyConfig()
    # Enable all signal types
    config.ft_shorts_enabled = True
    config.ft_ema9_pb_enabled = True
    config.ft_trend_cont_enabled = True
    config.ft_trend_cont_adx_min = 25.0
    config.ft_adx_min = 20.0
    config.ft_adx_max = 35.0  # Signal A/D ADX range

    strategy = EsFifteenMinStrategy(config)

    all_trades: List[SimTrade] = []
    all_signals_count = 0
    chop_blocked_count = 0

    # We need enough context for the strategy (OR computation, etc.)
    # Process bar-by-bar, feeding growing DataFrame slices
    print(f"\nRunning strategy on {len(df)} bars...")

    # Track session date for OR resets
    prev_date = None

    for i in range(60, len(df)):
        row = df.iloc[i]
        ts = df.index[i]

        # Check if this bar is in a CHOP regime
        trend = df.iloc[i]["hybrid_trend"]
        is_chop = trend in ("CHOP", "CHOP_RANGE", "RANGE")

        # Feed growing window to strategy (last 100 bars for context)
        window_start = max(0, i - 100)
        features_slice = df.iloc[window_start:i + 1].copy()

        try:
            signal = strategy.generate(features_slice)
        except Exception:
            continue

        if signal.action in ("HOLD", ""):
            continue

        # We have a real signal
        all_signals_count += 1
        reason = signal.metadata.get("reason", "") if isinstance(signal.metadata, dict) else ""

        # Check if this would be CHOP-blocked
        is_pullback = "_PB_" in reason
        is_trend_cont = "TREND_CONT" in reason
        is_or_break = "OR_BREAK" in reason

        would_be_blocked = is_chop and (is_pullback or is_trend_cont) and not is_or_break

        if not would_be_blocked:
            continue

        chop_blocked_count += 1

        # Determine direction
        is_long = signal.action in ("BUY", "SCALP_BUY")
        direction = "LONG" if is_long else "SHORT"
        entry_price = row["close"]

        # Get SL/TP from signal metadata
        meta = signal.metadata if isinstance(signal.metadata, dict) else {}
        stop_loss = meta.get("stop_loss", 0.0)
        take_profit = meta.get("take_profit", 0.0)

        # If SL/TP not in metadata, compute from strategy defaults
        atr = row.get("ATR_14", 10.0)
        if stop_loss == 0.0 or take_profit == 0.0:
            if "EMA21_PB" in reason or "OR_BREAK" in reason:
                sl_pts = 6.0
                tp_pts = 8.0
            elif "EMA9_PB" in reason:
                sl_pts = min(20.0, max(8.0, atr * 1.0))
                tp_pts = sl_pts * 1.25
            elif "TREND_CONT" in reason:
                sl_pts = min(20.0, max(6.0, atr * 1.0))
                tp_pts = sl_pts * 1.25
            else:
                sl_pts = 6.0
                tp_pts = 8.0

            if is_long:
                stop_loss = entry_price - sl_pts
                take_profit = entry_price + tp_pts
            else:
                stop_loss = entry_price + sl_pts
                take_profit = entry_price - tp_pts

        adx_val = row.get("ADX_14", 0.0)
        rsi_val = row.get("RSI_14", 50.0)
        t_score = df.iloc[i]["trend_score"]
        e_spread = df.iloc[i]["ema_spread"]

        month_str = ts.strftime("%Y-%m") if hasattr(ts, "strftime") else str(ts)[:7]

        trade = SimTrade(
            bar_idx=i,
            timestamp=str(ts),
            signal_type=reason.split("|")[0].strip() if reason else "UNKNOWN",
            action=signal.action,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=signal.confidence,
            adx=adx_val,
            rsi=rsi_val,
            trend_score=t_score,
            trend_label=trend,
            ema_spread=e_spread,
            month=month_str,
        )

        # Simulate outcome
        trade = simulate_trade(trade, df, max_hold_bars=6)
        all_trades.append(trade)

    print(f"\n✅ Analysis complete:")
    print(f"   Total bars processed: {len(df) - 60}")
    print(f"   Total signals fired:  {all_signals_count}")
    print(f"   CHOP-blocked signals: {chop_blocked_count} ({len(all_trades)} simulated)")

    if not all_trades:
        print("\n⚠️  No CHOP-blocked trades found!")
        return

    # ──────────────────────────────────────────────────────────
    # 4. Results
    # ──────────────────────────────────────────────────────────

    # Convert to DataFrame for analysis
    results = pd.DataFrame([vars(t) for t in all_trades])

    print("\n" + "=" * 70)
    print("OVERALL RESULTS — ALL CHOP-BLOCKED TRADES")
    print("=" * 70)
    print_metrics(results, "All CHOP-blocked")

    # Direction breakdown
    longs = results[results["direction"] == "LONG"]
    shorts = results[results["direction"] == "SHORT"]

    print("\n" + "=" * 70)
    print("DIRECTION ASYMMETRY — THE KEY QUESTION")
    print("=" * 70)
    if len(longs) > 0:
        print(f"\n{'─' * 35}")
        print_metrics(longs, "LONGs in CHOP")
    if len(shorts) > 0:
        print(f"\n{'─' * 35}")
        print_metrics(shorts, "SHORTs in CHOP")

    # Statistical significance
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    if len(longs) >= 5 and len(shorts) >= 5:
        long_wr = (longs["outcome"] == "TP_HIT").mean()
        short_wr = (shorts["outcome"] == "TP_HIT").mean()
        long_n = len(longs)
        short_n = len(shorts)

        # Wilson confidence interval for win rates
        def wilson_ci(p, n, z=1.96):
            denom = 1 + z**2/n
            center = (p + z**2/(2*n)) / denom
            spread = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
            return max(0, center - spread), min(1, center + spread)

        long_lo, long_hi = wilson_ci(long_wr, long_n)
        short_lo, short_hi = wilson_ci(short_wr, short_n)

        print(f"\n  LONG  win rate: {long_wr:.1%} (n={long_n}), 95% CI: [{long_lo:.1%}, {long_hi:.1%}]")
        print(f"  SHORT win rate: {short_wr:.1%} (n={short_n}), 95% CI: [{short_lo:.1%}, {short_hi:.1%}]")

        # Check if CIs overlap
        if long_lo > short_hi:
            print(f"\n  ✅ STATISTICALLY SIGNIFICANT: LONG CI lower bound ({long_lo:.1%}) > SHORT CI upper bound ({short_hi:.1%})")
        elif long_hi < short_lo:
            print(f"\n  ❌ SHORTs actually better — CIs don't overlap in the other direction")
        else:
            print(f"\n  ⚠️  CIs overlap — difference not statistically significant at 95%")
            print(f"     LONG  CI: [{long_lo:.1%}, {long_hi:.1%}]")
            print(f"     SHORT CI: [{short_lo:.1%}, {short_hi:.1%}]")

        # Expectancy comparison
        long_exp = longs["pnl_dollars"].mean()
        short_exp = shorts["pnl_dollars"].mean()
        print(f"\n  LONG  expectancy: ${long_exp:+.2f}/trade")
        print(f"  SHORT expectancy: ${short_exp:+.2f}/trade")
        print(f"  Edge difference:  ${long_exp - short_exp:+.2f}/trade")

    # Monthly breakdown
    print("\n" + "=" * 70)
    print("MONTHLY BREAKDOWN")
    print("=" * 70)

    months = sorted(results["month"].unique())
    print(f"\n  {'Month':<10} {'Trades':>7} {'L':>3} {'S':>3} {'L-WR':>6} {'S-WR':>6} {'L-PnL':>9} {'S-PnL':>9} {'Net':>9}")
    print("  " + "─" * 68)

    monthly_data = []
    for m in months:
        month_df = results[results["month"] == m]
        ml = month_df[month_df["direction"] == "LONG"]
        ms = month_df[month_df["direction"] == "SHORT"]
        l_wr = (ml["outcome"] == "TP_HIT").mean() if len(ml) > 0 else 0
        s_wr = (ms["outcome"] == "TP_HIT").mean() if len(ms) > 0 else 0
        l_pnl = ml["pnl_dollars"].sum() if len(ml) > 0 else 0
        s_pnl = ms["pnl_dollars"].sum() if len(ms) > 0 else 0
        print(f"  {m:<10} {len(month_df):>7} {len(ml):>3} {len(ms):>3} {l_wr:>5.0%} {s_wr:>6.0%} {l_pnl:>+9.2f} {s_pnl:>+9.2f} {l_pnl+s_pnl:>+9.2f}")
        monthly_data.append({
            "month": m, "total": len(month_df),
            "longs": len(ml), "shorts": len(ms),
            "long_wr": l_wr, "short_wr": s_wr,
            "long_pnl": l_pnl, "short_pnl": s_pnl,
        })

    # Count months where LONGs > SHORTs
    if monthly_data:
        months_long_better = sum(1 for m in monthly_data if m["long_pnl"] > m["short_pnl"] and m["longs"] > 0)
        months_with_longs = sum(1 for m in monthly_data if m["longs"] > 0)
        print(f"\n  LONGs outperformed SHORTs in {months_long_better}/{months_with_longs} months with LONG trades")

    # Signal type breakdown
    print("\n" + "=" * 70)
    print("SIGNAL TYPE BREAKDOWN")
    print("=" * 70)

    for sig_type in sorted(results["signal_type"].unique()):
        sig_df = results[results["signal_type"] == sig_type]
        print(f"\n  {sig_type}:")
        sig_l = sig_df[sig_df["direction"] == "LONG"]
        sig_s = sig_df[sig_df["direction"] == "SHORT"]
        if len(sig_l) > 0:
            l_wr = (sig_l["outcome"] == "TP_HIT").mean()
            print(f"    LONG:  n={len(sig_l):3d}, WR={l_wr:.0%}, PnL=${sig_l['pnl_dollars'].sum():+.2f}, Avg=${sig_l['pnl_dollars'].mean():+.2f}")
        if len(sig_s) > 0:
            s_wr = (sig_s["outcome"] == "TP_HIT").mean()
            print(f"    SHORT: n={len(sig_s):3d}, WR={s_wr:.0%}, PnL=${sig_s['pnl_dollars'].sum():+.2f}, Avg=${sig_s['pnl_dollars'].mean():+.2f}")

    # ADX distribution for CHOP-blocked trades
    print("\n" + "=" * 70)
    print("ADX DISTRIBUTION (CHOP-blocked trades)")
    print("=" * 70)
    adx_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 50)]
    for lo, hi in adx_bins:
        bin_df = results[(results["adx"] >= lo) & (results["adx"] < hi)]
        if len(bin_df) > 0:
            bl = bin_df[bin_df["direction"] == "LONG"]
            bs = bin_df[bin_df["direction"] == "SHORT"]
            l_wr = (bl["outcome"] == "TP_HIT").mean() if len(bl) > 0 else 0
            s_wr = (bs["outcome"] == "TP_HIT").mean() if len(bs) > 0 else 0
            print(f"  ADX [{lo:2d}-{hi:2d}): n={len(bin_df):3d} (L:{len(bl)}/S:{len(bs)}) "
                  f"L-WR={l_wr:.0%} S-WR={s_wr:.0%} "
                  f"PnL=${bin_df['pnl_dollars'].sum():+.2f}")

    # Outcome distribution
    print("\n" + "=" * 70)
    print("OUTCOME DISTRIBUTION")
    print("=" * 70)
    for direction in ["LONG", "SHORT"]:
        dir_df = results[results["direction"] == direction]
        if len(dir_df) > 0:
            print(f"\n  {direction} ({len(dir_df)} trades):")
            for outcome in ["TP_HIT", "SL_HIT", "TIMEOUT"]:
                o_df = dir_df[dir_df["outcome"] == outcome]
                if len(o_df) > 0:
                    print(f"    {outcome:10s}: {len(o_df):3d} ({len(o_df)/len(dir_df)*100:.0f}%) avg PnL=${o_df['pnl_dollars'].mean():+.2f}")

    # Direction-aware filter impact
    print("\n" + "=" * 70)
    print("DIRECTION-AWARE FILTER IMPACT")
    print("=" * 70)
    total_pnl = results["pnl_dollars"].sum()
    long_pnl = longs["pnl_dollars"].sum() if len(longs) > 0 else 0
    short_pnl = shorts["pnl_dollars"].sum() if len(shorts) > 0 else 0

    long_wins = (longs["outcome"] == "TP_HIT").sum() if len(longs) > 0 else 0
    long_losses = len(longs) - long_wins if len(longs) > 0 else 0
    long_gross_wins = longs[longs["pnl_dollars"] > 0]["pnl_dollars"].sum() if len(longs) > 0 else 0
    long_gross_losses = abs(longs[longs["pnl_dollars"] <= 0]["pnl_dollars"].sum()) if len(longs) > 0 else 0
    long_pf = long_gross_wins / long_gross_losses if long_gross_losses > 0 else float("inf")

    print(f"\n  Strategy A (current): Block ALL → Net PnL = $0.00 (blocked {len(results)} trades)")
    print(f"  Strategy B (blind):   Unblock ALL → Net PnL = ${total_pnl:+.2f} ({len(results)} trades)")
    print(f"  Strategy C (dir):     LONGs pass, SHORTs block → Net PnL = ${long_pnl:+.2f} ({len(longs)} trades)")
    print(f"\n  Direction filter value: ${long_pnl - total_pnl:+.2f} (vs blind unblock)")
    print(f"  Direction filter value: ${long_pnl - 0:+.2f} (vs current block-all)")
    print(f"\n  LONG profit factor: {long_pf:.2f}")
    long_wr_all = (longs["outcome"] == "TP_HIT").mean() if len(longs) > 0 else 0
    print(f"  LONG win rate: {long_wr_all:.1%} ({long_wins}W/{long_losses}L)")
    print(f"  LONG expectancy: ${longs['pnl_dollars'].mean():+.2f}/trade" if len(longs) > 0 else "")

    # Max drawdown for LONG-only
    if len(longs) > 0:
        cum_pnl = longs["pnl_dollars"].cumsum()
        peak = cum_pnl.cummax()
        dd = cum_pnl - peak
        max_dd = dd.min()
        print(f"  LONG max drawdown: ${max_dd:.2f}")

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if len(longs) >= 20:
        long_wr_final = (longs["outcome"] == "TP_HIT").mean()
        short_wr_final = (shorts["outcome"] == "TP_HIT").mean() if len(shorts) > 0 else 0

        if long_wr_final > 0.45 and long_pnl > 0 and long_pf > 1.0:
            print(f"\n  ✅ DIRECTION-AWARE FILTER VALIDATED AT SCALE")
            print(f"     {len(longs)} LONG trades over {len(months)} months")
            print(f"     WR={long_wr_final:.0%}, PF={long_pf:.2f}, Net=${long_pnl:+.2f}")
            print(f"     vs SHORT WR={short_wr_final:.0%}, Net=${short_pnl:+.2f}")
            print(f"\n     → Allow LONGs in CHOP with dampen (current implementation)")
            print(f"     → Keep blocking SHORTs in CHOP")
        elif long_pnl > 0:
            print(f"\n  ⚠️  LONGs positive but marginal (WR={long_wr_final:.0%}, PF={long_pf:.2f})")
            print(f"     Consider more conservative dampen or additional filters")
        else:
            print(f"\n  ❌ LONGs negative at scale — reconsider direction-aware filter")
            print(f"     Block ALL signals in CHOP (revert to original)")
    else:
        print(f"\n  ⚠️  Only {len(longs)} LONG trades — insufficient sample for robust conclusion")
        print(f"     Need 20+ trades for meaningful statistical analysis")

    # Save detailed results
    output_path = Path("tools/chop_guard_year_results.csv")
    results.to_csv(output_path, index=False)
    print(f"\n  Detailed results saved to {output_path}")


def print_metrics(df: pd.DataFrame, label: str):
    """Print summary metrics for a set of trades."""
    n = len(df)
    wins = (df["outcome"] == "TP_HIT").sum()
    losses = (df["outcome"] == "SL_HIT").sum()
    timeouts = (df["outcome"] == "TIMEOUT").sum()
    wr = wins / n if n > 0 else 0

    gross_wins = df[df["pnl_dollars"] > 0]["pnl_dollars"].sum()
    gross_losses = abs(df[df["pnl_dollars"] <= 0]["pnl_dollars"].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    net = df["pnl_dollars"].sum()
    avg = df["pnl_dollars"].mean()

    cum = df["pnl_dollars"].cumsum()
    peak = cum.cummax()
    dd = cum - peak
    max_dd = dd.min()

    print(f"\n  {label} ({n} trades):")
    print(f"    Win rate:     {wr:.1%} ({wins}W / {losses}L / {timeouts}T)")
    print(f"    Net PnL:      ${net:+.2f}")
    print(f"    Profit factor: {pf:.2f}")
    print(f"    Avg PnL/trade: ${avg:+.2f}")
    print(f"    Max drawdown: ${max_dd:.2f}")
    print(f"    Avg ADX:      {df['adx'].mean():.1f}")
    print(f"    Avg RSI:      {df['rsi'].mean():.1f}")


if __name__ == "__main__":
    run_analysis()
