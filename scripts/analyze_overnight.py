#!/usr/bin/env python3
"""
Overnight Market Microstructure Analysis
=========================================
Analyze 30m and 15m ES data to determine if any overnight edge exists.

Sessions (ET):
  RTH:      09:30 - 16:00
  Evening:  16:00 - 20:00  (post-close, US)
  Asia:     20:00 - 02:00  (Tokyo/HK overlap)
  London:   02:00 - 06:00  (London open, EU overlap)
  Pre-RTH:  06:00 - 09:30  (pre-market, highest overnight volume)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")

# ── Load data ────────────────────────────────────────────────────────
df = pd.read_parquet("data/ib/ES_30m_1y.parquet")
if df.index.tzinfo is None:
    df.index = df.index.tz_localize("UTC")
df.index = df.index.tz_convert(ET)

# Filter to Feb 2025 - Jan 2026 (same as backtest period)
df = df["2025-02-01":"2026-01-30"]

print(f"Total 30m bars: {len(df)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print()

# ── Classify sessions ────────────────────────────────────────────────
from datetime import time

def classify_session(ts):
    t = ts.time()
    if time(9, 30) <= t < time(16, 0):
        return "RTH"
    elif time(16, 0) <= t < time(20, 0):
        return "Evening"
    elif time(20, 0) <= t or t < time(2, 0):
        return "Asia"
    elif time(2, 0) <= t < time(6, 0):
        return "London"
    elif time(6, 0) <= t < time(9, 30):
        return "Pre_RTH"
    else:
        return "Unknown"

df["session"] = df.index.map(classify_session)
df["bar_range"] = df["high"] - df["low"]
df["bar_return"] = df["close"] - df["open"]
df["bar_return_pct"] = df["bar_return"] / df["open"] * 100

# ── 1. Session distribution ──────────────────────────────────────────
print("=" * 70)
print("1. SESSION DISTRIBUTION")
print("=" * 70)
sess_counts = df["session"].value_counts()
for s in ["RTH", "Evening", "Asia", "London", "Pre_RTH"]:
    n = sess_counts.get(s, 0)
    pct = n / len(df) * 100
    print(f"  {s:10s}: {n:5d} bars ({pct:5.1f}%)")
print()

# ── 2. Bar characteristics by session ────────────────────────────────
print("=" * 70)
print("2. BAR CHARACTERISTICS BY SESSION (30m bars)")
print("=" * 70)
print(f"  {'Session':10s} {'AvgRange':>9s} {'MedRange':>9s} {'StdRange':>9s} {'AvgVol':>10s} {'ZeroRng%':>9s} {'AvgReturn':>10s}")
print("  " + "-" * 68)
for s in ["RTH", "Evening", "Asia", "London", "Pre_RTH"]:
    sub = df[df["session"] == s]
    avg_range = sub["bar_range"].mean()
    med_range = sub["bar_range"].median()
    std_range = sub["bar_range"].std()
    avg_vol = sub["volume"].mean()
    zero_pct = (sub["bar_range"] == 0).mean() * 100
    avg_ret = sub["bar_return"].mean()
    print(f"  {s:10s} {avg_range:9.2f} {med_range:9.2f} {std_range:9.2f} {avg_vol:10.0f} {zero_pct:9.1f} {avg_ret:10.4f}")
print()

# ── 3. Directional bias by session ───────────────────────────────────
print("=" * 70)
print("3. DIRECTIONAL BIAS BY SESSION")
print("=" * 70)
for s in ["RTH", "Evening", "Asia", "London", "Pre_RTH"]:
    sub = df[df["session"] == s]
    up = (sub["bar_return"] > 0).sum()
    dn = (sub["bar_return"] < 0).sum()
    flat = (sub["bar_return"] == 0).sum()
    total = len(sub)
    up_pct = up / total * 100 if total > 0 else 0
    avg_up = sub.loc[sub["bar_return"] > 0, "bar_return"].mean() if up > 0 else 0
    avg_dn = sub.loc[sub["bar_return"] < 0, "bar_return"].mean() if dn > 0 else 0
    print(f"  {s:10s}: Up {up_pct:5.1f}% | Avg Up: {avg_up:+.2f} pts | Avg Down: {avg_dn:+.2f} pts | Flat: {flat}")
print()

# ── 4. Mean reversion analysis ───────────────────────────────────────
print("=" * 70)
print("4. MEAN REVERSION ANALYSIS (overnight only)")
print("=" * 70)
# For each overnight session, check if large moves revert in the next bar
overnight = df[df["session"] != "RTH"].copy()
overnight["next_return"] = overnight["bar_return"].shift(-1)
overnight["large_up"] = overnight["bar_return"] > overnight["bar_range"].rolling(20).mean()
overnight["large_dn"] = overnight["bar_return"] < -overnight["bar_range"].rolling(20).mean()

# After a large up bar, does next bar go down? (mean reversion)
large_up = overnight[overnight["large_up"] == True]
large_dn = overnight[overnight["large_dn"] == True]

if len(large_up) > 0:
    revert_after_up = (large_up["next_return"] < 0).mean() * 100
    avg_revert_up = large_up["next_return"].mean()
    print(f"  After large UP bar (n={len(large_up)}): {revert_after_up:.1f}% revert | Avg next: {avg_revert_up:+.3f}")

if len(large_dn) > 0:
    revert_after_dn = (large_dn["next_return"] > 0).mean() * 100
    avg_revert_dn = large_dn["next_return"].mean()
    print(f"  After large DN bar (n={len(large_dn)}): {revert_after_dn:.1f}% revert | Avg next: {avg_revert_dn:+.3f}")
print()

# ── 5. Volatility compression → expansion ────────────────────────────
print("=" * 70)
print("5. VOLATILITY COMPRESSION ANALYSIS (overnight)")
print("=" * 70)
overnight["rolling_range_3"] = overnight["bar_range"].rolling(3).mean()
overnight["rolling_range_10"] = overnight["bar_range"].rolling(10).mean()
overnight["compression"] = overnight["rolling_range_3"] / overnight["rolling_range_10"]

# When 3-bar range is < 0.5× the 10-bar range, what happens?
compressed = overnight[overnight["compression"] < 0.5]
if len(compressed) > 0:
    # Next bar characteristics after compression
    compressed_next = overnight.loc[compressed.index].shift(-1).dropna()
    print(f"  Compression events (3bar/10bar < 0.5): n={len(compressed)}")
    print(f"  Next bar avg range: {compressed_next['bar_range'].mean():.2f}")
    print(f"  Next bar avg |return|: {compressed_next['bar_return'].abs().mean():.3f}")
    print(f"  Normal overnight avg range: {overnight['bar_range'].mean():.2f}")
else:
    print(f"  No compression events found (threshold too strict)")
    compressed = overnight[overnight["compression"] < 0.7]
    print(f"  Relaxed (< 0.7): n={len(compressed)}")
print()

# ── 6. Session-open gaps (London open, Pre-RTH) ──────────────────────
print("=" * 70)
print("6. SESSION TRANSITION ANALYSIS")
print("=" * 70)
# Check if first bar of each sub-session has exploitable patterns
for session_name in ["Evening", "Asia", "London", "Pre_RTH"]:
    sess_bars = df[df["session"] == session_name]
    if len(sess_bars) == 0:
        continue
    # Group by date, get first bar of each session
    sess_bars_copy = sess_bars.copy()
    sess_bars_copy["date"] = sess_bars_copy.index.date
    first_bars = sess_bars_copy.groupby("date").first()
    avg_range = first_bars["bar_range"].mean()
    avg_abs_ret = first_bars["bar_return"].abs().mean()
    up_pct = (first_bars["bar_return"] > 0).mean() * 100
    print(f"  {session_name:10s} first bar: AvgRange={avg_range:.2f} AvgAbsReturn={avg_abs_ret:.3f} Up%={up_pct:.1f}% (n={len(first_bars)})")
print()

# ── 7. Autocorrelation by session ────────────────────────────────────
print("=" * 70)
print("7. AUTOCORRELATION (lag-1 return correlation)")
print("=" * 70)
for s in ["RTH", "Evening", "Asia", "London", "Pre_RTH"]:
    sub = df[df["session"] == s].copy()
    sub["prev_return"] = sub["bar_return"].shift(1)
    ac = sub["bar_return"].corr(sub["prev_return"])
    print(f"  {s:10s}: autocorrelation = {ac:+.4f}")
print()

# ── 8. Overnight cumulative drift ────────────────────────────────────
print("=" * 70)
print("8. OVERNIGHT CUMULATIVE DRIFT")
print("=" * 70)
# Group by date, compute overnight return (16:00 close to 09:30 open)
df_rth = df[df["session"] == "RTH"].copy()
df_rth["date"] = df_rth.index.date
daily_close = df_rth.groupby("date")["close"].last()
daily_open = df_rth.groupby("date")["open"].first()

overnight_returns = []
dates = sorted(daily_close.index)
for i in range(1, len(dates)):
    prev_close = daily_close[dates[i-1]]
    today_open = daily_open[dates[i]]
    overnight_ret = today_open - prev_close
    overnight_returns.append({"date": dates[i], "overnight_return": overnight_ret})

on_df = pd.DataFrame(overnight_returns)
if len(on_df) > 0:
    avg_on = on_df["overnight_return"].mean()
    std_on = on_df["overnight_return"].std()
    pos_pct = (on_df["overnight_return"] > 0).mean() * 100
    cum_on = on_df["overnight_return"].sum()
    print(f"  Avg overnight return: {avg_on:+.3f} pts/night")
    print(f"  Std overnight return: {std_on:.3f} pts")
    print(f"  Positive nights: {pos_pct:.1f}%")
    print(f"  Cumulative drift: {cum_on:+.1f} pts over {len(on_df)} nights")
    print(f"  Annualized edge: {avg_on * 252:+.1f} pts (before fees)")
    print(f"  MES value: ${avg_on * 252 * 5:+,.0f}/yr (1 contract)")
print()

# ── 9. Risk/reward at $5K ────────────────────────────────────────────
print("=" * 70)
print("9. RISK BUDGET ANALYSIS")
print("=" * 70)
capital = 5000
max_risk_pct = 0.0025  # 0.25%
max_risk_usd = capital * max_risk_pct
point_value = 5  # MES
max_stop_pts = max_risk_usd / point_value
print(f"  Capital: ${capital:,}")
print(f"  Max risk/trade: {max_risk_pct*100:.2f}% = ${max_risk_usd:.2f}")
print(f"  Max stop distance: {max_stop_pts:.2f} points (MES $5/pt)")
print(f"  Overnight avg bar range: {overnight['bar_range'].mean():.2f} pts")
print(f"  Overnight median bar range: {overnight['bar_range'].median():.2f} pts")
print(f"  → Stop must be < {max_stop_pts:.1f} pts but avg bar = {overnight['bar_range'].mean():.1f} pts")
if max_stop_pts < overnight["bar_range"].mean():
    print(f"  ⚠️  STOP IS TIGHTER THAN AVERAGE BAR RANGE — extremely constrained")
print()

# ── 10. What if we just held long overnight? ──────────────────────────
print("=" * 70)
print("10. PASSIVE OVERNIGHT LONG BENCHMARK")
print("=" * 70)
if len(on_df) > 0:
    # Simulate: buy at RTH close, sell at next RTH open
    commission = 2.40  # round trip
    slippage = 0.25 * 2  # 1 tick each way = $2.50 round trip
    cost_per_trade = commission + slippage * point_value
    
    on_df["pnl"] = on_df["overnight_return"] * point_value - cost_per_trade
    on_df["cum_pnl"] = on_df["pnl"].cumsum()
    total_pnl = on_df["pnl"].sum()
    win_rate = (on_df["pnl"] > 0).mean() * 100
    avg_pnl = on_df["pnl"].mean()
    max_dd = (on_df["cum_pnl"] - on_df["cum_pnl"].cummax()).min()
    
    print(f"  Strategy: Buy close, sell next open (every night)")
    print(f"  Total P&L: ${total_pnl:+,.2f}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg P&L/trade: ${avg_pnl:+,.2f}")
    print(f"  Max Drawdown: ${max_dd:+,.2f}")
    print(f"  Trades: {len(on_df)}")
    sharpe = on_df["pnl"].mean() / on_df["pnl"].std() * np.sqrt(252) if on_df["pnl"].std() > 0 else 0
    print(f"  Sharpe: {sharpe:.2f}")
print()

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
