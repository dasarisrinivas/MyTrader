#!/usr/bin/env python3
"""
Overnight Edge Deep Dive
========================
The base analysis revealed:
1. Overnight drift = +2.4 pts/night (positive long bias — structural S&P premium)
2. But std = 41.5 pts — massive noise, signal-to-noise ratio is 0.058
3. Max stop of 2.5 pts vs avg bar range 8.2 pts — can't use stops at 0.25% risk
4. Mean reversion: 50-54% hit rate — barely above coin flip
5. Autocorrelation: near zero everywhere

Let's check:
- Can we filter nights by regime (VIX proxy, prior RTH trend)?
- Is there a time-of-night entry that captures the drift better?
- What's the realistic trade after fees with a 5-pt stop?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
from datetime import time

ET = ZoneInfo("US/Eastern")

df = pd.read_parquet("data/ib/ES_30m_1y.parquet")
if df.index.tzinfo is None:
    df.index = df.index.tz_localize("UTC")
df.index = df.index.tz_convert(ET)
df = df["2025-02-01":"2026-01-30"]

# Compute indicators
df["bar_range"] = df["high"] - df["low"]
df["bar_return"] = df["close"] - df["open"]
df["ATR_10"] = df["bar_range"].rolling(10).mean()

def classify_session(ts):
    t = ts.time()
    if time(9, 30) <= t < time(16, 0): return "RTH"
    elif time(16, 0) <= t < time(20, 0): return "Evening"
    elif time(20, 0) <= t or t < time(2, 0): return "Asia"
    elif time(2, 0) <= t < time(6, 0): return "London"
    elif time(6, 0) <= t < time(9, 30): return "Pre_RTH"
    else: return "Unknown"

df["session"] = df.index.map(classify_session)

# Get RTH daily data
df_rth = df[df["session"] == "RTH"].copy()
df_rth["date"] = df_rth.index.date

# Daily RTH metrics
daily = df_rth.groupby("date").agg(
    rth_open=("open", "first"),
    rth_close=("close", "last"),
    rth_high=("high", "max"),
    rth_low=("low", "min"),
    rth_volume=("volume", "sum"),
    rth_range=("bar_range", "sum"),  # approx
).copy()
daily["rth_return"] = daily["rth_close"] - daily["rth_open"]
daily["rth_range_hl"] = daily["rth_high"] - daily["rth_low"]

# ── Overnight returns (close-to-open) ──
dates = sorted(daily.index)
records = []
for i in range(1, len(dates)):
    prev_date = dates[i - 1]
    curr_date = dates[i]
    prev_close = daily.loc[prev_date, "rth_close"]
    curr_open = daily.loc[curr_date, "rth_open"]
    prev_return = daily.loc[prev_date, "rth_return"]
    prev_range = daily.loc[prev_date, "rth_range_hl"]
    
    overnight_ret = curr_open - prev_close
    
    records.append({
        "date": curr_date,
        "prev_close": prev_close,
        "today_open": curr_open,
        "overnight_return": overnight_ret,
        "prev_rth_return": prev_return,
        "prev_rth_range": prev_range,
    })

on = pd.DataFrame(records)
on["date"] = pd.to_datetime(on["date"])

COST = 2.40 + 0.50  # commission + 1 tick slippage each way (conservative)
PV = 5  # MES point value

# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("A. OVERNIGHT RETURN DISTRIBUTION")
print("=" * 70)
for pct in [10, 25, 50, 75, 90]:
    val = on["overnight_return"].quantile(pct / 100)
    print(f"  {pct}th percentile: {val:+.2f} pts")

print(f"\n  Mean: {on['overnight_return'].mean():+.3f}")
print(f"  Std:  {on['overnight_return'].std():.3f}")
print(f"  Skew: {on['overnight_return'].skew():.3f}")
print(f"  Kurt: {on['overnight_return'].kurtosis():.3f}")
print()

# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("B. FILTERED OVERNIGHT LONG: AFTER UP RTH DAY vs DOWN RTH DAY")
print("=" * 70)
for label, mask in [
    ("After UP RTH", on["prev_rth_return"] > 0),
    ("After DOWN RTH", on["prev_rth_return"] < 0),
    ("After BIG UP (>10pts)", on["prev_rth_return"] > 10),
    ("After BIG DOWN (<-10pts)", on["prev_rth_return"] < -10),
    ("After LOW RANGE (<20pts)", on["prev_rth_range"] < 20),
    ("After HIGH RANGE (>40pts)", on["prev_rth_range"] > 40),
]:
    sub = on[mask]
    if len(sub) < 10:
        print(f"  {label}: n={len(sub)} (too few)")
        continue
    pnl = sub["overnight_return"] * PV - COST
    total = pnl.sum()
    wr = (pnl > 0).mean() * 100
    avg = pnl.mean()
    dd = (pnl.cumsum() - pnl.cumsum().cummax()).min()
    print(f"  {label:30s}: n={len(sub):3d} | P&L=${total:+7.0f} | WR={wr:4.1f}% | Avg=${avg:+5.2f} | DD=${dd:+7.0f}")
print()

# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("C. STOP ANALYSIS: What stop distance avoids getting stopped out?")
print("=" * 70)
# For each night, compute the worst adverse excursion (MAE)
# We need intrabar data — use 30m bars during overnight
# Simulate: enter long at RTH close, what's the max drawdown during the night?
overnight_bars = df[df["session"] != "RTH"].copy()
overnight_bars["date_key"] = overnight_bars.index.date

# For each night, calculate MAE from entry (prev RTH close)
mae_records = []
for _, row in on.iterrows():
    d = row["date"]
    entry = row["prev_close"]
    
    # Get overnight bars for this night (from prev close to this open)
    night_bars = overnight_bars[
        (overnight_bars.index >= pd.Timestamp(row["date"] - pd.Timedelta(days=1), tz=ET).replace(hour=16)) &
        (overnight_bars.index < pd.Timestamp(row["date"], tz=ET).replace(hour=9, minute=30))
    ]
    
    if len(night_bars) == 0:
        continue
    
    # Max adverse excursion for a LONG from entry
    worst_low = night_bars["low"].min()
    mae = entry - worst_low  # positive = drawdown for long
    
    # Max favorable excursion
    best_high = night_bars["high"].max()
    mfe = best_high - entry  # positive = profit for long
    
    mae_records.append({
        "date": d,
        "mae": mae,
        "mfe": mfe,
        "overnight_return": row["overnight_return"],
    })

mae_df = pd.DataFrame(mae_records)

print(f"  MAE (max adverse excursion for overnight long):")
for pct in [50, 75, 90, 95, 99]:
    val = mae_df["mae"].quantile(pct / 100)
    print(f"    {pct}th percentile: {val:.2f} pts")

print(f"\n  MFE (max favorable excursion for overnight long):")
for pct in [50, 75, 90, 95]:
    val = mae_df["mfe"].quantile(pct / 100)
    print(f"    {pct}th percentile: {val:.2f} pts")

print()

# Simulate with various stop sizes
print("=" * 70)
print("D. PASSIVE LONG WITH STOP LOSS (various sizes)")
print("=" * 70)
print(f"  {'Stop':>6s} {'Trades':>7s} {'Stopped':>8s} {'P&L':>9s} {'WinRate':>8s} {'PF':>6s} {'MaxDD':>9s}")
print("  " + "-" * 55)

for stop_pts in [2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 999]:
    stopped = 0
    pnls = []
    for _, row in mae_df.iterrows():
        if row["mae"] >= stop_pts:
            # Stopped out
            pnl_pts = -stop_pts
            stopped += 1
        else:
            pnl_pts = row["overnight_return"]
        pnls.append(pnl_pts * PV - COST)
    
    pnl_series = pd.Series(pnls)
    total = pnl_series.sum()
    wr = (pnl_series > 0).mean() * 100
    wins = pnl_series[pnl_series > 0].sum()
    losses = pnl_series[pnl_series <= 0].sum()
    pf = abs(wins / losses) if losses != 0 else 999
    dd = (pnl_series.cumsum() - pnl_series.cumsum().cummax()).min()
    
    label = f"{stop_pts:.1f}" if stop_pts < 999 else "None"
    print(f"  {label:>6s} {len(pnls):>7d} {stopped:>8d} ${total:>+8.0f} {wr:>7.1f}% {pf:>5.2f} ${dd:>+8.0f}")
print()

# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("E. RISK-ADJUSTED: Can we meet 0.25% risk ($12.50)?")
print("=" * 70)
max_risk = 12.50
max_stop = max_risk / PV
print(f"  Max stop at 0.25% risk: {max_stop:.2f} pts")
print(f"  Median overnight MAE: {mae_df['mae'].median():.2f} pts")
print(f"  75th pctl MAE: {mae_df['mae'].quantile(0.75):.2f} pts")
print(f"  90th pctl MAE: {mae_df['mae'].quantile(0.90):.2f} pts")
print()
print(f"  With 2.5pt stop: {(mae_df['mae'] >= 2.5).mean()*100:.0f}% of nights get stopped out")
print(f"  With 5.0pt stop: {(mae_df['mae'] >= 5.0).mean()*100:.0f}% of nights get stopped out")
print(f"  With 10pt stop:  {(mae_df['mae'] >= 10.0).mean()*100:.0f}% of nights get stopped out")
print()

# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("F. CONCLUSION METRICS")
print("=" * 70)
# The real question: is there ANY overnight strategy that meets:
# - PF > 1.0 after fees
# - DD < 10% ($500)
# - Risk ≤ 0.25% per trade

# Best realistic case: 10pt stop
stop = 10.0
pnls_10 = []
for _, row in mae_df.iterrows():
    if row["mae"] >= stop:
        pnl_pts = -stop
    else:
        pnl_pts = row["overnight_return"]
    pnls_10.append(pnl_pts * PV - COST)

s = pd.Series(pnls_10)
total = s.sum()
dd = (s.cumsum() - s.cumsum().cummax()).min()
dd_pct = abs(dd) / 5000 * 100
risk_per = stop * PV
risk_pct = risk_per / 5000 * 100
sharpe = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0

print(f"  Best realistic (10pt stop, every night):")
print(f"    Total P&L:     ${total:+,.0f}")
print(f"    Max Drawdown:  ${dd:+,.0f} ({dd_pct:.1f}% of $5K)")
print(f"    Risk per trade: ${risk_per:.0f} ({risk_pct:.1f}% of $5K)")
print(f"    Sharpe:         {sharpe:.2f}")
print()
if risk_pct > 0.25:
    print(f"  ⚠️  10pt stop = {risk_pct:.1f}% risk — EXCEEDS 0.25% constraint by {risk_pct/0.25:.0f}×")
if dd_pct > 10:
    print(f"  ⚠️  Drawdown {dd_pct:.1f}% EXCEEDS 10% constraint")
print()

# What about fractional position sizing?
print(f"  Fractional sizing to meet 0.25% risk ({max_risk:.2f}/trade):")
print(f"    With 10pt stop: need ${stop*PV:.0f} risk budget → cannot trade 1 MES contract")
print(f"    With 5pt stop:  need ${5*PV:.0f} risk budget → cannot trade 1 MES contract") 
print(f"    With 2.5pt stop: ${2.5*PV:.2f} = exactly $12.50 → CAN trade but 71% stop-out rate")
print()
print(f"  BOTTOM LINE: MES minimum = 1 contract. Cannot fractionally size below 1 contract.")
print(f"  At 1 contract, ANY stop > 2.5 pts exceeds 0.25% risk on $5K.")
