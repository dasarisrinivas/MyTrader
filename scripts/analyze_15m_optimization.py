"""
Analyze 15m strategy for optimization opportunities.
Reads the latest backtest trades CSV and analyzes exit types, signal types,
time-of-day PnL, hold time, ATR/ADX, monthly PnL.
"""
import sys, os, ast, glob
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo


def run_analysis():
    # Find latest trades CSV
    pattern = "reports/backtest_ES_2025-02-01_2026-01-30_*_trades.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        print("No trades CSV found! Run backtest first.")
        return
    csv_path = files[-1]
    print(f"Reading: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
    df['exit_time'] = pd.to_datetime(df['exit_time'], utc=True)
    df['pnl_dollars'] = df['realized_pnl']
    df['pnl_points'] = df['realized_pnl'] / 5.0
    
    # Parse entry_metadata
    def parse_meta(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) else {}
        except:
            return {}
    df['meta'] = df['entry_metadata'].apply(parse_meta)
    
    # Signal type
    df['signal_type'] = df['meta'].apply(
        lambda m: 'EMA21_PB' if 'PB' in str(m.get('signal_reason', '')) 
                  else ('OR_BREAK' if 'OR' in str(m.get('signal_reason', '')) else 'OTHER')
    )
    
    # ET times
    et = ZoneInfo("US/Eastern")
    df['entry_hour_et'] = df['entry_time'].apply(lambda t: t.astimezone(et).hour)
    df['entry_minute_et'] = df['entry_time'].apply(lambda t: t.astimezone(et).minute)
    
    # ATR/ADX
    df['atr'] = df['meta'].apply(lambda m: m.get('atr_value', 0))
    df['adx'] = df['meta'].apply(lambda m: m.get('adx_value', 0))
    
    # Hold time
    df['hold_minutes'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
    
    print(f"\n{'='*70}")
    print(f"15M STRATEGY OPTIMIZATION ANALYSIS ({len(df)} trades)")
    print(f"{'='*70}")
    print(f"Total PnL: ${df['pnl_dollars'].sum():.0f}")
    print(f"Win Rate: {(df['pnl_dollars'] > 0).mean()*100:.1f}%")
    gw = df[df['pnl_dollars'] > 0]['pnl_dollars'].sum()
    gl = abs(df[df['pnl_dollars'] < 0]['pnl_dollars'].sum())
    print(f"Profit Factor: {gw/gl:.2f}" if gl > 0 else "PF: inf")
    
    # ── 1. Exit Type ──
    print(f"\n{'─'*60}")
    print("1. EXIT TYPE ANALYSIS")
    print(f"{'─'*60}")
    for exit_type in sorted(df['exit_reason'].unique()):
        s = df[df['exit_reason'] == exit_type]
        tp = s['pnl_dollars'].sum()
        avg = s['pnl_dollars'].mean()
        wr = (s['pnl_dollars'] > 0).mean() * 100
        sw = s[s['pnl_dollars'] > 0]['pnl_dollars'].sum()
        sl = abs(s[s['pnl_dollars'] < 0]['pnl_dollars'].sum())
        pf = sw/sl if sl > 0 else float('inf')
        print(f"  {exit_type:20s}: {len(s):4d} trades | ${tp:>8.0f} | avg ${avg:>7.1f} | WR {wr:.0f}% | PF {pf:.2f}")
    
    # ── 2. Signal Type ──
    print(f"\n{'─'*60}")
    print("2. SIGNAL TYPE ANALYSIS")
    print(f"{'─'*60}")
    for sig_type in sorted(df['signal_type'].unique()):
        s = df[df['signal_type'] == sig_type]
        tp = s['pnl_dollars'].sum()
        wr = (s['pnl_dollars'] > 0).mean() * 100
        sw = s[s['pnl_dollars'] > 0]['pnl_dollars'].sum()
        sl = abs(s[s['pnl_dollars'] < 0]['pnl_dollars'].sum())
        pf = sw/sl if sl > 0 else float('inf')
        avg_win = s[s['pnl_dollars'] > 0]['pnl_dollars'].mean() if (s['pnl_dollars'] > 0).any() else 0
        avg_loss = s[s['pnl_dollars'] <= 0]['pnl_dollars'].mean() if (s['pnl_dollars'] <= 0).any() else 0
        print(f"  {sig_type:15s}: {len(s):4d} trades | ${tp:>8.0f} | WR {wr:.0f}% | PF {pf:.2f} | avg_W ${avg_win:.0f} avg_L ${avg_loss:.0f}")
    
    # ── 3. Entry Hour ──
    print(f"\n{'─'*60}")
    print("3. ENTRY HOUR ANALYSIS (ET)")
    print(f"{'─'*60}")
    for hour in sorted(df['entry_hour_et'].unique()):
        s = df[df['entry_hour_et'] == hour]
        tp = s['pnl_dollars'].sum()
        avg = s['pnl_dollars'].mean()
        wr = (s['pnl_dollars'] > 0).mean() * 100
        sw = s[s['pnl_dollars'] > 0]['pnl_dollars'].sum()
        sl_val = abs(s[s['pnl_dollars'] < 0]['pnl_dollars'].sum())
        pf = sw/sl_val if sl_val > 0 else float('inf')
        print(f"  {hour:02d}:xx ET: {len(s):4d} trades | ${tp:>8.0f} | avg ${avg:>7.1f} | WR {wr:.0f}% | PF {pf:.2f}")
    
    # 30-min buckets
    print(f"\n  Entry time (30-min buckets):")
    df['entry_slot'] = df['entry_hour_et'] * 100 + (df['entry_minute_et'] // 30) * 30
    for slot in sorted(df['entry_slot'].unique()):
        s = df[df['entry_slot'] == slot]
        tp = s['pnl_dollars'].sum()
        wr = (s['pnl_dollars'] > 0).mean() * 100
        h, m = divmod(slot, 100)
        print(f"    {h:02d}:{m:02d} ET: {len(s):3d} trades | ${tp:>7.0f} | WR {wr:.0f}%")
    
    # ── 4. Exit reason × signal type ──
    print(f"\n{'─'*60}")
    print("4. EXIT REASON BY SIGNAL TYPE")
    print(f"{'─'*60}")
    for sig in sorted(df['signal_type'].unique()):
        print(f"\n  {sig}:")
        subset = df[df['signal_type'] == sig]
        for er in sorted(subset['exit_reason'].unique()):
            s2 = subset[subset['exit_reason'] == er]
            tp = s2['pnl_dollars'].sum()
            wr = (s2['pnl_dollars'] > 0).mean() * 100
            print(f"    {er:20s}: {len(s2):3d} trades | ${tp:>7.0f} | WR {wr:.0f}%")
    
    # ── 5. Flatten/Max-hold deep analysis ──
    print(f"\n{'─'*60}")
    print("5. FLATTEN / MAX-HOLD EXIT DEEP ANALYSIS")
    print(f"{'─'*60}")
    mh = df[df['exit_reason'].str.contains('flatten|MAX_HOLD|max_hold', case=False, na=False)]
    if len(mh) > 0:
        w = (mh['pnl_dollars'] > 0).sum()
        l = (mh['pnl_dollars'] <= 0).sum()
        print(f"  Count: {len(mh)} ({w} W / {l} L)")
        print(f"  Total PnL: ${mh['pnl_dollars'].sum():.0f}")
        print(f"  Avg PnL: ${mh['pnl_dollars'].mean():.1f} ({mh['pnl_points'].mean():.1f} pts)")
        print(f"  Avg hold: {mh['hold_minutes'].mean():.0f} min")
        
        print(f"\n  PnL distribution (pts):")
        bins = [-100, -30, -15, -5, 0, 5, 15, 30, 100]
        mh_cut = pd.cut(mh['pnl_points'], bins=bins)
        for b, cnt in mh_cut.value_counts().sort_index().items():
            if cnt > 0:
                print(f"    {str(b):20s}: {cnt:3d} trades")
    
    # ── 6. Monthly PnL ──
    print(f"\n{'─'*60}")
    print("6. MONTHLY PNL")
    print(f"{'─'*60}")
    df['month'] = df['entry_time'].apply(lambda t: f"{t.year}-{t.month:02d}")
    for month in sorted(df['month'].unique()):
        s = df[df['month'] == month]
        tp = s['pnl_dollars'].sum()
        wr = (s['pnl_dollars'] > 0).mean() * 100
        print(f"  {month}: {len(s):3d} trades | ${tp:>8.0f} | WR {wr:.0f}%")
    
    # ── 7. ATR / Volatility ──
    print(f"\n{'─'*60}")
    print("7. ATR / ADX ANALYSIS")
    print(f"{'─'*60}")
    atr_q = df['atr'].quantile([0.25, 0.5, 0.75])
    print(f"  ATR: min={df['atr'].min():.1f}  Q1={atr_q[0.25]:.1f}  med={atr_q[0.5]:.1f}  Q3={atr_q[0.75]:.1f}  max={df['atr'].max():.1f}")
    
    for label, lo, hi in [("Low (<Q1)", 0, atr_q[0.25]), ("Med (Q1-Q3)", atr_q[0.25], atr_q[0.75]), ("High (>Q3)", atr_q[0.75], 999)]:
        s = df[(df['atr'] >= lo) & (df['atr'] < hi)]
        if len(s) == 0: continue
        tp = s['pnl_dollars'].sum()
        wr = (s['pnl_dollars'] > 0).mean() * 100
        sw = s[s['pnl_dollars'] > 0]['pnl_dollars'].sum()
        slv = abs(s[s['pnl_dollars'] < 0]['pnl_dollars'].sum())
        pf = sw/slv if slv > 0 else float('inf')
        print(f"  {label:15s}: {len(s):3d} trades | ${tp:>7.0f} | WR {wr:.0f}% | PF {pf:.2f}")
    
    print(f"\n  ADX buckets:")
    for lo, hi in [(20,25), (25,30), (30,35), (35,40), (40,100)]:
        s = df[(df['adx'] >= lo) & (df['adx'] < hi)]
        if len(s) == 0: continue
        tp = s['pnl_dollars'].sum()
        wr = (s['pnl_dollars'] > 0).mean() * 100
        sw = s[s['pnl_dollars'] > 0]['pnl_dollars'].sum()
        slv = abs(s[s['pnl_dollars'] < 0]['pnl_dollars'].sum())
        pf = sw/slv if slv > 0 else float('inf')
        print(f"    ADX {lo:2d}-{hi:2d}: {len(s):3d} trades | ${tp:>7.0f} | WR {wr:.0f}% | PF {pf:.2f}")
    
    # ── 8. Hold time ──
    print(f"\n{'─'*60}")
    print("8. HOLD TIME ANALYSIS")
    print(f"{'─'*60}")
    for lo, hi in [(0,15), (15,30), (30,45), (45,60), (60,75), (75,91)]:
        s = df[(df['hold_minutes'] >= lo) & (df['hold_minutes'] <= hi)]
        if len(s) == 0: continue
        tp = s['pnl_dollars'].sum()
        wr = (s['pnl_dollars'] > 0).mean() * 100
        print(f"  {lo:2d}-{hi:2d}min: {len(s):3d} trades | ${tp:>7.0f} | WR {wr:.0f}% | avg_pts {s['pnl_points'].mean():+.1f}")
    
    # ── 9. Summary ──
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")
    
    tp_hits = df[df['exit_reason'].str.contains('PROFIT|bracket_tp', case=False, na=False)]
    sl_hits = df[df['exit_reason'].str.contains('STOP|bracket_sl', case=False, na=False)]
    flattens = df[df['exit_reason'].str.contains('flatten', case=False, na=False)]
    
    print(f"\n  Exit breakdown:")
    print(f"    Bracket TP: {len(tp_hits):3d} ({len(tp_hits)/len(df)*100:.0f}%) | ${tp_hits['pnl_dollars'].sum():>7.0f}")
    print(f"    Bracket SL: {len(sl_hits):3d} ({len(sl_hits)/len(df)*100:.0f}%) | ${sl_hits['pnl_dollars'].sum():>7.0f}")
    print(f"    Flatten:    {len(flattens):3d} ({len(flattens)/len(df)*100:.0f}%) | ${flattens['pnl_dollars'].sum():>7.0f}")
    
    if len(flattens) > 0:
        fw = (flattens['pnl_dollars'] > 0).sum()
        fl = (flattens['pnl_dollars'] <= 0).sum()
        if fw / len(flattens) > 0.55:
            print(f"\n  💡 {fw/len(flattens)*100:.0f}% of flatten exits are winners — longer hold may let them reach TP")
    
    or_trades = df[df['signal_type'] == 'OR_BREAK']
    if len(or_trades) > 0:
        or_pnl = or_trades['pnl_dollars'].sum()
        or_wr = (or_trades['pnl_dollars'] > 0).mean() * 100
        if or_pnl < 0:
            print(f"\n  ⚠️ OR_BREAK is net NEGATIVE (${or_pnl:.0f}, WR {or_wr:.0f}%)")


if __name__ == "__main__":
    run_analysis()
