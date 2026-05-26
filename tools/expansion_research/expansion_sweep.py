"""
Variant sweep on top of the range-expansion prototype.

Re-uses the indicator + impulse-labeling layer from expansion_proto.py, then
tests post-filters to see whether the trigger can be conditioned into positive
expectancy without killing capture:

  V0  baseline (compression + expansion + vol + body + escape)
  V1  drop the open      (fire only >= 09:30 CT)
  V2  trend-aligned      (long: close>ema50 & ema21>ema50 ; short mirror)
  V3  non-climactic      (ADX_at_fire < 30 AND tr <= 3xATR ; not a blowoff)
  V4  V1 + V2 + V3 combined
  V5  V4 but afternoon-only (>= 11:00 CT)

For each: fires/day, capture (early/any), false-break% (all / open / lunch /
pm), expectancy at 1.5R and 2.0R, and % new coverage vs EMA9_PB.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expansion_proto import (enrich, in_rth, ema9_pb_long, label_impulse_legs,
                             P as BASE, DATA)
from datetime import time


def base_components(df, P):
    atr = df['atr']
    compressed = df['box_rng'] <= P['comp_atr_mult'] * atr
    expanded = df['tr'] >= P['exp_atr_mult'] * atr
    vol_exp = df['volume'] >= P['vol_mult'] * df['vol_sma']
    body_big = df['body'] >= P['body_mult'] * df['body_sma']
    body_conv = (df['body'] / df['rng']) >= P['body_frac']
    bdir = df['close'] - df['open']
    escape_up = df['close'] > df['box_hi']
    escape_dn = df['close'] < df['box_lo']
    base = compressed & expanded & vol_exp & body_big & body_conv & in_rth(df, P)
    return base, escape_up, escape_dn, bdir


def make_signal(df, P, variant):
    base, eu, ed, bdir = base_components(df, P)
    long = base & eu & (bdir > 0)
    short = base & ed & (bdir < 0)
    # ---- variant post-filters ----
    if variant.get("min_ct"):
        mc = variant["min_ct"]
        ok = pd.Series([t >= mc for t in df['ct']], index=df.index)
        long &= ok; short &= ok
    if variant.get("trend_align"):
        lt = (df['close'] > df['ema50']) & (df['ema21'] > df['ema50'])
        st = (df['close'] < df['ema50']) & (df['ema21'] < df['ema50'])
        long &= lt; short &= st
    if variant.get("non_climactic"):
        nc = (df['adx'] < 30) & (df['tr'] <= 3.0 * df['atr'])
        long &= nc; short &= nc
    d = pd.Series(0, index=df.index)
    d[long] = 1; d[short] = -1
    return d


def metrics(df, sarr, legs, pb_bars, P, n_days):
    H, L, C, A = (df['high'].values, df['low'].values,
                  df['close'].values, df['atr'].values)
    n = len(df)
    fires = np.where(sarr != 0)[0]
    fire_set = {1: set(np.where(sarr == 1)[0]), -1: set(np.where(sarr == -1)[0])}
    # capture
    ce = ca = 0
    for lg in legs:
        d, s, pk = lg['dir'], lg['start'], lg['peak']
        fb = fire_set[d]
        if any(s-1 <= f <= s+P['early_bars'] for f in fb): ce += 1
        if any(s-1 <= f <= pk for f in fb): ca += 1
    # false-break by bucket + expectancy
    def bucket(hr):
        if hr < 9: return 'open'
        if 11 <= hr <= 13: return 'lunch'
        if hr >= 14: return 'pm'
        return 'mid'
    from collections import defaultdict
    tot = defaultdict(int); fail = defaultdict(int)
    def exp_at(k):
        w = l = 0
        for i in fires:
            d, e, atr = sarr[i], C[i], A[i]
            if not np.isfinite(atr) or atr <= 0: continue
            tp = e + d*k*atr; sl = e - d*atr; res = 0
            for j in range(i+1, min(i+1+P['hold'], n)):
                if d == 1:
                    if L[j] <= sl: res=-1; break
                    if H[j] >= tp: res=1; break
                else:
                    if H[j] >= sl: res=-1; break
                    if L[j] <= tp: res=1; break
            if res==1: w+=1
            elif res==-1: l+=1
        dec = w+l
        return (w*k - l)/dec if dec else 0, (w/dec*100 if dec else 0), w, l
    for i in fires:
        d, e, atr = sarr[i], C[i], A[i]
        if not np.isfinite(atr) or atr <= 0: continue
        b = bucket(df.index[i].hour); tot[b]+=1
        failed=True
        for j in range(i+1, min(i+1+P['confirm'], n)):
            fav = (H[j]-e) if d==1 else (e-L[j])
            adv = (e-L[j]) if d==1 else (H[j]-e)
            if fav >= P['follow_atr']*atr: failed=False; break
            if adv >= P['fail_atr']*atr: failed=True; break
        else: failed=False
        if failed: fail[b]+=1
    def fr(b):
        return (fail[b]/tot[b]*100) if tot[b] else float('nan')
    all_t = sum(tot.values()); all_f = sum(fail.values())
    e15, wr15, *_ = exp_at(1.5)
    e20, wr20, *_ = exp_at(2.0)
    # new coverage
    long_bars = np.where(sarr==1)[0]
    near = sum(1 for i in long_bars if any(abs(i-p)<=2 for p in pb_bars))
    newcov = (100 - near/len(long_bars)*100) if len(long_bars) else float('nan')
    return dict(
        fires=len(fires), per_day=len(fires)/max(n_days,1),
        cap_e=ce/max(len(legs),1)*100, cap_a=ca/max(len(legs),1)*100,
        false_all=(all_f/all_t*100 if all_t else float('nan')),
        f_open=fr('open'), f_lunch=fr('lunch'), f_pm=fr('pm'),
        e15=e15, wr15=wr15, e20=e20, wr20=wr20, newcov=newcov,
    )


def main():
    P = dict(BASE)
    df = enrich(pd.read_parquet(DATA), P)
    legs = label_impulse_legs(df, P)
    pb_bars = set(np.where(ema9_pb_long(df, P).values)[0])
    n_days = df.loc[in_rth(df, P), 'date'].nunique()

    variants = [
        ("V0 baseline", {}),
        ("V1 drop-open >=09:30", {"min_ct": time(9,30)}),
        ("V2 trend-aligned", {"trend_align": True}),
        ("V3 non-climactic", {"non_climactic": True}),
        ("V4 V1+V2+V3", {"min_ct": time(9,30), "trend_align": True, "non_climactic": True}),
        ("V5 V4 + pm>=11:00", {"min_ct": time(11,0), "trend_align": True, "non_climactic": True}),
    ]
    print("="*112)
    print(f"VARIANT SWEEP — ES 15m, {df.index.min().date()}→{df.index.max().date()} "
          f"({n_days} RTH days, {len(legs)} impulse legs)")
    print("="*112)
    hdr = (f"{'variant':22} {'fires/d':>7} {'capE%':>6} {'capAny%':>7} "
           f"{'false%':>6} {'open':>5} {'lunch':>6} {'pm':>5} "
           f"{'exp1.5R':>8} {'WR15':>5} {'exp2R':>7} {'WR2':>4} {'new%':>5}")
    print(hdr); print("-"*112)
    for name, var in variants:
        sarr = make_signal(df, P, var).values
        m = metrics(df, sarr, legs, pb_bars, P, n_days)
        print(f"{name:22} {m['per_day']:7.2f} {m['cap_e']:6.1f} {m['cap_a']:7.1f} "
              f"{m['false_all']:6.0f} {m['f_open']:5.0f} {m['f_lunch']:6.0f} {m['f_pm']:5.0f} "
              f"{m['e15']:+8.2f} {m['wr15']:5.0f} {m['e20']:+7.2f} {m['wr20']:4.0f} {m['newcov']:5.0f}")
    print("-"*112)
    print("exp = expectancy per trade in R (target k*ATR, stop 1*ATR, hold "
          f"{P['hold']} bars). BE WR: 1.5R=40%, 2R=33%.")
    print("capE = captured within start+2 bars; capAny = before leg peak. "
          "false = reversed 1xATR before 1xATR follow-through within "
          f"{P['confirm']} bars.")


if __name__ == "__main__":
    main()
