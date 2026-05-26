"""
Earlier continuation-arming prototype (ADX slope / relaxed stack).

Question: today's move armed EMA9_PB_LONG only at 14:15 — AFTER the impulse,
in the "late continuation" regime. Can we arm earlier ("trend strength is
INCREASING rapidly") without buying noise?

Method — per contiguous bullish "continuation episode" (RTH, same day):
  * find the first bar the CURRENT EMA9_PB logic would arm  (BASE)
  * find the first bar each EARLIER-ARMING variant would arm (Vx)
  * lead_bars = base_first - variant_first  (>0 = variant earlier)
  * simulate each entry (stop 1xATR, targets k*ATR, hold H bars) and record
    forward MFE, MAE, and outcome.

Measures the 4 things asked:
  1. how many bars earlier eligibility occurs
  2. MAE/MFE shift vs current timing
  3. whether earlier arming degrades win rate
  4. whether expectancy improves (entering before extension exhaustion)

Longs only (shorts are symmetric). Research only; no live code touched.
"""
from __future__ import annotations
import sys, os, numpy as np, pandas as pd
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from expansion_proto import enrich, in_rth, P as BASE, DATA

Q = dict(
    adx_abs=22.0,          # current absolute ADX floor
    adx_slope_k=3,         # bars over which to measure ADX slope (45 min)
    adx_slope_min=3.0,     # "rising rapidly" = +this ADX over k bars
    adx_slope_floor=15.0,  # but require at least this ADX (avoid dead chop)
    touch_pct=0.0015,      # pullback touch tolerance to ema9
    rsi_lo=40, rsi_hi=70,
    hold=8,                # forward bars
    stop_atr=1.0,
)


def add_adx_slope(df, k):
    df = df.copy()
    df['adx_slope'] = df['adx'] - df['adx'].shift(k)
    return df


def arm_masks(df, Q):
    """Return dict of boolean Series: BASE + earlier-arming variants (longs)."""
    e9, e21, e50 = df['ema9'], df['ema21'], df['ema50']
    touch = df['low'] <= e9 * (1 + Q['touch_pct'])
    closeok = df['close'] > e9
    bull = df['close'] > df['open']
    rsiok = (df['rsi'] >= Q['rsi_lo']) & (df['rsi'] <= Q['rsi_hi'])
    rth = in_rth(df, Q if 'rth_start' in Q else BASE)
    full_stack = (e9 > e21) & (e21 > e50)
    loose_stack = (e9 > e21) & (df['close'] > e50)   # don't wait for e21>e50
    adx_abs = df['adx'] >= Q['adx_abs']
    adx_slope = (df['adx_slope'] >= Q['adx_slope_min']) & (df['adx'] >= Q['adx_slope_floor'])
    common = touch & closeok & bull & rsiok & rth
    return {
        'BASE  (stack+adx>=22)':        common & full_stack & adx_abs,
        'V1 adx-slope OR abs':          common & full_stack & (adx_abs | adx_slope),
        'V2 loose-stack':               common & loose_stack & adx_abs,
        'V3 slope OR + loose-stack':    common & loose_stack & (adx_abs | adx_slope),
    }


def episodes(df, Q):
    """Contiguous bullish micro-context within an RTH day: ema9>ema21 & close>ema50."""
    e9, e21, e50 = df['ema9'].values, df['ema21'].values, df['ema50'].values
    rth = in_rth(df, BASE).values
    dates = df['date'].values
    ctx = (e9 > e21) & (df['close'].values > e50) & rth
    eps = []
    n = len(df); i = 0
    while i < n:
        if ctx[i]:
            j = i
            while j + 1 < n and ctx[j+1] and dates[j+1] == dates[i]:
                j += 1
            if j - i >= 1:           # at least 2 bars
                eps.append((i, j))
            i = j + 1
        else:
            i += 1
    return eps


def first_true(mask_vals, lo, hi):
    idxs = np.where(mask_vals[lo:hi+1])[0]
    return lo + idxs[0] if len(idxs) else None


def sim(df, i, Q, k_targets):
    H = df['high'].values; L = df['low'].values; C = df['close'].values; A = df['atr'].values
    n = len(df); entry = C[i]; atr = A[i]
    if not np.isfinite(atr) or atr <= 0:
        return None
    mfe = mae = 0.0
    out = {}
    sl = entry - Q['stop_atr'] * atr
    res_by_k = {k: 0 for k in k_targets}
    done = {k: False for k in k_targets}
    for j in range(i+1, min(i+1+Q['hold'], n)):
        mfe = max(mfe, H[j] - entry); mae = max(mae, entry - L[j])
        for k in k_targets:
            if done[k]:
                continue
            tp = entry + k*atr
            if L[j] <= sl:
                res_by_k[k] = -1; done[k] = True
            elif H[j] >= tp:
                res_by_k[k] = +1; done[k] = True
    return dict(mfe=mfe, mae=mae, mfe_atr=mfe/atr, mae_atr=mae/atr,
                atr=atr, res=res_by_k)


def expectancy(results, k):
    w = sum(1 for r in results if r['res'][k] == 1)
    l = sum(1 for r in results if r['res'][k] == -1)
    dec = w + l
    wr = w/dec*100 if dec else 0
    exp = (w*k - l)/dec if dec else 0
    return wr, exp, w, l


def main():
    df = add_adx_slope(enrich(pd.read_parquet(DATA), BASE), Q['adx_slope_k'])
    masks = {name: m.values for name, m in arm_masks(df, Q).items()}
    eps = episodes(df, Q)
    n_days = df.loc[in_rth(df, BASE), 'date'].nunique()
    k_targets = (1.25, 1.5, 2.0)

    print("="*100)
    print(f"CONTINUATION-ARMING TIMING — ES 15m, {df.index.min().date()}→{df.index.max().date()} "
          f"({n_days} RTH days, {len(eps)} bullish episodes)")
    print("="*100)
    print(f"ADX-slope rule: +{Q['adx_slope_min']:.0f} over {Q['adx_slope_k']} bars "
          f"AND adx>={Q['adx_slope_floor']:.0f}  (vs current absolute adx>={Q['adx_abs']:.0f})\n")

    # First-arm bar per episode, per variant
    base_name = 'BASE  (stack+adx>=22)'
    per_variant = {name: dict(entries=[], leads=[], first_idx=[]) for name in masks}
    base_first = {}
    for (a, b) in eps:
        bf = first_true(masks[base_name], a, b)
        base_first[(a, b)] = bf
    for name, mv in masks.items():
        for (a, b) in eps:
            f = first_true(mv, a, b)
            if f is None:
                continue
            per_variant[name]['first_idx'].append(f)
            r = sim(df, f, Q, k_targets)
            if r:
                per_variant[name]['entries'].append(r)
            bf = base_first[(a, b)]
            if bf is not None:
                per_variant[name]['leads'].append(bf - f)   # >0 = variant earlier

    # ---- Table ----
    hdr = (f"{'variant':26} {'entries':>7} {'/day':>5} {'medLead':>7} {'%earlier':>8} "
           f"{'MFE(xATR)':>9} {'MAE(xATR)':>9} {'WR1.5':>6} {'exp1.5':>7} {'WR2':>5} {'exp2':>6}")
    print(hdr); print("-"*100)
    for name, mv in masks.items():
        E = per_variant[name]['entries']
        leads = [x for x in per_variant[name]['leads']]
        earlier = [x for x in leads if x > 0]
        n = len(E)
        if n == 0:
            print(f"{name:26} {0:>7}"); continue
        mfe = np.median([r['mfe_atr'] for r in E])
        mae = np.median([r['mae_atr'] for r in E])
        wr15, e15, *_ = expectancy(E, 1.5)
        wr2, e2, *_ = expectancy(E, 2.0)
        med_lead = np.median(earlier) if earlier else 0
        pct_earlier = (len(earlier)/len(leads)*100) if leads else 0
        print(f"{name:26} {n:>7} {n/n_days:>5.2f} {med_lead:>7.1f} {pct_earlier:>7.0f}% "
              f"{mfe:>9.2f} {mae:>9.2f} {wr15:>5.0f}% {e15:>+7.2f} {wr2:>4.0f}% {e2:>+6.2f}")
    print("-"*100)

    # ---- The key question: quality of the EARLY-ONLY entries ----
    # bars where V3 arms but BASE does NOT (within the episode, before base first-arm)
    v3 = masks['V3 slope OR + loose-stack']; bs = masks[base_name]
    early_only = []
    base_entries = []
    for (a, b) in eps:
        bf = first_true(bs, a, b)
        vf = first_true(v3, a, b)
        if bf is not None:
            r = sim(df, bf, Q, k_targets)
            if r: base_entries.append(r)
        if vf is not None and (bf is None or vf < bf):
            r = sim(df, vf, Q, k_targets)
            if r: early_only.append(r)
    print("\n-- EARLY-ONLY entries (V3 arms before BASE, or where BASE never arms) --")
    for label, S in (("BASE first-arm", base_entries), ("EARLY-ONLY (the marginal adds)", early_only)):
        if not S:
            print(f"   {label:34} n=0"); continue
        wr15,e15,w,l = expectancy(S,1.5); wr2,e2,*_ = expectancy(S,2.0)
        mfe=np.median([r['mfe_atr'] for r in S]); mae=np.median([r['mae_atr'] for r in S])
        print(f"   {label:34} n={len(S):4d}  MFE {mfe:.2f}xATR  MAE {mae:.2f}xATR  "
              f"WR1.5 {wr15:.0f}% exp {e15:+.2f}R | WR2 {wr2:.0f}% exp {e2:+.2f}R")
    print("\nIf EARLY-ONLY expectancy >= BASE, earlier arming is 'free' participation.")
    print("MFE compares run-room remaining at entry (higher = entered before exhaustion).")


if __name__ == "__main__":
    main()
