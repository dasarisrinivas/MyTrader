"""
Range-expansion ("transition energy") trigger — historical prototype.

GOAL (per design discussion): detect when the market *escapes a balanced
auction with abnormal participation* during the session — the state today's
es_fifteen_min strategy is structurally blind to (OR-cross already consumed,
EMA stack not yet aligned, ADX not yet >= absolute floor).

This is RESEARCH ONLY. It does not touch live strategy code. It loads a year
of 15-min ES bars and measures, for a parameterized trigger:
  1. capture rate of major intraday impulse legs (ground-truth labeled)
  2. average excursion before first pullback (run room)
  3. false-break frequency by time-of-day (lunchtime chop risk)
  4. overlap / conflict with existing EMA9_PB pullback entries
  5. whether wider ATR targets pair better than the pullback logic's ~1.25R

Trigger = compression precedent + range expansion + volume expansion
          + body displacement + escape beyond the balance range.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import time

DATA = "data/ib/ES_15m_1y.parquet"

# ---- tunable parameters -------------------------------------------------
P = dict(
    # indicators
    atr_n=14, adx_n=14, rsi_n=14,
    # compression precedent (the "balance" before the break)
    n_comp=6,            # bars of balance to look back (90 min)
    comp_atr_mult=3.0,   # prior n_comp-bar range <= this * ATR  => "balanced"
    # the expansion bar itself
    exp_atr_mult=1.5,    # bar true-range >= this * ATR
    vol_n=20, vol_mult=1.5,     # bar volume >= this * SMA(vol)
    body_n=10, body_mult=1.5,   # |close-open| >= this * SMA(body)
    body_frac=0.5,       # |close-open| / (high-low) >= this (directional conviction)
    # session
    rth_start=time(8, 30), rth_end=time(15, 0),  # CT cash session
    # impulse-leg ground truth
    imp_atr_mult=2.5,    # a "major" leg advances >= this * ATR ...
    imp_window=8,        # ... within this many bars (2h) ...
    imp_retrace_cap=0.40,# ... with adverse giveback before the peak <= this frac
    early_bars=2,        # trigger "participates" if it fires <= start+early_bars
    # excursion / target analysis
    hold=8,              # forward bars to evaluate MFE/targets
    pullback_atr=1.0,    # "first pullback" = retrace >= this * ATR from running MFE
    fail_atr=1.0,        # false break = adverse move >= this * ATR before any follow-through
    follow_atr=1.0,      # follow-through = favorable move >= this * ATR within confirm
    confirm=4,           # bars to judge follow-through vs failure
)


def wilder_atr(df, n):
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr, tr.ewm(alpha=1/n, adjust=False).mean()


def wilder_adx(df, n):
    h, l, c = df['high'], df['low'], df['close']
    up = h.diff(); dn = -l.diff()
    plus_dm = ((up > dn) & (up > 0)) * up.clip(lower=0)
    minus_dm = ((dn > up) & (dn > 0)) * dn.clip(lower=0)
    _, atr = wilder_atr(df, n)
    atr_s = atr.replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def rsi(df, n):
    d = df['close'].diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    ls = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = g / ls.replace(0, np.nan)
    return 100 - 100/(1+rs)


def enrich(df, P):
    df = df.copy()
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    tr, atr = wilder_atr(df, P['atr_n'])
    df['tr'] = tr; df['atr'] = atr
    df['adx'] = wilder_adx(df, P['adx_n'])
    df['rsi'] = rsi(df, P['rsi_n'])
    df['body'] = (df['close'] - df['open']).abs()
    df['rng'] = (df['high'] - df['low']).replace(0, np.nan)
    df['vol_sma'] = df['volume'].rolling(P['vol_n']).mean()
    df['body_sma'] = df['body'].rolling(P['body_n']).mean()
    # prior n_comp-bar balance box (exclusive of current bar)
    nc = P['n_comp']
    df['box_hi'] = df['high'].rolling(nc).max().shift(1)
    df['box_lo'] = df['low'].rolling(nc).min().shift(1)
    df['box_rng'] = df['box_hi'] - df['box_lo']
    df['ct'] = df.index.time
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    return df


def in_rth(df, P):
    return (df['ct'] >= P['rth_start']) & (df['ct'] < P['rth_end'])


def expansion_trigger(df, P):
    """Vectorized long/short transition-energy trigger. Returns dir series in {+1,-1,0}."""
    atr = df['atr']
    compressed = df['box_rng'] <= P['comp_atr_mult'] * atr
    expanded = df['tr'] >= P['exp_atr_mult'] * atr
    vol_exp = df['volume'] >= P['vol_mult'] * df['vol_sma']
    body_big = df['body'] >= P['body_mult'] * df['body_sma']
    body_dir = (df['close'] - df['open'])
    body_conv = (df['body'] / df['rng']) >= P['body_frac']
    escape_up = df['close'] > df['box_hi']
    escape_dn = df['close'] < df['box_lo']
    base = compressed & expanded & vol_exp & body_big & body_conv & in_rth(df, P)
    long_sig = base & escape_up & (body_dir > 0)
    short_sig = base & escape_dn & (body_dir < 0)
    d = pd.Series(0, index=df.index)
    d[long_sig] = 1
    d[short_sig] = -1
    return d


def ema9_pb_long(df, P):
    """Simplified replica of the live EMA9_PB_LONG arming condition, for overlap."""
    stack = (df['ema9'] > df['ema21']) & (df['ema21'] > df['ema50'])
    touch = df['low'] <= df['ema9'] * (1 + 0.0015)   # pulled back to ema9
    closeok = df['close'] > df['ema9']               # but closed back above
    adxok = df['adx'] >= 22
    rsiok = (df['rsi'] >= 40) & (df['rsi'] <= 70)
    sig = stack & touch & closeok & adxok & rsiok & in_rth(df, P) & (df['close'] > df['open'])
    return sig


def label_impulse_legs(df, P):
    """Independent ground truth: discrete major directional legs in RTH.
    For each candidate start bar, look forward imp_window bars; a long leg
    requires forward MFE >= imp_atr_mult*ATR with adverse giveback before the
    MFE peak <= imp_retrace_cap*MFE. Dedupe by skipping past each leg's peak."""
    legs = []
    idx = df.index
    H, L, C, A = df['high'].values, df['low'].values, df['close'].values, df['atr'].values
    rth = in_rth(df, P).values
    dates = df['date'].values
    n = len(df); W = P['imp_window']
    i = 0
    while i < n - 1:
        if not rth[i] or not np.isfinite(A[i]) or A[i] <= 0:
            i += 1; continue
        thr = P['imp_atr_mult'] * A[i]
        c0 = C[i]
        # window stays within same RTH date
        j_end = i + 1
        while j_end < n and j_end <= i + W and rth[j_end] and dates[j_end] == dates[i]:
            j_end += 1
        if j_end <= i + 1:
            i += 1; continue
        seg = slice(i + 1, j_end)
        up_mfe = H[seg].max() - c0
        dn_mfe = c0 - L[seg].min()
        direction = 0
        if up_mfe >= thr or dn_mfe >= thr:
            if up_mfe >= dn_mfe:
                peak = i + 1 + int(np.argmax(H[seg]))
                adverse = c0 - L[i+1:peak+1].min() if peak > i else 0.0
                if up_mfe > 0 and adverse <= P['imp_retrace_cap'] * up_mfe:
                    direction = 1; mfe = up_mfe; pk = peak
            else:
                peak = i + 1 + int(np.argmin(L[seg]))
                adverse = H[i+1:peak+1].max() - c0 if peak > i else 0.0
                if dn_mfe > 0 and adverse <= P['imp_retrace_cap'] * dn_mfe:
                    direction = -1; mfe = dn_mfe; pk = peak
        if direction != 0:
            legs.append(dict(start=i, peak=pk, dir=direction, mfe=mfe,
                             atr=A[i], start_ts=idx[i], hour=idx[i].hour))
            i = pk + 1   # dedupe: jump past this leg
        else:
            i += 1
    return legs


def main():
    df = pd.read_parquet(DATA)
    df = enrich(df, P)
    sig = expansion_trigger(df, P)
    pb = ema9_pb_long(df, P)
    df['sig'] = sig; df['pb'] = pb
    legs = label_impulse_legs(df, P)

    rth = in_rth(df, P)
    n_days = df.loc[rth, 'date'].nunique()
    n_fires = int((sig != 0).sum())
    n_long = int((sig == 1).sum()); n_short = int((sig == -1).sum())
    print("="*72)
    print("RANGE-EXPANSION TRIGGER PROTOTYPE — ES 15m, "
          f"{df.index.min().date()}→{df.index.max().date()} ({n_days} RTH days)")
    print("="*72)
    print(f"Params: comp<= {P['comp_atr_mult']}xATR/{P['n_comp']}bars, "
          f"TR>= {P['exp_atr_mult']}xATR, vol>= {P['vol_mult']}x, "
          f"body>= {P['body_mult']}x & {P['body_frac']} of range, escape box")
    print(f"\nFires: {n_fires} total ({n_long} long / {n_short} short) "
          f"= {n_fires/max(n_days,1):.2f}/day")
    print(f"Ground-truth major impulse legs: {len(legs)} "
          f"({len(legs)/max(n_days,1):.2f}/day)")

    # ---- 1. capture rate -------------------------------------------------
    fire_bars = {k: set() for k in (1, -1)}
    sarr = sig.values
    for k in (1, -1):
        fire_bars[k] = set(np.where(sarr == k)[0])
    captured_early = captured_any = 0
    excur_at_capture = []
    for lg in legs:
        d = lg['dir']; s = lg['start']; pk = lg['peak']
        early_hi = s + P['early_bars']
        fb = fire_bars[d]
        if any(s - 1 <= f <= early_hi for f in fb):
            captured_early += 1
        if any(s - 1 <= f <= pk for f in fb):
            captured_any += 1
    print("\n-- 1. CAPTURE RATE of major impulse legs --")
    print(f"   early (<= start+{P['early_bars']} bars): {captured_early}/{len(legs)} "
          f"= {captured_early/max(len(legs),1)*100:.1f}%")
    print(f"   any time before peak             : {captured_any}/{len(legs)} "
          f"= {captured_any/max(len(legs),1)*100:.1f}%")

    # ---- 2. excursion before first pullback (per fire) ------------------
    H, L, C, A, O = (df['high'].values, df['low'].values, df['close'].values,
                     df['atr'].values, df['open'].values)
    n = len(df)
    mfes = []
    for i in np.where(sarr != 0)[0]:
        d = sarr[i]; entry = C[i]; atr = A[i]
        if not np.isfinite(atr) or atr <= 0: continue
        run_mfe = 0.0
        for j in range(i+1, min(i+1+P['hold'], n)):
            fav = (H[j]-entry) if d == 1 else (entry-L[j])
            adv = (entry-L[j]) if d == 1 else (H[j]-entry)
            run_mfe = max(run_mfe, fav)
            # first pullback: gave back pullback_atr*ATR from running MFE
            if run_mfe > 0 and (run_mfe - fav) >= P['pullback_atr']*atr:
                break
        mfes.append((run_mfe, run_mfe/atr))
    if mfes:
        pts = np.array([m[0] for m in mfes]); xatr = np.array([m[1] for m in mfes])
        print("\n-- 2. EXCURSION before first pullback (per fire) --")
        print(f"   mean {pts.mean():.1f} pts ({xatr.mean():.2f}xATR) | "
              f"median {np.median(pts):.1f} pts ({np.median(xatr):.2f}xATR) | "
              f"90th {np.percentile(pts,90):.1f} pts")

    # ---- 3. false-break frequency by time-of-day ------------------------
    from collections import defaultdict
    tod_tot = defaultdict(int); tod_fail = defaultdict(int)
    for i in np.where(sarr != 0)[0]:
        d = sarr[i]; entry = C[i]; atr = A[i]
        if not np.isfinite(atr) or atr <= 0: continue
        hr = df.index[i].hour
        tod_tot[hr] += 1
        failed = True
        for j in range(i+1, min(i+1+P['confirm'], n)):
            fav = (H[j]-entry) if d == 1 else (entry-L[j])
            adv = (entry-L[j]) if d == 1 else (H[j]-entry)
            if fav >= P['follow_atr']*atr:   # got follow-through first
                failed = False; break
            if adv >= P['fail_atr']*atr:     # broke against us first
                failed = True; break
        else:
            failed = False
        if failed: tod_fail[hr] += 1
    print("\n-- 3. FALSE-BREAK rate by CT hour --")
    for hr in sorted(tod_tot):
        t = tod_tot[hr]; f = tod_fail[hr]
        print(f"   {hr:02d}:00 CT  fires={t:3d}  false={f:3d}  ({f/t*100:4.0f}%)")

    # ---- 4. overlap with EMA9_PB ----------------------------------------
    pb_bars = set(np.where(df['pb'].values)[0])
    exp_long_bars = np.where(sarr == 1)[0]
    near = 0
    for i in exp_long_bars:
        if any(abs(i - p) <= 2 for p in pb_bars):
            near += 1
    print("\n-- 4. OVERLAP with EMA9_PB longs (within +/-2 bars) --")
    print(f"   EMA9_PB long arming bars: {len(pb_bars)}  "
          f"({len(pb_bars)/max(n_days,1):.2f}/day)")
    print(f"   expansion-long fires that coincide w/ EMA9_PB: {near}/{len(exp_long_bars)} "
          f"= {near/max(len(exp_long_bars),1)*100:.0f}%  "
          f"(=> {100-near/max(len(exp_long_bars),1)*100:.0f}% are NEW coverage)")

    # ---- 5. target pairing: expansion fires vs ATR-multiple targets ------
    print("\n-- 5. TARGET PAIRING (expansion fires; TP/SL = k*ATR, hold "
          f"{P['hold']} bars) --")
    for k in (1.25, 1.5, 2.0, 3.0, 4.0):
        wins = losses = neither = 0
        for i in np.where(sarr != 0)[0]:
            d = sarr[i]; entry = C[i]; atr = A[i]
            if not np.isfinite(atr) or atr <= 0: continue
            tp = entry + d*k*atr; sl = entry - d*atr   # 1xATR stop, k*ATR target
            res = 0
            for j in range(i+1, min(i+1+P['hold'], n)):
                hi, lo = H[j], L[j]
                if d == 1:
                    if lo <= sl: res = -1; break
                    if hi >= tp: res = 1; break
                else:
                    if hi >= sl: res = -1; break
                    if lo <= tp: res = 1; break
            if res == 1: wins += 1
            elif res == -1: losses += 1
            else: neither += 1
        dec = wins + losses
        wr = wins/dec*100 if dec else 0
        # expectancy per trade in R (target=k R, stop=1 R), open trades ignored
        exp_R = (wins*k - losses*1)/dec if dec else 0
        be = 1/(1+k)*100
        print(f"   target {k:>4.2f}R/1R: WR {wr:4.0f}% (BE {be:4.0f}%)  "
              f"exp/trade {exp_R:+.2f}R  [{wins}W/{losses}L/{neither} open]")

    # quick sanity: show a few sample fires
    print("\n-- sample expansion fires --")
    sf = df[df['sig'] != 0].head(6)
    for ts, r in sf.iterrows():
        print(f"   {ts}  dir={int(r['sig']):+d} close={r['close']:.2f} "
              f"atr={r['atr']:.1f} adx={r['adx']:.0f} boxRng={r['box_rng']:.1f} "
              f"tr={r['tr']:.1f} vol/{r['vol_sma']:.0f}")


if __name__ == "__main__":
    main()
