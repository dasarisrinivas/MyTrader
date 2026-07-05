"""Backtest the SPY options technical-level rules on real 5-min data.

Validates each rule currently implemented in technical_levels.py and
dynamic_confidence.py blocks 13-18:

  1. ORB breakout follow-through (overall, by width bucket, by gap size, by weekday)
  2. VWAP +/-2sd fade vs continuation
  3. EDR exhaustion (>=85%) -> further range extension probability
  4. RSI divergence forward returns
  5. Pivot touch mean reversion

Data: yfinance SPY 5-min (60d max) + ^VIX daily.
"""
import math
import sys
from collections import defaultdict
from datetime import time

import numpy as np
import pandas as pd
import yfinance as yf

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shree.spy_options.technical_levels import _rsi_series, _detect_divergence, _floor_pivots

ET = "America/New_York"

print("Downloading SPY 5-min (60d) and VIX daily...")
spy = yf.download("SPY", period="60d", interval="5m", progress=False, auto_adjust=False)
vix = yf.download("^VIX", period="70d", interval="1d", progress=False, auto_adjust=False)

if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

spy.index = spy.index.tz_convert(ET)
vix_close = {d.date(): float(c) for d, c in vix["Close"].items()}

# RTH only
spy = spy.between_time("09:30", "15:55")
days = sorted(set(spy.index.date))
print(f"Sessions: {len(days)}  ({days[0]} .. {days[-1]})\n")

# Build per-day frames
day_frames = {d: spy[spy.index.date == d] for d in days}

# ============================================================================
# 1. ORB BREAKOUT FOLLOW-THROUGH
# ============================================================================
# Rule under test: first confirmed 5-min close outside the 9:30-10:00 range
# after 10:00. Follow-through = SPY moves >= +0.15% further in breakout
# direction before reverting -0.15% against entry (first-touch), plus
# direction-correct-at-close stats.

orb_results = []   # dict per breakout event

for i, d in enumerate(days):
    df = day_frames[d]
    if len(df) < 20:
        continue
    orb = df.between_time("09:30", "09:55")   # 6 bars, 9:30-10:00
    post = df.between_time("10:00", "15:55")
    if len(orb) < 6 or len(post) < 10:
        continue
    orb_h, orb_l = float(orb["High"].max()), float(orb["Low"].min())
    day_open = float(df["Open"].iloc[0])
    orb_width_pct = (orb_h - orb_l) / day_open * 100

    prev_close = None
    if i > 0:
        prev_df = day_frames[days[i - 1]]
        if len(prev_df):
            prev_close = float(prev_df["Close"].iloc[-1])
    gap_pct = (day_open - prev_close) / prev_close * 100 if prev_close else 0.0

    # find first confirmed close outside ORB
    breakout = None
    for j in range(len(post)):
        c = float(post["Close"].iloc[j])
        if c > orb_h:
            breakout = ("UP", j, c)
            break
        if c < orb_l:
            breakout = ("DOWN", j, c)
            break
    if breakout is None:
        continue
    direction, j, entry = breakout
    entry_time = post.index[j].time()

    # forward path after entry bar
    fwd = post.iloc[j + 1:]
    if len(fwd) < 3:
        continue
    sgn = 1 if direction == "UP" else -1
    # first-touch: +0.15% target vs -0.15% stop (in breakout direction)
    tgt = entry * (1 + sgn * 0.0015)
    stp = entry * (1 - sgn * 0.0015)
    outcome = None
    for k in range(len(fwd)):
        hi, lo = float(fwd["High"].iloc[k]), float(fwd["Low"].iloc[k])
        hit_t = hi >= tgt if sgn == 1 else lo <= tgt
        hit_s = lo <= stp if sgn == 1 else hi >= stp
        if hit_t and hit_s:
            outcome = "AMBIG"
            break
        if hit_t:
            outcome = "WIN"
            break
        if hit_s:
            outcome = "LOSS"
            break
    if outcome is None:
        outcome = "FLAT"
    close_px = float(fwd["Close"].iloc[-1])
    close_ret = sgn * (close_px - entry) / entry * 100

    orb_results.append(dict(
        day=d, weekday=pd.Timestamp(d).weekday(), direction=direction,
        width=orb_width_pct, gap=gap_pct, entry_time=entry_time,
        outcome=outcome, close_ret=close_ret,
    ))

res = pd.DataFrame(orb_results)
res_clean = res[res.outcome.isin(["WIN", "LOSS"])]

def wr(sub):
    if len(sub) == 0:
        return float("nan"), 0
    w = (sub.outcome == "WIN").sum()
    return w / len(sub) * 100, len(sub)

print("=" * 70)
print("1. ORB BREAKOUT (first confirmed close outside 30-min range)")
print("=" * 70)
w, n = wr(res_clean)
print(f"Overall first-touch win rate (+-0.15%): {w:.0f}%  (n={n})")
print(f"Direction correct at close: {(res.close_ret > 0).mean() * 100:.0f}%  avg close ret {res.close_ret.mean():+.2f}%  (n={len(res)})")

print("\nBy ORB width bucket:")
for lbl, lo, hi in [("<0.20% (tight)", 0, .20), ("0.20-0.40%", .20, .40), ("0.40-0.60%", .40, .60), (">0.60% (wide)", .60, 99)]:
    sub = res_clean[(res_clean.width >= lo) & (res_clean.width < hi)]
    w, n = wr(sub)
    suba = res[(res.width >= lo) & (res.width < hi)]
    print(f"  {lbl:16s} first-touch {w:5.0f}% (n={n:2d})   close-correct {(suba.close_ret>0).mean()*100 if len(suba) else float('nan'):3.0f}%  avg {suba.close_ret.mean() if len(suba) else float('nan'):+.2f}%")

print("\nBy opening gap:")
for lbl, cond in [
    ("|gap| < 0.30%", res_clean.gap.abs() < .30),
    ("|gap| 0.30-0.70%", (res_clean.gap.abs() >= .30) & (res_clean.gap.abs() < .70)),
    ("|gap| >= 0.70%", res_clean.gap.abs() >= .70),
]:
    sub = res_clean[cond]
    w, n = wr(sub)
    print(f"  {lbl:18s} first-touch {w:5.0f}% (n={n})")

print("\nGap-aligned vs gap-opposed breakouts (|gap| >= 0.30%):")
g = res_clean[res_clean.gap.abs() >= 0.30]
aligned = g[((g.gap > 0) & (g.direction == "UP")) | ((g.gap < 0) & (g.direction == "DOWN"))]
opposed = g[((g.gap > 0) & (g.direction == "DOWN")) | ((g.gap < 0) & (g.direction == "UP"))]
w, n = wr(aligned); print(f"  aligned  {w:.0f}% (n={n})")
w, n = wr(opposed); print(f"  opposed  {w:.0f}% (n={n})")

print("\nBy weekday (0=Mon):")
for wd in range(5):
    sub = res_clean[res_clean.weekday == wd]
    w, n = wr(sub)
    print(f"  wd={wd}  first-touch {w:5.0f}% (n={n})")

print("\nBy entry time:")
for lbl, t0, t1 in [("10:00-11:00", time(10), time(11)), ("11:00-13:00", time(11), time(13)), ("13:00-16:00", time(13), time(16))]:
    sub = res_clean[[t0 <= t < t1 for t in res_clean.entry_time]]
    w, n = wr(sub)
    print(f"  {lbl}  first-touch {w:5.0f}% (n={n})")

# ============================================================================
# 2. VWAP +/-2SD FADE
# ============================================================================
print("\n" + "=" * 70)
print("2. VWAP BAND EXTREMES (+/-2sd) — fade vs continuation, 30/60-min fwd")
print("=" * 70)

band_events = []
for d in days:
    df = day_frames[d]
    if len(df) < 30:
        continue
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    v = df["Volume"].clip(lower=1)
    cum_pv = (tp * v).cumsum()
    cum_pv2 = (tp * tp * v).cumsum()
    cum_v = v.cumsum()
    vwap = cum_pv / cum_v
    sd = np.sqrt((cum_pv2 / cum_v - vwap ** 2).clip(lower=0))
    in_2sd_prev = False
    for j in range(12, len(df) - 6):   # after 10:30, need 30m fwd
        c = float(df["Close"].iloc[j])
        u2, l2 = float(vwap.iloc[j] + 2 * sd.iloc[j]), float(vwap.iloc[j] - 2 * sd.iloc[j])
        pos = "ABOVE" if c >= u2 else "BELOW" if c <= l2 else None
        if pos and not in_2sd_prev:   # first entry into the zone
            vw = float(vwap.iloc[j])
            fwd6 = float(df["Close"].iloc[min(j + 6, len(df) - 1)])
            fwd12 = float(df["Close"].iloc[min(j + 12, len(df) - 1)])
            sgn = -1 if pos == "ABOVE" else 1   # fade direction
            band_events.append(dict(
                day=d, pos=pos,
                fade_ret_30m=sgn * (fwd6 - c) / c * 100,
                fade_ret_60m=sgn * (fwd12 - c) / c * 100,
                reverted_to_1sd_60m=abs(fwd12 - vw) < abs(c - vw) * 0.5,
            ))
        in_2sd_prev = pos is not None

be = pd.DataFrame(band_events)
if len(be):
    print(f"Events (first entry into +/-2sd, after 10:30): n={len(be)}")
    print(f"Fade profitable 30m fwd: {(be.fade_ret_30m > 0).mean()*100:.0f}%   avg {be.fade_ret_30m.mean():+.3f}%")
    print(f"Fade profitable 60m fwd: {(be.fade_ret_60m > 0).mean()*100:.0f}%   avg {be.fade_ret_60m.mean():+.3f}%")
    print(f"Reverted >=50% toward VWAP within 60m: {be.reverted_to_1sd_60m.mean()*100:.0f}%")
    for p in ["ABOVE", "BELOW"]:
        sub = be[be.pos == p]
        if len(sub):
            print(f"  {p}_2SD: n={len(sub)}  fade-win-60m {(sub.fade_ret_60m>0).mean()*100:.0f}%  avg {sub.fade_ret_60m.mean():+.3f}%")
else:
    print("No 2sd events found")

# ============================================================================
# 3. EDR EXHAUSTION
# ============================================================================
print("\n" + "=" * 70)
print("3. EDR EXHAUSTION — after 85% of VIX-implied range is used")
print("=" * 70)

edr_events = []
for i, d in enumerate(days):
    df = day_frames[d]
    if len(df) < 30:
        continue
    # prior day's VIX close (what you'd know at the open)
    vix_prior = None
    for back in range(1, 6):
        if i - back >= 0 and days[i - back] in vix_close:
            vix_prior = vix_close[days[i - back]]
            break
    if vix_prior is None:
        continue
    day_open = float(df["Open"].iloc[0])
    edr = vix_prior / math.sqrt(252) / 100 * day_open
    hh, ll = day_open, day_open
    crossed = False
    for j in range(len(df)):
        hh = max(hh, float(df["High"].iloc[j]))
        ll = min(ll, float(df["Low"].iloc[j]))
        used = max(hh - day_open, day_open - ll) / edr * 100
        if used >= 85 and not crossed:
            crossed = True
            # how much further does the range extend by close?
            rest = df.iloc[j:]
            hh_end = float(rest["High"].max()) if len(rest) else hh
            ll_end = float(rest["Low"].min()) if len(rest) else ll
            used_end = max(max(hh_end, hh) - day_open, day_open - min(ll_end, ll)) / edr * 100
            dominant_up = (hh - day_open) >= (day_open - ll)
            c_now = float(df["Close"].iloc[j])
            c_end = float(df["Close"].iloc[-1])
            cont_ret = (c_end - c_now) / c_now * 100 * (1 if dominant_up else -1)
            edr_events.append(dict(
                day=d, cross_time=df.index[j].time(), used_end=used_end,
                cont_ret=cont_ret,
            ))
            break

ee = pd.DataFrame(edr_events)
print(f"Sessions hitting 85% EDR: {len(ee)}/{len(days)}")
if len(ee):
    print(f"Avg final range: {ee.used_end.mean():.0f}% of EDR  (median {ee.used_end.median():.0f}%)")
    print(f"Range extended past 120% after crossing 85%: {(ee.used_end >= 120).mean()*100:.0f}%")
    print(f"Continuation (close further in dominant direction): {(ee.cont_ret > 0).mean()*100:.0f}%  avg {ee.cont_ret.mean():+.2f}%")
    early = ee[[t < time(13) for t in ee.cross_time]]
    late = ee[[t >= time(13) for t in ee.cross_time]]
    if len(early):
        print(f"  crossed before 13:00 (n={len(early)}): continuation {(early.cont_ret>0).mean()*100:.0f}%  avg {early.cont_ret.mean():+.2f}%")
    if len(late):
        print(f"  crossed after  13:00 (n={len(late)}): continuation {(late.cont_ret>0).mean()*100:.0f}%  avg {late.cont_ret.mean():+.2f}%")

# ============================================================================
# 4. RSI DIVERGENCE
# ============================================================================
print("\n" + "=" * 70)
print("4. RSI(14, 5m) DIVERGENCE — forward 30/60-min return in divergence direction")
print("=" * 70)

div_events = []
for d in days:
    df = day_frames[d]
    closes = [float(x) for x in df["Close"]]
    if len(closes) < 30:
        continue
    rsi = _rsi_series(closes, 14)
    last_div_bar = -99
    for j in range(22, len(closes) - 6):
        div = _detect_divergence(closes[: j + 1], rsi[: j + 1], lookback=8)
        if div != "NONE" and j - last_div_bar > 6:   # dedupe clusters
            last_div_bar = j
            sgn = 1 if div == "BULLISH_DIV" else -1
            c = closes[j]
            f6 = closes[min(j + 6, len(closes) - 1)]
            f12 = closes[min(j + 12, len(closes) - 1)]
            div_events.append(dict(
                day=d, div=div,
                ret30=sgn * (f6 - c) / c * 100,
                ret60=sgn * (f12 - c) / c * 100,
            ))

de = pd.DataFrame(div_events)
if len(de):
    n_bear = int((de['div'] == 'BEARISH_DIV').sum())
    n_bull = int((de['div'] == 'BULLISH_DIV').sum())
    print(f"Divergence events: n={len(de)}  ({n_bear} bearish, {n_bull} bullish)")
    print(f"Profitable 30m: {(de.ret30 > 0).mean()*100:.0f}%  avg {de.ret30.mean():+.3f}%")
    print(f"Profitable 60m: {(de.ret60 > 0).mean()*100:.0f}%  avg {de.ret60.mean():+.3f}%")
    for dv in ["BEARISH_DIV", "BULLISH_DIV"]:
        sub = de[de['div'] == dv]
        if len(sub):
            print(f"  {dv}: n={len(sub)}  win-60m {(sub.ret60>0).mean()*100:.0f}%  avg {sub.ret60.mean():+.3f}%")
else:
    print("No divergence events")

# ============================================================================
# 5. PIVOT TOUCHES
# ============================================================================
print("\n" + "=" * 70)
print("5. PIVOT TOUCHES — bounce rate within 30 min of first touch (>=0.10% bounce)")
print("=" * 70)

piv_events = []
for i in range(1, len(days)):
    prev, cur = day_frames[days[i - 1]], day_frames[days[i]]
    if len(prev) < 10 or len(cur) < 20:
        continue
    ph, pl, pc = float(prev["High"].max()), float(prev["Low"].min()), float(prev["Close"].iloc[-1])
    pp, r1, r2, s1, s2 = _floor_pivots(ph, pl, pc)
    for name, lvl, kind in [("R1", r1, "RES"), ("R2", r2, "RES"), ("S1", s1, "SUP"), ("S2", s2, "SUP")]:
        # first touch after 10:00
        post = cur.between_time("10:00", "15:25")
        touched = None
        for j in range(len(post)):
            hi, lo = float(post["High"].iloc[j]), float(post["Low"].iloc[j])
            if lo <= lvl <= hi:
                touched = j
                break
        if touched is None:
            continue
        c = float(post["Close"].iloc[touched])
        f6 = float(post["Close"].iloc[min(touched + 6, len(post) - 1)])
        sgn = -1 if kind == "RES" else 1   # bounce direction
        piv_events.append(dict(level=name, bounce_ret=sgn * (f6 - c) / c * 100))

pe = pd.DataFrame(piv_events)
if len(pe):
    for name in ["R1", "R2", "S1", "S2"]:
        sub = pe[pe.level == name]
        if len(sub):
            print(f"  {name}: n={len(sub):2d}  bounce-profitable-30m {(sub.bounce_ret>0).mean()*100:3.0f}%  avg {sub.bounce_ret.mean():+.3f}%")
    print(f"  ALL: n={len(pe)}  {(pe.bounce_ret>0).mean()*100:.0f}%  avg {pe.bounce_ret.mean():+.3f}%")
else:
    print("No pivot touches")

print("\nDone.")
