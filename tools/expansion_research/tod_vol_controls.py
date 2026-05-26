"""
Is box_rng_atr a genuine market-state classifier, or a proxy for time-of-day /
volatility regime? (Srini's "most important hidden risk".)

The worry: box_rng_atr (prior 6-bar range / ATR) might separate re-timed from
net-new only because re-timed cluster in a particular session segment or vol
regime, not because it captures "auction has escaped balance." fold-3's AUC
wobble (0.54) is a hint that some dependency exists.

Cleanest test of "proxy vs orthogonal": if re-timed and net-new are NOT
separable by time-of-day or by ATR level (their hour/ATR AUCs ~ 0.5), yet
box_rng_atr separates them at AUC ~0.78, then box_rng_atr carries information
orthogonal to TOD/vol -> not a proxy. Plus a Simpson check: within-segment and
within-ATR-tercile AUCs (small n, directional only) and rank correlations.

Longs only. Research only.
"""
from __future__ import annotations
import sys, os, numpy as np, pandas as pd
from datetime import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expansion_proto import enrich, DATA, P as BASE, in_rth
from continuation_timing import Q, add_adx_slope
from entry_filter_proto import build_entries, rank_auc


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return np.nan
    ra = pd.Series(a[m]).rank().values
    rb = pd.Series(b[m]).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def seg(t: time):
    if t < time(10, 0):
        return "open"
    if t < time(13, 30):
        return "midday"
    return "close"


def main():
    df = add_adx_slope(enrich(pd.read_parquet(DATA), BASE), Q["adx_slope_k"])
    rows, _ = build_entries(df)
    # attach TOD + ATR-level at the entry bar
    rth = df[in_rth(df, BASE)]
    atr_terciles = np.nanpercentile(rth["atr"].values, [33.33, 66.67])
    for r in rows:
        i = r["idx"]; ts = df.index[i]
        r["box"] = r["feats"]["box_rng_atr"]
        r["min_since_open"] = (ts.hour - 8) * 60 + ts.minute - 30
        r["seg"] = seg(ts.time())
        r["atr"] = df["atr"].values[i]
        r["atr_reg"] = ("lo" if r["atr"] <= atr_terciles[0]
                        else "hi" if r["atr"] >= atr_terciles[1] else "mid")

    re = [r for r in rows if r["cohort"] == "retimed"]
    nn = [r for r in rows if r["cohort"] == "netnew"]
    print("=" * 92)
    print("PROXY CONTROL — is box_rng_atr market-state, or a TOD / volatility proxy?")
    print(f"re-timed n={len(re)}  net-new n={len(nn)}  (ATR terciles @ {atr_terciles[0]:.1f} / {atr_terciles[1]:.1f})")
    print("=" * 92)

    # ---- 1. can TOD or ATR alone separate the cohorts? -----------------
    auc_box = rank_auc([r["box"] for r in re], [r["box"] for r in nn])
    auc_min = rank_auc([r["min_since_open"] for r in re], [r["min_since_open"] for r in nn])
    auc_atr = rank_auc([r["atr"] for r in re], [r["atr"] for r in nn])
    print("\n-- cohort separability by each axis (AUC re-timed vs net-new) --")
    print(f"   box_rng_atr        AUC {auc_box:.2f}   <- the feature")
    print(f"   minutes-since-open AUC {auc_min:.2f}   (|AUC-0.5| small => TOD does NOT separate cohorts)")
    print(f"   ATR level          AUC {auc_atr:.2f}   (|AUC-0.5| small => vol level does NOT separate cohorts)")
    print("   READ: if box AUC is high while TOD/ATR AUC ~0.5, box carries orthogonal info -> not a proxy.")

    # ---- 2. does box_rng_atr co-move with TOD / ATR? -------------------
    allbox = [r["box"] for r in rows]
    print("\n-- does box_rng_atr co-move with TOD / vol? (Spearman, all relaxed entries) --")
    print(f"   corr(box, minutes-since-open) = {spearman(allbox, [r['min_since_open'] for r in rows]):+.2f}")
    print(f"   corr(box, ATR level)          = {spearman(allbox, [r['atr'] for r in rows]):+.2f}  "
          f"(near 0 expected — box is range/ATR, vol-normalized by construction)")

    # ---- 3. Simpson check: within-stratum separation -------------------
    print("\n-- within-stratum AUC (box_rng_atr; small n => directional only) --")
    print("   by session segment:")
    for s in ("open", "midday", "close"):
        p = [r["box"] for r in re if r["seg"] == s]
        q = [r["box"] for r in nn if r["seg"] == s]
        a = rank_auc(p, q)
        astr = f"{a:.2f}" if np.isfinite(a) else " n/a"
        print(f"     {s:7} re={len(p):2d} nn={len(q):2d}  AUC {astr}")
    print("   by ATR regime:")
    for g in ("lo", "mid", "hi"):
        p = [r["box"] for r in re if r["atr_reg"] == g]
        q = [r["box"] for r in nn if r["atr_reg"] == g]
        a = rank_auc(p, q)
        astr = f"{a:.2f}" if np.isfinite(a) else " n/a"
        print(f"     {g:7} re={len(p):2d} nn={len(q):2d}  AUC {astr}")

    # ---- 4. cohort composition by stratum (clustering?) ----------------
    print("\n-- cohort mix by stratum (is one cohort concentrated in a segment/regime?) --")
    for axis, key, vals in (("segment", "seg", ("open", "midday", "close")),
                            ("ATR reg", "atr_reg", ("lo", "mid", "hi"))):
        print(f"   {axis}:")
        for v in vals:
            r_ = sum(1 for r in re if r[key] == v)
            n_ = sum(1 for r in nn if r[key] == v)
            print(f"     {v:7} re-timed {r_:2d} ({r_/len(re)*100:3.0f}%)  "
                  f"net-new {n_:2d} ({n_/len(nn)*100:3.0f}%)")


if __name__ == "__main__":
    main()
