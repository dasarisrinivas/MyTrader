"""
Decision-time discriminator prototype: can entry-time observables separate the
RE-TIMED cohort (relaxed arms early, base confirms later -> +0.88R) from the
NET-NEW cohort (base never confirms -> -0.35R)?

If yes, the confirmation lag is partly removable: a live filter could keep the
re-timed-like setups and drop the net-new-like ones, recovering the timing edge
without the participation drag. If no, the lag is the unavoidable price of the
EMA stack's trend-quality discrimination.

Method (deliberately conservative — only 25 vs 32 labeled points):
  1. At each relaxed (V3) arming bar, compute CAUSAL entry-time features only.
  2. Univariate rank-AUC per feature = P(feature_retimed > feature_netnew).
     No multi-feature classifier (would overfit 57 rows).
  3. Multiple-comparison-aware permutation test: shuffle labels, recompute the
     MAX |AUC-0.5| across all features, repeat -> p-value for the best feature,
     so "we looked at a dozen features" is paid for.
  4. For the top discriminator, apply a principled threshold as an extra gate on
     the FULL relaxed population and report what it keeps/drops + expectancy.
     (Threshold chosen on the same data -> optimistic; flagged as such.)

Longs only. Research only; no live strategy code is touched.
"""
from __future__ import annotations
import sys, os, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expansion_proto import enrich, DATA, P as BASE
from continuation_timing import Q, add_adx_slope, arm_masks, episodes, expectancy
from timing_attribution import BASE_NAME, classify, realized_R

RELAXED_NAME = "V3 slope OR + loose-stack"
HEADLINE_K = 1.5
SLOPE_K = 3          # bars for ema/adx slope (45 min)
N_PERM = 5000
RNG = np.random.default_rng(7)


def entry_features(df, i):
    """All features use data up to & including bar i (enrich() is fully causal:
    ewm adjust=False, trailing rolling, box uses shift(1)). No look-ahead."""
    e9, e21, e50 = df["ema9"].values, df["ema21"].values, df["ema50"].values
    C, H, L = df["close"].values, df["high"].values, df["low"].values
    atr = df["atr"].values[i]
    body, rng_, tr = df["body"].values[i], df["rng"].values[i], df["tr"].values[i]
    vsma = df["vol_sma"].values[i]
    box = df["box_rng"].values[i]
    if not np.isfinite(atr) or atr <= 0 or i < SLOPE_K:
        return None
    return {
        "ema9_slope":    (e9[i] - e9[i - SLOPE_K]) / atr,
        "ema21_slope":   (e21[i] - e21[i - SLOPE_K]) / atr,
        "ema50_slope":   (e50[i] - e50[i - SLOPE_K]) / atr,
        "close_vs_ema50": (C[i] - e50[i]) / atr,
        "close_vs_ema21": (C[i] - e21[i]) / atr,
        "stack_gap":     (e21[i] - e50[i]) / atr,   # <0 => stack not yet matured
        "adx":           df["adx"].values[i],
        "adx_slope":     df["adx_slope"].values[i],
        "rsi":           df["rsi"].values[i],
        "body_eff":      body / rng_ if rng_ and np.isfinite(rng_) else np.nan,
        "tr_atr":        tr / atr,
        "box_rng_atr":   box / atr if np.isfinite(box) else np.nan,
        "vol_ratio":     df["volume"].values[i] / vsma if vsma and np.isfinite(vsma) else np.nan,
        "pullback_depth": (e9[i] - L[i]) / atr,     # how deep the low dipped below ema9
    }


def rank_auc(pos, neg):
    """AUC = P(random pos > random neg) via Mann-Whitney, ties at 0.5."""
    pos = np.asarray([x for x in pos if np.isfinite(x)], float)
    neg = np.asarray([x for x in neg if np.isfinite(x)], float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().values
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def build_entries(df):
    masks = {n: m.values for n, m in arm_masks(df, Q).items()}
    eps = episodes(df, Q)
    coh = classify(df, eps, masks[BASE_NAME], masks[RELAXED_NAME], Q, (1.25, 1.5, 2.0))
    rows = []
    for cohort in ("retimed", "nochange", "netnew"):
        for r in coh[cohort]:
            i = r["vf"]
            f = entry_features(df, i)
            if f is None:
                continue
            rows.append(dict(idx=i, cohort=cohort, feats=f,
                             R=realized_R(df, i, HEADLINE_K, Q),
                             res15=r["v"]["res"][HEADLINE_K]))
    return rows, coh


def decided_stats(Rlist, reslist):
    """blended expectancy: decided-only WR/exp (matches prior metric) +
    portfolio mean R (timeouts marked-to-market)."""
    w = sum(1 for r in reslist if r == 1)
    l = sum(1 for r in reslist if r == -1)
    dec = w + l
    wr = w / dec * 100 if dec else 0.0
    exp = (w * HEADLINE_K - l) / dec if dec else 0.0
    meanR = float(np.mean([r for r in Rlist if r is not None])) if Rlist else 0.0
    return wr, exp, meanR, w, l, len(Rlist) - dec


def main():
    df = add_adx_slope(enrich(pd.read_parquet(DATA), BASE), Q["adx_slope_k"])
    rows, coh = build_entries(df)
    feat_names = list(rows[0]["feats"].keys())

    re_rows = [r for r in rows if r["cohort"] == "retimed"]
    nn_rows = [r for r in rows if r["cohort"] == "netnew"]
    nc_rows = [r for r in rows if r["cohort"] == "nochange"]

    print("=" * 100)
    print("DECISION-TIME DISCRIMINATOR — can entry-time observables separate "
          "RE-TIMED from NET-NEW?")
    print(f"ES 15m  |  relaxed rule: {RELAXED_NAME}  |  target {HEADLINE_K}R  |  "
          f"slope window {SLOPE_K} bars")
    print(f"labeled: RE-TIMED n={len(re_rows)} (+edge)   NET-NEW n={len(nn_rows)} (-edge)   "
          f"NO-CHANGE n={len(nc_rows)} (=base)")
    print("=" * 100)

    # ---- 1. univariate separation --------------------------------------
    aucs = {}
    for fn in feat_names:
        pos = [r["feats"][fn] for r in re_rows]
        neg = [r["feats"][fn] for r in nn_rows]
        aucs[fn] = rank_auc(pos, neg)
    order = sorted(feat_names, key=lambda f: -abs((aucs[f] or 0.5) - 0.5))

    print(f"\n{'feature':16} {'AUC':>6} {'sep':>6} {'retimed med':>12} "
          f"{'netnew med':>11} {'direction':>22}")
    print("-" * 80)
    for fn in order:
        a = aucs[fn]
        rm = np.nanmedian([r["feats"][fn] for r in re_rows])
        nm = np.nanmedian([r["feats"][fn] for r in nn_rows])
        if a >= 0.5:
            d = "higher in RE-TIMED"
        else:
            d = "higher in NET-NEW"
        print(f"{fn:16} {a:>6.2f} {abs(a-0.5):>6.2f} {rm:>12.2f} {nm:>11.2f} {d:>22}")

    # ---- 2. multiple-comparison-aware permutation test -----------------
    best_feat = order[0]
    best_sep = abs(aucs[best_feat] - 0.5)
    pooled = re_rows + nn_rows
    n_re = len(re_rows)
    feat_arrays = {fn: np.array([r["feats"][fn] for r in pooled], float) for fn in feat_names}
    ge = 0
    for _ in range(N_PERM):
        perm = RNG.permutation(len(pooled))
        pos_idx, neg_idx = perm[:n_re], perm[n_re:]
        max_sep = 0.0
        for fn in feat_names:
            arr = feat_arrays[fn]
            a = rank_auc(arr[pos_idx], arr[neg_idx])
            if np.isfinite(a):
                max_sep = max(max_sep, abs(a - 0.5))
        if max_sep >= best_sep:
            ge += 1
    p_fw = (ge + 1) / (N_PERM + 1)
    print(f"\nbest discriminator: {best_feat}  (AUC {aucs[best_feat]:.2f}, sep {best_sep:.2f})")
    print(f"family-wise permutation p (max-sep across {len(feat_names)} features, "
          f"{N_PERM} shuffles): p = {p_fw:.4f}")
    print("  -> p<0.05: separation survives the multiple-comparison penalty (real candidate)")
    print("  -> p>0.05: best feature is within what label-shuffling produces by chance")

    # ---- 3. gate effect on the FULL relaxed population -----------------
    # principled threshold: median of the best feature across all relaxed entries
    allvals = np.array([r["feats"][best_feat] for r in rows], float)
    thr = np.nanmedian(allvals)
    # direction: keep the side that re-timed sits on
    keep_high = aucs[best_feat] >= 0.5
    def passes(r):
        v = r["feats"][best_feat]
        if not np.isfinite(v):
            return False
        return (v >= thr) if keep_high else (v <= thr)

    print(f"\n-- GATE: keep relaxed entries with {best_feat} "
          f"{'>=' if keep_high else '<='} {thr:.2f} (median split; IN-SAMPLE, optimistic) --")
    base_paired = [r["base"] for r in coh["retimed"]] + [r["base"] for r in coh["nochange"]]
    bwr, bexp, bR, *_ = decided_stats(
        [realized_R(df, s_i["bf"], HEADLINE_K, Q) for s_i in coh["retimed"] + coh["nochange"]],
        [s_i["base"]["res"][HEADLINE_K] for s_i in coh["retimed"] + coh["nochange"]])
    print(f"   {'population':28} {'n':>4} {'reK/ncK/nnK':>12} {'WR':>5} {'exp':>7} {'meanR':>7}")
    awr, aexp, aR, *_ = decided_stats([r["R"] for r in rows], [r["res15"] for r in rows])
    print(f"   {'BASE (paired)':28} {len(coh['retimed'])+len(coh['nochange']):>4} "
          f"{'-':>12} {bwr:>4.0f}% {bexp:>+7.2f} {bR:>+7.2f}")
    print(f"   {'RELAXED unfiltered':28} {len(rows):>4} "
          f"{f'{len(re_rows)}/{len(nc_rows)}/{len(nn_rows)}':>12} {awr:>4.0f}% {aexp:>+7.2f} {aR:>+7.2f}")
    kept = [r for r in rows if passes(r)]
    drop = [r for r in rows if not passes(r)]
    for label, grp in (("RELAXED + gate (kept)", kept), ("  gate dropped", drop)):
        reK = sum(1 for r in grp if r["cohort"] == "retimed")
        ncK = sum(1 for r in grp if r["cohort"] == "nochange")
        nnK = sum(1 for r in grp if r["cohort"] == "netnew")
        wr, exp, mR, *_ = decided_stats([r["R"] for r in grp], [r["res15"] for r in grp])
        print(f"   {label:28} {len(grp):>4} {f'{reK}/{ncK}/{nnK}':>12} "
              f"{wr:>4.0f}% {exp:>+7.2f} {mR:>+7.2f}")
    reK = sum(1 for r in kept if r["cohort"] == "retimed")
    nnK = sum(1 for r in kept if r["cohort"] == "netnew")
    print(f"\n   gate retains {reK}/{len(re_rows)} re-timed, "
          f"{nnK}/{len(nn_rows)} net-new "
          f"(ideal: keep re-timed, drop net-new).")
    print("   NOTE: threshold picked on this same data; treat magnitudes as an upper "
          "bound, not an expectation. The AUC + permutation p is the honest signal.")


if __name__ == "__main__":
    main()
