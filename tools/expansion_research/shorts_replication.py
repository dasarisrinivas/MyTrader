#!/usr/bin/env python3
"""
Short-side replication of the box_rng_atr discriminator — held frozen.

Mirrors the entire long-side chain for shorts with NO new degrees of freedom:
same feature (box_rng_atr), SAME frozen threshold 3.01 (re-deriving on short
data would be a refit), no combos, no session logic, no vol conditioning. The
only changes are the direction mirror: bearish stack, rally-to-EMA9 touch,
RSI band reflected [40,70]->[30,60], short sim/exits. Data used as-is (the
Mar-23 pre-market spike is NOT cleaned — cleaning then re-running would reopen
the optimization loop).

Tests three hidden assumptions at once (Srini): symmetry, microstructure
dependence, and mechanism stability. A short-side PASS (within-segment
separation preserved, re-timed asymmetry preserved, not pure loser-suppression)
graduates this from "timing overlay" to "credible market-state discriminator."
A short-side FAIL is also informative: the feature would then be specific to
bullish-continuation-after-expansion microstructure — itself a tradable insight.

    python tools/expansion_research/shorts_replication.py
"""
from __future__ import annotations
import sys, os, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expansion_proto import enrich, P as BASE, in_rth
from continuation_timing import Q, add_adx_slope, first_true
from entry_filter_proto import entry_features, rank_auc
from walk_forward import day_labels, dexp
from tod_vol_controls import seg
from forward_test import TRAIN, FWD, enrich_with_warmup

THRESHOLD = 3.01      # FROZEN (long-derived); deliberately NOT refit for shorts
K = 1.5
FEATURE = "box_rng_atr"
RSI_LO, RSI_HI = 30, 60   # mirror of long [40,70] reflected about 50
N_PERM = 5000
RNG = np.random.default_rng(11)


# ---- direction-mirrored arming / episodes / sim -------------------------
def short_arm_masks(df, Q):
    e9, e21, e50 = df["ema9"], df["ema21"], df["ema50"]
    touch = df["high"] >= e9 * (1 - Q["touch_pct"])     # rallied UP to ema9
    closeok = df["close"] < e9                          # but closed back below
    bear = df["close"] < df["open"]
    rsiok = (df["rsi"] >= RSI_LO) & (df["rsi"] <= RSI_HI)
    rth = in_rth(df, BASE)
    full_stack = (e9 < e21) & (e21 < e50)
    loose_stack = (e9 < e21) & (df["close"] < e50)
    adx_abs = df["adx"] >= Q["adx_abs"]
    adx_slope = (df["adx_slope"] >= Q["adx_slope_min"]) & (df["adx"] >= Q["adx_slope_floor"])
    common = touch & closeok & bear & rsiok & rth
    return {
        "BASE":  (common & full_stack & adx_abs).values,
        "V3":    (common & loose_stack & (adx_abs | adx_slope)).values,
    }


def short_episodes(df, Q):
    e9, e21, e50 = df["ema9"].values, df["ema21"].values, df["ema50"].values
    rth = in_rth(df, BASE).values
    dates = df["date"].values
    ctx = (e9 < e21) & (df["close"].values < e50) & rth
    eps, n, i = [], len(df), 0
    while i < n:
        if ctx[i]:
            j = i
            while j + 1 < n and ctx[j + 1] and dates[j + 1] == dates[i]:
                j += 1
            if j - i >= 1:
                eps.append((i, j))
            i = j + 1
        else:
            i += 1
    return eps


def short_sim(df, i, Q, k_targets):
    H, L, C, A = (df["high"].values, df["low"].values, df["close"].values, df["atr"].values)
    n = len(df); entry = C[i]; atr = A[i]
    if not np.isfinite(atr) or atr <= 0:
        return None
    mfe = mae = 0.0
    sl = entry + Q["stop_atr"] * atr                    # stop ABOVE for shorts
    res = {k: 0 for k in k_targets}; done = {k: False for k in k_targets}
    for j in range(i + 1, min(i + 1 + Q["hold"], n)):
        mfe = max(mfe, entry - L[j]); mae = max(mae, H[j] - entry)   # favorable=down
        for k in k_targets:
            if done[k]:
                continue
            tp = entry - k * atr                        # target BELOW
            if H[j] >= sl:
                res[k] = -1; done[k] = True
            elif L[j] <= tp:
                res[k] = +1; done[k] = True
    return dict(mfe=mfe, mae=mae, mfe_atr=mfe / atr, mae_atr=mae / atr, atr=atr, res=res)


def short_realized_R(df, i, k, Q):
    H, L, C, A = (df["high"].values, df["low"].values, df["close"].values, df["atr"].values)
    n = len(df); entry = C[i]; atr = A[i]
    if not np.isfinite(atr) or atr <= 0:
        return None
    sl = entry + Q["stop_atr"] * atr; tp = entry - k * atr; last = i
    for j in range(i + 1, min(i + 1 + Q["hold"], n)):
        last = j
        if H[j] >= sl:
            return -1.0
        if L[j] <= tp:
            return float(k)
    return (entry - C[last]) / atr                      # timeout: favorable=down


def classify_short(df, eps, base_mask, relaxed_mask, k_targets):
    coh = dict(retimed=[], nochange=[], netnew=[], bug=[])
    for (a, b) in eps:
        bf = first_true(base_mask, a, b); vf = first_true(relaxed_mask, a, b)
        if bf is None and vf is None:
            continue
        if bf is not None and vf is None:
            coh["bug"].append((a, b)); continue
        if bf is None:
            rv = short_sim(df, vf, Q, k_targets)
            if rv:
                coh["netnew"].append(dict(vf=vf, bf=None, v=rv, base=None)); continue
        rv = short_sim(df, vf, Q, k_targets); rb = short_sim(df, bf, Q, k_targets)
        if not rv or not rb:
            continue
        rec = dict(vf=vf, bf=bf, v=rv, base=rb)
        coh["retimed" if vf < bf else "nochange"].append(rec)
    return coh


def build_short(df, eval_dates):
    masks = short_arm_masks(df, Q)
    eps = short_episodes(df, Q)
    coh = classify_short(df, eps, masks["BASE"], masks["V3"], (1.25, 1.5, 2.0))
    rth = df[in_rth(df, BASE)]
    terc = np.nanpercentile(rth["atr"].values, [33.33, 66.67]) if len(rth) else (np.nan, np.nan)
    relaxed, base = [], []
    for cohort in ("retimed", "nochange", "netnew"):
        for r in coh[cohort]:
            i = r["vf"]; d = df.index[i].date()
            if d not in eval_dates:
                continue
            f = entry_features(df, i)
            if f is None or not np.isfinite(f[FEATURE]):
                continue
            ts = df.index[i]; atr = df["atr"].values[i]
            relaxed.append(dict(cohort=cohort, x=f[FEATURE], R=short_realized_R(df, i, K, Q),
                                res=r["v"]["res"][K], mfe=r["v"]["mfe_atr"], mae=r["v"]["mae_atr"],
                                seg=seg(ts.time()), date=d,
                                atr_reg=("lo" if atr <= terc[0] else "hi" if atr >= terc[1] else "mid")))
    for cohort in ("retimed", "nochange"):
        for r in coh[cohort]:
            i = r["bf"]; d = df.index[i].date()
            if d in eval_dates:
                base.append(dict(R=short_realized_R(df, i, K, Q), res=r["base"]["res"][K], date=d))
    return relaxed, base


def auc_in(rows, pred):
    return rank_auc([r["x"] for r in rows if r["cohort"] == "retimed" and pred(r)],
                    [r["x"] for r in rows if r["cohort"] == "netnew" and pred(r)])


def metrics(df, eval_dates):
    relaxed, base = build_short(df, eval_dates)
    re = [r for r in relaxed if r["cohort"] == "retimed"]
    nn = [r for r in relaxed if r["cohort"] == "netnew"]
    nc = [r for r in relaxed if r["cohort"] == "nochange"]
    kept = [r for r in relaxed if r["x"] >= THRESHOLD]
    drop = [r for r in relaxed if r["x"] < THRESHOLD]
    labels, _ = day_labels(df[df["date"].isin(eval_dates)]) if eval_dates else ({}, 0)
    m = dict(
        n_re=len(re), n_nn=len(nn), n_nc=len(nc), n_rel=len(relaxed), n_base=len(base),
        re_exp=dexp(re)["exp"], nn_exp=dexp(nn)["exp"],
        auc_pool=auc_in(relaxed, lambda r: True),
        auc_open=auc_in(relaxed, lambda r: r["seg"] == "open"),
        auc_mid=auc_in(relaxed, lambda r: r["seg"] == "midday"),
        base_exp=dexp(base)["exp"], rel_exp=dexp(relaxed)["exp"], gated_exp=dexp(kept)["exp"],
        drop_meanR=float(np.mean([r["R"] for r in drop if r["R"] is not None])) if drop else 0.0,
        drop_loser=(sum(1 for r in drop if r["res"] == -1) /
                    max(sum(1 for r in drop if r["res"] != 0), 1) * 100),
        kept_re_mfe=float(np.median([r["mfe"] for r in kept if r["cohort"] == "retimed"])) if any(r["cohort"]=="retimed" for r in kept) else float("nan"),
        kept_re_mae=float(np.median([r["mae"] for r in kept if r["cohort"] == "retimed"])) if any(r["cohort"]=="retimed" for r in kept) else float("nan"),
        re_mfe=float(np.median([r["mfe"] for r in re])) if re else float("nan"),
        re_mae=float(np.median([r["mae"] for r in re])) if re else float("nan"),
    )
    m["edge"] = m["gated_exp"] - m["base_exp"]
    for reg in ("trend", "rot"):
        kr = [r for r in kept if labels.get(r["date"]) == reg]
        br = [r for r in base if labels.get(r["date"]) == reg]
        m[f"edge_{reg}"] = dexp(kr)["exp"] - dexp(br)["exp"]
    m["_relaxed"] = relaxed
    return m


def perm_p(relaxed):
    re = [r["x"] for r in relaxed if r["cohort"] == "retimed"]
    nn = [r["x"] for r in relaxed if r["cohort"] == "netnew"]
    obs = abs(rank_auc(re, nn) - 0.5)
    pooled = np.array(re + nn, float); n_re = len(re)
    ge = 0
    for _ in range(N_PERM):
        p = RNG.permutation(len(pooled))
        a = rank_auc(pooled[p[:n_re]], pooled[p[n_re:]])
        if np.isfinite(a) and abs(a - 0.5) >= obs:
            ge += 1
    return (ge + 1) / (N_PERM + 1)


def fa(a):
    return f"{a:.2f}" if np.isfinite(a) else " n/a"


def main():
    train = pd.read_parquet(TRAIN)
    df_is = add_adx_slope(enrich(train, BASE), Q["adx_slope_k"])
    is_dates = set(df_is["date"])
    is_m = metrics(df_is, is_dates)
    p_is = perm_p(is_m["_relaxed"])

    fw = None
    if FWD.exists():
        df_fw, fstart = enrich_with_warmup(FWD, TRAIN, warmup_bars=60)
        fw_dates = {d for d in df_fw["date"] if d >= fstart.date()}
        fw = metrics(df_fw, fw_dates)

    print("=" * 92)
    print("SHORT-SIDE REPLICATION — box_rng_atr >= 3.01 gate (FROZEN, long-derived), ES 15m SHORTS")
    print(f"RSI band [{RSI_LO},{RSI_HI}] (mirror of long [40,70]); threshold NOT refit; data as-is")
    print("=" * 92)
    cols = ["IN-SAMPLE", "FORWARD"] if fw else ["IN-SAMPLE"]
    def row(name, ai, af=None):
        s = f"{name:32} {ai:>16}"
        if fw:
            s += f" {af:>16}"
        print(s)
    print(f"\n{'metric':32} {'IN-SAMPLE':>16}" + (f" {'FORWARD':>16}" if fw else ""))
    print("-" * (50 + (17 if fw else 0)))
    g = lambda m, k: m[k] if m else None
    row("re-timed / net-new / no-change",
        f"{is_m['n_re']}/{is_m['n_nn']}/{is_m['n_nc']}", f"{fw['n_re']}/{fw['n_nn']}/{fw['n_nc']}" if fw else None)
    row("re-timed exp / net-new exp", f"{is_m['re_exp']:+.2f}/{is_m['nn_exp']:+.2f}",
        f"{fw['re_exp']:+.2f}/{fw['nn_exp']:+.2f}" if fw else None)
    row("re-timed MFE/MAE (all)", f"{is_m['re_mfe']:.2f}/{is_m['re_mae']:.2f}",
        f"{fw['re_mfe']:.2f}/{fw['re_mae']:.2f}" if fw else None)
    row("AUC pooled", fa(is_m['auc_pool']), fa(fw['auc_pool']) if fw else None)
    row("AUC open / midday", f"{fa(is_m['auc_open'])}/{fa(is_m['auc_mid'])}",
        f"{fa(fw['auc_open'])}/{fa(fw['auc_mid'])}" if fw else None)
    row("perm p (single-feature, IS)", f"{p_is:.4f}", "" if fw else None)
    row("base / ungated / gated exp", f"{is_m['base_exp']:+.2f}/{is_m['rel_exp']:+.2f}/{is_m['gated_exp']:+.2f}",
        f"{fw['base_exp']:+.2f}/{fw['rel_exp']:+.2f}/{fw['gated_exp']:+.2f}" if fw else None)
    row("edge (gated-base)", f"{is_m['edge']:+.2f}", f"{fw['edge']:+.2f}" if fw else None)
    row("edge trend / rot", f"{is_m['edge_trend']:+.2f}/{is_m['edge_rot']:+.2f}",
        f"{fw['edge_trend']:+.2f}/{fw['edge_rot']:+.2f}" if fw else None)
    row("kept re-timed MFE/MAE", f"{is_m['kept_re_mfe']:.2f}/{is_m['kept_re_mae']:.2f}",
        f"{fw['kept_re_mfe']:.2f}/{fw['kept_re_mae']:.2f}" if fw else None)
    row("dropped meanR / loser%", f"{is_m['drop_meanR']:+.2f}/{is_m['drop_loser']:.0f}%",
        f"{fw['drop_meanR']:+.2f}/{fw['drop_loser']:.0f}%" if fw else None)

    # ---- three success criteria (judged on the richer of IS/FWD) -------
    print("\n" + "=" * 92)
    print("SHORT-SIDE SUCCESS CRITERIA (vs long-side signature)")
    print("=" * 92)
    print("Long-side ref: attribution re-timed>>net-new; IS AUC 0.78 (within-seg 0.68/0.67); "
          "fwd within-seg 0.85/0.75; re-timed asymmetry ~2-2.3 / ~0.6-0.8")
    sign_ok = is_m["re_exp"] > is_m["nn_exp"]
    sep_ok = np.isfinite(is_m["auc_pool"]) and is_m["auc_pool"] >= 0.60 and p_is < 0.05
    asym_ok = np.isfinite(is_m["re_mfe"]) and is_m["re_mfe"] >= 1.6 and is_m["re_mae"] <= 1.0
    supp = is_m["drop_meanR"]
    print(f"  1. attribution signature replicates (re-timed exp > net-new exp): "
          f"{'YES' if sign_ok else 'NO'}  ({is_m['re_exp']:+.2f} vs {is_m['nn_exp']:+.2f})")
    print(f"  2. box_rng_atr separates (IS AUC>=0.60 & perm p<0.05): "
          f"{'YES' if sep_ok else 'NO'}  (AUC {fa(is_m['auc_pool'])}, p {p_is:.4f})")
    print(f"  3. re-timed asymmetry preserved (MFE>=1.6, MAE<=1.0): "
          f"{'YES' if asym_ok else 'NO'}  ({is_m['re_mfe']:.2f}/{is_m['re_mae']:.2f})")
    print(f"  (mechanism check: dropped meanR {supp:+.2f} — strongly negative => leaning loser-suppression)")
    verdict = "REPLICATES — credible market-state discriminator (both sides)" if (sign_ok and sep_ok and asym_ok) \
        else "DOES NOT cleanly replicate — feature may be specific to LONG continuation microstructure (itself tradable insight)"
    print(f"\n  SHORT-SIDE VERDICT: {verdict}")
    print("  (samples are thin on both sides — treat as directional; do not refit anything.)")


if __name__ == "__main__":
    main()
