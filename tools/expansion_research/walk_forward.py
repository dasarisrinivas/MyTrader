"""
Walk-forward validation of the box_rng_atr gate on the relaxed continuation rule.

This tests a FALSIFIABLE structural hypothesis, not a feature search:
  H: a single frozen entry-time feature (box_rng_atr = prior 6-bar range / ATR)
     separates re-timed (real, already-expanded trends awaiting confirmation)
     from net-new (rotational drift that never matures) OUT OF SAMPLE, and a
     gate on it preserves the relaxed rule's edge over baseline OOS.

Discipline (pre-committed, do not relax after seeing results):
  * ONE feature, frozen. No stacking, no AND/OR combos, no adaptive gating.
  * Threshold set by a FIXED RULE (median of the TRAIN-window relaxed entries),
    never optimized against test outcomes.
  * Expanding window: block0 = initial train; folds 1..N-1 are OOS. Every OOS
    event is predicted exactly once; pooled OOS is the headline (per-fold counts
    are too small to read individually — shown only for stability).
  * Results segmented by trend vs rotational session.
  * Mechanism decomposition: does the OOS benefit come from SUPPRESSING LOSERS
    (regime-fragile) or PRESERVING MFE ASYMMETRY on re-timed winners (structural)?

Acceptance criteria (a priori; set before reading output):
  C1  pooled OOS AUC (re-timed vs net-new) >= 0.60
  C2  pooled OOS gated blended expectancy > base  AND  >= ungated relaxed
  C3  pooled OOS (gated - base) edge >= 50% of the in-sample (gated - base) edge
  C4  gated > base in BOTH trend and rotational buckets (no sign reversal)
  C5  threshold frozen / fixed-rule (structural — satisfied by construction)

Longs only. Research only; no live strategy code is touched.
"""
from __future__ import annotations
import sys, os, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expansion_proto import enrich, DATA, P as BASE, in_rth
from continuation_timing import Q, add_adx_slope, arm_masks, episodes
from timing_attribution import BASE_NAME, classify, realized_R
from entry_filter_proto import entry_features, rank_auc, RELAXED_NAME

K = 1.5
N_FOLDS = 5
FEATURE = "box_rng_atr"
# a priori acceptance thresholds
C1_AUC_MIN = 0.60
C3_RETAIN = 0.50


def build(df):
    """relaxed entries (re-timed/no-change/net-new) + base entries, with ts/feature/outcome."""
    masks = {n: m.values for n, m in arm_masks(df, Q).items()}
    eps = episodes(df, Q)
    coh = classify(df, eps, masks[BASE_NAME], masks[RELAXED_NAME], Q, (1.25, 1.5, 2.0))
    relaxed, base = [], []
    for cohort in ("retimed", "nochange", "netnew"):
        for r in coh[cohort]:
            i = r["vf"]; f = entry_features(df, i)
            if f is None or not np.isfinite(f[FEATURE]):
                continue
            relaxed.append(dict(idx=i, ts=df.index[i], date=df.index[i].date(),
                                cohort=cohort, x=f[FEATURE],
                                R=realized_R(df, i, K, Q), res=r["v"]["res"][K],
                                mfe=r["v"]["mfe_atr"], mae=r["v"]["mae_atr"]))
    for cohort in ("retimed", "nochange"):
        for r in coh[cohort]:
            i = r["bf"]
            base.append(dict(idx=i, ts=df.index[i], date=df.index[i].date(),
                             R=realized_R(df, i, K, Q), res=r["base"]["res"][K]))
    return relaxed, base


def day_labels(df):
    """trend vs rotational per RTH day: directional efficiency |close-open|/(hi-lo),
    median split. Analysis-only label (not used by the live gate)."""
    rth = df[in_rth(df, BASE)]
    eff = {}
    for d, g in rth.groupby("date"):
        rng = g["high"].max() - g["low"].min()
        if rng <= 0:
            continue
        eff[d] = abs(g["close"].iloc[-1] - g["open"].iloc[0]) / rng
    med = np.median(list(eff.values()))
    return {d: ("trend" if e >= med else "rot") for d, e in eff.items()}, med


def dexp(entries):
    """decided-only WR/expectancy (matches prior metric) + portfolio mean R + counts."""
    w = sum(1 for e in entries if e["res"] == 1)
    l = sum(1 for e in entries if e["res"] == -1)
    dec = w + l
    wr = w / dec * 100 if dec else 0.0
    exp = (w * K - l) / dec if dec else 0.0
    mR = float(np.mean([e["R"] for e in entries if e["R"] is not None])) if entries else 0.0
    return dict(n=len(entries), w=w, l=l, wr=wr, exp=exp, meanR=mR)


def auc_cohorts(entries):
    pos = [e["x"] for e in entries if e["cohort"] == "retimed"]
    neg = [e["x"] for e in entries if e["cohort"] == "netnew"]
    return rank_auc(pos, neg), len(pos), len(neg)


def main():
    df = add_adx_slope(enrich(pd.read_parquet(DATA), BASE), Q["adx_slope_k"])
    relaxed, base = build(df)
    labels, eff_med = day_labels(df)
    dates = sorted(set(df["date"]))
    blocks = [set(b) for b in np.array_split(dates, N_FOLDS)]

    print("=" * 100)
    print("WALK-FORWARD — box_rng_atr gate on relaxed continuation rule (ES 15m, longs)")
    print(f"{N_FOLDS} blocks, expanding window (block0 train-only, folds 1..{N_FOLDS-1} OOS, pooled)")
    print(f"feature FROZEN = {FEATURE}; threshold = median of train-window relaxed entries (fixed rule)")
    print(f"trend/rotational split at directional-efficiency median {eff_med:.2f}")
    print("=" * 100)

    # ---------- in-sample reference (full-sample median gate) -------------
    is_thr = np.median([e["x"] for e in relaxed])
    is_kept = [e for e in relaxed if e["x"] >= is_thr]
    is_gated, is_base = dexp(is_kept), dexp(base)
    is_edge = is_gated["exp"] - is_base["exp"]
    print(f"\nIN-SAMPLE (full-sample median thr={is_thr:.2f}): "
          f"base exp {is_base['exp']:+.2f} | relaxed-all {dexp(relaxed)['exp']:+.2f} | "
          f"gated {is_gated['exp']:+.2f}  -> IS edge over base {is_edge:+.2f}R")

    # ---------- expanding-window pooled OOS -------------------------------
    oos_kept, oos_all, oos_dropped = [], [], []
    print(f"\n{'fold':5} {'train d':>8} {'thr':>6} {'test re/nn/nc':>14} "
          f"{'OOS AUC':>8} {'gated exp':>10} {'base exp':>9}")
    print("-" * 70)
    oos_dates = set()
    for k in range(1, N_FOLDS):
        train_dates = set().union(*blocks[:k])
        test_dates = blocks[k]
        oos_dates |= test_dates
        train_rel = [e for e in relaxed if e["date"] in train_dates]
        if not train_rel:
            continue
        thr = np.median([e["x"] for e in train_rel])          # FIXED RULE
        test_rel = [e for e in relaxed if e["date"] in test_dates]
        test_base = [e for e in base if e["date"] in test_dates]
        kept = [e for e in test_rel if e["x"] >= thr]
        dropped = [e for e in test_rel if e["x"] < thr]
        oos_kept += kept; oos_all += test_rel; oos_dropped += dropped
        a, npos, nneg = auc_cohorts(test_rel)
        re_ = sum(1 for e in test_rel if e["cohort"] == "retimed")
        nn_ = sum(1 for e in test_rel if e["cohort"] == "netnew")
        nc_ = sum(1 for e in test_rel if e["cohort"] == "nochange")
        astr = f"{a:.2f}" if np.isfinite(a) else "  n/a"
        print(f"{k:>5} {len(train_dates):>8} {thr:>6.2f} {f'{re_}/{nn_}/{nc_}':>14} "
              f"{astr:>8} {dexp(kept)['exp']:>+10.2f} {dexp(test_base)['exp']:>+9.2f}")
    print("-" * 70)

    # ---------- pooled OOS headline --------------------------------------
    pooled_auc, npos, nneg = auc_cohorts(oos_all)
    g, a_all, base_oos = dexp(oos_kept), dexp(oos_all), dexp([e for e in base if e["date"] in oos_dates])
    oos_edge = g["exp"] - base_oos["exp"]
    print(f"\nPOOLED OOS ({npos} re-timed vs {nneg} net-new; {a_all['n']} relaxed entries)")
    print(f"  separation AUC (re-timed vs net-new)     : {pooled_auc:.2f}")
    print(f"  base (OOS windows)        : n={base_oos['n']:3d}  WR {base_oos['wr']:.0f}%  "
          f"exp {base_oos['exp']:+.2f}  meanR {base_oos['meanR']:+.2f}")
    print(f"  relaxed-all (ungated)     : n={a_all['n']:3d}  WR {a_all['wr']:.0f}%  "
          f"exp {a_all['exp']:+.2f}  meanR {a_all['meanR']:+.2f}")
    print(f"  relaxed + gate (kept)     : n={g['n']:3d}  WR {g['wr']:.0f}%  "
          f"exp {g['exp']:+.2f}  meanR {g['meanR']:+.2f}   -> OOS edge over base {oos_edge:+.2f}R")
    rk = sum(1 for e in oos_kept if e["cohort"] == "retimed")
    nk = sum(1 for e in oos_kept if e["cohort"] == "netnew")
    print(f"  gate keeps {rk}/{npos} re-timed, {nk}/{nneg} net-new")

    # ---------- regime segmentation --------------------------------------
    print("\n-- REGIME SPLIT (pooled OOS) --")
    for reg in ("trend", "rot"):
        kept_r = [e for e in oos_kept if labels.get(e["date"]) == reg]
        base_r = [e for e in base if e["date"] in oos_dates and labels.get(e["date"]) == reg]
        gk, bk = dexp(kept_r), dexp(base_r)
        print(f"   {reg:5}: gated n={gk['n']:3d} exp {gk['exp']:+.2f} | "
              f"base n={bk['n']:3d} exp {bk['exp']:+.2f} | edge {gk['exp']-bk['exp']:+.2f}R")

    # ---------- mechanism: suppression vs preserved asymmetry -------------
    print("\n-- MECHANISM (pooled OOS) --")
    drop = dexp(oos_dropped)
    drop_loser_rate = (oos_dropped and sum(1 for e in oos_dropped if e["res"] == -1) /
                       max(sum(1 for e in oos_dropped if e["res"] != 0), 1) * 100)
    print(f"   loser-suppression: dropped n={drop['n']} meanR {drop['meanR']:+.2f} "
          f"loser-rate {drop_loser_rate:.0f}%  (very negative => gate mostly discards losers)")
    kept_re = [e for e in oos_kept if e["cohort"] == "retimed"]
    all_re = [e for e in oos_all if e["cohort"] == "retimed"]
    if kept_re:
        print(f"   winner-quality: kept re-timed MFE {np.median([e['mfe'] for e in kept_re]):.2f} / "
              f"MAE {np.median([e['mae'] for e in kept_re]):.2f} xATR  "
              f"(IS re-timed was 2.22 / 0.61; asymmetry preserved => structural)")
    if all_re:
        print(f"   ref: ALL OOS re-timed MFE {np.median([e['mfe'] for e in all_re]):.2f} / "
              f"MAE {np.median([e['mae'] for e in all_re]):.2f}")

    # ---------- strict frozen-threshold variant --------------------------
    b0_thr = np.median([e["x"] for e in relaxed if e["date"] in blocks[0]])
    fz_kept = [e for e in oos_all if e["x"] >= b0_thr]
    print(f"\n-- STRICT VARIANT: single frozen thr={b0_thr:.2f} from block0, applied to all OOS --")
    fz = dexp(fz_kept)
    print(f"   gated exp {fz['exp']:+.2f} (n={fz['n']})  vs base {base_oos['exp']:+.2f}  "
          f"edge {fz['exp']-base_oos['exp']:+.2f}R")

    # ---------- pre-registered verdicts ----------------------------------
    print("\n" + "=" * 100)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 100)
    c1 = np.isfinite(pooled_auc) and pooled_auc >= C1_AUC_MIN
    c2 = (g["exp"] > base_oos["exp"]) and (g["exp"] >= a_all["exp"])
    c3 = (is_edge > 0) and (oos_edge >= C3_RETAIN * is_edge)
    tr = [e for e in oos_kept if labels.get(e["date"]) == "trend"]
    ro = [e for e in oos_kept if labels.get(e["date"]) == "rot"]
    trb = [e for e in base if e["date"] in oos_dates and labels.get(e["date"]) == "trend"]
    rob = [e for e in base if e["date"] in oos_dates and labels.get(e["date"]) == "rot"]
    c4 = (dexp(tr)["exp"] > dexp(trb)["exp"]) and (dexp(ro)["exp"] > dexp(rob)["exp"])
    def mark(b): return "PASS" if b else "FAIL"
    print(f"  C1 OOS AUC >= {C1_AUC_MIN}                : {mark(c1)}  (AUC {pooled_auc:.2f})")
    print(f"  C2 gated > base AND >= ungated relaxed : {mark(c2)}  "
          f"(gated {g['exp']:+.2f}, base {base_oos['exp']:+.2f}, ungated {a_all['exp']:+.2f})")
    print(f"  C3 OOS edge >= {C3_RETAIN:.0%} of IS edge       : {mark(c3)}  "
          f"(OOS {oos_edge:+.2f}R vs IS {is_edge:+.2f}R)")
    print(f"  C4 gated > base in BOTH regimes        : {mark(c4)}  "
          f"(trend {dexp(tr)['exp']-dexp(trb)['exp']:+.2f}R, rot {dexp(ro)['exp']-dexp(rob)['exp']:+.2f}R)")
    print(f"  C5 threshold frozen / fixed-rule       : PASS  (structural)")
    print(f"\n  OVERALL: {'CONFIRMED' if (c1 and c2 and c3 and c4) else 'NOT CONFIRMED'} "
          f"(thin OOS sample — read magnitudes as directional)")


if __name__ == "__main__":
    main()
