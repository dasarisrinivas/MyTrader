#!/usr/bin/env python3
"""
Frozen forward test — executes FORWARD_TEST_PROTOCOL.md verbatim.

Applies the FROZEN rule + gate (box_rng_atr >= 3.01) to the unseen Jan-2026 ->
present slice and prints the pre-registered F1-F6 verdict. Computes the
in-sample reference from the training parquet itself so the comparison is always
self-consistent. Takes NO tunable arguments (per protocol §9).

    python tools/expansion_research/forward_test.py            # real forward test
    python tools/expansion_research/forward_test.py --selftest # sandbox smoke test

Reads : data/ib/ES_15m_1y.parquet (train + warm-up), data/ib/ES_15m_fwd_2026.parquet
Writes: tools/expansion_research/forward_test_result.txt
"""
from __future__ import annotations
import sys, os, contextlib
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expansion_proto import enrich, P as BASE, in_rth
from continuation_timing import Q, add_adx_slope, arm_masks, episodes
from timing_attribution import BASE_NAME, classify, realized_R
from entry_filter_proto import entry_features, rank_auc, RELAXED_NAME
from walk_forward import day_labels, dexp
from tod_vol_controls import seg

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/ib/ES_15m_1y.parquet"
FWD = ROOT / "data/ib/ES_15m_fwd_2026.parquet"
RESULT = Path(__file__).resolve().parent / "forward_test_result.txt"

# ---- FROZEN per protocol (do not change) --------------------------------
THRESHOLD = 3.01
K = 1.5
FEATURE = "box_rng_atr"
# pre-registered acceptance bands (FORWARD_TEST_PROTOCOL.md §6)
F1_AUC = 0.62      # within-segment (open & midday)
F2_AUC = 0.60      # pooled
F3_EDGE = 0.12     # gated-minus-base R
F5_MFE, F5_MAE = 1.8, 0.8
MIN_EVENTS = 8


def enrich_with_warmup(data_path, warmup_path, warmup_bars=60):
    """Enrich the slice with a lead-in tail from the prior file so EMA/ATR/ADX
    are warm at the slice's first bar; tag warm-up rows for exclusion."""
    df = pd.read_parquet(data_path)
    start = df.index.min()
    if warmup_path and Path(warmup_path).exists() and warmup_path != data_path:
        w = pd.read_parquet(warmup_path)
        w = w[w.index < start].tail(warmup_bars)
        df = pd.concat([w, df]).sort_index()
        df = df[~df.index.duplicated(keep="first")]
    df = add_adx_slope(enrich(df, BASE), Q["adx_slope_k"])
    return df, start


def build(df, eval_dates):
    """relaxed + base entries whose entry bar date is in eval_dates."""
    masks = {n: m.values for n, m in arm_masks(df, Q).items()}
    eps = episodes(df, Q)
    coh = classify(df, eps, masks[BASE_NAME], masks[RELAXED_NAME], Q, (1.25, 1.5, 2.0))
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
            relaxed.append(dict(cohort=cohort, x=f[FEATURE], R=realized_R(df, i, K, Q),
                                res=r["v"]["res"][K], mfe=r["v"]["mfe_atr"], mae=r["v"]["mae_atr"],
                                seg=seg(ts.time()), date=d,
                                atr_reg=("lo" if atr <= terc[0] else "hi" if atr >= terc[1] else "mid")))
    for cohort in ("retimed", "nochange"):
        for r in coh[cohort]:
            i = r["bf"]; d = df.index[i].date()
            if d in eval_dates:
                base.append(dict(R=realized_R(df, i, K, Q), res=r["base"]["res"][K], date=d))
    return relaxed, base


def auc_in(rows, pred):
    return rank_auc([r["x"] for r in rows if r["cohort"] == "retimed" and pred(r)],
                    [r["x"] for r in rows if r["cohort"] == "netnew" and pred(r)])


def metrics(df, eval_dates, label):
    relaxed, base = build(df, eval_dates)
    re = [r for r in relaxed if r["cohort"] == "retimed"]
    nn = [r for r in relaxed if r["cohort"] == "netnew"]
    nc = [r for r in relaxed if r["cohort"] == "nochange"]
    kept = [r for r in relaxed if r["x"] >= THRESHOLD]
    drop = [r for r in relaxed if r["x"] < THRESHOLD]
    labels, _ = day_labels(df[df["date"].isin(eval_dates)]) if eval_dates else ({}, 0)

    def edge(grp_kept):
        return dexp(grp_kept)["exp"]
    m = dict(
        label=label, n_re=len(re), n_nn=len(nn), n_nc=len(nc), n_rel=len(relaxed), n_base=len(base),
        auc_pool=auc_in(relaxed, lambda r: True),
        auc_open=auc_in(relaxed, lambda r: r["seg"] == "open"),
        auc_mid=auc_in(relaxed, lambda r: r["seg"] == "midday"),
        auc_close=auc_in(relaxed, lambda r: r["seg"] == "close"),
        auc_lo=auc_in(relaxed, lambda r: r["atr_reg"] == "lo"),
        auc_mid_v=auc_in(relaxed, lambda r: r["atr_reg"] == "mid"),
        auc_hi=auc_in(relaxed, lambda r: r["atr_reg"] == "hi"),
        base_exp=dexp(base)["exp"], rel_exp=dexp(relaxed)["exp"], gated_exp=dexp(kept)["exp"],
        gated_n=len(kept), kept_re=sum(1 for r in kept if r["cohort"] == "retimed"),
        kept_nn=sum(1 for r in kept if r["cohort"] == "netnew"),
        drop_meanR=float(np.mean([r["R"] for r in drop if r["R"] is not None])) if drop else 0.0,
        drop_loser=(sum(1 for r in drop if r["res"] == -1) /
                    max(sum(1 for r in drop if r["res"] != 0), 1) * 100),
        kept_re_mfe=float(np.median([r["mfe"] for r in kept if r["cohort"] == "retimed"])) if any(r["cohort"]=="retimed" for r in kept) else float("nan"),
        kept_re_mae=float(np.median([r["mae"] for r in kept if r["cohort"] == "retimed"])) if any(r["cohort"]=="retimed" for r in kept) else float("nan"),
    )
    m["edge"] = m["gated_exp"] - m["base_exp"]
    # regime split
    for reg in ("trend", "rot"):
        kr = [r for r in kept if labels.get(r["date"]) == reg]
        br = [r for r in base if labels.get(r["date"]) == reg]
        m[f"gated_{reg}"] = dexp(kr)["exp"]; m[f"base_{reg}"] = dexp(br)["exp"]
        m[f"edge_{reg}"] = dexp(kr)["exp"] - dexp(br)["exp"]
    # TOD / ATR composition (% of relaxed entries)
    tot = max(len(relaxed), 1)
    for s in ("open", "midday", "close"):
        m[f"tod_{s}"] = sum(1 for r in relaxed if r["seg"] == s) / tot * 100
    for g in ("lo", "mid", "hi"):
        m[f"atr_{g}"] = sum(1 for r in relaxed if r["atr_reg"] == g) / tot * 100
    return m


def fmt_auc(a):
    return f"{a:.2f}" if np.isfinite(a) else " n/a"


def report(is_m, fw, out):
    p = lambda *a: print(*a, file=out)
    p("=" * 92)
    p("FROZEN FORWARD TEST — box_rng_atr >= 3.01 gate on relaxed continuation rule (ES 15m, longs)")
    p("FORWARD_TEST_PROTOCOL.md — no tuning, no re-thresholding")
    p("=" * 92)
    p(f"\n{'metric':34} {'IN-SAMPLE (ref)':>18} {'FORWARD':>16}")
    p("-" * 70)
    rows = [
        ("re-timed / net-new / no-change", f"{is_m['n_re']}/{is_m['n_nn']}/{is_m['n_nc']}", f"{fw['n_re']}/{fw['n_nn']}/{fw['n_nc']}"),
        ("relaxed entries / base", f"{is_m['n_rel']}/{is_m['n_base']}", f"{fw['n_rel']}/{fw['n_base']}"),
        ("AUC pooled", fmt_auc(is_m['auc_pool']), fmt_auc(fw['auc_pool'])),
        ("AUC open / midday", f"{fmt_auc(is_m['auc_open'])}/{fmt_auc(is_m['auc_mid'])}", f"{fmt_auc(fw['auc_open'])}/{fmt_auc(fw['auc_mid'])}"),
        ("AUC atr lo/mid/hi", f"{fmt_auc(is_m['auc_lo'])}/{fmt_auc(is_m['auc_mid_v'])}/{fmt_auc(is_m['auc_hi'])}", f"{fmt_auc(fw['auc_lo'])}/{fmt_auc(fw['auc_mid_v'])}/{fmt_auc(fw['auc_hi'])}"),
        ("base exp", f"{is_m['base_exp']:+.2f}", f"{fw['base_exp']:+.2f}"),
        ("relaxed ungated exp", f"{is_m['rel_exp']:+.2f}", f"{fw['rel_exp']:+.2f}"),
        ("gated exp", f"{is_m['gated_exp']:+.2f}", f"{fw['gated_exp']:+.2f}"),
        ("edge (gated - base)", f"{is_m['edge']:+.2f}", f"{fw['edge']:+.2f}"),
        ("edge trend / rot", f"{is_m['edge_trend']:+.2f}/{is_m['edge_rot']:+.2f}", f"{fw['edge_trend']:+.2f}/{fw['edge_rot']:+.2f}"),
        ("kept re-timed MFE/MAE", f"{is_m['kept_re_mfe']:.2f}/{is_m['kept_re_mae']:.2f}", f"{fw['kept_re_mfe']:.2f}/{fw['kept_re_mae']:.2f}"),
        ("dropped meanR / loser%", f"{is_m['drop_meanR']:+.2f}/{is_m['drop_loser']:.0f}%", f"{fw['drop_meanR']:+.2f}/{fw['drop_loser']:.0f}%"),
        ("TOD open/mid/close %", f"{is_m['tod_open']:.0f}/{is_m['tod_midday']:.0f}/{is_m['tod_close']:.0f}", f"{fw['tod_open']:.0f}/{fw['tod_midday']:.0f}/{fw['tod_close']:.0f}"),
    ]
    for name, a, b in rows:
        p(f"{name:34} {a:>18} {b:>16}")

    p("\n" + "=" * 92)
    p("PRE-REGISTERED VERDICT")
    p("=" * 92)
    underpowered = fw["n_re"] < MIN_EVENTS and fw["n_nn"] < MIN_EVENTS
    if underpowered:
        p(f"  INCONCLUSIVE — underpowered: {fw['n_re']} re-timed & {fw['n_nn']} net-new "
          f"(< {MIN_EVENTS} each). Extend the window; do NOT soften criteria.")
        return
    f1 = (np.isfinite(fw["auc_open"]) and fw["auc_open"] >= F1_AUC and
          np.isfinite(fw["auc_mid"]) and fw["auc_mid"] >= F1_AUC)
    f2 = np.isfinite(fw["auc_pool"]) and fw["auc_pool"] >= F2_AUC
    f3 = (fw["gated_exp"] > fw["base_exp"]) and (fw["gated_exp"] >= fw["rel_exp"]) and (fw["edge"] >= F3_EDGE)
    f4 = (fw["edge_trend"] > 0) and (fw["edge_rot"] > 0)
    f5 = (np.isfinite(fw["kept_re_mfe"]) and fw["kept_re_mfe"] >= F5_MFE and fw["kept_re_mae"] <= F5_MAE)
    mark = lambda b: "PASS" if b else "FAIL"
    p(f"  F1 within-seg AUC >= {F1_AUC} (open & midday) : {mark(f1)}  ({fmt_auc(fw['auc_open'])}/{fmt_auc(fw['auc_mid'])})")
    p(f"  F2 pooled AUC >= {F2_AUC}                      : {mark(f2)}  ({fmt_auc(fw['auc_pool'])})")
    p(f"  F3 gated>base & >=ungated & edge>=+{F3_EDGE}R    : {mark(f3)}  (edge {fw['edge']:+.2f}R)")
    p(f"  F4 edge>0 in BOTH regimes                   : {mark(f4)}  (trend {fw['edge_trend']:+.2f} / rot {fw['edge_rot']:+.2f})")
    p(f"  F5 kept re-timed MFE>={F5_MFE} & MAE<={F5_MAE}       : {mark(f5)}  ({fw['kept_re_mfe']:.2f}/{fw['kept_re_mae']:.2f})")
    p(f"\n  OVERALL (F1 AND F3 AND F4): "
      f"{'CONFIRMED' if (f1 and f3 and f4) else 'NOT CONFIRMED'}")
    p("  (F2/F5 are diagnostic context; magnitudes are directional given thin samples.)")


def main():
    selftest = "--selftest" in sys.argv
    train = pd.read_parquet(TRAIN)
    train_dates = set(add_adx_slope(enrich(train, BASE), Q["adx_slope_k"])["date"])

    if selftest:
        # split training: last ~110 cal days = pseudo-forward, rest = train/warm-up
        dts = sorted(train_dates)
        cut = dts[int(len(dts) * 0.6)]
        df = add_adx_slope(enrich(train, BASE), Q["adx_slope_k"])
        is_dates = {d for d in train_dates if d < cut}
        fw_dates = {d for d in train_dates if d >= cut}
        is_m = metrics(df, is_dates, "IS")
        fw = metrics(df, fw_dates, "FWD(selftest)")
    else:
        if not FWD.exists():
            sys.exit(f"Forward slice not found: {FWD}\nRun fetch_forward_slice.py on the IB-connected machine first.")
        df_is = add_adx_slope(enrich(train, BASE), Q["adx_slope_k"])
        is_m = metrics(df_is, train_dates, "IS")
        df_fw, fstart = enrich_with_warmup(FWD, TRAIN, warmup_bars=60)
        fw_dates = {d for d in df_fw["date"] if d >= fstart.date()}
        fw = metrics(df_fw, fw_dates, "FWD")

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT, "w") as fh:
        class Tee:
            def write(self, s): sys.__stdout__.write(s); fh.write(s)
            def flush(self): sys.__stdout__.flush(); fh.flush()
        with contextlib.redirect_stdout(Tee()):
            report(is_m, fw, sys.stdout)
    print(f"\n[written to {RESULT}]", file=sys.__stdout__)


if __name__ == "__main__":
    main()
