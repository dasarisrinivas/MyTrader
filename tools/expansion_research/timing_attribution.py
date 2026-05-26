"""
Re-timing vs net-new attribution for relaxed-stack continuation arming.

The continuation-timing prototype showed the relaxed stack ("early-only") adds
entries with BETTER expectancy than baseline (+0.18R vs +0.06R). But that
"early-only" cohort conflates two structurally different claims:

  RE-TIMED  -> "same edge, better entry point"   (high credibility, regime-stable)
  NET-NEW   -> "new edge from relaxing constraints" (high overfit risk)

Out-of-sample these behave nothing alike, so before any walk-forward we need to
know which one the +0.18R actually is. This script splits the marginal entries
into the strict classes Srini specified and decomposes the relaxed-vs-base
portfolio R improvement into a TIMING part and a NET-NEW part.

Per contiguous bullish episode (RTH, same day; ema9>ema21 & close>ema50):
  bf = first bar BASE EMA9_PB would arm     (full stack + adx>=22)
  vf = first bar the RELAXED rule would arm  (loose stack [+ ADX-accel OR for V3])

  RE-TIMED      bf and vf exist, vf <  bf     (base would have traded anyway)
  NO-CHANGE     bf and vf exist, vf == bf     (relaxed added nothing here)
  NET-NEW       vf exists, bf is None         (base never arms in the episode)
  BUG/MISMATCH  bf exists, vf is None         (impossible by construction -> index bug)

Because an episode is *defined* by loose-stack being true on every bar, every
base arm bar is also a relaxed arm bar, so vf <= bf always and the BUG class
must be empty. A non-empty BUG class means an episode/index mismatch.

Longs only. Research only; no live strategy code is touched.
"""
from __future__ import annotations
import sys, os, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expansion_proto import enrich, DATA, P as BASE
from continuation_timing import (
    Q, add_adx_slope, arm_masks, episodes, first_true, sim, expectancy,
)

BASE_NAME = "BASE  (stack+adx>=22)"
# headline first; V2 isolates the pure stack relaxation from the ADX-slope clause
RELAXED_RULES = [
    ("V3 slope OR + loose-stack", "Phase-1 rule: loose stack + ADX-accel OR-clause"),
    ("V2 loose-stack",            "loose stack only (isolates the EMA-stack relaxation)"),
]
K_TARGETS = (1.25, 1.5, 2.0)
HEADLINE_K = 1.5


def realized_R(df, i, k, Q):
    """Realized R for one trade at target k (1R = 1xATR stop), stop-wins-ties
    (mirrors sim()). Timeout within hold -> mark-to-market at exit close, so
    every trade contributes to a portfolio sum. Returns None if ATR invalid."""
    H, L, C, A = (df["high"].values, df["low"].values,
                  df["close"].values, df["atr"].values)
    n = len(df); entry = C[i]; atr = A[i]
    if not np.isfinite(atr) or atr <= 0:
        return None
    sl = entry - Q["stop_atr"] * atr
    tp = entry + k * atr
    last = i
    for j in range(i + 1, min(i + 1 + Q["hold"], n)):
        last = j
        if L[j] <= sl:                 # stop checked first (conservative, matches sim)
            return -1.0
        if H[j] >= tp:
            return float(k)
    return (C[last] - entry) / atr     # timeout -> R at hold-exit close


def classify(df, eps, base_mask, relaxed_mask, Q, k_targets):
    """Split episodes into the four classes; carry both sim results where paired."""
    coh = dict(retimed=[], nochange=[], netnew=[], bug=[])
    for (a, b) in eps:
        bf = first_true(base_mask, a, b)
        vf = first_true(relaxed_mask, a, b)
        if bf is None and vf is None:
            continue
        if bf is not None and vf is None:          # construction-impossible
            coh["bug"].append(dict(a=a, b=b, bf=bf, vf=vf))
            continue
        if bf is None:                             # net-new admission
            rv = sim(df, vf, Q, k_targets)
            if rv:
                coh["netnew"].append(dict(a=a, b=b, vf=vf, bf=None, v=rv, base=None))
            continue
        rv = sim(df, vf, Q, k_targets)             # both arm
        rb = sim(df, bf, Q, k_targets)
        if not rv or not rb:
            continue
        rec = dict(a=a, b=b, vf=vf, bf=bf, v=rv, base=rb, lead=bf - vf)
        coh["retimed" if vf < bf else "nochange"].append(rec)
    return coh


def cohort_stats(sims, k_targets):
    """WR / expectancy / open-count per target + median MFE/MAE, over a list of
    sim() result dicts."""
    rows = {}
    for k in k_targets:
        wr, exp, w, l = expectancy(sims, k)
        opn = sum(1 for r in sims if r["res"][k] == 0)
        rows[k] = dict(wr=wr, exp=exp, w=w, l=l, open=opn)
    mfe = float(np.median([r["mfe_atr"] for r in sims])) if sims else 0.0
    mae = float(np.median([r["mae_atr"] for r in sims])) if sims else 0.0
    return rows, mfe, mae, len(sims)


def print_cohort_table(coh, k_targets, n_days):
    relaxed_all = coh["retimed"] + coh["nochange"] + coh["netnew"]
    base_paired = [r["base"] for r in coh["retimed"]] + [r["base"] for r in coh["nochange"]]
    groups = [
        ("BASE (paired episodes)",        base_paired),
        ("RELAXED total",                 [r["v"] for r in relaxed_all]),
        ("  RE-TIMED (relaxed entry)",    [r["v"] for r in coh["retimed"]]),
        ("  RE-TIMED (base entry, same ep)", [r["base"] for r in coh["retimed"]]),
        ("  NO-CHANGE",                   [r["v"] for r in coh["nochange"]]),
        ("  NET-NEW",                     [r["v"] for r in coh["netnew"]]),
    ]
    hdr = (f"{'cohort':32} {'n':>5} {'/day':>5} {'MFE':>5} {'MAE':>5} "
           f"{'WR1.25':>7} {'e1.25':>7} {'WR1.5':>6} {'e1.5':>7} {'WR2':>5} {'e2':>7} {'open@1.5':>9}")
    print(hdr); print("-" * len(hdr))
    for label, sims in groups:
        if not sims:
            print(f"{label:32} {0:>5}"); continue
        rows, mfe, mae, n = cohort_stats(sims, k_targets)
        print(f"{label:32} {n:>5} {n/n_days:>5.2f} {mfe:>5.2f} {mae:>5.2f} "
              f"{rows[1.25]['wr']:>6.0f}% {rows[1.25]['exp']:>+7.2f} "
              f"{rows[1.5]['wr']:>5.0f}% {rows[1.5]['exp']:>+7.2f} "
              f"{rows[2.0]['wr']:>4.0f}% {rows[2.0]['exp']:>+7.2f} {rows[1.5]['open']:>9}")


def decompose(coh, df, k, Q):
    """Portfolio R decomposition at target k:
       total relaxed-minus-base R = sum over re-timed (R_vf - R_bf)   [TIMING]
                                  + sum over net-new  (R_vf - 0)      [NET-NEW]
                                  + 0 over no-change."""
    timing = []
    for r in coh["retimed"]:
        rv = realized_R(df, r["vf"], k, Q); rb = realized_R(df, r["bf"], k, Q)
        if rv is None or rb is None:
            continue
        timing.append(rv - rb)
    netnew = [realized_R(df, r["vf"], k, Q) for r in coh["netnew"]]
    netnew = [x for x in netnew if x is not None]
    # base portfolio mean-R (paired episodes) for reference
    base_R = []
    for r in coh["retimed"] + coh["nochange"]:
        rb = realized_R(df, r["bf"], k, Q)
        if rb is not None:
            base_R.append(rb)
    return timing, netnew, base_R


def run_rule(df, eps, masks, relaxed_name, desc, n_days):
    print("\n" + "=" * 104)
    print(f"RELAXED RULE: {relaxed_name}   [{desc}]")
    print("=" * 104)
    coh = classify(df, eps, masks[BASE_NAME], masks[relaxed_name], Q, K_TARGETS)

    n_re, n_nc, n_nn, n_bug = (len(coh["retimed"]), len(coh["nochange"]),
                               len(coh["netnew"]), len(coh["bug"]))
    print(f"episodes classified: RE-TIMED={n_re}  NO-CHANGE={n_nc}  "
          f"NET-NEW={n_nn}  BUG/MISMATCH={n_bug}")
    if n_bug:
        print(f"  !! BUG class non-empty ({n_bug}) — base arms but relaxed doesn't, "
              f"which is impossible if episodes==loose-stack. Investigate index alignment.")
    else:
        print("  sanity OK: BUG class empty (every base arm is also a relaxed arm).")
    if n_re:
        leads = [r["lead"] for r in coh["retimed"]]
        print(f"  re-timing lead bars: median {np.median(leads):.1f}  "
              f"mean {np.mean(leads):.1f}  max {max(leads)}  (1 bar = 15 min)")
    print()
    print_cohort_table(coh, K_TARGETS, n_days)

    # ---- portfolio R decomposition --------------------------------------
    timing, netnew, base_R = decompose(coh, df, HEADLINE_K, Q)
    sum_t, sum_n = float(np.sum(timing)), float(np.sum(netnew))
    total = sum_t + sum_n
    print(f"\n-- PORTFOLIO R DECOMPOSITION @ {HEADLINE_K}R target "
          f"(timeouts marked-to-market at hold exit) --")
    print(f"   base portfolio mean R/trade (paired eps) : {np.mean(base_R):+.3f}R "
          f"over {len(base_R)} trades")
    if timing:
        print(f"   TIMING  (re-timed: R_relaxed - R_base)   : sum {sum_t:+.2f}R  | "
              f"mean delta {np.mean(timing):+.3f}R/episode over {len(timing)} eps")
    if netnew:
        print(f"   NET-NEW (new participation, base=0)      : sum {sum_n:+.2f}R  | "
              f"mean {np.mean(netnew):+.3f}R/trade over {len(netnew)} trades")
    if abs(total) > 1e-9:
        print(f"   total relaxed-minus-base R               : {total:+.2f}R   "
              f"=>  TIMING {sum_t/total*100:4.0f}%   NET-NEW {sum_n/total*100:4.0f}%")
    # verdict heuristic (descriptive, not advice)
    print("\n   READ:")
    if timing:
        md = np.mean(timing)
        print(f"     - per-episode timing delta is {'POSITIVE' if md > 0 else 'NON-POSITIVE'} "
              f"({md:+.3f}R): earlier entry on the SAME trades "
              f"{'genuinely improves them' if md > 0 else 'does NOT improve them'}.")
    if netnew and base_R:
        nn_mean = np.mean(netnew); bs_mean = np.mean(base_R)
        print(f"     - net-new trades average {nn_mean:+.3f}R vs base {bs_mean:+.3f}R: "
              f"the relaxed-only signals are {'≥' if nn_mean >= bs_mean else '<'} base quality.")
    if abs(total) > 1e-9:
        share = sum_t / total
        if share >= 0.6:
            print("     - improvement is TIMING-DOMINATED -> 'same edge, better entry' "
                  "-> regime-stable, walk-forward likely confirmatory.")
        elif share <= 0.4:
            print("     - improvement is NET-NEW-DOMINATED -> 'new edge from relaxed "
                  "constraints' -> regime-dependent, walk-forward likely to compress.")
        else:
            print("     - improvement is MIXED -> walk-forward must segment by regime "
                  "and weight the net-new portion skeptically.")
    return coh


def main():
    df = add_adx_slope(enrich(pd.read_parquet(DATA), BASE), Q["adx_slope_k"])
    masks = {name: m.values for name, m in arm_masks(df, Q).items()}
    eps = episodes(df, Q)
    n_days = df.loc[df["ct"].between(BASE["rth_start"], BASE["rth_end"], inclusive="left"),
                    "date"].nunique()
    n_days = max(n_days, 1)

    print("=" * 104)
    print(f"RE-TIMING vs NET-NEW ATTRIBUTION — ES 15m, "
          f"{df.index.min().date()}->{df.index.max().date()} "
          f"({n_days} RTH days, {len(eps)} bullish episodes)")
    print("=" * 104)
    print("Question: of the relaxed rule's expectancy gain over BASE EMA9_PB, how much is")
    print("earlier entry on trades base would take anyway (TIMING) vs trades base never")
    print("takes at all (NET-NEW)?  Timing = robust; net-new = overfit risk.")

    for name, desc in RELAXED_RULES:
        run_rule(df, eps, masks, name, desc, n_days)


if __name__ == "__main__":
    main()
