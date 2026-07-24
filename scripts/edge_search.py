#!/usr/bin/env python3
"""Exploratory edge search — quant data-mining, NOT strategy design.

Question asked: "what structural/calendar/context subset consistently shows
positive expectancy?" — NOT "what indicator". Price-derived indicators, flow,
GEX, max-pain, mean-reversion, ORB, TC variants all already failed and are NOT
re-tested. This mines the META dimensions (day-of-week, time-of-day, month, DTE,
signal_type, VIX regime, tier, regime) and their 2-/3-way interactions, then
tries to DESTROY every positive pocket (multiple comparisons, bootstrap, OOS).

Read-only. No production/strategy changes. Observation only.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from itertools import combinations
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder

ET = ZoneInfo("America/New_York")
SPY = "file:/Users/svss/Documents/code/ShreeBot/data/spy_options_signals.db?mode=ro&immutable=1"
MIN_N = 20          # below this a subgroup is meaningless
RNG = np.random.default_rng(20260724)


def load() -> pd.DataFrame:
    c = sqlite3.connect(SPY, uri=True)
    df = pd.read_sql_query(
        "SELECT id, sent_at, signal_type, dte, vix, regime, confidence_tier, "
        "confidence_time_bucket, iv_rank, sentiment_label, pnl_pct "
        "FROM spy_signals WHERE pnl_pct IS NOT NULL", c)
    c.close()

    def et(s):
        s = s.split(".")[0].rstrip("Z")
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).astimezone(ET)
    d = df["sent_at"].apply(et)
    df["dow"] = d.apply(lambda x: x.strftime("%a"))
    df["month"] = d.apply(lambda x: x.strftime("%Y-%m"))
    df["hour"] = d.apply(lambda x: x.hour)
    df["tod"] = pd.cut(d.apply(lambda x: x.hour + x.minute / 60),
                       [9.5, 10.5, 12, 14, 15, 16],
                       labels=["OPEN", "MORN", "MID", "POWER", "CLOSE"])
    df["dte_b"] = pd.cut(df["dte"].fillna(-1),
                         [-2, 0, 2, 7, 999], labels=["0", "1-2", "3-7", "8+"])
    df["vix_b"] = pd.cut(df["vix"].fillna(-1),
                         [-2, 15, 20, 25, 999], labels=["lo", "mid", "hi", "xhi"])
    df["win"] = (df["pnl_pct"] > 0).astype(int)
    df["date"] = d.apply(lambda x: x.date().isoformat())
    return df


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    means = [RNG.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def perm_p(x, n=2000):
    """P(mean >= observed) under sign-flip null centered at 0 — one-sample."""
    x = np.asarray(x, float)
    obs = x.mean()
    c = 0
    for _ in range(n):
        s = RNG.choice([-1, 1], len(x))
        if (x * s).mean() >= abs(obs):
            c += 1
    return c / n


FEATURES = ["dow", "tod", "dte_b", "vix_b", "signal_type", "confidence_tier",
            "regime", "sentiment_label"]


def subgroups(df, dims):
    """Yield (label, mask) for every value-combination of the given dims."""
    grp = df.groupby(list(dims), observed=True)
    for key, idx in grp.groups.items():
        if not isinstance(key, tuple):
            key = (key,)
        label = " & ".join(f"{d}={v}" for d, v in zip(dims, key))
        yield label, df.loc[idx]


def mine(df):
    rows = []
    dimsets = ([(f,) for f in FEATURES] +
               list(combinations(FEATURES, 2)) +
               list(combinations(FEATURES, 3)))
    for dims in dimsets:
        for label, sub in subgroups(df, dims):
            if len(sub) < MIN_N:
                continue
            x = sub["pnl_pct"].values
            rows.append({
                "level": len(dims), "subgroup": label, "n": len(sub),
                "n_sess": sub["date"].nunique(),
                "pct_jul": (sub["month"] == "2026-07").mean(),
                "mean_pnl": x.mean(), "win": sub["win"].mean(),
                "p": perm_p(x, 1500),
            })
    return pd.DataFrame(rows)


def session_test(df, dims, vals):
    """Honest independent-unit test: collapse the pocket to per-session means,
    then bootstrap across SESSIONS (the real independent unit). Intra-session
    signals are correlated, so per-signal tests massively overstate n."""
    m = np.ones(len(df), bool)
    for d, v in zip(dims, vals):
        m &= df[d].astype(str).values == v
    sub = df[m]
    per = sub.groupby("date")["pnl_pct"].mean().values
    if len(per) < 3:
        return len(per), float("nan"), float("nan"), float("nan")
    lo, hi = boot_ci(per, 1500)
    return len(per), per.mean(), lo, hi


def bh_fdr(pvals, q=0.10):
    p = np.sort(pvals); n = len(p)
    passed = [i for i in range(n) if p[i] <= (i + 1) / n * q]
    return (p[max(passed)] if passed else -1.0)


def main():
    df = load()
    print(f"dataset: {len(df)} signals with pnl_pct, "
          f"{df['date'].nunique()} sessions, mean_pnl={df['pnl_pct'].mean():.4f}, "
          f"win={df['win'].mean():.3f}")

    res = mine(df)
    print(f"\nsubgroups tested (n>={MIN_N}): {len(res)}")
    pos = res[res["mean_pnl"] > 0].sort_values("p")
    print(f"positive-expectancy subgroups: {len(pos)}")

    print("\n=== top 12 positive pockets by permutation p (with SESSION count) ===")
    print(f"{'lvl':<4}{'n':>5}{'sess':>5}{'%Jul':>6}{'mean':>8}{'win':>6}{'perm_p':>8}  subgroup")
    for _, r in pos.head(12).iterrows():
        print(f"{r['level']:<4}{r['n']:>5}{r['n_sess']:>5}{r['pct_jul']:>6.2f}"
              f"{r['mean_pnl']:>8.3f}{r['win']:>6.2f}{r['p']:>8.3f}  {r['subgroup']}")
    print(f"\n  median sessions spanned by positive pockets: "
          f"{pos['n_sess'].median():.0f}   median %-in-July: {pos['pct_jul'].median():.2f}")

    # DESTRUCTION 0 — session-clustered test on the top 8 pockets
    print("\n=== DESTRUCTION: honest independent-unit (per-SESSION) test ===")
    print("  a pocket's signals are correlated within a session; the real n is "
          "sessions.\n  bootstrap the per-session means across sessions:")
    for _, r in pos.head(8).iterrows():
        dims = [x.split("=")[0] for x in r["subgroup"].split(" & ")]
        vals = [x.split("=")[1] for x in r["subgroup"].split(" & ")]
        ns, mu, lo, hi = session_test(df, dims, vals)
        verdict = ("<3 sessions — meaningless" if ns < 3
                   else "CI excludes 0" if lo > 0 else "CI INCLUDES 0 -> noise")
        print(f"  {r['subgroup'][:55]:<55} sess={ns} "
              f"mean={mu if mu==mu else float('nan'):.3f} "
              f"CI=({lo:.3f},{hi:.3f}) {verdict}")

    # DESTRUCTION 1 — multiple comparisons across ALL tested subgroups
    thr = bh_fdr(res["p"].values, 0.10)
    n_raw = (res["p"] < 0.05).sum()
    exp_fp = 0.05 * len(res)
    print(f"\n=== DESTRUCTION: multiple comparisons ===")
    print(f"subgroups with raw p<0.05: {n_raw}  (expected by chance: {exp_fp:.0f})")
    print(f"BH-FDR q=0.10 threshold: {thr:.4f}  -> "
          f"{'survivors: '+str((res['p']<=thr).sum()) if thr>0 else 'NONE survive'}")

    # DESTRUCTION 2 — bootstrap CI + OOS on the single best positive pocket
    if len(pos):
        best = pos.iloc[0]
        dims = [x.split("=")[0] for x in best["subgroup"].split(" & ")]
        vals = [x.split("=")[1] for x in best["subgroup"].split(" & ")]
        m = np.ones(len(df), bool)
        for d, v in zip(dims, vals):
            m &= df[d].astype(str).values == v
        sub = df[m]
        lo, hi = boot_ci(sub["pnl_pct"].values)
        print(f"\n=== DESTRUCTION: best pocket '{best['subgroup']}' ===")
        print(f"n={len(sub)} mean={sub['pnl_pct'].mean():.3f} "
              f"boot95%CI=({lo:.3f},{hi:.3f})  {'excludes 0' if lo>0 else 'INCLUDES 0 -> noise'}")
        tr = sub[sub["month"] <= "2026-05"]; te = sub[sub["month"] >= "2026-06"]
        print(f"OOS: train(Apr-May) n={len(tr)} mean="
              f"{tr['pnl_pct'].mean() if len(tr) else float('nan'):.3f} | "
              f"test(Jun-Jul) n={len(te)} mean="
              f"{te['pnl_pct'].mean() if len(te) else float('nan'):.3f}")

    # DESTRUCTION 3 — RF permutation importance vs shuffle (does ANY meta-feature
    # carry win/loss information at all?)
    X = df[FEATURES].astype(str).fillna("NA")
    enc = OneHotEncoder(handle_unknown="ignore").fit(X)
    Xe = enc.transform(X)
    y = df["win"].values
    rf = RandomForestClassifier(n_estimators=300, max_depth=4,
                                random_state=1, n_jobs=-1).fit(Xe, y)
    pi = permutation_importance(rf, Xe.toarray(), y, n_repeats=20,
                                random_state=1, scoring="roc_auc")
    print(f"\n=== DESTRUCTION: RF permutation importance (AUC) ===")
    print(f"in-sample AUC drop from permuting features (mean±std): "
          f"{pi.importances_mean.mean():.4f} ± {pi.importances_std.mean():.4f}")
    print("  (≈0 => meta-features carry no win/loss information)")


if __name__ == "__main__":
    main()
