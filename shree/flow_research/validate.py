"""Validation battery — does flow predict anything price does not?

Same standard as prior audits: quintile forward-return lift, option-P&L proxy,
correlation with existing (price-derived) features, a direction-confound guard,
and a SHUFFLE control that must flatten any claimed edge.

Outcome source for the POC: the existing spy_signals rows already carry
`pnl_pct` (option-P&L proxy) and spy_price / spy_price_exit (underlying move),
so retro-attached SIGNAL snapshots validate immediately with zero new data.
Fixed-horizon forward returns (1/5/15/30 min) require a price tape and plug in
here later via an added outcome column — the framework is identical.

Nothing in this module trades, sizes, or gates. It reports numbers.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Flow measures under test (Tier A first — the orthogonal ones).
FLOW_MEASURES = [
    "pc_prem_imbalance", "dw_flow", "net_call_prem", "net_put_prem",
    "sweep_intensity", "block_prem", "oc_open_ratio",
    "expiry_concentration", "strike_repetition", "atm_vs_wing",
    "iv_weighted_side",
]

# Existing price-derived features to test correlation against. If flow is just a
# restatement of these, correlation is high and there is no new information.
EXISTING_FEATURES = [
    "confidence", "vix", "iv_rank", "volume_spike_mult", "spread_pct",
    "intraday_pc_ratio", "external_composite", "sentiment_score", "flow_score",
]


def load_joined(flow_db: str, spy_db: str) -> pd.DataFrame:
    """Join SIGNAL snapshots to their spy_signals row. Read-only on production."""
    fconn = sqlite3.connect(flow_db)
    shadow = pd.read_sql_query(
        "SELECT * FROM shadow_flow WHERE snapshot_kind='SIGNAL' "
        "AND signal_id IS NOT NULL",
        fconn,
    )
    fconn.close()
    if shadow.empty:
        return shadow

    sconn = sqlite3.connect(f"file:{os.path.abspath(spy_db)}?mode=ro&immutable=1",
                            uri=True)
    sig = pd.read_sql_query(
        "SELECT id AS signal_id, sent_at, signal_type, spy_price, "
        "spy_price_exit, pnl_pct, outcome, "
        + ", ".join(f for f in EXISTING_FEATURES) +
        " FROM spy_signals",
        sconn,
    )
    sconn.close()

    df = shadow.merge(sig, on="signal_id", how="inner")
    # derived outcomes
    df["opt_pnl"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
    df["win"] = (df["opt_pnl"] > 0).astype(float)
    with np.errstate(all="ignore"):
        df["underlying_move"] = (
            (df["spy_price_exit"] - df["spy_price"]) / df["spy_price"]
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def quintile_lift(df: pd.DataFrame, measure: str,
                  outcome: str = "opt_pnl", q: int = 5) -> Dict:
    """Mean outcome by measure quintile; report top-minus-bottom spread and
    monotonicity. A flat spread => no edge in this measure."""
    d = df[[measure, outcome]].dropna()
    if len(d) < q * 5:
        return {"measure": measure, "n": len(d), "status": "INSUFFICIENT"}
    try:
        d = d.assign(bucket=pd.qcut(d[measure], q, labels=False, duplicates="drop"))
    except ValueError:
        return {"measure": measure, "n": len(d), "status": "DEGENERATE"}
    means = d.groupby("bucket")[outcome].mean()
    spread = float(means.iloc[-1] - means.iloc[0])
    monotone = bool(means.is_monotonic_increasing or means.is_monotonic_decreasing)
    return {
        "measure": measure,
        "n": int(len(d)),
        "bucket_means": [round(float(x), 5) for x in means.tolist()],
        "top_minus_bottom": round(spread, 5),
        "monotone": monotone,
        "status": "OK",
    }


def feature_correlation(df: pd.DataFrame,
                        measures: Optional[List[str]] = None) -> pd.DataFrame:
    """Correlation of each flow measure with each existing feature.

    LOW correlation is REQUIRED — high correlation means the flow measure is a
    price restatement, not new information.
    """
    measures = measures or FLOW_MEASURES
    cols_m = [m for m in measures if m in df.columns]
    cols_e = [e for e in EXISTING_FEATURES if e in df.columns]
    sub = df[cols_m + cols_e].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="spearman")
    return corr.loc[cols_m, cols_e]


def shuffle_control(df: pd.DataFrame, measure: str,
                    outcome: str = "opt_pnl", n_shuffles: int = 1000,
                    seed: int = 12345) -> Dict:
    """Permutation test. Shuffle the measure vs outcomes; the observed
    top-minus-bottom spread must exceed the shuffled distribution. If a broken
    classifier is manufacturing structure, the observed effect sits inside the
    shuffle noise and this p-value is large."""
    d = df[[measure, outcome]].dropna()
    if len(d) < 25:
        return {"measure": measure, "n": len(d), "status": "INSUFFICIENT"}
    rng = np.random.default_rng(seed)
    x = d[measure].to_numpy()
    y = d[outcome].to_numpy()

    def _spread(xx, yy):
        order = np.argsort(xx)
        k = max(1, len(xx) // 5)
        return yy[order][-k:].mean() - yy[order][:k].mean()

    obs = _spread(x, y)
    null = np.empty(n_shuffles)
    for i in range(n_shuffles):
        null[i] = _spread(x, rng.permutation(y))
    # two-sided p
    p = float((np.abs(null) >= abs(obs)).mean())
    return {
        "measure": measure,
        "n": int(len(d)),
        "observed_spread": round(float(obs), 5),
        "null_std": round(float(null.std()), 5),
        "p_value": round(p, 4),
        "passes": bool(p < 0.05),
        "status": "OK",
    }


def direction_confound(df: pd.DataFrame, measure: str,
                       outcome: str = "opt_pnl",
                       control: str = "underlying_move") -> Dict:
    """Partial correlation of measure vs outcome, controlling for the
    contemporaneous underlying move (the CALL_SWEEP beta-not-alpha trap).

    If the raw correlation vanishes after residualizing on the move, the
    'edge' was just directional beta.
    """
    d = df[[measure, outcome, control]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 30:
        return {"measure": measure, "n": len(d), "status": "INSUFFICIENT"}
    raw = float(d[measure].corr(d[outcome], method="spearman"))

    def _resid(target, ctrl):
        c = np.polyfit(ctrl, target, 1)
        return target - (c[0] * ctrl + c[1])

    rx = _resid(d[measure].to_numpy(), d[control].to_numpy())
    ry = _resid(d[outcome].to_numpy(), d[control].to_numpy())
    from scipy.stats import spearmanr
    partial = float(spearmanr(rx, ry).correlation)
    survives = bool(abs(partial) >= 0.5 * abs(raw) and abs(partial) > 0.05)
    return {
        "measure": measure,
        "n": int(len(d)),
        "raw_corr": round(raw, 4),
        "partial_corr_ctrl_move": round(partial, 4),
        "survives_confound": survives,
        "status": "OK",
    }


def run_battery(flow_db: str, spy_db: str) -> Dict:
    """Full battery over all Tier-A/B measures. Returns a report dict."""
    df = load_joined(flow_db, spy_db)
    report: Dict = {"n_joined": int(len(df)), "measures": {}}
    if df.empty:
        report["status"] = "NO_DATA"
        return report
    corr = feature_correlation(df)
    report["max_abs_corr_with_existing"] = {
        m: round(float(corr.loc[m].abs().max()), 3)
        for m in corr.index
    }
    for m in FLOW_MEASURES:
        if m not in df.columns:
            continue
        report["measures"][m] = {
            "lift": quintile_lift(df, m),
            "shuffle": shuffle_control(df, m),
            "confound": direction_confound(df, m),
        }
    report["status"] = "OK"
    return report
