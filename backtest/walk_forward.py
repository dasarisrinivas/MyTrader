#!/usr/bin/env python3
"""
Walk-Forward Optimization (T4)
==============================
Rolling 3-month IS / 1-month OOS windows across the Feb 2025–Jan 2026
backtest period.

Scores each parameter combination separately for:
  - Low-vol regime  (ATR < ATR_THRESHOLD, default 13)
  - High-vol regime (ATR >= ATR_THRESHOLD)

Selects the regime-optimal params and evaluates on the OOS month.

Usage:
    python3 -m backtest.walk_forward \
        --data-file data/raw/MES/FUT_MESH6/15_mins/MES_15m_20250201_20260131.parquet \
        [--atr-threshold 13.0] [--is-months 3] [--oos-months 1]
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger

warnings.filterwarnings("ignore", category=FutureWarning)

# ── logging ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="WARNING", format="{time:HH:mm:ss} | {level} | {message}")
logger.add("logs/walk_forward.log", level="DEBUG", rotation="10 MB")

# ── parameter grid ──────────────────────────────────────────────────────────────
# Keep it tractable: 4×3×3×3 = 108 combos per IS window
PARAM_GRID: Dict[str, List[Any]] = {
    "ft_adx_min":           [15.0, 18.0, 22.0, 26.0],
    "ft_ema_touch_pct":     [0.0010, 0.0015, 0.0020],
    "ft_ema_touch_atr_mult":[0.5,   0.75,   1.0  ],
    "ft_pb_stop_mult":      [1.2,   1.5,    2.0  ],
}

# Fixed params (inherit from bt_t3_proximity baseline)
BASE_PARAMS: Dict[str, Any] = {
    "ft_pb_target_mult":        1.0,
    "ft_max_hold_bars":         8,
    "ft_or_minutes":            30,
    "ft_or_break_max_per_day":  2,
    "ft_or_target_r":           1.0,
    "ft_adx_max":               45.0,
    "ft_atr_very_low_threshold":8.0,
    "ft_atr_high_threshold":    13.0,
    "ft_atr_extreme_threshold": 20.0,
    "ft_fixed_tp_points":       8.0,
    "ft_fixed_tp_points_ema9":  10.0,
    "ft_fixed_tp_points_trend": 12.0,
    "ft_fixed_sl_points":       6.0,
    "ft_fixed_sl_points_ema9":  8.0,
    "ft_fixed_sl_points_trend": 8.0,
    "ft_ema9_sl_atr_mult":      1.0,
    "ft_ema9_sl_floor_pts":     8.0,
    "ft_ema9_sl_ceiling_pts":   20.0,
    "ft_ema9_rr_ratio":         1.25,
    "ft_ema9_pb_enabled":       True,
    "ft_ema9_pb_stop_mult":     1.2,
    "ft_ema9_pb_target_mult":   1.5,
    "ft_ema9_touch_pct":        0.0015,
    "ft_shorts_enabled":        True,
    "ft_short_pb_stop_mult":    1.5,
    "ft_short_pb_target_mult":  1.0,
    "ft_short_or_target_r":     1.0,
    "ft_trend_cont_enabled":    True,
    "ft_trend_cont_stop_mult":  1.0,
    "ft_trend_cont_target_mult":2.0,
    "ft_trend_cont_adx_min":    25.0,
    "ft_trend_cont_ema9_pct":   0.003,
    "ft_trend_cont_max_per_day":2,
    "ft_trend_cont_max_ext_pts":30.0,
    "ft_trend_cont_gap_adx_min":25.0,
    "ft_proximity_enabled":     True,
    "ft_proximity_gap_mult":    0.3,
    "ft_proximity_size_mult":   0.7,
    "ft_proximity_sl_mult":     0.8,
    "ft_proximity_tp_mult":     0.8,
    "ft_proximity_max_per_day": 2,
    "ft_entry_start_hour":      0,
    "ft_entry_start_minute":    0,
    "ft_entry_end_hour":        23,
    "ft_entry_end_minute":      59,
    "rth_start_hour":           0,
    "rth_start_minute":         0,
    "rth_end_hour":             23,
    "rth_end_minute":           59,
    "use_15m_strategy":         True,
    "use_scoring_system":       False,
}


# ── ATR computation ──────────────────────────────────────────────────────────────

def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute ATR_14 on raw OHLCV data and add it as a column."""
    df = df.copy()
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder smoothing (same as ta-lib ATR)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    df["ATR_14"] = atr
    return df


# ── helpers ─────────────────────────────────────────────────────────────────────

def _profit_factor(pnls: List[float]) -> float:
    """Profit factor = gross_profit / gross_loss. Returns 0 if no trades or all losers."""
    if not pnls:
        return 0.0
    wins  = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    if losses == 0:
        return wins if wins > 0 else 0.0
    return wins / losses


def _sharpe(pnls: List[float]) -> float:
    """Annualised per-trade Sharpe (0 if < 2 trades)."""
    if len(pnls) < 2:
        return 0.0
    arr = np.array(pnls, dtype=float)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return float(arr.mean() / std * np.sqrt(252 * 2))


def _score(pnls: List[float], min_trades: int = 3) -> float:
    """
    Composite score for IS selection.
    Uses profit_factor × log(n_trades) so that edge quality and
    sample size are both rewarded. Returns 0 for insufficient trades.
    """
    if len(pnls) < min_trades:
        return 0.0
    pf = _profit_factor(pnls)
    return pf * np.log1p(len(pnls))


def _regime_stats(
    trades: List[Dict],
    df_15m: pd.DataFrame,
    atr_threshold: float,
) -> Dict:
    """
    Split trades into low-vol / high-vol by ATR at entry bar.
    Returns a dict with per-regime stats.
    """
    if not trades:
        return {"low": {}, "high": {}, "all": {}}

    atr_col = "ATR_14" if "ATR_14" in df_15m.columns else None
    atr_map = df_15m[atr_col].to_dict() if atr_col else {}

    low_pnl: List[float] = []
    high_pnl: List[float] = []

    for trade in trades:
        entry_ts = trade.get("entry_time")
        pnl = trade["realized_pnl"]
        atr_val = 0.0
        if entry_ts is not None and atr_map:
            atr_val = float(atr_map.get(entry_ts, 0.0))
            if atr_val == 0.0:
                try:
                    floored = entry_ts.floor("15min")
                    atr_val = float(atr_map.get(floored, 0.0))
                except Exception:
                    pass
        if atr_val >= atr_threshold:
            high_pnl.append(pnl)
        else:
            low_pnl.append(pnl)

    all_pnl = [t["realized_pnl"] for t in trades]

    def _stats(pnls):
        return {
            "n":      len(pnls),
            "pnl":    round(sum(pnls), 2),
            "pf":     round(_profit_factor(pnls), 3),
            "sharpe": round(_sharpe(pnls), 3),
            "score":  round(_score(pnls), 3),
        }

    return {
        "low":  _stats(low_pnl),
        "high": _stats(high_pnl),
        "all":  _stats(all_pnl),
    }


# (legacy compat — kept for _regime_sharpes callers)
def _regime_sharpes(
    trades: List[Dict],
    df_15m: pd.DataFrame,
    atr_threshold: float,
) -> Tuple[float, float, int, int]:
    rs = _regime_stats(trades, df_15m, atr_threshold)
    return (
        rs["low"].get("sharpe", 0.0),
        rs["high"].get("sharpe", 0.0),
        rs["low"].get("n", 0),
        rs["high"].get("n", 0),
    )




def _build_strategy_config(params: Dict[str, Any]):
    """Create an OneMinuteStrategyConfig with the given params."""
    from shree.config import OneMinuteStrategyConfig
    cfg = OneMinuteStrategyConfig()
    merged = {**BASE_PARAMS, **params}
    for k, v in merged.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    # Wire ft_max_hold_bars → max_hold_minutes
    ft_bars = merged.get("ft_max_hold_bars", 8)
    cfg.max_hold_minutes = ft_bars * 15
    return cfg


def _run_single(
    df_slice: pd.DataFrame,
    params: Dict[str, Any],
    start_dt: datetime,
    end_dt: datetime,
    capital: float = 50_000.0,
    silent: bool = True,
) -> Optional[Dict]:
    """
    Run one 15m backtest on df_slice with the given params.
    Returns the results dict, or None on error.
    """
    import io
    from backtest.engine import BacktestEngine, BacktestConfig
    from shree.risk.risk_gate import RiskGateConfig

    strategy_cfg = _build_strategy_config(params)
    rg_cfg = RiskGateConfig()

    bt_cfg = BacktestConfig(
        symbol="MES",
        start_date=start_dt,
        end_date=end_dt,
        initial_capital=capital,
        slippage_ticks=1.0,
        commission_per_contract=2.40,
        session_type="full",
        strategy_config=strategy_cfg,
        risk_gate_config=rg_cfg,
        enable_trend_optimizer=False,  # speed
    )

    try:
        engine = BacktestEngine(bt_cfg)
        engine.load_15m_only(df_slice)
        results = engine.run_15m_only()
        return results
    except Exception as exc:
        logger.debug(f"_run_single error: {exc}")
        return None


# ── window generation ───────────────────────────────────────────────────────────

def _generate_windows(
    df: pd.DataFrame,
    is_months: int,
    oos_months: int,
) -> List[Tuple[datetime, datetime, datetime, datetime]]:
    """
    Generate (is_start, is_end, oos_start, oos_end) tuples.
    Roll by 1 month.
    """
    from dateutil.relativedelta import relativedelta

    # Determine monthly boundaries present in data
    all_months = sorted(df.index.tz_localize(None).to_period("M").unique())
    windows = []

    for i in range(len(all_months) - is_months - oos_months + 1):
        is_periods  = all_months[i : i + is_months]
        oos_periods = all_months[i + is_months : i + is_months + oos_months]

        if len(is_periods) < is_months or len(oos_periods) < oos_months:
            continue

        is_start  = is_periods[0].start_time.replace(tzinfo=timezone.utc)
        is_end    = is_periods[-1].end_time.replace(tzinfo=timezone.utc)
        oos_start = oos_periods[0].start_time.replace(tzinfo=timezone.utc)
        oos_end   = oos_periods[-1].end_time.replace(tzinfo=timezone.utc)

        windows.append((is_start, is_end, oos_start, oos_end))

    return windows


# ── grid search ─────────────────────────────────────────────────────────────────

def _grid_search(
    df_is: pd.DataFrame,
    is_start: datetime,
    is_end: datetime,
    atr_threshold: float,
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    Run full grid on IS data.
    Returns (best_low_vol_params, best_high_vol_params, full_results_list).
    """
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    total = len(combos)
    logger.warning(f"  IS grid: {total} combos for {is_start.date()} → {is_end.date()}")

    all_results: List[Dict] = []

    for idx, combo_vals in enumerate(combos):
        params = dict(zip(keys, combo_vals))
        results = _run_single(df_is, params, is_start, is_end)
        if results is None:
            continue

        trades = results.get("trades", [])
        rs = _regime_stats(trades, df_is, atr_threshold)

        all_results.append({
            "params":       params,
            "regime":       rs,
            "score_all":    rs["all"].get("score", 0.0),
            "score_low":    rs["low"].get("score", 0.0),
            "score_high":   rs["high"].get("score", 0.0),
            "n_trades":     rs["all"].get("n", 0),
            "total_pnl":    rs["all"].get("pnl", 0.0),
        })

        if (idx + 1) % 20 == 0:
            logger.warning(f"    {idx + 1}/{total} done")

    if not all_results:
        return BASE_PARAMS.copy(), BASE_PARAMS.copy(), []

    # Candidates must have at least 3 trades in the relevant regime
    # Best for low-vol: max score_low, then score_all as tiebreak
    cands_low = [r for r in all_results if r["regime"]["low"].get("n", 0) >= 3]
    if not cands_low:
        cands_low = all_results
    best_low = max(cands_low, key=lambda r: (r["score_low"], r["score_all"]))

    # Best for high-vol: max score_high, then score_all
    cands_high = [r for r in all_results if r["regime"]["high"].get("n", 0) >= 3]
    if not cands_high:
        cands_high = all_results
    best_high = max(cands_high, key=lambda r: (r["score_high"], r["score_all"]))

    return best_low["params"], best_high["params"], all_results


# ── OOS evaluation ──────────────────────────────────────────────────────────────

def _eval_oos(
    df_oos: pd.DataFrame,
    params_low: Dict,
    params_high: Dict,
    oos_start: datetime,
    oos_end: datetime,
    atr_threshold: float,
) -> Dict:
    """Evaluate OOS with low-vol and high-vol optimal params separately."""
    res_low  = _run_single(df_oos, params_low,  oos_start, oos_end) or {}
    res_high = _run_single(df_oos, params_high, oos_start, oos_end) or {}

    trades_low  = res_low.get("trades",  [])
    trades_high = res_high.get("trades", [])

    rs_low  = _regime_stats(trades_low,  df_oos, atr_threshold)
    rs_high = _regime_stats(trades_high, df_oos, atr_threshold)

    return {
        "low_vol_params":  params_low,
        "high_vol_params": params_high,
        "low_opt": {
            "n_trades":   rs_low["all"].get("n", 0),
            "total_pnl":  rs_low["all"].get("pnl", 0.0),
            "sharpe":     rs_low["all"].get("sharpe", 0.0),
            "pf":         rs_low["all"].get("pf", 0.0),
            "regime_low": rs_low["low"],
            "regime_high":rs_low["high"],
        },
        "high_opt": {
            "n_trades":   rs_high["all"].get("n", 0),
            "total_pnl":  rs_high["all"].get("pnl", 0.0),
            "sharpe":     rs_high["all"].get("sharpe", 0.0),
            "pf":         rs_high["all"].get("pf", 0.0),
            "regime_low": rs_high["low"],
            "regime_high":rs_high["high"],
        },
    }


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization (T4)")
    parser.add_argument("--data-file", required=True, help="Path to 15m parquet file")
    parser.add_argument("--atr-threshold", type=float, default=13.0,
                        help="ATR boundary for low vs high-vol regime (default 13)")
    parser.add_argument("--is-months",  type=int, default=3,
                        help="In-sample window length in months (default 3)")
    parser.add_argument("--oos-months", type=int, default=1,
                        help="Out-of-sample window length in months (default 1)")
    parser.add_argument("--output-dir", default="reports/wfo",
                        help="Output directory (default reports/wfo)")
    args = parser.parse_args()

    # ── load data ────────────────────────────────────────────────────────────
    data_path = Path(args.data_file)
    if not data_path.exists():
        sys.exit(f"Data file not found: {data_path}")

    print(f"\nLoading {data_path} ...", flush=True)
    df = pd.read_parquet(data_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")

    print(f"Loaded {len(df)} bars: {df.index.min().date()} → {df.index.max().date()}", flush=True)

    # Pre-compute ATR_14 on the full dataset (raw parquet has OHLCV only)
    print("Computing ATR_14 ...", flush=True)
    df = _add_atr(df, period=14)
    atr_pct_high = (df["ATR_14"] >= args.atr_threshold).mean() * 100
    print(f"ATR >= {args.atr_threshold}: {atr_pct_high:.1f}% of bars", flush=True)

    # ── generate windows ─────────────────────────────────────────────────────
    windows = _generate_windows(df, args.is_months, args.oos_months)
    if not windows:
        sys.exit("Not enough data to generate IS/OOS windows")

    print(f"\nWalk-forward windows: {len(windows)}", flush=True)
    for w in windows:
        is_s, is_e, oos_s, oos_e = w
        print(f"  IS: {is_s.date()} → {is_e.date()}  |  OOS: {oos_s.date()} → {oos_e.date()}", flush=True)

    # ── walk-forward loop ─────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wfo_summary: List[Dict] = []

    for win_idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
        print(f"\n{'='*60}", flush=True)
        print(f"Window {win_idx+1}/{len(windows)}: IS {is_start.date()}→{is_end.date()} | OOS {oos_start.date()}→{oos_end.date()}", flush=True)

        df_is  = df[(df.index >= is_start) & (df.index <= is_end)].copy()
        df_oos = df[(df.index >= oos_start) & (df.index <= oos_end)].copy()

        print(f"  IS bars: {len(df_is)}  |  OOS bars: {len(df_oos)}", flush=True)

        if len(df_is) < 50 or len(df_oos) < 10:
            print("  SKIP: insufficient data", flush=True)
            continue

        # Grid search on IS
        best_low, best_high, grid_results = _grid_search(
            df_is, is_start, is_end, args.atr_threshold
        )

        print(f"  Best low-vol params:  {best_low}", flush=True)
        print(f"  Best high-vol params: {best_high}", flush=True)

        # Evaluate OOS
        oos_result = _eval_oos(df_oos, best_low, best_high, oos_start, oos_end, args.atr_threshold)

        low_oos  = oos_result["low_opt"]
        high_oos = oos_result["high_opt"]
        print(f"\n  OOS (low-vol params):  trades={low_oos['n_trades']}  "
              f"pnl=${low_oos['total_pnl']:.0f}  pf={low_oos['pf']:.2f}  sharpe={low_oos['sharpe']:.2f}", flush=True)
        print(f"  OOS (high-vol params): trades={high_oos['n_trades']}  "
              f"pnl=${high_oos['total_pnl']:.0f}  pf={high_oos['pf']:.2f}  sharpe={high_oos['sharpe']:.2f}", flush=True)

        # Save per-window detail
        win_file = output_dir / f"wfo_window_{win_idx+1:02d}_{oos_start.date()}.json"
        with open(win_file, "w") as f:
            json.dump({
                "window": win_idx + 1,
                "is_start":  str(is_start.date()),
                "is_end":    str(is_end.date()),
                "oos_start": str(oos_start.date()),
                "oos_end":   str(oos_end.date()),
                "best_low_vol_params":  best_low,
                "best_high_vol_params": best_high,
                "oos_low_opt":          low_oos,
                "oos_high_opt":         high_oos,
                "is_grid_top10": sorted(
                    grid_results,
                    key=lambda r: r["score_all"],
                    reverse=True,
                )[:10],
            }, f, indent=2, default=str)

        wfo_summary.append({
            "window":    win_idx + 1,
            "oos_month": str(oos_start.date())[:7],
            "best_low_vol_params":  best_low,
            "best_high_vol_params": best_high,
            "oos_low_pnl":    low_oos["total_pnl"],
            "oos_high_pnl":   high_oos["total_pnl"],
            "oos_low_sharpe": low_oos["sharpe"],
            "oos_high_sharpe":high_oos["sharpe"],
            "oos_low_pf":     low_oos["pf"],
            "oos_high_pf":    high_oos["pf"],
            "oos_low_trades": low_oos["n_trades"],
            "oos_high_trades":high_oos["n_trades"],
        })

    # ── final summary ─────────────────────────────────────────────────────────
    if not wfo_summary:
        print("\nNo windows completed.", flush=True)
        return

    summary_path = output_dir / "wfo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(wfo_summary, f, indent=2, default=str)

    print(f"\n{'='*60}", flush=True)
    print("WALK-FORWARD SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    hdr = f"{'Month':<10} {'LowOpt PnL':>12} {'LowOpt PF':>10} {'HiOpt PnL':>12} {'HiOpt PF':>10} {'Trades(L)':>10} {'Trades(H)':>10}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    total_low_pnl  = 0.0
    total_high_pnl = 0.0
    for row in wfo_summary:
        print(f"{row['oos_month']:<10} "
              f"${row['oos_low_pnl']:>+10.0f}   "
              f"{row['oos_low_pf']:>9.2f}   "
              f"${row['oos_high_pnl']:>+10.0f}   "
              f"{row['oos_high_pf']:>9.2f}   "
              f"{row['oos_low_trades']:>8}   "
              f"{row['oos_high_trades']:>8}", flush=True)
        total_low_pnl  += row["oos_low_pnl"]
        total_high_pnl += row["oos_high_pnl"]

    print("-" * len(hdr), flush=True)
    print(f"{'TOTAL':<10} ${total_low_pnl:>+10.0f}{'':>13}${total_high_pnl:>+10.0f}", flush=True)
    print(f"\nResults saved to: {output_dir}/", flush=True)

    # ── generate recommended live params ─────────────────────────────────────
    # Take the most recent window's best params as live recommendation
    latest = wfo_summary[-1]
    rec_path = output_dir / "recommended_params.yaml"
    rec = {
        "_comment": (
            f"WFO recommended params — generated {datetime.now().date()} | "
            f"Latest IS window: {wfo_summary[-1]['oos_month']}"
        ),
        "low_vol_regime":  latest["best_low_vol_params"],
        "high_vol_regime": latest["best_high_vol_params"],
    }
    with open(rec_path, "w") as f:
        yaml.dump(rec, f, default_flow_style=False, sort_keys=False)
    print(f"Recommended params → {rec_path}", flush=True)


if __name__ == "__main__":
    main()
