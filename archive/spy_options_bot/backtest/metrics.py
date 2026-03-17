"""Performance metrics calculator for SPY options backtest results.

All metrics derived from the SimulatedTrade list and daily equity curve.
Returns a flat dict suitable for JSON serialisation and console display.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

from backtest.backtest_engine import BacktestResults, SimulatedTrade

RISK_FREE_DAILY = 0.053 / 252.0   # approximate daily risk-free return


def calculate_metrics(results: BacktestResults) -> dict:
    """Compute all performance metrics from BacktestResults.

    Returns a dict with keys matching the dashboard sections in reporter.py.
    """
    trades = results.trades
    eq = results.equity_curve
    initial = results.initial_capital

    # Filter out end_of_backtest force-closes for cleaner stats (include in equity)
    real_trades = [t for t in trades if t.exit_reason != "end_of_backtest"]

    m: dict = {}

    # ------------------------------------------------------------------
    # Basic counts
    # ------------------------------------------------------------------
    m["total_trades"] = len(real_trades)
    m["start_date"] = results.start_date.isoformat()
    m["end_date"] = results.end_date.isoformat()
    m["initial_capital"] = round(initial, 2)
    m["pdt_blocked_weeks"] = results.pdt_blocked_weeks
    m["no_trade_days"] = len(results.no_trade_records)

    if not real_trades:
        _empty = {k: 0.0 for k in [
            "total_return_pct", "annualized_return_pct", "win_rate_pct",
            "avg_win", "avg_loss", "profit_factor", "expected_value",
            "max_drawdown_pct", "max_drawdown_duration_days",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio", "var_95",
            "avg_days_held", "avg_premium_collected", "avg_pnl_per_trade",
        ]}
        m.update(_empty)
        m["final_equity"] = round(initial, 2)
        m["exit_reason_counts"] = {}
        m["monthly_returns"] = []
        m["winning_trades"] = 0
        m["losing_trades"] = 0
        return m

    pnls = [t.net_pnl for t in real_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------
    final_equity = float(eq["equity"].iloc[-1]) if not eq.empty else initial
    m["final_equity"] = round(final_equity, 2)

    trading_days_count = len(eq)
    total_return = (final_equity - initial) / initial
    m["total_return_pct"] = round(total_return * 100, 2)

    if trading_days_count > 1:
        ann_return = (1 + total_return) ** (252 / trading_days_count) - 1
    else:
        ann_return = 0.0
    m["annualized_return_pct"] = round(ann_return * 100, 2)

    m["winning_trades"] = len(wins)
    m["losing_trades"] = len(losses)
    m["win_rate_pct"] = round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0
    m["avg_win"] = round(np.mean(wins), 2) if wins else 0.0
    m["avg_loss"] = round(np.mean(losses), 2) if losses else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    m["profit_factor"] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    wr = len(wins) / len(pnls)
    m["expected_value"] = round(wr * m["avg_win"] + (1 - wr) * m["avg_loss"], 2)

    # ------------------------------------------------------------------
    # Risk metrics (from equity curve)
    # ------------------------------------------------------------------
    if not eq.empty:
        daily_equity = eq["equity"].values.astype(float)
        daily_returns = np.diff(daily_equity) / daily_equity[:-1]

        # Drawdown
        peak = np.maximum.accumulate(daily_equity)
        drawdown = (daily_equity - peak) / peak
        max_dd = float(np.min(drawdown))
        m["max_drawdown_pct"] = round(max_dd * 100, 2)

        # Drawdown duration
        in_drawdown = drawdown < 0
        dd_duration = 0
        current_duration = 0
        for flag in in_drawdown:
            if flag:
                current_duration += 1
                dd_duration = max(dd_duration, current_duration)
            else:
                current_duration = 0
        m["max_drawdown_duration_days"] = dd_duration

        # Sharpe
        excess = daily_returns - RISK_FREE_DAILY
        std_all = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
        m["sharpe_ratio"] = round(
            float(np.mean(excess)) / std_all * np.sqrt(252) if std_all > 0 else 0.0, 2
        )

        # Sortino (downside only)
        downside = daily_returns[daily_returns < RISK_FREE_DAILY]
        std_down = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        m["sortino_ratio"] = round(
            float(np.mean(excess)) / std_down * np.sqrt(252) if std_down > 0 else 0.0, 2
        )

        # Calmar
        m["calmar_ratio"] = round(
            ann_return / abs(max_dd) if max_dd < 0 else float("inf"), 2
        )

        # VaR 95% (5th percentile of daily returns)
        m["var_95"] = round(float(np.percentile(daily_returns, 5)) * 100, 2) if len(daily_returns) > 0 else 0.0
    else:
        for k in ["max_drawdown_pct", "max_drawdown_duration_days",
                  "sharpe_ratio", "sortino_ratio", "calmar_ratio", "var_95"]:
            m[k] = 0.0

    # ------------------------------------------------------------------
    # Trade-level metrics
    # ------------------------------------------------------------------
    m["avg_days_held"] = round(np.mean([t.days_held for t in real_trades]), 1)
    m["avg_premium_collected"] = round(np.mean([t.entry_premium for t in real_trades]), 2)
    m["avg_pnl_per_trade"] = round(np.mean(pnls), 2)
    m["total_commissions"] = round(sum(t.commissions for t in real_trades), 2)

    # Exit reason breakdown
    exit_counts: dict[str, int] = defaultdict(int)
    for t in real_trades:
        exit_counts[t.exit_reason] += 1
    m["exit_reason_counts"] = dict(exit_counts)

    # ------------------------------------------------------------------
    # Monthly P&L breakdown
    # ------------------------------------------------------------------
    monthly: dict[str, dict] = defaultdict(
        lambda: {"trades": 0, "wins": 0, "losses": 0, "gross_pnl": 0.0, "net_pnl": 0.0}
    )
    for t in real_trades:
        key = t.close_date.strftime("%Y-%m")
        monthly[key]["trades"] += 1
        monthly[key]["net_pnl"] += t.net_pnl
        monthly[key]["gross_pnl"] += t.gross_pnl
        if t.net_pnl > 0:
            monthly[key]["wins"] += 1
        else:
            monthly[key]["losses"] += 1

    monthly_rows = []
    for month_key in sorted(monthly.keys()):
        row = {"month": month_key, **monthly[month_key]}
        row["return_pct"] = round(row["net_pnl"] / initial * 100, 2)
        row["gross_pnl"] = round(row["gross_pnl"], 2)
        row["net_pnl"] = round(row["net_pnl"], 2)
        monthly_rows.append(row)
    m["monthly_returns"] = monthly_rows

    # ------------------------------------------------------------------
    # Peak drawdown timing (for console display)
    # ------------------------------------------------------------------
    if not eq.empty:
        equity_series = eq["equity"]
        peak_idx = (equity_series / equity_series.cummax()).idxmin()
        # Find prior peak
        prior_peak_idx = equity_series[:peak_idx].idxmax() if len(equity_series[:peak_idx]) > 0 else peak_idx
        m["drawdown_peak_date"] = str(prior_peak_idx.date())
        # Recovery: first date after peak_idx where equity >= prior peak value
        prior_peak_val = float(equity_series.loc[prior_peak_idx]) if prior_peak_idx != peak_idx else initial
        recovery_dates = equity_series[peak_idx:][equity_series[peak_idx:] >= prior_peak_val]
        m["drawdown_recovery_date"] = str(recovery_dates.index[0].date()) if not recovery_dates.empty else "not_recovered"

    return m
