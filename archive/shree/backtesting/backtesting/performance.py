"""Performance analytics utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def summarize_performance(equity_curve: pd.Series, trades: List[Dict] | None = None) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    if equity_curve.empty:
        return _empty_metrics()
    
    returns = equity_curve.pct_change().dropna()
    
    # Basic metrics
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    initial_capital = float(equity_curve.iloc[0])
    final_capital = float(equity_curve.iloc[-1])
    total_pnl = final_capital - initial_capital
    
    # CAGR (Compound Annual Growth Rate)
    days = len(equity_curve)
    years = days / 252
    cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0
    
    # Sharpe Ratio
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = float(mean_return / std_return * (252 ** 0.5)) if std_return > 0 else 0
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return
    sortino = float(mean_return / downside_std * (252 ** 0.5)) if downside_std > 0 else 0
    
    # Drawdown analysis
    cummax = equity_curve.cummax()
    drawdown = (equity_curve / cummax - 1)
    max_drawdown = float(drawdown.min())
    
    # Average drawdown
    avg_drawdown = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0
    
    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = float(gains / losses) if losses != 0 else float("inf")
    
    # Calmar Ratio (CAGR / Max Drawdown)
    calmar = float(abs(cagr / max_drawdown)) if max_drawdown != 0 else 0
    
    metrics = {
        "total_return": total_return,
        "total_pnl": total_pnl,
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "avg_drawdown": avg_drawdown,
        "profit_factor": profit_factor,
        "calmar_ratio": calmar,
        "volatility": float(std_return * (252 ** 0.5)),
    }
    
    # Trade-specific metrics if trades provided
    if trades:
        trade_metrics = analyze_trades(trades)
        metrics.update(trade_metrics)
    
    return metrics


def analyze_trades(trades: List[Dict]) -> Dict[str, float]:
    """Analyze individual trades for detailed statistics."""
    if not trades:
        return {}
    
    # Filter for completed trades with realized PnL
    completed_trades = [t for t in trades if "realized" in t]
    
    if not completed_trades:
        return {
            "total_trades": len(trades),
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_trade": 0.0,
        }
    
    pnls = [t["realized"] for t in completed_trades]
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p < 0]
    
    total_trades = len(completed_trades)
    winning_trades = len(winning)
    losing_trades = len(losing)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    avg_win = float(np.mean(winning)) if winning else 0.0
    avg_loss = float(np.mean(losing)) if losing else 0.0
    largest_win = float(max(pnls)) if pnls else 0.0
    largest_loss = float(min(pnls)) if pnls else 0.0
    avg_trade = float(np.mean(pnls)) if pnls else 0.0
    
    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Average holding time (if timestamps available)
    holding_times = []
    for i in range(0, len(trades) - 1, 2):
        if i + 1 < len(trades) and "timestamp" in trades[i] and "timestamp" in trades[i+1]:
            try:
                entry_time = pd.Timestamp(trades[i]["timestamp"])
                exit_time = pd.Timestamp(trades[i+1]["timestamp"])
                holding_times.append((exit_time - entry_time).total_seconds() / 3600)  # hours
            except:
                pass
    
    avg_holding_time = float(np.mean(holding_times)) if holding_times else 0.0
    
    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "avg_trade": avg_trade,
        "expectancy": expectancy,
        "avg_holding_hours": avg_holding_time,
    }


def _empty_metrics() -> Dict[str, float]:
    """Return empty metrics dict."""
    return {
        "total_return": 0.0,
        "cagr": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "avg_drawdown": 0.0,
        "profit_factor": 0.0,
        "calmar_ratio": 0.0,
        "volatility": 0.0,
    }


def export_report(metrics: Dict[str, float], trades: List[Dict], path: Path, format: str = "json") -> None:
    """Export performance report to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        report = {
            "metrics": metrics,
            "trades": trades,
            "generated_at": pd.Timestamp.utcnow().isoformat()
        }
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
    elif format == "csv":
        # Export metrics as CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_path = path.parent / f"{path.stem}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        # Export trades as CSV
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_path = path.parent / f"{path.stem}_trades.csv"
            trades_df.to_csv(trades_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
