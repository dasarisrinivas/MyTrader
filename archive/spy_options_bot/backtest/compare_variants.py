"""Compare multiple backtest variants side-by-side."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "spy_options_bot"))
sys.path.insert(0, str(Path(__file__).parents[1]))

from backtest.backtest_engine import BacktestEngine
from backtest.metrics import calculate_metrics

CACHE_DIR = Path(__file__).parents[2] / "backtest_results" / "cache"
START = date(2025, 3, 13)
END   = date(2026, 3, 13)
CAP   = 5_000.0

VARIANTS = [
    {"name": "Baseline",     "min_vix": 15.0, "entry_days": (0,1,2),   "strategy": "auto"},
    {"name": "Low VIX (12)", "min_vix": 12.0, "entry_days": (0,1,2),   "strategy": "auto"},
    {"name": "Thu entries",  "min_vix": 15.0, "entry_days": (0,1,2,3), "strategy": "auto"},
    {"name": "Puts-only",    "min_vix": 15.0, "entry_days": (0,1,2),   "strategy": "puts_only"},
    {"name": "Combined",     "min_vix": 12.0, "entry_days": (0,1,2,3), "strategy": "puts_only"},
]


def run_variant(v: dict) -> dict:
    engine = BacktestEngine(cache_dir=CACHE_DIR)
    results = engine.run(
        start_date=START,
        end_date=END,
        initial_capital=CAP,
        strategy=v["strategy"],
        entry_days=v["entry_days"],
        min_vix=v["min_vix"],
    )
    m = calculate_metrics(results)
    return {
        "name":          v["name"],
        "trades":        m["total_trades"],
        "return_pct":    m["total_return_pct"],
        "ann_return":    m["annualized_return_pct"],
        "win_rate":      m["win_rate_pct"],
        "sharpe":        m["sharpe_ratio"],
        "max_dd":        m["max_drawdown_pct"],
        "profit_factor": m["profit_factor"],
        "avg_pnl":       m["avg_pnl_per_trade"],
        "final_equity":  m["final_equity"],
        "pdt_blocked":   m["pdt_blocked_weeks"],
    }


def main():
    rows = []
    for v in VARIANTS:
        print(f"Running: {v['name']}...")
        rows.append(run_variant(v))

    header = (
        f"  {'Variant':<18} {'Trades':>7} {'Return':>8} {'Ann Ret':>8} {'Win%':>6} "
        f"{'Sharpe':>7} {'MaxDD':>7} {'PF':>5} {'AvgP&L':>8} {'Final $':>9} {'PDT-Blk':>8}"
    )
    divider = "=" * len(header)

    print()
    print(divider)
    print(header)
    print(divider)
    for r in rows:
        print(
            f"  {r['name']:<18} {r['trades']:>7} {r['return_pct']:>+7.1f}% {r['ann_return']:>+7.1f}% "
            f"{r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} {r['max_dd']:>+6.1f}% "
            f"{r['profit_factor']:>5.2f} {r['avg_pnl']:>+7.2f}  ${r['final_equity']:>8,.0f}  {r['pdt_blocked']:>5}wk"
        )
    print(divider)
    print()


if __name__ == "__main__":
    main()
