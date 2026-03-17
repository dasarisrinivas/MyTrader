"""Run the strategy through multiple historical regimes.

Downloads data for each period via yfinance (no IBKR needed) into
separate cache subdirectories, runs the backtest, and prints a
side-by-side comparison table.

Periods covered:
  2013–2015  Low-vol bull  : Jan 2013 – Dec 2015  (VIX <15 most of year)
  2018  Q4 crash           : Oct 2018 – Mar 2019  (-20% peak-to-trough)
  2020  COVID crash        : Jan 2020 – Aug 2020  (-34% peak-to-trough)
  2022  Bear market        : Jan 2022 – Dec 2022  (-25% peak-to-trough)
  2023  Bull recovery      : Jan 2023 – Dec 2023  (+24% recovery year)

Usage:
    python spy_options_bot/backtest/run_crash_tests.py
    python spy_options_bot/backtest/run_crash_tests.py --capital 5000
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

_file = Path(__file__).resolve()
_backtest_dir = _file.parent
_bot_dir = _backtest_dir.parent
_repo_root = _bot_dir.parent

sys.path.insert(0, str(_bot_dir))
sys.path.insert(0, str(_backtest_dir.parent))

from backtest.backtest_engine import BacktestEngine
from backtest.metrics import calculate_metrics

OUTPUT_DIR = _repo_root / "backtest_results"

CRASH_PERIODS = [
    {
        "label": "2013-2015 Low Vol",
        "download_start": date(2012, 6, 1),   # need SMA200 lookback
        "backtest_start": date(2013, 1, 1),
        "backtest_end":   date(2015, 12, 31),
        "note": "Low-vol bull run, VIX <15 most of 2013-14 — ideal premium-selling env",
    },
    {
        "label": "2018 Q4 Crash",
        "download_start": date(2017, 6, 1),
        "backtest_start": date(2018, 9, 1),
        "backtest_end":   date(2019, 3, 31),
        "note": "Fed rate hike + trade war selloff, -20%",
    },
    {
        "label": "2020 COVID Crash",
        "download_start": date(2019, 6, 1),
        "backtest_start": date(2020, 1, 15),
        "backtest_end":   date(2020, 8, 31),
        "note": "COVID pandemic selloff, -34% in 33 days",
    },
    {
        "label": "2022 Bear Market",
        "download_start": date(2021, 6, 1),
        "backtest_start": date(2022, 1, 1),
        "backtest_end":   date(2022, 12, 31),
        "note": "Fed tightening cycle, -25% over full year",
    },
    {
        "label": "2023 Bull Recovery",
        "download_start": date(2022, 6, 1),
        "backtest_start": date(2023, 1, 1),
        "backtest_end":   date(2023, 12, 31),
        "note": "Post-bear recovery, +24% SPY — moderate vol, trend following",
    },
]


def _seed_cache(cache_dir: Path, dl_start: date, dl_end: date, label: str) -> None:
    """Download SPY + VIX daily data for the crash period into cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = dl_end.strftime("%Y%m%d")

    spy_path = cache_dir / f"SPY_daily_{tag}.csv"
    vix_path = cache_dir / f"VIX_daily_{tag}.csv"
    spy_5m_path = cache_dir / f"SPY_5min_{tag}.csv"

    if not spy_path.exists():
        print(f"  Downloading SPY daily for {label}...")
        raw = yf.download("SPY", start=dl_start.isoformat(), end=dl_end.isoformat(),
                          interval="1d", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(spy_path, index=False)
        print(f"    {len(df)} bars → {spy_path.name}")

    if not vix_path.exists():
        print(f"  Downloading VIX daily for {label}...")
        raw = yf.download("^VIX", start=dl_start.isoformat(), end=dl_end.isoformat(),
                          interval="1d", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        df = raw[["Open", "High", "Low", "Close"]].copy()
        df.columns = ["open", "high", "low", "close"]
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(vix_path, index=False)
        print(f"    {len(df)} bars → {vix_path.name}")

    # No 5-min data available for historical crashes — engine falls back to daily range
    if not spy_5m_path.exists():
        pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"]).to_csv(
            spy_5m_path, index=False
        )


def _run_period(period: dict, capital: float) -> dict:
    """Run backtest for one crash period and return metrics dict."""
    cache_dir = OUTPUT_DIR / "cache_crash" / period["label"].replace(" ", "_")
    dl_end = period["backtest_end"] + __import__("datetime").timedelta(days=5)

    _seed_cache(cache_dir, period["download_start"], dl_end, period["label"])

    engine = BacktestEngine(cache_dir=cache_dir)
    results = engine.run(
        start_date=period["backtest_start"],
        end_date=period["backtest_end"],
        initial_capital=capital,
        strategy="auto",
    )
    m = calculate_metrics(results)
    m["_label"] = period["label"]
    m["_note"] = period["note"]
    m["_trades"] = len(results.trades)
    return m


def _print_comparison(rows: list[dict], capital: float) -> None:
    print()
    print("=" * 75)
    print(f"  CRASH BACKTEST COMPARISON  |  Starting Capital: ${capital:,.0f}")
    print("=" * 75)
    header = f"{'Period':<24} {'Return':>8} {'WinRate':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'AvgPnL':>9}"
    print(header)
    print("-" * 75)
    for m in rows:
        print(
            f"{m['_label']:<24} "
            f"{m.get('total_return_pct', 0):>+7.1f}% "
            f"{m.get('win_rate_pct', 0):>7.1f}% "
            f"{m.get('sharpe_ratio', 0):>8.2f} "
            f"{m.get('max_drawdown_pct', 0):>7.1f}% "
            f"{m['_trades']:>7} "
            f"${m.get('avg_pnl_per_trade', 0):>+8.2f}"
        )
        print(f"  ↳ {m['_note']}")
    print("=" * 75)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="SPY Options Bot — Crash Backtests")
    parser.add_argument("--capital", type=float, default=5_000.0,
                        help="Starting capital (default: 5000)")
    args = parser.parse_args()

    print("\nRunning regime backtests (5 periods)...")
    rows = []
    for period in CRASH_PERIODS:
        print(f"\n[{period['label']}]")
        m = _run_period(period, args.capital)
        rows.append(m)

    _print_comparison(rows, args.capital)


if __name__ == "__main__":
    main()
