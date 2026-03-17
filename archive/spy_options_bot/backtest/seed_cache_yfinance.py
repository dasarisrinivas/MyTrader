"""Seed the backtest cache using yfinance (no IBKR needed).

Saves SPY daily, VIX daily, and SPY 5-min data in the exact CSV format
expected by data_downloader.py's load_* functions.

Usage:
    python spy_options_bot/backtest/seed_cache_yfinance.py
    python spy_options_bot/backtest/seed_cache_yfinance.py --start 2024-01-01
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parents[2] / "backtest_results" / "cache"


def seed(start: date, end: date) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = end.strftime("%Y%m%d")

    # ── SPY daily ─────────────────────────────────────────────────────────
    spy_daily_path = CACHE_DIR / f"SPY_daily_{tag}.csv"
    if spy_daily_path.exists():
        print(f"[skip] SPY daily already cached: {spy_daily_path}")
    else:
        print("Downloading SPY daily bars via yfinance...")
        raw = yf.download("SPY", start=start.isoformat(), end=end.isoformat(),
                          interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned no SPY daily data")
        # Flatten multi-level columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(spy_daily_path, index=False)
        print(f"  SPY daily: {len(df)} bars → {spy_daily_path}")

    # ── VIX daily ─────────────────────────────────────────────────────────
    vix_daily_path = CACHE_DIR / f"VIX_daily_{tag}.csv"
    if vix_daily_path.exists():
        print(f"[skip] VIX daily already cached: {vix_daily_path}")
    else:
        print("Downloading VIX daily bars via yfinance...")
        raw = yf.download("^VIX", start=start.isoformat(), end=end.isoformat(),
                          interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned no VIX daily data")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        df = raw[["Open", "High", "Low", "Close"]].copy()
        df.columns = ["open", "high", "low", "close"]
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(vix_daily_path, index=False)
        print(f"  VIX daily: {len(df)} bars → {vix_daily_path}")

    # ── SPY 5-minute (last 30 days, yfinance limit is ~60 days for 5m) ────
    spy_5min_path = CACHE_DIR / f"SPY_5min_{tag}.csv"
    if spy_5min_path.exists():
        print(f"[skip] SPY 5-min already cached: {spy_5min_path}")
    else:
        print("Downloading SPY 5-min bars via yfinance (last 30 days)...")
        start_5m = end - timedelta(days=30)
        raw = yf.download("SPY", start=start_5m.isoformat(), end=end.isoformat(),
                          interval="5m", auto_adjust=True, progress=False)
        if raw.empty:
            print("  WARNING: No SPY 5-min data — emergency gamma will use daily range")
            pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"]).to_csv(
                spy_5min_path, index=False
            )
        else:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index.name = "datetime"
            df = df.reset_index()
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.sort_values("datetime").reset_index(drop=True)
            df.to_csv(spy_5min_path, index=False)
            print(f"  SPY 5-min: {len(df)} bars → {spy_5min_path}")

    print("\nCache seeded. Run the backtest with:")
    print("  python spy_options_bot/backtest/run_backtest.py --use-cache")


def main() -> None:
    one_year_ago = date.today() - timedelta(days=365)
    today = date.today()

    parser = argparse.ArgumentParser(description="Seed backtest cache from yfinance")
    parser.add_argument("--start", default=one_year_ago.isoformat())
    parser.add_argument("--end", default=today.isoformat())
    args = parser.parse_args()

    seed(date.fromisoformat(args.start), date.fromisoformat(args.end))


if __name__ == "__main__":
    main()
