"""Historical data downloader for SPY options backtest.

Downloads from IBKR and caches to CSV files so reruns are instant.
Uses clientId=3 (separate from live bot=20 and MES bot=11).

Rate-limiting:
  IBKR limits historical data requests to ~60 per 10 minutes.
  IBKRRateLimiter enforces a 55-request ceiling with auto-pause.
"""
from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_here = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(_here / "spy_options_bot"))

from ib_insync import IB, Index, Option, Stock
from logger import logger

BACKTEST_CLIENT_ID = 3


class IBKRRateLimiter:
    """Enforces IBKR's ~60 historical data requests per 10-minute window."""

    MAX_REQUESTS = 55
    WINDOW_SECONDS = 600

    def __init__(self) -> None:
        self._request_times: list[float] = []

    async def throttle(self) -> None:
        now = time.monotonic()
        self._request_times = [
            t for t in self._request_times if now - t < self.WINDOW_SECONDS
        ]
        if len(self._request_times) >= self.MAX_REQUESTS:
            wait = self.WINDOW_SECONDS - (now - self._request_times[0]) + 2
            logger.info(
                f"[RateLimiter] {len(self._request_times)}/{self.MAX_REQUESTS} requests used "
                f"in last {self.WINDOW_SECONDS}s — sleeping {wait:.0f}s"
            )
            await asyncio.sleep(wait)
        self._request_times.append(time.monotonic())


class DataDownloader:
    """Downloads SPY and VIX historical data from IBKR and caches to CSV."""

    SPY_DAILY_FILE = "spy_daily.csv"
    SPY_5MIN_FILE = "spy_5min.csv"
    VIX_DAILY_FILE = "vix_daily.csv"

    def __init__(self, ib: IB, cache_dir: Path) -> None:
        self.ib = ib
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limiter = IBKRRateLimiter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def download_all(self, end_date: str = "") -> dict[str, Path]:
        """Download all required data. Returns dict of {name: csv_path}."""
        results: dict[str, Path] = {}
        results["spy_daily"] = await self._download_spy_daily(end_date)
        results["vix_daily"] = await self._download_vix_daily(end_date)
        results["spy_5min"] = await self._download_spy_5min()
        return results

    # ------------------------------------------------------------------
    # SPY daily bars — 1 year
    # ------------------------------------------------------------------

    async def _download_spy_daily(self, end_date: str = "") -> Path:
        cache_key = self._cache_key("SPY_daily", end_date or date.today().strftime("%Y%m%d"))
        if cache_key.exists():
            logger.info(f"Using cached SPY daily: {cache_key}")
            return cache_key

        logger.info("Downloading SPY daily bars (1 year)...")
        spy = Stock("SPY", "SMART", "USD")
        await self.ib.qualifyContractsAsync(spy)
        await self.rate_limiter.throttle()

        bars = await self.ib.reqHistoricalDataAsync(
            spy,
            endDateTime=end_date,
            durationStr="1 Y",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            raise RuntimeError("No SPY daily bars returned from IBKR")

        df = pd.DataFrame(
            [{"date": b.date, "open": b.open, "high": b.high,
              "low": b.low, "close": b.close, "volume": b.volume}
             for b in bars]
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(cache_key, index=False)
        logger.info(f"SPY daily saved: {len(df)} bars → {cache_key}")
        return cache_key

    # ------------------------------------------------------------------
    # SPY 5-minute bars — last 30 days
    # ------------------------------------------------------------------

    async def _download_spy_5min(self) -> Path:
        cache_key = self._cache_key("SPY_5min", date.today().strftime("%Y%m%d"))
        if cache_key.exists():
            logger.info(f"Using cached SPY 5-min: {cache_key}")
            return cache_key

        logger.info("Downloading SPY 5-min bars (30 days)...")
        spy = Stock("SPY", "SMART", "USD")
        await self.ib.qualifyContractsAsync(spy)
        await self.rate_limiter.throttle()

        bars = await self.ib.reqHistoricalDataAsync(
            spy,
            endDateTime="",
            durationStr="30 D",
            barSizeSetting="5 mins",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,  # UTC timestamps
        )
        if not bars:
            logger.warning("No SPY 5-min bars returned — emergency gamma check will use daily range")
            # Write empty file so we don't retry every run
            pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"]).to_csv(
                cache_key, index=False
            )
            return cache_key

        df = pd.DataFrame(
            [{"datetime": b.date, "open": b.open, "high": b.high,
              "low": b.low, "close": b.close, "volume": b.volume}
             for b in bars]
        )
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime").reset_index(drop=True)
        df.to_csv(cache_key, index=False)
        logger.info(f"SPY 5-min saved: {len(df)} bars → {cache_key}")
        return cache_key

    # ------------------------------------------------------------------
    # VIX daily bars — 1 year
    # ------------------------------------------------------------------

    async def _download_vix_daily(self, end_date: str = "") -> Path:
        cache_key = self._cache_key("VIX_daily", end_date or date.today().strftime("%Y%m%d"))
        if cache_key.exists():
            logger.info(f"Using cached VIX daily: {cache_key}")
            return cache_key

        logger.info("Downloading VIX daily bars (1 year)...")
        vix = Index("VIX", "CBOE", "USD")
        qualified = await self.ib.qualifyContractsAsync(vix)
        if not qualified:
            raise RuntimeError("Could not qualify VIX contract")
        await self.rate_limiter.throttle()

        bars = await self.ib.reqHistoricalDataAsync(
            qualified[0],
            endDateTime=end_date,
            durationStr="1 Y",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            raise RuntimeError("No VIX bars returned from IBKR")

        df = pd.DataFrame(
            [{"date": b.date, "open": b.open, "high": b.high,
              "low": b.low, "close": b.close}
             for b in bars]
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(cache_key, index=False)
        logger.info(f"VIX daily saved: {len(df)} bars → {cache_key}")
        return cache_key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, name: str, date_tag: str) -> Path:
        return self.cache_dir / f"{name}_{date_tag}.csv"


# ------------------------------------------------------------------
# Standalone loader (reads from cache, no IBKR needed)
# ------------------------------------------------------------------

def load_spy_daily(cache_dir: Path) -> pd.DataFrame:
    """Load the most recent SPY daily CSV from cache."""
    files = sorted(cache_dir.glob("SPY_daily_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No SPY daily cache found in {cache_dir}. Run with --download-only first."
        )
    df = pd.read_csv(files[-1], parse_dates=["date"])
    return df.set_index("date").sort_index()


def load_vix_daily(cache_dir: Path) -> pd.DataFrame:
    """Load the most recent VIX daily CSV from cache."""
    files = sorted(cache_dir.glob("VIX_daily_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No VIX daily cache found in {cache_dir}. Run with --download-only first."
        )
    df = pd.read_csv(files[-1], parse_dates=["date"])
    return df.set_index("date").sort_index()


def load_spy_5min(cache_dir: Path) -> pd.DataFrame | None:
    """Load SPY 5-min bars from cache. Returns None if not available."""
    files = sorted(cache_dir.glob("SPY_5min_*.csv"))
    if not files:
        return None
    df = pd.read_csv(files[-1], parse_dates=["datetime"])
    if df.empty:
        return None
    df = df.set_index("datetime").sort_index()
    return df
