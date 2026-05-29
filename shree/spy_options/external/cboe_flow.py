"""
CBOE total options put/call ratio from the public daily CSV.
Refreshed once per day.

Source: https://www.cboe.com/us/options/market_statistics/daily/
The daily CSV (options_pc_ratio_p.csv) contains SPX, SPY, total equity, and total index P/C.
"""
from __future__ import annotations

import asyncio
import io
import time
from dataclasses import dataclass
from datetime import date
from typing import Optional
import aiohttp
from loguru import logger

_CBOE_TOTAL_URL = (
    "https://www.cboe.com/us/options/market_statistics/daily/"
    "?mkt=cone&type=put_call_ratios&csv=1"
)

# Fallback: simpler public CSV for equity P/C ratio
_CBOE_EQ_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/PC_EQUITY.csv"


@dataclass
class CboeFlowState:
    equity_pc: Optional[float] = None   # equity P/C ratio
    index_pc: Optional[float] = None    # index P/C ratio
    total_pc: Optional[float] = None    # total P/C ratio
    fetched_date: Optional[date] = None
    available: bool = False

    def is_stale(self) -> bool:
        return self.fetched_date is None or self.fetched_date < date.today()

    @property
    def sentiment_bias(self) -> float:
        """
        Converts P/C ratio to a sentiment score -1..+1.
        High P/C (> 1.2) = bearish retail positioning → contrarian bullish signal.
        Low P/C  (< 0.7) = bullish retail → contrarian bearish signal.
        """
        pc = self.equity_pc or self.total_pc
        if pc is None:
            return 0.0
        if pc > 1.2:
            return 0.3   # contrarian: retail too bearish
        if pc > 1.0:
            return 0.1
        if pc < 0.7:
            return -0.3  # contrarian: retail too bullish
        if pc < 0.85:
            return -0.1
        return 0.0


class CboeFlow:
    def __init__(self, timeout_s: float = 10.0):
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._state = CboeFlowState()
        self._lock = asyncio.Lock()

    async def refresh_if_stale(self) -> None:
        if not self._state.is_stale():
            return
        async with self._lock:
            if not self._state.is_stale():
                return
            await self._fetch()

    async def _fetch(self) -> None:
        # Try the simpler equity P/C CSV first (most reliable endpoint)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.get(_CBOE_EQ_URL, headers=headers) as resp:
                    resp.raise_for_status()
                    text = await resp.text()
            eq_pc = self._parse_equity_csv(text)
            self._state = CboeFlowState(
                equity_pc=eq_pc,
                fetched_date=date.today(),
                available=eq_pc is not None,
            )
            if eq_pc is not None:
                logger.debug(f"[CBOE] Equity P/C ratio: {eq_pc:.3f}")
        except Exception as exc:
            logger.warning(f"[CBOE] Fetch failed: {exc}")
            self._state = CboeFlowState(
                fetched_date=date.today(), available=False
            )

    def _parse_equity_csv(self, text: str) -> Optional[float]:
        """Parse last row of CBOE equity P/C CSV; return the ratio."""
        try:
            lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
            if len(lines) < 2:
                return None
            last = lines[-1].split(",")
            # Format: DATE, PC_RATIO  (or similar)
            if len(last) >= 2:
                return float(last[-1])
        except Exception:
            pass
        return None

    @property
    def state(self) -> CboeFlowState:
        return self._state
