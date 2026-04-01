"""Market breadth internals via free yfinance data.

Approach (all free, no API key):
  - 11 SPDR sector ETFs measured against their day-open → breadth ratio
  - SPY 1-min bars → up/down volume proxy
  - NYSE TICK attempted via ^TICK (yfinance, not always available)

Refresh: 10-minute TTL (intraday)

BreadthState fields:
    breadth_ratio       0.0–1.0 fraction of sectors above their day-open
    breadth_label       STRONG / MODERATE / WEAK / NEUTRAL
    up_vol_ratio        up-volume / (up+down) from SPY 1-min bars
    sector_count_up     sectors trading above open
    sector_count_down   sectors trading below open
    tick_available      whether ^TICK data was fetchable
    tick_value          latest TICK snapshot (None if unavailable)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_SECTOR_ETFS = [
    "XLK",  # Technology
    "XLF",  # Financials
    "XLE",  # Energy
    "XLI",  # Industrials
    "XLV",  # Health Care
    "XLB",  # Materials
    "XLU",  # Utilities
    "XLRE", # Real Estate
    "XLP",  # Consumer Staples
    "XLY",  # Consumer Discretionary
    "XLC",  # Communication Services
]

_TTL_S = 600  # 10-minute refresh


@dataclass
class BreadthState:
    breadth_ratio: float = 0.5
    breadth_label: str = "NEUTRAL"
    up_vol_ratio: float = 0.5
    sector_count_up: int = 0
    sector_count_down: int = 0
    tick_available: bool = False
    tick_value: Optional[float] = None
    fetched_at: float = 0.0
    available: bool = False


class BreadthSignals:
    """Intraday market breadth via sector ETF participation + volume proxy."""

    def __init__(self) -> None:
        self._state = BreadthState()
        self._lock = asyncio.Lock()

    def is_stale(self) -> bool:
        return (time.monotonic() - self._state.fetched_at) > _TTL_S

    @property
    def state(self) -> BreadthState:
        return self._state

    async def refresh_if_stale(self) -> None:
        if not self.is_stale():
            return
        async with self._lock:
            if not self.is_stale():
                return
            loop = asyncio.get_event_loop()
            try:
                state = await loop.run_in_executor(None, self._fetch_sync)
                state.fetched_at = time.monotonic()
                self._state = state
            except Exception as exc:
                logger.warning("BreadthSignals fetch failed: %s", exc)

    # ------------------------------------------------------------------
    # Synchronous fetch (runs in executor thread)
    # ------------------------------------------------------------------

    def _fetch_sync(self) -> BreadthState:
        import yfinance as yf

        state = BreadthState()

        # --- Sector breadth -------------------------------------------
        tickers_to_fetch = _SECTOR_ETFS + ["SPY"]
        try:
            data = yf.download(
                tickers=" ".join(tickers_to_fetch),
                period="1d",
                interval="5m",
                progress=False,
                auto_adjust=True,
                threads=True,
            )
        except Exception as exc:
            logger.warning("BreadthSignals: yfinance download failed: %s", exc)
            return state

        # yfinance returns MultiIndex columns when multiple tickers
        if data.empty:
            return state

        up = 0
        down = 0
        for etf in _SECTOR_ETFS:
            try:
                if "Close" in data.columns.get_level_values(0):
                    close_series = data["Close"][etf].dropna()
                    open_series = data["Open"][etf].dropna()
                else:
                    # Single-ticker fallback (shouldn't happen in multi-ticker case)
                    close_series = data["Close"].dropna()
                    open_series = data["Open"].dropna()

                if close_series.empty or open_series.empty:
                    continue

                day_open = float(open_series.iloc[0])
                last_close = float(close_series.iloc[-1])

                if last_close > day_open:
                    up += 1
                elif last_close < day_open:
                    down += 1
            except Exception:
                continue

        total_sectors = up + down
        state.sector_count_up = up
        state.sector_count_down = down
        if total_sectors > 0:
            state.breadth_ratio = up / total_sectors
        else:
            state.breadth_ratio = 0.5

        # --- Breadth label -------------------------------------------
        state.breadth_label = _breadth_label(state.breadth_ratio, total_sectors)

        # --- Up/Down volume proxy from SPY 1-min bars ----------------
        try:
            if "Close" in data.columns.get_level_values(0):
                spy_close = data["Close"]["SPY"].dropna()
                spy_vol = data["Volume"]["SPY"].dropna()
            else:
                spy_close = data["Close"].dropna()
                spy_vol = data["Volume"].dropna()

            if len(spy_close) >= 2 and len(spy_vol) >= 2:
                closes = spy_close.values
                vols = spy_vol.values
                up_vol = 0.0
                dn_vol = 0.0
                for i in range(1, len(closes)):
                    if closes[i] > closes[i - 1]:
                        up_vol += vols[i]
                    elif closes[i] < closes[i - 1]:
                        dn_vol += vols[i]
                total_vol = up_vol + dn_vol
                state.up_vol_ratio = up_vol / total_vol if total_vol > 0 else 0.5
        except Exception as exc:
            logger.debug("BreadthSignals: up/down vol calc failed: %s", exc)
            state.up_vol_ratio = 0.5

        # --- NYSE TICK attempt (^TICK via yfinance, often unavailable) -
        try:
            tick_data = yf.download(
                "^TICK",
                period="1d",
                interval="5m",
                progress=False,
                auto_adjust=False,
            )
            if not tick_data.empty:
                tick_close = tick_data["Close"].dropna()
                if not tick_close.empty:
                    state.tick_value = float(tick_close.iloc[-1])
                    state.tick_available = True
        except Exception:
            pass  # TICK not available — fine

        state.available = total_sectors >= 6  # Need at least 6 sectors
        return state


def _breadth_label(ratio: float, total: int) -> str:
    if total < 6:
        return "NEUTRAL"
    if ratio >= 0.75:
        return "STRONG"
    if ratio >= 0.55:
        return "MODERATE"
    if ratio <= 0.25:
        return "WEAK"
    if ratio <= 0.45:
        return "MODERATE_WEAK"
    return "NEUTRAL"
