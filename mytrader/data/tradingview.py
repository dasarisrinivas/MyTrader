"""TradingView webhook and REST integration."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, AsyncIterator

import httpx
import pandas as pd

from ..utils.logger import logger
from .base import DataCollector


class TradingViewCollector(DataCollector):
    """Pulls candle data from a TradingView-compatible REST endpoint with rate limiting."""

    def __init__(
        self, 
        base_url: str, 
        symbol: str, 
        interval: str = "1m", 
        client: httpx.AsyncClient | None = None,
        max_retries: int = 3,
        rate_limit_delay: float = 1.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.symbol = symbol
        self.interval = interval
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = asyncio.get_event_loop().time()

    async def collect(self) -> pd.DataFrame:
        """Collect historical data with retry logic and rate limiting."""
        await self._rate_limit()
        
        params = {"symbol": self.symbol, "interval": self.interval, "limit": 500}
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.get(f"{self.base_url}/history", params=params)
                response.raise_for_status()
                payload = response.json()
                df = pd.DataFrame(payload["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                df.set_index("timestamp", inplace=True)
                return df.sort_index()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    retry_after = int(e.response.headers.get("Retry-After", 60))
                    logger.warning("TradingView rate limited. Waiting %d seconds...", retry_after)
                    await asyncio.sleep(retry_after)
                elif attempt < self.max_retries - 1:
                    delay = 2 ** attempt
                    logger.warning("TradingView HTTP error %d (attempt %d/%d). Retrying in %ds...", 
                                 e.response.status_code, attempt + 1, self.max_retries, delay)
                    await asyncio.sleep(delay)
                else:
                    logger.error("TradingView request failed after %d attempts", self.max_retries)
                    raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt
                    logger.warning("TradingView collection error (attempt %d/%d): %s. Retrying in %ds...", 
                                 attempt + 1, self.max_retries, e, delay)
                    await asyncio.sleep(delay)
                else:
                    logger.error("TradingView collection failed: %s", e)
                    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Stream data with error handling and rate limiting."""
        while True:
            try:
                df = await self.collect()
                if not df.empty:
                    latest = df.iloc[-1]
                    yield {
                        "timestamp": latest.name.to_pydatetime(),
                        "open": float(latest["open"]),
                        "high": float(latest["high"]),
                        "low": float(latest["low"]),
                        "close": float(latest["close"]),
                        "volume": float(latest["volume"]),
                        "source": "tradingview",
                    }
            except Exception as exc:  # noqa: BLE001
                logger.error("TradingView stream error: %s", exc)
                yield {"timestamp": datetime.utcnow(), "error": str(exc), "source": "tradingview"}
            await asyncio.sleep(60)
