"""
StockTwits retail sentiment for $SPY via free public API (no auth required).
Returns bullish_pct, bearish_pct, and a normalised score.
TTL-cached (default 10 min).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional
import aiohttp
from loguru import logger

_API_URL = "https://api.stocktwits.com/api/2/streams/symbol/SPY.json"


@dataclass
class StockTwitsState:
    score: float = 0.0          # -1.0 (bearish) to +1.0 (bullish)
    bullish_pct: float = 0.0    # 0–100
    bearish_pct: float = 0.0    # 0–100
    message_count: int = 0
    fetched_at: float = 0.0
    available: bool = False

    def is_stale(self, ttl_s: float) -> bool:
        return (time.monotonic() - self.fetched_at) > ttl_s


class StockTwitsClient:
    def __init__(self, ttl_minutes: float = 10.0, timeout_s: float = 8.0):
        self._ttl_s = ttl_minutes * 60.0
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._state = StockTwitsState()

    async def refresh_if_stale(self) -> None:
        if not self._state.is_stale(self._ttl_s):
            return
        await self._fetch()

    async def _fetch(self) -> None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/javascript, */*; q=0.01",
        }
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.get(_API_URL, headers=headers) as resp:
                    if resp.status == 429:
                        logger.debug("[StockTwits] Rate limited — skipping")
                        self._state = StockTwitsState(
                            fetched_at=time.monotonic(), available=False
                        )
                        return
                    resp.raise_for_status()
                    data = await resp.json()

            messages = data.get("messages", [])
            bull = sum(
                1 for m in messages
                if (m.get("entities", {}).get("sentiment") or {}).get("basic") == "Bullish"
            )
            bear = sum(
                1 for m in messages
                if (m.get("entities", {}).get("sentiment") or {}).get("basic") == "Bearish"
            )
            total = bull + bear
            if total == 0:
                score = 0.0
                bull_pct = bear_pct = 0.0
            else:
                bull_pct = bull / total * 100.0
                bear_pct = bear / total * 100.0
                # Normalise to -1..+1
                score = (bull - bear) / total

            self._state = StockTwitsState(
                score=score,
                bullish_pct=bull_pct,
                bearish_pct=bear_pct,
                message_count=len(messages),
                fetched_at=time.monotonic(),
                available=True,
            )
            logger.debug(
                f"[StockTwits] {len(messages)} msgs, bull={bull_pct:.0f}% bear={bear_pct:.0f}%"
            )
        except Exception as exc:
            logger.warning(f"[StockTwits] Fetch failed: {exc}")
            self._state = StockTwitsState(
                fetched_at=time.monotonic(), available=False
            )

    @property
    def state(self) -> StockTwitsState:
        return self._state
