"""Interactive Brokers data access via ib_insync."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, AsyncIterator

import pandas as pd
from ib_insync import IB, Contract, Future

from ..utils.logger import logger
from .base import DataCollector


class IBKRCollector(DataCollector):
    """Handles historical and streaming data collection from IBKR with reconnection logic."""

    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        symbol: str,
        exchange: str = "GLOBEX",
        currency: str = "USD",
        bar_size: str = "1 min",
        duration: str = "1 D",
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.symbol = symbol
        self.exchange = exchange
        self.currency = currency
        self.bar_size = bar_size
        self.duration = duration
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.ib = IB()
        self.connection_attempts = 0

    def connect(self) -> None:
        """Synchronous connection method for compatibility."""
        if not self.ib.isConnected():
            try:
                logger.info("Connecting to IBKR at %s:%s", self.host, self.port)
                self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=20)
                logger.info("Successfully connected to IBKR")
            except Exception as e:
                logger.error("Failed to connect to IBKR: %s", e)
                raise
    
    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    async def _connect(self) -> None:
        """Connect to IBKR with exponential backoff retry logic."""
        if self.ib.isConnected():
            return
        
        for attempt in range(self.max_retries):
            try:
                logger.info("Connecting to IBKR at %s:%s (attempt %d/%d)", 
                           self.host, self.port, attempt + 1, self.max_retries)
                await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
                self.connection_attempts = 0
                logger.info("Successfully connected to IBKR")
                return
            except Exception as e:
                self.connection_attempts += 1
                delay = self.base_delay * (2 ** attempt)
                logger.warning("IBKR connection failed (attempt %d/%d): %s. Retrying in %.1fs...", 
                             attempt + 1, self.max_retries, e, delay)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    logger.error("Failed to connect to IBKR after %d attempts", self.max_retries)
                    raise

    def _contract(self) -> Contract:
        return Future(symbol=self.symbol, exchange=self.exchange, currency=self.currency)

    async def collect(self) -> pd.DataFrame:
        """Collect historical data with automatic reconnection on failure."""
        await self._connect()
        contract = self._contract()
        
        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=self.duration,
                barSizeSetting=self.bar_size,
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
            )
            df = pd.DataFrame(bars, columns=["date", "open", "high", "low", "close", "volume", "barCount", "average"])
            df["timestamp"] = pd.to_datetime(df["date"], utc=True)
            df.set_index("timestamp", inplace=True)
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error("Failed to collect IBKR data: %s. Reconnecting...", e)
            self.ib.disconnect()
            await self._connect()
            raise

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Stream real-time bars with automatic reconnection and error handling."""
        await self._connect()
        contract = self._contract()
        data_queue = asyncio.Queue()
        sub = None

        def on_bar_update(bid_ask: Any, has_new_bar: bool) -> None:
            bar = bid_ask
            payload = {
                "timestamp": pd.Timestamp(bar.time, tz="UTC").to_pydatetime(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "source": "ibkr",
            }
            data_queue.put_nowait(payload)

        try:
            sub = self.ib.reqRealTimeBars(contract, barSize=5, whatToShow="TRADES", useRTH=False)
            sub.updateEvent += on_bar_update

            while True:
                try:
                    item = await asyncio.wait_for(data_queue.get(), timeout=5.0)
                    yield item
                except asyncio.TimeoutError:
                    # Check connection and reconnect if needed
                    if not self.ib.isConnected():
                        logger.warning("IBKR connection lost. Reconnecting...")
                        if sub:
                            try:
                                sub.updateEvent -= on_bar_update
                                self.ib.cancelRealTimeBars(sub)
                            except:
                                pass
                        await self._connect()
                        sub = self.ib.reqRealTimeBars(contract, barSize=5, whatToShow="TRADES", useRTH=False)
                        sub.updateEvent += on_bar_update
                    else:
                        await self.ib.waitOnUpdate(timeout=0.2)
                except Exception as e:
                    logger.error("Error in IBKR stream: %s. Attempting reconnection...", e)
                    if sub:
                        try:
                            sub.updateEvent -= on_bar_update
                            self.ib.cancelRealTimeBars(sub)
                        except:
                            pass
                    self.ib.disconnect()
                    await asyncio.sleep(self.base_delay)
                    await self._connect()
                    sub = self.ib.reqRealTimeBars(contract, barSize=5, whatToShow="TRADES", useRTH=False)
                    sub.updateEvent += on_bar_update
        finally:
            if sub:
                try:
                    sub.updateEvent -= on_bar_update
                    self.ib.cancelRealTimeBars(sub)
                except:
                    pass
            if self.ib.isConnected():
                self.ib.disconnect()
