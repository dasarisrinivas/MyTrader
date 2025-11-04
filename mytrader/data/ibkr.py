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

    async def _connect(self) -> None:
        """Connect to IBKR with exponential backoff retry logic."""
        if self.ib.isConnected():
            return
        
        # Ensure clean state before connecting
        if self.ib.isConnected():
            logger.info("Disconnecting existing stale connection...")
            self.ib.disconnect()
            await asyncio.sleep(1)
        
        for attempt in range(self.max_retries):
            try:
                logger.info("Connecting to IBKR at %s:%s with client_id=%d (attempt %d/%d)", 
                           self.host, self.port, self.client_id, attempt + 1, self.max_retries)
                await self.ib.connectAsync(self.host, self.port, clientId=self.client_id, timeout=30)
                self.connection_attempts = 0
                
                # Small delay to let connection stabilize
                await asyncio.sleep(0.5)
                
                # Request delayed market data (free, no subscription needed)
                # This provides 15-minute delayed data for paper trading
                self.ib.reqMarketDataType(3)  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen
                logger.info("✅ Data collector connected successfully (client_id=%d, delayed data)", self.client_id)
                return
            except TimeoutError:
                self.connection_attempts += 1
                logger.error("❌ Connection timeout (attempt %d/%d). IB Gateway may be in bad state.", 
                           attempt + 1, self.max_retries)
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning("Retrying in %.1fs...", delay)
                    await asyncio.sleep(delay)
                else:
                    logger.error("Failed to connect after %d attempts", self.max_retries)
                    logger.error("SOLUTION: Restart IB Gateway completely:")
                    logger.error("  1. Close IB Gateway window")
                    logger.error("  2. Wait 30 seconds")  
                    logger.error("  3. Restart IB Gateway and login")
                    logger.error("  4. Run: ./restart_clean.sh")
                    logger.error("  5. Run: ./start_trading.sh")
                    raise
            except Exception as e:
                self.connection_attempts += 1
                error_msg = str(e).lower()
                
                # Check for specific Error 162 (IP address issue)
                if "162" in error_msg or "different ip address" in error_msg:
                    logger.error("❌ Error 162: TWS/Gateway session conflict detected")
                    logger.error("CAUSE: Another application or session is using this TWS login")
                    logger.error("")
                    logger.error("SOLUTIONS:")
                    logger.error("  1. RESTART IB GATEWAY:")
                    logger.error("     - File > Exit in IB Gateway")
                    logger.error("     - Wait 30 seconds")
                    logger.error("     - Restart IB Gateway and login again")
                    logger.error("     - Run: ./restart_clean.sh")
                    logger.error("")
                    logger.error("  2. CHECK IB GATEWAY API SETTINGS:")
                    logger.error("     - Edit > Global Configuration > API > Settings")
                    logger.error("     - UNCHECK 'Read-Only API'")
                    logger.error("     - Add 127.0.0.1 to Trusted IPs")
                    logger.error("     - Enable 'ActiveX and Socket Clients'")
                    logger.error("")
                    logger.error("  3. CHECK FOR OTHER TWS/GATEWAY SESSIONS:")
                    logger.error("     - Close TWS Workstation if running")
                    logger.error("     - Ensure no other apps are connected")
                    logger.error("")
                    raise
                
                delay = self.base_delay * (2 ** attempt)
                logger.warning("IBKR connection failed (attempt %d/%d): %s. Retrying in %.1fs...", 
                             attempt + 1, self.max_retries, e, delay)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    logger.error("Failed to connect to IBKR after %d attempts", self.max_retries)
                    raise

    def _contract(self) -> Contract:
        """Create contract - will be qualified with IB on first use."""
        # For futures, don't specify lastTradeDateOrContractMonth
        # Let IB qualify it to get the front month contract
        return Future(symbol=self.symbol, exchange=self.exchange, currency=self.currency)
    
    async def _qualified_contract(self) -> Contract:
        """Get IB-qualified contract (resolves to actual tradeable contract)."""
        contract = self._contract()
        qualified = await self.ib.qualifyContractsAsync(contract)
        if not qualified:
            # If qualification fails, try requesting contract details to find front month
            logger.warning("Direct qualification failed, searching for front month contract...")
            details = await self.ib.reqContractDetailsAsync(contract)
            if not details:
                raise ValueError(
                    f"Could not qualify contract: {contract}. "
                    f"Make sure the symbol '{self.symbol}' is correct and you have market data permissions."
                )
            # Get the front month (first contract in the list)
            front_month = details[0].contract
            logger.info("Using front month contract: %s", front_month)
            return front_month
        return qualified[0]

    async def collect(self) -> pd.DataFrame:
        """Collect historical data with automatic reconnection on failure."""
        await self._connect()
        contract = await self._qualified_contract()
        
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
        """Stream data by polling historical bars (paper trading compatible)."""
        await self._connect()
        contract = await self._qualified_contract()
        logger.info("Starting historical data polling for %s (paper trading mode)", contract.localSymbol)
        
        last_bar_time = None
        poll_interval = 5  # Poll every 5 seconds
        
        try:
            while True:
                try:
                    # Request last 2 bars of 5-second data
                    bars = await self.ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime="",
                        durationStr="60 S",  # Last 60 seconds
                        barSizeSetting="5 secs",
                        whatToShow="TRADES",
                        useRTH=False,
                        formatDate=1,
                    )
                    
                    if bars:
                        # Only yield new bars we haven't seen
                        for bar in bars:
                            bar_time = pd.Timestamp(bar.date).to_pydatetime()
                            if last_bar_time is None or bar_time > last_bar_time:
                                payload = {
                                    "timestamp": bar_time,
                                    "open": float(bar.open),
                                    "high": float(bar.high),
                                    "low": float(bar.low),
                                    "close": float(bar.close),
                                    "volume": float(bar.volume),
                                    "source": "ibkr",
                                }
                                last_bar_time = bar_time
                                yield payload
                    
                    await asyncio.sleep(poll_interval)
                    
                except Exception as e:
                    logger.error("Error polling historical data: %s. Reconnecting...", e)
                    if not self.ib.isConnected():
                        self.ib.disconnect()
                        await asyncio.sleep(self.base_delay)
                        await self._connect()
                        contract = await self._qualified_contract()
                    await asyncio.sleep(poll_interval)
                    
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()
