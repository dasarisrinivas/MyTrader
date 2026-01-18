"""
Interactive Brokers Historical Data Downloader
==============================================

Downloads historical bars from IBKR with:
- Pacing compliance (max 60 requests per 10 minutes)
- Chunking for long date ranges (IB limitations)
- Exponential backoff retry logic
- Local Parquet caching to avoid re-downloads
- Support for 1m and 5m bars
- Fallback to SPY/VIX proxies when IB data unavailable
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib

import pandas as pd
import numpy as np
from loguru import logger

try:
    from ib_insync import IB, Contract, Future, ContFuture, Stock, Index
    HAS_IB_INSYNC = True
except ImportError:
    HAS_IB_INSYNC = False
    logger.warning("ib_insync not installed. IB downloads will be unavailable.")


@dataclass
class DownloadConfig:
    """Configuration for historical data download."""
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 99  # Dedicated client ID for downloads
    
    # Pacing settings (IB limits: max 60 requests per 10 minutes)
    requests_per_window: int = 50  # Conservative buffer
    window_seconds: int = 600
    request_delay_seconds: float = 2.0  # Min delay between requests
    
    # Retry settings
    max_retries: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0
    
    # Data paths
    cache_dir: Path = field(default_factory=lambda: Path("data/raw"))
    
    # Bar size durations (IB max durations per bar size)
    # For 1 min bars: max 1 day per request, so chunk by days
    # For 5 min bars: max 1 week per request
    chunk_durations: Dict[str, str] = field(default_factory=lambda: {
        "1 min": "1 D",
        "5 mins": "1 W",
        "15 mins": "2 W",
        "1 hour": "1 M",
        "1 day": "1 Y",
    })


class IBHistoricalDownloader:
    """
    Downloads historical data from Interactive Brokers.
    
    Features:
    - Respects IB pacing limits (60 requests per 10 min window)
    - Chunks large date ranges into manageable pieces
    - Caches downloaded data in Parquet format
    - Handles reconnection and retries gracefully
    - Supports ES/MES futures, VIX/VX, and equity proxies (SPY)
    """
    
    def __init__(self, config: Optional[DownloadConfig] = None):
        if not HAS_IB_INSYNC:
            raise ImportError("ib_insync is required for IB downloads. Install with: pip install ib_insync")
        
        self.config = config or DownloadConfig()
        self.ib: Optional[IB] = None
        self._request_timestamps: List[float] = []
        self._qualified_contracts: Dict[str, Contract] = {}
        
        # Ensure cache directory exists
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(
        self, 
        symbol: str, 
        contract_id: str, 
        bar_size: str, 
        start: datetime, 
        end: datetime
    ) -> Path:
        """Generate cache file path for a data chunk."""
        bar_size_clean = bar_size.replace(" ", "_")
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        
        cache_path = (
            self.config.cache_dir 
            / symbol 
            / contract_id 
            / bar_size_clean 
            / f"{start_str}_{end_str}.parquet"
        )
        return cache_path
    
    def _check_cache(
        self, 
        symbol: str, 
        contract_id: str, 
        bar_size: str, 
        start: datetime, 
        end: datetime
    ) -> Optional[pd.DataFrame]:
        """Check if data is already cached."""
        cache_path = self._get_cache_path(symbol, contract_id, bar_size, start, end)
        
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                logger.info(f"Cache hit: {cache_path} ({len(df)} bars)")
                return df
            except Exception as e:
                logger.warning(f"Cache read failed: {e}. Will re-download.")
        
        return None
    
    def _save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        contract_id: str,
        bar_size: str,
        start: datetime,
        end: datetime
    ) -> None:
        """Save data to cache (parquet with CSV fallback)."""
        if df.empty:
            return
            
        cache_path = self._get_cache_path(symbol, contract_id, bar_size, start, end)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df.to_parquet(cache_path, engine="pyarrow", compression="snappy")
            logger.info(f"Cached: {cache_path} ({len(df)} bars)")
        except ImportError:
            # Fallback to CSV if pyarrow not installed
            csv_path = cache_path.with_suffix(".csv")
            df.to_csv(csv_path)
            logger.info(f"Cached (CSV fallback): {csv_path} ({len(df)} bars)")
    
    async def _wait_for_pacing(self) -> None:
        """Wait if necessary to comply with IB pacing limits."""
        now = time.time()
        window_start = now - self.config.window_seconds
        
        # Remove old timestamps
        self._request_timestamps = [
            ts for ts in self._request_timestamps if ts > window_start
        ]
        
        # Check if we're at limit
        if len(self._request_timestamps) >= self.config.requests_per_window:
            oldest = min(self._request_timestamps)
            wait_time = oldest + self.config.window_seconds - now + 1
            if wait_time > 0:
                logger.warning(f"Pacing limit reached. Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
        
        # Minimum delay between requests
        await asyncio.sleep(self.config.request_delay_seconds)
        
        # Record this request
        self._request_timestamps.append(time.time())
    
    async def connect(self) -> None:
        """Connect to IB Gateway/TWS."""
        if self.ib and self.ib.isConnected():
            return
        
        self.ib = IB()
        
        for attempt in range(self.config.max_retries):
            try:
                logger.info(
                    f"Connecting to IB at {self.config.host}:{self.config.port} "
                    f"(client_id={self.config.client_id}, attempt {attempt + 1}/{self.config.max_retries})"
                )
                await self.ib.connectAsync(
                    self.config.host, 
                    self.config.port, 
                    clientId=self.config.client_id,
                    timeout=30
                )
                
                # Request delayed data (works without market data subscription)
                self.ib.reqMarketDataType(3)  # 3 = Delayed
                
                logger.info("✅ Connected to IB Gateway")
                return
                
            except Exception as e:
                delay = min(
                    self.config.base_delay * (2 ** attempt), 
                    self.config.max_delay
                )
                logger.warning(
                    f"Connection failed: {e}. Retrying in {delay:.1f}s..."
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    raise ConnectionError(f"Failed to connect to IB after {self.config.max_retries} attempts")
    
    def disconnect(self) -> None:
        """Disconnect from IB."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB")
    
    async def _qualify_contract(self, contract: Contract) -> Contract:
        """Qualify a contract with IB to get full details."""
        key = f"{contract.symbol}_{contract.secType}_{contract.exchange}"
        
        if key in self._qualified_contracts:
            return self._qualified_contracts[key]
        
        qualified = await self.ib.qualifyContractsAsync(contract)
        
        if not qualified:
            # Try contract details for futures
            if contract.secType == "FUT":
                details = await self.ib.reqContractDetailsAsync(contract)
                if details:
                    # Use front month
                    qualified_contract = details[0].contract
                    self._qualified_contracts[key] = qualified_contract
                    logger.info(f"Qualified contract: {qualified_contract.localSymbol}")
                    return qualified_contract
            
            raise ValueError(f"Could not qualify contract: {contract}")
        
        self._qualified_contracts[key] = qualified[0]
        return qualified[0]
    
    def _create_contract(
        self, 
        symbol: str, 
        sec_type: str = "FUT",
        exchange: str = "CME",
        currency: str = "USD",
        expiry: Optional[str] = None
    ) -> Contract:
        """Create an IB contract object."""
        if sec_type == "FUT":
            contract = Future(
                symbol=symbol,
                exchange=exchange,
                currency=currency
            )
            if expiry:
                contract.lastTradeDateOrContractMonth = expiry
        elif sec_type == "STK":
            contract = Stock(symbol=symbol, exchange=exchange, currency=currency)
        elif sec_type == "IND":
            contract = Index(symbol=symbol, exchange=exchange, currency=currency)
        else:
            raise ValueError(f"Unknown security type: {sec_type}")
        
        return contract
    
    def _generate_date_chunks(
        self,
        start: datetime,
        end: datetime,
        bar_size: str
    ) -> List[Tuple[datetime, datetime]]:
        """Generate date chunks based on IB duration limits for bar size."""
        chunks = []
        
        # Determine chunk size based on bar size
        if "min" in bar_size:
            # For minute bars, chunk by day
            chunk_delta = timedelta(days=1)
        elif "hour" in bar_size:
            # For hourly bars, chunk by week
            chunk_delta = timedelta(weeks=1)
        else:
            # For daily bars, chunk by month
            chunk_delta = timedelta(days=30)
        
        current = start
        while current < end:
            chunk_end = min(current + chunk_delta, end)
            chunks.append((current, chunk_end))
            current = chunk_end
        
        return chunks
    
    async def _download_chunk(
        self,
        contract: Contract,
        end_datetime: datetime,
        duration: str,
        bar_size: str,
        what_to_show: str = "TRADES"
    ) -> pd.DataFrame:
        """Download a single chunk of historical data."""
        await self._wait_for_pacing()
        
        for attempt in range(self.config.max_retries):
            try:
                # Format end datetime for IB
                end_str = end_datetime.strftime("%Y%m%d %H:%M:%S")
                
                bars = await self.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_str,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=False,  # Include extended hours
                    formatDate=1,
                )
                
                if not bars:
                    logger.warning(f"No data returned for {contract.localSymbol} ending {end_str}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        "timestamp": pd.Timestamp(bar.date).tz_localize("UTC") if bar.date.tzinfo is None else pd.Timestamp(bar.date).tz_convert("UTC"),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                        "vwap": float(bar.average) if hasattr(bar, "average") else np.nan,
                        "trade_count": int(bar.barCount) if hasattr(bar, "barCount") else 0,
                    }
                    for bar in bars
                ])
                
                df.set_index("timestamp", inplace=True)
                return df
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle pacing violation
                if "pacing" in error_str or "too many" in error_str:
                    wait_time = 60  # Wait a full minute on pacing error
                    logger.warning(f"Pacing violation! Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                # Handle other errors with backoff
                delay = min(
                    self.config.base_delay * (2 ** attempt),
                    self.config.max_delay
                )
                logger.warning(
                    f"Download error (attempt {attempt + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    raise
        
        return pd.DataFrame()
    
    async def download(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_size: str = "1 min",
        sec_type: str = "FUT",
        exchange: str = "CME",
        currency: str = "USD",
        expiry: Optional[str] = None,
        what_to_show: str = "TRADES",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download historical data for a symbol over a date range.
        
        Args:
            symbol: Instrument symbol (e.g., "MES", "ES", "SPY")
            start: Start datetime (UTC)
            end: End datetime (UTC)
            bar_size: Bar size (e.g., "1 min", "5 mins")
            sec_type: Security type ("FUT", "STK", "IND")
            exchange: Exchange (e.g., "CME", "SMART")
            currency: Currency (e.g., "USD")
            expiry: Optional contract expiry (e.g., "202403")
            what_to_show: Data type ("TRADES", "MIDPOINT", etc.)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp (UTC)
        """
        await self.connect()
        
        # Create and qualify contract
        contract = self._create_contract(symbol, sec_type, exchange, currency, expiry)
        qualified = await self._qualify_contract(contract)
        contract_id = qualified.localSymbol or f"{symbol}_{expiry or 'FRONT'}"
        
        logger.info(
            f"Downloading {symbol} ({contract_id}) from {start} to {end} "
            f"[bar_size={bar_size}]"
        )
        
        # Check cache first
        if use_cache:
            cached = self._check_cache(symbol, contract_id, bar_size, start, end)
            if cached is not None:
                return cached
        
        # Generate date chunks
        chunks = self._generate_date_chunks(start, end, bar_size)
        logger.info(f"Downloading {len(chunks)} chunks...")
        
        # Determine duration string for IB
        duration = self.config.chunk_durations.get(bar_size, "1 D")
        
        all_data: List[pd.DataFrame] = []
        
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            logger.info(f"Chunk {i + 1}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()}")
            
            try:
                df = await self._download_chunk(
                    qualified,
                    chunk_end,
                    duration,
                    bar_size,
                    what_to_show
                )
                
                if not df.empty:
                    # Filter to requested range
                    df = df[(df.index >= start) & (df.index <= end)]
                    all_data.append(df)
                    
            except Exception as e:
                logger.error(f"Failed to download chunk {chunk_start} to {chunk_end}: {e}")
                # Continue with other chunks
        
        if not all_data:
            logger.warning(f"No data downloaded for {symbol}")
            return pd.DataFrame()
        
        # Combine all chunks
        combined = pd.concat(all_data, axis=0)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined.sort_index(inplace=True)
        
        # Cache the result
        if use_cache and not combined.empty:
            self._save_to_cache(combined, symbol, contract_id, bar_size, start, end)
        
        logger.info(f"Downloaded {len(combined)} bars for {symbol}")
        return combined
    
    async def download_multiple_contracts(
        self,
        symbol: str,
        contracts: List[str],  # List of expiry dates like ["202403", "202406"]
        start: datetime,
        end: datetime,
        bar_size: str = "1 min",
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple contract months.
        
        Useful for building continuous futures series.
        """
        results = {}
        
        for expiry in contracts:
            logger.info(f"Downloading {symbol} {expiry}...")
            
            try:
                df = await self.download(
                    symbol=symbol,
                    start=start,
                    end=end,
                    bar_size=bar_size,
                    expiry=expiry,
                    **kwargs
                )
                results[expiry] = df
                
            except Exception as e:
                logger.error(f"Failed to download {symbol} {expiry}: {e}")
        
        return results

    def _generate_quarterly_expiries(
        self,
        start: datetime,
        end: datetime
    ) -> List[str]:
        """
        Generate list of quarterly futures expiry codes (H, M, U, Z) for date range.
        
        ES/MES futures expire on the 3rd Friday of H (Mar), M (Jun), U (Sep), Z (Dec).
        """
        expiries = []
        
        # Quarterly month codes
        quarters = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}
        
        # Start from the quarter before start date to ensure coverage
        year = start.year
        if start.month <= 3:
            year -= 1
            month = 12
        elif start.month <= 6:
            month = 3
        elif start.month <= 9:
            month = 6
        elif start.month <= 12:
            month = 9
        else:
            month = 12
        
        # Generate expiries until we're past end date
        while True:
            # Move to next quarter
            month += 3
            if month > 12:
                month = 3
                year += 1
            
            expiry_str = f"{year}{month:02d}"
            expiries.append(expiry_str)
            
            # Stop when we're past the end date
            if year > end.year + 1:
                break
            if year == end.year + 1 and month > 3:
                break
        
        return expiries

    async def download_continuous(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_size: str = "1 min",
        exchange: str = "CME",
        currency: str = "USD",
        what_to_show: str = "TRADES",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download continuous futures data from IB.
        
        Strategy: Use the front-month Future contract (not ContFuture) since
        ContFuture doesn't support endDateTime for historical chunking.
        
        For recent data, we can download as much as the contract allows.
        IB keeps ~2 years of 1-day data, ~1 year of 1-hour data, and
        ~60 days of 1-minute data for active contracts.
        
        IB Data Limits per request:
        - 1-min bars: max 7 days per request
        - 5-min bars: max 30 days per request  
        - 1-hour bars: max 1 year per request
        - 1-day bars: max 2+ years per request
        
        Args:
            symbol: Futures symbol (e.g., "MES", "ES")
            start: Start datetime (UTC)
            end: End datetime (UTC)
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour", "1 day")
            
        Returns:
            DataFrame with futures data
        """
        await self.connect()
        
        # Get the front-month contract by requesting details
        front_month = Future(symbol=symbol, exchange=exchange, currency=currency)
        
        try:
            # Request contract details to get all available contracts
            details = await self.ib.reqContractDetailsAsync(front_month)
            if not details:
                raise ValueError(f"No contracts found for {symbol}")
            
            # Sort by expiry and pick the front-month (earliest expiry)
            sorted_details = sorted(details, key=lambda x: x.contract.lastTradeDateOrContractMonth)
            front_month = sorted_details[0].contract
            
            # Qualify it to ensure we have full details
            qualified_list = await self.ib.qualifyContractsAsync(front_month)
            if qualified_list:
                front_month = qualified_list[0]
            
            logger.info(f"Using Future: {front_month.localSymbol} (conId={front_month.conId})")
        except Exception as e:
            logger.error(f"Failed to qualify Future {symbol}: {e}")
            raise
        
        cache_key = f"FUT_{front_month.localSymbol}"
        
        # Check cache
        if use_cache:
            cached = self._check_cache(symbol, cache_key, bar_size, start, end)
            if cached is not None:
                return cached
        
        # Determine chunk size based on bar size (IB limits)
        bar_size_lower = bar_size.lower()
        if "1 min" in bar_size_lower or bar_size_lower == "1m":
            chunk_days = 6  # Conservative: 6 days to stay under 7-day limit
        elif "5 min" in bar_size_lower or bar_size_lower == "5m":
            chunk_days = 25
        elif "15 min" in bar_size_lower or bar_size_lower == "15m":
            chunk_days = 50
        elif "hour" in bar_size_lower or bar_size_lower == "1h":
            chunk_days = 300
        elif "day" in bar_size_lower or bar_size_lower == "1d":
            chunk_days = 700
        else:
            chunk_days = 6
        
        # Generate chunks working backwards from end to start
        chunks: List[Tuple[datetime, datetime]] = []
        current_end = end
        
        while current_end > start:
            chunk_start = current_end - timedelta(days=chunk_days)
            if chunk_start < start:
                chunk_start = start
            
            chunks.append((chunk_start, current_end))
            current_end = chunk_start
        
        # Reverse so we download chronologically
        chunks.reverse()
        
        logger.info(f"Downloading {symbol} ({front_month.localSymbol}): {len(chunks)} chunks from {start.date()} to {end.date()}")
        
        all_data: List[pd.DataFrame] = []
        
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            actual_days = (chunk_end - chunk_start).days
            if actual_days < 1:
                actual_days = 1
            duration_str = f"{actual_days} D"
            
            logger.info(f"Chunk {i + 1}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()} (duration={duration_str})")
            
            await self._wait_for_pacing()
            
            for attempt in range(self.config.max_retries):
                try:
                    end_str = chunk_end.strftime("%Y%m%d %H:%M:%S")
                    
                    bars = await self.ib.reqHistoricalDataAsync(
                        front_month,
                        endDateTime=end_str,
                        durationStr=duration_str,
                        barSizeSetting=bar_size,
                        whatToShow=what_to_show,
                        useRTH=False,
                        formatDate=1,
                        timeout=120
                    )
                    
                    if bars:
                        df = pd.DataFrame([
                            {
                                "timestamp": pd.Timestamp(bar.date).tz_localize("UTC") if bar.date.tzinfo is None else pd.Timestamp(bar.date).tz_convert("UTC"),
                                "open": float(bar.open),
                                "high": float(bar.high),
                                "low": float(bar.low),
                                "close": float(bar.close),
                                "volume": int(bar.volume),
                            }
                            for bar in bars
                        ])
                        df.set_index("timestamp", inplace=True)
                        
                        # Filter to chunk range
                        df = df[(df.index >= chunk_start) & (df.index <= chunk_end)]
                        
                        if not df.empty:
                            all_data.append(df)
                            logger.info(f"  Got {len(df)} bars")
                    else:
                        logger.warning(f"  No data for chunk {chunk_start.date()} to {chunk_end.date()}")
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e).lower()
                    if "timeout" in error_str or "pacing" in error_str:
                        wait_time = 30 if "timeout" in error_str else 60
                        logger.warning(f"  {e}. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        delay = self.config.base_delay * (2 ** attempt)
                        logger.warning(f"  Error: {e}. Retry in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                    
                    if attempt == self.config.max_retries - 1:
                        logger.error(f"  Failed chunk after {self.config.max_retries} attempts")
        
        if not all_data:
            logger.warning(f"No data downloaded for {symbol}")
            return pd.DataFrame()
        
        # Combine all chunks
        combined = pd.concat(all_data, axis=0)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined.sort_index(inplace=True)
        
        # Filter to requested range
        combined = combined[(combined.index >= start) & (combined.index <= end)]
        
        # Cache the result
        if use_cache and not combined.empty:
            self._save_to_cache(combined, symbol, cache_key, bar_size, start, end)
        
        logger.info(f"✅ Downloaded {len(combined)} bars for {symbol} ({front_month.localSymbol})")
        return combined


class FallbackDataDownloader:
    """
    Fallback data downloader when IB data is unavailable.
    
    Supports:
    - SPY as ES/MES proxy (clearly labeled)
    - VIX index data
    - External data providers (Databento/Polygon) if configured
    """
    
    def __init__(self, cache_dir: Path = Path("data/raw/fallback")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for external provider configuration
        self.databento_api_key = os.environ.get("DATABENTO_API_KEY")
        self.polygon_api_key = os.environ.get("POLYGON_API_KEY")
    
    def download_spy_proxy(
        self,
        start: datetime,
        end: datetime,
        bar_size: str = "1 min"
    ) -> Tuple[pd.DataFrame, str]:
        """
        Download SPY data as an ES proxy.
        
        Returns:
            Tuple of (DataFrame, mode_label) where mode_label is "proxy_spy"
        
        Note:
            yfinance has limitations on intraday data (max 8 days for 1m).
            For longer periods, we automatically fall back to daily data
            and use it directly for the backtest.
        """
        try:
            import yfinance as yf
            
            # Determine interval for yfinance
            interval_map = {
                "1 min": "1m",
                "5 mins": "5m",
                "15 mins": "15m",
                "1 hour": "1h",
                "1 day": "1d",
            }
            interval = interval_map.get(bar_size, "1m")
            
            logger.warning(
                f"⚠️  PROXY MODE: Downloading SPY as ES proxy. "
                f"Results will be labeled 'proxy_spy'."
            )
            
            # Check date range - yfinance limits 1m data to 8 days
            date_range_days = (end - start).days
            use_daily_fallback = False
            
            if interval in ("1m", "5m") and date_range_days > 7:
                logger.warning(
                    f"⚠️  Date range ({date_range_days} days) exceeds yfinance limit for {interval} data. "
                    f"Falling back to daily data for proxy backtest."
                )
                interval = "1d"
                use_daily_fallback = True
            
            ticker = yf.Ticker("SPY")
            df = ticker.history(
                start=start,
                end=end,
                interval=interval,
                prepost=True  # Include pre/post market
            )
            
            if df.empty:
                raise ValueError("No SPY data returned from yfinance")
            
            # Normalize column names
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"adj close": "adj_close"})
            
            # Ensure UTC timezone
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("America/New_York").tz_convert("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            
            df.index.name = "timestamp"
            
            # Scale SPY prices to approximate ES point values (SPY ~1/10 of ES)
            # ES ~ SPY * 10
            scale_factor = 10.0
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col] * scale_factor
            
            # Log data info
            logger.info(
                f"SPY proxy data: {len(df)} bars from {df.index[0]} to {df.index[-1]} "
                f"({'daily' if use_daily_fallback else interval} interval)"
            )
            
            # Cache result (optional - continue if parquet not available)
            try:
                cache_path = self.cache_dir / f"SPY_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{interval}.parquet"
                df.to_parquet(cache_path)
            except ImportError:
                # pyarrow/fastparquet not installed, skip caching
                logger.debug("Parquet caching skipped (pyarrow not installed)")
            
            mode_label = "proxy_spy_daily" if use_daily_fallback else "proxy_spy"
            return df[["open", "high", "low", "close", "volume"]], mode_label
            
        except ImportError as e:
            if "yfinance" in str(e):
                raise ImportError("yfinance is required for SPY proxy. Install with: pip install yfinance")
            raise
    
    def download_vix(
        self,
        start: datetime,
        end: datetime,
        bar_size: str = "1 day"
    ) -> pd.DataFrame:
        """Download VIX index data."""
        try:
            import yfinance as yf
            
            interval_map = {"1 day": "1d", "1 hour": "1h"}
            interval = interval_map.get(bar_size, "1d")
            
            ticker = yf.Ticker("^VIX")
            df = ticker.history(start=start, end=end, interval=interval)
            
            if df.empty:
                logger.warning("No VIX data available")
                return pd.DataFrame()
            
            df.columns = [c.lower() for c in df.columns]
            
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("America/New_York").tz_convert("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            
            df.index.name = "timestamp"
            return df[["open", "high", "low", "close", "volume"]]
            
        except Exception as e:
            logger.warning(f"VIX download failed: {e}")
            return pd.DataFrame()


class DatabentoDownloader:
    """
    Download historical futures data from Databento.
    
    Databento provides high-quality continuous futures data with:
    - Proper roll adjustment (back-adjusted or raw)
    - Full historical coverage going back years
    - 1-minute and higher resolution bars
    
    Requires: DATABENTO_API_KEY environment variable
    """
    
    # Databento symbol mapping for CME micro e-mini futures
    SYMBOL_MAP = {
        "MES": "MES.c.0",    # Front-month continuous MES
        "MNQ": "MNQ.c.0",    # Front-month continuous MNQ
        "ES": "ES.c.0",      # Front-month continuous ES
        "NQ": "NQ.c.0",      # Front-month continuous NQ
        "MYM": "MYM.c.0",    # Front-month continuous Mini Dow
        "M2K": "M2K.c.0",    # Front-month continuous Mini Russell
    }
    
    def __init__(self, cache_dir: Path = Path("data/raw/databento")):
        self.api_key = os.environ.get("DATABENTO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DATABENTO_API_KEY environment variable required. "
                "Get your API key at https://databento.com"
            )
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import databento as db
            self.db = db
            self.client = db.Historical(self.api_key)
        except ImportError:
            raise ImportError(
                "databento package required. Install with: pip install databento"
            )
    
    def download(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_size: str = "1 min"
    ) -> pd.DataFrame:
        """
        Download continuous futures data from Databento.
        
        Args:
            symbol: Futures symbol (MES, ES, NQ, etc.)
            start: Start datetime (UTC)
            end: End datetime (UTC)
            bar_size: Bar size ("1 min", "5 mins", "1 hour", "1 day")
            
        Returns:
            DataFrame with OHLCV data, UTC timezone
        """
        # Check cache first
        cache_key = f"{symbol}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{bar_size.replace(' ', '_')}"
        cache_path = self.cache_dir / f"{cache_key}.csv"
        
        if cache_path.exists():
            logger.info(f"Loading cached Databento data from {cache_path}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            return df
        
        # Map symbol to Databento format
        db_symbol = self.SYMBOL_MAP.get(symbol.upper())
        if not db_symbol:
            # Try direct symbol if not in map
            db_symbol = f"{symbol.upper()}.c.0"
            logger.warning(f"Symbol {symbol} not in map, trying {db_symbol}")
        
        # Map bar size to Databento schema
        schema_map = {
            "1 min": "ohlcv-1m",
            "5 mins": "ohlcv-5m",
            "15 mins": "ohlcv-15m",
            "1 hour": "ohlcv-1h",
            "1 day": "ohlcv-1d",
        }
        schema = schema_map.get(bar_size, "ohlcv-1m")
        
        logger.info(
            f"Downloading {symbol} from Databento: "
            f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} "
            f"({schema})"
        )
        
        try:
            # Download from Databento
            # Using GLBX.MDP3 dataset for CME futures
            data = self.client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=[db_symbol],
                schema=schema,
                start=start.strftime("%Y-%m-%dT%H:%M:%S"),
                end=end.strftime("%Y-%m-%dT%H:%M:%S"),
                stype_in="continuous",
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            if df.empty:
                logger.warning(f"No data returned from Databento for {symbol}")
                return pd.DataFrame()
            
            # Normalize column names to match our format
            # Databento returns: open, high, low, close, volume
            df = df.rename(columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            })
            
            # Databento prices are in instrument units - for MES/ES this is already correct
            # No scaling needed
            
            # Ensure UTC timezone
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            
            df.index.name = "timestamp"
            
            # Keep only OHLCV columns
            result = df[["open", "high", "low", "close", "volume"]].copy()
            
            # Cache the result
            result.to_csv(cache_path)
            logger.info(f"Cached {len(result)} bars to {cache_path}")
            
            logger.info(
                f"✅ Downloaded {len(result)} bars from Databento: "
                f"{result.index[0]} to {result.index[-1]}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Databento download failed: {e}")
            raise
    
    def get_cost_estimate(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_size: str = "1 min"
    ) -> dict:
        """
        Get cost estimate for downloading data.
        
        Returns dict with estimated cost and data size.
        """
        db_symbol = self.SYMBOL_MAP.get(symbol.upper(), f"{symbol.upper()}.c.0")
        schema_map = {
            "1 min": "ohlcv-1m",
            "5 mins": "ohlcv-5m",
            "1 hour": "ohlcv-1h",
            "1 day": "ohlcv-1d",
        }
        schema = schema_map.get(bar_size, "ohlcv-1m")
        
        try:
            cost = self.client.metadata.get_cost(
                dataset="GLBX.MDP3",
                symbols=[db_symbol],
                schema=schema,
                start=start.strftime("%Y-%m-%dT%H:%M:%S"),
                end=end.strftime("%Y-%m-%dT%H:%M:%S"),
                stype_in="continuous",
            )
            return {
                "cost_usd": cost,
                "symbol": db_symbol,
                "schema": schema,
                "start": start,
                "end": end
            }
        except Exception as e:
            logger.warning(f"Could not get cost estimate: {e}")
            return {"error": str(e)}


class PolygonDownloader:
    """
    Download historical futures data from Polygon.io.
    
    Polygon provides continuous futures data with:
    - Good historical coverage
    - 1-minute resolution
    - Affordable pricing
    
    Requires: POLYGON_API_KEY environment variable
    """
    
    SYMBOL_MAP = {
        "MES": "I:MES1!",   # Front-month continuous MES
        "ES": "I:ES1!",     # Front-month continuous ES
        "NQ": "I:NQ1!",     # Front-month continuous NQ
        "MNQ": "I:MNQ1!",   # Front-month continuous MNQ
    }
    
    def __init__(self, cache_dir: Path = Path("data/raw/polygon")):
        self.api_key = os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError(
                "POLYGON_API_KEY environment variable required. "
                "Get your API key at https://polygon.io"
            )
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://api.polygon.io"
    
    def download(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_size: str = "1 min"
    ) -> pd.DataFrame:
        """
        Download continuous futures data from Polygon.
        
        Args:
            symbol: Futures symbol (MES, ES, NQ, etc.)
            start: Start datetime (UTC)
            end: End datetime (UTC)
            bar_size: Bar size ("1 min", "5 mins", "1 hour", "1 day")
            
        Returns:
            DataFrame with OHLCV data, UTC timezone
        """
        import requests
        
        # Check cache first
        cache_key = f"{symbol}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{bar_size.replace(' ', '_')}"
        cache_path = self.cache_dir / f"{cache_key}.csv"
        
        if cache_path.exists():
            logger.info(f"Loading cached Polygon data from {cache_path}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            return df
        
        # Map symbol to Polygon format
        poly_symbol = self.SYMBOL_MAP.get(symbol.upper())
        if not poly_symbol:
            poly_symbol = f"I:{symbol.upper()}1!"
            logger.warning(f"Symbol {symbol} not in map, trying {poly_symbol}")
        
        # Map bar size to Polygon params
        timespan_map = {
            "1 min": ("minute", 1),
            "5 mins": ("minute", 5),
            "15 mins": ("minute", 15),
            "1 hour": ("hour", 1),
            "1 day": ("day", 1),
        }
        timespan, multiplier = timespan_map.get(bar_size, ("minute", 1))
        
        logger.info(
            f"Downloading {symbol} from Polygon: "
            f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        )
        
        all_data = []
        current_start = start
        
        # Polygon has pagination limits, download in chunks
        while current_start < end:
            url = (
                f"{self.base_url}/v2/aggs/ticker/{poly_symbol}/range/"
                f"{multiplier}/{timespan}/"
                f"{int(current_start.timestamp() * 1000)}/"
                f"{int(end.timestamp() * 1000)}"
            )
            
            params = {
                "apiKey": self.api_key,
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "OK" or "results" not in data:
                logger.warning(f"Polygon API error: {data}")
                break
            
            results = data["results"]
            if not results:
                break
            
            all_data.extend(results)
            
            # Update start for next iteration
            last_ts = results[-1]["t"]
            current_start = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            current_start += timedelta(minutes=1)
            
            if len(results) < 50000:
                break  # No more data
        
        if not all_data:
            logger.warning(f"No data returned from Polygon for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        
        df = df.rename(columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume"
        })
        
        df.index.name = "timestamp"
        result = df[["open", "high", "low", "close", "volume"]].copy()
        
        # Cache the result
        result.to_csv(cache_path)
        logger.info(f"Cached {len(result)} bars to {cache_path}")
        
        logger.info(
            f"✅ Downloaded {len(result)} bars from Polygon: "
            f"{result.index[0]} to {result.index[-1]}"
        )
        
        return result


def get_best_available_downloader(
    symbol: str,
    start: datetime,
    end: datetime,
    bar_size: str = "1 min",
    prefer_source: str = "auto"
) -> Tuple[object, str]:
    """
    Get the best available data downloader based on configuration and data needs.
    
    Priority order (when prefer_source="auto"):
    1. Databento (best quality for futures, requires API key)
    2. Polygon (good quality, requires API key)
    3. IB (if connected and contracts available)
    4. SPY proxy via yfinance (fallback, limited to 8 days for 1m data)
    
    Args:
        symbol: Symbol to download
        start: Start datetime
        end: End datetime
        bar_size: Bar size
        prefer_source: "auto", "databento", "polygon", "ib", "proxy"
        
    Returns:
        Tuple of (downloader_instance, source_name)
    """
    date_range_days = (end - start).days
    
    # Check for third-party API keys
    has_databento = bool(os.environ.get("DATABENTO_API_KEY"))
    has_polygon = bool(os.environ.get("POLYGON_API_KEY"))
    
    if prefer_source == "databento" or (prefer_source == "auto" and has_databento):
        try:
            downloader = DatabentoDownloader()
            logger.info("Using Databento for historical data")
            return downloader, "databento"
        except (ImportError, ValueError) as e:
            logger.warning(f"Databento unavailable: {e}")
    
    if prefer_source == "polygon" or (prefer_source == "auto" and has_polygon):
        try:
            downloader = PolygonDownloader()
            logger.info("Using Polygon for historical data")
            return downloader, "polygon"
        except ValueError as e:
            logger.warning(f"Polygon unavailable: {e}")
    
    if prefer_source in ("ib", "auto"):
        # For short date ranges, IB might work
        if date_range_days <= 30:
            logger.info("IB may be available for short-range data")
            return None, "ib"  # Caller should use IBHistoricalDownloader
    
    # Fallback to SPY proxy
    logger.warning(
        f"No premium data source available. Using SPY proxy. "
        f"For {date_range_days}-day backtest, consider setting DATABENTO_API_KEY or POLYGON_API_KEY"
    )
    return FallbackDataDownloader(), "proxy"


async def main():
    """Test the downloader."""
    config = DownloadConfig()
    downloader = IBHistoricalDownloader(config)
    
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=5)
        
        df = await downloader.download(
            symbol="MES",
            start=start,
            end=end,
            bar_size="1 min"
        )
        
        print(f"Downloaded {len(df)} bars")
        print(df.head())
        print(df.tail())
        
    finally:
        downloader.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
