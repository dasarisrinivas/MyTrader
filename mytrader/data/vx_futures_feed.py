"""VIX Futures (VX) real-time market data feed from Interactive Brokers.

This module provides a real-time VX futures feed for use as a volatility/risk-off
factor in MES/ES trading decisions. When VIX is elevated, position sizing and
signal scores can be reduced to be more conservative.

Requirements:
- CFE Enhanced (NP,L1) market data subscription in IBKR
- IB Gateway or TWS running and logged in

Usage:
    from mytrader.data.vx_futures_feed import VxFuturesFeed
    
    vx_feed = VxFuturesFeed(host="127.0.0.1", port=7497, client_id=71)
    vx_feed.start_in_background()
    
    # Later in signal processing:
    multiplier = vx_feed.get_volatility_multiplier()
    final_score = base_score * multiplier
    
    # When shutting down:
    vx_feed.stop_background()
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from ib_insync import IB, Contract, Future, Ticker

from ..utils.logger import logger


@dataclass
class VxState:
    """Thread-safe container for VX futures state."""
    price: Optional[float] = None
    last_update: Optional[datetime] = None
    contract_symbol: Optional[str] = None
    error_count: int = 0
    
    
@dataclass
class VxConfig:
    """Configuration for VX futures feed."""
    host: str = "127.0.0.1"
    port: int = 7497  # 7497=paper, 7496=live
    client_id: int = 71
    market_data_type: int = 1  # 1=live, 3=delayed
    stale_seconds: int = 120
    conservative_on_stale: bool = False
    # Thresholds for multiplier calculation
    extreme_threshold: float = 30.0  # VX >= 30 -> 0.4 multiplier
    elevated_threshold: float = 20.0  # VX >= 20 -> 0.7 multiplier
    # Reconnection settings
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0


class VxFuturesFeed:
    """Real-time VX futures feed from Interactive Brokers.
    
    Provides volatility-based position sizing multipliers based on VIX futures levels.
    When VIX is high (fear/uncertainty), the multiplier reduces position sizes.
    
    Multiplier thresholds (configurable):
        - VX >= 30: 0.4 (extreme fear - reduce positions significantly)
        - VX >= 20: 0.7 (elevated volatility - be cautious)
        - VX < 20: 1.0 (normal conditions)
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 71,
        market_data_type: int = 1,
        stale_seconds: int = 120,
        conservative_on_stale: bool = False,
        extreme_threshold: float = 30.0,
        elevated_threshold: float = 20.0,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ):
        """Initialize VX futures feed.
        
        Args:
            host: IB Gateway host address
            port: IB Gateway port (7497=paper, 7496=live)
            client_id: Unique client ID for this connection
            market_data_type: 1=live, 3=delayed
            stale_seconds: Seconds after which data is considered stale
            conservative_on_stale: If True, return 0.5 multiplier when data is stale
            extreme_threshold: VX level for extreme fear (0.4 multiplier)
            elevated_threshold: VX level for elevated volatility (0.7 multiplier)
            max_retries: Maximum connection retry attempts
            base_delay: Base delay for exponential backoff
        """
        self.config = VxConfig(
            host=host,
            port=port,
            client_id=client_id,
            market_data_type=market_data_type,
            stale_seconds=stale_seconds,
            conservative_on_stale=conservative_on_stale,
            extreme_threshold=extreme_threshold,
            elevated_threshold=elevated_threshold,
            max_retries=max_retries,
            base_delay=base_delay,
        )
        
        self.ib = IB()
        self._state = VxState()
        # Use RLock (reentrant lock) because get_volatility_multiplier() calls is_stale()
        # and both need to acquire the lock. Also avoids deadlock in background thread
        # when logging multiplier value while holding the lock.
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._front_contract: Optional[Contract] = None
        self._ticker: Optional[Ticker] = None
        
    def connect(self) -> bool:
        """Synchronous connect for simple usage.
        
        Returns:
            True if connected successfully, False otherwise.
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._connect_async())
            loop.close()
            return result
        except Exception as e:
            logger.error("VX feed connect failed: {}", e)
            return False
    
    async def _connect_async(self) -> bool:
        """Connect to IBKR with exponential backoff retry logic."""
        if self.ib.isConnected():
            return True
            
        for attempt in range(self.config.max_retries):
            try:
                logger.info(
                    "VX Feed: Connecting to IBKR at {}:{} with client_id={} (attempt {}/{})",
                    self.config.host,
                    self.config.port,
                    self.config.client_id,
                    attempt + 1,
                    self.config.max_retries,
                )
                await self.ib.connectAsync(
                    self.config.host,
                    self.config.port,
                    clientId=self.config.client_id,
                    timeout=30
                )
                
                # Small delay to let connection stabilize
                await asyncio.sleep(0.5)
                
                # Set market data type
                self.ib.reqMarketDataType(self.config.market_data_type)
                data_type_str = "Live" if self.config.market_data_type == 1 else "Delayed"
                logger.info(
                    "✅ VX Feed connected successfully (client_id={}, {} data)",
                    self.config.client_id,
                    data_type_str
                )
                return True
                
            except TimeoutError:
                logger.error(
                    "❌ VX Feed connection timeout (attempt {}/{})",
                    attempt + 1,
                    self.config.max_retries,
                )
            except Exception as e:
                logger.warning(
                    "VX Feed connection failed (attempt {}/{}): {}",
                    attempt + 1,
                    self.config.max_retries,
                    e,
                )
            
            if attempt < self.config.max_retries - 1:
                delay = min(
                    self.config.base_delay * (2 ** attempt),
                    self.config.max_delay
                )
                logger.info("VX Feed: Retrying in {:.1f}s...", delay)
                await asyncio.sleep(delay)
        
        logger.error("VX Feed: Failed to connect after {} attempts", self.config.max_retries)
        return False
    
    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.ib.isConnected():
            try:
                if self._ticker:
                    self.ib.cancelMktData(self._front_contract)
                self.ib.disconnect()
                logger.info("VX Feed disconnected")
            except Exception as e:
                logger.warning("VX Feed disconnect error: {}", e)
        
        self._front_contract = None
        self._ticker = None
    
    def start_in_background(self) -> bool:
        """Start VX feed in a background thread.
        
        Returns:
            True if started successfully, False otherwise.
        """
        if self._running:
            logger.warning("VX Feed already running in background")
            return True
        
        self._running = True
        self._thread = threading.Thread(target=self._run_background, daemon=True)
        self._thread.start()
        
        # Wait briefly for connection to establish
        time.sleep(2)
        
        with self._lock:
            connected = self.ib.isConnected()
        
        if connected:
            logger.info("✅ VX Feed background thread started successfully")
            return True
        else:
            logger.warning("⚠️ VX Feed background thread started but not yet connected")
            return True  # Thread is running, will retry connection
    
    def _run_background(self) -> None:
        """Background thread main loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._background_main())
        except Exception as e:
            logger.error("VX Feed background thread error: {}", e)
        finally:
            self._loop.close()
            self._running = False
    
    async def _background_main(self) -> None:
        """Main async loop for background thread."""
        reconnect_delay = 30
        poll_interval = 10  # Check for new price every 10 seconds
        
        while self._running:
            try:
                # Connect if not connected
                if not self.ib.isConnected():
                    connected = await self._connect_async()
                    if not connected:
                        await asyncio.sleep(reconnect_delay)
                        continue
                
                # Get front month contract if not set
                if not self._front_contract:
                    self._front_contract = await self._get_front_month_contract()
                    if not self._front_contract:
                        logger.error("VX Feed: Could not find front month contract")
                        await asyncio.sleep(60)
                        continue
                    else:
                        logger.info("VX Feed: Using contract {} (exp={})", 
                                   self._front_contract.localSymbol,
                                   self._front_contract.lastTradeDateOrContractMonth)
                
                # Subscribe to market data if not subscribed
                if not self._ticker:
                    await self._subscribe_market_data()
                
                # Poll for price updates (simpler than streaming)
                if self._ticker:
                    price = self._get_price_from_ticker()
                    if price is not None and price > 0:
                        with self._lock:
                            old_price = self._state.price
                            self._state.price = price
                            self._state.last_update = datetime.now()
                            self._state.error_count = 0
                        
                        # Only log when price changes significantly
                        if old_price is None or abs(price - old_price) > 0.05:
                            logger.info("📈 VX Feed: Price={:.2f} (multiplier={:.1f}x)", 
                                       price, self.get_volatility_multiplier())
                
                # Sleep between polls
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error("VX Feed background loop error: {}", e)
                with self._lock:
                    self._state.error_count += 1
                
                # Reset state on error
                self._front_contract = None
                self._ticker = None
                
                if self.ib.isConnected():
                    self.ib.disconnect()
                
                await asyncio.sleep(10)
    
    async def _get_front_month_contract(self) -> Optional[Contract]:
        """Get the front-month VX futures contract.
        
        Returns:
            Front-month VX contract or None if not found.
        """
        try:
            # Request all VIX futures contracts
            # NOTE: IBKR uses symbol "VIX" (not "VX") for VIX futures on CFE
            vx_base = Future(symbol="VIX", exchange="CFE", currency="USD")
            details_list = await self.ib.reqContractDetailsAsync(vx_base)
            
            if not details_list:
                logger.warning("VX Feed: No VIX futures contracts found (check CFE subscription)")
                return None
            
            # Filter for non-expired contracts and sort by expiry
            today = datetime.now().strftime("%Y%m%d")
            valid_contracts = []
            
            for detail in details_list:
                contract = detail.contract
                expiry = contract.lastTradeDateOrContractMonth
                
                # Only include contracts that haven't expired
                if expiry and expiry >= today:
                    valid_contracts.append((expiry, contract))
            
            if not valid_contracts:
                logger.warning("VX Feed: No valid (non-expired) VX contracts found")
                return None
            
            # Sort by expiry date and get the front month
            valid_contracts.sort(key=lambda x: x[0])
            front_month = valid_contracts[0][1]
            
            logger.info(
                "VX Feed: Selected front-month contract: {} (expiry: {})",
                front_month.localSymbol,
                front_month.lastTradeDateOrContractMonth
            )
            
            # Store contract symbol in state
            with self._lock:
                self._state.contract_symbol = front_month.localSymbol
            
            return front_month
            
        except Exception as e:
            logger.error("VX Feed: Error getting front month contract: {}", e)
            return None
    
    async def _subscribe_market_data(self) -> None:
        """Subscribe to market data for the front-month contract."""
        if not self._front_contract:
            return
        
        try:
            self._ticker = self.ib.reqMktData(
                self._front_contract,
                genericTickList="",
                snapshot=False,
                regulatorySnapshot=False
            )
            logger.info(
                "VX Feed: Subscribed to market data for {}",
                self._front_contract.localSymbol
            )
        except Exception as e:
            logger.error("VX Feed: Error subscribing to market data: {}", e)
            self._ticker = None
    
    async def _process_updates(self) -> None:
        """Process market data updates."""
        poll_interval = 5  # Check every 5 seconds
        
        while self._running and self.ib.isConnected():
            try:
                if self._ticker:
                    price = self._get_price_from_ticker()
                    
                    if price is not None and price > 0:
                        with self._lock:
                            self._state.price = price
                            self._state.last_update = datetime.now()
                            self._state.error_count = 0
                        
                        logger.debug("VX Feed: Price update: {:.2f}", price)
                
                # Use asyncio.sleep instead of ib.sleep to avoid event loop conflict
                await asyncio.sleep(poll_interval)
                
            except asyncio.CancelledError:
                logger.info("VX Feed: Update loop cancelled")
                break
            except Exception as e:
                logger.error("VX Feed: Error processing updates: {}", e)
                await asyncio.sleep(10)  # Wait before retrying
    
    def _get_price_from_ticker(self) -> Optional[float]:
        """Extract price from ticker with fallbacks.
        
        Priority: marketPrice() -> last -> close -> midpoint -> bid/ask
        
        Returns:
            Price value or None if unavailable.
        """
        import math
        
        if not self._ticker:
            return None
        
        def is_valid_price(p) -> bool:
            """Check if price is valid (not None, NaN, or negative)."""
            if p is None:
                return False
            try:
                pf = float(p)
                return pf > 0 and not math.isnan(pf)
            except (ValueError, TypeError):
                return False
        
        # Try marketPrice() first (handles all cases internally)
        try:
            price = self._ticker.marketPrice()
            if is_valid_price(price):
                return float(price)
        except Exception:
            pass
        
        # Fallback to last price
        if is_valid_price(self._ticker.last):
            return float(self._ticker.last)
        
        # Fallback to close price (useful when market just opened)
        if is_valid_price(self._ticker.close):
            return float(self._ticker.close)
        
        # Fallback to midpoint (only if both bid and ask are valid)
        bid = self._ticker.bid
        ask = self._ticker.ask
        if is_valid_price(bid) and is_valid_price(ask):
            return float((bid + ask) / 2)
        
        # Last resort: bid only (ask might be -1.0 in thin markets)
        if is_valid_price(bid):
            return float(bid)
        if is_valid_price(ask):
            return float(ask)
        
        return None
    
    def stop_background(self) -> None:
        """Stop the background thread and disconnect."""
        logger.info("VX Feed: Stopping background thread...")
        self._running = False
        
        # Give thread time to clean up
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        self.disconnect()
        logger.info("VX Feed: Background thread stopped")
    
    def get_latest(self) -> dict:
        """Get the latest VX state as a dictionary.
        
        Returns:
            Dict with price, last_update, contract, is_stale, multiplier.
        """
        with self._lock:
            is_stale = self.is_stale()
            return {
                "price": self._state.price,
                "last_update": self._state.last_update.isoformat() if self._state.last_update else None,
                "contract": self._state.contract_symbol,
                "is_stale": is_stale,
                "multiplier": self._calculate_multiplier(self._state.price, is_stale),
                "error_count": self._state.error_count,
                "connected": self.ib.isConnected(),
            }
    
    def get_vx_price(self) -> Optional[float]:
        """Get the current VX price.
        
        Returns:
            Current VX price or None if unavailable.
        """
        with self._lock:
            return self._state.price
    
    def is_stale(self) -> bool:
        """Check if the VX data is stale.
        
        Returns:
            True if data is stale (older than stale_seconds), False otherwise.
        """
        with self._lock:
            if self._state.last_update is None:
                return True
            
            age = (datetime.now() - self._state.last_update).total_seconds()
            return age > self.config.stale_seconds
    
    def get_volatility_multiplier(self) -> float:
        """Get the volatility-based position sizing multiplier.
        
        The multiplier reduces position sizes when VIX is elevated:
        - VX >= extreme_threshold (30): 0.4 (extreme fear)
        - VX >= elevated_threshold (20): 0.7 (elevated volatility)
        - VX < elevated_threshold: 1.0 (normal conditions)
        
        If data is stale and conservative_on_stale is True, returns 0.5.
        
        Returns:
            Multiplier value between 0.4 and 1.0.
        """
        with self._lock:
            is_stale = self.is_stale()
            return self._calculate_multiplier(self._state.price, is_stale)
    
    def _calculate_multiplier(self, price: Optional[float], is_stale: bool) -> float:
        """Calculate the volatility multiplier based on VX price.
        
        Args:
            price: Current VX price
            is_stale: Whether the data is stale
            
        Returns:
            Multiplier value between 0.4 and 1.0.
        """
        # Handle stale data
        if is_stale and self.config.conservative_on_stale:
            return 0.5
        
        # No price available - use neutral multiplier
        if price is None or price <= 0:
            return 1.0
        
        # Calculate multiplier based on thresholds
        if price >= self.config.extreme_threshold:
            return 0.4
        elif price >= self.config.elevated_threshold:
            return 0.7
        else:
            return 1.0


# Module-level singleton for easy access
_vx_feed_instance: Optional[VxFuturesFeed] = None


def get_vx_feed() -> Optional[VxFuturesFeed]:
    """Get the global VX feed instance.
    
    Returns:
        VxFuturesFeed instance or None if not initialized.
    """
    return _vx_feed_instance


def init_vx_feed(
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 71,
    **kwargs
) -> VxFuturesFeed:
    """Initialize the global VX feed instance.
    
    Args:
        host: IB Gateway host address
        port: IB Gateway port
        client_id: Unique client ID
        **kwargs: Additional VxFuturesFeed arguments
        
    Returns:
        Initialized VxFuturesFeed instance.
    """
    global _vx_feed_instance
    
    if _vx_feed_instance is not None:
        logger.warning("VX Feed already initialized, stopping existing instance")
        _vx_feed_instance.stop_background()
    
    _vx_feed_instance = VxFuturesFeed(
        host=host,
        port=port,
        client_id=client_id,
        **kwargs
    )
    
    return _vx_feed_instance


def shutdown_vx_feed() -> None:
    """Shutdown the global VX feed instance."""
    global _vx_feed_instance
    
    if _vx_feed_instance is not None:
        _vx_feed_instance.stop_background()
        _vx_feed_instance = None
        logger.info("VX Feed shutdown complete")
