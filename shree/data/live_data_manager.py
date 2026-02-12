"""
LiveDataManager - Event-driven live market data subscription manager.
=====================================================================
Replaces 15-minute polling with real-time tick subscriptions.
Provides event callbacks for strategies to react to market data.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ib_insync import IB, Contract, Future, Ticker

from ..utils.logger import logger


class TickType(Enum):
    """Types of tick data events."""
    LAST = "LAST"
    BID = "BID"
    ASK = "ASK"
    BID_SIZE = "BID_SIZE"
    ASK_SIZE = "ASK_SIZE"
    LAST_SIZE = "LAST_SIZE"
    VOLUME = "VOLUME"
    HIGH = "HIGH"
    LOW = "LOW"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    UNKNOWN = "UNKNOWN"


@dataclass
class NormalizedTick:
    """Normalized tick data structure for all tick events."""
    timestamp: datetime
    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    last_size: Optional[int] = None
    volume: Optional[int] = None
    tick_type: TickType = TickType.UNKNOWN
    raw_tick_type: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['tick_type'] = self.tick_type.value
        return d


@dataclass  
class NormalizedQuote:
    """Normalized quote (bid/ask) update."""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    mid: float = field(init=False)
    spread: float = field(init=False)
    
    def __post_init__(self):
        self.mid = (self.bid + self.ask) / 2 if self.bid and self.ask else 0.0
        self.spread = self.ask - self.bid if self.bid and self.ask else 0.0


@dataclass
class NormalizedCandle:
    """OHLCV candle reconstructed from ticks."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_count: int
    interval_seconds: int


@dataclass
class MarketDepthLevel:
    """Single level of market depth."""
    price: float
    size: int
    market_maker: str = ""


@dataclass
class NormalizedDepth:
    """Normalized market depth update."""
    timestamp: datetime
    symbol: str
    bids: List[MarketDepthLevel]
    asks: List[MarketDepthLevel]


@dataclass
class ConnectionStatus:
    """Connection status tracking."""
    connected: bool = False
    last_connect_attempt: Optional[datetime] = None
    last_successful_connect: Optional[datetime] = None
    reconnect_count: int = 0
    last_error: Optional[str] = None


class LiveDataConfig:
    """Configuration for LiveDataManager."""
    
    def __init__(
        self,
        enable_tick_subscriptions: bool = True,
        candle_interval_seconds: int = 60,
        max_event_queue_size: int = 10000,
        backpressure_drop_count: int = 100,
        reconnect_max_retries: int = 5,
        reconnect_base_delay_seconds: float = 1.0,
        reconnect_max_delay_seconds: float = 60.0,
        log_dir: str = "./logs",
    ):
        self.enable_tick_subscriptions = enable_tick_subscriptions
        self.candle_interval_seconds = candle_interval_seconds
        self.max_event_queue_size = max_event_queue_size
        self.backpressure_drop_count = backpressure_drop_count
        self.reconnect_max_retries = reconnect_max_retries
        self.reconnect_base_delay_seconds = reconnect_base_delay_seconds
        self.reconnect_max_delay_seconds = reconnect_max_delay_seconds
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LiveDataConfig":
        """Load config from YAML file."""
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        live_data = data.get("LIVE_DATA", {})
        return cls(
            enable_tick_subscriptions=live_data.get("enable_tick_subscriptions", True),
            candle_interval_seconds=live_data.get("candle_interval_seconds", 60),
            max_event_queue_size=live_data.get("max_event_queue_size", 10000),
            backpressure_drop_count=live_data.get("backpressure_drop_count", 100),
            reconnect_max_retries=live_data.get("reconnect_max_retries", 5),
            reconnect_base_delay_seconds=live_data.get("reconnect_base_delay_seconds", 1.0),
            reconnect_max_delay_seconds=live_data.get("reconnect_max_delay_seconds", 60.0),
            log_dir=data.get("LOG_DIR", "./logs"),
        )


class CandleBuilder:
    """Builds candles from ticks in real-time."""
    
    def __init__(self, symbol: str, interval_seconds: int = 60):
        self.symbol = symbol
        self.interval_seconds = interval_seconds
        self.current_candle: Optional[NormalizedCandle] = None
        self.candle_start_time: Optional[datetime] = None
        self.completed_candles: deque[NormalizedCandle] = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def _get_candle_start(self, timestamp: datetime) -> datetime:
        """Calculate the start time of the candle period containing timestamp."""
        # Round down to the nearest interval
        ts = timestamp.timestamp()
        candle_ts = (ts // self.interval_seconds) * self.interval_seconds
        return datetime.fromtimestamp(candle_ts, tz=timezone.utc)
    
    def process_tick(self, tick: NormalizedTick) -> Optional[NormalizedCandle]:
        """
        Process a tick and potentially return a completed candle.
        
        Returns:
            Completed candle if a new candle period started, None otherwise.
        """
        if tick.last is None:
            return None
        
        with self._lock:
            candle_start = self._get_candle_start(tick.timestamp)
            completed_candle = None
            
            # Check if we need to start a new candle
            if self.candle_start_time is None or candle_start > self.candle_start_time:
                # Complete current candle if it exists
                if self.current_candle is not None:
                    completed_candle = self.current_candle
                    self.completed_candles.append(completed_candle)
                
                # Start new candle
                self.candle_start_time = candle_start
                self.current_candle = NormalizedCandle(
                    timestamp=candle_start,
                    symbol=self.symbol,
                    open=tick.last,
                    high=tick.last,
                    low=tick.last,
                    close=tick.last,
                    volume=tick.last_size or 0,
                    tick_count=1,
                    interval_seconds=self.interval_seconds,
                )
            else:
                # Update current candle
                if self.current_candle is not None:
                    self.current_candle.high = max(self.current_candle.high, tick.last)
                    self.current_candle.low = min(self.current_candle.low, tick.last)
                    self.current_candle.close = tick.last
                    self.current_candle.volume += tick.last_size or 0
                    self.current_candle.tick_count += 1
            
            return completed_candle
    
    def get_current_candle(self) -> Optional[NormalizedCandle]:
        """Get the current in-progress candle."""
        with self._lock:
            return self.current_candle
    
    def get_completed_candles(self, count: int = 100) -> List[NormalizedCandle]:
        """Get the most recent completed candles."""
        with self._lock:
            return list(self.completed_candles)[-count:]
    
    def rebuild_candles_from_ticks(
        self, 
        ticks: List[NormalizedTick]
    ) -> List[NormalizedCandle]:
        """
        Rebuild candles from a list of historical ticks.
        Useful for warming up indicators after reconnection.
        
        Args:
            ticks: List of ticks sorted by timestamp ascending
            
        Returns:
            List of completed candles
        """
        candles = []
        temp_builder = CandleBuilder(self.symbol, self.interval_seconds)
        
        for tick in ticks:
            completed = temp_builder.process_tick(tick)
            if completed:
                candles.append(completed)
        
        # Add final incomplete candle if it has data
        if temp_builder.current_candle is not None:
            candles.append(temp_builder.current_candle)
        
        return candles


class LiveDataManager:
    """
    Manages live market data subscriptions from Interactive Brokers.
    
    Replaces 15-minute polling with real-time tick subscriptions.
    Provides event callbacks for strategies to react to market changes.
    """
    
    def __init__(
        self,
        ib: IB,
        config: Optional[LiveDataConfig] = None,
    ):
        self.ib = ib
        self.config = config or LiveDataConfig()
        
        # Connection status
        self.connection_status = ConnectionStatus()
        self._connection_lock: Optional[asyncio.Lock] = None  # Lazy init for Python 3.9 compat
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Subscriptions
        self._subscribed_contracts: Dict[str, Contract] = {}
        self._tickers: Dict[str, Ticker] = {}
        self._candle_builders: Dict[str, CandleBuilder] = {}
        
        # Event queue with backpressure
        self._event_queue: deque[Dict[str, Any]] = deque(
            maxlen=self.config.max_event_queue_size
        )
        self._dropped_events_count = 0
        
        # Callbacks - strategies register these
        self._on_tick_callbacks: List[Callable[[NormalizedTick], None]] = []
        self._on_quote_callbacks: List[Callable[[NormalizedQuote], None]] = []
        self._on_depth_callbacks: List[Callable[[NormalizedDepth], None]] = []
        self._on_candle_callbacks: List[Callable[[NormalizedCandle], None]] = []
        self._on_connection_callbacks: List[Callable[[ConnectionStatus], None]] = []
        
        # Latest data cache (for quick access)
        self._latest_ticks: Dict[str, NormalizedTick] = {}
        self._latest_quotes: Dict[str, NormalizedQuote] = {}
        
        # Correlation ID for logging
        self._correlation_id = self._generate_correlation_id()
        
        # Register IB event handlers
        self._setup_ib_handlers()
    
    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID for this session."""
        return hashlib.md5(
            f"{datetime.now().isoformat()}-{id(self)}".encode()
        ).hexdigest()[:12]
    
    def _setup_ib_handlers(self) -> None:
        """Set up IB event handlers for data updates."""
        self.ib.pendingTickersEvent += self._on_pending_tickers
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.connectedEvent += self._on_connected
        self.ib.errorEvent += self._on_ib_error
    
    def _log_event(
        self, 
        event_type: str, 
        data: Dict[str, Any], 
        level: str = "INFO"
    ) -> None:
        """Log structured event to reconcile.log."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": self._correlation_id,
            "event_type": event_type,
            "level": level,
            "data": data,
        }
        
        log_file = self.config.log_dir / "reconcile.log"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Also log to standard logger
        log_msg = f"[{event_type}] {json.dumps(data)}"
        if level == "ERROR":
            logger.error(log_msg)
        elif level == "WARN":
            logger.warning(log_msg)
        else:
            logger.debug(log_msg)
    
    # =========================================================================
    # Connection Management
    # =========================================================================
    
    async def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """
        Wait for IB connection to be established.
        
        Args:
            timeout: Maximum seconds to wait
            
        Returns:
            True if connected, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.ib.isConnected():
                self.connection_status.connected = True
                self.connection_status.last_successful_connect = datetime.now(timezone.utc)
                self._log_event("connection_status", {"status": "connected"})
                return True
            await asyncio.sleep(0.5)
        
        self._log_event(
            "connection_status", 
            {"status": "timeout", "timeout_seconds": timeout},
            level="ERROR"
        )
        return False
    
    def is_connected(self) -> bool:
        """Check if currently connected to IB."""
        return self.ib.isConnected()
    
    def _on_connected(self) -> None:
        """Handle IB connection established."""
        self.connection_status.connected = True
        self.connection_status.last_successful_connect = datetime.now(timezone.utc)
        self._log_event("connection_status", {"status": "connected"})
        
        # Notify callbacks
        for callback in self._on_connection_callbacks:
            try:
                callback(self.connection_status)
            except Exception as e:
                logger.error(f"Connection callback error: {e}")
    
    def _on_disconnected(self) -> None:
        """Handle IB disconnection."""
        self.connection_status.connected = False
        self._log_event(
            "connection_status", 
            {"status": "disconnected"},
            level="WARN"
        )
        
        # Notify callbacks
        for callback in self._on_connection_callbacks:
            try:
                callback(self.connection_status)
            except Exception as e:
                logger.error(f"Connection callback error: {e}")
        
        # Start reconnection if not already running
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())
    
    def _on_ib_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract) -> None:
        """Handle IB errors."""
        self._log_event(
            "ib_error",
            {
                "req_id": reqId,
                "error_code": errorCode,
                "error_string": errorString,
                "contract": str(contract) if contract else None,
            },
            level="ERROR" if errorCode >= 200 else "WARN"
        )
    
    async def _reconnect_loop(self) -> None:
        """Reconnection loop with exponential backoff."""
        retries = 0
        delay = self.config.reconnect_base_delay_seconds
        
        while retries < self.config.reconnect_max_retries:
            self.connection_status.last_connect_attempt = datetime.now(timezone.utc)
            self.connection_status.reconnect_count += 1
            retries += 1
            
            self._log_event(
                "reconnect_attempt",
                {"attempt": retries, "delay_seconds": delay}
            )
            
            try:
                if not self.ib.isConnected():
                    # IB connection is managed externally - just wait
                    await asyncio.sleep(delay)
                    
                    if self.ib.isConnected():
                        self._log_event("reconnect_success", {"attempts": retries})
                        # Re-subscribe to contracts
                        await self._resubscribe_all()
                        return
                else:
                    return
                    
            except Exception as e:
                self.connection_status.last_error = str(e)
                self._log_event(
                    "reconnect_failed",
                    {"attempt": retries, "error": str(e)},
                    level="ERROR"
                )
            
            # Exponential backoff
            delay = min(delay * 2, self.config.reconnect_max_delay_seconds)
        
        self._log_event(
            "reconnect_exhausted",
            {"max_retries": self.config.reconnect_max_retries},
            level="ERROR"
        )
    
    async def _resubscribe_all(self) -> None:
        """Re-subscribe to all contracts after reconnection."""
        for symbol, contract in list(self._subscribed_contracts.items()):
            try:
                await self.subscribe(contract)
                self._log_event("resubscribe_success", {"symbol": symbol})
            except Exception as e:
                self._log_event(
                    "resubscribe_failed",
                    {"symbol": symbol, "error": str(e)},
                    level="ERROR"
                )
    
    # =========================================================================
    # Subscription Management
    # =========================================================================
    
    async def subscribe(
        self,
        contract: Contract,
        market_depth: bool = False,
    ) -> bool:
        """
        Subscribe to real-time data for a contract.
        
        Args:
            contract: IB Contract to subscribe to
            market_depth: Whether to also subscribe to market depth (L2)
            
        Returns:
            True if subscription successful
        """
        if not self.ib.isConnected():
            self._log_event(
                "subscribe_failed",
                {"symbol": contract.symbol, "reason": "not_connected"},
                level="ERROR"
            )
            return False
        
        symbol = contract.symbol
        
        try:
            # Qualify contract if needed
            qualified = await self.ib.qualifyContractsAsync(contract)
            if qualified:
                contract = qualified[0]
            
            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            
            self._tickers[symbol] = ticker
            self._subscribed_contracts[symbol] = contract
            
            # Create candle builder
            self._candle_builders[symbol] = CandleBuilder(
                symbol, 
                self.config.candle_interval_seconds
            )
            
            # Request market depth if requested
            if market_depth:
                self.ib.reqMktDepth(contract, 5)
            
            self._log_event(
                "subscription_started",
                {"symbol": symbol, "contract": str(contract), "market_depth": market_depth}
            )
            
            return True
            
        except Exception as e:
            self._log_event(
                "subscribe_failed",
                {"symbol": symbol, "error": str(e)},
                level="ERROR"
            )
            return False
    
    async def unsubscribe(self, symbol: str) -> bool:
        """Unsubscribe from real-time data for a symbol."""
        if symbol not in self._subscribed_contracts:
            return False
        
        try:
            contract = self._subscribed_contracts[symbol]
            
            if symbol in self._tickers:
                self.ib.cancelMktData(contract)
                del self._tickers[symbol]
            
            del self._subscribed_contracts[symbol]
            
            if symbol in self._candle_builders:
                del self._candle_builders[symbol]
            
            self._log_event("unsubscribed", {"symbol": symbol})
            return True
            
        except Exception as e:
            self._log_event(
                "unsubscribe_failed",
                {"symbol": symbol, "error": str(e)},
                level="ERROR"
            )
            return False
    
    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all active subscriptions."""
        for symbol in list(self._subscribed_contracts.keys()):
            await self.unsubscribe(symbol)
    
    # =========================================================================
    # Event Handling
    # =========================================================================
    
    def _on_pending_tickers(self, tickers: Set[Ticker]) -> None:
        """Handle incoming ticker updates from IB."""
        for ticker in tickers:
            if ticker.contract is None:
                continue
            
            symbol = ticker.contract.symbol
            timestamp = datetime.now(timezone.utc)
            
            # Create normalized tick
            tick = NormalizedTick(
                timestamp=timestamp,
                symbol=symbol,
                bid=ticker.bid if ticker.bid > 0 else None,
                ask=ticker.ask if ticker.ask > 0 else None,
                last=ticker.last if ticker.last > 0 else None,
                bid_size=ticker.bidSize if ticker.bidSize >= 0 else None,
                ask_size=ticker.askSize if ticker.askSize >= 0 else None,
                last_size=ticker.lastSize if ticker.lastSize >= 0 else None,
                volume=ticker.volume if ticker.volume >= 0 else None,
            )
            
            # Check backpressure
            if len(self._event_queue) >= self.config.max_event_queue_size:
                # Drop oldest events
                for _ in range(self.config.backpressure_drop_count):
                    if self._event_queue:
                        self._event_queue.popleft()
                        self._dropped_events_count += 1
                
                if self._dropped_events_count % 1000 == 0:
                    self._log_event(
                        "live_data_backpressure_drop",
                        {"dropped_total": self._dropped_events_count},
                        level="WARN"
                    )
            
            # Store latest tick
            self._latest_ticks[symbol] = tick
            
            # Add to event queue
            self._event_queue.append({"type": "tick", "data": tick})
            
            # Fire tick callbacks
            for callback in self._on_tick_callbacks:
                try:
                    callback(tick)
                except Exception as e:
                    logger.error(f"Tick callback error: {e}")
            
            # Build candle
            if symbol in self._candle_builders and tick.last is not None:
                completed_candle = self._candle_builders[symbol].process_tick(tick)
                if completed_candle:
                    self._event_queue.append({"type": "candle", "data": completed_candle})
                    
                    # Fire candle callbacks
                    for callback in self._on_candle_callbacks:
                        try:
                            callback(completed_candle)
                        except Exception as e:
                            logger.error(f"Candle callback error: {e}")
            
            # Create quote update if bid/ask changed
            if tick.bid and tick.ask:
                quote = NormalizedQuote(
                    timestamp=timestamp,
                    symbol=symbol,
                    bid=tick.bid,
                    ask=tick.ask,
                    bid_size=tick.bid_size or 0,
                    ask_size=tick.ask_size or 0,
                )
                
                self._latest_quotes[symbol] = quote
                
                # Fire quote callbacks
                for callback in self._on_quote_callbacks:
                    try:
                        callback(quote)
                    except Exception as e:
                        logger.error(f"Quote callback error: {e}")
    
    # =========================================================================
    # Callback Registration
    # =========================================================================
    
    def register_on_tick(self, callback: Callable[[NormalizedTick], None]) -> None:
        """Register callback for tick events."""
        self._on_tick_callbacks.append(callback)
    
    def register_on_quote(self, callback: Callable[[NormalizedQuote], None]) -> None:
        """Register callback for quote updates."""
        self._on_quote_callbacks.append(callback)
    
    def register_on_candle(self, callback: Callable[[NormalizedCandle], None]) -> None:
        """Register callback for new candle events."""
        self._on_candle_callbacks.append(callback)
    
    def register_on_depth(self, callback: Callable[[NormalizedDepth], None]) -> None:
        """Register callback for market depth updates."""
        self._on_depth_callbacks.append(callback)
    
    def register_on_connection(self, callback: Callable[[ConnectionStatus], None]) -> None:
        """Register callback for connection status changes."""
        self._on_connection_callbacks.append(callback)
    
    def unregister_on_tick(self, callback: Callable[[NormalizedTick], None]) -> None:
        """Unregister tick callback."""
        if callback in self._on_tick_callbacks:
            self._on_tick_callbacks.remove(callback)
    
    def unregister_on_quote(self, callback: Callable[[NormalizedQuote], None]) -> None:
        """Unregister quote callback."""
        if callback in self._on_quote_callbacks:
            self._on_quote_callbacks.remove(callback)
    
    def unregister_on_candle(self, callback: Callable[[NormalizedCandle], None]) -> None:
        """Unregister candle callback."""
        if callback in self._on_candle_callbacks:
            self._on_candle_callbacks.remove(callback)
    
    # =========================================================================
    # Data Access
    # =========================================================================
    
    def get_latest_tick(self, symbol: str) -> Optional[NormalizedTick]:
        """Get the latest tick for a symbol."""
        return self._latest_ticks.get(symbol)
    
    def get_latest_quote(self, symbol: str) -> Optional[NormalizedQuote]:
        """Get the latest quote for a symbol."""
        return self._latest_quotes.get(symbol)
    
    def get_current_candle(self, symbol: str) -> Optional[NormalizedCandle]:
        """Get the current in-progress candle."""
        builder = self._candle_builders.get(symbol)
        return builder.get_current_candle() if builder else None
    
    def get_completed_candles(self, symbol: str, count: int = 100) -> List[NormalizedCandle]:
        """Get completed candles for a symbol."""
        builder = self._candle_builders.get(symbol)
        return builder.get_completed_candles(count) if builder else []
    
    def rebuild_candles_from_ticks(
        self, 
        symbol: str,
        ticks: List[NormalizedTick]
    ) -> List[NormalizedCandle]:
        """
        Rebuild candles from historical ticks.
        Useful for warming up indicators after reconnection.
        """
        builder = self._candle_builders.get(symbol)
        if builder:
            return builder.rebuild_candles_from_ticks(ticks)
        
        # Create temporary builder if symbol not subscribed
        temp_builder = CandleBuilder(symbol, self.config.candle_interval_seconds)
        return temp_builder.rebuild_candles_from_ticks(ticks)
    
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of currently subscribed symbols."""
        return list(self._subscribed_contracts.keys())
    
    def get_connection_status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self.connection_status
    
    def get_event_queue_size(self) -> int:
        """Get current event queue size."""
        return len(self._event_queue)
    
    def get_dropped_events_count(self) -> int:
        """Get total count of dropped events due to backpressure."""
        return self._dropped_events_count
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    async def shutdown(self) -> None:
        """Shutdown the LiveDataManager gracefully."""
        self._log_event("shutdown_started", {})
        
        # Unsubscribe from all
        await self.unsubscribe_all()
        
        # Cancel reconnect task
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        
        # Clear callbacks
        self._on_tick_callbacks.clear()
        self._on_quote_callbacks.clear()
        self._on_candle_callbacks.clear()
        self._on_depth_callbacks.clear()
        self._on_connection_callbacks.clear()
        
        self._log_event("shutdown_completed", {})
