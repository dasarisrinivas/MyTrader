"""Trade execution via Interactive Brokers with reconciliation support."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from uuid import uuid4
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING, Any

from ib_insync import Contract, Future, IB, LimitOrder, MarketOrder, Order, StopOrder, StopLimitOrder, Trade

from ..config import TradingConfig
from ..risk.trade_math import (
    ContractSpec,
    TradingMode,
    calculate_realized_pnl,
    get_commission_per_side,
    get_contract_spec,
)
from ..monitoring.order_tracker import OrderTracker
from ..utils.logger import logger
from ..utils.structured_logging import log_structured_event
from ..utils.telegram_notifier import TelegramNotifier
from .order_builder import format_bracket_snapshot, validate_bracket_prices

if TYPE_CHECKING:
    from .reconcile import ReconcileManager, ReconcileLock
    from ..data.live_data_manager import LiveDataManager


@dataclass
class OrderResult:
    trade: Trade
    status: str
    message: Optional[str] = None
    fill_price: Optional[float] = None
    filled_quantity: int = 0


@dataclass
class PositionInfo:
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_atr_multiplier: Optional[float] = None
    trailing_percent: Optional[float] = None
    atr_value: Optional[float] = None
    entry_metadata: Optional[Dict] = None


@dataclass
class CloseFill:
    """Represents the result of closing part of a position."""

    contracts: float
    entry_price: float
    exit_price: float
    direction: int  # +1 for closing long, -1 for closing short
    gross_pnl: float
    points: float


class TradeExecutor:
    """Enhanced trade executor with real-time PnL tracking and order monitoring.
    
    Integrates with:
    - ReconcileManager for startup reconciliation
    - LiveDataManager for event-driven market data
    - Safety features: DRY_RUN, SAFE_MODE, reconcile_lock
    """
    
    def __init__(
        self, 
        ib: IB, 
        config: TradingConfig, 
        symbol: str, 
        exchange: str = "GLOBEX", 
        currency: str = "USD",
        telegram_notifier: Optional[TelegramNotifier] = None,
        reconcile_manager: Optional["ReconcileManager"] = None,
        live_data_manager: Optional["LiveDataManager"] = None,
        trading_mode: TradingMode = "live",
        contract_spec: Optional[ContractSpec] = None,
        commission_per_side: Optional[float] = None,
    ) -> None:
        self.ib = ib
        self.config = config
        self.symbol = symbol
        self.exchange = exchange
        self.currency = currency
        self.active_orders: Dict[int, Trade] = {}
        self.positions: Dict[str, PositionInfo] = {}
        self.order_tracker = OrderTracker()  # SQLite-based order tracking
        self.order_history: List[Dict] = []
        self.contract_spec: ContractSpec = contract_spec or get_contract_spec(symbol, config)
        self.trading_mode: TradingMode = trading_mode
        self._commission_per_side = (
            commission_per_side
            if commission_per_side is not None
            else get_commission_per_side(
                self.contract_spec,
                self.trading_mode,
                getattr(config, "commission_per_contract", None),
            )
        )
        self.realized_pnl_gross = 0.0
        self.realized_pnl_net = 0.0
        self.realized_pnl = 0.0  # Backwards compatibility alias (net PnL)
        self.total_commission_paid = 0.0
        self._local_position_qty = 0.0
        self._local_avg_price = 0.0
        self._open_position_commission = 0.0
        self.order_history: List[Dict] = []
        self._qualified_contract: Contract | None = None  # Cache the qualified front month contract
        self._last_contract_refresh: Optional[float] = None
        self._contract_call_timestamps: deque[float] = deque()
        self._price_snapshot_timestamps: deque[float] = deque()
        self._contract_cache_ttl = getattr(config, "contract_cache_ttl_seconds", 300)
        self._price_snapshot_min_interval = getattr(config, "price_snapshot_min_interval_seconds", 5)
        self._contract_warning_threshold = getattr(config, "contract_call_warning_threshold", 5)
        self._snapshot_warning_threshold = getattr(config, "snapshot_call_warning_threshold", 30)
        self._last_price_snapshot: Optional[float] = None
        self._last_price_value: Optional[float] = None
        
        # Store stop loss / take profit for each order (for Telegram notifications)
        self.order_targets: Dict[int, Dict[str, float]] = {}  # {order_id: {'stop_loss': float, 'take_profit': float}}
        self.order_metadata: Dict[int, Dict[str, Any]] = {}
        
        # Track order creation times to detect stuck orders
        self.order_creation_times: Dict[int, datetime] = {}  # {order_id: creation_time}
        
        # Store connection parameters for auto-reconnect
        self._connection_host: str = "127.0.0.1"
        self._connection_port: int = 4002
        self._connection_client_id: int = 2
        self._connection_client_id: int = 2
        self._keepalive_task: Optional[object] = None
        
        # Telegram notifications
        self.telegram = telegram_notifier
        
        # NEW: Reconciliation and Live Data integration
        self._reconcile_manager: Optional["ReconcileManager"] = reconcile_manager
        self._live_data_manager: Optional["LiveDataManager"] = live_data_manager
        self._reconcile_lock: Optional["ReconcileLock"] = None
        
        # NEW: Idempotency tracking for order submissions
        self._submission_signatures: Dict[str, datetime] = {}
        self._signature_ttl_seconds: int = getattr(config, "idempotency_signature_ttl_seconds", 900)
        
        # NEW: Initial state tracking (orders found on IB at startup)
        self._initial_state_order_ids: set = set()

        # NEW: Hard order lock to prevent overlapping brackets
        self._order_locked: bool = False
        self._order_lock_reason: Optional[str] = None
        self._order_lock_engaged_at: Optional[datetime] = None
        self._order_lock_parent_id: Optional[int] = None
        self._order_lock_order_ids: set[int] = set()
        self._order_lock_timeout_seconds: int = getattr(config, "order_lock_timeout_seconds", 300)
        self._pending_order_timeout_seconds: int = getattr(config, "pending_order_timeout_seconds", 180)
        
        # Initialize PositionManager
        from .position_manager import PositionManager
        self.position_manager = PositionManager(ib, config, symbol)

    # =========================================================================
    # NEW: Reconciliation Integration Methods
    # =========================================================================
    
    def set_reconcile_manager(self, manager: "ReconcileManager") -> None:
        """Set the reconciliation manager for startup sync."""
        self._reconcile_manager = manager
        self._reconcile_lock = manager.lock
        logger.info("ReconcileManager attached to executor")
    
    def set_live_data_manager(self, manager: "LiveDataManager") -> None:
        """Set the live data manager for event-driven data."""
        self._live_data_manager = manager
        logger.info("LiveDataManager attached to executor")
    
    def can_trade(self) -> bool:
        """Check if trading is allowed (reconciliation complete, not in safe mode)."""
        if self._reconcile_lock is None:
            return True  # No lock configured, allow trading
        return self._reconcile_lock.can_trade()
    
    def is_safe_mode(self) -> bool:
        """Check if safe mode is active."""
        if self._reconcile_lock is None:
            return False
        return self._reconcile_lock.is_safe_mode()

    def is_order_locked(self) -> bool:
        """Expose order lock status to trading manager."""
        self._enforce_order_lock_timeout()
        return self._order_locked
    
    def get_order_lock_reason(self) -> Optional[str]:
        """Return current order lock reason for telemetry."""
        self._enforce_order_lock_timeout()
        return self._order_lock_reason
    
    def get_order_lock_age_seconds(self) -> float:
        """Return how long the current lock has been engaged (seconds)."""
        self._enforce_order_lock_timeout()
        return self._order_lock_age_seconds()

    def force_release_order_lock(self, reason: str = "manual override", cancel_tracked: bool = True) -> None:
        """Manually release any lock (optionally canceling tracked orders first)."""
        if cancel_tracked:
            self._cancel_locked_orders(reason)
        self._release_order_lock(reason)

    def _engage_order_lock(self, reason: str) -> None:
        """Engage hard order lock until bracket placement is confirmed."""
        self._order_locked = True
        self._order_lock_reason = reason
        self._order_lock_engaged_at = datetime.utcnow()
        self._order_lock_order_ids.clear()
        self._order_lock_parent_id = None
        logger.warning(f"🔒 Order lock engaged: {reason}")

    def _release_order_lock(self, context: str = "") -> None:
        """Release the hard order lock."""
        if self._order_locked:
            logger.info(f"🔓 Order lock released ({context})")
        self._order_locked = False
        self._order_lock_reason = None
        self._order_lock_engaged_at = None
        self._order_lock_parent_id = None
        self._order_lock_order_ids.clear()
    
    def _register_lock_order_id(self, order_id: Optional[int]) -> None:
        """Track orders associated with the current lock for watchdog clean-up."""
        if not self._order_locked or order_id is None:
            return
        self._order_lock_order_ids.add(order_id)
        if self._order_lock_parent_id is None:
            self._order_lock_parent_id = order_id
    
    def _cancel_locked_orders(self, reason: str) -> None:
        """Cancel any tracked orders associated with the lock."""
        if not self._order_lock_order_ids:
            return
        logger.warning("🧹 Canceling %d locked orders (%s)", len(self._order_lock_order_ids), reason)
        for order_id in list(self._order_lock_order_ids):
            trade = self.active_orders.get(order_id)
            if not trade:
                continue
            self._cancel_trade(trade, reason, warn_if_missing=False)
        self._order_lock_order_ids.clear()
    
    def _order_lock_age_seconds(self) -> float:
        if not self._order_locked or not self._order_lock_engaged_at:
            return 0.0
        return (datetime.utcnow() - self._order_lock_engaged_at).total_seconds()
    
    def _enforce_order_lock_timeout(self) -> None:
        """Watchdog to automatically clear locks that overstay the timeout."""
        if not self._order_locked:
            return
        if self._order_lock_timeout_seconds <= 0:
            return
        age = self._order_lock_age_seconds()
        if age < self._order_lock_timeout_seconds:
            return
        logger.error(
            "⏰ Order lock watchdog triggered after %.0fs (reason=%s, parent=%s)",
            age,
            self._order_lock_reason,
            self._order_lock_parent_id,
        )
        self._cancel_locked_orders("lock_timeout")
        self._release_order_lock("watchdog timeout")
    
    def is_initial_state_order(self, order_id: int) -> bool:
        """Check if an order was found on IB at startup (external order)."""
        if self._reconcile_manager:
            return self._reconcile_manager.is_initial_state_order(order_id)
        return order_id in self._initial_state_order_ids
    
    def mark_initial_state_orders(self, order_ids: set) -> None:
        """Mark orders as initial state (found on IB at startup)."""
        self._initial_state_order_ids = order_ids
        logger.info(f"Marked {len(order_ids)} orders as initial state")
    
    def generate_submission_signature(
        self,
        action: str,
        quantity: int,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Generate idempotency signature for order submission.
        Includes trading context so restarts avoid replaying entries.
        """
        bar_ts = metadata.get("bar_close_timestamp") or ""
        signal_id = metadata.get("signal_id") or metadata.get("trade_cycle_id") or ""
        strategy = metadata.get("strategy_name") or metadata.get("signal_source") or "unknown"
        bucket = metadata.get("entry_price_bucket")
        bucket_str = f"{bucket:.2f}" if bucket is not None else ""
        payload = "|".join(
            [
                self.symbol,
                action.upper(),
                str(bar_ts),
                str(signal_id),
                strategy,
                str(quantity),
                bucket_str,
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]
    
    def is_duplicate_submission(self, signature: str) -> bool:
        """Check if a submission signature is a duplicate."""
        if signature in self._submission_signatures:
            created_at = self._submission_signatures[signature]
            if (datetime.now(timezone.utc) - created_at).total_seconds() < self._signature_ttl_seconds:
                return True
            del self._submission_signatures[signature]
        try:
            if self.order_tracker.signature_exists(signature, self._signature_ttl_seconds):
                return True
        except Exception as exc:
            logger.debug(f"Signature lookup skipped: {exc}")
        return False
    
    def record_submission(
        self,
        signature: str,
        action: str,
        quantity: int,
        metadata: Dict[str, Any],
    ) -> None:
        """Record a submission signature for idempotency."""
        now = datetime.now(timezone.utc)
        self._submission_signatures[signature] = now
        try:
            self.order_tracker.record_submission_signature(
                signature=signature,
                symbol=self.symbol,
                action=action,
                quantity=quantity,
                price_bucket=metadata.get("entry_price_bucket"),
                bar_timestamp=metadata.get("bar_close_timestamp"),
                signal_id=metadata.get("signal_id") or metadata.get("trade_cycle_id"),
                strategy_name=metadata.get("strategy_name") or metadata.get("signal_source"),
            )
        except Exception as exc:
            logger.debug(f"Signature persistence failed: {exc}")
        
        # Clean up old signatures (older than 5 minutes)
        cutoff = datetime.now(timezone.utc)
        old_sigs = [
            sig for sig, ts in self._submission_signatures.items()
            if (cutoff - ts).total_seconds() > self._signature_ttl_seconds
        ]
        for sig in old_sigs:
            del self._submission_signatures[sig]
    
    def _log_reconcile_event(self, event_type: str, data: dict, level: str = "INFO") -> None:
        """Log event to reconcile.log for audit trail."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "level": level,
            "data": data,
        }
        
        log_dir = Path("./logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        with open(log_dir / "reconcile.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def contract(self) -> Contract:
        return Future(symbol=self.symbol, exchange=self.exchange, currency=self.currency)
    
    async def get_qualified_contract(self) -> Contract | None:
        """Get the fully qualified front month contract."""
        self._record_metric(self._contract_call_timestamps, "qualified_contract_calls", self._contract_warning_threshold)
        if (
            self._qualified_contract
            and self._last_contract_refresh
            and time.time() - self._last_contract_refresh < self._contract_cache_ttl
        ):
            logger.debug("Using cached contract %s", self._qualified_contract.localSymbol)
            return self._qualified_contract
        try:
            # Check connection health before making API call
            if not self.ib.isConnected():
                logger.warning("IB connection lost, attempting to reconnect...")
                try:
                    # Try to reconnect with stored connection parameters
                    await self.ib.connectAsync(
                        self._connection_host, 
                        self._connection_port, 
                        clientId=self._connection_client_id, 
                        timeout=30
                    )
                    logger.info("Reconnection successful")
                except Exception as reconnect_error:
                    logger.error(f"Reconnection failed: {reconnect_error}")
                    return None
            
            contract = self.contract()
            details = await self.ib.reqContractDetailsAsync(contract)
            if not details:
                logger.error(f"Could not find any contracts for {self.symbol}")
                return None
            
            # Sort by expiration date to get the desired month (offset driven)
            details.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
            offset = max(0, min(getattr(self.config, "contract_month_offset", 0), len(details) - 1))
            front_month = details[offset].contract
            logger.info(
                "Qualified contract: {symbol} (exp: {expiry}) [offset={offset}]",
                symbol=front_month.localSymbol,
                expiry=front_month.lastTradeDateOrContractMonth,
                offset=offset,
            )
            
            # Cache it
            self._qualified_contract = front_month
            self._last_contract_refresh = time.time()
            return front_month
        except Exception as e:
            logger.error(f"Failed to get qualified contract: {e}")
            return None

    async def connect(self, host: str, port: int, client_id: int, timeout: int = 120) -> None:
        """Connect to IBKR and set up event handlers."""
        if self.ib.isConnected():
            return
        
        # Store connection parameters for auto-reconnect
        self._connection_host = host
        self._connection_port = port
        self._connection_client_id = client_id
        
        logger.info(
            "Connecting executor to IBKR {host}:{port} (client_id={client_id}, timeout={timeout}s)",
            host=host,
            port=port,
            client_id=client_id,
            timeout=timeout,
        )
        try:
            await self.ib.connectAsync(host, port, clientId=client_id, timeout=timeout)
        except TimeoutError:
            logger.error(
                "Connection timeout after {timeout}s. IB Gateway may be in bad state.",
                timeout=timeout,
            )
            logger.error("Try restarting IB Gateway: Close it completely, wait 30s, then restart")
            raise
        
        # Note: We use snapshot=True for price requests, which doesn't require
        # reqMarketDataType(). This avoids "competing live session" errors.
        logger.info("Connected to IBKR - using snapshot market data requests")
        
        # Set up event handlers for order updates
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_execution
        
        # Cancel all existing orders on startup to avoid "too many orders" error
        await self._cancel_all_existing_orders()
        
        # Request initial positions
        await self._reconcile_positions()
        
        # Start connection keepalive task
        await self._start_keepalive()
    
    async def _cancel_all_existing_orders(self) -> None:
        """Cancel any existing orders for this symbol and sync order state.
        
        This is called on startup to ensure a clean slate. We aggressively
        cancel and wait for confirmation to avoid the "active orders pending" issue.
        """
        try:
            # Get current open trades from IB
            open_trades = self.ib.openTrades()
            trades_to_cancel = []
            
            # Collect orders for our symbol
            for trade in open_trades:
                if trade.contract.symbol == self.symbol:
                    trades_to_cancel.append(trade)
                    logger.info(f"🎯 Found existing order to cancel: {trade.order.orderId} - {trade.order.action} {trade.order.totalQuantity} {self.symbol} (status: {trade.orderStatus.status})")
            
            if not trades_to_cancel:
                logger.info(f"✅ No existing orders for {self.symbol} - clean slate")
                self.active_orders.clear()
                return
            
            logger.info(f"🔄 Canceling {len(trades_to_cancel)} existing orders for {self.symbol}...")
            
            # Cancel each order and wait for status change
            for trade in trades_to_cancel:
                order_id = trade.order.orderId
                initial_status = trade.orderStatus.status
                
                # Skip if already cancelled/filled
                if initial_status in ['Cancelled', 'Filled', 'Inactive']:
                    logger.info(f"   Order {order_id} already {initial_status}, skipping")
                    continue
                
                logger.info(f"   Canceling order {order_id}...")
                self.ib.cancelOrder(trade.order)
            
            # Wait for cancellations with polling (more reliable than fixed sleep)
            max_wait_seconds = 10
            poll_interval = 0.5
            elapsed = 0.0
            
            while elapsed < max_wait_seconds:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                
                # Check if all our orders are now cancelled
                still_active = 0
                for trade in self.ib.openTrades():
                    if trade.contract.symbol == self.symbol:
                        if trade.orderStatus.status in ['PreSubmitted', 'Submitted', 'PendingSubmit', 'PendingCancel']:
                            still_active += 1
                
                if still_active == 0:
                    logger.info(f"✅ All orders cancelled after {elapsed:.1f}s")
                    break
                
                logger.debug(f"   Waiting for {still_active} orders to cancel... ({elapsed:.1f}s)")
            
            # Final check - if orders still exist after timeout, log a warning
            remaining_orders = []
            for trade in self.ib.openTrades():
                if trade.contract.symbol == self.symbol:
                    status = trade.orderStatus.status
                    if status in ['PreSubmitted', 'Submitted', 'PendingSubmit']:
                        remaining_orders.append(f"{trade.order.orderId}({status})")
            
            if remaining_orders:
                logger.warning(f"⚠️  {len(remaining_orders)} orders could not be cancelled after {max_wait_seconds}s: {remaining_orders}")
                logger.warning("   These may be from a different client_id or stuck in IB Gateway.")
                logger.warning("   Consider manually canceling via IB Gateway or TWS.")
            
            # Clear our tracking regardless - we'll re-sync in _reconcile_orders
            self.active_orders.clear()
            self.order_creation_times.clear()
            
            # Reconcile order state with IB after cancellations
            await self._reconcile_orders()
            
        except Exception as e:
            logger.error(f"Failed to cancel existing orders: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _reconcile_orders(self) -> None:
        """Reconcile active orders with IB Gateway."""
        try:
            # Clear current tracking
            old_count = len(self.active_orders)
            self.active_orders.clear()
            
            # Get current open trades from IB
            open_trades = self.ib.openTrades()
            
            for trade in open_trades:
                if trade.contract.symbol == self.symbol:
                    order_id = trade.order.orderId
                    status = trade.orderStatus.status
                    
                    # Only track truly active orders (not cancelled/filled)
                    if status in ['PreSubmitted', 'Submitted', 'PendingSubmit']:
                        self.active_orders[order_id] = trade
                        logger.info(f"📋 Active order found: {order_id} - {trade.order.action} {trade.order.totalQuantity} {self.symbol} (status: {status})")
                    else:
                        logger.debug(f"Skipping order {order_id} with status: {status}")
            
            if old_count > 0 or len(self.active_orders) > 0:
                logger.info(f"🔄 Order reconciliation: {old_count} → {len(self.active_orders)} active orders")
            else:
                logger.info("✅ No active orders")
                
        except Exception as e:
            logger.error(f"Failed to reconcile orders: {e}")

    async def force_cancel_all_orders(self) -> int:
        """Force cancel ALL orders for this symbol (use with caution).
        
        This is a nuclear option for when orders are stuck and blocking new trades.
        It uses reqGlobalCancel() which cancels ALL orders across all client IDs.
        
        Returns:
            Number of orders that were canceled
        """
        logger.warning("🚨 FORCE CANCEL ALL ORDERS requested")
        
        try:
            # First, try symbol-specific cancel
            open_trades = self.ib.openTrades()
            symbol_orders = [t for t in open_trades if t.contract.symbol == self.symbol]
            
            if not symbol_orders:
                logger.info("No orders to cancel")
                self.active_orders.clear()
                return 0
            
            logger.warning(f"🔴 Force canceling {len(symbol_orders)} orders for {self.symbol}...")
            
            for trade in symbol_orders:
                logger.info(f"   Canceling: {trade.order.orderId} - {trade.order.action} {trade.order.totalQuantity}")
                self.ib.cancelOrder(trade.order)
            
            # Wait longer for force cancel
            await asyncio.sleep(5)
            
            # Check if any remain
            remaining = [t for t in self.ib.openTrades() 
                        if t.contract.symbol == self.symbol 
                        and t.orderStatus.status in ['PreSubmitted', 'Submitted', 'PendingSubmit']]
            
            if remaining:
                logger.warning(f"⚠️  {len(remaining)} orders still remaining after cancel attempt")
                logger.warning("   Attempting global cancel request...")
                # Last resort: global cancel
                self.ib.reqGlobalCancel()
                await asyncio.sleep(3)
            
            # Clear internal tracking
            self.active_orders.clear()
            self.order_creation_times.clear()
            
            # Re-sync
            await self._reconcile_orders()
            
            canceled_count = len(symbol_orders) - len(self.active_orders)
            logger.info(f"✅ Force cancel complete: {canceled_count} orders canceled")
            return canceled_count
            
        except Exception as e:
            logger.error(f"Force cancel failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0

    async def _reconcile_positions(self) -> None:
        """Reconcile current positions with IBKR."""
        try:
            positions = self.ib.positions()
            for position in positions:
                if position.contract.symbol == self.symbol:
                    # IBKR avgCost is total cost, divide by quantity to get per-contract price
                    per_contract_cost = float(position.avgCost) / abs(position.position) if position.position != 0 else 0.0
                    self.positions[self.symbol] = PositionInfo(
                        symbol=self.symbol,
                        quantity=int(position.position),
                        avg_cost=per_contract_cost,
                        market_value=float(position.position * per_contract_cost),
                        unrealized_pnl=float(position.unrealizedPNL) if hasattr(position, 'unrealizedPNL') else 0.0,
                        realized_pnl=0.0
                    )
                    logger.info(f"Reconciled position: {self.symbol} qty={position.position} avg_cost={per_contract_cost:.2f} (total_cost={position.avgCost:.2f})")
        except Exception as e:
            logger.error("Failed to reconcile positions: {}", e)

    def _apply_fill_to_position(
        self,
        action: str,
        quantity: float,
        price: float,
    ) -> Optional[CloseFill]:
        """Update local position tracker and return closing stats if any."""
        normalized = action.upper()
        closing: Optional[CloseFill] = None
        remaining = float(quantity)
        if normalized == "BUY":
            if self._local_position_qty < 0:
                close_qty = min(remaining, abs(self._local_position_qty))
                gross, points = calculate_realized_pnl(
                    self._local_avg_price or price,
                    price,
                    -close_qty,
                    self.contract_spec,
                )
                closing = CloseFill(
                    contracts=close_qty,
                    entry_price=self._local_avg_price or price,
                    exit_price=price,
                    direction=-1,
                    gross_pnl=gross,
                    points=points,
                )
                self._local_position_qty += close_qty
                remaining -= close_qty
                if abs(self._local_position_qty) < 1e-9:
                    self._local_position_qty = 0.0
                    self._local_avg_price = 0.0
            if remaining > 0:
                if self._local_position_qty <= 0:
                    self._local_avg_price = price
                    self._local_position_qty = remaining
                else:
                    total_cost = self._local_avg_price * self._local_position_qty + price * remaining
                    self._local_position_qty += remaining
                    self._local_avg_price = total_cost / max(self._local_position_qty, 1e-9)
        else:  # SELL
            if self._local_position_qty > 0:
                close_qty = min(remaining, self._local_position_qty)
                gross, points = calculate_realized_pnl(
                    self._local_avg_price or price,
                    price,
                    close_qty,
                    self.contract_spec,
                )
                closing = CloseFill(
                    contracts=close_qty,
                    entry_price=self._local_avg_price or price,
                    exit_price=price,
                    direction=1,
                    gross_pnl=gross,
                    points=points,
                )
                self._local_position_qty -= close_qty
                remaining -= close_qty
                if abs(self._local_position_qty) < 1e-9:
                    self._local_position_qty = 0.0
                    self._local_avg_price = 0.0
            if remaining > 0:
                if self._local_position_qty >= 0:
                    self._local_avg_price = price
                    self._local_position_qty = -remaining
                else:
                    total_cost = self._local_avg_price * abs(self._local_position_qty) + price * remaining
                    self._local_position_qty -= remaining
                    self._local_avg_price = total_cost / max(abs(self._local_position_qty), 1e-9)
        return closing

    def _allocate_entry_commission(
        self,
        closed_qty: float,
        prev_abs_position: float,
    ) -> float:
        """Remove proportional entry commissions for contracts that just closed."""
        if closed_qty <= 0 or prev_abs_position <= 0 or self._open_position_commission <= 0:
            return 0.0
        proportion = closed_qty / prev_abs_position
        allocated = self._open_position_commission * proportion
        self._open_position_commission -= allocated
        if self._open_position_commission < 1e-6:
            self._open_position_commission = 0.0
        return allocated

    def _estimate_fill_commission(
        self,
        quantity: float,
        reported_commission: Optional[float],
    ) -> float:
        """Return commission paid for a fill, estimating when IB doesn't report it."""
        if reported_commission is not None and abs(reported_commission) > 1e-6:
            return float(reported_commission)
        if quantity <= 0:
            return 0.0
        per_side = get_commission_per_side(
            self.contract_spec,
            self.trading_mode,
            self._commission_per_side,
        )
        if self.trading_mode == "paper":
            return 0.0
        return per_side * quantity

    def _on_order_status(self, trade: Trade) -> None:
        """Callback for order status updates."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        filled = trade.orderStatus.filled
        remaining = trade.orderStatus.remaining
        avg_fill_price = trade.orderStatus.avgFillPrice
        
        logger.info(f"📊 Order {order_id} status: {status} (filled={filled}, remaining={remaining}, avg={avg_fill_price:.2f})")
        
        # Update order tracker
        self.order_tracker.update_order_status(
            order_id=order_id,
            status=status,
            filled=filled,
            remaining=remaining,
            avg_fill_price=avg_fill_price if avg_fill_price > 0 else None,
            message=f"Status: {status}"
        )
        
        if status in ("Filled", "Cancelled", "Inactive"):
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            # Clean up tracking dictionaries
            if order_id in self.order_creation_times:
                del self.order_creation_times[order_id]
            if order_id in self.order_targets:
                del self.order_targets[order_id]
        
        # Log to history
        self.order_history.append({
            "timestamp": datetime.utcnow(),
            "order_id": order_id,
            "status": status,
            "filled": filled,
            "remaining": remaining,
            "avg_fill_price": avg_fill_price
        })

    def _on_execution(self, trade: Trade, fill) -> None:
        """Callback for execution details with slippage protection."""
        order_id = trade.order.orderId
        quantity = fill.execution.shares
        price = fill.execution.price
        commission_report = (
            float(fill.commissionReport.commission)
            if hasattr(fill, "commissionReport") and hasattr(fill.commissionReport, "commission")
            else None
        )
        realized_pnl = 0.0
        
        # FIX: Check for excessive slippage
        order = trade.order
        expected_price = None
        
        if hasattr(order, 'lmtPrice') and order.lmtPrice > 0:
            expected_price = order.lmtPrice
        
        if expected_price:
            slippage = abs(price - expected_price)
            slippage_pct = (slippage / expected_price) * 100
            
            # Alert on excessive slippage (> $5 or 0.1%)
            max_slippage_dollars = 5.0
            max_slippage_pct = 0.1
            
            if slippage > max_slippage_dollars or slippage_pct > max_slippage_pct:
                logger.warning(f"⚠️  HIGH SLIPPAGE: Order {order_id}, expected={expected_price:.2f}, filled={price:.2f}, slippage=${slippage:.2f} ({slippage_pct:.3f}%)")
            else:
                logger.info(f"✅ Execution: Order {order_id}, qty={quantity}, price={price:.2f}, slippage=${slippage:.2f}")
        else:
            logger.info(f"✅ Execution: Order {order_id}, qty={quantity}, price={price:.2f}, commission={commission_report}")
        
        abs_quantity = abs(quantity)
        commission_paid = self._estimate_fill_commission(abs_quantity, commission_report)
        per_contract_commission = commission_paid / abs_quantity if abs_quantity else 0.0
        prev_abs_position = abs(self._local_position_qty)
        close_result = self._apply_fill_to_position(order.action, abs_quantity, price)
        closed_qty = close_result.contracts if close_result else 0.0
        entry_commission_alloc = self._allocate_entry_commission(closed_qty, prev_abs_position)
        exit_commission_paid = per_contract_commission * closed_qty
        round_trip_commission = entry_commission_alloc + exit_commission_paid
        open_qty = max(abs_quantity - closed_qty, 0.0)
        if open_qty > 0:
            self._open_position_commission += per_contract_commission * open_qty
        gross_pnl = close_result.gross_pnl if close_result else 0.0
        realized_pnl = gross_pnl - round_trip_commission
        self.total_commission_paid += commission_paid
        if close_result:
            self.realized_pnl_gross += gross_pnl
            self.realized_pnl_net += realized_pnl
            self.realized_pnl = self.realized_pnl_net
            logger.info(
                "💰 Realized PnL gross=%.2f net=%.2f (points=%.2f, commission=%.2f, totals gross=%.2f net=%.2f)",
                gross_pnl,
                realized_pnl,
                close_result.points,
                round_trip_commission,
                self.realized_pnl_gross,
                self.realized_pnl_net,
            )
        else:
            logger.debug("Position updated without closure; tracking unrealized PnL only.")
        
        # Record execution in tracker
        self.order_tracker.record_execution(
            order_id=order_id,
            quantity=quantity,
            price=price,
            commission=commission_paid,
            realized_pnl=realized_pnl,
            gross_pnl=gross_pnl,
            net_pnl=realized_pnl,
        )
        
        # Reconcile positions after execution to get updated position from IB
        # Note: This is called from sync callback, need to schedule as task
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._reconcile_positions())
            else:
                asyncio.create_task(self._reconcile_positions())
        except RuntimeError:
            pass  # No event loop, skip reconciliation
        
        # EMERGENCY STOP CHECK: Verify bracket protection after entry fill
        # Only check for entry orders (not exit orders which have metadata["exit_order"]=True)
        order_meta = self.order_metadata.get(order_id, {})
        parent_id = getattr(order, "parentId", None)
        parent_meta = self.order_metadata.get(parent_id) if parent_id is not None else None
        metadata_source = parent_meta or order_meta
        is_exit_order = order_meta.get("exit_order", False)
        targets = self.order_targets.get(order_id, {})
        if not targets and parent_id in self.order_targets:
            targets = self.order_targets.get(parent_id, {})
        expected_stop_loss = targets.get('stop_loss')
        expected_take_profit = targets.get('take_profit')
        
        if not is_exit_order and (expected_stop_loss is not None or expected_take_profit is not None):
            # This is an entry order that should have bracket protection
            # Check if bracket orders are active (children with parentId = order_id)
            bracket_active = False
            try:
                # Check IB directly for active bracket orders
                all_trades = self.ib.trades()
                for t in all_trades:
                    if hasattr(t.order, 'parentId') and t.order.parentId == order_id:
                        if t.orderStatus.status not in ("Filled", "Cancelled", "Inactive"):
                            bracket_active = True
                            logger.debug(f"✅ Found active bracket order {t.order.orderId} for parent {order_id}")
                            break
                
                # Also check active_orders dict for bracket children
                if not bracket_active:
                    for child_id, child_trade in self.active_orders.items():
                        if hasattr(child_trade.order, 'parentId') and child_trade.order.parentId == order_id:
                            if child_trade.orderStatus.status not in ("Filled", "Cancelled", "Inactive"):
                                bracket_active = True
                                logger.debug(f"✅ Found active bracket order {child_id} in active_orders for parent {order_id}")
                                break
            except Exception as bracket_check_error:
                logger.warning(f"⚠️ Error checking bracket status: {bracket_check_error}")
            
            if not bracket_active and expected_stop_loss is not None:
                # CRITICAL: Entry fill without bracket protection - place emergency stop
                logger.error(
                    f"🚨 EMERGENCY STOP REQUIRED: Entry order {order_id} filled @ {price} "
                    f"but bracket orders missing or inactive. Expected SL={expected_stop_loss}"
                )
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(self._place_emergency_stop(
                            order_id=order_id,
                            entry_price=price,
                            entry_action=order.action,
                            quantity=quantity,
                            expected_stop_loss=expected_stop_loss,
                            trade_cycle_id=order_meta.get("trade_cycle_id"),
                        ))
                    else:
                        asyncio.create_task(self._place_emergency_stop(
                            order_id=order_id,
                            entry_price=price,
                            entry_action=order.action,
                            quantity=quantity,
                            expected_stop_loss=expected_stop_loss,
                            trade_cycle_id=order_meta.get("trade_cycle_id"),
                        ))
                except Exception as emergency_error:
                    logger.critical(f"❌ FAILED to place emergency stop: {emergency_error}")
        
        # Send Telegram notification (non-blocking)
        logger.info(f"📱 Checking Telegram: enabled={self.telegram.enabled if self.telegram else False}")
        if self.telegram and self.telegram.enabled:
            try:
                # Get current position
                current_position = self.positions.get(self.symbol)
                position_qty = current_position.quantity if current_position else None
                
                # Determine side from order action
                side = order.action  # "BUY" or "SELL"
                
                logger.info(f"📱 Sending Telegram alert: {side} {quantity} @ {price}, SL={expected_stop_loss}, TP={expected_take_profit}")
                # Send notification in background (fire-and-forget)
                entry_price_msg = close_result.entry_price if close_result else metadata_source.get("entry_price")
                commission_msg = round_trip_commission if close_result else commission_paid
                gross_msg = gross_pnl if close_result else 0.0
                net_msg = realized_pnl if close_result else 0.0
                points_msg = close_result.points if close_result else None
                self.telegram.send_trade_alert_background(
                    symbol=self.symbol,
                    side=side,
                    quantity=abs_quantity,
                    fill_price=price,
                    timestamp=datetime.utcnow(),
                    current_position=position_qty,
                    order_id=order_id,
                    entry_price=entry_price_msg,
                    exit_price=price if close_result else None,
                    commission=commission_msg,
                    gross_pnl=gross_msg,
                    net_pnl=net_msg,
                    points=points_msg,
                    risk_reward=metadata_source.get("risk_reward"),
                    stop_loss=expected_stop_loss,
                    take_profit=expected_take_profit
                )
                logger.info(f"📱 Telegram alert queued successfully")
            except Exception as e:
                # Never let Telegram errors affect trading
                logger.error(f"❌ Failed to send Telegram notification: {e}")

    async def place_order(
        self,
        action: str,
        quantity: int,
        limit_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        metadata: Dict | None = None,
        rationale: Dict | None = None,
        features: Dict | None = None,
        market_regime: str | None = None,
        entry_price: float | None = None,
        reduce_only: bool = False,
    ) -> OrderResult:
        """Place an order with optional bracket orders for risk management.
        
        NEW: Integrates with reconciliation safety features:
        - Checks reconcile_lock before placing orders
        - Respects SAFE_MODE to prevent trading
        - Uses idempotency signatures to prevent duplicates
        """
        # CRITICAL: Convert SCALP_BUY/SCALP_SELL to BUY/SELL for IB API
        # IB API only accepts 'BUY' or 'SELL' as valid action values
        original_action = action
        if action in ("SCALP_BUY", "scalp_buy"):
            action = "BUY"
        elif action in ("SCALP_SELL", "scalp_sell"):
            action = "SELL"
        
        if original_action != action:
            logger.info(f"📝 Converted action {original_action} -> {action} for IB API")

        metadata = dict(metadata or {})
        entry_price_hint = entry_price
        if entry_price_hint is None:
            entry_price_hint = metadata.get("entry_price")
        if entry_price_hint is None and limit_price is not None:
            entry_price_hint = limit_price
        if entry_price_hint is not None:
            metadata.setdefault("entry_price", entry_price_hint)

        if reduce_only:
            current_position = await self.get_current_position()
            if not current_position or current_position.quantity == 0:
                logger.warning("Reduce-only order requested but no open position")
                from ib_insync import Trade as IBTrade
                dummy_trade = IBTrade()
                return OrderResult(trade=dummy_trade, status="Cancelled", message="No position to reduce")
            required_action = "SELL" if current_position.quantity > 0 else "BUY"
            if action != required_action:
                logger.warning(f"Adjusting exit action {action} -> {required_action} for reduce-only order")
                action = required_action
            quantity = min(quantity, abs(current_position.quantity))
            metadata["reduce_only"] = True

        guard_entry_price = entry_price_hint if entry_price_hint is not None else limit_price if limit_price is not None else metadata.get("entry_price")
        guard_reason = self._validate_protective_invariants(
            action=action,
            entry_price=guard_entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reduce_only=reduce_only,
        )
        if guard_reason:
            incident_id = uuid4().hex[:10]
            message = f"Protective guard blocked order: {guard_reason} (incident={incident_id})"
            logger.error(message)
            log_structured_event(
                agent="ib_executor",
                event_type="risk.order_blocked",
                message=message,
                payload={
                    "incident_id": incident_id,
                    "reason": guard_reason,
                    "action": action,
                    "quantity": quantity,
                    "entry_price": guard_entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "reduce_only": reduce_only,
                    "trade_cycle_id": metadata.get("trade_cycle_id"),
                },
            )
            from ib_insync import Trade as IBTrade
            dummy_trade = IBTrade()
            return OrderResult(
                trade=dummy_trade,
                status="Cancelled",
                message=message,
            )
        
        # NEW: Check if trading is allowed (reconciliation complete, not in safe mode)
        if not self.can_trade():
            if self.is_safe_mode():
                logger.warning("SAFE_MODE active - order blocked")
                self._log_reconcile_event("order_blocked_safe_mode", {
                    "action": action, "quantity": quantity, "symbol": self.symbol
                }, "WARN")
            else:
                logger.warning("Reconciliation not complete - order blocked")
                self._log_reconcile_event("order_blocked_reconcile_pending", {
                    "action": action, "quantity": quantity, "symbol": self.symbol
                }, "WARN")
            
            from ib_insync import Trade as IBTrade
            dummy_trade = IBTrade()
            return OrderResult(trade=dummy_trade, status="Cancelled", message="Trading not allowed - reconciliation pending or safe mode active")
        
        # NEW: Check for duplicate submission (idempotency)
        submission_sig = self.generate_submission_signature(action, quantity, metadata)
        if self.is_duplicate_submission(submission_sig):
            logger.warning(f"Duplicate order submission blocked: {action} {quantity} @ {limit_price}")
            self._log_reconcile_event("duplicate_submission_blocked", {
                "action": action, "quantity": quantity, "limit_price": limit_price,
                "signature": submission_sig
            }, "WARN")
            log_structured_event(
                agent="ib_executor",
                event_type="duplicate_submission_blocked",
                message="Duplicate submission blocked",
                payload={
                    "signature": submission_sig,
                    "action": action,
                    "quantity": quantity,
                    "symbol": self.symbol,
                },
            )
            from ib_insync import Trade as IBTrade
            dummy_trade = IBTrade()
            return OrderResult(trade=dummy_trade, status="Cancelled", message="Duplicate submission blocked")
        
        if self._order_locked:
            lock_age = self._order_lock_age_seconds() if hasattr(self, '_order_lock_age_seconds') else None
            logger.warning(
                f"Skipping trade; order lock active for {lock_age:.1f} seconds (reason: {self._order_lock_reason})" if lock_age is not None else f"Skipping trade; order lock active (reason: {self._order_lock_reason})"
            )
            from ib_insync import Trade as IBTrade
            dummy_trade = IBTrade()
            return OrderResult(trade=dummy_trade, status="Cancelled", message="Order lock active")
        
        # Check connection health before placing order
        if not self.ib.isConnected():
            logger.error("IB not connected - cannot place order")
            from ib_insync import Trade as IBTrade
            dummy_trade = IBTrade()
            return OrderResult(trade=dummy_trade, status="Cancelled", message="Not connected to IB")
        
        # Check if we have too many active orders (IB limit is 15 per side)
        open_trades = self.ib.openTrades()
        active_count = sum(1 for t in open_trades if t.contract.symbol == self.symbol and t.orderStatus.status in ('PreSubmitted', 'Submitted'))
        if active_count >= 12:
            logger.warning(f"Skipping trade; too many active orders ({active_count}) - canceling old orders first...")
            await self._cancel_all_existing_orders()
            await asyncio.sleep(2)
        
        # Get the qualified front month contract
        contract = await self.get_qualified_contract()
        if not contract:
            logger.error("Failed to get qualified contract - cannot place order")
            # Return a failed order result
            from ib_insync import Trade as IBTrade
            dummy_trade = IBTrade()
            return OrderResult(trade=dummy_trade, status="Cancelled", message="Failed to qualify contract")
        
        # NEW: Record submission signature for idempotency
        self.record_submission(submission_sig, action, quantity, metadata)
        
        # Log order attempt for audit trail
        self._log_reconcile_event("order_placed", {
            "action": action, "quantity": quantity, "limit_price": limit_price,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "symbol": self.symbol, "signature": submission_sig
        })

        # SAFETY CHECK: Position Manager
        # Note: quantity is always positive in the argument, action determines direction
        signed_quantity = quantity if action == "BUY" else -quantity
        decision = await self.position_manager.can_place_order(signed_quantity)
        if decision.allowed_contracts == 0:
            logger.warning(f"Skipping trade due to filter/PositionManager: {decision.reason}")
            from ib_insync import Trade as IBTrade
            dummy_trade = IBTrade()
            return OrderResult(trade=dummy_trade, status="Cancelled", message=f"Rejected: {decision.reason}")
        if abs(decision.allowed_contracts) < quantity:
            logger.warning(f"Order size reduced by PositionManager: {quantity} -> {abs(decision.allowed_contracts)} ({decision.reason})")
            quantity = abs(decision.allowed_contracts)
        logger.info(f"✅ PositionManager approved: {quantity} contracts (Reason: {decision.reason})")

        order: Order
        if limit_price is not None:
            order = LimitOrder(action, quantity, limit_price)
        else:
            # PROFESSIONAL ENTRY: Use LIMIT orders with reasonable buffer
            # Get current price and add buffer to ensure fill without excessive slippage
            current_price = await self.get_current_price()
            if current_price is None:
                logger.error("Cannot get current price for limit order")
                from ib_insync import Trade as IBTrade
                dummy_trade = IBTrade()
                return OrderResult(trade=dummy_trade, status="Cancelled", message="No current price")
            
            # Use aggressive buffer for faster fills (4 ticks = 1 point for ES)
            # This ensures fill quickly while still using LIMIT orders
            tick_buffer = 1.00  # 4 ticks for ES = $50 buffer for faster fills
            limit_price = current_price + tick_buffer if action == "BUY" else current_price - tick_buffer
            order = LimitOrder(action, quantity, limit_price)
            logger.info(f"📊 Using LIMIT order @ {limit_price:.2f} (market: {current_price:.2f}, buffer: {tick_buffer})")

        # Allow trading outside regular trading hours (ES futures are nearly 24hr)
        order.outsideRth = True
        
        # Add metadata to order reference for tracking
        if metadata:
            order.orderRef = f"MyTrader_{metadata.get('signal_source', 'manual')}"
        
        entry_price_estimate = guard_entry_price
        if entry_price_estimate is None:
            entry_price_estimate = limit_price if limit_price is not None else metadata.get("entry_price")
        bracket_children: list[Order] = []
        placed_child_trades: list[Trade] = []
        if stop_loss is not None or take_profit is not None:
            validation = validate_bracket_prices(
                action=action,
                entry_price=entry_price_estimate,
                stop_loss=stop_loss,
                take_profit=take_profit,
                tick_size=self.config.tick_size,
            )
            if not validation.valid:
                rejection = validation.reason or "Invalid protective levels"
                logger.error(f"❌ Bracket rejected: {rejection}")
                self._log_reconcile_event(
                    "bracket_rejected",
                    {
                        "action": action,
                        "quantity": quantity,
                        "entry_price": entry_price_estimate,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "reason": validation.reason,
                    },
                    "ERROR",
                )
                from ib_insync import Trade as IBTrade
                dummy_trade = IBTrade()
                return OrderResult(
                    trade=dummy_trade,
                    status="Cancelled",
                    message=f"reject bracket: {rejection}",
                )

            stop_loss = validation.stop_loss
            take_profit = validation.take_profit
            if validation.adjusted_fields:
                logger.warning(
                    f"🔧 Protective levels adjusted to maintain tick spacing: {', '.join(validation.adjusted_fields)}"
                )

            order.transmit = False
            opposite = "SELL" if action == "BUY" else "BUY"
            
            if take_profit is not None:
                tp_order = LimitOrder(opposite, quantity, take_profit)
                tp_order.transmit = False
                tp_order.outsideRth = True
                bracket_children.append(tp_order)
                logger.info(f"Adding take-profit order at {take_profit:.2f}")
            
            if stop_loss is not None:
                # PROFESSIONAL STOP-LOSS: Use STOP-LIMIT with wider buffer
                # For ES futures, allow 1-2 points (4-8 ticks) of slippage on stop
                tick_size = self.config.tick_size
                offset_ticks = 4  # Allow 1 point (4 ticks = $50) of slippage
                
                if action == "BUY":  # Long position, stop is below
                    limit_price_sl = stop_loss - (offset_ticks * tick_size)
                else:  # Short position, stop is above
                    limit_price_sl = stop_loss + (offset_ticks * tick_size)
                
                sl_order = StopLimitOrder(opposite, quantity, stop_loss, limit_price_sl)
                sl_order.transmit = False
                sl_order.outsideRth = True
                bracket_children.append(sl_order)
                logger.info(f"Adding stop-loss order: stop={stop_loss:.2f}, limit={limit_price_sl:.2f} (STOP-LIMIT with {abs(stop_loss - limit_price_sl):.2f} buffer)")
            
            if bracket_children:
                bracket_children[-1].transmit = True
        else:
            order.transmit = True

        if bracket_children:
            metadata["validated_stop_loss"] = stop_loss
            metadata["validated_take_profit"] = take_profit
            snapshot = format_bracket_snapshot(
                entry_price_estimate,
                stop_loss,
                take_profit,
                metadata.get("atr_fallback_used"),
            )
            logger.info(
                "📡 Bracket telemetry | qty={qty} lock={locked} {snapshot}",
                qty=quantity,
                locked=self._order_locked,
                snapshot=snapshot,
            )

        # Place parent order
        lock_engaged = False
        if bracket_children:
            reason = f"{action} {quantity} @ {entry_price_estimate or limit_price}"
            self._engage_order_lock(reason)
            lock_engaged = True

        try:
            parent_trade = self.ib.placeOrder(contract, order)
            self._log_trade_status(parent_trade, label="parent-submit")
            parent_id = parent_trade.order.orderId
            self._register_lock_order_id(parent_id)
            self.order_metadata[parent_id] = dict(metadata)
            
            # Track order creation time
            self.order_creation_times[parent_id] = datetime.utcnow()
            
            # Store stop_loss and take_profit for this order (for Telegram notifications)
            self.order_targets[parent_id] = {
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            # Get current price for entry tracking
            current_price = limit_price if limit_price else None
            if metadata and 'entry_price' in metadata:
                current_price = metadata['entry_price']
            
            # Record order placement in tracker
            import json
            rationale_str = json.dumps(rationale) if rationale else None
            features_str = json.dumps(features) if features else None
            
            self.order_tracker.record_order_placement(
                order_id=parent_id,
                symbol=self.symbol,
                action=action,
                quantity=quantity,
                order_type="LIMIT" if limit_price else "MARKET",
                limit_price=limit_price,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=metadata.get('confidence') if metadata else None,
                atr=metadata.get('atr_value') if metadata else None,
                rationale=rationale_str,
                features=features_str,
                market_regime=market_regime,
                trade_cycle_id=metadata.get("trade_cycle_id"),
            )
            
            # Track active order
            self.active_orders[parent_id] = parent_trade
            
            # Place bracket orders
            for child in bracket_children:
                child.parentId = parent_id
            for i, child in enumerate(bracket_children):
                child_trade = self.ib.placeOrder(contract, child)
                self._log_trade_status(child_trade, label=f"child-submit-{i+1}")
                child_id = child_trade.order.orderId
                self._register_lock_order_id(child_id)
                self.active_orders[child_id] = child_trade
                placed_child_trades.append(child_trade)
                await self._await_order_submission(child_trade, label=f"child-{child_id}")
                
                # Track bracket order creation time
                self.order_creation_times[child_id] = datetime.utcnow()
                
                # Record child order (SL/TP)
                child_type = "STOP_LIMIT" if isinstance(child, StopLimitOrder) else "STOP" if isinstance(child, StopOrder) else "LIMIT"
                # StopLimitOrder has both auxPrice (stop trigger) and lmtPrice (limit price)
                if isinstance(child, StopLimitOrder):
                    child_stop_price = child.auxPrice  # Stop trigger price
                    child_limit_price = child.lmtPrice  # Limit price after stop triggered
                elif isinstance(child, StopOrder):
                    child_stop_price = child.auxPrice
                    child_limit_price = None
                else:  # LimitOrder
                    child_stop_price = None
                    child_limit_price = child.lmtPrice
                
                self.order_tracker.record_order_placement(
                    order_id=child_id,
                    parent_order_id=parent_id,
                    symbol=self.symbol,
                    action=child.action,
                    quantity=child.totalQuantity,
                    order_type=child_type,
                    stop_price=child_stop_price,
                    limit_price=child_limit_price,
                    trade_cycle_id=metadata.get("trade_cycle_id") if metadata else None,
                )
                
                logger.info(f"📝 Placed bracket order {child_id} ({child_type}) (parent={parent_id})")

            await self._await_order_submission(parent_trade, label="parent")
            confirmation_ok = await self._confirm_protective_orders(parent_trade, placed_child_trades)
            if not confirmation_ok:
                message = "Protective orders not confirmed"
                self._handle_protective_confirmation_failure(parent_trade, placed_child_trades, message)
                return OrderResult(
                    trade=parent_trade,
                    status="Cancelled",
                    message=message,
                )
        finally:
            if lock_engaged:
                self._release_order_lock("bracket submission complete")
        status = parent_trade.orderStatus.status
        
        # Store position metadata for trailing stops
        if status in ("Filled", "PreSubmitted", "Submitted"):
            fill_price = float(parent_trade.orderStatus.avgFillPrice) if parent_trade.orderStatus.avgFillPrice > 0 else None
            if fill_price and metadata:
                self.positions[self.symbol] = PositionInfo(
                    symbol=self.symbol,
                    quantity=quantity if action == "BUY" else -quantity,
                    avg_cost=fill_price,
                    market_value=fill_price * quantity,
                    unrealized_pnl=0.0,
                    realized_pnl=self.realized_pnl,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_atr_multiplier=metadata.get("trailing_atr_multiplier"),
                    trailing_percent=metadata.get("trailing_percent"),
                    atr_value=metadata.get("atr_value"),
                    entry_metadata=metadata.get("entry_metadata")
                )
                logger.info("Position metadata stored for trailing stops")
        
        logger.info(
            "Order {action} placed: orderId={order_id} status={status}",
            action=action,
            order_id=parent_id,
            status=status,
        )
        
        return OrderResult(
            trade=parent_trade, 
            status=status,
            fill_price=float(parent_trade.orderStatus.avgFillPrice) if parent_trade.orderStatus.avgFillPrice > 0 else None,
            filled_quantity=int(parent_trade.orderStatus.filled)
        )

    def _validate_protective_invariants(
        self,
        action: str,
        entry_price: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        reduce_only: bool,
    ) -> Optional[str]:
        """Ensure every non-reduce order ships with sane protective levels."""
        if reduce_only:
            return None
        if stop_loss is None or stop_loss <= 0:
            return "missing or non-positive stop-loss"
        if take_profit is None or take_profit <= 0:
            return "missing or non-positive take-profit"
        if entry_price is None or entry_price <= 0:
            return "missing entry price for protective validation"
        normalized_action = action.upper()
        if normalized_action == "BUY":
            if stop_loss >= entry_price:
                return f"BUY stop-loss {stop_loss:.2f} must be below entry {entry_price:.2f}"
            if take_profit <= entry_price:
                return f"BUY take-profit {take_profit:.2f} must be above entry {entry_price:.2f}"
        else:
            if stop_loss <= entry_price:
                return f"SELL stop-loss {stop_loss:.2f} must be above entry {entry_price:.2f}"
            if take_profit >= entry_price:
                return f"SELL take-profit {take_profit:.2f} must be below entry {entry_price:.2f}"
        return None
    
    def _record_metric(self, timestamps: deque, label: str, threshold: int) -> None:
        """Track call counts per minute and emit warnings when thresholds exceeded."""
        now = time.time()
        timestamps.append(now)
        while timestamps and now - timestamps[0] > 60:
            timestamps.popleft()
        if len(timestamps) > threshold:
            logger.warning(
                "{} per minute high: {}",
                label,
                len(timestamps),
            )
            log_structured_event(
                agent="ib_executor",
                event_type=f"metrics.{label}",
                message="Rate limit warning",
                payload={"count_last_minute": len(timestamps)},
            )

    def _log_trade_status(self, trade: Trade | None, label: str) -> None:
        """Emit a detailed snapshot of the IB response for troubleshooting."""
        if trade is None or getattr(trade, "order", None) is None:
            logger.info("🛰️ {label} -> no trade/order data available", label=label)
            return
        order = trade.order
        order_status = trade.orderStatus
        status = getattr(order_status, "status", "UNKNOWN")
        client_id = getattr(order_status, "clientId", getattr(order, "clientId", "NA"))
        perm_id = getattr(order, "permId", None) or getattr(order_status, "permId", "NA")
        filled = getattr(order_status, "filled", "NA")
        remaining = getattr(order_status, "remaining", "NA")
        backend_name = type(self.ib).__name__ if self.ib else "Unknown"
        is_mock = backend_name.lower().startswith("mock")
        logger.info(
            "🛰️ {label} | backend={backend} mock={mock} orderId={order_id} permId={perm_id} status={status} filled={filled} remaining={remaining} clientId={client_id}",
            label=label,
            backend=backend_name,
            mock=is_mock,
            order_id=getattr(order, "orderId", "NA"),
            perm_id=perm_id,
            status=status,
            filled=filled,
            remaining=remaining,
            client_id=client_id,
        )

    async def _await_order_submission(self, trade: Trade, label: str = "order", timeout: float = 3.0) -> None:
        """Wait for IB to acknowledge submission so we don't leave PendingSubmit ghosts."""
        if not self.ib or trade is None:
            return
        self._log_trade_status(trade, f"{label}-ack-start")
        start = time.monotonic()
        while trade.orderStatus.status in ("PendingSubmit", "PendingCancel"):
            await asyncio.sleep(0.2)
            if time.monotonic() - start > timeout:
                logger.warning(
                    f"⏳ {label} {trade.order.orderId} still {trade.orderStatus.status} after {timeout}s"
                )
                break
        self._log_trade_status(trade, f"{label}-ack-complete")
    
    async def _confirm_protective_orders(
        self,
        parent_trade: Trade,
        child_trades: List[Trade],
        timeout: float = 8.0,
        poll_interval: float = 0.5,
    ) -> bool:
        """Ensure parent and children advance beyond PendingSubmit shortly after placement."""
        if not child_trades:
            return True
        required_statuses = {"PreSubmitted", "Submitted", "Filled"}
        start = time.monotonic()
        parent_id = getattr(parent_trade.order, "orderId", "NA")
        child_ids = [getattr(child.order, "orderId", "NA") for child in child_trades]
        while time.monotonic() - start < timeout:
            parent_status = getattr(parent_trade.orderStatus, "status", "UNKNOWN")
            children_ready = all(
                getattr(child.orderStatus, "status", "UNKNOWN") in required_statuses
                for child in child_trades
            )
            if parent_status in required_statuses and children_ready:
                return True
            await asyncio.sleep(poll_interval)
        logger.error(
            "⚠️ Protective orders not confirmed for parent %s (children=%s)",
            parent_id,
            child_ids,
        )
        return False

    def _handle_protective_confirmation_failure(
        self,
        parent_trade: Trade,
        child_trades: List[Trade],
        message: str,
    ) -> None:
        """Cancel parent/children and emit structured events when bracket confirmation fails."""
        incident_id = uuid4().hex[:10]
        payload = {
            "incident_id": incident_id,
            "parent_order_id": getattr(parent_trade.order, "orderId", None),
            "child_order_ids": [getattr(child.order, "orderId", None) for child in child_trades],
            "symbol": self.symbol,
        }
        logger.error("❌ %s (incident=%s)", message, incident_id)
        log_structured_event(
            agent="ib_executor",
            event_type="protective_orders_not_confirmed",
            message=message,
            payload=payload,
        )
        self._log_reconcile_event("protective_orders_not_confirmed", payload, "ERROR")
        for trade in [parent_trade, *child_trades]:
            try:
                self.ib.cancelOrder(trade.order)
            except Exception as cancel_error:
                logger.error("Failed to cancel order %s: %s", getattr(trade.order, "orderId", "NA"), cancel_error)
            self.active_orders.pop(getattr(trade.order, "orderId", None), None)
        parent_id = getattr(parent_trade.order, "orderId", None)
        if parent_id in self.order_targets:
            self.order_targets.pop(parent_id, None)
        if parent_id is not None:
            self.order_tracker.update_order_status(parent_id, "Cancelled", message=message)
        for child in child_trades:
            child_id = getattr(child.order, "orderId", None)
            if child_id is not None:
                self.order_tracker.update_order_status(child_id, "Cancelled", message=message)

    def _cancel_trade(
        self,
        trade_ref: Trade | Order | None,
        reason: str,
        warn_if_missing: bool = True,
    ) -> bool:
        """Cancel a trade/order object and clean up tracking structures."""
        if trade_ref is None:
            if warn_if_missing:
                logger.warning("Cancel requested for missing trade (%s)", reason)
            return False
        order = trade_ref.order if hasattr(trade_ref, "order") else trade_ref
        if order is None:
            if warn_if_missing:
                logger.warning("Cancel requested but no order handle available (%s)", reason)
            return False
        order_id = getattr(order, "orderId", None)
        try:
            self.ib.cancelOrder(order)
            logger.info("✅ Cancelled order %s (%s)", order_id, reason)
            success = True
        except Exception as exc:
            logger.error("Failed to cancel order %s (%s): %s", order_id, reason, exc)
            success = False
        if order_id in self.active_orders:
            self.active_orders.pop(order_id, None)
        if order_id in self.order_creation_times:
            self.order_creation_times.pop(order_id, None)
        if order_id in self.order_targets:
            self.order_targets.pop(order_id, None)
        if order_id is not None:
            try:
                self.order_tracker.update_order_status(order_id, "Cancelled", message=reason)
            except Exception as tracker_error:
                logger.debug("Order tracker update failed for %s: %s", order_id, tracker_error)
        return success

    async def cancel_order(self, order_id: int) -> bool:
        """Cancel a specific order by ID."""
        if order_id not in self.active_orders:
            logger.warning("Order {order_id} not found in active orders", order_id=order_id)
            return False
        
        try:
            trade = self.active_orders[order_id]
            order = trade.order if hasattr(trade, "order") else trade
            self.ib.cancelOrder(order)
            self.ib.waitOnUpdate()  # waitOnUpdate is synchronous, not async
            logger.info("Cancelled order {order_id}", order_id=order_id)
            self.active_orders.pop(order_id, None)
            return True
        except Exception as e:
            logger.error("Failed to cancel order {order_id}: {error}", order_id=order_id, error=e)
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel all active orders."""
        count = 0
        for order_id in list(self.active_orders.keys()):
            if await self.cancel_order(order_id):
                count += 1
        logger.info("Cancelled {count} orders", count=count)
        return count

    async def get_current_position(self) -> PositionInfo | None:
        """Get current position for the trading symbol."""
        await self._reconcile_positions()
        return self.positions.get(self.symbol)

    async def get_unrealized_pnl(self) -> float:
        """Get current unrealized PnL."""
        position = await self.get_current_position()
        return position.unrealized_pnl if position else 0.0

    def get_realized_pnl(self) -> float:
        """Get total realized PnL."""
        return self.realized_pnl

    def get_active_order_count(self, sync: bool = False) -> int:
        """Get number of active orders.
        
        Args:
            sync: If True, reconcile with IB Gateway first (slower but accurate)
        """
        if sync:
            self._enforce_order_lock_timeout()
            # Note: self.ib.openTrades() returns cached state, which is updated
            # automatically by ib_insync when running in an async event loop.
            # No need to call ib.sleep() which can conflict with asyncio.
            
            # Do a quick sync with IB to ensure accuracy
            open_trades = self.ib.openTrades()
            synced_orders: Dict[int, Trade] = {}
            current_time = datetime.utcnow()
            pending_statuses = {'PreSubmitted', 'Submitted', 'PendingSubmit'}
            stuck_threshold = max(30, self._pending_order_timeout_seconds)
            
            symbol_trades = [t for t in open_trades if t.contract.symbol == self.symbol]
            submitted_trades = [t for t in symbol_trades if t.orderStatus.status in pending_statuses]
            pending_submit_trades = [t for t in symbol_trades if t.orderStatus.status == 'PendingSubmit']
            
            logger.info(
                "🔍 Sync: %d active orders (%d PendingSubmit, lock=%s)",
                len(submitted_trades),
                len(pending_submit_trades),
                f"{self._order_lock_reason} {self._order_lock_age_seconds():.0f}s"
                if self._order_locked
                else "idle",
            )
            
            order_snapshots: List[str] = []
            for trade in symbol_trades:
                status = trade.orderStatus.status
                order_id = trade.order.orderId
                created_at = self.order_creation_times.get(order_id)
                age_seconds = (current_time - created_at).total_seconds() if created_at else None
                age_display = f"{age_seconds:.1f}s" if age_seconds is not None else "unknown"
                order_snapshots.append(
                    f"#{order_id} {trade.order.action} {trade.order.totalQuantity} status={status} age={age_display}"
                )
                
                if status in pending_statuses:
                    if (
                        age_seconds is not None
                        and age_seconds > stuck_threshold
                        and status in {'PendingSubmit', 'Submitted'}
                    ):
                        logger.warning(
                            "⚠️  Order %s stuck in %s for %.1fs (threshold=%ss) – canceling",
                            order_id,
                            status,
                            age_seconds,
                            stuck_threshold,
                        )
                        self._cancel_trade(trade, "pending_status_watchdog")
                        if self._order_locked:
                            self._release_order_lock("stuck order canceled during sync")
                        continue
                    synced_orders[order_id] = trade
                else:
                    logger.debug("Skipping order %s with status: %s", order_id, status)
            
            if order_snapshots:
                logger.info("   Active order detail: %s", "; ".join(order_snapshots))
            
            # ALWAYS update to match IB's state (this fixes the sync issue)
            old_count = len(self.active_orders)
            old_ids = list(self.active_orders.keys())
            self.active_orders = synced_orders
            if not synced_orders and self._order_locked:
                self._release_order_lock("active order sync cleared state")
            
            # Log sync result for debugging
            if old_count != len(synced_orders):
                logger.info(f"🔄 Order count sync: {old_count} → {len(synced_orders)} active orders")
                if old_count > 0 and len(synced_orders) == 0:
                    logger.info(f"   ✅ Cleared stale orders: {old_ids}")
                if synced_orders:
                    logger.info(f"   Active order IDs: {list(synced_orders.keys())}")
            elif synced_orders:
                # Log even when count is same but periodically
                logger.debug(f"Order sync: {len(synced_orders)} orders remain active: {list(synced_orders.keys())}")
        
        return len(self.active_orders)

    def get_order_history(self, limit: int = 100) -> List[Dict]:
        """Get recent order history."""
        return self.order_history[-limit:]

    async def close_position(self) -> OrderResult | None:
        """Close current position at market."""
        position = await self.get_current_position()
        if not position or position.quantity == 0:
            logger.info("No position to close")
            return None
        
        action = "SELL" if position.quantity > 0 else "BUY"
        quantity = abs(position.quantity)
        
        logger.info("Closing position: {action} {quantity} contracts", action=action, quantity=quantity)
        return await self.place_order(action, quantity)

    async def update_trailing_stops(self, current_price: float, current_atr: float | None = None) -> bool:
        """
        Update trailing stops for open positions (matching backtest logic).
        Returns True if stop was updated.
        """
        position = self.positions.get(self.symbol)
        if not position or position.quantity == 0:
            return False
        
        if not position.trailing_atr_multiplier and not position.trailing_percent:
            # No trailing configured
            return False
        
        direction = 1 if position.quantity > 0 else -1
        entry_price = position.avg_cost
        current_stop = position.stop_loss
        
        # Use stored ATR if current not provided
        atr_value = current_atr if current_atr else position.atr_value
        
        new_stop = None
        
        # ATR-based trailing (matches backtest)
        if position.trailing_atr_multiplier and atr_value and atr_value > 0:
            trail_distance = atr_value * position.trailing_atr_multiplier
            
            if direction > 0:  # Long position
                potential_stop = current_price - trail_distance
                if current_stop is None or potential_stop > current_stop:
                    new_stop = potential_stop
            else:  # Short position
                potential_stop = current_price + trail_distance
                if current_stop is None or potential_stop < current_stop:
                    new_stop = potential_stop
        
        # Percent-based trailing (matches backtest)
        if position.trailing_percent and position.trailing_percent > 0:
            profit = (current_price - entry_price) * direction
            if profit > 0:
                trail_price = entry_price + direction * profit * (1 - position.trailing_percent)
                
                if direction > 0:
                    potential_stop = max(current_stop or trail_price, trail_price)
                    if current_stop is None or potential_stop > current_stop:
                        new_stop = potential_stop
                else:
                    potential_stop = min(current_stop or trail_price, trail_price)
                    if current_stop is None or potential_stop < current_stop:
                        new_stop = potential_stop
        
        # Update stop if changed
        if new_stop and new_stop != current_stop:
            # Cancel existing stop order and place new one
            # Find the stop order in active orders
            stop_order_id = None
            for order_id, trade in self.active_orders.items():
                if isinstance(trade.order, StopOrder):
                    stop_order_id = order_id
                    break
            
            if stop_order_id:
                await self.cancel_order(stop_order_id)
            
            # Place new stop order
            opposite = "SELL" if direction > 0 else "BUY"
            quantity = abs(position.quantity)
            
            # Get qualified contract
            contract = await self.get_qualified_contract()
            if not contract:
                logger.error("Failed to get qualified contract for trailing stop")
                return False
            
            sl_order = StopOrder(opposite, quantity, new_stop)
            sl_order.transmit = True
            
            stop_trade = self.ib.placeOrder(contract, sl_order)
            self.active_orders[stop_trade.order.orderId] = stop_trade
            
            # Update position info
            position.stop_loss = new_stop
            
            logger.info(
                "Trailing stop updated: {old:.2f} -> {new:.2f} (price={price:.2f})",
                old=current_stop or 0,
                new=new_stop,
                price=current_price,
            )
            return True
        
        return False

    async def _place_emergency_stop(
        self,
        order_id: int,
        entry_price: float,
        entry_action: str,
        quantity: int,
        expected_stop_loss: float,
        trade_cycle_id: Optional[str] = None,
    ) -> None:
        """Place emergency stop loss when entry fill occurs without bracket protection.
        
        This is a safety mechanism to ensure positions are never left unprotected.
        Idempotent: checks if stop already exists before placing.
        """
        try:
            logger.warning(
                f"🚨 Placing EMERGENCY STOP for order {order_id}: "
                f"{entry_action} {quantity} @ {entry_price}, SL={expected_stop_loss}"
            )
            
            # Check if stop already exists (idempotent check)
            current_position = await self.get_current_position()
            if not current_position or current_position.quantity == 0:
                logger.warning("⚠️ No position found - emergency stop not needed")
                return
            
            # Check if we already have an active stop order
            all_trades = self.ib.trades()
            has_active_stop = False
            for t in all_trades:
                if isinstance(t.order, (StopOrder, StopLimitOrder)):
                    if t.orderStatus.status not in ("Filled", "Cancelled", "Inactive"):
                        # Check if this stop is for our position
                        if abs(t.order.totalQuantity) == abs(current_position.quantity):
                            has_active_stop = True
                            logger.info(f"✅ Active stop order {t.order.orderId} already exists")
                            break
            
            if has_active_stop:
                logger.info("✅ Emergency stop not needed - active stop already exists")
                return
            
            # Determine stop action (opposite of entry)
            stop_action = "SELL" if entry_action == "BUY" else "BUY"
            
            # Get contract
            contract = await self.get_qualified_contract()
            if not contract:
                logger.error("❌ Cannot place emergency stop - no contract available")
                return
            
            # Place stop limit order (more reliable than stop market)
            tick_size = self.config.tick_size
            offset_ticks = 4  # Allow 1 point slippage
            
            if entry_action == "BUY":  # Long position, stop below
                limit_price = expected_stop_loss - (offset_ticks * tick_size)
            else:  # Short position, stop above
                limit_price = expected_stop_loss + (offset_ticks * tick_size)
            
            sl_order = StopLimitOrder(stop_action, quantity, expected_stop_loss, limit_price)
            sl_order.transmit = True
            sl_order.outsideRth = True
            
            stop_trade = self.ib.placeOrder(contract, sl_order)
            stop_order_id = stop_trade.order.orderId
            
            self.active_orders[stop_order_id] = stop_trade
            self.order_creation_times[stop_order_id] = datetime.utcnow()
            
            # Record emergency stop in tracker
            self.order_tracker.record_order_placement(
                order_id=stop_order_id,
                parent_order_id=order_id,
                symbol=self.symbol,
                action=stop_action,
                quantity=quantity,
                order_type="STOP_LIMIT",
                stop_price=expected_stop_loss,
                limit_price=limit_price,
                trade_cycle_id=trade_cycle_id,
            )
            
            logger.info(
                f"✅ EMERGENCY STOP placed: order {stop_order_id} "
                f"({stop_action} {quantity} @ stop={expected_stop_loss:.2f}, limit={limit_price:.2f})"
            )
            
            # Send alert
            if self.telegram and self.telegram.enabled:
                try:
                    self.telegram.send_message(
                        f"🚨 EMERGENCY STOP placed for unprotected position:\n"
                        f"Entry: {entry_action} {quantity} @ {entry_price:.2f} (order {order_id})\n"
                        f"Stop: {stop_action} {quantity} @ {expected_stop_loss:.2f} (order {stop_order_id})"
                    )
                except Exception:
                    pass  # Don't fail on Telegram errors
            
        except Exception as e:
            logger.critical(f"❌ CRITICAL: Failed to place emergency stop: {e}")
            import traceback
            logger.critical(traceback.format_exc())

    async def get_current_price(self) -> float | None:
        """Get current market price for the contract."""
        import asyncio
        self._record_metric(self._price_snapshot_timestamps, "snapshot_price_requests", self._snapshot_warning_threshold)
        now = time.time()
        if (
            self._last_price_value is not None
            and self._last_price_snapshot
            and now - self._last_price_snapshot < self._price_snapshot_min_interval
        ):
            logger.debug("Using cached price snapshot %.2f", self._last_price_value)
            return self._last_price_value
        try:
            # Check connection health before making API call
            if not self.ib.isConnected():
                logger.warning("IB connection lost, attempting to reconnect...")
                try:
                    await self.ib.connectAsync(
                        self._connection_host, 
                        self._connection_port, 
                        clientId=self._connection_client_id, 
                        timeout=30
                    )
                    logger.info("Reconnection successful")
                except Exception as reconnect_error:
                    logger.error(f"Reconnection failed: {reconnect_error}")
                    return None
            
            # Get the qualified front month contract
            front_month = await self.get_qualified_contract()
            if not front_month:
                logger.warning("Could not get qualified contract")
                return None
            
            logger.info(f"Requesting market data for {front_month.localSymbol}...")
            
            # Request market data snapshot - snapshot=True auto-cancels after first update
            # No manual cancellation needed for snapshots
            ticker = self.ib.reqMktData(front_month, snapshot=True)
            derived_price: Optional[float] = None
            
            # Wait for ticker to populate with data
            for attempt in range(10):  # Try for up to 5 seconds
                await asyncio.sleep(0.5)
                
                logger.debug(f"Attempt {attempt+1}/10: last={ticker.last}, close={ticker.close}, bid={ticker.bid}, ask={ticker.ask}")
                
                # Check if we have any price data
                if ticker.last and ticker.last > 0:
                    logger.info(f"Got last price: {ticker.last}")
                    derived_price = float(ticker.last)
                    break
                elif ticker.close and ticker.close > 0:
                    logger.info(f"Got close price: {ticker.close}")
                    derived_price = float(ticker.close)
                    break
                elif ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                    midpoint = (ticker.bid + ticker.ask) / 2
                    logger.info(f"Got bid/ask: {ticker.bid}/{ticker.ask}, midpoint: {midpoint}")
                    derived_price = float(midpoint)
                    break
            if derived_price is not None:
                self._last_price_value = derived_price
                self._last_price_snapshot = time.time()
                return derived_price
            
            # Timeout - no data received
            logger.warning(f"Timeout waiting for price data. Ticker state: last={ticker.last}, close={ticker.close}, bid={ticker.bid}, ask={ticker.ask}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get current price: {e}")
            logger.exception("Full traceback:")
            return None

    async def _start_keepalive(self) -> None:
        """Start background keepalive task to monitor connection health."""
        import asyncio
        
        if self._keepalive_task is not None:
            logger.warning("Keepalive task already running")
            return
        
        async def keepalive_loop():
            """Background task that checks connection every 30 seconds."""
            logger.info("Starting connection keepalive task (checking every 30s)")
            while True:
                try:
                    await asyncio.sleep(30)
                    
                    if not self.ib.isConnected():
                        logger.warning("⚠️  Connection lost, attempting auto-reconnect...")
                        
                        # Clear stale orders since we lost connection
                        if self.active_orders:
                            logger.warning(f"🧹 Clearing {len(self.active_orders)} stale orders due to disconnection")
                            self.active_orders.clear()
                            self.order_creation_times.clear()
                        
                        try:
                            await self.ib.connectAsync(
                                self._connection_host,
                                self._connection_port,
                                clientId=self._connection_client_id,
                                timeout=30
                            )
                            # Re-setup after reconnection (no reqMarketDataType needed for snapshots)
                            self.ib.orderStatusEvent += self._on_order_status
                            self.ib.execDetailsEvent += self._on_execution
                            logger.info("✅ Auto-reconnection successful")
                            
                            # Reconcile orders with IB after reconnection
                            await self._reconcile_orders()
                            await self._reconcile_positions()
                            
                        except Exception as reconnect_error:
                            logger.error(f"❌ Auto-reconnection failed: {reconnect_error}")
                    else:
                        # Connection is healthy, just log periodically
                        logger.debug("Connection health check: OK")
                        
                except Exception as e:
                    logger.error(f"Error in keepalive loop: {e}")
                    
        # Start the background task
        self._keepalive_task = asyncio.create_task(keepalive_loop())
        logger.info("Keepalive task started")

    async def stop_keepalive(self) -> None:
        """Stop the keepalive background task."""
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None
            logger.info("Keepalive task stopped")
    
    def clear_active_orders(self):
        """Manually clear all tracked active orders (one-time fix)."""
        self.active_orders.clear()
