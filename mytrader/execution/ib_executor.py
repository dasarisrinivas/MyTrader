"""Trade execution via Interactive Brokers with reconciliation support."""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from ib_insync import Contract, Future, IB, LimitOrder, MarketOrder, Order, StopOrder, StopLimitOrder, Trade

from ..config import TradingConfig
from ..monitoring.order_tracker import OrderTracker
from ..utils.logger import logger
from ..utils.telegram_notifier import TelegramNotifier

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
        self.realized_pnl = 0.0
        self.order_history: List[Dict] = []
        self._qualified_contract: Contract | None = None  # Cache the qualified front month contract
        
        # Store stop loss / take profit for each order (for Telegram notifications)
        self.order_targets: Dict[int, Dict[str, float]] = {}  # {order_id: {'stop_loss': float, 'take_profit': float}}
        
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
        
        # NEW: Initial state tracking (orders found on IB at startup)
        self._initial_state_order_ids: set = set()
        
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
        price: Optional[float],
    ) -> str:
        """
        Generate idempotency signature for order submission.
        Prevents duplicate orders during reconnects/duplicate events.
        """
        timestamp_rounded = (datetime.now(timezone.utc).timestamp() // 10) * 10
        data = f"{self.symbol}|{quantity}|{price or 'MKT'}|{action}|{timestamp_rounded}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    def is_duplicate_submission(self, signature: str) -> bool:
        """Check if a submission signature is a duplicate."""
        if signature in self._submission_signatures:
            # Check if signature is still valid (within 60 seconds)
            created_at = self._submission_signatures[signature]
            if (datetime.now(timezone.utc) - created_at).total_seconds() < 60:
                return True
            else:
                # Expired, remove it
                del self._submission_signatures[signature]
        return False
    
    def record_submission(self, signature: str) -> None:
        """Record a submission signature for idempotency."""
        self._submission_signatures[signature] = datetime.now(timezone.utc)
        
        # Clean up old signatures (older than 5 minutes)
        cutoff = datetime.now(timezone.utc)
        old_sigs = [
            sig for sig, ts in self._submission_signatures.items()
            if (cutoff - ts).total_seconds() > 300
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
            
            # Sort by expiration date to get the front month (earliest expiration)
            details.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
            front_month = details[0].contract
            logger.info(f"Qualified contract: {front_month.localSymbol} (exp: {front_month.lastTradeDateOrContractMonth})")
            
            # Cache it
            self._qualified_contract = front_month
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
        
        logger.info("Connecting executor to IBKR %s:%s (client_id=%d, timeout=%ds)", 
                   host, port, client_id, timeout)
        try:
            await self.ib.connectAsync(host, port, clientId=client_id, timeout=timeout)
        except TimeoutError:
            logger.error("Connection timeout after %ds. IB Gateway may be in bad state.", timeout)
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
                        self.active_orders[order_id] = trade.order
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
            logger.error("Failed to reconcile positions: %s", e)

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
        commission = fill.commissionReport.commission if hasattr(fill, 'commissionReport') else None
        realized_pnl = None
        
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
            logger.info(f"✅ Execution: Order {order_id}, qty={quantity}, price={price:.2f}, commission={commission}")
        
        # Update realized PnL
        if hasattr(fill, 'commissionReport') and hasattr(fill.commissionReport, 'realizedPNL'):
            realized_pnl = float(fill.commissionReport.realizedPNL)
            self.realized_pnl += realized_pnl
            logger.info(f"💰 Realized PnL: {realized_pnl:.2f} (total: {self.realized_pnl:.2f})")
        
        # Record execution in tracker
        self.order_tracker.record_execution(
            order_id=order_id,
            quantity=quantity,
            price=price,
            commission=commission,
            realized_pnl=realized_pnl
        )
        
        # Reconcile positions after execution to get updated position from IB
        self._reconcile_positions()
        
        # Send Telegram notification (non-blocking)
        logger.info(f"📱 Checking Telegram: enabled={self.telegram.enabled if self.telegram else False}")
        if self.telegram and self.telegram.enabled:
            try:
                # Get current position
                current_position = self.positions.get(self.symbol)
                position_qty = current_position.quantity if current_position else None
                
                # Get stop loss and take profit from stored order targets
                targets = self.order_targets.get(order_id, {})
                stop_loss = targets.get('stop_loss')
                take_profit = targets.get('take_profit')
                
                # Determine side from order action
                side = order.action  # "BUY" or "SELL"
                
                logger.info(f"📱 Sending Telegram alert: {side} {quantity} @ {price}, SL={stop_loss}, TP={take_profit}")
                # Send notification in background (fire-and-forget)
                self.telegram.send_trade_alert_background(
                    symbol=self.symbol,
                    side=side,
                    quantity=quantity,
                    fill_price=price,
                    timestamp=datetime.utcnow(),
                    current_position=position_qty,
                    order_id=order_id,
                    commission=commission,
                    realized_pnl=realized_pnl,
                    stop_loss=stop_loss,
                    take_profit=take_profit
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
    ) -> OrderResult:
        """Place an order with optional bracket orders for risk management.
        
        NEW: Integrates with reconciliation safety features:
        - Checks reconcile_lock before placing orders
        - Respects SAFE_MODE to prevent trading
        - Uses idempotency signatures to prevent duplicates
        """
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
        submission_sig = self.generate_submission_signature(action, quantity, limit_price)
        if self.is_duplicate_submission(submission_sig):
            logger.warning(f"Duplicate order submission blocked: {action} {quantity} @ {limit_price}")
            self._log_reconcile_event("duplicate_submission_blocked", {
                "action": action, "quantity": quantity, "limit_price": limit_price,
                "signature": submission_sig
            }, "WARN")
            from ib_insync import Trade as IBTrade
            dummy_trade = IBTrade()
            return OrderResult(trade=dummy_trade, status="Cancelled", message="Duplicate submission blocked")
        
        # Check connection health before placing order
        if not self.ib.isConnected():
            logger.error("IB not connected - cannot place order")
            from ib_insync import Trade as IBTrade
            dummy_trade = IBTrade()
            return OrderResult(trade=dummy_trade, status="Cancelled", message="Not connected to IB")
        
        # Check if we have too many active orders (IB limit is 15 per side)
        open_trades = self.ib.openTrades()
        active_count = sum(1 for t in open_trades if t.contract.symbol == self.symbol and t.orderStatus.status in ('PreSubmitted', 'Submitted'))
        
        if active_count >= 12:  # Stay below the 15 limit with some buffer
            logger.warning(f"⚠️  Too many active orders ({active_count}), canceling old orders first...")
            await self._cancel_all_existing_orders()
            await asyncio.sleep(2)  # Wait for cancellations
        
        # Get the qualified front month contract
        contract = await self.get_qualified_contract()
        if not contract:
            logger.error("Failed to get qualified contract - cannot place order")
            # Return a failed order result
            from ib_insync import Trade as IBTrade
            dummy_trade = IBTrade()
            return OrderResult(trade=dummy_trade, status="Cancelled", message="Failed to qualify contract")
        
        # NEW: Record submission signature for idempotency
        self.record_submission(submission_sig)
        
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
            logger.warning(f"⛔ Order rejected by PositionManager: {decision.reason}")
            from ib_insync import Trade as IBTrade
            dummy_trade = IBTrade()
            return OrderResult(trade=dummy_trade, status="Cancelled", message=f"Rejected: {decision.reason}")
            
        if abs(decision.allowed_contracts) < quantity:
            logger.warning(f"⚠️ Order size reduced by PositionManager: {quantity} -> {abs(decision.allowed_contracts)} ({decision.reason})")
            quantity = abs(decision.allowed_contracts)
            # Update signed quantity for logic below if needed, though we just use 'quantity'
        
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
            
            # Use 2-tick buffer for ES (0.25 * 2 = 0.50 points)
            # This ensures fill while avoiding market orders
            tick_buffer = 0.50  # 2 ticks for ES
            limit_price = current_price + tick_buffer if action == "BUY" else current_price - tick_buffer
            order = LimitOrder(action, quantity, limit_price)
            logger.info(f"📊 Using LIMIT order @ {limit_price:.2f} (market: {current_price:.2f}, buffer: {tick_buffer})")

        # Add metadata to order reference for tracking
        if metadata:
            order.orderRef = f"MyTrader_{metadata.get('signal_source', 'manual')}"
        
        bracket_children: list[Order] = []
        if stop_loss is not None or take_profit is not None:
            order.transmit = False
            opposite = "SELL" if action == "BUY" else "BUY"
            
            if take_profit is not None:
                tp_order = LimitOrder(opposite, quantity, take_profit)
                tp_order.transmit = False
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
                bracket_children.append(sl_order)
                logger.info(f"Adding stop-loss order: stop={stop_loss:.2f}, limit={limit_price_sl:.2f} (STOP-LIMIT with {abs(stop_loss - limit_price_sl):.2f} buffer)")
            
            if bracket_children:
                bracket_children[-1].transmit = True
        else:
            order.transmit = True

        # Place parent order
        parent_trade = self.ib.placeOrder(contract, order)
        parent_id = parent_trade.order.orderId
        
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
            market_regime=market_regime
        )
        
        # Track active order
        self.active_orders[parent_id] = parent_trade
        
        # Place bracket orders
        for child in bracket_children:
            child.parentId = parent_id
        for i, child in enumerate(bracket_children):
            child_trade = self.ib.placeOrder(contract, child)
            child_id = child_trade.order.orderId
            self.active_orders[child_id] = child_trade
            
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
            )
            
            logger.info(f"📝 Placed bracket order {child_id} ({child_type}) (parent={parent_id})")

        # Wait briefly for order status updates (use asyncio.sleep instead of blocking waitOnUpdate)
        await asyncio.sleep(0.1)  # Small delay to allow order status callbacks to fire
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
        
        logger.info("Order %s placed: orderId=%d status=%s", 
                   action, parent_id, status)
        
        return OrderResult(
            trade=parent_trade, 
            status=status,
            fill_price=float(parent_trade.orderStatus.avgFillPrice) if parent_trade.orderStatus.avgFillPrice > 0 else None,
            filled_quantity=int(parent_trade.orderStatus.filled)
        )

    async def cancel_order(self, order_id: int) -> bool:
        """Cancel a specific order by ID."""
        if order_id not in self.active_orders:
            logger.warning("Order %d not found in active orders", order_id)
            return False
        
        try:
            trade = self.active_orders[order_id]
            self.ib.cancelOrder(trade.order)
            self.ib.waitOnUpdate()  # waitOnUpdate is synchronous, not async
            logger.info("Cancelled order %d", order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel order %d: %s", order_id, e)
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel all active orders."""
        count = 0
        for order_id in list(self.active_orders.keys()):
            if await self.cancel_order(order_id):
                count += 1
        logger.info("Cancelled %d orders", count)
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
            # Do a quick sync with IB to ensure accuracy
            open_trades = self.ib.openTrades()
            synced_orders = {}
            current_time = datetime.utcnow()
            stuck_order_threshold_minutes = 60  # Cancel orders stuck for > 60 minutes
            
            for trade in open_trades:
                if trade.contract.symbol == self.symbol:
                    status = trade.orderStatus.status
                    order_id = trade.order.orderId
                    
                    # Log what IB is reporting
                    logger.debug(f"IB reports order {order_id}: {trade.order.action} {trade.order.totalQuantity} @ {getattr(trade.order, 'lmtPrice', 'MKT')}, status={status}")
                    
                    if status in ['PreSubmitted', 'Submitted', 'PendingSubmit']:
                        # Check if order is stuck (been active too long)
                        if order_id in self.order_creation_times:
                            order_age = (current_time - self.order_creation_times[order_id]).total_seconds() / 60
                            if order_age > stuck_order_threshold_minutes:
                                logger.warning(f"⚠️  Order {order_id} has been active for {order_age:.1f} minutes, canceling stuck order")
                                try:
                                    self.ib.cancelOrder(trade.order)
                                    logger.info(f"✅ Canceled stuck order {order_id}")
                                    continue  # Don't add to synced_orders
                                except Exception as e:
                                    logger.error(f"Failed to cancel stuck order {order_id}: {e}")
                        
                        synced_orders[order_id] = trade.order
            
            # ALWAYS update to match IB's state (this fixes the sync issue)
            old_count = len(self.active_orders)
            self.active_orders = synced_orders
            
            # Log sync result for debugging
            if old_count != len(synced_orders):
                logger.info(f"🔄 Order count sync: {old_count} → {len(synced_orders)} active orders")
                if synced_orders:
                    logger.info(f"   Active order IDs: {list(synced_orders.keys())}")
                elif old_count > 0:
                    logger.info(f"   ✅ Cleared {old_count} stale orders from tracking")
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
        
        logger.info("Closing position: %s %d contracts", action, quantity)
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
            
            logger.info("Trailing stop updated: %.2f -> %.2f (price=%.2f)", 
                       current_stop or 0, new_stop, current_price)
            return True
        
        return False

    async def get_current_price(self) -> float | None:
        """Get current market price for the contract."""
        import asyncio
        
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
            
            # Wait for ticker to populate with data
            for attempt in range(10):  # Try for up to 5 seconds
                await asyncio.sleep(0.5)
                
                logger.debug(f"Attempt {attempt+1}/10: last={ticker.last}, close={ticker.close}, bid={ticker.bid}, ask={ticker.ask}")
                
                # Check if we have any price data
                if ticker.last and ticker.last > 0:
                    logger.info(f"Got last price: {ticker.last}")
                    return float(ticker.last)
                elif ticker.close and ticker.close > 0:
                    logger.info(f"Got close price: {ticker.close}")
                    return float(ticker.close)
                elif ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                    midpoint = (ticker.bid + ticker.ask) / 2
                    logger.info(f"Got bid/ask: {ticker.bid}/{ticker.ask}, midpoint: {midpoint}")
                    return float(midpoint)
            
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


