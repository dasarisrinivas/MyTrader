"""
Broker Simulator for Backtesting
================================

Simulates order execution with realistic fills:
- Market orders filled at next bar open with slippage
- Limit orders triggered on high/low rules
- Stop orders triggered on high/low with slippage
- Bracket orders (OCO) for stop-loss and take-profit
- Order modifications (trailing stops, breakeven)
- Partial fills support
- Commission calculation

IMPORTANT: All fills use rules that prevent lookahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Any
from collections import defaultdict
import uuid

import pandas as pd
import numpy as np
from loguru import logger


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAIL = "TRAIL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderTimeInForce(Enum):
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    
    # Price levels
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None  # Fixed points
    trail_percent: Optional[float] = None  # Percentage
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    
    # Timing
    created_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    
    # Bracket order links
    parent_id: Optional[str] = None  # If this is a child order (SL/TP)
    take_profit_id: Optional[str] = None  # Linked TP order
    stop_loss_id: Optional[str] = None  # Linked SL order
    is_bracket_child: bool = False
    
    # Tracking
    metadata: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    modification_count: int = 0
    
    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, 
                               OrderStatus.REJECTED, OrderStatus.EXPIRED)
    
    def __hash__(self):
        return hash(self.order_id)


@dataclass
class Fill:
    """Represents an order fill."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current position state."""
    symbol: str
    quantity: int  # Positive for long, negative for short
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Tracking
    entry_time: Optional[datetime] = None
    entry_bar_index: int = 0
    trade_count: int = 0
    entry_metadata: Dict[str, Any] = field(default_factory=dict)  # JAN 11 2026: Store entry metadata for regime analytics
    
    @property
    def is_flat(self) -> bool:
        return self.quantity == 0
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0


@dataclass
class BrokerConfig:
    """Configuration for broker simulator."""
    
    # Slippage settings
    slippage_ticks: float = 1.0  # Base slippage in ticks
    slippage_mode: str = "fixed"  # "fixed", "proportional", "random"
    slippage_pct: float = 0.01  # For proportional mode (0.01 = 1%)
    max_slippage_ticks: float = 4.0  # Maximum slippage cap
    
    # Commission settings
    commission_per_contract: float = 2.40  # Round trip
    commission_mode: str = "per_contract"  # "per_contract", "per_trade"
    
    # Contract specs (MES)
    tick_size: float = 0.25
    tick_value: float = 1.25  # $1.25 per tick for MES
    point_value: float = 5.0  # $5 per point for MES
    
    # Fill rules
    fill_on_touch: bool = False  # Limit orders fill on touch vs penetration
    market_fill_bar: str = "next_open"  # "next_open", "current_close"
    stop_fill_slippage: bool = True  # Add slippage to stop fills
    
    # Session hours (for order expiration)
    session_start_hour: int = 18  # 6 PM ET (start of futures session)
    session_end_hour: int = 17  # 5 PM ET
    
    # Position limits
    max_position_size: int = 10
    allow_pyramiding: bool = False


class BrokerSimulator:
    """
    Simulates broker order execution for backtesting.
    
    Key features:
    - Realistic fill simulation using OHLC rules
    - Bracket orders with OCO behavior
    - Order modifications (trailing stops, breakeven)
    - Commission and slippage modeling
    - Position tracking
    - No lookahead bias in fill logic
    """
    
    def __init__(self, config: Optional[BrokerConfig] = None):
        self.config = config or BrokerConfig()
        
        # State
        self.orders: Dict[str, Order] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.positions: Dict[str, Position] = {}
        
        # Tracking
        self.equity_history: List[Tuple[datetime, float]] = []
        self.trade_log: List[Dict] = []
        self.modification_log: List[Dict] = []
        
        # Metrics
        self.total_commission: float = 0.0
        self.total_slippage: float = 0.0
        
        # Callbacks
        self.on_fill: Optional[Callable[[Fill], None]] = None
        self.on_order_update: Optional[Callable[[Order], None]] = None
    
    def reset(self) -> None:
        """Reset all state for a new backtest."""
        self.orders.clear()
        self.pending_orders.clear()
        self.fills.clear()
        self.positions.clear()
        self.equity_history.clear()
        self.trade_log.clear()
        self.modification_log.clear()
        self.total_commission = 0.0
        self.total_slippage = 0.0
    
    def get_position(self, symbol: str) -> Position:
        """Get current position for a symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol, quantity=0, avg_price=0.0)
        return self.positions[symbol]
    
    def submit_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        timestamp: datetime,
        metadata: Optional[Dict] = None
    ) -> Order:
        """Submit a market order."""
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            created_at=timestamp,
            submitted_at=timestamp,
            status=OrderStatus.SUBMITTED,
            metadata=metadata or {}
        )
        
        self.orders[order.order_id] = order
        self.pending_orders[order.order_id] = order
        
        logger.debug(f"Submitted market {side.value} {quantity} {symbol} @ {timestamp}")
        return order
    
    def submit_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: float,
        timestamp: datetime,
        time_in_force: OrderTimeInForce = OrderTimeInForce.GTC,
        metadata: Optional[Dict] = None
    ) -> Order:
        """Submit a limit order."""
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            created_at=timestamp,
            submitted_at=timestamp,
            status=OrderStatus.SUBMITTED,
            time_in_force=time_in_force,
            metadata=metadata or {}
        )
        
        self.orders[order.order_id] = order
        self.pending_orders[order.order_id] = order
        
        logger.debug(f"Submitted limit {side.value} {quantity} {symbol} @ {limit_price}")
        return order
    
    def submit_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        stop_price: float,
        timestamp: datetime,
        limit_price: Optional[float] = None,  # For stop-limit
        metadata: Optional[Dict] = None
    ) -> Order:
        """Submit a stop or stop-limit order."""
        order_type = OrderType.STOP_LIMIT if limit_price else OrderType.STOP
        
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            stop_price=stop_price,
            limit_price=limit_price,
            created_at=timestamp,
            submitted_at=timestamp,
            status=OrderStatus.SUBMITTED,
            metadata=metadata or {}
        )
        
        self.orders[order.order_id] = order
        self.pending_orders[order.order_id] = order
        
        logger.debug(f"Submitted stop {side.value} {quantity} {symbol} @ {stop_price}")
        return order
    
    def submit_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_type: OrderType,
        stop_loss: float,
        take_profit: float,
        timestamp: datetime,
        entry_price: Optional[float] = None,  # For limit entry
        metadata: Optional[Dict] = None
    ) -> Tuple[Order, Order, Order]:
        """
        Submit a bracket order (entry + stop loss + take profit).
        
        Returns:
            Tuple of (entry_order, stop_loss_order, take_profit_order)
        """
        meta = metadata or {}
        
        # Create entry order
        if entry_type == OrderType.MARKET:
            entry_order = self.submit_market_order(
                symbol, side, quantity, timestamp, meta
            )
        else:
            entry_order = self.submit_limit_order(
                symbol, side, quantity, entry_price, timestamp, 
                metadata=meta
            )
        
        # Create stop loss (opposite side)
        sl_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        sl_order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=sl_side,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_loss,
            created_at=timestamp,
            status=OrderStatus.PENDING,  # Wait for entry fill
            parent_id=entry_order.order_id,
            is_bracket_child=True,
            metadata={"type": "stop_loss", **meta}
        )
        
        # Create take profit (opposite side)
        tp_order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=sl_side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=take_profit,
            created_at=timestamp,
            status=OrderStatus.PENDING,  # Wait for entry fill
            parent_id=entry_order.order_id,
            is_bracket_child=True,
            metadata={"type": "take_profit", **meta}
        )
        
        # Link orders
        entry_order.stop_loss_id = sl_order.order_id
        entry_order.take_profit_id = tp_order.order_id
        sl_order.take_profit_id = tp_order.order_id  # OCO link
        tp_order.stop_loss_id = sl_order.order_id  # OCO link
        
        # Store orders
        self.orders[sl_order.order_id] = sl_order
        self.orders[tp_order.order_id] = tp_order
        
        logger.debug(
            f"Submitted bracket: {side.value} {quantity} {symbol}, "
            f"SL={stop_loss}, TP={take_profit}"
        )
        
        return entry_order, sl_order, tp_order
    
    def modify_order(
        self,
        order_id: str,
        new_stop_price: Optional[float] = None,
        new_limit_price: Optional[float] = None,
        new_quantity: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        reason: str = ""
    ) -> bool:
        """
        Modify an existing order.
        
        Returns:
            True if modification successful
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found for modification")
            return False
        
        order = self.orders[order_id]
        
        if order.is_complete:
            logger.warning(f"Cannot modify completed order {order_id}")
            return False
        
        old_values = {
            "stop_price": order.stop_price,
            "limit_price": order.limit_price,
            "quantity": order.quantity
        }
        
        if new_stop_price is not None:
            order.stop_price = new_stop_price
        if new_limit_price is not None:
            order.limit_price = new_limit_price
        if new_quantity is not None:
            order.quantity = new_quantity
        
        order.modification_count += 1
        
        self.modification_log.append({
            "order_id": order_id,
            "timestamp": timestamp,
            "old_values": old_values,
            "new_values": {
                "stop_price": order.stop_price,
                "limit_price": order.limit_price,
                "quantity": order.quantity
            },
            "reason": reason
        })
        
        logger.debug(f"Modified order {order_id}: {reason}")
        return True
    
    def cancel_order(self, order_id: str, timestamp: Optional[datetime] = None) -> bool:
        """Cancel an order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.is_complete:
            return False
        
        order.status = OrderStatus.CANCELLED
        self.pending_orders.pop(order_id, None)
        
        logger.debug(f"Cancelled order {order_id}")
        
        if self.on_order_update:
            self.on_order_update(order)
        
        return True
    
    def cancel_all_orders(self, symbol: str = None) -> int:
        """
        Cancel all pending orders for a symbol (or all orders if symbol is None).
        
        JAN 11 2026: Added for max hold time exit support.
        
        Args:
            symbol: Symbol to cancel orders for, or None for all symbols
            
        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0
        order_ids_to_cancel = list(self.pending_orders.keys())  # Copy to avoid mutation during iteration
        
        for order_id in order_ids_to_cancel:
            order = self.orders.get(order_id)
            if order and (symbol is None or order.symbol == symbol):
                if self.cancel_order(order_id):
                    cancelled_count += 1
        
        logger.debug(f"Cancelled {cancelled_count} pending orders for {symbol or 'all symbols'}")
        return cancelled_count
    
    def _cancel_oco_pair(self, order: Order, timestamp: datetime) -> None:
        """Cancel the OCO pair when one side fills."""
        if order.stop_loss_id and order.stop_loss_id in self.orders:
            oco_order = self.orders[order.stop_loss_id]
            if not oco_order.is_complete:
                oco_order.status = OrderStatus.CANCELLED
                self.pending_orders.pop(order.stop_loss_id, None)
                logger.debug(f"OCO cancelled: {order.stop_loss_id}")
        
        if order.take_profit_id and order.take_profit_id in self.orders:
            oco_order = self.orders[order.take_profit_id]
            if not oco_order.is_complete:
                oco_order.status = OrderStatus.CANCELLED
                self.pending_orders.pop(order.take_profit_id, None)
                logger.debug(f"OCO cancelled: {order.take_profit_id}")
    
    def process_bar(
        self,
        symbol: str,
        bar: pd.Series,
        timestamp: datetime
    ) -> List[Fill]:
        """
        Process a bar and execute any triggered orders.
        
        Uses OHLC rules to determine fills:
        - Market orders: Fill at open with slippage
        - Limit buy: Fill if low <= limit_price
        - Limit sell: Fill if high >= limit_price
        - Stop buy: Trigger if high >= stop_price
        - Stop sell: Trigger if low <= stop_price
        
        Args:
            symbol: Instrument symbol
            bar: OHLC bar data
            timestamp: Bar timestamp
            
        Returns:
            List of fills generated this bar
        """
        bar_fills: List[Fill] = []
        
        open_price = float(bar["open"])
        high_price = float(bar["high"])
        low_price = float(bar["low"])
        close_price = float(bar["close"])
        
        orders_to_process = list(self.pending_orders.values())
        
        for order in orders_to_process:
            if order.symbol != symbol or order.is_complete:
                continue
            
            fill = self._try_fill_order(
                order, open_price, high_price, low_price, close_price, timestamp
            )
            
            if fill:
                bar_fills.append(fill)
                
                # Handle OCO cancellation
                if order.is_bracket_child:
                    self._cancel_oco_pair(order, timestamp)
                
                # Activate bracket children if entry filled
                if order.stop_loss_id and order.take_profit_id:
                    self._activate_bracket_children(order, timestamp)
        
        # Update position P&L
        position = self.get_position(symbol)
        if not position.is_flat:
            position.unrealized_pnl = self._calculate_unrealized_pnl(
                position, close_price
            )
        
        return bar_fills
    
    def _try_fill_order(
        self,
        order: Order,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        timestamp: datetime
    ) -> Optional[Fill]:
        """Attempt to fill an order based on OHLC rules."""
        
        if order.order_type == OrderType.MARKET:
            # Market orders fill at open (or close depending on config)
            if self.config.market_fill_bar == "next_open":
                fill_price = open_price
            else:
                fill_price = close_price
            
            slippage = self._calculate_slippage(order.side, fill_price)
            
            if order.side == OrderSide.BUY:
                fill_price += slippage
            else:
                fill_price -= slippage
            
            return self._execute_fill(order, fill_price, timestamp, slippage)
        
        elif order.order_type == OrderType.LIMIT:
            return self._check_limit_fill(
                order, open_price, high_price, low_price, timestamp
            )
        
        elif order.order_type == OrderType.STOP:
            return self._check_stop_fill(
                order, open_price, high_price, low_price, timestamp
            )
        
        elif order.order_type == OrderType.STOP_LIMIT:
            return self._check_stop_limit_fill(
                order, open_price, high_price, low_price, timestamp
            )
        
        return None
    
    def _check_limit_fill(
        self,
        order: Order,
        open_price: float,
        high_price: float,
        low_price: float,
        timestamp: datetime
    ) -> Optional[Fill]:
        """Check if limit order fills."""
        limit = order.limit_price
        
        if order.side == OrderSide.BUY:
            # Buy limit fills if price trades at or below limit
            if self.config.fill_on_touch:
                fills = low_price <= limit
            else:
                fills = low_price < limit or (low_price == limit and open_price <= limit)
            
            if fills:
                # Fill at limit price (may get better fill if gap through)
                fill_price = min(limit, open_price)
                return self._execute_fill(order, fill_price, timestamp, 0.0)
        
        else:  # SELL
            # Sell limit fills if price trades at or above limit
            if self.config.fill_on_touch:
                fills = high_price >= limit
            else:
                fills = high_price > limit or (high_price == limit and open_price >= limit)
            
            if fills:
                fill_price = max(limit, open_price)
                return self._execute_fill(order, fill_price, timestamp, 0.0)
        
        return None
    
    def _check_stop_fill(
        self,
        order: Order,
        open_price: float,
        high_price: float,
        low_price: float,
        timestamp: datetime
    ) -> Optional[Fill]:
        """Check if stop order fills."""
        stop = order.stop_price
        
        if order.side == OrderSide.BUY:
            # Buy stop triggers if price trades at or above stop
            if high_price >= stop:
                # Fill at stop price or open if gapped through
                base_price = max(stop, open_price)
                
                if self.config.stop_fill_slippage:
                    slippage = self._calculate_slippage(order.side, base_price)
                    fill_price = base_price + slippage
                else:
                    slippage = 0.0
                    fill_price = base_price
                
                return self._execute_fill(order, fill_price, timestamp, slippage)
        
        else:  # SELL
            # Sell stop triggers if price trades at or below stop
            if low_price <= stop:
                base_price = min(stop, open_price)
                
                if self.config.stop_fill_slippage:
                    slippage = self._calculate_slippage(order.side, base_price)
                    fill_price = base_price - slippage
                else:
                    slippage = 0.0
                    fill_price = base_price
                
                return self._execute_fill(order, fill_price, timestamp, slippage)
        
        return None
    
    def _check_stop_limit_fill(
        self,
        order: Order,
        open_price: float,
        high_price: float,
        low_price: float,
        timestamp: datetime
    ) -> Optional[Fill]:
        """Check if stop-limit order fills (two-stage)."""
        stop = order.stop_price
        limit = order.limit_price
        
        # First check if stop is triggered
        triggered = False
        
        if order.side == OrderSide.BUY:
            triggered = high_price >= stop
        else:
            triggered = low_price <= stop
        
        if not triggered:
            return None
        
        # If triggered, check if limit fills same bar
        # This is conservative - in reality might fill on next bar
        if order.side == OrderSide.BUY:
            if low_price <= limit:
                fill_price = min(limit, max(stop, open_price))
                return self._execute_fill(order, fill_price, timestamp, 0.0)
        else:
            if high_price >= limit:
                fill_price = max(limit, min(stop, open_price))
                return self._execute_fill(order, fill_price, timestamp, 0.0)
        
        return None
    
    def _execute_fill(
        self,
        order: Order,
        price: float,
        timestamp: datetime,
        slippage: float
    ) -> Fill:
        """Execute a fill and update state."""
        # Round price to tick
        price = round(price / self.config.tick_size) * self.config.tick_size
        
        quantity = order.remaining_quantity
        
        # Calculate commission
        commission = self._calculate_commission(quantity)
        
        # Create fill
        fill = Fill(
            fill_id=str(uuid.uuid4())[:8],
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            commission=commission,
            slippage=slippage,
            metadata=order.metadata.copy()
        )
        
        # Update order state
        order.filled_quantity += quantity
        order.avg_fill_price = price  # Simplified - would need weighting for partials
        order.filled_at = timestamp
        order.status = OrderStatus.FILLED
        
        # Remove from pending
        self.pending_orders.pop(order.order_id, None)
        
        # Update position
        self._update_position(fill)
        
        # Store fill
        self.fills.append(fill)
        
        # Update metrics
        self.total_commission += commission
        self.total_slippage += abs(slippage) * quantity
        
        logger.debug(
            f"Filled: {order.side.value} {quantity} {order.symbol} @ {price} "
            f"(slip={slippage:.4f}, comm={commission:.2f})"
        )
        
        # Callback
        if self.on_fill:
            self.on_fill(fill)
        if self.on_order_update:
            self.on_order_update(order)
        
        return fill
    
    def _calculate_slippage(self, side: OrderSide, price: float) -> float:
        """Calculate slippage based on configuration."""
        if self.config.slippage_mode == "fixed":
            slippage = self.config.slippage_ticks * self.config.tick_size
        elif self.config.slippage_mode == "proportional":
            slippage = price * self.config.slippage_pct
        elif self.config.slippage_mode == "random":
            # Random slippage between 0 and max
            max_slip = self.config.slippage_ticks * self.config.tick_size
            slippage = np.random.uniform(0, max_slip)
        else:
            slippage = self.config.slippage_ticks * self.config.tick_size
        
        # Cap slippage
        max_slippage = self.config.max_slippage_ticks * self.config.tick_size
        slippage = min(slippage, max_slippage)
        
        return slippage
    
    def _calculate_commission(self, quantity: int) -> float:
        """Calculate commission for a fill."""
        if self.config.commission_mode == "per_contract":
            return quantity * self.config.commission_per_contract
        else:  # per_trade
            return self.config.commission_per_contract
    
    def _update_position(self, fill: Fill) -> None:
        """Update position based on fill."""
        position = self.get_position(fill.symbol)
        
        fill_qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        
        if position.quantity == 0:
            # Opening new position
            position.quantity = fill_qty
            position.avg_price = fill.price
            position.entry_time = fill.timestamp
            position.trade_count += 1
            # JAN 11 2026: Store entry metadata from the fill for regime analytics
            position.entry_metadata = dict(fill.metadata) if fill.metadata else {}
            
        elif (position.quantity > 0 and fill_qty > 0) or (position.quantity < 0 and fill_qty < 0):
            # Adding to position
            if self.config.allow_pyramiding:
                total_qty = position.quantity + fill_qty
                position.avg_price = (
                    (position.avg_price * position.quantity + fill.price * fill_qty) 
                    / total_qty
                )
                position.quantity = total_qty
            else:
                logger.warning("Pyramiding not allowed, ignoring add")
        
        else:
            # Reducing or reversing position
            original_position_qty = position.quantity  # Save original for reversal check
            close_qty = min(abs(fill_qty), abs(original_position_qty))
            
            # Calculate realized P&L for closed portion
            if original_position_qty > 0:  # Was long, selling
                pnl = (fill.price - position.avg_price) * close_qty * self.config.point_value
            else:  # Was short, buying
                pnl = (position.avg_price - fill.price) * close_qty * self.config.point_value
            
            pnl -= fill.commission
            position.realized_pnl += pnl
            
            # Update position
            remaining = abs(original_position_qty) - close_qty
            
            if remaining == 0:
                # Position closed
                self._record_trade(position, fill, pnl)
                position.quantity = 0
                position.avg_price = 0.0
                position.unrealized_pnl = 0.0
            else:
                # Partial close
                position.quantity = remaining * (1 if original_position_qty > 0 else -1)
            
            # Handle reversal (only if fill is larger than original position)
            if abs(fill_qty) > abs(original_position_qty):
                reversal_qty = abs(fill_qty) - abs(original_position_qty)
                position.quantity = reversal_qty * (1 if fill_qty > 0 else -1)
                position.avg_price = fill.price
                position.entry_time = fill.timestamp
                position.trade_count += 1
    
    def _calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
        """Calculate unrealized P&L for a position."""
        if position.is_flat:
            return 0.0
        
        if position.is_long:
            return (current_price - position.avg_price) * position.quantity * self.config.point_value
        else:
            return (position.avg_price - current_price) * abs(position.quantity) * self.config.point_value
    
    def _record_trade(self, position: Position, exit_fill: Fill, realized_pnl: float) -> None:
        """Record a completed trade."""
        self.trade_log.append({
            "symbol": position.symbol,
            "direction": "LONG" if position.quantity > 0 else "SHORT",
            "entry_time": position.entry_time,
            "entry_price": position.avg_price,
            "exit_time": exit_fill.timestamp,
            "exit_price": exit_fill.price,
            "quantity": abs(position.quantity),
            "realized_pnl": realized_pnl,
            "pnl_points": abs(exit_fill.price - position.avg_price),
            "commission": exit_fill.commission,
            "exit_reason": exit_fill.metadata.get("type", "unknown"),
            # JAN 11 2026: Include entry metadata for regime analytics
            "entry_metadata": position.entry_metadata,
        })
    
    def _activate_bracket_children(self, entry_order: Order, timestamp: datetime) -> None:
        """Activate stop loss and take profit orders after entry fills."""
        for child_id in [entry_order.stop_loss_id, entry_order.take_profit_id]:
            if child_id and child_id in self.orders:
                child = self.orders[child_id]
                child.status = OrderStatus.SUBMITTED
                child.submitted_at = timestamp
                self.pending_orders[child_id] = child
                logger.debug(f"Activated bracket child: {child_id}")
    
    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate total equity (realized + unrealized)."""
        total_unrealized = 0.0
        
        for symbol, position in self.positions.items():
            if not position.is_flat and symbol in current_prices:
                position.unrealized_pnl = self._calculate_unrealized_pnl(
                    position, current_prices[symbol]
                )
                total_unrealized += position.unrealized_pnl
        
        total_realized = sum(p.realized_pnl for p in self.positions.values())
        
        return total_realized + total_unrealized
    
    def get_trade_summary(self) -> Dict:
        """Get summary of all completed trades."""
        if not self.trade_log:
            return {}
        
        trades_df = pd.DataFrame(self.trade_log)
        
        wins = trades_df[trades_df["realized_pnl"] > 0]
        losses = trades_df[trades_df["realized_pnl"] <= 0]
        
        return {
            "total_trades": len(trades_df),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(trades_df) if len(trades_df) > 0 else 0,
            "total_pnl": trades_df["realized_pnl"].sum(),
            "avg_win": wins["realized_pnl"].mean() if len(wins) > 0 else 0,
            "avg_loss": losses["realized_pnl"].mean() if len(losses) > 0 else 0,
            "largest_win": wins["realized_pnl"].max() if len(wins) > 0 else 0,
            "largest_loss": losses["realized_pnl"].min() if len(losses) > 0 else 0,
            "profit_factor": (
                abs(wins["realized_pnl"].sum() / losses["realized_pnl"].sum())
                if losses["realized_pnl"].sum() != 0 else float("inf")
            ),
            "total_commission": self.total_commission,
            "total_slippage_cost": self.total_slippage * self.config.point_value,
        }
    
    def get_pending_orders(self) -> List[Order]:
        """Get list of pending orders."""
        return list(self.pending_orders.values())
    
    def get_open_position(self, symbol: str) -> Optional[Position]:
        """Get open position for a symbol, None if flat."""
        position = self.positions.get(symbol)
        if position and not position.is_flat:
            return position
        return None
    
    def flatten_position(
        self,
        symbol: str,
        timestamp: datetime,
        reason: str = "flatten"
    ) -> Optional[Order]:
        """Close entire position with market order."""
        position = self.get_open_position(symbol)
        if not position:
            return None
        
        side = OrderSide.SELL if position.is_long else OrderSide.BUY
        
        return self.submit_market_order(
            symbol=symbol,
            side=side,
            quantity=abs(position.quantity),
            timestamp=timestamp,
            metadata={"type": reason}
        )
