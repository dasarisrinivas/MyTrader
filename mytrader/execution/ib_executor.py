"""Trade execution via Interactive Brokers."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from ib_insync import Contract, Future, IB, LimitOrder, MarketOrder, Order, StopOrder, Trade

from ..config import TradingConfig
from ..utils.logger import logger


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


class TradeExecutor:
    """Enhanced trade executor with real-time PnL tracking and order monitoring."""
    
    def __init__(
        self, 
        ib: IB, 
        config: TradingConfig, 
        symbol: str, 
        exchange: str = "GLOBEX", 
        currency: str = "USD"
    ) -> None:
        self.ib = ib
        self.config = config
        self.symbol = symbol
        self.exchange = exchange
        self.currency = currency
        self.active_orders: Dict[int, Trade] = {}
        self.positions: Dict[str, PositionInfo] = {}
        self.realized_pnl = 0.0
        self.order_history: List[Dict] = []

    def contract(self) -> Contract:
        return Future(symbol=self.symbol, exchange=self.exchange, currency=self.currency)

    async def connect(self, host: str, port: int, client_id: int) -> None:
        """Connect to IBKR and set up event handlers."""
        if self.ib.isConnected():
            return
        logger.info("Connecting executor to IBKR %s:%s", host, port)
        await self.ib.connectAsync(host, port, clientId=client_id)
        
        # Set up event handlers for order updates
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_execution
        
        # Request initial positions
        await self._reconcile_positions()

    async def _reconcile_positions(self) -> None:
        """Reconcile current positions with IBKR."""
        try:
            positions = self.ib.positions()
            for position in positions:
                if position.contract.symbol == self.symbol:
                    self.positions[self.symbol] = PositionInfo(
                        symbol=self.symbol,
                        quantity=int(position.position),
                        avg_cost=float(position.avgCost),
                        market_value=float(position.position * position.avgCost),
                        unrealized_pnl=float(position.unrealizedPNL) if hasattr(position, 'unrealizedPNL') else 0.0,
                        realized_pnl=0.0
                    )
                    logger.info("Reconciled position: %s qty=%d avg_cost=%.2f", 
                               self.symbol, position.position, position.avgCost)
        except Exception as e:
            logger.error("Failed to reconcile positions: %s", e)

    def _on_order_status(self, trade: Trade) -> None:
        """Callback for order status updates."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        
        logger.info("Order %d status update: %s", order_id, status)
        
        if status in ("Filled", "Cancelled", "Inactive"):
            if order_id in self.active_orders:
                del self.active_orders[order_id]
        
        # Log to history
        self.order_history.append({
            "timestamp": datetime.utcnow(),
            "order_id": order_id,
            "status": status,
            "filled": trade.orderStatus.filled,
            "remaining": trade.orderStatus.remaining,
            "avg_fill_price": trade.orderStatus.avgFillPrice
        })

    def _on_execution(self, trade: Trade, fill) -> None:
        """Callback for execution details."""
        logger.info("Execution: order_id=%d qty=%d price=%.2f", 
                   trade.order.orderId, fill.execution.shares, fill.execution.price)
        
        # Update realized PnL
        if hasattr(fill.commissionReport, 'realizedPNL'):
            self.realized_pnl += float(fill.commissionReport.realizedPNL)
            logger.info("Realized PnL updated: %.2f (total: %.2f)", 
                       fill.commissionReport.realizedPNL, self.realized_pnl)

    async def place_order(
        self,
        action: str,
        quantity: int,
        limit_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrderResult:
        """Place an order with optional bracket orders for risk management."""
        contract = self.contract()
        await self.ib.qualifyContractsAsync(contract)

        order: Order
        if limit_price is not None:
            order = LimitOrder(action, quantity, limit_price)
        else:
            order = MarketOrder(action, quantity)

        bracket_children: list[Order] = []
        if stop_loss is not None or take_profit is not None:
            order.transmit = False
            opposite = "SELL" if action == "BUY" else "BUY"
            
            if take_profit is not None:
                tp_order = LimitOrder(opposite, quantity, take_profit)
                tp_order.transmit = False
                bracket_children.append(tp_order)
                logger.info("Adding take-profit order at %.2f", take_profit)
            
            if stop_loss is not None:
                sl_order = StopOrder(opposite, quantity, stop_loss)
                sl_order.transmit = False
                bracket_children.append(sl_order)
                logger.info("Adding stop-loss order at %.2f", stop_loss)
            
            if bracket_children:
                bracket_children[-1].transmit = True
        else:
            order.transmit = True

        # Place parent order
        parent_trade = self.ib.placeOrder(contract, order)
        parent_id = parent_trade.order.orderId
        
        # Track active order
        self.active_orders[parent_id] = parent_trade
        
        # Place bracket orders
        for child in bracket_children:
            child.parentId = parent_id
        for i, child in enumerate(bracket_children):
            child_trade = self.ib.placeOrder(contract, child)
            self.active_orders[child_trade.order.orderId] = child_trade
            logger.info("Placed bracket order %d (parent=%d)", 
                       child_trade.order.orderId, parent_id)

        # Wait for initial status update
        await self.ib.waitOnUpdate()
        status = parent_trade.orderStatus.status
        
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
            await self.ib.waitOnUpdate()
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

    def get_active_order_count(self) -> int:
        """Get number of active orders."""
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
