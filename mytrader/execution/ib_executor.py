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
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_atr_multiplier: Optional[float] = None
    trailing_percent: Optional[float] = None
    atr_value: Optional[float] = None
    entry_metadata: Optional[Dict] = None


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
        self._qualified_contract: Contract | None = None  # Cache the qualified front month contract

    def contract(self) -> Contract:
        return Future(symbol=self.symbol, exchange=self.exchange, currency=self.currency)
    
    async def get_qualified_contract(self) -> Contract | None:
        """Get the fully qualified front month contract."""
        try:
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

    async def connect(self, host: str, port: int, client_id: int, timeout: int = 30) -> None:
        """Connect to IBKR and set up event handlers."""
        if self.ib.isConnected():
            return
        logger.info("Connecting executor to IBKR %s:%s (client_id=%d, timeout=%ds)", 
                   host, port, client_id, timeout)
        try:
            await self.ib.connectAsync(host, port, clientId=client_id, timeout=timeout)
        except TimeoutError:
            logger.error("Connection timeout after %ds. IB Gateway may be in bad state.", timeout)
            logger.error("Try restarting IB Gateway: Close it completely, wait 30s, then restart")
            raise
        
        # Request delayed market data (free, available for ES futures)
        self.ib.reqMarketDataType(3)  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen
        logger.info("Using delayed market data (15-min delay, free)")
        logger.info("To use live data: Subscribe to 'US Equity and Options Add-On Streaming Bundle' in IBKR Account Management > Market Data Subscriptions")
        
        # Set up event handlers for order updates
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_execution
        
        # Cancel all existing orders on startup to avoid "too many orders" error
        await self._cancel_all_existing_orders()
        
        # Request initial positions
        await self._reconcile_positions()
    
    async def _cancel_all_existing_orders(self) -> None:
        """Cancel all existing orders for this symbol on startup."""
        try:
            open_trades = self.ib.openTrades()
            canceled_count = 0
            for trade in open_trades:
                # Cancel orders for our symbol
                if trade.contract.symbol == self.symbol:
                    self.ib.cancelOrder(trade.order)
                    canceled_count += 1
                    logger.info(f"Canceling existing order {trade.order.orderId} for {self.symbol}")
            
            if canceled_count > 0:
                await self.ib.sleep(2)  # Give time for cancellations to process
                logger.info(f"✅ Canceled {canceled_count} existing orders for {self.symbol}")
            else:
                logger.info("No existing orders to cancel")
        except Exception as e:
            logger.error(f"Failed to cancel existing orders: {e}")

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
        metadata: Dict | None = None,
    ) -> OrderResult:
        """Place an order with optional bracket orders for risk management."""
        # Check if we have too many active orders (IB limit is 15 per side)
        open_trades = self.ib.openTrades()
        active_count = sum(1 for t in open_trades if t.contract.symbol == self.symbol and t.orderStatus.status in ('PreSubmitted', 'Submitted'))
        
        if active_count >= 12:  # Stay below the 15 limit with some buffer
            logger.warning(f"⚠️  Too many active orders ({active_count}), canceling old orders first...")
            await self._cancel_all_existing_orders()
            await self.ib.sleep(2)  # Wait for cancellations
        
        # Get the qualified front month contract
        contract = await self.get_qualified_contract()
        if not contract:
            logger.error("Failed to get qualified contract - cannot place order")
            # Return a failed order result
            from ib_insync import Trade as IBTrade
            dummy_trade = IBTrade()
            return OrderResult(trade=dummy_trade, status="Cancelled", message="Failed to qualify contract")

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
            # Get the qualified front month contract
            front_month = await self.get_qualified_contract()
            if not front_month:
                return None
            
            # Request market data snapshot - snapshot=True auto-cancels after first update
            # No manual cancellation needed for snapshots
            ticker = self.ib.reqMktData(front_month, snapshot=True)
            
            # Wait for ticker to populate with data
            for attempt in range(10):  # Try for up to 5 seconds
                await asyncio.sleep(0.5)
                
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


