"""
Mock IB Server for Local Testing
=================================
Simulates IB Gateway responses for testing reconciliation and live data
without requiring a real IB connection.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
from unittest.mock import MagicMock, AsyncMock


@dataclass
class MockContract:
    """Mock IB Contract."""
    symbol: str = "ES"
    secType: str = "FUT"
    exchange: str = "CME"
    currency: str = "USD"
    localSymbol: str = "ESZ4"
    lastTradeDateOrContractMonth: str = "20251219"
    conId: int = 12345


@dataclass
class MockOrder:
    """Mock IB Order."""
    orderId: int = 0
    permId: int = 0
    clientId: int = 1
    action: str = "BUY"
    totalQuantity: int = 1
    orderType: str = "LMT"
    lmtPrice: float = 0.0
    auxPrice: float = 0.0  # Stop price
    transmit: bool = True
    parentId: int = 0
    orderRef: str = ""
    
    def __post_init__(self):
        if self.permId == 0:
            self.permId = random.randint(100000, 999999)


@dataclass
class MockOrderStatus:
    """Mock IB Order Status."""
    status: str = "Submitted"
    filled: int = 0
    remaining: int = 1
    avgFillPrice: float = 0.0
    permId: int = 0
    parentId: int = 0
    lastFillPrice: float = 0.0
    clientId: int = 1
    whyHeld: str = ""


@dataclass
class MockTrade:
    """Mock IB Trade (Order + Status + Contract)."""
    contract: MockContract = field(default_factory=MockContract)
    order: MockOrder = field(default_factory=MockOrder)
    orderStatus: MockOrderStatus = field(default_factory=MockOrderStatus)
    fills: List[Any] = field(default_factory=list)
    log: List[Any] = field(default_factory=list)


@dataclass
class MockPosition:
    """Mock IB Position."""
    account: str = "DU123456"
    contract: MockContract = field(default_factory=MockContract)
    position: int = 0
    avgCost: float = 0.0
    
    @property
    def unrealizedPNL(self) -> float:
        return 0.0


@dataclass
class MockExecution:
    """Mock IB Execution."""
    orderId: int = 0
    permId: int = 0
    clientId: int = 1
    symbol: str = "ES"
    side: str = "BOT"
    shares: int = 1
    price: float = 0.0
    time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MockFill:
    """Mock IB Fill."""
    contract: MockContract = field(default_factory=MockContract)
    execution: MockExecution = field(default_factory=MockExecution)
    commissionReport: Any = None


@dataclass
class MockTicker:
    """Mock IB Ticker for market data."""
    contract: MockContract = field(default_factory=MockContract)
    bid: float = 0.0
    bidSize: int = 0
    ask: float = 0.0
    askSize: int = 0
    last: float = 0.0
    lastSize: int = 0
    volume: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0


class MockIB:
    """
    Mock Interactive Brokers connection for testing.
    
    Simulates IB Gateway behavior including:
    - Open trades/orders
    - Positions
    - Fills
    - Market data
    - Event callbacks
    """
    
    def __init__(self):
        self._connected = False
        self._next_order_id = 1000
        
        # State
        self._open_trades: Dict[int, MockTrade] = {}
        self._positions: List[MockPosition] = []
        self._fills: List[MockFill] = []
        self._market_data_type = 3  # Delayed by default
        
        # Event callbacks
        self.orderStatusEvent = EventEmitter()
        self.execDetailsEvent = EventEmitter()
        self.disconnectedEvent = EventEmitter()
        self.connectedEvent = EventEmitter()
        self.errorEvent = EventEmitter()
        self.pendingTickersEvent = EventEmitter()
        
        # Tickers for market data
        self._tickers: Dict[str, MockTicker] = {}
        self._market_data_subscriptions: Set[str] = set()
        
        # Simulation settings
        self.simulate_fills = True
        self.fill_delay_seconds = 0.1
        self.simulate_price_movement = True
        self.base_price = 5000.0  # Base ES price
        
    def isConnected(self) -> bool:
        """Check if connected."""
        return self._connected
    
    async def connectAsync(
        self, 
        host: str, 
        port: int, 
        clientId: int, 
        timeout: int = 30
    ) -> None:
        """Simulate async connection."""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self._connected = True
        self.connectedEvent.emit()
    
    def disconnect(self) -> None:
        """Disconnect."""
        self._connected = False
        self.disconnectedEvent.emit()
    
    def reqMarketDataType(self, dataType: int) -> None:
        """Set market data type."""
        self._market_data_type = dataType
    
    def positions(self) -> List[MockPosition]:
        """Get all positions."""
        return self._positions
    
    def openTrades(self) -> List[MockTrade]:
        """Get all open trades."""
        return list(self._open_trades.values())
    
    def fills(self) -> List[MockFill]:
        """Get all fills."""
        return self._fills
    
    async def qualifyContractsAsync(self, *contracts) -> List[MockContract]:
        """Qualify contracts."""
        await asyncio.sleep(0.05)
        return [MockContract(symbol=c.symbol) for c in contracts]
    
    async def reqContractDetailsAsync(self, contract) -> List[Any]:
        """Get contract details."""
        await asyncio.sleep(0.05)
        mock_details = MagicMock()
        mock_details.contract = MockContract(symbol=contract.symbol)
        return [mock_details]
    
    def placeOrder(self, contract: MockContract, order: MockOrder) -> MockTrade:
        """Place an order."""
        # Assign order ID if not set
        if order.orderId == 0:
            order.orderId = self._next_order_id
            self._next_order_id += 1
        
        # Create trade
        status = MockOrderStatus(
            status="Submitted",
            filled=0,
            remaining=int(order.totalQuantity),
            permId=order.permId,
        )
        
        trade = MockTrade(
            contract=contract,
            order=order,
            orderStatus=status,
        )
        
        self._open_trades[order.orderId] = trade
        
        # Emit order status event
        self.orderStatusEvent.emit(trade)
        
        # Schedule fill simulation if enabled
        if self.simulate_fills and order.transmit:
            asyncio.create_task(self._simulate_fill(trade))
        
        return trade
    
    async def _simulate_fill(self, trade: MockTrade) -> None:
        """Simulate order fill after delay."""
        await asyncio.sleep(self.fill_delay_seconds)
        
        # Update status to filled
        fill_price = trade.order.lmtPrice if trade.order.lmtPrice > 0 else self.base_price
        
        trade.orderStatus.status = "Filled"
        trade.orderStatus.filled = int(trade.order.totalQuantity)
        trade.orderStatus.remaining = 0
        trade.orderStatus.avgFillPrice = fill_price
        
        # Create fill
        execution = MockExecution(
            orderId=trade.order.orderId,
            permId=trade.order.permId,
            symbol=trade.contract.symbol,
            side="BOT" if trade.order.action == "BUY" else "SLD",
            shares=int(trade.order.totalQuantity),
            price=fill_price,
        )
        
        fill = MockFill(
            contract=trade.contract,
            execution=execution,
        )
        
        self._fills.append(fill)
        
        # Emit events
        self.orderStatusEvent.emit(trade)
        self.execDetailsEvent.emit(trade, fill)
        
        # Update position
        self._update_position(trade.contract.symbol, trade.order.action, int(trade.order.totalQuantity), fill_price)
        
        # Remove from open trades
        if trade.order.orderId in self._open_trades:
            del self._open_trades[trade.order.orderId]
    
    def _update_position(self, symbol: str, action: str, quantity: int, price: float) -> None:
        """Update position after fill."""
        # Find existing position
        for pos in self._positions:
            if pos.contract.symbol == symbol:
                if action == "BUY":
                    pos.position += quantity
                else:
                    pos.position -= quantity
                # Update avg cost (simplified)
                pos.avgCost = price * abs(pos.position)
                return
        
        # Create new position
        qty = quantity if action == "BUY" else -quantity
        self._positions.append(MockPosition(
            contract=MockContract(symbol=symbol),
            position=qty,
            avgCost=price * abs(qty),
        ))
    
    def cancelOrder(self, order: MockOrder) -> None:
        """Cancel an order."""
        if order.orderId in self._open_trades:
            trade = self._open_trades[order.orderId]
            trade.orderStatus.status = "Cancelled"
            self.orderStatusEvent.emit(trade)
            del self._open_trades[order.orderId]
    
    def reqMktData(self, contract: MockContract, genericTicks: str = '', snapshot: bool = False, regulatorySnapshot: bool = False) -> MockTicker:
        """Request market data."""
        symbol = contract.symbol
        
        if symbol not in self._tickers:
            # Create new ticker with simulated prices
            price = self.base_price + random.uniform(-10, 10)
            spread = 0.25
            
            self._tickers[symbol] = MockTicker(
                contract=contract,
                bid=price,
                bidSize=100,
                ask=price + spread,
                askSize=100,
                last=price + spread/2,
                lastSize=1,
                volume=10000,
            )
        
        if not snapshot:
            self._market_data_subscriptions.add(symbol)
        
        return self._tickers[symbol]
    
    def cancelMktData(self, contract: MockContract) -> None:
        """Cancel market data subscription."""
        symbol = contract.symbol
        if symbol in self._market_data_subscriptions:
            self._market_data_subscriptions.remove(symbol)
    
    def reqMktDepth(self, contract: MockContract, numRows: int = 5) -> None:
        """Request market depth (no-op in mock)."""
        pass
    
    # =========================================================================
    # Test Setup Methods
    # =========================================================================
    
    def add_open_order(
        self,
        order_id: int,
        symbol: str = "ES",
        action: str = "BUY",
        quantity: int = 1,
        limit_price: float = 5000.0,
        status: str = "Submitted",
    ) -> MockTrade:
        """Add an open order for testing."""
        contract = MockContract(symbol=symbol)
        order = MockOrder(
            orderId=order_id,
            action=action,
            totalQuantity=quantity,
            orderType="LMT",
            lmtPrice=limit_price,
        )
        order_status = MockOrderStatus(
            status=status,
            filled=0,
            remaining=quantity,
        )
        
        trade = MockTrade(
            contract=contract,
            order=order,
            orderStatus=order_status,
        )
        
        self._open_trades[order_id] = trade
        return trade
    
    def add_position(
        self,
        symbol: str = "ES",
        quantity: int = 1,
        avg_cost: float = 5000.0,
    ) -> MockPosition:
        """Add a position for testing."""
        pos = MockPosition(
            contract=MockContract(symbol=symbol),
            position=quantity,
            avgCost=avg_cost * abs(quantity),
        )
        self._positions.append(pos)
        return pos
    
    def clear_state(self) -> None:
        """Clear all state."""
        self._open_trades.clear()
        self._positions.clear()
        self._fills.clear()
        self._tickers.clear()
        self._market_data_subscriptions.clear()
    
    def set_base_price(self, price: float) -> None:
        """Set base price for simulations."""
        self.base_price = price
        
        # Update existing tickers
        for ticker in self._tickers.values():
            spread = ticker.ask - ticker.bid
            ticker.bid = price
            ticker.ask = price + spread
            ticker.last = price + spread/2


class EventEmitter:
    """Simple event emitter for callbacks."""
    
    def __init__(self):
        self._callbacks: List[Callable] = []
    
    def __iadd__(self, callback: Callable):
        """Add callback with += operator."""
        self._callbacks.append(callback)
        return self
    
    def __isub__(self, callback: Callable):
        """Remove callback with -= operator."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
        return self
    
    def emit(self, *args, **kwargs):
        """Emit event to all callbacks."""
        for callback in self._callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Error in event callback: {e}")


# =========================================================================
# Factory Functions for Testing
# =========================================================================

def create_mock_ib_with_orders(orders: List[Dict[str, Any]]) -> MockIB:
    """
    Create a MockIB with pre-configured orders.
    
    Args:
        orders: List of dicts with keys: order_id, symbol, action, quantity, limit_price, status
    
    Returns:
        Configured MockIB instance
    """
    ib = MockIB()
    for order in orders:
        ib.add_open_order(
            order_id=order.get("order_id", random.randint(1, 999)),
            symbol=order.get("symbol", "ES"),
            action=order.get("action", "BUY"),
            quantity=order.get("quantity", 1),
            limit_price=order.get("limit_price", 5000.0),
            status=order.get("status", "Submitted"),
        )
    return ib


def create_mock_ib_with_positions(positions: List[Dict[str, Any]]) -> MockIB:
    """
    Create a MockIB with pre-configured positions.
    
    Args:
        positions: List of dicts with keys: symbol, quantity, avg_cost
    
    Returns:
        Configured MockIB instance
    """
    ib = MockIB()
    for pos in positions:
        ib.add_position(
            symbol=pos.get("symbol", "ES"),
            quantity=pos.get("quantity", 1),
            avg_cost=pos.get("avg_cost", 5000.0),
        )
    return ib


def create_mock_ib_for_reconcile_test() -> MockIB:
    """
    Create a MockIB configured for reconciliation testing.
    
    Sets up:
    - Order A: Exists on both IB and DB (needs update)
    - Order B: Exists only on IB (needs insert)
    - Order C: Will be in DB only (needs delete)
    - ES position: SPY futures initial state
    """
    ib = MockIB()
    ib._connected = True
    
    # Order A - exists on both, status different
    ib.add_open_order(
        order_id=100,
        symbol="ES",
        action="BUY",
        quantity=1,
        limit_price=5000.0,
        status="Submitted",
    )
    
    # Order B - exists only on IB
    ib.add_open_order(
        order_id=200,
        symbol="ES",
        action="SELL",
        quantity=2,
        limit_price=5010.0,
        status="PreSubmitted",
    )
    
    # ES position - SPY futures
    ib.add_position(
        symbol="ES",
        quantity=1,
        avg_cost=4990.0,
    )
    
    return ib


# =========================================================================
# Async Test Helpers
# =========================================================================

async def run_mock_ib_scenario(
    ib: MockIB,
    duration_seconds: float = 5.0,
    tick_interval: float = 0.5,
) -> None:
    """
    Run a mock IB scenario with simulated price movements.
    
    Args:
        ib: MockIB instance
        duration_seconds: How long to run
        tick_interval: Time between tick updates
    """
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        if ib.simulate_price_movement:
            # Update prices randomly
            ib.base_price += random.uniform(-2, 2)
            
            # Update all tickers
            for symbol, ticker in ib._tickers.items():
                spread = 0.25
                ticker.bid = ib.base_price
                ticker.ask = ib.base_price + spread
                ticker.last = ib.base_price + spread/2
                ticker.lastSize = random.randint(1, 10)
            
            # Emit pending tickers event
            if ib._market_data_subscriptions:
                active_tickers = {
                    ib._tickers[s] for s in ib._market_data_subscriptions 
                    if s in ib._tickers
                }
                if active_tickers:
                    ib.pendingTickersEvent.emit(active_tickers)
        
        await asyncio.sleep(tick_interval)


if __name__ == "__main__":
    # Simple test
    async def main():
        ib = create_mock_ib_for_reconcile_test()
        
        print("Open trades:", len(ib.openTrades()))
        for trade in ib.openTrades():
            print(f"  Order {trade.order.orderId}: {trade.order.action} {trade.order.totalQuantity} @ {trade.order.lmtPrice}")
        
        print("\nPositions:", len(ib.positions()))
        for pos in ib.positions():
            print(f"  {pos.contract.symbol}: {pos.position} @ {pos.avgCost}")
        
        # Test market data
        ticker = ib.reqMktData(MockContract(symbol="ES"))
        print(f"\nTicker: bid={ticker.bid}, ask={ticker.ask}, last={ticker.last}")
    
    asyncio.run(main())
