#!/usr/bin/env python3
"""
AWS Agent Live Trading Integration

This script integrates the deployed AWS Bedrock Agents with the live trading bot.
It can run standalone or be imported into the existing trading system.

Usage:
    # Standalone test mode (no real orders)
    python bin/run_aws_agent_trading.py --test
    
    # Live mode with IBKR (simulation)
    python bin/run_aws_agent_trading.py --simulation
    
    # Live mode (real orders - BE CAREFUL)
    python bin/run_aws_agent_trading.py --live
"""

import argparse
import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from mytrader.config import Settings
from mytrader.aws import AgentInvoker, MarketSnapshotBuilder, PnLUpdater
from mytrader.utils.timezone_utils import now_cst, format_cst


class AWSAgentTradingIntegration:
    """Integrates AWS Bedrock Agents with live trading.
    
    This class:
    - Collects real-time market data from IBKR
    - Builds market snapshots for agents
    - Invokes Decision and Risk agents
    - Executes trades based on agent decisions
    """
    
    def __init__(
        self,
        settings: Settings,
        simulation_mode: bool = True,
        use_step_functions: bool = False,
    ):
        """Initialize AWS agent trading integration.
        
        Args:
            settings: Trading bot settings
            simulation_mode: If True, no real orders are placed
            use_step_functions: Use Step Functions for agent orchestration
        """
        self.settings = settings
        self.simulation_mode = simulation_mode
        self.use_step_functions = use_step_functions
        
        # Initialize components
        self.invoker: Optional[AgentInvoker] = None
        self.snapshot_builder: Optional[MarketSnapshotBuilder] = None
        self.pnl_updater: Optional[PnLUpdater] = None
        
        # IBKR components (will be initialized if available)
        self.ib = None
        self.executor = None
        
        # State
        self.running = False
        self._last_trade_time: Optional[datetime] = None
        self._cooldown_seconds = 300  # 5 minutes between trades
        
        if simulation_mode:
            logger.warning("🔶 SIMULATION MODE - No real orders will be placed")
    
    async def initialize(self) -> bool:
        """Initialize all components.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize AWS Agent Invoker from deployed config
            logger.info("Initializing AWS Agent Invoker...")
            self.invoker = AgentInvoker.from_deployed_config(
                use_step_functions=self.use_step_functions
            )
            logger.info("✅ AWS Agent Invoker initialized")
            
            # Initialize market snapshot builder
            self.snapshot_builder = MarketSnapshotBuilder()
            logger.info("✅ Market Snapshot Builder initialized")
            
            # Initialize P&L updater
            self.pnl_updater = PnLUpdater()
            logger.info("✅ P&L Updater initialized")
            
            # Initialize IBKR connection
            try:
                from ib_insync import IB
                from mytrader.execution.ib_executor import TradeExecutor
                import random
                
                self.ib = IB()
                
                # Get IBKR settings from config
                ibkr_host = getattr(self.settings.data, 'ibkr_host', '127.0.0.1')
                ibkr_port = getattr(self.settings.data, 'ibkr_port', 4002)
                # Use random client ID to avoid conflicts
                ibkr_client_id = random.randint(200, 299)
                
                logger.info(f"Connecting to IBKR at {ibkr_host}:{ibkr_port} (client_id={ibkr_client_id})...")
                
                await self.ib.connectAsync(
                    host=ibkr_host,
                    port=ibkr_port,
                    clientId=ibkr_client_id,
                )
                
                # Use trading config from settings
                self.executor = TradeExecutor(
                    self.ib,
                    self.settings.trading,  # Pass the trading config directly
                    self.settings.data.ibkr_symbol,
                    self.settings.data.ibkr_exchange,
                )
                
                logger.info(f"✅ IBKR connection established (symbol: {self.settings.data.ibkr_symbol})")
            except Exception as e:
                logger.warning(f"⚠️ IBKR connection failed: {e}")
                logger.warning("Running in offline mode - will use mock data")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    async def get_market_snapshot(self) -> Dict[str, Any]:
        """Get current market snapshot.
        
        Returns:
            Market snapshot dictionary for agent invocation
        """
        if self.executor and self.ib and self.ib.isConnected():
            try:
                # Get real data from IBKR
                current_price = await self.executor.get_current_price()
                
                if current_price:
                    # Store price history for indicator calculation
                    if not hasattr(self, '_price_history'):
                        self._price_history = []
                    
                    self._price_history.append({
                        'price': current_price,
                        'timestamp': datetime.now(timezone.utc)
                    })
                    
                    # Keep last 100 prices
                    if len(self._price_history) > 100:
                        self._price_history = self._price_history[-100:]
                    
                    # Calculate simple indicators from price history
                    prices = [p['price'] for p in self._price_history]
                    
                    # Simple RSI calculation (14-period)
                    rsi = self._calculate_simple_rsi(prices) if len(prices) > 14 else 50
                    
                    # EMAs
                    ema_9 = self._calculate_ema(prices, 9) if len(prices) > 9 else current_price
                    ema_20 = self._calculate_ema(prices, 20) if len(prices) > 20 else current_price
                    
                    # ATR approximation (use price range)
                    atr = self._calculate_simple_atr(prices) if len(prices) > 14 else 10
                    
                    # Determine trend
                    if ema_9 > ema_20 and current_price > ema_9:
                        trend = "UPTREND"
                    elif ema_9 < ema_20 and current_price < ema_9:
                        trend = "DOWNTREND"
                    else:
                        trend = "RANGE"
                    
                    # Determine volatility
                    if atr > 15:
                        volatility = "HIGH"
                    elif atr > 8:
                        volatility = "MED"
                    else:
                        volatility = "LOW"
                    
                    snapshot = self.snapshot_builder.build(
                        symbol=self.settings.data.ibkr_symbol,
                        price=current_price,
                        trend=trend,
                        volatility=volatility,
                        rsi=rsi,
                        atr=atr,
                        ema_9=ema_9,
                        ema_20=ema_20,
                        volume=50000,  # Would get from ticker
                    )
                    
                    logger.debug(f"Real market snapshot: price={current_price:.2f}, RSI={rsi:.1f}, trend={trend}")
                    return snapshot
                    
            except Exception as e:
                logger.warning(f"Error getting real market data: {e}")
        
        # Fall back to mock data
        snapshot = self.snapshot_builder.build_mock()
        logger.debug(f"Using mock snapshot: price={snapshot.get('price'):.2f}")
        return snapshot
    
    def _calculate_simple_rsi(self, prices: list, period: int = 14) -> float:
        """Calculate simple RSI from price list."""
        if len(prices) < period + 1:
            return 50
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, c) for c in changes[-period:]]
        losses = [abs(min(0, c)) for c in changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_ema(self, prices: list, period: int) -> float:
        """Calculate EMA from price list."""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calculate_simple_atr(self, prices: list, period: int = 14) -> float:
        """Calculate simple ATR approximation from price list."""
        if len(prices) < period:
            return 10
        
        # Use price changes as approximation
        changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        return sum(changes[-period:]) / period
    
    async def get_account_metrics(self) -> Dict[str, Any]:
        """Get current account metrics.
        
        Returns:
            Account metrics dictionary for risk evaluation
        """
        if self.pnl_updater and self.executor:
            # Get real account metrics
            position = await self.executor.get_current_position()
            position_qty = position.quantity if position else 0
            
            # Use PnLUpdater's get_account_metrics method
            return self.pnl_updater.get_account_metrics(current_position=position_qty)
        else:
            # Mock metrics for testing
            return {
                'current_pnl_today': 0,
                'current_position': 0,
                'losing_streak': 0,
                'trades_today': 0,
                'account_balance': self.settings.trading.initial_capital,
                'open_risk': 0,
            }
    
    def is_in_cooldown(self) -> bool:
        """Check if we're in trade cooldown period."""
        if not self._last_trade_time:
            return False
        
        elapsed = (now_cst() - self._last_trade_time).total_seconds()
        return elapsed < self._cooldown_seconds
    
    async def process_trading_cycle(self) -> Dict[str, Any]:
        """Process one trading cycle with AWS agents.
        
        Returns:
            Trading decision result
        """
        # Check cooldown
        if self.is_in_cooldown():
            elapsed = (now_cst() - self._last_trade_time).total_seconds()
            remaining = self._cooldown_seconds - elapsed
            logger.debug(f"⏳ Cooldown active: {remaining:.0f}s remaining")
            return {'decision': 'WAIT', 'reason': 'Cooldown active'}
        
        # Get market snapshot
        market_snapshot = await self.get_market_snapshot()
        account_metrics = await self.get_account_metrics()
        
        logger.info(f"📊 Market snapshot: price={market_snapshot.get('price')}, "
                    f"trend={market_snapshot.get('trend')}, vol={market_snapshot.get('volatility')}")
        
        # Invoke AWS agents for decision
        logger.info("🤖 Invoking AWS Bedrock Agents...")
        decision = self.invoker.get_trading_decision(
            market_snapshot=market_snapshot,
            account_metrics=account_metrics,
        )
        
        logger.info(
            f"📈 Agent Decision: {decision.get('decision')} "
            f"(confidence={decision.get('confidence', 0):.2%}, "
            f"allowed={decision.get('allowed_to_trade')})"
        )
        
        # Execute trade if approved
        if decision.get('allowed_to_trade') and decision.get('decision') in ['BUY', 'SELL']:
            await self._execute_trade(decision, market_snapshot)
        
        return decision
    
    async def _execute_trade(self, decision: Dict[str, Any], snapshot: Dict[str, Any]):
        """Execute a trade based on agent decision.
        
        Args:
            decision: Agent decision result
            snapshot: Market snapshot
        """
        action = decision.get('decision')
        size = max(1, int(decision.get('adjusted_size', 1)))
        stop_loss = decision.get('stop_loss')
        take_profit = decision.get('take_profit')
        
        if not self.executor:
            logger.warning("⚠️ No executor available - cannot execute trade")
            return
        
        try:
            current_price = snapshot.get('price', 0)
            
            logger.info(f"📍 Placing order: {action} {size} contracts @ ~{current_price:.2f}")
            
            # Place order with stop loss and take profit
            order_id = await self.executor.place_order(
                action=action,
                quantity=size,
                limit_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            
            logger.info(f"✅ Order placed: {action} {size} contracts (order_id={order_id})")
            
            self._last_trade_time = now_cst()
            
            # Upload trade data to S3 for learning
            await self._upload_trade_data(action, size, snapshot, decision)
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
    
    async def _upload_trade_data(
        self,
        action: str,
        size: int,
        snapshot: Dict[str, Any],
        decision: Dict[str, Any],
    ):
        """Upload trade data to S3 for agent learning.
        
        Args:
            action: Trade action (BUY/SELL)
            size: Position size
            snapshot: Market snapshot at time of trade
            decision: Full agent decision
        """
        try:
            trade_record = {
                'timestamp': now_cst().isoformat(),
                'action': action,
                'size': size,
                'entry_price': snapshot.get('price'),
                'market_snapshot': snapshot,
                'agent_decision': decision,
            }
            
            result = self.invoker.upload_trade_data([trade_record])
            logger.debug(f"Uploaded trade data: {result}")
            
        except Exception as e:
            logger.warning(f"Failed to upload trade data: {e}")
    
    async def run(self, poll_interval: int = 60):
        """Run the trading loop.
        
        Args:
            poll_interval: Seconds between trading cycles
        """
        if not await self.initialize():
            logger.error("Failed to initialize - exiting")
            return
        
        logger.info(f"🚀 Starting AWS Agent Trading Loop (poll every {poll_interval}s)")
        self.running = True
        
        try:
            while self.running:
                try:
                    await self.process_trading_cycle()
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                
                await asyncio.sleep(poll_interval)
                
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown all components."""
        logger.info("Shutting down AWS Agent Trading Integration...")
        self.running = False
        
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")


async def test_agent_integration():
    """Test agent integration with mock data."""
    logger.info("=" * 60)
    logger.info("Testing AWS Agent Integration")
    logger.info("=" * 60)
    
    # Create mock settings
    from dataclasses import dataclass
    
    @dataclass
    class MockSettings:
        @dataclass
        class Trading:
            initial_capital: float = 100000.0
        trading = Trading()
    
    settings = MockSettings()
    
    # Initialize integration in test mode
    integration = AWSAgentTradingIntegration(
        settings=settings,
        simulation_mode=True,
        use_step_functions=False,
    )
    
    # Initialize components
    if not await integration.initialize():
        logger.error("Failed to initialize")
        return
    
    # Run a few test cycles
    for i in range(3):
        logger.info(f"\n--- Test Cycle {i+1} ---")
        decision = await integration.process_trading_cycle()
        logger.info(f"Result: {json.dumps(decision, indent=2, default=str)}")
        await asyncio.sleep(2)
    
    await integration.shutdown()
    logger.info("Test complete!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AWS Agent Live Trading Integration"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with mock data (no IBKR connection)"
    )
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="Run in simulation mode (connect to IBKR but no real orders)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode (REAL ORDERS - BE CAREFUL!)"
    )
    parser.add_argument(
        "--step-functions",
        action="store_true",
        help="Use Step Functions for agent orchestration"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Poll interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/aws_agent_trading.log", level="DEBUG", rotation="10 MB")
    
    if args.test:
        # Test mode - use mock data
        await test_agent_integration()
        return
    
    # Load settings
    from mytrader.utils.settings_loader import load_settings
    settings = load_settings(args.config)
    
    # Determine simulation mode
    if args.live:
        simulation_mode = False
        logger.warning("=" * 60)
        logger.warning("🔴 LIVE TRADING MODE - REAL ORDERS WILL BE PLACED")
        logger.warning("=" * 60)
        await asyncio.sleep(5)  # Give user time to cancel
    else:
        simulation_mode = True
        logger.info("🔶 SIMULATION MODE - No real orders will be placed")
    
    # Create integration
    integration = AWSAgentTradingIntegration(
        settings=settings,
        simulation_mode=simulation_mode,
        use_step_functions=args.step_functions,
    )
    
    # Handle shutdown signals
    def handle_signal(sig, frame):
        logger.info("Shutdown signal received")
        integration.running = False
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Run trading loop
    await integration.run(poll_interval=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
