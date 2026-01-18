#!/usr/bin/env python3
"""
30-MINUTE OVERNIGHT TRADING BOT
===============================
Dedicated runner for the 30-minute strategy, optimized for overnight/globex sessions.

Strategy: MesThirtyMinuteStrategy (TREND_CONTINUATION only)
Backtest Results: 68 trades, +$64.85, 1.04 PF, 38.2% WR

Config:
- Stop: 1x ATR (~6-8 pts)
- Target: 2R (2x risk)
- ADX threshold: 30.0
- Signal type: TREND_CONTINUATION only (EMA_RECLAIM disabled)

Usage:
    python run_30m_overnight.py --simulation  # Paper trade mode
    python run_30m_overnight.py               # Live mode (use with caution!)
"""
import argparse
import asyncio
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

import pandas as pd
from ib_insync import IB, Contract, util

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mytrader.config import Settings, ThirtyMinuteStrategyConfig
from mytrader.strategies.mes_thirty_minute import MesThirtyMinuteStrategy
from mytrader.utils.logger import configure_logging, logger
from mytrader.utils.settings_loader import load_settings
from mytrader.utils.timezone_utils import now_cst, CST


class ThirtyMinuteOvernightBot:
    """Lightweight 30-minute overnight trading bot."""
    
    def __init__(
        self,
        settings: Settings,
        simulation_mode: bool = True,
    ):
        self.settings = settings
        self.simulation_mode = simulation_mode
        self.running = False
        
        # IB connection
        self.ib: Optional[IB] = None
        
        # Strategy config (use defaults tuned from backtest)
        self.strategy_config = ThirtyMinuteStrategyConfig(
            enabled=True,
            stop_atr_multiplier=1.0,  # 1x ATR = ~6-8 pts
            take_profit_multiplier=2.0,  # 2R target
            adx_trend_threshold=30.0,  # Raised ADX requirement
            enable_ema_reclaim=False,  # Disabled - losing signal type
            enable_trend_continuation=True,  # Profitable signal type
            max_trades_per_day=4,
            cooldown_bars=2,  # 1 hour cooldown
            trade_all_sessions=True,  # Overnight trading
        )
        
        # Strategy instance
        self.strategy: Optional[MesThirtyMinuteStrategy] = None
        
        # State tracking
        self.price_history: List[Dict] = []
        self.last_bar_time: Optional[datetime] = None
        self.last_signal_time: Optional[datetime] = None
        self.trades_today: int = 0
        self.daily_pnl: float = 0.0
        self.position: int = 0
        self.entry_price: Optional[float] = None
        self.current_stop: Optional[float] = None
        self.current_target: Optional[float] = None
        
        # Contract
        self.contract: Optional[Contract] = None
        
    async def connect(self) -> bool:
        """Connect to IB Gateway."""
        try:
            self.ib = IB()
            
            host = getattr(self.settings.data, "ibkr_host", "127.0.0.1")
            port = getattr(self.settings.data, "ibkr_port", 4002)
            client_id = getattr(self.settings.data, "ibkr_client_id", 1) + 100  # Use different client ID
            
            logger.info(f"🔌 Connecting to IB Gateway at {host}:{port} (client {client_id})")
            
            await self.ib.connectAsync(host, port, clientId=client_id)
            
            if not self.ib.isConnected():
                logger.error("❌ Failed to connect to IB Gateway")
                return False
            
            logger.info("✅ Connected to IB Gateway")
            
            # Create MES contract for the front month (March 2026 = H6)
            # MES contract months: H (Mar), M (Jun), U (Sep), Z (Dec)
            from datetime import datetime
            now = datetime.now()
            year = now.year
            month = now.month
            
            # Determine current front month
            if month <= 3:
                contract_month = f"{year}03"  # March
            elif month <= 6:
                contract_month = f"{year}06"  # June
            elif month <= 9:
                contract_month = f"{year}09"  # September
            else:
                contract_month = f"{year}12"  # December
            
            self.contract = Contract(
                symbol="MES",
                secType="FUT",
                exchange="CME",
                currency="USD",
                lastTradeDateOrContractMonth=contract_month,
            )
            
            # Qualify contract
            contracts = await self.ib.qualifyContractsAsync(self.contract)
            if contracts:
                self.contract = contracts[0]
                logger.info(f"✅ Qualified contract: {self.contract.localSymbol}")
            else:
                # Fallback: try to find any valid MES contract
                logger.warning("⚠️ Could not qualify contract, trying continuous search...")
                search_contract = Contract(
                    symbol="MES",
                    secType="FUT",
                    exchange="CME",
                    currency="USD",
                )
                details = await self.ib.reqContractDetailsAsync(search_contract)
                if details:
                    self.contract = details[0].contract
                    logger.info(f"✅ Found contract via search: {self.contract.localSymbol}")
                else:
                    logger.error("❌ Could not find valid MES contract")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False
    
    async def fetch_30m_bars(self, bars: int = 100) -> pd.DataFrame:
        """Fetch 30-minute historical bars from IB."""
        try:
            # Request 30-minute bars
            historical = await self.ib.reqHistoricalDataAsync(
                self.contract,
                endDateTime="",
                durationStr=f"{bars * 30 + 60} S" if bars < 10 else f"{bars // 2 + 1} D",
                barSizeSetting="30 mins",
                whatToShow="TRADES",
                useRTH=False,  # Include overnight
                formatDate=1,
            )
            
            if not historical:
                logger.warning("⚠️ No historical data received")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = util.df(historical)
            
            # Rename columns to match strategy expectations
            df = df.rename(columns={
                "date": "timestamp",
            })
            
            # Ensure timestamp is datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            
            logger.info(f"📊 Fetched {len(df)} 30-min bars")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching bars: {e}")
            return pd.DataFrame()
    
    def initialize_strategy(self, df: pd.DataFrame) -> bool:
        """Initialize the 30-minute strategy with historical data."""
        try:
            # MesThirtyMinuteStrategy takes only config
            self.strategy = MesThirtyMinuteStrategy(
                config=self.strategy_config,
            )
            
            # Warm up strategy with historical data
            for i, (timestamp, row) in enumerate(df.iterrows()):
                bar = {
                    "timestamp": timestamp,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row.get("volume", 0),
                }
                self.price_history.append(bar)
            
            logger.info(f"✅ Strategy initialized with {len(self.price_history)} bars")
            return True
            
        except Exception as e:
            logger.error(f"❌ Strategy initialization error: {e}")
            return False
    
    async def get_current_price(self) -> Optional[float]:
        """Get current price from IB."""
        try:
            ticker = self.ib.reqMktData(self.contract, "", False, False)
            await asyncio.sleep(0.5)  # Wait for data
            
            if ticker.last and ticker.last > 0:
                return ticker.last
            elif ticker.close and ticker.close > 0:
                return ticker.close
            
            return None
        except Exception as e:
            logger.warning(f"⚠️ Error getting price: {e}")
            return None
    
    async def place_order(
        self,
        action: str,
        quantity: int,
        limit_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Place order (simulation or live)."""
        mode_str = "SIMULATION" if self.simulation_mode else "LIVE"
        
        logger.info("=" * 60)
        logger.info(f"📤 [{mode_str}] ORDER: {action} {quantity} MES @ {limit_price:.2f}")
        if stop_loss:
            logger.info(f"   Stop Loss: {stop_loss:.2f}")
        if take_profit:
            logger.info(f"   Take Profit: {take_profit:.2f}")
        logger.info("=" * 60)
        
        if self.simulation_mode:
            # Simulate fill
            if action == "BUY":
                self.position = quantity
                self.entry_price = limit_price
            elif action == "SELL":
                if self.position > 0:
                    # Close long
                    pnl = (limit_price - self.entry_price) * 5 * quantity
                    self.daily_pnl += pnl
                    logger.info(f"💰 Closed LONG for ${pnl:.2f}")
                    self.position = 0
                    self.entry_price = None
                else:
                    self.position = -quantity
                    self.entry_price = limit_price
            
            self.current_stop = stop_loss
            self.current_target = take_profit
            self.trades_today += 1
            return True
        
        # Live order placement would go here
        logger.warning("🚨 LIVE ORDERS NOT IMPLEMENTED - Use simulation mode")
        return False
    
    async def check_exit_conditions(self, current_price: float) -> bool:
        """Check if position should be exited."""
        if self.position == 0:
            return False
        
        is_long = self.position > 0
        
        # Check stop loss
        if self.current_stop:
            if is_long and current_price <= self.current_stop:
                logger.info(f"🛑 STOP LOSS HIT: {current_price:.2f} <= {self.current_stop:.2f}")
                await self.place_order("SELL", abs(self.position), current_price)
                return True
            elif not is_long and current_price >= self.current_stop:
                logger.info(f"🛑 STOP LOSS HIT: {current_price:.2f} >= {self.current_stop:.2f}")
                await self.place_order("BUY", abs(self.position), current_price)
                return True
        
        # Check take profit
        if self.current_target:
            if is_long and current_price >= self.current_target:
                logger.info(f"🎯 TAKE PROFIT HIT: {current_price:.2f} >= {self.current_target:.2f}")
                await self.place_order("SELL", abs(self.position), current_price)
                return True
            elif not is_long and current_price <= self.current_target:
                logger.info(f"🎯 TAKE PROFIT HIT: {current_price:.2f} <= {self.current_target:.2f}")
                await self.place_order("BUY", abs(self.position), current_price)
                return True
        
        return False
    
    async def process_bar(self, bar: Dict) -> None:
        """Process a new 30-minute bar."""
        self.price_history.append(bar)
        
        # Keep only last 200 bars
        if len(self.price_history) > 200:
            self.price_history = self.price_history[-200:]
        
        # Convert to DataFrame for strategy
        df = pd.DataFrame(self.price_history)
        df.set_index("timestamp", inplace=True)
        
        current_price = bar["close"]
        
        # Check exit conditions first
        if await self.check_exit_conditions(current_price):
            return
        
        # Generate signal (pass current bar index and position)
        bar_idx = len(df) - 1
        signal = self.strategy.generate_signal(df, bar_idx, self.position)
        
        if signal is None:
            return
        
        # ThirtyMinuteSignal is a dataclass, access attributes directly
        action = signal.action
        confidence = signal.confidence
        signal_type = signal.signal_type
        
        # Log signal
        now = now_cst()
        logger.info(f"📊 [{now.strftime('%H:%M')}] Signal: {action} ({signal_type}) conf={confidence:.2f}")
        
        if action == "HOLD":
            return
        
        # Check if we can trade
        if self.position != 0:
            logger.info("⏸️ Already in position, skipping signal")
            return
        
        if self.trades_today >= self.strategy_config.max_trades_per_day:
            logger.info(f"⏸️ Max trades reached ({self.trades_today})")
            return
        
        # Check cooldown
        if self.last_signal_time:
            cooldown_minutes = self.strategy_config.cooldown_bars * 30
            elapsed = (now - self.last_signal_time).total_seconds() / 60
            if elapsed < cooldown_minutes:
                logger.info(f"⏸️ Cooldown: {cooldown_minutes - elapsed:.1f} min remaining")
                return
        
        # Get stops from signal (dataclass attributes)
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit
        
        # Place order
        if action == "BUY":
            await self.place_order("BUY", 1, current_price, stop_loss, take_profit)
            self.last_signal_time = now
        elif action == "SELL":
            await self.place_order("SELL", 1, current_price, stop_loss, take_profit)
            self.last_signal_time = now
    
    async def run(self):
        """Main trading loop."""
        logger.info("=" * 70)
        logger.info("🌙 30-MINUTE OVERNIGHT TRADING BOT STARTING")
        logger.info("=" * 70)
        mode_str = "SIMULATION" if self.simulation_mode else "⚠️ LIVE ⚠️"
        logger.info(f"Mode: {mode_str}")
        logger.info(f"Strategy: TREND_CONTINUATION only")
        logger.info(f"Stop: 1x ATR (~6-8 pts)")
        logger.info(f"Target: 2R")
        logger.info(f"ADX threshold: {self.strategy_config.adx_trend_threshold}")
        logger.info("=" * 70)
        
        # Connect to IB
        if not await self.connect():
            logger.error("❌ Failed to connect - exiting")
            return
        
        # Fetch initial bars
        df = await self.fetch_30m_bars(100)
        if df.empty:
            logger.error("❌ No historical data - exiting")
            return
        
        # Initialize strategy
        if not self.initialize_strategy(df):
            logger.error("❌ Strategy init failed - exiting")
            return
        
        self.running = True
        last_bar_minute = None
        
        logger.info("🚀 Trading loop started - waiting for 30-min bar closes")
        
        while self.running:
            try:
                now = now_cst()
                
                # Check for 30-minute bar close (xx:00 or xx:30)
                current_minute = now.minute
                is_bar_close = current_minute in [0, 30]
                
                if is_bar_close and current_minute != last_bar_minute:
                    last_bar_minute = current_minute
                    
                    # Fetch latest bar
                    df = await self.fetch_30m_bars(5)
                    if not df.empty:
                        latest_bar = df.iloc[-1]
                        bar = {
                            "timestamp": df.index[-1],
                            "open": latest_bar["open"],
                            "high": latest_bar["high"],
                            "low": latest_bar["low"],
                            "close": latest_bar["close"],
                            "volume": latest_bar.get("volume", 0),
                        }
                        
                        logger.info(f"📊 New 30m bar: O={bar['open']:.2f} H={bar['high']:.2f} L={bar['low']:.2f} C={bar['close']:.2f}")
                        
                        await self.process_bar(bar)
                
                # Status update every 5 minutes
                if now.minute % 5 == 0 and now.second < 10:
                    current_price = await self.get_current_price()
                    if current_price:
                        pos_str = f"LONG {self.position}" if self.position > 0 else (f"SHORT {abs(self.position)}" if self.position < 0 else "FLAT")
                        logger.info(f"📈 [{now.strftime('%H:%M')}] Price: {current_price:.2f} | Position: {pos_str} | PnL: ${self.daily_pnl:.2f}")
                        
                        # Check exit conditions on tick
                        if self.position != 0:
                            await self.check_exit_conditions(current_price)
                
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("🛑 Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                await asyncio.sleep(10)
        
        await self.stop()
    
    async def stop(self):
        """Stop the bot."""
        self.running = False
        
        logger.info("=" * 70)
        logger.info("📊 30-MINUTE BOT SESSION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Trades: {self.trades_today}")
        logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")
        pos_str = f"LONG {self.position}" if self.position > 0 else (f"SHORT {abs(self.position)}" if self.position < 0 else "FLAT")
        logger.info(f"Final Position: {pos_str}")
        logger.info("=" * 70)
        
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("🔌 Disconnected from IB Gateway")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="30-Minute Overnight Trading Bot")
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        default=True,
        help="Run in simulation mode (default: True)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in LIVE mode (use with caution!)"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file"
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Configure logging
    configure_logging(log_file="logs/overnight_30m.log", level="INFO")
    
    # Determine mode
    simulation_mode = not args.live
    
    if not simulation_mode:
        logger.warning("=" * 70)
        logger.warning("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!")
        logger.warning("=" * 70)
        response = input("Type 'YES' to confirm live trading: ")
        if response != "YES":
            logger.info("Live trading cancelled")
            return
    
    # Load settings
    settings = load_settings(args.config)
    
    # Create and run bot
    bot = ThirtyMinuteOvernightBot(settings, simulation_mode=simulation_mode)
    
    # Handle signals
    def handle_signal(sig, frame):
        logger.info("🛑 Shutdown signal received")
        bot.running = False
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
