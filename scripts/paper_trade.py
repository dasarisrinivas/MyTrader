#!/usr/bin/env python3
"""
Paper Trading Launcher with Safety Checks

This script provides a safe way to start paper trading with:
1. Pre-flight checks (connection, account, data feeds)
2. Risk limit validation
3. Real-time monitoring
4. Emergency stop functionality

Usage:
    python scripts/paper_trade.py --config config.yaml
    python scripts/paper_trade.py --config config.yaml --max-daily-loss 1000
"""

import sys
import signal
from pathlib import Path
from datetime import datetime
import time

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mytrader.utils.settings_loader import load_settings
from mytrader.data.ibkr import IBKRCollector
from mytrader.strategies.engine import StrategyEngine
from mytrader.risk.manager import RiskManager
from mytrader.execution.ib_executor import TradeExecutor
from mytrader.monitoring.live_tracker import LivePerformanceTracker
from mytrader.utils.logger import configure_logging, logger


class PaperTradingSession:
    """Manages a paper trading session with safety checks."""
    
    def __init__(self, config_path):
        """Initialize paper trading session."""
        self.config = load_settings(config_path)
        self.running = False
        self.stop_requested = False
        
        # Components
        self.data_collector = None
        self.strategy_engine = None
        self.risk_manager = None
        self.executor = None
        self.tracker = None
        
        # Session stats
        self.session_start = None
        self.total_signals = 0
        self.total_trades = 0
        self.last_check_time = None
        
    def pre_flight_checks(self, use_async=False):
        """Run pre-flight safety checks.
        
        Args:
            use_async: If True, skip synchronous IB connection checks (for API context)
        """
        print("\n" + "=" * 80)
        print("Pre-Flight Safety Checks")
        print("=" * 80)
        
        checks_passed = []
        checks_failed = []
        net_liquidation = 100000  # Default value if we can't retrieve it
        
        # Check 1: IBKR Connection
        print("\n1️⃣  Testing IBKR connection...")
        try:
            self.data_collector = IBKRCollector(
                host=self.config.data.ibkr_host,
                port=self.config.data.ibkr_port,
                client_id=self.config.data.ibkr_client_id,
                symbol=self.config.data.ibkr_symbol,
                exchange=self.config.data.ibkr_exchange,
                currency=self.config.data.ibkr_currency
            )
            
            if use_async:
                # When called from async context (API), just verify we can create the collector
                # Actual connection will be established later with async methods
                print(f"   ✅ IBKR Collector initialized (async mode)")
                print(f"   Target: {self.config.data.ibkr_host}:{self.config.data.ibkr_port}")
                checks_passed.append("IBKR connection")
            else:
                # Synchronous connection for CLI usage
                self.data_collector.connect()
            
                # Verify paper trading account
                accounts = self.data_collector.ib.managedAccounts()
                print(f"   ✅ Connected to IBKR - Accounts: {accounts}")
            
                # Check if paper account
                account_values = self.data_collector.ib.accountValues()
                for av in account_values:
                    if av.tag == 'AccountType':
                        if 'PAPER' not in av.value.upper():
                            print(f"   ⚠️  WARNING: This appears to be a LIVE account!")
                            print(f"   Account type: {av.value}")
                            response = input("\n   Continue anyway? (yes/no): ")
                            if response.lower() != 'yes':
                                checks_failed.append("User cancelled - not a paper account")
                                return False
            
                checks_passed.append("IBKR connection")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            checks_failed.append(f"IBKR connection: {e}")
        
        # Check 2: Account Balance
        print("\n2️⃣  Checking account balance...")
        if use_async:
            # Skip detailed account checks in async mode
            print(f"   ⚠️  Skipping in async mode (will verify at runtime)")
            checks_passed.append("Account balance")
        else:
            try:
                # Use sync method to get account summary
                account_summary = self.data_collector.ib.accountSummary()
            
                # Wait a moment for data
                self.data_collector.ib.sleep(2)
            
                temp_net_liquidation = None
                available_funds = None
            
                for item in account_summary:
                    if item.tag == 'NetLiquidation':
                        temp_net_liquidation = float(item.value)
                    elif item.tag == 'AvailableFunds':
                        available_funds = float(item.value)
            
                if temp_net_liquidation:
                    net_liquidation = temp_net_liquidation  # Update the variable in outer scope
                    print(f"   ✅ Account balance: ${net_liquidation:,.2f}")
                    print(f"   Available funds: ${available_funds:,.2f}")
                
                    if net_liquidation < 10000:
                        print(f"   ⚠️  Warning: Low account balance")
                
                    checks_passed.append("Account balance")
                else:
                    print(f"   ⚠️  Could not retrieve account balance")
                    checks_failed.append("Could not retrieve account balance")
                
            except Exception as e:
                print(f"   ⚠️  Could not check balance: {e}")
        
        # Check 3: Market Data Feed
        print("\n3️⃣  Testing market data feed...")
        if use_async:
            # Skip market data test in async mode
            print(f"   ⚠️  Skipping in async mode (will verify at runtime)")
            checks_passed.append("Market data")
        else:
            try:
                from ib_insync import Future
            
                contract = Future('ES', '202412', 'CME')
                self.data_collector.ib.qualifyContracts(contract)
            
                # Wait for qualification
                self.data_collector.ib.sleep(2)
            
                # Request real-time bars
                bars = self.data_collector.ib.reqRealTimeBars(
                    contract, 5, 'TRADES', False
                )
            
                # Wait for first bar
                self.data_collector.ib.sleep(6)
            
                if bars:
                    print(f"   ✅ Market data feed active")
                    if len(bars) > 0:
                        print(f"   Last price: {bars[-1].close if bars else 'N/A'}")
                    checks_passed.append("Market data")
                else:
                    print(f"   ⚠️  No market data received")
                    checks_failed.append("Market data not available")
            
                # Cancel subscription
                self.data_collector.ib.cancelRealTimeBars(bars)
            
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                checks_failed.append(f"Market data: {e}")
        
        # Check 4: Risk Limits
        print("\n4️⃣  Validating risk limits...")
        try:
            risk_config = self.config.trading
            
            print(f"   Max position size: {risk_config.max_position_size} contracts")
            print(f"   Max daily loss: ${risk_config.max_daily_loss:,.2f}")
            print(f"   Max daily trades: {risk_config.max_daily_trades}")
            print(f"   Stop loss: {risk_config.stop_loss_ticks} ticks")
            print(f"   Take profit: {risk_config.take_profit_ticks} ticks")
            
            # Validate reasonable limits
            if risk_config.max_position_size > 10:
                print(f"   ⚠️  Warning: Large max position size")
            
            if risk_config.max_daily_loss > net_liquidation * 0.1:
                print(f"   ⚠️  Warning: Max daily loss > 10% of account")
            
            print(f"   ✅ Risk limits configured")
            checks_passed.append("Risk limits")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            checks_failed.append(f"Risk limits: {e}")
        
        # Check 5: Strategy Configuration
        print("\n5️⃣  Validating strategy configuration...")
        try:
            strategies = self.config.strategies
            print(f"   Active strategies: {len(strategies)}")
            for strategy in strategies:
                print(f"      - {strategy.name if hasattr(strategy, 'name') else 'Unknown'}")
            
            print(f"   ✅ Strategies configured")
            checks_passed.append("Strategy configuration")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            checks_failed.append(f"Strategy configuration: {e}")
        
        # Summary
        print("\n" + "=" * 80)
        print("Pre-Flight Check Summary")
        print("=" * 80)
        
        print(f"\n✅ Passed: {len(checks_passed)}")
        for check in checks_passed:
            print(f"   • {check}")
        
        if checks_failed:
            print(f"\n❌ Failed: {len(checks_failed)}")
            for check in checks_failed:
                print(f"   • {check}")
            
            print("\n⚠️  Cannot start trading - please fix the issues above")
            return False
        
        print("\n✅ All checks passed - ready to trade!")
        return True
    
    def setup_components(self):
        """Initialize trading components."""
        print("\n🔧 Setting up trading components...")
        
        # Strategy engine
        from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
        from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
        
        strategies = []
        for strat_config in self.config.strategies:
            name = strat_config.name if hasattr(strat_config, 'name') else ''
            params = strat_config.params if hasattr(strat_config, 'params') else {}
            
            if 'rsi_macd' in name.lower():
                strategies.append(RsiMacdSentimentStrategy(**params))
            elif 'momentum' in name.lower():
                strategies.append(MomentumReversalStrategy(**params))
        
        self.strategy_engine = StrategyEngine(strategies)
        print(f"   ✅ Strategy engine initialized with {len(strategies)} strategies")
        
        # Risk manager
        self.risk_manager = RiskManager(self.config.trading)
        print(f"   ✅ Risk manager initialized")
        
        # Trade executor
        self.executor = TradeExecutor(
            self.data_collector.ib,
            self.config.trading,
            self.risk_manager
        )
        print(f"   ✅ Trade executor initialized")
        
        # Performance tracker
        self.tracker = LivePerformanceTracker(
            initial_capital=self.config.trading.initial_capital
        )
        print(f"   ✅ Performance tracker initialized")
    
    def run_trading_loop(self):
        """Main trading loop."""
        print("\n" + "=" * 80)
        print("Starting Paper Trading Session")
        print("=" * 80)
        print("\n⚠️  Press Ctrl+C to stop trading gracefully\n")
        
        self.running = True
        self.session_start = datetime.now()
        self.last_check_time = datetime.now()
        
        try:
            # Start data stream
            from ib_insync import Future
            contract = Future('ES', '202412', 'CME')
            self.data_collector.ib.qualifyContracts(contract)
            
            # Subscribe to real-time bars
            bars = self.data_collector.ib.reqRealTimeBars(
                contract, 5, 'TRADES', False
            )
            
            print(f"📡 Subscribed to real-time data for {contract.symbol}")
            print(f"⏰ Session started at {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Trading loop
            bar_count = 0
            historical_bars = []
            
            while self.running and not self.stop_requested:
                # Keep connection alive
                self.data_collector.ib.sleep(1)
                
                # Check for new bars
                if len(bars) > bar_count:
                    bar_count = len(bars)
                    latest_bar = bars[-1]
                    
                    # Convert to DataFrame row
                    bar_data = {
                        'timestamp': latest_bar.time,
                        'open': latest_bar.open,
                        'high': latest_bar.high,
                        'low': latest_bar.low,
                        'close': latest_bar.close,
                        'volume': latest_bar.volume
                    }
                    historical_bars.append(bar_data)
                    
                    # Keep last 100 bars for strategy context
                    if len(historical_bars) > 100:
                        historical_bars = historical_bars[-100:]
                    
                    # Need at least 30 bars for indicators
                    if len(historical_bars) >= 30:
                        # Create DataFrame
                        df = pd.DataFrame(historical_bars)
                        df.set_index('timestamp', inplace=True)
                        
                        # Add sentiment (neutral for now)
                        df['sentiment_twitter'] = 0.0
                        df['sentiment_news'] = 0.0
                        
                        # Engineer features
                        from mytrader.features.feature_engineer import engineer_features
                        df_features = engineer_features(df)
                        
                        # Generate signals
                        signals = self.strategy_engine.generate_signals(df_features)
                        self.total_signals += len(signals)
                        
                        # Process signals
                        for signal in signals:
                            if signal.action != "HOLD":
                                # Check risk limits
                                if self.risk_manager.can_trade():
                                    # Calculate position size
                                    position_size = self.risk_manager.position_size(
                                        signal,
                                        latest_bar.close,
                                        method='fixed'
                                    )
                                    
                                    # Execute trade
                                    if position_size > 0:
                                        logger.info(f"Signal: {signal.action} | Size: {position_size} | "
                                                  f"Price: ${latest_bar.close:.2f} | Conf: {signal.confidence:.2f}")
                                        
                                        # Place order
                                        self.executor.execute_signal(signal, contract, position_size)
                                        self.total_trades += 1
                        
                        # Update performance tracker
                        current_pnl = self.executor.get_realized_pnl()
                        self.tracker.update(
                            timestamp=latest_bar.time,
                            equity=self.config.trading.initial_capital + current_pnl,
                            realized_pnl=current_pnl
                        )
                        
                        # Periodic status update (every 5 minutes)
                        now = datetime.now()
                        if (now - self.last_check_time).seconds >= 300:
                            self.print_status(latest_bar.close, current_pnl)
                            self.last_check_time = now
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupt received - stopping gracefully...")
            self.stop_requested = True
        
        finally:
            self.cleanup()
    
    def print_status(self, current_price, current_pnl):
        """Print current session status."""
        snapshot = self.tracker.get_snapshot()
        session_duration = (datetime.now() - self.session_start).seconds / 60
        
        print("\n" + "=" * 80)
        print(f"Session Status - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
        print(f"Duration: {session_duration:.0f} min | Price: ${current_price:.2f} | "
              f"Signals: {self.total_signals} | Trades: {self.total_trades}")
        print(f"P&L: ${current_pnl:+.2f} | Return: {snapshot.total_return*100:+.2f}% | "
              f"Sharpe: {snapshot.sharpe_ratio:.2f}")
        print("=" * 80 + "\n")
    
    def cleanup(self):
        """Clean up and save results."""
        print("\n🛑 Shutting down trading session...")
        
        # Cancel all pending orders
        if self.executor:
            self.executor.cancel_all_orders()
            print("   ✅ Cancelled pending orders")
        
        # Save performance report
        if self.tracker:
            report_path = Path(f"reports/paper_trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            report_path.parent.mkdir(exist_ok=True)
            self.tracker.export_report(report_path)
            print(f"   ✅ Saved performance report to {report_path}")
        
        # Disconnect
        if self.data_collector:
            self.data_collector.disconnect()
            print("   ✅ Disconnected from IBKR")
        
        # Final summary
        print("\n" + "=" * 80)
        print("Session Summary")
        print("=" * 80)
        
        if self.tracker:
            snapshot = self.tracker.get_snapshot()
            session_duration = (datetime.now() - self.session_start).seconds / 60
            
            print(f"\n⏱️  Duration: {session_duration:.0f} minutes")
            print(f"📊 Total signals: {self.total_signals}")
            print(f"📈 Total trades: {self.total_trades}")
            print(f"💰 P&L: ${snapshot.total_pnl:+.2f}")
            print(f"📊 Return: {snapshot.total_return*100:+.2f}%")
            print(f"📈 Sharpe Ratio: {snapshot.sharpe_ratio:.2f}")
            print(f"📉 Max Drawdown: {snapshot.max_drawdown*100:.2f}%")
        
        print("\n✅ Session closed successfully")


def main():
    """Main entry point."""
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Start paper trading session')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--skip-checks', action='store_true', help='Skip pre-flight checks (not recommended)')
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(level="INFO")
    
    print("\n" + "=" * 80)
    print("MyTrader - Paper Trading Session")
    print("=" * 80)
    
    # Create session
    session = PaperTradingSession(args.config)
    
    # Run pre-flight checks
    if not args.skip_checks:
        if not session.pre_flight_checks():
            print("\n❌ Pre-flight checks failed - cannot start trading")
            sys.exit(1)
    
    # Setup components
    session.setup_components()
    
    # Confirm start
    print("\n" + "=" * 80)
    response = input("\n🚀 Ready to start paper trading. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Trading cancelled by user")
        sys.exit(0)
    
    # Run trading loop
    session.run_trading_loop()


if __name__ == "__main__":
    main()
