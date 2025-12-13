"""
Backtest Runner for 30-Day Agent Integration Test

Runs a backtest that verifies all four agents are invoked correctly,
produce expected artifacts, and integrate end-to-end.
"""
import argparse
import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import pytz

from ..config import Settings
from ..utils.logger import configure_logging, logger
from ..utils.settings_loader import load_settings
from ..features.feature_engineer import engineer_features
from ..risk.manager import RiskManager
from .artifacts import ArtifactLogger
from ..agents.scheduler_emulator import SchedulerEmulator
from ..agents.lambda_wrappers import (
    Agent2DecisionEngineWrapper,
    Agent3RiskControlWrapper,
)


class BacktestRunner:
    """Runs backtest with full agent integration."""
    
    CST = pytz.timezone('America/Chicago')
    
    def __init__(
        self,
        settings: Settings,
        start_date: str,
        end_date: str,
        symbol: str = "ES",
        data_source: str = "local",
        artifacts_dir: Optional[Path] = None,
    ):
        """
        Initialize backtest runner.
        
        Args:
            settings: Trading configuration
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            symbol: Trading symbol (default: ES)
            data_source: Data source ("local" or "s3")
            artifacts_dir: Directory for artifacts (default: artifacts/backtest)
        """
        self.settings = settings
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        self.symbol = symbol
        self.data_source = data_source
        
        # Set up artifacts directory
        if artifacts_dir is None:
            artifacts_dir = Path("artifacts/backtest")
        self.artifacts_dir = artifacts_dir
        self.artifact_logger = ArtifactLogger(artifacts_dir)
        
        # Initialize scheduler emulator
        self.scheduler = SchedulerEmulator(artifacts_dir)
        
        # Initialize agent wrappers
        self.agent2_wrapper = Agent2DecisionEngineWrapper(artifacts_dir)
        self.agent3_wrapper = Agent3RiskControlWrapper(artifacts_dir)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(settings.trading)
        
        # Trading state
        self.capital = settings.trading.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.losing_streak = 0
        self.trades: List[Dict[str, Any]] = []
        
        # Feature flags
        os.environ['FF_BACKTEST_MODE'] = '1'
        os.environ['FF_LOCAL_LAMBDA'] = '1'
        os.environ['FF_ARTIFACT_LOGGING'] = '1'
    
    def load_historical_data(self, date: str) -> Optional[pd.DataFrame]:
        """
        Load historical data for a specific date.
        
        Args:
            date: Date string in YYYY-MM-DD format
        
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        if self.data_source == "s3":
            # TODO: Implement S3 download
            logger.warning("S3 data source not yet implemented")
            return None
        
        # Try local files (prioritize IB downloaded data)
        data_paths = [
            Path(f"data/ib/{self.symbol}_1m_last30d.parquet"),  # IB downloaded data
            Path(f"data/historical_{self.symbol.lower()}.parquet"),
            Path(f"data/{self.symbol.lower()}_{self.start_date}_to_{self.end_date}.parquet"),
            Path(f"data/{self.symbol.lower()}_{date}.parquet"),
            Path(f"data/{self.symbol.lower()}_{date}.csv"),
        ]
        
        for path in data_paths:
            if path.exists():
                logger.info(f"Loading data from {path}")
                try:
                    if path.suffix == '.parquet':
                        df = pd.read_parquet(path)
                    else:
                        df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
                    
                    # Ensure timestamp is the index
                    if 'timestamp' in df.columns and df.index.name != 'timestamp':
                        df = df.set_index('timestamp')
                    
                    # Filter to specific date if needed
                    date_obj = datetime.strptime(date, '%Y-%m-%d').date()
                    if not df.empty:
                        # Filter to the specific date
                        df_filtered = df[df.index.date == date_obj]
                        if not df_filtered.empty:
                            return df_filtered
                        # If no data for this specific date, return empty (will be handled by caller)
                        logger.debug(f"No data for {date} in {path}, but file exists")
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
                    continue
        
        logger.warning(f"No data found for {date}")
        return None
    
    def build_market_snapshot(self, row: pd.Series, features: pd.DataFrame) -> Dict[str, Any]:
        """Build market snapshot for Agent 2."""
        return {
            'price': float(row.get('close', 0)),
            'trend': self._detect_trend(features),
            'volatility': self._classify_volatility(row),
            'regime': self._detect_regime(features),
            'PDH_delta': float(row.get('PDH_delta', 0)) if 'PDH_delta' in row else 0,
            'PDL_delta': float(row.get('PDL_delta', 0)) if 'PDL_delta' in row else 0,
            'rsi': float(row.get('RSI_14', 50)) if 'RSI_14' in row else 50,
            'atr': float(row.get('ATR_14', 1.0)) if 'ATR_14' in row else 1.0,
            'macd_histogram': float(row.get('MACD_histogram', 0)) if 'MACD_histogram' in row else 0,
            'volume': int(row.get('volume', 0)) if 'volume' in row else 0,
        }
    
    def _detect_trend(self, features: pd.DataFrame) -> str:
        """Detect market trend from features."""
        if features.empty:
            return 'UNKNOWN'
        
        # Simple trend detection based on EMA
        if 'EMA_9' in features.columns and 'EMA_20' in features.columns:
            latest = features.iloc[-1]
            if latest['EMA_9'] > latest['EMA_20']:
                return 'UPTREND'
            elif latest['EMA_9'] < latest['EMA_20']:
                return 'DOWNTREND'
        return 'RANGE'
    
    def _classify_volatility(self, row: pd.Series) -> str:
        """Classify volatility."""
        atr = float(row.get('ATR_14', 1.0)) if 'ATR_14' in row else 1.0
        if atr > 2.0:
            return 'HIGH'
        elif atr > 1.0:
            return 'MED'
        return 'LOW'
    
    def _detect_regime(self, features: pd.DataFrame) -> str:
        """Detect market regime."""
        return self._detect_trend(features)  # Simplified
    
    def get_similar_trades(self, market_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get similar historical trades for RAG search.
        
        In backtest mode, this can use:
        - Local vector store
        - Stubbed KB client
        - Historical trades from previous backtest days
        
        For now, returns empty list (stub).
        """
        # TODO: Implement actual RAG search using local KB or historical data
        return []
    
    def run_day(self, date: str) -> Dict[str, Any]:
        """
        Run backtest for a single day.
        
        Args:
            date: Date string in YYYY-MM-DD format
        
        Returns:
            Day results with trades and statistics
        """
        logger.info(f"📅 Processing day: {date}")
        
        # Reset daily state
        self.daily_pnl = 0.0
        self.trades_today = 0
        
        # Run Agent 1: Data Ingestion (nightly, at start-of-day)
        logger.info(f"🔵 Running Agent 1 for {date}")
        agent1_result = self.scheduler.run_nightly_ingestion(date, raw_trade_data=[])
        
        # Load historical data for the day
        df = self.load_historical_data(date)
        if df is None or df.empty:
            logger.warning(f"No data for {date}, skipping")
            return {
                'date': date,
                'trades': 0,
                'pnl': 0.0,
                'agent1_run': True,
                'agent4_run': False,
            }
        
        # Engineer features
        features = engineer_features(df[['open', 'high', 'low', 'close', 'volume']])
        
        # Simulate trading loop (minute-by-minute or bar-by-bar)
        day_trades = []
        
        for idx, row in df.iterrows():
            # Build market snapshot
            market_snapshot = self.build_market_snapshot(row, features.loc[:idx])
            
            # Get similar trades (RAG search)
            similar_trades = self.get_similar_trades(market_snapshot)
            
            # Invoke Agent 2: Decision Engine
            agent2_event = {
                'similar_trades': similar_trades,
                'current_context': market_snapshot
            }
            agent2_result = self.agent2_wrapper.invoke(agent2_event)
            
            decision = agent2_result.get('decision', 'WAIT')
            confidence = agent2_result.get('confidence', 0.0)
            
            # Only proceed if decision is not WAIT
            if decision == 'WAIT':
                continue
            
            # Invoke Agent 3: Risk Control
            account_metrics = {
                'current_pnl_today': self.daily_pnl,
                'current_position': self.position,
                'losing_streak': self.losing_streak,
                'trades_today': self.trades_today,
                'account_balance': self.capital,
                'open_risk': abs(self.position) * 50 * 2.0,  # Simplified
            }
            
            market_conditions = {
                'volatility': market_snapshot['volatility'],
                'regime': market_snapshot['regime'],
                'atr': market_snapshot['atr'],
                'vix': 20.0,  # Stub
            }
            
            agent3_event = {
                'trade_decision': {
                    'action': decision,
                    'confidence': confidence,
                    'symbol': self.symbol,
                    'proposed_size': 1,
                },
                'account_metrics': account_metrics,
                'market_conditions': market_conditions,
            }
            
            agent3_result = self.agent3_wrapper.invoke(agent3_event)
            
            # Execute trade if approved
            if agent3_result.get('allowed_to_trade', False):
                size = agent3_result.get('adjusted_size', 1)
                price = float(row['close'])
                
                # Simulate execution
                if decision == 'BUY' and self.position == 0:
                    self.position = size
                    self.entry_price = price
                    commission = size * self.settings.trading.commission_per_contract
                    self.capital -= commission
                    
                    trade = {
                        'timestamp': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                        'action': 'BUY',
                        'price': price,
                        'quantity': size,
                        'confidence': confidence,
                        'stop_loss': agent2_result.get('stop_loss'),
                        'take_profit': agent2_result.get('take_profit'),
                    }
                    day_trades.append(trade)
                    self.artifact_logger.log_trade(date, trade)
                    self.trades_today += 1
                    
                elif decision == 'SELL' and self.position == 0:
                    self.position = -size
                    self.entry_price = price
                    commission = size * self.settings.trading.commission_per_contract
                    self.capital -= commission
                    
                    trade = {
                        'timestamp': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                        'action': 'SELL',
                        'price': price,
                        'quantity': size,
                        'confidence': confidence,
                        'stop_loss': agent2_result.get('stop_loss'),
                        'take_profit': agent2_result.get('take_profit'),
                    }
                    day_trades.append(trade)
                    self.artifact_logger.log_trade(date, trade)
                    self.trades_today += 1
            
            # Check exits for open positions
            if self.position != 0:
                current_price = float(row['close'])
                pnl = (current_price - self.entry_price) * self.position * self.settings.trading.contract_multiplier
                
                # Simple exit logic (stop/target from Agent 2)
                # In a full implementation, would parse stop_loss/take_profit strings
                stop_distance = 2.0 * market_snapshot['atr']  # Simplified
                target_distance = 3.0 * market_snapshot['atr']  # Simplified
                
                should_exit = False
                exit_reason = ""
                
                if self.position > 0:  # Long position
                    if current_price <= self.entry_price - stop_distance:
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif current_price >= self.entry_price + target_distance:
                        should_exit = True
                        exit_reason = "take_profit"
                else:  # Short position
                    if current_price >= self.entry_price + stop_distance:
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif current_price <= self.entry_price - target_distance:
                        should_exit = True
                        exit_reason = "take_profit"
                
                if should_exit:
                    exit_price = current_price
                    realized_pnl = (exit_price - self.entry_price) * self.position * self.settings.trading.contract_multiplier
                    commission = abs(self.position) * self.settings.trading.commission_per_contract
                    self.capital += realized_pnl - commission
                    self.daily_pnl += realized_pnl
                    
                    # Update losing streak
                    if realized_pnl < 0:
                        self.losing_streak += 1
                    else:
                        self.losing_streak = 0
                    
                    exit_trade = {
                        'timestamp': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                        'action': 'EXIT',
                        'price': exit_price,
                        'quantity': abs(self.position),
                        'realized_pnl': realized_pnl,
                        'exit_reason': exit_reason,
                    }
                    day_trades.append(exit_trade)
                    self.artifact_logger.log_trade(date, exit_trade)
                    
                    self.position = 0
                    self.entry_price = 0.0
        
        # Run Agent 4: Learning (nightly at 11 PM CST)
        logger.info(f"🟣 Running Agent 4 for {date}")
        losing_trades = [t for t in day_trades if t.get('realized_pnl', 0) < 0]
        agent4_result = self.scheduler.run_nightly_learning(date, losing_trades=losing_trades)
        
        return {
            'date': date,
            'trades': len(day_trades),
            'pnl': self.daily_pnl,
            'agent1_run': True,
            'agent4_run': True,
            'day_trades': day_trades,
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full backtest across all dates.
        
        Returns:
            Backtest results with summary statistics
        """
        logger.info(f"🚀 Starting backtest: {self.start_date} to {self.end_date}")
        
        # Generate date range
        current_date = self.start_date
        dates = []
        while current_date <= self.end_date:
            dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        logger.info(f"📅 Processing {len(dates)} days")
        
        # Process each day
        day_results = []
        for date in dates:
            try:
                result = self.run_day(date)
                day_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {date}: {e}", exc_info=True)
                day_results.append({
                    'date': date,
                    'trades': 0,
                    'pnl': 0.0,
                    'agent1_run': False,
                    'agent4_run': False,
                    'error': str(e),
                })
        
        # Generate summary
        total_trades = sum(r.get('trades', 0) for r in day_results)
        total_pnl = sum(r.get('pnl', 0.0) for r in day_results)
        agent1_runs = sum(1 for r in day_results if r.get('agent1_run', False))
        agent4_runs = sum(1 for r in day_results if r.get('agent4_run', False))
        
        # Validate artifacts
        artifact_summary = self.artifact_logger.generate_summary(dates)
        
        results = {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'days_processed': len(dates),
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'agent1_runs': agent1_runs,
            'agent4_runs': agent4_runs,
            'artifact_summary': artifact_summary,
            'day_results': day_results,
        }
        
        return results


def main():
    """Main entry point for backtest runner."""
    parser = argparse.ArgumentParser(description="Run 30-day backtest with agent integration")
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default='ES', help='Trading symbol')
    parser.add_argument('--data-source', type=str, default='local', choices=['local', 's3'], help='Data source')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Set defaults
    if args.end_date is None:
        args.end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    if args.start_date is None:
        end_date_obj = datetime.strptime(args.end_date, '%Y-%m-%d')
        args.start_date = (end_date_obj - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Configure logging
    configure_logging(log_file="logs/backtest.log", level="INFO", serialize=False)
    
    # Load settings
    settings = load_settings(args.config)
    
    # Create runner
    runner = BacktestRunner(
        settings=settings,
        start_date=args.start_date,
        end_date=args.end_date,
        symbol=args.symbol,
        data_source=args.data_source,
    )
    
    # Run backtest
    results = runner.run()
    
    # Generate summary report
    try:
        from .summary import generate_summary_report
        summary_path = Path("reports/backtest_last30_summary.md")
        generate_summary_report(results, summary_path)
    except ImportError:
        logger.warning("Summary generator not available, skipping report generation")
        summary_path = None
    
    logger.info(f"✅ Backtest complete! Summary: {summary_path}")
    
    # Check for missing artifacts
    missing_count = results['artifact_summary'].get('missing_artifacts', [])
    if missing_count:
        logger.error(f"❌ Found {len(missing_count)} days with missing artifacts!")
        return 1
    
    logger.info("✅ All artifacts present!")
    return 0


if __name__ == '__main__':
    import os
    import sys
    sys.exit(main())
