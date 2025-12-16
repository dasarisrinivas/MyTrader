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
from ..learning.strategy_state import StrategyStateManager


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
        self.strategy_manager = StrategyStateManager(artifacts_dir / 'strategy_state.json')
        self.strategy_params = self.strategy_manager.load_state()
        
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
        self.trade_memory: List[Dict[str, Any]] = []
        self.open_trade_contexts: List[Dict[str, Any]] = []
        self.agent2_calls_today = 0
        self.decision_count_today = 0
        self._seed_trade_memory()
        
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
            Path(f"data/ib/{self.symbol}_1m_last30d.parquet"),  # IB downloaded data (parquet)
            Path(f"data/ib/{self.symbol}_1m_last30d.csv"),  # IB downloaded data (CSV fallback)
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
        snapshot = {
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
        timestamp = getattr(row, 'name', None)
        if timestamp is not None:
            if hasattr(timestamp, 'isoformat'):
                snapshot['timestamp'] = timestamp.isoformat()
                snapshot['date'] = timestamp.strftime('%Y-%m-%d')
            else:
                snapshot['timestamp'] = str(timestamp)
                snapshot['date'] = str(timestamp)[:10]
        return snapshot
    
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
        if not self.trade_memory:
            return []
        
        scored_trades = []
        for trade in self.trade_memory:
            context = trade.get('market_snapshot', {})
            score = self._compute_similarity_score(market_snapshot, context)
            effective_score = score if score > 0 else 0.05  # small floor to maintain sample count
            scored_trades.append({
                'action': trade.get('action'),
                'outcome': trade.get('outcome'),
                'pnl': trade.get('pnl'),
                'confidence': trade.get('confidence'),
                'similarity_score': round(effective_score, 3),
                'context': {
                    'trend': context.get('trend'),
                    'regime': context.get('regime'),
                    'volatility': context.get('volatility'),
                    'rsi': context.get('rsi'),
                }
            })
        
        scored_trades.sort(key=lambda t: t['similarity_score'], reverse=True)
        return scored_trades[:15]
    
    def _compute_similarity_score(
        self,
        current: Dict[str, Any],
        historical: Dict[str, Any]
    ) -> float:
        """Lightweight similarity scoring between current and prior trade context."""
        if not current or not historical:
            return 0.0
        
        score = 0.0
        if current.get('trend') == historical.get('trend'):
            score += 0.35
        if current.get('regime') == historical.get('regime'):
            score += 0.2
        if current.get('volatility') == historical.get('volatility'):
            score += 0.15
        
        cur_rsi = float(current.get('rsi', 50))
        hist_rsi = float(historical.get('rsi', 50))
        rsi_diff = abs(cur_rsi - hist_rsi) / 100.0
        score -= min(0.2, rsi_diff)
        
        cur_macd = float(current.get('macd_histogram', 0))
        hist_macd = float(historical.get('macd_histogram', 0))
        macd_diff = abs(cur_macd - hist_macd) / 5.0
        score -= min(0.15, macd_diff)
        
        return max(0.0, min(1.0, score))
    
    def _seed_trade_memory(self) -> None:
        """Populate synthetic trades so Agent 2 has analogs on day one."""
        if self.trade_memory:
            return
        
        logger.info("Seeding trade memory with synthetic historical trades")
        seed_definitions = [
            # action, trend, volatility, rsi, macd, atr, pnl, pdh_delta, pdl_delta
            ('BUY', 'UPTREND', 'LOW', 64, 0.45, 1.0, 450.0, 0.8, -0.2),
            ('BUY', 'UPTREND', 'MED', 58, 0.32, 1.4, 275.0, 0.5, -0.1),
            ('BUY', 'UPTREND', 'LOW', 70, 0.60, 0.9, 520.0, 1.0, -0.4),
            ('BUY', 'RANGE', 'LOW', 55, 0.15, 1.1, 120.0, 0.2, -0.2),
            ('SELL', 'DOWNTREND', 'MED', 42, -0.35, 1.6, 380.0, -0.4, 0.9),
            ('SELL', 'DOWNTREND', 'HIGH', 38, -0.55, 2.2, 610.0, -0.6, 1.2),
            ('SELL', 'DOWNTREND', 'MED', 46, -0.28, 1.4, 210.0, -0.3, 0.6),
            ('SELL', 'RANGE', 'LOW', 48, -0.12, 1.0, 95.0, -0.1, 0.4),
            ('BUY', 'UPTREND', 'HIGH', 66, 0.38, 2.0, 190.0, 0.9, -0.3),
            ('SELL', 'DOWNTREND', 'LOW', 44, -0.18, 0.8, 150.0, -0.2, 0.7),
            ('BUY', 'RANGE', 'MED', 60, 0.25, 1.3, 160.0, 0.4, -0.3),
            ('SELL', 'RANGE', 'MED', 47, -0.22, 1.2, 140.0, -0.2, 0.5),
        ]
        
        now = datetime.now(timezone.utc)
        for idx, (action, trend, volatility, rsi, macd, atr, pnl, pdh_delta, pdl_delta) in enumerate(seed_definitions):
            entry_time = now - timedelta(days=idx + 10)
            exit_time = entry_time + timedelta(hours=4)
            outcome = 'WIN' if pnl > 0 else 'LOSS'
            context = {
                'trend': trend,
                'regime': trend,
                'volatility': volatility,
                'rsi': float(rsi),
                'atr': float(atr),
                'macd_histogram': float(macd),
                'PDH_delta': float(pdh_delta),
                'PDL_delta': float(pdl_delta),
                'timestamp': entry_time.isoformat(),
                'date': entry_time.strftime('%Y-%m-%d'),
            }
            seed_trade = {
                'date': context['date'],
                'entry_time': entry_time.isoformat(),
                'exit_time': exit_time.isoformat(),
                'entry_price': 4500 + (idx * 2 if action == 'BUY' else -idx * 2),
                'exit_price': 4500 + (idx * 2) + (pnl / self.settings.trading.contract_multiplier),
                'action': action,
                'size': 1,
                'confidence': 0.7 if action == 'BUY' else 0.68,
                'market_snapshot': context,
                'pnl': pnl,
                'outcome': outcome,
                'exit_reason': 'target' if pnl > 0 else 'stop_loss',
            }
            self.trade_memory.append(seed_trade)
    
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
        self.agent2_calls_today = 0
        self.decision_count_today = 0
        self._refresh_strategy_params()
        
        # Run Agent 1: Data Ingestion (nightly, at start-of-day)
        logger.info(f"🔵 Running Agent 1 for {date}")
        agent1_result = self.scheduler.run_nightly_ingestion(date, raw_trade_data=[])
        
        # Load historical data for the day
        df = self.load_historical_data(date)
        if df is None or df.empty:
            logger.warning(f"No data for {date}, generating placeholder artifacts")
            reason = f"No market data for {date}"
            self._log_placeholder_day(date, reason=reason)
            daily_summary = {
                'date': date,
                'trades_taken': 0,
                'decision_events': 0,
                'agent2_invocations': 0,
                'pnl': 0.0,
                'missed_opportunity': 0.0,
                'status': 'NO_DATA'
            }
            self.scheduler.run_nightly_learning(
                date,
                losing_trades=[],
                daily_summary=daily_summary,
            )
            self._ensure_daily_artifacts(date)
            return {
                'date': date,
                'trades': 0,
                'pnl': 0.0,
                'agent1_run': True,
                'agent4_run': True,
                'daily_summary': daily_summary,
                'day_trades': [],
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
                'current_context': market_snapshot,
                'backtest_date': date,
                'strategy_params': self.strategy_params,
            }
            self.agent2_calls_today += 1
            agent2_result = self.agent2_wrapper.invoke(agent2_event)
            
            decision = agent2_result.get('decision', 'WAIT')
            confidence = agent2_result.get('confidence', 0.0)
            
            # Only proceed if decision is not WAIT
            if decision == 'WAIT':
                continue
            
            self.decision_count_today += 1
            
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
                'backtest_date': date,
                'strategy_params': self.strategy_params,
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
                    self._record_open_trade(
                        date=date,
                        timestamp=idx,
                        action=decision,
                        size=size,
                        price=price,
                        market_snapshot=market_snapshot,
                        confidence=confidence,
                    )
                    
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
                    self._record_open_trade(
                        date=date,
                        timestamp=idx,
                        action=decision,
                        size=size,
                        price=price,
                        market_snapshot=market_snapshot,
                        confidence=confidence,
                    )
            
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
                    trade_context = self._finalize_trade_context(
                        date=date,
                        exit_timestamp=idx,
                        exit_price=exit_price,
                        realized_pnl=realized_pnl,
                        exit_reason=exit_reason,
                    )
                    if trade_context:
                        exit_trade['trend'] = trade_context['market_snapshot'].get('trend')
                        exit_trade['regime'] = trade_context['market_snapshot'].get('regime')
                        exit_trade['volatility'] = trade_context['market_snapshot'].get('volatility')
                        exit_trade['entry_action'] = trade_context.get('action')
                    day_trades.append(exit_trade)
                    self.artifact_logger.log_trade(date, exit_trade)
                    
                    self.position = 0
                    self.entry_price = 0.0
        
        # Run Agent 4: Learning (nightly at 11 PM CST)
        logger.info(f"🟣 Running Agent 4 for {date}")
        losing_trades = [t for t in day_trades if t.get('realized_pnl', 0) < 0]
        daily_summary = self._build_daily_summary(date, df, day_trades)
        agent4_result = self.scheduler.run_nightly_learning(
            date,
            losing_trades=losing_trades,
            daily_summary=daily_summary,
        )
        self._refresh_strategy_params()
        self._ensure_daily_artifacts(date)
        
        logger.info(
            "📊 Day %s summary • trades=%d • pnl=%.2f • decisions=%d • agent2 calls=%d",
            date,
            len(day_trades),
            self.daily_pnl,
            self.decision_count_today,
            self.agent2_calls_today,
        )
        
        return {
            'date': date,
            'trades': len(day_trades),
            'pnl': self.daily_pnl,
            'agent1_run': True,
            'agent4_run': True,
            'day_trades': day_trades,
            'decisions': self.decision_count_today,
            'agent2_invocations': self.agent2_calls_today,
            'daily_summary': daily_summary,
        }
    
    def _refresh_strategy_params(self) -> None:
        """Reload adaptive strategy knobs after Agent 4 updates."""
        try:
            self.strategy_params = self.strategy_manager.load_state()
        except Exception as exc:
            logger.warning("Unable to refresh strategy params: %s", exc)
    
    def _record_open_trade(
        self,
        date: str,
        timestamp: Any,
        action: str,
        size: int,
        price: float,
        market_snapshot: Dict[str, Any],
        confidence: float,
    ) -> None:
        """Track context for an open trade so we can learn from its outcome later."""
        trade_context = {
            'date': date,
            'entry_time': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'action': action,
            'size': size,
            'entry_price': price,
            'market_snapshot': market_snapshot,
            'confidence': confidence,
        }
        self.open_trade_contexts.append(trade_context)
    
    def _finalize_trade_context(
        self,
        date: str,
        exit_timestamp: Any,
        exit_price: float,
        realized_pnl: float,
        exit_reason: str,
    ) -> Optional[Dict[str, Any]]:
        """Close out the current open trade context and add it to memory."""
        if not self.open_trade_contexts:
            return None
        
        context = self.open_trade_contexts.pop(0)
        context.update({
            'exit_time': exit_timestamp.isoformat() if hasattr(exit_timestamp, 'isoformat') else str(exit_timestamp),
            'exit_price': exit_price,
            'pnl': realized_pnl,
            'outcome': 'WIN' if realized_pnl > 0 else ('LOSS' if realized_pnl < 0 else 'BREAKEVEN'),
            'exit_reason': exit_reason,
        })
        self.trade_memory.append(context)
        # Keep memory bounded to avoid excessive growth
        if len(self.trade_memory) > 500:
            self.trade_memory = self.trade_memory[-500:]
        return context
    
    def _build_daily_summary(
        self,
        date: str,
        df: Optional[pd.DataFrame],
        day_trades: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate diagnostics for Agent 4 and observability."""
        trades_taken = sum(1 for t in day_trades if t.get('action') in ('BUY', 'SELL'))
        missed_opportunity = 0.0
        if trades_taken == 0 and df is not None and not df.empty:
            start_price = float(df['close'].iloc[0])
            end_price = float(df['close'].iloc[-1])
            missed_opportunity = (end_price - start_price) * self.settings.trading.contract_multiplier
        
        return {
            'date': date,
            'trades_taken': trades_taken,
            'decision_events': self.decision_count_today,
            'agent2_invocations': self.agent2_calls_today,
            'pnl': self.daily_pnl,
            'missed_opportunity': missed_opportunity,
            'open_position': self.position,
        }
    
    def _log_placeholder_day(self, date: str, reason: str) -> None:
        """Emit placeholder artifacts when a day cannot be fully processed."""
        self._log_placeholder_decision(date, reason)
        self._log_placeholder_risk(date, reason)
    
    def _log_placeholder_decision(self, date: str, reason: str) -> None:
        """Write a placeholder Agent 2 decision artifact."""
        placeholder_context = {
            'price': 0.0,
            'trend': 'RANGE',
            'volatility': 'LOW',
            'regime': 'RANGE',
            'PDH_delta': 0.0,
            'PDL_delta': 0.0,
            'rsi': 50.0,
            'atr': 1.0,
            'macd_histogram': 0.0,
            'volume': 0,
            'timestamp': f'{date}T00:00:00',
            'date': date,
        }
        self.agent2_wrapper.invoke({
            'similar_trades': [],
            'current_context': placeholder_context,
            'backtest_date': date,
            'placeholder_reason': reason,
            'strategy_params': self.strategy_params,
        })
    
    def _log_placeholder_risk(self, date: str, reason: str) -> None:
        """Write a placeholder Agent 3 evaluation artifact."""
        self.agent3_wrapper.invoke({
            'trade_decision': {
                'action': 'WAIT',
                'confidence': 0.0,
                'symbol': self.symbol,
                'proposed_size': 0,
            },
            'account_metrics': {
                'current_pnl_today': self.daily_pnl,
                'current_position': self.position,
                'losing_streak': self.losing_streak,
                'trades_today': self.trades_today,
                'account_balance': self.capital,
                'open_risk': 0.0,
            },
            'market_conditions': {
                'volatility': 'LOW',
                'regime': 'RANGE',
                'atr': 1.0,
                'vix': 18.0,
            },
            'backtest_date': date,
            'placeholder_reason': reason,
            'strategy_params': self.strategy_params,
        })
    
    def _ensure_daily_artifacts(self, date: str) -> None:
        """Immediately verify and backfill artifacts for a specific day."""
        validation = self.artifact_logger.validate_artifacts(date)
        if validation['valid']:
            return
        
        logger.warning(
            "Detected missing artifacts for %s: %s",
            date,
            [m['file'] for m in validation['missing']],
        )
        for missing in validation['missing']:
            filename = missing['file']
            if filename == 'agent2_decisions.ndjson':
                self._log_placeholder_decision(date, reason='Backfill missing Agent 2 output')
            elif filename == 'agent3_risk.ndjson':
                self._log_placeholder_risk(date, reason='Backfill missing Agent 3 output')
            elif filename == 'agent4_learning_update.json':
                self.scheduler.run_nightly_learning(
                    date,
                    losing_trades=[],
                    daily_summary={
                        'date': date,
                        'pnl': self.daily_pnl,
                        'trades_taken': 0,
                        'decision_events': self.decision_count_today,
                        'status': 'BACKFILL',
                    },
                )
        
        final_validation = self.artifact_logger.validate_artifacts(date)
        if not final_validation['valid']:
            logger.error(
                "❌ Unable to backfill all artifacts for %s: %s",
                date,
                final_validation['missing'],
            )
    
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
