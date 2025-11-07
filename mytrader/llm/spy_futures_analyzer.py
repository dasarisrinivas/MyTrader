"""SPY Futures Daily Performance Analyzer.

Analyzes SPY Futures (ES/MES) paper trading performance and generates
structured insights for dashboard consumption.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..utils.logger import logger


@dataclass
class SPYFuturesTrade:
    """Individual SPY Futures trade record."""
    timestamp: str
    symbol: str  # ES or MES
    action: str  # BUY or SELL
    quantity: int
    price: float
    signal_type: str
    llm_confidence: Optional[float] = None
    llm_reasoning: Optional[str] = None
    position_id: Optional[str] = None
    exit_timestamp: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    holding_time_minutes: Optional[int] = None
    exit_reason: Optional[str] = None


@dataclass
class SPYFuturesPerformance:
    """Daily SPY Futures performance summary."""
    date: str
    symbol: str  # ES or MES
    total_trades: int
    closed_trades: int
    open_positions: int
    
    # Win/Loss metrics
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L metrics
    total_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    max_drawdown: float
    
    # Trading behavior
    average_holding_time_minutes: float
    trades_by_hour: Dict[int, int]
    pnl_by_hour: Dict[int, float]
    
    # Signal performance
    trades_by_signal: Dict[str, int]
    pnl_by_signal: Dict[str, float]
    win_rate_by_signal: Dict[str, float]
    
    # LLM enhancement
    llm_enhanced_trades: int
    llm_enhanced_pnl: float
    llm_average_confidence: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SPYFuturesAnalyzer:
    """Analyzer for SPY Futures paper trading performance."""
    
    # Supported SPY Futures symbols
    VALID_SYMBOLS = ["ES", "MES", "SPY"]  # ES = E-mini S&P, MES = Micro E-mini
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        csv_path: Optional[Path] = None
    ):
        """Initialize SPY Futures analyzer.
        
        Args:
            db_path: Path to trade database
            csv_path: Path to CSV trade logs
        """
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "llm_trades.db"
        
        if csv_path is None:
            project_root = Path(__file__).parent.parent.parent
            csv_path = project_root / "logs" / "trades.csv"
        
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)
        
        logger.info(f"SPYFuturesAnalyzer initialized")
        logger.info(f"  Database: {self.db_path}")
        logger.info(f"  CSV: {self.csv_path}")
    
    def load_trades_from_db(
        self,
        days: int = 1,
        symbol_filter: Optional[str] = None
    ) -> List[SPYFuturesTrade]:
        """Load SPY Futures trades from database.
        
        Args:
            days: Number of recent days to load
            symbol_filter: Filter by specific symbol (ES, MES, SPY)
            
        Returns:
            List of SPY Futures trade records
        """
        if not self.db_path.exists():
            logger.warning(f"Database not found: {self.db_path}")
            return []
        
        # Calculate date threshold
        date_threshold = (datetime.now() - timedelta(days=days)).isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Query trades - filter for SPY Futures only
            symbol_list = [symbol_filter] if symbol_filter else self.VALID_SYMBOLS
            placeholders = ','.join('?' * len(symbol_list))
            
            query = f"""
                SELECT 
                    timestamp, symbol, action, quantity, price,
                    signal_type, llm_confidence, llm_reasoning,
                    position_id, exit_timestamp, exit_price,
                    pnl, exit_reason
                FROM trades
                WHERE timestamp >= ?
                AND symbol IN ({placeholders})
                ORDER BY timestamp DESC
            """
            
            cursor = conn.cursor()
            cursor.execute(query, [date_threshold] + symbol_list)
            
            trades = []
            for row in cursor.fetchall():
                # Calculate holding time if trade is closed
                holding_time = None
                if row[9]:  # exit_timestamp exists
                    try:
                        entry_time = datetime.fromisoformat(row[0])
                        exit_time = datetime.fromisoformat(row[9])
                        holding_time = int((exit_time - entry_time).total_seconds() / 60)
                    except:
                        pass
                
                trade = SPYFuturesTrade(
                    timestamp=row[0],
                    symbol=row[1],
                    action=row[2],
                    quantity=row[3],
                    price=row[4],
                    signal_type=row[5],
                    llm_confidence=row[6],
                    llm_reasoning=row[7],
                    position_id=row[8],
                    exit_timestamp=row[9],
                    exit_price=row[10],
                    pnl=row[11],
                    holding_time_minutes=holding_time,
                    exit_reason=row[12]
                )
                trades.append(trade)
            
            conn.close()
            
            logger.info(f"Loaded {len(trades)} SPY Futures trades from database")
            return trades
        
        except Exception as e:
            logger.error(f"Error loading from database: {e}", exc_info=True)
            return []
    
    def load_trades_from_csv(
        self,
        days: int = 1,
        symbol_filter: Optional[str] = None
    ) -> List[SPYFuturesTrade]:
        """Load SPY Futures trades from CSV.
        
        Args:
            days: Number of recent days to load
            symbol_filter: Filter by specific symbol (ES, MES, SPY)
            
        Returns:
            List of SPY Futures trade records
        """
        if not self.csv_path.exists():
            logger.warning(f"CSV not found: {self.csv_path}")
            return []
        
        try:
            df = pd.read_csv(self.csv_path)
            
            # Filter for SPY Futures symbols only
            symbol_list = [symbol_filter] if symbol_filter else self.VALID_SYMBOLS
            df = df[df['symbol'].isin(symbol_list)]
            
            # Filter by date
            date_threshold = datetime.now() - timedelta(days=days)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[df['timestamp'] >= date_threshold]
            
            # Sort by timestamp descending
            df = df.sort_values('timestamp', ascending=False)
            
            trades = []
            for _, row in df.iterrows():
                # Calculate holding time
                holding_time = None
                if pd.notna(row.get('exit_timestamp')):
                    try:
                        entry_time = pd.to_datetime(row['timestamp'])
                        exit_time = pd.to_datetime(row['exit_timestamp'])
                        holding_time = int((exit_time - entry_time).total_seconds() / 60)
                    except:
                        pass
                
                # Handle timestamp - could be datetime or string
                timestamp_str = row['timestamp']
                if hasattr(timestamp_str, 'isoformat'):
                    timestamp_str = timestamp_str.isoformat()
                
                exit_timestamp_str = None
                if pd.notna(row.get('exit_timestamp')):
                    exit_timestamp_str = row['exit_timestamp']
                    if hasattr(exit_timestamp_str, 'isoformat'):
                        exit_timestamp_str = exit_timestamp_str.isoformat()
                
                trade = SPYFuturesTrade(
                    timestamp=timestamp_str,
                    symbol=row['symbol'],
                    action=row['action'],
                    quantity=int(row['quantity']),
                    price=float(row['price']),
                    signal_type=row.get('signal_type', 'unknown'),
                    llm_confidence=float(row['llm_confidence']) if pd.notna(row.get('llm_confidence')) else None,
                    llm_reasoning=row.get('llm_reasoning'),
                    position_id=row.get('position_id'),
                    exit_timestamp=exit_timestamp_str,
                    exit_price=float(row['exit_price']) if pd.notna(row.get('exit_price')) else None,
                    pnl=float(row['pnl']) if pd.notna(row.get('pnl')) else None,
                    holding_time_minutes=holding_time,
                    exit_reason=row.get('exit_reason')
                )
                trades.append(trade)
            
            logger.info(f"Loaded {len(trades)} SPY Futures trades from CSV")
            return trades
        
        except Exception as e:
            logger.error(f"Error loading from CSV: {e}", exc_info=True)
            return []
    
    def compute_performance(
        self,
        trades: List[SPYFuturesTrade],
        target_date: Optional[str] = None
    ) -> SPYFuturesPerformance:
        """Compute comprehensive performance metrics for SPY Futures.
        
        Args:
            trades: List of trade records
            target_date: Date string (YYYY-MM-DD), defaults to today
            
        Returns:
            Performance summary
        """
        if not trades:
            # Return empty performance
            return SPYFuturesPerformance(
                date=target_date or datetime.now().strftime("%Y-%m-%d"),
                symbol="ES",
                total_trades=0,
                closed_trades=0,
                open_positions=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                gross_profit=0.0,
                gross_loss=0.0,
                profit_factor=0.0,
                average_win=0.0,
                average_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                max_drawdown=0.0,
                average_holding_time_minutes=0.0,
                trades_by_hour={},
                pnl_by_hour={},
                trades_by_signal={},
                pnl_by_signal={},
                win_rate_by_signal={},
                llm_enhanced_trades=0,
                llm_enhanced_pnl=0.0
            )
        
        # Determine primary symbol
        symbol_counts = {}
        for trade in trades:
            symbol_counts[trade.symbol] = symbol_counts.get(trade.symbol, 0) + 1
        primary_symbol = max(symbol_counts, key=symbol_counts.get)
        
        # Basic counts
        total_trades = len(trades)
        closed_trades = sum(1 for t in trades if t.pnl is not None)
        open_positions = total_trades - closed_trades
        
        # Filter closed trades for detailed metrics
        closed = [t for t in trades if t.pnl is not None]
        
        if not closed:
            return SPYFuturesPerformance(
                date=target_date or datetime.now().strftime("%Y-%m-%d"),
                symbol=primary_symbol,
                total_trades=total_trades,
                closed_trades=0,
                open_positions=open_positions,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                gross_profit=0.0,
                gross_loss=0.0,
                profit_factor=0.0,
                average_win=0.0,
                average_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                max_drawdown=0.0,
                average_holding_time_minutes=0.0,
                trades_by_hour={},
                pnl_by_hour={},
                trades_by_signal={},
                pnl_by_signal={},
                win_rate_by_signal={},
                llm_enhanced_trades=0,
                llm_enhanced_pnl=0.0
            )
        
        # Win/Loss metrics
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / closed_trades if closed_trades > 0 else 0.0
        
        # P&L metrics
        total_pnl = sum(t.pnl for t in closed)
        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        average_win = gross_profit / winning_trades if winning_trades > 0 else 0.0
        average_loss = gross_loss / losing_trades if losing_trades > 0 else 0.0
        
        largest_win = max((t.pnl for t in wins), default=0.0)
        largest_loss = min((t.pnl for t in losses), default=0.0)
        
        # Calculate max drawdown
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0
        for trade in sorted(closed, key=lambda t: t.timestamp):
            cumulative_pnl += trade.pnl
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        # Holding time analysis
        holding_times = [t.holding_time_minutes for t in closed if t.holding_time_minutes]
        avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0.0
        
        # Hour-based analysis
        trades_by_hour = {}
        pnl_by_hour = {}
        for trade in closed:
            try:
                hour = datetime.fromisoformat(trade.timestamp).hour
                trades_by_hour[hour] = trades_by_hour.get(hour, 0) + 1
                pnl_by_hour[hour] = pnl_by_hour.get(hour, 0.0) + trade.pnl
            except:
                pass
        
        # Signal type analysis
        trades_by_signal = {}
        pnl_by_signal = {}
        wins_by_signal = {}
        
        for trade in closed:
            signal = trade.signal_type
            trades_by_signal[signal] = trades_by_signal.get(signal, 0) + 1
            pnl_by_signal[signal] = pnl_by_signal.get(signal, 0.0) + trade.pnl
            if trade.pnl > 0:
                wins_by_signal[signal] = wins_by_signal.get(signal, 0) + 1
        
        win_rate_by_signal = {}
        for signal, count in trades_by_signal.items():
            wins = wins_by_signal.get(signal, 0)
            win_rate_by_signal[signal] = wins / count if count > 0 else 0.0
        
        # LLM enhancement analysis
        llm_trades = [t for t in closed if t.llm_confidence is not None]
        llm_enhanced_trades = len(llm_trades)
        llm_enhanced_pnl = sum(t.pnl for t in llm_trades) if llm_trades else 0.0
        llm_avg_confidence = sum(t.llm_confidence for t in llm_trades) / len(llm_trades) if llm_trades else None
        
        return SPYFuturesPerformance(
            date=target_date or datetime.now().strftime("%Y-%m-%d"),
            symbol=primary_symbol,
            total_trades=total_trades,
            closed_trades=closed_trades,
            open_positions=open_positions,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_drawdown=max_drawdown,
            average_holding_time_minutes=avg_holding_time,
            trades_by_hour=trades_by_hour,
            pnl_by_hour=pnl_by_hour,
            trades_by_signal=trades_by_signal,
            pnl_by_signal=pnl_by_signal,
            win_rate_by_signal=win_rate_by_signal,
            llm_enhanced_trades=llm_enhanced_trades,
            llm_enhanced_pnl=llm_enhanced_pnl,
            llm_average_confidence=llm_avg_confidence
        )
    
    def analyze_daily_performance(
        self,
        days: int = 1,
        use_database: bool = True,
        symbol_filter: Optional[str] = None
    ) -> Tuple[SPYFuturesPerformance, List[SPYFuturesTrade]]:
        """Analyze recent SPY Futures performance.
        
        Args:
            days: Number of recent days to analyze
            use_database: Use database vs CSV
            symbol_filter: Filter by specific symbol
            
        Returns:
            Tuple of (performance summary, trade list)
        """
        logger.info(f"Analyzing SPY Futures performance for last {days} day(s)")
        
        # Load trades
        if use_database:
            trades = self.load_trades_from_db(days=days, symbol_filter=symbol_filter)
        else:
            trades = self.load_trades_from_csv(days=days, symbol_filter=symbol_filter)
        
        # Compute performance
        performance = self.compute_performance(trades)
        
        logger.info(f"Analysis complete: {performance.total_trades} trades, "
                   f"${performance.total_pnl:,.2f} P&L")
        
        return performance, trades
