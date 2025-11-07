"""Live trade analyzer for paper trading performance review."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..utils.logger import logger


@dataclass
class TradeRecord:
    """Individual trade record."""
    
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    entry_price: float
    exit_price: Optional[float]
    realized_pnl: float
    holding_time_minutes: float
    outcome: str  # WIN, LOSS, BREAKEVEN, OPEN
    signal_type: str  # e.g., "rsi_macd", "sentiment", "momentum"
    sentiment_score: Optional[float]
    llm_confidence: Optional[float]
    llm_reasoning: Optional[str]
    market_conditions: Optional[Dict]


@dataclass
class TradeSummary:
    """Summary of trading performance."""
    
    period_start: str
    period_end: str
    total_trades: int
    open_trades: int
    closed_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    total_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_holding_time: float
    max_drawdown: float
    sharpe_ratio: float
    
    # By signal type
    trades_by_signal: Dict[str, int]
    win_rate_by_signal: Dict[str, float]
    pnl_by_signal: Dict[str, float]
    
    # By market condition
    trades_by_condition: Dict[str, int]
    win_rate_by_condition: Dict[str, float]
    
    # LLM performance
    llm_enhanced_trades: int
    llm_accuracy: float
    avg_llm_confidence: float
    
    # Timing analysis
    trades_by_hour: Dict[int, int]
    win_rate_by_hour: Dict[int, float]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "total_trades": self.total_trades,
            "open_trades": self.open_trades,
            "closed_trades": self.closed_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_holding_time": self.avg_holding_time,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "trades_by_signal": self.trades_by_signal,
            "win_rate_by_signal": self.win_rate_by_signal,
            "pnl_by_signal": self.pnl_by_signal,
            "trades_by_condition": self.trades_by_condition,
            "win_rate_by_condition": self.win_rate_by_condition,
            "llm_enhanced_trades": self.llm_enhanced_trades,
            "llm_accuracy": self.llm_accuracy,
            "avg_llm_confidence": self.avg_llm_confidence,
            "trades_by_hour": self.trades_by_hour,
            "win_rate_by_hour": self.win_rate_by_hour,
        }


class LiveTradeAnalyzer:
    """Analyzes live paper trading performance."""
    
    def __init__(
        self,
        trades_csv_path: Optional[Path] = None,
        logs_dir: Optional[Path] = None
    ):
        """Initialize live trade analyzer.
        
        Args:
            trades_csv_path: Path to trades CSV file
            logs_dir: Directory containing trade logs
        """
        if trades_csv_path is None:
            project_root = Path(__file__).parent.parent.parent
            trades_csv_path = project_root / "logs" / "trades.csv"
        
        if logs_dir is None:
            project_root = Path(__file__).parent.parent.parent
            logs_dir = project_root / "logs"
        
        self.trades_csv_path = Path(trades_csv_path)
        self.logs_dir = Path(logs_dir)
        
        logger.info(f"LiveTradeAnalyzer initialized: {self.trades_csv_path}")
    
    def load_trades_from_csv(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[TradeRecord]:
        """Load trades from CSV file.
        
        Args:
            start_date: Filter trades from this date
            end_date: Filter trades to this date
            
        Returns:
            List of trade records
        """
        if not self.trades_csv_path.exists():
            logger.warning(f"Trades CSV not found: {self.trades_csv_path}")
            return []
        
        trades = []
        
        try:
            with open(self.trades_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        timestamp = datetime.fromisoformat(row['timestamp'])
                        
                        # Apply date filters
                        if start_date and timestamp < start_date:
                            continue
                        if end_date and timestamp > end_date:
                            continue
                        
                        trade = TradeRecord(
                            timestamp=timestamp,
                            symbol=row.get('symbol', 'ES'),
                            action=row.get('action', 'BUY'),
                            quantity=int(row.get('quantity', 1)),
                            entry_price=float(row.get('entry_price', 0)),
                            exit_price=float(row['exit_price']) if row.get('exit_price') else None,
                            realized_pnl=float(row.get('realized_pnl', 0)),
                            holding_time_minutes=float(row.get('holding_time_minutes', 0)),
                            outcome=row.get('outcome', 'OPEN'),
                            signal_type=row.get('signal_type', 'unknown'),
                            sentiment_score=float(row['sentiment_score']) if row.get('sentiment_score') else None,
                            llm_confidence=float(row['llm_confidence']) if row.get('llm_confidence') else None,
                            llm_reasoning=row.get('llm_reasoning'),
                            market_conditions=json.loads(row['market_conditions']) if row.get('market_conditions') else None
                        )
                        
                        trades.append(trade)
                        
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Skipping invalid trade row: {e}")
                        continue
            
            logger.info(f"Loaded {len(trades)} trades from CSV")
            
        except Exception as e:
            logger.error(f"Error loading trades from CSV: {e}")
            return []
        
        return trades
    
    def load_trades_from_db(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[TradeRecord]:
        """Load trades from trade logger database.
        
        Args:
            start_date: Filter trades from this date
            end_date: Filter trades to this date
            
        Returns:
            List of trade records
        """
        from .trade_logger import TradeLogger
        
        try:
            trade_logger = TradeLogger()
            
            # Get trades from database
            if start_date and end_date:
                days = (end_date - start_date).days + 1
            else:
                days = 3  # Default to last 3 days
            
            db_trades = trade_logger.get_recent_trades(limit=1000)
            
            trades = []
            for db_trade in db_trades:
                try:
                    timestamp = datetime.fromisoformat(db_trade['timestamp'])
                    
                    # Apply date filters
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue
                    
                    # Parse market conditions if available
                    market_conditions = None
                    if db_trade.get('entry_context'):
                        try:
                            context = json.loads(db_trade['entry_context'])
                            market_conditions = {
                                'price': context.get('price'),
                                'volume': context.get('volume'),
                                'volatility': context.get('volatility'),
                                'trend': context.get('trend')
                            }
                        except (json.JSONDecodeError, KeyError):
                            pass
                    
                    trade = TradeRecord(
                        timestamp=timestamp,
                        symbol=db_trade.get('symbol', 'ES'),
                        action=db_trade.get('action', 'BUY'),
                        quantity=db_trade.get('quantity', 1),
                        entry_price=db_trade.get('entry_price', 0),
                        exit_price=db_trade.get('exit_price'),
                        realized_pnl=db_trade.get('realized_pnl', 0),
                        holding_time_minutes=db_trade.get('trade_duration_minutes', 0),
                        outcome=db_trade.get('outcome', 'OPEN'),
                        signal_type='llm_enhanced',  # Assume LLM-enhanced if from DB
                        sentiment_score=None,  # Could parse from context
                        llm_confidence=db_trade.get('confidence'),
                        llm_reasoning=db_trade.get('reasoning'),
                        market_conditions=market_conditions
                    )
                    
                    trades.append(trade)
                    
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping invalid DB trade: {e}")
                    continue
            
            logger.info(f"Loaded {len(trades)} trades from database")
            return trades
            
        except Exception as e:
            logger.error(f"Error loading trades from database: {e}")
            return []
    
    def compute_summary(
        self,
        trades: List[TradeRecord],
        period_start: Optional[str] = None,
        period_end: Optional[str] = None
    ) -> TradeSummary:
        """Compute comprehensive trade summary.
        
        Args:
            trades: List of trade records
            period_start: Period start date string
            period_end: Period end date string
            
        Returns:
            Trade summary object
        """
        if not trades:
            # Return empty summary
            return TradeSummary(
                period_start=period_start or datetime.now().strftime("%Y-%m-%d"),
                period_end=period_end or datetime.now().strftime("%Y-%m-%d"),
                total_trades=0, open_trades=0, closed_trades=0,
                winning_trades=0, losing_trades=0, breakeven_trades=0,
                win_rate=0.0, total_pnl=0.0, gross_profit=0.0, gross_loss=0.0,
                profit_factor=0.0, avg_win=0.0, avg_loss=0.0,
                largest_win=0.0, largest_loss=0.0, avg_holding_time=0.0,
                max_drawdown=0.0, sharpe_ratio=0.0,
                trades_by_signal={}, win_rate_by_signal={}, pnl_by_signal={},
                trades_by_condition={}, win_rate_by_condition={},
                llm_enhanced_trades=0, llm_accuracy=0.0, avg_llm_confidence=0.0,
                trades_by_hour={}, win_rate_by_hour={}
            )
        
        # Basic counts
        total_trades = len(trades)
        open_trades = sum(1 for t in trades if t.outcome == 'OPEN')
        closed_trades = total_trades - open_trades
        winning_trades = sum(1 for t in trades if t.outcome == 'WIN')
        losing_trades = sum(1 for t in trades if t.outcome == 'LOSS')
        breakeven_trades = sum(1 for t in trades if t.outcome == 'BREAKEVEN')
        
        # P&L metrics
        closed_trades_list = [t for t in trades if t.outcome != 'OPEN']
        
        if closed_trades_list:
            total_pnl = sum(t.realized_pnl for t in closed_trades_list)
            wins = [t.realized_pnl for t in closed_trades_list if t.outcome == 'WIN']
            losses = [t.realized_pnl for t in closed_trades_list if t.outcome == 'LOSS']
            
            gross_profit = sum(wins) if wins else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            largest_win = max(wins) if wins else 0.0
            largest_loss = min(losses) if losses else 0.0
            
            win_rate = winning_trades / closed_trades if closed_trades > 0 else 0.0
            
            # Holding time
            avg_holding_time = sum(t.holding_time_minutes for t in closed_trades_list) / len(closed_trades_list)
            
            # Drawdown
            pnls = [t.realized_pnl for t in closed_trades_list]
            cumulative = pd.Series(pnls).cumsum()
            running_max = cumulative.expanding().max()
            drawdown = running_max - cumulative
            max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0
            
            # Sharpe ratio
            if len(pnls) > 1:
                sharpe_ratio = pd.Series(pnls).mean() / pd.Series(pnls).std() if pd.Series(pnls).std() > 0 else 0.0
            else:
                sharpe_ratio = 0.0
        else:
            total_pnl = gross_profit = gross_loss = profit_factor = 0.0
            avg_win = avg_loss = largest_win = largest_loss = 0.0
            win_rate = avg_holding_time = max_drawdown = sharpe_ratio = 0.0
        
        # By signal type
        trades_by_signal = {}
        wins_by_signal = {}
        pnl_by_signal = {}
        
        for trade in closed_trades_list:
            signal = trade.signal_type or 'unknown'
            trades_by_signal[signal] = trades_by_signal.get(signal, 0) + 1
            
            if trade.outcome == 'WIN':
                wins_by_signal[signal] = wins_by_signal.get(signal, 0) + 1
            
            pnl_by_signal[signal] = pnl_by_signal.get(signal, 0.0) + trade.realized_pnl
        
        win_rate_by_signal = {
            signal: (wins_by_signal.get(signal, 0) / count) if count > 0 else 0.0
            for signal, count in trades_by_signal.items()
        }
        
        # By market condition
        trades_by_condition = {}
        wins_by_condition = {}
        
        for trade in closed_trades_list:
            if trade.market_conditions:
                condition = trade.market_conditions.get('trend', 'unknown')
                trades_by_condition[condition] = trades_by_condition.get(condition, 0) + 1
                
                if trade.outcome == 'WIN':
                    wins_by_condition[condition] = wins_by_condition.get(condition, 0) + 1
        
        win_rate_by_condition = {
            condition: (wins_by_condition.get(condition, 0) / count) if count > 0 else 0.0
            for condition, count in trades_by_condition.items()
        }
        
        # LLM performance
        llm_trades = [t for t in closed_trades_list if t.llm_confidence is not None]
        llm_enhanced_trades = len(llm_trades)
        
        if llm_trades:
            llm_correct = sum(1 for t in llm_trades if t.outcome == 'WIN')
            llm_accuracy = llm_correct / len(llm_trades)
            avg_llm_confidence = sum(t.llm_confidence for t in llm_trades) / len(llm_trades)
        else:
            llm_accuracy = avg_llm_confidence = 0.0
        
        # By hour
        trades_by_hour = {}
        wins_by_hour = {}
        
        for trade in closed_trades_list:
            hour = trade.timestamp.hour
            trades_by_hour[hour] = trades_by_hour.get(hour, 0) + 1
            
            if trade.outcome == 'WIN':
                wins_by_hour[hour] = wins_by_hour.get(hour, 0) + 1
        
        win_rate_by_hour = {
            hour: (wins_by_hour.get(hour, 0) / count) if count > 0 else 0.0
            for hour, count in trades_by_hour.items()
        }
        
        # Determine period
        if not period_start:
            period_start = min(t.timestamp for t in trades).strftime("%Y-%m-%d")
        if not period_end:
            period_end = max(t.timestamp for t in trades).strftime("%Y-%m-%d")
        
        return TradeSummary(
            period_start=period_start,
            period_end=period_end,
            total_trades=total_trades,
            open_trades=open_trades,
            closed_trades=closed_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=breakeven_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_holding_time=avg_holding_time,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades_by_signal=trades_by_signal,
            win_rate_by_signal=win_rate_by_signal,
            pnl_by_signal=pnl_by_signal,
            trades_by_condition=trades_by_condition,
            win_rate_by_condition=win_rate_by_condition,
            llm_enhanced_trades=llm_enhanced_trades,
            llm_accuracy=llm_accuracy,
            avg_llm_confidence=avg_llm_confidence,
            trades_by_hour=trades_by_hour,
            win_rate_by_hour=win_rate_by_hour
        )
    
    def analyze_recent_performance(
        self,
        days: int = 3,
        use_database: bool = True
    ) -> Tuple[TradeSummary, List[TradeRecord]]:
        """Analyze recent trading performance.
        
        Args:
            days: Number of days to analyze
            use_database: Whether to use trade database (vs CSV)
            
        Returns:
            Tuple of (summary, trades list)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Load trades
        if use_database:
            trades = self.load_trades_from_db(start_date, end_date)
        else:
            trades = self.load_trades_from_csv(start_date, end_date)
        
        # Compute summary
        summary = self.compute_summary(
            trades,
            period_start=start_date.strftime("%Y-%m-%d"),
            period_end=end_date.strftime("%Y-%m-%d")
        )
        
        logger.info(
            f"Analyzed {summary.total_trades} trades over {days} days: "
            f"{summary.win_rate:.1%} WR, ${summary.total_pnl:.2f} P&L"
        )
        
        return summary, trades
