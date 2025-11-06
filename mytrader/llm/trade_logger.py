"""Trade logger for storing LLM predictions and outcomes."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from ..utils.logger import logger
from .data_schema import TradeOutcome, TradeRecommendation, TradingContext


class TradeLogger:
    """Logger for storing trades with LLM predictions for learning pipeline."""
    
    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """Initialize trade logger with SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            # Use project root/data/llm_trades.db by default
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "llm_trades.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Table for trade outcomes
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    realized_pnl REAL DEFAULT 0.0,
                    trade_duration_minutes REAL DEFAULT 0.0,
                    outcome TEXT DEFAULT 'OPEN',
                    entry_context TEXT,
                    exit_context TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Table for LLM recommendations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_outcome_id INTEGER,
                    trade_decision TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    suggested_position_size INTEGER,
                    suggested_stop_loss REAL,
                    suggested_take_profit REAL,
                    reasoning TEXT,
                    key_factors TEXT,
                    risk_assessment TEXT,
                    model_name TEXT,
                    timestamp TEXT NOT NULL,
                    processing_time_ms REAL,
                    raw_response TEXT,
                    FOREIGN KEY (trade_outcome_id) REFERENCES trade_outcomes (id)
                )
            """)
            
            # Table for aggregated performance metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    total_pnl REAL DEFAULT 0.0,
                    avg_win REAL DEFAULT 0.0,
                    avg_loss REAL DEFAULT 0.0,
                    profit_factor REAL DEFAULT 0.0,
                    sharpe_ratio REAL DEFAULT 0.0,
                    max_drawdown REAL DEFAULT 0.0,
                    llm_accuracy REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_outcomes_timestamp 
                ON trade_outcomes (timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_outcomes_symbol 
                ON trade_outcomes (symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_outcomes_outcome 
                ON trade_outcomes (outcome)
            """)
            
            conn.commit()
            logger.info(f"Trade logger database initialized: {self.db_path}")
    
    def log_trade_entry(
        self,
        outcome: TradeOutcome,
        llm_recommendation: Optional[TradeRecommendation] = None,
    ) -> int:
        """Log a new trade entry.
        
        Args:
            outcome: Trade outcome record
            llm_recommendation: LLM recommendation at entry
            
        Returns:
            Trade outcome ID
        """
        now = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert trade outcome
            cursor = conn.execute(
                """
                INSERT INTO trade_outcomes (
                    order_id, symbol, timestamp, action, quantity,
                    entry_price, exit_price, realized_pnl,
                    trade_duration_minutes, outcome,
                    entry_context, exit_context,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome.order_id,
                    outcome.symbol,
                    outcome.timestamp.isoformat(),
                    outcome.action,
                    outcome.quantity,
                    outcome.entry_price,
                    outcome.exit_price,
                    outcome.realized_pnl,
                    outcome.trade_duration_minutes,
                    outcome.outcome,
                    json.dumps(outcome.entry_context.to_dict()) if outcome.entry_context else None,
                    json.dumps(outcome.exit_context.to_dict()) if outcome.exit_context else None,
                    now,
                    now,
                )
            )
            trade_outcome_id = cursor.lastrowid
            
            # Insert LLM recommendation if provided
            if llm_recommendation:
                conn.execute(
                    """
                    INSERT INTO llm_recommendations (
                        trade_outcome_id, trade_decision, confidence,
                        suggested_position_size, suggested_stop_loss, suggested_take_profit,
                        reasoning, key_factors, risk_assessment,
                        model_name, timestamp, processing_time_ms, raw_response
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_outcome_id,
                        llm_recommendation.trade_decision,
                        llm_recommendation.confidence,
                        llm_recommendation.suggested_position_size,
                        llm_recommendation.suggested_stop_loss,
                        llm_recommendation.suggested_take_profit,
                        llm_recommendation.reasoning,
                        json.dumps(llm_recommendation.key_factors),
                        llm_recommendation.risk_assessment,
                        llm_recommendation.model_name,
                        llm_recommendation.timestamp.isoformat(),
                        llm_recommendation.processing_time_ms,
                        json.dumps(llm_recommendation.raw_response) if llm_recommendation.raw_response else None,
                    )
                )
            
            conn.commit()
            
            logger.info(
                f"Logged trade entry: {outcome.action} {outcome.quantity} {outcome.symbol} "
                f"@ ${outcome.entry_price:.2f} (ID: {trade_outcome_id})"
            )
            
            return trade_outcome_id
    
    def update_trade_exit(
        self,
        order_id: int,
        exit_price: float,
        realized_pnl: float,
        trade_duration_minutes: float,
        outcome: str,
        exit_context: Optional[TradingContext] = None,
    ) -> None:
        """Update trade with exit information.
        
        Args:
            order_id: Original order ID
            exit_price: Exit price
            realized_pnl: Realized profit/loss
            trade_duration_minutes: Trade duration in minutes
            outcome: Trade outcome ("WIN", "LOSS", "BREAKEVEN")
            exit_context: Market context at exit
        """
        now = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE trade_outcomes
                SET exit_price = ?,
                    realized_pnl = ?,
                    trade_duration_minutes = ?,
                    outcome = ?,
                    exit_context = ?,
                    updated_at = ?
                WHERE order_id = ?
                """,
                (
                    exit_price,
                    realized_pnl,
                    trade_duration_minutes,
                    outcome,
                    json.dumps(exit_context.to_dict()) if exit_context else None,
                    now,
                    order_id,
                )
            )
            conn.commit()
            
            logger.info(
                f"Updated trade exit: Order {order_id} @ ${exit_price:.2f} "
                f"P&L: ${realized_pnl:.2f} ({outcome})"
            )
    
    def get_recent_trades(
        self,
        limit: int = 100,
        outcome_filter: Optional[str] = None
    ) -> List[dict]:
        """Get recent trade records.
        
        Args:
            limit: Maximum number of records to return
            outcome_filter: Filter by outcome ("WIN", "LOSS", "OPEN", etc.)
            
        Returns:
            List of trade records with LLM recommendations
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = """
                SELECT 
                    t.*,
                    l.trade_decision, l.confidence, l.reasoning,
                    l.key_factors, l.model_name
                FROM trade_outcomes t
                LEFT JOIN llm_recommendations l ON t.id = l.trade_outcome_id
            """
            
            if outcome_filter:
                query += " WHERE t.outcome = ?"
                cursor = conn.execute(query + " ORDER BY t.timestamp DESC LIMIT ?", 
                                    (outcome_filter, limit))
            else:
                cursor = conn.execute(query + " ORDER BY t.timestamp DESC LIMIT ?", (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_performance_summary(self, days: int = 30) -> dict:
        """Get performance summary for recent period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with performance metrics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losing_trades,
                    SUM(realized_pnl) as total_pnl,
                    AVG(CASE WHEN outcome = 'WIN' THEN realized_pnl ELSE NULL END) as avg_win,
                    AVG(CASE WHEN outcome = 'LOSS' THEN realized_pnl ELSE NULL END) as avg_loss,
                    AVG(trade_duration_minutes) as avg_duration
                FROM trade_outcomes
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                    AND outcome IN ('WIN', 'LOSS', 'BREAKEVEN')
                """,
                (days,)
            )
            
            row = cursor.fetchone()
            
            total_trades = row[0] or 0
            winning_trades = row[1] or 0
            losing_trades = row[2] or 0
            total_pnl = row[3] or 0.0
            avg_win = row[4] or 0.0
            avg_loss = row[5] or 0.0
            avg_duration = row[6] or 0.0
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0.0
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "avg_duration_minutes": avg_duration,
                "profit_factor": profit_factor,
            }
    
    def export_training_data(self, output_path: Union[str, Path]) -> int:
        """Export trade data for LLM training.
        
        Args:
            output_path: Path to output JSON file
            
        Returns:
            Number of records exported
        """
        trades = self.get_recent_trades(limit=10000, outcome_filter=None)
        
        # Filter to only include closed trades with LLM recommendations
        training_data = [
            trade for trade in trades
            if trade.get("outcome") in ("WIN", "LOSS", "BREAKEVEN")
            and trade.get("trade_decision") is not None
        ]
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(training_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(training_data)} trades to {output_path}")
        return len(training_data)
