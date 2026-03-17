"""SQLite Manager for Bedrock calls logging and state management.

This module provides persistent storage for:
- Bedrock API call logs (prompts, responses, costs)
- Daily quota tracking
- Event trigger history
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import logger


class BedrockSQLiteManager:
    """SQLite manager with WAL mode for Bedrock call logging."""
    
    # Default daily quota (can be overridden in config)
    DEFAULT_DAILY_QUOTA = 1000  # API calls per day
    DEFAULT_DAILY_COST_LIMIT = 50.0  # USD per day
    
    def __init__(
        self,
        db_path: str = "data/bedrock_hybrid.db",
        daily_quota: int = DEFAULT_DAILY_QUOTA,
        daily_cost_limit: float = DEFAULT_DAILY_COST_LIMIT,
    ):
        """Initialize SQLite manager.
        
        Args:
            db_path: Path to SQLite database file
            daily_quota: Maximum API calls per day
            daily_cost_limit: Maximum cost in USD per day
        """
        self.db_path = Path(db_path)
        self.daily_quota = daily_quota
        self.daily_cost_limit = daily_cost_limit
        
        # Thread-local storage for connections
        self._local = threading.local()
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        logger.info(f"Initialized BedrockSQLiteManager at {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
        return self._local.connection
    
    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database transaction error: {e}")
            raise
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._transaction() as conn:
            # Bedrock API calls table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bedrock_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    trigger TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT,
                    model TEXT NOT NULL,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    cost_estimate REAL DEFAULT 0.0,
                    latency_ms REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'success',
                    error_message TEXT,
                    context_hash TEXT,
                    bias_result TEXT,
                    confidence REAL
                )
            """)
            
            # Create indices for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bedrock_calls_ts 
                ON bedrock_calls(ts)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bedrock_calls_trigger 
                ON bedrock_calls(trigger)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bedrock_calls_context_hash 
                ON bedrock_calls(context_hash)
            """)
            
            # Event trigger history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS event_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    trigger_type TEXT NOT NULL,
                    reason TEXT,
                    payload TEXT,
                    bedrock_call_id INTEGER,
                    FOREIGN KEY (bedrock_call_id) REFERENCES bedrock_calls(id)
                )
            """)
            
            # Daily quota tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_quota (
                    date TEXT PRIMARY KEY,
                    call_count INTEGER DEFAULT 0,
                    total_tokens_in INTEGER DEFAULT 0,
                    total_tokens_out INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            logger.info("Database schema initialized")
    
    def log_bedrock_call(
        self,
        trigger: str,
        prompt: str,
        response: Optional[str],
        model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_estimate: float = 0.0,
        latency_ms: float = 0.0,
        status: str = "success",
        error_message: Optional[str] = None,
        context_hash: Optional[str] = None,
        bias_result: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """Log a Bedrock API call.
        
        Args:
            trigger: What triggered the call (e.g., 'market_open', 'volatility_spike')
            prompt: The prompt sent to Bedrock
            response: The response from Bedrock
            model: Model ID used
            tokens_in: Input tokens
            tokens_out: Output tokens
            cost_estimate: Estimated cost in USD
            latency_ms: Response latency in milliseconds
            status: 'success' or 'error'
            error_message: Error message if status is 'error'
            context_hash: Hash of context for caching
            bias_result: The bias result from response (BUY/SELL/HOLD)
            confidence: Confidence score from response
            
        Returns:
            ID of the inserted record
        """
        with self._transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO bedrock_calls 
                (trigger, prompt, response, model, tokens_in, tokens_out, 
                 cost_estimate, latency_ms, status, error_message, 
                 context_hash, bias_result, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trigger, prompt, response, model, tokens_in, tokens_out,
                cost_estimate, latency_ms, status, error_message,
                context_hash, bias_result, confidence
            ))
            call_id = cursor.lastrowid
            
            # Update daily quota
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            conn.execute("""
                INSERT INTO daily_quota (date, call_count, total_tokens_in, total_tokens_out, total_cost)
                VALUES (?, 1, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    call_count = call_count + 1,
                    total_tokens_in = total_tokens_in + excluded.total_tokens_in,
                    total_tokens_out = total_tokens_out + excluded.total_tokens_out,
                    total_cost = total_cost + excluded.total_cost,
                    last_updated = CURRENT_TIMESTAMP
            """, (today, tokens_in, tokens_out, cost_estimate))
            
            logger.debug(f"Logged Bedrock call {call_id}: trigger={trigger}, status={status}")
            return call_id
    
    def log_event_trigger(
        self,
        trigger_type: str,
        reason: str,
        payload: Optional[Dict] = None,
        bedrock_call_id: Optional[int] = None,
    ) -> int:
        """Log an event trigger.
        
        Args:
            trigger_type: Type of trigger (e.g., 'market_open', 'volatility_spike')
            reason: Human-readable reason
            payload: Event payload as dict
            bedrock_call_id: Associated Bedrock call ID (if any)
            
        Returns:
            ID of the inserted record
        """
        payload_json = json.dumps(payload) if payload else None
        
        with self._transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO event_triggers (trigger_type, reason, payload, bedrock_call_id)
                VALUES (?, ?, ?, ?)
            """, (trigger_type, reason, payload_json, bedrock_call_id))
            
            return cursor.lastrowid
    
    def get_recent_bedrock_calls(
        self,
        limit: int = 10,
        trigger_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent Bedrock calls.
        
        Args:
            limit: Maximum number of calls to return
            trigger_filter: Optional filter by trigger type
            
        Returns:
            List of call records as dicts
        """
        conn = self._get_connection()
        
        if trigger_filter:
            cursor = conn.execute("""
                SELECT * FROM bedrock_calls 
                WHERE trigger = ?
                ORDER BY ts DESC 
                LIMIT ?
            """, (trigger_filter, limit))
        else:
            cursor = conn.execute("""
                SELECT * FROM bedrock_calls 
                ORDER BY ts DESC 
                LIMIT ?
            """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_call_by_context_hash(self, context_hash: str) -> Optional[Dict[str, Any]]:
        """Get a Bedrock call by context hash (for caching).
        
        Args:
            context_hash: Hash of the context
            
        Returns:
            Call record or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM bedrock_calls 
            WHERE context_hash = ? AND status = 'success'
            ORDER BY ts DESC 
            LIMIT 1
        """, (context_hash,))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_daily_stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get daily statistics.
        
        Args:
            date: Date string (YYYY-MM-DD), defaults to today
            
        Returns:
            Dictionary with daily stats
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM daily_quota WHERE date = ?
        """, (date,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        else:
            return {
                "date": date,
                "call_count": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_cost": 0.0,
            }
    
    def check_quota(self) -> Tuple[bool, str]:
        """Check if daily quota is within limits.
        
        Returns:
            Tuple of (is_within_quota, message)
        """
        stats = self.get_daily_stats()
        
        if stats["call_count"] >= self.daily_quota:
            return False, f"Daily call quota exceeded: {stats['call_count']}/{self.daily_quota}"
        
        if stats["total_cost"] >= self.daily_cost_limit:
            return False, f"Daily cost limit exceeded: ${stats['total_cost']:.2f}/${self.daily_cost_limit:.2f}"
        
        return True, f"Quota OK: {stats['call_count']}/{self.daily_quota} calls, ${stats['total_cost']:.2f}/${self.daily_cost_limit:.2f}"
    
    def get_total_cost_today(self) -> float:
        """Get total cost for today.
        
        Returns:
            Total cost in USD
        """
        stats = self.get_daily_stats()
        return stats.get("total_cost", 0.0)
    
    def get_total_calls_today(self) -> int:
        """Get total calls for today.
        
        Returns:
            Number of calls
        """
        stats = self.get_daily_stats()
        return stats.get("call_count", 0)
    
    def get_cost_summary(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get cost summary for the last N days.
        
        Args:
            days: Number of days to include
            
        Returns:
            List of daily summaries
        """
        conn = self._get_connection()
        
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        
        cursor = conn.execute("""
            SELECT * FROM daily_quota 
            WHERE date >= ?
            ORDER BY date DESC
        """, (start_date,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_records(self, days_to_keep: int = 30) -> int:
        """Clean up old records.
        
        Args:
            days_to_keep: Number of days of records to keep
            
        Returns:
            Number of deleted records
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).strftime("%Y-%m-%d %H:%M:%S")
        
        with self._transaction() as conn:
            # Delete old event triggers first (foreign key constraint)
            conn.execute("""
                DELETE FROM event_triggers 
                WHERE ts < ?
            """, (cutoff,))
            
            # Delete old Bedrock calls
            cursor = conn.execute("""
                DELETE FROM bedrock_calls 
                WHERE ts < ?
            """, (cutoff,))
            
            deleted_count = cursor.rowcount
            
            # Delete old daily quota records
            date_cutoff = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).strftime("%Y-%m-%d")
            conn.execute("""
                DELETE FROM daily_quota 
                WHERE date < ?
            """, (date_cutoff,))
            
            logger.info(f"Cleaned up {deleted_count} old Bedrock call records")
            return deleted_count
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            logger.debug("Database connection closed")


# Singleton instance for module-level access
_manager_instance: Optional[BedrockSQLiteManager] = None


def get_sqlite_manager(
    db_path: str = "data/bedrock_hybrid.db",
    daily_quota: int = BedrockSQLiteManager.DEFAULT_DAILY_QUOTA,
    daily_cost_limit: float = BedrockSQLiteManager.DEFAULT_DAILY_COST_LIMIT,
) -> BedrockSQLiteManager:
    """Get or create the singleton SQLite manager instance.
    
    Args:
        db_path: Path to SQLite database file
        daily_quota: Maximum API calls per day
        daily_cost_limit: Maximum cost in USD per day
        
    Returns:
        BedrockSQLiteManager instance
    """
    global _manager_instance
    
    if _manager_instance is None:
        _manager_instance = BedrockSQLiteManager(
            db_path=db_path,
            daily_quota=daily_quota,
            daily_cost_limit=daily_cost_limit,
        )
    
    return _manager_instance
