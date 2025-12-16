"""Order tracking system with SQLite persistence."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from mytrader.utils.logger import logger


class OrderTracker:
    """Track orders and their lifecycle in SQLite database."""
    
    def __init__(self, db_path: str | Path = None):
        """Initialize order tracker with SQLite database."""
        if db_path is None:
            # Use project root/data/orders.db by default
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "orders.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id INTEGER PRIMARY KEY,
                    parent_order_id INTEGER,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_type TEXT NOT NULL,
                    limit_price REAL,
                    stop_price REAL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    confidence REAL,
                    atr REAL,
                    rationale TEXT,
                    features TEXT,
                    market_regime TEXT,
                    trade_cycle_id TEXT,
                    status TEXT NOT NULL,
                    filled_quantity INTEGER DEFAULT 0,
                    avg_fill_price REAL,
                    commission REAL,
                    realized_pnl REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Migration: Add new columns if they don't exist
            try:
                cursor = conn.execute("PRAGMA table_info(orders)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if "rationale" not in columns:
                    conn.execute("ALTER TABLE orders ADD COLUMN rationale TEXT")
                if "features" not in columns:
                    conn.execute("ALTER TABLE orders ADD COLUMN features TEXT")
                if "market_regime" not in columns:
                    conn.execute("ALTER TABLE orders ADD COLUMN market_regime TEXT")
                if "trade_cycle_id" not in columns:
                    conn.execute("ALTER TABLE orders ADD COLUMN trade_cycle_id TEXT")
            except Exception as e:
                logger.error(f"Migration failed: {e}")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS order_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT,
                    filled INTEGER,
                    remaining INTEGER,
                    avg_fill_price REAL,
                    message TEXT,
                    FOREIGN KEY (order_id) REFERENCES orders (order_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    commission REAL,
                    realized_pnl REAL,
                    FOREIGN KEY (order_id) REFERENCES orders (order_id)
                )
            """)
            
            conn.commit()
            logger.info(f"Order tracker database initialized: {self.db_path}")
    
    def record_order_placement(
        self,
        order_id: int,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        confidence: Optional[float] = None,
        atr: Optional[float] = None,
        parent_order_id: Optional[int] = None,
        rationale: Optional[str] = None,
        features: Optional[str] = None,
        market_regime: Optional[str] = None,
        trade_cycle_id: Optional[str] = None,
    ) -> None:
        """Record a new order placement."""
        now = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO orders (
                    order_id, parent_order_id, timestamp, symbol, action, quantity,
                    order_type, limit_price, stop_price, entry_price, stop_loss,
                    take_profit, confidence, atr, rationale, features, market_regime,
                    trade_cycle_id,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_id, parent_order_id, now, symbol, action, quantity,
                order_type, limit_price, stop_price, entry_price, stop_loss,
                take_profit, confidence, atr, rationale, features, market_regime,
                trade_cycle_id, "Placed", now, now
            ))
            
            conn.execute("""
                INSERT INTO order_events (order_id, timestamp, event_type, status, message)
                VALUES (?, ?, ?, ?, ?)
            """, (order_id, now, "PLACED", "Placed", f"{action} {quantity} {symbol}"))
            
            conn.commit()
        
        logger.info(f"📝 Recorded order placement: ID={order_id}, {action} {quantity} {symbol}")
    
    def update_order_status(
        self,
        order_id: int,
        status: str,
        filled: Optional[int] = None,
        remaining: Optional[int] = None,
        avg_fill_price: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        """Update order status."""
        now = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Update order
            updates = ["status = ?", "updated_at = ?"]
            values = [status, now]
            
            if filled is not None:
                updates.append("filled_quantity = ?")
                values.append(filled)
            
            if avg_fill_price is not None:
                updates.append("avg_fill_price = ?")
                values.append(avg_fill_price)
            
            values.append(order_id)
            
            conn.execute(f"""
                UPDATE orders
                SET {', '.join(updates)}
                WHERE order_id = ?
            """, values)
            
            # Record event
            conn.execute("""
                INSERT INTO order_events (
                    order_id, timestamp, event_type, status, filled, remaining,
                    avg_fill_price, message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (order_id, now, "STATUS_UPDATE", status, filled, remaining, avg_fill_price, message))
            
            conn.commit()
        
        logger.info(f"📊 Order {order_id} status: {status}" + (f" ({filled} filled)" if filled else ""))
    
    def record_execution(
        self,
        order_id: int,
        quantity: int,
        price: float,
        commission: Optional[float] = None,
        realized_pnl: Optional[float] = None,
    ) -> None:
        """Record an order execution."""
        now = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO executions (order_id, timestamp, quantity, price, commission, realized_pnl)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (order_id, now, quantity, price, commission, realized_pnl))
            
            # Update order with execution info
            conn.execute("""
                UPDATE orders
                SET avg_fill_price = ?,
                    commission = COALESCE(commission, 0) + COALESCE(?, 0),
                    realized_pnl = COALESCE(realized_pnl, 0) + COALESCE(?, 0),
                    updated_at = ?
                WHERE order_id = ?
            """, (price, commission, realized_pnl, now, order_id))
            
            # Record event
            conn.execute("""
                INSERT INTO order_events (order_id, timestamp, event_type, message)
                VALUES (?, ?, ?, ?)
            """, (order_id, now, "EXECUTION", f"Filled {quantity} @ {price:.2f}"))
            
            conn.commit()
        
        logger.info(f"✅ Execution recorded: Order {order_id}, {quantity} @ {price:.2f}")
    
    def get_all_orders(self, limit: int = 100) -> List[Dict]:
        """Get all orders with their latest status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    o.*,
                    (SELECT COUNT(*) FROM executions e WHERE e.order_id = o.order_id) as execution_count,
                    (SELECT COUNT(*) FROM order_events ev WHERE ev.order_id = o.order_id) as event_count
                FROM orders o
                ORDER BY o.timestamp DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_order_details(self, order_id: int) -> Optional[Dict]:
        """Get detailed information about a specific order."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get order
            cursor = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
            order = cursor.fetchone()
            if not order:
                return None
            
            order_dict = dict(order)
            
            # Get events
            cursor = conn.execute("""
                SELECT * FROM order_events
                WHERE order_id = ?
                ORDER BY timestamp ASC
            """, (order_id,))
            order_dict['events'] = [dict(row) for row in cursor.fetchall()]
            
            # Get executions
            cursor = conn.execute("""
                SELECT * FROM executions
                WHERE order_id = ?
                ORDER BY timestamp ASC
            """, (order_id,))
            order_dict['executions'] = [dict(row) for row in cursor.fetchall()]
            
            return order_dict
    
    def get_active_orders(self) -> List[Dict]:
        """Get all active (not filled/cancelled) orders."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM orders
                WHERE status NOT IN ('Filled', 'Cancelled', 'Inactive')
                ORDER BY timestamp DESC
            """)
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_orders_by_date(self, start_date: str, end_date: Optional[str] = None) -> List[Dict]:
        """Get orders within a date range."""
        if end_date is None:
            end_date = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM orders
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            """, (start_date, end_date))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary from all orders."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_orders,
                    SUM(CASE WHEN status = 'Filled' THEN 1 ELSE 0 END) as filled_orders,
                    SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END) as cancelled_orders,
                    SUM(COALESCE(realized_pnl, 0)) as total_pnl,
                    SUM(COALESCE(commission, 0)) as total_commission,
                    AVG(CASE WHEN realized_pnl IS NOT NULL THEN realized_pnl ELSE NULL END) as avg_pnl_per_trade,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losing_trades
                FROM orders
                WHERE status = 'Filled'
            """)
            
            row = cursor.fetchone()
            return {
                "total_orders": row[0] or 0,
                "filled_orders": row[1] or 0,
                "cancelled_orders": row[2] or 0,
                "total_pnl": row[3] or 0.0,
                "total_commission": row[4] or 0.0,
                "avg_pnl_per_trade": row[5] or 0.0,
                "winning_trades": row[6] or 0,
                "losing_trades": row[7] or 0,
            }
    
    def clear_old_orders(self, days: int = 7) -> None:
        """Clear orders older than specified days."""
        cutoff = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM order_events
                WHERE order_id IN (
                    SELECT order_id FROM orders
                    WHERE timestamp < datetime('now', '-' || ? || ' days')
                )
            """, (days,))
            
            conn.execute("""
                DELETE FROM executions
                WHERE order_id IN (
                    SELECT order_id FROM orders
                    WHERE timestamp < datetime('now', '-' || ? || ' days')
                )
            """, (days,))
            
            conn.execute("""
                DELETE FROM orders
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            conn.commit()
        
        logger.info(f"🧹 Cleared orders older than {days} days")
