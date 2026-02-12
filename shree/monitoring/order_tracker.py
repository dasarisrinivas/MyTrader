"""Order tracking system with SQLite persistence."""
from __future__ import annotations

import sqlite3
import os
import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import json

from shree.utils.logger import logger


class OrderTracker:
    """Track orders and their lifecycle in an SQLite database."""

    def validate_last_trade_time(self, symbol: str, max_past_days: int = 7) -> bool:
        """Check if last trade time is not in the future or implausibly old."""
        ts = self.get_last_trade_time(symbol)
        if not ts:
            return True
        now = datetime.now(timezone.utc)
        if ts > now:
            logger.warning(f"Last trade time for {symbol} is in the future: {ts}")
            return False
        if (now - ts).total_seconds() > max_past_days * 86400:
            logger.warning(
                f"Last trade time for {symbol} is more than {max_past_days} days ago: {ts}"
            )
            return False
        return True
    
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
                    gross_pnl REAL,
                    net_pnl REAL,
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
                if "gross_pnl" not in columns:
                    conn.execute("ALTER TABLE orders ADD COLUMN gross_pnl REAL")
                if "net_pnl" not in columns:
                    conn.execute("ALTER TABLE orders ADD COLUMN net_pnl REAL")
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
                    gross_pnl REAL,
                    net_pnl REAL,
                    FOREIGN KEY (order_id) REFERENCES orders (order_id)
                )
            """)
            try:
                cursor = conn.execute("PRAGMA table_info(executions)")
                exec_columns = [row[1] for row in cursor.fetchall()]
                if "gross_pnl" not in exec_columns:
                    conn.execute("ALTER TABLE executions ADD COLUMN gross_pnl REAL")
                if "net_pnl" not in exec_columns:
                    conn.execute("ALTER TABLE executions ADD COLUMN net_pnl REAL")
            except Exception as exec_err:
                logger.error(f"Execution table migration failed: {exec_err}")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS submission_signatures (
                    signature TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price_bucket REAL,
                    bar_timestamp TEXT,
                    signal_id TEXT,
                    strategy_name TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbol_state (
                    symbol TEXT PRIMARY KEY,
                    last_trade_time TEXT,
                    updated_at TEXT NOT NULL
                )
            """)

            # Trade-level outcomes table (deterministic closure record).
            # This exists alongside order-level rows so audits can query closures without log parsing.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    trade_cycle_id TEXT PRIMARY KEY,
                    root_order_id INTEGER,
                    symbol TEXT NOT NULL,
                    entry_time TEXT,
                    exit_time TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    exit_reason TEXT,
                    realized_pnl REAL,
                    gross_pnl REAL,
                    net_pnl REAL,
                    commission REAL,
                    extra_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trade_outcomes_symbol_time
                ON trade_outcomes(symbol, exit_time)
                """
            )
            
            conn.commit()
            logger.info(f"Order tracker database initialized: {self.db_path}")

    def upsert_trade_entry(
        self,
        trade_cycle_id: str,
        root_order_id: int,
        symbol: str,
        entry_time: Optional[str] = None,
        entry_price: Optional[float] = None,
        quantity: Optional[int] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        """Persist a trade entry snapshot keyed by trade_cycle_id."""
        if not trade_cycle_id:
            return
        now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        payload = json.dumps(extra or {}) if extra is not None else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trade_outcomes (
                    trade_cycle_id, root_order_id, symbol, entry_time, entry_price, quantity,
                    extra_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_cycle_id) DO UPDATE SET
                    root_order_id = COALESCE(excluded.root_order_id, trade_outcomes.root_order_id),
                    symbol = COALESCE(excluded.symbol, trade_outcomes.symbol),
                    entry_time = COALESCE(excluded.entry_time, trade_outcomes.entry_time),
                    entry_price = COALESCE(excluded.entry_price, trade_outcomes.entry_price),
                    quantity = COALESCE(excluded.quantity, trade_outcomes.quantity),
                    extra_json = COALESCE(excluded.extra_json, trade_outcomes.extra_json),
                    updated_at = excluded.updated_at
                """,
                (trade_cycle_id, root_order_id, symbol, entry_time, entry_price, quantity, payload, now, now),
            )
            conn.commit()

    def finalize_trade_exit(
        self,
        trade_cycle_id: str,
        exit_time: Optional[str] = None,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> Optional[Dict[str, float]]:
        """Finalize trade outcome using rolled-up P&L on the trade root order.

        Returns:
            Dict with realized/gross/net/commission if updated, else None.
        """
        if not trade_cycle_id:
            return None
        now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        payload = json.dumps(extra or {}) if extra is not None else None
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            root = conn.execute(
                """
                SELECT order_id, symbol, avg_fill_price, quantity, realized_pnl, gross_pnl, net_pnl, commission
                FROM orders
                WHERE trade_cycle_id = ? AND parent_order_id IS NULL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (trade_cycle_id,),
            ).fetchone()
            if not root:
                try:
                    logger.warning(
                        "trade_outcomes finalize skipped: no root order for trade_cycle_id=%s (exit_reason=%s)",
                        str(trade_cycle_id),
                        str(exit_reason) if exit_reason is not None else "",
                    )
                except Exception:
                    pass
                return None

            pnl = {
                "realized_pnl": float(root["realized_pnl"] or 0.0),
                "gross_pnl": float(root["gross_pnl"] or 0.0),
                "net_pnl": float(root["net_pnl"] or 0.0),
                "commission": float(root["commission"] or 0.0),
            }

            conn.execute(
                """
                INSERT INTO trade_outcomes (
                    trade_cycle_id, root_order_id, symbol, entry_time, exit_time,
                    entry_price, exit_price, quantity, exit_reason,
                    realized_pnl, gross_pnl, net_pnl, commission,
                    extra_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_cycle_id) DO UPDATE SET
                    root_order_id = excluded.root_order_id,
                    symbol = excluded.symbol,
                    exit_time = COALESCE(excluded.exit_time, trade_outcomes.exit_time),
                    exit_price = COALESCE(excluded.exit_price, trade_outcomes.exit_price),
                    exit_reason = COALESCE(excluded.exit_reason, trade_outcomes.exit_reason),
                    realized_pnl = excluded.realized_pnl,
                    gross_pnl = excluded.gross_pnl,
                    net_pnl = excluded.net_pnl,
                    commission = excluded.commission,
                    extra_json = COALESCE(excluded.extra_json, trade_outcomes.extra_json),
                    updated_at = excluded.updated_at
                """,
                (
                    trade_cycle_id,
                    int(root["order_id"]),
                    str(root["symbol"]),
                    None,
                    exit_time,
                    float(root["avg_fill_price"] or 0.0),
                    exit_price,
                    int(root["quantity"] or 0),
                    exit_reason,
                    pnl["realized_pnl"],
                    pnl["gross_pnl"],
                    pnl["net_pnl"],
                    pnl["commission"],
                    payload,
                    now,
                    now,
                ),
            )
            conn.commit()

        return pnl
    
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
        rationale: Optional[object] = None,
        features: Optional[object] = None,
        market_regime: Optional[str] = None,
        trade_cycle_id: Optional[str] = None,
    ) -> None:
        """Record a new order placement."""
        now = datetime.utcnow().isoformat()

        def _coerce_json_payload(value: Optional[object]) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, str):
                return value
            try:
                return json.dumps(value)
            except Exception:
                # Best-effort: keep going without blocking persistence.
                return None

        features_json = _coerce_json_payload(features)
        rationale_json = _coerce_json_payload(rationale)

        def _is_missing_snapshot(value: Optional[str]) -> bool:
            if value is None:
                return True
            if not str(value).strip():
                return True
            if str(value).strip() == "{}":
                return True
            return False

        # For forensic/audit integrity: root entries should carry a feature/rationale snapshot.
        # Child bracket orders intentionally omit these.
        if parent_order_id is None and trade_cycle_id and (
            _is_missing_snapshot(features_json) or _is_missing_snapshot(rationale_json)
        ):
            callsite = self._format_order_placement_callsite()

            logger.warning(
                "Order placement missing features/rationale (root order). trade_cycle_id=%s order_id=%s symbol=%s features=%s rationale=%s%s",
                trade_cycle_id,
                order_id,
                symbol,
                "None" if features_json is None else "set",
                "None" if rationale_json is None else "set",
                callsite,
            )

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
                    take_profit, confidence, atr, rationale_json, features_json, market_regime,
                trade_cycle_id, "Placed", now, now
            ))
            
            conn.execute("""
                INSERT INTO order_events (order_id, timestamp, event_type, status, message)
                VALUES (?, ?, ?, ?, ?)
            """, (order_id, now, "PLACED", "Placed", f"{action} {quantity} {symbol}"))
            
            conn.commit()
        
        logger.info(f"📝 Recorded order placement: ID={order_id}, {action} {quantity} {symbol}")

    @staticmethod
    def _format_order_placement_callsite() -> str:
        """Return optional callsite suffix for root-order placement warnings.

        Gated behind env var SHREE_ORDER_TRACKER_CALLSITE to avoid overhead/noise.
        """

        if os.getenv("SHREE_ORDER_TRACKER_CALLSITE", "").strip().lower() not in {"1", "true", "yes"}:
            return ""
        try:
            stack = inspect.stack()

            # 0: _format_order_placement_callsite
            # 1: record_order_placement OR direct caller (if helper called directly)
            # 2+: upstream
            for frame in stack[1:]:
                filename = frame.filename or ""
                function = frame.function or ""

                # Skip our own tracker frames.
                if "order_tracker.py" in filename and function in {
                    "_format_order_placement_callsite",
                    "record_order_placement",
                }:
                    continue

                # Skip pytest harness frames so tests can assert on their own function name.
                if "site-packages/_pytest/" in filename or filename.endswith("/_pytest/python.py"):
                    continue
                if function.startswith("pytest_"):
                    continue

                return f" caller={filename}:{frame.lineno} {function}"
        except Exception:
            return ""
        return ""
    
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
        gross_pnl: Optional[float] = None,
        net_pnl: Optional[float] = None,
    ) -> None:
        """Record an order execution."""
        now = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO executions (order_id, timestamp, quantity, price, commission, realized_pnl, gross_pnl, net_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (order_id, now, quantity, price, commission, realized_pnl, gross_pnl, net_pnl))
            
            # Update order with execution info
            conn.execute("""
                UPDATE orders
                SET avg_fill_price = ?,
                    commission = COALESCE(commission, 0) + COALESCE(?, 0),
                    realized_pnl = COALESCE(realized_pnl, 0) + COALESCE(?, 0),
                    gross_pnl = COALESCE(gross_pnl, 0) + COALESCE(?, 0),
                    net_pnl = COALESCE(net_pnl, 0) + COALESCE(?, 0),
                    updated_at = ?
                WHERE order_id = ?
            """, (price, commission, realized_pnl, gross_pnl, net_pnl, now, order_id))
            
            # Record event
            conn.execute("""
                INSERT INTO order_events (order_id, timestamp, event_type, message)
                VALUES (?, ?, ?, ?)
            """, (order_id, now, "EXECUTION", f"Filled {quantity} @ {price:.2f}"))

            # IMPORTANT: realized P&L is often recorded on the *exit* (child) order_id.
            # Roll that P&L up to the trade root so trade-level analytics using the parent
            # row (or trade_cycle_id group) reflect the actual outcome.
            self._rollup_trade_pnl(conn, order_id)
            
            conn.commit()
        
        logger.info(f"✅ Execution recorded: Order {order_id}, {quantity} @ {price:.2f}")

    def _get_trade_root_order_id(self, conn: sqlite3.Connection, order_id: int) -> int:
        """Return the top-level parent order_id for a bracket tree."""
        conn.row_factory = sqlite3.Row
        current = order_id
        # Walk up parent pointers until we reach a node with no parent.
        for _ in range(20):  # defensive bound against cycles
            row = conn.execute(
                "SELECT parent_order_id FROM orders WHERE order_id = ?",
                (current,),
            ).fetchone()
            if not row:
                return current
            parent = row["parent_order_id"]
            if parent is None:
                return current
            if parent == current:
                return current
            current = int(parent)
        return current

    def _rollup_trade_pnl(self, conn: sqlite3.Connection, order_id: int) -> None:
        """Aggregate child order P&L onto the trade root row.

        We roll up realized/gross/net/commission across all orders in the same bracket tree
        *and* trade_cycle_id (when present) onto the root order row. This makes the root
        order act like the canonical "trade" record.
        """
        conn.row_factory = sqlite3.Row
        root_id = self._get_trade_root_order_id(conn, order_id)
        root = conn.execute(
            "SELECT trade_cycle_id FROM orders WHERE order_id = ?",
            (root_id,),
        ).fetchone()
        trade_cycle_id = root["trade_cycle_id"] if root else None

        # Prefer trade_cycle_id grouping when available; otherwise fall back to bracket tree.
        if trade_cycle_id:
            agg = conn.execute(
                """
                SELECT
                    SUM(COALESCE(realized_pnl, 0)) AS realized,
                    SUM(COALESCE(gross_pnl, 0)) AS gross,
                    SUM(COALESCE(net_pnl, 0)) AS net,
                    SUM(COALESCE(commission, 0)) AS commission
                FROM orders
                WHERE trade_cycle_id = ?
                """,
                (trade_cycle_id,),
            ).fetchone()
        else:
            agg = conn.execute(
                """
                SELECT
                    SUM(COALESCE(realized_pnl, 0)) AS realized,
                    SUM(COALESCE(gross_pnl, 0)) AS gross,
                    SUM(COALESCE(net_pnl, 0)) AS net,
                    SUM(COALESCE(commission, 0)) AS commission
                FROM orders
                WHERE order_id = ? OR parent_order_id = ?
                """,
                (root_id, root_id),
            ).fetchone()

        if not agg:
            return

        conn.execute(
            """
            UPDATE orders
            SET
                realized_pnl = ?,
                gross_pnl = ?,
                net_pnl = ?,
                commission = ?,
                updated_at = ?
            WHERE order_id = ?
            """,
            (
                float(agg["realized"] or 0.0),
                float(agg["gross"] or 0.0),
                float(agg["net"] or 0.0),
                float(agg["commission"] or 0.0),
                datetime.utcnow().isoformat(),
                root_id,
            ),
        )
    
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
                    SUM(COALESCE(gross_pnl, 0)) as total_gross_pnl,
                    SUM(COALESCE(net_pnl, 0)) as total_net_pnl,
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
                "total_gross_pnl": row[4] or 0.0,
                "total_net_pnl": row[5] or 0.0,
                "total_commission": row[6] or 0.0,
                "avg_pnl_per_trade": row[7] or 0.0,
                "winning_trades": row[8] or 0,
                "losing_trades": row[9] or 0,
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

    def record_submission_signature(
        self,
        signature: str,
        symbol: str,
        action: str,
        quantity: int,
        price_bucket: float | None,
        bar_timestamp: str | None,
        signal_id: str | None,
        strategy_name: str | None,
    ) -> None:
        """Persist an idempotency signature so restarts remember submissions."""
        now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO submission_signatures (
                    signature, symbol, action, quantity, price_bucket,
                    bar_timestamp, signal_id, strategy_name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signature,
                    symbol,
                    action,
                    quantity,
                    price_bucket,
                    bar_timestamp,
                    signal_id,
                    strategy_name,
                    now,
                ),
            )

    def signature_exists(self, signature: str, ttl_seconds: int) -> bool:
        """Check if an idempotency signature exists and is still valid."""
        cutoff = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(seconds=ttl_seconds)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT created_at FROM submission_signatures WHERE signature = ?",
                (signature,),
            )
            row = cursor.fetchone()
            if not row:
                return False
            created_at = row["created_at"]
            if not created_at:
                conn.execute("DELETE FROM submission_signatures WHERE signature = ?", (signature,))
                return False
            created_dt = datetime.fromisoformat(created_at)
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            if created_dt < cutoff:
                conn.execute("DELETE FROM submission_signatures WHERE signature = ?", (signature,))
                return False
        return True

    def record_last_trade_time(self, symbol: str, timestamp: datetime) -> None:
        """Persist the last trade time per symbol to enforce cooldown across restarts."""
        ts = timestamp.astimezone(timezone.utc).isoformat()
        now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO symbol_state(symbol, last_trade_time, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    last_trade_time = excluded.last_trade_time,
                    updated_at = excluded.updated_at
                """,
                (symbol, ts, now),
            )

    def get_last_trade_time(self, symbol: str) -> datetime | None:
        """Retrieve the last trade timestamp for a symbol if recorded."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT last_trade_time FROM symbol_state WHERE symbol = ?",
                (symbol,),
            )
            row = cursor.fetchone()
            if not row or not row["last_trade_time"]:
                return None
            ts = datetime.fromisoformat(row["last_trade_time"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts

    def get_order_protection(self, order_id: int) -> Optional[Dict[str, Optional[float]]]:
        """Return persisted protection levels for an order or its parent if available."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT order_id, parent_order_id, entry_price, stop_loss, take_profit
                FROM orders
                WHERE order_id = ? OR parent_order_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (order_id, order_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "order_id": row["order_id"],
                "parent_order_id": row["parent_order_id"],
                "entry_price": row["entry_price"],
                "stop_loss": row["stop_loss"],
                "take_profit": row["take_profit"],
            }

    def reset_symbol_state(self, symbol: str) -> None:
        """Remove persisted cooldown/lock state for a symbol."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM symbol_state WHERE symbol = ?", (symbol,))
            conn.commit()
        logger.info(f"♻️  Cleared symbol state for {symbol}")
