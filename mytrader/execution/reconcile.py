"""
Reconciliation Module for MyTrader
===================================
Handles startup reconciliation between IB orders/positions and local database.
Implements safety features: DRY_RUN, SAFE_MODE, staged actions, and audit logging.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from ib_insync import IB, Contract, Future, Order, Trade

from ..utils.logger import logger
from ..monitoring.order_tracker import OrderTracker


class ReconcileStatus(Enum):
    """Status of reconciliation process."""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    RECONCILE_DRY_RUN_COMPLETED = "RECONCILE_DRY_RUN_COMPLETED"
    AWAITING_DRY_RUN_CONFIRM = "AWAITING_DRY_RUN_CONFIRM"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SAFE_MODE_ACTIVE = "SAFE_MODE_ACTIVE"


class MatchStrategy(Enum):
    """Order matching strategies."""
    IB_ORDER_ID = "ib_order_id"
    SIGNATURE = "signature"


@dataclass
class ReconcileConfig:
    """Configuration for reconciliation process."""
    dry_run: bool = True
    safe_mode: bool = True
    dry_run_confirm: bool = False
    auto_manage_external_orders: bool = True
    force_sync_delete: bool = False
    backup_dir: str = "./backups"
    log_dir: str = "./logs"
    status_dir: str = "./status"
    alerts_dir: str = "./alerts"
    timeout_seconds: int = 60
    match_strategy: str = "ib_order_id"
    timestamp_match_window_seconds: int = 300
    require_exact_price_match: bool = True
    db_path: str = "./data/orders.db"
    audit_table: str = "order_audit"
    
    @classmethod
    def from_yaml(cls, yaml_path: str = "config/local_reconcile.yml") -> "ReconcileConfig":
        """Load configuration from YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            logger.warning(f"Config file not found at {yaml_path}, using defaults")
            return cls()
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        reconcile = data.get("RECONCILE", {})
        
        return cls(
            dry_run=data.get("DRY_RUN", True),
            safe_mode=data.get("SAFE_MODE", True),
            dry_run_confirm=data.get("DRY_RUN_CONFIRM", False),
            auto_manage_external_orders=data.get("AUTO_MANAGE_EXTERNAL_ORDERS", True),
            force_sync_delete=data.get("FORCE_SYNC_DELETE", False),
            backup_dir=data.get("BACKUP_DIR", "./backups"),
            log_dir=data.get("LOG_DIR", "./logs"),
            status_dir=data.get("STATUS_DIR", "./status"),
            alerts_dir=data.get("ALERTS_DIR", "./alerts"),
            timeout_seconds=reconcile.get("timeout_seconds", 60),
            match_strategy=reconcile.get("match_strategy", "ib_order_id"),
            timestamp_match_window_seconds=reconcile.get("timestamp_match_window_seconds", 300),
            require_exact_price_match=reconcile.get("require_exact_price_match", True),
            db_path=data.get("DATABASE", {}).get("orders_db_path", "./data/orders.db"),
            audit_table=data.get("DATABASE", {}).get("audit_table", "order_audit"),
        )


@dataclass
class IBOrderInfo:
    """Information about an order from IB."""
    order_id: int
    perm_id: int
    client_id: int
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    order_type: str
    limit_price: Optional[float]
    stop_price: Optional[float]
    status: str
    filled: int
    remaining: int
    avg_fill_price: Optional[float]
    parent_id: int
    order_ref: Optional[str]
    
    @classmethod
    def from_trade(cls, trade: Trade) -> "IBOrderInfo":
        """Create from ib_insync Trade object."""
        order = trade.order
        status = trade.orderStatus
        
        return cls(
            order_id=order.orderId,
            perm_id=order.permId,
            client_id=order.clientId,
            symbol=trade.contract.symbol if trade.contract else "",
            action=order.action,
            quantity=int(order.totalQuantity),
            order_type=order.orderType,
            limit_price=float(order.lmtPrice) if order.lmtPrice else None,
            stop_price=float(order.auxPrice) if order.auxPrice else None,
            status=status.status,
            filled=int(status.filled),
            remaining=int(status.remaining),
            avg_fill_price=float(status.avgFillPrice) if status.avgFillPrice > 0 else None,
            parent_id=order.parentId,
            order_ref=order.orderRef if order.orderRef else None,
        )


@dataclass
class IBPositionInfo:
    """Information about a position from IB."""
    account: str
    symbol: str
    security_type: str
    exchange: str
    quantity: int
    avg_cost: float
    
    @classmethod
    def from_position(cls, position) -> "IBPositionInfo":
        """Create from ib_insync Position object."""
        return cls(
            account=position.account,
            symbol=position.contract.symbol,
            security_type=position.contract.secType,
            exchange=position.contract.exchange,
            quantity=int(position.position),
            avg_cost=float(position.avgCost) / abs(position.position) if position.position != 0 else 0.0,
        )


@dataclass
class DBOrderInfo:
    """Information about an order from the database."""
    order_id: int
    symbol: str
    action: str
    quantity: int
    order_type: str
    limit_price: Optional[float]
    stop_price: Optional[float]
    status: str
    filled_quantity: int
    avg_fill_price: Optional[float]
    parent_order_id: Optional[int]
    timestamp: str
    created_at: str
    
    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "DBOrderInfo":
        """Create from database row dictionary."""
        return cls(
            order_id=row["order_id"],
            symbol=row["symbol"],
            action=row["action"],
            quantity=row["quantity"],
            order_type=row["order_type"],
            limit_price=row.get("limit_price"),
            stop_price=row.get("stop_price"),
            status=row["status"],
            filled_quantity=row.get("filled_quantity", 0),
            avg_fill_price=row.get("avg_fill_price"),
            parent_order_id=row.get("parent_order_id"),
            timestamp=row["timestamp"],
            created_at=row["created_at"],
        )


@dataclass
class ReconcileAction:
    """An action to be taken during reconciliation."""
    action_type: str  # "update", "delete", "insert", "close_recommendation", "take_profit"
    order_id: Optional[int]
    ib_order_id: Optional[int]
    db_order_id: Optional[int]
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    executed: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReconcilePlan:
    """Plan for reconciliation actions."""
    to_update: List[ReconcileAction] = field(default_factory=list)
    to_delete: List[ReconcileAction] = field(default_factory=list)
    to_insert: List[ReconcileAction] = field(default_factory=list)
    ambiguous_matches: List[ReconcileAction] = field(default_factory=list)
    spy_futures_actions: List[ReconcileAction] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "to_update": [a.to_dict() for a in self.to_update],
            "to_delete": [a.to_dict() for a in self.to_delete],
            "to_insert": [a.to_dict() for a in self.to_insert],
            "ambiguous_matches": [a.to_dict() for a in self.ambiguous_matches],
            "spy_futures_actions": [a.to_dict() for a in self.spy_futures_actions],
            "summary": {
                "update_count": len(self.to_update),
                "delete_count": len(self.to_delete),
                "insert_count": len(self.to_insert),
                "ambiguous_count": len(self.ambiguous_matches),
                "spy_futures_count": len(self.spy_futures_actions),
            }
        }


@dataclass
class ReconcileResult:
    """Result of reconciliation process."""
    status: ReconcileStatus
    plan: ReconcilePlan
    backup_file: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    inserted_count: int = 0
    deleted_count: int = 0
    updated_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "plan": self.plan.to_dict(),
            "backup_file": self.backup_file,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "inserted_count": self.inserted_count,
            "deleted_count": self.deleted_count,
            "updated_count": self.updated_count,
        }


class ReconcileLock:
    """
    Lock to prevent trading operations until reconciliation completes.
    Thread-safe implementation.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._reconcile_complete = threading.Event()
        self._safe_mode = False
    
    def acquire_reconcile_lock(self) -> bool:
        """Acquire the reconciliation lock. Returns True if successful."""
        return self._lock.acquire(blocking=False)
    
    def release_reconcile_lock(self) -> None:
        """Release the reconciliation lock and signal completion."""
        try:
            self._lock.release()
        except RuntimeError:
            pass  # Lock wasn't held
        self._reconcile_complete.set()
    
    def wait_for_reconcile(self, timeout: float = 60.0) -> bool:
        """Wait for reconciliation to complete. Returns True if completed."""
        return self._reconcile_complete.wait(timeout=timeout)
    
    def is_reconcile_complete(self) -> bool:
        """Check if reconciliation is complete."""
        return self._reconcile_complete.is_set()
    
    def set_safe_mode(self, enabled: bool) -> None:
        """Enable or disable safe mode."""
        self._safe_mode = enabled
    
    def is_safe_mode(self) -> bool:
        """Check if safe mode is active."""
        return self._safe_mode
    
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        if self._safe_mode:
            return False
        return self._reconcile_complete.is_set()
    
    def reset(self) -> None:
        """Reset the lock state for a new reconciliation."""
        self._reconcile_complete.clear()


class ReconcileManager:
    """
    Manages startup reconciliation between IB and local database.
    
    Safety features:
    - DRY_RUN mode: Simulates without executing
    - SAFE_MODE: Read-only reconciliation
    - Staged actions with confirmation
    - Database backups before any changes
    - Audit logging for all operations
    """
    
    # SPY futures symbols to handle specially
    SPY_FUTURES_SYMBOLS = {"ES", "MES", "SPY"}
    
    def __init__(
        self,
        ib: IB,
        config: Optional[ReconcileConfig] = None,
        order_tracker: Optional[OrderTracker] = None,
    ):
        self.ib = ib
        self.config = config or ReconcileConfig.from_yaml()
        self.order_tracker = order_tracker or OrderTracker(self.config.db_path)
        
        # Reconciliation state
        self.status = ReconcileStatus.NOT_STARTED
        self.lock = ReconcileLock()
        self.last_result: Optional[ReconcileResult] = None
        
        # Track initial state from IB
        self.initial_ib_orders: Dict[int, IBOrderInfo] = {}
        self.initial_ib_positions: Dict[str, IBPositionInfo] = {}
        
        # Idempotency tracking
        self._submission_signatures: Set[str] = set()
        
        # Ensure directories exist
        for dir_attr in ["backup_dir", "log_dir", "status_dir", "alerts_dir"]:
            Path(getattr(self.config, dir_attr)).mkdir(parents=True, exist_ok=True)
        
        # Initialize audit table
        self._init_audit_table()
        
        # Set safe mode if configured
        if self.config.safe_mode:
            self.lock.set_safe_mode(True)
    
    def _init_audit_table(self) -> None:
        """Initialize the order_audit table in the database."""
        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.audit_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    correlation_id TEXT,
                    action_type TEXT NOT NULL,
                    order_id INTEGER,
                    ib_order_id INTEGER,
                    reason TEXT,
                    details TEXT,
                    backup_file TEXT,
                    executed BOOLEAN DEFAULT FALSE,
                    error TEXT
                )
            """)
            conn.commit()
    
    def _log_event(
        self, 
        event_type: str, 
        data: Dict[str, Any], 
        level: str = "INFO",
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log structured event to reconcile.log."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id or self._generate_correlation_id(),
            "event_type": event_type,
            "level": level,
            "data": data,
        }
        
        log_file = Path(self.config.log_dir) / "reconcile.log"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Also log to standard logger
        log_msg = f"[{event_type}] {json.dumps(data)}"
        if level == "ERROR":
            logger.error(log_msg)
        elif level == "WARN":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID."""
        return hashlib.md5(
            f"{datetime.now().isoformat()}-{id(self)}".encode()
        ).hexdigest()[:12]
    
    def generate_submission_signature(
        self,
        symbol: str,
        quantity: int,
        price: Optional[float],
        side: str,
        timestamp: datetime,
    ) -> str:
        """
        Generate idempotency signature for order submission.
        
        Signature = sha256(symbol|qty|price|side|timestamp_rounded)
        Timestamp is rounded to 10-second windows.
        """
        # Round timestamp to 10-second window
        ts_rounded = (timestamp.timestamp() // 10) * 10
        
        data = f"{symbol}|{quantity}|{price or 'MKT'}|{side}|{ts_rounded}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    def is_duplicate_submission(self, signature: str) -> bool:
        """Check if submission signature is a duplicate."""
        return signature in self._submission_signatures
    
    def record_submission(self, signature: str) -> None:
        """Record a submission signature."""
        self._submission_signatures.add(signature)
    
    # =========================================================================
    # Backup Operations
    # =========================================================================
    
    async def run_backup(self) -> Tuple[bool, Optional[str]]:
        """
        Run database backup script.
        
        Returns:
            Tuple of (success, backup_file_path)
        """
        self._log_event("backup_started", {})
        
        backup_script = Path("scripts/backup_db.sh")
        if not backup_script.exists():
            self._log_event("backup_failed", {"reason": "script_not_found"}, level="ERROR")
            return False, None
        
        try:
            result = subprocess.run(
                ["bash", str(backup_script)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode != 0:
                self._log_event(
                    "backup_failed",
                    {"returncode": result.returncode, "stderr": result.stderr},
                    level="ERROR"
                )
                return False, None
            
            # Parse backup file from output
            backup_file = None
            for line in result.stdout.split("\n"):
                if "SQL Dump:" in line or "Binary:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        backup_file = parts[1].strip()
                        break
            
            self._log_event("backup_completed", {"backup_file": backup_file})
            return True, backup_file
            
        except subprocess.TimeoutExpired:
            self._log_event("backup_failed", {"reason": "timeout"}, level="ERROR")
            return False, None
        except Exception as e:
            self._log_event("backup_failed", {"reason": str(e)}, level="ERROR")
            return False, None
    
    # =========================================================================
    # IB Data Fetching
    # =========================================================================
    
    async def fetch_ib_orders(self, symbol: Optional[str] = None) -> List[IBOrderInfo]:
        """Fetch all open/working orders from IB."""
        if not self.ib.isConnected():
            self._log_event("fetch_ib_orders_failed", {"reason": "not_connected"}, level="ERROR")
            return []
        
        try:
            open_trades = self.ib.openTrades()
            orders = []
            
            for trade in open_trades:
                if symbol and trade.contract and trade.contract.symbol != symbol:
                    continue
                
                order_info = IBOrderInfo.from_trade(trade)
                orders.append(order_info)
                
                # Store as initial state
                self.initial_ib_orders[order_info.order_id] = order_info
            
            self._log_event(
                "fetch_ib_orders_completed",
                {"count": len(orders), "symbol_filter": symbol}
            )
            return orders
            
        except Exception as e:
            self._log_event("fetch_ib_orders_failed", {"error": str(e)}, level="ERROR")
            return []
    
    async def fetch_ib_positions(self, symbol: Optional[str] = None) -> List[IBPositionInfo]:
        """Fetch all positions from IB."""
        if not self.ib.isConnected():
            self._log_event("fetch_ib_positions_failed", {"reason": "not_connected"}, level="ERROR")
            return []
        
        try:
            positions = self.ib.positions()
            result = []
            
            for pos in positions:
                if pos.position == 0:
                    continue
                if symbol and pos.contract.symbol != symbol:
                    continue
                
                pos_info = IBPositionInfo.from_position(pos)
                result.append(pos_info)
                
                # Store as initial state
                self.initial_ib_positions[pos_info.symbol] = pos_info
            
            self._log_event(
                "fetch_ib_positions_completed",
                {"count": len(result), "symbol_filter": symbol}
            )
            return result
            
        except Exception as e:
            self._log_event("fetch_ib_positions_failed", {"error": str(e)}, level="ERROR")
            return []
    
    async def fetch_ib_fills(self) -> List[Dict[str, Any]]:
        """Fetch recent fills from IB."""
        if not self.ib.isConnected():
            return []
        
        try:
            fills = self.ib.fills()
            result = []
            
            for fill in fills:
                result.append({
                    "order_id": fill.execution.orderId,
                    "perm_id": fill.execution.permId,
                    "symbol": fill.contract.symbol,
                    "side": fill.execution.side,
                    "shares": fill.execution.shares,
                    "price": fill.execution.price,
                    "time": fill.execution.time.isoformat() if fill.execution.time else None,
                })
            
            self._log_event("fetch_ib_fills_completed", {"count": len(result)})
            return result
            
        except Exception as e:
            self._log_event("fetch_ib_fills_failed", {"error": str(e)}, level="ERROR")
            return []
    
    # =========================================================================
    # DB Data Fetching
    # =========================================================================
    
    def fetch_db_orders(
        self, 
        status_filter: Optional[List[str]] = None,
        symbol: Optional[str] = None,
    ) -> List[DBOrderInfo]:
        """
        Fetch orders from database.
        
        Args:
            status_filter: Filter by status values (e.g., ["Placed", "Submitted"])
            symbol: Filter by symbol
        """
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM orders WHERE 1=1"
                params = []
                
                if status_filter:
                    placeholders = ",".join("?" * len(status_filter))
                    query += f" AND status IN ({placeholders})"
                    params.extend(status_filter)
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                query += " ORDER BY timestamp DESC"
                
                cursor = conn.execute(query, params)
                rows = [dict(row) for row in cursor.fetchall()]
                
                orders = [DBOrderInfo.from_db_row(row) for row in rows]
                
                self._log_event(
                    "fetch_db_orders_completed",
                    {
                        "count": len(orders),
                        "status_filter": status_filter,
                        "symbol": symbol,
                    }
                )
                return orders
                
        except Exception as e:
            self._log_event("fetch_db_orders_failed", {"error": str(e)}, level="ERROR")
            return []
    
    # =========================================================================
    # Matching Logic
    # =========================================================================
    
    def _match_by_order_id(
        self,
        ib_orders: List[IBOrderInfo],
        db_orders: List[DBOrderInfo],
    ) -> Tuple[Dict[int, Tuple[IBOrderInfo, DBOrderInfo]], List[IBOrderInfo], List[DBOrderInfo]]:
        """
        Primary matching by IB order ID.
        
        Returns:
            Tuple of (matched, unmatched_ib, unmatched_db)
        """
        matched = {}
        ib_by_id = {o.order_id: o for o in ib_orders}
        db_by_id = {o.order_id: o for o in db_orders}
        
        for order_id, ib_order in ib_by_id.items():
            if order_id in db_by_id:
                matched[order_id] = (ib_order, db_by_id[order_id])
        
        matched_ids = set(matched.keys())
        unmatched_ib = [o for o in ib_orders if o.order_id not in matched_ids]
        unmatched_db = [o for o in db_orders if o.order_id not in matched_ids]
        
        return matched, unmatched_ib, unmatched_db
    
    def _match_by_signature(
        self,
        ib_order: IBOrderInfo,
        db_orders: List[DBOrderInfo],
    ) -> Optional[DBOrderInfo]:
        """
        Secondary matching by signature (symbol + side + qty + price + timestamp).
        Used when order ID matching fails.
        """
        for db_order in db_orders:
            # Check symbol and action
            if ib_order.symbol != db_order.symbol:
                continue
            if ib_order.action != db_order.action:
                continue
            if ib_order.quantity != db_order.quantity:
                continue
            
            # Check price if required
            if self.config.require_exact_price_match:
                if ib_order.limit_price != db_order.limit_price:
                    continue
            
            # Check timestamp window
            try:
                db_ts = datetime.fromisoformat(db_order.timestamp.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                time_diff = abs((now - db_ts).total_seconds())
                
                if time_diff <= self.config.timestamp_match_window_seconds:
                    return db_order
            except Exception:
                continue
        
        return None
    
    # =========================================================================
    # Reconciliation Plan Building
    # =========================================================================
    
    async def build_reconcile_plan(
        self,
        symbol: Optional[str] = None,
    ) -> ReconcilePlan:
        """
        Build reconciliation plan by comparing IB state with DB state.
        
        Args:
            symbol: Optional symbol to filter reconciliation
            
        Returns:
            ReconcilePlan with actions to take
        """
        plan = ReconcilePlan()
        
        # Fetch data
        ib_orders = await self.fetch_ib_orders(symbol)
        ib_positions = await self.fetch_ib_positions(symbol)
        
        # Fetch DB orders that are in pending/submitted states
        active_statuses = ["Placed", "Submitted", "PreSubmitted", "PendingSubmit"]
        db_orders = self.fetch_db_orders(status_filter=active_statuses, symbol=symbol)
        
        # Primary matching by order ID
        matched, unmatched_ib, unmatched_db = self._match_by_order_id(ib_orders, db_orders)
        
        # Process matched orders - check for status updates needed
        for order_id, (ib_order, db_order) in matched.items():
            if ib_order.status != db_order.status:
                plan.to_update.append(ReconcileAction(
                    action_type="update",
                    order_id=order_id,
                    ib_order_id=ib_order.order_id,
                    db_order_id=db_order.order_id,
                    reason="status_mismatch",
                    details={
                        "ib_status": ib_order.status,
                        "db_status": db_order.status,
                        "filled": ib_order.filled,
                        "avg_fill_price": ib_order.avg_fill_price,
                    }
                ))
        
        # Process unmatched DB orders (orphaned - exist in DB but not on IB)
        # Try secondary matching first
        still_unmatched_db = []
        for db_order in unmatched_db:
            # Only consider truly orphaned orders (not filled/cancelled)
            if db_order.status in ["Filled", "Cancelled", "Inactive"]:
                continue
            
            # Try signature matching
            secondary_match = None
            for ib_order in unmatched_ib:
                match = self._match_by_signature(ib_order, [db_order])
                if match:
                    secondary_match = ib_order
                    break
            
            if secondary_match:
                # Ambiguous match found
                plan.ambiguous_matches.append(ReconcileAction(
                    action_type="ambiguous",
                    order_id=db_order.order_id,
                    ib_order_id=secondary_match.order_id,
                    db_order_id=db_order.order_id,
                    reason="secondary_match_ambiguous",
                    details={
                        "ib_order": asdict(secondary_match),
                        "db_order": asdict(db_order),
                    }
                ))
            else:
                still_unmatched_db.append(db_order)
        
        # Mark truly orphaned DB orders for deletion
        for db_order in still_unmatched_db:
            plan.to_delete.append(ReconcileAction(
                action_type="delete",
                order_id=db_order.order_id,
                ib_order_id=None,
                db_order_id=db_order.order_id,
                reason="not_on_ib",
                details=asdict(db_order),
            ))
        
        # Process unmatched IB orders (exist on IB but not in DB)
        for ib_order in unmatched_ib:
            # Skip if already matched ambiguously
            ambiguous_ids = {a.ib_order_id for a in plan.ambiguous_matches}
            if ib_order.order_id in ambiguous_ids:
                continue
            
            plan.to_insert.append(ReconcileAction(
                action_type="insert",
                order_id=ib_order.order_id,
                ib_order_id=ib_order.order_id,
                db_order_id=None,
                reason="not_in_db",
                details=asdict(ib_order),
            ))
        
        # Handle SPY futures positions
        for pos in ib_positions:
            if pos.symbol in self.SPY_FUTURES_SYMBOLS:
                plan.spy_futures_actions.append(ReconcileAction(
                    action_type="initial_state",
                    order_id=None,
                    ib_order_id=None,
                    db_order_id=None,
                    reason="spy_futures_position_recognized",
                    details={
                        "symbol": pos.symbol,
                        "quantity": pos.quantity,
                        "avg_cost": pos.avg_cost,
                        "account": pos.account,
                    }
                ))
        
        self._log_event(
            "reconcile_plan_built",
            plan.to_dict()["summary"]
        )
        
        return plan
    
    # =========================================================================
    # Staged Actions Management
    # =========================================================================
    
    def save_staged_actions(self, plan: ReconcilePlan) -> str:
        """Save staged actions to JSON file."""
        staged_file = Path("staged_actions.json")
        
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "dry_run": self.config.dry_run,
                "safe_mode": self.config.safe_mode,
                "dry_run_confirm": self.config.dry_run_confirm,
            },
            "plan": plan.to_dict(),
        }
        
        with open(staged_file, "w") as f:
            json.dump(data, f, indent=2)
        
        self._log_event("staged_actions_saved", {"file": str(staged_file)})
        return str(staged_file)
    
    def load_staged_actions(self) -> Optional[Dict[str, Any]]:
        """Load staged actions from JSON file."""
        staged_file = Path("staged_actions.json")
        
        if not staged_file.exists():
            return None
        
        with open(staged_file) as f:
            return json.load(f)
    
    # =========================================================================
    # Action Execution
    # =========================================================================
    
    def _record_audit(
        self,
        action: ReconcileAction,
        backup_file: Optional[str],
        correlation_id: str,
        conn: sqlite3.Connection,
    ) -> None:
        """Record audit entry for an action."""
        conn.execute(f"""
            INSERT INTO {self.config.audit_table}
            (timestamp, correlation_id, action_type, order_id, ib_order_id, reason, details, backup_file, executed, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            correlation_id,
            action.action_type,
            action.order_id,
            action.ib_order_id,
            action.reason,
            json.dumps(action.details),
            backup_file,
            action.executed,
            action.error,
        ))
    
    async def execute_plan(
        self,
        plan: ReconcilePlan,
        backup_file: Optional[str],
    ) -> ReconcileResult:
        """
        Execute the reconciliation plan.
        
        Respects DRY_RUN, SAFE_MODE, and DRY_RUN_CONFIRM settings.
        """
        correlation_id = self._generate_correlation_id()
        result = ReconcileResult(
            status=ReconcileStatus.IN_PROGRESS,
            plan=plan,
            backup_file=backup_file,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        
        # Check if we should execute
        if self.config.dry_run:
            result.status = ReconcileStatus.RECONCILE_DRY_RUN_COMPLETED
            self._log_event(
                "reconcile_dry_run_completed",
                {"correlation_id": correlation_id},
            )
            result.completed_at = datetime.now(timezone.utc).isoformat()
            return result
        
        if self.config.safe_mode:
            result.status = ReconcileStatus.SAFE_MODE_ACTIVE
            self._log_event(
                "reconcile_safe_mode_active",
                {"correlation_id": correlation_id},
            )
            result.completed_at = datetime.now(timezone.utc).isoformat()
            return result
        
        if not self.config.dry_run_confirm:
            result.status = ReconcileStatus.AWAITING_DRY_RUN_CONFIRM
            self._log_event(
                "AWAITING_DRY_RUN_CONFIRM",
                {"correlation_id": correlation_id},
            )
            result.completed_at = datetime.now(timezone.utc).isoformat()
            return result
        
        # Execute actions in a transaction
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                conn.execute("BEGIN TRANSACTION")
                
                try:
                    # Execute deletions
                    for action in plan.to_delete:
                        await self._execute_delete(action, conn, correlation_id, backup_file)
                        if action.executed:
                            result.deleted_count += 1
                    
                    # Execute inserts
                    for action in plan.to_insert:
                        await self._execute_insert(action, conn, correlation_id, backup_file)
                        if action.executed:
                            result.inserted_count += 1
                    
                    # Execute updates
                    for action in plan.to_update:
                        await self._execute_update(action, conn, correlation_id, backup_file)
                        if action.executed:
                            result.updated_count += 1
                    
                    conn.commit()
                    result.status = ReconcileStatus.COMPLETED
                    
                except Exception as e:
                    conn.rollback()
                    raise
                    
        except Exception as e:
            result.status = ReconcileStatus.FAILED
            result.error = str(e)
            self._log_event(
                "reconcile_failed",
                {"error": str(e), "correlation_id": correlation_id},
                level="ERROR"
            )
            
            # Enter safe mode on failure
            self.lock.set_safe_mode(True)
        
        result.completed_at = datetime.now(timezone.utc).isoformat()
        
        # Log completion
        self._log_event(
            "reconcile_completed",
            {
                "status": result.status.value,
                "inserted": result.inserted_count,
                "deleted": result.deleted_count,
                "updated": result.updated_count,
                "correlation_id": correlation_id,
            }
        )
        
        return result
    
    async def _execute_delete(
        self,
        action: ReconcileAction,
        conn: sqlite3.Connection,
        correlation_id: str,
        backup_file: Optional[str],
    ) -> None:
        """Execute a delete action."""
        try:
            if self.config.force_sync_delete:
                # Hard delete
                conn.execute("DELETE FROM orders WHERE order_id = ?", (action.db_order_id,))
            else:
                # Soft delete - mark as orphaned_deleted
                conn.execute(
                    "UPDATE orders SET status = 'orphaned_deleted', updated_at = ? WHERE order_id = ?",
                    (datetime.now(timezone.utc).isoformat(), action.db_order_id)
                )
            
            action.executed = True
            self._record_audit(action, backup_file, correlation_id, conn)
            
            self._log_event(
                "order_deleted",
                {"order_id": action.db_order_id, "hard_delete": self.config.force_sync_delete},
            )
            
        except Exception as e:
            action.error = str(e)
            self._record_audit(action, backup_file, correlation_id, conn)
    
    async def _execute_insert(
        self,
        action: ReconcileAction,
        conn: sqlite3.Connection,
        correlation_id: str,
        backup_file: Optional[str],
    ) -> None:
        """Execute an insert action."""
        try:
            details = action.details
            now = datetime.now(timezone.utc).isoformat()
            
            conn.execute("""
                INSERT OR REPLACE INTO orders (
                    order_id, timestamp, symbol, action, quantity,
                    order_type, limit_price, stop_price, status,
                    filled_quantity, avg_fill_price, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                details.get("order_id"),
                now,
                details.get("symbol"),
                details.get("action"),
                details.get("quantity"),
                details.get("order_type"),
                details.get("limit_price"),
                details.get("stop_price"),
                details.get("status"),
                details.get("filled", 0),
                details.get("avg_fill_price"),
                now,
                now,
            ))
            
            action.executed = True
            self._record_audit(action, backup_file, correlation_id, conn)
            
            self._log_event(
                "order_inserted",
                {"order_id": details.get("order_id"), "symbol": details.get("symbol")},
            )
            
        except Exception as e:
            action.error = str(e)
            self._record_audit(action, backup_file, correlation_id, conn)
    
    async def _execute_update(
        self,
        action: ReconcileAction,
        conn: sqlite3.Connection,
        correlation_id: str,
        backup_file: Optional[str],
    ) -> None:
        """Execute an update action."""
        try:
            details = action.details
            now = datetime.now(timezone.utc).isoformat()
            
            conn.execute("""
                UPDATE orders
                SET status = ?, filled_quantity = ?, avg_fill_price = ?, updated_at = ?
                WHERE order_id = ?
            """, (
                details.get("ib_status"),
                details.get("filled", 0),
                details.get("avg_fill_price"),
                now,
                action.db_order_id,
            ))
            
            action.executed = True
            self._record_audit(action, backup_file, correlation_id, conn)
            
            self._log_event(
                "order_updated",
                {"order_id": action.db_order_id, "new_status": details.get("ib_status")},
            )
            
        except Exception as e:
            action.error = str(e)
            self._record_audit(action, backup_file, correlation_id, conn)
    
    # =========================================================================
    # Status and Snapshot Management
    # =========================================================================
    
    def save_status(self, result: ReconcileResult) -> None:
        """Save reconciliation status to status file."""
        status_file = Path(self.config.status_dir) / "reconcile_status.json"
        
        data = {
            "status": result.status.value,
            "last_run": result.completed_at or result.started_at,
            "inserted": result.inserted_count,
            "deleted": result.deleted_count,
            "updated": result.updated_count,
            "ambiguous_count": len(result.plan.ambiguous_matches),
            "backup_file": result.backup_file,
            "error": result.error,
        }
        
        with open(status_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def save_snapshot(self, result: ReconcileResult) -> str:
        """Save reconciliation snapshot with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = Path(self.config.status_dir) / f"reconcile_snapshot_{timestamp}.json"
        
        with open(snapshot_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return str(snapshot_file)
    
    def save_ambiguous_matches(self, plan: ReconcilePlan) -> Optional[str]:
        """Save ambiguous matches to separate file for review."""
        if not plan.ambiguous_matches:
            return None
        
        ambiguous_file = Path(self.config.log_dir) / "ambiguous_matches.json"
        
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(plan.ambiguous_matches),
            "matches": [a.to_dict() for a in plan.ambiguous_matches],
        }
        
        with open(ambiguous_file, "w") as f:
            json.dump(data, f, indent=2)
        
        self._log_event(
            "ambiguous_matches_found",
            {"count": len(plan.ambiguous_matches), "file": str(ambiguous_file)},
            level="WARN"
        )
        
        return str(ambiguous_file)
    
    # =========================================================================
    # Main Reconciliation Entry Point
    # =========================================================================
    
    async def run_reconciliation(
        self,
        symbol: Optional[str] = None,
    ) -> ReconcileResult:
        """
        Run the full reconciliation process.
        
        Steps:
        1. Acquire reconcile lock
        2. Run database backup
        3. Fetch IB and DB state
        4. Build reconciliation plan
        5. Save staged actions
        6. Execute plan (respecting safety settings)
        7. Save status and snapshot
        8. Release lock
        
        Args:
            symbol: Optional symbol to filter reconciliation
            
        Returns:
            ReconcileResult with reconciliation outcome
        """
        self._log_event("reconcile_started", {"symbol": symbol})
        self.status = ReconcileStatus.IN_PROGRESS
        
        # Acquire lock
        if not self.lock.acquire_reconcile_lock():
            self._log_event("reconcile_lock_failed", {}, level="ERROR")
            return ReconcileResult(
                status=ReconcileStatus.FAILED,
                plan=ReconcilePlan(),
                error="Could not acquire reconcile lock",
            )
        
        try:
            # Step 1: Run backup
            backup_success, backup_file = await self.run_backup()
            if not backup_success:
                self._log_event("backup_failed", {}, level="ERROR")
                return ReconcileResult(
                    status=ReconcileStatus.FAILED,
                    plan=ReconcilePlan(),
                    error="Backup failed - aborting reconciliation",
                )
            
            # Step 2: Build reconciliation plan
            plan = await self.build_reconcile_plan(symbol)
            
            # Step 3: Save staged actions
            self.save_staged_actions(plan)
            
            # Step 4: Check for ambiguous matches
            if plan.ambiguous_matches:
                self.save_ambiguous_matches(plan)
            
            # Step 5: Execute plan
            result = await self.execute_plan(plan, backup_file)
            
            # Step 6: Save status and snapshot
            self.save_status(result)
            snapshot_file = self.save_snapshot(result)
            
            self._log_event(
                "reconcile_finished",
                {
                    "status": result.status.value,
                    "snapshot": snapshot_file,
                }
            )
            
            self.last_result = result
            self.status = result.status
            
            return result
            
        except Exception as e:
            self._log_event("reconcile_error", {"error": str(e)}, level="ERROR")
            self.lock.set_safe_mode(True)
            return ReconcileResult(
                status=ReconcileStatus.FAILED,
                plan=ReconcilePlan(),
                error=str(e),
            )
            
        finally:
            self.lock.release_reconcile_lock()
    
    # =========================================================================
    # SPY Futures Handling
    # =========================================================================
    
    def is_initial_state_order(self, order_id: int) -> bool:
        """Check if an order is from initial IB state (external order)."""
        return order_id in self.initial_ib_orders
    
    def is_initial_state_position(self, symbol: str) -> bool:
        """Check if a position is from initial IB state."""
        return symbol in self.initial_ib_positions
    
    def get_initial_position(self, symbol: str) -> Optional[IBPositionInfo]:
        """Get initial position info for a symbol."""
        return self.initial_ib_positions.get(symbol)
    
    async def evaluate_spy_futures_position(
        self,
        symbol: str,
        current_price: float,
        max_unrealized_loss_pct: float = 2.0,
        take_profit_threshold_pct: float = 1.5,
    ) -> Optional[ReconcileAction]:
        """
        Evaluate a SPY futures position for risk/profit actions.
        
        Returns:
            ReconcileAction if action needed, None otherwise
        """
        position = self.initial_ib_positions.get(symbol)
        if not position:
            return None
        
        if position.quantity == 0:
            return None
        
        # Calculate unrealized PnL
        if position.quantity > 0:  # Long
            unrealized_pnl_pct = ((current_price - position.avg_cost) / position.avg_cost) * 100
        else:  # Short
            unrealized_pnl_pct = ((position.avg_cost - current_price) / position.avg_cost) * 100
        
        # Check risk threshold
        if unrealized_pnl_pct < -max_unrealized_loss_pct:
            return ReconcileAction(
                action_type="close_recommendation",
                order_id=None,
                ib_order_id=None,
                db_order_id=None,
                reason="risk_threshold_exceeded",
                details={
                    "symbol": symbol,
                    "quantity": position.quantity,
                    "avg_cost": position.avg_cost,
                    "current_price": current_price,
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                    "threshold": -max_unrealized_loss_pct,
                }
            )
        
        # Check take profit threshold
        if unrealized_pnl_pct >= take_profit_threshold_pct:
            return ReconcileAction(
                action_type="take_profit",
                order_id=None,
                ib_order_id=None,
                db_order_id=None,
                reason="take_profit_threshold_met",
                details={
                    "symbol": symbol,
                    "quantity": position.quantity,
                    "avg_cost": position.avg_cost,
                    "current_price": current_price,
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                    "threshold": take_profit_threshold_pct,
                }
            )
        
        return None
