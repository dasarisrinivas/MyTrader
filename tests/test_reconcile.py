"""
Unit Tests for Reconciliation Module
=====================================
Tests for startup reconciliation, order matching, and safety features.
"""
import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.mock_ib_server import MockIB, create_mock_ib_for_reconcile_test


class TestReconcileDeletedAndInserted(unittest.TestCase):
    """
    Test: test_reconcile_deleted_and_inserted
    
    Scenario:
    - IB has orders A, B
    - DB has orders A, C
    
    Expected:
    - A: matched (may need update)
    - B: in to_insert (not in DB)
    - C: in to_delete (not on IB)
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_orders.db")
        
        # Initialize database with schema
        self._init_test_db()
        
        # Create mock IB
        self.mock_ib = MockIB()
        self.mock_ib._connected = True
        
        # Add order A and B to IB
        self.mock_ib.add_open_order(
            order_id=100,
            symbol="ES",
            action="BUY",
            quantity=1,
            limit_price=5000.0,
            status="Submitted",
        )
        self.mock_ib.add_open_order(
            order_id=200,
            symbol="ES",
            action="SELL",
            quantity=2,
            limit_price=5010.0,
            status="Submitted",
        )
        
        # Add order A and C to DB
        self._add_db_order(100, "ES", "BUY", 1, "Placed")  # A
        self._add_db_order(300, "ES", "BUY", 1, "Submitted")  # C
    
    def _init_test_db(self):
        """Initialize test database with required schema."""
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
                    status TEXT NOT NULL,
                    filled_quantity INTEGER DEFAULT 0,
                    avg_fill_price REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS order_audit (
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
    
    def _add_db_order(self, order_id, symbol, action, quantity, status):
        """Add an order to the test database."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO orders (order_id, timestamp, symbol, action, quantity, order_type, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'LMT', ?, ?, ?)
            """, (order_id, now, symbol, action, quantity, status, now, now))
            conn.commit()
    
    def test_reconcile_plan_building(self):
        """Test that reconciliation correctly identifies to_insert and to_delete."""
        from mytrader.execution.reconcile import ReconcileManager, ReconcileConfig
        
        # Create config pointing to test DB
        config = ReconcileConfig(
            dry_run=True,
            safe_mode=True,
            db_path=self.db_path,
            backup_dir=self.temp_dir,
            log_dir=self.temp_dir,
            status_dir=self.temp_dir,
            alerts_dir=self.temp_dir,
        )
        
        # Create reconcile manager with mock IB
        manager = ReconcileManager(self.mock_ib, config)
        
        # Run reconciliation plan building
        async def run_test():
            plan = await manager.build_reconcile_plan("ES")
            return plan
        
        plan = asyncio.run(run_test())
        
        # Verify results
        # Order A (100): Should be in neither to_insert nor to_delete (matched)
        # Order B (200): Should be in to_insert (on IB but not in DB)
        # Order C (300): Should be in to_delete (in DB but not on IB)
        
        to_insert_ids = [a.ib_order_id for a in plan.to_insert]
        to_delete_ids = [a.db_order_id for a in plan.to_delete]
        
        self.assertIn(200, to_insert_ids, "Order B should be in to_insert")
        self.assertIn(300, to_delete_ids, "Order C should be in to_delete")
        self.assertNotIn(100, to_insert_ids, "Order A should not be in to_insert")
        self.assertNotIn(100, to_delete_ids, "Order A should not be in to_delete")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestIdempotenceReconcile(unittest.TestCase):
    """
    Test: test_idempotence_reconcile
    
    Run reconcile twice and expect no duplicate inserts.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_orders.db")
        self._init_test_db()
        
        self.mock_ib = MockIB()
        self.mock_ib._connected = True
        
        # Add one order to IB that's not in DB
        self.mock_ib.add_open_order(
            order_id=100,
            symbol="ES",
            action="BUY",
            quantity=1,
            limit_price=5000.0,
            status="Submitted",
        )
    
    def _init_test_db(self):
        """Initialize test database."""
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
                    status TEXT NOT NULL,
                    filled_quantity INTEGER DEFAULT 0,
                    avg_fill_price REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS order_audit (
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
    
    def test_idempotent_inserts(self):
        """Test that running reconcile twice doesn't create duplicate inserts."""
        from mytrader.execution.reconcile import ReconcileManager, ReconcileConfig
        
        config = ReconcileConfig(
            dry_run=False,
            safe_mode=False,
            dry_run_confirm=True,  # Allow execution
            db_path=self.db_path,
            backup_dir=self.temp_dir,
            log_dir=self.temp_dir,
            status_dir=self.temp_dir,
            alerts_dir=self.temp_dir,
        )
        
        manager = ReconcileManager(self.mock_ib, config)
        
        async def run_test():
            # First run - should insert
            with patch.object(manager, 'run_backup', new_callable=AsyncMock) as mock_backup:
                mock_backup.return_value = (True, "backup.sql")
                result1 = await manager.run_reconciliation("ES")
            
            # Second run - should not insert again
            with patch.object(manager, 'run_backup', new_callable=AsyncMock) as mock_backup:
                mock_backup.return_value = (True, "backup.sql")
                result2 = await manager.run_reconciliation("ES")
            
            return result1, result2
        
        result1, result2 = asyncio.run(run_test())
        
        # First run should have inserted
        self.assertEqual(result1.inserted_count, 1, "First run should insert 1 order")
        
        # Second run should not insert (already exists)
        self.assertEqual(result2.inserted_count, 0, "Second run should not insert any orders")
        
        # Verify only one record in DB
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM orders WHERE order_id = 100")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1, "Should have exactly one order 100 in DB")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestSPYHandling(unittest.TestCase):
    """
    Test: test_spy_handling
    
    Simulate SPY futures position on IB at startup and verify bot recognizes it.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_orders.db")
        self._init_test_db()
        
        self.mock_ib = MockIB()
        self.mock_ib._connected = True
        
        # Add ES position (SPY futures)
        self.mock_ib.add_position(
            symbol="ES",
            quantity=2,
            avg_cost=4990.0,
        )
        
        # Also add MES position
        self.mock_ib.add_position(
            symbol="MES",
            quantity=-1,
            avg_cost=5010.0,
        )
    
    def _init_test_db(self):
        """Initialize test database."""
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
                    status TEXT NOT NULL,
                    filled_quantity INTEGER DEFAULT 0,
                    avg_fill_price REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS order_audit (
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
    
    def test_spy_futures_recognized(self):
        """Test that SPY futures positions are recognized as initial state."""
        from mytrader.execution.reconcile import ReconcileManager, ReconcileConfig
        
        config = ReconcileConfig(
            dry_run=True,
            safe_mode=True,
            db_path=self.db_path,
            backup_dir=self.temp_dir,
            log_dir=self.temp_dir,
            status_dir=self.temp_dir,
            alerts_dir=self.temp_dir,
        )
        
        manager = ReconcileManager(self.mock_ib, config)
        
        async def run_test():
            plan = await manager.build_reconcile_plan()
            return plan
        
        plan = asyncio.run(run_test())
        
        # Should have SPY futures actions
        self.assertTrue(len(plan.spy_futures_actions) > 0, "Should have SPY futures actions")
        
        # Check that ES and MES are recognized
        spy_symbols = [a.details.get("symbol") for a in plan.spy_futures_actions]
        self.assertIn("ES", spy_symbols, "ES should be recognized as SPY futures")
        self.assertIn("MES", spy_symbols, "MES should be recognized as SPY futures")
        
        # Verify positions are in initial state
        self.assertTrue(manager.is_initial_state_position("ES"), "ES should be initial state")
        self.assertTrue(manager.is_initial_state_position("MES"), "MES should be initial state")
    
    def test_spy_futures_evaluation(self):
        """Test evaluation of SPY futures positions for risk/profit actions."""
        from mytrader.execution.reconcile import ReconcileManager, ReconcileConfig
        
        config = ReconcileConfig(
            dry_run=True,
            safe_mode=True,
            db_path=self.db_path,
            backup_dir=self.temp_dir,
            log_dir=self.temp_dir,
            status_dir=self.temp_dir,
            alerts_dir=self.temp_dir,
        )
        
        manager = ReconcileManager(self.mock_ib, config)
        
        async def run_test():
            # First fetch positions to populate initial state
            await manager.fetch_ib_positions()
            
            # Test close recommendation (price dropped significantly)
            close_action = await manager.evaluate_spy_futures_position(
                symbol="ES",
                current_price=4850.0,  # Price dropped from 4990
                max_unrealized_loss_pct=2.0,
            )
            
            # Test take profit (price increased)
            profit_action = await manager.evaluate_spy_futures_position(
                symbol="ES",
                current_price=5100.0,  # Price rose from 4990
                take_profit_threshold_pct=1.5,
            )
            
            return close_action, profit_action
        
        close_action, profit_action = asyncio.run(run_test())
        
        # Should recommend closing due to loss
        self.assertIsNotNone(close_action, "Should have close recommendation")
        self.assertEqual(close_action.action_type, "close_recommendation")
        
        # Should recommend take profit
        self.assertIsNotNone(profit_action, "Should have take profit recommendation")
        self.assertEqual(profit_action.action_type, "take_profit")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestReconcileLock(unittest.TestCase):
    """Test the reconcile lock mechanism."""
    
    def test_lock_prevents_trading_before_reconcile(self):
        """Test that lock prevents trading before reconciliation completes."""
        from mytrader.execution.reconcile import ReconcileLock
        
        lock = ReconcileLock()
        
        # Initially, trading should not be allowed
        self.assertFalse(lock.can_trade(), "Trading should be blocked before reconcile")
        
        # Acquire lock for reconciliation
        acquired = lock.acquire_reconcile_lock()
        self.assertTrue(acquired, "Should be able to acquire lock")
        
        # Still can't trade
        self.assertFalse(lock.can_trade(), "Trading should be blocked during reconcile")
        
        # Release lock (complete reconciliation)
        lock.release_reconcile_lock()
        
        # Now trading should be allowed
        self.assertTrue(lock.can_trade(), "Trading should be allowed after reconcile")
    
    def test_safe_mode_blocks_trading(self):
        """Test that safe mode blocks trading even after reconciliation."""
        from mytrader.execution.reconcile import ReconcileLock
        
        lock = ReconcileLock()
        lock.release_reconcile_lock()  # Complete reconcile
        
        # Trading should be allowed
        self.assertTrue(lock.can_trade())
        
        # Enable safe mode
        lock.set_safe_mode(True)
        
        # Trading should be blocked
        self.assertFalse(lock.can_trade(), "Safe mode should block trading")
        
        # Disable safe mode
        lock.set_safe_mode(False)
        
        # Trading should be allowed again
        self.assertTrue(lock.can_trade())


if __name__ == "__main__":
    unittest.main()
