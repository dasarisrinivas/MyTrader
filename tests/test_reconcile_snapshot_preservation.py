import sqlite3
from datetime import datetime, timezone


def _create_orders_table(conn: sqlite3.Connection) -> None:
    # Minimal schema subset for this test.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            quantity REAL,
            order_type TEXT,
            limit_price REAL,
            stop_price REAL,
            status TEXT,
            filled_quantity REAL,
            avg_fill_price REAL,
            trade_cycle_id TEXT,
            features TEXT,
            rationale TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )


def test_reconcile_insert_preserves_existing_snapshots():
    """Reconcile INSERT OR REPLACE must not wipe features/rationale/trade_cycle_id if already present."""
    from mytrader.execution.reconcile import ReconcileManager, ReconcileAction, ReconcileConfig

    conn = sqlite3.connect(":memory:")
    _create_orders_table(conn)

    now = datetime.now(timezone.utc).isoformat()

    # Seed an order as if it was written by OrderTracker (with snapshots).
    conn.execute(
        """
        INSERT INTO orders (
            order_id, timestamp, symbol, action, quantity,
            order_type, limit_price, stop_price, status,
            filled_quantity, avg_fill_price,
            trade_cycle_id, features, rationale,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            123,
            now,
            "MES",
            "BUY",
            1,
            "LIMIT",
            5000.0,
            None,
            "Submitted",
            0,
            None,
            "cycle123",
            "{\"rsi\": 55}",
            "{\"why\": \"entry\"}",
            now,
            now,
        ),
    )
    conn.commit()

    # Build a reconcile action that *doesn't* include snapshot fields.
    action = ReconcileAction(
        action_type="insert",
        order_id=123,
        ib_order_id=123,
        db_order_id=123,
        reason="test",
        details={
            "order_id": 123,
            "symbol": "MES",
            "action": "BUY",
            "quantity": 1,
            "order_type": "LIMIT",
            "limit_price": 5000.0,
            "stop_price": None,
            "status": "Filled",
            "filled": 1,
            "avg_fill_price": 5000.25,
        },
    )

    # ReconcileManager requires an IB instance, but _execute_insert doesn't use it.
    # Use a lightweight stub to avoid pulling in ib_insync.
    class _IBStub:  # pragma: no cover
        pass

    reconciler = ReconcileManager(ib=_IBStub(), config=ReconcileConfig(db_path=":memory:"))

    # The manager creates the audit table in its own connection. Since we're passing
    # a separate in-memory connection into _execute_insert, we need to create the
    # audit table in this connection too.
    conn.execute(
        """
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
        """
    )

    # Call the internal insert executor; we don't want to pull in IB / filesystem.
    import asyncio

    asyncio.run(reconciler._execute_insert(action, conn, "corr", None))

    row = conn.execute(
        "SELECT trade_cycle_id, features, rationale, status, filled_quantity FROM orders WHERE order_id=123"
    ).fetchone()
    assert row is not None

    trade_cycle_id, features, rationale, status, filled_qty = row
    assert trade_cycle_id == "cycle123"
    assert features == "{\"rsi\": 55}"
    assert rationale == "{\"why\": \"entry\"}"
    assert status == "Filled"
    assert filled_qty == 1
