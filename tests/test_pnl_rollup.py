import sqlite3
from pathlib import Path

from shree.monitoring.order_tracker import OrderTracker


def _init_in_memory_schema(conn: sqlite3.Connection) -> None:
    # Mimic OrderTracker schema (subset needed for roll-up)
    conn.execute(
        """
        CREATE TABLE orders (
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
        """
    )
    conn.execute(
        """
        CREATE TABLE executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            commission REAL,
            realized_pnl REAL,
            gross_pnl REAL,
            net_pnl REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE order_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            status TEXT,
            filled INTEGER,
            remaining INTEGER,
            avg_fill_price REAL,
            message TEXT
        )
        """
    )


def test_rollup_realized_pnl_from_child_to_parent(tmp_path: Path) -> None:
    db_path = tmp_path / "orders.db"
    tracker = OrderTracker(db_path=db_path)

    # Parent entry order
    tracker.record_order_placement(
        order_id=100,
        symbol="ES",
        action="BUY",
        quantity=1,
        order_type="LIMIT",
        entry_price=5000.0,
        stop_loss=4990.0,
        take_profit=5010.0,
        trade_cycle_id="tc1",
    )

    # Child stop order that realizes the loss
    tracker.record_order_placement(
        order_id=101,
        parent_order_id=100,
        symbol="ES",
        action="SELL",
        quantity=1,
        order_type="STOP",
        stop_price=4990.0,
        trade_cycle_id="tc1",
    )

    # Record execution against child order
    tracker.record_execution(
        order_id=101,
        quantity=1,
        price=4990.0,
        commission=1.25,
        realized_pnl=-50.0,
        gross_pnl=-50.0,
        net_pnl=-51.25,
    )

    # Parent should have rolled-up PnL
    parent = tracker.get_order_details(100)
    assert parent is not None
    assert (parent.get("realized_pnl") or 0.0) == -50.0
    assert (parent.get("gross_pnl") or 0.0) == -50.0
    assert (parent.get("net_pnl") or 0.0) == -51.25
    assert (parent.get("commission") or 0.0) == 1.25


def test_trade_outcomes_entry_and_exit_finalize(tmp_path: Path) -> None:
    db_path = tmp_path / "orders.db"
    tracker = OrderTracker(db_path=db_path)

    # Simulate an entry snapshot for a trade cycle (even before orders fill)
    tracker.upsert_trade_entry(
        trade_cycle_id="tc2",
        root_order_id=200,
        symbol="ES",
        entry_time="2026-01-13T00:00:00+00:00",
        entry_price=5000.0,
        quantity=1,
        extra={"note": "entry"},
    )

    # Create the root order row (what finalize_trade_exit uses for roll-up totals)
    tracker.record_order_placement(
        order_id=200,
        symbol="ES",
        action="BUY",
        quantity=1,
        order_type="LIMIT",
        entry_price=5000.0,
        stop_loss=4990.0,
        take_profit=5010.0,
        trade_cycle_id="tc2",
    )

    # Record an execution directly on the parent for simplicity
    tracker.record_execution(
        order_id=200,
        quantity=1,
        price=5000.0,
        commission=1.0,
        realized_pnl=25.0,
        gross_pnl=25.0,
        net_pnl=24.0,
    )

    updated = tracker.finalize_trade_exit(
        trade_cycle_id="tc2",
        exit_time="2026-01-13T00:10:00+00:00",
        exit_price=5005.0,
        exit_reason="TEST",
        extra={"note": "exit"},
    )
    assert updated is not None
    assert updated["realized_pnl"] == 25.0

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT trade_cycle_id, exit_reason, realized_pnl FROM trade_outcomes WHERE trade_cycle_id = ?",
            ("tc2",),
        ).fetchone()
    assert row is not None
    assert row["trade_cycle_id"] == "tc2"
    assert row["exit_reason"] == "TEST"
    assert float(row["realized_pnl"]) == 25.0
