import sqlite3

from shree.execution.live_trading_manager import LiveTradingManager


def _mk_db(tmp_path):
    db_path = tmp_path / "orders.db"
    conn = sqlite3.connect(db_path)
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
    conn.commit()
    return conn, db_path


def test_infers_profit_target_from_limit_child(tmp_path, monkeypatch):
    conn, db_path = _mk_db(tmp_path)

    # Root BUY @ 100, TP LIMIT child @ 110
    conn.execute(
        """INSERT INTO orders (
            order_id, parent_order_id, timestamp, symbol, action, quantity, order_type,
            limit_price, stop_price, entry_price, stop_loss, take_profit, trade_cycle_id,
            status, avg_fill_price, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            1,
            None,
            "2026-01-01T00:00:00",
            "TEST",
            "BUY",
            1,
            "MARKET",
            None,
            None,
            100.0,
            95.0,
            110.0,
            "tc1",
            "Filled",
            100.0,
            "2026-01-01T00:00:00",
            "2026-01-01T00:00:00",
        ),
    )
    conn.execute(
        """INSERT INTO orders (
            order_id, parent_order_id, timestamp, symbol, action, quantity, order_type,
            limit_price, stop_price, entry_price, trade_cycle_id,
            status, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            2,
            1,
            "2026-01-01T01:00:00",
            "TEST",
            "SELL",
            1,
            "LIMIT",
            110.0,
            None,
            100.0,
            "tc1",
            "Filled",
            "2026-01-01T01:00:00",
            "2026-01-01T01:00:00",
        ),
    )
    conn.execute(
        """INSERT INTO executions (order_id, timestamp, quantity, price, realized_pnl)
        VALUES (?,?,?,?,?)""",
        (2, "2026-01-01T01:00:00", 1, 110.0, 10.0),
    )
    conn.commit()

    # Hook LiveTradingManager helper in place
    manager = LiveTradingManager.__new__(LiveTradingManager)

    # We monkeypatch a tiny helper onto the instance for test purposes.
    # Implementation lives in live_trading_manager.py and reads tracker DB.
    from shree.monitoring.order_tracker import OrderTracker

    tracker = OrderTracker(db_path=db_path)
    # Avoid the tracker creating/migrating new tables in our scratch DB.
    tracker.db_path = db_path

    # Call the inference helper (added in main code)
    reason = LiveTradingManager._infer_bracket_fill_reason(
        manager,
        conn=conn,
        trade_cycle_id="tc1",
        current_direction="LONG",
    )

    assert reason == "PROFIT_TARGET"


def test_infers_stop_loss_from_stop_child(tmp_path):
    conn, db_path = _mk_db(tmp_path)

    # Root BUY @ 100, SL STOP child @ 95
    conn.execute(
        """INSERT INTO orders (
            order_id, parent_order_id, timestamp, symbol, action, quantity, order_type,
            limit_price, stop_price, entry_price, stop_loss, take_profit, trade_cycle_id,
            status, avg_fill_price, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            1,
            None,
            "2026-01-01T00:00:00",
            "TEST",
            "BUY",
            1,
            "MARKET",
            None,
            None,
            100.0,
            95.0,
            110.0,
            "tc2",
            "Filled",
            100.0,
            "2026-01-01T00:00:00",
            "2026-01-01T00:00:00",
        ),
    )
    conn.execute(
        """INSERT INTO orders (
            order_id, parent_order_id, timestamp, symbol, action, quantity, order_type,
            limit_price, stop_price, entry_price, trade_cycle_id,
            status, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            3,
            1,
            "2026-01-01T01:00:00",
            "TEST",
            "SELL",
            1,
            "STOP",
            None,
            95.0,
            100.0,
            "tc2",
            "Filled",
            "2026-01-01T01:00:00",
            "2026-01-01T01:00:00",
        ),
    )
    conn.execute(
        """INSERT INTO executions (order_id, timestamp, quantity, price, realized_pnl)
        VALUES (?,?,?,?,?)""",
        (3, "2026-01-01T01:00:00", 1, 95.0, -5.0),
    )
    conn.commit()

    manager = LiveTradingManager.__new__(LiveTradingManager)
    reason = LiveTradingManager._infer_bracket_fill_reason(
        manager,
        conn=conn,
        trade_cycle_id="tc2",
        current_direction="LONG",
    )
    assert reason == "STOP_LOSS"
