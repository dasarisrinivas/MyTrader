import sqlite3


def _create_orders_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            parent_order_id INTEGER,
            trade_cycle_id TEXT,
            features TEXT,
            rationale TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )


def test_backfill_only_fills_missing_fields(tmp_path):
    from tools.backfill_missing_order_snapshots import repair_missing_snapshots

    db = tmp_path / "orders.db"
    conn = sqlite3.connect(db)
    _create_orders_table(conn)

    # Target root order missing both
    conn.execute(
        "INSERT INTO orders(order_id,parent_order_id,trade_cycle_id,features,rationale,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
        (1, None, "cycleA", "", "", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )

    # Donor root within same cycle with snapshots
    conn.execute(
        "INSERT INTO orders(order_id,parent_order_id,trade_cycle_id,features,rationale,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
        (2, None, "cycleA", "{\"rsi\":55}", "{\"why\":\"x\"}", "2026-01-01T00:01:00", "2026-01-01T00:01:00"),
    )

    # Another cycle should not be used
    conn.execute(
        "INSERT INTO orders(order_id,parent_order_id,trade_cycle_id,features,rationale,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
        (3, None, "cycleB", "{\"rsi\":99}", "{\"why\":\"y\"}", "2026-01-01T00:02:00", "2026-01-01T00:02:00"),
    )

    # Target with existing features should not be overwritten
    conn.execute(
        "INSERT INTO orders(order_id,parent_order_id,trade_cycle_id,features,rationale,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
        (4, None, "cycleA", "{\"keep\":true}", "", "2026-01-01T00:03:00", "2026-01-01T00:03:00"),
    )

    conn.commit()
    conn.close()

    counts = repair_missing_snapshots(str(db), apply=True)
    assert counts.targets == 2  # order_id 1 and 4
    assert counts.updated == 2

    conn = sqlite3.connect(db)

    f1, r1 = conn.execute("SELECT features, rationale FROM orders WHERE order_id=1").fetchone()
    assert f1 == "{\"rsi\":55}"
    assert "__backfilled" in r1

    f4, r4 = conn.execute("SELECT features, rationale FROM orders WHERE order_id=4").fetchone()
    assert f4 == "{\"keep\":true}"  # not overwritten
    assert "__backfilled" in r4

    conn.close()
