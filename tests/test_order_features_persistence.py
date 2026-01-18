import sqlite3
from pathlib import Path

from mytrader.monitoring.order_tracker import OrderTracker


def test_root_order_persists_features_and_rationale(tmp_path: Path):
    """Root entries must persist features/rationale for forensic audits.

    This guards against silent NULLs that later show up as UNKNOWN-heavy buckets
    in tools/trade_outcomes_report.py feature audits.
    """

    db_path = tmp_path / "orders.db"
    tracker = OrderTracker(db_path=str(db_path))

    tracker.record_order_placement(
        order_id=123,
        symbol="MES",
        action="BUY",
        quantity=1,
        order_type="LIMIT",
        limit_price=100.0,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=101.0,
        parent_order_id=None,
        trade_cycle_id="tc_abc",
        features='{"rsi": 55.0, "close": 100.0}',
        rationale='{"strategy_name": "hybrid", "decision_confidence": 0.6}',
        market_regime="UPTREND",
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT features, rationale
            FROM orders
            WHERE order_id = ?
            """,
            (123,),
        ).fetchone()

    assert row is not None
    assert row["features"] is not None and str(row["features"]).strip() != ""
    assert row["rationale"] is not None and str(row["rationale"]).strip() != ""
