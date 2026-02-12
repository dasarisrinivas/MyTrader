import os
from pathlib import Path

from shree.monitoring.order_tracker import OrderTracker


def _trigger_missing_features(tracker: OrderTracker) -> None:
    # Intentionally omit features/rationale to hit the warning path.
    tracker.record_order_placement(
        order_id=999,
        symbol="MES",
        action="BUY",
        quantity=1,
        order_type="LIMIT",
        limit_price=100.0,
        entry_price=100.0,
        trade_cycle_id="tc_missing",
        parent_order_id=None,
        features=None,
        rationale=None,
    )


def test_callsite_suffix_present_when_enabled(tmp_path: Path):
    os.environ["SHREE_ORDER_TRACKER_CALLSITE"] = "1"

    db_path = tmp_path / "orders.db"
    tracker = OrderTracker(db_path=str(db_path))

    # Call the helper from a known frame to assert it names this function.
    suffix = tracker._format_order_placement_callsite()
    assert suffix
    assert "caller=" in suffix
    assert "test_callsite_suffix_present_when_enabled" in suffix


def test_callsite_suffix_empty_by_default(tmp_path: Path):
    os.environ.pop("SHREE_ORDER_TRACKER_CALLSITE", None)

    db_path = tmp_path / "orders.db"
    tracker = OrderTracker(db_path=str(db_path))

    suffix = tracker._format_order_placement_callsite()
    assert suffix == ""
