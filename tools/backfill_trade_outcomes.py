"""Backfill deterministic trade closures into orders.db.

This script reads existing order rows (and rolled-up parent PnL) and writes a
canonical record into orders.db:trade_outcomes keyed by trade_cycle_id.

It is safe to run multiple times: trade_outcomes is upserted.

Usage (optional):
    python3 tools/backfill_trade_outcomes.py --days 30
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import sqlite3
import json

from mytrader.monitoring.order_tracker import OrderTracker


def _sum_executions_for_order_ids(conn: sqlite3.Connection, order_ids: list[int]) -> dict:
    if not order_ids:
        return {"realized_pnl": 0.0, "gross_pnl": 0.0, "net_pnl": 0.0, "commission": 0.0}

    placeholders = ",".join(["?"] * len(order_ids))
    row = conn.execute(
        f"""
        SELECT
            COALESCE(SUM(COALESCE(realized_pnl, 0)), 0) AS realized_pnl,
            COALESCE(SUM(COALESCE(gross_pnl, 0)), 0) AS gross_pnl,
            COALESCE(SUM(COALESCE(net_pnl, 0)), 0) AS net_pnl,
            COALESCE(SUM(COALESCE(commission, 0)), 0) AS commission
        FROM executions
        WHERE order_id IN ({placeholders})
        """,
        tuple(order_ids),
    ).fetchone()
    return {
        "realized_pnl": float(row["realized_pnl"] or 0.0),
        "gross_pnl": float(row["gross_pnl"] or 0.0),
        "net_pnl": float(row["net_pnl"] or 0.0),
        "commission": float(row["commission"] or 0.0),
    }


def _get_bracket_tree_order_ids(conn: sqlite3.Connection, root_order_id: int) -> list[int]:
    """Return root + all descendants via parent_order_id links."""
    order_ids: list[int] = []
    frontier: list[int] = [root_order_id]

    while frontier:
        current = frontier.pop()
        if current in order_ids:
            continue
        order_ids.append(current)
        child_rows = conn.execute(
            "SELECT order_id FROM orders WHERE parent_order_id = ?",
            (current,),
        ).fetchall()
        for r in child_rows:
            frontier.append(int(r["order_id"]))

    return order_ids


def _latest_execution_exit_snapshot(
    conn: sqlite3.Connection, order_ids: list[int]
) -> tuple[str | None, float | None]:
    if not order_ids:
        return None, None
    placeholders = ",".join(["?"] * len(order_ids))
    row = conn.execute(
        f"""
        SELECT timestamp, price
        FROM executions
        WHERE order_id IN ({placeholders})
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        tuple(order_ids),
    ).fetchone()
    if not row:
        return None, None
    return row["timestamp"], float(row["price"] or 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    tracker = OrderTracker()

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    cutoff_iso = cutoff.replace(microsecond=0).isoformat()

    # Pull candidate roots (parent_order_id is NULL) with a trade_cycle_id.
    with sqlite3.connect(tracker.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT order_id, trade_cycle_id, symbol, timestamp, avg_fill_price, quantity
            FROM orders
            WHERE parent_order_id IS NULL
              AND trade_cycle_id IS NOT NULL
              AND timestamp >= ?
            ORDER BY timestamp DESC
            """,
            (cutoff_iso,),
        ).fetchall()

    updated = 0
    with sqlite3.connect(tracker.db_path) as conn:
        conn.row_factory = sqlite3.Row

        for row in rows:
            tcid = row["trade_cycle_id"]
            if not tcid:
                continue

            root_order_id = int(row["order_id"])
            order_ids = _get_bracket_tree_order_ids(conn, root_order_id)
            pnl = _sum_executions_for_order_ids(conn, order_ids)
            exit_time, exit_price = _latest_execution_exit_snapshot(conn, order_ids)

            now = datetime.now(timezone.utc).isoformat()
            payload = {
                "root_order_id": root_order_id,
                "order_ids": order_ids,
                "source": "backfill",
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
                    entry_time = COALESCE(excluded.entry_time, trade_outcomes.entry_time),
                    exit_time = COALESCE(excluded.exit_time, trade_outcomes.exit_time),
                    entry_price = COALESCE(excluded.entry_price, trade_outcomes.entry_price),
                    exit_price = COALESCE(excluded.exit_price, trade_outcomes.exit_price),
                    quantity = COALESCE(excluded.quantity, trade_outcomes.quantity),
                    exit_reason = COALESCE(excluded.exit_reason, trade_outcomes.exit_reason),
                    realized_pnl = excluded.realized_pnl,
                    gross_pnl = excluded.gross_pnl,
                    net_pnl = excluded.net_pnl,
                    commission = excluded.commission,
                    updated_at = excluded.updated_at
                """,
                (
                    str(tcid),
                    root_order_id,
                    str(row["symbol"]),
                    str(row["timestamp"]),
                    exit_time,
                    float(row["avg_fill_price"] or 0.0),
                    exit_price,
                    int(row["quantity"] or 0),
                    "BACKFILL",
                    pnl["realized_pnl"],
                    pnl["gross_pnl"],
                    pnl["net_pnl"],
                    pnl["commission"],
                    json.dumps(payload),
                    now,
                    now,
                ),
            )
            updated += 1

        conn.commit()

    print(f"Backfilled/updated trade_outcomes rows: {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
