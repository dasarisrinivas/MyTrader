"""Backfill missing feature/rationale snapshots on root orders.

Why this exists
---------------
We observed many root orders in `data/orders.db` with a `trade_cycle_id` but empty/NULL
`features` and `rationale`. These rows break attribution in the trade outcome forensic
pipeline.

This tool performs a *conservative* backfill:
- Only targets *root* orders (`parent_order_id IS NULL`) with `trade_cycle_id` set
  and missing/empty `features` OR `rationale`.
- For each such root, it looks for a donor order within the same `trade_cycle_id`
  that has non-empty snapshots.
- It copies `features` and/or `rationale` only if the target field is missing.
- It annotates the rationale JSON with backfill metadata:
    - `__backfilled`: true
    - `__backfilled_from_order_id`: donor_order_id
    - `__backfilled_at`: ISO timestamp (UTC)

Safety
------
- Dry-run by default.
- Never overwrites non-empty snapshots.

Usage
-----
python3 tools/backfill_missing_order_snapshots.py --db data/orders.db --dry-run
python3 tools/backfill_missing_order_snapshots.py --db data/orders.db --apply

"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


def _is_missing_json(value: Optional[str]) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s == "{}"


def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {"_value": obj}
    except Exception:
        # Don't fail the whole repair because of one bad blob.
        return {"_raw": s}


def _merge_backfill_metadata(rationale_json: Optional[str], donor_order_id: int) -> str:
    base: Dict[str, Any] = {}
    if rationale_json and not _is_missing_json(rationale_json):
        base = _safe_json_loads(rationale_json)

    base["__backfilled"] = True
    base["__backfilled_from_order_id"] = donor_order_id
    base["__backfilled_at"] = datetime.now(timezone.utc).isoformat()

    return json.dumps(base, separators=(",", ":"), sort_keys=True)


@dataclass
class RepairCounts:
    cycles_scanned: int = 0
    targets: int = 0
    with_donor: int = 0
    updated: int = 0
    skipped_no_donor: int = 0
    skipped_already_complete: int = 0


def _get_donor_for_cycle(
    conn: sqlite3.Connection, trade_cycle_id: str, require_features: bool, require_rationale: bool
) -> Optional[Tuple[int, Optional[str], Optional[str]]]:
    """Pick a donor order_id/features/rationale for a cycle.

    Preference order:
    1) root orders with snapshots
    2) any order with snapshots

    We require at least the fields that are missing on target.
    """

    clauses = []
    if require_features:
        clauses.append("features IS NOT NULL AND TRIM(features) != '' AND TRIM(features) != '{}' ")
    if require_rationale:
        clauses.append("rationale IS NOT NULL AND TRIM(rationale) != '' AND TRIM(rationale) != '{}' ")

    where_need = " AND ".join(clauses) if clauses else "1=1"

    # Prefer root orders first.
    row = conn.execute(
        f"""
        SELECT order_id, features, rationale
        FROM orders
        WHERE trade_cycle_id = ?
          AND parent_order_id IS NULL
          AND {where_need}
        ORDER BY created_at ASC
        LIMIT 1
        """,
        (trade_cycle_id,),
    ).fetchone()

    if row:
        return int(row[0]), row[1], row[2]

    # Fallback: any order within cycle.
    row = conn.execute(
        f"""
        SELECT order_id, features, rationale
        FROM orders
        WHERE trade_cycle_id = ?
          AND {where_need}
        ORDER BY created_at ASC
        LIMIT 1
        """,
        (trade_cycle_id,),
    ).fetchone()

    if not row:
        return None

    return int(row[0]), row[1], row[2]


def repair_missing_snapshots(db_path: str, apply: bool, limit: Optional[int] = None) -> RepairCounts:
    counts = RepairCounts()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    targets_sql = """
    SELECT order_id, trade_cycle_id, features, rationale
    FROM orders
    WHERE parent_order_id IS NULL
      AND trade_cycle_id IS NOT NULL
      AND (features IS NULL OR TRIM(features) = '' OR TRIM(features) = '{}' 
           OR rationale IS NULL OR TRIM(rationale) = '' OR TRIM(rationale) = '{}' )
    ORDER BY created_at ASC
    """

    if limit is not None:
        targets_sql += " LIMIT ?"
        rows = conn.execute(targets_sql, (limit,)).fetchall()
    else:
        rows = conn.execute(targets_sql).fetchall()

    counts.targets = len(rows)

    for r in rows:
        order_id = int(r["order_id"])
        trade_cycle_id = str(r["trade_cycle_id"])
        features = r["features"]
        rationale = r["rationale"]

        missing_features = _is_missing_json(features)
        missing_rationale = _is_missing_json(rationale)

        if not (missing_features or missing_rationale):
            counts.skipped_already_complete += 1
            continue

        donor = _get_donor_for_cycle(
            conn,
            trade_cycle_id,
            require_features=missing_features,
            require_rationale=missing_rationale,
        )

        if donor is None:
            counts.skipped_no_donor += 1
            continue

        counts.with_donor += 1
        donor_order_id, donor_features, donor_rationale = donor

        new_features = features
        new_rationale = rationale

        if missing_features:
            new_features = donor_features

        if missing_rationale:
            # If donor rationale exists, copy it first; then annotate.
            new_rationale = donor_rationale

        # Always annotate rationale when we changed something.
        if (missing_features or missing_rationale) and donor_order_id is not None:
            new_rationale = _merge_backfill_metadata(new_rationale, donor_order_id)

        if apply:
            conn.execute(
                """
                UPDATE orders
                SET features = COALESCE(?, features),
                    rationale = COALESCE(?, rationale),
                    updated_at = ?
                WHERE order_id = ?
                """,
                (
                    new_features,
                    new_rationale,
                    datetime.now(timezone.utc).isoformat(),
                    order_id,
                ),
            )
            counts.updated += 1

    if apply:
        conn.commit()

    conn.close()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/orders.db")
    ap.add_argument("--apply", action="store_true", help="Apply updates (default is dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="Explicit dry-run (default)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of target root orders")
    args = ap.parse_args()

    apply = bool(args.apply)

    counts = repair_missing_snapshots(db_path=args.db, apply=apply, limit=args.limit)

    mode = "APPLY" if apply else "DRY_RUN"
    print(f"[{mode}] targets={counts.targets} with_donor={counts.with_donor} updated={counts.updated} skipped_no_donor={counts.skipped_no_donor}")


if __name__ == "__main__":
    main()
