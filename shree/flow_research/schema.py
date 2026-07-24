"""Flow-research database schema and access.

Its OWN sqlite file (default data/flow_research.db), never the production
spy_options_signals.db. Creating/opening this DB touches nothing the bot reads.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Iterable

from .models import Print, Snapshot


DDL_PRINTS = """
CREATE TABLE IF NOT EXISTS spy_flow_prints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc          TEXT    NOT NULL,
    ts_et           TEXT    NOT NULL,
    session_date    TEXT    NOT NULL,
    underlying_px   REAL,
    root            TEXT    NOT NULL,
    expiry          TEXT    NOT NULL,
    dte             INTEGER,
    strike          REAL    NOT NULL,
    right           TEXT    NOT NULL,
    trade_px        REAL    NOT NULL,
    size            INTEGER NOT NULL,
    premium         REAL    NOT NULL,
    exchange        TEXT,
    condition_codes TEXT,
    bid             REAL,
    ask             REAL,
    aggressor       TEXT,
    aggressor_src   TEXT,
    is_sweep        INTEGER DEFAULT 0,
    is_block        INTEGER DEFAULT 0,
    oc_estimate     TEXT,
    delta           REAL,
    gamma           REAL,
    iv              REAL,
    greeks_src      TEXT,
    data_source     TEXT,
    ingested_at     TEXT
);
"""

DDL_PRINTS_IX = [
    "CREATE INDEX IF NOT EXISTS ix_flow_session ON spy_flow_prints(session_date, ts_et);",
    "CREATE INDEX IF NOT EXISTS ix_flow_contract ON spy_flow_prints(session_date, expiry, strike, right);",
]

DDL_SHADOW = """
CREATE TABLE IF NOT EXISTS shadow_flow (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_kind        TEXT NOT NULL,
    signal_id            INTEGER,
    session_date         TEXT NOT NULL,
    ts_et                TEXT NOT NULL,
    window_s             INTEGER NOT NULL,
    net_call_prem        REAL,
    net_put_prem         REAL,
    pc_prem_imbalance    REAL,
    dw_flow              REAL,
    sweep_intensity      REAL,
    block_prem           REAL,
    oc_open_ratio        REAL,
    expiry_concentration REAL,
    strike_repetition    REAL,
    atm_vs_wing          REAL,
    iv_weighted_side     REAL,
    n_prints             INTEGER,
    n_prints_used        INTEGER,
    computed_at          TEXT
);
"""

DDL_SHADOW_IX = [
    "CREATE INDEX IF NOT EXISTS ix_shadow_sig ON shadow_flow(signal_id);",
    "CREATE INDEX IF NOT EXISTS ix_shadow_kind ON shadow_flow(snapshot_kind, session_date);",
]

_PRINT_COLS = [
    "ts_utc", "ts_et", "session_date", "underlying_px", "root", "expiry",
    "dte", "strike", "right", "trade_px", "size", "premium", "exchange",
    "condition_codes", "bid", "ask", "aggressor", "aggressor_src",
    "is_sweep", "is_block", "oc_estimate", "delta", "gamma", "iv",
    "greeks_src", "data_source",
]

_SHADOW_COLS = [
    "snapshot_kind", "signal_id", "session_date", "ts_et", "window_s",
    "net_call_prem", "net_put_prem", "pc_prem_imbalance", "dw_flow",
    "sweep_intensity", "block_prem", "oc_open_ratio", "expiry_concentration",
    "strike_repetition", "atm_vs_wing", "iv_weighted_side",
    "n_prints", "n_prints_used",
]


def open_db(path: str = "data/flow_research.db") -> sqlite3.Connection:
    """Open (creating if needed) the flow-research DB with both tables."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(DDL_PRINTS)
    for ix in DDL_PRINTS_IX:
        conn.execute(ix)
    conn.execute(DDL_SHADOW)
    for ix in DDL_SHADOW_IX:
        conn.execute(ix)
    conn.commit()
    return conn


def insert_prints(conn: sqlite3.Connection, prints: Iterable[Print]) -> int:
    """Bulk-insert raw prints. Returns count inserted."""
    rows = []
    for p in prints:
        r = p.to_row()
        rows.append(tuple(r.get(c) for c in _PRINT_COLS))
    placeholders = ",".join(["?"] * (len(_PRINT_COLS) + 1))  # +1 for ingested_at
    cols = ",".join(_PRINT_COLS + ["ingested_at"])
    conn.executemany(
        f"INSERT INTO spy_flow_prints ({cols}) VALUES ({placeholders})",
        [r + (_now_iso(),) for r in rows],
    )
    conn.commit()
    return len(rows)


def insert_snapshot(conn: sqlite3.Connection, snap: Snapshot) -> int:
    r = snap.to_row()
    placeholders = ",".join(["?"] * (len(_SHADOW_COLS) + 1))
    cols = ",".join(_SHADOW_COLS + ["computed_at"])
    vals = tuple(r.get(c) for c in _SHADOW_COLS) + (_now_iso(),)
    cur = conn.execute(
        f"INSERT INTO shadow_flow ({cols}) VALUES ({placeholders})", vals
    )
    conn.commit()
    return int(cur.lastrowid)


def _now_iso() -> str:
    # local import so the module is importable in restricted contexts
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
