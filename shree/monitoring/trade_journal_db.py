#!/usr/bin/env python3
"""
Trade Journal Database — SQLite schema and helper functions.

Stores daily analysis, blocked signals, near-misses, trades, and observations
for driving Phase 3+4 optimization decisions.

Database: data/trade_journal.db
"""

import os
import sqlite3

# Project root is 3 levels up from shree/monitoring/trade_journal_db.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(_PROJECT_ROOT, "data", "trade_journal.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Get a connection to the trade journal database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str = DB_PATH):
    """Create all tables if they don't exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            date            TEXT PRIMARY KEY,
            open_price      REAL,
            close_price     REAL,
            high_price      REAL,
            low_price       REAL,
            range_pts       REAL,
            vx_level        REAL,
            or_high         REAL,
            or_low          REAL,
            support_floor   REAL,
            total_trades    INTEGER DEFAULT 0,
            total_wins      INTEGER DEFAULT 0,
            total_losses    INTEGER DEFAULT 0,
            gross_pnl       REAL DEFAULT 0.0,
            net_pnl         REAL DEFAULT 0.0,
            signals_generated   INTEGER DEFAULT 0,
            chop_blocks         INTEGER DEFAULT 0,
            confidence_blocks   INTEGER DEFAULT 0,
            d_near_misses       INTEGER DEFAULT 0,
            d_near_misses_5pt   INTEGER DEFAULT 0,
            notes           TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            time            TEXT NOT NULL,
            signal_type     TEXT NOT NULL,
            direction       TEXT NOT NULL,
            confidence      REAL,
            adx             REAL,
            rsi             REAL,
            atr             REAL,
            macd_h          REAL,
            entry_price     REAL,
            stop_loss       REAL,
            take_profit     REAL,
            outcome         TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS blocked_signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            time            TEXT NOT NULL,
            signal_type     TEXT NOT NULL,
            direction       TEXT NOT NULL,
            block_reason    TEXT NOT NULL,
            confidence_at_block REAL,
            adx             REAL,
            rsi             REAL,
            atr             REAL,
            entry_price     REAL,
            stop_loss       REAL,
            take_profit     REAL,
            hypo_outcome    TEXT,
            hypo_pnl        REAL,
            hypo_notes      TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS near_misses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            time            TEXT NOT NULL,
            signal_type     TEXT NOT NULL,
            miss_reason     TEXT NOT NULL,
            miss_detail     TEXT,
            gap_pts         REAL,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            entry_time      TEXT NOT NULL,
            exit_time       TEXT,
            signal_type     TEXT NOT NULL,
            direction       TEXT NOT NULL,
            entry_price     REAL NOT NULL,
            exit_price      REAL,
            stop_loss       REAL,
            take_profit     REAL,
            outcome         TEXT,
            gross_pnl       REAL,
            net_pnl         REAL,
            commission      REAL,
            confidence      REAL,
            adx             REAL,
            vx_level        REAL,
            hold_duration_m INTEGER,
            notes           TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            category        TEXT NOT NULL,
            severity        TEXT DEFAULT 'INFO',
            title           TEXT NOT NULL,
            detail          TEXT,
            recommended_fix TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gate_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            gate_date       TEXT NOT NULL,
            metric_name     TEXT NOT NULL,
            metric_value    REAL NOT NULL,
            threshold       REAL,
            gate_met        INTEGER,
            computed_from   TEXT,
            notes           TEXT,
            updated_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_blocked_date ON blocked_signals(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_near_misses_date ON near_misses(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_observations_date ON observations(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_observations_category ON observations(category)")

    # ── Gold futures trades (separate table — no MES assumptions) ─────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id        TEXT NOT NULL,
            date            TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            action          TEXT NOT NULL,
            signal_type     TEXT NOT NULL,
            contracts       INTEGER NOT NULL,
            entry_price     REAL NOT NULL,
            stop_loss       REAL NOT NULL,
            take_profit     REAL NOT NULL,
            exit_price      REAL NOT NULL,
            realized_pnl    REAL NOT NULL,
            commission      REAL NOT NULL DEFAULT 0.0,
            net_pnl         REAL NOT NULL,
            entry_time      TEXT NOT NULL,
            exit_time       TEXT NOT NULL,
            hold_bars       INTEGER,
            exit_reason     TEXT NOT NULL,
            regime          TEXT,
            atr_at_entry    REAL,
            adx_at_entry    REAL,
            win             INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_gold_trades_id ON gold_trades(trade_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_gold_trades_date ON gold_trades(date)"
    )

    # ── Gold daily summary (separate from MES daily_summary) ─────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold_daily_summary (
            date            TEXT PRIMARY KEY,
            symbol          TEXT NOT NULL DEFAULT 'MGC',
            total_trades    INTEGER DEFAULT 0,
            wins            INTEGER DEFAULT 0,
            losses          INTEGER DEFAULT 0,
            win_rate        REAL DEFAULT 0.0,
            gross_pnl       REAL DEFAULT 0.0,
            net_pnl         REAL DEFAULT 0.0,
            profit_factor   REAL,
            avg_win         REAL,
            avg_loss        REAL,
            avg_hold_bars   REAL,
            signal_vwap_pb  INTEGER DEFAULT 0,
            signal_ema_pb   INTEGER DEFAULT 0,
            signal_orb      INTEGER DEFAULT 0,
            exit_tp         INTEGER DEFAULT 0,
            exit_sl         INTEGER DEFAULT 0,
            exit_time       INTEGER DEFAULT 0,
            exit_flatten    INTEGER DEFAULT 0,
            notes           TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()
    return db_path


def insert_daily_summary(conn, data: dict):
    """Insert or update a daily summary row."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    updates = ", ".join([f"{k}=excluded.{k}" for k in data.keys() if k != "date"])
    sql = f"""
        INSERT INTO daily_summary ({cols}) VALUES ({placeholders})
        ON CONFLICT(date) DO UPDATE SET {updates}
    """
    conn.execute(sql, list(data.values()))


def insert_signal(conn, data: dict):
    """Insert a signal record."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    conn.execute(f"INSERT INTO signals ({cols}) VALUES ({placeholders})", list(data.values()))


def insert_blocked_signal(conn, data: dict):
    """Insert a blocked signal record."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    conn.execute(f"INSERT INTO blocked_signals ({cols}) VALUES ({placeholders})", list(data.values()))


def insert_near_miss(conn, data: dict):
    """Insert a near-miss record."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    conn.execute(f"INSERT INTO near_misses ({cols}) VALUES ({placeholders})", list(data.values()))


def insert_trade(conn, data: dict):
    """Insert a trade record."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    conn.execute(f"INSERT INTO trades ({cols}) VALUES ({placeholders})", list(data.values()))


def insert_observation(conn, data: dict):
    """Insert an observation."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    conn.execute(f"INSERT INTO observations ({cols}) VALUES ({placeholders})", list(data.values()))


def insert_gate_metric(conn, data: dict):
    """Insert or update a gate metric."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    conn.execute(f"INSERT INTO gate_metrics ({cols}) VALUES ({placeholders})", list(data.values()))


def get_weekly_summary(conn, start_date: str, end_date: str) -> dict:
    """Get aggregated metrics for a date range."""
    row = conn.execute("""
        SELECT
            COUNT(*) as trading_days,
            SUM(total_trades) as total_trades,
            SUM(total_wins) as total_wins,
            SUM(total_losses) as total_losses,
            SUM(net_pnl) as total_pnl,
            SUM(signals_generated) as total_signals,
            SUM(chop_blocks) as total_chop_blocks,
            SUM(d_near_misses) as total_d_near_misses,
            ROUND(CAST(SUM(total_trades) AS REAL) / COUNT(*), 2) as trades_per_day
        FROM daily_summary
        WHERE date BETWEEN ? AND ?
    """, (start_date, end_date)).fetchone()
    return dict(row) if row else {}


def get_blocked_by_reason(conn, start_date: str, end_date: str) -> list:
    """Get blocked signal counts by reason."""
    rows = conn.execute("""
        SELECT block_reason, COUNT(*) as count,
               SUM(CASE WHEN hypo_outcome = 'TP_HIT' THEN 1 ELSE 0 END) as would_have_won,
               SUM(CASE WHEN hypo_outcome = 'SL_HIT' THEN 1 ELSE 0 END) as would_have_lost
        FROM blocked_signals
        WHERE date BETWEEN ? AND ?
        GROUP BY block_reason
        ORDER BY count DESC
    """, (start_date, end_date)).fetchall()
    return [dict(r) for r in rows]


def get_observations_by_category(conn, category: str = None, severity: str = None) -> list:
    """Get observations, optionally filtered."""
    sql = "SELECT * FROM observations WHERE 1=1"
    params = []
    if category:
        sql += " AND category = ?"
        params.append(category)
    if severity:
        sql += " AND severity = ?"
        params.append(severity)
    sql += " ORDER BY date DESC, id DESC"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_near_miss_summary(conn, start_date: str, end_date: str) -> dict:
    """Get near-miss stats for decision gate evaluation."""
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN gap_pts <= 1.0 THEN 1 ELSE 0 END) as within_1pt,
            SUM(CASE WHEN gap_pts <= 3.0 THEN 1 ELSE 0 END) as within_3pt,
            SUM(CASE WHEN gap_pts <= 5.0 THEN 1 ELSE 0 END) as within_5pt,
            MIN(gap_pts) as closest_miss,
            AVG(gap_pts) as avg_gap
        FROM near_misses
        WHERE date BETWEEN ? AND ?
    """, (start_date, end_date)).fetchone()
    return dict(row) if row else {}


# ── Gold-specific helpers ─────────────────────────────────────────────────────

def upsert_gold_trade(conn, data: dict) -> None:
    """Insert a gold trade record, skipping duplicates on trade_id."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    updates = ", ".join(
        [f"{k}=excluded.{k}" for k in data.keys() if k != "trade_id"]
    )
    sql = f"""
        INSERT INTO gold_trades ({cols}) VALUES ({placeholders})
        ON CONFLICT(trade_id) DO UPDATE SET {updates}
    """
    conn.execute(sql, list(data.values()))


def upsert_gold_daily_summary(conn, data: dict) -> None:
    """Insert or replace a gold daily summary row."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    updates = ", ".join(
        [f"{k}=excluded.{k}" for k in data.keys() if k != "date"]
    )
    sql = f"""
        INSERT INTO gold_daily_summary ({cols}) VALUES ({placeholders})
        ON CONFLICT(date) DO UPDATE SET {updates}
    """
    conn.execute(sql, list(data.values()))


def get_gold_trades(conn, start_date: str, end_date: str) -> list:
    """Return all gold trades in the date range, newest first."""
    rows = conn.execute(
        """
        SELECT * FROM gold_trades
        WHERE date BETWEEN ? AND ?
        ORDER BY date DESC, entry_time DESC
        """,
        (start_date, end_date),
    ).fetchall()
    return [dict(r) for r in rows]


def get_gold_summary(conn, start_date: str, end_date: str) -> dict:
    """Aggregate gold performance across a date range."""
    row = conn.execute(
        """
        SELECT
            COUNT(*)                                            AS total_trades,
            SUM(win)                                            AS wins,
            SUM(1 - win)                                        AS losses,
            ROUND(100.0 * AVG(win), 1)                          AS win_rate_pct,
            ROUND(SUM(net_pnl), 2)                              AS net_pnl,
            ROUND(SUM(CASE WHEN net_pnl > 0 THEN net_pnl ELSE 0 END), 2) AS gross_win,
            ROUND(SUM(CASE WHEN net_pnl < 0 THEN net_pnl ELSE 0 END), 2) AS gross_loss,
            ROUND(AVG(CASE WHEN win=1 THEN net_pnl END), 2)    AS avg_win,
            ROUND(AVG(CASE WHEN win=0 THEN net_pnl END), 2)    AS avg_loss,
            ROUND(AVG(hold_bars), 1)                            AS avg_hold_bars,
            COUNT(DISTINCT date)                                AS trading_days
        FROM gold_trades
        WHERE date BETWEEN ? AND ?
        """,
        (start_date, end_date),
    ).fetchone()
    if not row or row["total_trades"] == 0:
        return {}
    result = dict(row)
    gross_win = result.get("gross_win") or 0.0
    gross_loss = abs(result.get("gross_loss") or 0.0)
    result["profit_factor"] = round(gross_win / gross_loss, 2) if gross_loss > 0 else None
    return result


def get_gold_signal_breakdown(conn, start_date: str, end_date: str) -> list:
    """Win rate and P&L grouped by signal type."""
    rows = conn.execute(
        """
        SELECT
            signal_type,
            COUNT(*)                        AS total,
            SUM(win)                        AS wins,
            ROUND(100.0 * AVG(win), 1)      AS win_rate_pct,
            ROUND(SUM(net_pnl), 2)          AS net_pnl,
            ROUND(AVG(net_pnl), 2)          AS avg_pnl
        FROM gold_trades
        WHERE date BETWEEN ? AND ?
        GROUP BY signal_type
        ORDER BY net_pnl DESC
        """,
        (start_date, end_date),
    ).fetchall()
    return [dict(r) for r in rows]


def get_gold_exit_breakdown(conn, start_date: str, end_date: str) -> list:
    """Count of each exit reason."""
    rows = conn.execute(
        """
        SELECT
            exit_reason,
            COUNT(*)                        AS total,
            ROUND(SUM(net_pnl), 2)          AS net_pnl,
            ROUND(AVG(net_pnl), 2)          AS avg_pnl
        FROM gold_trades
        WHERE date BETWEEN ? AND ?
        GROUP BY exit_reason
        ORDER BY total DESC
        """,
        (start_date, end_date),
    ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    path = init_db()
    print(f"✅ Trade journal database initialized: {path}")
