"""Read realized PnL from the bot's orders.db (executions table).

This is the *truth* source for daily PnL — much more reliable than the
trade_outcomes table which has reconciliation gaps and corrupted rows.
Each fill in `executions` carries net_pnl; we sum closing fills for the
session.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple


@dataclass
class TradeOutcome:
    order_id: int
    timestamp: str   # ISO
    net_pnl: float
    gross_pnl: float
    commission: float

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def is_loss(self) -> bool:
        return self.net_pnl < 0


# Heuristic: a fill is a "closing" fill if its net_pnl is non-zero.
# Opening fills always show 0 PnL in the bot's executions table.
def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=5.0)
    con.row_factory = sqlite3.Row
    return con


def realized_pnl_for_session(db_path: str, session_date: str) -> Tuple[float, int]:
    """Return (sum_net_pnl, n_closed_trades) for the given YYYY-MM-DD.

    Uses CT day boundaries via the timestamp string prefix (the bot writes
    timestamps in UTC ISO; for safety we filter on a 24h window around the
    CT session date instead of naive prefix matching).
    """
    # Build UTC bounds: CT session ~ [date 05:00 UTC, date+1 05:00 UTC]
    # (CT = UTC-5 / UTC-6 depending on DST). We use a generous 30-hour
    # window centered on the CT day to absorb any DST/timezone drift —
    # this is still tight enough that prior/next sessions don't bleed in
    # because trading sessions break at 16:00 CT for maintenance.
    try:
        d = datetime.strptime(session_date, "%Y-%m-%d")
    except ValueError:
        return 0.0, 0

    # Approx UTC window: 04:00 UTC of session_date → 06:00 UTC next day.
    # Wide enough for either CST or CDT, narrow enough not to grab adjacent days.
    start_utc = f"{session_date}T04:00:00"
    end_utc = (d.replace(hour=23) ).strftime("%Y-%m-%dT") + "23:59:59"
    # Just take 30h window simpler:
    from datetime import timedelta
    end_dt = d + timedelta(hours=30)
    end_utc = end_dt.strftime("%Y-%m-%dT%H:%M:%S")

    con = _connect(db_path)
    try:
        cur = con.cursor()
        rows = cur.execute(
            """SELECT timestamp, net_pnl FROM executions
               WHERE timestamp >= ? AND timestamp < ?
                 AND net_pnl IS NOT NULL AND net_pnl != 0
                 AND ABS(net_pnl) < 5000  -- exclude corrupted IBKR cumulative rows
               ORDER BY timestamp""",
            (start_utc, end_utc),
        ).fetchall()
    finally:
        con.close()
    total = sum(r["net_pnl"] for r in rows)
    return float(total), len(rows)


def last_n_closed_trades(db_path: str, n: int = 20) -> List[TradeOutcome]:
    """Return the last N completed trades (non-zero net_pnl) ordered newest first."""
    con = _connect(db_path)
    try:
        cur = con.cursor()
        rows = cur.execute(
            """SELECT order_id, timestamp, net_pnl, gross_pnl, commission
               FROM executions
               WHERE net_pnl IS NOT NULL AND net_pnl != 0
                 AND ABS(net_pnl) < 5000
               ORDER BY timestamp DESC LIMIT ?""",
            (n,),
        ).fetchall()
    finally:
        con.close()
    return [
        TradeOutcome(
            order_id=int(r["order_id"]),
            timestamp=r["timestamp"],
            net_pnl=float(r["net_pnl"]),
            gross_pnl=float(r["gross_pnl"] or 0.0),
            commission=float(r["commission"] or 0.0),
        )
        for r in rows
    ]


def open_position_count(db_path: str) -> int:
    """Count entries in trade_outcomes with NULL exit_time as a proxy for open positions.

    NOTE: this is not authoritative — IBKR is the truth source. But it's
    a useful "is there work in flight" signal.
    """
    con = _connect(db_path)
    try:
        cur = con.cursor()
        n = cur.execute(
            "SELECT COUNT(*) FROM trade_outcomes WHERE exit_time IS NULL"
        ).fetchone()[0]
    finally:
        con.close()
    return int(n)


def streaks_from_recent(
    trades: List[TradeOutcome],
    *,
    since_iso: Optional[str] = None,
) -> Tuple[int, int]:
    """Compute current consecutive (wins, losses) from most-recent trades.

    Only one of the two is non-zero — whichever the latest trade is on.

    MAY 8 2026 fix: optionally bound the streak window to trades AFTER
    `since_iso`. Without this bound, a 6-loss streak from 9 days ago
    permanently locks the system out — the streak never breaks because
    no new trades can fire (catch-22). The manager passes the start of
    the current CT trading session here so streaks are session-scoped.
    Pre-session trades are *informational* (still in the trade log) but
    don't gate today's risk posture.
    """
    if not trades:
        return 0, 0

    # Filter by since_iso if provided
    if since_iso:
        trades = [t for t in trades if t.timestamp and t.timestamp >= since_iso]
        if not trades:
            return 0, 0

    # newest first
    first = trades[0]
    if first.is_win:
        wins = 0
        for t in trades:
            if t.is_win:
                wins += 1
            else:
                break
        return wins, 0
    elif first.is_loss:
        losses = 0
        for t in trades:
            if t.is_loss:
                losses += 1
            else:
                break
        return 0, losses
    return 0, 0
