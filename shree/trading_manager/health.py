"""Layer 1.5: Health Score — bad-day vs bad-logic detector.

The session-scoped streak counter (Layer 1) only sees within today. That's
right for daily decisions but blind to multi-day strategy decay: if the bot
loses 3 trades every single day for 10 days, each new session starts fresh
at 0 losses and the TM never escalates beyond DEFENSIVE.

This module looks at the *forest*, not the trees. Four rolling indicators:

  1. Weekly cumulative PnL       — capital preservation breach
  2. Rolling 30-trade win rate   — strategy decay
  3. Avg-winner / avg-loser      — stop/target sizing breakdown
  4. Multi-day red streak        — regime mismatch

Each metric is binary (triggered or not). The aggregate determines the
TM's strategic posture:

   0 triggers  → HEALTHY   (no override)
   1 trigger   → DEGRADED  (escalate to DEFENSIVE for the day)
   2 triggers  → SUSPECT   (escalate to SIT_OUT for the day)
   3+ triggers → LOCKED    (halt indefinitely; manual unlock required)

LOCKED only clears when:
  (a) The user writes an unlock marker file via the unlock CLI, AND
  (b) A subsequent PROBATION trade (1 small probe) wins.

If the probation trade loses, system reverts to LOCKED and requires
another manual unlock. This prevents an immediate rip-cord-pull retry.

All metrics are computed from data already in orders.db. No new feeds.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


CT = ZoneInfo("America/Chicago")


# ─────────────────────────────────────────────────────────────────────────────
# Metric thresholds — defaults; can be overridden via env in config.py
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "weekly_dd_pct":          5.0,    # weekly cumulative PnL drawdown limit
    "rolling_n":              30,     # rolling-WR window size
    "rolling_wr_floor":       30.0,   # if WR over rolling window drops below this %
    "winloser_n":             20,     # avg W/L ratio window size
    "winloser_floor":         0.7,    # if avg_winner/avg_loser < this, sizing is broken
    "multi_day_red_count":    3,      # consecutive losing sessions to trigger
    "multi_day_red_dd_pct":   3.0,    # AND total drawdown > this % over those sessions
}


@dataclass
class HealthMetrics:
    """Snapshot of all four health indicators."""
    weekly_pnl: float
    weekly_pnl_pct: float           # vs equity
    rolling_wr: float               # 0-100
    rolling_wr_n: int               # how many trades in window
    winloser_ratio: float           # avg(W$) / avg(L$)
    winloser_n: int
    multi_day_red_count: int        # consecutive red sessions
    multi_day_red_pnl: float        # total $ drawdown over those sessions

    # Per-trigger flags
    weekly_dd_triggered: bool = False
    rolling_wr_triggered: bool = False
    winloser_triggered: bool = False
    multi_day_red_triggered: bool = False

    @property
    def trigger_count(self) -> int:
        return sum([
            self.weekly_dd_triggered,
            self.rolling_wr_triggered,
            self.winloser_triggered,
            self.multi_day_red_triggered,
        ])

    @property
    def triggers(self) -> List[str]:
        out = []
        if self.weekly_dd_triggered:        out.append("weekly_dd")
        if self.rolling_wr_triggered:       out.append("rolling_wr")
        if self.winloser_triggered:         out.append("winloser_ratio")
        if self.multi_day_red_triggered:    out.append("multi_day_red")
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Status states (lifecycle)
# ─────────────────────────────────────────────────────────────────────────────

HEALTH_HEALTHY   = "HEALTHY"
HEALTH_DEGRADED  = "DEGRADED"     # 1 trigger
HEALTH_SUSPECT   = "SUSPECT"      # 2 triggers
HEALTH_LOCKED    = "LOCKED"       # 3+ triggers; manual unlock required
HEALTH_PROBATION = "PROBATION"    # post-unlock; 1 winning trade clears


def status_from_triggers(n: int) -> str:
    if n >= 3:
        return HEALTH_LOCKED
    if n == 2:
        return HEALTH_SUSPECT
    if n == 1:
        return HEALTH_DEGRADED
    return HEALTH_HEALTHY


# ─────────────────────────────────────────────────────────────────────────────
# DB-backed metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=5.0)
    con.row_factory = sqlite3.Row
    return con


def _ct_date_of(ts_iso: str) -> Optional[str]:
    """Convert a UTC-naive ISO timestamp to its CT calendar date (YYYY-MM-DD).
    The executions.timestamp column is stored as UTC ISO without offset."""
    if not ts_iso:
        return None
    try:
        # Parse as UTC (executions.timestamp is naive UTC ISO)
        if "+" in ts_iso or ts_iso.endswith("Z"):
            dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(ts_iso).replace(tzinfo=timezone.utc)
        return dt.astimezone(CT).strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return None


def compute_metrics(
    orders_db: str,
    *,
    account_equity: float,
    rolling_n: int = DEFAULTS["rolling_n"],
    winloser_n: int = DEFAULTS["winloser_n"],
    weekly_dd_pct: float = DEFAULTS["weekly_dd_pct"],
    rolling_wr_floor: float = DEFAULTS["rolling_wr_floor"],
    winloser_floor: float = DEFAULTS["winloser_floor"],
    multi_day_red_count: int = DEFAULTS["multi_day_red_count"],
    multi_day_red_dd_pct: float = DEFAULTS["multi_day_red_dd_pct"],
    since_iso: str | None = None,
) -> HealthMetrics:
    """Compute all four health metrics from orders.db.executions.

    Reads only closed trades (net_pnl != 0 AND |net_pnl| < 5000 — drops the
    known corrupted CORRUPTED_PNL_CUMULATIVE_IBKR rows from Feb 2026).

    2026-05-28: `since_iso` scopes the window to trades AFTER a calibration date,
    preventing the cold-start deadlock where pre-config-change / paper / old-book
    trades stay in the rolling window indefinitely (the bot hasn't traded since
    May 7 → rolling_wr was frozen at 20%, multi_day_red permanently triggered →
    permanent SUSPECT → MODIFY-at-small-size → no recovery). Default None = no
    cutoff, original behavior preserved.
    """
    con = _connect(orders_db)
    try:
        if since_iso:
            rows = con.execute(
                """SELECT order_id, timestamp, net_pnl
                   FROM executions
                   WHERE net_pnl IS NOT NULL AND net_pnl != 0
                     AND ABS(net_pnl) < 5000
                     AND timestamp >= ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (since_iso, max(rolling_n, winloser_n, 100)),
            ).fetchall()
        else:
            rows = con.execute(
                """SELECT order_id, timestamp, net_pnl
                   FROM executions
                   WHERE net_pnl IS NOT NULL AND net_pnl != 0
                     AND ABS(net_pnl) < 5000
                   ORDER BY timestamp DESC LIMIT ?""",
                (max(rolling_n, winloser_n, 100),),
            ).fetchall()
    finally:
        con.close()

    closed = [(r["timestamp"], float(r["net_pnl"])) for r in rows]

    # === Metric 1: Weekly cumulative PnL ====================================
    # Sum of trades in the last 7 calendar days.
    now_utc = datetime.now(timezone.utc)
    week_ago_utc = (now_utc - timedelta(days=7)).isoformat(timespec="seconds")
    week_ago_naive = week_ago_utc.replace("+00:00", "")
    weekly_pnl = sum(p for ts, p in closed if ts >= week_ago_naive)
    weekly_pnl_pct = -(weekly_pnl / account_equity) * 100.0 if account_equity else 0.0
    weekly_dd_triggered = weekly_pnl_pct >= weekly_dd_pct

    # === Metric 2: Rolling N-trade win rate =================================
    rolling_window = closed[:rolling_n]
    n_rolling = len(rolling_window)
    n_wins = sum(1 for _, p in rolling_window if p > 0)
    rolling_wr = (n_wins / n_rolling * 100.0) if n_rolling else 0.0
    # Only trigger if we have a meaningful sample (at least half the window)
    rolling_wr_triggered = (
        n_rolling >= max(10, rolling_n // 2)
        and rolling_wr < rolling_wr_floor
    )

    # === Metric 3: Avg-winner / avg-loser ratio =============================
    wl_window = closed[:winloser_n]
    n_wl = len(wl_window)
    wins = [p for _, p in wl_window if p > 0]
    losses = [-p for _, p in wl_window if p < 0]   # absolute values
    avg_w = sum(wins) / len(wins) if wins else 0.0
    avg_l = sum(losses) / len(losses) if losses else 0.0
    winloser_ratio = (avg_w / avg_l) if avg_l > 0 else (999.0 if avg_w > 0 else 0.0)
    # Only trigger if we have meaningful sample AND both sides exist
    winloser_triggered = (
        n_wl >= max(10, winloser_n // 2)
        and len(wins) >= 2
        and len(losses) >= 2
        and winloser_ratio < winloser_floor
    )

    # === Metric 4: Multi-day red streak =====================================
    # Group recent trades by CT calendar date; walk newest-first counting
    # consecutive days with negative net_pnl until we hit a green day.
    # Trigger if (count >= threshold) AND (total drawdown > threshold% of equity).
    by_day: Dict[str, float] = {}
    for ts, p in closed:
        d = _ct_date_of(ts)
        if d:
            by_day[d] = by_day.get(d, 0.0) + p
    days_sorted = sorted(by_day.keys(), reverse=True)  # newest first
    red_streak = 0
    red_dd = 0.0
    for d in days_sorted:
        if by_day[d] < 0:
            red_streak += 1
            red_dd += by_day[d]   # negative value
        else:
            break
    red_dd_pct = -(red_dd / account_equity) * 100.0 if account_equity else 0.0
    multi_day_red_triggered = (
        red_streak >= multi_day_red_count
        and red_dd_pct >= multi_day_red_dd_pct
    )

    return HealthMetrics(
        weekly_pnl=round(weekly_pnl, 2),
        weekly_pnl_pct=round(weekly_pnl_pct, 2),
        rolling_wr=round(rolling_wr, 1),
        rolling_wr_n=n_rolling,
        winloser_ratio=round(winloser_ratio, 2),
        winloser_n=n_wl,
        multi_day_red_count=red_streak,
        multi_day_red_pnl=round(red_dd, 2),
        weekly_dd_triggered=weekly_dd_triggered,
        rolling_wr_triggered=rolling_wr_triggered,
        winloser_triggered=winloser_triggered,
        multi_day_red_triggered=multi_day_red_triggered,
    )


def render_summary(m: HealthMetrics) -> str:
    """One-line human-readable summary for the heartbeat / decision log."""
    return (
        f"weekly={m.weekly_pnl:+.2f} ({m.weekly_pnl_pct:.1f}% dd"
        f"{'!' if m.weekly_dd_triggered else ''}) | "
        f"WR={m.rolling_wr:.0f}% over {m.rolling_wr_n} trades"
        f"{'!' if m.rolling_wr_triggered else ''} | "
        f"W/L={m.winloser_ratio:.2f}{'!' if m.winloser_triggered else ''} | "
        f"red_days={m.multi_day_red_count} ({m.multi_day_red_pnl:+.2f})"
        f"{'!' if m.multi_day_red_triggered else ''}"
    )
