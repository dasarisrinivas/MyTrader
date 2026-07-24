"""Attach flow snapshots to context — signals, rejected signals, intervals.

READ-ONLY against production: opens spy_options_signals.db in immutable mode to
retro-attach flow to the existing 657 dated signals, and reads
logs/blocked_signals.jsonl for the rejected control group. Writes ONLY to the
flow-research DB (shadow_flow).

Nothing here can influence a live order. It reads outcomes that already exist.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo

from .models import Print
from .features import compute_features
from .classify import classify_all, mark_blocks, mark_sweeps
from .schema import insert_snapshot


DEFAULT_WINDOW_S = 1800  # 30-minute trailing window for a snapshot
_ET = ZoneInfo("America/New_York")  # spy_signals.sent_at is UTC -> convert to ET


# ─────────────────────────────────────────────────────────────────────────────
# Print store — queries raw prints from the flow-research DB
# ─────────────────────────────────────────────────────────────────────────────

class PrintStore:
    """Reads raw prints from spy_flow_prints, materialised as Print objects,
    classified on the way out (aggressor + blocks + sweeps per session)."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def window(self, session_date: str, end_et: str, window_s: int) -> List[Print]:
        start_et = _iso_shift(end_et, -window_s)
        rows = self.conn.execute(
            """SELECT * FROM spy_flow_prints
               WHERE session_date = ? AND ts_et > ? AND ts_et <= ?
               ORDER BY ts_utc ASC""",
            (session_date, start_et, end_et),
        ).fetchall()
        prints = [_row_to_print(r) for r in rows]
        # classify within the window context
        classify_all(prints)
        mark_blocks(prints)
        mark_sweeps(prints)
        return prints

    def sessions(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT session_date FROM spy_flow_prints ORDER BY 1"
        ).fetchall()
        return [r[0] for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot drivers
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_signals(
    store: PrintStore,
    out_conn: sqlite3.Connection,
    spy_db_path: str,
    *,
    window_s: int = DEFAULT_WINDOW_S,
) -> int:
    """Retro-attach a flow snapshot to every dispatched signal in spy_signals.

    spy_db is opened read-only/immutable — production is never modified.
    """
    written = 0
    src = _open_readonly(spy_db_path)
    try:
        rows = src.execute(
            "SELECT id, sent_at FROM spy_signals ORDER BY sent_at"
        ).fetchall()
    finally:
        src.close()

    for sid, sent_at in rows:
        et = _to_et_iso(sent_at)
        if not et:
            continue
        session = et[:10]
        prints = store.window(session, et, window_s)
        if not prints:
            continue
        snap = compute_features(
            prints,
            snapshot_kind="SIGNAL",
            session_date=session,
            ts_et=et,
            window_s=window_s,
            signal_id=int(sid),
        )
        insert_snapshot(out_conn, snap)
        written += 1
    return written


def snapshot_rejected(
    store: PrintStore,
    out_conn: sqlite3.Connection,
    blocked_jsonl: str,
    *,
    window_s: int = DEFAULT_WINDOW_S,
) -> int:
    """Attach snapshots to blocked/rejected signals (the control group)."""
    if not os.path.exists(blocked_jsonl):
        return 0
    written = 0
    with open(blocked_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("sent_at") or rec.get("ts") or rec.get("timestamp")
            et = _to_et_iso(ts)
            if not et:
                continue
            session = et[:10]
            prints = store.window(session, et, window_s)
            if not prints:
                continue
            snap = compute_features(
                prints,
                snapshot_kind="REJECTED",
                session_date=session,
                ts_et=et,
                window_s=window_s,
                signal_id=rec.get("signal_id"),
            )
            insert_snapshot(out_conn, snap)
            written += 1
    return written


def snapshot_intervals(
    store: PrintStore,
    out_conn: sqlite3.Connection,
    *,
    clock_times_et: Sequence[str] = (
        "09:45", "10:15", "10:45", "11:15", "11:45",
        "12:15", "12:45", "13:15", "13:45", "14:15",
        "14:45", "15:15", "15:45",
    ),
    window_s: int = DEFAULT_WINDOW_S,
) -> int:
    """Unconditional interval snapshots — the unbiased sample (no signal
    selection). One per clock time per session that has prints."""
    written = 0
    for session in store.sessions():
        for hhmm in clock_times_et:
            et = f"{session}T{hhmm}:00"
            prints = store.window(session, et, window_s)
            if not prints:
                continue
            snap = compute_features(
                prints,
                snapshot_kind="INTERVAL",
                session_date=session,
                ts_et=et,
                window_s=window_s,
                signal_id=None,
            )
            insert_snapshot(out_conn, snap)
            written += 1
    return written


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _open_readonly(path: str) -> sqlite3.Connection:
    """Open a sqlite DB strictly read-only (immutable) — production safety."""
    uri = f"file:{os.path.abspath(path)}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def _row_to_print(r: sqlite3.Row) -> Print:
    conds = (r["condition_codes"] or "")
    conds_list = [c for c in conds.split(",") if c]
    return Print(
        ts_utc=r["ts_utc"],
        ts_et=r["ts_et"],
        session_date=r["session_date"],
        root=r["root"],
        expiry=r["expiry"],
        strike=r["strike"],
        right=r["right"],
        trade_px=r["trade_px"],
        size=r["size"],
        exchange=r["exchange"],
        condition_codes=conds_list,
        underlying_px=r["underlying_px"],
        dte=r["dte"],
        bid=r["bid"],
        ask=r["ask"],
        delta=r["delta"],
        gamma=r["gamma"],
        iv=r["iv"],
        greeks_src=r["greeks_src"],
        data_source=r["data_source"],
    )


def _to_et_iso(ts: Optional[str]) -> Optional[str]:
    """Normalise a signal timestamp to a naive ET ISO string 'YYYY-MM-DDTHH:MM:SS'.

    CRITICAL (verified 2026-07-23): spy_signals.sent_at is stored in **UTC**
    (naive, written by datetime.utcnow()), NOT ET. Empirical proof: the sent_at
    hour histogram clusters 13:00-19:00, i.e. RTH (09:30-16:00 ET) shifted +4h =
    13:30-20:00 UTC; and a sent_at of 18:35 maps to the ET-derived time_bucket
    PRE_POWER (= 14:35 ET). An earlier code comment claimed 'naive local ET' —
    that was WRONG and would mis-join every signal by 4h, silently voiding the
    flow battery. We therefore treat the naive stamp as UTC and convert to ET,
    matching the true-ET `ts_et` on the flow prints. (Blocked-signal JSONL uses
    the same logger, so it is treated as UTC too.)
    """
    if not ts:
        return None
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1]
    # drop an explicit +HH:MM offset (we re-attach UTC below)
    idx = s.find("+", 11)  # after the date part
    if idx > 0:
        s = s[:idx]
    # drop microseconds for clean parsing
    if "." in s:
        s = s.split(".", 1)[0]
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return s  # last-resort: return as-is rather than crash the run
    # naive stamp OR 'Z'/'+00:00' => UTC; convert to ET, return naive ET ISO
    dt = dt.replace(tzinfo=timezone.utc)
    et = dt.astimezone(_ET)
    return et.replace(tzinfo=None).isoformat()


def _iso_shift(et_iso: str, seconds: int) -> str:
    """Shift a naive ET ISO timestamp by `seconds` (may be negative)."""
    try:
        base = datetime.fromisoformat(et_iso)
    except ValueError:
        # tolerate 'YYYY-MM-DDTHH:MM' form
        base = datetime.fromisoformat(et_iso + ":00")
    return (base + timedelta(seconds=seconds)).isoformat()
