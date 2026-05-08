"""Backfill the learning DB from existing trade history.

Two data sources:

  1. MES futures
       Closed trades:    data/orders.db.trade_outcomes (where exit_time IS NOT
                         NULL and net_pnl is not corrupted)
       Signal context:   logs/decisions.jsonl (the SIGNAL rows, with regime,
                         ADX, ATR, etc.)
       Join: each closed trade's `entry_time` is matched to the nearest
             SIGNAL row in decisions.jsonl preceding it (within 120 seconds).

  2. SPY options
       Closed trades:    data/spy_options_signals.db.spy_signals — already
                         contains both signal context AND the outcome /
                         pnl_pct / exit_trigger fields. Single-table query.
       VIX is recorded directly on the signal row.

Idempotent: each event_id is unique, so re-running is safe.

Usage:
    python3 -m shree.trading_manager.backfill_learning
or:
    from shree.trading_manager.backfill_learning import run_backfill
    new = run_backfill()
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from .learning import (
    bucketize_time,
    bucketize_vix,
    normalize_regime,
    open_db,
    upsert_event,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_corrupted_pnl(reason: str, pnl: float) -> bool:
    """Heuristic for the bot's CORRUPTED_PNL_CUMULATIVE_IBKR rows. These showed
    up as $17K+ "PnL" entries in early Feb 2026 and would poison the stats."""
    if "CORRUPTED" in (reason or ""):
        return True
    # MES: a single contract risking ~$75 cannot legitimately PnL >= $500 on
    # one trade given our stop/target sizing. Treat anything outside [-500, +500]
    # as corrupted.
    if abs(pnl) > 500.0:
        return True
    return False


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return None


def _load_mes_signals(decisions_jsonl: str) -> List[Dict]:
    """Load all SIGNAL rows from decisions.jsonl, sorted by ts ascending."""
    if not os.path.exists(decisions_jsonl):
        return []
    rows = []
    with open(decisions_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("outcome") != "SIGNAL":
                continue
            if d.get("action") not in ("BUY", "SELL"):
                continue
            ts = _parse_iso(d.get("ts", ""))
            if ts is None:
                continue
            rows.append({"ts": ts, "raw": d})
    rows.sort(key=lambda r: r["ts"])
    return rows


def _signal_type_from_reason(reason: str) -> str:
    """The MES bot writes reason like
       "EMA21_PB_LONG | ADX=23 | RSI=54 | MACD_H=0.25 | ATR=6.4 | ...".
    First token is the signal type."""
    if not reason:
        return "UNKNOWN"
    return reason.split(" ")[0].split("|")[0].strip() or "UNKNOWN"


def _mes_regime_from_signal(d: Dict) -> str:
    """MES doesn't record an explicit regime label, but the rules engine
    derives one from ADX. Keep the same logic here so backfilled buckets
    match what the rules engine will look up at runtime."""
    try:
        adx = float(d.get("adx") or 0.0)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if adx >= 22:
        return "TRENDING"
    if adx <= 15:
        return "CHOPPY"
    return "MIXED"


def _match_signal_to_trade(
    signals: List[Dict],
    entry_time: datetime,
    window_seconds: int = 120,
) -> Optional[Dict]:
    """Find the most recent SIGNAL row that fired at or just before
    entry_time, within the time window. Returns the raw dict or None."""
    if not signals or entry_time is None:
        return None
    best = None
    best_delta = None
    # Linear scan from end backwards (signals are sorted asc).
    for row in reversed(signals):
        delta = (entry_time - row["ts"]).total_seconds()
        if delta < 0:
            continue  # signal AFTER entry — keep looking earlier
        if delta > window_seconds:
            break  # too old; everything earlier is even older
        if best_delta is None or delta < best_delta:
            best = row
            best_delta = delta
    return best["raw"] if best else None


# ─────────────────────────────────────────────────────────────────────────────
# Backfill — MES
# ─────────────────────────────────────────────────────────────────────────────

def backfill_mes(
    learning_db: str,
    orders_db: str,
    decisions_jsonl: str,
) -> Tuple[int, int, int]:
    """Returns (events_processed, events_new, events_skipped_no_signal)."""
    if not os.path.exists(orders_db):
        return (0, 0, 0)
    signals = _load_mes_signals(decisions_jsonl)

    src = sqlite3.connect(orders_db)
    src.row_factory = sqlite3.Row
    closed = src.execute(
        """SELECT trade_cycle_id, root_order_id, entry_time, exit_time,
                  entry_price, exit_price, exit_reason, net_pnl
           FROM trade_outcomes
           WHERE exit_time IS NOT NULL AND net_pnl IS NOT NULL"""
    ).fetchall()
    src.close()

    learning = open_db(learning_db)

    processed = new = skipped = 0
    for r in closed:
        processed += 1
        pnl = float(r["net_pnl"] or 0.0)
        reason = r["exit_reason"] or ""
        if _is_corrupted_pnl(reason, pnl):
            skipped += 1
            continue
        entry_dt = _parse_iso(r["entry_time"])
        if entry_dt is None:
            skipped += 1
            continue
        sig_raw = _match_signal_to_trade(signals, entry_dt)
        if sig_raw is None:
            skipped += 1
            continue

        signal_type = _signal_type_from_reason(sig_raw.get("reason", ""))
        regime = _mes_regime_from_signal(sig_raw)
        time_bucket = bucketize_time(sig_raw.get("ts", ""))
        # MES: no VIX recorded in decisions.jsonl
        vix_bucket = "UNKNOWN"
        event_id = f"mes:{r['trade_cycle_id']}"
        if upsert_event(
            learning,
            event_id=event_id,
            bot="mes",
            signal_type=signal_type,
            regime=regime,
            time_bucket=time_bucket,
            vix_bucket=vix_bucket,
            pnl=pnl,
            entry_time=r["entry_time"],
            exit_time=r["exit_time"],
        ):
            new += 1

    learning.close()
    return (processed, new, skipped)


# ─────────────────────────────────────────────────────────────────────────────
# Backfill — SPY options
# ─────────────────────────────────────────────────────────────────────────────

def backfill_spy(
    learning_db: str,
    spy_db: str,
) -> Tuple[int, int, int]:
    """The spy_signals table has built-in outcome/pnl_pct/exit_trigger fields.
    Returns (events_processed, events_new, events_skipped)."""
    if not os.path.exists(spy_db):
        return (0, 0, 0)

    src = sqlite3.connect(spy_db)
    src.row_factory = sqlite3.Row
    closed = src.execute(
        """SELECT id, sent_at, signal_type, outcome, pnl_pct, exit_at,
                  vix, regime, dollar_pnl_1ct
           FROM spy_signals
           WHERE outcome IS NOT NULL AND outcome != ''"""
    ).fetchall()
    src.close()

    learning = open_db(learning_db)

    processed = new = skipped = 0
    for r in closed:
        processed += 1
        # Convert pct PnL to a dollar proxy. If dollar_pnl_1ct is populated
        # use it directly; otherwise fall back to pnl_pct as a normalized
        # dimensionless score (still preserves win/loss sign).
        if r["dollar_pnl_1ct"] is not None:
            pnl = float(r["dollar_pnl_1ct"] or 0.0)
        elif r["pnl_pct"] is not None:
            pnl = float(r["pnl_pct"] or 0.0)
        else:
            skipped += 1
            continue
        if _is_corrupted_pnl("", pnl):
            skipped += 1
            continue

        signal_type = (r["signal_type"] or "UNKNOWN").upper()
        regime = normalize_regime(r["regime"] or "UNKNOWN")
        time_bucket = bucketize_time(r["sent_at"] or "")
        vix_bucket = bucketize_vix(r["vix"])
        event_id = f"spy:{r['id']}"

        if upsert_event(
            learning,
            event_id=event_id,
            bot="spy_options",
            signal_type=signal_type,
            regime=regime,
            time_bucket=time_bucket,
            vix_bucket=vix_bucket,
            pnl=pnl,
            entry_time=r["sent_at"] or "",
            exit_time=r["exit_at"] or "",
        ):
            new += 1

    learning.close()
    return (processed, new, skipped)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_backfill(
    learning_db: str = "data/learning.db",
    orders_db: str = "data/orders.db",
    decisions_jsonl: str = "logs/decisions.jsonl",
    spy_db: str = "data/spy_options_signals.db",
) -> Dict:
    """Run both backfills, return a summary dict."""
    mes_p, mes_n, mes_s = backfill_mes(learning_db, orders_db, decisions_jsonl)
    spy_p, spy_n, spy_s = backfill_spy(learning_db, spy_db)
    return {
        "mes": {"processed": mes_p, "new": mes_n, "skipped": mes_s},
        "spy": {"processed": spy_p, "new": spy_n, "skipped": spy_s},
    }


if __name__ == "__main__":
    res = run_backfill()
    print("Backfill complete:")
    print(f"  MES futures: {res['mes']['processed']} processed, "
          f"{res['mes']['new']} new, {res['mes']['skipped']} skipped")
    print(f"  SPY options: {res['spy']['processed']} processed, "
          f"{res['spy']['new']} new, {res['spy']['skipped']} skipped")
