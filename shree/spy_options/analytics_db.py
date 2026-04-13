"""SQLite analytics store for SPY options signals.

Every signal sent via Telegram is persisted here for win-rate analysis,
regime performance tracking, and confidence model tuning.

Uses synchronous sqlite3 — the write takes <1ms and does not block
the async polling loop meaningfully.

Schema v2 additions:
  expiry_date, dte
  external_composite, news_score, retail_score
  macro_headwind, macro_label, tnx_trend, dxy_trend, equity_pc
  flow_confirmation_score, dark_pool_bias, gex_bias, intraday_pc_ratio
  dynamic_confidence_delta, confidence_time_bucket, confidence_dte_rule
  conflict_detected

Schema v3 additions (outcome tracking / feedback loop):
  outcome          — 'open' | 'win' | 'loss' | 'scratch'
  spy_price_exit   — SPY price when exit alert fired (or manually recorded)
  exit_at          — UTC timestamp of exit
  pnl_pct          — estimated SPY move % from entry to exit (+ = favourable)
  exit_trigger     — which trigger fired: 'time_stop' | 'profit_target' |
                     'adverse_move' | 'regime_flip' | 'vwap_reversion' | 'manual'

Performance queries: see win_rate_summary(), win_rate_by_signal_type(),
  win_rate_by_time_bucket(), win_rate_by_regime().
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from ..utils.logger import logger

if TYPE_CHECKING:
    from .signal_engine import SpySignal


_DDL_TABLE = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS spy_signals (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    sent_at                  TEXT NOT NULL,
    signal_type              TEXT NOT NULL,
    strike                   REAL NOT NULL,
    right                    TEXT NOT NULL,
    expiry                   TEXT NOT NULL,
    expiry_date              TEXT,
    dte                      INTEGER,
    confidence               REAL NOT NULL,
    confidence_tier          TEXT NOT NULL,
    spy_price                REAL,
    vix                      REAL,
    iv_rank                  REAL,
    volume                   INTEGER,
    volume_spike_mult        REAL,
    bid                      REAL,
    ask                      REAL,
    bid_size                 INTEGER,
    ask_size                 INTEGER,
    spread_pct               REAL,
    open_interest            INTEGER,
    delta                    REAL,
    gamma                    REAL,
    theta                    REAL,
    vega                     REAL,
    impl_vol                 REAL,
    regime                   TEXT,
    sentiment_score          REAL,
    sentiment_label          TEXT,
    flow_score               REAL,
    external_composite       REAL,
    news_score               REAL,
    retail_score             REAL,
    macro_headwind           REAL,
    macro_label              TEXT,
    tnx_trend                TEXT,
    dxy_trend                TEXT,
    equity_pc                REAL,
    flow_confirmation_score  REAL,
    dark_pool_bias           TEXT,
    gex_bias                 TEXT,
    intraday_pc_ratio        REAL,
    dynamic_confidence_delta REAL,
    confidence_time_bucket   TEXT,
    confidence_dte_rule      TEXT,
    conflict_detected        INTEGER DEFAULT 0,
    reasoning                TEXT,
    suggested_trade          TEXT,
    -- v3: outcome tracking (populated via record_outcome())
    outcome                  TEXT DEFAULT 'open',   -- open | win | loss | scratch
    spy_price_exit           REAL,
    exit_at                  TEXT,
    pnl_pct                  REAL,                  -- SPY move % entry→exit (+= favourable)
    exit_trigger             TEXT,                  -- which exit rule fired
    created_at               TEXT DEFAULT (datetime('now'))
);
"""

# Indexes created AFTER migrations so columns exist on legacy databases
_DDL_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_spy_signals_sent_at    ON spy_signals(sent_at);
CREATE INDEX IF NOT EXISTS idx_spy_signals_type       ON spy_signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_spy_signals_confidence ON spy_signals(confidence);
CREATE INDEX IF NOT EXISTS idx_spy_signals_regime     ON spy_signals(regime);
CREATE INDEX IF NOT EXISTS idx_spy_signals_dte        ON spy_signals(dte);
CREATE INDEX IF NOT EXISTS idx_spy_signals_flow       ON spy_signals(flow_confirmation_score);
"""

# Migration: add new columns to existing DB without dropping data
_MIGRATIONS = [
    "ALTER TABLE spy_signals ADD COLUMN expiry_date TEXT",
    "ALTER TABLE spy_signals ADD COLUMN dte INTEGER",
    "ALTER TABLE spy_signals ADD COLUMN external_composite REAL",
    "ALTER TABLE spy_signals ADD COLUMN news_score REAL",
    "ALTER TABLE spy_signals ADD COLUMN retail_score REAL",
    "ALTER TABLE spy_signals ADD COLUMN macro_headwind REAL",
    "ALTER TABLE spy_signals ADD COLUMN macro_label TEXT",
    "ALTER TABLE spy_signals ADD COLUMN tnx_trend TEXT",
    "ALTER TABLE spy_signals ADD COLUMN dxy_trend TEXT",
    "ALTER TABLE spy_signals ADD COLUMN equity_pc REAL",
    "ALTER TABLE spy_signals ADD COLUMN flow_confirmation_score REAL",
    "ALTER TABLE spy_signals ADD COLUMN dark_pool_bias TEXT",
    "ALTER TABLE spy_signals ADD COLUMN gex_bias TEXT",
    "ALTER TABLE spy_signals ADD COLUMN intraday_pc_ratio REAL",
    "ALTER TABLE spy_signals ADD COLUMN dynamic_confidence_delta REAL",
    "ALTER TABLE spy_signals ADD COLUMN confidence_time_bucket TEXT",
    "ALTER TABLE spy_signals ADD COLUMN confidence_dte_rule TEXT",
    "ALTER TABLE spy_signals ADD COLUMN conflict_detected INTEGER DEFAULT 0",
    # v3 outcome tracking
    "ALTER TABLE spy_signals ADD COLUMN outcome TEXT DEFAULT 'open'",
    "ALTER TABLE spy_signals ADD COLUMN spy_price_exit REAL",
    "ALTER TABLE spy_signals ADD COLUMN exit_at TEXT",
    "ALTER TABLE spy_signals ADD COLUMN pnl_pct REAL",
    "ALTER TABLE spy_signals ADD COLUMN exit_trigger TEXT",
]


class AnalyticsDB:
    """Synchronous SQLite writer for SPY option signals."""

    def __init__(self, db_path: str = "data/spy_options_signals.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(db_path)
        # 1) Create table (no-op if it already exists with old schema)
        self._conn.executescript(_DDL_TABLE)
        self._conn.commit()
        # 2) Add any missing columns so legacy DBs get new fields
        self._apply_migrations()
        # 3) Now safe to create indexes — all columns guaranteed to exist
        self._conn.executescript(_DDL_INDEXES)
        self._conn.commit()
        logger.info("AnalyticsDB initialised → {}", db_path)

    def _apply_migrations(self) -> None:
        """Add new columns to an existing database without dropping data."""
        for sql in _MIGRATIONS:
            try:
                self._conn.execute(sql)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists — safe to skip

    def insert(self, sig: "SpySignal") -> None:
        """Persist a sent signal. Called synchronously after Telegram delivery."""
        try:
            self._conn.execute(
                """
                INSERT INTO spy_signals (
                    sent_at, signal_type, strike, right, expiry,
                    expiry_date, dte,
                    confidence, confidence_tier,
                    spy_price, vix, iv_rank,
                    volume, volume_spike_mult,
                    bid, ask, bid_size, ask_size, spread_pct,
                    open_interest,
                    delta, gamma, theta, vega, impl_vol,
                    regime, sentiment_score, sentiment_label,
                    flow_score,
                    external_composite, news_score, retail_score,
                    macro_headwind, macro_label, tnx_trend, dxy_trend, equity_pc,
                    flow_confirmation_score, dark_pool_bias, gex_bias, intraday_pc_ratio,
                    dynamic_confidence_delta, confidence_time_bucket,
                    confidence_dte_rule, conflict_detected,
                    reasoning, suggested_trade
                ) VALUES (
                    ?,?,?,?,?, ?,?, ?,?, ?,?,?, ?,?, ?,?,?,?,?, ?,
                    ?,?,?,?,?, ?,?,?, ?,
                    ?,?,?, ?,?,?,?,?, ?,?,?,?,
                    ?,?,?, ?,
                    ?,?
                )
                """,
                (
                    datetime.utcnow().isoformat(),
                    sig.signal_type.value,
                    sig.strike, sig.right, sig.expiry,
                    getattr(sig, "expiry_date", None),
                    getattr(sig, "dte", None),
                    sig.confidence, sig.confidence_tier,
                    sig.spy_price, sig.vix, sig.iv_rank,
                    sig.volume, sig.volume_spike_mult,
                    sig.bid, sig.ask, sig.bid_size, sig.ask_size, sig.spread_pct,
                    sig.open_interest,
                    sig.delta, sig.gamma, sig.theta, sig.vega, sig.impl_vol,
                    sig.regime, sig.sentiment_score, sig.sentiment_label,
                    sig.flow_score,
                    getattr(sig, "external_composite", None),
                    getattr(sig, "news_score", None),
                    getattr(sig, "retail_score", None),
                    getattr(sig, "macro_headwind", None),
                    getattr(sig, "macro_label", None),
                    getattr(sig, "tnx_trend", None),
                    getattr(sig, "dxy_trend", None),
                    getattr(sig, "equity_pc", None),
                    getattr(sig, "flow_confirmation_score", None),
                    getattr(sig, "dark_pool_bias", None),
                    getattr(sig, "gex_bias", None),
                    getattr(sig, "intraday_pc_ratio", None),
                    getattr(sig, "dynamic_confidence_delta", None),
                    getattr(sig, "confidence_time_bucket", None),
                    getattr(sig, "confidence_dte_rule", None),
                    int(getattr(sig, "conflict_detected", False)),
                    json.dumps(sig.reasoning),
                    sig.suggested_trade,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("AnalyticsDB insert failed: {}", exc)

    def find_signal_id(self, sig: "SpySignal") -> Optional[int]:
        """Return the DB row id of the most recently inserted row matching
        this signal's type, strike, right, and expiry.  Returns None if not found.
        Called by manager to look up the row before recording an outcome.
        """
        try:
            row = self._conn.execute(
                """
                SELECT id FROM spy_signals
                WHERE signal_type=? AND strike=? AND right=? AND expiry=?
                  AND outcome='open'
                ORDER BY id DESC LIMIT 1
                """,
                (sig.signal_type.value, sig.strike, sig.right, sig.expiry),
            ).fetchone()
            return row[0] if row else None
        except Exception as exc:
            logger.warning("find_signal_id failed: {}", exc)
            return None

    def record_outcome(
        self,
        signal_id: int,
        outcome: str,                # 'win' | 'loss' | 'scratch'
        spy_price_exit: float,
        exit_trigger: str,           # 'time_stop' | 'profit_target' | 'adverse_move' |
                                     # 'regime_flip' | 'vwap_reversion' | 'manual'
        direction: str = "NEUTRAL",  # 'BULLISH' | 'BEARISH' | 'NEUTRAL'
        entry_price: Optional[float] = None,
    ) -> None:
        """Record the exit outcome for a previously inserted signal.

        Called from manager._check_exit_conditions() when an exit trigger
        fires.  Computes pnl_pct as the SPY move from entry to exit in the
        signal direction (positive = favourable, negative = adverse).
        """
        pnl_pct: Optional[float] = None
        if entry_price and entry_price > 0 and spy_price_exit > 0:
            raw_pct = (spy_price_exit - entry_price) / entry_price * 100.0
            if direction == "BULLISH":
                pnl_pct = round(raw_pct, 4)
            elif direction == "BEARISH":
                pnl_pct = round(-raw_pct, 4)

        try:
            self._conn.execute(
                """
                UPDATE spy_signals
                SET outcome=?, spy_price_exit=?, exit_at=?, pnl_pct=?, exit_trigger=?
                WHERE id=?
                """,
                (
                    outcome,
                    spy_price_exit,
                    datetime.utcnow().isoformat(),
                    pnl_pct,
                    exit_trigger,
                    signal_id,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("AnalyticsDB record_outcome failed: {}", exc)

    # ── Performance queries (feedback loop) ──────────────────────────────────

    def win_rate_summary(self) -> Dict:
        """Return overall win-rate statistics across all closed signals.

        Only counts signals where outcome != 'open'.
        Returns a dict with keys: total, wins, losses, scratches, win_rate_pct,
          avg_pnl_pct, best_pnl_pct, worst_pnl_pct.
        """
        try:
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*)                                          AS total,
                    SUM(CASE WHEN outcome='win'     THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN outcome='loss'    THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN outcome='scratch' THEN 1 ELSE 0 END) AS scratches,
                    AVG(pnl_pct)                                      AS avg_pnl,
                    MAX(pnl_pct)                                      AS best_pnl,
                    MIN(pnl_pct)                                      AS worst_pnl
                FROM spy_signals
                WHERE outcome != 'open'
                """
            ).fetchone()
            if not row or row[0] == 0:
                return {"total": 0, "wins": 0, "losses": 0, "scratches": 0,
                        "win_rate_pct": 0.0, "avg_pnl_pct": 0.0,
                        "best_pnl_pct": None, "worst_pnl_pct": None}
            total, wins, losses, scratches, avg_pnl, best_pnl, worst_pnl = row
            win_rate = (wins / total * 100.0) if total else 0.0
            return {
                "total": total, "wins": wins, "losses": losses,
                "scratches": scratches,
                "win_rate_pct": round(win_rate, 1),
                "avg_pnl_pct": round(avg_pnl or 0.0, 3),
                "best_pnl_pct": round(best_pnl, 3) if best_pnl else None,
                "worst_pnl_pct": round(worst_pnl, 3) if worst_pnl else None,
            }
        except Exception as exc:
            logger.warning("win_rate_summary failed: {}", exc)
            return {}

    def win_rate_by_signal_type(self) -> List[Dict]:
        """Win rate broken down by signal_type.  Useful for disabling
        underperforming signal types."""
        return self._win_rate_group("signal_type")

    def win_rate_by_time_bucket(self) -> List[Dict]:
        """Win rate broken down by confidence_time_bucket (OPEN/MIDDAY/etc)."""
        return self._win_rate_group("confidence_time_bucket")

    def win_rate_by_regime(self) -> List[Dict]:
        """Win rate broken down by market regime at signal time."""
        return self._win_rate_group("regime")

    def _win_rate_group(self, group_col: str) -> List[Dict]:
        try:
            rows = self._conn.execute(
                f"""
                SELECT
                    {group_col}                                           AS group_key,
                    COUNT(*)                                              AS total,
                    SUM(CASE WHEN outcome='win'  THEN 1 ELSE 0 END)      AS wins,
                    SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END)      AS losses,
                    AVG(pnl_pct)                                         AS avg_pnl
                FROM spy_signals
                WHERE outcome != 'open'
                GROUP BY {group_col}
                ORDER BY total DESC
                """
            ).fetchall()
            result = []
            for group_key, total, wins, losses, avg_pnl in rows:
                win_rate = (wins / total * 100.0) if total else 0.0
                result.append({
                    "group": group_key,
                    "total": total,
                    "wins": wins,
                    "losses": losses,
                    "win_rate_pct": round(win_rate, 1),
                    "avg_pnl_pct": round(avg_pnl or 0.0, 3),
                })
            return result
        except Exception as exc:
            logger.warning("_win_rate_group({}) failed: {}", group_col, exc)
            return []

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
