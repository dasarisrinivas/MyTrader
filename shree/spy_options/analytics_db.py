"""SQLite analytics store for SPY options signals.

Every signal sent via Telegram is persisted here for future win-rate
analysis, regime performance tracking, and confidence model tuning.

Uses synchronous sqlite3 — the write takes <1ms and does not block
the async polling loop meaningfully.

Schema v2 additions:
  expiry_date, dte
  external_composite, news_score, retail_score
  macro_headwind, macro_label, tnx_trend, dxy_trend, equity_pc
  flow_confirmation_score, dark_pool_bias, gex_bias, intraday_pc_ratio
  dynamic_confidence_delta, confidence_time_bucket, confidence_dte_rule
  conflict_detected
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

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

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
