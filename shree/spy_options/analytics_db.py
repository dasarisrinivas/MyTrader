"""SQLite analytics store for SPY options signals.

Every signal sent via Telegram is persisted here for future win-rate
analysis, regime performance tracking, and confidence model tuning.

Uses synchronous sqlite3 — the write takes <1ms and does not block
the async polling loop meaningfully.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils.logger import logger

if TYPE_CHECKING:
    from .signal_engine import SpySignal


_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS spy_signals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    sent_at           TEXT NOT NULL,
    signal_type       TEXT NOT NULL,
    strike            REAL NOT NULL,
    right             TEXT NOT NULL,
    expiry            TEXT NOT NULL,
    confidence        REAL NOT NULL,
    confidence_tier   TEXT NOT NULL,
    spy_price         REAL,
    vix               REAL,
    iv_rank           REAL,
    volume            INTEGER,
    volume_spike_mult REAL,
    bid               REAL,
    ask               REAL,
    bid_size          INTEGER,
    ask_size          INTEGER,
    spread_pct        REAL,
    open_interest     INTEGER,
    delta             REAL,
    gamma             REAL,
    theta             REAL,
    vega              REAL,
    impl_vol          REAL,
    regime            TEXT,
    sentiment_score   REAL,
    sentiment_label   TEXT,
    flow_score        REAL,
    reasoning         TEXT,
    suggested_trade   TEXT,
    created_at        TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_spy_signals_sent_at    ON spy_signals(sent_at);
CREATE INDEX IF NOT EXISTS idx_spy_signals_type       ON spy_signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_spy_signals_confidence ON spy_signals(confidence);
CREATE INDEX IF NOT EXISTS idx_spy_signals_regime     ON spy_signals(regime);
"""


class AnalyticsDB:
    """Synchronous SQLite writer for SPY option signals."""

    def __init__(self, db_path: str = "data/spy_options_signals.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(db_path)
        self._conn.executescript(_DDL)
        self._conn.commit()
        logger.info("AnalyticsDB initialised → {}", db_path)

    def insert(self, sig: "SpySignal") -> None:
        """Persist a sent signal. Called synchronously after Telegram delivery."""
        try:
            self._conn.execute(
                """
                INSERT INTO spy_signals (
                    sent_at, signal_type, strike, right, expiry,
                    confidence, confidence_tier,
                    spy_price, vix, iv_rank,
                    volume, volume_spike_mult,
                    bid, ask, bid_size, ask_size, spread_pct,
                    open_interest,
                    delta, gamma, theta, vega, impl_vol,
                    regime, sentiment_score, sentiment_label,
                    flow_score, reasoning, suggested_trade
                ) VALUES (
                    ?,?,?,?,?, ?,?, ?,?,?, ?,?, ?,?,?,?,?, ?,
                    ?,?,?,?,?, ?,?,?, ?,?,?
                )
                """,
                (
                    datetime.utcnow().isoformat(),
                    sig.signal_type.value,
                    sig.strike,
                    sig.right,
                    sig.expiry,
                    sig.confidence,
                    sig.confidence_tier,
                    sig.spy_price,
                    sig.vix,
                    sig.iv_rank,
                    sig.volume,
                    sig.volume_spike_mult,
                    sig.bid,
                    sig.ask,
                    sig.bid_size,
                    sig.ask_size,
                    sig.spread_pct,
                    sig.open_interest,
                    sig.delta,
                    sig.gamma,
                    sig.theta,
                    sig.vega,
                    sig.impl_vol,
                    sig.regime,
                    sig.sentiment_score,
                    sig.sentiment_label,
                    sig.flow_score,
                    json.dumps(sig.reasoning),
                    sig.suggested_trade,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("AnalyticsDB insert failed: {}", exc)

    def close(self) -> None:
        """Close the database connection cleanly."""
        try:
            self._conn.close()
        except Exception:
            pass
