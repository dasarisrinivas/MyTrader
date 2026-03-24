"""Tests for gold_trades / gold_daily_summary tables in trade_journal_db."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from shree.monitoring.trade_journal_db import (
    get_connection,
    get_gold_exit_breakdown,
    get_gold_signal_breakdown,
    get_gold_summary,
    get_gold_trades,
    init_db,
    upsert_gold_daily_summary,
    upsert_gold_trade,
)


@pytest.fixture()
def db_conn():
    """Temporary isolated DB for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    conn = get_connection(db_path)
    yield conn
    conn.close()
    Path(db_path).unlink(missing_ok=True)


def _trade(
    trade_id: str = "abc123",
    date: str = "2026-03-10",
    symbol: str = "MGC",
    action: str = "BUY",
    signal_type: str = "VWAP_PB_LONG",
    contracts: int = 1,
    entry_price: float = 2350.0,
    stop_loss: float = 2338.0,
    take_profit: float = 2380.0,
    exit_price: float = 2380.0,
    realized_pnl: float = 300.0,
    commission: float = 1.20,
    net_pnl: float = 298.80,
    entry_time: str = "2026-03-10T09:30:00+00:00",
    exit_time: str = "2026-03-10T10:15:00+00:00",
    hold_bars: int = 45,
    exit_reason: str = "PROFIT_TARGET",
    regime: str = "TRENDING_BULL",
    atr_at_entry: float = 8.5,
    adx_at_entry: float = 30.0,
    win: int = 1,
) -> dict:
    return dict(
        trade_id=trade_id, date=date, symbol=symbol, action=action,
        signal_type=signal_type, contracts=contracts, entry_price=entry_price,
        stop_loss=stop_loss, take_profit=take_profit, exit_price=exit_price,
        realized_pnl=realized_pnl, commission=commission, net_pnl=net_pnl,
        entry_time=entry_time, exit_time=exit_time, hold_bars=hold_bars,
        exit_reason=exit_reason, regime=regime, atr_at_entry=atr_at_entry,
        adx_at_entry=adx_at_entry, win=win,
    )


class TestUpsertGoldTrade:
    def test_insert_and_retrieve(self, db_conn) -> None:
        upsert_gold_trade(db_conn, _trade())
        db_conn.commit()
        rows = get_gold_trades(db_conn, "2026-03-10", "2026-03-10")
        assert len(rows) == 1
        assert rows[0]["trade_id"] == "abc123"
        assert rows[0]["net_pnl"] == pytest.approx(298.80)

    def test_upsert_replaces_on_duplicate_trade_id(self, db_conn) -> None:
        upsert_gold_trade(db_conn, _trade(net_pnl=100.0))
        upsert_gold_trade(db_conn, _trade(net_pnl=200.0))   # same trade_id
        db_conn.commit()
        rows = get_gold_trades(db_conn, "2026-03-10", "2026-03-10")
        assert len(rows) == 1
        assert rows[0]["net_pnl"] == pytest.approx(200.0)

    def test_multiple_trades_same_day(self, db_conn) -> None:
        upsert_gold_trade(db_conn, _trade(trade_id="t1", net_pnl=100.0))
        upsert_gold_trade(db_conn, _trade(trade_id="t2", net_pnl=-50.0, win=0))
        db_conn.commit()
        rows = get_gold_trades(db_conn, "2026-03-10", "2026-03-10")
        assert len(rows) == 2


class TestGetGoldSummary:
    def test_empty_returns_empty_dict(self, db_conn) -> None:
        assert get_gold_summary(db_conn, "2026-01-01", "2026-01-31") == {}

    def test_summary_metrics(self, db_conn) -> None:
        upsert_gold_trade(db_conn, _trade(trade_id="t1", net_pnl=100.0, win=1))
        upsert_gold_trade(db_conn, _trade(trade_id="t2", net_pnl=-40.0, win=0))
        db_conn.commit()
        s = get_gold_summary(db_conn, "2026-03-10", "2026-03-10")
        assert s["total_trades"] == 2
        assert s["wins"] == 1
        assert s["losses"] == 1
        assert abs(s["net_pnl"] - 60.0) < 0.01
        assert s["profit_factor"] == pytest.approx(100.0 / 40.0, abs=0.05)
        assert s["win_rate_pct"] == pytest.approx(50.0)

    def test_profit_factor_none_when_no_losses(self, db_conn) -> None:
        upsert_gold_trade(db_conn, _trade(trade_id="t1", net_pnl=100.0, win=1))
        db_conn.commit()
        s = get_gold_summary(db_conn, "2026-03-10", "2026-03-10")
        assert s["profit_factor"] is None   # No losses → undefined PF


class TestSignalAndExitBreakdown:
    def test_signal_breakdown(self, db_conn) -> None:
        upsert_gold_trade(db_conn, _trade(trade_id="t1", signal_type="VWAP_PB_LONG", net_pnl=100.0, win=1))
        upsert_gold_trade(db_conn, _trade(trade_id="t2", signal_type="ORB_LONG", net_pnl=-30.0, win=0))
        upsert_gold_trade(db_conn, _trade(trade_id="t3", signal_type="VWAP_PB_LONG", net_pnl=80.0, win=1))
        db_conn.commit()
        rows = get_gold_signal_breakdown(db_conn, "2026-03-10", "2026-03-10")
        by_type = {r["signal_type"]: r for r in rows}
        assert by_type["VWAP_PB_LONG"]["total"] == 2
        assert by_type["VWAP_PB_LONG"]["wins"] == 2
        assert by_type["ORB_LONG"]["total"] == 1
        assert by_type["ORB_LONG"]["wins"] == 0

    def test_exit_breakdown(self, db_conn) -> None:
        upsert_gold_trade(db_conn, _trade(trade_id="t1", exit_reason="PROFIT_TARGET", net_pnl=100.0, win=1))
        upsert_gold_trade(db_conn, _trade(trade_id="t2", exit_reason="STOP_LOSS", net_pnl=-40.0, win=0))
        upsert_gold_trade(db_conn, _trade(trade_id="t3", exit_reason="TIME_STOP", net_pnl=-10.0, win=0))
        db_conn.commit()
        rows = get_gold_exit_breakdown(db_conn, "2026-03-10", "2026-03-10")
        by_reason = {r["exit_reason"]: r for r in rows}
        assert by_reason["PROFIT_TARGET"]["net_pnl"] == pytest.approx(100.0)
        assert by_reason["STOP_LOSS"]["net_pnl"] == pytest.approx(-40.0)


class TestUpsertGoldDailySummary:
    def test_insert_and_upsert(self, db_conn) -> None:
        upsert_gold_daily_summary(db_conn, {
            "date": "2026-03-10", "symbol": "MGC",
            "total_trades": 3, "wins": 2, "losses": 1,
            "win_rate": 66.7, "net_pnl": 120.0, "gross_pnl": 125.0,
        })
        db_conn.commit()
        row = db_conn.execute(
            "SELECT * FROM gold_daily_summary WHERE date='2026-03-10'"
        ).fetchone()
        assert row is not None
        assert row["total_trades"] == 3

        # Upsert updates
        upsert_gold_daily_summary(db_conn, {
            "date": "2026-03-10", "symbol": "MGC",
            "total_trades": 4, "wins": 3, "losses": 1,
            "win_rate": 75.0, "net_pnl": 150.0, "gross_pnl": 155.0,
        })
        db_conn.commit()
        row2 = db_conn.execute(
            "SELECT * FROM gold_daily_summary WHERE date='2026-03-10'"
        ).fetchone()
        assert row2["total_trades"] == 4
        assert row2["net_pnl"] == pytest.approx(150.0)
