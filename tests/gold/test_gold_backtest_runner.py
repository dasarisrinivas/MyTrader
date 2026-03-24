"""Tests for shree/backtest/gold_runner.py — the library backtest runner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shree.backtest.gold_runner import BacktestResult, BacktestSummary, GoldBacktestRunner
from shree.config.gold import GoldStrategyConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cfg(warmup: int = 30) -> GoldStrategyConfig:
    cfg = GoldStrategyConfig(enabled=True, symbol="MGC")
    cfg.indicators.warmup_bars = warmup
    cfg.exit.time_stop_bars = 20
    cfg.exit.atr_sl_multiplier = 1.5
    cfg.exit.atr_tp_multiplier = 2.5
    cfg.risk.max_risk_per_trade_usd = 50.0
    cfg.risk.daily_loss_limit_usd = 500.0
    cfg.risk.max_trades_per_day = 10
    return cfg


def _make_df(n: int = 200, trend: float = 1.0) -> pd.DataFrame:
    np.random.seed(7)
    closes = 2350.0 + np.arange(n) * trend + np.random.normal(0, 0.5, n)
    highs = closes + np.abs(np.random.normal(0, 1.5, n))
    lows = closes - np.abs(np.random.normal(0, 1.5, n))
    volumes = np.random.randint(20, 200, n).astype(float)
    idx = pd.date_range("2026-01-15 08:20", periods=n, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


# ── GoldBacktestRunner ────────────────────────────────────────────────────────

class TestGoldBacktestRunner:
    def test_returns_list(self) -> None:
        runner = GoldBacktestRunner(_cfg())
        assert isinstance(runner.run(_make_df()), list)

    def test_no_trades_before_warmup(self) -> None:
        runner = GoldBacktestRunner(_cfg(warmup=500))
        trades = runner.run(_make_df(200))
        assert len(trades) == 0

    def test_all_results_are_backtest_result(self) -> None:
        runner = GoldBacktestRunner(_cfg())
        for t in runner.run(_make_df()):
            assert isinstance(t, BacktestResult)

    def test_all_exits_have_reason(self) -> None:
        valid = {"STOP_LOSS", "PROFIT_TARGET", "TIME_STOP", "FLATTEN_SESSION"}
        runner = GoldBacktestRunner(_cfg())
        for t in runner.run(_make_df()):
            assert t.exit_reason in valid

    def test_all_exits_have_nonzero_price(self) -> None:
        runner = GoldBacktestRunner(_cfg())
        for t in runner.run(_make_df()):
            assert t.exit_price > 0

    def test_pnl_consistent_long(self) -> None:
        from shree.risk.trade_math import get_contract_spec
        cfg = _cfg()
        spec = get_contract_spec(cfg.symbol)
        runner = GoldBacktestRunner(cfg)
        for t in runner.run(_make_df()):
            if t.action == "BUY":
                expected = (t.exit_price - t.entry_price) * spec.point_value * t.contracts
                assert abs(t.gross_pnl - expected) < 0.01
            else:
                expected = (t.entry_price - t.exit_price) * spec.point_value * t.contracts
                assert abs(t.gross_pnl - expected) < 0.01

    def test_net_pnl_less_than_or_equal_gross_abs(self) -> None:
        runner = GoldBacktestRunner(_cfg())
        for t in runner.run(_make_df()):
            assert abs(t.net_pnl) <= abs(t.gross_pnl) + 0.01

    def test_win_flag_matches_net_pnl(self) -> None:
        runner = GoldBacktestRunner(_cfg())
        for t in runner.run(_make_df()):
            assert t.win == (t.net_pnl > 0)

    def test_hold_bars_non_negative(self) -> None:
        runner = GoldBacktestRunner(_cfg())
        for t in runner.run(_make_df()):
            assert t.hold_bars >= 0

    def test_empty_df_returns_empty(self) -> None:
        runner = GoldBacktestRunner(_cfg())
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        assert runner.run(df) == []

    def test_reproducible(self) -> None:
        df = _make_df(200)
        a = GoldBacktestRunner(_cfg()).run(df)
        b = GoldBacktestRunner(_cfg()).run(df)
        assert len(a) == len(b)
        for ta, tb in zip(a, b):
            assert ta.action == tb.action
            assert abs(ta.net_pnl - tb.net_pnl) < 0.001


# ── BacktestSummary ───────────────────────────────────────────────────────────

class TestBacktestSummary:
    def test_empty_results(self) -> None:
        runner = GoldBacktestRunner(_cfg())
        s = runner.summary([])
        assert s.total_trades == 0
        assert s.net_pnl == 0.0
        assert s.profit_factor is None

    def test_summary_totals_match(self) -> None:
        runner = GoldBacktestRunner(_cfg())
        trades = runner.run(_make_df(300, trend=1.0))
        if not trades:
            pytest.skip("No trades generated")
        s = runner.summary(trades)
        assert s.total_trades == len(trades)
        assert s.wins + s.losses == s.total_trades
        assert abs(s.net_pnl - sum(t.net_pnl for t in trades)) < 0.01

    def test_win_rate_between_0_and_100(self) -> None:
        runner = GoldBacktestRunner(_cfg())
        trades = runner.run(_make_df(300))
        if not trades:
            pytest.skip("No trades generated")
        s = runner.summary(trades)
        assert 0.0 <= s.win_rate_pct <= 100.0

    def test_profit_factor_none_when_no_losses(self) -> None:
        runner = GoldBacktestRunner(_cfg())
        # Manufacture results with only wins
        wins = [
            BacktestResult(
                date=pd.Timestamp("2026-01-15").date(),
                action="BUY", signal_type="VWAP_PB_LONG",
                contracts=1, entry_price=2350.0, exit_price=2360.0,
                stop_loss=2340.0, take_profit=2375.0, atr=5.0,
                regime="TRENDING_BULL", gross_pnl=100.0, commission=1.2,
                net_pnl=98.8, exit_reason="PROFIT_TARGET", hold_bars=10, win=True,
            )
        ]
        s = runner.summary(wins)
        assert s.profit_factor is None

    def test_by_signal_and_by_exit_populated(self) -> None:
        runner = GoldBacktestRunner(_cfg())
        trades = runner.run(_make_df(300))
        if not trades:
            pytest.skip("No trades generated")
        s = runner.summary(trades)
        assert len(s.by_signal) > 0
        assert len(s.by_exit) > 0
