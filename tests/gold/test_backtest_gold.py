"""Tests for the GoldBacktest engine in scripts/backtest_gold.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The backtest script lives in scripts/ (not a package), so add it to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from backtest_gold import GoldBacktest, load_csv  # noqa: E402

from shree.config.gold import GoldStrategyConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def _make_trending_df(n: int = 200, trend: float = 1.0) -> pd.DataFrame:
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


# ─────────────────────────────────────────────────────────────────────────────
# GoldBacktest unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGoldBacktest:
    def test_no_trades_before_warmup(self) -> None:
        cfg = _cfg(warmup=200)   # Warmup > data
        bt = GoldBacktest(cfg)
        df = _make_trending_df(100)
        trades = bt.run(df)
        assert len(trades) == 0

    def test_returns_list(self) -> None:
        bt = GoldBacktest(_cfg())
        df = _make_trending_df(200, trend=1.0)
        trades = bt.run(df)
        assert isinstance(trades, list)

    def test_all_trades_have_exit(self) -> None:
        bt = GoldBacktest(_cfg())
        df = _make_trending_df(200, trend=1.0)
        trades = bt.run(df)
        for t in trades:
            assert t.exit_price > 0
            assert t.exit_reason != ""

    def test_pnl_consistent(self) -> None:
        """gross_pnl = (exit - entry) × multiplier × contracts (long)."""
        bt = GoldBacktest(_cfg())
        df = _make_trending_df(200, trend=1.0)
        trades = bt.run(df)
        spec = bt._spec
        for t in trades:
            if t.action == "BUY":
                expected_gross = (t.exit_price - t.entry_price) * spec.point_value * t.contracts
            else:
                expected_gross = (t.entry_price - t.exit_price) * spec.point_value * t.contracts
            assert abs(t.gross_pnl - expected_gross) < 0.01

    def test_net_pnl_less_than_gross(self) -> None:
        bt = GoldBacktest(_cfg())
        df = _make_trending_df(200, trend=1.0)
        trades = bt.run(df)
        for t in trades:
            # Commission reduces net; losing trades may have |gross| > |net| (both negative)
            assert abs(t.net_pnl) <= abs(t.gross_pnl) + 0.01

    def test_win_flag_matches_net_pnl(self) -> None:
        bt = GoldBacktest(_cfg())
        df = _make_trending_df(200, trend=1.0)
        for t in bt.run(df):
            assert t.win == (t.net_pnl > 0)

    def test_hold_bars_positive(self) -> None:
        bt = GoldBacktest(_cfg())
        df = _make_trending_df(200, trend=1.0)
        for t in bt.run(df):
            assert t.hold_bars >= 0

    def test_daily_loss_limit_respected(self) -> None:
        """After hitting the daily loss limit, no more trades should fire that day."""
        cfg = _cfg()
        cfg.risk.daily_loss_limit_usd = 1.0   # Almost instant limit
        bt = GoldBacktest(cfg)
        df = _make_trending_df(200)
        trades = bt.run(df)
        # Each day can have at most a handful of trades before hitting the $1 limit
        from collections import Counter
        trades_per_day = Counter(t.date for t in trades)
        # If limit is $1, most days should have ≤ 2 trades
        for day, count in trades_per_day.items():
            assert count <= 5, f"Day {day} had {count} trades — daily limit not enforced"

    def test_reproducible_results(self) -> None:
        """Same data → same trade list."""
        df = _make_trending_df(200, trend=1.0)
        trades_a = GoldBacktest(_cfg()).run(df)
        trades_b = GoldBacktest(_cfg()).run(df)
        assert len(trades_a) == len(trades_b)
        for a, b in zip(trades_a, trades_b):
            assert a.action == b.action
            assert abs(a.net_pnl - b.net_pnl) < 0.001

    def test_exit_reasons_valid(self) -> None:
        valid = {"STOP_LOSS", "PROFIT_TARGET", "TIME_STOP", "FLATTEN_SESSION"}
        bt = GoldBacktest(_cfg())
        for t in bt.run(_make_trending_df(200)):
            assert t.exit_reason in valid, f"Unexpected exit reason: {t.exit_reason}"
