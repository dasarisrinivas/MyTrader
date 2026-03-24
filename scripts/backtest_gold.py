#!/usr/bin/env python3
"""
Gold Futures Intraday Strategy — Bar-by-Bar Backtest Runner.

Loads historical 1-minute OHLCV data (CSV or IB download), runs
GoldIntradayStrategy bar-by-bar with no lookahead, simulates fills,
and prints a detailed performance report.

Usage:
    python3 scripts/backtest_gold.py data/gold_bars_2026.csv
    python3 scripts/backtest_gold.py data/gold_bars_2026.csv --config config.gold.yaml
    python3 scripts/backtest_gold.py data/gold_bars_2026.csv --symbol GC --allow-gc
    python3 scripts/backtest_gold.py data/gold_bars_2026.csv --start 2026-01-01 --end 2026-03-01

CSV format (required columns):
    datetime,open,high,low,close,volume
    2026-01-02 08:20:00-05:00,2650.0,2651.5,2648.8,2650.2,320

Exit simulation (conservative — assumes fills at SL/TP levels):
    - TP fill:   high (long) or low (short) touches take_profit price
    - SL fill:   low (long) or high (short) touches stop_loss price
    - When both are touched in the same bar, SL is assumed (worst case)
    - Time stop: bar count since entry >= exit.time_stop_bars
    - Session flatten: last bar of configured session date
"""
from __future__ import annotations

import argparse
import csv
import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from shree.config.gold import GoldStrategyConfig
from shree.execution.gold.risk import DailyState, GoldRiskManager
from shree.risk.trade_math import get_contract_spec
from shree.strategies.gold.signals import GoldSignalType
from shree.strategies.gold.strategy import GoldIntradayStrategy, compute_indicators
from shree.utils.settings_loader import load_settings


# ── Data loading ─────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV CSV into a timezone-aware DataFrame sorted by time."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "time"})
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("America/New_York")
    else:
        df["time"] = df["time"].dt.tz_convert("America/New_York")
    df = df.set_index("time").sort_index()
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def _iter_sessions(df: pd.DataFrame, open_et: str = "08:20") -> Iterator[pd.DataFrame]:
    """Yield one DataFrame per trading day (bars from session open onward)."""
    open_h, open_m = int(open_et.split(":")[0]), int(open_et.split(":")[1])
    for day, grp in df.groupby(df.index.date):
        session = grp[
            (grp.index.hour > open_h) |
            ((grp.index.hour == open_h) & (grp.index.minute >= open_m))
        ]
        if len(session) > 0:
            yield session


# ── Trade simulation ──────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    trade_id: int
    date: str
    action: str
    signal_type: str
    contracts: int
    entry_bar_idx: int
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_price: float = 0.0
    exit_bar_idx: int = 0
    exit_reason: str = ""
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    hold_bars: int = 0
    atr: float = 0.0
    regime: str = ""
    win: bool = False


@dataclass
class BacktestState:
    """Per-session (day) state reset each morning."""
    trades_today: int = 0
    consecutive_losses: int = 0
    realized_pnl_today: float = 0.0
    cooldown_until: Optional[datetime] = None
    in_position: bool = False
    open_trade: Optional[BacktestTrade] = None


class GoldBacktest:
    """Bar-by-bar backtest simulator.

    Builds a rolling window of `warmup_bars` completed bars, then on each
    new bar: computes indicators → strategy.generate_gold() → if signal and
    no open position → enter → scan forward for TP/SL/time-stop exit.
    """

    def __init__(self, cfg: GoldStrategyConfig) -> None:
        self._cfg = cfg
        self._spec = get_contract_spec(cfg.symbol)
        self._strategy = GoldIntradayStrategy(cfg)
        self._risk_mgr = GoldRiskManager(cfg.risk, self._spec)
        self._trades: List[BacktestTrade] = []
        self._trade_counter = 0

    def run(self, df: pd.DataFrame) -> List[BacktestTrade]:
        """Run the full backtest on a pre-loaded OHLCV DataFrame."""
        self._trades = []
        self._trade_counter = 0
        state = BacktestState()
        current_date: Optional[date] = None

        bars = list(df.iterrows())   # (timestamp, row) pairs

        for i, (ts, row) in enumerate(bars):
            bar_date = ts.date()

            # Day rollover
            if bar_date != current_date:
                if state.open_trade is not None:
                    # Close any open trade at end of prior session
                    self._close_trade(state.open_trade, float(bars[i - 1][1]["close"]),
                                      "FLATTEN_SESSION", i - 1, state)
                state = BacktestState()
                current_date = bar_date
                self._strategy.notify_loss()  # Doesn't reset, just re-uses instance

            # Manage open position
            if state.in_position and state.open_trade is not None:
                t = state.open_trade
                bar_high = float(row["high"])
                bar_low = float(row["low"])
                bars_held = i - t.entry_bar_idx

                # Time stop
                if (self._cfg.exit.time_stop_enabled and
                        bars_held >= self._cfg.exit.time_stop_bars):
                    self._close_trade(t, float(row["close"]), "TIME_STOP", i, state)
                    continue

                # Check TP/SL — SL wins when both touched (worst-case)
                if t.action == "BUY":
                    sl_hit = bar_low <= t.stop_loss
                    tp_hit = bar_high >= t.take_profit
                else:
                    sl_hit = bar_high >= t.stop_loss
                    tp_hit = bar_low <= t.take_profit

                if sl_hit:
                    self._close_trade(t, t.stop_loss, "STOP_LOSS", i, state)
                elif tp_hit:
                    self._close_trade(t, t.take_profit, "PROFIT_TARGET", i, state)
                continue

            # Attempt entry on this bar (bars[0..i] = completed bars up to i)
            window = df.iloc[: i + 1]
            if len(window) < self._cfg.indicators.warmup_bars:
                continue

            signal = self._strategy.generate_gold(window, bar_timestamp=ts)
            if not signal.is_actionable:
                continue

            stop_dist = abs(signal.entry_ref_price - signal.stop_loss)
            daily = DailyState(
                realized_pnl=state.realized_pnl_today,
                trades_today=state.trades_today,
                consecutive_losses=state.consecutive_losses,
                cooldown_until=state.cooldown_until,
            )
            sizing = self._risk_mgr.size_position(stop_dist, daily)
            if not sizing.approved:
                continue

            self._trade_counter += 1
            t = BacktestTrade(
                trade_id=self._trade_counter,
                date=bar_date.isoformat(),
                action=signal.action,
                signal_type=signal.signal_type.value,
                contracts=sizing.contracts,
                entry_bar_idx=i,
                entry_price=signal.entry_ref_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                atr=signal.atr,
                regime=signal.regime.value,
            )
            state.in_position = True
            state.open_trade = t

        # Close any remaining open trade
        if state.open_trade is not None and bars:
            self._close_trade(state.open_trade,
                              float(bars[-1][1]["close"]),
                              "FLATTEN_SESSION",
                              len(bars) - 1, state)

        return self._trades

    def _close_trade(
        self,
        t: BacktestTrade,
        exit_price: float,
        reason: str,
        exit_bar_idx: int,
        state: BacktestState,
    ) -> None:
        t.exit_price = exit_price
        t.exit_bar_idx = exit_bar_idx
        t.exit_reason = reason
        t.hold_bars = exit_bar_idx - t.entry_bar_idx

        if t.action == "BUY":
            gross = (exit_price - t.entry_price) * self._spec.point_value * t.contracts
        else:
            gross = (t.entry_price - exit_price) * self._spec.point_value * t.contracts

        commission = (self._spec.live_commission_per_side or 0.0) * 2 * t.contracts
        net = gross - commission
        t.gross_pnl = round(gross, 2)
        t.net_pnl = round(net, 2)
        t.win = net > 0

        state.realized_pnl_today += net
        state.trades_today += 1
        state.in_position = False
        state.open_trade = None

        if t.win:
            state.consecutive_losses = 0
        else:
            state.consecutive_losses += 1
            cooldown = self._risk_mgr.compute_cooldown_until(state.consecutive_losses)
            if cooldown:
                state.cooldown_until = cooldown
            # Notify strategy generator of loss (bar cooldown)
            self._strategy.notify_loss()

        self._trades.append(t)


# ── Report ────────────────────────────────────────────────────────────────────

def _print_report(trades: List[BacktestTrade], cfg: GoldStrategyConfig) -> None:
    if not trades:
        print("\n⚠️  No trades generated — check warmup_bars or date range.\n")
        return

    wins = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    net_pnl = sum(t.net_pnl for t in trades)
    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")
    wr = round(100 * len(wins) / len(trades), 1)
    avg_win = round(gross_win / len(wins), 2) if wins else 0
    avg_loss = round(gross_loss / len(losses), 2) if losses else 0
    avg_hold = round(sum(t.hold_bars for t in trades) / len(trades), 1)

    days = len({t.date for t in trades})

    print(f"\n{'='*65}")
    print(f"  GOLD BACKTEST — {cfg.symbol} on {cfg.exchange}")
    print(f"  Period: {trades[0].date} → {trades[-1].date}  ({days} trading days)")
    print(f"{'='*65}")
    print(f"\n  Trades:          {len(trades)}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Win Rate:        {wr:.1f}%")
    print(f"  Net P&L:         ${net_pnl:.2f}")
    print(f"  Profit Factor:   {pf:.2f}")
    print(f"  Avg Win:         ${avg_win:.2f}   Avg Loss: ${avg_loss:.2f}")
    print(f"  Avg Hold (bars): {avg_hold:.1f}")
    print(f"  Trades/day:      {len(trades) / days:.1f}")

    # Signal breakdown
    by_sig: dict = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
    for t in trades:
        by_sig[t.signal_type]["n"] += 1
        by_sig[t.signal_type]["w"] += int(t.win)
        by_sig[t.signal_type]["pnl"] += t.net_pnl
    print(f"\n  {'Signal':<22} {'N':>4} {'W':>4} {'WR%':>6} {'Net P&L':>10}")
    print(f"  {'─'*22} {'─'*4} {'─'*4} {'─'*6} {'─'*10}")
    for sig, d in sorted(by_sig.items(), key=lambda x: x[1]["pnl"], reverse=True):
        wr_s = round(100 * d["w"] / d["n"], 1) if d["n"] else 0
        print(f"  {sig:<22} {d['n']:>4} {d['w']:>4} {wr_s:>5.1f}% ${d['pnl']:>9.2f}")

    # Exit breakdown
    by_exit: dict = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for t in trades:
        by_exit[t.exit_reason]["n"] += 1
        by_exit[t.exit_reason]["pnl"] += t.net_pnl
    print(f"\n  {'Exit Reason':<22} {'N':>4} {'Net P&L':>10}")
    print(f"  {'─'*22} {'─'*4} {'─'*10}")
    for reason, d in sorted(by_exit.items(), key=lambda x: x[1]["n"], reverse=True):
        print(f"  {reason:<22} {d['n']:>4} ${d['pnl']:>9.2f}")

    # Daily equity curve (running P&L)
    daily_pnl: dict = defaultdict(float)
    for t in trades:
        daily_pnl[t.date] += t.net_pnl
    running = 0.0
    print(f"\n  {'Date':<12} {'Trades':>7} {'Day P&L':>10} {'Cumulative':>11}")
    print(f"  {'─'*12} {'─'*7} {'─'*10} {'─'*11}")
    day_trade_count: dict = defaultdict(int)
    for t in trades:
        day_trade_count[t.date] += 1
    for d in sorted(daily_pnl):
        running += daily_pnl[d]
        marker = "  ▲" if daily_pnl[d] > 0 else "  ▼"
        print(f"  {d:<12} {day_trade_count[d]:>7} ${daily_pnl[d]:>9.2f} ${running:>10.2f}{marker}")

    print(f"\n{'='*65}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gold Futures Backtest")
    p.add_argument("csv", help="Path to 1-min OHLCV CSV (datetime,open,high,low,close,volume)")
    p.add_argument("--config", default=None, help="YAML config file (uses gold section)")
    p.add_argument("--symbol", default=None, choices=["MGC", "GC"],
                   help="Override symbol (default: from config or MGC)")
    p.add_argument("--allow-gc", action="store_true",
                   help="Allow GC (full Gold, $100/pt) — required when --symbol GC")
    p.add_argument("--start", default=None, help="Filter start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="Filter end date YYYY-MM-DD")
    p.add_argument("--warmup", type=int, default=None,
                   help="Override warmup_bars (default: from config)")
    p.add_argument("--sl-mult", type=float, default=None,
                   help="Override ATR SL multiplier")
    p.add_argument("--tp-mult", type=float, default=None,
                   help="Override ATR TP multiplier")
    p.add_argument("--save-csv", default=None,
                   help="Write trade records to this CSV path")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    if args.config:
        settings = load_settings(args.config)
        cfg = settings.gold
    else:
        cfg = GoldStrategyConfig(enabled=True)

    if args.symbol:
        cfg.symbol = args.symbol
    if args.allow_gc:
        cfg.allow_gc = True
    if args.warmup:
        cfg.indicators.warmup_bars = args.warmup
    if args.sl_mult:
        cfg.exit.atr_sl_multiplier = args.sl_mult
    if args.tp_mult:
        cfg.exit.atr_tp_multiplier = args.tp_mult

    cfg.validate()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {args.csv} …")
    df = load_csv(args.csv)
    print(f"  {len(df)} bars, {df.index[0]} → {df.index[-1]}")

    if args.start:
        df = df[df.index.date >= date.fromisoformat(args.start)]
    if args.end:
        df = df[df.index.date <= date.fromisoformat(args.end)]

    if df.empty:
        print("No data in specified range.")
        sys.exit(1)

    print(f"  Running on {len(df)} bars "
          f"({df.index[0].date()} → {df.index[-1].date()}) …")

    # ── Run backtest ──────────────────────────────────────────────────────────
    bt = GoldBacktest(cfg)
    trades = bt.run(df)

    # ── Report ────────────────────────────────────────────────────────────────
    _print_report(trades, cfg)

    # ── Optional CSV export ───────────────────────────────────────────────────
    if args.save_csv and trades:
        import csv as csv_mod
        out_path = Path(args.save_csv)
        fields = [
            "trade_id", "date", "action", "signal_type", "contracts",
            "entry_price", "stop_loss", "take_profit",
            "exit_price", "exit_reason", "gross_pnl", "net_pnl",
            "hold_bars", "atr", "regime", "win",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for t in trades:
                writer.writerow({f: getattr(t, f) for f in fields})
        print(f"Trade records saved to {out_path}")


if __name__ == "__main__":
    main()
