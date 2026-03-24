"""GoldBacktestRunner — reusable bar-by-bar Gold backtest in the shree library.

Wraps ``GoldIntradayStrategy`` and ``GoldRiskManager`` in a clean interface
that can be imported from notebooks, scripts, or other test harnesses without
adding ``scripts/`` to the Python path.

The standalone CLI lives in ``scripts/backtest_gold.py``; this module
provides the same engine as an importable class.

Typical usage::

    from shree.backtest.gold_runner import GoldBacktestRunner, BacktestResult
    from shree.config.gold import GoldStrategyConfig
    import pandas as pd

    cfg = GoldStrategyConfig(enabled=True, symbol="MGC")
    runner = GoldBacktestRunner(cfg)
    results = runner.run(df)          # df: timezone-aware OHLCV DataFrame
    summary = runner.summary(results)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from ..config.gold import GoldStrategyConfig
from ..execution.gold.risk import DailyState, GoldRiskManager, SizingResult
from ..risk.trade_math import get_contract_spec
from ..strategies.gold.strategy import GoldIntradayStrategy
from ..utils.logger import logger


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """One completed simulated trade."""

    date: date
    action: str                  # "BUY" | "SELL"
    signal_type: str
    contracts: int
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    atr: float
    regime: str
    gross_pnl: float             # (exit − entry) × point_value × contracts (signed)
    commission: float
    net_pnl: float
    exit_reason: str             # STOP_LOSS | PROFIT_TARGET | TIME_STOP | FLATTEN_SESSION
    hold_bars: int
    win: bool


@dataclass
class BacktestSummary:
    """Aggregate metrics over a list of BacktestResult objects."""

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate_pct: float = 0.0
    gross_pnl: float = 0.0
    total_commission: float = 0.0
    net_pnl: float = 0.0
    profit_factor: Optional[float] = None
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    by_signal: Dict[str, Dict] = field(default_factory=dict)
    by_exit: Dict[str, Dict] = field(default_factory=dict)


# ── Runner ────────────────────────────────────────────────────────────────────

class GoldBacktestRunner:
    """Bar-by-bar backtest engine for the Gold intraday strategy.

    Args:
        cfg: Gold strategy configuration.  ``enabled`` need not be True.

    Key simulation rules:
        - Rolling window fed to ``GoldIntradayStrategy.generate_gold()``.
        - SL wins when both TP and SL are touched in the same bar (conservative).
        - Commission is charged round-trip at ``ContractSpec.live_commission_per_side × 2``.
        - Positions are flattened at the close of each session's last bar
          (``FLATTEN_SESSION``).
        - Day state (daily loss limit, max trades) is reset each calendar date.
    """

    def __init__(self, cfg: GoldStrategyConfig) -> None:
        self._cfg = cfg
        self._strategy = GoldIntradayStrategy(cfg)
        self._risk = GoldRiskManager(
            cfg.risk,
            get_contract_spec(cfg.symbol),
            effective_max_risk=cfg.gc_adjusted_risk_usd(),
        )
        self._spec = get_contract_spec(cfg.symbol)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> List[BacktestResult]:
        """Replay every bar in *df* and return completed trades.

        Args:
            df: Timezone-aware OHLCV DataFrame with columns
                ``[open, high, low, close, volume]``, indexed by timestamp.
                Must be sorted in ascending time order.
        """
        if df.empty:
            return []

        results: List[BacktestResult] = []
        daily_state: DailyState = DailyState()
        current_date: Optional[date] = None

        # Open-position tracking
        position: Optional[_Position] = None

        bars = list(df.iterrows())
        for i, (ts, _row) in enumerate(bars):
            bar_date = ts.date() if hasattr(ts, "date") else ts.to_pydatetime().date()

            # ── Day rollover ──────────────────────────────────────────────────
            if bar_date != current_date:
                if position is not None:
                    # Flatten at end of previous session
                    close_price = float(df.iloc[i - 1]["close"])
                    result = self._close_position(position, close_price, "FLATTEN_SESSION", i - 1)
                    results.append(result)
                    position = None
                    self._strategy.notify_loss()  # Conservative: treat flatten as neutral
                daily_state = DailyState()
                current_date = bar_date

            window = df.iloc[: i + 1]

            # ── Manage open position ──────────────────────────────────────────
            if position is not None:
                row = df.iloc[i]
                exit_result = self._check_exits(position, row, i)
                if exit_result is not None:
                    results.append(exit_result)
                    if exit_result.exit_reason == "STOP_LOSS":
                        self._strategy.notify_loss()
                    position = None
                    # Update day state
                    daily_state.realized_pnl += exit_result.net_pnl
                    daily_state.trades_today += 1
                    if exit_result.net_pnl < 0:
                        daily_state.consecutive_loss_count += 1
                    else:
                        daily_state.consecutive_loss_count = 0
                continue  # No new entry while in a position

            # ── Attempt entry ─────────────────────────────────────────────────
            gold_signal = self._strategy.generate_gold(window, bar_timestamp=ts)
            if not gold_signal.is_actionable:
                continue

            # Risk sizing — stop distance (always positive)
            if gold_signal.action == "BUY":
                stop_dist = gold_signal.entry_ref_price - gold_signal.stop_loss
            else:
                stop_dist = gold_signal.stop_loss - gold_signal.entry_ref_price
            sizing = self._risk.size_position(
                stop_distance_points=stop_dist,
                daily=daily_state,
            )
            if not sizing.approved:
                logger.debug("GoldBacktestRunner: sizing rejected — {}", sizing.reason)
                continue

            position = _Position(
                action=gold_signal.action,
                signal_type=gold_signal.signal_type.value,
                contracts=sizing.contracts,
                entry_price=gold_signal.entry_ref_price,
                stop_loss=gold_signal.stop_loss,
                take_profit=gold_signal.take_profit,
                atr=gold_signal.atr,
                regime=gold_signal.regime.value,
                entry_bar=i,
                entry_date=bar_date,
            )

        # ── Flush any open position at end of data ────────────────────────────
        if position is not None and len(df) > 0:
            close_price = float(df.iloc[-1]["close"])
            results.append(
                self._close_position(position, close_price, "FLATTEN_SESSION", len(df) - 1)
            )

        return results

    def summary(self, results: List[BacktestResult]) -> BacktestSummary:
        """Compute aggregate metrics from a list of ``BacktestResult`` objects."""
        if not results:
            return BacktestSummary()

        wins = [r for r in results if r.win]
        losses = [r for r in results if not r.win]
        gross_wins = sum(r.gross_pnl for r in wins)
        gross_losses = abs(sum(r.gross_pnl for r in losses))

        # Max drawdown (equity curve)
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in results:
            equity += r.net_pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)

        # Signal breakdown
        by_signal: Dict[str, Dict] = {}
        for r in results:
            s = by_signal.setdefault(r.signal_type, {"total": 0, "wins": 0, "net_pnl": 0.0})
            s["total"] += 1
            s["wins"] += int(r.win)
            s["net_pnl"] += r.net_pnl

        # Exit breakdown
        by_exit: Dict[str, Dict] = {}
        for r in results:
            e = by_exit.setdefault(r.exit_reason, {"total": 0, "net_pnl": 0.0})
            e["total"] += 1
            e["net_pnl"] += r.net_pnl

        return BacktestSummary(
            total_trades=len(results),
            wins=len(wins),
            losses=len(losses),
            win_rate_pct=100.0 * len(wins) / len(results),
            gross_pnl=sum(r.gross_pnl for r in results),
            total_commission=sum(r.commission for r in results),
            net_pnl=sum(r.net_pnl for r in results),
            profit_factor=gross_wins / gross_losses if gross_losses > 0 else None,
            avg_win=sum(r.net_pnl for r in wins) / len(wins) if wins else 0.0,
            avg_loss=sum(r.net_pnl for r in losses) / len(losses) if losses else 0.0,
            max_drawdown=max_dd,
            by_signal=by_signal,
            by_exit=by_exit,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_exits(
        self,
        pos: "_Position",
        row: pd.Series,
        bar_idx: int,
    ) -> Optional[BacktestResult]:
        """Check TP/SL/time-stop against a completed bar.  Returns result or None."""
        high = float(row.get("high", row["close"]))
        low = float(row.get("low", row["close"]))
        close = float(row["close"])

        if pos.action == "BUY":
            tp_hit = high >= pos.take_profit
            sl_hit = low <= pos.stop_loss
        else:
            tp_hit = low <= pos.take_profit
            sl_hit = high >= pos.stop_loss

        # Worst-case: SL wins when both touch in the same bar
        if sl_hit:
            return self._close_position(pos, pos.stop_loss, "STOP_LOSS", bar_idx)
        if tp_hit:
            return self._close_position(pos, pos.take_profit, "PROFIT_TARGET", bar_idx)

        # Time stop
        hold = bar_idx - pos.entry_bar
        if (
            self._cfg.exit.time_stop_enabled
            and self._cfg.exit.time_stop_bars > 0
            and hold >= self._cfg.exit.time_stop_bars
        ):
            return self._close_position(pos, close, "TIME_STOP", bar_idx)

        return None

    def _close_position(
        self,
        pos: "_Position",
        exit_price: float,
        reason: str,
        bar_idx: int,
    ) -> BacktestResult:
        spec = self._spec
        if pos.action == "BUY":
            gross = (exit_price - pos.entry_price) * spec.point_value * pos.contracts
        else:
            gross = (pos.entry_price - exit_price) * spec.point_value * pos.contracts

        commission = spec.live_commission_per_side * 2 * pos.contracts
        net = gross - commission

        return BacktestResult(
            date=pos.entry_date,
            action=pos.action,
            signal_type=pos.signal_type,
            contracts=pos.contracts,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            atr=pos.atr,
            regime=pos.regime,
            gross_pnl=gross,
            commission=commission,
            net_pnl=net,
            exit_reason=reason,
            hold_bars=bar_idx - pos.entry_bar,
            win=net > 0,
        )


@dataclass
class _Position:
    """Internal open-position tracker."""
    action: str
    signal_type: str
    contracts: int
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    regime: str
    entry_bar: int
    entry_date: date
