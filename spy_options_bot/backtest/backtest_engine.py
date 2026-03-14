"""Backtest engine — replays the SPY options strategy against historical data.

Strategy mirrors the live bot exactly:
  - Entry: Mon/Tue/Wed, VIX 12–30, SPY vs SMA20 trend, SPY vs SMA200 regime, PDT compliance
  - Exit: 50% profit target, 2× loss stop, delta stop, Thursday EOD, emergency gamma
  - Position sizing: capped at 5% account risk per leg
  - Slippage: 1% of premium at entry + exit (2% on stressed stops/emergency fills)
  - Commission: $0.65 per contract per side

Filters added vs original:
  - Delta target lowered to 0.16 (further OTM, higher win rate)
  - Gamma protection: DTE ≥ 4 calendar days (blocks Wednesday entries)
  - Market regime: SPY must be above 200-day SMA to sell puts; below for calls
  - Realistic slippage model: percentage of premium, not fixed dollar

No lookahead bias:
  - Signal uses previous trading day's close and SMAs
  - VIX gate uses current day's open as proxy for 9:35 AM snapshot
  - Entry execution price is current day's open
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from backtest.options_simulator import (
    RISK_FREE_RATE,
    BSResult,
    bs_price,
    dte_years,
    find_strike_for_delta,
    vix_to_sigma,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMISSION_PER_CONTRACT: float = 0.65       # per side, per contract
ENTRY_SLIPPAGE_PCT: float = 0.01            # 1% of premium — sell slightly below mid
EXIT_SLIPPAGE_PCT: float = 0.01             # 1% — normal limit-order exit
EXIT_SLIPPAGE_STRESSED_PCT: float = 0.02   # 2% — market order on stops / emergency

MAX_ACCOUNT_RISK_PCT: float = 0.05
MAX_LOSS_MULTIPLE: float = 2.0
PROFIT_TARGET_PCT: float = 0.50
DELTA_STOP: float = 0.50
MIN_VIX: float = 12.0
MAX_VIX: float = 30.0
SMA_DAYS: int = 20
SMA200_DAYS: int = 200                      # Market regime filter
PDT_MAX_TRADES: int = 3
PDT_WINDOW_DAYS: int = 5
MAX_CONTRACTS: int = 1
TARGET_DELTA: float = 0.16                  # Further OTM → higher win rate

# Gamma protection: require at least 4 calendar days to expiry
# Blocks Wednesday entries (2 DTE) — only Mon (4 DTE) and Tue (3 DTE) allowed
MIN_DTE_CAL: int = 3   # Tuesday entry to this Friday (3 cal days)
MAX_DTE_CAL: int = 5   # Monday entry to this Friday

StrategyType = Literal["sell_put", "sell_call", "sell_strangle", "auto",
                        "puts_only", "calls_only", "strangles_only"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SimulatedTrade:
    trade_id: str
    entry_date: date
    expiry: date
    right: str           # 'P' or 'C'
    strike: float
    contracts: int
    entry_premium: float
    entry_delta: float
    entry_theta: float
    entry_iv: float
    entry_spy_price: float
    entry_vix: float
    close_date: date
    close_premium: float
    gross_pnl: float     # (entry_premium - close_premium) * 100 * contracts
    commissions: float   # entry + exit costs
    net_pnl: float       # gross_pnl - commissions
    exit_reason: str     # profit_target | loss_stop | delta_stop | thursday_eod | emergency_gamma
    days_held: int
    strategy: str        # sell_put | sell_call (per leg; strangle has 2 trades)


@dataclass
class NoTradeRecord:
    date: date
    reason: str


@dataclass
class BacktestResults:
    trades: list[SimulatedTrade]
    equity_curve: pd.DataFrame         # columns: date, equity, unrealized_pnl
    no_trade_records: list[NoTradeRecord]
    initial_capital: float
    start_date: date
    end_date: date
    pdt_blocked_weeks: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_friday(d: date) -> date:
    """Return nearest upcoming Friday (same week if d is Mon-Thu, else next week)."""
    days_ahead = 4 - d.weekday()  # Friday = weekday 4
    if days_ahead <= 0:
        days_ahead += 7
    return d + timedelta(days=days_ahead)


def _get_trading_days(df_index: pd.DatetimeIndex, start: date, end: date) -> list[date]:
    """Return all dates in df_index between start and end inclusive."""
    return [
        d.date() for d in df_index
        if start <= d.date() <= end
    ]


def _prev_trading_day(trading_days: list[date], ref: date) -> date | None:
    """Return the trading day before ref, or None."""
    before = [d for d in trading_days if d < ref]
    return before[-1] if before else None


def _pdt_slots_used(closed_dates: list[date], ref: date) -> int:
    """Count round-trips in the rolling 5-trading-day window ending at ref."""
    cutoff = ref - timedelta(days=14)  # rough lookback; will filter precisely below
    recent = [d for d in closed_dates if d >= cutoff and d <= ref]
    # Count only the last 5 CALENDAR occurrences of trading (approximate)
    # For simplicity, use a 7-calendar-day window (Mon-Fri = 5 trading days max)
    window_start = ref - timedelta(days=6)
    return sum(1 for d in closed_dates if window_start <= d <= ref)


def _pdt_slots_remaining(closed_dates: list[date], ref: date) -> int:
    return max(0, PDT_MAX_TRADES - _pdt_slots_used(closed_dates, ref))


def _calc_position_size(account_value: float, premium: float) -> int:
    """Max contracts limited by 5% account risk, hard-capped at MAX_CONTRACTS."""
    max_risk = account_value * MAX_ACCOUNT_RISK_PCT
    risk_per = premium * 100.0 * MAX_LOSS_MULTIPLE
    return min(MAX_CONTRACTS, max(1, int(max_risk / risk_per)))


def _trade_cost(
    contracts: int,
    entry_premium: float,
    close_premium: float,
    exit_reason: str,
) -> float:
    """Total round-trip cost: commission + realistic slippage.

    Slippage is a percentage of premium (not a fixed dollar amount) to
    reflect that wider-premium options have wider bid-ask spreads.
    Stressed exits (stops, emergency) use a higher slippage multiplier
    to simulate market-order fills in adverse conditions.
    """
    commission = COMMISSION_PER_CONTRACT * contracts * 2
    entry_slip = entry_premium * ENTRY_SLIPPAGE_PCT * 100 * contracts
    stressed = exit_reason in ("loss_stop", "delta_stop", "emergency_gamma")
    exit_pct = EXIT_SLIPPAGE_STRESSED_PCT if stressed else EXIT_SLIPPAGE_PCT
    exit_slip = close_premium * exit_pct * 100 * contracts
    return commission + entry_slip + exit_slip


def _check_emergency_gamma_daily(spy_row: pd.Series) -> bool:
    """Approximate Thursday emergency gamma check using daily high-low range.
    If (high - low) / open > 1.5%, treat as emergency gamma event.
    """
    if spy_row["open"] <= 0:
        return False
    return (spy_row["high"] - spy_row["low"]) / spy_row["open"] > 0.015


def _check_emergency_gamma_5min(spy_5min: pd.DataFrame, day: date) -> bool:
    """Check if any 5-min bar on `day` moved > 1.5% vs previous bar close."""
    day_bars = spy_5min[spy_5min.index.date == day]
    if len(day_bars) < 2:
        return False
    closes = day_bars["close"].values
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and abs(closes[i] - closes[i - 1]) / closes[i - 1] > 0.015:
            return True
    return False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Simulates 1 year of SPY weekly options selling against historical data."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    def run(
        self,
        start_date: date,
        end_date: date,
        initial_capital: float = 25_000.0,
        strategy: StrategyType = "auto",
        entry_days: tuple[int, ...] = (0, 1, 2),
        min_vix: float = MIN_VIX,
        max_vix: float = MAX_VIX,
    ) -> BacktestResults:
        """Execute the full backtest simulation.

        Args:
            start_date:       First date to consider for entries.
            end_date:         Last date in simulation.
            initial_capital:  Starting portfolio value.
            strategy:         'auto' | 'puts_only' | 'calls_only' | 'strangles_only'

        Returns:
            BacktestResults with full trade log and daily equity curve.
        """
        from backtest.data_downloader import load_spy_daily, load_spy_5min, load_vix_daily

        spy_df = load_spy_daily(self.cache_dir)
        vix_df = load_vix_daily(self.cache_dir)
        spy_5min = load_spy_5min(self.cache_dir)

        # Align DataFrames to backtest window (+SMA_DAYS lookback)
        lookback_start = start_date - timedelta(days=SMA_DAYS * 2)
        spy_df = spy_df[spy_df.index >= pd.Timestamp(lookback_start)]
        vix_df = vix_df[vix_df.index >= pd.Timestamp(lookback_start)]

        trading_days = _get_trading_days(spy_df.index, start_date, end_date)
        if not trading_days:
            raise ValueError(f"No trading days found between {start_date} and {end_date}")

        # Simulation state
        open_legs: list[_OpenLeg] = []
        closed_trades: list[SimulatedTrade] = []
        pdt_closed_dates: list[date] = []   # one entry per closed leg (for PDT counting)
        no_trade_records: list[NoTradeRecord] = []
        realized_pnl: float = 0.0
        equity_rows: list[dict] = []
        pdt_blocked_weeks: int = 0
        _pdt_blocked_this_week: bool = False
        _last_week_num: int = -1

        trade_counter: int = 0

        for day in tqdm(trading_days, desc="Simulating", unit="day"):
            ts = pd.Timestamp(day)
            if ts not in spy_df.index or ts not in vix_df.index:
                continue

            spy_row = spy_df.loc[ts]
            vix_row = vix_df.loc[ts]
            prev_day = _prev_trading_day(trading_days, day)

            # ----------------------------------------------------------------
            # 1. Mark-to-market existing positions and check exits
            # ----------------------------------------------------------------
            legs_to_close: list[tuple[_OpenLeg, str]] = []

            is_thursday = day.weekday() == 3
            emergency = False

            if open_legs and is_thursday:
                if spy_5min is not None:
                    emergency = _check_emergency_gamma_5min(spy_5min, day)
                else:
                    emergency = _check_emergency_gamma_daily(spy_row)

            for leg in open_legs:
                dte_remaining = (leg.expiry - day).days
                T = dte_years(dte_remaining)
                sigma = vix_to_sigma(vix_row["close"])
                bs = bs_price(spy_row["close"], leg.strike, T, RISK_FREE_RATE, sigma, leg.right)
                current_price = bs.price
                current_delta = bs.delta

                if emergency:
                    legs_to_close.append((leg, "emergency_gamma"))
                elif current_price <= leg.entry_premium * PROFIT_TARGET_PCT:
                    legs_to_close.append((leg, "profit_target"))
                elif current_price >= leg.entry_premium * MAX_LOSS_MULTIPLE:
                    legs_to_close.append((leg, "loss_stop"))
                elif abs(current_delta) >= DELTA_STOP:
                    legs_to_close.append((leg, "delta_stop"))
                elif is_thursday:
                    legs_to_close.append((leg, "thursday_eod"))

            for leg, reason in legs_to_close:
                dte_remaining = max((leg.expiry - day).days, 0)
                T = dte_years(dte_remaining)
                sigma = vix_to_sigma(vix_row["close"])
                bs = bs_price(spy_row["close"], leg.strike, T, RISK_FREE_RATE, sigma, leg.right)

                # Profit target closes at exactly the target price (realistic mid fill)
                if reason == "profit_target":
                    close_price = round(leg.entry_premium * PROFIT_TARGET_PCT, 4)
                else:
                    close_price = round(bs.price, 4)

                gross = (leg.entry_premium - close_price) * 100.0 * leg.contracts
                cost = _trade_cost(leg.contracts, leg.entry_premium, close_price, reason)
                net = gross - cost

                trade_counter += 1
                trade = SimulatedTrade(
                    trade_id=f"T{trade_counter:04d}",
                    entry_date=leg.entry_date,
                    expiry=leg.expiry,
                    right=leg.right,
                    strike=leg.strike,
                    contracts=leg.contracts,
                    entry_premium=leg.entry_premium,
                    entry_delta=leg.entry_delta,
                    entry_theta=leg.entry_theta,
                    entry_iv=leg.entry_iv,
                    entry_spy_price=leg.entry_spy_price,
                    entry_vix=leg.entry_vix,
                    close_date=day,
                    close_premium=round(close_price, 4),
                    gross_pnl=round(gross, 2),
                    commissions=round(cost, 2),
                    net_pnl=round(net, 2),
                    exit_reason=reason,
                    days_held=(day - leg.entry_date).days,
                    strategy=leg.strategy,
                )
                closed_trades.append(trade)
                pdt_closed_dates.append(day)
                realized_pnl += net
                open_legs.remove(leg)

            # ----------------------------------------------------------------
            # 2. Compute unrealized P&L for remaining open legs
            # ----------------------------------------------------------------
            unrealized = 0.0
            for leg in open_legs:
                dte_remaining = max((leg.expiry - day).days, 0)
                T = dte_years(dte_remaining)
                sigma = vix_to_sigma(vix_row["close"])
                bs = bs_price(spy_row["close"], leg.strike, T, RISK_FREE_RATE, sigma, leg.right)
                unrealized += (leg.entry_premium - bs.price) * 100.0 * leg.contracts

            equity = initial_capital + realized_pnl + unrealized
            equity_rows.append({"date": day, "equity": round(equity, 2),
                                 "unrealized_pnl": round(unrealized, 2)})

            # ----------------------------------------------------------------
            # 3. Consider new entry (Mon/Tue/Wed only, no open legs)
            # ----------------------------------------------------------------
            week_num = day.isocalendar()[1]
            if week_num != _last_week_num:
                _pdt_blocked_this_week = False
                _last_week_num = week_num

            if open_legs or day.weekday() not in entry_days:
                continue

            if prev_day is None:
                continue

            # PDT check
            slots = _pdt_slots_remaining(pdt_closed_dates, day)
            if slots <= 0:
                no_trade_records.append(NoTradeRecord(day, "pdt_limit"))
                if not _pdt_blocked_this_week:
                    pdt_blocked_weeks += 1
                    _pdt_blocked_this_week = True
                continue

            # VIX gate — use current day's open as proxy for 9:35 AM intraday snapshot
            # (matches live bot's reqMktData snapshot=True at cycle start)
            # Fall back to prior close only if today's bar is missing.
            curr_ts = pd.Timestamp(day)
            prev_ts = pd.Timestamp(prev_day)
            if curr_ts in vix_df.index:
                entry_vix = float(vix_df.loc[curr_ts, "open"])
            elif prev_ts in vix_df.index:
                entry_vix = float(vix_df.loc[prev_ts, "close"])
            else:
                no_trade_records.append(NoTradeRecord(day, "vix_data_missing"))
                continue
            if entry_vix < min_vix:
                no_trade_records.append(NoTradeRecord(day, f"vix_too_low_{entry_vix:.1f}"))
                continue
            if entry_vix > max_vix:
                no_trade_records.append(NoTradeRecord(day, f"vix_too_high_{entry_vix:.1f}"))
                continue

            # SMA20 + SMA200 through PREVIOUS day (no lookahead)
            spy_hist = spy_df[spy_df.index <= prev_ts]
            if len(spy_hist) < SMA_DAYS:
                no_trade_records.append(NoTradeRecord(day, "insufficient_sma_data"))
                continue
            sma20 = float(spy_hist["close"].iloc[-SMA_DAYS:].mean())
            sma200 = float(spy_hist["close"].iloc[-SMA200_DAYS:].mean()) \
                if len(spy_hist) >= SMA200_DAYS else None

            # Entry price: current day's open (available at 9:35 AM, no lookahead)
            spy_open = float(spy_row["open"])
            if spy_open <= 0:
                continue

            # Market regime filter: SPY vs 200-day SMA
            # Bear regime (SPY < SMA200) → restrict to calls only, no puts/strangles
            # Bull regime (SPY > SMA200) → full strategy as configured
            bear_regime = sma200 is not None and spy_open < sma200
            if bear_regime:
                regime_strategy = "calls_only"
            else:
                regime_strategy = strategy

            trend = "up" if spy_open > sma20 else ("down" if spy_open < sma20 else "neutral")

            # PDT-aware strategy selection (regime-adjusted)
            selected_strategy = _select_strategy(regime_strategy, trend, slots)
            if selected_strategy is None:
                no_trade_records.append(NoTradeRecord(day, f"no_strategy_trend={trend}_slots={slots}"))
                continue

            # Target expiry — nearest Friday
            target_expiry = _next_friday(day)
            dte_cal = (target_expiry - day).days
            if not (MIN_DTE_CAL <= dte_cal <= MAX_DTE_CAL):
                no_trade_records.append(NoTradeRecord(day, f"dte_out_of_range_{dte_cal}"))
                continue

            T = dte_years(dte_cal)
            sigma = vix_to_sigma(entry_vix)
            current_equity = initial_capital + realized_pnl

            # Build legs based on strategy
            new_legs: list[_OpenLeg] = []

            if selected_strategy in ("sell_put", "sell_strangle"):
                put_k = round(find_strike_for_delta(spy_open, T, RISK_FREE_RATE, sigma, -TARGET_DELTA, "P"))
                put_bs = bs_price(spy_open, put_k, T, RISK_FREE_RATE, sigma, "P")
                if put_bs.price > 0.05:  # Minimum premium sanity check
                    contracts = _calc_position_size(current_equity, put_bs.price)
                    new_legs.append(_OpenLeg(
                        entry_date=day, expiry=target_expiry, right="P",
                        strike=float(put_k), contracts=contracts,
                        entry_premium=put_bs.price, entry_delta=put_bs.delta,
                        entry_theta=put_bs.theta, entry_iv=sigma,
                        entry_spy_price=spy_open, entry_vix=entry_vix,
                        strategy="sell_put",
                    ))

            if selected_strategy in ("sell_call", "sell_strangle"):
                call_k = round(find_strike_for_delta(spy_open, T, RISK_FREE_RATE, sigma, TARGET_DELTA, "C"))
                call_bs = bs_price(spy_open, call_k, T, RISK_FREE_RATE, sigma, "C")
                if call_bs.price > 0.05:
                    contracts = _calc_position_size(current_equity, call_bs.price)
                    new_legs.append(_OpenLeg(
                        entry_date=day, expiry=target_expiry, right="C",
                        strike=float(call_k), contracts=contracts,
                        entry_premium=call_bs.price, entry_delta=call_bs.delta,
                        entry_theta=call_bs.theta, entry_iv=sigma,
                        entry_spy_price=spy_open, entry_vix=entry_vix,
                        strategy="sell_call",
                    ))

            if not new_legs:
                no_trade_records.append(NoTradeRecord(day, "no_valid_premium"))
                continue

            open_legs.extend(new_legs)

        # ----------------------------------------------------------------
        # Force-close any positions still open at end_date
        # ----------------------------------------------------------------
        last_day = trading_days[-1]
        if open_legs:
            last_ts = pd.Timestamp(last_day)
            spy_last = spy_df.loc[last_ts] if last_ts in spy_df.index else None
            vix_last = vix_df.loc[last_ts] if last_ts in vix_df.index else None
            for leg in list(open_legs):
                dte_remaining = max((leg.expiry - last_day).days, 0)
                T = dte_years(dte_remaining)
                sigma = vix_to_sigma(float(vix_last["close"])) if vix_last is not None else 0.20
                spy_close = float(spy_last["close"]) if spy_last is not None else leg.entry_spy_price
                bs = bs_price(spy_close, leg.strike, T, RISK_FREE_RATE, sigma, leg.right)
                close_price = round(bs.price, 4)
                gross = (leg.entry_premium - close_price) * 100.0 * leg.contracts
                cost = _trade_cost(leg.contracts, leg.entry_premium, close_price, "end_of_backtest")
                trade_counter += 1
                closed_trades.append(SimulatedTrade(
                    trade_id=f"T{trade_counter:04d}",
                    entry_date=leg.entry_date, expiry=leg.expiry,
                    right=leg.right, strike=leg.strike, contracts=leg.contracts,
                    entry_premium=leg.entry_premium, entry_delta=leg.entry_delta,
                    entry_theta=leg.entry_theta, entry_iv=leg.entry_iv,
                    entry_spy_price=leg.entry_spy_price, entry_vix=leg.entry_vix,
                    close_date=last_day, close_premium=round(close_price, 4),
                    gross_pnl=round(gross, 2), commissions=round(cost, 2),
                    net_pnl=round(gross - cost, 2),
                    exit_reason="end_of_backtest", days_held=(last_day - leg.entry_date).days,
                    strategy=leg.strategy,
                ))
            open_legs.clear()

        equity_df = pd.DataFrame(equity_rows)
        if not equity_df.empty:
            equity_df["date"] = pd.to_datetime(equity_df["date"])
            equity_df = equity_df.set_index("date").sort_index()

        return BacktestResults(
            trades=closed_trades,
            equity_curve=equity_df,
            no_trade_records=no_trade_records,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            pdt_blocked_weeks=pdt_blocked_weeks,
        )


# ---------------------------------------------------------------------------
# Internal state for open positions
# ---------------------------------------------------------------------------

@dataclass
class _OpenLeg:
    entry_date: date
    expiry: date
    right: str
    strike: float
    contracts: int
    entry_premium: float
    entry_delta: float
    entry_theta: float
    entry_iv: float
    entry_spy_price: float
    entry_vix: float
    strategy: str


def _select_strategy(
    mode: StrategyType,
    trend: str,
    slots: int,
) -> str | None:
    """Map mode + trend + PDT slots to a concrete single-leg strategy string."""
    if mode == "puts_only":
        return "sell_put"
    if mode == "calls_only":
        return "sell_call"
    if mode == "strangles_only":
        return "sell_strangle" if slots >= 2 else None

    # Auto mode: trend-following with graceful strangle degradation
    if trend == "up":
        return "sell_put"
    if trend == "down":
        return "sell_call"
    # Neutral
    if slots >= 2:
        return "sell_strangle"
    return "sell_put"  # degrade strangle to single when 1 slot left
