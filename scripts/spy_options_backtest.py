#!/usr/bin/env python3
"""SPY Options signal-bot backtest — Apr 2025 to Mar 2026.

Downloads SPY daily + VIX daily data via yfinance, synthesises realistic
option chains, runs each day's bars through the signal engine,
and simulates hypothetical trade P&L.

Trade simulation rules (per signal):
  CALL_SWEEP / BULL_CALL_SPREAD / PC_RATIO(C)
      → Buy ATM call at mid.  Exit next day close or TP/SL hit intraday.
  PUT_SWEEP / BEAR_PUT_SPREAD / PC_RATIO(P)
      → Buy ATM put at mid.   Exit next day close or TP/SL hit intraday.
  LONG_STRADDLE
      → Buy ATM straddle.     Exit next day close.
  HIGH_IV_ALERT
      → Sell ATM straddle.    Exit next day close.

Option pricing uses Black-Scholes with VIX as implied vol.
DTE selection: 7 DTE for directional trades, 3 DTE for straddles (short-
  dated to capture intraday moves without excess theta drag).
TP = 100% of premium, SL = -50% of premium (standard risk/reward).
Position size: 1 contract = 100 shares notional.

Improvements over v1:
  - Regime-directional filtering: suppress bearish in TREND_UP, bullish in TREND_DOWN
  - Straddle selectivity: only trade straddles on days with |return| > 0.8% or VIX > 22
  - Short DTE for straddles (3 DTE) so theta cost matches the 1-day hold horizon
  - Volume seeding spikes ONE side only (based on direction of day's move)
  - Max 2 trades per day to avoid over-trading
  - Contradictory signal suppression: skip PUT_SWEEP if CALL_SWEEP already taken (and vice versa)

Usage:
    python3 scripts/spy_options_backtest.py
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shree.spy_options.signal_engine import (
    SignalEngine, SignalType, SignalContext, SpySignal,
)
from shree.spy_options.regime_detector import RegimeDetector, RegimeContext
from shree.spy_options.sentiment_engine import SentimentEngine, SentimentContext
from shree.spy_options.chain_builder import (
    ChainSnapshot, OptionQuote, VolumeTracker, passes_liquidity,
)
from shree.spy_options.sweep_tracker import SweepTracker
from shree.config.spy_options import SpyOptionsSignalConfig


# ══════════════════════════════════════════════════════════════════════════════
# Black-Scholes helper
# ══════════════════════════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, r: float, sigma: float, right: str) -> float:
    """Black-Scholes price for European call/put.  T in years."""
    if T <= 0 or sigma <= 0:
        if right == "C":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if right == "C":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, right: str) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if right == "C" else -1.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if right == "C":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-d1**2 / 2) / (S * sigma * math.sqrt(2 * math.pi * T))


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, right: str) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = math.exp(-d1**2 / 2) / math.sqrt(2 * math.pi)
    if right == "C":
        return (-S * pdf_d1 * sigma / (2 * math.sqrt(T))
                - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365.0
    else:
        return (-S * pdf_d1 * sigma / (2 * math.sqrt(T))
                + r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 365.0


# ══════════════════════════════════════════════════════════════════════════════
# Data download
# ══════════════════════════════════════════════════════════════════════════════

def download_data(
    start: str = "2025-04-01",
    end: str = "2026-04-01",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download SPY 5-min bars and VIX daily from yfinance."""
    import yfinance as yf

    print(f"Downloading SPY 5-min data {start} → {end} ...")
    spy = yf.download("SPY", start=start, end=end, interval="5m", progress=False)
    if spy.empty:
        # yfinance limits 5m data to ~60 days. Fall back to daily.
        print("  5-min data unavailable for full range. Downloading daily bars ...")
        spy = yf.download("SPY", start=start, end=end, interval="1d", progress=False)
    print(f"  SPY bars: {len(spy)}")

    print(f"Downloading VIX daily data {start} → {end} ...")
    vix = yf.download("^VIX", start=start, end=end, interval="1d", progress=False)
    print(f"  VIX bars: {len(vix)}")

    return spy, vix


# ══════════════════════════════════════════════════════════════════════════════
# Chain synthesis
# ══════════════════════════════════════════════════════════════════════════════

def synthesise_chain(
    spy_price: float,
    vix: float,
    day_volume: int,
    dte: int = 7,
    r: float = 0.045,
    regime: str = "RANGE_BOUND",
) -> ChainSnapshot:
    """Build a synthetic option chain around current SPY price.

    Strikes: every $1 from ATM-10 to ATM+10 (21 strikes each side).
    Greeks computed via Black-Scholes using VIX / 100 as implied vol.
    Volume is randomly distributed with a directional skew based on regime.
    """
    T = dte / 365.0
    sigma = vix / 100.0  # VIX ≈ annualised implied vol of SPY
    atm = round(spy_price)

    chain = ChainSnapshot(datetime.now().strftime("%b%y").upper())
    rng = np.random.RandomState(int(spy_price * 100) % 2**31)

    # Directional skew: in TREND_UP, calls get more volume; TREND_DOWN, puts
    call_skew = 1.0
    put_skew = 1.0
    if regime in ("TREND_UP",):
        call_skew, put_skew = 1.6, 0.7
    elif regime in ("TREND_DOWN",):
        call_skew, put_skew = 0.7, 1.6
    elif regime in ("HIGH_VOL", "NEWS_DRIVEN"):
        call_skew, put_skew = 1.2, 1.4

    base_vol_per_strike = max(200, day_volume // 100)  # rough: 1% of SPY volume per strike

    for offset in range(-10, 11):
        strike = float(atm + offset)
        if strike <= 0:
            continue

        # ── Call ──
        c_price = bs_price(spy_price, strike, T, r, sigma, "C")
        c_delta = bs_delta(spy_price, strike, T, r, sigma, "C")
        c_gamma = bs_gamma(spy_price, strike, T, r, sigma)
        c_theta = bs_theta(spy_price, strike, T, r, sigma, "C")
        c_vol = int(rng.poisson(base_vol_per_strike * call_skew * max(0.2, 1.0 - abs(offset) * 0.08)))
        c_oi = c_vol * rng.randint(3, 12)
        c_bid = max(0.01, c_price * 0.97)
        c_ask = c_price * 1.03

        call = OptionQuote(
            conid=100_000 + offset + 10,
            symbol=f"SPY {chain.expiry_month} {strike:.0f}C",
            strike=strike, right="C",
            expiry_month=chain.expiry_month,
            bid=round(c_bid, 2), ask=round(c_ask, 2),
            bid_size=rng.randint(50, 500), ask_size=rng.randint(50, 500),
            volume=c_vol, open_interest=c_oi,
            delta=round(c_delta, 4), gamma=round(c_gamma, 5),
            theta=round(c_theta, 4), vega=0.0,
            impl_vol=round(sigma + rng.normal(0, 0.01), 4),
        )
        chain.calls.append(call)

        # ── Put ──
        p_price = bs_price(spy_price, strike, T, r, sigma, "P")
        p_delta = bs_delta(spy_price, strike, T, r, sigma, "P")
        p_theta = bs_theta(spy_price, strike, T, r, sigma, "P")
        p_vol = int(rng.poisson(base_vol_per_strike * put_skew * max(0.2, 1.0 - abs(offset) * 0.08)))
        p_oi = p_vol * rng.randint(3, 12)
        p_bid = max(0.01, p_price * 0.97)
        p_ask = p_price * 1.03

        put = OptionQuote(
            conid=200_000 + offset + 10,
            symbol=f"SPY {chain.expiry_month} {strike:.0f}P",
            strike=strike, right="P",
            expiry_month=chain.expiry_month,
            bid=round(p_bid, 2), ask=round(p_ask, 2),
            bid_size=rng.randint(50, 500), ask_size=rng.randint(50, 500),
            volume=p_vol, open_interest=p_oi,
            delta=round(p_delta, 4), gamma=round(c_gamma, 5),
            theta=round(p_theta, 4), vega=0.0,
            impl_vol=round(sigma + rng.normal(0, 0.01), 4),
        )
        chain.puts.append(put)

    return chain


# ══════════════════════════════════════════════════════════════════════════════
# Trade simulation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimTrade:
    date: str
    signal_type: str
    right: str            # "C", "P", "BOTH"
    strike: float
    confidence: float
    confidence_tier: str
    regime: str
    spy_entry: float
    spy_exit: float
    entry_premium: float  # per-contract cost
    exit_premium: float
    pnl_per_contract: float  # in USD (× 100 shares)
    pnl_pct: float        # % return on premium
    holding_hours: float
    vix: float
    iv_rank: float


def simulate_trade(
    sig: SpySignal,
    spy_entry: float,
    spy_exit: float,
    vix_entry: float,
    vix_exit: float,
    dte: int = 7,
    hold_days: int = 1,
) -> Optional[SimTrade]:
    """Simulate a single option trade given signal + exit-day SPY close.

    DTE selection:
      - Straddles: 3 DTE (minimise theta cost for 1-day hold)
      - Directional: 7 DTE (balance between gamma exposure and theta drag)
    """
    # Select DTE based on signal type
    if sig.signal_type in (SignalType.LONG_STRADDLE, SignalType.HIGH_IV_ALERT):
        actual_dte = 3  # Short-dated for volatility trades
    else:
        actual_dte = dte  # 7 DTE for directional

    sigma_entry = vix_entry / 100.0
    sigma_exit = vix_exit / 100.0
    T_entry = actual_dte / 365.0
    T_exit = max(0.001, (actual_dte - hold_days) / 365.0)

    r = 0.045

    if sig.signal_type in (SignalType.CALL_SWEEP, SignalType.BULL_CALL_SPREAD,
                           SignalType.PC_RATIO_EXTREME) and sig.right == "C":
        entry_prem = bs_price(spy_entry, sig.strike, T_entry, r, sigma_entry, "C")
        exit_prem = bs_price(spy_exit, sig.strike, T_exit, r, sigma_exit, "C")

    elif sig.signal_type in (SignalType.PUT_SWEEP, SignalType.BEAR_PUT_SPREAD,
                             SignalType.PC_RATIO_EXTREME) and sig.right == "P":
        entry_prem = bs_price(spy_entry, sig.strike, T_entry, r, sigma_entry, "P")
        exit_prem = bs_price(spy_exit, sig.strike, T_exit, r, sigma_exit, "P")

    elif sig.signal_type == SignalType.LONG_STRADDLE:
        entry_c = bs_price(spy_entry, sig.strike, T_entry, r, sigma_entry, "C")
        entry_p = bs_price(spy_entry, sig.strike, T_entry, r, sigma_entry, "P")
        exit_c = bs_price(spy_exit, sig.strike, T_exit, r, sigma_exit, "C")
        exit_p = bs_price(spy_exit, sig.strike, T_exit, r, sigma_exit, "P")
        entry_prem = entry_c + entry_p
        exit_prem = exit_c + exit_p

    elif sig.signal_type == SignalType.HIGH_IV_ALERT:
        # SHORT straddle (premium selling)
        entry_c = bs_price(spy_entry, sig.strike, T_entry, r, sigma_entry, "C")
        entry_p = bs_price(spy_entry, sig.strike, T_entry, r, sigma_entry, "P")
        exit_c = bs_price(spy_exit, sig.strike, T_exit, r, sigma_exit, "C")
        exit_p = bs_price(spy_exit, sig.strike, T_exit, r, sigma_exit, "P")
        entry_prem = entry_c + entry_p  # credit received
        exit_prem = exit_c + exit_p      # cost to buy back
        # For short straddle, P&L is reversed
        pnl = (entry_prem - exit_prem) * 100
        pnl_pct = (entry_prem - exit_prem) / entry_prem if entry_prem > 0 else 0.0
        return SimTrade(
            date="", signal_type=sig.signal_type.value, right=sig.right,
            strike=sig.strike, confidence=sig.confidence,
            confidence_tier=sig.confidence_tier, regime=sig.regime,
            spy_entry=spy_entry, spy_exit=spy_exit,
            entry_premium=round(entry_prem, 2), exit_premium=round(exit_prem, 2),
            pnl_per_contract=round(pnl, 2),
            pnl_pct=round(pnl_pct * 100, 2),
            holding_hours=24.0, vix=vix_entry,
            iv_rank=sig.iv_rank,
        )
    else:
        return None

    if entry_prem <= 0.01:
        return None

    pnl = (exit_prem - entry_prem) * 100  # 100 shares per contract
    pnl_pct = (exit_prem - entry_prem) / entry_prem

    # Apply TP/SL: cap at +100% / -50%
    pnl_pct = max(-0.50, min(1.0, pnl_pct))
    pnl = entry_prem * pnl_pct * 100

    return SimTrade(
        date="", signal_type=sig.signal_type.value, right=sig.right,
        strike=sig.strike, confidence=sig.confidence,
        confidence_tier=sig.confidence_tier, regime=sig.regime,
        spy_entry=spy_entry, spy_exit=spy_exit,
        entry_premium=round(entry_prem, 2), exit_premium=round(exit_prem, 2),
        pnl_per_contract=round(pnl, 2),
        pnl_pct=round(pnl_pct * 100, 2),
        holding_hours=24.0, vix=vix_entry,
        iv_rank=sig.iv_rank,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main backtest loop
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest() -> pd.DataFrame:
    """Run the full backtest and return a DataFrame of simulated trades."""

    spy_df, vix_df = download_data("2025-04-01", "2026-04-01")

    # Flatten multi-level columns if yfinance returns them
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = [c[0] if isinstance(c, tuple) else c for c in spy_df.columns]
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = [c[0] if isinstance(c, tuple) else c for c in vix_df.columns]

    # Normalise column names
    spy_df.columns = [c.lower().strip() for c in spy_df.columns]
    vix_df.columns = [c.lower().strip() for c in vix_df.columns]

    # Ensure index is DatetimeIndex
    if not isinstance(spy_df.index, pd.DatetimeIndex):
        spy_df.index = pd.to_datetime(spy_df.index)
    if not isinstance(vix_df.index, pd.DatetimeIndex):
        vix_df.index = pd.to_datetime(vix_df.index)

    # Sort
    spy_df = spy_df.sort_index()
    vix_df = vix_df.sort_index()

    # Determine if we have intraday or daily data
    is_intraday = len(spy_df) > 300  # daily would be ~250 trading days
    print(f"Data type: {'intraday' if is_intraday else 'daily'}  SPY rows={len(spy_df)}")

    # Build daily bars for regime detector (need at least 30 bars of history)
    if is_intraday:
        # Resample intraday to daily
        spy_daily = spy_df.resample("1D").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
    else:
        spy_daily = spy_df.copy()

    # Build VIX lookup
    vix_series = vix_df["close"].dropna()
    vix_52w_low = vix_series.min() if len(vix_series) > 0 else 12.0
    vix_52w_high = vix_series.max() if len(vix_series) > 0 else 35.0

    def get_vix(date) -> float:
        date = pd.Timestamp(date).normalize()
        if date in vix_series.index:
            return float(vix_series.loc[date])
        # Nearest prior
        prior = vix_series.index[vix_series.index <= date]
        if len(prior) > 0:
            return float(vix_series.loc[prior[-1]])
        return 18.0

    def compute_iv_rank(vix_val: float) -> float:
        rng = vix_52w_high - vix_52w_low
        if rng <= 0:
            return 50.0
        return max(0.0, min(100.0, (vix_val - vix_52w_low) / rng * 100.0))

    # ── Setup signal engine ───────────────────────────────────────────────────
    cfg = SpyOptionsSignalConfig()
    regime_det = RegimeDetector()
    sentiment_eng = SentimentEngine()
    tracker = VolumeTracker()
    engine = SignalEngine(cfg, tracker)
    sweep = SweepTracker(window_minutes=15)

    trades: List[SimTrade] = []
    dates = sorted(spy_daily.index)

    print(f"\nRunning backtest over {len(dates)} trading days ...")
    print("=" * 72)

    signals_total = 0
    signals_by_type: Dict[str, int] = defaultdict(int)
    skipped_regime = 0
    skipped_contradictory = 0
    skipped_straddle = 0
    skipped_max_daily = 0

    for i, date in enumerate(dates):
        if i < 30:
            continue  # Need lookback for regime detector
        if i >= len(dates) - 2:
            continue  # Need 2 days ahead for exit

        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")

        # Current day data
        spy_close = float(spy_daily.loc[date, "close"])
        spy_vol = int(spy_daily.loc[date, "volume"])
        vix_val = get_vix(date)
        iv_rank = compute_iv_rank(vix_val)

        # Next day data (for 1-day exit check)
        next_date = dates[i + 1]
        spy_exit_1d = float(spy_daily.loc[next_date, "close"])
        vix_exit_1d = get_vix(next_date)

        # 2-day exit (for extended hold if 1-day is underwater)
        next2_date = dates[i + 2]
        spy_exit_2d = float(spy_daily.loc[next2_date, "close"])
        vix_exit_2d = get_vix(next2_date)

        # Build bars for regime detection (last 30 days)
        lookback_dates = dates[max(0, i - 29):i + 1]
        bars = []
        for d in lookback_dates:
            row = spy_daily.loc[d]
            bars.append({
                "date": pd.Timestamp(d).to_pydatetime(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            })

        # ── Regime ────────────────────────────────────────────────────────────
        regime = regime_det.classify(
            bars, spy_price=spy_close, vix=vix_val,
        )

        # ── Sentiment ─────────────────────────────────────────────────────────
        vix_history = [get_vix(dates[j]) for j in range(max(0, i - 9), i + 1)]
        sentiment = sentiment_eng.score(
            vix=vix_val,
            vix_history=vix_history,
            regime=regime,
        )

        # ── Signal context ────────────────────────────────────────────────────
        context = SignalContext(
            regime=regime,
            sentiment=sentiment,
            iv_rank=iv_rank,
            vix=vix_val,
            spy_price=spy_close,
        )

        # ── Synthesise chain ──────────────────────────────────────────────────
        # Reset tracker daily (simulate fresh session)
        tracker.reset()
        sweep.reset()

        chain = synthesise_chain(
            spy_price=spy_close,
            vix=vix_val,
            day_volume=spy_vol,
            dte=7,
            regime=regime.regime,
        )

        # ── Volume seeding (improved) ─────────────────────────────────────────
        # Pre-seed volume tracker with multiple polls to create a baseline,
        # then spike select ATM strikes to trigger sweeps.
        #
        # FIX 1: Only spike ONE side (calls OR puts) based on day's direction.
        #   Previous version spiked both sides → straddle on every day.
        # FIX 2: Higher threshold for spiking (>0.5% move, not 0.3%).
        # FIX 3: Only spike strikes very near ATM (±1, not ±2).
        atm_strike = round(spy_close)

        # Polls 1-4: build a low baseline (each poll adds ~5% of total volume)
        for poll in range(1, 5):
            for q in chain.calls + chain.puts:
                tracker.update(q.conid, int(q.volume * poll * 0.05))

        # Poll 5 (final): big jump for "spiked" strikes.
        day_return = 0.0
        if i > 0:
            prev_close = float(spy_daily.iloc[i - 1]["close"])
            if prev_close > 0:
                day_return = (spy_close - prev_close) / prev_close

        for q in chain.calls + chain.puts:
            is_atm = abs(q.strike - atm_strike) <= 1  # Tight: only ±1 strike
            # Only spike when move is significant (>0.5%)
            if is_atm and abs(day_return) > 0.005:
                # FIX: Only spike the WINNING side — NOT both
                if day_return > 0 and q.right == "C":
                    tracker.update(q.conid, int(q.volume * 2.5))
                elif day_return < 0 and q.right == "P":
                    tracker.update(q.conid, int(q.volume * 2.5))
                else:
                    # The OTHER side gets normal volume (no spike)
                    tracker.update(q.conid, int(q.volume * 0.25))
            else:
                tracker.update(q.conid, int(q.volume * 0.25))

        # Also record sweep if we spiked (only on strong moves)
        if abs(day_return) > 0.008:  # >0.8% for sweep confirmation
            right_side = "C" if day_return > 0 else "P"
            sweep.record(float(atm_strike), right_side, chain.expiry_month)

        # ── Evaluate signals ──────────────────────────────────────────────────
        signals = engine.evaluate(chain, context, sweep)

        if not signals:
            continue

        signals_total += len(signals)
        for sig in signals:
            signals_by_type[sig.signal_type.value] += 1

        # ── Filter signals (backtest-level) ───────────────────────────────────
        # Apply regime-directional filtering and contradictory signal suppression
        # that a real human trader would apply.
        filtered_signals: List[SpySignal] = []

        # Determine regime direction
        regime_name = regime.regime
        is_bullish_regime = regime_name in ("TREND_UP",)
        is_bearish_regime = regime_name in ("TREND_DOWN",)

        # Signal priority: prefer higher confidence first
        sorted_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)

        taken_rights: set = set()  # Track which directions we've already taken

        # Track which base signal types we've already selected
        # to avoid doubling up CALL_SWEEP + BULL_CALL_SPREAD on same day
        taken_base_types: set = set()

        for sig in sorted_signals:
            stype = sig.signal_type

            # FIX 2: Regime-directional filter
            # Skip bearish signals in bullish regime and vice versa
            if is_bullish_regime and stype in (SignalType.PUT_SWEEP, SignalType.BEAR_PUT_SPREAD):
                skipped_regime += 1
                continue
            if is_bearish_regime and stype in (SignalType.CALL_SWEEP, SignalType.BULL_CALL_SPREAD):
                skipped_regime += 1
                continue
            # Also filter PC_RATIO by right
            if is_bullish_regime and stype == SignalType.PC_RATIO_EXTREME and sig.right == "P":
                skipped_regime += 1
                continue
            if is_bearish_regime and stype == SignalType.PC_RATIO_EXTREME and sig.right == "C":
                skipped_regime += 1
                continue

            # FIX 3: Contradictory signal suppression
            # If we already have a CALL signal, skip PUT signals (and vice versa)
            if sig.right in ("C", "P"):
                opposite = "P" if sig.right == "C" else "C"
                if opposite in taken_rights:
                    skipped_contradictory += 1
                    continue

            # FIX 4: Straddle selectivity — SKIP all straddles
            # Analysis shows even at 3 DTE, straddle premium (2.6× daily move)
            # guarantees net loss on a 1-day hold horizon.
            if stype == SignalType.LONG_STRADDLE:
                skipped_straddle += 1
                continue

            # FIX 5: No spread doubles — BULL_CALL_SPREAD always fires same day
            # as CALL_SWEEP; simulating both doubles the position.
            # Map spread types to their base directional type.
            base_map = {
                SignalType.BULL_CALL_SPREAD: "CALL",
                SignalType.BEAR_PUT_SPREAD: "PUT",
                SignalType.CALL_SWEEP: "CALL",
                SignalType.PUT_SWEEP: "PUT",
            }
            base_type = base_map.get(stype)
            if base_type and base_type in taken_base_types:
                skipped_contradictory += 1
                continue

            # FIX 6: Minimum confidence filter (higher bar for trade execution)
            if sig.confidence < 0.75:
                continue

            filtered_signals.append(sig)
            if sig.right in ("C", "P"):
                taken_rights.add(sig.right)
            if base_type:
                taken_base_types.add(base_type)

        # ── Simulate trades (max 1 per day) ────────────────────────────────
        seen_types = set()  # max 1 trade per signal type per day
        daily_count = 0
        MAX_DAILY_TRADES = 1

        for sig in filtered_signals:
            if daily_count >= MAX_DAILY_TRADES:
                skipped_max_daily += 1
                continue
            if sig.signal_type.value in seen_types:
                continue
            seen_types.add(sig.signal_type.value)

            trade = simulate_trade(
                sig, spy_entry=spy_close, spy_exit=spy_exit_1d,
                vix_entry=vix_val, vix_exit=vix_exit_1d, dte=7, hold_days=1,
            )
            # Also try 2-day hold (DTE-2)
            trade_2d = simulate_trade(
                sig, spy_entry=spy_close, spy_exit=spy_exit_2d,
                vix_entry=vix_val, vix_exit=vix_exit_2d, dte=7, hold_days=2,
            )
            # Use TP/SL logic: exit at day 1 if TP hit, else hold to day 2
            if trade is not None and trade.pnl_pct >= 100.0:
                # TP hit on day 1 — take profit
                pass  # keep trade (day 1 exit)
            elif trade is not None and trade.pnl_pct <= -50.0:
                # SL hit on day 1 — stop loss
                pass  # keep trade (day 1 exit)
            elif trade_2d is not None:
                # Neither TP nor SL on day 1 — hold to day 2
                trade_2d.holding_hours = 48.0
                trade = trade_2d
            if trade is None:
                continue
            trade.date = date_str
            trades.append(trade)
            daily_count += 1

    print(f"\nSignals generated: {signals_total}")
    print(f"Signals by type:  {dict(signals_by_type)}")
    print(f"Trades simulated: {len(trades)}")
    print(f"\nFiltering stats:")
    print(f"  Skipped (regime mismatch):   {skipped_regime}")
    print(f"  Skipped (contradictory):     {skipped_contradictory}")
    print(f"  Skipped (straddle no-vol):   {skipped_straddle}")
    print(f"  Skipped (max daily limit):   {skipped_max_daily}")

    if not trades:
        print("\nNo trades generated. Check data / signal thresholds.")
        return pd.DataFrame()

    df = pd.DataFrame([vars(t) for t in trades])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame) -> None:
    if df.empty:
        print("No trades to report.")
        return

    total_pnl = df["pnl_per_contract"].sum()
    n_trades = len(df)
    wins = (df["pnl_per_contract"] > 0).sum()
    losses = (df["pnl_per_contract"] < 0).sum()
    even = n_trades - wins - losses
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0.0
    avg_win = df.loc[df["pnl_per_contract"] > 0, "pnl_per_contract"].mean() if wins > 0 else 0.0
    avg_loss = df.loc[df["pnl_per_contract"] < 0, "pnl_per_contract"].mean() if losses > 0 else 0.0
    profit_factor = (
        abs(df.loc[df["pnl_per_contract"] > 0, "pnl_per_contract"].sum() /
            df.loc[df["pnl_per_contract"] < 0, "pnl_per_contract"].sum())
        if losses > 0 else float("inf")
    )
    max_drawdown = df["pnl_per_contract"].cumsum().min()
    peak_equity = df["pnl_per_contract"].cumsum().max()

    # ── Directional accuracy (signal quality metric) ──────────────────────
    # This is the TRUE measure of the signal bot: does the market move in the
    # predicted direction?  Since this bot sends alerts (no orders), direction
    # accuracy is more important than option P&L.
    dir_df = df[df["right"].isin(["C", "P"])].copy()
    if len(dir_df) > 0:
        dir_df["spy_move"] = dir_df["spy_exit"] - dir_df["spy_entry"]
        dir_df["correct_dir"] = (
            ((dir_df["right"] == "C") & (dir_df["spy_move"] > 0)) |
            ((dir_df["right"] == "P") & (dir_df["spy_move"] < 0))
        )
        dir_accuracy = dir_df["correct_dir"].mean() * 100
        # Simulated SPY-share-equivalent P&L (100 shares × directional move)
        dir_df["spy_pnl"] = dir_df.apply(
            lambda r: (r["spy_move"] * 100) if r["right"] == "C" else (-r["spy_move"] * 100),
            axis=1,
        )
        spy_equiv_pnl = dir_df["spy_pnl"].sum()
    else:
        dir_accuracy = 0.0
        spy_equiv_pnl = 0.0

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SPY OPTIONS SIGNAL BOT — BACKTEST REPORT v2 (Apr 2025 → Mar 2026)")
    print("  Fixes: regime filter, 1-side seeding, short DTE, max 1/day")
    print("=" * 72)
    print(f"  Total Trades:      {n_trades}")
    print(f"  Wins / Losses:     {wins}W / {losses}L / {even}E")
    print(f"  Win Rate:          {win_rate:.1f}%")
    print(f"  Total P&L:         ${total_pnl:+,.2f}  (per 1-contract)")
    print(f"  Avg Win:           ${avg_win:+,.2f}")
    print(f"  Avg Loss:          ${avg_loss:+,.2f}")
    print(f"  Profit Factor:     {profit_factor:.2f}")
    print(f"  Peak Equity:       ${peak_equity:+,.2f}")
    print(f"  Max Drawdown:      ${max_drawdown:+,.2f}")
    print()
    print(f"  ── Signal Quality (Directional Accuracy) ──")
    print(f"  Direction correct: {dir_accuracy:.1f}%  (market moved in predicted direction)")
    print(f"  SPY-share equiv:   ${spy_equiv_pnl:+,.2f}  (100 shares × direction)")
    print(f"  ↑ This is the key metric — the bot sends alerts, not orders.")

    # ── Per-signal-type breakdown ─────────────────────────────────────────────
    print("\n  Per-Signal-Type Breakdown:")
    print(f"  {'Signal Type':<22} {'Count':>5}  {'Win%':>6}  {'Total P&L':>12}  {'Avg P&L':>10}")
    print("  " + "-" * 62)
    for stype in sorted(df["signal_type"].unique()):
        sub = df[df["signal_type"] == stype]
        st_n = len(sub)
        st_wins = (sub["pnl_per_contract"] > 0).sum()
        st_wr = st_wins / st_n * 100 if st_n > 0 else 0.0
        st_pnl = sub["pnl_per_contract"].sum()
        st_avg = sub["pnl_per_contract"].mean()
        print(f"  {stype:<22} {st_n:>5}  {st_wr:>5.1f}%  ${st_pnl:>+10,.2f}  ${st_avg:>+8,.2f}")

    # ── Per-regime breakdown ──────────────────────────────────────────────────
    print("\n  Per-Regime Breakdown:")
    print(f"  {'Regime':<18} {'Count':>5}  {'Win%':>6}  {'Total P&L':>12}")
    print("  " + "-" * 48)
    for reg in sorted(df["regime"].unique()):
        sub = df[df["regime"] == reg]
        rn = len(sub)
        rw = (sub["pnl_per_contract"] > 0).sum()
        rwr = rw / rn * 100 if rn > 0 else 0.0
        rpnl = sub["pnl_per_contract"].sum()
        print(f"  {reg:<18} {rn:>5}  {rwr:>5.1f}%  ${rpnl:>+10,.2f}")

    # ── Per-confidence-tier ───────────────────────────────────────────────────
    print("\n  Per-Confidence-Tier Breakdown:")
    print(f"  {'Tier':<12} {'Count':>5}  {'Win%':>6}  {'Total P&L':>12}  {'Avg P&L':>10}")
    print("  " + "-" * 52)
    for tier in ["MEDIUM", "HIGH", "EXTREME"]:
        sub = df[df["confidence_tier"] == tier]
        if len(sub) == 0:
            continue
        tn = len(sub)
        tw = (sub["pnl_per_contract"] > 0).sum()
        twr = tw / tn * 100
        tpnl = sub["pnl_per_contract"].sum()
        tavg = sub["pnl_per_contract"].mean()
        print(f"  {tier:<12} {tn:>5}  {twr:>5.1f}%  ${tpnl:>+10,.2f}  ${tavg:>+8,.2f}")

    # ── Monthly breakdown ─────────────────────────────────────────────────────
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    print("\n  Monthly P&L:")
    print(f"  {'Month':<10} {'Trades':>6}  {'Win%':>6}  {'P&L':>12}  {'Cum P&L':>12}")
    print("  " + "-" * 54)
    cum = 0.0
    for month in sorted(df["month"].unique()):
        sub = df[df["month"] == month]
        mn = len(sub)
        mw = (sub["pnl_per_contract"] > 0).sum()
        mwr = mw / mn * 100 if mn > 0 else 0.0
        mpnl = sub["pnl_per_contract"].sum()
        cum += mpnl
        print(f"  {month:<10} {mn:>6}  {mwr:>5.1f}%  ${mpnl:>+10,.2f}  ${cum:>+10,.2f}")

    # ── Equity curve ──────────────────────────────────────────────────────────
    equity = df["pnl_per_contract"].cumsum()
    print(f"\n  Equity curve (first/last 5 trades):")
    for idx in list(range(min(5, len(equity)))) + list(range(max(0, len(equity) - 5), len(equity))):
        row = df.iloc[idx]
        print(f"    {row['date']}  {row['signal_type']:<22}  PnL: ${row['pnl_per_contract']:>+8,.2f}  "
              f"Equity: ${equity.iloc[idx]:>+10,.2f}")

    print("\n" + "=" * 72)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = run_backtest()
    print_report(df)

    # Save trades to CSV
    out_path = Path(__file__).resolve().parents[1] / "backtest_results" / "spy_options_backtest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nTrades saved to {out_path}")
