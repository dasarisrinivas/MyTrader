"""
ES 15-Minute Strategy
=====================

FEB 2026 — Complete replacement of failed 1-minute strategies.

WHY 15 MINUTES
--------------
1-minute bars on ES are pure noise:
  - Zero-range bars, 2-pt avg range, anti-predictive EMA crosses
  - 1,552-trade backtest: -$5,209, 34.1% WR (scoring strategy)
  - 82-trade structural reversion: -$1,278, 55% WR but R:R 0.48

15-minute bars are dramatically better:
  - 12.5 pt avg range, 40K avg volume, zero zero-range bars
  - EMA21 pullback long: PF 1.75, 422 trades, +$6,700 (1 year)
  - Robust to look-ahead test (next-bar entry PF 1.81)
  - Sharpe 4.05, max DD $616, only 1 losing month (Sep)

DATA-DRIVEN DESIGN
------------------
From simulation of 15m bars (277 trading days, Feb 2025 - Jan 2026):

  Primary signal:  EMA21 pullback long in uptrend
    - EMA21 > EMA50 (uptrend)
    - Bar low touches EMA21 (±0.1%)
    - Bar closes above EMA21 (bounce confirmed)
    - Bar is bullish (close > open)
    - ADX > 20 (trending, not choppy)
    - Stop: 1.5× ATR, Target: 2.5× ATR, Max hold: 6 bars (90 min)
    - 422 trades, PF 1.75, avg win 13.5 pts, avg loss -9.7 pts

  Secondary signal: Opening Range breakout long
    - Close crosses above OR high (first time)
    - EMA9 > EMA21 (short-term uptrend)
    - ADX > 20
    - Stop: OR low - 1 pt, Target: 1.5× risk distance
    - 158 trades, PF 1.91

  Combined (day-by-day): +$8,309, Sharpe 4.05, 56% winning days

  SHORT SIDE: Consistently negative (ES has upward drift).
  DO NOT trade shorts with this strategy.

ENTRY MODEL
-----------
1. Compute Opening Range from first 30 min of RTH (9:30-10:00 ET)
2. Compute EMA9, EMA21, EMA50, RSI, ATR, ADX on 15m bars
3. During RTH (9:30-15:00 ET):
   Signal A (EMA21 Pullback):
     - EMA21 > EMA50 (uptrend)
     - Bar low ≤ EMA21 × 1.001 (touches EMA21)
     - Close > EMA21 AND close > open (bullish bounce)
     - ADX > 20
   Signal B (OR Breakout):
     - Close > OR_HIGH and prev close ≤ OR_HIGH (breakout bar)
     - EMA9 > EMA21 (short-term aligned)
     - ADX > 20

EXIT MODEL
----------
- Stop: 1.5× ATR (Signal A) or OR_LOW - 1 pt (Signal B)
- Target: 2.5× ATR (Signal A) or 1.5× risk (Signal B)
- Time-stop: 6 bars (90 min) — if neither hit, exit at market
- No artificial daily trade cap (avg 3.5/day, max ~10)

Author: Quantitative Trading Redesign — Feb 2026
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta, datetime
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
from loguru import logger

from ..config import OneMinuteStrategyConfig
from ..features.feature_engineer import _adx, _atr, _ema, _rsi
from .base import BaseStrategy, Signal


@dataclass
class FifteenMinLevels:
    """Structural levels for 15m strategy."""
    or_high: float = 0.0
    or_low: float = 0.0
    ema9: float = 0.0
    ema21: float = 0.0
    ema50: float = 0.0
    atr: float = 0.0
    adx: float = 0.0
    rsi: float = 50.0


class EsFifteenMinStrategy(BaseStrategy):
    """
    15-minute ES strategy: EMA21 pullback in uptrend + OR breakout.
    Long-only — shorts are consistently negative on ES.

    Designed from scratch based on 15m bar simulation showing
    PF 1.75 over 277 trading days with 422 trades.
    """

    name = "es_fifteen_min"

    def __init__(self, config: OneMinuteStrategyConfig):
        self.config = config

        # Session state (reset daily)
        self._session_date: Optional[Any] = None
        self._or_high: float = 0.0
        self._or_low: float = 0.0
        self._or_computed: bool = False
        self._opening_bars: List[dict] = []
        self._or_broken_today: bool = False  # Only first OR breakout per day
        self._prev_close: float = 0.0  # For detecting first cross above OR_HIGH

        # Parameters (from simulation sweep)
        self._pb_stop_mult: float = getattr(config, 'ft_pb_stop_mult', 1.5)
        self._pb_target_mult: float = getattr(config, 'ft_pb_target_mult', 2.5)
        self._or_target_r: float = getattr(config, 'ft_or_target_r', 1.5)
        self._adx_min: float = getattr(config, 'ft_adx_min', 20.0)
        self._ema_touch_pct: float = getattr(config, 'ft_ema_touch_pct', 0.001)
        self._or_minutes: int = getattr(config, 'ft_or_minutes', 30)

        # FEB 7 2026: Entry time filter (ET)
        # Data analysis on DST-correct backtest shows:
        #   10:xx entries: -$1,208 (87 trades, 55% WR but terrible R:R)
        #   15:xx entries: -$397 (22 trades, 9% WR)
        # Skipping these hours recovers +$1,479 on 128 fewer trades.
        # Default: 11:00-14:59 ET (midday sweet spot)
        self._entry_start_et: time = time(
            getattr(config, 'ft_entry_start_hour', 11),
            getattr(config, 'ft_entry_start_minute', 0)
        )
        self._entry_end_et: time = time(
            getattr(config, 'ft_entry_end_hour', 15),
            getattr(config, 'ft_entry_end_minute', 0)
        )

        # Session times (ET)
        self._rth_start: time = time(
            getattr(config, 'rth_start_hour', 9),
            getattr(config, 'rth_start_minute', 30)
        )
        self._rth_end: time = time(
            getattr(config, 'rth_end_hour', 16),
            getattr(config, 'rth_end_minute', 0)
        )

    # ------------------------------------------------------------------
    #  BaseStrategy interface
    # ------------------------------------------------------------------
    def generate(self, features: pd.DataFrame) -> Signal:
        """
        Main entry point — called by engine on every 15m bar.
        
        Args:
            features: DataFrame with 15m OHLCV + indicators.
                      Index is UTC DatetimeIndex.
        """
        if len(features) < 60:
            return Signal("HOLD", 0.0, {"reason": "WARMUP"})

        enriched = self._ensure_indicators(features)
        latest = enriched.iloc[-1]
        current_time = enriched.index[-1]

        # Convert to ET for time logic
        et_time = self._to_et(current_time)
        et_date = et_time.date()

        # ---- Daily reset ----
        if self._session_date != et_date:
            self._reset_session(et_date)

        # ---- Opening Range collection (first 30 min of RTH) ----
        or_end = self._add_minutes_to_time(self._rth_start, self._or_minutes)

        if self._rth_start <= et_time.time() < or_end:
            self._collect_opening_bar(latest)
            self._prev_close = float(latest["close"])
            return Signal("HOLD", 0.0, {"reason": "OPENING_RANGE_FORMING"})

        # Compute OR once after opening range window
        if not self._or_computed and self._opening_bars:
            self._compute_opening_range()

        # ---- Must be within RTH ----
        if not (self._rth_start <= et_time.time() < self._rth_end):
            self._prev_close = float(latest["close"])
            return Signal("HOLD", 0.0, {"reason": "OUTSIDE_RTH"})

        # ---- Entry time filter (FEB 7 2026) ----
        # Skip hours with negative expectancy (10:xx = -$1,208, 15:xx = -$397)
        if not (self._entry_start_et <= et_time.time() < self._entry_end_et):
            self._prev_close = float(latest["close"])
            return Signal("HOLD", 0.0, {"reason": "OUTSIDE_ENTRY_WINDOW"})

        # ---- Extract indicators ----
        close = float(latest["close"])
        open_price = float(latest.get("open", close))
        low = float(latest["low"])
        high = float(latest["high"])
        ema9 = float(latest.get("EMA_9", close))
        ema21 = float(latest.get("EMA_21", close))
        ema50 = float(latest.get("EMA_50", close))
        atr = float(latest.get("ATR_14", 0))
        adx = float(latest.get("ADX_14", 0))
        rsi = float(latest.get("RSI_14", 50))

        # Fix NaN
        for name, val in [("atr", atr), ("adx", adx), ("rsi", rsi)]:
            if np.isnan(val):
                if name == "atr": atr = 10.0
                elif name == "adx": adx = 0.0
                elif name == "rsi": rsi = 50.0

        if atr <= 0:
            self._prev_close = close
            return Signal("HOLD", 0.0, {"reason": "ZERO_ATR"})

        # ---- Signal A: EMA21 Pullback Long ----
        signal_a = self._check_ema21_pullback(
            close, open_price, low, ema21, ema50, atr, adx
        )

        # ---- Signal B: OR Breakout Long ----
        signal_b = self._check_or_breakout(
            close, high, ema9, ema21, atr, adx
        )

        # Prefer Signal A (higher frequency, better tested).
        # If both fire on same bar, take A (more conservative stop).
        chosen = None
        if signal_a is not None:
            chosen = signal_a
        elif signal_b is not None:
            chosen = signal_b

        if chosen is None:
            self._prev_close = close
            return Signal("HOLD", 0.0, {"reason": "NO_SIGNAL"})

        action, stop_loss, take_profit, reason = chosen

        metadata = {
            "reason": reason,
            "strategy_type": self.name,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr,
            "adx_value": adx,
            "rsi": rsi,
            "ema9": ema9,
            "ema21": ema21,
            "ema50": ema50,
            "or_high": self._or_high,
            "or_low": self._or_low,
            "market_state": "UPTREND_PULLBACK" if "PB" in reason else "OR_BREAKOUT",
            "position_size": 1.0,
            "entry_type": "ema21_pullback" if "PB" in reason else "or_breakout",
            "session_type": "RTH",
        }

        logger.info(
            f"📊 {current_time} | {action}: close={close:.2f} "
            f"ema21={ema21:.2f} atr={atr:.1f} adx={adx:.0f} "
            f"SL={stop_loss:.2f} TP={take_profit:.2f} | {reason}"
        )

        self._prev_close = close
        return Signal(action=action, confidence=0.7, metadata=metadata)

    # ------------------------------------------------------------------
    #  Signal A: EMA21 Pullback Long
    # ------------------------------------------------------------------
    def _check_ema21_pullback(
        self, close: float, open_p: float, low: float,
        ema21: float, ema50: float, atr: float, adx: float
    ) -> Optional[tuple]:
        """
        EMA21 pullback in uptrend.
        
        Conditions:
          1. EMA21 > EMA50 (uptrend)
          2. Bar low touches EMA21 (within 0.1%)
          3. Close > EMA21 (bounced back above)
          4. Close > Open (bullish bar)
          5. ADX > threshold (trending)
        
        Returns: (action, stop, target, reason) or None
        """
        # 1. Uptrend
        if ema21 <= ema50:
            return None

        # 2. Low touches EMA21
        touch_threshold = ema21 * (1 + self._ema_touch_pct)
        if low > touch_threshold:
            return None

        # 3. Close above EMA21
        if close <= ema21:
            return None

        # 4. Bullish bar
        if close <= open_p:
            return None

        # 5. ADX filter
        if adx < self._adx_min:
            return None

        # ---- Compute stops/targets ----
        stop_loss = close - atr * self._pb_stop_mult
        take_profit = close + atr * self._pb_target_mult

        reason = f"EMA21_PB_LONG | ADX={adx:.0f} | ATR={atr:.1f}"
        return ("BUY", stop_loss, take_profit, reason)

    # ------------------------------------------------------------------
    #  Signal B: Opening Range Breakout Long
    # ------------------------------------------------------------------
    def _check_or_breakout(
        self, close: float, high: float,
        ema9: float, ema21: float, atr: float, adx: float
    ) -> Optional[tuple]:
        """
        Opening Range breakout long (first breakout per day).
        
        Conditions:
          1. OR has been computed
          2. Close > OR_HIGH and previous close ≤ OR_HIGH (first cross)
          3. EMA9 > EMA21 (short-term uptrend)
          4. ADX > threshold
          5. Haven't already fired this signal today
        
        Returns: (action, stop, target, reason) or None
        """
        if not self._or_computed or self._or_high <= 0:
            return None

        if self._or_broken_today:
            return None

        # First cross above OR_HIGH
        if not (close > self._or_high and self._prev_close <= self._or_high):
            return None

        # EMA alignment
        if ema9 <= ema21:
            return None

        # ADX filter
        if adx < self._adx_min:
            return None

        # ---- Compute stops/targets ----
        risk_dist = close - self._or_low + 1.0  # OR low - 1 pt buffer
        stop_loss = self._or_low - 1.0
        take_profit = close + risk_dist * self._or_target_r

        # Cap stop distance to prevent enormous risk on wide ORs
        max_stop_dist = atr * 3.0
        if (close - stop_loss) > max_stop_dist:
            stop_loss = close - max_stop_dist
            risk_dist = max_stop_dist
            take_profit = close + risk_dist * self._or_target_r

        self._or_broken_today = True

        reason = f"OR_BREAK_LONG | ADX={adx:.0f} | OR_H={self._or_high:.2f}"
        return ("BUY", stop_loss, take_profit, reason)

    # ------------------------------------------------------------------
    #  Opening range helpers
    # ------------------------------------------------------------------
    def _collect_opening_bar(self, bar: pd.Series) -> None:
        """Collect bars during opening range period."""
        self._opening_bars.append({
            'high': float(bar['high']),
            'low': float(bar['low']),
            'close': float(bar['close']),
        })

    def _compute_opening_range(self) -> None:
        """Compute OR from collected bars."""
        if not self._opening_bars:
            return
        self._or_high = max(b['high'] for b in self._opening_bars)
        self._or_low = min(b['low'] for b in self._opening_bars)
        self._or_computed = True
        logger.debug(
            f"OR computed: HIGH={self._or_high:.2f} LOW={self._or_low:.2f} "
            f"({len(self._opening_bars)} bars)"
        )

    # ------------------------------------------------------------------
    #  Session management
    # ------------------------------------------------------------------
    def _reset_session(self, date) -> None:
        """Reset all session state for a new trading day."""
        self._session_date = date
        self._or_high = 0.0
        self._or_low = 0.0
        self._or_computed = False
        self._opening_bars = []
        self._or_broken_today = False
        self._prev_close = 0.0

    def _to_et(self, ts: pd.Timestamp) -> Any:
        """Convert timestamp to US/Eastern."""
        if ts.tz is not None:
            try:
                return ts.tz_convert("US/Eastern")
            except Exception:
                return ts
        return ts

    @staticmethod
    def _add_minutes_to_time(t: time, minutes: int) -> time:
        from datetime import datetime as dt
        d = dt.combine(dt.today(), t) + timedelta(minutes=minutes)
        return d.time()

    # ------------------------------------------------------------------
    #  Indicator computation
    # ------------------------------------------------------------------
    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required indicators are present in the dataframe."""
        enriched = df.copy()
        close = enriched["close"]
        high = enriched["high"]
        low = enriched["low"]

        if "EMA_9" not in enriched:
            enriched["EMA_9"] = _ema(close, 9)
        if "EMA_21" not in enriched:
            enriched["EMA_21"] = _ema(close, 21)
        if "EMA_50" not in enriched:
            enriched["EMA_50"] = _ema(close, 50)
        if "RSI_14" not in enriched:
            enriched["RSI_14"] = _rsi(close, 14)
        if "ATR_14" not in enriched:
            enriched["ATR_14"] = _atr(high, low, close, 14)
        if "ADX_14" not in enriched:
            enriched["ADX_14"] = _adx(high, low, close, 14)

        # Forward-fill NaN from warmup
        for col in ["EMA_9", "EMA_21", "EMA_50", "RSI_14", "ATR_14", "ADX_14"]:
            if col in enriched.columns:
                enriched[col] = enriched[col].ffill().bfill()

        return enriched
