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

  SHORT SIDE: Enabled via ft_shorts_enabled config toggle (FEB 12 2026).
  Signal D: EMA21 pullback short (mirror of A) in downtrend.
  Signal E: OR breakdown short (mirror of B) below OR low.

  TREND CONTINUATION: Added FEB 13 2026 for strong rally/selloff days.
  Signal F: Trend continuation long/short — fires when price runs away
            from EMA21 without pulling back.  Uses EMA9 as dynamic support,
            requires full EMA stack alignment + 2 ascending/descending closes
            + ADX >= 22.  Max 2 fires per day per side.

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
    15-minute ES strategy: EMA21 pullback + OR breakout (long & short).

    Long side: EMA21 pullback, EMA9 pullback, OR breakout (Signals A, B, C)
    Short side: EMA21 pullback short, OR breakdown short (Signals D, E)
    Shorts gated by ft_shorts_enabled config toggle.

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
        self._adx_max: float = getattr(config, 'ft_adx_max', 999.0)  # FEB 7 2026: ADX cap
        self._ema_touch_pct: float = getattr(config, 'ft_ema_touch_pct', 0.001)
        self._or_minutes: int = getattr(config, 'ft_or_minutes', 30)

        # FEB 7 2026: EMA9 pullback parameters (Signal C)
        self._ema9_pb_enabled: bool = getattr(config, 'ft_ema9_pb_enabled', False)
        self._ema9_pb_stop_mult: float = getattr(config, 'ft_ema9_pb_stop_mult', 1.2)
        self._ema9_pb_target_mult: float = getattr(config, 'ft_ema9_pb_target_mult', 1.5)
        self._ema9_touch_pct: float = getattr(config, 'ft_ema9_touch_pct', 0.0015)  # 0.15%

        # FEB 12 2026: Short-side signals (D, E) — mirror of long signals
        self._shorts_enabled: bool = getattr(config, 'ft_shorts_enabled', False)
        self._short_pb_stop_mult: float = getattr(config, 'ft_short_pb_stop_mult', 1.5)
        self._short_pb_target_mult: float = getattr(config, 'ft_short_pb_target_mult', 1.0)
        self._short_or_target_r: float = getattr(config, 'ft_short_or_target_r', 1.0)
        self._or_broken_below_today: bool = False  # Only first OR breakdown per day

        # FEB 18 2026: Fixed-point take-profit system
        # FEB 19 2026: Fixed-point stop-loss system (matching TP for consistent R:R)
        # In futures, each point = fixed $ (MES: $5/pt).  ATR-based stops
        # produce inconsistent R:R when paired with fixed TP.
        # Fixed-point SL/TP gives guaranteed R:R at every entry:
        #   A/B/D/E: SL=6pts($30), TP=8pts($40) → R:R 1.33:1
        #   C:       SL=ATR-adaptive (see below), TP=SL×1.25 → R:R 1.25:1
        #   F:       SL=8pts($40), TP=12pts($60) → R:R 1.50:1
        self._fixed_tp_points: float = getattr(config, 'ft_fixed_tp_points', 8.0)          # Signals A, B, D, E
        self._fixed_tp_points_ema9: float = getattr(config, 'ft_fixed_tp_points_ema9', 10.0)  # Signal C fallback
        self._fixed_tp_points_trend: float = getattr(config, 'ft_fixed_tp_points_trend', 12.0) # Signal F (trend cont)
        self._fixed_sl_points: float = getattr(config, 'ft_fixed_sl_points', 6.0)          # Signals A, B, D, E
        self._fixed_sl_points_ema9: float = getattr(config, 'ft_fixed_sl_points_ema9', 8.0)  # Signal C fallback
        self._fixed_sl_points_trend: float = getattr(config, 'ft_fixed_sl_points_trend', 8.0) # Signal F

        # FEB 20 2026: ATR-adaptive stops for Signal C
        # Fixed 8pt SL was only 0.50× ATR at typical vol (mean ATR 16.1) → noise band stops.
        # Dynamic: SL = min(ceiling, max(floor, ATR × mult)), TP = SL × rr_ratio
        # Simulation (Oct 2025): WR 37.5% → 50%, PF 0.54 → 0.95 at ATR×1.0
        self._ema9_sl_atr_mult: float = getattr(config, 'ft_ema9_sl_atr_mult', 1.0)
        self._ema9_sl_floor: float = getattr(config, 'ft_ema9_sl_floor_pts', 8.0)
        self._ema9_sl_ceiling: float = getattr(config, 'ft_ema9_sl_ceiling_pts', 20.0)
        self._ema9_rr_ratio: float = getattr(config, 'ft_ema9_rr_ratio', 1.25)

        # FEB 13 2026: Trend continuation signal (Signal F) — captures
        # strong rally / selloff days when price runs away from EMA21
        # without pulling back.  Uses EMA9 as dynamic support instead.
        self._trend_cont_enabled: bool = getattr(config, 'ft_trend_cont_enabled', True)
        self._trend_cont_stop_mult: float = getattr(config, 'ft_trend_cont_stop_mult', 1.0)
        self._trend_cont_target_mult: float = getattr(config, 'ft_trend_cont_target_mult', 2.0)
        self._trend_cont_adx_min: float = getattr(config, 'ft_trend_cont_adx_min', 18.0)
        self._trend_cont_ema9_pct: float = getattr(config, 'ft_trend_cont_ema9_pct', 0.003)  # 0.3% proximity to EMA9
        # FEB 20 2026: Gap/extension handling — on gap-up/down days, ATR is
        # small (overnight range) but price is far from EMAs.  The old
        # 1.5×ATR limit was ~4 pts, blocking 20-30pt gap-ups entirely.
        # Use max(multiplier × ATR, fixed ceiling) so gaps can still trade.
        self._trend_cont_max_ext_pts: float = getattr(config, 'ft_trend_cont_max_ext_pts', 30.0)  # Max pts from EMA9 allowed
        self._trend_cont_gap_adx_min: float = getattr(config, 'ft_trend_cont_gap_adx_min', 25.0)  # Higher ADX required when extended >3×ATR
        self._trend_cont_fired_long: bool = False   # Max 2 per day per side
        self._trend_cont_fired_short: bool = False
        self._trend_cont_long_count: int = 0   # Track how many fired
        self._trend_cont_short_count: int = 0
        self._trend_cont_max_per_day: int = getattr(config, 'ft_trend_cont_max_per_day', 2)

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

        # ---- FEB 9 2026: Reconstruct OR from historical bars on mid-day restart ----
        # If we're past the OR window and OR was never computed (e.g. bot restarted
        # after 10:00 ET), scan the bootstrapped historical bars in `enriched` for
        # the 9:30-10:00 ET window and reconstruct the Opening Range.
        if et_time.time() >= or_end and not self._or_computed and not self._opening_bars:
            logger.info(
                f"🔁 OR reconstruction check: et_time={et_time.time()} or_end={or_end} "
                f"or_computed={self._or_computed} opening_bars={len(self._opening_bars)}"
            )
            # log enriched index bounds to help debugging
            try:
                logger.info(
                    f"🔁 Enriched index range: {enriched.index[0]} -> {enriched.index[-1]}"
                )
            except Exception:
                pass
            self._reconstruct_or_from_history(enriched, et_date, or_end)

        if self._rth_start <= et_time.time() < or_end:
            self._collect_opening_bar(latest)
            self._prev_close = float(latest["close"])
            logger.info(
                f"🔍 OR collecting: bars={len(self._opening_bars)} "
                f"H={max(b['high'] for b in self._opening_bars):.2f} "
                f"L={min(b['low'] for b in self._opening_bars):.2f}"
            )
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
        macd_hist = float(latest.get("MACDhist_12_26_9", 0))

        # FEB 20 2026: Seed prev_close from prior bar on first evaluation
        # after startup.  Without this, OR cross-detection (Signals B, E)
        # is broken on the first bar (prev_close=0.0).
        if self._prev_close == 0.0 and len(enriched) >= 2:
            self._prev_close = float(enriched.iloc[-2]["close"])
            logger.info(f"📌 Seeded prev_close={self._prev_close:.2f} from prior bar")

        # Fix NaN
        for name, val in [("atr", atr), ("adx", adx), ("rsi", rsi), ("macd_hist", macd_hist)]:
            if np.isnan(val):
                if name == "atr": atr = 10.0
                elif name == "adx": adx = 0.0
                elif name == "rsi": rsi = 50.0
                elif name == "macd_hist": macd_hist = 0.0

        if atr <= 0:
            self._prev_close = close
            return Signal("HOLD", 0.0, {"reason": "ZERO_ATR"})

        # ---- Signal A: EMA21 Pullback Long ----
        signal_a = self._check_ema21_pullback(
            close, open_price, low, ema21, ema50, atr, adx,
            rsi, macd_hist,
        )

        # ---- Signal B: OR Breakout Long ----
        signal_b = self._check_or_breakout(
            close, high, ema9, ema21, atr, adx, macd_hist,
        )

        # ---- Signal C: EMA9 Pullback Long (faster trend) ----
        signal_c = None
        if self._ema9_pb_enabled:
            signal_c = self._check_ema9_pullback(
                close, open_price, low, ema9, ema21, ema50, atr, adx, rsi,
                macd_hist,
            )

        # ---- Signal D: EMA21 Pullback Short (downtrend mirror of A) ----
        signal_d = None
        if self._shorts_enabled:
            signal_d = self._check_ema21_pullback_short(
                close, open_price, high, ema21, ema50, atr, adx,
                rsi, macd_hist,
            )

        # ---- Signal E: OR Breakdown Short (downtrend mirror of B) ----
        signal_e = None
        if self._shorts_enabled:
            signal_e = self._check_or_breakdown(
                close, low, ema9, ema21, atr, adx, macd_hist,
            )

        # ---- Signal F: Trend Continuation (strong momentum days) ----
        # FEB 13 2026: Catches rallies/selloffs where price runs away
        # from EMA21 without pulling back.  Uses EMA9 as dynamic support.
        signal_f_long = None
        signal_f_short = None
        if self._trend_cont_enabled:
            signal_f_long = self._check_trend_continuation_long(
                enriched, close, open_price, low, high,
                ema9, ema21, ema50, atr, adx, rsi, macd_hist,
            )
            if self._shorts_enabled and signal_f_long is None:
                signal_f_short = self._check_trend_continuation_short(
                    enriched, close, open_price, low, high,
                    ema9, ema21, ema50, atr, adx, rsi, macd_hist,
                )

        # Priority: A (EMA21 PB Long) > C (EMA9 PB Long) > B (OR breakout Long)
        #         > F_long (Trend Cont Long)
        #         > D (EMA21 PB Short) > E (OR breakdown Short)
        #         > F_short (Trend Cont Short)
        # Long signals take priority over short signals.
        # If both fire on same bar, take higher priority.
        chosen = None
        if signal_a is not None:
            chosen = signal_a
        elif signal_c is not None:
            chosen = signal_c
        elif signal_b is not None:
            chosen = signal_b
        elif signal_f_long is not None:
            chosen = signal_f_long
        elif signal_d is not None:
            chosen = signal_d
        elif signal_e is not None:
            chosen = signal_e
        elif signal_f_short is not None:
            chosen = signal_f_short

        if chosen is None:
            # ── Diagnostic: why no signal fired ──
            _diag_parts = []
            # Signal A diagnostics (Long EMA21 PB)
            if ema21 <= ema50:
                _diag_parts.append(f"A:ema21({ema21:.1f})<=ema50({ema50:.1f})")
            else:
                _touch = ema21 * (1 + self._ema_touch_pct)
                if low > _touch:
                    _diag_parts.append(f"A:low({low:.1f})>touch({_touch:.1f})")
                elif close <= ema21:
                    _diag_parts.append(f"A:close({close:.1f})<=ema21({ema21:.1f})")
                elif close <= open_price:
                    _diag_parts.append(f"A:bearish(c={close:.1f},o={open_price:.1f})")
                elif adx < self._adx_min or adx > self._adx_max:
                    _diag_parts.append(f"A:adx({adx:.0f})out[{self._adx_min}-{self._adx_max}]")
                elif macd_hist <= 0:
                    _diag_parts.append(f"A:macd_hist({macd_hist:.2f})<=0")
                elif rsi > 70 or rsi < 35:
                    _diag_parts.append(f"A:rsi({rsi:.0f})out[35-70]")
            # Signal B diagnostics (Long OR Breakout)
            if not self._or_computed or self._or_high <= 0:
                _diag_parts.append(f"B:no_OR(computed={self._or_computed},h={self._or_high:.1f})")
            elif self._or_broken_today:
                _diag_parts.append("B:already_broken")
            elif not (close > self._or_high and self._prev_close <= self._or_high):
                _diag_parts.append(f"B:no_cross(c={close:.1f},prev={self._prev_close:.1f},OR_H={self._or_high:.1f})")
            elif ema9 <= ema21:
                _diag_parts.append(f"B:ema9({ema9:.1f})<=ema21({ema21:.1f})")
            elif adx < self._adx_min:
                _diag_parts.append(f"B:adx({adx:.0f})<{self._adx_min}")
            elif macd_hist <= 0:
                _diag_parts.append(f"B:macd_hist({macd_hist:.2f})<=0")
            # Signal D diagnostics (Short EMA21 PB)
            if self._shorts_enabled:
                if ema21 >= ema50:
                    _diag_parts.append(f"D:ema21({ema21:.1f})>=ema50({ema50:.1f})")
                else:
                    _touch_s = ema21 * (1 - self._ema_touch_pct)
                    if high < _touch_s:
                        _diag_parts.append(f"D:high({high:.1f})<touch({_touch_s:.1f})")
                    elif close >= ema21:
                        _diag_parts.append(f"D:close({close:.1f})>=ema21({ema21:.1f})")
                    elif close >= open_price:
                        _diag_parts.append(f"D:bullish(c={close:.1f},o={open_price:.1f})")
                    elif adx < self._adx_min or adx > self._adx_max:
                        _diag_parts.append(f"D:adx({adx:.0f})out[{self._adx_min}-{self._adx_max}]")
                    elif macd_hist >= 0:
                        _diag_parts.append(f"D:macd_hist({macd_hist:.2f})>=0")
                    elif rsi < 30 or rsi > 65:
                        _diag_parts.append(f"D:rsi({rsi:.0f})out[30-65]")
            else:
                _diag_parts.append("D:shorts_disabled")
            # Signal E diagnostics (Short OR Breakdown)
            if self._shorts_enabled:
                if not self._or_computed or self._or_low <= 0:
                    _diag_parts.append(f"E:no_OR(computed={self._or_computed},l={self._or_low:.1f})")
                elif self._or_broken_below_today:
                    _diag_parts.append("E:already_broken_below")
                elif not (close < self._or_low and self._prev_close >= self._or_low):
                    _diag_parts.append(f"E:no_cross(c={close:.1f},prev={self._prev_close:.1f},OR_L={self._or_low:.1f})")
                elif ema9 >= ema21:
                    _diag_parts.append(f"E:ema9({ema9:.1f})>=ema21({ema21:.1f})")
                elif adx < self._adx_min:
                    _diag_parts.append(f"E:adx({adx:.0f})<{self._adx_min}")
                elif macd_hist >= 0:
                    _diag_parts.append(f"E:macd_hist({macd_hist:.2f})>=0")
            # Signal F diagnostics (Trend Continuation)
            if self._trend_cont_enabled:
                if self._trend_cont_long_count >= self._trend_cont_max_per_day:
                    _f_reason = f"F:maxed({self._trend_cont_long_count})"
                elif not (ema9 > ema21 > ema50):
                    _f_reason = f"F:stack(e9={ema9:.0f},e21={ema21:.0f},e50={ema50:.0f})"
                elif adx < self._trend_cont_adx_min:
                    _f_reason = f"F:adx({adx:.0f})<{self._trend_cont_adx_min:.0f}"
                elif close <= ema9:
                    _f_reason = f"F:close({close:.1f})<=ema9({ema9:.1f})"
                elif close <= open_price:
                    _f_reason = f"F:bearish(c={close:.1f},o={open_price:.1f})"
                elif macd_hist <= 0:
                    _f_reason = f"F:macd({macd_hist:.2f})<=0"
                elif rsi < 45 or rsi > 78:
                    _f_reason = f"F:rsi({rsi:.0f})out[45-78]"
                else:
                    # Proximity / ascending closes / gap-adx
                    _spread = close - ema9
                    _max_ext = max(atr * 1.5, self._trend_cont_max_ext_pts)
                    if _spread > _max_ext:
                        _f_reason = f"F:ext({_spread:.1f})>max({_max_ext:.1f})"
                    elif atr > 0 and _spread > atr * 3 and adx < self._trend_cont_gap_adx_min:
                        _f_reason = f"F:gap_adx({adx:.0f})<{self._trend_cont_gap_adx_min:.0f}(ext={_spread:.1f})"
                    elif len(enriched) >= 4:
                        c1 = float(enriched.iloc[-1]["close"])
                        c2 = float(enriched.iloc[-2]["close"])
                        c3 = float(enriched.iloc[-3]["close"])
                        _f_reason = f"F:asc({c1:.1f},{c2:.1f},{c3:.1f})"
                    else:
                        _f_reason = f"F:bars<4({len(enriched)})"
                _diag_parts.append(_f_reason)
            _diag = " | ".join(_diag_parts) if _diag_parts else "unknown"
            logger.info(f"🔍 NO_SIGNAL diag: {_diag}")
            self._prev_close = close
            return Signal("HOLD", 0.0, {"reason": "NO_SIGNAL"})

        action, stop_loss, take_profit, reason = chosen

        is_short = action == "SELL"
        if is_short:
            if "TREND_CONT" in reason:
                market_state = "DOWNTREND_CONTINUATION"
                entry_type = "trend_continuation_short"
            elif "PB" in reason:
                market_state = "DOWNTREND_PULLBACK"
                entry_type = "ema21_pullback_short"
            else:
                market_state = "OR_BREAKDOWN"
                entry_type = "or_breakdown"
        else:
            if "TREND_CONT" in reason:
                market_state = "UPTREND_CONTINUATION"
                entry_type = "trend_continuation_long"
            elif "PB" in reason:
                market_state = "UPTREND_PULLBACK"
                entry_type = "ema21_pullback" if "EMA21_PB" in reason else "ema9_pullback"
            else:
                market_state = "OR_BREAKOUT"
                entry_type = "or_breakout"

        metadata = {
            "reason": reason,
            "strategy_type": self.name,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr_value": atr,
            "adx_value": adx,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "ema9": ema9,
            "ema21": ema21,
            "ema50": ema50,
            "or_high": self._or_high,
            "or_low": self._or_low,
            "market_state": market_state,
            "position_size": 1.0,
            "entry_type": entry_type,
            "session_type": "RTH",
        }

        logger.info(
            f"📊 {current_time} | {action}: close={close:.2f} "
            f"ema21={ema21:.2f} atr={atr:.1f} adx={adx:.0f} rsi={rsi:.0f} "
            f"macd_h={macd_hist:.2f} SL={stop_loss:.2f} TP={take_profit:.2f} | {reason}"
        )

        self._prev_close = close
        return Signal(action=action, confidence=0.7, metadata=metadata)

    # ------------------------------------------------------------------
    #  Signal A: EMA21 Pullback Long
    # ------------------------------------------------------------------
    def _check_ema21_pullback(
        self, close: float, open_p: float, low: float,
        ema21: float, ema50: float, atr: float, adx: float,
        rsi: float = 50.0, macd_hist: float = 0.0,
    ) -> Optional[tuple]:
        """
        EMA21 pullback in uptrend.
        
        Conditions:
          1. EMA21 > EMA50 (uptrend)
          2. Bar low touches EMA21 (within 0.1%)
          3. Close > EMA21 (bounced back above)
          4. Close > Open (bullish bar)
          5. ADX > threshold (trending)
          6. MACD histogram > 0 (momentum confirming uptrend) — FEB 10 2026
          7. RSI 35-70 (not overbought, not deeply oversold) — FEB 10 2026
        
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
        if adx > self._adx_max:
            return None

        # 6. MACD histogram must be positive (momentum confirming uptrend)
        #    FEB 10 2026: All 4 losing signals on Feb 9-10 had negative or
        #    near-zero MACD histograms — trend structure (EMA21>EMA50) was
        #    intact but momentum had already rolled over.
        if macd_hist <= 0:
            return None

        # 7. RSI filter: avoid overbought exhaustion and deep oversold
        #    FEB 10 2026: Feb 9 signals had RSI ~67-70 (overbought zone)
        if rsi > 70 or rsi < 35:
            return None

        # ---- Compute stops/targets ----
        stop_loss = close - self._fixed_sl_points    # Fixed-point SL ($30 at 6 pts)
        take_profit = close + self._fixed_tp_points  # Fixed-point TP ($40 at 8 pts)

        reason = f"EMA21_PB_LONG | ADX={adx:.0f} | RSI={rsi:.0f} | MACD_H={macd_hist:.2f} | ATR={atr:.1f}"
        return ("BUY", stop_loss, take_profit, reason)

    # ------------------------------------------------------------------
    #  Signal B: Opening Range Breakout Long
    # ------------------------------------------------------------------
    def _check_or_breakout(
        self, close: float, high: float,
        ema9: float, ema21: float, atr: float, adx: float,
        macd_hist: float = 0.0,
    ) -> Optional[tuple]:
        """
        Opening Range breakout long (first breakout per day).
        
        Conditions:
          1. OR has been computed
          2. Close > OR_HIGH and previous close ≤ OR_HIGH (first cross)
          3. EMA9 > EMA21 (short-term uptrend)
          4. ADX > threshold
          5. Haven't already fired this signal today
          6. MACD histogram > 0 (momentum confirming breakout) — FEB 10 2026
        
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
        if adx > self._adx_max:
            return None

        # 6. MACD histogram must be positive (momentum confirming breakout)
        #    FEB 10 2026: Don't chase OR breakouts with fading momentum
        if macd_hist <= 0:
            return None

        # ---- Compute stops/targets ----
        # FEB 19 2026: Fixed-point SL/TP for consistent R:R
        stop_loss = close - self._fixed_sl_points    # Fixed-point SL ($30 at 6 pts)
        take_profit = close + self._fixed_tp_points  # Fixed-point TP ($40 at 8 pts)

        self._or_broken_today = True

        reason = f"OR_BREAK_LONG | ADX={adx:.0f} | OR_H={self._or_high:.2f}"
        return ("BUY", stop_loss, take_profit, reason)

    # ------------------------------------------------------------------
    #  Signal C: EMA9 Pullback Long (fast trend)
    # ------------------------------------------------------------------
    def _check_ema9_pullback(
        self, close: float, open_p: float, low: float,
        ema9: float, ema21: float, ema50: float, atr: float, adx: float,
        rsi: float, macd_hist: float = 0.0,
    ) -> Optional[tuple]:
        """
        EMA9 pullback in strong uptrend — captures faster moves.
        
        This is a higher-frequency companion to Signal A. It fires when
        price pulls back to EMA9 (not EMA21) in a strong trend, catching
        shallow pullbacks that EMA21 misses.
        
        Conditions:
          1. Strong uptrend: EMA9 > EMA21 > EMA50
          2. Bar low touches EMA9 (within 0.15%)
          3. Close > EMA9 (bounced back above)
          4. Bullish bar (close > open)
          5. ADX 22-35 (trending but not exhausted)
          6. RSI 40-70 (not overbought, not oversold)
          7. Bar did NOT also touch EMA21 (avoid overlap with Signal A)
          8. MACD histogram > 0 (momentum confirming) — FEB 10 2026
        
        Returns: (action, stop, target, reason) or None
        """
        # 1. Strong uptrend (all EMAs aligned)
        if not (ema9 > ema21 > ema50):
            return None

        # 2. Low touches EMA9
        touch_threshold = ema9 * (1 + self._ema9_touch_pct)
        if low > touch_threshold:
            return None

        # 3. Close above EMA9
        if close <= ema9:
            return None

        # 4. Bullish bar
        if close <= open_p:
            return None

        # 5. ADX filter (22-35: trending but not exhausted)
        if adx < 22 or adx > self._adx_max:
            return None

        # 6. RSI filter (avoid overbought)
        if rsi > 70 or rsi < 40:
            return None

        # 7. No overlap with Signal A (if bar also touched EMA21, let A handle it)
        ema21_touch = ema21 * (1 + self._ema_touch_pct)
        if low <= ema21_touch:
            return None

        # 8. MACD histogram must be positive (momentum confirming)
        if macd_hist <= 0:
            return None

        # ---- Compute ATR-adaptive stops/targets (FEB 20 2026) ----
        # SL = min(ceiling, max(floor, ATR × mult))  — adapts to volatility
        # TP = SL × rr_ratio                         — preserves R:R
        sl_pts = min(self._ema9_sl_ceiling,
                     max(self._ema9_sl_floor, atr * self._ema9_sl_atr_mult))
        tp_pts = sl_pts * self._ema9_rr_ratio
        stop_loss = close - sl_pts
        take_profit = close + tp_pts

        reason = (f"EMA9_PB_LONG | ADX={adx:.0f} | RSI={rsi:.0f} | ATR={atr:.1f}"
                  f" | SL={sl_pts:.1f}pts | TP={tp_pts:.1f}pts")
        return ("BUY", stop_loss, take_profit, reason)

    # ------------------------------------------------------------------
    #  Signal D: EMA21 Pullback Short (downtrend mirror of Signal A)
    # ------------------------------------------------------------------
    def _check_ema21_pullback_short(
        self, close: float, open_p: float, high: float,
        ema21: float, ema50: float, atr: float, adx: float,
        rsi: float = 50.0, macd_hist: float = 0.0,
    ) -> Optional[tuple]:
        """
        EMA21 pullback in downtrend — short-side mirror of Signal A.

        Conditions (exact inverse of long):
          1. EMA21 < EMA50 (downtrend)
          2. Bar high touches EMA21 from below (within 0.1%)
          3. Close < EMA21 (rejected back below)
          4. Close < Open (bearish bar)
          5. ADX > threshold (trending)
          6. MACD histogram < 0 (momentum confirming downtrend)
          7. RSI 30-65 (not oversold, not overbought)

        Returns: (action, stop, target, reason) or None
        """
        # 1. Downtrend
        if ema21 >= ema50:
            return None

        # 2. High touches EMA21 from below
        touch_threshold = ema21 * (1 - self._ema_touch_pct)
        if high < touch_threshold:
            return None

        # 3. Close below EMA21
        if close >= ema21:
            return None

        # 4. Bearish bar
        if close >= open_p:
            return None

        # 5. ADX filter
        if adx < self._adx_min:
            return None
        if adx > self._adx_max:
            return None

        # 6. MACD histogram must be negative (momentum confirming downtrend)
        if macd_hist >= 0:
            return None

        # 7. RSI filter: avoid deeply oversold and overbought
        if rsi < 30 or rsi > 65:
            return None

        # ---- Compute stops/targets (inverted) ----
        stop_loss = close + self._fixed_sl_points    # Fixed-point SL ($30 at 6 pts)
        take_profit = close - self._fixed_tp_points  # Fixed-point TP ($40 at 8 pts)

        reason = f"EMA21_PB_SHORT | ADX={adx:.0f} | RSI={rsi:.0f} | MACD_H={macd_hist:.2f} | ATR={atr:.1f}"
        return ("SELL", stop_loss, take_profit, reason)

    # ------------------------------------------------------------------
    #  Signal E: Opening Range Breakdown Short (mirror of Signal B)
    # ------------------------------------------------------------------
    def _check_or_breakdown(
        self, close: float, low: float,
        ema9: float, ema21: float, atr: float, adx: float,
        macd_hist: float = 0.0,
    ) -> Optional[tuple]:
        """
        Opening Range breakdown short — first breakdown per day.

        Conditions (exact inverse of long OR breakout):
          1. OR has been computed
          2. Close < OR_LOW and previous close >= OR_LOW (first cross below)
          3. EMA9 < EMA21 (short-term downtrend)
          4. ADX > threshold
          5. Haven't already fired this signal today
          6. MACD histogram < 0 (momentum confirming breakdown)

        Returns: (action, stop, target, reason) or None
        """
        if not self._or_computed or self._or_low <= 0:
            return None

        if self._or_broken_below_today:
            return None

        # First cross below OR_LOW
        if not (close < self._or_low and self._prev_close >= self._or_low):
            return None

        # EMA alignment (bearish)
        if ema9 >= ema21:
            return None

        # ADX filter
        if adx < self._adx_min:
            return None
        if adx > self._adx_max:
            return None

        # MACD histogram must be negative (momentum confirming breakdown)
        if macd_hist >= 0:
            return None

        # ---- Compute stops/targets (inverted) ----
        # FEB 19 2026: Fixed-point SL/TP for consistent R:R
        stop_loss = close + self._fixed_sl_points    # Fixed-point SL ($30 at 6 pts)
        take_profit = close - self._fixed_tp_points  # Fixed-point TP ($40 at 8 pts)

        self._or_broken_below_today = True

        reason = f"OR_BREAK_SHORT | ADX={adx:.0f} | OR_L={self._or_low:.2f}"
        return ("SELL", stop_loss, take_profit, reason)

    # ------------------------------------------------------------------
    #  Signal F: Trend Continuation Long (strong rally days)
    # ------------------------------------------------------------------
    def _check_trend_continuation_long(
        self, df: pd.DataFrame, close: float, open_p: float,
        low: float, high: float,
        ema9: float, ema21: float, ema50: float,
        atr: float, adx: float, rsi: float, macd_hist: float,
    ) -> Optional[tuple]:
        """
        Trend continuation in strong uptrend — catches strong rally days
        when price runs away from EMA21 without pulling back.

        FEB 13 2026: On strong rally days the pullback signals (A, C)
        never fire because price never dips to EMA21/EMA9.  This signal
        enters on momentum continuation using EMA9 as dynamic support.

        Conditions:
          1. Full EMA stack: EMA9 > EMA21 > EMA50 (strong uptrend)
          2. Close > EMA9 (price above all EMAs — running)
          3. Low is near EMA9 (within 0.3%) — shallow dip toward EMA9
             OR current bar is bullish and prev bar close > EMA9 (sustained trend)
          4. ADX >= 22 (strong trending)
          5. MACD histogram > 0 (momentum confirming)
          6. RSI 45-78 (not exhausted, not weak)
          7. Last 3 closes are ascending (c[-1] > c[-2] > c[-3]) — momentum
          8. Max N fires per day per side (default 2)
          9. Bullish bar (close > open)

        Risk: tighter stop (1.0× ATR below EMA9), wider target (2.0× ATR).
        Stop is placed below EMA9 to use it as structural support.

        Returns: (action, stop, target, reason) or None
        """
        # 8. Daily limit
        if self._trend_cont_long_count >= self._trend_cont_max_per_day:
            return None

        # 1. Full EMA stack
        if not (ema9 > ema21 > ema50):
            return None

        # 2. Close above EMA9
        if close <= ema9:
            return None

        # 9. Bullish bar
        if close <= open_p:
            return None

        # 4. ADX filter (need strong trend)
        if adx < self._trend_cont_adx_min:
            return None

        # 5. MACD momentum
        if macd_hist <= 0:
            return None

        # 6. RSI filter
        if rsi < 45 or rsi > 78:
            return None

        # 3. EMA9 proximity check — low should be near EMA9 (shallow dip)
        #    This prevents buying at the very top of a spike.
        #    FEB 20 2026: Use max(1.5×ATR, max_ext_pts) to handle gap-up days
        #    where ATR is small but price legitimately gapped away from EMAs.
        #    When extension is large (>3×ATR), require higher ADX for safety.
        ema9_zone = ema9 * (1 + self._trend_cont_ema9_pct)
        spread = close - ema9
        max_allowed = max(atr * 1.5, self._trend_cont_max_ext_pts)
        if low > ema9_zone:
            if spread > max_allowed:
                return None
            # Extra safety: if very extended (>3×ATR), demand stronger trend
            if atr > 0 and spread > atr * 3:
                if adx < self._trend_cont_gap_adx_min:
                    return None

        # 7. Ascending closes — last 2 bars must show rising closes
        # FEB 20 2026: Relaxed from 3-bar to 2-bar. Simulation showed
        # F_long improved from 3W/10L to 5W/7L with this change.
        if len(df) < 3:
            return None
        c1 = float(df.iloc[-1]["close"])  # current
        c2 = float(df.iloc[-2]["close"])  # prev
        if not (c1 > c2):
            return None

        # ---- Compute stops/targets ----
        # FEB 19 2026: Fixed-point SL/TP for consistent R:R
        stop_loss = close - self._fixed_sl_points_trend      # Fixed-point SL ($40 at 8 pts)
        take_profit = close + self._fixed_tp_points_trend    # Fixed-point TP ($60 at 12 pts)

        self._trend_cont_long_count += 1

        reason = (f"TREND_CONT_LONG | ADX={adx:.0f} | RSI={rsi:.0f} "
                  f"| MACD_H={macd_hist:.2f} | e9={ema9:.1f} | #{self._trend_cont_long_count}")
        return ("BUY", stop_loss, take_profit, reason)

    # ------------------------------------------------------------------
    #  Signal F: Trend Continuation Short (strong selloff days)
    # ------------------------------------------------------------------
    def _check_trend_continuation_short(
        self, df: pd.DataFrame, close: float, open_p: float,
        low: float, high: float,
        ema9: float, ema21: float, ema50: float,
        atr: float, adx: float, rsi: float, macd_hist: float,
    ) -> Optional[tuple]:
        """
        Trend continuation in strong downtrend — mirror of long version.

        Conditions (exact inverse):
          1. Full EMA stack: EMA9 < EMA21 < EMA50 (strong downtrend)
          2. Close < EMA9 (price below all EMAs — selling)
          3. High is near EMA9 (within 0.3%) — shallow bounce toward EMA9
             OR current bar bearish and prev close < EMA9
          4. ADX >= 22 (strong trending)
          5. MACD histogram < 0 (momentum confirming)
          6. RSI 22-55 (not oversold, not strong)
          7. Last 3 closes are descending (c[-1] < c[-2] < c[-3])
          8. Max N fires per day (default 2)
          9. Bearish bar (close < open)

        Returns: (action, stop, target, reason) or None
        """
        # 8. Daily limit
        if self._trend_cont_short_count >= self._trend_cont_max_per_day:
            return None

        # 1. Full EMA stack (bearish)
        if not (ema9 < ema21 < ema50):
            return None

        # 2. Close below EMA9
        if close >= ema9:
            return None

        # 9. Bearish bar
        if close >= open_p:
            return None

        # 4. ADX filter
        if adx < self._trend_cont_adx_min:
            return None

        # 5. MACD momentum (negative)
        if macd_hist >= 0:
            return None

        # 6. RSI filter
        if rsi < 22 or rsi > 55:
            return None

        # 3. EMA9 proximity — high near EMA9 (shallow bounce)
        #    FEB 20 2026: Use max(1.5×ATR, max_ext_pts) for gap-down days
        ema9_zone = ema9 * (1 - self._trend_cont_ema9_pct)
        spread = ema9 - close
        max_allowed = max(atr * 1.5, self._trend_cont_max_ext_pts)
        if high < ema9_zone:
            if spread > max_allowed:
                return None
            # Extra safety: very extended gap-down needs stronger trend
            if atr > 0 and spread > atr * 3:
                if adx < self._trend_cont_gap_adx_min:
                    return None

        # 7. Descending closes — last 2 bars must show falling closes
        # FEB 20 2026: Relaxed from 3-bar to 2-bar (mirrors long side change).
        if len(df) < 3:
            return None
        c1 = float(df.iloc[-1]["close"])
        c2 = float(df.iloc[-2]["close"])
        if not (c1 < c2):
            return None

        # ---- Compute stops/targets (inverted) ----
        # FEB 19 2026: Fixed-point SL/TP for consistent R:R
        stop_loss = close + self._fixed_sl_points_trend      # Fixed-point SL ($40 at 8 pts)
        take_profit = close - self._fixed_tp_points_trend    # Fixed-point TP ($60 at 12 pts)

        self._trend_cont_short_count += 1

        reason = (f"TREND_CONT_SHORT | ADX={adx:.0f} | RSI={rsi:.0f} "
                  f"| MACD_H={macd_hist:.2f} | e9={ema9:.1f} | #{self._trend_cont_short_count}")
        return ("SELL", stop_loss, take_profit, reason)

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
        logger.info(
            f"📊 OR computed: HIGH={self._or_high:.2f} LOW={self._or_low:.2f} "
            f"({len(self._opening_bars)} bars)"
        )
        # Persist OR to disk so restarts can reconstruct if needed
        try:
            import json, os
            date_str = self._session_date.isoformat() if self._session_date else datetime.now().date().isoformat()
            path = os.path.join("data", f"or_{date_str}.json")
            with open(path, "w") as fh:
                json.dump({"date": date_str, "or_high": self._or_high, "or_low": self._or_low}, fh)
            logger.info(f"💾 OR persisted to {path}")
        except Exception as _:
            logger.debug("OR persistence skipped/failed")

    def _reconstruct_or_from_history(
        self, df: pd.DataFrame, target_date, or_end: time
    ) -> None:
        """Reconstruct Opening Range from bootstrapped historical bars.

        FEB 9 2026: When the bot restarts after the OR window (10:00 ET),
        the in-memory OR state is lost.  This scans the historical bars in
        the features DataFrame for today's 9:30-10:00 ET window and
        reconstructs the Opening Range so Signal B (OR breakout) can fire.
        """
        try:
            or_bars_found = []
            for ts, row in df.iterrows():
                # Ensure timestamp is a pandas Timestamp with timezone info.
                try:
                    if not isinstance(ts, pd.Timestamp):
                        ts = pd.Timestamp(ts)
                    if ts.tz is None:
                        # DataFrames from some sources may be UTC-naive; assume UTC
                        ts = ts.tz_localize('UTC')
                    et_ts = ts.tz_convert('US/Eastern')
                except Exception:
                    # Fallback to helper
                    et_ts = self._to_et(ts)

                if et_ts.date() != target_date:
                    continue
                if self._rth_start <= et_ts.time() < or_end:
                    or_bars_found.append({
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                    })

            if or_bars_found:
                self._opening_bars = or_bars_found
                self._compute_opening_range()
                logger.info(
                    f"🔄 OR RECONSTRUCTED from {len(or_bars_found)} historical bars: "
                    f"HIGH={self._or_high:.2f} LOW={self._or_low:.2f}"
                )
            else:
                # Try loading persisted OR from disk as a fallback
                try:
                    import os, json
                    date_str = target_date.isoformat() if hasattr(target_date, 'isoformat') else str(target_date)
                    path = os.path.join("data", f"or_{date_str}.json")
                    if os.path.exists(path):
                        with open(path, "r") as fh:
                            payload = json.load(fh)
                            self._or_high = float(payload.get("or_high", 0.0))
                            self._or_low = float(payload.get("or_low", 0.0))
                            self._or_computed = True
                            logger.info(f"🔄 OR loaded from disk fallback: HIGH={self._or_high:.2f} LOW={self._or_low:.2f}")
                            return
                except Exception:
                    pass
                logger.warning(
                    f"⚠️ No OR bars found in history for {target_date} "
                    f"(window {self._rth_start}-{or_end} ET). "
                    f"DataFrame range: {df.index[0]} to {df.index[-1]}"
                )
        except Exception as exc:
            logger.error(f"Failed to reconstruct OR from history: {exc}")

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
        self._or_broken_below_today = False
        self._prev_close = 0.0
        # Signal F counters
        self._trend_cont_long_count = 0
        self._trend_cont_short_count = 0

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
