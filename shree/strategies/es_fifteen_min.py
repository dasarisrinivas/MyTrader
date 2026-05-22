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

import json
from dataclasses import dataclass
from datetime import time, timedelta, datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
from loguru import logger

from ..config import OneMinuteStrategyConfig
from ..features.feature_engineer import _adx, _atr, _ema, _rsi
from .base import BaseStrategy, Signal

try:
    from ..data.sentiment_aggregator import update_price_momentum
    _PRICE_MOMENTUM_AVAILABLE = True
except ImportError:
    _PRICE_MOMENTUM_AVAILABLE = False


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
        self._or_break_long_count: int = 0   # FEB 26 2026: counter (was bool)
        self._prev_close: float = 0.0  # For detecting first cross above OR_HIGH

        # Parameters (from simulation sweep)
        self._pb_stop_mult: float = getattr(config, 'ft_pb_stop_mult', 1.5)
        self._pb_target_mult: float = getattr(config, 'ft_pb_target_mult', 2.5)
        self._or_target_r: float = getattr(config, 'ft_or_target_r', 1.5)
        self._adx_min: float = getattr(config, 'ft_adx_min', 20.0)
        self._adx_max: float = getattr(config, 'ft_adx_max', 999.0)  # FEB 7 2026: ADX cap
        self._ema_touch_pct: float = getattr(config, 'ft_ema_touch_pct', 0.001)
        # MAR 2026 Fix #6: ATR-adaptive touch band for A/D signals.
        # When > 0, replaces pct-based band: threshold = ema21 ± atr * mult.
        # Scales with volatility: tighter when quiet, wider on high-vol days.
        # 0.0 = disabled (falls back to ft_ema_touch_pct).
        self._ema_touch_atr_mult: float = getattr(config, 'ft_ema_touch_atr_mult', 0.0)
        # MAR 2026 T2: Regime-adaptive band — ATR boundaries for 4-bucket system.
        # Very-low (<8): tighten to 0.0010  Normal (8-13): current pct  High (13-20): ATR-scaled  Extreme (>20): ATR*1.0 (live only)
        self._atr_very_low: float = getattr(config, 'ft_atr_very_low_threshold', 8.0)
        self._atr_high: float = getattr(config, 'ft_atr_high_threshold', 13.0)
        self._atr_extreme: float = getattr(config, 'ft_atr_extreme_threshold', 20.0)
        # MAR 2026 T3: Proximity entry (Signal A-prime / D-prime)
        # Fires when price nearly touches the band but misses by ≤ proximity_gap_mult × ATR.
        # Only active in high-vol regime (ATR > _atr_high). Max 2 per day per side.
        self._proximity_enabled: bool = getattr(config, 'ft_proximity_enabled', False)
        self._proximity_gap_mult: float = getattr(config, 'ft_proximity_gap_mult', 0.3)   # gap allowed above band
        self._proximity_size_mult: float = getattr(config, 'ft_proximity_size_mult', 0.7)  # 70% position size
        self._proximity_sl_mult: float = getattr(config, 'ft_proximity_sl_mult', 0.8)      # tighter SL
        self._proximity_tp_mult: float = getattr(config, 'ft_proximity_tp_mult', 0.8)      # tighter TP
        self._proximity_max_per_day: int = getattr(config, 'ft_proximity_max_per_day', 2)
        self._proximity_long_count: int = 0
        self._proximity_short_count: int = 0
        self._or_minutes: int = getattr(config, 'ft_or_minutes', 30)

        # APR 10 2026: Medium-ATR regime block.
        # Backtest evidence: Low ATR +$203, Medium ATR (8-13) -$258, High ATR +$158.
        # Medium-vol chop has no edge for pullback/ORB signals.
        # APR 22 2026: Converted from binary toggle to 3-mode control.
        #   ft_medium_atr_block_mode:
        #     "block"    — null all A/B/D/E signals in medium-ATR (original behavior)
        #     "adx_gate" — allow signal only if ADX ≥ (ft_adx_min + ft_medium_atr_adx_bump)
        #                  requires a stronger trend to overcome chop-regime edge loss.
        #     "off"      — no medium-ATR filter (pure experimentation)
        # ft_medium_atr_block_enabled (legacy) still honored: true ⇒ block, false ⇒ off.
        _legacy_enabled = getattr(config, 'ft_medium_atr_block_enabled', True)
        self._medium_atr_block_enabled: bool = bool(_legacy_enabled)
        self._medium_atr_block_mode: str = str(
            getattr(config, 'ft_medium_atr_block_mode',
                    "block" if _legacy_enabled else "off")
        ).lower()
        self._medium_atr_adx_bump: float = float(
            getattr(config, 'ft_medium_atr_adx_bump', 7.0) or 0.0
        )

        # APR 10 2026: Day-of-week + time-of-day entry blocks.
        # Monday: -$270, 20% win rate across baseline backtest.
        # 20 UTC (3 PM CT / 4 PM ET): -$197, 0% win rate — near-close liquidity drain.
        self._monday_block_enabled: bool = getattr(config, 'ft_monday_block_enabled', True)
        self._late_afternoon_block_hour_utc: int = int(getattr(config, 'ft_late_afternoon_block_hour_utc', 20))

        # MAR 16 2026 Fix #1: A/D per-session overnight cap.
        # Signal A (EMA21_PB_LONG) and D (EMA21_PB_SHORT) have no daily counter,
        # unlike every other signal (B, E, F, G all have max_per_day guards).
        # Root cause of the 01:30 CT duplicate trade on Mar 16: second A-signal
        # fired identically to the first with nothing changed except a 90-min gap.
        # Overnight: max=1 (one pullback per direction is enough in thin market).
        # RTH: max=3 (multiple valid pullbacks can occur in a trending RTH session).
        self._ema21_pb_long_count: int = 0
        self._ema21_pb_short_count: int = 0
        self._ema21_pb_max_overnight: int = int(getattr(config, 'ft_ema21_pb_max_overnight', 1))
        self._ema21_pb_max_rth: int = int(getattr(config, 'ft_ema21_pb_max_rth', 3))

        # FEB 7 2026: EMA9 pullback parameters (Signal C)
        self._ema9_pb_enabled: bool = getattr(config, 'ft_ema9_pb_enabled', False)
        self._ema9_pb_stop_mult: float = getattr(config, 'ft_ema9_pb_stop_mult', 1.2)
        self._ema9_pb_target_mult: float = getattr(config, 'ft_ema9_pb_target_mult', 1.5)
        self._ema9_touch_pct: float = getattr(config, 'ft_ema9_touch_pct', 0.0015)  # 0.15%
        # MAR 16 2026 Fix #3: MACD floor for Signal C.
        # Was `macd_hist > 0` — fires on near-zero momentum (e.g. MACD=0.20)
        # which has no edge. Raised to 0.3 to require meaningful momentum.
        self._ema9_pb_macd_min: float = float(
            getattr(config, 'ft_ema9_pb_macd_min', 0.3) or 0.0
        )

        # FEB 12 2026: Short-side signals (D, E) — mirror of long signals
        self._shorts_enabled: bool = getattr(config, 'ft_shorts_enabled', False)
        self._short_pb_stop_mult: float = getattr(config, 'ft_short_pb_stop_mult', 1.5)
        self._short_pb_target_mult: float = getattr(config, 'ft_short_pb_target_mult', 1.0)
        self._short_or_target_r: float = getattr(config, 'ft_short_or_target_r', 1.0)
        self._or_break_short_count: int = 0  # FEB 26 2026: counter (was bool)

        # FEB 26 2026: Allow up to N OR breaks per day per side.
        # The cross condition (prev_close on other side of OR level) naturally
        # requires price to retest the OR level before re-triggering — a high-
        # probability pattern when the initial OR break confirms the trend.
        self._or_break_max_per_day: int = getattr(config, 'ft_or_break_max_per_day', 2)
        # MAY 12 2026 FIX #6: Per-direction enable for OR_BREAK signals.
        # Live data Feb 18 - May 7 2026 showed OR_BREAK_LONG with 0/5 wins,
        # -$197.32.  Disabled by default until a backtest finds a regime
        # where it has positive expectancy.
        self._or_break_long_enabled: bool = bool(
            getattr(config, 'ft_or_break_long_enabled', True)
        )
        self._or_break_short_enabled: bool = bool(
            getattr(config, 'ft_or_break_short_enabled', True)
        )
        # MAR 12 2026 Fix A: RSI guards on OR signals
        # OR_BREAK_SHORT: block if RSI < threshold (already oversold → high bounce risk)
        # OR_BREAK_LONG:  block if RSI > threshold (already overbought → high reversal risk)
        self._or_break_short_rsi_min: float = getattr(config, 'ft_or_break_short_rsi_min', 40.0)
        self._or_break_long_rsi_max: float = getattr(config, 'ft_or_break_long_rsi_max', 60.0)

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

        # MAR 10 2026: Overnight SL/TP scaling — Signals A/B/C/D/E/F
        # Outside core RTH (9:30-16:00 ET), overnight wick noise is ~1.5-2×
        # wider than RTH.  SL is widened so normal overnight swings don't stop
        # out good setups; TP is tightened because overnight moves extend less.
        # Signal G (London) is excluded — it already has its own calibrated stops.
        self._overnight_sl_mult: float = getattr(config, 'ft_overnight_sl_mult', 1.2)
        self._overnight_tp_mult: float = getattr(config, 'ft_overnight_tp_mult', 1.5)  # MAR 11 2026: raised from 0.85 → 1.5
        # MAR 11 2026 Fix #12: Hard minimum R:R for overnight trades.
        # After overnight scaling, a trade that still doesn't meet this floor is HOLD'd.
        # At 1.2× SL and 1.5× TP: R:R = (8×1.5)/(6×1.2) = 12/7.2 = 1.67:1
        # Break-even WR at 1.67:1 = 1/(1+1.67) = 37.5% — matches current live rate,
        # so the floor ensures any improvement in WR directly converts to profit.
        self._overnight_min_rr: float = float(getattr(config, 'ft_overnight_min_rr', 1.5) or 1.5)
        # MAR 16 2026 Fix #2: Slippage buffer in overnight R:R gate.
        # The gate computes R:R from `close` (signal price), but market orders
        # fill above/below close. Both Mar 16 trades were computed at R:R=1.67
        # from close but actually filled at 1.41/1.48 due to market-order slippage.
        # This adds an expected slippage buffer to SL (widens it) and subtracts
        # it from TP (shrinks it) before checking vs overnight_min_rr floor.
        # Default 0.5 pts: (12.0-0.5)/(7.2+0.5) = 11.5/7.7 = 1.49 — just below 1.5 floor.
        # Set to 0.0 to disable. Tune based on observed average slippage from trade journal.
        self._entry_slippage_pts: float = float(getattr(config, 'ft_entry_slippage_pts', 0.5) or 0.0)
        # MAR 15 2026: Overnight entry quality guards — block signals that have
        # no edge during low-liquidity hours (EVENING 16:00-23:00 ET, OVERNIGHT).
        #
        # Guard 1 — RSI extreme: TREND_CONT shorts with RSI < threshold are
        # oversold exhaustion; TREND_CONT longs with RSI > (100-threshold) are
        # overbought. Both are counter-trend re-entry traps overnight.
        # Evidence: 2 overnight TREND_CONT losses (RSI=30, RSI=32) totalling −$110.
        # Threshold raised from 30 → 35: original 30 caught neither problem trade
        # (RSI=30 strict-< boundary miss; RSI=32 above 30). 35 catches ≤34.
        self._overnight_rsi_extreme_block: float = float(
            getattr(config, 'ft_overnight_rsi_extreme_block', 35.0) or 35.0
        )
        # Guard 2 — MACD divergence: Block D/E (short) signals overnight when
        # MACD histogram is positive (momentum opposes direction). MACD was removed
        # from D/E signals globally (Mar 2026) to reduce over-filtering, but during
        # overnight low-liquidity periods a MACD divergence is a reliable reversal
        # warning that outweighs the weaker trend signal.
        # Evidence: 2 overnight D/E short losses with MACD=+1.03 and +1.63 totalling −$78.
        # Symmetric: block A/C (long) overnight when MACD is negative.
        # Set to 0.0 to disable; default 0.4 catches meaningful divergence only.
        # MAR 20 2026: Lowered 0.5 → 0.4 — EMA21_PB_SHORT with MACD_H=+0.45 slipped
        # under 0.5 threshold and stopped out (07:15 overnight trade, −$40).
        self._overnight_macd_divergence_threshold: float = float(
            getattr(config, 'ft_overnight_macd_divergence_threshold', 0.4) or 0.0
        )
        # Guard 3 — Overnight ATR floor for A/D (FIX #18, MAR 17 2026):
        # Block A/D/A-prime/D-prime pullback signals overnight when ATR < threshold.
        # Evidence: 3/3 overnight A/D fills with ATR < 6.0 were SL_HIT losers (−$110).
        # Low ATR overnight = thin liquidity, stop easily clipped by random noise.
        # The sole overnight A/D winner had ATR = 7.7.
        # Set to 0 to disable; default 6.0.
        self._overnight_min_atr_ad: float = float(
            getattr(config, 'ft_overnight_min_atr_ad', 6.0) or 0.0
        )
        # Core RTH bounds used ONLY for detecting whether to apply overnight scaling.
        # These are hardcoded to true RTH hours regardless of the session-gate config.
        self._core_rth_start: time = time(9, 30)
        self._core_rth_end: time = time(16, 0)

        # MAR 12 2026: ATR-adaptive stops for OR_BREAK signals (B/E)
        # Root cause of 3 losses today: ATR=10-12pt, fixed SL=6pt → sub-ATR stop = noise-band stop.
        # Fix: SL = clamp(ATR × 0.75, floor=6, ceiling=12), TP = SL × rr_ratio (default 1.33)
        # At ATR=10: SL=7.5pt → above noise, TP=10pt. At ATR=6: SL=6pt (floor), TP=8pt.
        self._or_break_sl_atr_mult: float = getattr(config, 'ft_or_break_sl_atr_mult', 0.75)
        self._or_break_sl_floor: float = getattr(config, 'ft_or_break_sl_floor_pts', 6.0)
        self._or_break_sl_ceiling: float = getattr(config, 'ft_or_break_sl_ceiling_pts', 12.0)
        self._or_break_rr_ratio: float = getattr(config, 'ft_or_break_rr_ratio', 1.33)

        # APR 2 2026: Anti-chase guard for OR breakout/breakdown signals
        # Block entry when bar close is already too far past the OR level.
        # Evidence: Apr 2 OR_BREAK_SHORT entered 13pts past OR_LOW (1.3×ATR) → SL hit in 19s.
        # Default 1.0×ATR max chase distance keeps entries near the breakout level.
        self._or_break_max_chase_atr: float = getattr(config, 'ft_or_break_max_chase_atr', 1.0)

        # APR 29 2026: Volume + VWAP confirmation for OR breakout/breakdown (B/E).
        # Volume_ratio is volume / 20-bar SMA (computed in feature_engineer).
        # Low-volume breakouts are common false-breakout patterns; require >= threshold
        # to confirm institutional participation. Default 0.7 (well below average — this
        # is a low bar; tune up after backtest validation against signal log).
        # Set to 0 to disable. Already-computed feature, just being read now.
        self._or_break_min_vol_ratio: float = getattr(config, 'ft_or_break_min_vol_ratio', 0.7)
        # VWAP confirmation: require close on the correct side of session VWAP.
        # For B (long break), close must be > VWAP_daily; for E (short break), close < VWAP.
        # Applied to breakouts ONLY — reversion signals (DynamicSupportFloor, A/D pullbacks)
        # are intentionally counter-VWAP and must not be filtered this way.
        self._or_break_vwap_filter_enabled: bool = bool(getattr(config, 'ft_or_break_vwap_filter', True))

        # MAR 16 2026 Fix: ATR-adaptive stops for EMA21 pullback signals (A/D)
        # Root cause: Mar 16 Trade #3 had fixed SL=6pt at ATR=10 → sub-ATR noise
        # stopped out in 2 min.  SL = clamp(ATR × 1.0, 6pt floor, 15pt ceiling).
        # At ATR=10: SL=10pt (above noise band), TP=10×1.33=13.25pt.
        # At ATR=6:  SL=6pt (floor), TP=8pt.  At ATR=18: SL=15pt (ceiling), TP=20pt.
        self._ema21_sl_atr_mult: float = getattr(config, 'ft_ema21_sl_atr_mult', 1.0)
        self._ema21_sl_floor: float = getattr(config, 'ft_ema21_sl_floor_pts', 6.0)
        self._ema21_sl_ceiling: float = getattr(config, 'ft_ema21_sl_ceiling_pts', 15.0)
        self._ema21_rr_ratio: float = getattr(config, 'ft_ema21_rr_ratio', 1.33)
        # MAY 12 2026 FIX #4: structural SL placement for Signals A & D.
        # When True, SL is placed beyond the rejection pivot (bar low for
        # longs / bar high for shorts) plus a small buffer.  When the
        # structural stop is wider than _ema21_sl_ceiling, the trade is
        # SKIPPED (returns None) rather than truncated.  See config.yaml
        # for the full rationale.
        self._ema21_sl_use_structural: bool = bool(
            getattr(config, 'ft_ema21_sl_use_structural', True)
        )
        self._ema21_sl_buffer_pts: float = float(
            getattr(config, 'ft_ema21_sl_buffer_pts', 0.5) or 0.5
        )

        # MAR 16 2026 Fix #5: MACD divergence filter for Signal A/D.
        # Root cause: Mar 16 12:30 trade — Signal A fired BUY with MACD_H=-1.40
        # (bearish momentum opposing long pullback). Lost -$57.50 in 20 min.
        # MACD was globally removed from A/D in Mar 2026 to reduce over-filtering,
        # but deep-negative MACD reliably signals "this is a slide, not a pullback".
        # Block A when MACD_H < -threshold (bearish opposes long).
        # Block D when MACD_H > +threshold (bullish opposes short).
        # MAR 19 2026: Threshold lowered 1.0 → 0.5.
    # Root cause: 11:00 RTH EMA21_PB_SHORT MACD_H=+0.95 slipped under 1.0 → SL hit.
    # Same-session 09:00 short MACD_H=-1.47 won. Meaningfully positive MACD on a
    # short pullback signals the counter-move has momentum; 0.5 catches it early
    # without blocking near-zero noise.
        # Set to 0 to disable.
        self._ema21_macd_divergence_block: float = float(
            getattr(config, 'ft_ema21_macd_divergence_block', 1.0) or 0.0
        )

        # FEB 24 2026: ATR-adaptive stops/targets for Signal F (TREND_CONT)
        # Backtest analysis (252 trades, Feb 2025 – Jan 2026):
        #   Fixed 8pt SL / 12pt TP → 50% WR, PnL = −$9 (break-even)
        #   ATR < 7:  TP=2.1×ATR (unreachable), SL=1.4×ATR → 49% WR, −$72
        #   ATR 7-10: TP=1.4×ATR, SL=1.0×ATR → 57% WR, +$247  ← sweet spot
        #   ATR 15-25: SL=0.5×ATR (noise band) → 31% WR, −$317
        # Fix: SL = clamp(ATR × 1.0, floor=6, ceiling=20)
        #      TP = SL × 1.25 (same R:R as Signal C's proven adaptive system)
        # This keeps SL ≈ 1×ATR at every vol level and TP within reach.
        self._trend_sl_atr_mult: float = getattr(config, 'ft_trend_sl_atr_mult', 1.0)
        self._trend_sl_floor: float = getattr(config, 'ft_trend_sl_floor_pts', 6.0)
        self._trend_sl_ceiling: float = getattr(config, 'ft_trend_sl_ceiling_pts', 20.0)
        self._trend_rr_ratio: float = getattr(config, 'ft_trend_rr_ratio', 1.25)
        # MAR 12 2026 Fix #16: Trend exhaustion guard for TREND_CONT signals.
        # Block TREND_CONT when session has already moved > N×ATR from OR midpoint.
        self._trend_exhaustion_atr_multiple: float = float(
            getattr(config, 'ft_trend_exhaustion_atr_multiple', 8.0) or 8.0
        )

        # MAR 16 2026 Fix #2: Post-exhaustion cooldown — when TREND_EXHAUSTION
        # fires, block ALL same-direction signals for N bars (default 4 = 60 min).
        # Root cause: Mar 16 Trade #3 at 09:45 — exhaustion fired at 09:15 but
        # only blocked TREND_CONT, not the A-signal that fired 30 min later into
        # the same exhausted trend.
        self._exhaustion_cooldown_bars: int = int(
            getattr(config, 'ft_exhaustion_cooldown_bars', 4)
        )
        # Runtime state — bars remaining in cooldown (0 = no cooldown active)
        self._exhaustion_long_bars_left: int = 0
        self._exhaustion_short_bars_left: int = 0

        # FEB 13 2026: Trend continuation signal (Signal F) — captures
        # strong rally / selloff days when price runs away from EMA21
        # without pulling back.  Uses EMA9 as dynamic support instead.
        #
        # FEB 24 2026 — ADX floor raised from 18 → 25 after deep analysis:
        #   224-trade backtest (Feb 2025 – Jan 2026, TREND_CONT only):
        #     ADX<18:  67 trades, 31% WR, −$1,243 (−$18.6/trade) ← disaster
        #     ADX 18-22: 31 trades, 42% WR, +$18 ($0.6/trade) ← break-even
        #     ADX 22-30: 50 trades, 40% WR, −$731 (−$14.6/trade)
        #     ADX≥30:  76 trades, 50% WR, +$178 ($2.3/trade) ← only edge
        #   ADX≥25 filter blocks 114/224 worst trades, saves $1,582.
        #   Today's losing trade (ADX=18.01) would have been blocked.
        #   MES-specific: ADX=18 means a directional burst just EXHAUSTED,
        #   not that a trend is continuing.  Require real trend strength.
        self._trend_cont_enabled: bool = getattr(config, 'ft_trend_cont_enabled', True)
        # MAY 12 2026 FIX #6: Per-direction enable for TREND_CONT.
        # Live data Feb 18 - May 7 2026:
        #   TREND_CONT_LONG  : 2/2 wins, +$237.91 ★ (only profitable signal)
        #   TREND_CONT_SHORT : 0/5 wins,  -$303.56
        # Disable the short side; keep the long side (the strategy's only
        # confirmed positive-expectancy signal in the live sample).
        self._trend_cont_long_enabled: bool = bool(
            getattr(config, 'ft_trend_cont_long_enabled', True)
        )
        self._trend_cont_short_enabled: bool = bool(
            getattr(config, 'ft_trend_cont_short_enabled', True)
        )
        self._trend_cont_stop_mult: float = getattr(config, 'ft_trend_cont_stop_mult', 1.0)
        self._trend_cont_target_mult: float = getattr(config, 'ft_trend_cont_target_mult', 2.0)
        self._trend_cont_adx_min: float = getattr(config, 'ft_trend_cont_adx_min', 25.0)
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

        # MAR 9 2026: Signal G — London Momentum Breakout
        # During London session (2-5 AM CST / 3-6 AM ET / 8-11 AM GMT),
        # MES sits in tight range (ATR 2-4pts, ADX 16-18) until European
        # liquidity injects momentum.  Pullback signals (A/D) fail because
        # EMAs are too tight, and OR signals (B/E) can't fire (no OR yet).
        #
        # Signal G detects the FIRST directional impulse after London open:
        #   - EMA9 crosses EMA21 (momentum shift in low-vol environment)
        #   - ADX rising (was <18, now ≥ threshold) — confirms breakout from chop
        #   - Close is beyond EMA9 in cross direction (momentum confirmed)
        #   - Bullish/bearish bar (directional candle)
        #   - Max 1 per day (first impulse only — don't chase)
        #   - Tighter SL/TP for low-vol environment
        self._london_enabled: bool = getattr(config, 'ft_london_enabled', False)
        self._london_start_ct: time = time(
            getattr(config, 'ft_london_start_hour', 2),
            getattr(config, 'ft_london_start_minute', 0)
        )
        self._london_end_ct: time = time(
            getattr(config, 'ft_london_end_hour', 5),
            getattr(config, 'ft_london_end_minute', 0)
        )
        self._london_adx_min: float = getattr(config, 'ft_london_adx_min', 15.0)
        self._london_sl_points: float = getattr(config, 'ft_london_sl_points', 5.0)   # Floor SL (pts from close)
        self._london_sl_atr_cap: float = getattr(config, 'ft_london_sl_atr_cap', 1.0) # Cap SL at N×ATR
        self._london_tp_points: float = getattr(config, 'ft_london_tp_points', 8.0)   # TP — above median 8.2pt London move
        self._london_max_per_day: int = getattr(config, 'ft_london_max_per_day', 1)
        self._london_fired_count: int = 0

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

        # APR 22 2026 Fix #1: PDH proximity block — configurable.
        # Previously hardcoded 0.25% (~18pt on ES). On range-bound days where price
        # consolidates below PDH, that filter vetoes every long pullback.
        # Now two-stage:
        #   - Tight zone (< ft_pdh_proximity_block_pct): always block when close ≤ PDH
        #   - Wide  zone (< ft_pdh_proximity_rsi_block_pct): block only when RSI ≥ rsi_max
        # Set ft_pdh_proximity_block_enabled: false to disable entirely.
        self._pdh_proximity_block_enabled: bool = bool(
            getattr(config, 'ft_pdh_proximity_block_enabled', True)
        )
        self._pdh_proximity_block_pct: float = float(
            getattr(config, 'ft_pdh_proximity_block_pct', 0.0010) or 0.0
        )
        self._pdh_proximity_rsi_block_pct: float = float(
            getattr(config, 'ft_pdh_proximity_rsi_block_pct', 0.0050) or 0.0
        )
        self._pdh_proximity_rsi_max: float = float(
            getattr(config, 'ft_pdh_proximity_rsi_max', 70.0) or 70.0
        )

        # MAY 12 2026 FIX #5: Higher-timeframe (30m) trend filter.
        # The 30m trend is computed and refreshed in the live_trading_manager
        # via direct IB 30m bar requests, then injected into features.attrs
        # by the signal_processor.  This strategy only consumes the result.
        # Modes:
        #   "block_counter"   — block signals against the HTF trend (default)
        #   "require_aligned" — additionally block when HTF is NEUTRAL
        #   "off"             — disabled
        self._htf_filter_enabled: bool = bool(
            getattr(config, 'ft_htf_filter_enabled', True)
        )
        self._htf_filter_mode: str = str(
            getattr(config, 'ft_htf_filter_mode', 'block_counter') or 'block_counter'
        ).lower()

        # MAR 10 2026: Load persisted counters from previous run (same CME session)
        self._load_counters()

    # ------------------------------------------------------------------
    #  Higher-TF trend filter (MAY 12 2026 FIX #5)
    # ------------------------------------------------------------------
    def _htf_blocks_signal(self, action: str, features: pd.DataFrame) -> Optional[str]:
        """Return a reason string if HTF filter blocks this signal, else None.

        Reads htf_30m_trend from features.attrs (populated by signal_processor
        from the live_trading_manager's IB-fetched 30m trend).  Counter-trend
        blocking only — does not fire if HTF data is unknown (graceful
        degradation when the bot is freshly started or IB is unreachable).
        """
        if not self._htf_filter_enabled or self._htf_filter_mode == "off":
            return None
        try:
            trend = str(features.attrs.get("htf_30m_trend", "UNKNOWN")).upper()
        except Exception:
            return None
        if trend == "UNKNOWN":
            # No HTF data yet — fail open (don't block).  Once IB poll
            # populates the trend, subsequent signals will be filtered.
            return None

        is_buy = action.upper() == "BUY"
        is_sell = action.upper() == "SELL"

        if is_buy:
            if trend == "DOWN":
                return f"HTF_30m_DOWN — counter-trend long blocked"
            if trend == "NEUTRAL" and self._htf_filter_mode == "require_aligned":
                return f"HTF_30m_NEUTRAL — strict-align mode blocks long"
        elif is_sell:
            if trend == "UP":
                return f"HTF_30m_UP — counter-trend short blocked"
            if trend == "NEUTRAL" and self._htf_filter_mode == "require_aligned":
                return f"HTF_30m_NEUTRAL — strict-align mode blocks short"
        return None

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

        # ---- Extract indicators (needed for Signal G before RTH gate) ----
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
        pdh = float(latest.get("PDH", 0))
        # APR 29 2026: Volume + VWAP for OR breakout confirmation (Signals B/E).
        # Defaults are NEUTRAL (1.0 ratio = at-average, vwap=close = no preference)
        # so missing-data bars don't get auto-blocked.
        volume_ratio = float(latest.get("volume_ratio", 1.0))
        if np.isnan(volume_ratio):
            volume_ratio = 1.0
        vwap_daily = float(latest.get("VWAP_daily", close))
        if np.isnan(vwap_daily) or vwap_daily <= 0:
            vwap_daily = close

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

        # ---- Signal G: London Momentum Breakout (before RTH gate) ----
        # Evaluated first because it fires during 2-5 AM CST (outside RTH).
        # The method itself time-gates to London hours only.
        signal_g = self._check_london_momentum(
            enriched, current_time,
            close, open_price, low, high,
            ema9, ema21, ema50, atr, adx,
        )
        if signal_g is not None:
            action, stop_loss, take_profit, reason = signal_g
            is_short = action == "SELL"
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
                "market_state": "LONDON_MOMENTUM_SHORT" if is_short else "LONDON_MOMENTUM_LONG",
                "position_size": 1.0,
                "position_scaler": 1.0,
                "entry_type": "london_momentum",
                "session_type": "LONDON",
            }
            logger.info(
                f"📊 {current_time} | {action}: close={close:.2f} "
                f"ema9={ema9:.2f} ema21={ema21:.2f} atr={atr:.1f} adx={adx:.0f} "
                f"SL={stop_loss:.2f} TP={take_profit:.2f} | {reason}"
            )
            # MAY 12 2026 FIX #5: HTF filter applies to Signal G too (defensive
            # — Signal G is disabled by Fix #2 today but if re-enabled it
            # should also respect higher-TF discipline).
            _htf_block_g = self._htf_blocks_signal(action, features)
            if _htf_block_g is not None:
                logger.info(
                    f"🚫 HTF_BLOCK (Signal G): {action} blocked — {_htf_block_g}"
                )
                self._prev_close = close
                return Signal(
                    "HOLD",
                    0.0,
                    {"reason": f"HTF_BLOCK | G | {_htf_block_g}"},
                )
            self._prev_close = close
            return Signal(action=action, confidence=0.7, metadata=metadata)

        # ---- Must be within RTH ----
        if not (self._rth_start <= et_time.time() < self._rth_end):
            self._prev_close = float(latest["close"])
            return Signal("HOLD", 0.0, {"reason": "OUTSIDE_RTH"})

        # ---- Entry time filter (FEB 7 2026) ----
        # Skip hours with negative expectancy (10:xx = -$1,208, 15:xx = -$397)
        if not (self._entry_start_et <= et_time.time() < self._entry_end_et):
            self._prev_close = float(latest["close"])
            return Signal("HOLD", 0.0, {"reason": "OUTSIDE_ENTRY_WINDOW"})

        # (Indicators already extracted above, before Signal G / RTH gate)

        # MAR 16 2026 Fix #1: Compute overnight flag for A/D cap.
        # True when outside core RTH (9:30-16:00 ET). Same boundary used for
        # overnight scaling and the overnight guards block below.
        _is_overnight_pb = not (self._core_rth_start <= et_time.time() < self._core_rth_end)

        # ── MAR 16 2026 Fix #2: Post-exhaustion cooldown tick ───────────────
        # Decrement cooldown counters each bar. When TREND_EXHAUSTION fires
        # (inside _check_trend_continuation_long/short), we set the counter
        # to _exhaustion_cooldown_bars. While > 0, same-direction signals
        # are nulled after signal evaluation (see block below priority select).
        if self._exhaustion_long_bars_left > 0:
            self._exhaustion_long_bars_left -= 1
        if self._exhaustion_short_bars_left > 0:
            self._exhaustion_short_bars_left -= 1

        # Pre-compute exhaustion state for this bar (used to set cooldowns
        # when TREND_CONT exhaustion fires in check methods AND to block
        # same-direction non-TREND_CONT signals in the priority block below).
        _exh_mult = self._trend_exhaustion_atr_multiple
        _exh_cooldown_n = self._exhaustion_cooldown_bars
        _exh_or_ok = (_exh_cooldown_n > 0 and self._or_computed
                      and self._or_high > 0 and atr > 0 and _exh_mult > 0)
        if _exh_or_ok:
            _exh_ref = (self._or_high + self._or_low) / 2.0
            _exh_move = close - _exh_ref
            _exh_atr_ratio = abs(_exh_move) / atr
            if _exh_atr_ratio > _exh_mult:
                if _exh_move > 0 and self._exhaustion_long_bars_left == 0:
                    self._exhaustion_long_bars_left = self._exhaustion_cooldown_bars
                    logger.info(
                        f"🚫 EXHAUSTION_COOLDOWN: LONG exhaustion detected | "
                        f"move={_exh_move:+.1f}pts ({_exh_atr_ratio:.1f}×ATR) "
                        f"— blocking BUY signals for {self._exhaustion_cooldown_bars} bars"
                    )
                elif _exh_move < 0 and self._exhaustion_short_bars_left == 0:
                    self._exhaustion_short_bars_left = self._exhaustion_cooldown_bars
                    logger.info(
                        f"🚫 EXHAUSTION_COOLDOWN: SHORT exhaustion detected | "
                        f"move={_exh_move:+.1f}pts ({_exh_atr_ratio:.1f}×ATR) "
                        f"— blocking SELL signals for {self._exhaustion_cooldown_bars} bars"
                    )

        # ── Price momentum sentiment (APR 27 2026) ──────────────────────────
        # Update market-structure sentiment each bar so combined sentiment
        # reflects actual price structure, not just social media noise.
        if _PRICE_MOMENTUM_AVAILABLE:
            try:
                update_price_momentum(close, ema9, ema21, ema50, adx)
            except Exception:
                pass

        # ---- Signal A: EMA21 Pullback Long ----
        signal_a = self._check_ema21_pullback(
            close, open_price, low, ema21, ema50, atr, adx,
            rsi, macd_hist, is_overnight=_is_overnight_pb, pdh=pdh,
        )

        # ---- Signal A-prime: EMA21 Proximity Long (near-miss, high-vol only) ----
        signal_aprox = None
        if signal_a is None:  # only evaluate when A missed
            signal_aprox = self._check_ema21_proximity_long(
                close, open_price, low, ema21, ema50, atr, adx,
            )

        # ---- Signal B: OR Breakout Long ----
        # MAY 12 2026 FIX #6: per-direction enable.
        signal_b = None
        if self._or_break_long_enabled:
            signal_b = self._check_or_breakout(
                close, high, ema9, ema21, atr, adx, macd_hist, rsi,
                volume_ratio=volume_ratio, vwap_daily=vwap_daily,
            )

        # ---- Signal C: EMA9 Pullback Long (faster trend) ----
        signal_c = None
        if self._ema9_pb_enabled:
            signal_c = self._check_ema9_pullback(
                close, open_price, low, ema9, ema21, ema50, atr, adx, rsi,
                macd_hist, pdh=pdh,
            )

        # ---- Signal D: EMA21 Pullback Short (downtrend mirror of A) ----
        signal_d = None
        signal_dprox = None
        if self._shorts_enabled:
            signal_d = self._check_ema21_pullback_short(
                close, open_price, high, ema21, ema50, atr, adx,
                rsi, macd_hist, is_overnight=_is_overnight_pb,
            )
            if signal_d is None:  # only evaluate when D missed
                signal_dprox = self._check_ema21_proximity_short(
                    close, open_price, high, ema21, ema50, atr, adx,
                )

        # ---- Signal E: OR Breakdown Short (downtrend mirror of B) ----
        # MAY 12 2026 FIX #6: also gated by per-direction enable for symmetry.
        signal_e = None
        if self._shorts_enabled and self._or_break_short_enabled:
            signal_e = self._check_or_breakdown(
                close, low, ema9, ema21, atr, adx, macd_hist, rsi,
                volume_ratio=volume_ratio, vwap_daily=vwap_daily,
            )

        # ---- Signal F: Trend Continuation (strong momentum days) ----
        # FEB 13 2026: Catches rallies/selloffs where price runs away
        # from EMA21 without pulling back.  Uses EMA9 as dynamic support.
        signal_f_long = None
        signal_f_short = None
        if self._trend_cont_enabled:
            # MAY 12 2026 FIX #6: per-direction enable.
            if self._trend_cont_long_enabled:
                signal_f_long = self._check_trend_continuation_long(
                    enriched, close, open_price, low, high,
                    ema9, ema21, ema50, atr, adx, rsi, macd_hist,
                )
            if (self._shorts_enabled
                    and self._trend_cont_short_enabled
                    and signal_f_long is None):
                signal_f_short = self._check_trend_continuation_short(
                    enriched, close, open_price, low, high,
                    ema9, ema21, ema50, atr, adx, rsi, macd_hist,
                )

        # ── Overnight entry-quality guards (MAR 15 2026) ────────────────────
        # During low-liquidity hours (outside core RTH 9:30-16:00 ET), two
        # additional filters protect against signal patterns with no overnight edge.
        _et_t_guard = et_time.time()
        _is_core_rth_guard = self._core_rth_start <= _et_t_guard < self._core_rth_end
        if not _is_core_rth_guard:
            _rsi_ext = self._overnight_rsi_extreme_block  # default 30
            _macd_thr = self._overnight_macd_divergence_threshold  # default 0.5

            # Guard 1: RSI extreme — TREND_CONT exhaustion trap overnight.
            # SELL continuation at RSI < threshold = already oversold, not continuing.
            # BUY  continuation at RSI > (100-threshold) = already overbought.
            if _rsi_ext > 0:
                if signal_f_short is not None and rsi < _rsi_ext:
                    logger.info(
                        f"🚫 ON_RSI_EXTREME: blocking TREND_CONT_SHORT overnight "
                        f"(RSI={rsi:.1f} < {_rsi_ext}) | oversold exhaustion trap"
                    )
                    signal_f_short = None
                if signal_f_long is not None and rsi > (100.0 - _rsi_ext):
                    logger.info(
                        f"🚫 ON_RSI_EXTREME: blocking TREND_CONT_LONG overnight "
                        f"(RSI={rsi:.1f} > {100.0 - _rsi_ext:.0f}) | overbought exhaustion trap"
                    )
                    signal_f_long = None

            # Guard 2: MACD divergence — D/E/A short (long) with opposing momentum overnight.
            # MACD removed from D/E globally (Mar 2026) to reduce RTH over-filtering.
            # Overnight, a positive MACD on a SHORT entry is a reliable reversal warning.
            if _macd_thr > 0:
                if signal_d is not None and macd_hist > _macd_thr:
                    logger.info(
                        f"🚫 ON_MACD_DIVERGE: blocking D-short overnight "
                        f"(MACD={macd_hist:+.2f} > +{_macd_thr}) | momentum opposes direction"
                    )
                    signal_d = None
                    signal_dprox = None  # proximity variant blocked together
                elif signal_dprox is not None and macd_hist > _macd_thr:
                    logger.info(
                        f"🚫 ON_MACD_DIVERGE: blocking D-prime-short overnight "
                        f"(MACD={macd_hist:+.2f} > +{_macd_thr}) | momentum opposes direction"
                    )
                    signal_dprox = None
                if signal_e is not None and macd_hist > _macd_thr:
                    logger.info(
                        f"🚫 ON_MACD_DIVERGE: blocking E-short overnight "
                        f"(MACD={macd_hist:+.2f} > +{_macd_thr}) | momentum opposes direction"
                    )
                    signal_e = None
                # Symmetric: block A/C long when MACD is negative overnight
                if signal_a is not None and macd_hist < -_macd_thr:
                    logger.info(
                        f"🚫 ON_MACD_DIVERGE: blocking A-long overnight "
                        f"(MACD={macd_hist:+.2f} < -{_macd_thr}) | momentum opposes direction"
                    )
                    signal_a = None
                if signal_c is not None and macd_hist < -_macd_thr:
                    logger.info(
                        f"🚫 ON_MACD_DIVERGE: blocking C-long overnight "
                        f"(MACD={macd_hist:+.2f} < -{_macd_thr}) | momentum opposes direction"
                    )
                    signal_c = None
                # MAR 20 2026: Extend Guard 2 to TREND_CONT (signal F).
                # Evidence: 06:45 TREND_CONT_SHORT bypassed guard entirely; price ground up.
                if signal_f_short is not None and macd_hist > _macd_thr:
                    logger.info(
                        f"🚫 ON_MACD_DIVERGE: blocking F-short overnight "
                        f"(MACD={macd_hist:+.2f} > +{_macd_thr}) | momentum opposes direction"
                    )
                    signal_f_short = None
                if signal_f_long is not None and macd_hist < -_macd_thr:
                    logger.info(
                        f"🚫 ON_MACD_DIVERGE: blocking F-long overnight "
                        f"(MACD={macd_hist:+.2f} < -{_macd_thr}) | momentum opposes direction"
                    )
                    signal_f_long = None

            # Guard 3: ATR floor for A/D overnight (FIX #18, MAR 17 2026).
            # Low ATR overnight = thin liquidity, SL easily clipped by noise.
            # Evidence: 3/3 overnight A/D fills with ATR < 6.0 were SL_HIT losers (−$110).
            _atr_floor = self._overnight_min_atr_ad
            if _atr_floor > 0:
                if signal_a is not None and atr < _atr_floor:
                    logger.info(
                        f"🚫 ON_ATR_FLOOR: blocking A-long overnight "
                        f"(ATR={atr:.1f} < {_atr_floor:.1f}) | thin liquidity, SL noise risk"
                    )
                    signal_a = None
                if signal_aprox is not None and atr < _atr_floor:
                    logger.info(
                        f"🚫 ON_ATR_FLOOR: blocking A-prime-long overnight "
                        f"(ATR={atr:.1f} < {_atr_floor:.1f}) | thin liquidity, SL noise risk"
                    )
                    signal_aprox = None
                if signal_d is not None and atr < _atr_floor:
                    logger.info(
                        f"🚫 ON_ATR_FLOOR: blocking D-short overnight "
                        f"(ATR={atr:.1f} < {_atr_floor:.1f}) | thin liquidity, SL noise risk"
                    )
                    signal_d = None
                if signal_dprox is not None and atr < _atr_floor:
                    logger.info(
                        f"🚫 ON_ATR_FLOOR: blocking D-prime-short overnight "
                        f"(ATR={atr:.1f} < {_atr_floor:.1f}) | thin liquidity, SL noise risk"
                    )
                    signal_dprox = None

        # ── MAR 16 2026 Fix #2: Post-exhaustion cooldown block ─────────────
        # When exhaustion cooldown is active, null ALL same-direction signals.
        if self._exhaustion_long_bars_left > 0:
            _blocked = [s for s in ["A", "A'", "C", "B", "F_long"]
                        if {"A": signal_a, "A'": signal_aprox, "C": signal_c,
                            "B": signal_b, "F_long": signal_f_long}.get(s) is not None]
            if _blocked:
                logger.info(
                    f"🚫 EXHAUSTION_COOLDOWN: blocking BUY signals {_blocked} "
                    f"({self._exhaustion_long_bars_left} bars remaining)"
                )
            signal_a = signal_aprox = signal_c = signal_b = signal_f_long = None
        if self._exhaustion_short_bars_left > 0:
            _blocked = [s for s in ["D", "D'", "E", "F_short"]
                        if {"D": signal_d, "D'": signal_dprox,
                            "E": signal_e, "F_short": signal_f_short}.get(s) is not None]
            if _blocked:
                logger.info(
                    f"🚫 EXHAUSTION_COOLDOWN: blocking SELL signals {_blocked} "
                    f"({self._exhaustion_short_bars_left} bars remaining)"
                )
            signal_d = signal_dprox = signal_e = signal_f_short = None

        # ── APR 10 2026: Medium-ATR regime filter (3-mode APR 22 2026) ────
        # Backtest: Medium ATR (8-13) = -$258, 35% WR. Proximity (A'/D')
        # already gated to high-vol only. Trend continuation (F) and London (G)
        # are exempt — different edge.
        #   mode "block"    — null A/B/D/E in medium regime (original)
        #   mode "adx_gate" — allow only if ADX ≥ (ft_adx_min + bump). Strong trends
        #                     can override chop-regime edge loss.
        #   mode "off"      — no filter
        if self._medium_atr_block_mode != "off" and self._atr_very_low <= atr < self._atr_high:
            if self._medium_atr_block_mode == "block":
                _blocked_med = [n for n, s in [("A", signal_a), ("B", signal_b),
                                                ("D", signal_d), ("E", signal_e)]
                                if s is not None]
                if _blocked_med:
                    logger.info(
                        f"🚫 MEDIUM_ATR_BLOCK: ATR={atr:.1f} in [{self._atr_very_low:.0f}, "
                        f"{self._atr_high:.0f}) — blocking {_blocked_med}"
                    )
                    signal_a = signal_b = signal_d = signal_e = None
            elif self._medium_atr_block_mode == "adx_gate":
                _adx_req = self._adx_min + self._medium_atr_adx_bump
                if adx < _adx_req:
                    _gated = [n for n, s in [("A", signal_a), ("B", signal_b),
                                              ("D", signal_d), ("E", signal_e)]
                              if s is not None]
                    if _gated:
                        logger.info(
                            f"🚫 MEDIUM_ATR_ADX_GATE: ATR={atr:.1f} in medium regime, "
                            f"ADX={adx:.0f}<{_adx_req:.0f} (min+{self._medium_atr_adx_bump:.0f}) "
                            f"— blocking {_gated}"
                        )
                        signal_a = signal_b = signal_d = signal_e = None
                else:
                    # Passed the ADX gate — allow but annotate for downstream logging
                    logger.info(
                        f"⚠️ MEDIUM_ATR_ADX_PASS: ATR={atr:.1f} in medium regime but "
                        f"ADX={adx:.0f}≥{_adx_req:.0f} — allowing A/B/D/E"
                    )

        # ── APR 10 2026: Monday entry block ──────────────────────────────
        # Backtest Mon P&L -$270, 20% WR. Entire day is negative expectancy.
        if self._monday_block_enabled and et_time.weekday() == 0:
            _any_active = any(s is not None for s in [
                signal_a, signal_aprox, signal_b, signal_c,
                signal_d, signal_dprox, signal_e,
                signal_f_long, signal_f_short, signal_g,
            ])
            if _any_active:
                logger.info(
                    f"🚫 MONDAY_BLOCK: blocking all signals on Monday "
                    f"({et_time.strftime('%Y-%m-%d')})"
                )
                signal_a = signal_aprox = signal_b = signal_c = None
                signal_d = signal_dprox = signal_e = None
                signal_f_long = signal_f_short = signal_g = None

        # ── APR 10 2026: Late-afternoon entry block ──────────────────────
        # Backtest 20 UTC (3 PM CT / 4 PM ET): -$197, 0% WR.
        # Block new entries at or after this hour; existing positions can run.
        if self._late_afternoon_block_hour_utc > 0:
            _bar_utc_hour = current_time.astimezone(
                __import__('zoneinfo').ZoneInfo('UTC')
            ).hour if current_time.tzinfo else current_time.hour
            if _bar_utc_hour >= self._late_afternoon_block_hour_utc:
                _any_active = any(s is not None for s in [
                    signal_a, signal_aprox, signal_b, signal_c,
                    signal_d, signal_dprox, signal_e,
                    signal_f_long, signal_f_short, signal_g,
                ])
                if _any_active:
                    logger.info(
                        f"🚫 LATE_AFTERNOON_BLOCK: bar hour={_bar_utc_hour} UTC "
                        f">= {self._late_afternoon_block_hour_utc} — blocking all signals"
                    )
                    signal_a = signal_aprox = signal_b = signal_c = None
                    signal_d = signal_dprox = signal_e = None
                    signal_f_long = signal_f_short = signal_g = None

        # Priority: A (EMA21 PB Long) > A-prime (proximity long)
        #         > C (EMA9 PB Long) > B (OR breakout Long) > F_long
        #         > D (EMA21 PB Short) > D-prime (proximity short)
        #         > E (OR breakdown Short) > F_short
        # Long signals take priority over short signals.
        chosen = None
        chosen_is_proximity = False
        if signal_a is not None:
            chosen = signal_a
        elif signal_aprox is not None:
            chosen = signal_aprox
            chosen_is_proximity = True
        elif signal_c is not None:
            chosen = signal_c
        elif signal_b is not None:
            chosen = signal_b
        elif signal_f_long is not None:
            chosen = signal_f_long
        elif signal_d is not None:
            chosen = signal_d
        elif signal_dprox is not None:
            chosen = signal_dprox
            chosen_is_proximity = True
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
                _touch = self._regime_touch_threshold(ema21, atr, side="long")
                if low > _touch:
                    _diag_parts.append(f"A:low({low:.1f})>touch({_touch:.1f})")
                elif close <= ema21:
                    _diag_parts.append(f"A:close({close:.1f})<=ema21({ema21:.1f})")
                elif close <= open_price:
                    _diag_parts.append(f"A:bearish(c={close:.1f},o={open_price:.1f})")
                elif adx < self._adx_min or adx > self._adx_max:
                    _diag_parts.append(f"A:adx({adx:.0f})out[{self._adx_min}-{self._adx_max}]")
                elif self._ema21_macd_divergence_block > 0 and macd_hist < -self._ema21_macd_divergence_block:
                    _diag_parts.append(f"A:macd_div({macd_hist:.2f})<-{self._ema21_macd_divergence_block:.1f}")
            # Signal B diagnostics (Long OR Breakout)
            if not self._or_computed or self._or_high <= 0:
                _diag_parts.append(f"B:no_OR(computed={self._or_computed},h={self._or_high:.1f})")
            elif self._or_break_long_count >= self._or_break_max_per_day:
                _diag_parts.append(f"B:maxed({self._or_break_long_count})")
            elif not (close > self._or_high and self._prev_close <= self._or_high):
                _diag_parts.append(f"B:no_cross(c={close:.1f},prev={self._prev_close:.1f},OR_H={self._or_high:.1f})")
            elif ema9 <= ema21:
                _diag_parts.append(f"B:ema9({ema9:.1f})<=ema21({ema21:.1f})")
            elif adx < self._adx_min:
                _diag_parts.append(f"B:adx({adx:.0f})<{self._adx_min}")
            # Signal C diagnostics (Long EMA9 PB — fast trend)
            if self._ema9_pb_enabled:
                if not (ema9 > ema21 > ema50):
                    _diag_parts.append(f"C:stack(e9={ema9:.0f},e21={ema21:.0f},e50={ema50:.0f})")
                else:
                    _c_touch = ema9 * (1 + self._ema9_touch_pct)
                    _ema21_c_touch = ema21 * (1 + self._ema_touch_pct)
                    if low > _c_touch:
                        _diag_parts.append(f"C:low({low:.1f})>touch({_c_touch:.1f})")
                    elif close <= ema9:
                        _diag_parts.append(f"C:close({close:.1f})<=ema9({ema9:.1f})")
                    elif close <= open_price:
                        _diag_parts.append(f"C:bearish(c={close:.1f},o={open_price:.1f})")
                    elif adx < 22 or adx > self._adx_max:
                        _diag_parts.append(f"C:adx({adx:.0f})out[22-{self._adx_max}]")
                    elif rsi > 70 or rsi < 40:
                        _diag_parts.append(f"C:rsi({rsi:.0f})out[40-70]")
                    elif low <= _ema21_c_touch:
                        _diag_parts.append(f"C:ema21_overlap(low={low:.1f}<=touch21={_ema21_c_touch:.1f})")
                    elif macd_hist < self._ema9_pb_macd_min:
                        _diag_parts.append(f"C:macd({macd_hist:.2f})<{self._ema9_pb_macd_min:.1f}")
            # Signal D diagnostics (Short EMA21 PB)
            if self._shorts_enabled:
                if ema21 >= ema50:
                    _diag_parts.append(f"D:ema21({ema21:.1f})>=ema50({ema50:.1f})")
                else:
                    _touch_s = self._regime_touch_threshold(ema21, atr, side="short")
                    if high < _touch_s:
                        _diag_parts.append(f"D:high({high:.1f})<touch({_touch_s:.1f})")
                    elif close >= ema21:
                        _diag_parts.append(f"D:close({close:.1f})>=ema21({ema21:.1f})")
                    elif close >= open_price:
                        _diag_parts.append(f"D:bullish(c={close:.1f},o={open_price:.1f})")
                    elif adx < self._adx_min or adx > self._adx_max:
                        _diag_parts.append(f"D:adx({adx:.0f})out[{self._adx_min}-{self._adx_max}]")
                    elif self._ema21_macd_divergence_block > 0 and macd_hist > self._ema21_macd_divergence_block:
                        _diag_parts.append(f"D:macd_div({macd_hist:.2f})>+{self._ema21_macd_divergence_block:.1f}")
            else:
                _diag_parts.append("D:shorts_disabled")
            # Signal A-prime / D-prime proximity diagnostics (only in high-vol)
            if self._proximity_enabled and atr >= self._atr_high:
                _tlong = self._regime_touch_threshold(ema21, atr, side="long")
                _prox_outer = _tlong + self._proximity_gap_mult * atr
                if low > _tlong and low <= _prox_outer and ema21 > ema50:
                    _diag_parts.append(
                        f"Aprox:near(low={low:.1f} thresh={_tlong:.1f} outer={_prox_outer:.1f})"
                    )
            # Signal E diagnostics (Short OR Breakdown)
            if self._shorts_enabled:
                if not self._or_computed or self._or_low <= 0:
                    _diag_parts.append(f"E:no_OR(computed={self._or_computed},l={self._or_low:.1f})")
                elif self._or_break_short_count >= self._or_break_max_per_day:
                    _diag_parts.append(f"E:maxed({self._or_break_short_count})")
                elif not (close < self._or_low and self._prev_close >= self._or_low):
                    _diag_parts.append(f"E:no_cross(c={close:.1f},prev={self._prev_close:.1f},OR_L={self._or_low:.1f})")
                elif ema9 >= ema21:
                    _diag_parts.append(f"E:ema9({ema9:.1f})>=ema21({ema21:.1f})")
                elif adx < self._adx_min:
                    _diag_parts.append(f"E:adx({adx:.0f})<{self._adx_min}")
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
            # Signal G diagnostics (London Momentum)
            if self._london_enabled:
                ct_time_diag = self._to_ct(current_time)
                ct_t_diag = ct_time_diag.time()
                if self._london_start_ct <= ct_t_diag < self._london_end_ct:
                    if self._london_fired_count >= self._london_max_per_day:
                        _diag_parts.append(f"G:maxed({self._london_fired_count})")
                    elif len(enriched) >= 2:
                        _pe9 = float(enriched.iloc[-2].get("EMA_9", 0))
                        _pe21 = float(enriched.iloc[-2].get("EMA_21", 0))
                        _bcross = _pe9 <= _pe21 and ema9 > ema21
                        _scross = _pe9 >= _pe21 and ema9 < ema21
                        if not _bcross and not _scross:
                            _diag_parts.append(f"G:no_cross(e9={ema9:.1f},e21={ema21:.1f},pe9={_pe9:.1f},pe21={_pe21:.1f})")
                        elif adx < self._london_adx_min:
                            _diag_parts.append(f"G:adx({adx:.0f})<{self._london_adx_min:.0f}")
                        else:
                            _diag_parts.append(f"G:candle_check(c={close:.1f},o={open_price:.1f},e9={ema9:.1f})")
                else:
                    _diag_parts.append(f"G:outside_london({ct_t_diag})")
            _diag = " | ".join(_diag_parts) if _diag_parts else "unknown"
            logger.info(f"🔍 NO_SIGNAL diag: {_diag}")
            # APR 22 2026 Fix #2: Structured shadow log for post-hoc analysis
            self._log_shadow_decision(
                ts=current_time, outcome="HOLD",
                close=close, ema9=ema9, ema21=ema21, ema50=ema50,
                atr=atr, adx=adx, rsi=rsi, macd_hist=macd_hist,
                pdh=pdh, or_high=self._or_high, or_low=self._or_low,
                action=None, stop_loss=None, take_profit=None,
                reason="NO_SIGNAL", gate_diag=_diag,
                counters={
                    "A": self._ema21_pb_long_count, "D": self._ema21_pb_short_count,
                    "B": self._or_break_long_count, "E": self._or_break_short_count,
                    "F_L": self._trend_cont_long_count, "F_S": self._trend_cont_short_count,
                    "A'": self._proximity_long_count, "D'": self._proximity_short_count,
                    "G": self._london_fired_count,
                },
            )
            self._prev_close = close
            return Signal("HOLD", 0.0, {"reason": "NO_SIGNAL"})

        action, stop_loss, take_profit, reason = chosen

        # MAY 12 2026 FIX #5: Higher-timeframe (30m) trend filter.
        # Block counter-trend signals — won't take a 15m long when the 30m
        # is trending down (or vice versa).  Reads htf_30m_trend from
        # features.attrs (populated by signal_processor from live IB 30m
        # bars).  Fails open if trend is UNKNOWN (no data yet).
        _htf_block_reason = self._htf_blocks_signal(action, features)
        if _htf_block_reason is not None:
            logger.info(
                f"🚫 HTF_BLOCK: {action} blocked — {_htf_block_reason} "
                f"| original signal: {reason}"
            )
            self._prev_close = close
            return Signal(
                "HOLD",
                0.0,
                {"reason": f"HTF_BLOCK | {_htf_block_reason} | was: {reason}"},
            )

        # ── Overnight SL/TP scaling ──────────────────────────────────────────
        # Signals A/B/C/D/E/F are calibrated for RTH liquidity.  Outside core
        # RTH (9:30-16:00 ET) the noise band is wider and moves extend less:
        #   SL × overnight_sl_mult (1.2) — survive overnight wicks
        #   TP × overnight_tp_mult (1.5) — wider TP to match and improve R:R
        # Signal G is excluded — it fires before this block and returns early.
        _et_t = et_time.time()
        _is_core_rth = self._core_rth_start <= _et_t < self._core_rth_end
        if not _is_core_rth and (self._overnight_sl_mult != 1.0 or self._overnight_tp_mult != 1.0):
            _sl_pts = abs(close - stop_loss)
            _tp_pts = abs(take_profit - close)
            if action == "BUY":
                stop_loss = close - _sl_pts * self._overnight_sl_mult
                take_profit = close + _tp_pts * self._overnight_tp_mult
            else:  # SELL
                stop_loss = close + _sl_pts * self._overnight_sl_mult
                take_profit = close - _tp_pts * self._overnight_tp_mult
            reason = f"{reason} | ONAdj(SL×{self._overnight_sl_mult:.2f},TP×{self._overnight_tp_mult:.2f})"

        # MAR 11 2026 Fix #12: Overnight minimum R:R gate.
        # After scaling, compute actual R:R and block trades that don't meet the floor.
        # This catches edge cases where ATR-adaptive SL (signals C/F) or proximity
        # signals (A-prime/D-prime) produce a sub-floor R:R even after overnight scaling.
        # MAR 16 2026 Fix #2: Add slippage buffer before comparing to floor.
        # Market orders fill above/below close; gate computed from close overstates R:R.
        # ft_entry_slippage_pts (default 0.5) widens effective SL and shrinks effective TP.
        if not _is_core_rth and self._overnight_min_rr > 0:
            _slippage = self._entry_slippage_pts
            _scaled_sl = abs(close - stop_loss) + _slippage
            _scaled_tp = abs(take_profit - close) - _slippage
            _actual_rr = _scaled_tp / _scaled_sl if _scaled_sl > 0 else 0.0
            if _actual_rr < self._overnight_min_rr:
                logger.info(
                    f"🚫 OVERNIGHT_RR_GATE: R:R {_actual_rr:.2f} < floor {self._overnight_min_rr:.2f} "
                    f"(SL={_scaled_sl:.1f}pts+slip, TP={_scaled_tp:.1f}pts-slip) | {reason}"
                )
                self._prev_close = close
                return Signal("HOLD", 0.0, {"reason": f"OVERNIGHT_RR_BELOW_FLOOR | rr={_actual_rr:.2f} floor={self._overnight_min_rr:.2f}"})
        _session_type = (
            "RTH" if _is_core_rth
            else "OVERNIGHT" if (_et_t >= time(23, 0) or _et_t < time(9, 30))
            else "EVENING"
        )

        # T3: track proximity counter and mark reduced size in metadata
        if chosen_is_proximity:
            if action == "BUY":
                self._proximity_long_count += 1
            else:
                self._proximity_short_count += 1
            self._save_counters()

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
            "position_scaler": self._proximity_size_mult if chosen_is_proximity else 1.0,
            "entry_type": entry_type,
            "session_type": _session_type,
        }

        logger.info(
            f"📊 {current_time} | {action}: close={close:.2f} "
            f"ema21={ema21:.2f} atr={atr:.1f} adx={adx:.0f} rsi={rsi:.0f} "
            f"macd_h={macd_hist:.2f} SL={stop_loss:.2f} TP={take_profit:.2f} "
            f"[{_session_type}] | {reason}"
        )
        # APR 22 2026 Fix #2: Structured shadow log for post-hoc analysis
        self._log_shadow_decision(
            ts=current_time, outcome="SIGNAL",
            close=close, ema9=ema9, ema21=ema21, ema50=ema50,
            atr=atr, adx=adx, rsi=rsi, macd_hist=macd_hist,
            pdh=(pdh if pdh is not None else 0.0),
            or_high=self._or_high, or_low=self._or_low,
            action=action, stop_loss=stop_loss, take_profit=take_profit,
            reason=reason, gate_diag=_session_type,
            counters={
                "A": self._ema21_pb_long_count, "D": self._ema21_pb_short_count,
                "B": self._or_break_long_count, "E": self._or_break_short_count,
                "F_L": self._trend_cont_long_count, "F_S": self._trend_cont_short_count,
                "A'": self._proximity_long_count, "D'": self._proximity_short_count,
                "G": self._london_fired_count,
            },
        )

        self._prev_close = close
        return Signal(action=action, confidence=0.7, metadata=metadata)

    # ------------------------------------------------------------------
    #  MAR 2026 T2: Regime-adaptive touch band
    # ------------------------------------------------------------------
    def _regime_touch_threshold(self, ema21: float, atr: float, side: str) -> float:
        """Return the EMA21 touch band threshold for the current ATR regime.

        Four buckets (ATR in ES/MES points):
          < ft_atr_very_low_threshold (8):   tight pct band (0.0010) — reduce false touches
          8 – ft_atr_high_threshold (13):    current ft_ema_touch_pct (0.0015) — unchanged
          13 – ft_atr_extreme_threshold (20): ATR-scaled (ft_ema_touch_atr_mult × ATR)
          > ft_atr_extreme_threshold (20):   ATR × 1.0 — extreme vol, widest band

        side = "long"  → threshold is above EMA21 (low must be ≤ threshold)
        side = "short" → threshold is below EMA21 (high must be ≥ threshold)
        """
        sign = 1.0 if side == "long" else -1.0

        if atr <= 0:
            return ema21 * (1.0 + sign * self._ema_touch_pct)

        if atr < self._atr_very_low:
            # Very low vol: tighter band to avoid noise touches
            return ema21 * (1.0 + sign * 0.0010)

        if atr < self._atr_high:
            # Normal regime: existing configured pct (0.0015)
            return ema21 * (1.0 + sign * self._ema_touch_pct)

        if atr < self._atr_extreme:
            # High vol (Fix #6 regime): ATR-scaled band
            mult = self._ema_touch_atr_mult if self._ema_touch_atr_mult > 0 else 0.75
            return ema21 + sign * atr * mult

        # Extreme vol (ATR > 20): widest band — ATR × 1.0
        return ema21 + sign * atr * 1.0

    # ------------------------------------------------------------------
    #  Signal A: EMA21 Pullback Long
    # ------------------------------------------------------------------
    def _check_ema21_pullback(
        self, close: float, open_p: float, low: float,
        ema21: float, ema50: float, atr: float, adx: float,
        rsi: float = 50.0, macd_hist: float = 0.0,
        is_overnight: bool = False,
        pdh: float = 0.0,
    ) -> Optional[tuple]:
        """
        EMA21 pullback in uptrend.

        Conditions:
          1. EMA21 > EMA50 (uptrend)
          2. Bar low touches EMA21 (within 0.1%)
          3. Close > EMA21 (bounced back above)
          4. Close > Open (bullish bar)
          5. ADX > threshold (trending)
          6. MACD histogram not deeply negative (MAR 16 2026 Fix #5 — restored)
             Block when MACD_H < -threshold (default -1.0). Set to 0 to disable.
             7. Not within 0.25% below PDH (MAR 17 2026) — buying into the prior
                 day high from below is buying into resistance; stop rate was 100%
                 in this zone.

        Returns: (action, stop, target, reason) or None
        """
        # MAR 16 2026 Fix #1: Per-session cap — overnight max=1, RTH max=3.
        _max = self._ema21_pb_max_overnight if is_overnight else self._ema21_pb_max_rth
        if self._ema21_pb_long_count >= _max:
            return None

        # 1. Uptrend
        if ema21 <= ema50:
            return None

        # 2. Low touches EMA21 — regime-adaptive band (T2 MAR 2026)
        touch_threshold = self._regime_touch_threshold(ema21, atr, side="long")
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

        # 6. MACD divergence filter (MAR 16 2026 Fix #5)
        # Block BUY when MACD strongly negative — momentum opposes the pullback
        if self._ema21_macd_divergence_block > 0 and macd_hist < -self._ema21_macd_divergence_block:
            return None

        # 7. PDH proximity filter (MAR 17 2026, refactored APR 22 2026)
        # Block BUY when trading into previous day high from below.
        # Tight zone (ft_pdh_proximity_block_pct, default 0.10%): always blocks.
        # Wide  zone (ft_pdh_proximity_rsi_block_pct, default 0.50%): blocks only
        #   if RSI ≥ ft_pdh_proximity_rsi_max (default 70) — overbought + at resistance.
        _pdh_gap_pct = (pdh - close) / close if pdh > 0 else 0.0
        if self._pdh_proximity_block_enabled and pdh > 0 and close <= pdh:
            if self._pdh_proximity_block_pct > 0 and _pdh_gap_pct <= self._pdh_proximity_block_pct:
                logger.info(
                    f"🚫 PDH_PROXIMITY_BLOCK: close={close:.2f} within "
                    f"{self._pdh_proximity_block_pct * 100:.2f}% of PDH={pdh:.2f} "
                    f"({_pdh_gap_pct * 100:.2f}% below) — blocking BUY"
                )
                return None
            if (self._pdh_proximity_rsi_block_pct > 0
                    and _pdh_gap_pct <= self._pdh_proximity_rsi_block_pct
                    and rsi >= self._pdh_proximity_rsi_max):
                logger.info(
                    f"🚫 PDH_RSI_BLOCK: close={close:.2f} within "
                    f"{self._pdh_proximity_rsi_block_pct * 100:.2f}% of PDH={pdh:.2f} "
                    f"({_pdh_gap_pct * 100:.2f}% below) + RSI={rsi:.0f}"
                    f"≥{self._pdh_proximity_rsi_max:.0f} — blocking BUY"
                )
                return None

        # ---- Compute SL: structural (FIX #4, MAY 12 2026) or ATR-adaptive ----
        # The bar's LOW is the EMA21 touch point (price was rejected from below
        # EMA21 and bounced back above).  Mechanical SL = close+ATR*mult often
        # falls INSIDE this wick, so the very next bar's retest stops us out.
        # Structural placement: SL = bar_low - buffer.  If wider than ceiling,
        # SKIP the trade — a setup whose structural stop exceeds the account
        # risk cap is not a setup we want to take.
        if self._ema21_sl_use_structural:
            sl_pts_structural = (close - low) + self._ema21_sl_buffer_pts
            sl_pts = max(self._ema21_sl_floor, sl_pts_structural)
            if sl_pts > self._ema21_sl_ceiling:
                logger.info(
                    f"🚫 EMA21_PB_LONG SKIPPED: structural SL {sl_pts:.1f}pt "
                    f"> ceiling {self._ema21_sl_ceiling:.1f}pt "
                    f"(close={close:.2f} low={low:.2f} buf={self._ema21_sl_buffer_pts:.2f})"
                )
                return None
            sl_method = "STRUCT"
        else:
            sl_pts = min(self._ema21_sl_ceiling,
                         max(self._ema21_sl_floor, atr * self._ema21_sl_atr_mult))
            sl_method = "ATR"
        tp_pts = round(sl_pts * self._ema21_rr_ratio * 4) / 4  # tick to 0.25pt

        stop_loss = close - sl_pts
        take_profit = close + tp_pts

        self._ema21_pb_long_count += 1
        self._save_counters()

        reason = (f"EMA21_PB_LONG | ADX={adx:.0f} | RSI={rsi:.0f} | MACD_H={macd_hist:.2f} "
                  f"| ATR={atr:.1f} | SL={sl_pts:.1f}pts[{sl_method}] | TP={tp_pts:.1f}pts")
        return ("BUY", stop_loss, take_profit, reason)

    # ------------------------------------------------------------------
    #  Signal B: Opening Range Breakout Long
    # ------------------------------------------------------------------
    def _check_or_breakout(
        self, close: float, high: float,
        ema9: float, ema21: float, atr: float, adx: float,
        macd_hist: float = 0.0,
        rsi: float = 50.0,
        volume_ratio: float = 1.0,
        vwap_daily: float = 0.0,
    ) -> Optional[tuple]:
        """
        Opening Range breakout long (up to N per day, default 2).

        FEB 26 2026: Changed from once-per-day to max N per day.
        The cross condition (prev_close <= OR_HIGH and close > OR_HIGH)
        naturally requires price to return below the OR level before
        re-triggering — this is a "retest breakout" pattern with
        strong backtest support for continuation entries.
        
        Conditions:
          1. OR has been computed
          2. Close > OR_HIGH and previous close ≤ OR_HIGH (cross above)
          3. EMA9 > EMA21 (short-term uptrend)
          4. ADX > threshold
          5. Haven't exceeded max fires per day
          6. (REMOVED MAR 3 2026: MACD histogram — redundant; breakout IS momentum)
        
        Returns: (action, stop, target, reason) or None
        """
        if not self._or_computed or self._or_high <= 0:
            return None

        if self._or_break_long_count >= self._or_break_max_per_day:
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

        # MAR 12 2026 Fix A: RSI overbought guard — don't buy breakout into overbought
        if rsi > self._or_break_long_rsi_max:
            logger.info(
                f"OR_BREAK_LONG blocked: RSI={rsi:.1f} > {self._or_break_long_rsi_max:.0f} (overbought)"
            )
            return None

        # APR 2 2026: Anti-chase guard — block if price already ran too far past OR level.
        # close is the 15m bar close, which can be well past OR_HIGH by the time signal fires.
        _chase_dist = close - self._or_high
        _max_chase = atr * self._or_break_max_chase_atr
        if _chase_dist > _max_chase:
            logger.info(
                f"OR_BREAK_LONG blocked: chase={_chase_dist:.1f}pts > {_max_chase:.1f}pts "
                f"({self._or_break_max_chase_atr:.1f}×ATR) past OR_H={self._or_high:.2f}"
            )
            return None

        # APR 29 2026: Volume confirmation — low-volume breakouts have weak follow-through.
        # Threshold default 0.7 (vol must be >= 70% of 20-bar avg). Set 0 to disable.
        if self._or_break_min_vol_ratio > 0 and volume_ratio < self._or_break_min_vol_ratio:
            logger.info(
                f"OR_BREAK_LONG blocked: vol_ratio={volume_ratio:.2f} < "
                f"{self._or_break_min_vol_ratio:.2f} (low-volume breakout)"
            )
            return None

        # APR 29 2026: VWAP confirmation — long breakout requires close above session VWAP.
        # Below-VWAP breakouts often fail (institutional reference rejects them).
        if self._or_break_vwap_filter_enabled and vwap_daily > 0 and close <= vwap_daily:
            logger.info(
                f"OR_BREAK_LONG blocked: close={close:.2f} <= VWAP={vwap_daily:.2f} "
                f"(below-VWAP breakout — weak institutional support)"
            )
            return None

        # ---- Compute stops/targets ----
        # MAR 12 2026: ATR-adaptive SL — fixed 6pt SL at ATR=10-12 is sub-ATR (noise band).
        # SL = clamp(ATR × mult, floor, ceiling), TP = SL × R:R ratio
        sl_pts = min(self._or_break_sl_ceiling, max(self._or_break_sl_floor, atr * self._or_break_sl_atr_mult))
        tp_pts = round(sl_pts * self._or_break_rr_ratio * 4) / 4  # round to 0.25-pt tick
        stop_loss = close - sl_pts
        take_profit = close + tp_pts

        self._or_break_long_count += 1
        self._save_counters()

        tag = "" if self._or_break_long_count == 1 else f" | retest#{self._or_break_long_count}"
        reason = f"OR_BREAK_LONG | ADX={adx:.0f} | OR_H={self._or_high:.2f} | SL={sl_pts:.1f}pt | TP={tp_pts:.1f}pt{tag}"
        return ("BUY", stop_loss, take_profit, reason)

    # ------------------------------------------------------------------
    #  Signal C: EMA9 Pullback Long (fast trend)
    # ------------------------------------------------------------------
    def _check_ema9_pullback(
        self, close: float, open_p: float, low: float,
        ema9: float, ema21: float, ema50: float, atr: float, adx: float,
        rsi: float, macd_hist: float = 0.0,
        pdh: float = 0.0,
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

        # 8. MACD histogram must show meaningful momentum (MAR 16 2026: raised floor)
        # Was `macd_hist > 0` — near-zero (e.g. 0.20) has no edge. Default 0.3.
        if macd_hist < self._ema9_pb_macd_min:
            return None

        # 9. PDH proximity filter (APR 6 2026, refactored APR 22 2026)
        # Uses the same configurable thresholds as Signal A. Tight zone always
        # blocks; wide zone only blocks when RSI ≥ rsi_max.
        if self._pdh_proximity_block_enabled and pdh > 0 and close > 0:
            _pdh_gap_pct = (pdh - close) / close
            if close <= pdh:
                if (self._pdh_proximity_block_pct > 0
                        and _pdh_gap_pct <= self._pdh_proximity_block_pct):
                    logger.info(
                        f"🚫 EMA9_PDH_PROXIMITY_BLOCK: close={close:.2f} within "
                        f"{self._pdh_proximity_block_pct * 100:.2f}% of PDH={pdh:.2f} "
                        f"({_pdh_gap_pct * 100:.2f}% below) — blocking BUY"
                    )
                    return None
                if (self._pdh_proximity_rsi_block_pct > 0
                        and _pdh_gap_pct <= self._pdh_proximity_rsi_block_pct
                        and rsi >= self._pdh_proximity_rsi_max):
                    logger.info(
                        f"🚫 EMA9_PDH_RSI_BLOCK: close={close:.2f} within "
                        f"{self._pdh_proximity_rsi_block_pct * 100:.2f}% of PDH={pdh:.2f} "
                        f"({_pdh_gap_pct * 100:.2f}% below) + RSI={rsi:.0f}"
                        f"≥{self._pdh_proximity_rsi_max:.0f} — blocking BUY"
                    )
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
        is_overnight: bool = False,
    ) -> Optional[tuple]:
        """
        EMA21 pullback in downtrend — short-side mirror of Signal A.

        Conditions (exact inverse of long):
          1. EMA21 < EMA50 (downtrend)
          2. Bar high touches EMA21 from below (within 0.1%)
          3. Close < EMA21 (rejected back below)
          4. Close < Open (bearish bar)
          5. ADX > threshold (trending)
             6. MACD histogram not deeply positive (MAR 16 2026 Fix #5 — restored)
                 Block when MACD_H > +threshold (default +0.5). Set to 0 to disable.

        Returns: (action, stop, target, reason) or None
        """
        # MAR 16 2026 Fix #1: Per-session cap — overnight max=1, RTH max=3.
        _max = self._ema21_pb_max_overnight if is_overnight else self._ema21_pb_max_rth
        if self._ema21_pb_short_count >= _max:
            return None

        # 1. Downtrend
        if ema21 >= ema50:
            return None

        # 2. High touches EMA21 from below — regime-adaptive band (T2 MAR 2026)
        touch_threshold = self._regime_touch_threshold(ema21, atr, side="short")
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

        # 6. MACD divergence filter (MAR 16 2026 Fix #5)
        # Block SELL when MACD strongly positive — momentum opposes the pullback
        if self._ema21_macd_divergence_block > 0 and macd_hist > self._ema21_macd_divergence_block:
            return None

        # ---- Compute SL: structural (FIX #4, MAY 12 2026) or ATR-adaptive ----
        # Mirror of Signal A: bar HIGH is the EMA21 rejection point.
        # Structural placement: SL = bar_high + buffer.  Skip if wider than ceiling.
        if self._ema21_sl_use_structural:
            sl_pts_structural = (high - close) + self._ema21_sl_buffer_pts
            sl_pts = max(self._ema21_sl_floor, sl_pts_structural)
            if sl_pts > self._ema21_sl_ceiling:
                logger.info(
                    f"🚫 EMA21_PB_SHORT SKIPPED: structural SL {sl_pts:.1f}pt "
                    f"> ceiling {self._ema21_sl_ceiling:.1f}pt "
                    f"(close={close:.2f} high={high:.2f} buf={self._ema21_sl_buffer_pts:.2f})"
                )
                return None
            sl_method = "STRUCT"
        else:
            sl_pts = min(self._ema21_sl_ceiling,
                         max(self._ema21_sl_floor, atr * self._ema21_sl_atr_mult))
            sl_method = "ATR"
        tp_pts = round(sl_pts * self._ema21_rr_ratio * 4) / 4  # tick to 0.25pt

        stop_loss = close + sl_pts
        take_profit = close - tp_pts

        self._ema21_pb_short_count += 1
        self._save_counters()

        reason = (f"EMA21_PB_SHORT | ADX={adx:.0f} | RSI={rsi:.0f} | MACD_H={macd_hist:.2f} "
                  f"| ATR={atr:.1f} | SL={sl_pts:.1f}pts[{sl_method}] | TP={tp_pts:.1f}pts")
        return ("SELL", stop_loss, take_profit, reason)

    # ------------------------------------------------------------------
    #  Signal A-prime: EMA21 Proximity Long (near-miss recovery)
    #  Signal D-prime: EMA21 Proximity Short (near-miss recovery)
    #
    #  MAR 2026 T3: Motivated by 29+43 near-misses in Feb-Mar 2026 high-vol
    #  regime where price pulled back to within 0.3 ATR of the band but the
    #  bar low didn't quite reach the touch threshold.  Enters at 70% size.
    # ------------------------------------------------------------------
    def _check_ema21_proximity_long(
        self,
        close: float, open_p: float, low: float,
        ema21: float, ema50: float, atr: float, adx: float,
    ) -> Optional[tuple]:
        """Near-miss long entry when low just misses the EMA21 touch band.

        Fires when ALL of:
          1. High-vol regime: ATR > _atr_high (13 pts)
          2. Uptrend: EMA21 > EMA50
          3. Low is in the proximity zone:
               touch_threshold < low ≤ touch_threshold + gap
             where gap = _proximity_gap_mult × ATR (default 0.3 × ATR)
             (i.e., price NEARLY touched but Signal A did NOT fire)
          4. Close > EMA21 (bar already bounced above EMA21)
          5. Bullish bar
          6. ADX in [_adx_min, _adx_max]
          7. Day count ≤ _proximity_max_per_day
        """
        if not self._proximity_enabled:
            return None
        if atr < self._atr_high:
            return None
        if self._proximity_long_count >= self._proximity_max_per_day:
            return None
        if ema21 <= ema50:
            return None

        touch_threshold = self._regime_touch_threshold(ema21, atr, side="long")
        proximity_gap = self._proximity_gap_mult * atr
        proximity_outer = touch_threshold + proximity_gap

        # Near-miss zone: low is above touch_threshold (Signal A missed)
        # but within proximity_gap of it
        if low <= touch_threshold:
            return None   # Signal A should have fired — don't double-count
        if low > proximity_outer:
            return None   # Too far away — not a near-miss

        if close <= ema21:
            return None
        if close <= open_p:
            return None
        if adx < self._adx_min or adx > self._adx_max:
            return None

        sl = close - self._fixed_sl_points * self._proximity_sl_mult
        tp = close + self._fixed_tp_points * self._proximity_tp_mult
        reason = (
            f"EMA21_PROX_LONG | ATR={atr:.1f} | ADX={adx:.0f} "
            f"| low={low:.2f} thresh={touch_threshold:.2f} gap={proximity_gap:.1f} "
            f"| SIZE={self._proximity_size_mult:.0%} | #{self._proximity_long_count + 1}"
        )
        return ("BUY", sl, tp, reason)

    def _check_ema21_proximity_short(
        self,
        close: float, open_p: float, high: float,
        ema21: float, ema50: float, atr: float, adx: float,
    ) -> Optional[tuple]:
        """Near-miss short entry — mirror of _check_ema21_proximity_long."""
        if not self._proximity_enabled:
            return None
        if atr < self._atr_high:
            return None
        if self._proximity_short_count >= self._proximity_max_per_day:
            return None
        if ema21 >= ema50:
            return None

        touch_threshold = self._regime_touch_threshold(ema21, atr, side="short")
        proximity_gap = self._proximity_gap_mult * atr
        proximity_outer = touch_threshold - proximity_gap

        if high >= touch_threshold:
            return None   # Signal D should have fired
        if high < proximity_outer:
            return None   # Too far away

        if close >= ema21:
            return None
        if close >= open_p:
            return None
        if adx < self._adx_min or adx > self._adx_max:
            return None

        sl = close + self._fixed_sl_points * self._proximity_sl_mult
        tp = close - self._fixed_tp_points * self._proximity_tp_mult
        reason = (
            f"EMA21_PROX_SHORT | ATR={atr:.1f} | ADX={adx:.0f} "
            f"| high={high:.2f} thresh={touch_threshold:.2f} gap={proximity_gap:.1f} "
            f"| SIZE={self._proximity_size_mult:.0%} | #{self._proximity_short_count + 1}"
        )
        return ("SELL", sl, tp, reason)

    # ------------------------------------------------------------------
    #  Signal G: London Momentum Breakout
    # ------------------------------------------------------------------
    def _check_london_momentum(
        self, df: pd.DataFrame, current_time: Any,
        close: float, open_p: float, low: float, high: float,
        ema9: float, ema21: float, ema50: float,
        atr: float, adx: float,
    ) -> Optional[tuple]:
        """
        London session momentum breakout — captures first directional
        impulse when European liquidity arrives.

        MAR 9 2026: During London session (2-5 AM CST), MES typically
        sits in tight range (ATR 2-4pts, ADX 16-18).  When London opens,
        volatility injects a directional move.  This signal detects the
        first EMA9/EMA21 crossover during London hours.

        Conditions:
          1. London time window: ft_london_start_hour to ft_london_end_hour CT
          2. EMA9 crossed EMA21 THIS bar (prev bar had opposite alignment)
          3. ADX >= ft_london_adx_min (15) — lower than RTH signals because
             London is emerging from chop, not already trending
          4. Directional confirmation:
             LONG:  EMA9 > EMA21, close > EMA9, bullish bar
             SHORT: EMA9 < EMA21, close < EMA9, bearish bar
          5. Max ft_london_max_per_day per day (default 1 — first impulse only)

        SL/TP sizing for London move (typical 8-20pts):
          SL = structural: max(ft_london_sl_points floor, EMA21_gap + 0.5pt),
               capped at ft_london_sl_atr_cap × ATR.  Placed just beyond EMA21
               so a normal wick doesn't stop out a valid cross.
          TP = ft_london_tp_points (8pts / $40) — above median 8.2pt directional move;
               ~55-60% hit rate vs 165-session MES London data (Feb 2025-Jan 2026)
          R:R ≈ 1.6:1 (8pt TP / 5pt SL floor)

        Returns: (action, stop, target, reason) or None
        """
        if not self._london_enabled:
            return None

        # 5. Daily limit
        if self._london_fired_count >= self._london_max_per_day:
            return None

        # 1. London time window (CT)
        ct_time = self._to_ct(current_time)
        ct_t = ct_time.time()
        if not (self._london_start_ct <= ct_t < self._london_end_ct):
            return None

        # 2. EMA9/EMA21 crossover THIS bar — need previous bar to check
        if len(df) < 2:
            return None
        prev = df.iloc[-2]
        prev_ema9 = float(prev.get("EMA_9", 0))
        prev_ema21 = float(prev.get("EMA_21", 0))

        # Bullish cross: EMA9 was below EMA21, now above
        bullish_cross = prev_ema9 <= prev_ema21 and ema9 > ema21
        # Bearish cross: EMA9 was above EMA21, now below
        bearish_cross = prev_ema9 >= prev_ema21 and ema9 < ema21

        if not bullish_cross and not bearish_cross:
            return None

        # 3. ADX filter (lower bar than RTH — London is emerging from chop)
        if adx < self._london_adx_min:
            return None

        # 4. Directional confirmation
        if bullish_cross:
            if close <= ema9:
                return None  # Price must be above EMA9 to confirm momentum
            if close <= open_p:
                return None  # Must be bullish bar

            # Structural SL: just below EMA21 (level that invalidates the cross),
            # floored at ft_london_sl_points and capped at ft_london_sl_atr_cap×ATR.
            sl_pts = max(self._london_sl_points, (close - ema21) + 0.5)
            sl_pts = min(sl_pts, atr * self._london_sl_atr_cap)
            sl_pts = max(sl_pts, self._london_sl_points)  # re-apply floor after cap
            sl = close - sl_pts
            tp = close + self._london_tp_points
            self._london_fired_count += 1
            self._save_counters()
            reason = (
                f"LONDON_MOMENTUM_LONG | ADX={adx:.0f} | ATR={atr:.1f} "
                f"| SL={sl_pts:.1f}pt | EMA9_cross_above_EMA21 | #{self._london_fired_count}"
            )
            return ("BUY", sl, tp, reason)
        else:
            # bearish cross — only if shorts enabled
            if not self._shorts_enabled:
                return None
            if close >= ema9:
                return None  # Price must be below EMA9
            if close >= open_p:
                return None  # Must be bearish bar

            # Mirror of long-side structural SL
            sl_pts = max(self._london_sl_points, (ema21 - close) + 0.5)
            sl_pts = min(sl_pts, atr * self._london_sl_atr_cap)
            sl_pts = max(sl_pts, self._london_sl_points)  # re-apply floor after cap
            sl = close + sl_pts
            tp = close - self._london_tp_points
            self._london_fired_count += 1
            self._save_counters()
            reason = (
                f"LONDON_MOMENTUM_SHORT | ADX={adx:.0f} | ATR={atr:.1f} "
                f"| SL={sl_pts:.1f}pt | EMA9_cross_below_EMA21 | #{self._london_fired_count}"
            )
            return ("SELL", sl, tp, reason)

    # ------------------------------------------------------------------
    #  Signal E: Opening Range Breakdown Short (mirror of Signal B)
    # ------------------------------------------------------------------
    def _check_or_breakdown(
        self, close: float, low: float,
        ema9: float, ema21: float, atr: float, adx: float,
        macd_hist: float = 0.0,
        rsi: float = 50.0,
        volume_ratio: float = 1.0,
        vwap_daily: float = 0.0,
    ) -> Optional[tuple]:
        """
        Opening Range breakdown short (up to N per day, default 2).

        FEB 26 2026: Changed from once-per-day to max N per day.
        The cross condition (prev_close >= OR_LOW and close < OR_LOW)
        naturally requires price to bounce back above OR_LOW before
        re-triggering — this is a "retest breakdown" pattern.

        Conditions (exact inverse of long OR breakout):
          1. OR has been computed
          2. Close < OR_LOW and previous close >= OR_LOW (cross below)
          3. EMA9 < EMA21 (short-term downtrend)
          4. ADX > threshold
          5. Haven't exceeded max fires per day
          6. (REMOVED MAR 3 2026: MACD histogram — redundant; breakdown IS momentum)

        Returns: (action, stop, target, reason) or None
        """
        if not self._or_computed or self._or_low <= 0:
            return None

        if self._or_break_short_count >= self._or_break_max_per_day:
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

        # MAR 12 2026 Fix A: RSI oversold guard — don't short breakdown into oversold
        if rsi < self._or_break_short_rsi_min:
            logger.info(
                f"OR_BREAK_SHORT blocked: RSI={rsi:.1f} < {self._or_break_short_rsi_min:.0f} (oversold — high bounce risk)"
            )
            return None

        # APR 2 2026: Anti-chase guard — block if price already ran too far past OR level.
        # Evidence: Apr 2 OR_BREAK_SHORT close=6520, OR_LOW=6533 → 13pts chase (1.3×ATR) → SL hit in 19s.
        _chase_dist = self._or_low - close  # positive when close is below OR_LOW
        _max_chase = atr * self._or_break_max_chase_atr
        if _chase_dist > _max_chase:
            logger.info(
                f"OR_BREAK_SHORT blocked: chase={_chase_dist:.1f}pts > {_max_chase:.1f}pts "
                f"({self._or_break_max_chase_atr:.1f}×ATR) past OR_L={self._or_low:.2f}"
            )
            return None

        # APR 29 2026: Volume confirmation — low-volume breakdowns have weak follow-through.
        if self._or_break_min_vol_ratio > 0 and volume_ratio < self._or_break_min_vol_ratio:
            logger.info(
                f"OR_BREAK_SHORT blocked: vol_ratio={volume_ratio:.2f} < "
                f"{self._or_break_min_vol_ratio:.2f} (low-volume breakdown)"
            )
            return None

        # APR 29 2026: VWAP confirmation — short breakdown requires close below session VWAP.
        # Above-VWAP breakdowns often fail (institutional reference supports them).
        if self._or_break_vwap_filter_enabled and vwap_daily > 0 and close >= vwap_daily:
            logger.info(
                f"OR_BREAK_SHORT blocked: close={close:.2f} >= VWAP={vwap_daily:.2f} "
                f"(above-VWAP breakdown — weak institutional rejection)"
            )
            return None

        # ---- Compute stops/targets (inverted) ----
        # MAR 12 2026: ATR-adaptive SL — fixed 6pt SL at ATR=10-12 is sub-ATR (noise band).
        # SL = clamp(ATR × mult, floor, ceiling), TP = SL × R:R ratio
        sl_pts = min(self._or_break_sl_ceiling, max(self._or_break_sl_floor, atr * self._or_break_sl_atr_mult))
        tp_pts = round(sl_pts * self._or_break_rr_ratio * 4) / 4  # round to 0.25-pt tick
        stop_loss = close + sl_pts
        take_profit = close - tp_pts

        self._or_break_short_count += 1
        self._save_counters()

        tag = "" if self._or_break_short_count == 1 else f" | retest#{self._or_break_short_count}"
        reason = f"OR_BREAK_SHORT | ADX={adx:.0f} | OR_L={self._or_low:.2f} | SL={sl_pts:.1f}pt | TP={tp_pts:.1f}pt{tag}"
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
          4. ADX >= 25 (strong trending — raised from 18 on FEB 24 2026)
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

        # Fix #16 (MAR 12 2026): Trend exhaustion guard.
        # Block TREND_CONT_LONG when the session has already moved > N×ATR
        # from the OR midpoint — price is overextended, not a continuation entry.
        _exhaustion_mult = getattr(self, "_trend_exhaustion_atr_multiple", 8.0)
        _or_computed = getattr(self, "_or_computed", False)
        _or_high = getattr(self, "_or_high", 0.0)
        _or_low = getattr(self, "_or_low", 0.0)
        if _exhaustion_mult > 0 and _or_computed and _or_high > 0 and atr > 0:
            _session_ref = (_or_high + _or_low) / 2.0
            _session_move = close - _session_ref  # positive = upside move
            _move_in_atr = abs(_session_move) / atr
            if _move_in_atr > _exhaustion_mult:
                logger.info(
                    f"TREND_EXHAUSTION_LONG: move={_session_move:+.1f}pts "
                    f"({_move_in_atr:.1f}×ATR) > {_exhaustion_mult}×ATR threshold — "
                    f"blocking TREND_CONT_LONG"
                )
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

        # 4. ADX filter (need strong trend — FEB 24: raised to 25)
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

        # ---- Compute ATR-adaptive stops/targets (FEB 24 2026) ----
        # SL = clamp(ATR × mult, floor, ceiling), TP = SL × rr_ratio
        # At ATR=8.7: SL=8.7pts($44), TP=10.9pts($54) → reachable
        # At ATR=15:  SL=15pts($75), TP=18.8pts($94) → room to breathe
        # At ATR=5:   SL=6pts($30),  TP=7.5pts($38) → floor protects
        sl_pts = min(self._trend_sl_ceiling,
                     max(self._trend_sl_floor, atr * self._trend_sl_atr_mult))
        tp_pts = sl_pts * self._trend_rr_ratio
        stop_loss = close - sl_pts
        take_profit = close + tp_pts

        self._trend_cont_long_count += 1
        self._save_counters()

        reason = (f"TREND_CONT_LONG | ADX={adx:.0f} | RSI={rsi:.0f} "
                  f"| MACD_H={macd_hist:.2f} | e9={ema9:.1f} | #{self._trend_cont_long_count}"
                  f" | SL={sl_pts:.1f}pts | TP={tp_pts:.1f}pts")
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
          4. ADX >= 25 (strong trending — raised from 18 on FEB 24 2026)
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

        # Fix #16 (MAR 12 2026): Trend exhaustion guard.
        # Block TREND_CONT_SHORT when the session has already moved > N×ATR
        # from the OR midpoint — the selloff is spent, not a continuation entry.
        _exhaustion_mult = getattr(self, "_trend_exhaustion_atr_multiple", 8.0)
        _or_computed = getattr(self, "_or_computed", False)
        _or_high = getattr(self, "_or_high", 0.0)
        _or_low = getattr(self, "_or_low", 0.0)
        if _exhaustion_mult > 0 and _or_computed and _or_high > 0 and atr > 0:
            _session_ref = (_or_high + _or_low) / 2.0
            _session_move = _session_ref - close  # positive = downside move
            _move_in_atr = abs(_session_move) / atr
            if _move_in_atr > _exhaustion_mult:
                logger.info(
                    f"TREND_EXHAUSTION_SHORT: move={_session_move:+.1f}pts "
                    f"({_move_in_atr:.1f}×ATR) > {_exhaustion_mult}×ATR threshold — "
                    f"blocking TREND_CONT_SHORT"
                )
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

        # ---- Compute ATR-adaptive stops/targets (FEB 24 2026) ----
        # Mirrors long side: SL = clamp(ATR × mult, floor, ceiling), TP = SL × rr_ratio
        sl_pts = min(self._trend_sl_ceiling,
                     max(self._trend_sl_floor, atr * self._trend_sl_atr_mult))
        tp_pts = sl_pts * self._trend_rr_ratio
        stop_loss = close + sl_pts
        take_profit = close - tp_pts

        self._trend_cont_short_count += 1
        self._save_counters()

        reason = (f"TREND_CONT_SHORT | ADX={adx:.0f} | RSI={rsi:.0f} "
                  f"| MACD_H={macd_hist:.2f} | e9={ema9:.1f} | #{self._trend_cont_short_count}"
                  f" | SL={sl_pts:.1f}pts | TP={tp_pts:.1f}pts")
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
    _COUNTER_FILE = Path("data/signal_counters.json")

    # APR 22 2026 Fix #2: Structured shadow-decision log.
    # Writes one JSON line per bar to logs/decisions.jsonl so every gate firing
    # and signal intent is auditable offline. Safe to parse with jq or pandas.
    _SHADOW_LOG_PATH: Optional[Path] = None

    def _log_shadow_decision(self, **kv) -> None:
        try:
            if self._SHADOW_LOG_PATH is None:
                # Lazy init — repo_root/logs/decisions.jsonl
                repo_root = Path(__file__).resolve().parents[2]
                path = repo_root / "logs" / "decisions.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                # Bind class-level attribute so all instances share the path
                type(self)._SHADOW_LOG_PATH = path
            ts = kv.get("ts")
            if hasattr(ts, "isoformat"):
                kv["ts"] = ts.isoformat()
            else:
                kv["ts"] = str(ts) if ts is not None else ""
            # Round floats to 3 decimals for compactness
            for k, v in list(kv.items()):
                if isinstance(v, float):
                    kv[k] = round(v, 4)
            kv["strategy"] = self.name
            kv["session_date"] = (
                self._session_date.isoformat() if self._session_date else ""
            )
            with self._SHADOW_LOG_PATH.open("a") as fh:
                fh.write(json.dumps(kv, default=str) + "\n")
        except Exception as exc:
            # Never let logging kill the strategy
            logger.debug(f"shadow_log write failed: {exc}")

    def _reset_session(self, date) -> None:
        """Reset all session state for a new trading day."""
        self._session_date = date
        self._or_high = 0.0
        self._or_low = 0.0
        self._or_computed = False
        self._opening_bars = []
        self._or_break_long_count = 0
        self._or_break_short_count = 0
        self._prev_close = 0.0
        # Signal F counters
        self._trend_cont_long_count = 0
        self._trend_cont_short_count = 0
        # Signal A/D (EMA21 pullback) counters — MAR 16 2026 Fix #1
        self._ema21_pb_long_count = 0
        self._ema21_pb_short_count = 0
        # Signal A-prime / D-prime (proximity) counters
        self._proximity_long_count = 0
        self._proximity_short_count = 0
        # Signal G (London momentum) counter
        self._london_fired_count = 0
        # MAR 16 2026 Fix #2: Reset exhaustion cooldown
        self._exhaustion_long_bars_left = 0
        self._exhaustion_short_bars_left = 0
        # Persist the zeroed counters
        self._save_counters()

    # ------------------------------------------------------------------
    #  Counter persistence — survive bot restarts
    # ------------------------------------------------------------------
    def _get_cme_session_date(self) -> str:
        """Return the CME trading session date as YYYY-MM-DD.

        CME session starts at 17:00 CT.  A bar at 2026-03-10 02:00 CT
        belongs to the session that opened on 2026-03-09 17:00 CT, so
        its session date is '2026-03-09'.  A bar at 2026-03-10 18:00 CT
        belongs to '2026-03-10'.
        """
        try:
            from ..utils.timezone_utils import now_cst
            ct_now = now_cst()
        except Exception:
            ct_now = datetime.now()
        if ct_now.hour < 17:
            # Before 5 PM CT → session started yesterday
            session_date = (ct_now - timedelta(days=1)).date()
        else:
            session_date = ct_now.date()
        return session_date.isoformat()

    def _save_counters(self) -> None:
        """Persist signal fire counters to disk."""
        data = {
            "cme_session_date": self._get_cme_session_date(),
            "or_break_long": getattr(self, "_or_break_long_count", 0),
            "or_break_short": getattr(self, "_or_break_short_count", 0),
            "trend_cont_long": getattr(self, "_trend_cont_long_count", 0),
            "trend_cont_short": getattr(self, "_trend_cont_short_count", 0),
            "proximity_long": getattr(self, "_proximity_long_count", 0),
            "proximity_short": getattr(self, "_proximity_short_count", 0),
            "london_fired": getattr(self, "_london_fired_count", 0),
            "ema21_pb_long": getattr(self, "_ema21_pb_long_count", 0),
            "ema21_pb_short": getattr(self, "_ema21_pb_short_count", 0),
        }
        try:
            self._COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._COUNTER_FILE.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning(f"Failed to save signal counters: {exc}")

    def _load_counters(self) -> None:
        """Load signal fire counters from disk if same CME session."""
        if not self._COUNTER_FILE.exists():
            return
        try:
            data = json.loads(self._COUNTER_FILE.read_text())
            saved_session = data.get("cme_session_date", "")
            current_session = self._get_cme_session_date()
            if saved_session != current_session:
                logger.info(
                    f"🔄 Counter file session {saved_session} != current {current_session}, "
                    f"starting fresh"
                )
                return  # Different CME session → counters already zeroed in __init__
            self._or_break_long_count = data.get("or_break_long", 0)
            self._or_break_short_count = data.get("or_break_short", 0)
            self._trend_cont_long_count = data.get("trend_cont_long", 0)
            self._trend_cont_short_count = data.get("trend_cont_short", 0)
            self._proximity_long_count = data.get("proximity_long", 0)
            self._proximity_short_count = data.get("proximity_short", 0)
            self._london_fired_count = data.get("london_fired", 0)
            self._ema21_pb_long_count = data.get("ema21_pb_long", 0)
            self._ema21_pb_short_count = data.get("ema21_pb_short", 0)
            logger.info(
                f"📂 Loaded signal counters from disk (session={saved_session}): "
                f"A={self._ema21_pb_long_count} D={self._ema21_pb_short_count} "
                f"B={self._or_break_long_count} E={self._or_break_short_count} "
                f"F_L={self._trend_cont_long_count} F_S={self._trend_cont_short_count} "
                f"A'={self._proximity_long_count} D'={self._proximity_short_count} "
                f"G={self._london_fired_count}"
            )
        except Exception as exc:
            logger.warning(f"Failed to load signal counters: {exc}")

    def rollback_counter(self, signal_reason: str) -> None:
        """Undo a counter increment when signal is blocked downstream.

        Called by signal_processor when CHOP guard, confidence gate, or
        other post-strategy filters block a signal.  The strategy already
        incremented the counter (needed for same-bar max-per-day gating),
        so we decrement it back and re-persist.

        MAR 10 2026: Fixes bug where CHOP-blocked signals consumed
        daily counter slots — e.g. 2 CHOP-blocked F signals at 2:30+3:00
        exhausted the 2/day limit, preventing valid RTH F signals.
        """
        rolled_back = False
        if "TREND_CONT_LONG" in signal_reason:
            if self._trend_cont_long_count > 0:
                self._trend_cont_long_count -= 1
                rolled_back = True
        elif "TREND_CONT_SHORT" in signal_reason:
            if self._trend_cont_short_count > 0:
                self._trend_cont_short_count -= 1
                rolled_back = True
        elif "EMA21_PB_LONG" in signal_reason:
            if self._ema21_pb_long_count > 0:
                self._ema21_pb_long_count -= 1
                rolled_back = True
        elif "EMA21_PB_SHORT" in signal_reason:
            if self._ema21_pb_short_count > 0:
                self._ema21_pb_short_count -= 1
                rolled_back = True
        elif "OR_BREAK_LONG" in signal_reason:
            if self._or_break_long_count > 0:
                self._or_break_long_count -= 1
                rolled_back = True
        elif "OR_BREAK_SHORT" in signal_reason:
            if self._or_break_short_count > 0:
                self._or_break_short_count -= 1
                rolled_back = True
        elif "LONDON_MOMENTUM" in signal_reason:
            if self._london_fired_count > 0:
                self._london_fired_count -= 1
                rolled_back = True
        elif "PROXIMITY" in signal_reason or "PRIME" in signal_reason:
            if "LONG" in signal_reason or "BUY" in signal_reason:
                if self._proximity_long_count > 0:
                    self._proximity_long_count -= 1
                    rolled_back = True
            else:
                if self._proximity_short_count > 0:
                    self._proximity_short_count -= 1
                    rolled_back = True

        if rolled_back:
            logger.info(f"↩️ Counter rolled back for blocked signal: {signal_reason}")
            self._save_counters()

    def _to_et(self, ts: pd.Timestamp) -> Any:
        """Convert timestamp to US/Eastern."""
        if ts.tz is not None:
            try:
                return ts.tz_convert("US/Eastern")
            except Exception:
                return ts
        return ts

    def _to_ct(self, ts: pd.Timestamp) -> Any:
        """Convert timestamp to US/Central (CT) for London session check."""
        if ts.tz is not None:
            try:
                return ts.tz_convert("America/Chicago")
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
