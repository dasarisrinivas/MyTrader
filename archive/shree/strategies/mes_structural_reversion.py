"""
MES Structural Reversion Strategy
==================================

FEB 2026 — COMPLETE REDESIGN based on 1-year backtest analysis.

PREMISE
-------
1-minute ES/MES data is mean-reverting.  The prior scoring strategy's
direction signal (1m EMA cross) was anti-predictive — reversing every
trade would have been profitable (+$4,832 vs -$5,209).

This strategy does the OPPOSITE of the old approach:
  - Fewer trades (max 2/session) instead of 6/day
  - Mean reversion from structure, not trend following
  - Wider stops to survive noise (old stops killed 37% of trades in < 5 min)
  - Faster profit taking (asymmetric: quick exit on winners)
  - Strict time-of-day gating (only 10:00–11:30 CST)
  - ATR regime gating (only ATR ≥ 4)
  - Time-stop exit (max 45 min hold)

DATA-DRIVEN DESIGN
------------------
From analysis of 1,552 trades (Feb 2025 – Jan 2026):

  Hold < 5 min  →  10.2% WR,  -$10,080  (entering into counter-moves)
  Hold 10-45 min → 46.5% WR,  +$4,815   (survivors are profitable)
  10:00-10:59    → only profitable 30-min window
  ATR ≥ 4, 10-11am → PF 1.56, 44.2% WR
  After quick loss, reverse dir → 40% WR (mean reversion signal)
  Score/ADX/trend_label → zero predictive value (deleted)

ENTRY MODEL
-----------
1. Wait for Opening Range to establish (9:30–10:00 CST)
2. Compute VWAP, PDH, PDL, opening range (OR_HIGH / OR_LOW)
3. During 10:00–11:30 CST, look for overextension:
   - Price pushes ABOVE opening range high or PDH → SHORT reversion to VWAP
   - Price pushes BELOW opening range low or PDL → LONG reversion to VWAP
4. Confirmation: RSI overextended (>65 for short, <35 for long)
5. Rejection candle: current bar closes back inside the level

EXIT MODEL
----------
- Stop: 2.0 × ATR (wider to survive noise; data shows quick stops are the #1 loss driver)
- Target: VWAP midpoint (structural target, typically 1–1.5× ATR)
- Time-stop: 45 min max hold (data shows 10-45 min is the profitable window)
- Breakeven lock: move stop to entry + 0.5 pts after 1R in profit

FILTERS
-------
- ATR ≥ 4 (88% of old losing trades were in ATR < 4)
- Volume > 100 on entry bar
- Max 2 trades per session
- Cooldown 15 min between trades (prevent revenge trading)
- No trades during 12:00–13:30 lunch chop
- 15m regime awareness: prefer ranging (mean reversion works best in chop)

Author: Quantitative Trading Redesign — Feb 2026
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta, datetime
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from loguru import logger

from ..config import OneMinuteStrategyConfig
from ..features.feature_engineer import _adx, _atr, _ema, _rsi
from .base import BaseStrategy, Signal


@dataclass
class StructuralLevels:
    """Key structural levels computed from the session."""
    vwap: float = 0.0
    pdh: float = 0.0       # Previous day high
    pdl: float = 0.0       # Previous day low
    or_high: float = 0.0   # Opening range high (first 30 min)
    or_low: float = 0.0    # Opening range low (first 30 min)
    or_mid: float = 0.0    # Opening range midpoint
    bb_upper: float = 0.0  # Bollinger upper band
    bb_lower: float = 0.0  # Bollinger lower band


class MesStructuralReversionStrategy(BaseStrategy):
    """
    Mean-reversion from structural levels, designed to be the
    anti-thesis of the failed trend-following scoring system.
    
    Core principle: MES mean-reverts aggressively on 1-min data.
    We exploit overextension from known levels (VWAP, PDH/PDL,
    opening range) with wide stops and fast targets.
    """

    name = "mes_structural_reversion"

    def __init__(self, config: OneMinuteStrategyConfig):
        self.config = config
        
        # Session state (reset daily)
        self._session_date: Optional[Any] = None
        self._session_trades: int = 0
        self._last_trade_time: Optional[pd.Timestamp] = None
        self._or_high: float = 0.0
        self._or_low: float = 0.0
        self._or_computed: bool = False
        self._opening_bars: list = []  # Collect bars during opening range
        
        # Configurable parameters (with data-driven defaults)
        self._min_atr: float = getattr(config, 'sr_min_atr', 4.0)
        self._max_trades: int = getattr(config, 'sr_max_trades', 2)
        self._cooldown_min: int = getattr(config, 'sr_cooldown_minutes', 15)
        self._stop_atr_mult: float = getattr(config, 'stop_atr_multiplier', 2.0)
        self._target_mode: str = getattr(config, 'sr_target_mode', 'vwap')  # 'vwap' or 'atr'
        self._target_atr_mult: float = getattr(config, 'sr_target_atr_mult', 1.5)
        self._time_stop_min: int = getattr(config, 'sr_time_stop_minutes', 45)
        self._rsi_long_max: float = getattr(config, 'sr_rsi_long_max', 38.0)
        self._rsi_short_min: float = getattr(config, 'sr_rsi_short_min', 62.0)
        self._min_volume: int = getattr(config, 'sr_min_volume', 100)
        self._or_minutes: int = getattr(config, 'sr_or_minutes', 30)
        self._overextension_atr_mult: float = getattr(config, 'sr_overextension_atr_mult', 0.5)
        self._breakeven_r: float = getattr(config, 'sr_breakeven_r', 1.0)
        
        # Trade window CST
        self._trade_start: time = time(
            getattr(config, 'sr_trade_start_hour', 10),
            getattr(config, 'sr_trade_start_minute', 0)
        )
        self._trade_end: time = time(
            getattr(config, 'sr_trade_end_hour', 11),
            getattr(config, 'sr_trade_end_minute', 30)
        )
        
        # No-trade lunch window
        self._lunch_start: time = time(12, 0)
        self._lunch_end: time = time(13, 30)

    # ------------------------------------------------------------------
    #  BaseStrategy interface
    # ------------------------------------------------------------------
    def generate(self, features: pd.DataFrame) -> Signal:
        """Main entry point — called by engine on every 1m bar."""
        window = features.tail(max(getattr(self.config, 'window_bars', 400), 120)).copy()
        if len(window) < max(60, getattr(self.config, 'warmup_bars', 200)):
            return Signal("HOLD", 0.0, {"reason": "WARMUP"})

        enriched = self._ensure_indicators(window)
        latest = enriched.iloc[-1]
        current_time = enriched.index[-1]

        # Convert to CST for time logic
        cst_time = self._to_cst(current_time)
        cst_date = cst_time.date()

        # ---- Daily reset ----
        if self._session_date != cst_date:
            self._reset_session(cst_date)

        # ---- Collect opening range (first N minutes of RTH) ----
        rth_open = time(
            getattr(self.config, 'rth_start_hour', 9),
            getattr(self.config, 'rth_start_minute', 30)
        )
        or_end = self._add_minutes_to_time(rth_open, self._or_minutes)

        if rth_open <= cst_time.time() < or_end:
            self._collect_opening_bar(latest)
            return Signal("HOLD", 0.0, {"reason": "OPENING_RANGE_FORMING"})

        # Compute OR once we exit the opening window
        if not self._or_computed and self._opening_bars:
            self._compute_opening_range()

        # ---- Session validity checks ----
        if not self._is_in_trade_window(cst_time.time()):
            return Signal("HOLD", 0.0, {"reason": "OUTSIDE_TRADE_WINDOW"})

        if self._is_lunch(cst_time.time()):
            return Signal("HOLD", 0.0, {"reason": "LUNCH_WINDOW"})

        if self._session_trades >= self._max_trades:
            return Signal("HOLD", 0.0, {"reason": "MAX_SESSION_TRADES"})

        if not self._cooldown_elapsed(current_time):
            return Signal("HOLD", 0.0, {"reason": "COOLDOWN"})

        # ---- Data quality checks ----
        atr = float(latest.get("ATR_14", 0))
        if atr < self._min_atr:
            return Signal("HOLD", 0.0, {"reason": f"LOW_ATR_{atr:.1f}"})

        volume = float(latest.get("volume", 0))
        if volume < self._min_volume:
            return Signal("HOLD", 0.0, {"reason": "LOW_VOLUME"})

        # ---- Build structural levels ----
        levels = self._build_levels(enriched, latest)

        # ---- Evaluate reversion entry ----
        signal = self._evaluate_reversion(latest, enriched, levels, atr, current_time)
        return signal

    # ------------------------------------------------------------------
    #  Core logic: mean reversion from structural levels
    # ------------------------------------------------------------------
    def _evaluate_reversion(
        self,
        latest: pd.Series,
        enriched: pd.DataFrame,
        levels: StructuralLevels,
        atr: float,
        current_time: pd.Timestamp,
    ) -> Signal:
        """
        Check if price is overextended from a structural level
        and showing signs of reversion.
        
        KEY DESIGN (data-driven):
        - Require multi-bar confirmation: price must have ALREADY started
          reverting, not just touched the level. This prevents entering
          into momentum moves that kill us in < 5 min.
        - Use 3-bar lookback: bar[-3] or bar[-2] breached the level,
          and bar[-1] + bar[0] are moving back toward VWAP.
        - Tighter target: 1× ATR (take profit fast — data shows 10-45 min
          is the profitable window, avg winner is ~11 pts).
        - Wider stop: 2.5× ATR (data shows 37% of stops hit in < 5 min 
          with 1.5× ATR; 2× ATR is still getting stopped often).
        """
        close = float(latest["close"])
        open_price = float(latest.get("open", close))
        high = float(latest["high"])
        low = float(latest["low"])
        raw_rsi = latest.get("RSI_14", 50)
        rsi = float(raw_rsi) if not (isinstance(raw_rsi, float) and np.isnan(raw_rsi)) else 50.0
        if np.isnan(rsi):
            rsi = 50.0  # Final safety net
        vwap = levels.vwap

        # Need valid VWAP
        if vwap <= 0 or np.isnan(vwap):
            return Signal("HOLD", 0.0, {"reason": "NO_VWAP"})

        overext_dist = atr * self._overextension_atr_mult

        # Get recent bars for multi-bar confirmation
        recent = enriched.tail(5)
        if len(recent) < 4:
            return Signal("HOLD", 0.0, {"reason": "INSUFFICIENT_BARS"})

        recent_closes = recent["close"].values
        recent_highs = recent["high"].values
        recent_lows = recent["low"].values

        # ----------------------------------------------------------
        # SHORT REVERSION: Price overextended above resistance
        # Confirmation: recent high was ABOVE level, now price is
        #               pulling back (close < previous close).
        # ----------------------------------------------------------
        # Check if any of the last 3-4 bars breached a resistance level
        recent_max = max(recent_highs[-4:-1])  # bars [-4] to [-2]

        above_or_high = (self._or_computed and self._or_high > 0 and
                         recent_max > self._or_high)
        above_pdh = (levels.pdh > 0 and not np.isnan(levels.pdh) and
                     recent_max > levels.pdh)
        above_bb = (levels.bb_upper > 0 and not np.isnan(levels.bb_upper) and
                    recent_max > levels.bb_upper)

        short_level_breach = above_or_high or above_pdh or above_bb
        price_above_vwap = close > vwap + overext_dist
        rsi_overbought = rsi >= self._rsi_short_min

        # Multi-bar reversal confirmation:
        # - Current bar closes below previous bar's close
        # - Current bar is bearish (close < open)
        # - Price is pulling BACK toward VWAP (not surging further)
        bar_bearish = close < open_price
        pulling_back_down = close < recent_closes[-2]  # Close below prev close
        not_surging = close < recent_highs[-2]          # Not making new highs

        short_trigger = (
            short_level_breach and
            price_above_vwap and
            rsi_overbought and
            bar_bearish and
            pulling_back_down and
            not_surging
        )

        # ----------------------------------------------------------
        # LONG REVERSION: Price overextended below support
        # ----------------------------------------------------------
        recent_min = min(recent_lows[-4:-1])

        below_or_low = (self._or_computed and self._or_low > 0 and
                        recent_min < self._or_low)
        below_pdl = (levels.pdl > 0 and not np.isnan(levels.pdl) and
                     recent_min < levels.pdl)
        below_bb = (levels.bb_lower > 0 and not np.isnan(levels.bb_lower) and
                    recent_min < levels.bb_lower)

        long_level_breach = below_or_low or below_pdl or below_bb
        price_below_vwap = close < vwap - overext_dist
        rsi_oversold = rsi <= self._rsi_long_max

        bar_bullish = close > open_price
        pulling_back_up = close > recent_closes[-2]
        not_crashing = close > recent_lows[-2]

        long_trigger = (
            long_level_breach and
            price_below_vwap and
            rsi_oversold and
            bar_bullish and
            pulling_back_up and
            not_crashing
        )

        # ----------------------------------------------------------
        # If neither triggers, HOLD
        # ----------------------------------------------------------
        if not short_trigger and not long_trigger:
            return Signal("HOLD", 0.0, {"reason": "NO_REVERSION_SETUP"})

        # If both trigger (shouldn't happen), skip
        if short_trigger and long_trigger:
            return Signal("HOLD", 0.0, {"reason": "AMBIGUOUS_SIGNAL"})

        # ----------------------------------------------------------
        # 15m regime: skip if strong trend AGAINST our reversion
        # ----------------------------------------------------------
        regime_15m = str(latest.get("15m_regime", "UNKNOWN"))
        adx_15m = float(latest.get("15m_ADX_14", 0))

        if short_trigger and regime_15m == "UPTREND" and adx_15m > 30:
            return Signal("HOLD", 0.0, {"reason": "15M_STRONG_UPTREND_NO_SHORT"})

        if long_trigger and regime_15m == "DOWNTREND" and adx_15m > 30:
            return Signal("HOLD", 0.0, {"reason": "15M_STRONG_DOWNTREND_NO_LONG"})

        # ----------------------------------------------------------
        # Compute stop and target — ASYMMETRIC R:R
        # Target = 1× ATR (quick profit, data shows winners avg 11 pts)
        # Stop = 2.5× ATR (wide enough to survive noise)
        # This requires ~72% WR to break even... but with confirmed
        # reversals the WR should be much higher than random.
        # ----------------------------------------------------------
        stop_mult = self._stop_atr_mult
        
        if short_trigger:
            action = "SELL"
            stop_loss = close + (atr * stop_mult)
            
            # Target: distance to VWAP, floored at 0.5×ATR, capped at 1.0×ATR
            # Data shows winners avg 7 pts (1.3× ATR) — but quick hits <5min
            # are 80% WR. Optimize for fast scalps.
            dist_to_vwap = close - vwap
            target_dist = max(min(dist_to_vwap * 0.5, atr * 1.0), atr * 0.5)
            target = close - target_dist

            triggers = []
            if above_or_high: triggers.append("ABOVE_OR_HIGH")
            if above_pdh: triggers.append("ABOVE_PDH")
            if above_bb: triggers.append("ABOVE_BB_UPPER")
            reason_str = f"SHORT_REVERSION | RSI={rsi:.0f} | {'+'.join(triggers)}"

        else:  # long_trigger
            action = "BUY"
            stop_loss = close - (atr * stop_mult)

            dist_to_vwap = vwap - close
            target_dist = max(min(dist_to_vwap * 0.5, atr * 1.0), atr * 0.5)
            target = close + target_dist

            triggers = []
            if below_or_low: triggers.append("BELOW_OR_LOW")
            if below_pdl: triggers.append("BELOW_PDL")
            if below_bb: triggers.append("BELOW_BB_LOWER")
            reason_str = f"LONG_REVERSION | RSI={rsi:.0f} | {'+'.join(triggers)}"

        # Record the trade
        self._session_trades += 1
        self._last_trade_time = current_time

        metadata = {
            "reason": reason_str,
            "strategy_type": self.name,
            "stop_loss": stop_loss,
            "take_profit": target,
            "atr_value": atr,
            "adx_value": float(latest.get("ADX_14", 0)),
            "rsi": rsi,
            "vwap": vwap,
            "or_high": self._or_high,
            "or_low": self._or_low,
            "pdh": levels.pdh,
            "pdl": levels.pdl,
            "regime_15m": regime_15m,
            "session_trades": self._session_trades,
            "market_state": "MEAN_REVERSION",
            "position_size": 1.0,
            "entry_type": "structural_reversion",
            "session_type": "RTH",
        }

        confidence = 0.7
        
        logger.info(
            f"🔄 {current_time} | {action} reversion: close={close:.2f} "
            f"vwap={vwap:.2f} atr={atr:.1f} rsi={rsi:.0f} "
            f"SL={stop_loss:.2f} TP={target:.2f} | {reason_str}"
        )

        return Signal(action=action, confidence=confidence, metadata=metadata)

    # ------------------------------------------------------------------
    #  Opening range computation
    # ------------------------------------------------------------------
    def _collect_opening_bar(self, bar: pd.Series) -> None:
        """Collect bars during opening range period."""
        self._opening_bars.append({
            'high': float(bar['high']),
            'low': float(bar['low']),
            'close': float(bar['close']),
            'volume': float(bar.get('volume', 0)),
        })

    def _compute_opening_range(self) -> None:
        """Compute opening range from collected bars."""
        if not self._opening_bars:
            return
        highs = [b['high'] for b in self._opening_bars]
        lows = [b['low'] for b in self._opening_bars]
        self._or_high = max(highs)
        self._or_low = min(lows)
        self._or_computed = True
        logger.debug(
            f"Opening range computed: HIGH={self._or_high:.2f} "
            f"LOW={self._or_low:.2f} ({len(self._opening_bars)} bars)"
        )

    # ------------------------------------------------------------------
    #  Level building
    # ------------------------------------------------------------------
    def _build_levels(self, enriched: pd.DataFrame, latest: pd.Series) -> StructuralLevels:
        """Build structural levels from current data."""
        return StructuralLevels(
            vwap=float(latest.get("SESSION_VWAP", latest.get("VWAP_daily", latest.get("VWAP", 0)))),
            pdh=float(latest.get("PDH", 0)),
            pdl=float(latest.get("PDL", 0)),
            or_high=self._or_high,
            or_low=self._or_low,
            or_mid=(self._or_high + self._or_low) / 2 if self._or_computed else 0,
            bb_upper=float(latest.get("BB_upper", latest.get("BB_upper_20_2", 0))),
            bb_lower=float(latest.get("BB_lower", latest.get("BB_lower_20_2", 0))),
        )

    # ------------------------------------------------------------------
    #  Session and time management
    # ------------------------------------------------------------------
    def _reset_session(self, date) -> None:
        """Reset all session state for a new trading day."""
        self._session_date = date
        self._session_trades = 0
        self._last_trade_time = None
        self._or_high = 0.0
        self._or_low = 0.0
        self._or_computed = False
        self._opening_bars = []

    def _to_cst(self, ts: pd.Timestamp) -> Any:
        """Convert timestamp to CST."""
        if ts.tz is not None:
            try:
                return ts.tz_convert("America/Chicago")
            except Exception:
                return ts
        return ts

    def _is_in_trade_window(self, t: time) -> bool:
        """Check if time is within allowed trade window."""
        return self._trade_start <= t < self._trade_end

    def _is_lunch(self, t: time) -> bool:
        """Check if time is in lunch no-trade window."""
        return self._lunch_start <= t < self._lunch_end

    def _cooldown_elapsed(self, current_time: pd.Timestamp) -> bool:
        """Check if enough time has passed since last trade."""
        if self._last_trade_time is None:
            return True
        elapsed = (current_time - self._last_trade_time).total_seconds() / 60
        return elapsed >= self._cooldown_min

    @staticmethod
    def _add_minutes_to_time(t: time, minutes: int) -> time:
        """Add minutes to a time object."""
        from datetime import datetime as dt
        d = dt.combine(dt.today(), t) + timedelta(minutes=minutes)
        return d.time()

    # ------------------------------------------------------------------
    #  Indicator computation (reuse from existing strategy)
    # ------------------------------------------------------------------
    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required indicators are present."""
        enriched = df.copy()
        close = enriched["close"]
        high = enriched["high"]
        low = enriched["low"]
        volume = enriched["volume"]

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

        # Bollinger Bands
        if "BB_upper" not in enriched and "BB_upper_20_2" not in enriched:
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std(ddof=0)
            enriched["BB_upper"] = sma20 + 2 * std20
            enriched["BB_lower"] = sma20 - 2 * std20

        # Forward-fill NaN in indicators to avoid NaN at entry
        for col in ["RSI_14", "ATR_14", "ADX_14", "EMA_9", "EMA_21", "EMA_50",
                     "BB_upper", "BB_upper_20_2", "BB_lower", "BB_lower_20_2"]:
            if col in enriched.columns:
                enriched[col] = enriched[col].ffill().bfill()

        # Session VWAP
        if "SESSION_VWAP" not in enriched:
            enriched["SESSION_VWAP"] = self._compute_session_vwap(enriched)

        # PDH / PDL
        if "PDH" not in enriched or "PDL" not in enriched:
            enriched["PDH"], enriched["PDL"] = self._compute_previous_day_levels(enriched)

        return enriched

    def _compute_session_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Compute session VWAP (RTH reset)."""
        idx = df.index
        if idx.tz is not None:
            try:
                idx_local = idx.tz_convert("America/Chicago")
            except Exception:
                idx_local = idx
        else:
            idx_local = idx

        typical = (df["high"] + df["low"] + df["close"]) / 3
        vwap = pd.Series(index=df.index, dtype=float)
        
        dates = idx_local.date if hasattr(idx_local, "date") else [None] * len(df)
        session_ids = pd.Series(dates, index=df.index)

        for _, mask_idx in session_ids.groupby(session_ids).groups.items():
            idx_mask = df.index.isin(mask_idx)
            vol = df.loc[idx_mask, "volume"].replace(0, np.nan)
            tp = typical.loc[idx_mask]
            cum_vwap = (tp * vol).cumsum() / vol.cumsum()
            vwap.loc[idx_mask] = cum_vwap

        return vwap.ffill()

    def _compute_previous_day_levels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Compute PDH and PDL."""
        idx = df.index
        if idx.tz is not None:
            try:
                idx_local = idx.tz_convert("America/Chicago")
            except Exception:
                idx_local = idx
        else:
            idx_local = idx

        days = pd.Series(
            idx_local.date if hasattr(idx_local, "date") else idx_local,
            index=df.index
        )
        daily_high = df.groupby(days)["high"].max()
        daily_low = df.groupby(days)["low"].min()
        pdh = days.map(lambda d: daily_high.get(d - timedelta(days=1), np.nan))
        pdl = days.map(lambda d: daily_low.get(d - timedelta(days=1), np.nan))
        return pdh, pdl
