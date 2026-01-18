"""MES 1-minute close strategy with minimal indicator stack."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta, datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..config import OneMinuteStrategyConfig
from ..features.feature_engineer import _adx, _atr, _ema, _rsi
from ..utils.structured_logging import log_structured_event
from .base import BaseStrategy, Signal


@dataclass
class StrategyDecision:
    action: str
    confidence: float
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, float]] = None


class MesOneMinuteTrendStrategy(BaseStrategy):
    """Single-instrument MES strategy evaluated strictly on 1-minute closes."""

    name = "mes_one_minute_trend"

    def __init__(self, config: OneMinuteStrategyConfig):
        self.config = config
        self._trade_log: list[pd.Timestamp] = []
        self._hourly_log: list[pd.Timestamp] = []

    def _is_rth(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within RTH (Regular Trading Hours)."""
        if not self.config.rth_only:
            return True
            
        # Convert to local time if needed
        if timestamp.tz is not None:
            try:
                local_ts = timestamp.tz_convert("America/Chicago")
            except Exception:
                local_ts = timestamp
        else:
            local_ts = timestamp
            
        t = local_ts.time() if hasattr(local_ts, 'time') else time(0, 0)
        rth_start = time(self.config.rth_start_hour, self.config.rth_start_minute)
        rth_end = time(self.config.rth_end_hour, self.config.rth_end_minute)
        
        return rth_start <= t < rth_end

    def _is_overnight_session(self, timestamp: pd.Timestamp) -> bool:
        """
        JAN 11 2026: Check if timestamp is within overnight/evening session.
        
        Overnight session: 6:00 PM - 9:30 AM ET (ES futures globex)
        This is the complement of RTH, but excludes the weekend close period.
        """
        allow_overnight = getattr(self.config, 'allow_overnight_trading', False)
        if not allow_overnight:
            return False
            
        # Convert to local time
        if timestamp.tz is not None:
            try:
                local_ts = timestamp.tz_convert("America/Chicago")
            except Exception:
                local_ts = timestamp
        else:
            local_ts = timestamp
            
        t = local_ts.time() if hasattr(local_ts, 'time') else time(0, 0)
        
        overnight_start = time(
            getattr(self.config, 'overnight_start_hour', 18), 0
        )
        overnight_end = time(
            getattr(self.config, 'overnight_end_hour', 9),
            getattr(self.config, 'overnight_end_minute', 30)
        )
        
        # Overnight spans midnight: 18:00 -> 09:30 next day
        # So we check: t >= 18:00 OR t < 09:30
        return t >= overnight_start or t < overnight_end

    def _get_session_type(self, timestamp: pd.Timestamp) -> str:
        """Return 'RTH', 'OVERNIGHT', or 'CLOSED' for the current timestamp."""
        if self._is_rth(timestamp):
            return "RTH"
        elif self._is_overnight_session(timestamp):
            return "OVERNIGHT"
        else:
            return "CLOSED"

    def generate(self, features: pd.DataFrame) -> Signal:
        window = features.tail(max(self.config.window_bars, 120)).copy()
        if len(window) < max(60, self.config.warmup_bars):
            return Signal("HOLD", 0.0, {"reason": "WARMUP"})

        enriched = self._ensure_indicators(window)
        latest = enriched.iloc[-1]
        prev = enriched.iloc[-2]
        current_time = enriched.index[-1]
        
        # Standard 1m values
        atr_series = enriched["ATR_14"].tail(120).dropna()
        atr_value = float(latest["ATR_14"])
        adx_value = float(latest["ADX_14"])
        
        # JAN 17 2026: Use 15m indicators if available per user request
        # Swapping these ensures log consistency and trade params (stops) match user intent
        use_mtf = getattr(self.config, 'use_mtf_regime', False)

        if use_mtf:
            if "15m_ATR_14" in latest and not np.isnan(latest["15m_ATR_14"]):
                atr_value = float(latest["15m_ATR_14"])
                if "15m_ATR_14" in enriched.columns:
                    atr_series = enriched["15m_ATR_14"].tail(120).dropna()
            
            if "15m_ADX_14" in latest and not np.isnan(latest["15m_ADX_14"]):
                adx_value = float(latest["15m_ADX_14"])

        market_state = "TRENDING" if adx_value >= self.config.trend_adx_threshold else "RANGING"
        
        # 2. Trend Label (Direction)
        if use_mtf and "15m_regime" in latest:
            regime_val = str(latest["15m_regime"])
            if regime_val == "UPTREND":
                trend_label = "UPTREND"
            elif regime_val == "DOWNTREND":
                trend_label = "DOWNTREND"
            elif regime_val == "RANGING":
                trend_label = "CHOP"
            else:
                trend_label = self._classify_trend(latest)
        else:
            trend_label = self._classify_trend(latest)

        atr_low, atr_high = self._atr_percentile_bounds(atr_series)
        candle_range = float(latest["high"] - latest["low"])

        reasons: list[str] = []
        filters_block = False
        
        # JAN 11 2026: Session-aware trading
        # - RTH: Use 1m signals with standard filters
        # - OVERNIGHT: Use 30m signals with stricter filters (only on 30m bar close)
        # - CLOSED: No trading
        session_type = self._get_session_type(current_time)
        
        if session_type == "CLOSED":
            return Signal("HOLD", 0.0, {"reason": "MARKET_CLOSED"})
        
        # JAN 11 2026: Overnight session uses 30m timeframe for cleaner signals
        is_overnight = session_type == "OVERNIGHT"
        
        if is_overnight:
            # CRITICAL: Only make decisions on 30m bar boundaries
            # A 30m bar closes at :00 and :30 minutes
            bar_minute = current_time.minute
            if bar_minute not in (0, 30):
                # Not a 30m bar close - skip this 1m bar
                return Signal("HOLD", 0.0, {"reason": "WAITING_30M_CLOSE"})
            
            # Check if 30m data is available
            regime_30m = str(latest.get("30m_regime", "UNKNOWN"))
            adx_30m = float(latest.get("30m_ADX_14", 0))
            atr_30m = float(latest.get("30m_ATR_14", 0))
            
            if regime_30m == "UNKNOWN" or adx_30m == 0:
                return Signal("HOLD", 0.0, {"reason": "NO_30M_DATA"})
            
            # Stricter overnight filters
            # TUNING (Jan 17 2026): Lowered ADX threshold from 25 to 20 to catch moves earlier
            overnight_adx_thresh = getattr(self.config, 'overnight_adx_threshold', 20.0)
            overnight_atr_pct = getattr(self.config, 'overnight_atr_percentile', 0.75)
            overnight_min_vol = getattr(self.config, 'overnight_min_volume', 50)
            
            # Use 30m ADX for trend check
            if adx_30m < overnight_adx_thresh:
                return Signal("HOLD", 0.0, {"reason": "30M_WEAK_TREND", "30m_adx": adx_30m})
            
            # Use 30m regime for direction
            if regime_30m == "RANGING":
                return Signal("HOLD", 0.0, {"reason": "30M_RANGING"})
            
            # Stricter volume for overnight (use aggregated 30m volume would be better)
            # For now, check 1m volume is reasonable
            bar_volume = float(latest.get("volume", 0))
            if bar_volume < overnight_min_vol:
                return Signal("HOLD", 0.0, {"reason": "OVERNIGHT_LOW_VOLUME"})
            
            # Stricter ATR percentile for overnight using 30m ATR
            atr_30m_series = enriched.get("30m_ATR_14", pd.Series()).tail(200).dropna()
            if len(atr_30m_series) >= 50:
                high_atr_30m_threshold = atr_30m_series.quantile(overnight_atr_pct)
                if atr_30m < high_atr_30m_threshold:
                    return Signal("HOLD", 0.0, {"reason": "OVERNIGHT_LOW_ATR"})
            
            # Override trend_label with 30m regime for signal generation
            if regime_30m == "UPTREND":
                trend_label = "UPTREND"
            elif regime_30m == "DOWNTREND":
                trend_label = "DOWNTREND"
        
        # RTH session: use original filters
        elif session_type == "RTH":
            # Standard RTH volume filter
            bar_volume = float(latest.get("volume", 0))
            if bar_volume < self.config.min_bar_volume:
                filters_block = True
                reasons.append("LOW_VOLUME")
        else:
            # Unknown session - skip
            return Signal("HOLD", 0.0, {"reason": "OUTSIDE_RTH"})
        
        # Only hard-block on invalid ATR (NaN or zero)
        if atr_value <= 0 or np.isnan(atr_value):
            filters_block = True
            reasons.append("ATR_INVALID")
        
        # JAN 11 2026: High ATR regime filter - only trade when volatility is elevated
        # Backtest showed: Low ATR 27% WR, Med ATR 20% WR, High ATR 63.6% WR
        require_high_atr = getattr(self.config, 'require_high_atr', False)
        if require_high_atr and not filters_block:
            # TUNING (Jan 17 2026): Lowered from 0.67 (top 33%) to 0.50 (median) to increase trade frequency
            high_atr_pct = getattr(self.config, 'high_atr_percentile', 0.50)
            lookback = getattr(self.config, 'high_atr_lookback', 200)
            
            # JAN 17 2026: Check 15m ATR for regime if available
            if use_mtf and "15m_ATR_14" in enriched.columns:
                 atr_check_series = enriched["15m_ATR_14"].tail(lookback).dropna()
                 atr_check_value = float(latest.get("15m_ATR_14", 0))
            else:
                 atr_check_series = enriched["ATR_14"].tail(lookback).dropna()
                 atr_check_value = atr_value

            if len(atr_check_series) >= 50:
                high_atr_threshold = atr_check_series.quantile(high_atr_pct)
                if atr_check_value < high_atr_threshold:
                    filters_block = True
                    reasons.append("LOW_ATR_REGIME")  # Not in top tercile
        
        # Soft warnings (logged but don't block) for ATR percentile extremes
        # These inform the decision but don't prevent trading
        if atr_value < atr_low:
            reasons.append("ATR_LOW_WARN")  # Changed from hard block
        elif atr_value > atr_high:
            reasons.append("ATR_HIGH_WARN")  # Changed from hard block

        # TINY_CANDLE: Only block if candle range is 0 AND we would trade
        # Non-zero small candles are fine - the market is just quiet
        if candle_range == 0:
            # Zero range = no price movement, can't trade this bar
            filters_block = True
            reasons.append("ZERO_RANGE")
        elif candle_range < self.config.tiny_candle_atr_factor * atr_value:
            # Small candle - warn but allow trading on valid setups
            reasons.append("SMALL_CANDLE_WARN")

        metadata: Dict[str, float | str] = {
            "market_state": market_state,
            "trend_label": trend_label,
            "atr_value": atr_value,
            "adx_value": adx_value,
            "ema9": float(latest["EMA_9"]),
            "ema21": float(latest["EMA_21"]),
            "vwap": float(latest["SESSION_VWAP"]),
            "pdh": float(latest.get("PDH", np.nan)),
            "pdl": float(latest.get("PDL", np.nan)),
            "candle_range": candle_range,
            "bar_volume": bar_volume,
            "session_type": session_type,  # JAN 11 2026: Track RTH vs OVERNIGHT
        }
        
        # JAN 11 2026: Add 30m metadata if in overnight session
        if is_overnight:
            metadata["30m_regime"] = str(latest.get("30m_regime", "UNKNOWN"))
            metadata["30m_adx"] = float(latest.get("30m_ADX_14", 0))
            metadata["30m_atr"] = float(latest.get("30m_ATR_14", 0))
        
        # JAN 11 2026: Check 15m regime alignment if enabled (RTH only)
        use_mtf_regime = getattr(self.config, 'use_mtf_regime', False)
        regime_15m = str(latest.get("15m_regime", "UNKNOWN"))
        metadata["15m_regime"] = regime_15m
        
        # Skip 15m checks for overnight - already using 30m
        if not is_overnight and use_mtf_regime and regime_15m not in ("UNKNOWN", ""):
            require_trend = getattr(self.config, 'require_regime_trend', True)
            
            if require_trend and regime_15m == "RANGING":
                # Don't trade when 15m shows ranging - wait for trend
                reasons.append("15M_RANGING")
                filters_block = True
            elif trend_label == "UPTREND" and regime_15m == "DOWNTREND":
                # 1m uptrend but 15m downtrend - counter-trend, skip
                reasons.append("15M_COUNTER_TREND")
                filters_block = True
            elif trend_label == "DOWNTREND" and regime_15m == "UPTREND":
                # 1m downtrend but 15m uptrend - counter-trend, skip  
                reasons.append("15M_COUNTER_TREND")
                filters_block = True

        decision = StrategyDecision(action="HOLD", confidence=0.0, reason="INIT", metadata=metadata)

        if filters_block or trend_label == "CHOP":
            decision = StrategyDecision("HOLD", 0.0, ",".join(reasons) or "CHOP", metadata=metadata)
        else:
            pullback_ok, pull_reason = self._pullback_confirmation(enriched)
            # JAN 11 2026: Disable extensions if configured
            breakout_decision = None
            if self.config.breakout_enabled and getattr(self.config, 'enable_trend_extensions', True):
                breakout_decision = self._breakout_check(latest, adx_value, metadata)

            if breakout_decision:
                decision = breakout_decision
                decision.metadata.update(metadata)
            elif trend_label == "UPTREND" and pullback_ok:
                decision = self._enter_with_brackets(
                    direction="BUY",
                    close=float(latest["close"]),
                    atr=atr_value,
                    reason=pull_reason or "PULLBACK_LONG",
                    base_conf=0.68,
                    extra_meta=metadata,
                )
            elif trend_label == "DOWNTREND" and pullback_ok:
                decision = self._enter_with_brackets(
                    direction="SELL",
                    close=float(latest["close"]),
                    atr=atr_value,
                    reason=pull_reason or "PULLBACK_SHORT",
                    base_conf=0.68,
                    extra_meta=metadata,
                )
            else:
                decision = StrategyDecision("HOLD", 0.0, "NO_SETUP", metadata=metadata)

        self._log_decision(enriched.index[-1], decision, latest)
        meta = decision.metadata or {}
        meta.setdefault("reason", decision.reason)
        meta.setdefault("strategy_type", self.name)
        meta["market_state"] = market_state
        meta["trend_label"] = trend_label

        return Signal(action=decision.action, confidence=decision.confidence, metadata=meta)

    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        close, high, low, volume = (
            enriched["close"],
            enriched["high"],
            enriched["low"],
            enriched["volume"],
        )
        if "EMA_9" not in enriched:
            enriched["EMA_9"] = _ema(close, 9)
        if "EMA_21" not in enriched:
            enriched["EMA_21"] = _ema(close, 21)
        if "RSI_14" not in enriched:
            enriched["RSI_14"] = _rsi(close, 14)
        if "ATR_14" not in enriched:
            enriched["ATR_14"] = _atr(high, low, close, 14)
        if "ADX_14" not in enriched:
            enriched["ADX_14"] = _adx(high, low, close, 14)
        if "SESSION_VWAP" not in enriched:
            enriched["SESSION_VWAP"] = self._compute_session_vwap(enriched, self.config.use_eth_session)
        if "PDH" not in enriched or "PDL" not in enriched:
            enriched["PDH"], enriched["PDL"] = self._compute_previous_day_levels(enriched)
        return enriched

    def _compute_session_vwap(self, df: pd.DataFrame, use_eth: bool) -> pd.Series:
        idx_local = df.index
        if idx_local.tz is not None:
            try:
                idx_local = idx_local.tz_convert("America/Chicago")
            except Exception:
                idx_local = idx_local.tz_localize(None)
        times = idx_local.time if hasattr(idx_local, "time") else [time(0, 0)] * len(df)
        active_mask = pd.Series(True, index=df.index)
        if not use_eth:
            active_mask = pd.Series(
                [(t >= time(8, 30)) and (t <= time(15, 0)) for t in times],
                index=df.index,
            )

        session_ids = pd.Series(idx_local.date if hasattr(idx_local, "date") else idx_local, index=df.index)
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vwap = pd.Series(index=df.index, dtype=float)

        for session, mask in session_ids.groupby(session_ids).groups.items():
            mask_idx = df.index.isin(mask)
            active_idx = mask_idx & active_mask
            if not active_idx.any():
                continue
            vol = df.loc[active_idx, "volume"].replace(0, np.nan)
            tp = typical.loc[active_idx]
            cum_vwap = (tp * vol).cumsum() / vol.cumsum()
            vwap.loc[active_idx] = cum_vwap
            # Forward-fill non-active bars within the same session with last active value
            inactive = mask_idx & (~active_mask)
            if inactive.any() and not cum_vwap.empty:
                vwap.loc[inactive] = cum_vwap.iloc[-1]

        return vwap.ffill()

    def _compute_previous_day_levels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        idx_local = df.index
        if idx_local.tz is not None:
            try:
                idx_local = idx_local.tz_convert("America/Chicago")
            except Exception:
                idx_local = idx_local.tz_localize(None)
        days = pd.Series(idx_local.date if hasattr(idx_local, "date") else idx_local, index=df.index)
        daily_high = df.groupby(days)["high"].max()
        daily_low = df.groupby(days)["low"].min()
        pdh = days.map(lambda d: daily_high.get(d - timedelta(days=1), np.nan))
        pdl = days.map(lambda d: daily_low.get(d - timedelta(days=1), np.nan))
        return pdh, pdl

    def _classify_trend(self, latest) -> str:
        adx = float(latest["ADX_14"])
        market_state = "TRENDING" if adx >= self.config.trend_adx_threshold else "RANGING"
        if market_state != "TRENDING":
            return "CHOP"
        close = float(latest["close"])
        vwap = float(latest["SESSION_VWAP"])
        ema9 = float(latest["EMA_9"])
        ema21 = float(latest["EMA_21"])
        if close > vwap and ema9 > ema21 and close > ema9:
            return "UPTREND"
        if close < vwap and ema9 < ema21 and close < ema9:
            return "DOWNTREND"
        return "CHOP"

    def _atr_percentile_bounds(self, atr_series: pd.Series) -> Tuple[float, float]:
        if atr_series.empty:
            return 0.0, float("inf")
        low = float(atr_series.quantile(self.config.atr_percentile_low))
        high = float(atr_series.quantile(self.config.atr_percentile_high))
        return low, high

    def _pullback_confirmation(self, df: pd.DataFrame) -> Tuple[bool, str]:
        if len(df) < 5:
            return False, "INSUFFICIENT_HISTORY"
        recent = df.tail(max(4, self.config.pullback_lookback + 1))
        rsi_now = float(recent["RSI_14"].iloc[-1])
        rsi_prev = float(recent["RSI_14"].iloc[-2])
        close_now = float(recent["close"].iloc[-1])
        close_prev = float(recent["close"].iloc[-2])
        ema9_now = float(recent["EMA_9"].iloc[-1])
        ema9_prev = float(recent["EMA_9"].iloc[-2])

        # RSI pullback long/short windows
        if 40 <= rsi_now <= 55 and rsi_now > rsi_prev:
            return True, "RSI_PULLBACK_LONG"
        if 45 <= rsi_now <= 60 and rsi_now < rsi_prev:
            return True, "RSI_PULLBACK_SHORT"

        # EMA reclaim across last few bars
        reclaim_window = recent.tail(self.config.pullback_lookback + 1)
        closes = reclaim_window["close"].values
        ema9_vals = reclaim_window["EMA_9"].values
        if len(closes) >= 2:
            prev_below = closes[-2] < ema9_vals[-2]
            now_above = closes[-1] > ema9_vals[-1]
            prev_above = closes[-2] > ema9_vals[-2]
            now_below = closes[-1] < ema9_vals[-1]
            if prev_below and now_above:
                return True, "EMA_RECLAIM_LONG"
            if prev_above and now_below:
                return True, "EMA_RECLAIM_SHORT"
        if close_prev < ema9_prev and close_now > ema9_now:
            return True, "EMA_RECLAIM_LONG"
        if close_prev > ema9_prev and close_now < ema9_now:
            return True, "EMA_RECLAIM_SHORT"
        return False, "NO_PULLBACK"

    def _enter_with_brackets(
        self,
        direction: str,
        close: float,
        atr: float,
        reason: str,
        base_conf: float,
        extra_meta: Dict[str, float | str],
    ) -> StrategyDecision:
        stop_dist = atr * self.config.stop_atr_multiplier
        
        # JAN 11 2026: Enforce minimum stop distance to pass RiskGate
        # RiskGate requires min_stop_points=6.0 (now tuned to 3.0), so ensure stop_dist >= 3.25 points
        min_stop_points = 3.25  # Slightly above RiskGate minimum to ensure passage
        if stop_dist < min_stop_points:
            stop_dist = min_stop_points
        
        take_profit_dist = stop_dist * self.config.take_profit_multiple
        if stop_dist <= 0 or take_profit_dist <= 0:
            return StrategyDecision("HOLD", 0.0, "INVALID_STOPS", metadata=extra_meta)
        if direction == "BUY":
            stop_loss = close - stop_dist
            take_profit = close + take_profit_dist
        else:
            stop_loss = close + stop_dist
            take_profit = close - take_profit_dist

        if (direction == "BUY" and (stop_loss >= close or take_profit <= close)) or (
            direction == "SELL" and (stop_loss <= close or take_profit >= close)
        ):
            return StrategyDecision("HOLD", 0.0, "STOP_TP_INVALID", metadata=extra_meta)

        meta = dict(extra_meta)
        meta.update(
            {
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr_value": atr,
                "decision_reason": reason,
            }
        )
        return StrategyDecision(direction, base_conf, reason, stop_loss, take_profit, meta)

    def _breakout_check(
        self, latest, adx_value: float, base_meta: Dict[str, float | str]
    ) -> Optional[StrategyDecision]:
        close = float(latest["close"])
        pdh = float(latest.get("PDH", np.nan))
        pdl = float(latest.get("PDL", np.nan))
        if self.config.breakout_use_or_levels:
            pdh = float(latest.get("ORH", pdh))
            pdl = float(latest.get("ORL", pdl))
        vwap = float(latest["SESSION_VWAP"])
        atr = float(latest["ATR_14"])
        high = float(latest["high"])
        low = float(latest["low"])
        candle_range = high - low if high >= low else 0.0
        body_top_threshold = low + candle_range * 0.70
        body_bottom_threshold = high - candle_range * 0.70

        # Long breakout
        if pd.notna(pdh) and close > pdh and close > vwap and adx_value >= self.config.breakout_adx_threshold:
            if getattr(self.config, "breakout_strength_filter", True) and close < body_top_threshold:
                return None
            return self._enter_with_brackets(
                direction="BUY",
                close=close,
                atr=atr,
                reason="BREAKOUT_LONG",
                base_conf=0.70,
                extra_meta=base_meta,
            )
        # Short breakout
        if pd.notna(pdl) and close < pdl and close < vwap and adx_value >= self.config.breakout_adx_threshold:
            if getattr(self.config, "breakout_strength_filter", True) and close > body_bottom_threshold:
                return None
            return self._enter_with_brackets(
                direction="SELL",
                close=close,
                atr=atr,
                reason="BREAKOUT_SHORT",
                base_conf=0.70,
                extra_meta=base_meta,
            )
        return None

    def _log_decision(self, ts, decision: StrategyDecision, latest) -> None:
        payload = {
            "timestamp": str(ts),
            "action": decision.action,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "trend": decision.metadata.get("trend_label") if decision.metadata else "",
            "state": decision.metadata.get("market_state") if decision.metadata else "",
            "close": float(latest.get("close", np.nan)),
            "ema9": float(latest.get("EMA_9", np.nan)),
            "ema21": float(latest.get("EMA_21", np.nan)),
            "vwap": float(latest.get("SESSION_VWAP", np.nan)),
            "atr": float(decision.metadata.get("atr_value", np.nan)) if decision.metadata else float(latest.get("ATR_14", np.nan)),
            "adx": float(decision.metadata.get("adx_value", np.nan)) if decision.metadata else float(latest.get("ADX_14", np.nan)),
            "pdh": float(latest.get("PDH", np.nan)),
            "pdl": float(latest.get("PDL", np.nan)),
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
        }
        logger.info(
            f"🕐 {ts} | state={payload['state']} trend={payload['trend']} act={decision.action} conf={decision.confidence:.2f} "
            f"reason={decision.reason} close={payload['close']:.2f} ema9={payload['ema9']:.2f} "
            f"ema21={payload['ema21']:.2f} vwap={payload['vwap']:.2f} atr={payload['atr']:.2f} adx={payload['adx']:.2f}"
        )
        log_structured_event(
            agent="mes_one_minute",
            event_type="decision",
            message=decision.reason,
            payload=payload,
        )
