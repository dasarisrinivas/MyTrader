"""BUY continuation entry module — buys pullbacks in bullish acceptance.

Optimised for morning RTH (09:30-11:00 CST).  Entry requires
``MarketState.is_acceptance == True``, a valid pullback into EMA9/EMA21/VWAP,
a bullish confirmation candle, and a minimum 2:1 risk/reward ratio.

Inputs:
    market_state (MarketStateResult): Current phase/trend from the detector.
    data (dict): Current OHLCV bar with indicator columns.
    recent_bars (DataFrame, optional): Recent price history for pullback analysis.
    timestamp (datetime, optional): Current time for session filtering.

Outputs:
    EntrySignal: ``BUY`` with confidence & stop/target, or ``HOLD``.

Side-effects:
    None — pure analysis, no I/O.
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from loguru import logger

from ..market_state import MarketStateResult, TrendDirection
from .session_time import SessionWindow
from .signals import EntrySignal, PullbackAnalysis


class BuyContinuationModule:
    """BUY continuation module — buys pullbacks in acceptance.

    See module docstring for full acceptance / entry / stop criteria.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # RSI thresholds
        self.rsi_min = config.get("rsi_min", 55)
        self.rsi_max = config.get("rsi_max", 70)
        self.rsi_overextension = config.get("rsi_overextension", 72)
        self.rsi_ideal_min = config.get("rsi_ideal_min", 58)
        self.rsi_ideal_max = config.get("rsi_ideal_max", 65)

        # MACD thresholds
        self.macd_acceptance_threshold = config.get("macd_acceptance_threshold", 0.20)
        self.macd_slope_min = config.get("macd_slope_min", 0.0)

        # ADX thresholds
        self.adx_trend_threshold = config.get("adx_trend_threshold", 25)
        self.adx_strong_threshold = config.get("adx_strong_threshold", 30)

        # Pullback parameters
        self.pullback_depth_min_pct = config.get("pullback_depth_min_pct", 0.05)
        self.pullback_depth_max_pct = config.get("pullback_depth_max_pct", 0.50)
        self.max_vwap_distance_atr = config.get("max_vwap_distance_atr", 2.0)
        self.ema_proximity_pct = config.get("ema_proximity_pct", 0.15)

        # Confirmation
        self.require_bullish_close = config.get("require_bullish_close", True)
        self.require_higher_low = config.get("require_higher_low", False)
        self.min_candle_body_ratio = config.get("min_candle_body_ratio", 0.4)
        self.engulfing_required = config.get("engulfing_required", False)

        # Risk management
        self.stop_atr_mult = config.get("stop_atr_mult", 1.5)
        self.target_risk_mult = config.get("target_risk_mult", 2.0)
        self.min_stop_points = config.get("min_stop_points", 3.25)
        self.max_stop_points = config.get("max_stop_points", 8.0)
        self.min_rr_ratio = config.get("min_rr_ratio", 2.0)

        # Higher-timeframe alignment
        self.require_htf_alignment = config.get("require_htf_alignment", True)

        # Session timing (CST)
        self.morning_open_start = time(9, 30)
        self.morning_open_end = time(10, 0)
        self.morning_prime_start = time(10, 0)
        self.morning_prime_end = time(11, 0)
        self.reversal_block_until = time(10, 15)
        self.midday_start = time(11, 0)
        self.midday_end = time(14, 0)

        # Minimum scores
        self.min_pullback_score = config.get("min_pullback_score", 35)
        self.min_confidence = config.get("min_confidence", 0.55)

        # Tracking
        self._last_signal_time: Optional[datetime] = None
        self._consecutive_holds = 0

        logger.info(
            f"BuyContinuationModule initialized: "
            f"RSI=[{self.rsi_min}-{self.rsi_max}], "
            f"MACD_threshold={self.macd_acceptance_threshold}, "
            f"ADX_trend={self.adx_trend_threshold}, "
            f"R:R_min={self.min_rr_ratio}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_session_window(self, timestamp: Optional[datetime]) -> SessionWindow:
        """Determine current session window for strategy selection."""
        if timestamp is None:
            return SessionWindow.MORNING_PRIME
        try:
            if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is not None:
                local_ts = timestamp
            else:
                local_ts = timestamp
            t = local_ts.time() if hasattr(local_ts, "time") else time(10, 0)
        except Exception:
            t = time(10, 0)

        if t < self.morning_open_start:
            return SessionWindow.PRE_MARKET
        elif t < self.morning_open_end:
            return SessionWindow.MORNING_OPEN
        elif t < self.morning_prime_end:
            return SessionWindow.MORNING_PRIME
        elif t < self.midday_end:
            return SessionWindow.MIDDAY
        elif t < time(15, 0):
            return SessionWindow.AFTERNOON
        elif t < time(16, 0):
            return SessionWindow.CLOSE
        else:
            return SessionWindow.OVERNIGHT

    def _is_reversal_blocked(
        self, timestamp: Optional[datetime], market_state: MarketStateResult
    ) -> Tuple[bool, str]:
        if timestamp is None:
            return False, ""
        try:
            t = timestamp.time() if hasattr(timestamp, "time") else time(10, 30)
        except Exception:
            return False, ""
        if t < self.reversal_block_until:
            if market_state.trend in [TrendDirection.STRONG_TREND_UP, TrendDirection.TREND_UP]:
                return True, f"REVERSAL_BLOCKED: Before {self.reversal_block_until} with bullish trend"
            if market_state.is_acceptance and not market_state.is_exhaustion:
                return True, f"REVERSAL_BLOCKED: Before {self.reversal_block_until} in acceptance"
        return False, ""

    # ------------------------------------------------------------------
    # Pullback analysis
    # ------------------------------------------------------------------

    def _analyze_pullback(
        self,
        price: float,
        open_price: float,
        high: float,
        low: float,
        ema9: float,
        ema21: float,
        ema50: float,
        vwap: float,
        atr: float,
        recent_bars: Optional[pd.DataFrame],
    ) -> PullbackAnalysis:
        """Return a detailed ``PullbackAnalysis`` for the current bar."""
        result = PullbackAnalysis()
        if price <= 0 or atr <= 0:
            return result

        ema9_dist_pct = abs(price - ema9) / price * 100 if ema9 > 0 else 999
        ema21_dist_pct = abs(price - ema21) / price * 100 if ema21 > 0 else 999

        # EMA9 touch / proximity
        ema9_touch = low <= ema9 <= high if ema9 > 0 else False
        ema9_near = ema9_dist_pct < self.ema_proximity_pct and price >= ema9
        if ema9_touch:
            result.score += 35
            result.touch_level = "EMA9"
            result.reasons.append("EMA9_BOUNCE")
        elif ema9_near:
            result.score += 25
            if not result.touch_level:
                result.touch_level = "EMA9"
            result.reasons.append("EMA9_SUPPORT")

        # EMA21 touch / proximity
        ema21_touch = low <= ema21 <= high if ema21 > 0 else False
        ema21_near = ema21_dist_pct < self.ema_proximity_pct * 1.5 and price >= ema21
        if ema21_touch:
            result.score += 30
            if not result.touch_level:
                result.touch_level = "EMA21"
            result.reasons.append("EMA21_BOUNCE")
        elif ema21_near:
            result.score += 20
            if not result.touch_level:
                result.touch_level = "EMA21"
            result.reasons.append("EMA21_SUPPORT")

        # VWAP touch / reclaim
        vwap_touch = low <= vwap <= high if vwap > 0 else False
        vwap_near = (abs(price - vwap) / price * 100 < self.ema_proximity_pct) and price >= vwap if vwap > 0 else False
        vwap_reclaim = (low < vwap < price) if vwap > 0 else False
        if vwap_reclaim:
            result.score += 35
            if not result.touch_level:
                result.touch_level = "VWAP"
            result.reasons.append("VWAP_RECLAIM")
        elif vwap_touch:
            result.score += 30
            if not result.touch_level:
                result.touch_level = "VWAP"
            result.reasons.append("VWAP_BOUNCE")
        elif vwap_near:
            result.score += 20
            if not result.touch_level:
                result.touch_level = "VWAP"
            result.reasons.append("VWAP_SUPPORT")

        # Candle confirmation
        candle_range = high - low if high > low else 0.01
        candle_body = abs(price - open_price)
        body_ratio = candle_body / candle_range
        is_bullish = price > open_price

        if recent_bars is not None and len(recent_bars) >= 2:
            prev_bar = recent_bars.iloc[-2]
            prev_open = float(prev_bar.get("open", 0))
            prev_close = float(prev_bar.get("close", 0))
            prev_was_bearish = prev_close < prev_open
            if is_bullish and prev_was_bearish:
                if price > prev_open and open_price < prev_close:
                    result.score += 20
                    result.confirmation = "BULLISH_ENGULF"
                    result.reasons.append("ENGULFING")

        if is_bullish and body_ratio > self.min_candle_body_ratio:
            if result.confirmation != "BULLISH_ENGULF":
                result.confirmation = "STRONG_CLOSE"
            result.score += 15
            result.reasons.append("BULLISH_CLOSE")
        elif is_bullish:
            result.score += 5
            result.reasons.append("WEAK_BULLISH")

        # Higher low
        if recent_bars is not None and len(recent_bars) >= 3:
            recent_lows = recent_bars["low"].tail(3).values
            if len(recent_lows) >= 3:
                if low > recent_lows[-2]:
                    result.score += 10
                    if not result.confirmation:
                        result.confirmation = "HIGHER_LOW"
                    result.reasons.append("HIGHER_LOW")
                if recent_lows[-2] > recent_lows[-3] and low > recent_lows[-2]:
                    result.score += 5
                    result.reasons.append("HL_SEQUENCE")

        # Pullback depth
        if recent_bars is not None and len(recent_bars) >= 5:
            recent_high = recent_bars["high"].tail(5).max()
            if recent_high > low:
                result.depth_pct = (recent_high - low) / recent_high * 100
                if self.pullback_depth_min_pct <= result.depth_pct <= self.pullback_depth_max_pct:
                    result.score += 10
                    result.reasons.append(f"IDEAL_DEPTH({result.depth_pct:.2f}%)")
                elif result.depth_pct > self.pullback_depth_max_pct:
                    result.score -= 10
                    result.reasons.append(f"DEEP_PULLBACK({result.depth_pct:.2f}%)")

        result.is_valid = (
            result.score >= self.min_pullback_score
            and result.touch_level != ""
            and len(result.reasons) >= 2
        )
        return result

    # ------------------------------------------------------------------
    # Stops & targets
    # ------------------------------------------------------------------

    def _calculate_stops_and_targets(
        self,
        price: float,
        ema21: float,
        atr: float,
        pdh: float,
        recent_bars: Optional[pd.DataFrame],
        pullback_analysis: PullbackAnalysis,
    ) -> Tuple[float, float, bool, str]:
        """Return ``(stop_loss, take_profit, is_valid, reason)``."""
        stop_candidates = []
        if recent_bars is not None and len(recent_bars) >= 3:
            pullback_low = recent_bars["low"].tail(3).min()
            stop_candidates.append(("PULLBACK_LOW", pullback_low - atr * 0.3))
        if ema21 > 0:
            stop_candidates.append(("EMA21", ema21 - atr * 0.3))
        stop_candidates.append(("ATR", price - atr * self.stop_atr_mult))

        if stop_candidates:
            stop_candidates.sort(key=lambda x: x[1], reverse=True)
            stop_type, stop_loss = stop_candidates[0]
        else:
            stop_loss = price - atr * self.stop_atr_mult
            stop_type = "DEFAULT"

        stop_distance = price - stop_loss
        if stop_distance < self.min_stop_points:
            stop_loss = price - self.min_stop_points
            stop_distance = self.min_stop_points
        if stop_distance > self.max_stop_points:
            stop_loss = price - self.max_stop_points
            stop_distance = self.max_stop_points

        risk = stop_distance
        min_reward = risk * self.min_rr_ratio

        target_candidates = []
        if pdh > 0 and pdh > price:
            pdh_reward = pdh - price
            if pdh_reward >= min_reward:
                target_candidates.append(("PDH", pdh - atr * 0.25))
        measured_target = price + (risk * self.target_risk_mult)
        target_candidates.append(("MEASURED", measured_target))
        if pullback_analysis.score >= 50:
            extended_target = price + (risk * 3.0)
            target_candidates.append(("EXTENDED", extended_target))

        if target_candidates:
            pdh_targets = [t for t in target_candidates if t[0] == "PDH"]
            if pdh_targets:
                target_type, take_profit = pdh_targets[0]
            else:
                target_type, take_profit = target_candidates[0]
        else:
            take_profit = price + min_reward
            target_type = "MIN_RR"

        actual_reward = take_profit - price
        actual_rr = actual_reward / risk if risk > 0 else 0
        if actual_rr < self.min_rr_ratio:
            return stop_loss, take_profit, False, f"INSUFFICIENT_RR: {actual_rr:.2f} < {self.min_rr_ratio}"
        return stop_loss, take_profit, True, f"{stop_type}_STOP, {target_type}_TARGET, RR={actual_rr:.2f}"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        market_state: MarketStateResult,
        data: Dict[str, Any],
        recent_bars: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None,
    ) -> EntrySignal:
        """Evaluate for BUY continuation entry.

        Returns:
            EntrySignal with action ``BUY`` or ``HOLD``.
        """
        session = self._get_session_window(timestamp)
        session_str = session.value

        metadata: Dict[str, Any] = {
            "session_window": session_str,
            "is_acceptance": market_state.is_acceptance,
            "is_exhaustion": market_state.is_exhaustion,
            "trend_score": market_state.trend_score,
            "continuation_score": market_state.continuation_score,
            "exhaustion_score": market_state.exhaustion_score,
            "allow_buy": market_state.allow_buy,
            "allow_short": market_state.allow_short,
            "phase": market_state.phase.value,
            "trend": market_state.trend.value,
        }

        if not market_state.allow_buy:
            return EntrySignal(action="HOLD", confidence=0.0, reason=f"BUY_BLOCKED: {market_state.phase.value}", entry_type="BLOCKED", session_window=session_str, metadata=metadata)

        if market_state.is_exhaustion:
            return EntrySignal(action="HOLD", confidence=0.0, reason="EXHAUSTION_PRESENT: Momentum loss detected, wait for resolution", entry_type="CAUTION", session_window=session_str, metadata=metadata)

        if not market_state.is_acceptance:
            if market_state.trend not in [TrendDirection.STRONG_TREND_UP, TrendDirection.TREND_UP, TrendDirection.WEAK_UP]:
                return EntrySignal(action="HOLD", confidence=0.0, reason=f"NOT_ACCEPTANCE: phase={market_state.phase.value}, trend={market_state.trend.value}", entry_type="WAIT", session_window=session_str, metadata=metadata)

        # Extract price data
        price = float(data.get("close", data.get("price", 0)))
        open_price = float(data.get("open", price))
        high = float(data.get("high", price))
        low = float(data.get("low", price))
        ema9 = float(data.get("ema_9", data.get("EMA_9", price)))
        ema21 = float(data.get("ema_21", data.get("EMA_21", price)))
        ema50 = float(data.get("ema_50", data.get("EMA_50", ema21)))
        vwap = float(data.get("vwap", data.get("SESSION_VWAP", price)))
        atr = float(data.get("atr", data.get("ATR_14", 1.0)))
        adx = float(data.get("adx", data.get("ADX_14", 20)))
        macd_hist = float(data.get("macd_hist", data.get("MACD_hist", 0)))
        pdh = float(data.get("pdh", data.get("PDH", 0)))
        rsi = market_state.rsi_value

        metadata.update({"rsi": rsi, "adx": adx, "macd_hist": macd_hist, "price": price, "ema9": ema9, "ema21": ema21, "vwap": vwap, "atr": atr})

        # Higher-timeframe alignment
        if self.require_htf_alignment:
            htf_ema21 = float(data.get("15m_ema21", data.get("15m_EMA_21", 0)))
            htf_ema50 = float(data.get("15m_ema50", data.get("15m_EMA_50", 0)))
            metadata["htf_ema21"] = htf_ema21
            metadata["htf_ema50"] = htf_ema50
            if htf_ema21 > 0 and htf_ema50 > 0 and htf_ema21 < htf_ema50:
                return EntrySignal(action="HOLD", confidence=0.0, reason="HTF_DOWNTREND: 15m EMA21 < EMA50", entry_type="BLOCKED", session_window=session_str, metadata=metadata)

        if rsi > self.rsi_overextension:
            return EntrySignal(action="HOLD", confidence=0.0, reason=f"RSI_OVEREXTENDED: {rsi:.1f} > {self.rsi_overextension} (chase risk)", entry_type="BLOCKED", session_window=session_str, metadata=metadata)

        if vwap > 0 and atr > 0:
            vwap_dist_atr = (price - vwap) / atr
            if vwap_dist_atr > self.max_vwap_distance_atr:
                return EntrySignal(action="HOLD", confidence=0.0, reason=f"VWAP_DISTANCE: {vwap_dist_atr:.1f} ATR > {self.max_vwap_distance_atr} (need pullback)", entry_type="WAIT", session_window=session_str, metadata=metadata)

        rsi_in_zone = self.rsi_min <= rsi <= self.rsi_max
        rsi_in_ideal = self.rsi_ideal_min <= rsi <= self.rsi_ideal_max

        if not rsi_in_zone and rsi < self.rsi_min - 5:
            return EntrySignal(action="HOLD", confidence=0.0, reason=f"RSI_TOO_LOW: {rsi:.1f} < {self.rsi_min - 5} (wait for momentum)", entry_type="WAIT", session_window=session_str, metadata=metadata)

        # Pullback analysis
        pullback = self._analyze_pullback(price, open_price, high, low, ema9, ema21, ema50, vwap, atr, recent_bars)
        metadata.update({"pullback_score": pullback.score, "pullback_level": pullback.touch_level, "pullback_confirmation": pullback.confirmation, "pullback_depth_pct": pullback.depth_pct, "pullback_reasons": pullback.reasons})

        if not pullback.is_valid:
            return EntrySignal(action="HOLD", confidence=0.0, reason=f"NO_VALID_PULLBACK: score={pullback.score:.0f}, level={pullback.touch_level or 'NONE'}", entry_type="WAIT", session_window=session_str, metadata=metadata)

        # Stops & targets
        stop_loss, take_profit, rr_valid, rr_reason = self._calculate_stops_and_targets(price, ema21, atr, pdh, recent_bars, pullback)
        metadata.update({"stop_loss": stop_loss, "take_profit": take_profit, "stop_reason": rr_reason})
        if not rr_valid:
            return EntrySignal(action="HOLD", confidence=0.0, reason=rr_reason, entry_type="BLOCKED", session_window=session_str, metadata=metadata)

        # Confidence scoring
        base_confidence = 0.55
        if session == SessionWindow.MORNING_PRIME:
            base_confidence += 0.10
        elif session == SessionWindow.MORNING_OPEN:
            base_confidence += 0.05
        elif session == SessionWindow.MIDDAY:
            base_confidence -= 0.10
        if market_state.is_acceptance:
            base_confidence += 0.10
        if rsi_in_ideal:
            base_confidence += 0.08
        elif rsi_in_zone:
            base_confidence += 0.04
        if adx >= self.adx_strong_threshold:
            base_confidence += 0.08
        elif adx >= self.adx_trend_threshold:
            base_confidence += 0.04
        if macd_hist > self.macd_acceptance_threshold:
            base_confidence += 0.06
        elif macd_hist > 0:
            base_confidence += 0.03
        pullback_bonus = (pullback.score - self.min_pullback_score) / 100
        base_confidence += min(0.15, pullback_bonus)
        if pullback.confirmation == "BULLISH_ENGULF":
            base_confidence += 0.05
        if market_state.continuation_score > 60:
            base_confidence += 0.05
        final_confidence = min(0.95, max(self.min_confidence, base_confidence))

        metadata["confidence_breakdown"] = {"base": 0.55, "session": session_str, "acceptance": market_state.is_acceptance, "rsi_in_ideal": rsi_in_ideal, "adx": adx, "macd_hist": macd_hist, "pullback_score": pullback.score, "final": final_confidence}

        reason_parts = ["BUY_CONTINUATION", f"{pullback.touch_level}_PULLBACK"]
        if pullback.confirmation:
            reason_parts.append(pullback.confirmation)
        if session == SessionWindow.MORNING_PRIME:
            reason_parts.append("MORNING_PRIME")
        if market_state.is_acceptance:
            reason_parts.append("ACCEPTANCE")

        return EntrySignal(action="BUY", confidence=final_confidence, reason=" | ".join(reason_parts), stop_loss=stop_loss, take_profit=take_profit, entry_type="CONTINUATION", session_window=session_str, metadata=metadata)
