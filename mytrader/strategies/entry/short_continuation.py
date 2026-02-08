"""SHORT continuation entry module — sells pullbacks in bearish acceptance.

Mirrors BuyContinuationModule for downtrend pullbacks.  Requires
``MarketState.is_acceptance == True`` in a bearish context, a valid
pullback into EMA resistance, and a minimum 2:1 risk/reward ratio.

Inputs:
    market_state, data, recent_bars, timestamp (same as BuyContinuation).

Outputs:
    EntrySignal: ``SELL`` with confidence & stop/target, or ``HOLD``.

Side-effects:
    None — pure analysis, no I/O.
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from ..market_state import MarketStateResult, TrendDirection
from .session_time import SessionWindow
from .signals import EntrySignal, PullbackAnalysis


class ShortContinuationModule:
    """SHORT continuation module - sells pullbacks in bearish acceptance.

    Mirrors BUY_CONTINUATION logic for downtrend pullbacks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # === RSI THRESHOLDS (bearish zone) ===
        self.rsi_min = config.get("rsi_min", 30)
        self.rsi_max = config.get("rsi_max", 45)
        self.rsi_overextension = config.get("rsi_overextension", 28)
        self.rsi_ideal_min = config.get("rsi_ideal_min", 35)
        self.rsi_ideal_max = config.get("rsi_ideal_max", 42)

        # === MACD THRESHOLDS (bearish) ===
        self.macd_acceptance_threshold = config.get("macd_acceptance_threshold", -0.20)
        self.macd_slope_min = config.get("macd_slope_min", 0.0)

        # === ADX THRESHOLD ===
        self.adx_trend_threshold = config.get("adx_trend_threshold", 25)
        self.adx_strong_threshold = config.get("adx_strong_threshold", 30)

        # === PULLBACK PARAMETERS ===
        self.pullback_depth_min_pct = config.get("pullback_depth_min_pct", 0.05)
        self.pullback_depth_max_pct = config.get("pullback_depth_max_pct", 0.50)
        self.max_vwap_distance_atr = config.get("max_vwap_distance_atr", 2.0)
        self.ema_proximity_pct = config.get("ema_proximity_pct", 0.15)

        # === CONFIRMATION REQUIREMENTS ===
        self.require_bearish_close = config.get("require_bearish_close", True)
        self.require_lower_high = config.get("require_lower_high", False)
        self.min_candle_body_ratio = config.get("min_candle_body_ratio", 0.4)
        self.engulfing_required = config.get("engulfing_required", False)

        # === RISK MANAGEMENT ===
        self.stop_atr_mult = config.get("stop_atr_mult", 1.5)
        self.target_risk_mult = config.get("target_risk_mult", 2.0)
        self.min_stop_points = config.get("min_stop_points", 3.25)
        self.max_stop_points = config.get("max_stop_points", 8.0)
        self.min_rr_ratio = config.get("min_rr_ratio", 2.0)

        # === HIGHER TIMEFRAME ALIGNMENT ===
        self.require_htf_alignment = config.get("require_htf_alignment", True)

        # === MINIMUM SCORES ===
        self.min_pullback_score = config.get("min_pullback_score", 35)
        self.min_confidence = config.get("min_confidence", 0.55)

        logger.info(
            f"ShortContinuationModule initialized: "
            f"RSI=[{self.rsi_min}-{self.rsi_max}], "
            f"MACD_threshold={self.macd_acceptance_threshold}, "
            f"ADX_trend={self.adx_trend_threshold}, "
            f"R:R_min={self.min_rr_ratio}"
        )

    def _get_session_window(self, timestamp: Optional[datetime]) -> SessionWindow:
        if timestamp is None:
            return SessionWindow.MORNING_PRIME

        try:
            if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                local_ts = timestamp
            else:
                local_ts = timestamp
            t = local_ts.time() if hasattr(local_ts, 'time') else time(10, 0)
        except Exception:
            t = time(10, 0)

        if t < time(9, 30):
            return SessionWindow.PRE_MARKET
        elif t < time(10, 0):
            return SessionWindow.MORNING_OPEN
        elif t < time(11, 0):
            return SessionWindow.MORNING_PRIME
        elif t < time(14, 0):
            return SessionWindow.MIDDAY
        elif t < time(15, 0):
            return SessionWindow.AFTERNOON
        elif t < time(16, 0):
            return SessionWindow.CLOSE
        else:
            return SessionWindow.OVERNIGHT

    def _analyze_pullback_short(
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
        recent_bars: Optional[pd.DataFrame]
    ) -> PullbackAnalysis:
        result = PullbackAnalysis()

        if price <= 0 or atr <= 0:
            return result

        ema9_dist_pct = abs(price - ema9) / price * 100 if ema9 > 0 else 999
        ema21_dist_pct = abs(price - ema21) / price * 100 if ema21 > 0 else 999
        vwap_dist_pct = abs(price - vwap) / price * 100 if vwap > 0 else 999
        vwap_dist_atr = abs(price - vwap) / atr if vwap > 0 else 999

        ema9_touch = low <= ema9 <= high if ema9 > 0 else False
        ema9_near = ema9_dist_pct < self.ema_proximity_pct and price <= ema9

        if ema9_touch:
            result.score += 35
            result.touch_level = "EMA9"
            result.reasons.append("EMA9_REJECT")
        elif ema9_near:
            result.score += 25
            if not result.touch_level:
                result.touch_level = "EMA9"
            result.reasons.append("EMA9_RESISTANCE")

        ema21_touch = low <= ema21 <= high if ema21 > 0 else False
        ema21_near = ema21_dist_pct < self.ema_proximity_pct * 1.5 and price <= ema21

        if ema21_touch:
            result.score += 30
            if not result.touch_level:
                result.touch_level = "EMA21"
            result.reasons.append("EMA21_REJECT")
        elif ema21_near:
            result.score += 20
            if not result.touch_level:
                result.touch_level = "EMA21"
            result.reasons.append("EMA21_RESISTANCE")

        vwap_touch = low <= vwap <= high if vwap > 0 else False
        vwap_near = vwap_dist_pct < self.ema_proximity_pct and price <= vwap
        vwap_reject = (high > vwap > price) if vwap > 0 else False

        if vwap_reject:
            result.score += 35
            if not result.touch_level:
                result.touch_level = "VWAP"
            result.reasons.append("VWAP_REJECT")
        elif vwap_touch:
            result.score += 30
            if not result.touch_level:
                result.touch_level = "VWAP"
            result.reasons.append("VWAP_REJECT")
        elif vwap_near:
            result.score += 20
            if not result.touch_level:
                result.touch_level = "VWAP"
            result.reasons.append("VWAP_RESISTANCE")

        candle_range = high - low if high > low else 0.01
        candle_body = abs(price - open_price)
        body_ratio = candle_body / candle_range
        is_bearish = price < open_price

        if recent_bars is not None and len(recent_bars) >= 2:
            prev_bar = recent_bars.iloc[-2]
            prev_open = float(prev_bar.get('open', 0))
            prev_close = float(prev_bar.get('close', 0))
            prev_was_bullish = prev_close > prev_open

            if is_bearish and prev_was_bullish:
                if open_price > prev_close and price < prev_open:
                    result.score += 20
                    result.confirmation = "BEARISH_ENGULF"
                    result.reasons.append("ENGULFING")

        if is_bearish and body_ratio > self.min_candle_body_ratio:
            if result.confirmation != "BEARISH_ENGULF":
                result.confirmation = "STRONG_CLOSE"
            result.score += 15
            result.reasons.append("BEARISH_CLOSE")
        elif is_bearish:
            result.score += 5
            result.reasons.append("WEAK_BEARISH")

        if recent_bars is not None and len(recent_bars) >= 3:
            recent_highs = recent_bars['high'].tail(3).values
            if len(recent_highs) >= 3:
                if high < recent_highs[-2]:
                    result.score += 10
                    if not result.confirmation:
                        result.confirmation = "LOWER_HIGH"
                    result.reasons.append("LOWER_HIGH")
                if recent_highs[-2] < recent_highs[-3] and high < recent_highs[-2]:
                    result.score += 5
                    result.reasons.append("LH_SEQUENCE")

        if recent_bars is not None and len(recent_bars) >= 5:
            recent_low = recent_bars['low'].tail(5).min()
            if recent_low > 0:
                result.depth_pct = (high - recent_low) / recent_low * 100

                if self.pullback_depth_min_pct <= result.depth_pct <= self.pullback_depth_max_pct:
                    result.score += 10
                    result.reasons.append(f"IDEAL_DEPTH({result.depth_pct:.2f}%)")
                elif result.depth_pct > self.pullback_depth_max_pct:
                    result.score -= 10
                    result.reasons.append(f"DEEP_PULLBACK({result.depth_pct:.2f}%)")

        result.is_valid = (
            result.score >= self.min_pullback_score and
            result.touch_level != "" and
            len(result.reasons) >= 2
        )

        return result

    def _calculate_stops_and_targets_short(
        self,
        price: float,
        ema21: float,
        atr: float,
        pdl: float,
        recent_bars: Optional[pd.DataFrame],
        pullback_analysis: PullbackAnalysis
    ) -> Tuple[float, float, bool, str]:
        stop_candidates = []

        if recent_bars is not None and len(recent_bars) >= 3:
            pullback_high = recent_bars['high'].tail(3).max()
            stop_candidates.append(("PULLBACK_HIGH", pullback_high + atr * 0.3))

        if ema21 > 0:
            stop_candidates.append(("EMA21", ema21 + atr * 0.3))

        stop_candidates.append(("ATR", price + atr * self.stop_atr_mult))

        if stop_candidates:
            stop_candidates.sort(key=lambda x: x[1])
            stop_type, stop_loss = stop_candidates[0]
        else:
            stop_loss = price + atr * self.stop_atr_mult
            stop_type = "DEFAULT"

        stop_distance = stop_loss - price

        if stop_distance < self.min_stop_points:
            stop_loss = price + self.min_stop_points
            stop_distance = self.min_stop_points

        if stop_distance > self.max_stop_points:
            stop_loss = price + self.max_stop_points
            stop_distance = self.max_stop_points

        risk = stop_distance
        min_reward = risk * self.min_rr_ratio

        target_candidates = []

        if pdl > 0 and pdl < price:
            pdl_reward = price - pdl
            if pdl_reward >= min_reward:
                target_candidates.append(("PDL", pdl + atr * 0.25))

        measured_target = price - (risk * self.target_risk_mult)
        target_candidates.append(("MEASURED", measured_target))

        if pullback_analysis.score >= 50:
            extended_target = price - (risk * 3.0)
            target_candidates.append(("EXTENDED", extended_target))

        if target_candidates:
            pdl_targets = [t for t in target_candidates if t[0] == "PDL"]
            if pdl_targets:
                target_type, take_profit = pdl_targets[0]
            else:
                target_type, take_profit = target_candidates[0]
        else:
            take_profit = price - min_reward
            target_type = "MIN_RR"

        actual_reward = price - take_profit
        actual_rr = actual_reward / risk if risk > 0 else 0

        if actual_rr < self.min_rr_ratio:
            return stop_loss, take_profit, False, f"INSUFFICIENT_RR: {actual_rr:.2f} < {self.min_rr_ratio}"

        return stop_loss, take_profit, True, f"{stop_type}_STOP, {target_type}_TARGET, RR={actual_rr:.2f}"

    def evaluate(
        self,
        market_state: MarketStateResult,
        data: Dict[str, Any],
        recent_bars: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None
    ) -> EntrySignal:
        session = self._get_session_window(timestamp)
        session_str = session.value

        metadata = {
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

        if not market_state.allow_short:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"SHORT_BLOCKED: {market_state.phase.value}",
                entry_type="BLOCKED",
                session_window=session_str,
                metadata=metadata
            )

        if market_state.is_exhaustion:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason="EXHAUSTION_PRESENT: Momentum loss detected, wait for resolution",
                entry_type="CAUTION",
                session_window=session_str,
                metadata=metadata
            )

        if not market_state.is_acceptance:
            if market_state.trend not in [TrendDirection.STRONG_TREND_DOWN, TrendDirection.TREND_DOWN, TrendDirection.WEAK_DOWN]:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"NOT_ACCEPTANCE: phase={market_state.phase.value}, trend={market_state.trend.value}",
                    entry_type="WAIT",
                    session_window=session_str,
                    metadata=metadata
                )

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
        pdl = float(data.get("pdl", data.get("PDL", 0)))

        rsi = market_state.rsi_value

        metadata.update({
            "rsi": rsi,
            "adx": adx,
            "macd_hist": macd_hist,
            "price": price,
            "ema9": ema9,
            "ema21": ema21,
            "vwap": vwap,
            "atr": atr,
        })

        if self.require_htf_alignment:
            htf_ema21 = float(data.get("15m_ema21", data.get("15m_EMA_21", 0)))
            htf_ema50 = float(data.get("15m_ema50", data.get("15m_EMA_50", 0)))
            metadata["htf_ema21"] = htf_ema21
            metadata["htf_ema50"] = htf_ema50

            if htf_ema21 > 0 and htf_ema50 > 0 and htf_ema21 > htf_ema50:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason="HTF_UPTREND: 15m EMA21 > EMA50",
                    entry_type="BLOCKED",
                    session_window=session_str,
                    metadata=metadata
                )

        if rsi < self.rsi_overextension:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"RSI_OVERSOLD: {rsi:.1f} < {self.rsi_overextension} (rebound risk)",
                entry_type="BLOCKED",
                session_window=session_str,
                metadata=metadata
            )

        if vwap > 0 and atr > 0:
            vwap_dist_atr = (vwap - price) / atr
            if vwap_dist_atr > self.max_vwap_distance_atr:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"VWAP_DISTANCE: {vwap_dist_atr:.1f} ATR > {self.max_vwap_distance_atr} (need pullback)",
                    entry_type="WAIT",
                    session_window=session_str,
                    metadata=metadata
                )

        rsi_in_zone = self.rsi_min <= rsi <= self.rsi_max
        rsi_in_ideal = self.rsi_ideal_min <= rsi <= self.rsi_ideal_max

        if not rsi_in_zone:
            if rsi > self.rsi_max + 5:
                return EntrySignal(
                    action="HOLD",
                    confidence=0.0,
                    reason=f"RSI_TOO_HIGH: {rsi:.1f} > {self.rsi_max + 5} (wait for momentum)",
                    entry_type="WAIT",
                    session_window=session_str,
                    metadata=metadata
                )

        pullback = self._analyze_pullback_short(
            price=price,
            open_price=open_price,
            high=high,
            low=low,
            ema9=ema9,
            ema21=ema21,
            ema50=ema50,
            vwap=vwap,
            atr=atr,
            recent_bars=recent_bars
        )

        metadata["pullback_score"] = pullback.score
        metadata["pullback_level"] = pullback.touch_level
        metadata["pullback_confirmation"] = pullback.confirmation
        metadata["pullback_depth_pct"] = pullback.depth_pct
        metadata["pullback_reasons"] = pullback.reasons

        if not pullback.is_valid:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=f"NO_VALID_PULLBACK: score={pullback.score:.0f}, level={pullback.touch_level or 'NONE'}",
                entry_type="WAIT",
                session_window=session_str,
                metadata=metadata
            )

        stop_loss, take_profit, rr_valid, rr_reason = self._calculate_stops_and_targets_short(
            price=price,
            ema21=ema21,
            atr=atr,
            pdl=pdl,
            recent_bars=recent_bars,
            pullback_analysis=pullback
        )

        metadata["stop_loss"] = stop_loss
        metadata["take_profit"] = take_profit
        metadata["stop_reason"] = rr_reason

        if not rr_valid:
            return EntrySignal(
                action="HOLD",
                confidence=0.0,
                reason=rr_reason,
                entry_type="BLOCKED",
                session_window=session_str,
                metadata=metadata
            )

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

        if macd_hist < self.macd_acceptance_threshold:
            base_confidence += 0.06
        elif macd_hist < 0:
            base_confidence += 0.03

        pullback_bonus = (pullback.score - self.min_pullback_score) / 100
        base_confidence += min(0.15, pullback_bonus)

        if pullback.confirmation == "BEARISH_ENGULF":
            base_confidence += 0.05

        if market_state.continuation_score > 60:
            base_confidence += 0.05

        final_confidence = min(0.95, max(self.min_confidence, base_confidence))

        metadata["confidence_breakdown"] = {
            "base": 0.55,
            "session": session_str,
            "acceptance": market_state.is_acceptance,
            "rsi_in_ideal": rsi_in_ideal,
            "adx": adx,
            "macd_hist": macd_hist,
            "pullback_score": pullback.score,
            "final": final_confidence
        }

        reason_parts = [
            "SHORT_CONTINUATION",
            f"{pullback.touch_level}_PULLBACK",
        ]
        if pullback.confirmation:
            reason_parts.append(pullback.confirmation)
        if session == SessionWindow.MORNING_PRIME:
            reason_parts.append("MORNING_PRIME")
        if market_state.is_acceptance:
            reason_parts.append("ACCEPTANCE")

        return EntrySignal(
            action="SELL",
            confidence=final_confidence,
            reason=" | ".join(reason_parts),
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_type="CONTINUATION",
            session_window=session_str,
            metadata=metadata
        )


