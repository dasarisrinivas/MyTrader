"""Trend Continuation Optimizer - Modify existing brackets instead of close+re-enter.

When in a profitable position with trend continuing:
- Instead of taking profit and immediately re-entering at similar/worse price
- Extend the take profit target and trail the stop loss to breakeven+

WHEN TO MODIFY (all/most should be true):
1. Higher timeframe trend is strong (price above rising MA, HH/HL structure)
2. Momentum hasn't rolled over (no divergence, no rejection wicks)
3. Room to run: next resistance > 0.5-1.0 ATR ahead
4. Volatility supports extension (ATR not collapsing)
5. Risk stays bounded: new stop doesn't increase worst-case loss

WHEN NOT TO MODIFY:
1. Trend weak/choppy (low ADX, mean reversion around VWAP)
2. Price hitting resistance/liquidity and getting rejected
3. Late session or approaching news events
4. Modification would widen stop or reduce expectancy

DECISION TRIGGER: When price is near TP (within 0.25 ATR or X ticks)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import TYPE_CHECKING, Optional, Dict, Any, Tuple, List

from ...utils.logger import logger
from ...utils.structured_logging import log_structured_event

if TYPE_CHECKING:
    from ..live_trading_manager import LiveTradingManager
    from ..ib_executor import PositionInfo


@dataclass
class ContinuationAnalysis:
    """Result of trend continuation analysis."""
    should_modify: bool
    reason: str
    current_pnl_points: float = 0.0
    current_rr_achieved: float = 0.0  # How much of original R:R captured
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None
    savings_estimate: float = 0.0  # Estimated commission/slippage savings
    trend_strength: float = 0.0
    risk_to_new_stop: float = 0.0
    
    # Detailed scoring
    scoring_details: Dict[str, Any] = field(default_factory=dict)
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass
class TrendScore:
    """Detailed trend analysis scoring."""
    ma_score: float = 0.0        # Price vs MA position (0-1)
    structure_score: float = 0.0  # HH/HL or LL/LH structure (0-1)
    momentum_score: float = 0.0   # RSI/MACD momentum (0-1)
    adx_score: float = 0.0        # Trend strength (0-1)
    vwap_score: float = 0.0       # Price vs VWAP (0-1)
    total: float = 0.0            # Weighted total (0-1)
    
    def calculate_total(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted total score."""
        if weights is None:
            weights = {
                "ma": 0.25,
                "structure": 0.20,
                "momentum": 0.25,
                "adx": 0.15,
                "vwap": 0.15,
            }
        self.total = (
            self.ma_score * weights["ma"] +
            self.structure_score * weights["structure"] +
            self.momentum_score * weights["momentum"] +
            self.adx_score * weights["adx"] +
            self.vwap_score * weights["vwap"]
        )
        return self.total


class TrendContinuationOptimizer:
    """Analyzes whether to modify existing bracket vs close+re-enter.
    
    The key insight: If we're about to hit TP and immediately re-enter,
    we're paying commissions and slippage for no net position change.
    Better to just extend the existing trade - IF conditions favor it.
    
    Key Decision Logic:
    1. TRIGGER: Price within decision_zone of TP (default: 0.25 ATR)
    2. SCORE: Evaluate trend, momentum, room-to-run, volatility, risk
    3. COMPARE: Expected value of extension vs taking profit
    4. EXECUTE: Only modify if EV(extend) > EV(take profit)
    """
    
    # Session times when we should NOT extend (EST)
    NO_EXTEND_TIMES = [
        (dt_time(9, 25), dt_time(9, 35)),   # Around open
        (dt_time(15, 45), dt_time(16, 0)),  # Near close
    ]
    
    def __init__(
        self,
        manager: "LiveTradingManager",
        # Decision trigger
        decision_zone_atr_mult: float = 0.25,  # Trigger when within 0.25 ATR of TP
        decision_zone_min_ticks: int = 4,       # Or within 4 ticks (1 point MES)
        # Trend requirements
        min_trend_score: float = 0.55,          # Minimum trend score to extend
        min_adx: float = 20.0,                  # Minimum ADX for extension
        max_rsi_long: float = 72.0,             # Don't extend longs above this RSI
        min_rsi_short: float = 28.0,            # Don't extend shorts below this RSI
        # Room to run
        min_room_atr_mult: float = 0.5,         # Need at least 0.5 ATR room to next resistance
        # Extension parameters
        tp_extension_atr_mult: float = 1.0,     # Extend TP by 1.0 ATR (conservative)
        max_extension_points: float = 15.0,     # Cap TP extension
        # Stop management
        breakeven_buffer_points: float = 1.0,   # Minimum profit to lock in
        sl_trail_atr_mult: float = 1.5,         # Trail stop by 1.5 ATR below price
        # Safety
        max_modifications_per_trade: int = 2,   # Don't extend more than twice
        enabled: bool = True,
    ):
        self.manager = manager
        
        # Decision trigger
        self.decision_zone_atr_mult = decision_zone_atr_mult
        self.decision_zone_min_ticks = decision_zone_min_ticks
        
        # Trend requirements
        self.min_trend_score = min_trend_score
        self.min_adx = min_adx
        self.max_rsi_long = max_rsi_long
        self.min_rsi_short = min_rsi_short
        
        # Room to run
        self.min_room_atr_mult = min_room_atr_mult
        
        # Extension parameters  
        self.tp_extension_atr_mult = tp_extension_atr_mult
        self.max_extension_points = max_extension_points
        
        # Stop management
        self.breakeven_buffer_points = breakeven_buffer_points
        self.sl_trail_atr_mult = sl_trail_atr_mult
        
        # Safety
        self.max_modifications_per_trade = max_modifications_per_trade
        self.enabled = enabled
        
        # Track modifications for logging
        self._modification_count = 0
        self._trade_modification_count: Dict[int, int] = {}  # order_id -> count
        self._last_modification_time: Optional[datetime] = None
    
    async def analyze_continuation(
        self,
        new_signal_action: str,
        current_price: float,
        new_stop_loss: float,
        new_take_profit: float,
        features: Dict[str, Any],
    ) -> ContinuationAnalysis:
        """Analyze whether to modify existing position instead of close+re-enter.
        
        Decision Process:
        1. Check prerequisites (position exists, direction matches, etc.)
        2. Check if in decision zone (near TP)
        3. Calculate comprehensive trend score
        4. Verify room to run (distance to resistance)
        5. Check volatility isn't collapsing
        6. Calculate new levels ensuring risk doesn't increase
        7. Compare EV of extension vs take profit
        
        Args:
            new_signal_action: The new signal (BUY, SELL, SCALP_BUY, SCALP_SELL)
            current_price: Current market price
            new_stop_loss: Stop loss for the proposed new trade
            new_take_profit: Take profit for the proposed new trade
            features: Current features DataFrame row
            
        Returns:
            ContinuationAnalysis with recommendation
        """
        rejection_reasons = []
        
        if not self.enabled:
            return ContinuationAnalysis(
                should_modify=False,
                reason="Trend continuation optimizer disabled"
            )
        
        # Check session timing
        session_ok, session_reason = self._check_session_timing()
        if not session_ok:
            return ContinuationAnalysis(
                should_modify=False,
                reason=session_reason,
                rejection_reasons=[session_reason]
            )
        
        # Get current position
        position = await self._get_current_position()
        if not position or position.quantity == 0:
            return ContinuationAnalysis(
                should_modify=False,
                reason="No existing position"
            )
        
        # Check direction match
        is_long = position.quantity > 0
        is_new_signal_long = new_signal_action.upper() in ("BUY", "SCALP_BUY")
        
        if is_long != is_new_signal_long:
            return ContinuationAnalysis(
                should_modify=False,
                reason="Signal direction doesn't match existing position"
            )
        
        # Check modification count for this trade
        trade_id = getattr(position, 'order_id', id(position))
        current_mods = self._trade_modification_count.get(trade_id, 0)
        if current_mods >= self.max_modifications_per_trade:
            return ContinuationAnalysis(
                should_modify=False,
                reason=f"Max modifications reached ({current_mods}/{self.max_modifications_per_trade})",
                rejection_reasons=[f"Already modified {current_mods} times"]
            )
        
        # Get key values
        entry_price = position.avg_cost
        direction = 1 if is_long else -1
        current_pnl_points = (current_price - entry_price) * direction
        
        original_stop = getattr(position, 'stop_loss', None)
        original_target = getattr(position, 'take_profit', None)
        
        if original_stop is None or original_target is None:
            return ContinuationAnalysis(
                should_modify=False,
                reason="Original stop/target not available",
                current_pnl_points=current_pnl_points
            )
        
        # Get ATR
        atr = float(features.get("ATR_14", features.get("atr", 0.0)))
        if atr <= 0:
            atr = 5.0  # Default for MES
        
        # ========================================
        # DECISION ZONE CHECK: Are we near TP?
        # ========================================
        distance_to_tp = abs(original_target - current_price)
        decision_zone_threshold = max(
            atr * self.decision_zone_atr_mult,
            self.decision_zone_min_ticks * 0.25  # Convert ticks to points
        )
        
        if distance_to_tp > decision_zone_threshold:
            return ContinuationAnalysis(
                should_modify=False,
                reason=f"Not in decision zone: {distance_to_tp:.2f} pts from TP (threshold: {decision_zone_threshold:.2f})",
                current_pnl_points=current_pnl_points
            )
        
        # Calculate R:R captured so far
        original_risk = abs(entry_price - original_stop)
        original_reward = abs(original_target - entry_price)
        
        if original_risk <= 0 or original_reward <= 0:
            return ContinuationAnalysis(
                should_modify=False,
                reason="Invalid original risk/reward",
                current_pnl_points=current_pnl_points
            )
        
        rr_captured = current_pnl_points / original_reward
        
        # ========================================
        # TREND ANALYSIS: Is trend still strong?
        # ========================================
        trend_score = self._calculate_trend_score(features, is_long)
        
        if trend_score.total < self.min_trend_score:
            rejection_reasons.append(f"Trend score too low: {trend_score.total:.2f} < {self.min_trend_score}")
            return ContinuationAnalysis(
                should_modify=False,
                reason=f"Trend not strong enough: score={trend_score.total:.2f}",
                current_pnl_points=current_pnl_points,
                current_rr_achieved=rr_captured,
                trend_strength=trend_score.total,
                scoring_details={"trend": trend_score.__dict__},
                rejection_reasons=rejection_reasons
            )
        
        # ========================================
        # MOMENTUM CHECK: Has momentum rolled over?
        # ========================================
        momentum_ok, momentum_reason = self._check_momentum_health(features, is_long)
        if not momentum_ok:
            rejection_reasons.append(f"Momentum: {momentum_reason}")
            return ContinuationAnalysis(
                should_modify=False,
                reason=f"Momentum unfavorable: {momentum_reason}",
                current_pnl_points=current_pnl_points,
                current_rr_achieved=rr_captured,
                trend_strength=trend_score.total,
                rejection_reasons=rejection_reasons
            )
        
        # ========================================
        # ROOM TO RUN: Is there space to next resistance?
        # ========================================
        room_to_resistance = self._estimate_room_to_resistance(
            features, current_price, is_long, atr
        )
        min_room_required = atr * self.min_room_atr_mult
        
        if room_to_resistance < min_room_required:
            rejection_reasons.append(f"Insufficient room: {room_to_resistance:.2f} < {min_room_required:.2f} pts")
            return ContinuationAnalysis(
                should_modify=False,
                reason=f"Not enough room to resistance: {room_to_resistance:.2f} pts",
                current_pnl_points=current_pnl_points,
                current_rr_achieved=rr_captured,
                trend_strength=trend_score.total,
                rejection_reasons=rejection_reasons
            )
        
        # ========================================
        # VOLATILITY CHECK: Is ATR stable/expanding?
        # ========================================
        vol_ok, vol_reason = self._check_volatility_health(features)
        if not vol_ok:
            rejection_reasons.append(f"Volatility: {vol_reason}")
            return ContinuationAnalysis(
                should_modify=False,
                reason=f"Volatility unfavorable: {vol_reason}",
                current_pnl_points=current_pnl_points,
                current_rr_achieved=rr_captured,
                trend_strength=trend_score.total,
                rejection_reasons=rejection_reasons
            )
        
        # ========================================
        # CALCULATE NEW LEVELS
        # ========================================
        new_stop, new_target = self._calculate_new_levels(
            entry_price=entry_price,
            current_price=current_price,
            original_stop=original_stop,
            original_target=original_target,
            is_long=is_long,
            atr=atr,
            features=features
        )
        
        # ========================================
        # RISK CHECK: Does new stop increase risk?
        # ========================================
        original_max_loss = abs(entry_price - original_stop)
        new_max_loss = abs(entry_price - new_stop)
        
        if new_max_loss > original_max_loss:
            rejection_reasons.append(f"Would increase risk: {new_max_loss:.2f} > {original_max_loss:.2f}")
            return ContinuationAnalysis(
                should_modify=False,
                reason="Modification would increase risk",
                current_pnl_points=current_pnl_points,
                current_rr_achieved=rr_captured,
                trend_strength=trend_score.total,
                rejection_reasons=rejection_reasons
            )
        
        # ========================================
        # EXPECTED VALUE COMPARISON
        # ========================================
        # EV(take profit now) = current_pnl - commissions
        # EV(extend) = P(win) * extended_reward - P(lose) * risk_to_new_stop - commissions_saved
        
        commission_per_round_trip = 2.50  # $2.50 per round trip
        slippage_estimate = 0.25 * 5  # 0.25 points * $5/point
        
        # Taking profit now:
        ev_take_profit = (current_pnl_points * 5) - commission_per_round_trip
        
        # Extending (probabilistic):
        risk_to_new_stop = abs(current_price - new_stop)
        potential_reward = abs(new_target - current_price)
        
        # Use trend score as proxy for P(win)
        p_win = min(0.7, 0.4 + trend_score.total * 0.4)  # Cap at 70%
        p_lose = 1 - p_win
        
        # Commission savings from not closing+reopening
        commission_savings = commission_per_round_trip * 2 + slippage_estimate * 2
        
        ev_extend = (
            p_win * potential_reward * 5 -
            p_lose * risk_to_new_stop * 5 +
            commission_savings
        )
        
        if ev_extend <= ev_take_profit:
            rejection_reasons.append(f"EV unfavorable: extend=${ev_extend:.2f} vs take=${ev_take_profit:.2f}")
            return ContinuationAnalysis(
                should_modify=False,
                reason=f"Extension EV ({ev_extend:.2f}) <= Take profit EV ({ev_take_profit:.2f})",
                current_pnl_points=current_pnl_points,
                current_rr_achieved=rr_captured,
                trend_strength=trend_score.total,
                scoring_details={
                    "ev_extend": ev_extend,
                    "ev_take_profit": ev_take_profit,
                    "p_win": p_win
                },
                rejection_reasons=rejection_reasons
            )
        
        # ========================================
        # APPROVAL: All checks passed
        # ========================================
        logger.info(
            "📈 TREND CONTINUATION APPROVED:\n"
            "   Position: {} @ {:.2f}, Current: {:.2f}\n"
            "   P&L: {:.2f} pts ({:.1%} of target)\n"
            "   Trend Score: {:.2f} (MA={:.2f}, Mom={:.2f}, ADX={:.2f})\n"
            "   Room to resistance: {:.2f} pts\n"
            "   EV: extend=${:.2f} vs take=${:.2f} (P(win)={:.1%})\n"
            "   New Stop: {:.2f} → {:.2f} (risk: {:.2f} → {:.2f})\n"
            "   New Target: {:.2f} → {:.2f} (+{:.2f} pts)\n"
            "   Commission savings: ${:.2f}",
            "LONG" if is_long else "SHORT",
            entry_price,
            current_price,
            current_pnl_points,
            rr_captured,
            trend_score.total,
            trend_score.ma_score,
            trend_score.momentum_score,
            trend_score.adx_score,
            room_to_resistance,
            ev_extend,
            ev_take_profit,
            p_win,
            original_stop,
            new_stop,
            original_max_loss,
            new_max_loss,
            original_target,
            new_target,
            abs(new_target - original_target),
            commission_savings
        )
        
        log_structured_event(
            agent="trend_continuation",
            event_type="optimization.approved",
            message=f"Modify bracket - trend continuation",
            payload={
                "position_direction": "LONG" if is_long else "SHORT",
                "entry_price": entry_price,
                "current_price": current_price,
                "pnl_points": current_pnl_points,
                "rr_captured": rr_captured,
                "trend_score": trend_score.total,
                "room_to_resistance": room_to_resistance,
                "ev_extend": ev_extend,
                "ev_take_profit": ev_take_profit,
                "p_win": p_win,
                "old_stop": original_stop,
                "new_stop": new_stop,
                "old_target": original_target,
                "new_target": new_target,
                "commission_savings": commission_savings,
            }
        )
        
        return ContinuationAnalysis(
            should_modify=True,
            reason="Trend continuation - EV favorable for extension",
            current_pnl_points=current_pnl_points,
            current_rr_achieved=rr_captured,
            new_stop_loss=new_stop,
            new_take_profit=new_target,
            savings_estimate=commission_savings,
            trend_strength=trend_score.total,
            risk_to_new_stop=risk_to_new_stop,
            scoring_details={
                "trend": trend_score.__dict__,
                "ev_extend": ev_extend,
                "ev_take_profit": ev_take_profit,
                "p_win": p_win,
                "room_to_resistance": room_to_resistance,
            }
        )
    
    def _check_session_timing(self) -> Tuple[bool, str]:
        """Check if current time is appropriate for extensions."""
        now = datetime.now()
        current_time = now.time()
        
        for start, end in self.NO_EXTEND_TIMES:
            if start <= current_time <= end:
                return False, f"No extensions during {start}-{end} (volatile period)"
        
        return True, "Session timing OK"
    
    def _calculate_trend_score(
        self,
        features: Dict[str, Any],
        is_long: bool
    ) -> TrendScore:
        """Calculate comprehensive trend score (0-1).
        
        Components:
        - MA Score: Price position relative to moving averages
        - Structure Score: Higher highs/lows (or lower for shorts)
        - Momentum Score: RSI, MACD alignment
        - ADX Score: Trend strength indicator
        - VWAP Score: Price relative to VWAP
        """
        score = TrendScore()
        
        # Get indicators
        close = float(features.get("close", features.get("Close", 0)))
        sma_20 = float(features.get("SMA_20", features.get("sma_20", close)))
        ema_9 = float(features.get("EMA_9", features.get("ema_9", close)))
        rsi = float(features.get("RSI_14", features.get("rsi", 50)))
        macd = float(features.get("MACD", features.get("macd", 0)))
        macd_signal = float(features.get("MACD_Signal", features.get("macd_signal", 0)))
        adx = float(features.get("ADX_14", features.get("adx", 20)))
        vwap = float(features.get("VWAP", features.get("vwap", close)))
        
        # Higher/lower structure (if available)
        high = float(features.get("high", features.get("High", close)))
        low = float(features.get("low", features.get("Low", close)))
        prev_high = float(features.get("prev_high", high))
        prev_low = float(features.get("prev_low", low))
        
        # 1. MA Score: Price above/below MAs and MA alignment
        if is_long:
            above_sma = 1.0 if close > sma_20 else 0.0
            above_ema = 1.0 if close > ema_9 else 0.0
            ema_above_sma = 1.0 if ema_9 > sma_20 else 0.0
            score.ma_score = (above_sma * 0.4 + above_ema * 0.3 + ema_above_sma * 0.3)
        else:
            below_sma = 1.0 if close < sma_20 else 0.0
            below_ema = 1.0 if close < ema_9 else 0.0
            ema_below_sma = 1.0 if ema_9 < sma_20 else 0.0
            score.ma_score = (below_sma * 0.4 + below_ema * 0.3 + ema_below_sma * 0.3)
        
        # 2. Structure Score: HH/HL for longs, LL/LH for shorts
        if is_long:
            hh = 1.0 if high > prev_high else 0.0
            hl = 1.0 if low > prev_low else 0.0
            score.structure_score = (hh * 0.5 + hl * 0.5)
        else:
            ll = 1.0 if low < prev_low else 0.0
            lh = 1.0 if high < prev_high else 0.0
            score.structure_score = (ll * 0.5 + lh * 0.5)
        
        # 3. Momentum Score: RSI and MACD
        if is_long:
            # RSI 50-70 is optimal for long continuation
            if 50 <= rsi <= 70:
                rsi_score = 1.0
            elif 45 <= rsi < 50:
                rsi_score = 0.6
            elif 70 < rsi <= self.max_rsi_long:
                rsi_score = 0.4
            elif rsi > self.max_rsi_long:
                rsi_score = 0.0  # Overbought, don't extend
            else:
                rsi_score = 0.3
            
            macd_score = 1.0 if macd > macd_signal else 0.3
        else:
            # RSI 30-50 is optimal for short continuation
            if 30 <= rsi <= 50:
                rsi_score = 1.0
            elif 50 < rsi <= 55:
                rsi_score = 0.6
            elif self.min_rsi_short <= rsi < 30:
                rsi_score = 0.4
            elif rsi < self.min_rsi_short:
                rsi_score = 0.0  # Oversold, don't extend
            else:
                rsi_score = 0.3
            
            macd_score = 1.0 if macd < macd_signal else 0.3
        
        score.momentum_score = (rsi_score * 0.5 + macd_score * 0.5)
        
        # 4. ADX Score: Trend strength
        if adx >= 30:
            score.adx_score = 1.0  # Strong trend
        elif adx >= 25:
            score.adx_score = 0.8
        elif adx >= self.min_adx:
            score.adx_score = 0.6
        elif adx >= 15:
            score.adx_score = 0.3
        else:
            score.adx_score = 0.0  # No trend
        
        # 5. VWAP Score: Price relative to VWAP
        if is_long:
            if close > vwap:
                score.vwap_score = 1.0
            elif close > vwap * 0.998:  # Within 0.2%
                score.vwap_score = 0.5
            else:
                score.vwap_score = 0.2
        else:
            if close < vwap:
                score.vwap_score = 1.0
            elif close < vwap * 1.002:
                score.vwap_score = 0.5
            else:
                score.vwap_score = 0.2
        
        score.calculate_total()
        return score
    
    def _check_momentum_health(
        self,
        features: Dict[str, Any],
        is_long: bool
    ) -> Tuple[bool, str]:
        """Check if momentum is healthy for continuation.
        
        Rejects if:
        - RSI divergence detected
        - MACD histogram shrinking significantly
        - Big rejection wick at resistance
        """
        rsi = float(features.get("RSI_14", features.get("rsi", 50)))
        macd_hist = float(features.get("MACD_Hist", features.get("macd_hist", 0)))
        prev_macd_hist = float(features.get("prev_macd_hist", macd_hist))
        
        high = float(features.get("high", features.get("High", 0)))
        low = float(features.get("low", features.get("Low", 0)))
        close = float(features.get("close", features.get("Close", 0)))
        
        # Check for overbought/oversold extremes
        if is_long and rsi > self.max_rsi_long:
            return False, f"RSI overbought ({rsi:.1f} > {self.max_rsi_long})"
        if not is_long and rsi < self.min_rsi_short:
            return False, f"RSI oversold ({rsi:.1f} < {self.min_rsi_short})"
        
        # Check MACD histogram - is momentum fading?
        if is_long:
            if macd_hist < 0 and prev_macd_hist > 0:
                return False, "MACD histogram crossed below zero"
            if prev_macd_hist > 0 and macd_hist > 0 and macd_hist < prev_macd_hist * 0.5:
                return False, "MACD momentum fading (histogram shrinking)"
        else:
            if macd_hist > 0 and prev_macd_hist < 0:
                return False, "MACD histogram crossed above zero"
            if prev_macd_hist < 0 and macd_hist < 0 and macd_hist > prev_macd_hist * 0.5:
                return False, "MACD momentum fading (histogram shrinking)"
        
        # Check for rejection wicks
        candle_range = high - low
        if candle_range > 0:
            if is_long:
                upper_wick = high - max(close, float(features.get("open", close)))
                wick_ratio = upper_wick / candle_range
                if wick_ratio > 0.6:  # 60%+ upper wick = rejection
                    return False, f"Rejection wick detected (upper wick {wick_ratio:.1%})"
            else:
                lower_wick = min(close, float(features.get("open", close))) - low
                wick_ratio = lower_wick / candle_range
                if wick_ratio > 0.6:
                    return False, f"Rejection wick detected (lower wick {wick_ratio:.1%})"
        
        return True, "Momentum healthy"
    
    def _estimate_room_to_resistance(
        self,
        features: Dict[str, Any],
        current_price: float,
        is_long: bool,
        atr: float
    ) -> float:
        """Estimate distance to next significant resistance/support.
        
        Uses available levels:
        - Prior day high/low
        - Overnight high/low
        - VWAP bands
        - Round numbers
        """
        resistance_levels = []
        support_levels = []
        
        # Prior day levels
        pdh = float(features.get("PDH", features.get("prior_day_high", 0)))
        pdl = float(features.get("PDL", features.get("prior_day_low", 0)))
        
        if pdh > 0:
            resistance_levels.append(pdh)
        if pdl > 0:
            support_levels.append(pdl)
        
        # Overnight levels
        onh = float(features.get("ONH", features.get("overnight_high", 0)))
        onl = float(features.get("ONL", features.get("overnight_low", 0)))
        
        if onh > 0:
            resistance_levels.append(onh)
        if onl > 0:
            support_levels.append(onl)
        
        # VWAP bands (if available)
        vwap_upper = float(features.get("VWAP_Upper", features.get("vwap_upper", 0)))
        vwap_lower = float(features.get("VWAP_Lower", features.get("vwap_lower", 0)))
        
        if vwap_upper > 0:
            resistance_levels.append(vwap_upper)
        if vwap_lower > 0:
            support_levels.append(vwap_lower)
        
        # Round number levels (every 25 points for ES/MES)
        round_25_above = math.ceil(current_price / 25) * 25
        round_25_below = math.floor(current_price / 25) * 25
        resistance_levels.append(round_25_above)
        support_levels.append(round_25_below)
        
        # Find nearest resistance (for longs) or support (for shorts)
        if is_long:
            # Find levels above current price
            levels_ahead = [l for l in resistance_levels if l > current_price + 0.5]
            if levels_ahead:
                nearest = min(levels_ahead)
                return nearest - current_price
            else:
                return atr * 2  # Default: assume room if no levels found
        else:
            # Find levels below current price
            levels_ahead = [l for l in support_levels if l < current_price - 0.5]
            if levels_ahead:
                nearest = max(levels_ahead)
                return current_price - nearest
            else:
                return atr * 2
    
    def _check_volatility_health(
        self,
        features: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check if volatility supports extension.
        
        Rejects if ATR is collapsing (entering chop zone).
        """
        atr = float(features.get("ATR_14", features.get("atr", 5)))
        prev_atr = float(features.get("prev_atr", atr))
        
        # Check if ATR is collapsing
        if prev_atr > 0 and atr < prev_atr * 0.7:
            return False, f"ATR collapsing ({atr:.2f} vs {prev_atr:.2f})"
        
        # Check minimum ATR (for MES, want at least 2 points of movement)
        if atr < 2.0:
            return False, f"ATR too low ({atr:.2f}), entering chop zone"
        
        return True, "Volatility healthy"
    
    def _calculate_new_levels(
        self,
        entry_price: float,
        current_price: float,
        original_stop: float,
        original_target: float,
        is_long: bool,
        atr: float,
        features: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate new stop and target levels.
        
        Stop Rules:
        - new_stop = max(current_stop, entry + breakeven_buffer, trailing_stop)
        - NEVER increase risk from original
        
        Target Rules:
        - Extend by tp_extension_atr_mult * ATR
        - Cap at max_extension_points
        """
        direction = 1 if is_long else -1
        
        # Calculate candidate stops
        # 1. Breakeven + buffer
        breakeven_stop = entry_price + (self.breakeven_buffer_points * direction)
        
        # 2. ATR trailing stop
        atr_trail_stop = current_price - (self.sl_trail_atr_mult * atr * direction)
        
        # 3. Swing low/high trail (if available)
        if is_long:
            swing_low = float(features.get("swing_low", features.get("low", current_price - atr)))
            swing_stop = swing_low - 0.5  # Buffer below swing
        else:
            swing_high = float(features.get("swing_high", features.get("high", current_price + atr)))
            swing_stop = swing_high + 0.5
        
        # Choose the most favorable (highest for long, lowest for short) that protects profit
        if is_long:
            # For longs: higher stop = more profit protection
            candidate_stops = [s for s in [original_stop, breakeven_stop, atr_trail_stop, swing_stop]
                            if s >= original_stop]  # Never move stop backward
            new_stop = max(candidate_stops) if candidate_stops else original_stop
        else:
            # For shorts: lower stop = more profit protection
            candidate_stops = [s for s in [original_stop, breakeven_stop, atr_trail_stop, swing_stop]
                            if s <= original_stop]
            new_stop = min(candidate_stops) if candidate_stops else original_stop
        
        # Calculate new target
        tp_extension = min(
            atr * self.tp_extension_atr_mult,
            self.max_extension_points
        )
        
        if is_long:
            new_target = original_target + tp_extension
        else:
            new_target = original_target - tp_extension
        
        # Round to tick size (0.25 for MES/ES)
        new_stop = round(new_stop * 4) / 4
        new_target = round(new_target * 4) / 4
        
        return new_stop, new_target
    
    async def _get_current_position(self) -> Optional["PositionInfo"]:
        """Get current position from executor."""
        if not self.manager.executor:
            return None
        return await self.manager.executor.get_current_position()
    
    async def execute_modification(
        self,
        analysis: ContinuationAnalysis
    ) -> bool:
        """Execute the bracket modification.
        
        Args:
            analysis: The ContinuationAnalysis with new levels
            
        Returns:
            True if modification was successful
        """
        if not analysis.should_modify:
            return False
        
        if analysis.new_stop_loss is None or analysis.new_take_profit is None:
            logger.error("Cannot execute modification: missing new levels")
            return False
        
        executor = self.manager.executor
        if not executor:
            logger.error("Cannot execute modification: no executor")
            return False
        
        try:
            # Get current position
            position = await self._get_current_position()
            if not position or position.quantity == 0:
                logger.warning("Position closed before modification could execute")
                return False
            
            is_long = position.quantity > 0
            quantity = abs(position.quantity)
            trade_id = getattr(position, 'order_id', id(position))
            
            # Cancel existing protective orders (stop and take profit)
            orders_to_cancel = []
            for order_id, trade in list(executor.active_orders.items()):
                order = trade.order
                order_type = getattr(order, "orderType", "")
                if order_type in ("STP", "STP LMT", "STOP"):
                    orders_to_cancel.append(order_id)
                elif order_type == "LMT":
                    order_action = getattr(order, "action", "")
                    if (is_long and order_action == "SELL") or (not is_long and order_action == "BUY"):
                        orders_to_cancel.append(order_id)
            
            cancelled_count = 0
            for order_id in orders_to_cancel:
                try:
                    await executor.cancel_order(order_id)
                    cancelled_count += 1
                except Exception as cancel_err:
                    logger.warning(f"Failed to cancel order {order_id}: {cancel_err}")
            
            logger.info(f"Cancelled {cancelled_count} existing protective orders")
            
            # Get qualified contract
            contract = await executor.get_qualified_contract()
            if not contract:
                logger.error("Cannot place new protective orders: no qualified contract")
                return False
            
            exit_action = "SELL" if is_long else "BUY"
            
            from ib_insync import StopOrder, LimitOrder
            
            # Place new stop order
            stop_order = StopOrder(exit_action, quantity, analysis.new_stop_loss)
            stop_order.transmit = True
            stop_order.outsideRth = True
            stop_trade = executor.ib.placeOrder(contract, stop_order)
            stop_id = getattr(stop_trade.order, "orderId", "NA")
            executor.active_orders[stop_id] = stop_trade
            
            logger.info(f"✅ New trailing stop {stop_id} at {analysis.new_stop_loss:.2f}")
            
            # Place new take profit order
            tp_order = LimitOrder(exit_action, quantity, analysis.new_take_profit)
            tp_order.transmit = True
            tp_order.outsideRth = True
            tp_trade = executor.ib.placeOrder(contract, tp_order)
            tp_id = getattr(tp_trade.order, "orderId", "NA")
            executor.active_orders[tp_id] = tp_trade
            
            logger.info(f"✅ New extended target {tp_id} at {analysis.new_take_profit:.2f}")
            
            # Update position tracking
            if hasattr(position, 'stop_loss'):
                position.stop_loss = analysis.new_stop_loss
            if hasattr(position, 'take_profit'):
                position.take_profit = analysis.new_take_profit
            
            # Track modification count for this trade
            self._trade_modification_count[trade_id] = self._trade_modification_count.get(trade_id, 0) + 1
            self._modification_count += 1
            self._last_modification_time = datetime.now()
            
            logger.info(
                "✅ Bracket modified successfully:\n"
                "   New Stop: {:.2f}\n"
                "   New Target: {:.2f}\n"
                "   Trade modifications: {}\n"
                "   Total modifications today: {}",
                analysis.new_stop_loss,
                analysis.new_take_profit,
                self._trade_modification_count[trade_id],
                self._modification_count
            )
            
            log_structured_event(
                agent="trend_continuation",
                event_type="optimization.executed",
                message=f"Bracket modified successfully",
                payload={
                    "trade_id": trade_id,
                    "new_stop": analysis.new_stop_loss,
                    "new_target": analysis.new_take_profit,
                    "modification_count": self._trade_modification_count[trade_id],
                    "savings_estimate": analysis.savings_estimate,
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute bracket modification: {e}")
            log_structured_event(
                agent="trend_continuation",
                event_type="optimization.failed",
                message=f"Bracket modification failed: {e}",
                level="error"
            )
            return False
