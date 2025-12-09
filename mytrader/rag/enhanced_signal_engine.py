"""Enhanced Signal Engine - Improved trend detection for low-volatility markets.

This module addresses the core issues identified in the bot's signal generation:
1. Trend detection too strict (requires price > EMA9 > EMA20)
2. No micro-trend detection (0.1%-0.3% moves)
3. ATR filter blocking signals in low-vol environments
4. Signal threshold (40) too high for subtle setups
5. No scalping mode for tight ranges

FIXES IMPLEMENTED:
- Multi-timeframe trend detection (micro, short, medium)
- Adaptive volatility regime classification
- Reduced signal threshold for scalp setups
- Mean reversion detection for range-bound days
- Momentum scoring per candle
- VWAP and EMA curl patterns
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    from ..utils.timezone_utils import now_cst
except ImportError:
    from datetime import timezone
    def now_cst():
        return datetime.now(timezone.utc)


class TrendType(Enum):
    """Enhanced trend classification."""
    STRONG_UPTREND = "STRONG_UPTREND"      # > 0.3% move, aligned EMAs
    MICRO_UPTREND = "MICRO_UPTREND"        # 0.1%-0.3% move
    WEAK_UPTREND = "WEAK_UPTREND"          # Price above EMAs but not aligned
    STRONG_DOWNTREND = "STRONG_DOWNTREND"  # > 0.3% move, aligned EMAs
    MICRO_DOWNTREND = "MICRO_DOWNTREND"    # -0.1% to -0.3% move
    WEAK_DOWNTREND = "WEAK_DOWNTREND"      # Price below EMAs but not aligned
    RANGE_BOUND = "RANGE_BOUND"            # Oscillating within tight range
    CHOP = "CHOP"                          # No clear direction, erratic
    BREAKOUT_UP = "BREAKOUT_UP"            # Breaking above resistance
    BREAKOUT_DOWN = "BREAKOUT_DOWN"        # Breaking below support


class VolatilityRegime(Enum):
    """Enhanced volatility classification."""
    VERY_LOW = "VERY_LOW"     # ATR ratio < 0.5 - scalp mode
    LOW = "LOW"               # ATR ratio 0.5-0.7
    NORMAL = "NORMAL"         # ATR ratio 0.7-1.3
    HIGH = "HIGH"             # ATR ratio 1.3-2.0
    EXTREME = "EXTREME"       # ATR ratio > 2.0


class SignalType(Enum):
    """Signal types with strength levels."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    SCALP_BUY = "SCALP_BUY"
    HOLD = "HOLD"
    SCALP_SELL = "SCALP_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class EnhancedSignalResult:
    """Result from enhanced signal engine."""
    # Primary signal
    signal: SignalType = SignalType.HOLD
    confidence: float = 0.0
    
    # Trend analysis
    trend_1m: TrendType = TrendType.CHOP
    trend_5m: TrendType = TrendType.CHOP
    trend_15m: TrendType = TrendType.CHOP
    htf_bias: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    
    # Volatility
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL
    atr_value: float = 0.0
    atr_ratio: float = 1.0
    
    # Pattern detection
    patterns_detected: List[str] = field(default_factory=list)
    
    # Momentum scoring
    momentum_score: float = 0.0  # -100 to +100
    
    # Price levels
    near_support: bool = False
    near_resistance: bool = False
    vwap_position: str = "AT"  # ABOVE, BELOW, AT
    
    # Signal components
    trend_score: float = 0.0
    momentum_component: float = 0.0
    level_component: float = 0.0
    pattern_component: float = 0.0
    
    # Filters
    filters_passed: List[str] = field(default_factory=list)
    filters_warned: List[str] = field(default_factory=list)
    filters_blocked: List[str] = field(default_factory=list)
    
    # Reasoning
    reasoning: str = ""
    
    @property
    def is_actionable(self) -> bool:
        """Whether this signal should generate a trade."""
        return self.signal not in [SignalType.HOLD] and not self.filters_blocked
    
    @property
    def is_scalp(self) -> bool:
        """Whether this is a scalp setup."""
        return self.signal in [SignalType.SCALP_BUY, SignalType.SCALP_SELL]


class EnhancedSignalEngine:
    """Enhanced signal engine with micro-trend detection.
    
    Key improvements over original RuleEngine:
    1. Detects 0.1%-0.3% micro trends
    2. Adapts to low-volatility environments
    3. Lower signal threshold for scalp setups
    4. Multi-timeframe trend alignment
    5. Pattern recognition (EMA curl, VWAP reclaim, etc.)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced signal engine."""
        config = config or {}
        
        # === THRESHOLDS (RELAXED FOR LOW-VOL) ===
        
        # ATR thresholds - MUCH lower for low-vol days
        self.atr_min = config.get("atr_min", 0.05)  # Was 0.15, now 0.05
        self.atr_max = config.get("atr_max", 10.0)
        
        # Signal thresholds - LOWER for micro trends
        self.strong_signal_threshold = config.get("strong_threshold", 50)
        self.normal_signal_threshold = config.get("normal_threshold", 30)  # Was 40
        self.scalp_signal_threshold = config.get("scalp_threshold", 20)    # NEW: for scalps
        
        # Trend detection - RELAXED
        self.micro_trend_pct = config.get("micro_trend_pct", 0.1)  # 0.1% = micro trend
        self.strong_trend_pct = config.get("strong_trend_pct", 0.3)  # 0.3% = strong trend
        
        # Momentum
        self.rsi_oversold = config.get("rsi_oversold", 35)  # Was 30, now more sensitive
        self.rsi_overbought = config.get("rsi_overbought", 65)  # Was 70
        
        # Proximity thresholds
        self.level_proximity_pct = config.get("level_proximity_pct", 0.15)  # 0.15%
        self.vwap_proximity_pct = config.get("vwap_proximity_pct", 0.1)  # 0.1%
        
        # Component weights - REBALANCED for micro trends
        self.trend_weight = config.get("trend_weight", 25)  # Was 30
        self.momentum_weight = config.get("momentum_weight", 30)  # Was 25 - increased
        self.level_weight = config.get("level_weight", 25)
        self.pattern_weight = config.get("pattern_weight", 20)  # NEW
        
        # Cooldown
        self.cooldown_minutes = config.get("cooldown_minutes", 10)  # Was 15
        self.last_trade_time: Optional[datetime] = None
        
        # Price history for pattern detection
        self.price_history: List[Dict] = []
        self.max_history = 50
        
        logger.info(f"EnhancedSignalEngine initialized (atr_min={self.atr_min}, scalp_threshold={self.scalp_signal_threshold})")
    
    def evaluate(self, market_data: Dict[str, Any]) -> EnhancedSignalResult:
        """Evaluate market data with enhanced signal generation.
        
        Args:
            market_data: Current market data with indicators
            
        Returns:
            EnhancedSignalResult with detailed analysis
        """
        result = EnhancedSignalResult()
        
        # Extract indicators
        price = float(market_data.get("close", market_data.get("price", 0)))
        open_price = float(market_data.get("open", price))
        high = float(market_data.get("high", price))
        low = float(market_data.get("low", price))
        
        ema_9 = float(market_data.get("ema_9", price))
        ema_20 = float(market_data.get("ema_20", price))
        ema_50 = float(market_data.get("ema_50", price))
        vwap = float(market_data.get("vwap", price))
        
        rsi = float(market_data.get("rsi", 50))
        macd_hist = float(market_data.get("macd_hist", 0))
        macd = float(market_data.get("macd", 0))
        macd_signal = float(market_data.get("macd_signal", 0))
        
        atr = float(market_data.get("atr", 0.5))
        atr_20_avg = float(market_data.get("atr_20_avg", atr))
        
        pdh = float(market_data.get("pdh", 0))
        pdl = float(market_data.get("pdl", 0))
        weekly_high = float(market_data.get("weekly_high", 0))
        weekly_low = float(market_data.get("weekly_low", 0))
        
        volume_ratio = float(market_data.get("volume_ratio", 1.0))
        
        # Store ATR values
        result.atr_value = atr
        result.atr_ratio = atr / atr_20_avg if atr_20_avg > 0 else 1.0
        
        # === 1. VOLATILITY REGIME DETECTION ===
        result.volatility_regime = self._classify_volatility(result.atr_ratio)
        
        # === 2. MULTI-TIMEFRAME TREND DETECTION ===
        result.trend_1m = self._detect_micro_trend(price, ema_9, ema_20, open_price)
        result.trend_5m = self._detect_short_trend(price, ema_9, ema_20, ema_50)
        result.trend_15m = self._detect_medium_trend(price, ema_20, ema_50)
        result.htf_bias = self._get_htf_bias(price, ema_50, pdh, pdl)
        
        # === 3. VWAP POSITION ===
        vwap_dist_pct = (price - vwap) / vwap * 100 if vwap > 0 else 0
        if vwap_dist_pct > self.vwap_proximity_pct:
            result.vwap_position = "ABOVE"
        elif vwap_dist_pct < -self.vwap_proximity_pct:
            result.vwap_position = "BELOW"
        else:
            result.vwap_position = "AT"
        
        # === 4. LEVEL PROXIMITY ===
        if pdh > 0 and pdl > 0:
            pdh_dist_pct = abs(price - pdh) / price * 100
            pdl_dist_pct = abs(price - pdl) / price * 100
            result.near_resistance = pdh_dist_pct < self.level_proximity_pct
            result.near_support = pdl_dist_pct < self.level_proximity_pct
        
        # === 5. PATTERN DETECTION ===
        result.patterns_detected = self._detect_patterns(
            price, ema_9, ema_20, vwap, rsi, macd_hist, 
            pdh, pdl, open_price, high, low
        )
        
        # === 6. MOMENTUM SCORING (-100 to +100) ===
        result.momentum_score = self._calculate_momentum_score(
            rsi, macd_hist, price, ema_9, ema_20, volume_ratio
        )
        
        # === 7. HARD FILTERS (with relaxed ATR) ===
        self._apply_filters(result, atr, price)
        
        # If blocked, return early
        if result.filters_blocked:
            result.signal = SignalType.HOLD
            result.reasoning = f"Blocked by: {', '.join(result.filters_blocked)}"
            return result
        
        # === 8. SIGNAL SCORING ===
        buy_score, sell_score = self._calculate_signal_scores(
            result, price, ema_9, ema_20, rsi, macd_hist, 
            pdh, pdl, vwap, volume_ratio
        )
        
        # Store components
        result.trend_score = buy_score - sell_score  # Net trend score
        
        # === 9. DETERMINE FINAL SIGNAL ===
        result.signal, result.confidence = self._determine_signal(
            buy_score, sell_score, result.volatility_regime
        )
        
        # === 10. BUILD REASONING ===
        result.reasoning = self._build_reasoning(result, buy_score, sell_score)
        
        return result
    
    def _classify_volatility(self, atr_ratio: float) -> VolatilityRegime:
        """Classify volatility regime based on ATR ratio."""
        if atr_ratio < 0.5:
            return VolatilityRegime.VERY_LOW
        elif atr_ratio < 0.7:
            return VolatilityRegime.LOW
        elif atr_ratio < 1.3:
            return VolatilityRegime.NORMAL
        elif atr_ratio < 2.0:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def _detect_micro_trend(
        self, price: float, ema_9: float, ema_20: float, open_price: float
    ) -> TrendType:
        """Detect micro trends (0.1%-0.3% moves) on 1-minute timeframe."""
        # Calculate price change from open
        if open_price > 0:
            pct_change = (price - open_price) / open_price * 100
        else:
            pct_change = 0
        
        # EMA slope (approximate)
        ema_diff_pct = (ema_9 - ema_20) / ema_20 * 100 if ema_20 > 0 else 0
        
        # Strong trend: > 0.3% with aligned EMAs
        if pct_change >= self.strong_trend_pct and price > ema_9 > ema_20:
            return TrendType.STRONG_UPTREND
        elif pct_change <= -self.strong_trend_pct and price < ema_9 < ema_20:
            return TrendType.STRONG_DOWNTREND
        
        # Micro trend: 0.1%-0.3%
        if self.micro_trend_pct <= pct_change < self.strong_trend_pct:
            if price > ema_9:
                return TrendType.MICRO_UPTREND
            else:
                return TrendType.WEAK_UPTREND
        elif -self.strong_trend_pct < pct_change <= -self.micro_trend_pct:
            if price < ema_9:
                return TrendType.MICRO_DOWNTREND
            else:
                return TrendType.WEAK_DOWNTREND
        
        # Weak trend: price above/below EMAs but small move
        if price > ema_9 and ema_diff_pct > 0.02:
            return TrendType.WEAK_UPTREND
        elif price < ema_9 and ema_diff_pct < -0.02:
            return TrendType.WEAK_DOWNTREND
        
        # Range-bound or chop
        if abs(ema_diff_pct) < 0.02:
            return TrendType.RANGE_BOUND
        
        return TrendType.CHOP
    
    def _detect_short_trend(
        self, price: float, ema_9: float, ema_20: float, ema_50: float
    ) -> TrendType:
        """Detect 5-minute trend based on EMA alignment."""
        if price > ema_9 > ema_20 > ema_50:
            return TrendType.STRONG_UPTREND
        elif price > ema_9 > ema_20:
            return TrendType.MICRO_UPTREND
        elif price > ema_9:
            return TrendType.WEAK_UPTREND
        elif price < ema_9 < ema_20 < ema_50:
            return TrendType.STRONG_DOWNTREND
        elif price < ema_9 < ema_20:
            return TrendType.MICRO_DOWNTREND
        elif price < ema_9:
            return TrendType.WEAK_DOWNTREND
        else:
            return TrendType.CHOP
    
    def _detect_medium_trend(
        self, price: float, ema_20: float, ema_50: float
    ) -> TrendType:
        """Detect 15-minute trend for HTF confirmation."""
        if price > ema_20 > ema_50:
            ema_spread = (ema_20 - ema_50) / ema_50 * 100
            if ema_spread > 0.2:
                return TrendType.STRONG_UPTREND
            return TrendType.MICRO_UPTREND
        elif price < ema_20 < ema_50:
            ema_spread = (ema_50 - ema_20) / ema_50 * 100
            if ema_spread > 0.2:
                return TrendType.STRONG_DOWNTREND
            return TrendType.MICRO_DOWNTREND
        elif price > ema_50:
            return TrendType.WEAK_UPTREND
        elif price < ema_50:
            return TrendType.WEAK_DOWNTREND
        return TrendType.RANGE_BOUND
    
    def _get_htf_bias(
        self, price: float, ema_50: float, pdh: float, pdl: float
    ) -> str:
        """Get higher timeframe bias."""
        bullish_score = 0
        bearish_score = 0
        
        if price > ema_50:
            bullish_score += 1
        else:
            bearish_score += 1
        
        if pdh > 0 and pdl > 0:
            mid_range = (pdh + pdl) / 2
            if price > mid_range:
                bullish_score += 1
            else:
                bearish_score += 1
        
        if bullish_score > bearish_score:
            return "BULLISH"
        elif bearish_score > bullish_score:
            return "BEARISH"
        return "NEUTRAL"
    
    def _detect_patterns(
        self, price: float, ema_9: float, ema_20: float, vwap: float,
        rsi: float, macd_hist: float, pdh: float, pdl: float,
        open_price: float, high: float, low: float
    ) -> List[str]:
        """Detect tradeable patterns."""
        patterns = []
        
        # EMA Curl Up (EMA9 crossing above EMA20)
        if 0 < (ema_9 - ema_20) / ema_20 * 100 < 0.05 and price > ema_9:
            patterns.append("EMA_CURL_UP")
        
        # EMA Curl Down
        if -0.05 < (ema_9 - ema_20) / ema_20 * 100 < 0 and price < ema_9:
            patterns.append("EMA_CURL_DOWN")
        
        # VWAP Reclaim (price crosses above VWAP)
        if vwap > 0:
            vwap_dist = (price - vwap) / vwap * 100
            if 0 < vwap_dist < 0.1 and price > open_price:
                patterns.append("VWAP_RECLAIM")
            elif -0.1 < vwap_dist < 0 and price < open_price:
                patterns.append("VWAP_REJECT")
        
        # PDH/PDL Bounce
        if pdl > 0:
            pdl_dist = (price - pdl) / pdl * 100
            if 0 < pdl_dist < 0.15 and low <= pdl * 1.001:
                patterns.append("PDL_BOUNCE")
        
        if pdh > 0:
            pdh_dist = (pdh - price) / pdh * 100
            if 0 < pdh_dist < 0.15 and high >= pdh * 0.999:
                patterns.append("PDH_REJECTION")
        
        # RSI Divergence (simplified)
        if rsi < 40 and macd_hist > 0:
            patterns.append("BULLISH_DIVERGENCE")
        elif rsi > 60 and macd_hist < 0:
            patterns.append("BEARISH_DIVERGENCE")
        
        # Momentum Burst
        candle_range = high - low if high > low else 0.01
        body = abs(price - open_price)
        if body / candle_range > 0.7:  # Strong candle body
            if price > open_price:
                patterns.append("BULLISH_MOMENTUM")
            else:
                patterns.append("BEARISH_MOMENTUM")
        
        # Range Breakout
        if pdh > 0 and price > pdh:
            patterns.append("BREAKOUT_UP")
        elif pdl > 0 and price < pdl:
            patterns.append("BREAKOUT_DOWN")
        
        return patterns
    
    def _calculate_momentum_score(
        self, rsi: float, macd_hist: float, price: float,
        ema_9: float, ema_20: float, volume_ratio: float
    ) -> float:
        """Calculate momentum score from -100 to +100."""
        score = 0.0
        
        # RSI component (-30 to +30)
        if rsi < 30:
            score += 30  # Oversold = bullish
        elif rsi > 70:
            score -= 30  # Overbought = bearish
        else:
            # Linear scale in middle zone
            score += (50 - rsi) * 0.5
        
        # MACD Histogram component (-25 to +25)
        macd_contribution = max(-25, min(25, macd_hist * 100))
        score += macd_contribution
        
        # Price vs EMA component (-25 to +25)
        if ema_20 > 0:
            ema_pct = (price - ema_20) / ema_20 * 100
            score += max(-25, min(25, ema_pct * 50))
        
        # Volume boost (-20 to +20)
        if volume_ratio > 1.5:
            # High volume amplifies existing direction
            if score > 0:
                score += 20
            elif score < 0:
                score -= 20
        
        return max(-100, min(100, score))
    
    def _apply_filters(
        self, result: EnhancedSignalResult, atr: float, price: float
    ) -> None:
        """Apply hard filters with relaxed thresholds."""
        # ATR filter - VERY relaxed
        if atr < self.atr_min:
            # Don't block, just warn - allow scalps
            result.filters_warned.append(f"VERY_LOW_ATR ({atr:.3f})")
        elif atr > self.atr_max:
            result.filters_blocked.append(f"EXTREME_ATR ({atr:.2f})")
        else:
            result.filters_passed.append("ATR_OK")
        
        # Cooldown - reduced
        if self.last_trade_time:
            elapsed = (now_cst() - self.last_trade_time).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                result.filters_blocked.append(f"COOLDOWN ({elapsed:.1f}m < {self.cooldown_minutes}m)")
            else:
                result.filters_passed.append("COOLDOWN_OK")
        else:
            result.filters_passed.append("COOLDOWN_OK")
        
        # Market hours
        current_time = now_cst()
        hour = current_time.hour
        if 8 <= hour <= 15:
            result.filters_passed.append("MARKET_HOURS_OK")
        else:
            result.filters_warned.append("OUTSIDE_CORE_HOURS")
    
    def _calculate_signal_scores(
        self, result: EnhancedSignalResult, price: float, ema_9: float,
        ema_20: float, rsi: float, macd_hist: float, pdh: float, pdl: float,
        vwap: float, volume_ratio: float
    ) -> Tuple[float, float]:
        """Calculate buy and sell scores with enhanced logic."""
        buy_score = 0.0
        sell_score = 0.0
        
        # === TREND COMPONENT (0-25 pts) ===
        trend_types = [result.trend_1m, result.trend_5m]
        
        for trend in trend_types:
            if trend in [TrendType.STRONG_UPTREND, TrendType.BREAKOUT_UP]:
                buy_score += self.trend_weight / 2
            elif trend in [TrendType.MICRO_UPTREND, TrendType.WEAK_UPTREND]:
                buy_score += self.trend_weight / 4  # Partial credit for micro
            elif trend in [TrendType.STRONG_DOWNTREND, TrendType.BREAKOUT_DOWN]:
                sell_score += self.trend_weight / 2
            elif trend in [TrendType.MICRO_DOWNTREND, TrendType.WEAK_DOWNTREND]:
                sell_score += self.trend_weight / 4
        
        # HTF bias bonus
        if result.htf_bias == "BULLISH":
            buy_score += 5
        elif result.htf_bias == "BEARISH":
            sell_score += 5
        
        result.trend_score = buy_score - sell_score
        
        # === MOMENTUM COMPONENT (0-30 pts) ===
        if result.momentum_score > 30:
            buy_score += self.momentum_weight * (result.momentum_score / 100)
        elif result.momentum_score < -30:
            sell_score += self.momentum_weight * (abs(result.momentum_score) / 100)
        
        # RSI extremes
        if rsi < self.rsi_oversold:
            buy_score += 10
            result.filters_passed.append("RSI_OVERSOLD")
        elif rsi > self.rsi_overbought:
            sell_score += 10
            result.filters_passed.append("RSI_OVERBOUGHT")
        
        # MACD
        if macd_hist > 0:
            buy_score += 5
        elif macd_hist < 0:
            sell_score += 5
        
        result.momentum_component = buy_score - sell_score - result.trend_score
        
        # === LEVEL COMPONENT (0-25 pts) ===
        if result.near_support:
            buy_score += self.level_weight * 0.7
            result.filters_passed.append("NEAR_SUPPORT")
        if result.near_resistance:
            sell_score += self.level_weight * 0.5
            result.filters_warned.append("NEAR_RESISTANCE")
        
        # VWAP
        if result.vwap_position == "ABOVE" and result.momentum_score > 0:
            buy_score += 5
        elif result.vwap_position == "BELOW" and result.momentum_score < 0:
            sell_score += 5
        
        result.level_component = (buy_score - sell_score - result.trend_score 
                                   - result.momentum_component)
        
        # === PATTERN COMPONENT (0-20 pts) ===
        bullish_patterns = ["EMA_CURL_UP", "VWAP_RECLAIM", "PDL_BOUNCE", 
                          "BULLISH_DIVERGENCE", "BULLISH_MOMENTUM", "BREAKOUT_UP"]
        bearish_patterns = ["EMA_CURL_DOWN", "VWAP_REJECT", "PDH_REJECTION",
                          "BEARISH_DIVERGENCE", "BEARISH_MOMENTUM", "BREAKOUT_DOWN"]
        
        for pattern in result.patterns_detected:
            if pattern in bullish_patterns:
                buy_score += self.pattern_weight / len(bullish_patterns)
            elif pattern in bearish_patterns:
                sell_score += self.pattern_weight / len(bearish_patterns)
        
        result.pattern_component = (buy_score - sell_score - result.trend_score 
                                     - result.momentum_component - result.level_component)
        
        # === VOLUME AMPLIFIER ===
        if volume_ratio > 1.5:
            buy_score *= 1.1
            sell_score *= 1.1
            result.filters_passed.append("HIGH_VOLUME")
        
        return buy_score, sell_score
    
    def _determine_signal(
        self, buy_score: float, sell_score: float, 
        volatility: VolatilityRegime
    ) -> Tuple[SignalType, float]:
        """Determine final signal with adaptive thresholds."""
        # Adjust thresholds based on volatility
        if volatility in [VolatilityRegime.VERY_LOW, VolatilityRegime.LOW]:
            # Low vol = use scalp threshold
            threshold = self.scalp_signal_threshold
            is_scalp_mode = True
        else:
            threshold = self.normal_signal_threshold
            is_scalp_mode = False
        
        net_score = buy_score - sell_score
        confidence = min(100, max(buy_score, sell_score))
        
        # Strong signals
        if buy_score >= self.strong_signal_threshold and buy_score > sell_score * 1.5:
            return SignalType.STRONG_BUY, confidence
        if sell_score >= self.strong_signal_threshold and sell_score > buy_score * 1.5:
            return SignalType.STRONG_SELL, confidence
        
        # Normal signals
        if buy_score > sell_score and buy_score >= threshold:
            if is_scalp_mode:
                return SignalType.SCALP_BUY, confidence
            return SignalType.BUY, confidence
        
        if sell_score > buy_score and sell_score >= threshold:
            if is_scalp_mode:
                return SignalType.SCALP_SELL, confidence
            return SignalType.SELL, confidence
        
        return SignalType.HOLD, max(buy_score, sell_score)
    
    def _build_reasoning(
        self, result: EnhancedSignalResult, buy_score: float, sell_score: float
    ) -> str:
        """Build human-readable reasoning."""
        parts = []
        
        parts.append(f"Signal: {result.signal.value} (conf={result.confidence:.0f}%)")
        parts.append(f"Volatility: {result.volatility_regime.value}")
        parts.append(f"Trends: 1m={result.trend_1m.value}, 5m={result.trend_5m.value}")
        parts.append(f"HTF Bias: {result.htf_bias}")
        parts.append(f"Momentum: {result.momentum_score:.1f}")
        
        if result.patterns_detected:
            parts.append(f"Patterns: {', '.join(result.patterns_detected)}")
        
        parts.append(f"Scores: BUY={buy_score:.1f}, SELL={sell_score:.1f}")
        
        if result.filters_blocked:
            parts.append(f"BLOCKED: {', '.join(result.filters_blocked)}")
        
        return " | ".join(parts)
    
    def record_trade(self) -> None:
        """Record that a trade was executed."""
        self.last_trade_time = now_cst()


def create_enhanced_engine(config: Optional[Dict[str, Any]] = None) -> EnhancedSignalEngine:
    """Factory function to create enhanced signal engine."""
    return EnhancedSignalEngine(config)
