"""
Multi-Strategy Trading System
Implements Trend-Following, Breakout, and Mean Reversion strategies
with comprehensive risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger


class MultiStrategy:
    """
    Comprehensive multi-strategy system with:
    - Trend Following (MA crossovers)
    - Breakout (price + volume confirmation)
    - Mean Reversion (Bollinger Bands + RSI)
    - Dynamic risk management (ATR-based stops)
    - Trailing stops
    """
    
    def __init__(
        self,
        strategy_mode: str = "trend_following",  # "trend_following", "breakout", "mean_reversion", "auto"
        reward_risk_ratio: float = 2.0,
        use_trailing_stop: bool = True,
        trailing_stop_pct: float = 0.5,  # 0.5% trailing stop
        min_confidence: float = 0.65,
        rag_engine=None  # Optional RAG engine for signal validation
    ):
        self.strategy_mode = strategy_mode
        self.reward_risk_ratio = reward_risk_ratio
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.min_confidence = min_confidence
        
        # Market context
        self.market_bias = "neutral"  # "bullish", "bearish", "neutral"
        self.volatility_level = "medium"  # "low", "medium", "high"
        
        # FIX: Strategy locking to prevent mid-position switching
        self.position_strategy = None  # Lock strategy when position is open
        
        # RAG validator for signal validation
        from .rag_validator import RAGSignalValidator
        self.rag_validator = RAGSignalValidator(rag_engine=rag_engine)
        
        logger.info(f"Initialized MultiStrategy: mode={strategy_mode}, R:R={reward_risk_ratio}, RAG={rag_engine is not None}")
    
    def analyze_market_context(self, df: pd.DataFrame) -> Dict:
        """
        Analyze overall market conditions to set bias and volatility context.
        """
        if len(df) < 50:
            return {"bias": "neutral", "volatility": "medium"}
        
        # Determine trend bias from longer-term MA
        df['ma_50'] = df['close'].rolling(50).mean()
        current_price = df['close'].iloc[-1]
        ma_50 = df['ma_50'].iloc[-1]
        
        if current_price > ma_50 * 1.01:
            bias = "bullish"
        elif current_price < ma_50 * 0.99:
            bias = "bearish"
        else:
            bias = "neutral"
        
        # Determine volatility from ATR
        atr = self._calculate_atr(df)
        avg_atr = atr.rolling(20).mean().iloc[-1] if len(atr) > 20 else atr.iloc[-1]
        current_atr = atr.iloc[-1]
        
        if current_atr > avg_atr * 1.3:
            volatility = "high"
        elif current_atr < avg_atr * 0.7:
            volatility = "low"
        else:
            volatility = "medium"
        
        self.market_bias = bias
        self.volatility_level = volatility
        
        logger.info(f"📊 Market Context: {bias} bias, {volatility} volatility")
        
        return {
            "bias": bias,
            "volatility": volatility,
            "atr": current_atr,
            "price": current_price,
            "ma_50": ma_50
        }
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_position: int = 0
    ) -> Tuple[str, float, Optional[Dict]]:
        """
        Generate trading signal with enhanced validation and safeguards.
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            current_position: Current position size
            
        Returns:
            Tuple of (action, confidence, risk_params)
        """
        # Safeguard 1: Validate input data
        if df is None or df.empty:
            logger.error("❌ Invalid data: DataFrame is None or empty")
            return "HOLD", 0.0, None
        
        # Safeguard 2: Check minimum data length
        min_required_rows = 50
        if len(df) < min_required_rows:
            logger.warning(f"⚠️  Insufficient data: {len(df)} rows (need {min_required_rows})")
            return "HOLD", 0.0, None
        
        # Safeguard 3: Validate required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return "HOLD", 0.0, None
        
        # Safeguard 4: Check for NaN values in critical columns
        critical_cols = ['close', 'high', 'low']
        for col in critical_cols:
            if df[col].isna().any():
                nan_count = df[col].isna().sum()
                logger.warning(f"⚠️  NaN values detected in {col}: {nan_count} rows")
                # Fill NaN with forward fill as a recovery measure
                df[col] = df[col].fillna(method='ffill')
        
        # Safeguard 5: Validate price data sanity
        last_close = df['close'].iloc[-1]
        if last_close <= 0 or not np.isfinite(last_close):
            logger.error(f"❌ Invalid price data: close={last_close}")
            return "HOLD", 0.0, None
        
        # Safeguard 6: Check for extreme price movements (possible data error)
        if len(df) > 1:
            price_change_pct = abs((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2])
            if price_change_pct > 0.2:  # 20% change in one bar
                logger.warning(f"⚠️  Extreme price movement detected: {price_change_pct*100:.1f}%")
                # Continue but with caution flag
        
        try:
            # Analyze market context first
            market_context = self.analyze_market_context(df)
            
            # Safeguard 7: Validate market context
            if not market_context or 'bias' not in market_context:
                logger.warning("⚠️  Invalid market context, using defaults")
                market_context = {'bias': 'neutral', 'volatility': 'medium', 'atr': 1.0}
            
            # Auto-select best strategy or use specified mode
            if self.strategy_mode == "auto":
                strategy_to_use = self._select_best_strategy(market_context)
            else:
                strategy_to_use = self.strategy_mode
            
            logger.debug(f"Using strategy: {strategy_to_use}")
            
            # Generate signal based on strategy
            if strategy_to_use == "trend_following":
                action, confidence, risk_params = self._trend_following_signal(df, current_position)
            elif strategy_to_use == "breakout":
                action, confidence, risk_params = self._breakout_signal(df, current_position)
            elif strategy_to_use == "mean_reversion":
                action, confidence, risk_params = self._mean_reversion_signal(df, current_position)
            else:
                logger.error(f"❌ Unknown strategy mode: {strategy_to_use}")
                return "HOLD", 0.0, None
            
            # Safeguard 8: Validate generated signal
            if action not in ["BUY", "SELL", "HOLD"]:
                logger.error(f"❌ Invalid action generated: {action}")
                return "HOLD", 0.0, None
            
            if not (0.0 <= confidence <= 1.0):
                logger.warning(f"⚠️  Confidence out of range: {confidence}, clamping to [0,1]")
                confidence = max(0.0, min(1.0, confidence))
            
            # Safeguard 9: Validate risk parameters
            if risk_params:
                risk_params = self._validate_risk_params(risk_params, df['close'].iloc[-1])
            
            # Apply minimum confidence threshold
            if confidence < self.min_confidence:
                logger.info(f"📉 Signal confidence {confidence:.2f} below threshold {self.min_confidence}")
                return "HOLD", confidence, risk_params
            
            # RAG validation (if enabled)
            if self.rag_validator.enabled:
                logger.info(f"🔍 Validating signal with RAG: {action} (confidence={confidence:.2f})")
                action, confidence, validation_reason = self.rag_validator.validate_signal(
                    action=action,
                    confidence=confidence,
                    risk_params=risk_params or {},
                    market_context=market_context,
                    df=df
                )
                
                if validation_reason:
                    logger.info(f"   RAG: {validation_reason}")
            
            return action, confidence, risk_params
            
        except Exception as e:
            logger.error(f"❌ Error generating signal: {e}")
            logger.exception("Signal generation traceback:")
            return "HOLD", 0.0, None
    
    def _validate_risk_params(self, risk_params: Dict, current_price: float) -> Dict:
        """Validate and sanitize risk parameters.
        
        Args:
            risk_params: Risk parameters dict
            current_price: Current market price
            
        Returns:
            Validated risk parameters
        """
        validated = risk_params.copy()
        
        # Validate stop loss levels
        for key in ['stop_loss_long', 'stop_loss_short']:
            if key in validated:
                stop_value = validated[key]
                if not isinstance(stop_value, (int, float)) or not np.isfinite(stop_value):
                    logger.warning(f"⚠️  Invalid {key}: {stop_value}, removing")
                    del validated[key]
                elif stop_value <= 0:
                    logger.warning(f"⚠️  Non-positive {key}: {stop_value}, removing")
                    del validated[key]
        
        # Validate take profit levels
        for key in ['take_profit_long', 'take_profit_short']:
            if key in validated:
                tp_value = validated[key]
                if not isinstance(tp_value, (int, float)) or not np.isfinite(tp_value):
                    logger.warning(f"⚠️  Invalid {key}: {tp_value}, removing")
                    del validated[key]
                elif tp_value <= 0:
                    logger.warning(f"⚠️  Non-positive {key}: {tp_value}, removing")
                    del validated[key]
        
        # Validate ATR
        if 'atr' in validated:
            atr = validated['atr']
            if not isinstance(atr, (int, float)) or not np.isfinite(atr) or atr <= 0:
                logger.warning(f"⚠️  Invalid ATR: {atr}, using default 1.0")
                validated['atr'] = 1.0
        
        return validated
        """
        Generate trading signal based on selected strategy.
        
        Returns:
            (action, confidence, risk_params)
            action: "BUY", "SELL", "HOLD"
            confidence: 0.0 to 1.0
            risk_params: {"stop_loss": price, "take_profit": price, "atr": value}
        """
        if len(df) < 50:
            return "HOLD", 0.0, None
        
        # Analyze market context first
        context = self.analyze_market_context(df)
        
        # Calculate risk parameters
        risk_params = self._calculate_risk_params(df)
        
        # FIX: Lock strategy when holding position to prevent mid-trade switching
        if current_position != 0:
            if self.position_strategy is None:
                # Position opened without strategy lock (shouldn't happen, but handle it)
                self.position_strategy = self.strategy_mode if self.strategy_mode != "auto" else "trend_following"
            strategy = self.position_strategy
            logger.info(f"🔒 Strategy locked: {strategy} (holding position)")
        else:
            # No position - can change strategy
            self.position_strategy = None
            
            # Auto-select strategy based on market conditions
            if self.strategy_mode == "auto":
                strategy = self._select_best_strategy(context, df)
            else:
                strategy = self.strategy_mode
            
            logger.info(f"🎯 Using strategy: {strategy}")
        
        # Generate signal based on selected strategy
        if strategy == "trend_following":
            action, confidence = self._trend_following_signal(df, context)
        elif strategy == "breakout":
            action, confidence = self._breakout_signal(df, context)
        elif strategy == "mean_reversion":
            action, confidence = self._mean_reversion_signal(df, context)
        else:
            return "HOLD", 0.0, None
        
        # Apply position rules
        if current_position > 0 and action == "BUY":
            action = "HOLD"  # Already long
        elif current_position < 0 and action == "SELL":
            action = "HOLD"  # Already short
        
        # Apply minimum confidence threshold (before RAG validation)
        if confidence < self.min_confidence:
            action = "HOLD"
        
        # RAG-enhanced signal validation
        if action != "HOLD" and current_position == 0:  # Only validate new entry signals
            action, confidence, validation_reason = self.rag_validator.validate_signal(
                action=action,
                confidence=confidence,
                risk_params=risk_params,
                market_context=context,
                df=df
            )
            if validation_reason:
                logger.info(f"📚 RAG: {validation_reason}")
            
            # Lock strategy if we're entering a position
            if action != "HOLD":
                self.position_strategy = strategy
                logger.info(f"🔒 Strategy locked to: {strategy}")
        
        logger.info(f"📊 SIGNAL: {action} (confidence={confidence:.2f})")
        
        return action, confidence, risk_params
    
    def _trend_following_signal(self, df: pd.DataFrame, context: Dict) -> Tuple[str, float]:
        """
        Trend-Following Strategy:
        - Short MA (10) crosses above Long MA (50) → BUY
        - Short MA (10) crosses below Long MA (50) → SELL
        - Confirm with price momentum
        """
        df['ma_10'] = df['close'].rolling(10).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        
        ma_10_current = df['ma_10'].iloc[-1]
        ma_10_prev = df['ma_10'].iloc[-2]
        ma_50_current = df['ma_50'].iloc[-1]
        ma_50_prev = df['ma_50'].iloc[-2]
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        
        # Detect crossover
        bullish_cross = (ma_10_prev <= ma_50_prev) and (ma_10_current > ma_50_current)
        bearish_cross = (ma_10_prev >= ma_50_prev) and (ma_10_current < ma_50_current)
        
        # Calculate confidence based on:
        # 1. Distance between MAs
        # 2. Price momentum
        # 3. Market bias alignment
        
        ma_separation = abs(ma_10_current - ma_50_current) / ma_50_current
        price_momentum = (current_price - prev_price) / prev_price
        
        if bullish_cross:
            confidence = 0.7  # Base confidence
            
            # Boost confidence with confirmations
            if price_momentum > 0:
                confidence += 0.1
            if context['bias'] == "bullish":
                confidence += 0.1
            if ma_separation > 0.005:  # 0.5% separation
                confidence += 0.1
            
            return "BUY", min(confidence, 1.0)
        
        elif bearish_cross:
            confidence = 0.7
            
            if price_momentum < 0:
                confidence += 0.1
            if context['bias'] == "bearish":
                confidence += 0.1
            if ma_separation > 0.005:
                confidence += 0.1
            
            return "SELL", min(confidence, 1.0)
        
        # Check if already in trend
        elif ma_10_current > ma_50_current * 1.002:
            # Strong uptrend, consider holding/buying
            return "BUY", 0.6
        elif ma_10_current < ma_50_current * 0.998:
            # Strong downtrend, consider holding/selling
            return "SELL", 0.6
        
        # NEW: Detect strong momentum moves (4+ point drops/rises on ES)
        # Calculate recent price change over last 5 bars
        if len(df) >= 5:
            price_5_bars_ago = df['close'].iloc[-5]
            price_change_points = current_price - price_5_bars_ago
            price_change_pct = (current_price - price_5_bars_ago) / price_5_bars_ago
            
            # Strong downward momentum: 3+ point drop (~0.045%) in 5 bars
            if price_change_points < -3.0 and price_change_pct < -0.0004:
                confidence = 0.65
                if price_change_points < -5.0:  # Very strong drop
                    confidence = 0.75
                return "SELL", confidence
            
            # Strong upward momentum: 3+ point rise (~0.045%) in 5 bars  
            elif price_change_points > 3.0 and price_change_pct > 0.0004:
                confidence = 0.65
                if price_change_points > 5.0:  # Very strong rise
                    confidence = 0.75
                return "BUY", confidence
        
        return "HOLD", 0.0
    
    def _breakout_signal(self, df: pd.DataFrame, context: Dict) -> Tuple[str, float]:
        """
        Breakout Strategy:
        - Price breaks above previous day high + volume confirmation → BUY
        - Price breaks below previous day low + volume confirmation → SELL
        """
        # Calculate rolling highs/lows (simulate "previous day")
        lookback = 20  # Use last 20 bars as reference
        df['rolling_high'] = df['high'].rolling(lookback).max()
        df['rolling_low'] = df['low'].rolling(lookback).min()
        
        # Volume analysis
        df['avg_volume'] = df['volume'].rolling(20).mean()
        
        current_price = df['close'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        prev_high = df['rolling_high'].iloc[-2]
        prev_low = df['rolling_low'].iloc[-2]
        avg_volume = df['avg_volume'].iloc[-1]
        
        # Detect breakout
        breakout_up = current_high > prev_high
        breakout_down = current_low < prev_low
        volume_surge = current_volume > avg_volume * 1.2
        
        if breakout_up:
            confidence = 0.65
            
            # Volume confirmation
            if volume_surge:
                confidence += 0.15
            
            # Trend alignment
            if context['bias'] == "bullish":
                confidence += 0.1
            
            # Strong breakout (> 0.05% above high = ~3 points on ES)
            breakout_strength = (current_price - prev_high) / prev_high
            if breakout_strength > 0.0005:  # Reduced from 0.003 to 0.0005 for ES sensitivity
                confidence += 0.1
            
            return "BUY", min(confidence, 1.0)
        
        elif breakout_down:
            confidence = 0.65
            
            if volume_surge:
                confidence += 0.15
            
            if context['bias'] == "bearish":
                confidence += 0.1
            
            breakout_strength = (prev_low - current_price) / prev_low
            if breakout_strength > 0.0005:  # Reduced from 0.003 to 0.0005 for ES sensitivity
                confidence += 0.1
            
            return "SELL", min(confidence, 1.0)
        
        return "HOLD", 0.0
    
    def _mean_reversion_signal(self, df: pd.DataFrame, context: Dict) -> Tuple[str, float]:
        """
        Mean Reversion Strategy:
        - Price below lower Bollinger Band + RSI < 30 → BUY (oversold)
        - Price above upper Bollinger Band + RSI > 70 → SELL (overbought)
        """
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        current_price = df['close'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        bb_middle = df['bb_middle'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        
        # Oversold condition (buy signal)
        if current_price < bb_lower and rsi < 30:
            confidence = 0.7
            
            # Extra oversold
            if rsi < 25:
                confidence += 0.1
            
            # Price significantly below lower band
            distance = (bb_lower - current_price) / bb_lower
            if distance > 0.01:  # 1% below
                confidence += 0.1
            
            # Avoid fighting strong downtrend
            if context['bias'] == "bearish":
                confidence -= 0.1
            
            return "BUY", max(min(confidence, 1.0), 0.6)
        
        # Overbought condition (sell signal)
        elif current_price > bb_upper and rsi > 70:
            confidence = 0.7
            
            if rsi > 75:
                confidence += 0.1
            
            distance = (current_price - bb_upper) / bb_upper
            if distance > 0.01:
                confidence += 0.1
            
            if context['bias'] == "bullish":
                confidence -= 0.1
            
            return "SELL", max(min(confidence, 1.0), 0.6)
        
        # Near middle band - neutral
        return "HOLD", 0.0
    
    def _calculate_risk_params(self, df: pd.DataFrame) -> Dict:
        """
        Calculate stop-loss and take-profit levels based on ATR.
        """
        atr = self._calculate_atr(df)
        current_atr = atr.iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # ATR-based stops (adjust multiplier based on volatility)
        if self.volatility_level == "high":
            atr_multiplier = 2.5
        elif self.volatility_level == "low":
            atr_multiplier = 1.5
        else:
            atr_multiplier = 2.0
        
        stop_distance = current_atr * atr_multiplier
        target_distance = stop_distance * self.reward_risk_ratio
        
        return {
            "atr": current_atr,
            "stop_loss_long": current_price - stop_distance,
            "stop_loss_short": current_price + stop_distance,
            "take_profit_long": current_price + target_distance,
            "take_profit_short": current_price - target_distance,
            "trailing_stop_pct": self.trailing_stop_pct
        }
    
    def _select_best_strategy(self, context: Dict, df: pd.DataFrame) -> str:
        """
        Auto-select best strategy based on market conditions.
        """
        # In high volatility + trending market → Trend Following
        if context['volatility'] == "high" and context['bias'] != "neutral":
            return "trend_following"
        
        # In low volatility + sideways market → Mean Reversion
        elif context['volatility'] == "low" and context['bias'] == "neutral":
            return "mean_reversion"
        
        # In medium volatility + any trend → Breakout
        elif context['volatility'] == "medium":
            return "breakout"
        
        # Default to trend following
        return "trend_following"
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def should_exit_position(
        self,
        df: pd.DataFrame,
        entry_price: float,
        position: int,
        risk_params: Dict,
        entry_index: int = None
    ) -> Tuple[bool, str]:
        """
        Determine if position should be exited based on:
        - Stop loss hit
        - Take profit hit
        - Trailing stop
        - Strategy reversal
        
        Args:
            entry_index: Optional index position where trade was entered
        """
        if position == 0:
            return False, "No position"
        
        current_price = df['close'].iloc[-1]
        
        # Check stop loss
        if position > 0:  # Long position
            if current_price <= risk_params['stop_loss_long']:
                return True, f"Stop loss hit: {current_price:.2f} <= {risk_params['stop_loss_long']:.2f}"
            
            if current_price >= risk_params['take_profit_long']:
                return True, f"Take profit hit: {current_price:.2f} >= {risk_params['take_profit_long']:.2f}"
            
            # Trailing stop - find max price since entry
            if self.use_trailing_stop:
                if entry_index is not None and entry_index < len(df):
                    max_price_since_entry = df['high'].iloc[entry_index:].max()
                else:
                    # If no entry index provided, use last 20 bars as estimate
                    max_price_since_entry = df['high'].tail(20).max()
                    
                trailing_stop = max_price_since_entry * (1 - self.trailing_stop_pct / 100)
                if current_price <= trailing_stop:
                    return True, f"Trailing stop: {current_price:.2f} <= {trailing_stop:.2f}"
        
        else:  # Short position
            if current_price >= risk_params['stop_loss_short']:
                return True, f"Stop loss hit: {current_price:.2f} >= {risk_params['stop_loss_short']:.2f}"
            
            if current_price <= risk_params['take_profit_short']:
                return True, f"Take profit hit: {current_price:.2f} <= {risk_params['take_profit_short']:.2f}"
            
            # Trailing stop
            if self.use_trailing_stop:
                if entry_index is not None and entry_index < len(df):
                    min_price_since_entry = df['low'].iloc[entry_index:].min()
                else:
                    min_price_since_entry = df['low'].tail(20).min()
                    
                trailing_stop = min_price_since_entry * (1 + self.trailing_stop_pct / 100)
                if current_price >= trailing_stop:
                    return True, f"Trailing stop: {current_price:.2f} >= {trailing_stop:.2f}"
        
        return False, "No exit condition"
