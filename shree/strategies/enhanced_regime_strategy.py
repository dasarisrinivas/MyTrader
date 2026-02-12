"""
Enhanced Trading Strategy with Market Regime Detection and Advanced Filters
Designed for ES/SPY Futures with focus on maximizing Sharpe, minimizing drawdown
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


@dataclass
class EnhancedRegimeStrategy(BaseStrategy):
    """
    Advanced strategy combining:
    - Market regime detection (Trending/Range/Volatile)
    - Multi-timeframe confirmation
    - Dynamic thresholds based on volatility
    - Volume profile analysis
    - Risk-adjusted position sizing signals
    """
    name: str = "enhanced_regime"
    
    # RSI parameters (adaptive based on regime)
    rsi_oversold_trending: float = 28.0
    rsi_overbought_trending: float = 68.0
    rsi_oversold_ranging: float = 25.0
    rsi_overbought_ranging: float = 75.0
    
    # Trend detection
    adx_trend_threshold: float = 28.0
    adx_strong_trend: float = 40.0
    
    # Volatility filters
    atr_percentile_low: float = 15.0
    atr_percentile_high: float = 87.0
    volatility_floor: float = 0.00015
    volatility_ceiling: float = 0.0035
    
    # Volume filters
    volume_ma_multiplier: float = 0.9
    
    # MACD settings
    use_macd_histogram: bool = True
    macd_momentum_threshold: float = 0.0
    
    # Bollinger Band mean reversion
    bb_extreme_lower: float = 0.06
    bb_extreme_upper: float = 0.86
    bb_mid_lower: float = 0.35
    bb_mid_upper: float = 0.65
    
    # Multi-factor confirmation requirement
    min_confirmations_trend: int = 2
    min_confirmations_range: int = 2
    
    # Risk controls
    atr_stop_trending: float = 1.8
    atr_stop_ranging: float = 1.3
    atr_stop_volatile: float = 2.4
    risk_reward_trending: float = 2.4
    risk_reward_ranging: float = 1.6
    risk_reward_volatile: float = 1.1
    trailing_trend_multiplier: float = 0.9
    trailing_range_multiplier: float = 0.0
    trailing_volatile_multiplier: float = 1.2

    # Sentiment filters
    sentiment_filter_trend: float = -0.2
    sentiment_filter_range: float = -0.4
    sentiment_filter_short: float = 0.1

    # Time-based filters
    avoid_first_minutes: int = 15  # Avoid first 15 minutes of trading
    avoid_last_minutes: int = 30   # Avoid last 30 minutes
    session_start: str = "09:40"
    session_end: str = "15:45"
    
    def generate(self, features: pd.DataFrame) -> Signal:
        """Generate signal with comprehensive market analysis."""
        if len(features) < 50:  # Need sufficient history
            return Signal(action="HOLD", confidence=0.0, metadata={"reason": "insufficient_data"})
        
        # Get latest and historical data
        latest = features.iloc[-1]
        window_20 = features.tail(20)
        window_50 = features.tail(50)
        
        # Detect market regime
        regime, regime_confidence = self._detect_market_regime(window_50, latest)
        
        # Check time-based filters
        if not self._check_time_filter(latest):
            return Signal(
                action="HOLD", 
                confidence=0.0, 
                metadata={"reason": "time_filter", "regime": regime}
            )
        
        # Get all technical indicators
        indicators = self._extract_indicators(latest, window_20, window_50)
        
        # Check volatility filter
        if not self._check_volatility_filter(indicators, window_50):
            return Signal(
                action="HOLD",
                confidence=0.0,
                metadata={"reason": "volatility_filter", "regime": regime, **indicators}
            )
        
        # Check volume filter
        if not self._check_volume_filter(latest, window_20):
            return Signal(
                action="HOLD",
                confidence=0.0,
                metadata={"reason": "volume_filter", "regime": regime, **indicators}
            )
        
        # Generate regime-specific signals
        if regime == "TRENDING":
            signal = self._trending_strategy(indicators, regime_confidence)
        elif regime == "RANGING":
            signal = self._ranging_strategy(indicators, regime_confidence)
        else:  # VOLATILE
            signal = self._volatile_strategy(indicators, regime_confidence)
        
        # Add regime info to metadata
        signal.metadata.setdefault("atr_value", indicators.get("atr", 0.0))
        signal.metadata.setdefault("volatility", indicators.get("volatility", 0.0))
        signal.metadata.setdefault("price", indicators.get("close", 0.0))
        signal.metadata["regime"] = regime
        signal.metadata["regime_confidence"] = regime_confidence
        for key, value in indicators.items():
            signal.metadata.setdefault(key, value)
        
        return signal
    
    def _detect_market_regime(self, window: pd.DataFrame, latest: pd.Series) -> Tuple[str, float]:
        """
        Detect market regime: TRENDING, RANGING, or VOLATILE
        Returns: (regime_name, confidence)
        """
        adx = latest.get("ADX_14", 0)
        atr = latest.get("ATR_14", 1)
        
        # Calculate price range vs ATR
        price_range = window["high"].max() - window["low"].min()
        avg_atr = window.get("ATR_14", pd.Series([atr])).mean()
        
        # Calculate directional movement
        close_returns = window["close"].pct_change()
        trend_consistency = abs(close_returns.mean()) / (close_returns.std() + 1e-6)
        
        # Bollinger Band width for volatility assessment
        bb_width = latest.get("BB_width", 0)
        
        # Regime detection logic
        if adx > self.adx_strong_trend:
            # Strong trend detected
            confidence = min(0.95, adx / 50.0)
            return "TRENDING", confidence
        elif adx > self.adx_trend_threshold and trend_consistency > 0.5:
            # Moderate trend with consistent direction
            confidence = min(0.85, (adx / 50.0) * (trend_consistency))
            return "TRENDING", confidence
        elif bb_width > avg_atr * 2:
            # High volatility, choppy market
            confidence = min(0.9, bb_width / (avg_atr * 3))
            return "VOLATILE", confidence
        else:
            # Range-bound market
            confidence = 0.7
            return "RANGING", confidence
    
    def _extract_indicators(self, latest: pd.Series, window_20: pd.DataFrame, window_50: pd.DataFrame) -> Dict[str, float]:
        """Extract and calculate all technical indicators."""
        return {
            "rsi": float(latest.get("RSI_14", 50)),
            "macd": float(latest.get("MACD_12_26_9", 0)),
            "macd_signal": float(latest.get("MACDsignal_12_26_9", 0)),
            "macd_hist": float(latest.get("MACDhist_12_26_9", 0)),
            "adx": float(latest.get("ADX_14", 0)),
            "atr": float(latest.get("ATR_14", 1)),
            "bb_percent": float(latest.get("BB_percent", 0.5)),
            "bb_width": float(latest.get("BB_width", 0)),
            "ema_20": float(latest.get("EMA_20", latest.get("close", 0))),
            "ema_50": float(latest.get("EMA_50", latest.get("close", 0))),
            "ema_200": float(latest.get("EMA_200", latest.get("close", 0))),
            "sma_20": float(latest.get("SMA_20", latest.get("close", 0))),
            "vwap": float(latest.get("VWAP_daily", latest.get("close", 0))),
            "volume": float(latest.get("volume", 0)),
            "close": float(latest.get("close", 0)),
            "sentiment": float(latest.get("sentiment_score", 0)),
            "price_vs_ema20": float((latest.get("close", 0) - latest.get("EMA_20", latest.get("close", 0))) / latest.get("EMA_20", 1)),
            "price_vs_ema50": float((latest.get("close", 0) - latest.get("EMA_50", latest.get("close", 0))) / latest.get("EMA_50", 1)),
            "price_vs_ema200": float((latest.get("close", 0) - latest.get("EMA_200", latest.get("close", 0))) / latest.get("EMA_200", 1)),
            "volatility": float(latest.get("volatility", 0.0)),
            "volume_ratio": float(latest.get("volume_ratio", 1.0)),
            "rolling_return": float(latest.get("rolling_return", 0.0)),
            "momentum_10": float(latest.get("momentum_10", 0.0)),
        }
    
    def _check_time_filter(self, latest: pd.Series) -> bool:
        """Check if we should trade based on time of day."""
        ts = getattr(latest, "name", None)
        if not isinstance(ts, pd.Timestamp):
            return True

        # Session boundaries
        start_minutes = self._session_minutes(self.session_start)
        end_minutes = self._session_minutes(self.session_end)
        current_minutes = ts.hour * 60 + ts.minute

        # Enforce start/end window and avoid extremes
        if current_minutes < start_minutes:
            return False
        if current_minutes > end_minutes:
            return False

        # Avoid first/last minutes buffers if provided
        open_minutes = 9 * 60 + 30
        if current_minutes - open_minutes < self.avoid_first_minutes:
            return False
        close_minutes = 16 * 60
        if close_minutes - current_minutes < self.avoid_last_minutes:
            return False

        return True
    
    def _check_volatility_filter(self, indicators: Dict[str, float], window: pd.DataFrame) -> bool:
        """Check if volatility is within acceptable range."""
        atr = indicators["atr"]
        atr_history = window.get("ATR_14", pd.Series([atr]))
        
        if len(atr_history) < 20:
            return True
        
        atr_percentile = (atr_history < atr).sum() / len(atr_history) * 100
        
        # Avoid extremely low or high volatility periods
        if atr_percentile < self.atr_percentile_low or atr_percentile > self.atr_percentile_high:
            return False
        
        vol = indicators.get("volatility", 0.0)
        if vol > 0:
            if vol < self.volatility_floor or vol > self.volatility_ceiling:
                return False

        return True
    
    def _check_volume_filter(self, latest: pd.Series, window: pd.DataFrame) -> bool:
        """Ensure sufficient volume for trade execution."""
        current_volume = latest.get("volume", 0)
        avg_volume = window["volume"].mean()

        volume_ratio = latest.get("volume_ratio")

        if volume_ratio is not None and not np.isnan(volume_ratio):
            if volume_ratio < self.volume_ma_multiplier:
                return False

        # Require above-average volume for reliability
        return current_volume >= (avg_volume * self.volume_ma_multiplier)

    def _build_trade_metadata(
        self,
        action: str,
        regime: str,
        regime_confidence: float,
        confirmations: list[str],
        indicators: Dict[str, float],
        base_confidence: float,
        strategy_tag: str,
    ) -> Dict[str, float]:
        atr_value = indicators.get("atr", 0.0)

        if regime == "TRENDING":
            atr_multiplier = self.atr_stop_trending
            risk_reward = self.risk_reward_trending
            trailing = self.trailing_trend_multiplier
        elif regime == "RANGING":
            atr_multiplier = self.atr_stop_ranging
            risk_reward = self.risk_reward_ranging
            trailing = self.trailing_range_multiplier
        else:
            atr_multiplier = self.atr_stop_volatile
            risk_reward = self.risk_reward_volatile
            trailing = self.trailing_volatile_multiplier

        position_scaler = float(np.clip(base_confidence * regime_confidence, 0.3, 1.0))

        metadata = {
            "strategy": strategy_tag,
            "confirmations": confirmations,
            "atr_stop_multiplier": float(max(0.5, atr_multiplier)),
            "risk_reward": float(max(0.8, risk_reward)),
            "position_scaler": position_scaler,
            "atr_value": atr_value,
            "direction_bias": "long" if action == "BUY" else "short",
            "entry_sentiment": indicators.get("sentiment", 0.0),
            "entry_volatility": indicators.get("volatility", 0.0),
            "entry_bb_percent": indicators.get("bb_percent", 0.5),
        }

        if trailing and trailing > 0:
            metadata["trailing_atr_multiplier"] = float(trailing)

        return metadata
    
    def _trending_strategy(self, indicators: Dict[str, float], regime_confidence: float) -> Signal:
        """Strategy optimized for trending markets - trend following."""
        confirmations_buy = []
        confirmations_sell = []
        confidence = 0.5
        
        rsi = indicators["rsi"]
        macd_hist = indicators["macd_hist"]
        adx = indicators["adx"]
        bb_percent = indicators["bb_percent"]
        price_vs_ema20 = indicators["price_vs_ema20"]
        price_vs_ema50 = indicators["price_vs_ema50"]
        price_vs_ema200 = indicators.get("price_vs_ema200", 0.0)
        sentiment = indicators["sentiment"]
        vwap = indicators.get("vwap", indicators.get("close", 0))
        close_price = indicators.get("close", 0)

        if adx < self.adx_trend_threshold or regime_confidence < 0.45:
            return Signal(action="HOLD", confidence=0.0, metadata={"reason": "weak_trend"})
        
        # BUY confirmations for trending market
        if rsi < self.rsi_oversold_trending:
            confirmations_buy.append("rsi_oversold")
            confidence += (self.rsi_oversold_trending - rsi) / 100
        
        if macd_hist > self.macd_momentum_threshold:
            confirmations_buy.append("macd_bullish")
            confidence += 0.15
        
        if price_vs_ema20 > 0 and price_vs_ema50 > 0 and price_vs_ema200 >= -0.001:
            confirmations_buy.append("above_emas")
            confidence += 0.15
        elif price_vs_ema20 > 0:
            confirmations_buy.append("above_ema20")
            confidence += 0.1
        
        if bb_percent < self.bb_mid_lower:
            confirmations_buy.append("bb_lower")
            confidence += 0.1
        
        if sentiment > self.sentiment_filter_trend:
            confirmations_buy.append("positive_sentiment")
            confidence += sentiment * 0.1
        
        if close_price >= vwap:
            confirmations_buy.append("above_vwap")
            confidence += 0.05

        if adx > self.adx_strong_trend:
            confirmations_buy.append("strong_trend")
            confidence += 0.1
        
        # SELL confirmations for trending market
        if rsi > self.rsi_overbought_trending:
            confirmations_sell.append("rsi_overbought")
            confidence_sell = 0.5 + (rsi - self.rsi_overbought_trending) / 100
        else:
            confidence_sell = 0.5
        
        if macd_hist < -self.macd_momentum_threshold:
            confirmations_sell.append("macd_bearish")
            confidence_sell += 0.15
        
        if price_vs_ema20 < 0 and price_vs_ema50 < 0 and price_vs_ema200 <= 0.001:
            confirmations_sell.append("below_emas")
            confidence_sell += 0.15
        elif price_vs_ema20 < 0:
            confirmations_sell.append("below_ema20")
            confidence_sell += 0.1
        
        if bb_percent > self.bb_mid_upper:
            confirmations_sell.append("bb_upper")
            confidence_sell += 0.1
        
        if sentiment < -self.sentiment_filter_short:
            confirmations_sell.append("negative_sentiment")
            confidence_sell += abs(sentiment) * 0.1
        
        if adx > self.adx_strong_trend:
            confirmations_sell.append("strong_trend")
            confidence_sell += 0.1
        
        # Generate signal based on confirmations
        if len(confirmations_buy) >= self.min_confirmations_trend:
            confidence = min(0.98, confidence * regime_confidence)
            metadata = self._build_trade_metadata("BUY", "TRENDING", regime_confidence, confirmations_buy, indicators, confidence, "trending_momentum")
            return Signal(action="BUY", confidence=float(confidence), metadata=metadata)
        
        if len(confirmations_sell) >= self.min_confirmations_trend:
            confidence_sell = min(0.98, confidence_sell * regime_confidence)
            metadata = self._build_trade_metadata("SELL", "TRENDING", regime_confidence, confirmations_sell, indicators, confidence_sell, "trending_momentum")
            return Signal(action="SELL", confidence=float(confidence_sell), metadata=metadata)
        
        return Signal(action="HOLD", confidence=0.0, metadata={"strategy": "trending_momentum"})
    
    def _ranging_strategy(self, indicators: Dict[str, float], regime_confidence: float) -> Signal:
        """Strategy for range-bound markets - mean reversion."""
        confirmations_buy = []
        confirmations_sell = []
        confidence = 0.5
        
        rsi = indicators["rsi"]
        bb_percent = indicators["bb_percent"]
        macd_hist = indicators["macd_hist"]
        sentiment = indicators["sentiment"]
        adx = indicators["adx"]

        if adx > self.adx_trend_threshold:
            return Signal(action="HOLD", confidence=0.0, metadata={"reason": "trend_strength"})
        
        # BUY confirmations for ranging market (mean reversion)
        if rsi < self.rsi_oversold_ranging:
            confirmations_buy.append("rsi_extreme_oversold")
            confidence += (self.rsi_oversold_ranging - rsi) / 80
        
        if bb_percent < self.bb_extreme_lower:
            confirmations_buy.append("bb_extreme_lower")
            confidence += 0.25
        elif bb_percent < 0.3:
            confirmations_buy.append("bb_lower_third")
            confidence += 0.15
        
        if macd_hist > 0 and indicators["macd"] < 0:
            confirmations_buy.append("macd_reversal")
            confidence += 0.15
        
        if sentiment > self.sentiment_filter_range:
            confirmations_buy.append("strong_positive_sentiment")
            confidence += 0.15
        
        # SELL confirmations for ranging market (mean reversion)
        confidence_sell = 0.5
        
        if rsi > self.rsi_overbought_ranging:
            confirmations_sell.append("rsi_extreme_overbought")
            confidence_sell += (rsi - self.rsi_overbought_ranging) / 80
        
        if bb_percent > self.bb_extreme_upper:
            confirmations_sell.append("bb_extreme_upper")
            confidence_sell += 0.25
        elif bb_percent > 0.7:
            confirmations_sell.append("bb_upper_third")
            confidence_sell += 0.15
        
        if macd_hist < 0 and indicators["macd"] > 0:
            confirmations_sell.append("macd_reversal")
            confidence_sell += 0.15
        
        if sentiment < -self.sentiment_filter_short:
            confirmations_sell.append("strong_negative_sentiment")
            confidence_sell += 0.15
        
        # Generate signal with lower confirmation requirement for ranging
        if len(confirmations_buy) >= self.min_confirmations_range:
            confidence = min(0.9, confidence * regime_confidence)
            metadata = self._build_trade_metadata("BUY", "RANGING", regime_confidence, confirmations_buy, indicators, confidence, "mean_reversion")
            return Signal(action="BUY", confidence=float(confidence), metadata=metadata)
        
        if len(confirmations_sell) >= self.min_confirmations_range:
            confidence_sell = min(0.9, confidence_sell * regime_confidence)
            metadata = self._build_trade_metadata("SELL", "RANGING", regime_confidence, confirmations_sell, indicators, confidence_sell, "mean_reversion")
            return Signal(action="SELL", confidence=float(confidence_sell), metadata=metadata)
        
        return Signal(action="HOLD", confidence=0.0, metadata={"strategy": "mean_reversion"})
    
    def _volatile_strategy(self, indicators: Dict[str, float], regime_confidence: float) -> Signal:
        """Conservative strategy for volatile markets - reduced trading."""
        rsi = indicators["rsi"]
        bb_percent = indicators["bb_percent"]
        adx = indicators["adx"]
        sentiment = indicators["sentiment"]

        confidence = 0.35  # Lower base confidence in volatile markets

        if rsi < 22 and bb_percent < 0.15 and sentiment > self.sentiment_filter_range:
            final_conf = min(0.75, confidence * regime_confidence)
            metadata = self._build_trade_metadata("BUY", "VOLATILE", regime_confidence, ["extreme_oversold"], indicators, final_conf, "volatile_extreme")
            metadata["reason"] = "extreme_oversold"
            return Signal(action="BUY", confidence=float(final_conf), metadata=metadata)

        if rsi > 78 and bb_percent > 0.85 and sentiment < -self.sentiment_filter_short:
            final_conf = min(0.75, confidence * regime_confidence)
            metadata = self._build_trade_metadata("SELL", "VOLATILE", regime_confidence, ["extreme_overbought"], indicators, final_conf, "volatile_extreme")
            metadata["reason"] = "extreme_overbought"
            return Signal(action="SELL", confidence=float(final_conf), metadata=metadata)

        return Signal(action="HOLD", confidence=0.0, metadata={"strategy": "volatile_conservative", "reason": "high_volatility"})

    @staticmethod
    def _session_minutes(ts_str: str) -> int:
        hour, minute = ts_str.split(":")
        return int(hour) * 60 + int(minute)
