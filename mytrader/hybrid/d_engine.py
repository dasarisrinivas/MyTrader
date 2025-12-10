"""D-Engine: Deterministic Rule-Based Trading Engine.

Fast, real-time decision engine that operates on every candle close.
Contains primary entry/exit rules, technical indicators, and breakout logic.
NO LLM calls - purely deterministic for low latency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from ..utils.logger import logger


@dataclass
class DEngineSignal:
    """Signal output from the deterministic engine."""
    
    # Core signal
    action: str  # "BUY", "SELL", "HOLD"
    is_candidate: bool  # True if passes initial filter (triggers H-engine)
    technical_score: float  # 0.0-1.0 aggregate technical score
    
    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    candle_close_time: Optional[datetime] = None
    
    # Price context
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Individual indicator scores
    indicator_scores: Dict[str, float] = field(default_factory=dict)
    
    # Level context
    near_pdh: bool = False
    near_pdl: bool = False
    near_weekly_high: bool = False
    near_weekly_low: bool = False
    
    # Trend alignment
    trend_aligned: bool = False
    ema_diff_pct: float = 0.0
    
    # Metadata
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "action": self.action,
            "is_candidate": self.is_candidate,
            "technical_score": self.technical_score,
            "timestamp": self.timestamp.isoformat(),
            "candle_close_time": self.candle_close_time.isoformat() if self.candle_close_time else None,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "indicator_scores": self.indicator_scores,
            "near_pdh": self.near_pdh,
            "near_pdl": self.near_pdl,
            "near_weekly_high": self.near_weekly_high,
            "near_weekly_low": self.near_weekly_low,
            "trend_aligned": self.trend_aligned,
            "ema_diff_pct": self.ema_diff_pct,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


class DeterministicEngine:
    """Fast deterministic trading engine for candle-close decisions.
    
    This engine:
    - Runs ONLY on candle close (explicit check)
    - Uses technical indicators (RSI, MACD, EMA, ATR)
    - Considers price levels (PDH/PDL, WH/WL)
    - Outputs candidates for H-engine confirmation
    - NEVER calls LLM - purely deterministic
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        # RSI thresholds
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0,
        "rsi_weight": 0.25,
        
        # MACD settings
        "macd_weight": 0.25,
        
        # EMA trend confirmation
        "ema_fast": 9,
        "ema_slow": 20,
        "ema_weight": 0.25,
        
        # Volatility (ATR)
        "atr_min_threshold": 0.5,
        "atr_max_threshold": 50.0,
        "atr_weight": 0.15,
        
        # Volume
        "volume_weight": 0.10,
        
        # Candidate threshold (minimum score to trigger H-engine)
        "candidate_threshold": 0.55,
        
        # Level proximity (as percentage of ATR)
        "level_proximity_atr_mult": 0.5,
        
        # Risk parameters
        "atr_stop_mult": 2.0,
        "atr_target_mult": 3.0,
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the deterministic engine.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        # Price levels (updated externally)
        self._pdh: Optional[float] = None
        self._pdl: Optional[float] = None
        self._week_high: Optional[float] = None
        self._week_low: Optional[float] = None
        
        # State tracking
        self._last_candle_processed: Optional[datetime] = None
        
        logger.info(f"DeterministicEngine initialized with config: {self.config}")
    
    def set_levels(
        self,
        pdh: Optional[float] = None,
        pdl: Optional[float] = None,
        week_high: Optional[float] = None,
        week_low: Optional[float] = None,
    ):
        """Update price levels for decision making.
        
        Args:
            pdh: Previous day high
            pdl: Previous day low
            week_high: Current week high
            week_low: Current week low
        """
        self._pdh = pdh
        self._pdl = pdl
        self._week_high = week_high
        self._week_low = week_low
        
        logger.debug(f"Levels updated: PDH={pdh}, PDL={pdl}, WH={week_high}, WL={week_low}")
    
    def is_candle_closed(self, candle_time: datetime, period_seconds: int = 60) -> bool:
        """Check if we're at a new candle boundary.
        
        Args:
            candle_time: Candle timestamp to check
            period_seconds: Candle period in seconds (default 60 for 1-min)
            
        Returns:
            True if this is a new candle close we haven't processed
        """
        # Normalize to candle boundary
        candle_boundary = candle_time.replace(second=0, microsecond=0)
        
        if self._last_candle_processed == candle_boundary:
            return False
        
        return True
    
    def evaluate(
        self,
        features: pd.DataFrame,
        current_price: float,
        candle_time: datetime,
        force: bool = False,
    ) -> DEngineSignal:
        """Evaluate market data and produce a trading signal.
        
        This method should ONLY be called at candle close.
        
        Args:
            features: DataFrame with OHLCV and indicators
            current_price: Current market price
            candle_time: Timestamp of the candle
            force: If True, skip candle close check (for testing)
            
        Returns:
            DEngineSignal with action, score, and metadata
        """
        # Candle close validation
        if not force and not self.is_candle_closed(candle_time):
            return DEngineSignal(
                action="HOLD",
                is_candidate=False,
                technical_score=0.0,
                reasons=["Not at candle close"],
            )
        
        # Mark this candle as processed
        candle_boundary = candle_time.replace(second=0, microsecond=0)
        self._last_candle_processed = candle_boundary
        
        if len(features) < 20:
            return DEngineSignal(
                action="HOLD",
                is_candidate=False,
                technical_score=0.0,
                reasons=["Insufficient data"],
            )
        
        latest = features.iloc[-1]
        reasons = []
        
        # Calculate individual indicator scores
        scores = {}
        
        # 1. RSI Score
        rsi = latest.get("RSI_14", 50.0)
        rsi_score = self._calculate_rsi_score(rsi)
        scores["rsi"] = rsi_score
        
        # 2. MACD Score
        macd = latest.get("MACD", 0.0)
        macd_signal = latest.get("MACD_signal", 0.0)
        macd_hist = latest.get("MACD_hist", macd - macd_signal)
        macd_score, macd_direction = self._calculate_macd_score(macd, macd_signal, macd_hist)
        scores["macd"] = macd_score
        
        # 3. EMA Trend Score
        ema_fast = latest.get(f"EMA_{self.config['ema_fast']}", latest.get("EMA_9", None))
        ema_slow = latest.get(f"EMA_{self.config['ema_slow']}", latest.get("EMA_20", None))
        ema_score, trend_direction, ema_diff_pct = self._calculate_ema_score(ema_fast, ema_slow)
        scores["ema"] = ema_score
        
        # 4. ATR / Volatility Score
        atr = latest.get("ATR_14", 0.0)
        atr_score, vol_ok = self._calculate_atr_score(atr, current_price)
        scores["atr"] = atr_score
        
        if not vol_ok:
            reasons.append(f"Volatility out of range: ATR={atr:.2f}")
        
        # 5. Volume Score
        volume = latest.get("volume", 0)
        avg_volume = features["volume"].rolling(20).mean().iloc[-1] if "volume" in features.columns else 0
        volume_score = self._calculate_volume_score(volume, avg_volume)
        scores["volume"] = volume_score
        
        # Calculate weighted technical score
        technical_score = (
            scores["rsi"] * self.config["rsi_weight"] +
            scores["macd"] * self.config["macd_weight"] +
            scores["ema"] * self.config["ema_weight"] +
            scores["atr"] * self.config["atr_weight"] +
            scores["volume"] * self.config["volume_weight"]
        )
        
        # Determine action based on directional bias
        if macd_direction > 0 and trend_direction > 0:
            proposed_action = "BUY"
        elif macd_direction < 0 and trend_direction < 0:
            proposed_action = "SELL"
        else:
            proposed_action = "HOLD"
            reasons.append("Mixed signals - no clear direction")
        
        # Check if this is a candidate for H-engine
        is_candidate = (
            technical_score >= self.config["candidate_threshold"] and
            vol_ok and
            proposed_action != "HOLD"
        )
        
        # Calculate stop loss and take profit
        stop_loss = 0.0
        take_profit = 0.0
        if atr > 0 and proposed_action != "HOLD":
            if proposed_action == "BUY":
                stop_loss = current_price - (atr * self.config["atr_stop_mult"])
                take_profit = current_price + (atr * self.config["atr_target_mult"])
            else:  # SELL
                stop_loss = current_price + (atr * self.config["atr_stop_mult"])
                take_profit = current_price - (atr * self.config["atr_target_mult"])
        
        # Check level proximity
        near_pdh, near_pdl = self._check_level_proximity(current_price, atr)
        near_wh, near_wl = self._check_weekly_proximity(current_price, atr)
        
        # Build reasons list
        if is_candidate:
            reasons.append(f"Candidate signal: {proposed_action} (score={technical_score:.3f})")
        reasons.append(f"RSI={rsi:.1f}, MACD_hist={macd_hist:.4f}, EMA_diff={ema_diff_pct:.2f}%")
        
        return DEngineSignal(
            action=proposed_action,
            is_candidate=is_candidate,
            technical_score=technical_score,
            candle_close_time=candle_boundary,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            indicator_scores=scores,
            near_pdh=near_pdh,
            near_pdl=near_pdl,
            near_weekly_high=near_wh,
            near_weekly_low=near_wl,
            trend_aligned=(proposed_action == "BUY" and trend_direction > 0) or 
                         (proposed_action == "SELL" and trend_direction < 0),
            ema_diff_pct=ema_diff_pct,
            reasons=reasons,
            metadata={
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_hist": macd_hist,
                "atr": atr,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
            }
        )
    
    def _calculate_rsi_score(self, rsi: float) -> float:
        """Calculate RSI contribution to technical score.
        
        Higher score for oversold (buy) or overbought (sell) conditions.
        """
        if rsi < self.config["rsi_oversold"]:
            # Oversold - bullish signal
            return min(1.0, (self.config["rsi_oversold"] - rsi) / 20 + 0.5)
        elif rsi > self.config["rsi_overbought"]:
            # Overbought - bearish signal
            return min(1.0, (rsi - self.config["rsi_overbought"]) / 20 + 0.5)
        else:
            # Neutral zone
            return 0.3
    
    def _calculate_macd_score(self, macd: float, signal: float, hist: float) -> tuple[float, int]:
        """Calculate MACD contribution and direction.
        
        Returns:
            Tuple of (score, direction) where direction is +1, -1, or 0
        """
        # Direction based on histogram
        if hist > 0:
            direction = 1
        elif hist < 0:
            direction = -1
        else:
            direction = 0
        
        # Score based on histogram strength
        hist_abs = abs(hist)
        if hist_abs > 0.5:
            score = 0.8
        elif hist_abs > 0.2:
            score = 0.6
        elif hist_abs > 0.05:
            score = 0.4
        else:
            score = 0.2
        
        # Bonus for MACD line crossing signal
        if direction > 0 and macd > signal:
            score = min(1.0, score + 0.1)
        elif direction < 0 and macd < signal:
            score = min(1.0, score + 0.1)
        
        return score, direction
    
    def _calculate_ema_score(
        self, 
        ema_fast: Optional[float], 
        ema_slow: Optional[float]
    ) -> tuple[float, int, float]:
        """Calculate EMA trend score and direction.
        
        Returns:
            Tuple of (score, direction, ema_diff_pct)
        """
        if ema_fast is None or ema_slow is None or ema_slow == 0:
            return 0.5, 0, 0.0
        
        ema_diff_pct = ((ema_fast - ema_slow) / ema_slow) * 100
        
        if ema_fast > ema_slow:
            direction = 1
            # Stronger trend = higher score
            if ema_diff_pct > 0.5:
                score = 0.9
            elif ema_diff_pct > 0.2:
                score = 0.7
            else:
                score = 0.5
        elif ema_fast < ema_slow:
            direction = -1
            if ema_diff_pct < -0.5:
                score = 0.9
            elif ema_diff_pct < -0.2:
                score = 0.7
            else:
                score = 0.5
        else:
            direction = 0
            score = 0.3
        
        return score, direction, ema_diff_pct
    
    def _calculate_atr_score(self, atr: float, price: float) -> tuple[float, bool]:
        """Calculate ATR score and check if within acceptable range.
        
        Returns:
            Tuple of (score, is_valid)
        """
        if atr < self.config["atr_min_threshold"]:
            return 0.2, False
        
        if atr > self.config["atr_max_threshold"]:
            return 0.2, False
        
        # ATR as percentage of price
        atr_pct = (atr / price) * 100 if price > 0 else 0
        
        # Ideal range: 0.1% to 0.5%
        if 0.1 <= atr_pct <= 0.5:
            score = 0.8
        elif 0.05 <= atr_pct <= 0.8:
            score = 0.6
        else:
            score = 0.4
        
        return score, True
    
    def _calculate_volume_score(self, volume: float, avg_volume: float) -> float:
        """Calculate volume score relative to average."""
        if avg_volume == 0:
            return 0.5
        
        vol_ratio = volume / avg_volume
        
        if vol_ratio > 2.0:
            return 0.9  # High volume confirmation
        elif vol_ratio > 1.5:
            return 0.7
        elif vol_ratio > 0.8:
            return 0.5
        else:
            return 0.3  # Low volume - less conviction
    
    def _check_level_proximity(
        self, 
        price: float, 
        atr: float
    ) -> tuple[bool, bool]:
        """Check proximity to PDH/PDL levels."""
        if atr <= 0:
            return False, False
        
        proximity = atr * self.config["level_proximity_atr_mult"]
        
        near_pdh = (
            self._pdh is not None and 
            abs(price - self._pdh) < proximity
        )
        near_pdl = (
            self._pdl is not None and 
            abs(price - self._pdl) < proximity
        )
        
        return near_pdh, near_pdl
    
    def _check_weekly_proximity(
        self, 
        price: float, 
        atr: float
    ) -> tuple[bool, bool]:
        """Check proximity to weekly high/low levels."""
        if atr <= 0:
            return False, False
        
        proximity = atr * self.config["level_proximity_atr_mult"]
        
        near_wh = (
            self._week_high is not None and 
            abs(price - self._week_high) < proximity
        )
        near_wl = (
            self._week_low is not None and 
            abs(price - self._week_low) < proximity
        )
        
        return near_wh, near_wl
