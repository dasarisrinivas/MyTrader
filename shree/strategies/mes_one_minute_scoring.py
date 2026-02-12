"""MES 1-Minute Strategy with Scoring-Based Entry System.

FEB 2026 REFACTOR: Scoring-Based Entry System
This is the refactored version that replaces hard rejection filters
with a weighted scoring system to increase trade frequency while 
preserving edge through intelligent condition weighting.

KEY CHANGES:
- Hard filters (ADX, regime, structure, etc.) → Weighted scoring
- All conditions contribute points, not binary pass/fail
- Position sizing based on score thresholds (full/half/none)
- Risk containment remains as HARD gates (non-negotiable)
- Increased trade frequency with preserved/improved edge

CORE PHILOSOPHY:
- Score >= 60: Full position (1.0x)
- Score >= 45: Half position (0.5x) 
- Score < 45: No trade
- Risk gates (max loss, daily loss, open risk) remain HARD

Author: Senior Quantitative Trading Engineer - Feb 2026
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
from ..utils.structured_logging import log_structured_event
from .base import BaseStrategy, Signal
from .scoring_integration import create_scoring_evaluator, ScoringDecision


@dataclass
class StrategyDecision:
    action: str
    confidence: float
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, float]] = None


class MesOneMinuteScoringStrategy(BaseStrategy):
    """MES 1-minute strategy using scoring-based entry system.
    
    FEB 2026 REFACTOR:
    - Replaced hard rejection filters with weighted scoring
    - Each condition adds/subtracts points instead of blocking
    - Position sizing based on total score (full/half/none)
    - Risk containment as HARD gates (unchanged)
    - Dramatically increased trade frequency
    
    Scoring Categories:
    - Trend/Structure: max +40 points
    - Momentum: max +25 points
    - Volatility/Regime: max +20 points
    - Entry Quality: max +20 points
    - Penalties: negative points for weak conditions
    """

    name = "mes_one_minute_scoring"

    def __init__(self, config: OneMinuteStrategyConfig):
        self.config = config
        self._trade_log: list[pd.Timestamp] = []
        self._hourly_log: list[pd.Timestamp] = []
        
        # Initialize scoring evaluator
        self._scoring_evaluator = create_scoring_evaluator(config)
        
        # Track previous bar for slope calculations
        self._prev_bar_data: Optional[pd.Series] = None
        
        # Risk tracking for hard gates
        self._daily_pnl = 0.0
        self._open_risk = 0.0

    def _is_rth(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within RTH (Regular Trading Hours)."""
        if not self.config.rth_only:
            return True
            
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

    def _is_in_rth_no_trade_window(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp falls within an optional RTH no-trade window."""
        start_hour = self.config.rth_no_trade_start_hour
        start_minute = self.config.rth_no_trade_start_minute
        end_hour = self.config.rth_no_trade_end_hour
        end_minute = self.config.rth_no_trade_end_minute

        if None in (start_hour, start_minute, end_hour, end_minute):
            return False

        if timestamp.tz is not None:
            try:
                local_ts = timestamp.tz_convert("America/Chicago")
            except Exception:
                local_ts = timestamp
        else:
            local_ts = timestamp

        t = local_ts.time() if hasattr(local_ts, "time") else time(0, 0)
        window_start = time(int(start_hour), int(start_minute))
        window_end = time(int(end_hour), int(end_minute))

        if window_start < window_end:
            return window_start <= t < window_end

        return t >= window_start or t < window_end

    def _get_session_type(self, timestamp: pd.Timestamp) -> str:
        """Return 'RTH', 'OVERNIGHT', or 'CLOSED' for the current timestamp."""
        if self._is_rth(timestamp):
            return "RTH"
        # For simplicity, treat non-RTH as closed (can add overnight logic if needed)
        return "CLOSED"

    def generate(self, features: pd.DataFrame) -> Signal:
        """Generate trading signal using scoring-based entry system.
        
        This is the main entry point that replaces hard filter logic
        with the scoring system while maintaining risk gates.
        """
        window = features.tail(max(self.config.window_bars, 120)).copy()
        if len(window) < max(60, self.config.warmup_bars):
            return Signal("HOLD", 0.0, {"reason": "WARMUP"})

        enriched = self._ensure_indicators(window)
        latest = enriched.iloc[-1]
        prev = enriched.iloc[-2] if len(enriched) >= 2 else None
        current_time = enriched.index[-1]
        
        # Get session type
        session_type = self._get_session_type(current_time)
        
        if session_type == "CLOSED":
            return Signal("HOLD", 0.0, {"reason": "MARKET_CLOSED"})
        
        if self._is_in_rth_no_trade_window(current_time):
            return Signal("HOLD", 0.0, {"reason": "RTH_NO_TRADE_WINDOW"})
        
        # Extract key values for metadata
        atr_value = float(latest.get("ATR_14", 1.0))
        adx_value = float(latest.get("ADX_14", 0))
        
        # Use 15m indicators if configured
        use_mtf = getattr(self.config, 'use_mtf_regime', False)
        if use_mtf:
            if "15m_ATR_14" in latest and not np.isnan(latest["15m_ATR_14"]):
                atr_value = float(latest["15m_ATR_14"])
            if "15m_ADX_14" in latest and not np.isnan(latest["15m_ADX_14"]):
                adx_value = float(latest["15m_ADX_14"])
        
        # Classify trend for metadata
        trend_label = self._classify_trend(latest, adx_value)
        market_state = "TRENDING" if adx_value >= self.config.trend_adx_threshold else "RANGING"
        
        # Get 15m regime if available
        regime_15m = str(latest.get("15m_regime", "UNKNOWN"))
        
        # Build metadata for scoring evaluator
        metadata: Dict[str, Any] = {
            "market_state": market_state,
            "trend_label": trend_label,
            "atr_value": atr_value,
            "adx_value": adx_value,
            "ema9": float(latest.get("EMA_9", 0)),
            "ema21": float(latest.get("EMA_21", 0)),
            "vwap": float(latest.get("SESSION_VWAP", 0)),
            "pdh": float(latest.get("PDH", np.nan)),
            "pdl": float(latest.get("PDL", np.nan)),
            "session_type": session_type,
            "15m_regime": regime_15m,
        }
        
        # =========================================================================
        # FEB 2026: SCORING-BASED ENTRY EVALUATION
        # =========================================================================
        # Use scoring evaluator instead of hard filters
        # The evaluator will:
        # 1. Calculate weighted scores across all conditions
        # 2. Determine position sizing (full/half/none)
        # 3. Check hard risk gates
        # 4. Return entry decision with stops/targets
        # =========================================================================
        
        recent_bars = enriched.tail(20)  # Last 20 bars for momentum/pullback analysis
        
        scoring_decision = self._scoring_evaluator.evaluate(
            latest=latest,
            prev=prev,
            recent_bars=recent_bars,
            current_time=current_time,
            metadata=metadata,
            daily_pnl=self._daily_pnl,
            open_risk=self._open_risk
        )
        
        # Convert scoring decision to strategy decision
        decision = StrategyDecision(
            action=scoring_decision.action,
            confidence=scoring_decision.confidence,
            reason=scoring_decision.reason,
            stop_loss=scoring_decision.stop_loss,
            take_profit=scoring_decision.take_profit,
            metadata=scoring_decision.metadata
        )
        
        # Store previous bar for next iteration
        self._prev_bar_data = prev
        
        # Log decision
        self._log_decision(current_time, decision, latest)
        
        # Build return signal
        meta = decision.metadata or {}
        meta.setdefault("reason", decision.reason)
        meta.setdefault("strategy_type", self.name)
        meta["market_state"] = market_state
        meta["trend_label"] = trend_label
        
        # Add position sizing and risk brackets to metadata
        if decision.action in ("BUY", "SELL"):
            meta["position_size"] = scoring_decision.position_size
            meta["stop_loss"] = decision.stop_loss  # ✅ Add stop loss
            meta["take_profit"] = decision.take_profit  # ✅ Add take profit
        
        return Signal(action=decision.action, confidence=decision.confidence, metadata=meta)

    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required indicators are present."""
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
        if "EMA_50" not in enriched:
            enriched["EMA_50"] = _ema(close, 50)
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
        
        # Compute MACD histogram if not present
        if "MACD_hist" not in enriched:
            enriched["MACD_hist"] = self._compute_macd_hist(close)
        
        return enriched

    def _compute_macd_hist(self, close: pd.Series) -> pd.Series:
        """Compute MACD histogram."""
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd_line = ema12 - ema26
        signal_line = _ema(macd_line, 9)
        return macd_line - signal_line

    def _compute_session_vwap(self, df: pd.DataFrame, use_eth: bool) -> pd.Series:
        """Compute session VWAP."""
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
            inactive = mask_idx & (~active_mask)
            if inactive.any() and not cum_vwap.empty:
                vwap.loc[inactive] = cum_vwap.iloc[-1]

        return vwap.ffill()

    def _compute_previous_day_levels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Compute previous day high/low levels."""
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

    def _classify_trend(self, latest: pd.Series, adx_value: float) -> str:
        """Classify trend direction (used for metadata)."""
        threshold = self.config.trend_adx_threshold
        market_state = "TRENDING" if adx_value >= threshold else "RANGING"
        if market_state != "TRENDING":
            return "CHOP"
        
        close = float(latest.get("close", 0))
        vwap = float(latest.get("SESSION_VWAP", close))
        ema9 = float(latest.get("EMA_9", close))
        ema21 = float(latest.get("EMA_21", close))
        
        if close > vwap and ema9 > ema21:
            return "UPTREND"
        if close < vwap and ema9 < ema21:
            return "DOWNTREND"
        return "CHOP"

    def _log_decision(self, ts: pd.Timestamp, decision: StrategyDecision, latest: pd.Series) -> None:
        """Log trading decision with comprehensive diagnostics."""
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
        
        # Add scoring-specific fields
        if decision.metadata:
            if "signal_score" in decision.metadata:
                payload["signal_score"] = decision.metadata["signal_score"]
            if "position_size" in decision.metadata:
                payload["position_size"] = decision.metadata["position_size"]
            if "score_breakdown" in decision.metadata:
                payload["score_breakdown"] = decision.metadata["score_breakdown"]
        
        logger.info(
            f"🕐 {ts} | state={payload['state']} trend={payload['trend']} act={decision.action} conf={decision.confidence:.2f} "
            f"reason={decision.reason} close={payload['close']:.2f} ema9={payload['ema9']:.2f} "
            f"ema21={payload['ema21']:.2f} vwap={payload['vwap']:.2f} atr={payload['atr']:.2f} adx={payload['adx']:.2f} "
            f"score={payload.get('signal_score', 'N/A')}"
        )
        log_structured_event(
            agent="mes_one_minute_scoring",
            event_type="decision",
            message=decision.reason,
            payload=payload,
        )
