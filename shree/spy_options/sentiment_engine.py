"""IB-sourced sentiment scorer for SPY options signal bot.

Produces a single score from -100 (max bearish) to +100 (max bullish)
using only data already available from IB Gateway — no external APIs needed.

Three inputs:
  1. VIX trend: rising VIX = bearish, falling = bullish (40% weight)
  2. SPY vs VWAP: normalized by ATR (35% weight)
  3. EMA slope: normalized by ATR (25% weight)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .regime_detector import RegimeContext


@dataclass
class SentimentContext:
    """Output of SentimentEngine.score()."""

    score: float          # -100 (bearish) to +100 (bullish)
    vix_trend: str        # "RISING" | "FALLING" | "FLAT"
    spy_vs_vwap: float    # raw points above/below VWAP
    ema_slope: float      # raw EMA-9 slope
    label: str            # "BEARISH" | "NEUTRAL" | "BULLISH"


class SentimentEngine:
    """Stateless IB-based sentiment scorer.

    Call score() once per poll. All inputs come from data already fetched
    (VIX history maintained by manager, regime from RegimeDetector).
    """

    def score(
        self,
        vix: float,
        vix_history: List[float],
        regime: RegimeContext,
    ) -> SentimentContext:
        """Compute sentiment score from VIX trend, VWAP position, and EMA slope.

        Args:
            vix: Current VIX level.
            vix_history: Recent VIX readings in chronological order (newest last).
                         Maintained as a deque(maxlen=10) in the manager.
            regime: RegimeContext from RegimeDetector.classify().

        Returns:
            SentimentContext with score, trend, and label.
        """
        atr = regime.atr14 if regime.atr14 > 0 else 1.0

        # ── Component 1: VIX trend (40 pts) ──────────────────────────────────
        vix_score = 0.0
        vix_trend = "FLAT"

        if len(vix_history) >= 3:
            # Compare current VIX to rolling 5-reading average
            window = vix_history[-5:] if len(vix_history) >= 5 else vix_history
            avg_vix = sum(window) / len(window)
            pct_change = (vix - avg_vix) / avg_vix if avg_vix > 0 else 0.0

            if pct_change > 0.05:      # VIX risen >5% above recent avg = bearish
                vix_score = -40.0
                vix_trend = "RISING"
            elif pct_change < -0.05:   # VIX fallen >5% below recent avg = bullish
                vix_score = +40.0
                vix_trend = "FALLING"
            else:
                # Proportional in the flat zone
                vix_score = -pct_change / 0.05 * 40.0
                vix_trend = "FLAT"

        # ── Component 2: SPY vs VWAP (35 pts) ────────────────────────────────
        # Normalize deviation by ATR so a 0.5-point gap in a 0.3-ATR day matters more
        vwap_score = 0.0
        if regime.vwap > 0:
            normalized = regime.spy_vs_vwap / atr
            vwap_score = max(-35.0, min(35.0, normalized * 35.0))

        # ── Component 3: EMA slope (25 pts) ──────────────────────────────────
        slope_score = 0.0
        if atr > 0:
            normalized_slope = regime.ema_slope / atr
            slope_score = max(-25.0, min(25.0, normalized_slope * 250.0))

        raw_score = vix_score + vwap_score + slope_score
        # Clamp to [-100, +100]
        final_score = max(-100.0, min(100.0, raw_score))

        if final_score > 25.0:
            label = "BULLISH"
        elif final_score < -25.0:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        return SentimentContext(
            score=final_score,
            vix_trend=vix_trend,
            spy_vs_vwap=regime.spy_vs_vwap,
            ema_slope=regime.ema_slope,
            label=label,
        )
