"""Trend-aware RSI pullback filter used by the deterministic engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class RSIPullbackConfig:
    """Configurable thresholds for trend-aware RSI logic."""

    rsi_period: int = 14
    ema_period: int = 50
    atr_period: int = 14
    rsi_long_pullback_low: float = 30.0
    rsi_long_pullback_reentry: float = 40.0
    rsi_short_pullback_high: float = 70.0
    rsi_short_pullback_reentry: float = 60.0
    ema_no_trade_band_pct: float = 0.0005  # +/-0.05% buffer around EMA to avoid chop
    atr_multiplier_sl: float = 1.2
    atr_multiplier_tp: float = 1.0
    atr_min_threshold: float = 0.0
    atr_max_threshold: float = 50.0
    enable_session_adjustments: bool = True
    overnight_rsi_buffer: float = 5.0
    overnight_tp_multiplier: float = 0.85
    overnight_sl_multiplier: float = 0.9
    session_rth_start: str = "08:30"
    session_rth_end: str = "15:15"


@dataclass
class RSITrendPullbackResult:
    """Result produced by the pullback filter."""

    trend: str = "FLAT"
    session: str = "UNKNOWN"
    rsi: float = 50.0
    prev_rsi: float = 50.0
    atr: float = 0.0
    atr_ok: bool = False
    in_no_trade_band: bool = False
    pullback_detected: bool = False
    pullback_reentry: bool = False
    allow_long: bool = False
    allow_short: bool = False
    stop_loss_mult: float = 0.0
    take_profit_mult: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_metadata(self) -> Dict[str, float | str | bool]:
        return {
            "trend": self.trend,
            "session": self.session,
            "rsi": self.rsi,
            "prev_rsi": self.prev_rsi,
            "atr": self.atr,
            "atr_ok": self.atr_ok,
            "in_no_trade_band": self.in_no_trade_band,
            "pullback_detected": self.pullback_detected,
            "pullback_reentry": self.pullback_reentry,
        }


def _parse_session_time(ts: str) -> int:
    hours, minutes = ts.split(":")
    return int(hours) * 60 + int(minutes)


def session_from_time(
    now: Optional[datetime] = None,
    rth_start: str = "08:30",
    rth_end: str = "15:15",
) -> str:
    """Return 'RTH' or 'OVERNIGHT' based on CST clock."""
    now = now or datetime.utcnow()
    minutes = now.hour * 60 + now.minute
    start = _parse_session_time(rth_start)
    end = _parse_session_time(rth_end)
    return "RTH" if start <= minutes <= end else "OVERNIGHT"


class RSITrendPullbackFilter:
    """RSI pullback confirmation aligned with EMA trend and ATR sanity checks."""

    def __init__(self, config: Optional[Dict[str, float]] = None):
        cfg = RSIPullbackConfig(**(config or {}))
        self.config = cfg
        self.rsi_col = f"RSI_{cfg.rsi_period}"
        self.ema_col = f"EMA_{cfg.ema_period}"
        self.atr_col = f"ATR_{cfg.atr_period}"

    def evaluate(self, features: pd.DataFrame, now: Optional[datetime] = None) -> RSITrendPullbackResult:
        """Evaluate latest candle context and return pullback decision."""
        result = RSITrendPullbackResult()
        if features is None or len(features) < 2:
            result.reasons.append("Insufficient history for RSI pullback")
            return result

        latest = features.iloc[-1]
        prev = features.iloc[-2]
        close = float(latest.get("close", 0.0))
        ema_trend = float(latest.get(self.ema_col, close))
        atr = float(latest.get(self.atr_col, latest.get("ATR_14", 0.0)))
        rsi = float(latest.get(self.rsi_col, 50.0))
        prev_rsi = float(prev.get(self.rsi_col, rsi))

        result.rsi = rsi
        result.prev_rsi = prev_rsi
        result.atr = atr
        result.session = session_from_time(now, self.config.session_rth_start, self.config.session_rth_end)

        # Trend assessment relative to EMA
        if close > ema_trend * (1 + self.config.ema_no_trade_band_pct):
            result.trend = "UPTREND"
        elif close < ema_trend * (1 - self.config.ema_no_trade_band_pct):
            result.trend = "DOWNTREND"
        else:
            result.trend = "FLAT"
            result.in_no_trade_band = True
            result.reasons.append("Price within EMA no-trade band")

        # ATR sanity
        result.atr_ok = (
            atr > 0
            and atr >= self.config.atr_min_threshold
            and atr <= self.config.atr_max_threshold
        )
        if not result.atr_ok:
            result.reasons.append(f"ATR out of bounds ({atr:.2f})")

        # Session-aware buffers
        long_low = self.config.rsi_long_pullback_low
        long_reentry = self.config.rsi_long_pullback_reentry
        short_high = self.config.rsi_short_pullback_high
        short_reentry = self.config.rsi_short_pullback_reentry
        sl_mult = self.config.atr_multiplier_sl
        tp_mult = self.config.atr_multiplier_tp

        if self.config.enable_session_adjustments and result.session == "OVERNIGHT":
            long_low = max(0.0, long_low - self.config.overnight_rsi_buffer)
            long_reentry = max(0.0, long_reentry - self.config.overnight_rsi_buffer)
            short_high = min(100.0, short_high + self.config.overnight_rsi_buffer)
            short_reentry = min(100.0, short_reentry + self.config.overnight_rsi_buffer)
            tp_mult *= self.config.overnight_tp_multiplier
            sl_mult *= self.config.overnight_sl_multiplier
            result.reasons.append("Overnight session adjustments applied")

        # Pullback detection
        result.pullback_detected = (
            (result.trend == "UPTREND" and rsi <= long_low)
            or (result.trend == "DOWNTREND" and rsi >= short_high)
        )

        long_reentry_hit = prev_rsi <= long_low and rsi >= long_reentry
        short_reentry_hit = prev_rsi >= short_high and rsi <= short_reentry
        result.pullback_reentry = long_reentry_hit or short_reentry_hit

        result.allow_long = (
            result.trend == "UPTREND"
            and result.pullback_reentry
            and not result.in_no_trade_band
            and result.atr_ok
        )
        result.allow_short = (
            result.trend == "DOWNTREND"
            and result.pullback_reentry
            and not result.in_no_trade_band
            and result.atr_ok
        )

        result.stop_loss_mult = sl_mult
        result.take_profit_mult = tp_mult

        if result.pullback_reentry:
            direction = "long" if result.allow_long else "short" if result.allow_short else "undecided"
            result.reasons.append(
                f"RSI pullback re-entry confirmed for {direction} (RSI {prev_rsi:.1f}->{rsi:.1f})"
            )

        return result
