"""MES 1-minute close strategy with minimal indicator stack."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..config import OneMinuteStrategyConfig
from ..features.feature_engineer import _adx, _atr, _ema, _rsi
from ..utils.structured_logging import log_structured_event
from .base import BaseStrategy, Signal


@dataclass
class StrategyDecision:
    action: str
    confidence: float
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, float]] = None


class MesOneMinuteTrendStrategy(BaseStrategy):
    """Single-instrument MES strategy evaluated strictly on 1-minute closes."""

    name = "mes_one_minute_trend"

    def __init__(self, config: OneMinuteStrategyConfig):
        self.config = config
        self._trade_log: list[pd.Timestamp] = []
        self._hourly_log: list[pd.Timestamp] = []

    def generate(self, features: pd.DataFrame) -> Signal:
        window = features.tail(max(self.config.window_bars, 120)).copy()
        if len(window) < max(60, self.config.warmup_bars):
            return Signal("HOLD", 0.0, {"reason": "WARMUP"})

        enriched = self._ensure_indicators(window)
        latest = enriched.iloc[-1]
        prev = enriched.iloc[-2]
        atr_series = enriched["ATR_14"].tail(120).dropna()
        atr_value = float(latest["ATR_14"])
        adx_value = float(latest["ADX_14"])

        market_state = "TRENDING" if adx_value >= self.config.trend_adx_threshold else "RANGING"
        trend_label = self._classify_trend(latest)
        atr_low, atr_high = self._atr_percentile_bounds(atr_series)
        candle_range = float(latest["high"] - latest["low"])

        reasons: list[str] = []
        filters_block = False

        if atr_value <= 0 or np.isnan(atr_value):
            filters_block = True
            reasons.append("ATR_INVALID")
        elif atr_value < atr_low:
            filters_block = True
            reasons.append("ATR_TOO_LOW")
        elif atr_value > atr_high:
            filters_block = True
            reasons.append("ATR_TOO_HIGH")

        if candle_range < self.config.tiny_candle_atr_factor * atr_value:
            filters_block = True
            reasons.append("TINY_CANDLE")

        metadata: Dict[str, float | str] = {
            "market_state": market_state,
            "trend_label": trend_label,
            "atr_value": atr_value,
            "adx_value": adx_value,
            "ema9": float(latest["EMA_9"]),
            "ema21": float(latest["EMA_21"]),
            "vwap": float(latest["SESSION_VWAP"]),
            "pdh": float(latest.get("PDH", np.nan)),
            "pdl": float(latest.get("PDL", np.nan)),
            "candle_range": candle_range,
        }

        decision = StrategyDecision(action="HOLD", confidence=0.0, reason="INIT", metadata=metadata)

        if filters_block or trend_label == "CHOP":
            decision = StrategyDecision("HOLD", 0.0, ",".join(reasons) or "CHOP", metadata=metadata)
        else:
            pullback_ok, pull_reason = self._pullback_confirmation(enriched)
            breakout_decision = (
                self._breakout_check(latest, adx_value, metadata)
                if self.config.breakout_enabled
                else None
            )

            if breakout_decision:
                decision = breakout_decision
                decision.metadata.update(metadata)
            elif trend_label == "UPTREND" and pullback_ok:
                decision = self._enter_with_brackets(
                    direction="BUY",
                    close=float(latest["close"]),
                    atr=atr_value,
                    reason=pull_reason or "PULLBACK_LONG",
                    base_conf=0.68,
                    extra_meta=metadata,
                )
            elif trend_label == "DOWNTREND" and pullback_ok:
                decision = self._enter_with_brackets(
                    direction="SELL",
                    close=float(latest["close"]),
                    atr=atr_value,
                    reason=pull_reason or "PULLBACK_SHORT",
                    base_conf=0.68,
                    extra_meta=metadata,
                )
            else:
                decision = StrategyDecision("HOLD", 0.0, "NO_SETUP", metadata=metadata)

        self._log_decision(enriched.index[-1], decision, latest)
        meta = decision.metadata or {}
        meta.setdefault("reason", decision.reason)
        meta.setdefault("strategy_type", self.name)
        meta["market_state"] = market_state
        meta["trend_label"] = trend_label

        return Signal(action=decision.action, confidence=decision.confidence, metadata=meta)

    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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
        return enriched

    def _compute_session_vwap(self, df: pd.DataFrame, use_eth: bool) -> pd.Series:
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
            # Forward-fill non-active bars within the same session with last active value
            inactive = mask_idx & (~active_mask)
            if inactive.any() and not cum_vwap.empty:
                vwap.loc[inactive] = cum_vwap.iloc[-1]

        return vwap.ffill()

    def _compute_previous_day_levels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
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

    def _classify_trend(self, latest) -> str:
        adx = float(latest["ADX_14"])
        market_state = "TRENDING" if adx >= self.config.trend_adx_threshold else "RANGING"
        if market_state != "TRENDING":
            return "CHOP"
        close = float(latest["close"])
        vwap = float(latest["SESSION_VWAP"])
        ema9 = float(latest["EMA_9"])
        ema21 = float(latest["EMA_21"])
        if close > vwap and ema9 > ema21 and close > ema9:
            return "UPTREND"
        if close < vwap and ema9 < ema21 and close < ema9:
            return "DOWNTREND"
        return "CHOP"

    def _atr_percentile_bounds(self, atr_series: pd.Series) -> Tuple[float, float]:
        if atr_series.empty:
            return 0.0, float("inf")
        low = float(atr_series.quantile(self.config.atr_percentile_low))
        high = float(atr_series.quantile(self.config.atr_percentile_high))
        return low, high

    def _pullback_confirmation(self, df: pd.DataFrame) -> Tuple[bool, str]:
        if len(df) < 5:
            return False, "INSUFFICIENT_HISTORY"
        recent = df.tail(max(4, self.config.pullback_lookback + 1))
        rsi_now = float(recent["RSI_14"].iloc[-1])
        rsi_prev = float(recent["RSI_14"].iloc[-2])
        close_now = float(recent["close"].iloc[-1])
        close_prev = float(recent["close"].iloc[-2])
        ema9_now = float(recent["EMA_9"].iloc[-1])
        ema9_prev = float(recent["EMA_9"].iloc[-2])

        # RSI pullback long/short windows
        if 40 <= rsi_now <= 55 and rsi_now > rsi_prev:
            return True, "RSI_PULLBACK_LONG"
        if 45 <= rsi_now <= 60 and rsi_now < rsi_prev:
            return True, "RSI_PULLBACK_SHORT"

        # EMA reclaim across last few bars
        reclaim_window = recent.tail(self.config.pullback_lookback + 1)
        closes = reclaim_window["close"].values
        ema9_vals = reclaim_window["EMA_9"].values
        if len(closes) >= 2:
            prev_below = closes[-2] < ema9_vals[-2]
            now_above = closes[-1] > ema9_vals[-1]
            prev_above = closes[-2] > ema9_vals[-2]
            now_below = closes[-1] < ema9_vals[-1]
            if prev_below and now_above:
                return True, "EMA_RECLAIM_LONG"
            if prev_above and now_below:
                return True, "EMA_RECLAIM_SHORT"
        if close_prev < ema9_prev and close_now > ema9_now:
            return True, "EMA_RECLAIM_LONG"
        if close_prev > ema9_prev and close_now < ema9_now:
            return True, "EMA_RECLAIM_SHORT"
        return False, "NO_PULLBACK"

    def _enter_with_brackets(
        self,
        direction: str,
        close: float,
        atr: float,
        reason: str,
        base_conf: float,
        extra_meta: Dict[str, float | str],
    ) -> StrategyDecision:
        stop_dist = atr * self.config.stop_atr_multiplier
        take_profit_dist = stop_dist * self.config.take_profit_multiple
        if stop_dist <= 0 or take_profit_dist <= 0:
            return StrategyDecision("HOLD", 0.0, "INVALID_STOPS", metadata=extra_meta)
        if direction == "BUY":
            stop_loss = close - stop_dist
            take_profit = close + take_profit_dist
        else:
            stop_loss = close + stop_dist
            take_profit = close - take_profit_dist

        if (direction == "BUY" and (stop_loss >= close or take_profit <= close)) or (
            direction == "SELL" and (stop_loss <= close or take_profit >= close)
        ):
            return StrategyDecision("HOLD", 0.0, "STOP_TP_INVALID", metadata=extra_meta)

        meta = dict(extra_meta)
        meta.update(
            {
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr_value": atr,
                "decision_reason": reason,
            }
        )
        return StrategyDecision(direction, base_conf, reason, stop_loss, take_profit, meta)

    def _breakout_check(
        self, latest, adx_value: float, base_meta: Dict[str, float | str]
    ) -> Optional[StrategyDecision]:
        close = float(latest["close"])
        pdh = float(latest.get("PDH", np.nan))
        pdl = float(latest.get("PDL", np.nan))
        if self.config.breakout_use_or_levels:
            pdh = float(latest.get("ORH", pdh))
            pdl = float(latest.get("ORL", pdl))
        vwap = float(latest["SESSION_VWAP"])
        atr = float(latest["ATR_14"])
        high = float(latest["high"])
        low = float(latest["low"])
        candle_range = high - low if high >= low else 0.0
        body_top_threshold = low + candle_range * 0.70
        body_bottom_threshold = high - candle_range * 0.70

        # Long breakout
        if pd.notna(pdh) and close > pdh and close > vwap and adx_value >= self.config.breakout_adx_threshold:
            if getattr(self.config, "breakout_strength_filter", True) and close < body_top_threshold:
                return None
            return self._enter_with_brackets(
                direction="BUY",
                close=close,
                atr=atr,
                reason="BREAKOUT_LONG",
                base_conf=0.70,
                extra_meta=base_meta,
            )
        # Short breakout
        if pd.notna(pdl) and close < pdl and close < vwap and adx_value >= self.config.breakout_adx_threshold:
            if getattr(self.config, "breakout_strength_filter", True) and close > body_bottom_threshold:
                return None
            return self._enter_with_brackets(
                direction="SELL",
                close=close,
                atr=atr,
                reason="BREAKOUT_SHORT",
                base_conf=0.70,
                extra_meta=base_meta,
            )
        return None

    def _log_decision(self, ts, decision: StrategyDecision, latest) -> None:
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
            "adx": float(latest.get("ADX_14", np.nan)),
            "pdh": float(latest.get("PDH", np.nan)),
            "pdl": float(latest.get("PDL", np.nan)),
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
        }
        logger.info(
            "🕐 %s | state=%s trend=%s act=%s conf=%.2f reason=%s close=%.2f ema9=%.2f ema21=%.2f vwap=%.2f atr=%.2f adx=%.2f",
            ts,
            payload["state"],
            payload["trend"],
            decision.action,
            decision.confidence,
            decision.reason,
            payload["close"],
            payload["ema9"],
            payload["ema21"],
            payload["vwap"],
            payload["atr"],
            payload["adx"],
        )
        log_structured_event(
            agent="mes_one_minute",
            event_type="decision",
            message=decision.reason,
            payload=payload,
        )
