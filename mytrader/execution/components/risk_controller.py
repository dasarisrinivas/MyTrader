"""Risk and structural context helpers."""

from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np

from ..utils.logger import logger


class RiskController:
    """Computes structural metrics and confidence adjustments."""

    def __init__(self, manager: "LiveTradingManager"):  # noqa: F821 (forward reference)
        self.manager = manager

    def compute_structural_metrics(self, features) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if features is None or features.empty:
            return metrics
        history = features.tail(min(len(features), 240)).copy()
        if history.empty:
            return metrics
        closes = history["close"].astype(float)
        highs = history["high"].astype(float)
        lows = history["low"].astype(float)
        idx = np.arange(len(closes))
        if len(idx) >= 5:
            slope = np.polyfit(idx, closes, 1)[0]
            metrics["trend_strength"] = float(slope / max(closes.iloc[-1], 1e-6))
            metrics["momentum_score"] = float(closes.pct_change(periods=5).dropna().sum())
        atr_series = history["ATR_14"].dropna() if "ATR_14" in history.columns else None
        if atr_series is not None and not atr_series.empty:
            metrics["atr_latest"] = float(atr_series.iloc[-1])
            metrics["volatility_rank"] = float(min(1.0, max(0.0, atr_series.rank(pct=True).iloc[-1])))
            metrics["atr_slope"] = float(atr_series.diff().rolling(5).mean().dropna().iloc[-1])
        if "RSI_14" in history.columns:
            metrics["rsi_latest"] = float(history["RSI_14"].dropna().iloc[-1])
        price_range = highs.max() - lows.min()
        if price_range > 0:
            metrics["range_position"] = float((closes.iloc[-1] - lows.min()) / price_range)
            metrics["range_width"] = float(price_range)
        metrics["structure_samples"] = len(history)
        metrics["historical_trend_strength"] = (
            float(closes.pct_change(periods=20).dropna().mean())
            if len(closes) >= 20
            else metrics.get("trend_strength", 0.0)
        )
        return metrics

    def apply_structural_weighting(self, signal, metrics: Dict[str, float]) -> float:
        if not metrics or signal.action == "HOLD":
            return 0.0
        is_buy = signal.action in ["BUY", "SCALP_BUY"]
        direction = 1 if is_buy else -1
        trend_bias = metrics.get("trend_strength", 0.0) * direction
        momentum_bias = metrics.get("momentum_score", 0.0) * direction
        range_position = metrics.get("range_position", 0.5)
        range_bias = (0.5 - range_position) * direction
        volatility_bias = (metrics.get("volatility_rank", 0.5) - 0.5)
        structure_score = (
            trend_bias * 0.5
            + momentum_bias * 0.3
            + range_bias * 0.2
            + volatility_bias * 0.1
        )
        structure_score = max(-0.18, min(0.18, structure_score))
        return structure_score

    def persist_structural_snapshot(self, structural_metrics: Dict[str, float], rag_context: Dict[str, Any], signal) -> None:
        if not structural_metrics or not self.manager.metrics_logger:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trend_strength": structural_metrics.get("trend_strength"),
            "volatility_rank": structural_metrics.get("volatility_rank"),
            "range_position": structural_metrics.get("range_position"),
            "atr": structural_metrics.get("atr_latest"),
            "rsi": structural_metrics.get("rsi_latest"),
            "rag_weighted_win_rate": (rag_context.get("stats") or {}).get("win_rate", 0.0),
            "rag_similar_trades": rag_context.get("similar_trades_count"),
            "decision": signal.action,
            "confidence": signal.confidence,
            "historical_avg_trend": structural_metrics.get("historical_trend_strength"),
        }
        try:
            self.manager.metrics_logger.record_market_metrics(payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Unable to persist structural snapshot: {exc}")

    def get_trend_from_features(self, row) -> str:
        ema_9 = float(row.get("EMA_9", row.get("ema_9", 0)))
        ema_20 = float(row.get("EMA_20", row.get("ema_20", 0)))
        close = float(row.get("close", 0))
        if ema_9 > ema_20 and close > ema_9:
            return "UPTREND"
        elif ema_9 < ema_20 and close < ema_9:
            return "DOWNTREND"
        return "RANGE"

    def get_volatility_from_features(self, row) -> str:
        atr = float(row.get("ATR_14", row.get("atr", 0)))
        if atr > 15:
            return "HIGH"
        elif atr > 8:
            return "MED"
        return "LOW"
