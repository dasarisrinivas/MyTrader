"""Risk and structural context helpers."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.logger import logger
from ...risk.atr_module import compute_protective_offsets


class RiskController:
    """Computes structural metrics and confidence adjustments."""

    def __init__(self, manager: "LiveTradingManager"):  # noqa: F821 (forward reference)
        self.manager = manager

    @staticmethod
    def _normalize_trend(label: str) -> str:
        txt = str(label or "").upper()
        if "UP" in txt:
            return "UPTREND"
        if "DOWN" in txt:
            return "DOWNTREND"
        return "RANGE"

    @staticmethod
    def _normalize_volatility(label: str) -> str:
        txt = str(label or "").upper()
        if txt.startswith("MED"):
            return "MEDIUM"
        if "HIGH" in txt:
            return "HIGH"
        if "LOW" in txt:
            return "LOW"
        return "MEDIUM"

    @staticmethod
    def _get_feature(row: Any, keys: List[str], default: float = 0.0) -> float:
        for key in keys:
            if key in row:
                try:
                    return float(row.get(key))
                except Exception:
                    continue
        return float(default)

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
        ema_9 = self._get_feature(row, ["EMA_9", "ema_9"])
        ema_20 = self._get_feature(row, ["EMA_20", "ema_20"])
        close = self._get_feature(row, ["close"])
        if ema_9 > ema_20 and close > ema_9:
            return self._normalize_trend("UPTREND")
        elif ema_9 < ema_20 and close < ema_9:
            return self._normalize_trend("DOWNTREND")
        return self._normalize_trend("RANGE")

    def get_volatility_from_features(self, row) -> str:
        atr = self._get_feature(row, ["ATR_14", "atr"], 0.0)
        if atr > 15:
            return self._normalize_volatility("HIGH")
        if atr > 8:
            return self._normalize_volatility("MED")
        return self._normalize_volatility("LOW")

    def validate_trade_risk(self, trade_request: "TradeRequest") -> "RiskResult":
        """Validate proposed trade against guardrails."""
        allowed = self.manager._validate_entry_guard(
            trade_request.entry_price,
            trade_request.stop_loss,
            trade_request.take_profit,
            trade_request.quantity,
            trade_request.action,
        )
        return RiskResult(allowed=allowed)

    def pre_trade_risk_checks(self) -> "RiskValidation":
        """Evaluate high-level risk gates before trade placement."""
        m = self.manager
        reasons: List[str] = []
        if not m.risk:
            return RiskValidation(allowed=True, reasons=reasons, suggested_size=0)

        stats = m.risk.get_statistics()
        max_daily_loss = getattr(m.settings.trading, "max_daily_loss", float("inf"))
        max_trades = getattr(m.settings.trading, "max_daily_trades", float("inf"))
        heat_limit = getattr(m.settings.trading, "margin_limit_pct", 1.0) * 100

        if stats.get("daily_loss", 0) >= max_daily_loss:
            reasons.append("DAILY_LOSS_LIMIT")
        if stats.get("daily_trade_count", 0) >= max_trades:
            reasons.append("DAILY_TRADE_LIMIT")
        if stats.get("portfolio_heat", 0) >= heat_limit:
            reasons.append("PORTFOLIO_HEAT")

        return RiskValidation(
            allowed=len(reasons) == 0,
            reasons=reasons,
            suggested_size=self.position_size_calculation(),
        )

    def position_size_calculation(self) -> int:
        """Return a conservative position size based on current stats."""
        m = self.manager
        if not m.risk:
            return 0
        stats = m.risk.get_statistics()
        confidence_floor = max(0.5, getattr(m, "_min_confidence_for_trade", 0.6))
        qty = m.risk.position_size(
            m.settings.trading.initial_capital,
            confidence_floor,
            win_rate=stats.get("win_rate"),
            avg_win=stats.get("avg_win"),
            avg_loss=stats.get("avg_loss"),
        )
        return min(max(1, qty), m.settings.trading.max_position_size)

    def stop_loss_optimization(self) -> float:
        """Suggest a stop level using ATR offsets and instrument tick size."""
        m = self.manager
        price = (
            m.status.current_price
            or (m.price_history[-1]["close"] if m.price_history else 0.0)
        )
        action = m.status.last_signal or "BUY"
        atr = 0.0
        if m.current_trade_features:
            atr = float(m.current_trade_features.get("atr", 0.0))
        if atr <= 0 and m.price_history:
            atr = float(m.price_history[-1].get("ATR_14", 0.0) or 0.0)
        offsets = compute_protective_offsets(
            atr_value=atr,
            tick_size=m.settings.trading.tick_size,
            scalper=False,
            volatility="MED",
            current_price=price,
        )
        is_buy = action.upper() in ("BUY", "SCALP_BUY")
        stop_price = price - offsets.stop_offset if is_buy else price + offsets.stop_offset
        return stop_price

    def exposure_monitoring(self) -> "ExposureReport":
        """Report current portfolio heat against configured limits."""
        m = self.manager
        stats = m.risk.get_statistics() if m.risk else {}
        heat = float(stats.get("portfolio_heat", 0.0) or 0.0)
        limit_pct = getattr(m.settings.trading, "margin_limit_pct", 1.0) * 100
        open_positions = 1 if getattr(m.status, "current_position", 0) else 0
        within_limits = heat <= limit_pct
        return ExposureReport(
            portfolio_heat=heat,
            limit=limit_pct,
            within_limits=within_limits,
            open_positions=open_positions,
        )

    def calculate_stop_loss(
        self,
        entry_price: float,
        action: str,
        atr: float,
        regime_params: Dict[str, float],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Compute protective stop/target using ATR + regime params."""
        metadata: Dict[str, Any] = {}
        m = self.manager
        atr_mult_sl = regime_params.get("atr_multiplier_sl", 2.0)
        atr_mult_tp = regime_params.get("atr_multiplier_tp", 4.0)

        if atr > 0:
            stop_offset = max(atr * atr_mult_sl, m._min_stop_distance)
            target_offset = max(
                atr * atr_mult_tp,
                stop_offset + m.settings.trading.tick_size,
            )
            metadata["atr_fallback_used"] = False
        else:
            offsets = compute_protective_offsets(
                atr_value=atr,
                tick_size=m.settings.trading.tick_size,
                scalper=False,
                volatility=regime_params.get("volatility", "MED"),
                current_price=entry_price,
            )
            stop_offset = offsets.stop_offset
            target_offset = offsets.target_offset
            metadata["atr_fallback_used"] = True
            logger.warning(
                f"⚠️ ATR invalid, using fallback offsets SL={stop_offset:.2f}, TP={target_offset:.2f}"
            )

        is_buy = action.upper() in ("BUY", "SCALP_BUY")
        stop_loss = entry_price - stop_offset if is_buy else entry_price + stop_offset
        take_profit = entry_price + target_offset if is_buy else entry_price - target_offset
        return stop_loss, take_profit, metadata


@dataclass
class TradeRequest:
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: int
    action: str = "BUY"


@dataclass
class RiskResult:
    allowed: bool
    reason: Optional[str] = None


@dataclass
class RiskValidation:
    allowed: bool
    reasons: List[str] = field(default_factory=list)
    suggested_size: int = 0


@dataclass
class ExposureReport:
    portfolio_heat: float
    limit: float
    within_limits: bool
    open_positions: int = 0
