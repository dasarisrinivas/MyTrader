"""Hybrid RAG Pipeline Integration for Live Trading Manager.

This module provides integration between the 3-layer hybrid RAG pipeline
and the live trading manager. It can be used as an alternative decision
engine that replaces or augments the existing signal generation.

Usage:
    from mytrader.rag.pipeline_integration import HybridPipelineIntegration
    
    # In LiveTradingManager
    self.hybrid_pipeline = HybridPipelineIntegration(settings)
    
    # In _process_trading_cycle
    result = await self.hybrid_pipeline.process(features, current_price)
    if result.should_trade:
        await self._place_order(result.signal, current_price, features)
"""
import asyncio
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from mytrader.hybrid.multi_factor_scorer import MultiFactorScorer
from mytrader.hybrid.coordination import AgentBus
from mytrader.utils.structured_logging import log_structured_event
from mytrader.utils.hold_reason import HoldReason
from mytrader.rag.hybrid_rag_pipeline import (
    HybridRAGPipeline,
    TradeAction,
    HybridPipelineResult,
    create_hybrid_pipeline,
)
from mytrader.risk.protection_validator import (
    ProtectionComputation,
    calculate_protection,
)
from mytrader.rag.rag_storage_manager import get_rag_storage
from mytrader.rag.embedding_builder import create_embedding_builder
from mytrader.rag.trade_logger import TradeLogger, get_trade_logger
from mytrader.rag.mistake_analyzer import MistakeAnalyzer, get_mistake_analyzer
from mytrader.rag.rag_daily_updater import RAGDailyUpdater, create_daily_updater

TRADE_ACTIONS = {"BUY", "SELL", "SCALP_BUY", "SCALP_SELL"}


class HybridSignal:
    """Signal object compatible with existing trading manager."""
    
    def __init__(
        self,
        action: str,
        confidence: float,
        metadata: Dict[str, Any] = None,
    ):
        self.action = action
        self.confidence = confidence
        self.metadata = metadata or {}


class HybridPipelineIntegration:
    """Integrates the 3-layer hybrid RAG pipeline with live trading.
    
    This class:
    1. Wraps the HybridRAGPipeline
    2. Converts between existing feature format and pipeline format
    3. Logs trades automatically
    4. Analyzes mistakes on losing trades
    """
    
    def __init__(
        self,
        settings: Any,
        llm_client: Optional[Any] = None,
        enabled: bool = True,
        context_bus: Optional[AgentBus] = None,
    ):
        """Initialize the hybrid pipeline integration.
        
        Args:
            settings: Application settings
            llm_client: Bedrock client for LLM calls
            enabled: Whether to use hybrid pipeline
        """
        self.settings = settings
        self.enabled = enabled
        self.context_bus = context_bus
        self._current_position = None
        data_cfg = getattr(settings, "data", None)
        self._symbol = getattr(data_cfg, "ibkr_symbol", "ES")
        self._timeframe = getattr(data_cfg, "tradingview_interval", "1m")
        
        # Get hybrid config from settings if available
        hybrid_config = {}
        if hasattr(settings, 'hybrid'):
            hybrid_config = {
                "rule_engine": {
                    "atr_min": getattr(settings.hybrid, 'atr_min', 0.15),  # Lowered for low-vol markets
                    "atr_max": getattr(settings.hybrid, 'atr_max', 5.0),
                    "rsi_oversold": getattr(settings.hybrid, 'rsi_oversold', 30),
                    "rsi_overbought": getattr(settings.hybrid, 'rsi_overbought', 70),
                    "cooldown_minutes": getattr(settings.hybrid, 'cooldown_minutes', 15),
                    "signal_threshold": getattr(settings.hybrid, 'signal_threshold', 40),
                },
                "llm": {
                    "min_confidence": getattr(settings.hybrid, 'min_confidence', 60),
                    "uncertainty_band": (
                        getattr(settings.hybrid, 'llm_uncertainty_band_low', 0.35),
                        getattr(settings.hybrid, 'llm_uncertainty_band_high', 0.65),
                    ),
                    "call_cooldown_seconds": getattr(settings.hybrid, 'llm_call_cooldown_seconds', 60),
                    "response_cache_ttl_seconds": getattr(settings.hybrid, 'llm_response_cache_ttl_seconds', 900),
                },
                "min_confidence_for_trade": getattr(settings.hybrid, 'min_confidence_for_trade', 40),
            }
            hybrid_config["level_confirmation_settings"] = {
                "level_confirmation_enabled": getattr(settings.hybrid, "level_confirmation_enabled", True),
                "level_confirm_proximity_pct": getattr(settings.hybrid, "level_confirm_proximity_pct", 0.15),
                "level_confirm_buffer_atr_mult": getattr(settings.hybrid, "level_confirm_buffer_atr_mult", 0.10),
                "level_confirm_min_buffer_points": getattr(settings.hybrid, "level_confirm_min_buffer_points", 0.50),
                "level_confirm_max_wait_candles": getattr(settings.hybrid, "level_confirm_max_wait_candles", 3),
                "level_confirm_timeout_mode": getattr(settings.hybrid, "level_confirm_timeout_mode", "SOFT_PENALTY"),
                "level_confirm_timeout_penalty": getattr(settings.hybrid, "level_confirm_timeout_penalty", 0.12),
            }
        
        rag_cfg = getattr(settings, 'rag', None)
        if rag_cfg:
            hybrid_config["min_similar_trades"] = getattr(rag_cfg, "min_similar_trades", 2)
            hybrid_config["min_weighted_win_rate"] = getattr(rag_cfg, "min_weighted_win_rate", 0.45)
            hybrid_config["min_weighted_win_rate_soft_floor"] = getattr(
                rag_cfg,
                "min_weighted_win_rate_soft_floor",
                getattr(rag_cfg, "min_weighted_win_rate", 0.45),
            )
            hybrid_config["min_similar_trades_for_full_threshold"] = getattr(
                rag_cfg,
                "min_similar_trades_for_full_threshold",
                0,
            )
            hybrid_config["rag_regime_filter"] = {
                "enabled": getattr(rag_cfg, "enabled", True),
                "min_win_rate": getattr(rag_cfg, "min_win_rate", getattr(rag_cfg, "min_weighted_win_rate", 0.15)),
                "mode": getattr(rag_cfg, "regime_mode", "relaxed"),
                "min_sample_for_hard_block": getattr(rag_cfg, "min_sample_for_hard_block", 30),
                "soft_penalty_when_below": getattr(rag_cfg, "soft_penalty_when_below", 0.10),
                "hard_block_when_below": getattr(rag_cfg, "hard_block_when_below", False),
                "min_similar_trades": getattr(rag_cfg, "min_similar_trades", 2),
                "min_weighted_win_rate_soft_floor": getattr(
                    rag_cfg,
                    "min_weighted_win_rate_soft_floor",
                    getattr(rag_cfg, "min_weighted_win_rate", 0.45),
                ),
                "min_similar_trades_for_full_threshold": getattr(
                    rag_cfg,
                    "min_similar_trades_for_full_threshold",
                    0,
                ),
            }
        
        # Initialize components
        self.storage = get_rag_storage()
        self.trade_logger = get_trade_logger()
        self.mistake_analyzer = get_mistake_analyzer()
        
        # Try to initialize embedding builder
        try:
            self.embedding_builder = create_embedding_builder()
        except Exception as e:
            logger.warning(f"Could not initialize embedding builder: {e}")
            self.embedding_builder = None
        
        rag_data_root = getattr(getattr(settings, 'hybrid', None), 'rag_data_path', 'rag_data')
        self.rag_data_path = Path(rag_data_root)
        self._initialize_local_rag_index()
        
        factor_weights = None
        if hasattr(settings, 'hybrid'):
            factor_weights = getattr(settings.hybrid, 'factor_weights', None)
        self.multi_factor_scorer = MultiFactorScorer(weights=factor_weights)

        # Initialize pipeline
        self.pipeline = create_hybrid_pipeline(
            config=hybrid_config,
            llm_client=llm_client,
            embedding_builder=self.embedding_builder,
            storage_manager=self.storage,
        )
        if self.pipeline and self.embedding_builder:
            self.pipeline.rag_retriever.embedding_builder = self.embedding_builder
        
        # Initialize daily updater
        self.daily_updater = create_daily_updater(storage=self.storage, embedding_builder=self.embedding_builder)
        
        # Track current trade for logging
        self._current_trade_id: Optional[str] = None
        
        logger.info("HybridPipelineIntegration initialized")
    
    def _collect_local_documents(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Collect documents from rag_data_path to bootstrap embeddings."""
        documents: List[Tuple[str, str, Dict[str, Any]]] = []
        if not self.rag_data_path.exists():
            return documents
        
        valid_ext = {".txt", ".md", ".json"}
        for file in self.rag_data_path.rglob("*"):
            if not file.is_file() or file.suffix.lower() not in valid_ext:
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            metadata = {
                "type": file.parent.name,
                "source": str(file.relative_to(self.rag_data_path)),
            }
            doc_id = f"{metadata['type']}::{file.stem}"
            documents.append((doc_id, text[:5000], metadata))
        return documents
    
    def _initialize_local_rag_index(self) -> None:
        """Ensure the embedding index has content, even when offline."""
        if not self.embedding_builder:
            return
        try:
            stats = self.embedding_builder.get_stats()
            if stats.get("documents", 0) > 0:
                return
        except Exception:
            pass
        
        documents = self._collect_local_documents()
        if documents:
            logger.info(f"Building local RAG index from {len(documents)} documents in {self.rag_data_path}")
            try:
                self.embedding_builder.build_index(documents, save=True)
            except Exception as e:
                logger.warning(f"Failed to build local RAG index: {e}")
        else:
            logger.warning(f"No local RAG documents found in {self.rag_data_path}")

    def _get_tick_size(self) -> float:
        """Return configured tick size with safe fallback."""
        trading_cfg = getattr(self.settings, "trading", None)
        tick_size = getattr(trading_cfg, "tick_size", 0.25) if trading_cfg else 0.25
        if not tick_size or not math.isfinite(tick_size):
            return 0.25
        return max(1e-6, float(tick_size))

    def ensure_ready(self, min_documents: int = 1) -> Dict[str, Any]:
        """Verify embedding builder has enough documents and warm it up if needed."""
        stats: Dict[str, Any] = {}
        if not self.embedding_builder:
            return stats
        try:
            stats = self.embedding_builder.get_stats()
        except Exception as exc:
            logger.debug(f"Unable to fetch embedding stats: {exc}")
            stats = {}
        if (stats.get("documents") or 0) < min_documents:
            self._initialize_local_rag_index()
            try:
                stats = self.embedding_builder.get_stats()
            except Exception:
                stats = {}
        return stats
    
    def convert_features_to_market_data(
        self,
        features,
        current_price: float,
        historical_metrics: Optional[Dict[str, Any]] = None,
        current_position: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Convert pandas features DataFrame to market data dict.
        
        Args:
            features: Features DataFrame with indicators
            current_price: Current price
            current_position: Optional position info (object with quantity/avg_cost)
            
        Returns:
            Market data dictionary for pipeline
        """
        if features.empty:
            return {"close": current_price, "price": current_price}
        
        row = features.iloc[-1]
        timestamp = features.index[-1]
        if hasattr(timestamp, "isoformat"):
            candle_ts = timestamp.isoformat()
        else:
            candle_ts = now_cst().isoformat()
        
        # Extract indicators with safe defaults
        market_data = {
            "price": current_price,
            "close": float(row.get("close", current_price)),
            "open": float(row.get("open", current_price)),
            "high": float(row.get("high", current_price)),
            "low": float(row.get("low", current_price)),
            
            # EMAs
            "ema_9": float(row.get("EMA_9", row.get("ema_9", current_price))),
            "ema_20": float(row.get("EMA_20", row.get("ema_20", current_price))),
            "ema_50": float(row.get("SMA_50", row.get("ema_50", current_price))),
            
            # Momentum
            "rsi": float(row.get("RSI_14", row.get("rsi", 50))),
            # MACD histogram: Try MACDhist_12_26_9 (standard column name from feature_engineer.py)
            "macd_hist": float(row.get("MACDhist_12_26_9", row.get("MACD_hist", row.get("MACDh_12_26_9", row.get("macd_hist", 0))))),
            
            # Volatility
            "atr": float(row.get("ATR_14", row.get("atr", 0))),
            "atr_20_avg": float(row.get("ATR_20_avg", row.get("ATR_14", row.get("atr", 1)))),
            
            # Levels (may need to be set elsewhere)
            "pdh": float(row.get("PDH", row.get("pdh", 0))),
            "pdl": float(row.get("PDL", row.get("pdl", 0))),
            "weekly_high": float(row.get("weekly_high", 0)),
            "weekly_low": float(row.get("weekly_low", 0)),
            "pivot": float(row.get("pivot", 0)),
            
            # Volume
            "volume_ratio": float(row.get("volume_ratio", 1.0)),
            
            # Additional
            "volatility": float(row.get("volatility_5m", row.get("volatility", 0))),
            "symbol": self._symbol,
            "timeframe": self._timeframe,
            "candle_timestamp": candle_ts,
        }
        if current_position:
            market_data["current_position_qty"] = getattr(current_position, "quantity", 0) or 0
            market_data["current_position_avg_cost"] = getattr(current_position, "avg_cost", current_price) or current_price
        recent = features.tail(50)
        recent_bars: List[Dict[str, float]] = []
        for idx, bar in recent.iterrows():
            if hasattr(idx, "isoformat"):
                ts_value = idx.isoformat()
            else:
                ts_value = str(idx)
            recent_bars.append(
                {
                    "ts": ts_value,
                    "close": float(bar.get("close", current_price)),
                }
            )
        market_data["recent_bars"] = recent_bars
        
        market_data.update(self._compute_historical_structure(features))
        
        if historical_metrics:
            market_data["historical_trend_strength"] = historical_metrics.get("avg_trend_strength", 0.0)
            market_data["historical_volatility_rank"] = historical_metrics.get("avg_volatility_rank", 0.0)
            market_data["historical_range_position"] = historical_metrics.get("avg_range_position", 0.0)
        
        logger.debug(
            "Market Data: price=%.2f RSI=%.1f trend=%.3f atr=%.3f range_pos=%.2f",
            market_data["price"],
            market_data["rsi"],
            market_data.get("trend_strength", 0.0),
            market_data["atr"],
            market_data.get("range_position", 0.0),
        )
        
        return market_data

    @staticmethod
    def _safe_float(value: object, default: Optional[float] = None) -> Optional[float]:
        """Convert arbitrary input to float without raising."""
        if value is None:
            return default
        try:
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped or stripped.lower() in {"nan", "none"}:
                    return default
                parsed = float(stripped)
            else:
                parsed = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(parsed):
            return default
        return parsed

    def _compute_historical_structure(self, features) -> Dict[str, float]:
        """Compute multi-candle metrics so the pipeline sees broader context."""
        metrics: Dict[str, float] = {}
        history = features.tail(min(len(features), 200)).copy()
        if history.empty:
            return metrics
        
        closes = history["close"].astype(float)
        highs = history["high"].astype(float)
        lows = history["low"].astype(float)
        atr_series = history["ATR_14"].astype(float) if "ATR_14" in history.columns else None
        
        if len(closes) >= 5:
            idx = np.arange(len(closes))
            try:
                slope = np.polyfit(idx, closes, 1)[0]
                metrics["trend_strength"] = float(slope / max(closes.iloc[-1], 1e-6))
            except Exception:
                metrics["trend_strength"] = 0.0
            
            window = closes.rolling(20).mean().iloc[-1] if len(closes) >= 20 else closes.mean()
            metrics["mean_reversion_score"] = float((closes.iloc[-1] - window) / max(window, 1e-6))
        
        if atr_series is not None and not atr_series.dropna().empty:
            atr_values = atr_series.dropna()
            metrics["atr_trend"] = float(atr_values.iloc[-1] - atr_values.mean())
            metrics["volatility_rank"] = float(min(1.0, max(0.0, atr_values.rank(pct=True).iloc[-1])))
        
        price_range = highs.max() - lows.min()
        if price_range > 0:
            metrics["range_position"] = float((closes.iloc[-1] - lows.min()) / price_range)
            metrics["support_level"] = float(lows.min())
            metrics["resistance_level"] = float(highs.max())
        else:
            metrics["range_position"] = 0.5
            metrics["support_level"] = float(closes.min())
            metrics["resistance_level"] = float(closes.max())
        
        recent = features.tail(3)
        if not recent.empty:
            metrics["momentum_score"] = float(recent["close"].pct_change().sum())
        
        return metrics

    def _get_context(self, channel: str, ttl_seconds: float) -> Optional[Dict[str, Any]]:
        """Fetch latest context from agent bus."""
        if not self.context_bus:
            return None
        message = self.context_bus.get_fresh(channel, ttl_seconds)
        return message.payload if message else None

    def set_current_position(self, position: Optional[Any]) -> None:
        """Inject current position for downstream process calls."""
        self._current_position = position
    
    def process_sync(
        self,
        features,
        current_price: float,
        current_position: Optional[Any] = None,
    ) -> Tuple[HybridSignal, HybridPipelineResult]:
        """Process features through the hybrid pipeline (synchronous).
        
        Args:
            features: Features DataFrame
            current_price: Current price
            current_position: Optional existing position
            
        Returns:
            Tuple of (HybridSignal, HybridPipelineResult)
        """
        if not self.enabled:
            return HybridSignal("HOLD", 0.0), None
        
        historical_summary = self._load_recent_metrics()
        
        # Convert features to market data
        market_data = self.convert_features_to_market_data(
            features,
            current_price,
            historical_metrics=historical_summary,
            current_position=current_position,
        )
        
        # Process through pipeline with fallbacks
        try:
            result = self.pipeline.process(
                market_data,
                current_position=current_position,
                features=features.to_dict() if hasattr(features, "to_dict") else None,
            )
            if result is None:
                logger.warning("Hybrid pipeline returned None, generating fallback signal")
                return self._generate_fallback_signal(market_data, features)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Hybrid pipeline error: {exc}")
            return self._generate_fallback_signal(market_data, features)

        news_context = self._get_context("news", 600)
        macro_context = self._get_context("macro", 900)
        risk_context = self._get_context("account", 180)
        macro_context = self._merge_structural_bias(macro_context, historical_summary)

        market_metrics_payload = {
            "trend_strength": market_data.get("trend_strength"),
            "range_position": market_data.get("range_position"),
            "volatility_rank": market_data.get("volatility_rank"),
            "historical_trend_strength": historical_summary.get("avg_trend_strength"),
            "historical_range_position": historical_summary.get("avg_range_position"),
        }
        decision = self.multi_factor_scorer.score(
            result,
            news_context=news_context,
            macro_context=macro_context,
            risk_context=risk_context,
            market_metrics=market_metrics_payload,
        )
        
        entry_price = market_data.get("price", current_price)
        atr_value = market_data.get("atr", 0.0)
        raw_stop = self._safe_float(result.stop_loss, default=0.0)
        raw_target = self._safe_float(result.take_profit, default=0.0)
        action_value = result.final_action.value
        protection: Optional[ProtectionComputation] = None

        if (
            action_value in TRADE_ACTIONS
            and (raw_stop <= 0 or raw_target <= 0)
        ):
            logger.warning(
                f"Hybrid pipeline produced non-positive protective levels "
                f"(action={action_value} stop={raw_stop:.4f} target={raw_target:.4f} entry={entry_price:.2f} atr={atr_value:.4f})"
            )

        if action_value in TRADE_ACTIONS:
            protection = calculate_protection(
                action=action_value,
                entry_price=entry_price,
                stop_points=result.stop_loss,
                target_points=result.take_profit,
                atr_value=atr_value,
                tick_size=self._get_tick_size(),
                volatility=result.rule_engine.volatility_regime,
            )
            stop_points = protection.stop_offset
            take_points = protection.target_offset
            if protection.source == "fallback_atr":
                logger.warning(
                    f"⚠️ Protective fallback used (action={action_value} reason={protection.fallback_reason or 'fallback'} atr={atr_value:.4f})"
                )
        else:
            stop_points = max(0.0, raw_stop)
            take_points = max(0.0, raw_target)

        metadata = {
            "hybrid_reasoning": result.final_reasoning,
            "rule_engine_score": result.rule_engine.score,
            "filters_passed": result.rule_engine.filters_passed,
            "filters_blocked": result.rule_engine.filters_blocked,
            "market_trend": result.rule_engine.market_trend,
            "volatility_regime": result.rule_engine.volatility_regime,
            "rag_docs_count": len(result.rag_retrieval.documents),
            "rag_similar_trades": result.rag_retrieval.similar_trade_count,
            "rag_weighted_win_rate": result.rag_retrieval.weighted_win_rate,
            "llm_confidence": result.llm_decision.confidence if result.llm_decision else None,
            "stop_loss_points": stop_points,
            "take_profit_points": take_points,
            "position_size_factor": result.position_size,
            "factor_scores": decision.scores.to_dict(),
            "confidence_band": decision.confidence_band,
            "trend_strength": market_data.get("trend_strength"),
            "range_position": market_data.get("range_position"),
            "momentum_score": market_data.get("momentum_score"),
        }
        
        if protection:
            metadata.update(
                {
                    "stop_loss_absolute": protection.stop_price,
                    "take_profit_absolute": protection.target_price,
                    "protection_source": protection.source,
                    "protection_fallback_reason": protection.fallback_reason,
                }
            )
        
        if result.hold_reason:
            metadata["hold_reason"] = {
                "gate": result.hold_reason.gate,
                "reason_code": result.hold_reason.reason_code,
                "reason_detail": result.hold_reason.reason_detail,
                "context": result.hold_reason.context,
            }

        pipeline_conf = result.final_confidence
        if pipeline_conf > 1:
            pipeline_conf = pipeline_conf / 100.0
        pipeline_conf = max(0.0, min(1.0, pipeline_conf))
        metadata["pipeline_final_confidence"] = result.final_confidence
        metadata["decision_confidence"] = decision.confidence
        
        signal = HybridSignal(
            action=decision.action if result.final_action != TradeAction.BLOCKED else "HOLD",
            confidence=pipeline_conf,
            metadata=metadata,
        )

        if protection:
            self._log_protection_snapshot(
                action_value=action_value,
                entry_price=entry_price,
                atr=atr_value,
                protection=protection,
                confidence=signal.confidence,
                trend=result.rule_engine.market_trend,
                volatility=result.rule_engine.volatility_regime,
            )

        if self.context_bus:
            self.context_bus.publish(
                "decision",
                {
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "metadata": metadata,
                },
                producer="hybrid_pipeline",
            )
        
        log_structured_event(
            agent="hybrid_pipeline",
            event_type="decision.generated",
            message=f"{signal.action} @ {signal.confidence:.2f}",
            payload={
                "trend": result.rule_engine.market_trend,
                "volatility": result.rule_engine.volatility_regime,
                "confidence_band": decision.confidence_band,
                "factor_scores": decision.scores.to_dict(),
                "news_bias": (news_context or {}).get("bias"),
                "macro_bias": (macro_context or {}).get("regime_bias"),
                "hold_reason": metadata.get("hold_reason"),
                "pipeline_confidence": result.final_confidence,
            },
        )
        
        self._persist_market_metrics(
            market_data=market_data,
            rag_result=result.rag_retrieval,
            decision=decision,
            historical_summary=historical_summary,
        )

        logger.info(
            f"Hybrid Pipeline: {signal.action} "
            f"(conf={signal.confidence:.2f}, "
            f"trend={result.rule_engine.market_trend}, "
            f"vol={result.rule_engine.volatility_regime})"
        )

        return signal, result

    def _generate_fallback_signal(
        self,
        market_data: Dict[str, Any],
        features: Any,
    ) -> Tuple[HybridSignal, Optional[HybridPipelineResult]]:
        """Generate a simple fallback signal when the pipeline fails or returns None."""
        def _feature_value(key: str, default: float = 0.0) -> Any:
            # Support both dict-like and DataFrame-like inputs
            if hasattr(features, "get"):
                try:
                    return features.get(key, default)
                except Exception:
                    pass
            try:
                if hasattr(features, "iloc") and len(features) > 0:
                    return features.iloc[-1].get(key, default)
            except Exception:
                pass
            return default

        trend = _feature_value("trend", "UNKNOWN") or _feature_value("market_trend", "UNKNOWN")
        current_price = market_data.get("close", market_data.get("price", 0))
        pdl = _feature_value("pdl", 0)
        pdh = _feature_value("pdh", 0)

        # Enhanced logic: use level position for directional bias
        if pdh and pdl and current_price:
            range_size = pdh - pdl
            price_position = (current_price - pdl) / range_size if range_size > 0 else 0.5
            if current_price > pdh:
                signal = "BUY"
                confidence = 0.25
                rationale = "Fallback: Above PDH"
            elif current_price > pdl:
                if price_position > 0.7:  # Near top of range
                    signal = "SELL"
                    confidence = 0.25
                    rationale = f"Fallback: Near range top ({price_position:.1%})"
                else:  # Middle-upper range
                    signal = "BUY"
                    confidence = 0.20
                    rationale = "Fallback: Above PDL, upward bias"
            elif current_price < pdl:
                signal = "SELL"
                confidence = 0.25
                rationale = "Fallback: Below PDL, downward bias"
            else:
                signal = "HOLD"
                confidence = 0.15
                rationale = "Fallback: At PDL, waiting"
        elif trend == "RANGE" and pdh and pdl:
            range_decision = self._handle_range_bound_market(current_price, pdh, pdl, features)
            signal = range_decision["signal"]
            confidence = range_decision["confidence"]
            rationale = range_decision["rationale"]
        elif trend == "UPTREND":
            signal = "BUY"
            confidence = 0.20
            rationale = "Fallback: Uptrend bias"
        elif trend == "DOWNTREND":
            signal = "SELL"
            confidence = 0.20
            rationale = "Fallback: Downtrend bias"
        else:
            signal = "HOLD"
            confidence = 0.15  # Do not zero out fallback holds
            rationale = "Fallback: No clear setup"

        # Downgrade to info to avoid noisy warnings when fallback logic chooses HOLD
        logger.info(f"🚧 Enhanced fallback signal: {signal} ({confidence:.2f}) reason={rationale}")

        return HybridSignal(
            action=signal,
            confidence=confidence,
            metadata={
                "source": "fallback",
                "rationale": rationale,
                "market_data": market_data,
            },
        ), None

    def _handle_range_bound_market(
        self,
        current_price: float,
        pdh: float,
        pdl: float,
        features: Any,
    ) -> Dict[str, Any]:
        """Handle range-bound market conditions."""
        range_size = pdh - pdl if pdh is not None and pdl is not None else 0
        if range_size <= 0:
            return {
                "signal": "HOLD",
                "confidence": 0.20,
                "rationale": "Range trading: invalid range, holding",
            }
        upper_threshold = pdh - (range_size * 0.1)  # 90% of range
        lower_threshold = pdl + (range_size * 0.1)  # 10% of range
        if current_price >= upper_threshold:
            return {
                "signal": "SELL",
                "confidence": 0.30,
                "rationale": f"Range trading: near resistance at {pdh}",
            }
        if current_price <= lower_threshold:
            return {
                "signal": "BUY",
                "confidence": 0.30,
                "rationale": f"Range trading: near support at {pdl}",
            }
        return {
            "signal": "HOLD",
            "confidence": 0.20,
            "rationale": "Range trading: mid-range, waiting for boundaries",
        }

    def _log_protection_snapshot(
        self,
        action_value: str,
        entry_price: float,
        atr: float,
        protection: ProtectionComputation,
        confidence: float,
        trend: Optional[str],
        volatility: Optional[str],
    ) -> None:
        """Emit telemetry about the computed protective bracket."""
        if not protection:
            return

        fallback_used = protection.source != "pipeline"
        message = (
            f"{action_value} entry={entry_price:.2f} "
            f"SL={protection.stop_price:.2f} ({protection.stop_offset:.2f}) "
            f"TP={protection.target_price:.2f} ({protection.target_offset:.2f})"
        )
        payload = {
            "action": action_value,
            "entry_price": entry_price,
            "stop_loss": protection.stop_price,
            "take_profit": protection.target_price,
            "stop_offset": protection.stop_offset,
            "target_offset": protection.target_offset,
            "atr": atr,
            "trend": trend,
            "volatility": volatility,
            "confidence": confidence,
            "source": protection.source,
            "fallback_reason": protection.fallback_reason,
            "fallback_used": fallback_used,
        }
        log_structured_event(
            agent="hybrid_pipeline",
            event_type="protection.snapshot",
            message=message,
            payload=payload,
        )
        logger.info(f"🛡️ Protection snapshot | {message} fallback={'yes' if fallback_used else 'no'}")
    
    async def process(
        self,
        features,
        current_price: float,
        current_position: Optional[Any] = None,
    ) -> Tuple[HybridSignal, HybridPipelineResult]:
        """Process features through the hybrid pipeline (async wrapper).
        
        Args:
            features: Features DataFrame
            current_price: Current price
            current_position: Optional existing position
            
        Returns:
            Tuple of (HybridSignal, HybridPipelineResult)
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        position_arg = current_position if current_position is not None else self._current_position
        return await loop.run_in_executor(
            None,
            self.process_sync,
            features,
            current_price,
            position_arg,
        )

    def _merge_structural_bias(
        self,
        macro_context: Optional[Dict[str, Any]],
        historical_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Blend stored metrics into macro context for scoring."""
        ctx = dict(macro_context or {})
        trend_strength = historical_summary.get("avg_trend_strength")
        if trend_strength is not None:
            if trend_strength > 0.05:
                ctx["regime_bias"] = "BULLISH"
            elif trend_strength < -0.05:
                ctx["regime_bias"] = "BEARISH"
            else:
                ctx.setdefault("regime_bias", "NEUTRAL")
        return ctx

    def _persist_market_metrics(
        self,
        market_data: Dict[str, Any],
        rag_result,
        decision,
        historical_summary: Dict[str, Any],
    ) -> None:
        """Persist blended metrics so future decisions can reuse them."""
        if not self.trade_logger or market_data is None:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trend_strength": market_data.get("trend_strength"),
            "volatility_rank": market_data.get("volatility_rank"),
            "range_position": market_data.get("range_position"),
            "atr": market_data.get("atr"),
            "rsi": market_data.get("rsi"),
            "rag_weighted_win_rate": rag_result.weighted_win_rate,
            "rag_similar_trades": rag_result.similar_trade_count,
            "decision": decision.action,
            "confidence": decision.confidence,
            "historical_avg_trend": historical_summary.get("avg_trend_strength"),
        }
        try:
            self.trade_logger.record_market_metrics(payload)
        except Exception as exc:
            logger.debug(f"Failed to persist market metrics: {exc}")

    def _load_recent_metrics(self, limit: int = 30) -> Dict[str, Any]:
        """Load summarized metrics from persistence for structural awareness."""
        if not self.trade_logger:
            return {}
        try:
            records = self.trade_logger.get_recent_market_metrics(limit=limit)
        except Exception as exc:
            logger.debug(f"Unable to load historical metrics: {exc}")
            return {}
        
        if not records:
            return {}
        
        def _avg(key: str) -> float:
            values = [row[key] for row in records if row.get(key) is not None]
            return fmean(values) if values else 0.0
        
        summary = {
            "sample_count": len(records),
            "avg_trend_strength": _avg("trend_strength"),
            "avg_volatility_rank": _avg("volatility_rank"),
            "avg_range_position": _avg("range_position"),
            "last_decision": records[0].get("decision"),
        }
        return summary
    
    def log_trade_entry(
        self,
        action: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        take_profit: float,
        market_data: Dict[str, Any],
        pipeline_result: Optional[HybridPipelineResult] = None,
    ) -> str:
        """Log a trade entry for RAG.
        
        Args:
            action: BUY or SELL
            entry_price: Entry price
            quantity: Position size
            stop_loss: Stop loss level
            take_profit: Take profit level
            market_data: Market data at entry
            pipeline_result: Pipeline result for context
            
        Returns:
            Trade ID
        """
        trade_id = self.trade_logger.log_entry(
            action=action,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            market_data=market_data,
            pipeline_result=pipeline_result,
        )
        
        self._current_trade_id = trade_id
        return trade_id
    
    def log_trade_exit(
        self,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[Any]:
        """Log a trade exit and analyze if it was a loss.
        
        Args:
            exit_price: Exit price
            exit_reason: Reason for exit
            
        Returns:
            Trade record or None
        """
        if not self._current_trade_id:
            logger.warning("No current trade ID for exit logging")
            return None
        
        trade = self.trade_logger.log_exit(
            trade_id=self._current_trade_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )
        
        # Analyze if it was a loss
        if trade and trade.result == "LOSS":
            logger.info(f"Analyzing losing trade {trade.trade_id}")
            self.mistake_analyzer.save_mistake_note(trade)
        
        self._current_trade_id = None
        return trade
    
    def record_trade_for_cooldown(self) -> None:
        """Record that a trade was executed (for pipeline cooldown)."""
        self.pipeline.record_trade()
    
    async def run_daily_update(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the daily RAG update process.
        
        Args:
            market_data: End of day market data
            
        Returns:
            Update results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.daily_updater.run_daily_update,
            market_data,
            False,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline and RAG statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "enabled": self.enabled,
            "trade_logger_active_trades": len(self.trade_logger.get_active_trades()),
        }
        
        # Add storage stats
        if self.storage:
            trade_stats = self.storage.get_trade_stats(days=30)
            stats["rag_trade_stats"] = trade_stats
        
        # Add embedding stats
        if self.embedding_builder:
            stats["embedding_stats"] = self.embedding_builder.get_stats()
        
        # Add mistake summary
        if self.mistake_analyzer:
            stats["mistake_summary"] = self.mistake_analyzer.get_mistake_summary(days=7, min_trades=3)
        
        return stats


def create_hybrid_integration(
    settings: Any,
    llm_client: Optional[Any] = None,
    context_bus: Optional[AgentBus] = None,
) -> HybridPipelineIntegration:
    """Factory function to create HybridPipelineIntegration.
    
    Args:
        settings: Application settings
        llm_client: Optional Bedrock client
        
    Returns:
        HybridPipelineIntegration instance
    """
    # Check if hybrid is enabled in settings
    enabled = True
    if hasattr(settings, 'hybrid'):
        enabled = getattr(settings.hybrid, 'enabled', True)
    
    return HybridPipelineIntegration(
        settings=settings,
        llm_client=llm_client,
        enabled=enabled,
        context_bus=context_bus,
    )
