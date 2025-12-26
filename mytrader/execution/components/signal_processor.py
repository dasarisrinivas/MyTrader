"""Signal processing pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ...config import Settings
from ...strategies.engine import StrategyEngine
from ...strategies.trading_filters import calculate_enhanced_confidence
from ...utils.logger import logger
from ...utils.structured_logging import log_structured_event
from ...utils.timezone_utils import now_cst
from ...features.feature_engineer import engineer_features
from .trade_decision_engine import TradingContext

if TYPE_CHECKING:  # pragma: no cover
    from ..live_trading_manager import LiveTradingManager


@dataclass
class SignalGenerationResult:
    """Container for signal pipeline output."""

    signal: Any
    pipeline_result: Any = None
    filters_passed: bool = True
    filters_applied: List[str] = field(default_factory=list)
    run_legacy_after_hybrid: bool = False


class EmergencySignalGenerator:
    """Force a directional signal after too many consecutive HOLDs."""

    def __init__(self, max_consecutive_holds: int = 10):
        self.consecutive_holds = 0
        self.max_consecutive_holds = max_consecutive_holds

    def apply(self, signal: Any, market_context: Dict[str, Any]) -> Any:
        action = getattr(signal, "action", None)
        if action == "HOLD":
            self.consecutive_holds += 1
        else:
            self.consecutive_holds = 0

        if self.consecutive_holds >= self.max_consecutive_holds:
            trend = market_context.get("trend", "UNKNOWN")
            logger.warning("🚨 Emergency mode: %s consecutive HOLDs (trend=%s)", self.consecutive_holds, trend)

            new_action: Optional[str] = None
            if trend in ["DOWNTREND", "WEAK_DOWN", "MICRO_DOWN"]:
                new_action = "SELL"
            elif trend in ["UPTREND", "WEAK_UP", "MICRO_UP"]:
                new_action = "BUY"

            if new_action:
                signal.action = new_action
                signal.confidence = max(getattr(signal, "confidence", 0.0), 0.25)
                metadata = getattr(signal, "metadata", {})
                metadata = metadata if isinstance(metadata, dict) else {}
                metadata.update(
                    {
                        "emergency_mode": True,
                        "emergency_reason": f"{self.consecutive_holds} consecutive HOLDs",
                        "emergency_trend": trend,
                    }
                )
                signal.metadata = metadata
                self.consecutive_holds = 0  # reset after firing

        return signal


class SignalProcessor:
    """Executes strategy and feature processing."""

    def __init__(
        self,
        settings: Settings,
        engine: Optional[StrategyEngine],
        manager: "LiveTradingManager",
    ):
        self.settings = settings
        self.engine = engine
        self.manager = manager
        # Hybrid pipeline can be injected later; default to manager's instance
        self.hybrid_pipeline = getattr(manager, "hybrid_pipeline", None)
        self._emergency_generator = EmergencySignalGenerator()

    async def generate_trading_signal(
        self,
        features,
        returns,
        current_price: float,
        structural_metrics: Dict[str, float],
    ) -> Optional[SignalGenerationResult]:
        """Generate a trading signal (hybrid or legacy) and confidence adjustments."""
        m = self.manager

        # === Hybrid RAG+LLM Pipeline ===
        if m._use_hybrid_pipeline and (self.hybrid_pipeline or m.hybrid_pipeline):
            pipeline = self.hybrid_pipeline or m.hybrid_pipeline
            try:
                position_for_pipeline = None
                if m.executor:
                    try:
                        position_for_pipeline = await m.executor.get_current_position()
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(f"Unable to fetch position for hybrid pipeline: {exc}")
                if hasattr(pipeline, "set_current_position"):
                    pipeline.set_current_position(position_for_pipeline)
                hybrid_signal, pipeline_result = await pipeline.process(
                    features,
                    current_price,
                    position_for_pipeline,
                )

                # Update status with hybrid pipeline info
                if pipeline_result:
                    m.context_manager.refresh_hybrid_context(pipeline_result)

                # Log hybrid pipeline decision
                logger.info(
                    f"🤖 Hybrid Pipeline: {hybrid_signal.action} "
                    f"(conf={hybrid_signal.confidence:.2f}, "
                    f"trend={m.status.hybrid_market_trend}, "
                    f"vol={m.status.hybrid_volatility_regime})"
                )
                log_structured_event(
                    agent="live_manager",
                    event_type="hybrid.signal",
                    message=f"{hybrid_signal.action} {hybrid_signal.confidence:.2f}",
                    payload={
                        "trend": m.status.hybrid_market_trend,
                        "volatility": m.status.hybrid_volatility_regime,
                        "metadata": hybrid_signal.metadata,
                    },
                )

                # Emergency mode: force a directional signal after too many HOLDs
                market_ctx = self._build_market_context_from_pipeline(pipeline_result)
                hybrid_signal = self._emergency_generator.apply(hybrid_signal, market_ctx)

                # Store pipeline result for trade logging
                m._current_pipeline_result = pipeline_result

                # Skip legacy RAG and filter processing unless fallback explicitly allowed
                run_legacy_after_hybrid = (
                    hybrid_signal.action == "HOLD"
                    and hybrid_signal.confidence <= 0
                    and m._allow_hybrid_legacy_fallback
                )
                if not run_legacy_after_hybrid:
                    return SignalGenerationResult(
                        signal=hybrid_signal,
                        pipeline_result=pipeline_result,
                        filters_passed=True,
                        filters_applied=[],
                        run_legacy_after_hybrid=False,
                    )
                logger.info("ℹ️  Hybrid HOLD detected; legacy evaluation allowed per config")
            except Exception as exc:  # noqa: BLE001
                await m._handle_hybrid_pipeline_failure(current_price, exc)
                return None

        # === Legacy path ===
        if not self.engine:
            logger.warning("Strategy engine not initialized; forcing HOLD")
            return SignalGenerationResult(
                signal=SimpleNamespace(
                    action="HOLD", confidence=0.0, metadata={"error": "Engine not ready"}
                ),
                filters_passed=True,
                filters_applied=[],
            )

        signal = self.engine.evaluate(features, returns)
        structure_bonus = m._apply_structural_weighting(signal, structural_metrics)
        if structure_bonus != 0:
            signal.confidence = max(0.0, min(1.0, signal.confidence + structure_bonus))
            logger.info(
                f"🧱 Structural bias applied: {structure_bonus:+.3f} "
                f"(trend={structural_metrics.get('trend_strength', 0):+.4f}, "
                f"momentum={structural_metrics.get('momentum_score', 0):+.4f})"
            )
            if hasattr(signal, "metadata"):
                metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
                metadata.update(
                    {
                        "structure_bonus": structure_bonus,
                        "structure_trend": structural_metrics.get("trend_strength"),
                        "structure_momentum": structural_metrics.get("momentum_score"),
                        "structure_range_position": structural_metrics.get("range_position"),
                    }
                )
                signal.metadata = metadata

        signal, filters_passed, filters_applied = await self._apply_confidence_layers(
            signal,
            features,
            current_price,
            structural_metrics,
        )

        # === Minimum confidence threshold check ===
        if signal.confidence < m._min_confidence_for_trade and signal.action != "HOLD":
            logger.info(
                f"🔽 Signal confidence {signal.confidence:.3f} below threshold "
                f"{m._min_confidence_for_trade:.3f}, converting to HOLD"
            )
            signal.action = "HOLD"

        # Emergency mode for legacy path
        market_ctx = self._build_market_context_from_features(features)
        signal = self._emergency_generator.apply(signal, market_ctx)

        m.status.last_signal = signal.action
        m.status.signal_confidence = signal.confidence

        return SignalGenerationResult(
            signal=signal,
            pipeline_result=None,
            filters_passed=filters_passed,
            filters_applied=filters_applied,
            run_legacy_after_hybrid=False,
        )

    def calculate_confidence(self, signal_data: Dict[str, float]) -> float:
        """Aggregate confidence adjustments."""
        base_conf = float(signal_data.get("base_confidence", 0.0))
        adjustments = signal_data.get("adjustments", [])
        if not isinstance(adjustments, (list, tuple)):
            adjustments = [float(signal_data.get("rag_adjustment", 0.0))]
            adjustments.append(float(signal_data.get("aws_kb_adjustment", 0.0)))
        total = base_conf + sum(float(adj) for adj in adjustments)
        return max(0.0, min(1.0, total))

    def apply_trading_filters(
        self,
        signal,
        features,
        current_price: float,
    ) -> Tuple[bool, List[str], float]:
        """Apply configured trading filters and enhanced confidence."""
        m = self.manager
        filters_passed = True
        filters_applied: List[str] = []
        enhanced_conf = signal.confidence

        if m.trading_filters is None:
            try:
                m.trading_filters = m._build_trading_filters()
                logger.info("✅ Trading filters initialized from config overrides")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"⚠️ Could not initialize trading filters: {exc}")

        if m.trading_filters and signal.action != "HOLD":
            try:
                if m.trading_filters._levels is None or len(features) > 200:
                    m.trading_filters.set_historical_data(features)

                filter_result = m.trading_filters.evaluate(
                    current_price=current_price,
                    proposed_action=signal.action,
                    features=features,
                )

                filters_passed = filter_result.can_trade
                filters_applied = filter_result.reasons

                calc_conf, conf_reasons = calculate_enhanced_confidence(
                    base_confidence=signal.confidence,
                    features=features,
                    action=signal.action,
                    price_levels=filter_result.levels,
                    current_price=current_price,
                )
                enhanced_conf = max(signal.confidence, calc_conf)
                enhanced_conf = min(1.0, max(0.0, enhanced_conf + filter_result.confidence_adjustment))

                if not filters_passed:
                    logger.warning(f"🚫 Signal BLOCKED by filters: {filter_result.reasons}")
                else:
                    logger.info(f"✅ Filters PASSED: {filters_applied}")
                    if conf_reasons:
                        logger.info(f"   Confidence factors: {conf_reasons}")
                    logger.info(
                        "   Enhanced confidence: %.3f -> %.3f (filter adj %+0.3f => %.3f)",
                        signal.confidence,
                        max(signal.confidence, calc_conf),
                        filter_result.confidence_adjustment,
                        enhanced_conf,
                    )

                m.status.filters_applied = filters_applied
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Filter evaluation error: {exc}")
                filters_passed = True

        return filters_passed, filters_applied, enhanced_conf

    async def process_trading_cycle(self, current_price: float):
        """Full trading cycle orchestration."""
        import pandas as pd
        import uuid
        from ...utils.timezone_utils import format_cst, utc_to_cst

        m = self.manager

        # === Cooldown check ===
        if m._last_trade_time:
            elapsed = (now_cst() - m._last_trade_time).total_seconds()
            cooldown_remaining = m._cooldown_seconds - elapsed
            if cooldown_remaining > 0:
                m.status.cooldown_remaining_seconds = int(cooldown_remaining)
                last_trade_display = format_cst(utc_to_cst(m._last_trade_time))
                logger.info(
                    "Skipping trade due to cooldown; %ds remaining (last trade %s)",
                    int(cooldown_remaining),
                    last_trade_display,
                )
                await m._broadcast_status()
                return
            m.status.cooldown_remaining_seconds = 0

        if m.executor and m.executor.is_order_locked():
            m.status.pending_order = True
            m.status.order_lock_reason = m.executor.get_order_lock_reason() or "Awaiting bracket confirmation"
            lock_age_sec = 0.0
            get_lock_age = getattr(m.executor, "get_order_lock_age_seconds", None)
            if callable(get_lock_age):
                try:
                    lock_age_sec = float(get_lock_age())
                except Exception:
                    lock_age_sec = 0.0
            logger.warning(
                "Skipping trade; order lock active for %.1fs (%s)",
                lock_age_sec,
                m.status.order_lock_reason,
            )
            await m._broadcast_status()
            return
        m.status.pending_order = False
        m.status.order_lock_reason = ""

        now = now_cst()
        current_candle_start = now.replace(second=0, microsecond=0)
        if m._last_candle_processed == current_candle_start:
            logger.debug("⏳ Waiting for next candle close (current minute already processed)")
            return
        m._last_candle_processed = current_candle_start
        logger.info("🕐 New candle close at {} CST", current_candle_start.strftime("%H:%M:%S"))

        cycle_id = uuid.uuid4().hex[:12]
        m._current_cycle_id = cycle_id
        m._cycle_context[cycle_id] = {
            "start_time": current_candle_start.isoformat(),
            "signal_type": None,
            "signal_confidence": None,
            "regime": None,
            "volatility": None,
            "reason_codes": set(),
            "aws": None,
        }
        log_structured_event(
            agent="live_manager",
            event_type="trade.cycle.start",
            message="Cycle start",
            payload={"price": current_price},
            correlation_id=cycle_id,
        )

        m._refresh_external_context()
        df = pd.DataFrame(m.price_history)
        df.set_index("timestamp", inplace=True)

        features = engineer_features(df[["open", "high", "low", "close", "volume"]], None)
        if features.empty:
            m.status.message = "Feature engineering returned empty"
            await m._broadcast_status()
            return

        returns = features["close"].pct_change().dropna()
        m._publish_feature_snapshot(features, current_price)
        m._publish_account_context()
        structural_metrics = m._compute_structural_metrics(features)

        signal_result = await self.generate_trading_signal(
            features=features,
            returns=returns,
            current_price=current_price,
            structural_metrics=structural_metrics,
        )
        if not signal_result or not signal_result.signal:
            return

        if signal_result.pipeline_result and not signal_result.run_legacy_after_hybrid:
            await m._process_hybrid_signal(
                signal_result.signal,
                signal_result.pipeline_result,
                current_price,
                features,
            )
            return

        signal = signal_result.signal
        filters_passed = signal_result.filters_passed
        filters_applied = signal_result.filters_applied

        cycle_ctx = m._cycle_context.get(m._current_cycle_id, {})
        cycle_ctx["signal_type"] = signal.action
        cycle_ctx["signal_confidence"] = signal.confidence
        cycle_ctx["regime"] = cycle_ctx.get("regime") or getattr(m.status, "hybrid_market_trend", None)
        cycle_ctx["volatility"] = cycle_ctx.get("volatility") or getattr(m.status, "hybrid_volatility_regime", None)

        await m._broadcast_signal(signal, current_price)

        cycle_ctx = m._cycle_context.get(m._current_cycle_id, {})
        cycle_ctx["signal_type"] = signal.action
        cycle_ctx["signal_confidence"] = signal.confidence
        cycle_ctx["regime"] = m.status.hybrid_market_trend
        cycle_ctx["volatility"] = m.status.hybrid_volatility_regime

        current_position = await m.executor.get_current_position()
        m.status.current_position = current_position.quantity if current_position else 0
        m.status.active_orders = m.executor.get_active_order_count()

        if current_position:
            m.status.unrealized_pnl = await m.executor.get_unrealized_pnl()
            m.tracker.update_equity(current_price, realized_pnl=0.0)
            m._update_status_from_tracker()

            atr_val = float(features.iloc[-1].get("ATR_14", 0.0))
            await m.executor.update_trailing_stops(current_price, atr_val)
            await m._log_position_status(current_position, current_price)

        await m._broadcast_status()

        active_orders = m.executor.get_active_order_count(sync=True)
        context = TradingContext(
            filters_passed=filters_passed,
            filters_applied=filters_applied,
            active_orders=active_orders,
            current_position=current_position,
            min_confidence=m._min_confidence_for_trade,
        )
        decision = m.trade_decision_engine.should_enter_trade(signal, context)

        if decision.exit_only and current_position:
            exit_qty = decision.exit_quantity or abs(current_position.quantity)
            logger.info(
                "  ↳ EXIT SIGNAL: Position=%s, Signal=%s, closing position",
                current_position.quantity,
                signal.action,
            )
            await m._place_exit_order(signal.action, exit_qty, current_price)
            return

        if not decision.allow:
            logger.info("Skipping trade (%s)", decision.reason or "blocked")
            return

        logger.info("  ↳ Attempting to place order: %s", signal.action)
        await m._place_order(signal, current_price, features)

    async def _apply_confidence_layers(
        self,
        signal,
        features,
        current_price: float,
        structural_metrics: Dict[str, float],
    ) -> Tuple[Any, bool, List[str]]:
        """Apply RAG, AWS KB, and trading filter adjustments."""
        m = self.manager
        rag_adjustment = 0.0
        rag_rationale: Dict[str, Any] = {}

        try:
            rag_rationale = m.context_manager.fetch_rag_context(features, signal.action, structural_metrics) or {}
            stats = rag_rationale.get("stats") or {}
            buckets = rag_rationale.get("buckets") or {}
            win_rate = stats.get("win_rate", 0.0)
            count = stats.get("count", 0)

            if count >= 5:
                if win_rate > 0.6:
                    rag_adjustment = 0.1
                    rag_rationale["adjustment"] = "Positive history"
                elif win_rate < 0.4:
                    rag_adjustment = -0.2
                    rag_rationale["adjustment"] = "Negative history"

            if buckets:
                m.current_trade_buckets = buckets
                m.current_trade_rationale = rag_rationale
                m.save_snapshot(features.iloc[-1], buckets)

        except Exception as exc:  # noqa: BLE001
            logger.error(f"RAG retrieval failed: {exc}")

        aws_kb_adjustment = 0.0
        if m._aws_agents_allowed and signal.action != "HOLD":
            try:
                aws_kb_result = await m._query_aws_knowledge_base(
                    features=features,
                    current_price=current_price,
                    proposed_action=signal.action,
                )

                if aws_kb_result:
                    aws_kb_adjustment = aws_kb_result.get("confidence_adjustment", 0.0)

                    if aws_kb_adjustment != 0:
                        logger.info(
                            f"🤖 AWS KB adjustment: {aws_kb_adjustment:+.2f} "
                            f"(similar_patterns={aws_kb_result.get('similar_patterns', 0)}, "
                            f"historical_win_rate={aws_kb_result.get('historical_win_rate', 0):.1%})"
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"AWS Knowledge Base query failed: {exc}")

        original_confidence = signal.confidence
        signal.confidence = self.calculate_confidence(
            {
                "base_confidence": original_confidence,
                "adjustments": [rag_adjustment, aws_kb_adjustment],
            }
        )

        logger.info(
            "📊 Signal: action=%s, confidence=%.3f (original=%.3f, rag_adj=%+.3f, aws_kb_adj=%+.3f)",
            signal.action,
            signal.confidence,
            original_confidence,
            rag_adjustment,
            aws_kb_adjustment,
        )

        if rag_adjustment != 0:
            logger.info(
                "RAG adjusted confidence: %.2f -> %.2f (%s)",
                original_confidence,
                signal.confidence,
                rag_rationale.get("adjustment"),
            )

        m._persist_structural_snapshot(structural_metrics, rag_rationale, signal)

        filters_passed, filters_applied, enhanced_conf = self.apply_trading_filters(
            signal=signal,
            features=features,
            current_price=current_price,
        )
        signal.confidence = enhanced_conf
        return signal, filters_passed, filters_applied

    def _build_market_context_from_pipeline(self, pipeline_result: Any) -> Dict[str, Any]:
        if not pipeline_result:
            return {}
        return {
            "trend": getattr(pipeline_result.rule_engine, "market_trend", None),
            "volatility": getattr(pipeline_result.rule_engine, "volatility_regime", None),
        }

    def _build_market_context_from_features(self, features) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        try:
            if hasattr(features, "iloc") and len(features) > 0:
                last = features.iloc[-1]
                context["trend"] = (
                    last.get("trend")
                    or last.get("market_trend")
                    or last.get("regime")
                )
                context["volatility"] = last.get("volatility_regime") or last.get("volatility")
        except Exception:
            pass
        return context
