"""Signal processing pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
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

# FEB 2026: Scoring system SUNSET — no longer used with 15m strategy
SCORING_AVAILABLE = False

# NEW: Multi-source sentiment aggregator (Jan 2026)
try:
    from ...data.sentiment_aggregator import (
        evaluate_sentiment_for_entry as multi_evaluate_entry,
        evaluate_sentiment_for_position as multi_evaluate_position,
        get_combined_mes_sentiment,
        get_mes_sentiment_score,
        set_cache_refresh_interval as multi_set_cache_interval,
        SentimentDecision,
        CombinedSentiment,
        # JAN 8 2026: Sentiment-derived trend
        get_sentiment_trend,
        combine_sentiment_with_technical_trend,
        SentimentTrend,
    )
    MULTI_SOURCE_SENTIMENT_AVAILABLE = True
except ImportError:
    MULTI_SOURCE_SENTIMENT_AVAILABLE = False

# LEGACY: Single-source Stocktwits sentiment (Jan 2026)
try:
    from ...data.stocktwits_sentiment import (
        evaluate_sentiment_for_entry,
        evaluate_sentiment_for_position,
        get_mes_sentiment,
        set_cache_refresh_interval,
    )
    STOCKTWITS_SENTIMENT_AVAILABLE = True
except ImportError:
    STOCKTWITS_SENTIMENT_AVAILABLE = False

# NEW: VX Futures Feed for volatility-based position sizing (Jan 2026)
try:
    from ...data.vx_futures_feed import (
        VxFuturesFeed,
        get_vx_feed,
        init_vx_feed,
        shutdown_vx_feed,
    )
    VX_FEED_AVAILABLE = True
except ImportError:
    VX_FEED_AVAILABLE = False
    VxFuturesFeed = None

# NEW: Multi-timeframe support (Jan 2026)
try:
    from ...data.candle_aggregator import MultiTimeframeCandleBuilder, MTFCandleManager
    MTF_AVAILABLE = True
except ImportError:
    MTF_AVAILABLE = False
    MultiTimeframeCandleBuilder = None
    MTFCandleManager = None

# NEW: MTF Trend Gate with state machine (Jan 12, 2026)
try:
    from .mtf_trend_gate import MTFTrendGate, TradingState
    MTF_GATE_AVAILABLE = True
except ImportError:
    MTF_GATE_AVAILABLE = False
    MTFTrendGate = None
    TradingState = None

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
    sentiment_modifier: Any = None  # SentimentDecisionModifier if applied


class EmergencySignalGenerator:
    """Monitor consecutive HOLDs and log warnings - ADVISORY ONLY.
    
    REVIEW FIX (Jan 2026): Changed from forcing trades to advisory-only mode.
    
    Previously, this class would force BUY/SELL after 10 consecutive HOLDs,
    which was identified as dangerous in range-bound markets (could trigger
    trades in unfavorable conditions). Now it only:
    1. Logs warnings when threshold is reached
    2. Adds metadata for analysis
    3. Does NOT override the strategy's HOLD decision
    
    Rationale: Forcing trades after 10 minutes of inactivity is too aggressive.
    Missing a trend is better than forcing a wrong trade. If activity is needed,
    implement a volatility breakout trigger instead of blind forced entry.
    """

    def __init__(self, max_consecutive_holds: int = 50):  # Increased from 10 to 50
        self.consecutive_holds = 0
        self.max_consecutive_holds = max_consecutive_holds
        self._warning_logged = False

    def apply(self, signal: Any, market_context: Dict[str, Any]) -> Any:
        action = getattr(signal, "action", None)
        if action == "HOLD":
            self.consecutive_holds += 1
        else:
            self.consecutive_holds = 0
            self._warning_logged = False

        # ADVISORY ONLY: Log warning but do NOT force trades
        if self.consecutive_holds >= self.max_consecutive_holds and not self._warning_logged:
            trend = market_context.get("trend", "UNKNOWN")
            adx = market_context.get("adx", 0)
            atr = market_context.get("atr", 0)
            
            logger.warning(
                f"⚠️ ADVISORY: {self.consecutive_holds} consecutive HOLDs "
                f"(trend={trend}, ADX={adx:.1f}, ATR={atr:.2f})"
            )
            logger.warning(
                "   Strategy is inactive - market may be range-bound or filters too strict. "
                "Review conditions if this persists."
            )
            
            # Add metadata for analysis but do NOT change the action
            metadata = getattr(signal, "metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            metadata.update({
                "consecutive_holds_warning": True,
                "consecutive_holds_count": self.consecutive_holds,
                "market_trend": trend,
                "advisory_only": True,  # Flag that we did NOT force a trade
            })
            signal.metadata = metadata
            self._warning_logged = True  # Only log once per streak

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

        # MAR 10 2026: Strategy counter rollback on blocked signals
        self._original_signal_reason: str = ""
        
        # NEW: Multi-timeframe candle builder (Jan 2026) - single 5m aggregator
        self._mtf_builder: Optional[MultiTimeframeCandleBuilder] = None
        
        # NEW: Full MTF Candle Manager for 5m, 15m, 30m (Jan 12, 2026)
        self._mtf_manager: Optional[MTFCandleManager] = None
        
        # NEW: MTF Trend Gate with state machine (Jan 12, 2026)
        self._mtf_gate: Optional[MTFTrendGate] = None

        # MTF gate manager helper (extracted for size reduction)
        from .mtf_gate_manager import MTFGateManager
        self._mtf = MTFGateManager(self)
        self._mtf.init_mtf_builder()
        self._mtf.init_mtf_manager()
        self._mtf.init_mtf_gate()
        # Expose references for backward compat
        self._mtf_builder = self._mtf._mtf_builder
        self._mtf_manager = self._mtf._mtf_manager
        self._mtf_gate = self._mtf._mtf_gate

        # Seed MTF trend gate from IB historical bars so we don't stay UNKNOWN for 30-60 minutes
        self._mtf_bootstrap_started: bool = False
        self._mtf_bootstrap_task = None
        
        # NEW: Multi-source sentiment configuration (Jan 2026)
        self._multi_source_enabled = False
        self._multi_source_config = None
        # LEGACY: Single-source Stocktwits sentiment
        self._stocktwits_enabled = False
        self._stocktwits_config = None
        
        # Sentiment + VX feed helper (extracted for size reduction)
        from .sentiment_evaluator import SentimentEvaluator
        self._sentiment = SentimentEvaluator(self)
        self._sentiment.init_sentiment()
        # Expose flags for backward compat
        self._multi_source_enabled = self._sentiment._multi_source_enabled
        self._multi_source_config = self._sentiment._multi_source_config
        self._stocktwits_enabled = self._sentiment._stocktwits_enabled
        self._stocktwits_config = self._sentiment._stocktwits_config
        
        # NEW: VX Futures Feed for volatility-based sizing (Jan 2026)
        self._vx_feed: Optional[VxFuturesFeed] = None
        self._vx_feed_enabled = False
        self._sentiment.init_vx_feed()
        self._vx_feed = self._sentiment._vx_feed
        self._vx_feed_enabled = self._sentiment._vx_feed_enabled
    
    def _init_mtf_manager(self) -> None:
        """Initialize MTFCandleManager (delegated to MTFGateManager)."""
        self._mtf.init_mtf_manager()
        self._mtf_manager = self._mtf._mtf_manager

    def _init_mtf_gate(self) -> None:
        """Initialize MTFTrendGate (delegated to MTFGateManager)."""
        self._mtf.init_mtf_gate()
        self._mtf_gate = self._mtf._mtf_gate

    def _ensure_mtf_gate_bootstrap(self) -> None:
        """Kick off MTF gate bootstrap (delegated to MTFGateManager)."""
        self._mtf.ensure_mtf_gate_bootstrap()
        self._mtf_bootstrap_started = self._mtf._mtf_bootstrap_started
        self._mtf_bootstrap_task = self._mtf._mtf_bootstrap_task

    async def _bootstrap_mtf_gate_from_ib(self) -> None:
        """Bootstrap MTF gate from IB historical bars (delegated to MTFGateManager)."""
        await self._mtf._bootstrap_mtf_gate_from_ib()

    def _init_sentiment(self) -> None:
        """Initialize sentiment integration (delegated to SentimentEvaluator)."""
        self._sentiment.init_sentiment()
    
    def _init_multi_source_sentiment(self, cfg) -> None:
        """Initialize multi-source sentiment aggregator (delegated)."""
        self._sentiment._init_multi_source_sentiment(cfg)
    
    def _init_stocktwits_sentiment(self, cfg) -> None:
        """Initialize legacy Stocktwits-only sentiment (delegated)."""
        self._sentiment._init_stocktwits_sentiment(cfg)
    
    def _init_vx_feed(self) -> None:
        """Initialize VX futures feed (delegated to SentimentEvaluator)."""
        self._sentiment.init_vx_feed()
        self._vx_feed = self._sentiment._vx_feed
        self._vx_feed_enabled = self._sentiment._vx_feed_enabled
    
    def _get_vx_multiplier(self) -> float:
        """Get the current VX-based volatility multiplier."""
        return self._sentiment.get_vx_multiplier()
    
    def shutdown_vx_feed(self) -> None:
        """Shutdown the VX futures feed (call on cleanup)."""
        self._sentiment.shutdown_vx_feed()
        self._vx_feed = None
        self._vx_feed_enabled = False
    
    def _get_current_session(self) -> str:
        """Determine current trading session based on CST time.
        
        Returns:
            'RTH' for Regular Trading Hours (8:30 AM - 3:00 PM CT)
            'EVENING' for Evening Session (5:00 PM - 11:00 PM CT)
            'OVERNIGHT' for Overnight (11:00 PM - 8:30 AM CT)
        """
        now = now_cst()
        minutes = now.hour * 60 + now.minute
        
        # RTH: 8:30 AM - 3:00 PM CT (510 - 900 minutes)
        if 510 <= minutes <= 900:
            return "RTH"
        # Evening: 5:00 PM - 11:00 PM CT (1020 - 1380 minutes)
        elif 1020 <= minutes <= 1380:
            return "EVENING"
        # Overnight: everything else
        else:
            return "OVERNIGHT"
    
    def _is_low_volume_session(self) -> bool:
        """Check if current session is low-volume (evening or overnight)."""
        session = self._get_current_session()
        return session in ("EVENING", "OVERNIGHT")

    def _evaluate_sentiment(self, signal: Any) -> Tuple[Any, Optional[Any]]:
        """Evaluate sentiment and apply to signal (delegated to SentimentEvaluator)."""
        return self._sentiment.evaluate_sentiment(signal)
    
    def _evaluate_multi_source_sentiment(self, signal: Any) -> Tuple[Any, Optional[Any]]:
        """Evaluate multi-source sentiment (delegated)."""
        return self._sentiment._evaluate_multi_source_sentiment(signal)
    
    def _evaluate_stocktwits_sentiment(self, signal: Any) -> Tuple[Any, Optional[Any]]:
        """Evaluate legacy Stocktwits-only sentiment (delegated)."""
        return self._sentiment._evaluate_stocktwits_sentiment(signal)

    async def generate_trading_signal(
        self,
        features,
        returns,
        current_price: float,
        structural_metrics: Dict[str, float],
    ) -> Optional[SignalGenerationResult]:
        """Generate a trading signal (hybrid or legacy) and confidence adjustments."""
        m = self.manager

        # Live-bar staleness guard: compute staleness and block entries if stale.
        # Skip this check during bootstrap (when no price bars have been collected yet).
        last_bar_ts = getattr(m, "_last_price_bar_ts", None)
        
        # Only enforce staleness if we've collected bars (skip during bootstrap)
        if last_bar_ts is not None:
            try:
                now = now_cst()
                if isinstance(last_bar_ts, datetime):
                    staleness_seconds = (now - last_bar_ts).total_seconds()
                else:
                    staleness_seconds = float("inf")
            except Exception:
                staleness_seconds = float("inf")

            # Configurable threshold (one_minute.live_bar_stale_seconds) default to 120s
            # FEB 2026: Scale default by active timeframe (15m bars have 900s period)
            active_tf = getattr(m, "_active_timeframe", "1m")
            live_threshold = 1200 if active_tf == "15m" else 120
            try:
                one_min_cfg = getattr(self.settings, "one_minute", None) or {}
                if isinstance(one_min_cfg, dict):
                    live_threshold = int(one_min_cfg.get("live_bar_stale_seconds", live_threshold))
                else:
                    live_threshold = int(getattr(one_min_cfg, "live_bar_stale_seconds", live_threshold))
            except Exception:
                pass  # keep timeframe-derived default

            if staleness_seconds > live_threshold:
                # Block new entries; allow exits (exit logic runs elsewhere).
                logger.warning(
                    f"⚠️ Blocking new entries: latest live bar is stale ({staleness_seconds:.0f}s > {live_threshold}s)"
                )
                # Return an explicit HOLD signal with block reason metadata so callers can log/act
                hold_signal = SimpleNamespace(action="HOLD", confidence=0.0, metadata={})
                hold_signal.metadata = {
                    "block_reasons": ["STALE_LIVE_BARS"],
                    "staleness_seconds": staleness_seconds,
                    "live_threshold_seconds": live_threshold,
                }
                return SignalGenerationResult(signal=hold_signal, pipeline_result=None, filters_passed=False)

        # ══════════════════════════════════════════════════════════════
        # FEB 7 2026: STRATEGY-FIRST ARCHITECTURE (ALL TIMEFRAMES)
        #
        # The strategy engine (15m, 30m, or 1m) is ALWAYS the PRIMARY
        # decision maker (BUY/SELL/HOLD). The hybrid pipeline (RAG,
        # LLM, sentiments, VIX) is a CONFIDENCE OVERLAY only:
        #   - Strategy says BUY → hybrid can boost or dampen confidence
        #   - Strategy says HOLD → stays HOLD (hybrid cannot override)
        #
        # This ensures backtest ↔ live parity: the strategy's generate()
        # is the sole signal source, exactly as in the backtest engine.
        # ══════════════════════════════════════════════════════════════
        return await self._generate_strategy_first_signal(
            features=features,
            returns=returns,
            current_price=current_price,
            structural_metrics=structural_metrics,
        )

    # ------------------------------------------------------------------
    # FEB 7 2026: Strategy-First Signal Generation (ALL timeframes)
    # ------------------------------------------------------------------

    def _rollback_strategy_counter(self, signal_reason: str) -> None:
        """Roll back the strategy's per-day signal counter when a signal is
        blocked by a post-strategy gate (CHOP guard, confidence threshold, etc).

        MAR 10 2026: The strategy increments counters at signal-generation time
        (needed for same-bar max-per-day gating).  If the signal is subsequently
        blocked, the counter must be decremented so the daily slot is not wasted.
        """
        if not self.engine or not signal_reason:
            return
        for strat in self.engine.strategies:
            if hasattr(strat, "rollback_counter"):
                strat.rollback_counter(signal_reason)
                break

    async def _generate_strategy_first_signal(
        self,
        features,
        returns,
        current_price: float,
        structural_metrics,
    ) -> Optional[SignalGenerationResult]:
        """Generate signal with strategy as primary, hybrid as confidence overlay.

        Architecture:
          1. Strategy engine evaluates features → primary signal (BUY/SELL/HOLD)
          2. If the strategy says HOLD → return HOLD immediately (no override)
          3. If the strategy says BUY/SELL → gather hybrid context as confidence
             modifiers (sentiment, VIX, RAG similarity, scoring) but NEVER change
             the action or convert a BUY/SELL to HOLD
          4. Return the strategy signal with adjusted confidence + hybrid metadata

        This matches the validated backtest (130 trades, PF 1.91, Sharpe 9.40)
        where the strategy's generate() is the sole decision maker.
        """
        m = self.manager

        if not self.engine:
            logger.warning("Strategy engine not initialized; forcing HOLD")
            return SignalGenerationResult(
                signal=SimpleNamespace(
                    action="HOLD", confidence=0.0, metadata={"error": "Engine not ready"}
                ),
                filters_passed=True,
                filters_applied=[],
            )

        # ── Step 1: Strategy evaluates (the ONLY trade decision) ──────
        signal = self.engine.evaluate(features, returns)
        original_action = signal.action  # Save for safety check later
        logger.info(
            f"📊 Strategy Signal: {signal.action} "
            f"(conf={signal.confidence:.2f}, "
            f"meta={signal.metadata.get('reason', '') if isinstance(signal.metadata, dict) else ''})"
        )

        # Feed OR levels into dynamic support floor (strategy puts them in metadata)
        sig_meta = signal.metadata if isinstance(signal.metadata, dict) else {}
        _or_h = sig_meta.get("or_high", 0.0)
        _or_l = sig_meta.get("or_low", 0.0)
        dsf = getattr(m, "dynamic_support_floor", None)
        if dsf and (_or_h > 0 or _or_l > 0):
            dsf.update_or_levels(_or_h, _or_l)

        m.status.last_signal = signal.action
        m.status.signal_confidence = signal.confidence

        # If HOLD, return immediately — hybrid pipeline cannot override
        if signal.action == "HOLD":
            return SignalGenerationResult(
                signal=signal,
                pipeline_result=None,
                filters_passed=True,
                filters_applied=[],
            )

        # ── Step 2: Gather hybrid context as confidence overlay ───────
        # These are ADDITIVE confidence modifiers — they can boost or
        # dampen the signal but NEVER change the action.
        pipeline_result = None
        confidence_adjustments = {}
        base_confidence = signal.confidence

        # 2a. Sentiment overlay
        sentiment_modifier = None
        if (self._multi_source_enabled or self._stocktwits_enabled):
            try:
                signal, sentiment_modifier = self._evaluate_sentiment(signal)
                if sentiment_modifier is not None:
                    sent_score = 0.0
                    if hasattr(sentiment_modifier, "combined_score"):
                        sent_score = sentiment_modifier.combined_score
                        m._last_sentiment_score = sent_score
                        m._last_sentiment_bias = (
                            "BULLISH" if sent_score > 0.2 else
                            "BEARISH" if sent_score < -0.2 else "NEUTRAL"
                        )
                    elif hasattr(sentiment_modifier, "mes_sentiment"):
                        sent_score = sentiment_modifier.mes_sentiment
                        m._last_sentiment_score = sent_score
                        m._last_sentiment_bias = (
                            "BULLISH" if sent_score > 0.2 else
                            "BEARISH" if sent_score < -0.2 else "NEUTRAL"
                        )
                    confidence_adjustments["sentiment"] = sent_score
                    logger.info(f"📡 Sentiment overlay: score={sent_score:.3f} bias={m._last_sentiment_bias}")
            except Exception as exc:
                logger.debug(f"Sentiment overlay skipped: {exc}")

        # 2b. VIX volatility — ADDITIVE signal-type-aware adjustment
        #
        # MAR 3 2026: Replaced multiplicative scaling (conf × 0.7) with
        # additive adjustment.  The old system was the #1 signal killer:
        #   0.70 × 0.70 = 0.49 → below 0.50 threshold → DEAD.
        # 27 of 45 signals (60%) were dampened, 26 failed conf threshold.
        #
        # New system: VX price → additive conf adjustment, varies by signal
        # type because different signals have different VX edge profiles:
        #   - OR breakouts/breakdowns THRIVE in elevated VX (range expansion)
        #   - Trend continuation works well in moderate VX (trends extend)
        #   - Pullbacks get wider noise in high VX (more stop-outs)
        #
        # VX tiers (MES median VX is ~18-22):
        #   VX < 16:  low vol — narrow ranges, targets harder to reach
        #   16-22:    normal MES baseline — neutral
        #   22-28:    elevated — good for trends, cautious for pullbacks
        #   28-35:    high fear — reduce pullbacks/continuation
        #   35+:      extreme — meaningful reduction except breakouts
        signal_reason_for_vx = signal.metadata.get("reason", "") if isinstance(signal.metadata, dict) else ""
        is_or_breakout_for_vx = "OR_BREAK" in signal_reason_for_vx
        is_trend_cont_for_vx = "TREND_CONT" in signal_reason_for_vx
        is_pullback_for_vx = "_PB_" in signal_reason_for_vx
        vx_multiplier = self._get_vx_multiplier()
        # Derive VX price from multiplier (reverse the piecewise-linear curve)
        # We need the actual VX price for tier logic, not the old multiplier
        vx_price = None
        if self._sentiment._vx_feed_enabled and self._sentiment._vx_feed:
            try:
                raw_vx = self._sentiment._vx_feed.get_vx_price()
                if isinstance(raw_vx, (int, float)) and raw_vx > 0:
                    vx_price = float(raw_vx)
            except Exception:
                pass  # VX feed unavailable — skip additive adjustment

        if vx_price is not None:
            # Signal-type-aware additive VX adjustment
            if vx_price < 16:
                # Low vol: narrow ranges, targets harder to reach
                vx_adj = -0.08 if is_trend_cont_for_vx else -0.03
            elif vx_price < 22:
                # Normal MES baseline — neutral
                vx_adj = 0.0
            elif vx_price < 28:
                # Elevated: trends extend, breakouts are real, pullbacks riskier
                if is_or_breakout_for_vx:
                    vx_adj = 0.03
                elif is_trend_cont_for_vx:
                    vx_adj = 0.05
                elif is_pullback_for_vx:
                    vx_adj = -0.05
                else:
                    vx_adj = 0.0
            elif vx_price < 35:
                # High fear: genuine caution warranted
                if is_or_breakout_for_vx:
                    vx_adj = 0.0  # Breakouts still valid
                else:
                    vx_adj = -0.08
            else:
                # Extreme fear (VX >= 35): only breakouts trade normally
                if is_or_breakout_for_vx:
                    vx_adj = -0.03
                else:
                    vx_adj = -0.15

            if vx_adj != 0.0:
                original_conf = signal.confidence
                signal.confidence = max(0.10, signal.confidence + vx_adj)
                confidence_adjustments["vx_additive"] = vx_adj
                logger.info(
                    f"📈 VX Adjustment: VX={vx_price:.1f} | {original_conf:.3f} "
                    f"{vx_adj:+.3f} = {signal.confidence:.3f} "
                    f"({'breakout' if is_or_breakout_for_vx else 'trend_cont' if is_trend_cont_for_vx else 'pullback' if is_pullback_for_vx else 'other'})"
                )
            else:
                confidence_adjustments["vx_neutral"] = 0.0
                logger.info(
                    f"📈 VX Neutral: VX={vx_price:.1f} — no adjustment "
                    f"(signal_type={'breakout' if is_or_breakout_for_vx else 'trend_cont' if is_trend_cont_for_vx else 'pullback' if is_pullback_for_vx else 'other'})"
                )

        # 2c. Hybrid pipeline as advisory (RAG similarity + LLM reasoning)
        if m._use_hybrid_pipeline and (self.hybrid_pipeline or m.hybrid_pipeline):
            pipeline = self.hybrid_pipeline or m.hybrid_pipeline
            try:
                position_for_pipeline = None
                if m.executor:
                    try:
                        position_for_pipeline = await m.executor.get_current_position()
                    except Exception:
                        pass

                hybrid_signal, pipeline_result = await pipeline.process(
                    features, current_price, position_for_pipeline,
                )

                if pipeline_result:
                    m.context_manager.refresh_hybrid_context(pipeline_result)

                    # Extract advisory confidence from hybrid pipeline
                    hybrid_conf = getattr(hybrid_signal, "confidence", 0.0) or 0.0
                    hybrid_action = getattr(hybrid_signal, "action", "HOLD")

                    # Agreement bonus: hybrid agrees with strategy direction → boost
                    # Disagreement penalty: hybrid opposes → dampen (but never flip)
                    if hybrid_action == signal.action:
                        # Aligned — boost confidence by up to +0.15
                        boost = min(0.15, hybrid_conf * 0.2)
                        signal.confidence = min(1.0, signal.confidence + boost)
                        confidence_adjustments["hybrid_agreement_boost"] = boost
                        logger.info(
                            f"🤖 Hybrid AGREES ({hybrid_action}): "
                            f"conf boost +{boost:.3f} → {signal.confidence:.3f}"
                        )
                    elif hybrid_action == "HOLD":
                        # Hybrid is uncertain — light dampening
                        # FEB 10 2026: Raised from 0.05 to 0.10.
                        # MAR 3 2026: Reduced from 0.10 to 0.05. With additive VX
                        # scaling, the confidence budget is tighter. Hybrid uncertain
                        # should nudge, not meaningfully reduce. The strategy's own
                        # filters (EMA alignment, ADX, bullish bar) are the quality gate.
                        dampen = 0.05
                        signal.confidence = max(0.1, signal.confidence - dampen)
                        confidence_adjustments["hybrid_uncertain_dampen"] = -dampen
                        logger.info(
                            f"🤖 Hybrid UNCERTAIN (HOLD): "
                            f"conf dampen -{dampen:.3f} → {signal.confidence:.3f}"
                        )
                    else:
                        # Hybrid opposes — tiered dampening (but NEVER flip action)
                        # FEB 10 2026: Strengthened from min(0.15, hybrid_conf * 0.15)
                        # to min(0.30, hybrid_conf * 0.40).
                        # FEB 20 2026: Capped at 0.10. Audit showed the old -0.30 cap
                        # was blocking ~50% of viable signals.
                        # MAR 3 2026: Capped at 0.05. Log analysis (Feb 6 – Mar 3) showed
                        # hybrid oppose hit 18/45 signals (40%). Combined with VX scaling,
                        # this created: 0.70 × 0.70 − 0.10 = 0.39 (killed). With additive
                        # VX and 0.05 cap: 0.70 − 0.05 − 0.05 = 0.60 (healthy).
                        # MAR 17 2026: Tiered cap. When hybrid opposes with conf >= 0.45
                        # (a meaningful disagreement, not just noise), allow up to -0.10.
                        # Below 0.45 keep the light -0.05 touch. Trade 7180 on 2026-03-17
                        # entered BUY at PDH with hybrid SELL @ 0.50 conf — the flat 0.05
                        # cap left final conf exactly at threshold (0.60) and it fired.
                        if hybrid_conf >= 0.45:
                            dampen = min(0.10, hybrid_conf * 0.20)
                        else:
                            dampen = min(0.05, hybrid_conf * 0.20)
                        signal.confidence = max(0.1, signal.confidence - dampen)
                        confidence_adjustments["hybrid_oppose_dampen"] = -dampen
                        logger.info(
                            f"🤖 Hybrid OPPOSES ({hybrid_action} vs {signal.action}): "
                            f"conf dampen -{dampen:.3f} → {signal.confidence:.3f}"
                        )

                    # Extract useful metadata from pipeline
                    hybrid_meta = getattr(hybrid_signal, "metadata", {}) or {}
                    if isinstance(signal.metadata, dict):
                        signal.metadata["hybrid_advisory"] = {
                            "action": hybrid_action,
                            "confidence": hybrid_conf,
                            "trend": hybrid_meta.get("market_trend"),
                            "volatility_regime": hybrid_meta.get("volatility_regime"),
                            "rag_similar_trades": hybrid_meta.get("rag_similar_trades"),
                            "rag_win_rate": hybrid_meta.get("rag_weighted_win_rate"),
                        }

                    # Update status with hybrid context
                    if hasattr(pipeline_result, "rule_engine"):
                        trend = getattr(pipeline_result.rule_engine, "market_trend", None)
                        vol = getattr(pipeline_result.rule_engine, "volatility_regime", None)
                        if trend:
                            m.status.hybrid_market_trend = trend
                        if vol:
                            m.status.hybrid_volatility_regime = vol

                m._current_pipeline_result = pipeline_result
            except Exception as exc:
                logger.warning(f"⚠️ Hybrid confidence overlay failed (non-fatal): {exc}")

        # 2d. Local knowledge base overlay — apply win-rate adjustment from
        # similar historical trades recorded during this and prior sessions.
        # The local KB is populated on every trade close (order_coordinator.py)
        # so by the second trade of a session it already reflects intraday outcomes.
        if m._local_kb:
            try:
                kb_context = {
                    "action": signal.action,
                    "trend": getattr(m.status, "hybrid_market_trend", None),
                    "volatility": getattr(m.status, "hybrid_volatility_regime", None),
                    "confidence": signal.confidence,
                }
                kb_result = m._query_local_knowledge_base(kb_context)
                kb_adj = kb_result.get("confidence_adjustment", 0.0)
                if kb_adj != 0.0:
                    original_conf = signal.confidence
                    signal.confidence = max(0.10, min(1.0, signal.confidence + kb_adj))
                    confidence_adjustments["local_kb"] = kb_adj
                    logger.info(
                        f"📚 Local KB overlay: {kb_result.get('similar_patterns', 0)} similar trades "
                        f"win_rate={kb_result.get('historical_win_rate', 0):.0%} "
                        f"adj={kb_adj:+.3f} | {original_conf:.3f} → {signal.confidence:.3f} "
                        f"| {kb_result.get('reasoning', '')}"
                    )
            except Exception as exc:
                logger.debug(f"Local KB overlay skipped: {exc}")

        # ── Step 2d½: CHOP regime guard ─────────────────────────────
        #
        # History:
        #   FEB 22 2026: Binary block for all pullback/trend_cont in CHOP.
        #   MAR 2 2026:  Direction-aware redesign (LONGs dampen, SHORTs block).
        #   MAR 2 2026:  REVERTED to block-all after 1-year backtest (25,646 bars,
        #                226 CHOP-blocked trades) showed LONGs are ALSO net losers:
        #                  LONGs:  26.5% WR, −$146.25, PF 0.90
        #                  SHORTs: 25.0% WR, −$421.25, PF 0.79
        #                CIs overlap [18.8-36.0%] vs [18.3-33.2%].
        #                14-trade live sample (80% LONG WR) was survivorship bias.
        #                Block-all ($0) > direction-aware (−$146.25) > blind (−$567.50).
        #
        # ADX micro-edge (from 1-year data):
        #   ADX 20-25: 152 trades, −$642.50 (deeply negative)
        #   ADX 25-30:  55 trades, +$65.00  (slight edge)
        #   ADX 30+:    19 trades, +$10.00  (near flat)
        #
        # Optional exception framework (ENABLE_CHOP_EXCEPTION env var, OFF
        # by default): allows bidirectional exceptions through a statistically
        # controlled gate.  Must satisfy ALL conditions before any CHOP
        # signal passes:
        #   1. ADX >= 25
        #   2. Daily sentiment bias ALIGNED with direction (BULLISH→LONG, BEARISH→SHORT)
        #   3. Confidence >= 0.70
        #   4. ATR expanding (current ATR > ATR 5 bars ago)
        #
        # MAR 9 2026: Made bidirectional. Live data Mar 3-9 showed 3 SHORT
        # signals in CHOP were all TP_HIT (+$120). Gate 2 (bias alignment)
        # ensures SHORTs only pass with BEARISH sentiment confirmation.
        # Enabled by default in start_bot.sh as of MAR 9.
        #
        # MAR 24 2026: Extended to also block OR breakout/breakdown (B/E)
        # in CHOP. Evidence: 5/5 executed OR breakout fills were SL_HIT losers
        # (−$180.87 total). Breakouts in CHOP are failed breakouts — price
        # crosses OR level but cannot sustain, reverting into the range.
        # Today's B-long in CHOP: −$51.05 (hybrid correctly called CHOP
        # but the guard didn't block it).
        signal_reason = signal.metadata.get("reason", "") if isinstance(signal.metadata, dict) else ""
        is_pullback_signal = "_PB_" in signal_reason
        is_trend_cont_signal = "TREND_CONT" in signal_reason
        is_or_breakout_signal = "OR_BREAK" in signal_reason
        hybrid_trend = getattr(m.status, "hybrid_market_trend", None)

        if (is_pullback_signal or is_trend_cont_signal or is_or_breakout_signal) and hybrid_trend == "CHOP" and signal.action in ("BUY", "SELL", "SCALP_BUY", "SCALP_SELL"):
            block_type = "pullback" if is_pullback_signal else ("trend_cont" if is_trend_cont_signal else "or_breakout")
            is_long = signal.action in ("BUY", "SCALP_BUY")

            # ── Check exception framework (OFF by default) ────────
            # MAR 24 2026: OR breakout/breakdown signals are NOT eligible for
            # the CHOP exception — the 4-gate framework is designed for
            # directional pullbacks, not range-bound breakouts.
            chop_exception_enabled = os.environ.get("ENABLE_CHOP_EXCEPTION", "").lower() in ("1", "true", "yes")
            exception_passed = False

            if chop_exception_enabled and not is_or_breakout_signal:
                # MAR 9 2026: Made bidirectional (was LONG-only).
                # Live data Mar 3-9: 3 SHORT signals in CHOP were all TP_HIT (+$120).
                # Exception requires ALL 5 gates aligned with signal direction.
                #
                # Gate 1: ADX >= 25 (only ADX 25+ showed positive edge in 1yr data)
                adx_val = 0.0
                reason_parts = signal_reason.split("|")
                for part in reason_parts:
                    part = part.strip()
                    if part.startswith("ADX="):
                        try:
                            adx_val = float(part.split("=")[1])
                        except (ValueError, IndexError):
                            pass
                        break

                # Gate 2: Daily sentiment bias ALIGNED with direction
                # LONG requires BULLISH, SHORT requires BEARISH
                daily_bias = getattr(m, "_last_sentiment_bias", "NEUTRAL")
                bias_aligned = (
                    (is_long and daily_bias == "BULLISH")
                    or (not is_long and daily_bias == "BEARISH")
                )

                # Gate 3: Confidence >= 0.70 (pre-dampen)
                conf_ok = signal.confidence >= 0.70

                # Gate 4: ATR expanding (current ATR > ATR 5 bars ago)
                atr_expanding = False
                if features is not None and len(features) >= 6 and "ATR_14" in features.columns:
                    try:
                        atr_current = float(features.iloc[-1]["ATR_14"])
                        atr_5ago = float(features.iloc[-6]["ATR_14"])
                        atr_expanding = atr_current > atr_5ago and atr_5ago > 0
                    except (ValueError, IndexError, KeyError):
                        pass

                exception_passed = (
                    adx_val >= 25
                    and bias_aligned
                    and conf_ok
                    and atr_expanding
                )

                if exception_passed:
                    # Exception activated — dampen confidence but allow through
                    chop_exception_dampen = 0.05
                    old_conf = signal.confidence
                    signal.confidence = max(0.15, signal.confidence - chop_exception_dampen)
                    confidence_adjustments["chop_exception_pass"] = -chop_exception_dampen
                    direction_label = "LONG" if is_long else "SHORT"
                    logger.warning(
                        f"⚠️ CHOP Exception Activated: {direction_label} allowed | "
                        f"ADX={adx_val:.0f} | conf={old_conf:.3f}→{signal.confidence:.3f} | "
                        f"bias={daily_bias} | ATR_expanding={atr_expanding} | "
                        f"signal={signal_reason}"
                    )
                    if isinstance(signal.metadata, dict):
                        signal.metadata["chop_guard"] = {
                            "original_action": original_action,
                            "original_confidence": base_confidence,
                            "hybrid_trend": hybrid_trend,
                            "block_type": block_type,
                            "exception_activated": True,
                            "exception_adx": adx_val,
                            "exception_bias": daily_bias,
                            "exception_atr_expanding": atr_expanding,
                            "dampen_applied": chop_exception_dampen,
                        }

            if not exception_passed:
                # Default: BLOCK ALL pullback/trend_cont in CHOP
                confidence_adjustments["chop_regime_block"] = -signal.confidence
                logger.warning(
                    f"🚫 CHOP Block-All Guard Activated: "
                    f"blocking {signal.action} {block_type} signal "
                    f"({signal_reason}) in CHOP regime. "
                    f"Was conf={signal.confidence:.3f}"
                )
                signal.action = "HOLD"
                signal.confidence = 0.0
                if isinstance(signal.metadata, dict):
                    signal.metadata["reason"] = f"CHOP_REGIME_BLOCK | original: {signal_reason}"
                    signal.metadata["chop_guard"] = {
                        "original_action": original_action,
                        "original_confidence": base_confidence,
                        "hybrid_trend": hybrid_trend,
                        "block_type": block_type,
                    }
                # MAR 10 2026: Roll back strategy counter so daily slot is not wasted
                self._rollback_strategy_counter(signal_reason)
                return SignalGenerationResult(
                    signal=signal,
                    pipeline_result=pipeline_result,
                    filters_passed=False,
                    filters_applied=list(confidence_adjustments.keys()),
                    run_legacy_after_hybrid=False,
                    sentiment_modifier=sentiment_modifier,
                )

        # ── Step 2d¾: MES dead-zone time guard (FEB 24 2026) ─────────
        # MES (Micro E-mini S&P 500) has a well-known liquidity structure:
        #   08:30-10:00 CT: Power hour (most volume, directional moves)
        #   10:30-13:00 CT: "Dead zone" — liquidity thins, chop dominates
        #   13:00-15:15 CT: Afternoon drift / final flush
        #
        # 224-trade backtest of TREND_CONT by session zone:
        #   Dead Zone (10-12 CST): 18 trades, 28% WR, −$447 (−$24.8/trade)
        #   Afternoon (13-14):      5 trades, 20% WR, −$163 (−$32.7/trade)
        #   vs Power Hour (09-10): 10 trades, 60% WR, +$269 (+$26.9/trade)
        #
        # Today's losing trade entered at 11:15 CST — dead zone entry.
        # This guard blocks TREND_CONT entries during 10:30-13:00 CST.
        # Pullback and OR breakout signals are EXEMPT (they have different
        # edge profiles and are already filtered by CHOP guard + ADX).
        if is_trend_cont_signal and signal.action in ("BUY", "SELL", "SCALP_BUY", "SCALP_SELL"):
            try:
                cst_now = now_cst()
                hour_ct = cst_now.hour
                minute_ct = cst_now.minute
                # Dead zone: 10:30 CT through 12:59 CT
                in_dead_zone = (hour_ct == 10 and minute_ct >= 30) or (11 <= hour_ct <= 12)
                if in_dead_zone:
                    confidence_adjustments["dead_zone_block"] = -signal.confidence
                    logger.warning(
                        f"🚫 MES dead-zone guard: blocking TREND_CONT signal "
                        f"({signal_reason}) at {hour_ct}:{minute_ct:02d} CST — "
                        f"MES liquidity thins 10:30-13:00, trend signals chop out. "
                        f"Was {signal.action} conf={signal.confidence:.3f}"
                    )
                    signal.action = "HOLD"
                    signal.confidence = 0.0
                    if isinstance(signal.metadata, dict):
                        signal.metadata["reason"] = f"DEAD_ZONE_BLOCK | original: {signal_reason}"
                        signal.metadata["dead_zone_guard"] = {
                            "original_action": original_action,
                            "original_confidence": base_confidence,
                            "time_cst": f"{hour_ct}:{minute_ct:02d}",
                        }
                    return SignalGenerationResult(
                        signal=signal,
                        pipeline_result=pipeline_result,
                        filters_passed=False,
                        filters_applied=list(confidence_adjustments.keys()),
                        run_legacy_after_hybrid=False,
                        sentiment_modifier=sentiment_modifier,
                    )
            except Exception:
                pass  # Graceful fallback if time resolution fails

        # ── Step 2e: Exhaustion dampening (FEB 9 2026) ────────────────
        # Block BUY signals near session highs when overbought indicators
        # are present.  This catches the class of failure where EMA-pullback
        # signals fire at the exhaustion point of an intra-day rally.
        # FEB 18 2026: Exempt OR breakout/breakdown signals — by definition
        # they fire at session extremes; exhaustion gate creates a catch-22.
        signal_reason = signal.metadata.get("reason", "") if isinstance(signal.metadata, dict) else ""
        is_or_breakout = "OR_BREAK" in signal_reason
        if is_or_breakout and signal.action in ("BUY", "SELL", "SCALP_BUY", "SCALP_SELL"):
            logger.info(f"✅ Exhaustion gate EXEMPT: OR breakout signal ({signal_reason})")
        if signal.action in ("BUY", "SCALP_BUY") and pipeline_result is not None and not is_or_breakout:
            exhaustion_result = self._apply_exhaustion_dampening(
                signal=signal,
                features=features,
                current_price=current_price,
                pipeline_result=pipeline_result,
            )
            if exhaustion_result is not None:
                confidence_adjustments["exhaustion_dampening"] = exhaustion_result["penalty"]
                if exhaustion_result["blocked"]:
                    # Convert to HOLD — strategy-first architecture allows
                    # gating via confidence overlay when conditions are extreme.
                    signal.action = "HOLD"
                    signal.confidence = 0.0
                    if isinstance(signal.metadata, dict):
                        signal.metadata["reason"] = "EXHAUSTION_NEAR_SESSION_HIGH"
                        signal.metadata["exhaustion_detail"] = exhaustion_result
                    # MAR 10 2026: Roll back strategy counter
                    self._rollback_strategy_counter(signal_reason)
                    logger.info(
                        f"🛑 Exhaustion gate: HOLD (was {original_action} "
                        f"conf={base_confidence:.3f})"
                    )
                    return SignalGenerationResult(
                        signal=signal,
                        pipeline_result=pipeline_result,
                        filters_passed=False,
                        filters_applied=list(confidence_adjustments.keys()),
                        run_legacy_after_hybrid=False,
                        sentiment_modifier=sentiment_modifier,
                    )

        # ── Step 3: Ensure action was NEVER changed ──────────────────
        # Safety net: the overlays should only adjust confidence, never flip action
        # (the _evaluate_sentiment method may have changed action in some edge cases)
        if signal.action != original_action:
            logger.warning(
                f"⚠️ Safety: overlay changed action from {original_action} to {signal.action}. "
                f"Restoring strategy action."
            )
            signal.action = original_action

        # Add overlay summary to metadata
        if isinstance(signal.metadata, dict):
            signal.metadata["confidence_overlays"] = confidence_adjustments
            signal.metadata["base_confidence"] = base_confidence
            signal.metadata["final_confidence"] = signal.confidence

        logger.info(
            f"📊 Final Signal: {signal.action} conf={signal.confidence:.3f} "
            f"(base={base_confidence:.3f}, overlays={confidence_adjustments})"
        )

        return SignalGenerationResult(
            signal=signal,
            pipeline_result=pipeline_result,
            filters_passed=True,
            filters_applied=list(confidence_adjustments.keys()),
            run_legacy_after_hybrid=False,
            sentiment_modifier=sentiment_modifier,
        )

    # FEB 8 2026: Removed dead code
    # - _apply_trend_pullback_enhancer (1m-era pullback enhancer, never called in strategy-first)
    # - calculate_confidence (only used by _apply_confidence_layers which is also dead)

    # ── Exhaustion Dampening (FEB 9 2026) ─────────────────────────
    # Named constants — tuneable without touching logic
    # FEB 18 2026: Loosened from 0.3%/RSI 65. Old values blocked ALL
    # momentum signals on trending days (RSI 65-70 is normal in a trend).
    # Now only blocks truly extreme exhaustion (RSI 75+ within 0.1% of high).
    EXHAUSTION_SESSION_HIGH_PCT = 0.1    # Within 0.1% of session high (was 0.3%)
    EXHAUSTION_RSI_THRESHOLD = 75.0      # RSI above this = overbought zone (was 65)
    EXHAUSTION_CONFIDENCE_PENALTY = 0.15  # Penalty applied on soft dampening
    EXHAUSTION_BLOCK_PENALTY = 1.0       # Full block (set conf to 0)
    EXHAUSTION_OVERBOUGHT_KEYWORDS = frozenset({
        "RSI_OVERBOUGHT",
        "RSI_HIGH_IN_ACCEPTANCE",
    })
    EXHAUSTION_SCORE_KEYWORDS = (
        "BULLISH_BUT_OVERBOUGHT",
        "RSI_HIGH(",
    )

    def _apply_exhaustion_dampening(
        self,
        signal,
        features,
        current_price: float,
        pipeline_result,
    ) -> Optional[Dict[str, Any]]:
        """Detect overbought exhaustion near session highs and dampen/block.

        Returns None if no exhaustion detected, otherwise a dict:
            {
                "blocked": bool,
                "penalty": float,        # negative adjustment applied
                "session_high": float,
                "proximity_pct": float,  # how close price is to session high
                "rsi": float,
                "overbought_flags": list,
                "reason": str,
            }

        Conditions for blocking (all must be true):
          1. Signal is BUY (long into exhaustion)
          2. Price is within EXHAUSTION_SESSION_HIGH_PCT of session high
          3. Hybrid pipeline flagged overbought/exhaustion indicators

        If condition 2 is met but 3 is borderline (RSI > threshold but no
        explicit flag), apply a softer penalty instead of full block.
        """
        try:
            rule_engine = getattr(pipeline_result, "rule_engine", None)
            if rule_engine is None:
                return None

            # ── 1. Compute session high from features (RTH bars) ──────
            session_high = 0.0
            try:
                if hasattr(features, "high"):
                    session_high = float(features["high"].max())
                elif hasattr(features, "High"):
                    session_high = float(features["High"].max())
            except Exception:
                pass

            if session_high <= 0:
                return None

            # ── 2. Check proximity to session high ────────────────────
            proximity_pct = abs(session_high - current_price) / session_high * 100
            near_session_high = proximity_pct <= self.EXHAUSTION_SESSION_HIGH_PCT

            if not near_session_high:
                return None  # Not near high — no exhaustion concern

            # ── 3. Check overbought flags from hybrid pipeline ────────
            filters_passed = getattr(rule_engine, "filters_passed", []) or []
            filters_warned = getattr(rule_engine, "filters_warned", []) or []
            all_flags = set(filters_passed) | set(filters_warned)
            indicators = getattr(rule_engine, "indicators", {}) or {}
            rsi = indicators.get("rsi", 50.0)

            # Check for explicit overbought flags
            overbought_flags = [
                f for f in all_flags
                if f in self.EXHAUSTION_OVERBOUGHT_KEYWORDS
            ]

            # Also check score_breakdown for string markers
            score_breakdown = indicators.get("score_breakdown", []) or []
            for item in score_breakdown:
                if isinstance(item, str):
                    for kw in self.EXHAUSTION_SCORE_KEYWORDS:
                        if kw in item and item not in overbought_flags:
                            overbought_flags.append(item)

            # ── 4. Determine severity ─────────────────────────────────
            has_explicit_overbought = len(overbought_flags) > 0
            rsi_overbought = rsi >= self.EXHAUSTION_RSI_THRESHOLD

            if has_explicit_overbought and rsi_overbought:
                # FULL BLOCK: explicit overbought flag + high RSI + near session high
                penalty = self.EXHAUSTION_BLOCK_PENALTY
                blocked = True
                reason = (
                    f"Blocked: overbought exhaustion near session high "
                    f"(price={current_price:.2f}, high={session_high:.2f}, "
                    f"proximity={proximity_pct:.2f}%, RSI={rsi:.1f}, "
                    f"flags={overbought_flags})"
                )
            elif rsi_overbought:
                # SOFT DAMPEN: RSI overbought + near high, but no explicit flag
                penalty = self.EXHAUSTION_CONFIDENCE_PENALTY
                blocked = False
                reason = (
                    f"Dampened: RSI overbought near session high "
                    f"(RSI={rsi:.1f}, proximity={proximity_pct:.2f}%)"
                )
            else:
                # Near session high but RSI not overbought — no action
                return None

            result = {
                "blocked": blocked,
                "penalty": -penalty,
                "session_high": session_high,
                "proximity_pct": proximity_pct,
                "rsi": rsi,
                "overbought_flags": overbought_flags,
                "reason": reason,
            }

            if blocked:
                logger.warning(f"🛑 EXHAUSTION GATE: {reason}")
            else:
                logger.info(f"⚠️ EXHAUSTION DAMPEN: {reason}")
                signal.confidence = max(0.1, signal.confidence - penalty)

            return result

        except Exception as exc:
            logger.debug(f"Exhaustion dampening check failed (non-fatal): {exc}")
            return None

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

    async def process_trading_cycle(self, current_price: float, bar_timestamp=None):
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

        ts = bar_timestamp or now_cst()
        # FEB 2026: Align candle dedup boundary to active timeframe
        active_tf = getattr(m, "_active_timeframe", "1m")
        if active_tf == "15m":
            # Align to 15-minute boundary for dedup
            current_candle_start = ts.replace(second=0, microsecond=0, minute=(ts.minute // 15) * 15)
        else:
            current_candle_start = ts.replace(second=0, microsecond=0)
        if m._last_candle_processed == current_candle_start:
            logger.debug(f"⏳ Waiting for next candle close (current {active_tf} bar already processed)")
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

        # === INJECT VX FUTURES PRICE INTO FEATURES ===
        # This allows the hybrid pipeline to use real-time VX for regime detection
        # NOTE: This is ADDITIVE - it doesn't replace existing sentiment/indicator logic
        vx_price = None
        if self._vx_feed_enabled and self._vx_feed:
            vx_price = self._vx_feed.get_vx_price()
            if vx_price is not None:
                features["vx_price"] = vx_price
                logger.debug(f"📈 Injected VX price into features: {vx_price:.2f}")

        returns = features["close"].pct_change().dropna()
        m._publish_feature_snapshot(features, current_price)
        m._publish_account_context()
        structural_metrics = m._compute_structural_metrics(features)
        try:
            last_row = features.iloc[-1]
            m.status.last_atr = float(last_row.get("ATR_14", 0.0))
            m.status.last_adx = float(last_row.get("ADX_14", 0.0))
        except Exception:
            m.status.last_atr = 0.0

        signal_result = await self.generate_trading_signal(
            features=features,
            returns=returns,
            current_price=current_price,
            structural_metrics=structural_metrics,
        )
        if not signal_result or not signal_result.signal:
            return

        # Emit decision metric (count decisions by action) and clear stale episode flag if present
        try:
            symbol = getattr(getattr(m, 'settings', None).data, 'ibkr_symbol', None)
            if getattr(m, 'prometheus_metrics', None) and symbol:
                m.prometheus_metrics.inc_decision(symbol, signal_result.signal.action)
                # If we got a non-stale decision, mark stale episode inactive
                block_reasons_local = getattr(signal_result.signal, 'metadata', {}) or {}
                if not (isinstance(block_reasons_local, dict) and 'STALE_LIVE_BARS' in block_reasons_local.get('block_reasons', [])):
                    try:
                        m.prometheus_metrics.set_stale_episode_active(symbol, False)
                    except Exception:
                        pass
        except Exception:
            pass

        # If we blocked entries due to stale live bars, cancel pending ENTRY orders
        try:
            metadata = getattr(signal_result.signal, "metadata", {}) or {}
            block_reasons = metadata.get("block_reasons", []) if isinstance(metadata, dict) else []
            if "STALE_LIVE_BARS" in block_reasons:
                one_min_cfg = getattr(self.settings, "one_minute", {}) or {}
                if isinstance(one_min_cfg, dict):
                    cancel_enabled = bool(one_min_cfg.get("cancel_entries_on_stale", True))
                    throttle_seconds = int(one_min_cfg.get("cancel_stale_throttle_seconds", 30))
                else:
                    cancel_enabled = bool(getattr(one_min_cfg, "cancel_entries_on_stale", True))
                    throttle_seconds = int(getattr(one_min_cfg, "cancel_stale_throttle_seconds", 30))

                # Metric: stale_live_bars_blocks_total
                try:
                    if getattr(self.manager, "metrics_logger", None):
                        payload = {
                            "timestamp": now_cst().isoformat(),
                            "stale_live_bars_blocked": 1,
                            "last_1m_bar_age_seconds": metadata.get("staleness_seconds"),
                        }
                        self.manager.metrics_logger.record_market_metrics(payload)
                except Exception:
                    pass

                # Emit Prometheus staleness metrics (signal owner of truth)
                try:
                    symbol = getattr(getattr(self.manager, 'settings', None).data, 'ibkr_symbol', None)
                    if getattr(self.manager, 'prometheus_metrics', None) and symbol:
                        # record bar age and increment stale block, mark stale episode active
                        self.manager.prometheus_metrics.set_bar_age(symbol, "1m", metadata.get("staleness_seconds", 0))
                        self.manager.prometheus_metrics.inc_stale_block(symbol)
                        self.manager.prometheus_metrics.set_stale_episode_active(symbol, True)
                except Exception:
                    pass

                if cancel_enabled and getattr(self.manager, "executor", None):
                    try:
                        canceled = await self.manager.executor.cancel_pending_entry_orders(
                            reason="STALE_LIVE_BARS",
                            throttle_seconds=throttle_seconds,
                        )
                        log_structured_event(
                            agent="live_manager",
                            event_type="stale.cancel_entries",
                            message="Cancelled pending entry orders due to stale live bars",
                            payload={
                                "symbol": getattr(getattr(self.manager, 'settings', None), 'data', {}).get('ibkr_symbol', None) if isinstance(getattr(self.manager, 'settings', None), dict) else getattr(getattr(self.manager, 'settings', None), 'data', None) and getattr(getattr(self.manager, 'settings', None).data, 'ibkr_symbol', None),
                                "cancel_result": canceled,
                                "age_seconds_last_1m": metadata.get("staleness_seconds"),
                            },
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(f"Error cancelling pending entries on stale live bars: {exc}")
        except Exception:
            pass

        if signal_result.pipeline_result and not signal_result.run_legacy_after_hybrid:
            # Update cycle context with regime/volatility BEFORE branching to
            # hybrid signal path — otherwise register_trade_entry reads None
            # from _cycle_context (BUG FIX: FEB 8 2026)
            cycle_ctx = m._cycle_context.get(m._current_cycle_id, {})
            cycle_ctx["signal_type"] = signal_result.signal.action
            cycle_ctx["signal_confidence"] = signal_result.signal.confidence
            cycle_ctx["regime"] = m.status.hybrid_market_trend or None
            cycle_ctx["volatility"] = m.status.hybrid_volatility_regime or None

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
        if isinstance(signal.metadata, dict):
            m.status.hybrid_market_trend = signal.metadata.get("trend_label", m.status.hybrid_market_trend)
            m.status.last_trend = signal.metadata.get("trend_label", getattr(m.status, "last_trend", ""))

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
            # MAR 10 2026: Roll back strategy counter if signal blocked by
            # confidence gate, filters, active orders, or open position
            sig_meta = signal.metadata if isinstance(signal.metadata, dict) else {}
            blocked_reason = sig_meta.get("reason", "")
            if blocked_reason and decision.reason in (
                "CONFIDENCE_BELOW_MIN", "filter_block", "HOLD",
            ):
                self._rollback_strategy_counter(blocked_reason)
            logger.info("Skipping trade ({})", decision.reason or "blocked")
            return

        logger.info("  ↳ Attempting to place order: {}", signal.action)
        await m._place_order(signal, current_price, features)

    # FEB 8 2026: Removed dead code
    # - _apply_confidence_layers (legacy RAG/AWS KB pipeline, replaced by strategy-first)

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

    # FEB 8 2026: Removed dead code
    # - _apply_adx_and_trend_gates (1m-era ADX/counter-trend gate, replaced by strategy's own ADX filter)

    def _apply_sentiment_trend_adjustment(self, signal: Any, technical_trend: str) -> Any:
        """Apply sentiment-derived trend adjustment to signal confidence (delegated)."""
        return self._sentiment.apply_sentiment_trend_adjustment(signal, technical_trend)

    def _apply_scoring_validation(
        self,
        hybrid_signal: Any,
        features,
        current_price: float,
    ) -> Any:
        """Scoring system validation — SUNSET FEB 2026.
        
        The 1m scoring system has been replaced by the 15m strategy's own
        entry quality filters (ADX cap, EMA touch, RSI band, entry window).
        This stub is kept for interface compatibility.
        """
        return hybrid_signal

    def _init_mtf_builder(self) -> None:
        """Initialize MTF candle builder (delegated to MTFGateManager)."""
        self._mtf.init_mtf_builder()
        self._mtf_builder = self._mtf._mtf_builder

    def update_mtf_candle(self, bar: Dict[str, Any]) -> None:
        """Feed a 1-minute bar to the multi-timeframe builder (delegated)."""
        self._mtf.update_mtf_candle(bar)

    def _update_mtf_manager_and_gate(self, bar: Dict[str, Any]) -> None:
        """Update MTF Manager and Gate with 1m bar (delegated)."""
        self._mtf._update_mtf_manager_and_gate(bar)

    def check_5m_trend_alignment(self, action: str) -> Tuple[bool, str]:
        """Check if action aligns with 5-minute trend (delegated)."""
        return self._mtf.check_5m_trend_alignment(action)

    def get_5m_trend(self) -> str:
        """Get the current 5-minute trend (delegated)."""
        return self._mtf.get_5m_trend()

    def apply_5m_trend_filter(self, signal: Any) -> Any:
        """Apply 5-minute trend filter to signal (delegated)."""
        return self._mtf.apply_5m_trend_filter(signal)

    def _apply_mtf_trend_gate(self, signal: Any) -> Any:
        """Apply the MTF Trend Gate - HARD GATE (delegated)."""
        return self._mtf.apply_mtf_trend_gate(signal)

    def notify_position_opened(self, direction: str) -> None:
        """Notify MTF gate that a position was opened (delegated)."""
        self._mtf.notify_position_opened(direction)

    def notify_position_closed(self, close_reason: str, direction: str, pnl: float) -> None:
        """Notify MTF gate that a position was closed (delegated)."""
        self._mtf.notify_position_closed(close_reason, direction, pnl)

    def get_mtf_gate_state(self) -> Optional[str]:
        """Get current MTF gate state (delegated)."""
        return self._mtf.get_mtf_gate_state()
