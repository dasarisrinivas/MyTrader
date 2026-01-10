"""Signal processing pipeline."""
from __future__ import annotations

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
    from ...data.candle_aggregator import MultiTimeframeCandleBuilder
    MTF_AVAILABLE = True
except ImportError:
    MTF_AVAILABLE = False
    MultiTimeframeCandleBuilder = None

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
        
        # NEW: Multi-timeframe candle builder (Jan 2026)
        self._mtf_builder: Optional[MultiTimeframeCandleBuilder] = None
        self._init_mtf_builder()
        
        # NEW: Multi-source sentiment configuration (Jan 2026)
        self._multi_source_enabled = False
        self._multi_source_config = None
        # LEGACY: Single-source Stocktwits sentiment
        self._stocktwits_enabled = False
        self._stocktwits_config = None
        self._init_sentiment()
        
        # NEW: VX Futures Feed for volatility-based sizing (Jan 2026)
        self._vx_feed: Optional[VxFuturesFeed] = None
        self._vx_feed_enabled = False
        self._init_vx_feed()

    def _init_sentiment(self) -> None:
        """Initialize sentiment integration (multi-source or legacy Stocktwits)."""
        # Prefer multi-source sentiment if available and enabled
        multi_cfg = getattr(self.settings, "multi_source_sentiment", None)
        if MULTI_SOURCE_SENTIMENT_AVAILABLE and multi_cfg and getattr(multi_cfg, "enabled", False):
            self._init_multi_source_sentiment(multi_cfg)
            return
        
        # Fall back to legacy Stocktwits-only sentiment
        stocktwits_cfg = getattr(self.settings, "stocktwits_sentiment", None)
        if STOCKTWITS_SENTIMENT_AVAILABLE and stocktwits_cfg and getattr(stocktwits_cfg, "enabled", False):
            self._init_stocktwits_sentiment(stocktwits_cfg)
            return
        
        logger.debug("Sentiment integration disabled in config")
    
    def _init_multi_source_sentiment(self, cfg) -> None:
        """Initialize multi-source sentiment aggregator (Stocktwits + Reddit)."""
        self._multi_source_enabled = True
        self._multi_source_config = cfg
        
        # Configure cache refresh interval
        refresh_interval = getattr(cfg, "refresh_interval_seconds", 300)
        multi_set_cache_interval(refresh_interval)
        
        # Check if Twitter is enabled
        twitter_enabled = getattr(cfg, "twitter_enabled", False)
        twitter_weight = getattr(cfg, "twitter_weight", 0.0)
        
        logger.info("=" * 70)
        logger.info("📊 MULTI-SOURCE SENTIMENT AGGREGATOR ENABLED")
        logger.info("=" * 70)
        logger.info(f"   Sources & Weights:")
        logger.info(f"      Stocktwits: {getattr(cfg, 'stocktwits_weight', 0.55):.0%}")
        logger.info(f"      Reddit:     {getattr(cfg, 'reddit_weight', 0.45):.0%}")
        if twitter_enabled and twitter_weight > 0:
            logger.info(f"      Twitter:    {twitter_weight:.0%}")
        else:
            logger.info(f"      Twitter:    DISABLED (no API credentials)")
        logger.info(f"   RTH Thresholds (8:30 AM - 3:00 PM CT):")
        logger.info(f"      Entry block: ±{getattr(cfg, 'entry_block_threshold', 0.4)}")
        logger.info(f"      Weak threshold: ±{getattr(cfg, 'weak_threshold', 0.2)}")
        logger.info(f"   Low-Volume Thresholds (Evening/Overnight):")
        logger.info(f"      Entry block: ±{getattr(cfg, 'low_volume_entry_block_threshold', 0.25)}")
        logger.info(f"      Weak threshold: ±{getattr(cfg, 'low_volume_weak_threshold', 0.10)}")
        logger.info(f"   Refresh interval: {refresh_interval}s")
        logger.info("=" * 70)
        
        # Warm the cache with initial fetch
        try:
            initial = get_combined_mes_sentiment(force_refresh=True)
            logger.info(f"   ✅ Initial combined sentiment: {initial.score:+.2f}")
            logger.info(f"      Stocktwits: {initial.stocktwits.score:+.2f} ({initial.stocktwits.sample_count} samples)")
            logger.info(f"      Reddit: {initial.reddit.score:+.2f} ({initial.reddit.sample_count} samples)")
            if twitter_enabled and twitter_weight > 0:
                logger.info(f"      Twitter: {initial.twitter.score:+.2f} ({initial.twitter.sample_count} samples)")
        except Exception as e:
            logger.warning(f"   ⚠️ Initial sentiment fetch failed: {e}")

    def _init_stocktwits_sentiment(self, cfg) -> None:
        """Initialize legacy Stocktwits-only sentiment."""
        self._stocktwits_enabled = True
        self._stocktwits_config = cfg
        
        refresh_interval = getattr(cfg, "refresh_interval_seconds", 300)
        set_cache_refresh_interval(refresh_interval)
        
        logger.info("=" * 60)
        logger.info("📊 STOCKTWITS SENTIMENT (LEGACY) ENABLED")
        logger.info("=" * 60)
        logger.info(f"   RTH thresholds (8:30 AM - 3:00 PM CT):")
        logger.info(f"      Entry block: ±{getattr(cfg, 'entry_block_threshold', 0.4)}")
        logger.info(f"      Weak threshold: ±{getattr(cfg, 'weak_threshold', 0.2)}")
        logger.info(f"   Refresh interval: {refresh_interval}s")
        logger.info("=" * 60)
        
        try:
            initial_sentiment = get_mes_sentiment(force_refresh=True)
            logger.info(f"   ✅ Initial MES sentiment: {initial_sentiment:.2f}")
        except Exception as e:
            logger.warning(f"   ⚠️ Initial sentiment fetch failed: {e}")
    
    def _init_vx_feed(self) -> None:
        """Initialize VX futures feed for volatility-based position sizing."""
        if not VX_FEED_AVAILABLE:
            logger.debug("VX Futures Feed module not available")
            return
        
        vix_cfg = getattr(self.settings, "vix_feed", None)
        if not vix_cfg or not getattr(vix_cfg, "enabled", False):
            logger.debug("VX Futures Feed disabled in config")
            return
        
        try:
            # Extract config values
            ib_host = getattr(vix_cfg, "ib_host", "127.0.0.1")
            ib_port = getattr(vix_cfg, "ib_port", 7497)
            client_id = getattr(vix_cfg, "client_id", 71)
            market_data_type = getattr(vix_cfg, "market_data_type", 1)
            stale_seconds = getattr(vix_cfg, "stale_seconds", 120)
            conservative_on_stale = getattr(vix_cfg, "conservative_on_stale", False)
            max_retries = getattr(vix_cfg, "max_retries", 5)
            base_delay = getattr(vix_cfg, "base_delay", 1.0)
            
            # Extract thresholds
            thresholds = getattr(vix_cfg, "thresholds", {})
            if isinstance(thresholds, dict):
                extreme_threshold = thresholds.get("extreme", 30.0)
                elevated_threshold = thresholds.get("elevated", 20.0)
            else:
                extreme_threshold = getattr(thresholds, "extreme", 30.0)
                elevated_threshold = getattr(thresholds, "elevated", 20.0)
            
            logger.info("=" * 70)
            logger.info("📈 VX FUTURES FEED INITIALIZATION")
            logger.info("=" * 70)
            logger.info(f"   IBKR Connection: {ib_host}:{ib_port} (client_id={client_id})")
            logger.info(f"   Market Data Type: {'Live' if market_data_type == 1 else 'Delayed'}")
            logger.info(f"   Stale Threshold: {stale_seconds}s")
            logger.info(f"   Volatility Thresholds:")
            logger.info(f"      VX >= {extreme_threshold}: 0.4x multiplier (extreme fear)")
            logger.info(f"      VX >= {elevated_threshold}: 0.7x multiplier (elevated)")
            logger.info(f"      VX < {elevated_threshold}: 1.0x multiplier (normal)")
            
            # Initialize the VX feed
            self._vx_feed = init_vx_feed(
                host=ib_host,
                port=ib_port,
                client_id=client_id,
                market_data_type=market_data_type,
                stale_seconds=stale_seconds,
                conservative_on_stale=conservative_on_stale,
                extreme_threshold=extreme_threshold,
                elevated_threshold=elevated_threshold,
                max_retries=max_retries,
                base_delay=base_delay,
            )
            
            # Start background thread
            started = self._vx_feed.start_in_background()
            if started:
                self._vx_feed_enabled = True
                logger.info("   ✅ VX Feed background thread started")
            else:
                logger.warning("   ⚠️ VX Feed failed to start background thread")
            
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"⚠️ VX Futures Feed initialization failed: {e}")
            logger.error("   Proceeding without VX volatility adjustment")
            self._vx_feed = None
            self._vx_feed_enabled = False
    
    def _get_vx_multiplier(self) -> float:
        """Get the current VX-based volatility multiplier.
        
        Returns:
            Multiplier between 0.4 and 1.0 based on VIX level.
            Returns 1.0 if VX feed is disabled or unavailable.
        """
        if not self._vx_feed_enabled or self._vx_feed is None:
            return 1.0
        
        try:
            multiplier = self._vx_feed.get_volatility_multiplier()
            vx_price = self._vx_feed.get_vx_price()
            is_stale = self._vx_feed.is_stale()
            
            if vx_price is not None and multiplier != 1.0:
                stale_str = " [STALE]" if is_stale else ""
                logger.info(
                    f"📈 VX Volatility Adjustment: VX={vx_price:.2f}{stale_str} → {multiplier:.1f}x multiplier"
                )
            
            return multiplier
        except Exception as e:
            logger.warning(f"⚠️ VX multiplier fetch failed: {e}")
            return 1.0
    
    def shutdown_vx_feed(self) -> None:
        """Shutdown the VX futures feed (call on cleanup)."""
        if self._vx_feed is not None:
            try:
                self._vx_feed.stop_background()
                logger.info("VX Futures Feed shutdown complete")
            except Exception as e:
                logger.warning(f"VX Feed shutdown error: {e}")
            finally:
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
        """Evaluate sentiment and apply to signal.
        
        Uses multi-source sentiment if available, otherwise falls back to Stocktwits-only.
        Uses stricter thresholds during low-volume sessions.
        
        Returns:
            Tuple of (modified_signal, sentiment_decision or None)
        """
        # Prefer multi-source sentiment
        if self._multi_source_enabled and MULTI_SOURCE_SENTIMENT_AVAILABLE:
            return self._evaluate_multi_source_sentiment(signal)
        
        # Fall back to legacy Stocktwits
        if self._stocktwits_enabled and STOCKTWITS_SENTIMENT_AVAILABLE:
            return self._evaluate_stocktwits_sentiment(signal)
        
        return signal, None
    
    def _evaluate_multi_source_sentiment(self, signal: Any) -> Tuple[Any, Optional[Any]]:
        """Evaluate multi-source sentiment (Stocktwits + Reddit + Twitter).
        
        Returns:
            Tuple of (modified_signal, SentimentDecision or None)
        """
        action = getattr(signal, "action", "HOLD")
        if action == "HOLD":
            return signal, None
        
        try:
            cfg = self._multi_source_config
            
            # Session-aware threshold selection
            is_low_volume = self._is_low_volume_session()
            current_session = self._get_current_session()
            
            if is_low_volume:
                entry_block = getattr(cfg, "low_volume_entry_block_threshold", 0.25)
                weak_threshold = getattr(cfg, "low_volume_weak_threshold", 0.10)
                low_volume_penalty = getattr(cfg, "low_volume_confidence_penalty", 0.9)
            else:
                entry_block = getattr(cfg, "entry_block_threshold", 0.4)
                weak_threshold = getattr(cfg, "weak_threshold", 0.2)
                low_volume_penalty = 1.0
            
            # JAN 8 2026 FIX: Extract RSI for contrarian logic
            # When sentiment is extreme but RSI is opposite extreme, it's a contrarian opportunity
            rsi_value = None
            try:
                metadata = getattr(signal, "metadata", {}) or {}
                # Try various RSI keys
                rsi_value = metadata.get("rsi") or metadata.get("indicators", {}).get("rsi")
                if rsi_value is None and "rule_result" in metadata:
                    rule_indicators = metadata.get("rule_result", {}).get("indicators", {})
                    rsi_value = rule_indicators.get("rsi")
                if rsi_value is not None:
                    rsi_value = float(rsi_value)
            except (ValueError, TypeError, AttributeError):
                rsi_value = None
            
            # Get sentiment decision with RSI for contrarian logic
            decision: SentimentDecision = multi_evaluate_entry(
                proposed_action=action,
                entry_block_threshold=entry_block,
                weak_threshold=weak_threshold,
                force_refresh=False,
                rsi_value=rsi_value,  # JAN 8 2026 FIX: Pass RSI for contrarian detection
            )
            
            # Log comprehensive info
            logger.info(f"📊 Multi-Source Sentiment Check ({current_session}):")
            logger.info(f"   Combined Score: {decision.combined_score:+.2f}")
            if decision.source_breakdown:
                logger.info(f"   Breakdown:")
                logger.info(f"      Stocktwits: {decision.source_breakdown.get('stocktwits', 0):+.2f}")
                logger.info(f"      Reddit:     {decision.source_breakdown.get('reddit', 0):+.2f}")
                logger.info(f"      Twitter:    {decision.source_breakdown.get('twitter', 0):+.2f}")
            logger.info(f"   Thresholds: block=±{entry_block}, weak=±{weak_threshold}")
            logger.info(f"   Recommendation: {decision.action_recommendation}")
            logger.info(f"   Reason: {decision.reason}")
            
            log_structured_event(
                agent="signal_processor",
                event_type="sentiment.multi_source.check",
                message=f"MultiSource Sentiment {decision.action_recommendation}",
                payload={
                    "action": action,
                    "session": current_session,
                    "is_low_volume": is_low_volume,
                    "combined_score": decision.combined_score,
                    "source_breakdown": decision.source_breakdown,
                    "entry_block_threshold": entry_block,
                    "weak_threshold": weak_threshold,
                    "allow_trade": decision.allow_trade,
                    "recommendation": decision.action_recommendation,
                    "confidence_modifier": decision.confidence_modifier,
                    "reason": decision.reason,
                },
            )
            
            # Handle sentiment-based blocking
            if not decision.allow_trade:
                logger.warning(f"🚫 MULTI-SOURCE SENTIMENT BLOCK ({current_session}): {decision.reason}")
                signal.action = "HOLD"
                signal.confidence = 0.0
                metadata = getattr(signal, "metadata", {}) or {}
                metadata["sentiment_blocked"] = True
                metadata["sentiment_reason"] = decision.reason
                metadata["combined_sentiment"] = decision.combined_score
                metadata["source_breakdown"] = decision.source_breakdown
                metadata["session"] = current_session
                signal.metadata = metadata
                return signal, decision
            
            # Apply confidence modifier with low-volume penalty
            effective_modifier = decision.confidence_modifier * low_volume_penalty
            if effective_modifier != 1.0:
                original_conf = getattr(signal, "confidence", 0.0)
                new_conf = original_conf * effective_modifier
                logger.info(
                    f"   Confidence: {original_conf:.3f} → {new_conf:.3f} "
                    f"(sentiment: {decision.confidence_modifier:.2f}, "
                    f"low_vol: {low_volume_penalty:.2f})"
                )
                signal.confidence = new_conf
                
                metadata = getattr(signal, "metadata", {}) or {}
                metadata["sentiment_modifier"] = decision.confidence_modifier
                metadata["low_volume_penalty"] = low_volume_penalty
                metadata["combined_sentiment"] = decision.combined_score
                metadata["source_breakdown"] = decision.source_breakdown
                metadata["session"] = current_session
                signal.metadata = metadata
            
            return signal, decision
            
        except Exception as e:
            logger.warning(f"⚠️ Multi-source sentiment check failed: {e}")
            logger.warning("   Proceeding without sentiment data")
            return signal, None

    def _evaluate_stocktwits_sentiment(self, signal: Any) -> Tuple[Any, Optional[Any]]:
        """Evaluate legacy Stocktwits-only sentiment and apply to signal.
        
        Uses stricter thresholds during low-volume sessions (evening/overnight).
        
        Returns:
            Tuple of (modified_signal, sentiment_modifier or None)
        """
        if not self._stocktwits_enabled or not STOCKTWITS_SENTIMENT_AVAILABLE:
            return signal, None
        
        action = getattr(signal, "action", "HOLD")
        if action == "HOLD":
            return signal, None
        
        try:
            cfg = self._stocktwits_config
            
            # Session-aware threshold selection
            is_low_volume = self._is_low_volume_session()
            current_session = self._get_current_session()
            
            if is_low_volume:
                # Stricter thresholds during low volume (evening/overnight)
                entry_block = getattr(cfg, "low_volume_entry_block_threshold", 0.25)
                weak_threshold = getattr(cfg, "low_volume_weak_threshold", 0.10)
                low_volume_penalty = getattr(cfg, "low_volume_confidence_penalty", 0.9)
                logger.info(f"📊 Low-volume session ({current_session}) - using stricter sentiment thresholds")
            else:
                # RTH thresholds (more relaxed due to higher liquidity)
                entry_block = getattr(cfg, "entry_block_threshold", 0.4)
                weak_threshold = getattr(cfg, "weak_threshold", 0.2)
                low_volume_penalty = 1.0  # No additional penalty during RTH
            
            sentiment_modifier = evaluate_sentiment_for_entry(
                proposed_action=action,
                entry_block_threshold=entry_block,
                weak_threshold=weak_threshold,
                force_refresh=False,  # Use cached data if fresh
            )
            
            logger.info(f"📊 Stocktwits Sentiment Check ({current_session} session):")
            logger.info(f"   MES Sentiment: {sentiment_modifier.mes_sentiment:.2f}")
            logger.info(f"   Entry block threshold: ±{entry_block}")
            logger.info(f"   Weak threshold: ±{weak_threshold}")
            logger.info(f"   Recommendation: {sentiment_modifier.action_recommendation}")
            logger.info(f"   Reason: {sentiment_modifier.reason}")
            
            log_structured_event(
                agent="signal_processor",
                event_type="sentiment.check",
                message=f"Sentiment {sentiment_modifier.action_recommendation}",
                payload={
                    "action": action,
                    "session": current_session,
                    "is_low_volume": is_low_volume,
                    "entry_block_threshold": entry_block,
                    "weak_threshold": weak_threshold,
                    "mes_sentiment": sentiment_modifier.mes_sentiment,
                    "allow_trade": sentiment_modifier.allow_trade,
                    "confidence_modifier": sentiment_modifier.confidence_modifier,
                    "reason": sentiment_modifier.reason,
                },
            )
            
            # Handle sentiment-based blocking
            if not sentiment_modifier.allow_trade:
                logger.warning(f"🚫 SENTIMENT BLOCK ({current_session}): {sentiment_modifier.reason}")
                signal.action = "HOLD"
                signal.confidence = 0.0
                metadata = getattr(signal, "metadata", {}) or {}
                metadata["sentiment_blocked"] = True
                metadata["sentiment_reason"] = sentiment_modifier.reason
                metadata["mes_sentiment"] = sentiment_modifier.mes_sentiment
                metadata["session"] = current_session
                signal.metadata = metadata
                return signal, sentiment_modifier
            
            # Apply confidence modifier (with additional low-volume penalty if applicable)
            effective_modifier = sentiment_modifier.confidence_modifier * low_volume_penalty
            if effective_modifier != 1.0:
                original_conf = getattr(signal, "confidence", 0.0)
                new_conf = original_conf * effective_modifier
                logger.info(
                    f"   Confidence adjusted: {original_conf:.3f} → {new_conf:.3f} "
                    f"(sentiment: {sentiment_modifier.confidence_modifier:.2f}, "
                    f"low_vol: {low_volume_penalty:.2f})"
                )
                signal.confidence = new_conf
                
                metadata = getattr(signal, "metadata", {}) or {}
                metadata["sentiment_modifier"] = sentiment_modifier.confidence_modifier
                metadata["low_volume_penalty"] = low_volume_penalty
                metadata["mes_sentiment"] = sentiment_modifier.mes_sentiment
                metadata["session"] = current_session
                signal.metadata = metadata
            
            return signal, sentiment_modifier
            
        except Exception as e:
            logger.warning(f"⚠️ Stocktwits sentiment check failed: {e}")
            logger.warning("   Proceeding without sentiment data")
            return signal, None

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
            live_threshold = 120
            try:
                one_min_cfg = getattr(self.settings, "one_minute", None) or {}
                if isinstance(one_min_cfg, dict):
                    live_threshold = int(one_min_cfg.get("live_bar_stale_seconds", live_threshold))
                else:
                    live_threshold = int(getattr(one_min_cfg, "live_bar_stale_seconds", live_threshold))
            except Exception:
                live_threshold = 120

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
                
                # === INJECT 5-MINUTE TREND INTO FEATURES FOR RULE ENGINE ===
                # This allows RuleEngine to use higher-timeframe trend in its scoring
                if self._mtf_builder is not None and self._mtf_builder.has_complete_candle():
                    htf_trend_5m = self._mtf_builder.get_trend()
                    features["5m_trend"] = htf_trend_5m
                    features["htf_trend"] = htf_trend_5m
                    logger.debug(f"Injected 5m trend into features: {htf_trend_5m}")
                
                # Also inject sentiment bias if available from recent sentiment check
                if hasattr(m, "_last_sentiment_bias"):
                    features["sentiment_bias"] = m._last_sentiment_bias
                if hasattr(m, "_last_sentiment_score"):
                    features["sentiment_score"] = m._last_sentiment_score
                
                hybrid_signal, pipeline_result = await pipeline.process(
                    features,
                    current_price,
                    position_for_pipeline,
                )

                # Update status with hybrid pipeline info
                if pipeline_result:
                    m.context_manager.refresh_hybrid_context(pipeline_result)

                    # Attach provenance details to structured logs for this decision cycle
                    try:
                        provenance = getattr(pipeline_result, "market_data", {})
                        provenance = provenance.get("levels_provenance") if isinstance(provenance, dict) else None
                    except Exception:
                        provenance = None
                    log_structured_event(
                        agent="live_manager",
                        event_type="hybrid.decision_provenance",
                        message=f"provenance {hybrid_signal.action}",
                        payload={
                            "levels_provenance": provenance,
                            "block_reasons": getattr(hybrid_signal, "metadata", {}).get("block_reasons"),
                            "age_seconds_last_1m": provenance.get("age_seconds_last_1m") if isinstance(provenance, dict) else None,
                            "timeframes_used": {"pipeline_timeframe": getattr(pipeline, "_timeframe", None)},
                        },
                    )

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

                # === JAN 2026 AUDIT FIX: ADX Gate + Counter-Trend Hard-Block ===
                hybrid_signal = self._apply_adx_and_trend_gates(
                    hybrid_signal, features, m.status.hybrid_market_trend
                )
                
                # === JAN 2026 AUDIT FIX: 5-Minute Trend Filter ===
                hybrid_signal = self.apply_5m_trend_filter(hybrid_signal)
                
                # === JAN 8 2026: Sentiment-Derived Trend Adjustment ===
                hybrid_signal = self._apply_sentiment_trend_adjustment(hybrid_signal, m.status.hybrid_market_trend)
                
                # === JAN 2026: Multi-Source Sentiment Integration ===
                sentiment_modifier = None
                if (self._multi_source_enabled or self._stocktwits_enabled) and hybrid_signal.action != "HOLD":
                    hybrid_signal, sentiment_modifier = self._evaluate_sentiment(hybrid_signal)
                    
                    # Cache sentiment for use in next cycle's trend calculation
                    if sentiment_modifier is not None:
                        if hasattr(sentiment_modifier, "combined_score"):
                            # Multi-source sentiment
                            m._last_sentiment_score = sentiment_modifier.combined_score
                            if sentiment_modifier.combined_score > 0.2:
                                m._last_sentiment_bias = "BULLISH"
                            elif sentiment_modifier.combined_score < -0.2:
                                m._last_sentiment_bias = "BEARISH"
                            else:
                                m._last_sentiment_bias = "NEUTRAL"
                        elif hasattr(sentiment_modifier, "mes_sentiment"):
                            # Stocktwits-only sentiment
                            m._last_sentiment_score = sentiment_modifier.mes_sentiment
                            if sentiment_modifier.mes_sentiment > 0.2:
                                m._last_sentiment_bias = "BULLISH"
                            elif sentiment_modifier.mes_sentiment < -0.2:
                                m._last_sentiment_bias = "BEARISH"
                            else:
                                m._last_sentiment_bias = "NEUTRAL"

                # === JAN 2026: VX Futures Volatility Multiplier ===
                # Apply VX-based scaling to reduce confidence when VIX is elevated
                vx_multiplier = self._get_vx_multiplier()
                if vx_multiplier != 1.0 and hybrid_signal.action != "HOLD":
                    original_conf = hybrid_signal.confidence
                    hybrid_signal.confidence = original_conf * vx_multiplier
                    logger.info(
                        f"📈 VX Volatility Scaling: {original_conf:.3f} × {vx_multiplier:.1f} = "
                        f"{hybrid_signal.confidence:.3f}"
                    )
                    metadata = getattr(hybrid_signal, "metadata", {}) or {}
                    metadata["vx_multiplier"] = vx_multiplier
                    metadata["vx_price"] = self._vx_feed.get_vx_price() if self._vx_feed else None
                    hybrid_signal.metadata = metadata

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
                        sentiment_modifier=sentiment_modifier,
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
        
        # === JAN 2026: Multi-Source Sentiment Integration (Legacy Path) ===
        sentiment_modifier = None
        if (self._multi_source_enabled or self._stocktwits_enabled) and signal.action != "HOLD":
            signal, sentiment_modifier = self._evaluate_sentiment(signal)

        m.status.last_signal = signal.action
        m.status.signal_confidence = signal.confidence

        return SignalGenerationResult(
            signal=signal,
            pipeline_result=None,
            filters_passed=filters_passed,
            filters_applied=filters_applied,
            run_legacy_after_hybrid=False,
            sentiment_modifier=sentiment_modifier,
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
        current_candle_start = ts.replace(second=0, microsecond=0)
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
            logger.info("Skipping trade ({})", decision.reason or "blocked")
            return

        logger.info("  ↳ Attempting to place order: {}", signal.action)
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

    def _apply_adx_and_trend_gates(self, signal: Any, features, market_trend: str) -> Any:
        """Apply ADX threshold gate and counter-trend hard-block.
        
        Added Jan 2, 2026 after audit showed:
        - 5/6 BUY signals in DOWNTREND caused losses
        - Low ADX trades had high failure rate
        
        Returns modified signal (HOLD if blocked).
        """
        if signal.action == "HOLD":
            return signal
        
        # Get config settings - handle both dict and config object
        trading_cfg = getattr(self.settings, "trading", None)
        entry_filters = None
        
        if trading_cfg is not None:
            if hasattr(trading_cfg, "entry_filters"):
                entry_filters = trading_cfg.entry_filters
            elif isinstance(trading_cfg, dict):
                entry_filters = trading_cfg.get("entry_filters", {})
        
        # Extract values handling both dict and config object
        # JAN 8 2026 FIX: Raised ADX threshold from 15 to 18 to filter out 
        # borderline trending signals that whipsaw. ADX 15-18 is "weak trend"
        # territory where trend-following strategies underperform.
        if entry_filters is None:
            require_adx = True
            min_adx = 18.0  # RAISED from 15.0 - filter weak trends
            allow_counter_trend = False
        elif hasattr(entry_filters, "require_adx_confirmation"):
            # Config object with attributes
            require_adx = getattr(entry_filters, "require_adx_confirmation", True)
            min_adx = getattr(entry_filters, "min_adx_threshold", 18.0)  # RAISED default
            allow_counter_trend = getattr(entry_filters, "allow_counter_trend", False)
        elif isinstance(entry_filters, dict):
            # Dict-based config
            require_adx = entry_filters.get("require_adx_confirmation", True)
            min_adx = entry_filters.get("min_adx_threshold", 18.0)  # RAISED default
            allow_counter_trend = entry_filters.get("allow_counter_trend", False)
        else:
            require_adx = True
            min_adx = 18.0  # RAISED from 15.0 - filter weak trends
            allow_counter_trend = False
        
        # Extract ADX from features
        adx_value = 0.0
        try:
            if hasattr(features, "iloc") and len(features) > 0:
                last_row = features.iloc[-1]
                adx_value = float(last_row.get("ADX_14", 0.0) or last_row.get("adx", 0.0))
        except Exception:
            pass
        
        metadata = getattr(signal, "metadata", {}) or {}
        block_reasons = metadata.get("block_reasons", [])
        
        # === ADX Gate ===
        if require_adx and adx_value > 0 and adx_value < min_adx:
            logger.warning(
                f"🚫 ADX GATE: Signal {signal.action} blocked - ADX {adx_value:.1f} < {min_adx:.1f} threshold"
            )
            block_reasons.append(f"LOW_ADX:{adx_value:.1f}<{min_adx:.1f}")
            signal.action = "HOLD"
            signal.confidence = 0.0
            metadata["block_reasons"] = block_reasons
            metadata["adx_blocked"] = True
            metadata["adx_value"] = adx_value
            signal.metadata = metadata
            return signal
        
        # === Counter-Trend Hard-Block ===
        if not allow_counter_trend:
            trend_upper = (market_trend or "").upper()
            is_counter_trend = False
            
            if signal.action in ("BUY", "SCALP_BUY") and "DOWN" in trend_upper:
                is_counter_trend = True
                block_reasons.append(f"COUNTER_TREND:BUY_IN_{trend_upper}")
            elif signal.action in ("SELL", "SCALP_SELL") and "UP" in trend_upper:
                is_counter_trend = True
                block_reasons.append(f"COUNTER_TREND:SELL_IN_{trend_upper}")
            
            if is_counter_trend:
                logger.warning(
                    "🚫 COUNTER-TREND BLOCK: {} signal blocked in {} market",
                    signal.action, trend_upper
                )
                signal.action = "HOLD"
                signal.confidence = 0.0
                metadata["block_reasons"] = block_reasons
                metadata["counter_trend_blocked"] = True
                metadata["blocked_trend"] = trend_upper
                signal.metadata = metadata
                return signal
        
        return signal

    def _apply_sentiment_trend_adjustment(self, signal: Any, technical_trend: str) -> Any:
        """Apply sentiment-derived trend adjustment to signal confidence.
        
        JAN 8 2026: Sentiment can be a leading indicator. When sentiment
        aligns with technicals, boost confidence. When they diverge, reduce.
        
        Args:
            signal: The trading signal to adjust
            technical_trend: Current technical trend ("UPTREND", "DOWNTREND", "NEUTRAL", etc.)
            
        Returns:
            Signal with potentially modified confidence
        """
        if signal.action == "HOLD":
            return signal
        
        if not self._multi_source_enabled or not MULTI_SOURCE_SENTIMENT_AVAILABLE:
            return signal
        
        try:
            # Get sentiment trend and combine with technical
            combined_trend, confidence_boost, explanation = combine_sentiment_with_technical_trend(
                technical_trend=technical_trend or "NEUTRAL",
                force_refresh=False,  # Use cached sentiment
            )
            
            # Apply confidence adjustment
            if confidence_boost != 0.0:
                original_conf = getattr(signal, "confidence", 0.0)
                new_conf = max(0.0, min(1.0, original_conf + confidence_boost))
                signal.confidence = new_conf
                
                logger.info(f"📊 SENTIMENT-TREND: {explanation}")
                logger.info(f"   Confidence: {original_conf:.3f} → {new_conf:.3f}")
                
                # Update metadata
                metadata = getattr(signal, "metadata", {}) or {}
                metadata["sentiment_trend_boost"] = confidence_boost
                metadata["sentiment_trend_explanation"] = explanation
                metadata["combined_trend"] = combined_trend
                signal.metadata = metadata
            
        except Exception as e:
            logger.debug(f"Sentiment trend adjustment skipped: {e}")
        
        return signal

    def _init_mtf_builder(self) -> None:
        """Initialize the multi-timeframe candle builder if enabled in config."""
        if not MTF_AVAILABLE or MultiTimeframeCandleBuilder is None:
            logger.debug("Multi-timeframe builder not available")
            return
        
        # Check config for MTF settings
        one_minute_cfg = getattr(self.settings, "one_minute", {})
        if isinstance(one_minute_cfg, dict):
            require_5m = one_minute_cfg.get("require_5m_trend_alignment", False)
            ema_period = one_minute_cfg.get("mtf_ema_period", 20)
        else:
            require_5m = getattr(one_minute_cfg, "require_5m_trend_alignment", False)
            ema_period = getattr(one_minute_cfg, "mtf_ema_period", 20)
        
        if require_5m:
            try:
                self._mtf_builder = MultiTimeframeCandleBuilder(
                    base_interval=1,
                    target_interval=5,
                    ema_period=ema_period,
                    max_history=100,
                )
                logger.info("✅ Multi-timeframe candle builder initialized (1m -> 5m)")
            except Exception as e:
                logger.warning(f"Failed to initialize MTF builder: {e}")
                self._mtf_builder = None
    
    def update_mtf_candle(self, bar: Dict[str, Any]) -> None:
        """Feed a 1-minute bar to the multi-timeframe builder.
        
        Call this from the trading loop whenever a new 1-min bar completes.
        """
        if self._mtf_builder is None:
            return
        
        try:
            timestamp = bar.get("timestamp")
            if timestamp is None:
                return
            
            completed_candle = self._mtf_builder.add_bar(
                timestamp=timestamp,
                open_price=float(bar.get("open", 0)),
                high_price=float(bar.get("high", 0)),
                low_price=float(bar.get("low", 0)),
                close_price=float(bar.get("close", 0)),
                volume=float(bar.get("volume", 0)),
            )
            
            # Log when a 5-minute candle completes
            if completed_candle is not None:
                trend_5m = self._mtf_builder.get_trend()
                logger.info(
                    f"📊 5-MIN CANDLE COMPLETE: O={completed_candle.open:.2f} "
                    f"H={completed_candle.high:.2f} L={completed_candle.low:.2f} "
                    f"C={completed_candle.close:.2f} | Trend: {trend_5m}"
                )
        except Exception as e:
            logger.debug(f"MTF candle update error: {e}")
    
    def check_5m_trend_alignment(self, action: str) -> Tuple[bool, str]:
        """Check if the proposed action aligns with the 5-minute trend.
        
        Args:
            action: 'BUY', 'SELL', etc.
        
        Returns:
            Tuple of (is_aligned, reason)
        """
        if self._mtf_builder is None:
            return True, "MTF_DISABLED"
        
        if not self._mtf_builder.has_complete_candle():
            return True, "MTF_INSUFFICIENT_DATA"
        
        return self._mtf_builder.is_trend_aligned(action)
    
    def get_5m_trend(self) -> str:
        """Get the current 5-minute trend."""
        if self._mtf_builder is None:
            return "UNKNOWN"
        return self._mtf_builder.get_trend()
    
    def apply_5m_trend_filter(self, signal: Any) -> Any:
        """Apply 5-minute trend filter to the signal.
        
        Blocks counter-trend trades based on higher timeframe analysis.
        Added Jan 2, 2026 to reduce whipsaw losses.
        """
        if signal.action == "HOLD":
            return signal
        
        if self._mtf_builder is None:
            return signal
        
        is_aligned, reason = self.check_5m_trend_alignment(signal.action)
        
        metadata = getattr(signal, "metadata", {}) or {}
        metadata["5m_trend"] = self.get_5m_trend()
        metadata["5m_alignment"] = reason
        
        if not is_aligned:
            logger.warning(
                "🚫 5-MIN TREND BLOCK: %s signal blocked - %s",
                signal.action, reason
            )
            block_reasons = metadata.get("block_reasons", [])
            block_reasons.append(reason)
            metadata["block_reasons"] = block_reasons
            metadata["5m_trend_blocked"] = True
            signal.action = "HOLD"
            signal.confidence = 0.0
        else:
            logger.info(f"✅ 5-min trend aligned: {reason}")
        
        signal.metadata = metadata
        return signal
