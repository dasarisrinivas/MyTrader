"""Sentiment evaluation helpers.

Extracted from ``SignalProcessor`` to reduce its size.
Covers multi-source sentiment (Stocktwits + Reddit + Twitter), legacy
single-source Stocktwits, VX Futures Feed, and sentiment-trend adjustments.

Usage inside ``SignalProcessor.__init__``::

    from .sentiment_evaluator import SentimentEvaluator
    self._sentiment = SentimentEvaluator(self)

Then each old method delegates::

    def _evaluate_sentiment(self, signal):
        return self._sentiment.evaluate_sentiment(signal)
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, TYPE_CHECKING

from ...utils.logger import logger
from ...utils.structured_logging import log_structured_event

# ── Conditional imports (mirror signal_processor.py) ──────────────────

# Multi-source sentiment aggregator
try:
    from ...data.sentiment_aggregator import (
        evaluate_sentiment_for_entry as multi_evaluate_entry,
        get_combined_mes_sentiment,
        set_cache_refresh_interval as multi_set_cache_interval,
        SentimentDecision,
        combine_sentiment_with_technical_trend,
    )
    MULTI_SOURCE_SENTIMENT_AVAILABLE = True
except ImportError:
    MULTI_SOURCE_SENTIMENT_AVAILABLE = False

# Legacy single-source Stocktwits
try:
    from ...data.stocktwits_sentiment import (
        evaluate_sentiment_for_entry,
        get_mes_sentiment,
        set_cache_refresh_interval,
    )
    STOCKTWITS_SENTIMENT_AVAILABLE = True
except ImportError:
    STOCKTWITS_SENTIMENT_AVAILABLE = False

# VX Futures Feed
try:
    from ...data.vx_futures_feed import (
        VxFuturesFeed,
        init_vx_feed,
    )
    VX_FEED_AVAILABLE = True
except ImportError:
    VX_FEED_AVAILABLE = False
    VxFuturesFeed = None  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from .signal_processor import SignalProcessor

__all__ = ["SentimentEvaluator"]


class SentimentEvaluator:
    """Encapsulates all sentiment / VIX-feed logic previously inlined in *SignalProcessor*."""

    def __init__(self, processor: "SignalProcessor") -> None:
        self._p = processor

        # ── sentiment state ──
        self._multi_source_enabled: bool = False
        self._multi_source_config: object = None
        self._stocktwits_enabled: bool = False
        self._stocktwits_config: object = None

        # ── VX feed state ──
        self._vx_feed: Optional[VxFuturesFeed] = None  # type: ignore[assignment]
        self._vx_feed_enabled: bool = False

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def init_sentiment(self) -> None:
        """Initialize sentiment integration (multi-source or legacy Stocktwits)."""
        multi_cfg = getattr(self._p.settings, "multi_source_sentiment", None)
        if MULTI_SOURCE_SENTIMENT_AVAILABLE and multi_cfg and getattr(multi_cfg, "enabled", False):
            self._init_multi_source_sentiment(multi_cfg)
            return

        stocktwits_cfg = getattr(self._p.settings, "stocktwits_sentiment", None)
        if STOCKTWITS_SENTIMENT_AVAILABLE and stocktwits_cfg and getattr(stocktwits_cfg, "enabled", False):
            self._init_stocktwits_sentiment(stocktwits_cfg)
            return

        logger.debug("Sentiment integration disabled in config")

    def _init_multi_source_sentiment(self, cfg: object) -> None:
        """Initialize multi-source sentiment aggregator (Stocktwits + Reddit)."""
        self._multi_source_enabled = True
        self._multi_source_config = cfg

        refresh_interval = getattr(cfg, "refresh_interval_seconds", 300)
        multi_set_cache_interval(refresh_interval)

        twitter_enabled = getattr(cfg, "twitter_enabled", False)
        twitter_weight = getattr(cfg, "twitter_weight", 0.0)

        logger.info("=" * 70)
        logger.info("📊 MULTI-SOURCE SENTIMENT AGGREGATOR ENABLED")
        logger.info("=" * 70)
        logger.info("   Sources & Weights:")
        logger.info(f"      Stocktwits: {getattr(cfg, 'stocktwits_weight', 0.55):.0%}")
        logger.info(f"      Reddit:     {getattr(cfg, 'reddit_weight', 0.45):.0%}")
        if twitter_enabled and twitter_weight > 0:
            logger.info(f"      Twitter:    {twitter_weight:.0%}")
        else:
            logger.info("      Twitter:    DISABLED (no API credentials)")
        logger.info("   RTH Thresholds (8:30 AM - 3:00 PM CT):")
        logger.info(f"      Entry block: ±{getattr(cfg, 'entry_block_threshold', 0.4)}")
        logger.info(f"      Weak threshold: ±{getattr(cfg, 'weak_threshold', 0.2)}")
        logger.info("   Low-Volume Thresholds (Evening/Overnight):")
        logger.info(f"      Entry block: ±{getattr(cfg, 'low_volume_entry_block_threshold', 0.25)}")
        logger.info(f"      Weak threshold: ±{getattr(cfg, 'low_volume_weak_threshold', 0.10)}")
        logger.info(f"   Refresh interval: {refresh_interval}s")
        logger.info("=" * 70)

        try:
            initial = get_combined_mes_sentiment(force_refresh=True)
            logger.info(f"   ✅ Initial combined sentiment: {initial.score:+.2f}")
            logger.info(f"      Stocktwits: {initial.stocktwits.score:+.2f} ({initial.stocktwits.sample_count} samples)")
            logger.info(f"      Reddit: {initial.reddit.score:+.2f} ({initial.reddit.sample_count} samples)")
            if twitter_enabled and twitter_weight > 0:
                logger.info(f"      Twitter: {initial.twitter.score:+.2f} ({initial.twitter.sample_count} samples)")
        except Exception as e:
            logger.warning(f"   ⚠️ Initial sentiment fetch failed: {e}")

    def _init_stocktwits_sentiment(self, cfg: object) -> None:
        """Initialize legacy Stocktwits-only sentiment."""
        self._stocktwits_enabled = True
        self._stocktwits_config = cfg

        refresh_interval = getattr(cfg, "refresh_interval_seconds", 300)
        set_cache_refresh_interval(refresh_interval)

        logger.info("=" * 60)
        logger.info("📊 STOCKTWITS SENTIMENT (LEGACY) ENABLED")
        logger.info("=" * 60)
        logger.info("   RTH thresholds (8:30 AM - 3:00 PM CT):")
        logger.info(f"      Entry block: ±{getattr(cfg, 'entry_block_threshold', 0.4)}")
        logger.info(f"      Weak threshold: ±{getattr(cfg, 'weak_threshold', 0.2)}")
        logger.info(f"   Refresh interval: {refresh_interval}s")
        logger.info("=" * 60)

        try:
            initial_sentiment = get_mes_sentiment(force_refresh=True)
            logger.info(f"   ✅ Initial MES sentiment: {initial_sentiment:.2f}")
        except Exception as e:
            logger.warning(f"   ⚠️ Initial sentiment fetch failed: {e}")

    # ------------------------------------------------------------------
    # VX Futures Feed
    # ------------------------------------------------------------------

    def init_vx_feed(self) -> None:
        """Initialize VX futures feed for volatility-based position sizing."""
        if not VX_FEED_AVAILABLE:
            logger.debug("VX Futures Feed module not available")
            return

        vix_cfg = getattr(self._p.settings, "vix_feed", None)
        if not vix_cfg or not getattr(vix_cfg, "enabled", False):
            logger.debug("VX Futures Feed disabled in config")
            return

        try:
            ib_host = getattr(vix_cfg, "ib_host", "127.0.0.1")
            ib_port = getattr(vix_cfg, "ib_port", 7497)
            client_id = getattr(vix_cfg, "client_id", 71)
            market_data_type = getattr(vix_cfg, "market_data_type", 1)
            stale_seconds = getattr(vix_cfg, "stale_seconds", 120)
            conservative_on_stale = getattr(vix_cfg, "conservative_on_stale", False)
            max_retries = getattr(vix_cfg, "max_retries", 5)
            base_delay = getattr(vix_cfg, "base_delay", 1.0)

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
            logger.info("   Volatility Thresholds:")
            logger.info(f"      VX >= {extreme_threshold}: 0.4x multiplier (extreme fear)")
            logger.info(f"      VX >= {elevated_threshold}: 0.7x multiplier (elevated)")
            logger.info(f"      VX < {elevated_threshold}: 1.0x multiplier (normal)")

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

    def get_vx_multiplier(self) -> float:
        """Get the current VX-based volatility multiplier (0.4 – 1.0)."""
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

    # ------------------------------------------------------------------
    # Runtime evaluation
    # ------------------------------------------------------------------

    def evaluate_sentiment(self, signal: Any) -> Tuple[Any, Optional[Any]]:
        """Evaluate sentiment and apply to signal.

        Uses multi-source sentiment if available, otherwise falls back to Stocktwits-only.
        Uses stricter thresholds during low-volume sessions.
        """
        if self._multi_source_enabled and MULTI_SOURCE_SENTIMENT_AVAILABLE:
            return self._evaluate_multi_source_sentiment(signal)

        if self._stocktwits_enabled and STOCKTWITS_SENTIMENT_AVAILABLE:
            return self._evaluate_stocktwits_sentiment(signal)

        return signal, None

    def _evaluate_multi_source_sentiment(self, signal: Any) -> Tuple[Any, Optional[Any]]:
        """Evaluate multi-source sentiment (Stocktwits + Reddit + Twitter)."""
        action = getattr(signal, "action", "HOLD")
        if action == "HOLD":
            return signal, None

        try:
            cfg = self._multi_source_config

            is_low_volume = self._p._is_low_volume_session()
            current_session = self._p._get_current_session()

            if is_low_volume:
                entry_block = getattr(cfg, "low_volume_entry_block_threshold", 0.25)
                weak_threshold = getattr(cfg, "low_volume_weak_threshold", 0.10)
                low_volume_penalty = getattr(cfg, "low_volume_confidence_penalty", 0.9)
            else:
                entry_block = getattr(cfg, "entry_block_threshold", 0.4)
                weak_threshold = getattr(cfg, "weak_threshold", 0.2)
                low_volume_penalty = 1.0

            # Extract RSI for contrarian logic
            rsi_value = None
            try:
                metadata = getattr(signal, "metadata", {}) or {}
                rsi_value = metadata.get("rsi") or metadata.get("indicators", {}).get("rsi")
                if rsi_value is None and "rule_result" in metadata:
                    rule_indicators = metadata.get("rule_result", {}).get("indicators", {})
                    rsi_value = rule_indicators.get("rsi")
                if rsi_value is not None:
                    rsi_value = float(rsi_value)
            except (ValueError, TypeError, AttributeError):
                rsi_value = None

            decision: SentimentDecision = multi_evaluate_entry(
                proposed_action=action,
                entry_block_threshold=entry_block,
                weak_threshold=weak_threshold,
                force_refresh=False,
                rsi_value=rsi_value,
            )

            logger.info(f"📊 Multi-Source Sentiment Check ({current_session}):")
            logger.info(f"   Combined Score: {decision.combined_score:+.2f}")
            if decision.source_breakdown:
                logger.info("   Breakdown:")
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
        """Evaluate legacy Stocktwits-only sentiment and apply to signal."""
        if not self._stocktwits_enabled or not STOCKTWITS_SENTIMENT_AVAILABLE:
            return signal, None

        action = getattr(signal, "action", "HOLD")
        if action == "HOLD":
            return signal, None

        try:
            cfg = self._stocktwits_config

            is_low_volume = self._p._is_low_volume_session()
            current_session = self._p._get_current_session()

            if is_low_volume:
                entry_block = getattr(cfg, "low_volume_entry_block_threshold", 0.25)
                weak_threshold = getattr(cfg, "low_volume_weak_threshold", 0.10)
                low_volume_penalty = getattr(cfg, "low_volume_confidence_penalty", 0.9)
                logger.info(f"📊 Low-volume session ({current_session}) - using stricter sentiment thresholds")
            else:
                entry_block = getattr(cfg, "entry_block_threshold", 0.4)
                weak_threshold = getattr(cfg, "weak_threshold", 0.2)
                low_volume_penalty = 1.0

            sentiment_modifier = evaluate_sentiment_for_entry(
                proposed_action=action,
                entry_block_threshold=entry_block,
                weak_threshold=weak_threshold,
                force_refresh=False,
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

    # ------------------------------------------------------------------
    # Sentiment-trend adjustment
    # ------------------------------------------------------------------

    def apply_sentiment_trend_adjustment(self, signal: Any, technical_trend: str) -> Any:
        """Apply sentiment-derived trend adjustment to signal confidence."""
        if signal.action == "HOLD":
            return signal

        if not self._multi_source_enabled or not MULTI_SOURCE_SENTIMENT_AVAILABLE:
            return signal

        try:
            combined_trend, confidence_boost, explanation = combine_sentiment_with_technical_trend(
                technical_trend=technical_trend or "NEUTRAL",
                force_refresh=False,
            )

            if confidence_boost != 0.0:
                original_conf = getattr(signal, "confidence", 0.0)
                new_conf = max(0.0, min(1.0, original_conf + confidence_boost))
                signal.confidence = new_conf

                logger.info(f"📊 SENTIMENT-TREND: {explanation}")
                logger.info(f"   Confidence: {original_conf:.3f} → {new_conf:.3f}")

                metadata = getattr(signal, "metadata", {}) or {}
                metadata["sentiment_trend_boost"] = confidence_boost
                metadata["sentiment_trend_explanation"] = explanation
                metadata["combined_trend"] = combined_trend
                signal.metadata = metadata

        except Exception as e:
            logger.debug(f"Sentiment trend adjustment skipped: {e}")

        return signal
