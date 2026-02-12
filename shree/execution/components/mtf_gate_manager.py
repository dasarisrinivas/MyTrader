"""Multi-timeframe (MTF) candle management and trend gating.

Extracted from ``SignalProcessor`` to reduce its size.
Covers MTF builder/manager initialization, 1m bar ingestion,
5m/15m/30m trend aggregation, IB bootstrap, and the hard MTF
Trend Gate that blocks counter-trend entries.

Usage inside ``SignalProcessor.__init__``::

    from .mtf_gate_manager import MTFGateManager
    self._mtf = MTFGateManager(self)

Then each old method delegates::

    def update_mtf_candle(self, bar):
        self._mtf.update_mtf_candle(bar)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from ...utils.logger import logger

# ── Conditional imports (mirror signal_processor.py) ──────────────────

try:
    from ...data.candle_aggregator import MultiTimeframeCandleBuilder, MTFCandleManager
    MTF_AVAILABLE = True
except ImportError:
    MTF_AVAILABLE = False
    MultiTimeframeCandleBuilder = None  # type: ignore[misc,assignment]
    MTFCandleManager = None  # type: ignore[misc,assignment]

try:
    from .mtf_trend_gate import MTFTrendGate, TradingState
    MTF_GATE_AVAILABLE = True
except ImportError:
    MTF_GATE_AVAILABLE = False
    MTFTrendGate = None  # type: ignore[misc,assignment]
    TradingState = None  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from .signal_processor import SignalProcessor

__all__ = ["MTFGateManager"]


class MTFGateManager:
    """Encapsulates all multi-timeframe candle aggregation and trend-gating logic."""

    def __init__(self, processor: "SignalProcessor") -> None:
        self._p = processor

        # ── MTF builder (1m → 5m single aggregator) ──
        self._mtf_builder: Optional[MultiTimeframeCandleBuilder] = None  # type: ignore[assignment]

        # ── MTF manager (5m, 15m, 30m full manager) ──
        self._mtf_manager: Optional[MTFCandleManager] = None  # type: ignore[assignment]

        # ── MTF Trend Gate (state machine) ──
        self._mtf_gate: Optional[MTFTrendGate] = None  # type: ignore[assignment]

        # ── Bootstrap tracking ──
        self._mtf_bootstrap_started: bool = False
        self._mtf_bootstrap_task = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_mtf_builder(self) -> None:
        """Initialize the multi-timeframe candle builder if enabled in config."""
        if not MTF_AVAILABLE or MultiTimeframeCandleBuilder is None:
            logger.debug("Multi-timeframe builder not available")
            return

        one_minute_cfg = getattr(self._p.settings, "one_minute", {})
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

    def init_mtf_manager(self) -> None:
        """Initialize the full multi-timeframe candle manager (5m, 15m, 30m)."""
        if not MTF_AVAILABLE or MTFCandleManager is None:
            logger.debug("MTFCandleManager not available")
            return

        try:
            self._mtf_manager = MTFCandleManager(
                ema_period_5m=20,
                ema_period_15m=20,
                ema_period_30m=20,
                max_history=100,
            )
            logger.info("✅ MTFCandleManager initialized (5m, 15m, 30m aggregators)")
        except Exception as e:
            logger.warning(f"Failed to initialize MTFCandleManager: {e}")
            self._mtf_manager = None

    def init_mtf_gate(self) -> None:
        """Initialize the MTF Trend Gate with state machine."""
        if not MTF_GATE_AVAILABLE or MTFTrendGate is None:
            logger.debug("MTFTrendGate not available")
            return

        one_minute_cfg = getattr(self._p.settings, "one_minute", {})
        if isinstance(one_minute_cfg, dict):
            mtf_gate_enabled = one_minute_cfg.get("mtf_gate_enabled", True)
        else:
            mtf_gate_enabled = getattr(one_minute_cfg, "mtf_gate_enabled", True)

        if not mtf_gate_enabled:
            logger.info("MTFTrendGate disabled via config")
            self._mtf_gate = None
            return

        try:
            self._mtf_gate = MTFTrendGate(
                min_15m_candles_after_close=1,
                require_full_mtf=True,
            )
            logger.info("✅ MTFTrendGate initialized (state machine for trend discipline)")
        except Exception as e:
            logger.warning(f"Failed to initialize MTFTrendGate: {e}")
            self._mtf_gate = None

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def ensure_mtf_gate_bootstrap(self) -> None:
        """Kick off a one-time async bootstrap of 5m/15m/30m trends from IB historical bars."""
        if self._mtf_bootstrap_started:
            return
        if self._mtf_gate is None:
            return
        if self._p.manager is None or getattr(self._p.manager, "executor", None) is None:
            return

        try:
            import asyncio

            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            self._mtf_bootstrap_started = True

            if loop is not None and loop.is_running():
                self._mtf_bootstrap_task = loop.create_task(self._bootstrap_mtf_gate_from_ib())
            else:
                self._mtf_bootstrap_task = None
        except Exception as exc:
            logger.debug(f"MTF gate bootstrap scheduling skipped: {exc}")

    async def _bootstrap_mtf_gate_from_ib(self) -> None:
        """Populate 5m/15m/30m trends using IB historical bars."""
        if self._mtf_gate is None:
            return
        executor = getattr(self._p.manager, "executor", None)
        if executor is None or getattr(executor, "ib", None) is None:
            return

        try:
            contract = await executor.get_qualified_contract()
            if not contract:
                return

            bar_requests = [
                ("5m", "43200 S", "5 mins"),
                ("15m", "3 D", "15 mins"),
                ("30m", "5 D", "30 mins"),
            ]

            def _trend_from_ema(closes, ema):
                if len(closes) < 3 or len(ema) < 3:
                    return "UNKNOWN", 0.0
                slope = float(ema[-1] - ema[-3])
                if slope > 0:
                    return "UPTREND", min(1.0, abs(slope) / 5.0)
                if slope < 0:
                    return "DOWNTREND", min(1.0, abs(slope) / 5.0)
                return "NEUTRAL", 0.2

            for tf, duration_str, bar_size in bar_requests:
                try:
                    bars = await executor.ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime="",
                        durationStr=duration_str,
                        barSizeSetting=bar_size,
                        whatToShow="TRADES",
                        useRTH=False,
                        formatDate=2,
                    )
                except AttributeError:
                    bars = executor.ib.reqHistoricalData(
                        contract,
                        endDateTime="",
                        durationStr=duration_str,
                        barSizeSetting=bar_size,
                        whatToShow="TRADES",
                        useRTH=False,
                        formatDate=2,
                    )

                if not bars:
                    continue

                closes = [
                    float(getattr(b, "close", 0.0) or 0.0)
                    for b in bars
                    if getattr(b, "close", None) is not None
                ]
                if len(closes) < 10:
                    continue

                ema_period = 20
                ema_vals = []
                k = 2.0 / (ema_period + 1.0)
                ema = closes[0]
                for c in closes:
                    ema = (c * k) + (ema * (1 - k))
                    ema_vals.append(ema)

                trend, conf = _trend_from_ema(closes, ema_vals)
                last_bar = bars[-1]
                candle_close_time = getattr(last_bar, "date", None)
                if isinstance(candle_close_time, str):
                    try:
                        candle_close_time = datetime.fromisoformat(
                            candle_close_time.replace("Z", "+00:00")
                        )
                    except Exception:
                        candle_close_time = None

                self._mtf_gate.update_trend(
                    timeframe=tf,
                    trend=trend,
                    confidence=float(conf),
                    ema_value=float(ema_vals[-1]) if ema_vals else None,
                    candle_close_time=candle_close_time,
                )

            logger.info("✅ MTF Gate bootstrapped from IB historical bars (5m/15m/30m)")

        except Exception as exc:
            logger.debug(f"MTF gate bootstrap failed: {exc}")

    # ------------------------------------------------------------------
    # Runtime: candle ingestion
    # ------------------------------------------------------------------

    def update_mtf_candle(self, bar: Dict[str, Any]) -> None:
        """Feed a 1-minute bar to the multi-timeframe builder."""
        if self._mtf_builder is None:
            return

        self.ensure_mtf_gate_bootstrap()

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

            if completed_candle is not None:
                trend_5m = self._mtf_builder.get_trend()
                logger.info(
                    f"📊 5-MIN CANDLE COMPLETE: O={completed_candle.open:.2f} "
                    f"H={completed_candle.high:.2f} L={completed_candle.low:.2f} "
                    f"C={completed_candle.close:.2f} | Trend: {trend_5m}"
                )
        except Exception as e:
            logger.debug(f"MTF candle update error: {e}")

        self._update_mtf_manager_and_gate(bar)

    def _update_mtf_manager_and_gate(self, bar: Dict[str, Any]) -> None:
        """Update the MTF Manager with 1m bar and sync trends to MTF Gate."""
        if self._mtf_manager is None:
            return

        try:
            timestamp = bar.get("timestamp")
            if timestamp is None:
                return

            completed = self._mtf_manager.add_1m_bar(
                timestamp=timestamp,
                open_price=float(bar.get("open", 0)),
                high_price=float(bar.get("high", 0)),
                low_price=float(bar.get("low", 0)),
                close_price=float(bar.get("close", 0)),
                volume=float(bar.get("volume", 0)),
            )

            if self._mtf_gate is not None:
                trends = self._mtf_manager.get_all_trends()

                for tf, trend_data in trends.items():
                    if not getattr(trend_data, "is_valid", False):
                        continue
                    if str(getattr(trend_data, "trend", "UNKNOWN") or "UNKNOWN").upper() == "UNKNOWN":
                        continue

                    self._mtf_gate.update_trend(
                        timeframe=tf,
                        trend=trend_data.trend,
                        confidence=trend_data.confidence,
                        ema_value=trend_data.ema_value,
                        candle_close_time=trend_data.candle_close_time,
                    )

                self._mtf_gate.update_trend(
                    timeframe="1m",
                    trend="TRIGGER_ONLY",
                    confidence=0.0,
                )

                if completed.get("15m") is not None:
                    summary = self._mtf_gate.get_state_summary()
                    logger.info(
                        f"📊 MTF STATE: state={summary['state']}, "
                        f"15m={summary['trends']['15m']['trend']}, "
                        f"30m={summary['trends']['30m']['trend']}, "
                        f"5m={summary['trends']['5m']['trend']}, "
                        f"cooldown_remaining={summary['cooldown']['candles_remaining']}"
                    )
        except Exception as e:
            logger.debug(f"MTF manager update error: {e}")

    # ------------------------------------------------------------------
    # 5-minute trend helpers
    # ------------------------------------------------------------------

    def check_5m_trend_alignment(self, action: str) -> Tuple[bool, str]:
        """Check if the proposed action aligns with the 5-minute trend."""
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
        """Apply 5-minute trend filter to the signal."""
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
                "🚫 5-MIN TREND BLOCK: {} signal blocked - {} (5m_trend={})",
                signal.action,
                reason,
                metadata.get("5m_trend", "UNKNOWN"),
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

    # ------------------------------------------------------------------
    # MTF Trend Gate (hard gate)
    # ------------------------------------------------------------------

    def apply_mtf_trend_gate(self, signal: Any) -> Any:
        """Apply the MTF Trend Gate — HARD GATE for multi-timeframe discipline."""
        if signal.action == "HOLD":
            return signal

        if self._mtf_gate is None:
            logger.debug("MTF Gate not initialized - skipping gate check")
            return signal

        sentiment_bias = getattr(self._p.manager, "_last_sentiment_bias", None)

        allowed, reason, metadata_from_gate = self._mtf_gate.evaluate_entry(
            proposed_action=signal.action,
            sentiment_bias=sentiment_bias,
        )

        metadata = getattr(signal, "metadata", {}) or {}
        metadata["mtf_gate"] = {
            "allowed": allowed,
            "reason": reason,
            "state": metadata_from_gate.get("state"),
            "trends": metadata_from_gate.get("trends"),
            "cooldown": metadata_from_gate.get("cooldown_candles_needed"),
        }

        if not allowed:
            logger.warning(f"🚫 MTF GATE BLOCK: {signal.action} blocked - {reason}")

            trends = metadata_from_gate.get("trends", {})
            if trends:
                logger.info(
                    f"   Trend snapshot: 15m={trends.get('15m', {}).get('trend', 'N/A')}, "
                    f"30m={trends.get('30m', {}).get('trend', 'N/A')}, "
                    f"5m={trends.get('5m', {}).get('trend', 'N/A')}"
                )

            block_reasons = metadata.get("block_reasons", [])
            block_reasons.append(f"MTF_GATE:{reason}")
            metadata["block_reasons"] = block_reasons
            metadata["mtf_gate_blocked"] = True

            signal.action = "HOLD"
            signal.confidence = 0.0
        else:
            logger.info(
                f"✅ MTF GATE PASSED: {signal.action} allowed "
                f"(state={metadata_from_gate.get('state')}, {reason})"
            )

        signal.metadata = metadata
        return signal

    # ------------------------------------------------------------------
    # Position lifecycle notifications
    # ------------------------------------------------------------------

    def notify_position_opened(self, direction: str) -> None:
        """Notify the MTF gate that a position was opened."""
        if self._mtf_gate is not None:
            self._mtf_gate.on_position_opened(direction)

    def notify_position_closed(
        self,
        close_reason: str,
        direction: str,
        pnl: float,
    ) -> None:
        """Notify the MTF gate that a position was closed — triggers cooldown."""
        if self._mtf_gate is not None:
            self._mtf_gate.on_position_closed(close_reason, direction, pnl)

            summary = self._mtf_gate.get_state_summary()
            logger.info(
                f"📊 MTF Gate post-close: state={summary['state']}, "
                f"cooldown={summary['cooldown']['candles_remaining']} x 15m candles"
            )

    def get_mtf_gate_state(self) -> Optional[str]:
        """Get the current MTF gate state."""
        if self._mtf_gate is None:
            return None
        return self._mtf_gate.state.value
