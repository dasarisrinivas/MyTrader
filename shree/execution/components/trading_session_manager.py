"""Session lifecycle operations for live trading."""

import asyncio
import time as _time
from datetime import time as _time_cls

from ib_insync import IB

from ...utils.logger import logger
from ...utils.timezone_utils import now_cst
from ...utils.session_utils import classify_session, TradingSession
from ..ib_executor import TradeExecutor
from ...monitoring.live_tracker import LivePerformanceTracker
from ...strategies.engine import StrategyEngine
from ...strategies.mes_one_minute import MesOneMinuteTrendStrategy
from ...strategies.mes_one_minute_scoring import MesOneMinuteScoringStrategy  # FEB 2026: Scoring system
from ...strategies.es_fifteen_min import EsFifteenMinStrategy  # FEB 2026: 15m strategy (replaces 1m)
from ...risk.manager import RiskManager
from ...utils.telegram_notifier import TelegramNotifier
# Use S3 RAGStorageManager instead of local SQLite RAGStorage
from ...rag.rag_storage_manager import RAGStorageManager as RAGStorage

# Optional hybrid pipeline
try:
    from ...rag.pipeline_integration import create_hybrid_integration
    HYBRID_PIPELINE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    create_hybrid_integration = None
    HYBRID_PIPELINE_AVAILABLE = False

# Optional decision metrics logger
try:
    from ...llm.trade_logger import TradeLogger as DecisionMetricsLogger
except ImportError:  # pragma: no cover - optional dependency
    DecisionMetricsLogger = None


class TradingSessionManager:
    """Starts, stops, and reconciles the live trading session."""

    def __init__(self, manager: "LiveTradingManager"):  # noqa: F821 (forward reference)
        self.manager = manager

    async def initialize(self):
        """Initialize trading components."""
        m = self.manager
        try:
            logger.info("Initializing trading components...")

            m.tracker = LivePerformanceTracker(
                initial_capital=m.settings.trading.initial_capital,
                risk_free_rate=m.settings.backtest.risk_free_rate,
                point_value=m.contract_spec.point_value,
            )

            # Fix #6 MAR 16 2026: Restore persisted daily P&L so the $250 daily
            # loss cap survives bot restarts mid-day.
            _persisted_pnl = getattr(m, "_persisted_daily_pnl", 0.0)
            if _persisted_pnl != 0.0:
                m.tracker.daily_pnl = _persisted_pnl
                m.tracker.total_realized_pnl += _persisted_pnl
                logger.info(
                    f"📊 Restored persisted daily P&L: ${_persisted_pnl:.2f} "
                    f"(daily loss cap = ${m.settings.risk_gate.daily_max_loss_usd})"
                )

            # FEB 2026: Strategy selection — 15m strategy is now the default
            one_min_cfg = m.one_minute_cfg or m.settings.one_minute
            use_15m = getattr(one_min_cfg, "use_15m_strategy", False)
            use_scoring = getattr(one_min_cfg, "use_scoring_system", False)

            if use_15m:
                strategy = EsFifteenMinStrategy(one_min_cfg)
                m._active_timeframe = "15m"
                m._bar_size_setting = "15 mins"
                m._candle_period_seconds = 15 * 60  # 900 seconds
                # Adjust warmup and bar window for 15m bars
                # 60 bars × 15min = 15 hours of warmup data
                m.status.min_bars_needed = max(60, getattr(one_min_cfg, "warmup_bars", 60))
                m._bar_window = max(200, getattr(one_min_cfg, "window_bars", 200))
                # Max hold tracking for 15m time-stop
                ft_hold_bars = getattr(one_min_cfg, "ft_max_hold_bars", 6)
                m._ft_max_hold_minutes = ft_hold_bars * 15
                logger.info("=" * 60)
                logger.info("📊 STRATEGY: ES 15-Minute (EMA21 Pullback + OR Breakout)")
                logger.info(f"   Timeframe: 15m | Max hold: {m._ft_max_hold_minutes} min")
                logger.info(f"   Warmup: {m.status.min_bars_needed} bars | Window: {m._bar_window} bars")
                shorts_on = getattr(one_min_cfg, 'ft_shorts_enabled', False)
                if shorts_on:
                    logger.info("   Mode: LONG + SHORT (short-side signals enabled)")
                else:
                    logger.info("   Mode: LONG-ONLY (shorts disabled)")
                logger.info("=" * 60)
            elif use_scoring:
                strategy = MesOneMinuteScoringStrategy(one_min_cfg)
                m._active_timeframe = "1m"
                m._bar_size_setting = "1 min"
                m._candle_period_seconds = 60
                logger.info("✅ Using SCORING-BASED entry system (1m experimental)")
            else:
                strategy = MesOneMinuteTrendStrategy(one_min_cfg)
                m._active_timeframe = "1m"
                m._bar_size_setting = "1 min"
                m._candle_period_seconds = 60
                logger.info("ℹ️  Using traditional hard-filter entry system (1m)")
            
            m.engine = StrategyEngine([strategy])
            m.signal_processor.engine = m.engine

            m.risk = RiskManager(m.settings.trading, position_sizing_method="kelly")

            if hasattr(m.settings, "telegram") and m.settings.telegram.enabled:
                m.telegram = TelegramNotifier(
                    bot_token=m.settings.telegram.bot_token,
                    chat_id=m.settings.telegram.chat_id,
                    enabled=True,
                )
                logger.info("✅ Telegram notifications initialized")
            else:
                m.telegram = None
                logger.info("ℹ️  Telegram notifications disabled")

            m.ib = IB()
            m.executor = TradeExecutor(
                m.ib,
                m.settings.trading,
                m.settings.data.ibkr_symbol,
                m.settings.data.ibkr_exchange,
                telegram_notifier=m.telegram,
                trading_mode=m.trading_mode,
                contract_spec=m.contract_spec,
                commission_per_side=m._commission_per_side,
            )

            # Attach prometheus metrics handle to executor if available
            try:
                if getattr(m, "prometheus_metrics", None):
                    setattr(m.executor, "prometheus_metrics", m.prometheus_metrics)
            except Exception:
                pass

            await m.executor.connect(
                m.settings.data.ibkr_host,
                m.settings.data.ibkr_port,
                client_id=11,
            )
            await m.force_order_reconciliation()
            m._load_persistent_cooldown_state()
            if m._reset_state_on_start:
                logger.warning("♻️  Reset-state flag detected - clearing cooldown/lock state at startup")
                m._reset_state_on_start = False
                m._apply_manual_state_reset()
            try:
                m.rag_storage = RAGStorage()
                logger.info("✅ RAG Storage initialized")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to initialize RAG Storage: {e}")

            if DecisionMetricsLogger:
                try:
                    m.metrics_logger = DecisionMetricsLogger()
                    logger.info("✅ Decision metrics logger initialized")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"⚠️ Could not initialize decision metrics logger: {e}")
                    m.metrics_logger = None
            else:
                logger.info("ℹ️ Decision metrics logger unavailable (module not installed)")

            if HYBRID_PIPELINE_AVAILABLE and create_hybrid_integration:
                try:
                    hybrid_enabled = getattr(m.settings, "hybrid", None)
                    if hybrid_enabled and getattr(hybrid_enabled, "enabled", False):
                        m.hybrid_pipeline = create_hybrid_integration(
                            settings=m.settings,
                            llm_client=None,
                            context_bus=m.agent_bus,
                        )
                        if hasattr(m.hybrid_pipeline, "ensure_ready"):
                            stats = m.hybrid_pipeline.ensure_ready(min_documents=5)
                            engine = stats.get("engine", "cpu")
                            doc_count = stats.get("documents", 0)
                            logger.info(f"🔎 Hybrid RAG index ready ({engine}, {doc_count} docs)")
                        m._use_hybrid_pipeline = True
                        m.status.hybrid_pipeline_enabled = True
                        m.signal_processor.hybrid_pipeline = m.hybrid_pipeline
                        logger.info("✅ Hybrid RAG+LLM Pipeline initialized (3-layer decision system)")
                    else:
                        logger.info("ℹ️  Hybrid pipeline disabled in config")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"⚠️  Failed to initialize Hybrid Pipeline: {e}")
                    m._use_hybrid_pipeline = False

            m._configure_aws_agents()
            await m._load_historical_context()
            
            # Wire historical context to hybrid pipeline for PDH/PDL decisions
            if m._use_hybrid_pipeline and m.hybrid_pipeline and m._historical_context:
                try:
                    prev_day = m._historical_context.get("previous_day", {})
                    weekly = m._historical_context.get("weekly", {})
                    m.hybrid_pipeline.set_price_levels(
                        pdh=prev_day.get("high"),
                        pdl=prev_day.get("low"),
                        weekly_high=weekly.get("high"),
                        weekly_low=weekly.get("low"),
                        source="ibkr_historical",
                    )
                    logger.info("✅ Historical context wired to hybrid pipeline for decision-making")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"⚠️ Failed to wire historical context to hybrid pipeline: {e}")
            
            await m._bootstrap_price_history(m.status.min_bars_needed)

            if m.executor and m.executor.ib:
                m.executor.ib.execDetailsEvent += m._on_execution_details

            logger.info("✅ Connected to IBKR")

            m.status.is_running = True
            m.status.session_start = now_cst().isoformat()
            m.status.message = "Initialized successfully"

            await m._broadcast_status()
            logger.info("✅ Live trading manager initialized")
            return True

        except Exception as e:  # noqa: BLE001
            m.status.message = f"Initialization failed: {str(e)}"
            await m._broadcast_error(str(e))
            logger.error(f"Failed to initialize: {e}")
            return False

    async def start(self):
        """Start the live trading loop."""
        m = self.manager
        if not await self.initialize():
            return

        logger.info("🔄 Starting trading loop...")
        m.running = True
        m.stop_requested = False

        # FEB 2026: Adjust poll interval based on active timeframe
        active_tf = getattr(m, "_active_timeframe", "1m")
        if active_tf == "15m":
            poll_interval = 30  # Poll every 30s for 15m bars (less frequent)
            tf_label = "15m"
        else:
            poll_interval = 5
            tf_label = "1m"

        try:
            logger.info(
                f"🔁 Entering main loop: running={m.running}, "
                f"stop_requested={m.stop_requested}, tf={active_tf}, poll={poll_interval}s"
            )
            while m.running and not m.stop_requested:
                try:
                    new_bar = await m._fetch_latest_bar()
                    if not new_bar:
                        m.status.message = f"Waiting for completed {tf_label} bar..."
                        # Periodic heartbeat so operator knows the bot is alive
                        if not hasattr(m, "_last_heartbeat_log"):
                            m._last_heartbeat_log = 0
                        now_mono = _time.monotonic()
                        if now_mono - m._last_heartbeat_log > 300:  # every 5 min
                            logger.info(
                                f"💓 Heartbeat: waiting for {tf_label} bar | "
                                f"bars={len(m.price_history)} | price={m.status.current_price}"
                            )
                            m._last_heartbeat_log = now_mono
                        await m._broadcast_status()
                        await asyncio.sleep(poll_interval)
                        continue

                    m._ingest_completed_bar(new_bar)
                    current_price = float(new_bar["close"])
                    m.status.current_price = current_price

                    if len(m.price_history) < m.status.min_bars_needed:
                        m.status.message = f"Collecting {tf_label} data: {len(m.price_history)}/{m.status.min_bars_needed} bars"
                        logger.info(m.status.message)
                        await m._broadcast_status()
                        await asyncio.sleep(poll_interval)
                        continue

                    if (
                        m.status.bars_collected == m.status.min_bars_needed
                        and not hasattr(m, "_position_verified")
                    ):
                        logger.info("🔍 Warmup complete. Verifying existing positions before trading...")
                        existing_position = await m.executor.get_current_position()
                        if existing_position and existing_position.quantity != 0:
                            logger.warning(
                                "⚠️  EXISTING POSITION DETECTED: %s contracts @ %.2f",
                                existing_position.quantity,
                                existing_position.avg_cost,
                            )
                            logger.warning("⚠️  Bot will manage this position. Use opposite signals to exit.")
                        else:
                            logger.info("✅ No existing positions. Ready to trade fresh.")
                        m._position_verified = True

                    logger.debug("Processing trading cycle...")

                    # ──────────────────────────────────────────────────────
                    # FEB 7 2026: SESSION ISOLATION FOR 15m STRATEGY
                    #
                    # Problem: Calling _process_trading_cycle() for non-RTH
                    # bars causes the strategy's generate() to update
                    # _prev_close with overnight prices. This produces
                    # different pullback signals than the backtest (which
                    # only calls strategy during RTH).
                    #
                    # Solution: For 15m, only run full trading cycle during
                    # RTH. For non-RTH bars, only check exit signals (so
                    # bracket fills / time-stops still work overnight).
                    # ──────────────────────────────────────────────────────
                    if active_tf == "15m":
                        bar_ts = new_bar.get("timestamp") or new_bar.get("date")
                        # Use strategy's configured RTH window (may be wider than
                        # default 9:30-16:00 ET if entry window was extended).
                        one_min_cfg = getattr(m.settings, "one_minute", None)
                        _rth_s = _time_cls(
                            getattr(one_min_cfg, "rth_start_hour", 9),
                            getattr(one_min_cfg, "rth_start_minute", 30),
                        ) if one_min_cfg else _time_cls(9, 30)
                        _rth_e = _time_cls(
                            getattr(one_min_cfg, "rth_end_hour", 16),
                            getattr(one_min_cfg, "rth_end_minute", 0),
                        ) if one_min_cfg else _time_cls(16, 0)
                        session = (
                            classify_session(bar_ts, rth_start=_rth_s, rth_end=_rth_e)
                            if bar_ts is not None
                            else TradingSession.RTH
                        )
                        if session == TradingSession.RTH:
                            await m._process_trading_cycle(current_price, bar_timestamp=new_bar["timestamp"])
                        else:
                            # Non-RTH: only monitor/exit existing positions
                            # (brackets fire via IB anyway, this catches time-stops)
                            position = None
                            if m.executor:
                                try:
                                    position = await m.executor.get_current_position()
                                except Exception:
                                    pass
                            qty = getattr(position, "quantity", 0) if position else 0
                            if qty != 0:
                                await m._check_position_exit_signals(current_price)
                            logger.info(
                                f"⏳ 15m non-RTH bar (session={session.name}): "
                                f"skipping strategy, exit-check only (pos={qty})"
                            )
                    else:
                        await m._process_trading_cycle(current_price, bar_timestamp=new_bar["timestamp"])

                    await asyncio.sleep(poll_interval)

                except Exception as cycle_error:  # noqa: BLE001
                    logger.error(f"Error in trading cycle: {cycle_error}")
                    import traceback
                    logger.error(traceback.format_exc())
                    await m._broadcast_error(str(cycle_error))
                    await asyncio.sleep(poll_interval)

            # Loop exited normally — log why
            logger.warning(
                f"⚠️ Trading loop exited: running={m.running}, "
                f"stop_requested={m.stop_requested}"
            )

        except Exception as e:  # noqa: BLE001
            logger.error(f"Fatal error in trading loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await m._broadcast_error(str(e))
        except BaseException as e:
            logger.critical(f"🛑 CRITICAL: Trading loop killed by {type(e).__name__}: {e}")
            raise
        finally:
            logger.info("🔚 Trading loop finally block — calling stop()")
            await m.stop()

    async def stop(self):
        m = self.manager
        m.running = False
        m.stop_requested = True
        m.status.is_running = False
        m.status.message = "Stopped"

        if m.ib and m.ib.isConnected():
            m.ib.disconnect()

        await m._broadcast_status()
        logger.info("Trading session stopped")

    async def force_order_reconciliation(self):
        if self.manager.executor:
            logger.info("Forcing reconciliation of active orders with IBKR...")
            await self.manager.executor._reconcile_orders()
            logger.info("Order reconciliation complete.")
