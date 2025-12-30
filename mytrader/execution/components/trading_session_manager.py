"""Session lifecycle operations for live trading."""

import asyncio

from ib_insync import IB

from ...utils.logger import logger
from ...utils.timezone_utils import now_cst
from ..ib_executor import TradeExecutor
from ...monitoring.live_tracker import LivePerformanceTracker
from ...strategies.engine import StrategyEngine
from ...strategies.mes_one_minute import MesOneMinuteTrendStrategy
from ...risk.manager import RiskManager
from ...utils.telegram_notifier import TelegramNotifier
from ...llm.rag_storage import RAGStorage

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

            m.engine = StrategyEngine(
                [MesOneMinuteTrendStrategy(m.one_minute_cfg or m.settings.one_minute)]
            )
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

        poll_interval = 5

        try:
            while m.running and not m.stop_requested:
                try:
                    new_bar = await m._fetch_latest_minute_bar()
                    if not new_bar:
                        m.status.message = "Waiting for completed 1m bar..."
                        await m._broadcast_status()
                        await asyncio.sleep(poll_interval)
                        continue

                    m._ingest_completed_bar(new_bar)
                    current_price = float(new_bar["close"])
                    m.status.current_price = current_price

                    if len(m.price_history) < m.status.min_bars_needed:
                        m.status.message = f"Collecting data: {len(m.price_history)}/{m.status.min_bars_needed} bars"
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
                    await m._process_trading_cycle(current_price, bar_timestamp=new_bar["timestamp"])

                    await asyncio.sleep(poll_interval)

                except Exception as cycle_error:  # noqa: BLE001
                    logger.error(f"Error in trading cycle: {cycle_error}")
                    await m._broadcast_error(str(cycle_error))
                    await asyncio.sleep(poll_interval)

        except Exception as e:  # noqa: BLE001
            logger.error(f"Fatal error in trading loop: {e}")
            await m._broadcast_error(str(e))
        finally:
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
