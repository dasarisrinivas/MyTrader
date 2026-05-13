"""Live Trading Manager with WebSocket broadcasting.

ENHANCED VERSION with:
- Trade cooldown period (prevents over-trading)
- Candle close validation
- Higher-timeframe level filters (PDH/PDL, WH/WL)
- Trend confirmation (EMA)
- Simulation mode (dry run)
- Enhanced confidence scoring
- Hybrid RAG+LLM Pipeline (3-layer decision system)
- CST timestamps throughout (Central Standard Time)
"""
import asyncio
import math
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
import threading

import sqlite3

import numpy as np

from ib_insync import IB
from ..config import Settings, FeatureFlagsConfig
from ..utils.logger import logger
from ..utils.structured_logging import log_structured_event
from ..utils.telegram_notifier import TelegramNotifier
from ..utils.timezone_utils import now_cst, today_cst, CST, utc_to_cst
from .ib_executor import TradeExecutor
from ..monitoring.live_tracker import LivePerformanceTracker
from ..strategies.engine import StrategyEngine
from ..features.feature_engineer import engineer_features
from ..risk.manager import RiskManager
from ..risk.risk_gate import RiskGate, RiskGateConfig
from ..risk.dynamic_support import DynamicSupportFloor, DynamicSupportFloorConfig
from ..risk.atr_module import compute_protective_offsets
from ..risk.trade_math import (
    ContractSpec,
    TradingMode,
    compute_risk_reward,
    enforce_min_take_profit,
    expected_target_outcome,
    get_commission_per_side,
    get_contract_spec,
)
from ..execution.guards import (
    WaitDecisionContext,
    compute_trade_risk_dollars,
    should_block_on_wait,
)
from ..learning.trade_learning import (
    TradeLearningRecorder,
)
from ..optimization.optimizer import ParameterOptimizer
from ..llm.rag_storage import RAGStorage
try:
    from ..llm.trade_logger import TradeLogger as DecisionMetricsLogger
except ImportError:
    DecisionMetricsLogger = None
from ..strategies.trading_filters import TradingFilters, PriceLevels
from ..hybrid.coordination import AgentBus
from .components import (
    CooldownManager,
    StatusBroadcaster,
    ContextManager,
    OrderCoordinator,
    RiskController,
    TradingSessionManager,
    MarketDataCoordinator,
    SignalProcessor,
    TradeDecisionEngine,
    SystemHealthMonitor,
)
from .components.risk_controller import TradeRequest

# NEW: Hybrid RAG+LLM Pipeline imports
try:
    from ..rag.pipeline_integration import HybridPipelineIntegration, create_hybrid_integration
    HYBRID_PIPELINE_AVAILABLE = True
except ImportError:
    HYBRID_PIPELINE_AVAILABLE = False
    logger.warning("Hybrid RAG pipeline not available - using legacy signal generation")

from ..rag.kb_monitor import kb_usage_tracker
try:
    from ..rag.local_knowledge_base import LocalKnowledgeBase
except ImportError:
    LocalKnowledgeBase = None

# NEW: AWS Bedrock Agents Pipeline imports
try:
    from ..aws import AgentInvoker, MarketSnapshotBuilder
    AWS_AGENTS_AVAILABLE = True
except ImportError:
    AWS_AGENTS_AVAILABLE = False
    logger.warning("AWS Agents not available - using local signal generation")


@dataclass
class TradingStatus:
    """Current trading status."""
    is_running: bool = False
    session_start: Optional[str] = None
    bars_collected: int = 0
    min_bars_needed: int = 50
    current_price: Optional[float] = None
    last_signal: Optional[str] = None
    signal_confidence: Optional[float] = None
    active_orders: int = 0
    current_position: int = 0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    message: str = ""
    # NEW: Cooldown and filter status
    cooldown_remaining_seconds: int = 0
    simulation_mode: bool = False
    filters_applied: List[str] = field(default_factory=list)
    # NEW: Hybrid pipeline status
    hybrid_pipeline_enabled: bool = False
    hybrid_market_trend: str = ""
    hybrid_volatility_regime: str = ""
    # NEW: AWS Agents status
    aws_agents_enabled: bool = False
    aws_agent_decision: str = ""
    # Order lock telemetry
    pending_order: bool = False
    order_lock_reason: str = ""
    # Feature cache
    last_atr: float = 0.0


class LiveTradingManager:
    """Manages live trading session with WebSocket broadcasting.
    
    ENHANCED with:
    - Trade cooldown period (configurable, default 5 minutes)
    - Candle close validation (wait for candle to close before entry)
    - Higher-timeframe levels (PDH/PDL, WH/WL, PWH/PWL)
    - Trend confirmation filters
    - Simulation mode for testing without real orders
    """
    def reset_state(self):
        """Manual override: reset last trade time and release order lock."""
        if self.executor:
            try:
                self.executor.force_release_order_lock(reason="manual override", cancel_tracked=True)
            except Exception as e:
                logger.error(f"Failed to release order lock: {e}")
        try:
            # Reset last trade time in tracker if available
            if hasattr(self, 'tracker') and self.tracker and hasattr(self.tracker, 'order_tracker'):
                self.tracker.order_tracker.reset_symbol_state(self.settings.data.ibkr_symbol)
            # Also try via executor's order_tracker if present
            if self.executor and hasattr(self.executor, 'order_tracker'):
                self.executor.order_tracker.reset_symbol_state(self.settings.data.ibkr_symbol)
        except Exception as e:
            logger.error(f"Failed to reset last trade time: {e}")

    def _note_pending_exit_reason(self, trade_cycle_id: Optional[str], reason: str) -> None:
        """Persist the 'intended' exit reason for the active trade.

        We often only deterministically detect the actual close when the position transitions to
        flat (e.g., bracket TP/SL fill). This helper lets us carry a more specific reason
        (SIGNAL_EXIT/TIME_EXIT/STOP_LOSS/PROFIT_TARGET/etc) into trade_outcomes at closure time.
        """
        if not trade_cycle_id:
            return
        try:
            if not hasattr(self, "_pending_exit_reasons"):
                self._pending_exit_reasons = {}
            self._pending_exit_reasons[str(trade_cycle_id)] = {
                "reason": str(reason),
                "noted_at": now_cst().isoformat(),
            }
        except Exception:
            pass

    def _get_pending_exit_reason(self, trade_cycle_id: Optional[str]) -> Optional[str]:
        if not trade_cycle_id or not hasattr(self, "_pending_exit_reasons"):
            return None
        try:
            entry = self._pending_exit_reasons.get(str(trade_cycle_id))
            return entry.get("reason") if isinstance(entry, dict) else None
        except Exception:
            return None

    def _clear_pending_exit_reason(self, trade_cycle_id: Optional[str]) -> None:
        if not trade_cycle_id or not hasattr(self, "_pending_exit_reasons"):
            return
        try:
            self._pending_exit_reasons.pop(str(trade_cycle_id), None)
        except Exception:
            pass

    def _infer_bracket_fill_reason(
        self,
        conn: sqlite3.Connection,
        trade_cycle_id: str,
        current_direction: Optional[str] = None,
    ) -> Optional[str]:
        """Infer PROFIT_TARGET vs STOP_LOSS on bracket-driven closes.

        Heuristic (best-effort): look at the latest execution across all orders in the trade_cycle_id
        group, then map the order_type of that order to TP vs SL.

        Returns:
            "PROFIT_TARGET" | "STOP_LOSS" | None
        """
        if not trade_cycle_id:
            return None

        conn.row_factory = sqlite3.Row
        latest = conn.execute(
            """
            SELECT e.order_id, e.timestamp, e.price, e.realized_pnl
            FROM executions e
            JOIN orders o ON o.order_id = e.order_id
            WHERE o.trade_cycle_id = ?
            ORDER BY e.timestamp DESC
            LIMIT 1
            """,
            (trade_cycle_id,),
        ).fetchone()
        if not latest:
            return None

        order_row = conn.execute(
            "SELECT order_type, limit_price, stop_price FROM orders WHERE order_id = ?",
            (int(latest["order_id"]),),
        ).fetchone()
        order_type = (order_row["order_type"] if order_row else None) or ""
        order_type = str(order_type).upper()

        # Common IB/resolved order types seen in this repo
        if order_type in {"STOP", "STP", "STOP_LIMIT"}:
            return "STOP_LOSS"
        if order_type in {"LIMIT", "LMT"}:
            return "PROFIT_TARGET"

        # Fallback: PnL sign. We don't know direction from this event alone; accept direction hint.
        pnl = float(latest["realized_pnl"] or 0.0)
        if pnl == 0.0:
            return None
        if current_direction and str(current_direction).upper() == "SHORT":
            # Closing a short: positive pnl means price fell => target hit; negative => stop hit.
            return "PROFIT_TARGET" if pnl > 0 else "STOP_LOSS"
        # Default LONG mapping
        return "PROFIT_TARGET" if pnl > 0 else "STOP_LOSS"
    
    # === CONFIGURATION CONSTANTS (sourced from config/cooldown manager defaults) ===
    DEFAULT_COOLDOWN_SECONDS = CooldownManager.DEFAULT_COOLDOWN_SECONDS
    MIN_CONFIDENCE_THRESHOLD = 0.60  # Fallback; overwritten by settings.trading.min_confidence_for_trade
    POLL_INTERVAL_SECONDS = 5  # Fallback; overridden by settings.trading.poll_interval_seconds if present
    CANDLE_PERIOD_SECONDS = 60
    MIN_COOLDOWN_MINUTES = CooldownManager.MIN_COOLDOWN_MINUTES
    MAX_COOLDOWN_MINUTES = CooldownManager.MAX_COOLDOWN_MINUTES
    COOLDOWN_WARNING_MINUTES = CooldownManager.COOLDOWN_WARNING_MINUTES
    PERSISTED_COOLDOWN_MAX_AGE = CooldownManager.PERSISTED_COOLDOWN_MAX_AGE
    FUTURE_COOLDOWN_TOLERANCE_SECONDS = CooldownManager.FUTURE_COOLDOWN_TOLERANCE_SECONDS
    
    def __init__(
        self,
        settings: Settings,
        simulation_mode: bool = False,
        reset_state_on_start: Optional[bool] = None,
    ):
        self.settings = settings
        self.simulation_mode = simulation_mode  # NEW: Dry run mode
        self.trading_mode: TradingMode = self._detect_trading_mode(settings)
        self.contract_spec: ContractSpec = get_contract_spec(
            settings.data.ibkr_symbol,
            settings.trading,
        )
        self._commission_per_side = get_commission_per_side(
            self.contract_spec,
            self.trading_mode,
            getattr(settings.trading, "commission_per_contract", None),
        )
        self.feature_flags: FeatureFlagsConfig = getattr(settings, "features", FeatureFlagsConfig())
        self._enforce_entry_risk_checks = self.feature_flags.enforce_entry_risk_checks
        self._enforce_wait_blocking = self.feature_flags.enforce_wait_blocking
        self._enforce_reduce_only_exits = self.feature_flags.enforce_reduce_only_exits
        self._enable_learning_hooks = self.feature_flags.enable_learning_hooks
        self.ib: Optional[IB] = None
        self.executor: Optional[TradeExecutor] = None
        self.tracker: Optional[LivePerformanceTracker] = None
        self.engine: Optional[StrategyEngine] = None
        self.risk: Optional[RiskManager] = None
        self.rag_storage: Optional[RAGStorage] = None
        self.metrics_logger: Optional["DecisionMetricsLogger"] = None
        self.telegram: Optional[TelegramNotifier] = None
        reset_flag = reset_state_on_start
        if reset_flag is None:
            reset_flag = getattr(getattr(settings, "trading", None), "reset_state_on_start", False)
        self._reset_state_on_start: bool = bool(reset_flag)
        
        # NEW: Trading filters for multi-timeframe analysis
        self.trading_filters: Optional[TradingFilters] = None
        self._entry_filter_cfg = getattr(getattr(settings, "trading", None), "entry_filters", None)
        self._min_confidence_for_trade = getattr(
            getattr(settings, "trading", None),
            "min_confidence_for_trade",
            self.MIN_CONFIDENCE_THRESHOLD,
        )
        self._min_stop_distance_ticks = getattr(
            getattr(settings, "trading", None),
            "min_stop_distance_ticks",
            4,
        )
        self._min_stop_distance = self.contract_spec.tick_size * max(1, self._min_stop_distance_ticks)
        
        # NEW: Hybrid RAG+LLM Pipeline
        self.hybrid_pipeline: Optional[HybridPipelineIntegration] = None
        self._use_hybrid_pipeline: bool = False
        hybrid_cfg = getattr(settings, "hybrid", None)
        self._allow_hybrid_legacy_fallback: bool = bool(
            getattr(hybrid_cfg, "allow_legacy_fallback", False)
        )
        
        # NEW: AWS Bedrock Agents Pipeline
        self.aws_agent_invoker: Optional[AgentInvoker] = None
        self.aws_snapshot_builder: Optional[MarketSnapshotBuilder] = None
        self._aws_agents_allowed: bool = False
        self._aws_agents_ready: bool = False

        # Hard risk gate - build RiskGateConfig from settings
        settings_gate = getattr(settings, "risk_gate", None)
        if settings_gate is None or isinstance(settings_gate, RiskGateConfig):
            gate_cfg = settings_gate or RiskGateConfig()
        else:
            # Convert settings object to RiskGateConfig with proper defaults
            gate_cfg = RiskGateConfig(
                max_contracts=getattr(settings_gate, "max_contracts", 1),
                risk_per_trade_usd=getattr(settings_gate, "risk_per_trade_usd", 60.0),
                risk_per_trade_min=getattr(settings_gate, "risk_per_trade_min", 25.0),
                risk_per_trade_max=getattr(settings_gate, "risk_per_trade_max", 75.0),
                min_stop_points=getattr(settings_gate, "min_stop_points", 4.0),
                max_stop_points=getattr(settings_gate, "max_stop_points", 12.0),
                margin_buffer_usd=getattr(settings_gate, "margin_buffer_usd", 1000.0),
                initial_margin_long=getattr(settings_gate, "initial_margin_long", 2464.0),
                initial_margin_short=getattr(settings_gate, "initial_margin_short", 2305.6),
                daily_max_loss_usd=getattr(settings_gate, "daily_max_loss_usd", 150.0),
                weekly_max_loss_usd=getattr(settings_gate, "weekly_max_loss_usd", 500.0),
                max_consecutive_losses=getattr(settings_gate, "max_consecutive_losses", 3),
                avoid_close_window_minutes=getattr(settings_gate, "avoid_close_window_minutes", 60),
                avoid_close_enabled=getattr(settings_gate, "avoid_close_enabled", True),
                peak_drawdown_enabled=getattr(settings_gate, "peak_drawdown_enabled", False),
                peak_drawdown_pct=getattr(settings_gate, "peak_drawdown_pct", 4.0),
                peak_drawdown_action=getattr(settings_gate, "peak_drawdown_action", "halt"),
                peak_drawdown_tighten_multiplier=getattr(settings_gate, "peak_drawdown_tighten_multiplier", 0.5),
                peak_drawdown_stop_buffer_points=getattr(settings_gate, "peak_drawdown_stop_buffer_points", 0.5),
                peak_drawdown_flatten_on_trigger=getattr(settings_gate, "peak_drawdown_flatten_on_trigger", True),
                peak_drawdown_reset_on_new_day=getattr(settings_gate, "peak_drawdown_reset_on_new_day", False),
            )
        gate_cfg.tick_size = getattr(settings.trading, "tick_size", gate_cfg.tick_size)
        self.risk_gate = RiskGate(gate_cfg)

        # Dynamic structural support floor — auto-computed from PDL / weekly low / OR low
        dsf_cfg = getattr(settings, "dynamic_support", None)
        self.dynamic_support_floor = DynamicSupportFloor(DynamicSupportFloorConfig(
            buffer_points=float(getattr(dsf_cfg, "buffer_points", 5.0)),
            min_sources=int(getattr(dsf_cfg, "min_sources", 1)),
            use_pdl=bool(getattr(dsf_cfg, "use_pdl", True)),
            use_weekly_low=bool(getattr(dsf_cfg, "use_weekly_low", True)),
            use_or_low=bool(getattr(dsf_cfg, "use_or_low", True)),
        ))
        
        # Knowledge base + telemetry
        rag_cfg = getattr(settings, "rag", None)
        self._rag_backend = getattr(rag_cfg, "backend", "off") if rag_cfg else "off"
        self._kb_cache_ttl = getattr(rag_cfg, "kb_cache_ttl_seconds", 120) if rag_cfg else 120
        self._kb_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._kb_cache_limit = 128
        self._local_kb: Optional[LocalKnowledgeBase] = None
        if rag_cfg and LocalKnowledgeBase is not None:
            try:
                store_path = getattr(rag_cfg, "local_store_path", "rag_data/local_kb/local_kb.sqlite")
                self._local_kb = LocalKnowledgeBase(store_path=store_path)
            except Exception as exc:
                logger.warning(f"⚠️  Local knowledge base unavailable: {exc}")
                self._local_kb = None
        kb_usage_tracker.configure(self._rag_backend, False)
        
        self.status = TradingStatus()
        self.status.simulation_mode = simulation_mode
        self.one_minute_cfg = getattr(settings, "one_minute", None)
        warmup_bars = getattr(self.one_minute_cfg, "warmup_bars", 320) if self.one_minute_cfg else 320
        self.status.min_bars_needed = max(300, warmup_bars)
        self._bar_window = max(500, getattr(self.one_minute_cfg, "window_bars", 400) if self.one_minute_cfg else 500)
        self.price_history: List[Dict] = []
        self.running = False
        self._last_price_bar_ts: Optional[datetime] = None
        # MAY 12 2026 FIX #5: Higher-timeframe (30m) trend filter state.
        # Populated by _refresh_htf_30m_trend() polling IB for native 30m bars.
        # Read by signal_processor and injected into features.attrs for the
        # strategy to apply a counter-trend block.
        self._htf_30m_closes: List[float] = []          # rolling close history
        self._htf_30m_last_bar_ts: Optional[datetime] = None
        self._htf_30m_last_fetch_ts: Optional[datetime] = None
        self._htf_30m_trend: str = "UNKNOWN"            # "UP" / "DOWN" / "NEUTRAL" / "UNKNOWN"
        self._htf_30m_ema_value: Optional[float] = None
        self._htf_30m_ema_prev: Optional[float] = None  # for slope detection
        self._trade_timestamps: List[datetime] = []
        self.stop_requested = False
        self._trade_time_lock = threading.Lock()
        self._last_submission_time: Optional[datetime] = None
        self._submission_backoff_seconds: int = max(
            5,
            getattr(getattr(settings, "trading", None), "decision_min_interval_seconds", 30),
        )
        self.agent_bus = AgentBus(default_ttl_seconds=240)
        context_dir = getattr(getattr(settings, 'data', None), 'external_context_dir', 'data/context')
        self._external_context_dir = Path(context_dir)
        self._context_refresh_mtimes: Dict[str, float] = {}
        # Component helpers
        self.cooldown_manager = CooldownManager(self)
        self.status_broadcaster = StatusBroadcaster(self)
        self.context_manager = ContextManager(self)
        self.order_coordinator = OrderCoordinator(self)
        self.risk_controller = RiskController(self)
        self.trading_session_manager = TradingSessionManager(self)
        self.market_data_coordinator = MarketDataCoordinator(self)
        self.signal_processor = SignalProcessor(self.settings, self.engine, self)
        self.trade_decision_engine = TradeDecisionEngine(self)
        self.system_health_monitor = SystemHealthMonitor(self)
        
        # Trend continuation optimizer - modifies brackets instead of close+re-enter
        from shree.execution.components.trend_continuation_optimizer import TrendContinuationOptimizer
        self.trend_continuation_optimizer = TrendContinuationOptimizer(self)

        # Exit logic helper (extracted to reduce file size)
        from shree.execution.components.exit_manager import ExitManager
        self._exit_mgr = ExitManager(self)

        learning_cfg = getattr(settings, "learning", None)
        if (
            self._enable_learning_hooks
            and learning_cfg
            and getattr(learning_cfg, "enabled", True)
        ):
            self.learning_recorder = TradeLearningRecorder(
                learning_cfg.outcomes_dir,
                learning_cfg.history_dir,
            )
        else:
            self.learning_recorder = None
        if self._local_kb and learning_cfg:
            self._bootstrap_local_kb(learning_cfg.outcomes_dir)
        
        # Trade context tracking
        self.current_trade_id: Optional[str] = None
        self.current_trade_entry_time: Optional[str] = None
        self.current_trade_entry_price: Optional[float] = None
        self.current_trade_action: Optional[str] = None  # BUY/SELL for RAG tracking
        self.current_trade_features: Optional[Dict] = None
        self._current_entry_cycle_id: Optional[str] = None  # Entry cycle ID for exit correlation
        self.current_trade_buckets: Optional[Dict] = None
        self.current_trade_rationale: Optional[Dict] = None
        self._open_trade_context: Optional[Dict[str, Any]] = None
        self._current_cycle_id: Optional[str] = None
        self._cycle_context: Dict[str, Dict[str, Any]] = {}
        self._active_reason_codes: Set[str] = set()
        
        # JAN 12, 2026: Position state tracking for MTF gate notifications
        self._last_known_position_qty: int = 0  # Track previous position for transition detection
        
        # APR 2, 2026: Dedup guard for _notify_position_closed.
        # Prevents double-counting when both the execution callback and the
        # position-transition poll detect the same close.
        self._last_close_entry_cycle_id: Optional[str] = None
        
        # FEB 5, 2026: Profit protection tracking for MES wave behavior
        self._breakeven_stop_set: bool = False  # Track if we've moved stop to breakeven
        self._partial_profit_taken: bool = False  # Track if we've taken partial profits (for 2+ contracts)
        
        # NEW: Cooldown tracking
        self._last_trade_time: Optional[datetime] = None
        raw_cooldown_minutes = getattr(
            settings.trading,
            'trade_cooldown_minutes',
            self.DEFAULT_COOLDOWN_SECONDS // 60,
        )
        if self.one_minute_cfg and getattr(self.one_minute_cfg, "cooldown_minutes", None):
            raw_cooldown_minutes = getattr(self.one_minute_cfg, "cooldown_minutes")
        sanitized_minutes = self.cooldown_manager.sanitize_cooldown_minutes(raw_cooldown_minutes)
        self.settings.trading.trade_cooldown_minutes = sanitized_minutes
        self._cooldown_seconds = sanitized_minutes * 60  # Convert minutes to seconds
        
        # NEW: Candle tracking for proper candle-close validation
        self._last_candle_processed: Optional[datetime] = None
        wait_for_close = True
        # --- Startup entry gating (prevents immediate post-restart entries) ---
        # Motivation: after restart we often have full history via bootstrap, which can trigger
        # an entry attempt on the very next completed 1m bar before state (cooldowns/MTF caches)
        # has stabilized. These knobs allow a short grace window and/or require N fresh bars.
        self._startup_utc: datetime = datetime.now(timezone.utc)
        self._startup_completed_bars: int = 0
        self._startup_grace_period_seconds: int = int(
            getattr(getattr(settings, "trading", None), "startup_grace_period_seconds", 0) or 0
        )
        self._startup_min_completed_bars: int = int(
            getattr(getattr(settings, "trading", None), "startup_min_completed_bars", 0) or 0
        )

        # Fix #14 MAR 13 2026: Restore consecutive-loss state from disk so that
        # bot restarts do not silently reset the 3-loss cooldown accumulator.
        # Fix #6 MAR 16 2026: Also restore realized_pnl_today so the $250 daily
        # loss cap survives restarts.
        # load_bot_state() handles: missing file, day rollover, stale cooldowns.
        try:
            from shree.utils.bot_state import load_bot_state
            _loss_count, _cooldown_until, _realized_pnl = load_bot_state()
            self._consecutive_loss_count: int = _loss_count
            self._extra_cooldown_until = _cooldown_until
            self._persisted_daily_pnl: float = _realized_pnl  # applied to tracker in TradingSessionManager.initialize()
        except Exception as _bs_exc:
            logger.warning(f"bot_state load failed (non-fatal): {_bs_exc}")
            self._consecutive_loss_count = 0
            self._extra_cooldown_until = None
            self._persisted_daily_pnl = 0.0
        if self._entry_filter_cfg and hasattr(self._entry_filter_cfg, "wait_for_candle_close"):
            wait_for_close = bool(self._entry_filter_cfg.wait_for_candle_close)
        self._waiting_for_candle_close: bool = wait_for_close
        
        # Callbacks for WebSocket broadcasting
        self.on_status_update: Optional[callable] = None
        self.on_signal_generated: Optional[callable] = None
        self.on_order_update: Optional[callable] = None
        self.on_trade_executed: Optional[callable] = None
        self.on_error: Optional[callable] = None
        
        if simulation_mode:
            logger.warning("🔶 SIMULATION MODE ENABLED - Orders will NOT be sent to IBKR")

        # Initialize Prometheus metrics (optional)
        try:
            from ..observability.prometheus_metrics import init_metrics, get_metrics

            prom = init_metrics(self.settings)
            if prom is not None:
                # Attach module-level handle for convenience
                self.prometheus_metrics = get_metrics()
                logger.info(f"✅ Prometheus metrics initialized on port {self.settings.observability.prometheus_port}")
            else:
                self.prometheus_metrics = None
                logger.info("⚠️  Prometheus metrics disabled (PROMETHEUS_ENABLED=false or not available)")
        except Exception as e:
            self.prometheus_metrics = None
            logger.warning(f"⚠️  Failed to initialize Prometheus metrics: {e}")
    
    def _sanitize_cooldown_minutes(self, raw_value: Any) -> int:
        """Clamp cooldown minutes to a safe range and emit warnings if needed."""
        return self.cooldown_manager.sanitize_cooldown_minutes(raw_value)
    
    def _build_trading_filters(self) -> TradingFilters:
        """Instantiate TradingFilters using YAML configuration overrides."""
        cfg = getattr(getattr(self.settings, "trading", None), "entry_filters", None)
        if not cfg:
            return TradingFilters()
        
        return TradingFilters(
            ema_fast=getattr(cfg, "ema_fast_period", 9),
            ema_slow=getattr(cfg, "ema_slow_period", 20),
            atr_period=getattr(cfg, "atr_period", 14),
            min_atr_threshold=getattr(cfg, "min_atr_threshold", 0.5),
            max_atr_threshold=getattr(cfg, "max_atr_threshold", 5.0),
            chop_zone_buffer_pct=getattr(cfg, "chop_zone_buffer_pct", 0.25),
            sr_proximity_ticks=getattr(cfg, "sr_proximity_ticks", 8),
            require_candle_close=getattr(cfg, "wait_for_candle_close", True),
            candle_period_seconds=getattr(cfg, "candle_period_seconds", 60),
            require_trend_alignment=getattr(cfg, "require_trend_alignment", True),
            allow_counter_trend=getattr(cfg, "allow_counter_trend", False),
            ema_alignment_tolerance_pct=getattr(cfg, "ema_alignment_tolerance_pct", 0.0002),
            counter_trend_penalty=getattr(cfg, "counter_trend_penalty", 0.10),
            min_atr_percentile=getattr(cfg, "min_atr_percentile", None),
            atr_percentile_lookback=getattr(cfg, "atr_percentile_lookback", 120),
            low_atr_penalty_mode=getattr(cfg, "low_atr_penalty_mode", False),
            low_atr_penalty=getattr(cfg, "low_atr_penalty", 0.10),
        )
    
    async def initialize(self):
        """Initialize trading components."""
        return await self.trading_session_manager.initialize()
    
    def _on_execution_details(self, trade, fill):
        """Handle execution details to track trade exits for RAG/learning.

        APR 2 2026 — Fix: Also update ``_last_known_position_qty`` when a
        bracket child (SL/TP) fills and the position goes flat.  Previously,
        the qty was only sampled once at the top of ``_process_trading_cycle``,
        so same-candle stop-outs (entry + exit within one 15m bar) produced a
        0→0 transition that was invisible — ``_notify_position_closed`` never
        fired and the loss was never persisted to ``bot_state.json``.
        """
        order = trade.order
        order_id = order.orderId
        parent_id = getattr(order, "parentId", None)
        is_bracket_child = parent_id is not None and parent_id > 0
        logger.debug(
            f"📩 _on_execution_details: order={order_id} parent={parent_id} "
            f"price={fill.execution.price} qty={fill.execution.shares}"
        )

        # ── Immediate position-transition bookkeeping ──────────────────
        # When an *entry* order fills, record that we now hold a position so
        # that the next cycle's transition check (prev≠0 → qty==0) can fire.
        # When a *bracket child* (SL/TP) fills, detect the close immediately
        # and invoke _notify_position_closed so the loss/win is persisted
        # even if the exit happens within the same candle as the entry.
        if is_bracket_child:
            # Bracket child fill → position is likely flat now.
            # Determine direction from the parent entry that opened the position.
            prev_qty = getattr(self, "_last_known_position_qty", 0)
            # The child's action is opposite the entry direction:
            # SL/TP for a LONG entry are SELL orders; for SHORT they are BUY.
            child_action = getattr(order, "action", "")
            direction = "LONG" if child_action == "SELL" else "SHORT"

            if prev_qty != 0:
                logger.info(
                    f"📊 Same-candle bracket fill detected: order={order_id} "
                    f"(parent={parent_id}) direction={direction} — "
                    f"updating _last_known_position_qty {prev_qty} → 0"
                )
                self._last_known_position_qty = 0
        else:
            # Entry fill → update qty so the next cycle knows we hold a position.
            action = getattr(order, "action", "")
            filled_qty = int(abs(fill.execution.shares))
            if action == "BUY":
                self._last_known_position_qty = filled_qty
            elif action == "SELL":
                self._last_known_position_qty = -filled_qty

        # This callback can fire while an event loop is already running.
        # Never call asyncio.run() from within a running loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            loop.create_task(self.order_coordinator.handle_order_fill(trade, fill))
            return

        # As a fallback (e.g., during synchronous tests), create a loop just for this call.
        asyncio.run(self.order_coordinator.handle_order_fill(trade, fill))

    def _bootstrap_local_kb(self, outcomes_dir: str) -> None:
        """Seed the local KB with any historical trade outcomes on disk."""
        if not self._local_kb or not outcomes_dir:
            return
        try:
            self._local_kb.bootstrap_from_outcomes(outcomes_dir)
        except Exception as exc:
            logger.debug(f"Local KB bootstrap skipped: {exc}")

    def _configure_aws_agents(self) -> None:
        """Decide whether AWS Agents/OpenSearch can be used and defer init."""
        rag_cfg = getattr(self.settings, "rag", None)
        backend = getattr(rag_cfg, "backend", "off") if rag_cfg else "off"
        remote_enabled = bool(
            rag_cfg
            and getattr(rag_cfg, "opensearch_enabled", False)
            and backend == "opensearch_serverless"
        )
        kb_usage_tracker.configure(self._rag_backend, remote_enabled)
        aws_cfg = getattr(self.settings, "aws_agents", None)
        allow = bool(
            AWS_AGENTS_AVAILABLE
            and remote_enabled
            and aws_cfg
            and getattr(aws_cfg, "enabled", False)
        )
        if not allow:
            self._aws_agents_allowed = False
            self._aws_agents_ready = False
            self.status.aws_agents_enabled = False
            if aws_cfg and getattr(aws_cfg, "enabled", False) and not remote_enabled:
                logger.info("ℹ️  AWS Agents disabled: OpenSearch backend not permitted")
            return
        self._aws_agents_allowed = True
        self.status.aws_agents_enabled = True
        logger.info("✅ AWS Agents permitted (lazy init)")

    def _ensure_aws_agent_invoker(self) -> bool:
        """Instantiate AWS AgentInvoker only when needed."""
        if not self._aws_agents_allowed:
            return False
        if self._aws_agents_ready and self.aws_agent_invoker and self.aws_snapshot_builder:
            return True
        try:
            self.aws_agent_invoker = AgentInvoker.from_deployed_config()
            self.aws_snapshot_builder = MarketSnapshotBuilder(
                symbol=self.settings.data.ibkr_symbol
            )
            self._aws_agents_ready = True
            logger.info("✅ AWS Agent Invoker ready (lazy load)")
            return True
        except Exception as exc:
            logger.warning(f"⚠️  Failed to initialize AWS Agents: {exc}")
            self._aws_agents_allowed = False
            self._aws_agents_ready = False
            self.status.aws_agents_enabled = False
            kb_usage_tracker.configure(self._rag_backend, False)
            return False

    def _build_kb_cache_key(
        self,
        trend: str,
        volatility: str,
        action: str,
    ) -> str:
        return self.context_manager.build_kb_cache_key(trend, volatility, action)

    def _get_cached_kb_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        return self.context_manager.get_cached_kb_result(cache_key)

    def _set_cached_kb_result(self, cache_key: str, payload: Dict[str, Any]) -> None:
        self.context_manager.set_cached_kb_result(cache_key, payload)

    def _query_local_knowledge_base(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.context_manager.query_local_knowledge_base(context)

    async def _load_historical_context(self):
        """Load historical market context at bot startup.
        
        This fetches previous day's high/low, today's high/low, weekly levels,
        and stores them for use by the RAG system and AWS agents.
        """
        try:
            logger.info("📊 Loading historical market context...")
            
            # Get historical bars from IBKR (1 day bars for PDH/PDL, 1 week for weekly levels)
            if self.executor and self.executor.ib:
                
                # Get the qualified contract
                contract = await self.executor.get_qualified_contract()
                if not contract:
                    logger.warning("⚠️ Could not get contract for historical data")
                    return

                # Fetch last 5 days of daily bars using the active loop when available
                daily_bars = None
                try:
                    loop: Optional[asyncio.AbstractEventLoop] = None
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None

                    async def _fetch(use_rth: bool):
                        ib = self.executor.ib
                        if loop and hasattr(ib, "reqHistoricalDataAsync"):
                            return await ib.reqHistoricalDataAsync(
                                contract,
                                endDateTime='',
                                durationStr='5 D',
                                barSizeSetting='1 day',
                                whatToShow='TRADES',
                                useRTH=use_rth,
                                formatDate=1,
                                timeout=10,
                            )
                        if loop:
                            return await loop.run_in_executor(
                                None,
                                lambda: ib.reqHistoricalData(
                                    contract,
                                    endDateTime='',
                                    durationStr='5 D',
                                    barSizeSetting='1 day',
                                    whatToShow='TRADES',
                                    useRTH=use_rth,
                                    formatDate=1,
                                    timeout=10,
                                ),
                            )
                        return self.executor.ib.reqHistoricalData(
                            contract,
                            endDateTime='',
                            durationStr='5 D',
                            barSizeSetting='1 day',
                            whatToShow='TRADES',
                            useRTH=use_rth,
                            formatDate=1,
                            timeout=10,
                        )

                    daily_bars = await _fetch(use_rth=True)
                    if not daily_bars:
                        daily_bars = await _fetch(use_rth=False)
                except Exception as hist_err:
                    logger.warning(f"⚠️ Historical data request failed: {hist_err}")
                    daily_bars = None
                
                if daily_bars and len(daily_bars) >= 2:
                    # Previous day's data (second to last bar)
                    prev_day = daily_bars[-2]
                    today = daily_bars[-1] if len(daily_bars) >= 1 else None
                    
                    self._historical_context = {
                        'previous_day': {
                            'date': str(prev_day.date),
                            'high': prev_day.high,
                            'low': prev_day.low,
                            'open': prev_day.open,
                            'close': prev_day.close,
                            'volume': prev_day.volume,
                        },
                        'today': {
                            'date': str(today.date) if today else str(now_cst().date()),
                            'high': today.high if today else 0,
                            'low': today.low if today else 0,
                            'open': today.open if today else 0,
                        } if today else {},
                        'weekly': {
                            'high': max(bar.high for bar in daily_bars),
                            'low': min(bar.low for bar in daily_bars),
                        },
                        'loaded_at': now_cst().isoformat(),
                    }
                    
                    pdh = self._historical_context['previous_day']['high']
                    pdl = self._historical_context['previous_day']['low']
                    prev_close = self._historical_context['previous_day']['close']
                    prev_date = self._historical_context['previous_day']['date']
                    today_date = self._historical_context.get('today', {}).get('date', 'N/A')
                    
                    logger.info(f"✅ Historical context loaded from IBKR daily bars:")
                    logger.info(f"   📈 Previous Day ({prev_date}): High={pdh:.2f}, Low={pdl:.2f}, Close={prev_close:.2f}")
                    if self._historical_context.get('today'):
                        th = self._historical_context['today'].get('high', 0)
                        tl = self._historical_context['today'].get('low', 0)
                        logger.info(f"   📊 Today ({today_date}): High={th:.2f}, Low={tl:.2f}")
                    logger.info(f"   📅 Weekly Range: {self._historical_context['weekly']['low']:.2f} - {self._historical_context['weekly']['high']:.2f}")
                    logger.info(f"   ⏱️  Loaded at: {self._historical_context['loaded_at']}")
                    
                    # Store in RAG for agents to query
                    await self._store_historical_context_in_rag()

                    # Feed PDL + weekly low into dynamic support floor
                    self.dynamic_support_floor.update_from_historical_context(self._historical_context)
                    
                else:
                    logger.warning("⚠️ Not enough historical bars received")
                    self._historical_context = {}
            else:
                logger.warning("⚠️ Executor not available for historical data")
                self._historical_context = {}
                
        except Exception as e:
            logger.error(f"❌ Failed to load historical context: {e}")
            self._historical_context = {}
    
    async def _store_historical_context_in_rag(self):
        """Store historical context in RAG system for agents to query."""
        try:
            if not self._historical_context:
                return
                
            context = self._historical_context
            today_str = now_cst().strftime("%Y-%m-%d")
            
            # Create a document with today's market context
            market_context_doc = f"""
DAILY MARKET CONTEXT - {today_str}
Symbol: {self.settings.data.ibkr_symbol}

PREVIOUS DAY LEVELS:
- Previous Day High (PDH): {context['previous_day']['high']:.2f}
- Previous Day Low (PDL): {context['previous_day']['low']:.2f}
- Previous Day Close: {context['previous_day']['close']:.2f}
- Previous Day Open: {context['previous_day']['open']:.2f}

TODAY'S LEVELS (so far):
- Today's High: {context.get('today', {}).get('high', 'N/A')}
- Today's Low: {context.get('today', {}).get('low', 'N/A')}
- Today's Open: {context.get('today', {}).get('open', 'N/A')}

WEEKLY RANGE:
- Weekly High: {context['weekly']['high']:.2f}
- Weekly Low: {context['weekly']['low']:.2f}

KEY LEVELS TO WATCH:
1. Support: PDL at {context['previous_day']['low']:.2f}
2. Resistance: PDH at {context['previous_day']['high']:.2f}
3. Pivot: Previous close at {context['previous_day']['close']:.2f}

TRADING GUIDANCE:
- If price > PDH: Bullish breakout, favor LONG positions
- If price < PDL: Bearish breakdown, favor SHORT positions
- If price between PDL and PDH: Range-bound, use mean reversion
- Watch for retests of PDH/PDL as potential entry points
"""
            
            # Store in local RAG if available
            if self.rag_storage:
                # Save as a dynamic document
                doc_path = f"rag_data/docs_dynamic/market_context_{today_str}.txt"
                import os
                os.makedirs("rag_data/docs_dynamic", exist_ok=True)
                with open(doc_path, 'w') as f:
                    f.write(market_context_doc)
                logger.info(f"✅ Stored market context in RAG: {doc_path}")
            
            # Also update the MarketSnapshotBuilder if available
            if hasattr(self, 'aws_snapshot_builder') and self.aws_snapshot_builder:
                self.aws_snapshot_builder.set_historical_levels(
                    pdh=context['previous_day']['high'],
                    pdl=context['previous_day']['low'],
                    prev_close=context['previous_day']['close'],
                    weekly_high=context['weekly']['high'],
                    weekly_low=context['weekly']['low'],
                )
                logger.info("✅ Updated AWS snapshot builder with historical levels")
                
        except Exception as e:
            logger.error(f"❌ Failed to store historical context in RAG: {e}")

    async def _bootstrap_price_history(self, min_bars: int = 60) -> None:
        """Seed price history with IB historical bars so multi-candle metrics have depth."""
        if self.price_history:
            return
        if not self.executor or not self.executor.ib:
            logger.warning("Cannot bootstrap price history - executor not ready")
            return
        contract = await self.executor.get_qualified_contract()
        if not contract:
            logger.warning("Cannot bootstrap price history - contract unavailable")
            return

        # FEB 2026: Dynamic bar size based on active timeframe
        active_tf = getattr(self, "_active_timeframe", "1m")
        bar_size_setting = getattr(self, "_bar_size_setting", "1 min")
        candle_seconds = getattr(self, "_candle_period_seconds", self.CANDLE_PERIOD_SECONDS)

        # FEB 7 2026: 15m strategy uses ALL bars for indicator computation.
        # The validated backtest (130 trades, PF 1.91) computes EMA/ATR/ADX
        # on the full 24/7 DataFrame (including overnight). RTH-only indicators
        # give different values (EMA differs ~2.75pts, ATR 7.69 vs 8.50) and
        # produce worse results (135 trades, PF 0.66).
        #
        # Session isolation is done in the strategy itself:
        #   - generate() only runs during RTH entry window (11:00-14:59 ET)
        #   - _prev_close is never updated during overnight
        #   - OR is only computed from RTH opening bars
        #
        # Therefore: useRTH=False for all timeframes to match backtest.
        use_rth_bootstrap = False

        if active_tf == "15m":
            # 15m bars: request enough duration for min_bars candles
            # IB max for seconds is 86400; for larger, use days
            needed_seconds = min_bars * candle_seconds  # e.g., 60 * 900 = 54000
            if needed_seconds > 86400:
                duration_str = f"{max(2, needed_seconds // 86400 + 1)} D"
            else:
                duration_str = f"{max(needed_seconds, 1800)} S"
        else:
            duration_seconds = max(min_bars * self.CANDLE_PERIOD_SECONDS, 900)
            duration_str = f"{duration_seconds} S"

        try:
            bars = await self.executor.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr=duration_str,
                barSizeSetting=bar_size_setting,
                whatToShow='TRADES',
                useRTH=use_rth_bootstrap,
                formatDate=2,
            )
        except AttributeError:
            bars = self.executor.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration_str,
                barSizeSetting=bar_size_setting,
                whatToShow='TRADES',
                useRTH=use_rth_bootstrap,
                formatDate=2,
            )
        except Exception as exc:
            logger.warning(f"Failed to bootstrap {active_tf} bars: {exc}")
            return
        if not bars:
            logger.warning("Historical bootstrap returned no bars")
            return
        history: List[Dict[str, Any]] = []
        for bar in bars[-max(min_bars, len(bars)) :]:
            ts = getattr(bar, "date", None)
            if isinstance(ts, datetime):
                dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            elif isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    dt = datetime.utcnow().replace(tzinfo=timezone.utc)
            else:
                dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            candle = {
                "timestamp": utc_to_cst(dt),
                "open": float(getattr(bar, "open", 0.0)),
                "high": float(getattr(bar, "high", 0.0)),
                "low": float(getattr(bar, "low", 0.0)),
                "close": float(getattr(bar, "close", 0.0)),
                "volume": int(getattr(bar, "volume", 0)),
            }
            history.append(candle)
        if history:
            window = max(self._bar_window, min_bars)
            self.price_history = history[-window:]
            self.status.bars_collected = len(self.price_history)
            self._last_price_bar_ts = history[-1]["timestamp"]
            
            # Staleness validation: Log first/last bar timestamps and check age
            first_bar_ts = history[0]["timestamp"]
            last_bar_ts = history[-1]["timestamp"]
            current_time = now_cst()
            staleness_seconds = (current_time - last_bar_ts).total_seconds() if isinstance(last_bar_ts, datetime) else 999
            
            logger.info(
                f"📚 Bootstrapped {len(self.price_history)} historical {active_tf} bars for structural context"
            )
            logger.info(
                f"   ⏱️  First bar: {first_bar_ts}, Last bar: {last_bar_ts}, Now: {current_time.isoformat()}"
            )
            # Staleness threshold scales with bar size
            default_stale = 1200 if active_tf == "15m" else 120
            logger.info(
                f"   ⏱️  Data age: {staleness_seconds:.0f}s (acceptable if <{default_stale}s)"
            )
            
            # Staleness behavior is configurable via one_minute config
            stale_threshold = default_stale
            fail_on_stale = False
            if getattr(self, "one_minute_cfg", None):
                stale_threshold = int(getattr(self.one_minute_cfg, "bootstrap_stale_seconds", stale_threshold))
                fail_on_stale = bool(getattr(self.one_minute_cfg, "fail_on_bootstrap_stale", False))

            # Warn if data is stale (older than threshold)
            if staleness_seconds > stale_threshold:
                msg = (
                    f"⚠️  Bootstrapped bars are STALE ({staleness_seconds:.0f}s old). "
                    "Market may be closed or data feed delayed."
                )
                if fail_on_stale:
                    # Fail early so initialization aborts and operator can investigate
                    logger.error(msg + " Failing startup due to configuration.")
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg)

    async def _fetch_latest_minute_bar(self) -> Optional[Dict[str, Any]]:
        """Fetch the most recent completed 1-minute bar from IBKR."""
        if not self.executor or not self.executor.ib:
            return None
        try:
            contract = await self.executor.get_qualified_contract()
            use_rth = not getattr(self.one_minute_cfg, "use_eth_session", False)
            bars = await self.executor.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr="180 S",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=use_rth,
                formatDate=2,
            )
            if not bars:
                return None
            last_bar = bars[-1]
            ts = getattr(last_bar, "date", None)
            if isinstance(ts, datetime):
                bar_ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            elif isinstance(ts, str):
                bar_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            else:
                bar_ts = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            if self._last_price_bar_ts and bar_ts <= self._last_price_bar_ts:
                return None
            candle = {
                "timestamp": utc_to_cst(bar_ts),
                "open": float(getattr(last_bar, "open", 0.0)),
                "high": float(getattr(last_bar, "high", 0.0)),
                "low": float(getattr(last_bar, "low", 0.0)),
                "close": float(getattr(last_bar, "close", 0.0)),
                "volume": int(getattr(last_bar, "volume", 0)),
            }
            return candle
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Latest minute bar fetch failed: {exc}")
            return None

    async def _fetch_latest_15m_bar(self) -> Optional[Dict[str, Any]]:
        """Fetch the most recent completed 15-minute bar from IBKR.
        
        FEB 2026: Added for live 15m strategy support. Requests 30 min of
        15-min bars to ensure we get at least one completed bar.
        
        FEB 7 2026: Uses useRTH=False — the validated backtest (130 trades,
        PF 1.91) computes indicators on ALL bars. RTH-only indicators give
        different EMA/ATR values. Session filtering is done in the strategy.
        
        MAR 25 2026: Upgraded error logging from debug→warning so fetch
        failures are visible in production logs. Added stale-data watchdog
        that logs a warning when no new bar arrives for >20 minutes.
        """
        if not self.executor or not self.executor.ib:
            logger.warning("15m bar fetch skipped: executor or IB connection not available")
            return None
        try:
            contract = await self.executor.get_qualified_contract()
            bars = await self.executor.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr="1800 S",  # 30 min of data to get at least 1 completed 15m bar
                barSizeSetting="15 mins",
                whatToShow="TRADES",
                useRTH=False,  # ALL bars — matches backtest indicator computation
                formatDate=2,
            )
            if not bars:
                logger.warning("15m bar fetch returned empty result from IBKR")
                return None
            last_bar = bars[-1]
            ts = getattr(last_bar, "date", None)
            if isinstance(ts, datetime):
                bar_ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            elif isinstance(ts, str):
                bar_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            else:
                bar_ts = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            if self._last_price_bar_ts and bar_ts <= self._last_price_bar_ts:
                # Stale-data watchdog: warn if we haven't received a new bar in >20 min
                if not hasattr(self, "_last_new_bar_time"):
                    self._last_new_bar_time = time.monotonic()
                stale_minutes = (time.monotonic() - self._last_new_bar_time) / 60
                if stale_minutes > 20 and not hasattr(self, "_stale_bar_warned"):
                    logger.warning(
                        f"⚠️  No new 15m bar in {stale_minutes:.0f}min! "
                        f"Last bar ts={bar_ts}, stored={self._last_price_bar_ts}, "
                        f"IB returned {len(bars)} bars, last close={getattr(last_bar, 'close', '?')}"
                    )
                    self._stale_bar_warned = True
                elif stale_minutes <= 20:
                    # Reset warning flag when freshness is restored
                    self._stale_bar_warned = False
                return None
            # New bar arrived — reset stale watchdog
            self._last_new_bar_time = time.monotonic()
            if hasattr(self, "_stale_bar_warned"):
                self._stale_bar_warned = False
            candle = {
                "timestamp": utc_to_cst(bar_ts),
                "open": float(getattr(last_bar, "open", 0.0)),
                "high": float(getattr(last_bar, "high", 0.0)),
                "low": float(getattr(last_bar, "low", 0.0)),
                "close": float(getattr(last_bar, "close", 0.0)),
                "volume": int(getattr(last_bar, "volume", 0)),
            }
            return candle
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Latest 15m bar fetch FAILED: {exc}")
            return None

    # ------------------------------------------------------------------
    #  MAY 12 2026 FIX #5: Higher-timeframe (30m) trend filter
    # ------------------------------------------------------------------
    async def _fetch_latest_30m_bars(self, limit: int = 50) -> Optional[List[Any]]:
        """Fetch the most recent 30-minute bars from IBKR (native 30m).

        Used by Fix #5 to compute a higher-timeframe trend that gates
        counter-trend signals from the 15m strategy.  IBKR provides
        native 30m bars — we don't resample 15m to 30m because IB's
        native aggregation aligns to the exchange's session boundaries
        and is the source of truth for 30m closes.
        """
        if not self.executor or not self.executor.ib:
            return None
        try:
            contract = await self.executor.get_qualified_contract()
            # Request enough history to compute a stable EMA(20).
            # 50 bars * 30m = 25 hours; safer to request '3 D' to absorb
            # weekends / holidays / partial-fill bootstrapping.
            bars = await self.executor.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr="3 D",
                barSizeSetting="30 mins",
                whatToShow="TRADES",
                useRTH=False,                # match 15m strategy convention
                formatDate=2,
            )
            if not bars:
                return None
            return list(bars)[-limit:]
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"30m bar fetch failed (non-fatal): {exc}")
            return None

    async def _refresh_htf_30m_trend(self) -> None:
        """Fetch latest 30m bars and recompute the HTF trend.

        Rate-limited: only fetches if last successful fetch was > 60s ago.
        Maintains a rolling buffer of closes and computes EMA(20).
        Trend = UP   if close > EMA AND EMA slope positive
                DOWN if close < EMA AND EMA slope negative
                NEUTRAL otherwise
        """
        # Skip if HTF filter is disabled in config
        cfg = self.one_minute_cfg
        if cfg is None:
            return
        if not bool(getattr(cfg, "ft_htf_filter_enabled", True)):
            return

        # Rate limit
        now = now_cst()
        interval_s = int(getattr(cfg, "ft_htf_refresh_interval_s", 60) or 60)
        if (self._htf_30m_last_fetch_ts is not None
                and (now - self._htf_30m_last_fetch_ts).total_seconds() < interval_s):
            return

        bars = await self._fetch_latest_30m_bars(limit=60)
        if not bars or len(bars) < 5:
            return

        # Extract closes.  Use the LATEST COMPLETED bar — drop the in-progress
        # last bar by checking timestamp continuity (IB returns only completed
        # bars when endDateTime="" in most cases, but be defensive).
        closes: List[float] = []
        for b in bars:
            c = getattr(b, "close", None)
            if c is not None:
                try:
                    closes.append(float(c))
                except (TypeError, ValueError):
                    pass
        if len(closes) < 5:
            return

        # Track the latest bar timestamp — used for staleness reporting
        last_b = bars[-1]
        ts = getattr(last_b, "date", None)
        try:
            if isinstance(ts, datetime):
                bar_ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            elif isinstance(ts, str):
                bar_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            else:
                bar_ts = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            self._htf_30m_last_bar_ts = utc_to_cst(bar_ts) if bar_ts.tzinfo else bar_ts
        except Exception:
            pass

        # Compute EMA(N)
        ema_period = int(getattr(cfg, "ft_htf_ema_period", 20) or 20)
        k = 2.0 / (ema_period + 1.0)
        ema_vals: List[float] = []
        ema = closes[0]
        for c in closes:
            ema = (c * k) + (ema * (1 - k))
            ema_vals.append(ema)

        # Persist closes (truncate to a reasonable window)
        self._htf_30m_closes = closes[-100:]
        self._htf_30m_ema_prev = self._htf_30m_ema_value
        self._htf_30m_ema_value = ema_vals[-1]
        self._htf_30m_last_fetch_ts = now

        # Trend classification
        close_now = closes[-1]
        ema_now = ema_vals[-1]
        # Slope: compare against ema 3 bars ago (1.5 hours of 30m history)
        ema_slope = ema_vals[-1] - ema_vals[-3] if len(ema_vals) >= 3 else 0.0

        prev_trend = self._htf_30m_trend
        if close_now > ema_now and ema_slope > 0:
            self._htf_30m_trend = "UP"
        elif close_now < ema_now and ema_slope < 0:
            self._htf_30m_trend = "DOWN"
        else:
            self._htf_30m_trend = "NEUTRAL"

        if prev_trend != self._htf_30m_trend:
            logger.info(
                f"📈 HTF 30m trend: {prev_trend} → {self._htf_30m_trend} "
                f"(close={close_now:.2f}, ema={ema_now:.2f}, slope={ema_slope:+.2f})"
            )

    async def _fetch_latest_bar(self) -> Optional[Dict[str, Any]]:
        """Fetch latest bar using the active timeframe (1m or 15m).
        
        FEB 2026: Dispatcher that routes to the correct bar fetcher based on
        the _active_timeframe attribute set during initialization.
        """
        active_tf = getattr(self, "_active_timeframe", "1m")
        if active_tf == "15m":
            return await self._fetch_latest_15m_bar()
        return await self._fetch_latest_minute_bar()

    def _ingest_completed_bar(self, bar: Dict[str, Any]) -> None:
        """Append a completed bar and maintain rolling window."""
        self.price_history.append(bar)
        if len(self.price_history) > self._bar_window:
            self.price_history = self.price_history[-self._bar_window :]
        self.status.bars_collected = len(self.price_history)
        self._last_price_bar_ts = bar.get("timestamp")

        # Track fresh completed bars since process startup (used for startup entry gating).
        try:
            self._startup_completed_bars += 1
        except Exception:
            # Be resilient if attribute isn't present for any reason
            pass
        
        # === JAN 8 2026 FIX: Feed 1m bar to 5m aggregator ===
        # This enables the 5-minute trend filter to actually work
        # FEB 2026: Skip MTF aggregation for 15m bars (already higher timeframe)
        if getattr(self, "_active_timeframe", "1m") == "1m":
            if hasattr(self, 'signal_processor') and self.signal_processor:
                self.signal_processor.update_mtf_candle(bar)
        
        # Update trend if not set by hybrid pipeline (ensures trend is always available)
        if not self.status.hybrid_market_trend and len(self.price_history) >= 10:
            self._compute_fallback_trend()

    def _compute_fallback_trend(self) -> None:
        """Compute trend from price history when hybrid pipeline hasn't set it.
        
        FEB 2026 FIX: Use 30 bars and steeper threshold (0.10 vs 0.05) to avoid
        classifying normal MES noise as trend. A slope of 0.05 over 20 bars is
        just ~1 point, which MES moves on a single tick.
        """
        try:
            closes = [float(bar.get("close", 0.0)) for bar in self.price_history[-30:] if isinstance(bar, dict)]
            if len(closes) >= 10:
                slope = np.polyfit(np.arange(len(closes)), closes, 1)[0]
                # Use threshold to avoid noise in flat markets
                if slope < -0.10:
                    trend = "DOWNTREND"
                elif slope > 0.10:
                    trend = "UPTREND"
                else:
                    trend = "NEUTRAL"
                self.status.hybrid_market_trend = trend
                logger.debug(f"Fallback trend computed: {trend} (slope={slope:.5f})")
        except Exception as exc:
            logger.debug(f"Fallback trend computation failed: {exc}")

    async def start(self):
        """Start the live trading loop."""
        return await self.trading_session_manager.start()
    
    async def _process_trading_cycle(self, current_price: float, bar_timestamp: Optional[datetime] = None):
        """Check position state first, then handle exits or entries."""
        # New cycle id for correlation/logging
        self._current_cycle_id = uuid.uuid4().hex[:12]
        self._cycle_context[self._current_cycle_id] = {}

        position = None
        if self.executor:
            try:
                position = await self.executor.get_current_position()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Unable to fetch position for trading cycle: {exc}")
        qty = getattr(position, "quantity", 0) if position else 0

        # === JAN 12, 2026: Detect bracket fill (TP/SL hit) by position transition ===
        # If we had a position last cycle but now flat, bracket must have filled
        prev_qty = self._last_known_position_qty
        if prev_qty != 0 and qty == 0:
            # Position was closed externally (bracket fill / TP / SL)
            direction = "LONG" if prev_qty > 0 else "SHORT"
            logger.info(f"📊 Position transition detected: {prev_qty} -> {qty} (bracket fill or external close)")
            # Deterministic trade closure persistence for audit trail
            trade_cycle_id = (
                getattr(self, "_current_entry_cycle_id", None)
                or getattr(self, "_current_cycle_id", None)
                or getattr(self, "current_trade_id", None)
            )
            specific_reason = self._get_pending_exit_reason(trade_cycle_id)
            if not specific_reason and trade_cycle_id and getattr(self, "executor", None) and getattr(self.executor, "order_tracker", None):
                try:
                    with sqlite3.connect(self.executor.order_tracker.db_path) as conn:
                        inferred = self._infer_bracket_fill_reason(
                            conn=conn,
                            trade_cycle_id=str(trade_cycle_id),
                            current_direction=direction,
                        )
                    specific_reason = inferred
                except Exception:
                    specific_reason = None
            specific_reason = specific_reason or "BRACKET_FILL"
            realized_pnl = 0.0
            if trade_cycle_id and getattr(self, "executor", None) and getattr(self.executor, "order_tracker", None):
                try:
                    pnl = self.executor.order_tracker.finalize_trade_exit(
                        trade_cycle_id=str(trade_cycle_id),
                        exit_time=now_cst().isoformat(),
                        exit_price=float(current_price) if current_price is not None else None,
                        exit_reason=specific_reason,
                        extra={
                            "direction": direction,
                            "source": "position_transition",
                            "pending_exit_reason": specific_reason,
                        },
                    )
                    if pnl and "realized_pnl" in pnl:
                        realized_pnl = float(pnl["realized_pnl"] or 0.0)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"trade_outcomes finalize skipped: {exc}")

            # Once we are flat, don't carry the prior exit reason into the next trade.
            self._clear_pending_exit_reason(trade_cycle_id)

            try:
                log_structured_event(
                    agent="live_manager",
                    event_type="TRADE_CLOSED",
                    message="Position transitioned to flat",
                    payload={
                        "trade_cycle_id": trade_cycle_id,
                        "direction": direction,
                        "exit_reason": specific_reason,
                        "realized_pnl": realized_pnl,
                    },
                )
            except Exception:
                pass

            self._notify_position_closed(
                close_reason=specific_reason,
                direction=direction,
                pnl=realized_pnl,
            )
        self._last_known_position_qty = qty

        if qty:
            logger.info(f"📊 Current position detected: {qty} contracts")
            exit_handled = await self._check_position_exit_signals(current_price)
            if exit_handled:
                return
            logger.debug("📊 Holding position ({}); skipping new entries", qty)
            return

        # Entry signals only - flat position
        if self._should_block_new_entry():
            return
        return await self.signal_processor.process_trading_cycle(current_price, bar_timestamp=bar_timestamp)

    async def _check_position_exit_signals(self, current_price: Optional[float]) -> bool:
        """Check if existing position should be closed."""
        return await self._exit_mgr.check_position_exit_signals(current_price)

    async def _check_position_exit_logic(self, position) -> Optional[Dict[str, Any]]:
        """Check if existing position should be exited based on simple P&L bands."""
        return await self._exit_mgr.check_position_exit_logic(position)

    async def _execute_position_exit(self, exit_signal: Dict[str, Any], position, current_price: Optional[float] = None) -> bool:
        """Execute position exit order honoring exit signal payload."""
        return await self._exit_mgr.execute_position_exit(exit_signal, position, current_price)
    
    def _notify_position_closed(
        self,
        close_reason: str,
        direction: str,
        pnl: float,
    ) -> None:
        """Notify signal processor about position close to trigger cooldown.
        
        JAN 12, 2026: This is called after any position close to:
        1. Reset decision state for fresh evaluation
        2. Trigger cooldown (wait for 15m candle)
        3. Clear cached trends to prevent immediate re-entry
        
        Args:
            close_reason: "TP", "SL", "MANUAL", "TIMEOUT", "TREND_FLIP", etc.
            direction: "LONG" or "SHORT"
            pnl: Realized P&L
        """
        # APR 2, 2026 — Dedup guard: both the execution callback path
        # (handle_order_fill → finalize_trade) and the position-transition
        # poll can invoke this for the same close.  Only process once per
        # trade cycle when pnl≠0 to avoid double-counting losses/wins.
        _entry_cycle = getattr(self, "_current_entry_cycle_id", None)
        if pnl != 0.0 and _entry_cycle:
            if _entry_cycle == getattr(self, "_last_close_entry_cycle_id", None):
                logger.debug(
                    f"📊 _notify_position_closed dedup: already processed "
                    f"cycle={_entry_cycle} — skipping"
                )
                return
            self._last_close_entry_cycle_id = _entry_cycle

        # APR 10 2026: Log every completed trade to CSV for live performance analysis.
        if pnl != 0.0:
            try:
                from shree.utils.live_trade_journal import log_live_trade
                _ctx = getattr(self, "_open_trade_context", None) or {}
                _dd_tier = 0
                if hasattr(self, "risk_gate") and self.risk_gate:
                    _dd_tier = self.risk_gate.get_drawdown_tier()
                _exit_px = _ctx.get("_last_exit_price", 0.0)
                if not _exit_px and pnl != 0.0 and _ctx.get("entry_price"):
                    _qty = _ctx.get("quantity", 1) or 1
                    _pv = 5.0
                    _ep = _ctx["entry_price"]
                    if _ctx.get("is_long"):
                        _exit_px = round(_ep + pnl / (_qty * _pv), 2)
                    else:
                        _exit_px = round(_ep - pnl / (_qty * _pv), 2)
                log_live_trade(
                    trade_context=_ctx,
                    exit_price=_exit_px,
                    realized_pnl=pnl,
                    exit_reason=close_reason,
                    dd_tier=_dd_tier,
                )
            except Exception as _journal_exc:
                logger.debug(f"live_trade_journal skipped: {_journal_exc}")

        # Fix #14 (MAR 12 2026): Consecutive-loss cooldown.
        # After ft_consecutive_loss_trigger (default 3) SL hits in a row,
        # impose an extended cooldown of cooldown_on_consecutive_losses_minutes (default 30).
        # Resets on any win or breakeven.
        #
        # NOTE: IB often reports 0.0 via finalize_trade_exit ("no root order" bug).
        # The real P&L is computed in order_coordinator.finalize_trade (price fallback)
        # and re-invokes this method with the correct value. Skip the consecutive-loss
        # accounting when pnl==0.0 to avoid the false-reset that would otherwise fire
        # from the first (zero-pnl) call from LTM's position-transition path.
        _trading_cfg = getattr(getattr(self, "settings", None), "trading", None)
        _loss_trigger = int(getattr(_trading_cfg, "ft_consecutive_loss_trigger", 3))
        _loss_cooldown_mins = int(getattr(_trading_cfg, "cooldown_on_consecutive_losses_minutes", 30))

        # Fix #6b MAR 17 2026: Accumulate trade P&L into tracker.daily_pnl
        # BEFORE reading it for bot_state persistence. Previously,
        # tracker.daily_pnl was never updated with trade results (update_equity
        # was only called with realized_pnl=0.0), so the persisted daily P&L
        # was always stale — missing the most recent trade's result.
        if pnl != 0.0 and self.tracker:
            self.tracker.daily_pnl += pnl
            self.tracker.total_realized_pnl += pnl
            logger.info(
                f"📊 Daily P&L updated: ${pnl:+.2f} → "
                f"daily=${self.tracker.daily_pnl:.2f}"
            )

        if pnl != 0.0:
            if pnl < 0:
                self._consecutive_loss_count = getattr(self, "_consecutive_loss_count", 0) + 1
                logger.warning(
                    f"📉 Consecutive losses: {self._consecutive_loss_count} "
                    f"(trigger at {_loss_trigger})"
                )
                if self._consecutive_loss_count >= _loss_trigger:
                    self._extra_cooldown_until = (
                        datetime.now(timezone.utc) + timedelta(minutes=_loss_cooldown_mins)
                    )
                    logger.warning(
                        f"🚫 CONSECUTIVE_LOSS_COOLDOWN: {self._consecutive_loss_count} losses "
                        f"→ blocking entries for {_loss_cooldown_mins} min "
                        f"(until {self._extra_cooldown_until.strftime('%H:%M:%S')} UTC)"
                    )
                # Persist after every loss increment so a restart picks up the count
                try:
                    from shree.utils.bot_state import save_bot_state
                    save_bot_state(
                        consecutive_loss_count=self._consecutive_loss_count,
                        extra_cooldown_until=getattr(self, "_extra_cooldown_until", None),
                        realized_pnl_today=self._get_daily_pnl_for_persist(),
                    )
                except Exception as _exc:
                    logger.debug(f"bot_state save skipped: {_exc}")
            else:
                if getattr(self, "_consecutive_loss_count", 0) > 0:
                    logger.info(
                        f"✅ Consecutive loss streak reset (was {self._consecutive_loss_count})"
                    )
                self._consecutive_loss_count = 0
                # Persist the reset so a restart doesn't re-load stale count
                try:
                    from shree.utils.bot_state import save_bot_state
                    save_bot_state(
                        consecutive_loss_count=0,
                        extra_cooldown_until=None,
                        realized_pnl_today=self._get_daily_pnl_for_persist(),
                    )
                except Exception as _exc:
                    logger.debug(f"bot_state save skipped: {_exc}")

        if hasattr(self, "signal_processor") and self.signal_processor:
            self.signal_processor.notify_position_closed(
                close_reason=close_reason,
                direction=direction,
                pnl=pnl,
            )
            logger.info(
                f"📊 MTF Gate notified of position close: {direction} {close_reason} "
                f"(pnl=${pnl:.2f})"
            )

    def _notify_position_opened(self, direction: str) -> None:
        """Notify signal processor about position open.
        
        Args:
            direction: "LONG" or "SHORT"
        """
        # Reset profit protection flags for new position
        self._breakeven_stop_set = False
        self._partial_profit_taken = False
        
        if hasattr(self, "signal_processor") and self.signal_processor:
            self.signal_processor.notify_position_opened(direction)
            logger.info(f"📊 MTF Gate notified of position open: {direction}")

    def _should_block_new_entry(self) -> bool:
        """Pre-entry gate: startup hold + lock/active orders/cooldown.

        Startup hold (NEW):
          - blocks *entries only* for a short time after process start, and/or until
            N newly completed bars have been processed.
          - exits/position management are unaffected because this is called only
            on the entry path (when flat).
        """

        # --- Startup gates ---
        if getattr(self, "_startup_grace_period_seconds", 0) > 0:
            elapsed = (datetime.now(timezone.utc) - self._startup_utc).total_seconds()
            remaining = self._startup_grace_period_seconds - elapsed
            if remaining > 0:
                logger.info(
                    f"Entry blocked: startup grace {remaining:.0f}s remaining (elapsed={elapsed:.0f}s)"
                )
                return True

        if (
            getattr(self, "_startup_min_completed_bars", 0) > 0
            and getattr(self, "_startup_completed_bars", 0) < self._startup_min_completed_bars
        ):
            logger.info(
                f"Entry blocked: waiting for {self._startup_completed_bars}/{self._startup_min_completed_bars} completed bars after startup"
            )
            return True
        # Fix #14: Consecutive-loss cooldown extended block
        _extra_until = getattr(self, "_extra_cooldown_until", None)
        if _extra_until and datetime.now(timezone.utc) < _extra_until:
            remaining_m = (_extra_until - datetime.now(timezone.utc)).total_seconds() / 60
            logger.warning(
                f"Entry blocked: consecutive-loss cooldown {remaining_m:.0f}m remaining"
            )
            return True

        # Order lock
        if self.executor and self.executor.is_order_locked():
            logger.info("Entry blocked: order lock active ({})", self.executor.get_order_lock_reason())
            return True

        # Active orders
        if self.executor:
            try:
                active = self.executor.get_active_order_count(sync=True)
                if active > 0:
                    logger.info("Entry blocked: {} active orders pending", active)
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Active order check skipped: {exc}")

        # Recent submission backoff
        if self._last_submission_time:
            elapsed = (datetime.now(timezone.utc) - self._last_submission_time).total_seconds()
            if elapsed < self._submission_backoff_seconds:
                logger.info(
                    "Entry blocked: submission backoff %.1fs remaining",
                    self._submission_backoff_seconds - elapsed,
                )
                return True

        # Cooldown
        if self._cooldown_seconds:
            last_trade = getattr(self, "_last_trade_time", None)
            if last_trade:
                now = datetime.now(timezone.utc)
                elapsed = (now - last_trade).total_seconds()
                remaining = self._cooldown_seconds - elapsed
                if remaining > 0:
                    logger.info("Entry blocked: cooldown {:.1f}s remaining", remaining)
                    return True

        if self.one_minute_cfg:
            cfg = self.one_minute_cfg
            now = datetime.now(timezone.utc)
            hour_cutoff = now - timedelta(hours=1)
            day_cutoff = now - timedelta(hours=24)
            trades_last_hour = [t for t in self._trade_timestamps if t >= hour_cutoff]
            trades_last_day = [t for t in self._trade_timestamps if t >= day_cutoff]
            if len(trades_last_hour) >= getattr(cfg, "max_trades_per_hour", 3):
                logger.info("Entry blocked: max trades per hour reached ({})", len(trades_last_hour))
                return True
            if len(trades_last_day) >= getattr(cfg, "max_trades_per_day", 8):
                logger.info("Entry blocked: max trades per day reached ({})", len(trades_last_day))
                return True

        return False

    def monitor_current_position(self) -> bool:
        """Monitor the existing 2-contract LONG position and trigger forced exit on large loss."""
        current_position = 2  # From logs
        current_price = 6979.75  # From logs
        entry_price = 34896.87 / (2 * 5)  # 3489.687 per contract

        pnl_per_contract = (current_price - entry_price) * 5
        total_pnl = pnl_per_contract * current_position

        logger.info("🔍 Current Position Monitor:")
        logger.info("   Position: +{} LONG", current_position)
        logger.info("   Entry: {:.2f}", entry_price)
        logger.info("   Current: {:.2f}", current_price)
        logger.info("   Total P&L: ${:.2f}", total_pnl)

        if total_pnl <= -150:
            logger.warning("🚨 Significant loss detected, forcing position exit")
            return True

        return False

    def _normalize_entry_price(self, entry_price: float, current_price: float) -> float:
        """Normalize notional entry costs (e.g., futures multiplier embedded).
        
        IB sometimes reports avg_cost in different formats:
        - Actual price: 6896.25
        - Notional (price * multiplier): 34481.25 (needs / 5)
        - Half notional: 3448.19 (needs * 2 for some reason)
        """
        if current_price <= 0:
            return entry_price
        
        ratio = entry_price / current_price
        configured_trading = getattr(getattr(self, "settings", None), "trading", None)
        custom_multipliers = getattr(configured_trading, "notional_multipliers", None) if configured_trading else None
        multipliers = tuple(custom_multipliers) if custom_multipliers else (5, 10, 20, 25, 50, 100, 200)
        
        # Check if entry_price is a multiple of current_price (notional value)
        for multiplier in multipliers:
            if abs(ratio - multiplier) <= 0.25:
                return entry_price / multiplier
        
        # FEB 5 2026 FIX: Check if entry_price is a fraction of current_price (half notional, etc)
        # This happens when IB reports avg_cost in a strange format
        for divisor in (0.5, 0.4, 0.6, 2.0, 2.5):  # Common fractions
            if abs(ratio - divisor) <= 0.05:
                # Entry price is much smaller than current - likely divided incorrectly
                normalized = entry_price / divisor
                # Verify the normalized value is reasonable (within 20% of current_price)
                if 0.80 <= (normalized / current_price) <= 1.20:
                    logger.warning(f"📊 Normalized entry price: {entry_price:.2f} → {normalized:.2f} (divisor={divisor})")
                    return normalized
        
        return entry_price

    def _get_max_exit_price_gap(self, current_price: float) -> float:
        """Return configurable/relative gap threshold for exit sanity checks."""
        return self._exit_mgr.get_max_exit_price_gap(current_price)

    def _get_exit_thresholds(self) -> Tuple[float, float, float]:
        """Return (profit_points, loss_points, max_hold_hours) using config, with sensible fallbacks."""
        return self._exit_mgr.get_exit_thresholds()

    def _generate_exit_signal_for_short(self, current_price: float, position) -> Optional[Dict[str, float]]:
        """Generate exit signals for short positions."""
        return self._exit_mgr.generate_exit_signal_for_short(current_price, position)

    def _generate_exit_signal_for_long(self, current_price: float, position) -> Optional[Dict[str, float]]:
        """Generate exit signals for long positions."""
        return self._exit_mgr.generate_exit_signal_for_long(current_price, position)

    def _get_position_age_hours(self, position) -> Optional[float]:
        """Return position age (hours) if timestamp is available."""
        return self._exit_mgr.get_position_age_hours(position)
    
    async def _log_position_status(self, position=None, current_price: Optional[float] = None) -> None:
        """Log current position, P&L, and duration for dashboard visibility."""
        if not self.executor:
            return
        pos = position
        if pos is None:
            pos = await self.executor.get_current_position()
        if not pos or pos.quantity == 0:
            return
        
        price = current_price
        if price is None:
            try:
                price = await self.executor.get_current_price()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Unable to fetch price for position dashboard: {exc}")
                return
        if price is None:
            return
        
        # Prefer position avg_cost; fallback to known entry from logs
        entry_price = float(getattr(pos, "avg_cost", 0.0) or 0.0)
        if entry_price <= 0:
            entry_price = 34914.38 / 5  # From logs: convert entry cost to price
        
        multiplier = getattr(self.contract_spec, "point_value", 5) or 5
        unrealized_pnl = (price - entry_price) * pos.quantity * multiplier
        duration_hours = self._get_position_age_hours(pos)
        duration_str = "unknown"
        if duration_hours is not None:
            hours = int(duration_hours)
            minutes = int((duration_hours - hours) * 60)
            duration_str = f"{hours}h {minutes}m"
        
        logger.info("📊 Position Status:")
        logger.info(f"   Quantity: {pos.quantity}")
        logger.info(f"   Entry Price: {entry_price:.2f}")
        logger.info(f"   Current Price: {price:.2f}")
        logger.info(f"   Unrealized P&L: ${unrealized_pnl:.2f}")
        logger.info(f"   Duration: {duration_str}")

    async def emergency_close_position(self, reason: str = "EMERGENCY_CLOSE") -> Optional[Any]:
        """Force close any existing position immediately (manual override)."""
        if not self.executor:
            logger.warning("⚠️ Emergency close requested but executor unavailable")
            return None
        
        position = await self.executor.get_current_position()
        if not position or position.quantity == 0:
            logger.info("ℹ️ Emergency close requested but no open position")
            return None
        
        action = "SELL" if position.quantity > 0 else "BUY"
        quantity = abs(position.quantity)
        
        price = None
        try:
            price = await self.executor.get_current_price()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"⚠️ Emergency close: could not fetch price, using avg cost. Error: {exc}")
        if price is None:
            price = getattr(position, "avg_cost", None) or 0.0
        
        logger.warning(f"🚨 Emergency position close: {action} {quantity} @ ~{price:.2f}")
        try:
            order_id = await self.executor.place_order(
                action=action,
                quantity=quantity,
                limit_price=price,
                stop_loss=None,
                take_profit=None,
                reduce_only=True,
                entry_price=price,
                metadata={
                    "emergency_close": True,
                    "reason": reason,
                    "original_position": position.quantity,
                    "trade_cycle_id": getattr(self, "_current_cycle_id", None),
                },
            )
            self._record_last_trade_timestamp()
            await self._broadcast_order_update(
                {
                    "type": "EXIT",
                    "action": action,
                    "quantity": quantity,
                    "price": price,
                    "order_id": order_id,
                    "emergency": True,
                }
            )
            return order_id
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ Emergency close failed: {exc}")
            await self._broadcast_error(f"Emergency close failed: {exc}")
            return None

    def _publish_feature_snapshot(self, features, current_price: float) -> None:
        """Publish most recent feature row to the agent bus for awareness."""
        if features is None or features.empty:
            return
        try:
            latest = features.iloc[-1]
            ts = features.index[-1]
            timestamp = ts.isoformat() if hasattr(ts, "isoformat") else now_cst().isoformat()
        except Exception:
            latest = features.iloc[-1]
            timestamp = now_cst().isoformat()
        self.market_data_coordinator.publish_feature_snapshot(latest, current_price, timestamp)

    def _refresh_external_context(self) -> None:
        """Reload cached news/macro context if files changed."""
        self.market_data_coordinator.refresh_external_context()

    def _publish_account_context(self) -> None:
        """Share account/risk state with other agents."""
        self.market_data_coordinator.publish_account_context()

    def _compute_structural_metrics(self, features) -> Dict[str, float]:
        """Derive structural metrics from the full candle buffer."""
        return self.risk_controller.compute_structural_metrics(features)

    def _apply_structural_weighting(self, signal, metrics: Dict[str, float]) -> float:
        """Adjust signal confidence based on structural context."""
        return self.risk_controller.apply_structural_weighting(signal, metrics)

    async def _handle_hybrid_pipeline_failure(self, current_price: float, exc: Exception) -> None:
        """Force HOLD when hybrid pipeline raises unexpected exception."""
        logger.exception("Hybrid pipeline error - forcing HOLD")
        self._add_reason_code("HYBRID_PIPELINE_ERROR")
        log_structured_event(
            agent="live_manager",
            event_type="hybrid.pipeline_error",
            message="Hybrid pipeline exception forced HOLD",
            payload={
                "trade_cycle_id": self._current_cycle_id,
                "exception": repr(exc),
            },
        )
        self.status.last_signal = "HOLD"
        self.status.signal_confidence = 0.0
        self.status.message = "Hybrid pipeline unavailable - holding"
        self._current_pipeline_result = None
        hold_signal = SimpleNamespace(action="HOLD", confidence=0.0, metadata={"error": str(exc)})
        await self._broadcast_signal(hold_signal, current_price)
        await self._broadcast_status()

    def _persist_structural_snapshot(
        self,
        structural_metrics: Dict[str, float],
        rag_context: Dict[str, Any],
        signal,
    ) -> None:
        """Persist blended metrics for downstream historical analysis."""
        self.risk_controller.persist_structural_snapshot(structural_metrics, rag_context, signal)

    def _update_status_from_tracker(self) -> None:
        """Sync status fields from tracker metrics."""
        self.status_broadcaster.update_status_from_tracker()

    async def _process_hybrid_signal(
        self,
        signal,
        pipeline_result,
        current_price: float,
        features,
    ):
        """Process a signal from the hybrid RAG+LLM pipeline.
        
        This is the new 3-layer decision path:
        1. Rule Engine has already evaluated filters
        2. RAG has retrieved similar trades and docs
        3. LLM has made final decision with reasoning
        
        Args:
            signal: HybridSignal from pipeline
            pipeline_result: HybridPipelineResult with full context
            current_price: Current price
            features: Features DataFrame
        """
        # Update status
        self.status.last_signal = signal.action
        self.status.signal_confidence = signal.confidence
        try:
            row = features.iloc[-1]
            self.status.last_atr = float(row.get("ATR_14", row.get("atr", 0.0)))
        except Exception:
            self.status.last_atr = 0.0
        
        # Enforce minimum confidence threshold (Safety Floor)
        # This prevents the pipeline from executing low-confidence signals (e.g. 0.24) even if deemed "valid" by internal logic
        if signal.action != "HOLD" and signal.confidence < self._min_confidence_for_trade:
            logger.info(
                f"  ↳ BLOCKED: Signal confidence {signal.confidence:.2f} < threshold {self._min_confidence_for_trade:.2f}"
            )
            return

        # Broadcast signal
        await self._broadcast_signal(signal, current_price)
        
        # Get current position snapshot
        current_position = await self.executor.get_current_position()
        self.status.current_position = current_position.quantity if current_position else 0
        self.status.active_orders = self.executor.get_active_order_count()
        if current_position:
            self.status.unrealized_pnl = await self.executor.get_unrealized_pnl()
            self.tracker.update_equity(current_price, realized_pnl=0.0)
            atr_val = float(features.iloc[-1].get("ATR_14", 0.0))
            await self.executor.update_trailing_stops(current_price, atr_val)
        await self._broadcast_status()
        
        # Check if pipeline blocked the trade
        if signal.action == "HOLD":
            if pipeline_result and pipeline_result.rule_engine.filters_blocked:
                logger.info(f"  ↳ Trade blocked by hybrid filters: {pipeline_result.rule_engine.filters_blocked}")
            else:
                logger.info(f"  ↳ Signal is HOLD, skipping order placement")
            return
        
        # === AWS AGENTS: Consult Decision Agent and Risk Agent for BUY/SELL signals ===
        aws_approved = True  # Default to approved if AWS agents disabled
        aws_adjustment = 0.0
        cycle_ctx = self._cycle_context.setdefault(self._current_cycle_id, {})
        
        if self._aws_agents_allowed:
            if not self._ensure_aws_agent_invoker():
                logger.debug("AWS Agents unavailable - skipping remote consult")
            else:
                try:
                    logger.info(f"🤖 Consulting AWS Agents for {signal.action} signal...")
                    
                    # Build market snapshot for agents
                    snapshot = self.aws_snapshot_builder.build(
                        current_price=current_price,
                        features=features
                    )
                    
                    # Build account metrics
                    account_metrics = {
                        'current_position': self.status.current_position,
                        'unrealized_pnl': self.status.unrealized_pnl,
                        'daily_pnl': self.status.daily_pnl,
                    }
                    
                    # Use the full agent pipeline via get_trading_decision
                    aws_result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.aws_agent_invoker.get_trading_decision(
                            market_snapshot=snapshot,
                            account_metrics=account_metrics,
                        )
                    )
                    
                    logger.info(f"  📊 AWS Decision: {aws_result.get('decision')} (conf={aws_result.get('confidence', 0):.2%})")
                    logger.info(
                        "  🛡️ AWS Risk: allowed={allowed}, flags={flags}, size_multiplier={multiplier:.2f}",
                        allowed=aws_result.get('allowed_to_trade'),
                        flags=aws_result.get('risk_flags', []),
                        multiplier=aws_result.get('size_multiplier', 0.0) or 0.0,
                    )
                    cycle_ctx["aws"] = aws_result
                    log_structured_event(
                        agent="live_manager",
                        event_type="aws.decision",
                        message=f"{aws_result.get('decision')} allowed={aws_result.get('allowed_to_trade')}",
                        payload=aws_result,
                    )
                    if self.agent_bus:
                        self.agent_bus.publish(
                            "aws_decision",
                            aws_result,
                            producer="aws_agents",
                            ttl_seconds=300,
                        )
                    
                    # Check if AWS allowed the trade
                    risk_flags = aws_result.get('risk_flags') or []
                    advisory_only = aws_result.get('advisory_only', False)
                    aws_allowed = aws_result.get('allowed_to_trade')
                    if aws_allowed is None:
                        aws_allowed = not bool(risk_flags)
                    if not aws_allowed and not risk_flags:
                        logger.info("  🔎 AWS disallowed trade without flags; treating as cautionary only")
                        aws_allowed = True
                    aws_decision = aws_result.get('decision', 'WAIT')
                    aws_confidence = aws_result.get('confidence', 0)

                    wait_ctx = WaitDecisionContext(
                        decision=aws_decision,
                        advisory_only=advisory_only,
                        confidence=aws_confidence,
                        size_multiplier=aws_result.get('size_multiplier'),
                    )
                    # HARD GUARDRAIL: Always check WAIT blocking (default block_on_wait=True)
                    # Only allow override if signal confidence exceeds threshold
                    wait_should_block = should_block_on_wait(
                        wait_ctx,
                        self.settings.aws_agents.block_on_wait,
                        self.settings.aws_agents.wait_override_confidence,
                        signal.confidence,
                    )
                    if wait_should_block:
                        self._add_reason_code("AWS_WAIT")
                        log_structured_event(
                            agent="live_manager",
                            event_type="aws.wait_block",
                            message="Trade blocked by WAIT advisory",
                            payload={
                                "trade_cycle_id": self._current_cycle_id,
                                "aws_decision": aws_decision,
                                "signal_confidence": signal.confidence,
                                "override_threshold": self.settings.aws_agents.wait_override_confidence,
                            },
                            correlation_id=self._current_cycle_id,
                        )
                        logger.warning(
                            f"🛑 Trade BLOCKED: AWS WAIT decision (conf={aws_confidence:.2%}, "
                            f"signal_conf={signal.confidence:.2f}, threshold={self.settings.aws_agents.wait_override_confidence:.2f})"
                        )
                        return
                    elif aws_decision.upper() == "WAIT":
                        # WAIT was overridden due to high signal confidence
                        self._add_reason_code("AWS_WAIT_OVERRIDE")
                        logger.info(
                            f"⚠️ AWS WAIT overridden: signal confidence {signal.confidence:.2f} > "
                            f"threshold {self.settings.aws_agents.wait_override_confidence:.2f}"
                        )
                    
                    force_block = bool(risk_flags and not aws_allowed)
                    
                    if force_block:
                        aws_approved = False
                        logger.warning(f"  ⚠️ AWS Risk Agent REJECTED trade: {risk_flags}")
                    elif aws_decision == 'WAIT':
                        wait_penalty = -0.10
                        if signal.confidence >= max(self._min_confidence_for_trade, 0.55) and not risk_flags:
                            wait_penalty = -0.05
                            logger.info("  📌 AWS issued WAIT but local signal strong; soft penalty applied")
                        aws_adjustment = wait_penalty
                        if advisory_only:
                            logger.info("  🛈 AWS WAIT advisory only - not blocking trade")
                    elif not aws_allowed:
                        aws_adjustment = min(aws_adjustment, -0.08)
                        logger.info("  ⚠️ AWS suggests reducing conviction (allowed=False with no flags)")
                    elif aws_decision == signal.action:
                        # AWS agrees with our signal
                        aws_adjustment = +0.05
                        logger.info(f"  📈 AWS Agents agree with {signal.action}: boosting confidence by {aws_adjustment:+.2f}")
                    elif aws_decision in ['BUY', 'SELL'] and aws_decision != signal.action:
                        aws_adjustment = -0.1
                        logger.warning(f"  ⚠️ AWS disagrees: we say {signal.action}, AWS says {aws_decision}")
                    
                    # Apply AWS adjustment to signal confidence
                    if aws_adjustment != 0:
                        signal.confidence = max(0.0, min(1.0, signal.confidence + aws_adjustment))
                        logger.info(f"  🔄 Adjusted confidence: {signal.confidence:.2f}")
                    aws_size_mult = aws_result.get('size_multiplier')
                    if aws_size_mult:
                        if not isinstance(signal.metadata, dict):
                            signal.metadata = {}
                        signal.metadata["aws_size_multiplier"] = aws_size_mult
                        logger.info(f"  📐 AWS size multiplier applied: {aws_size_mult:.2f}")
                    
                except Exception as e:
                    logger.warning(f"  ⚠️ AWS Agent consultation failed: {e} - proceeding with original signal")
        
        # Refresh position/active orders after potential AWS latency
        current_position = await self.executor.get_current_position()
        self.status.current_position = current_position.quantity if current_position else 0
        try:
            self.status.active_orders = self.executor.get_active_order_count(sync=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Active order count skipped after AWS consult: {exc}")

        # If AWS agents rejected the trade, convert to HOLD
        if not aws_approved:
            logger.info(f"  🛑 Trade blocked by AWS Risk Agent - converting to HOLD")
            return

        # CRITICAL: Check for active orders FIRST
        active_orders = self.executor.get_active_order_count(sync=True)
        if active_orders > 0:
            logger.info(f"  ↳ {active_orders} active orders pending, waiting for completion")
            return
        
        # FINAL SAFETY: Re-check confidence after all adjustments (sentiment, AWS, etc.)
        # If adjustments (like sentiment REDUCE_SIZE) dropped it below the floor, we must ABORT.
        min_conf = self._min_confidence_for_trade
        if signal.confidence < min_conf:
            logger.warning(
                f"🛑 Trade ABORTED: Weighted confidence {signal.confidence:.3f} dropped below floor {min_conf:.2f} "
                f"after adjustments (Sentiment/AWS)"
            )
            return

        # Check if we should exit existing position
        if current_position and current_position.quantity != 0:
            is_buy_signal = signal.action in ["BUY", "SCALP_BUY"]
            is_sell_signal = signal.action in ["SELL", "SCALP_SELL"]
            if (current_position.quantity > 0 and is_sell_signal) or \
               (current_position.quantity < 0 and is_buy_signal):
                logger.info(f"  ↳ HYBRID EXIT: Position={current_position.quantity}, Signal={signal.action}")
                exit_qty = abs(current_position.quantity)
                await self._place_exit_order(signal.action, exit_qty, current_price)
                
                # Log trade exit through hybrid pipeline
                if self.hybrid_pipeline and hasattr(self, '_current_pipeline_result'):
                    pd = getattr(self, '_current_pipeline_result', None)
                    market_data_for_exit = getattr(pd, 'market_data', None) if pd is not None else None
                    self.hybrid_pipeline.log_trade_exit(
                        exit_price=current_price,
                        exit_reason="SIGNAL_EXIT",
                        market_data=market_data_for_exit,
                        pipeline_result=pd,
                    )
                return
            else:
                logger.info(f"  ↳ Position open (qty={current_position.quantity}), same direction, no action")
                return
        
        # Place order using hybrid pipeline's risk parameters
        logger.info(f"  ↳ Placing HYBRID order: {signal.action}")
        await self._place_hybrid_order(signal, pipeline_result, current_price, features)
    
    async def _place_exit_order(self, action: str, quantity: int, exit_price: Optional[float] = None):
        """Place a market order to exit existing position."""
        return await self._exit_mgr.place_exit_order(action, quantity, exit_price)
    
    async def _query_aws_knowledge_base(
        self,
        features,
        current_price: float,
        proposed_action: str,
    ) -> dict:
        """Return KB-derived adjustment using AWS (remote) or local fallback."""
        try:
            row = features.iloc[-1]
            ema_9 = float(row.get('EMA_9', row.get('ema_9', current_price)))
            ema_20 = float(row.get('EMA_20', row.get('ema_20', current_price)))
            atr = float(row.get('ATR_14', row.get('atr', 10)))
            if ema_9 > ema_20 and current_price > ema_9:
                trend = 'UPTREND'
            elif ema_9 < ema_20 and current_price < ema_9:
                trend = 'DOWNTREND'
            else:
                trend = 'RANGE'
            if atr > 15:
                volatility = 'HIGH'
            elif atr > 8:
                volatility = 'MED'
            else:
                volatility = 'LOW'

            query_context = {
                "action": proposed_action,
                "trend": trend,
                "volatility": volatility,
                "confidence": float(row.get('signal_confidence', 0.0)),
            }
            cache_key = self._build_kb_cache_key(trend, volatility, proposed_action)
            cached = self._get_cached_kb_result(cache_key)
            if cached:
                kb_usage_tracker.record_query(cache_hit=True, remote_call=False)
                return cached

            rag_cfg = getattr(self.settings, "rag", None)
            remote_enabled = bool(
                self._aws_agents_allowed
                and rag_cfg
                and getattr(rag_cfg, "opensearch_enabled", False)
                and self._rag_backend == "opensearch_serverless"
            )

            if not remote_enabled or not self._ensure_aws_agent_invoker():
                local_result = self._query_local_knowledge_base(query_context)
                if local_result:
                    kb_usage_tracker.record_query(cache_hit=False, remote_call=False)
                    local_result["trend"] = trend
                    local_result["volatility"] = volatility
                    self._set_cached_kb_result(cache_key, local_result)
                    return local_result
                kb_usage_tracker.record_avoidance()
                return {}

            snapshot = self.aws_snapshot_builder.build(
                price=current_price,
                trend=trend,
                volatility=volatility,
                rsi=float(row.get('RSI_14', row.get('rsi', 50))),
                atr=atr,
                ema_9=ema_9,
                ema_20=ema_20,
            )
            response = self.aws_agent_invoker.agent_client.invoke_decision_agent(
                market_snapshot=snapshot,
            )
            similar_patterns = response.get('similar_patterns', 0)
            kb_confidence = response.get('confidence', 0.5)
            reasoning = response.get('reason', '')

            confidence_adjustment = 0.0
            if similar_patterns > 0:
                if kb_confidence >= 0.7:
                    confidence_adjustment = 0.15
                elif kb_confidence >= 0.6:
                    confidence_adjustment = 0.08
                elif kb_confidence <= 0.4:
                    confidence_adjustment = -0.15
                elif kb_confidence <= 0.5:
                    confidence_adjustment = -0.05

            payload = {
                'confidence_adjustment': confidence_adjustment,
                'similar_patterns': similar_patterns,
                'historical_win_rate': kb_confidence,
                'reasoning': reasoning[:200] if reasoning else '',
                'trend': trend,
                'volatility': volatility,
            }
            kb_usage_tracker.record_query(cache_hit=False, remote_call=True)
            self._set_cached_kb_result(cache_key, payload)
            return payload
            
        except Exception as e:
            logger.warning(f"AWS KB query error: {e}")
            kb_usage_tracker.record_avoidance()
            return {}
    
    async def _process_aws_agent_signal(
        self,
        features,
        current_price: float,
    ):
        """Process a signal from the AWS Bedrock Agents pipeline.
        
        This is the 4-agent decision path:
        1. Data Ingestion Agent cleans and structures data
        2. Decision Engine Agent analyzes and recommends trade
        3. Risk Control Agent validates and sizes position
        4. Learning Agent updates knowledge base
        
        Args:
            features: Features DataFrame
            current_price: Current price
        """
        # Build market snapshot from features
        row = features.iloc[-1]
        self.status.last_atr = float(row.get("ATR_14", row.get("atr", 0.0)))
        
        snapshot = self.aws_snapshot_builder.build(
            price=current_price,
            trend=self._get_trend_from_features(row),
            volatility=self._get_volatility_from_features(row),
            rsi=float(row.get('RSI_14', row.get('rsi', 50))),
            atr=float(row.get('ATR_14', row.get('atr', 10))),
            ema_9=float(row.get('EMA_9', row.get('ema_9', current_price))),
            ema_20=float(row.get('EMA_20', row.get('ema_20', current_price))),
            volume=int(row.get('volume', 50000)),
        )
        
        # Get account metrics
        current_position = await self.executor.get_current_position()
        position_qty = current_position.quantity if current_position else 0
        
        account_metrics = {
            'current_pnl_today': self.tracker.get_daily_pnl() if self.tracker else 0,
            'current_position': position_qty,
            'losing_streak': getattr(self, '_losing_streak', 0),
            'trades_today': getattr(self, '_trades_today', 0),
            'account_balance': self.settings.trading.initial_capital,
            'open_risk': abs(position_qty * 50) if position_qty else 0,
        }
        
        # Invoke AWS Agents for decision
        logger.info("🤖 Invoking AWS Bedrock Agents...")
        loop = asyncio.get_event_loop()
        decision = await loop.run_in_executor(
            None,
            lambda: self.aws_agent_invoker.get_trading_decision(
                market_snapshot=snapshot,
                account_metrics=account_metrics,
            ),
        )
        
        # Update status
        action = decision.get('decision', 'WAIT')
        confidence = decision.get('confidence', 0)
        
        self.status.last_signal = action
        self.status.signal_confidence = confidence
        self.status.aws_agent_decision = f"{action} ({confidence:.0%})"
        
        logger.info(
            f"📈 AWS Agent Decision: {action} "
            f"(conf={confidence:.2%}, allowed={decision.get('allowed_to_trade')})"
        )
        
        # Broadcast signal
        await self._broadcast_aws_agent_signal(decision, current_price)
        
        # Update position status
        self.status.current_position = position_qty
        self.status.active_orders = self.executor.get_active_order_count()
        
        if current_position:
            self.status.unrealized_pnl = await self.executor.get_unrealized_pnl()
        
        await self._broadcast_status()
        
        # If WAIT or not allowed, skip order placement
        if action == 'WAIT' or not decision.get('allowed_to_trade', False):
            reason = decision.get('reason', 'No signal')
            risk_flags = decision.get('risk_flags', [])
            logger.info(f"  ↳ No trade: {reason}")
            if risk_flags:
                logger.info(f"  ↳ Risk flags: {risk_flags}")
            return
        
        # Check for active orders
        active_orders = self.executor.get_active_order_count(sync=True)
        if active_orders > 0:
            logger.info(f"  ↳ {active_orders} active orders pending, waiting for completion")
            return
        
        # Check if we should exit existing position
        if position_qty != 0:
            is_buy_signal = action == "BUY"
            is_sell_signal = action == "SELL"
            
            if (position_qty > 0 and is_sell_signal) or (position_qty < 0 and is_buy_signal):
                logger.info(f"  ↳ AWS AGENT EXIT: Position={position_qty}, Signal={action}")
                exit_qty = abs(position_qty)
                await self._place_exit_order(action, exit_qty, current_price)
                return
            else:
                logger.info(f"  ↳ Position open (qty={position_qty}), same direction, no action")
                return
        
        # Place new order
        adjusted_size = max(1, int(decision.get('adjusted_size', 1)))
        stop_loss = decision.get('stop_loss')
        take_profit = decision.get('take_profit')
        
        logger.info(f"  ↳ Placing AWS AGENT order: {action} {adjusted_size} contracts")
        await self._place_aws_agent_order(action, adjusted_size, current_price, stop_loss, take_profit, decision)
    
    def _get_trend_from_features(self, row) -> str:
        """Extract trend classification from feature row."""
        return self.risk_controller.get_trend_from_features(row)
    
    def _get_volatility_from_features(self, row) -> str:
        """Extract volatility classification from feature row."""
        return self.risk_controller.get_volatility_from_features(row)

    def _get_live_vix_price(self) -> Optional[float]:
        """Return the current VX front-month price (VIX proxy) from the feed.

        Returns None if the feed is unavailable or stale, letting callers
        fall back to ATR-only behaviour.
        """
        sp = getattr(self, "signal_processor", None)
        if sp is None:
            return None
        vx_feed = getattr(sp, "_vx_feed", None)
        if vx_feed is None:
            return None
        try:
            if vx_feed.is_stale():
                return None
            return vx_feed.get_vx_price()
        except Exception:  # noqa: BLE001
            return None
    
    async def _broadcast_aws_agent_signal(self, decision: dict, current_price: float):
        """Broadcast AWS agent signal to WebSocket clients."""
        if self.on_signal_generated:
            try:
                await self.on_signal_generated({
                    'type': 'aws_agent_signal',
                    'decision': decision.get('decision'),
                    'confidence': decision.get('confidence', 0),
                    'allowed_to_trade': decision.get('allowed_to_trade'),
                    'adjusted_size': decision.get('adjusted_size', 0),
                    'risk_flags': decision.get('risk_flags', []),
                    'reason': decision.get('reason', ''),
                    'price': current_price,
                    'timestamp': now_cst().isoformat(),
                    'latency_ms': decision.get('latency_ms', 0),
                })
            except Exception as e:
                logger.error(f"Failed to broadcast AWS agent signal: {e}")
    
    async def _place_aws_agent_order(
        self,
        action: str,
        quantity: int,
        current_price: float,
        stop_loss: float = None,
        take_profit: float = None,
        decision: dict = None,
    ):
        """Place an order based on AWS agent decision.
        
        Args:
            action: BUY or SELL
            quantity: Number of contracts
            current_price: Current market price
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            decision: Full agent decision for logging
        """
        import uuid
        
        logger.info(
            f"📍 Placing AWS AGENT order: {action} {quantity} @ ~{current_price:.2f} "
            f"(SL={stop_loss}, TP={take_profit})"
        )

        # MAY 8 2026: Trading Manager veto hook for the AWS_AGENT path.
        # AWS Agents are currently disabled (OpenSearch backend not permitted)
        # but if they're re-enabled this path must not bypass the TM gate.
        # Only the latest_decision freshness fallback applies here — AWS path
        # doesn't carry a per-signal ts on the decision dict.
        try:
            from ..trading_manager.decision_log import latest_decision
            tm_path = "logs/manager_decisions.jsonl"
            latest = latest_decision(tm_path)
            if latest:
                import time as _time
                import datetime as _dt
                tm_rec = None
                try:
                    latest_ts = _dt.datetime.fromisoformat(
                        latest.get("ts", "")
                    ).timestamp()
                    if _time.time() - latest_ts <= 60:
                        tm_rec = latest
                except Exception:
                    pass
                if tm_rec and tm_rec.get("decision") == "REJECT":
                    logger.warning(
                        "🛡️  TRADING MANAGER VETO (AWS_AGENT): {} | reason: {}",
                        tm_rec.get("decision"),
                        str(tm_rec.get("reasoning", ""))[:200],
                    )
                    return
        except Exception as _tm_err:
            logger.error(
                "Trading Manager veto hook error in _place_aws_agent_order "
                "(fail-open): {}", _tm_err,
            )

        try:
            metadata = self._prepare_order_metadata(
                {"strategy_name": "aws_agents", "signal_source": "aws_agents"},
                current_price,
                "aws_agents",
            )

            allowed, reason, signal_key = await self.order_coordinator.enforce_entry_gates(
                action,
                metadata,
            )
            if not allowed:
                logger.info("AWS agent entry blocked by gate: {}", reason)
                self._add_reason_code(reason)
                return

            if self.simulation_mode:
                # Simulation mode - just log, don't place real order
                logger.info(f"🔶 [SIMULATION] Would place: {action} {quantity} @ {current_price:.2f}")
                self._record_submission_timestamp()
                # 🔄 MTF GATE: Notify position opened - simulation
                position_direction = "long" if action in ("BUY", "SCALP_BUY") else "short"
                self._notify_position_opened(position_direction)
                return

            allowed_gate, gate_levels = await self._enforce_risk_gate(
                action,
                quantity,
                current_price,
                decision.get("atr", 0.0) if decision else 0.0,
                stop_loss,
                take_profit,
            )
            if not allowed_gate:
                return
            if gate_levels:
                stop_loss = gate_levels.get("stop_loss", stop_loss)
                take_profit = gate_levels.get("take_profit", take_profit)
            
            if not self._validate_entry_guard(
                current_price,
                stop_loss,
                take_profit,
                quantity,
                action,
            ):
                self._add_reason_code("RISK_BLOCKED_INVALID_PROTECTION")
                log_structured_event(
                    agent="live_manager",
                    event_type="risk.entry_blocked",
                    message="Entry blocked by hard guardrails",
                    payload={"trade_cycle_id": self._current_cycle_id},
                )
                return
            self.order_coordinator.record_signal_key(signal_key)
            
            # Place the order
            try:
                current_pos = await self.executor.get_current_position()
                if current_pos and getattr(current_pos, "quantity", 0) != 0:
                    logger.info("Position changed before AWS submit (qty={}); skipping entry", current_pos.quantity)
                    return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Position recheck failed; skipping AWS entry: {}", exc)
                return
            result = await self.executor.place_order(
                action=action,
                quantity=quantity,
                limit_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_price=current_price,
                metadata=metadata,
            )
            order_id = result.trade.order.orderId if result and result.trade else None
            
            logger.info(f"✅ AWS Agent order placed: ID={order_id}")
            
            # Track trade for RAG updates.
            # CRITICAL: ensure we retain a stable trade_cycle_id that the executor/tracker
            # uses for orders + trade_outcomes. Without this, finalize_trade_exit can't find
            # the root order and we end up relying on BACKFILL-only outcomes.
            entry_trade_cycle_id = None
            try:
                entry_trade_cycle_id = (
                    (metadata or {}).get("trade_cycle_id")
                    or getattr(self, "_current_cycle_id", None)
                )
            except Exception:
                entry_trade_cycle_id = getattr(self, "_current_cycle_id", None)

            if entry_trade_cycle_id:
                # Used by position-transition close detection.
                self._current_entry_cycle_id = str(entry_trade_cycle_id)
                # Prefer using the trade_cycle_id as the canonical trade identifier.
                self.current_trade_id = str(entry_trade_cycle_id)
            else:
                # Fallback for legacy callsites.
                self.current_trade_id = str(uuid.uuid4())
            self.current_trade_entry_time = now_cst().isoformat()
            self.current_trade_entry_price = current_price
            self.current_trade_features = decision.get('decision_details', {}) if decision else {}
            self.current_trade_rationale = {
                'source': 'aws_agents',
                'decision': decision,
            }
            
            # Update counters (submission recorded; cooldown on fill)
            if result and (result.fill_price or result.filled_quantity):
                self._record_last_trade_timestamp()
                # 🔄 MTF GATE: Notify position opened (transition to IN_POSITION state)
                position_direction = "long" if action in ("BUY", "SCALP_BUY") else "short"
                self._notify_position_opened(position_direction)
            else:
                self._record_submission_timestamp()
            self._trades_today = getattr(self, '_trades_today', 0) + 1
            
            # Broadcast order update
            await self._broadcast_order_update({
                'type': 'AWS_AGENT_ENTRY',
                'action': action,
                'quantity': quantity,
                'price': current_price,
                'order_id': order_id,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'agent_decision': decision,
            })
            
            # Send Telegram notification
            if self.telegram:
                await self.telegram.send_trade_alert(
                    action=action,
                    quantity=quantity,
                    price=current_price,
                    source='AWS_AGENTS',
                )
            
        except Exception as e:
            logger.error(f"❌ Failed to place AWS agent order: {e}")
            await self._broadcast_error(f"AWS Agent order failed: {e}")
    
    async def _place_hybrid_order(self, signal, pipeline_result, current_price: float, features):
        """Place an order using hybrid pipeline's risk parameters.

        Uses stop loss and take profit from the pipeline result,
        which incorporates LLM suggestions and ATR-based calculations.
        """
        try:
            logger.info(f"🤖 HYBRID: Placing {signal.action} order")

            # MAY 8 2026: Trading Manager veto hook for the HYBRID order path.
            # Mirrors the hook in signal_processor.py:1660 — required because the
            # hybrid path was previously bypassing the TM gate, allowing a 5/7
            # trade ($-42.92) to execute despite the TM's REJECT verdict.
            # Read the latest matching decision from logs/manager_decisions.jsonl
            # and abort placement on REJECT. Fail-open if the file is unavailable
            # or the daemon hasn't written a verdict yet (logged loudly so the
            # operator notices).
            try:
                from ..trading_manager.decision_log import (
                    latest_decision_for_signal_ts,
                    latest_decision,
                )
                tm_path = "logs/manager_decisions.jsonl"
                sig_meta_for_ts = signal.metadata if isinstance(
                    getattr(signal, "metadata", None), dict
                ) else {}
                sig_ts = str(
                    sig_meta_for_ts.get("ts")
                    or sig_meta_for_ts.get("timestamp")
                    or ""
                )
                tm_rec = None
                if sig_ts:
                    tm_rec = latest_decision_for_signal_ts(tm_path, sig_ts)
                if tm_rec is None:
                    # Fall back to most-recent decision if it's fresh (<60s old).
                    latest = latest_decision(tm_path)
                    if latest:
                        import time as _time
                        import datetime as _dt
                        try:
                            latest_ts = _dt.datetime.fromisoformat(
                                latest.get("ts", "")
                            ).timestamp()
                            if _time.time() - latest_ts <= 60:
                                tm_rec = latest
                        except Exception:
                            pass
                if tm_rec and tm_rec.get("decision") == "REJECT":
                    logger.warning(
                        "🛡️  TRADING MANAGER VETO (HYBRID): {} | reason: {}",
                        tm_rec.get("decision"),
                        str(tm_rec.get("reasoning", ""))[:200],
                    )
                    return
                if tm_rec and tm_rec.get("decision") == "MODIFY":
                    logger.warning(
                        "🛡️  TRADING MANAGER MODIFY (HYBRID, size-down TBD): {}",
                        str(tm_rec.get("reasoning", ""))[:200],
                    )
                if tm_rec is None:
                    logger.warning(
                        "⚠️  Trading Manager decision not found for HYBRID signal — "
                        "proceeding fail-open. Verify the TM daemon is running."
                    )
            except Exception as _tm_err:
                # Never let the veto hook block trading on its own bug.
                logger.error(
                    "Trading Manager veto hook error in _place_hybrid_order "
                    "(fail-open): {}", _tm_err,
                )

            # Strict scalp intent mapping (Option B): allow upstream signals to remain BUY/SELL
            # and opt-in to SCALP_* execution behavior through metadata.is_scalp.
            base_meta_for_mapping = signal.metadata if isinstance(getattr(signal, "metadata", None), dict) else {}
            if bool(base_meta_for_mapping.get("is_scalp")) and isinstance(getattr(signal, "action", None), str):
                action_upper = signal.action.upper()
                mapped_action = None
                if action_upper == "BUY":
                    mapped_action = "SCALP_BUY"
                elif action_upper == "SELL":
                    mapped_action = "SCALP_SELL"
                if mapped_action and mapped_action != signal.action:
                    logger.info(
                        "🎯 Scalp intent enabled via metadata.is_scalp: %s -> %s",
                        signal.action,
                        mapped_action,
                    )
                    # Preserve provenance for audit trails.
                    base_meta_for_mapping.setdefault("scalp_intent_source", "metadata.is_scalp")
                    base_meta_for_mapping.setdefault("original_action", signal.action)
                    signal.metadata = base_meta_for_mapping
                    signal.action = mapped_action
            
            # Position sizing
            risk_stats = self.risk.get_statistics()
            qty = self.risk.position_size(
                self.settings.trading.initial_capital,
                signal.confidence,
                win_rate=risk_stats.get("win_rate"),
                avg_win=risk_stats.get("avg_win"),
                avg_loss=risk_stats.get("avg_loss")
            )
            
            # Apply position size factor from hybrid pipeline
            if pipeline_result:
                position_factor = pipeline_result.position_size
                qty = max(1, int(round(qty * position_factor)))
            aws_size_multiplier = None
            if isinstance(signal.metadata, dict):
                aws_size_multiplier = signal.metadata.get("aws_size_multiplier")
            if aws_size_multiplier:
                qty = max(1, int(round(qty * max(0.1, float(aws_size_multiplier)))))
            
            qty = min(qty, self.settings.trading.max_position_size)
            
            if not self.risk.can_trade(qty):
                await self._broadcast_error("Risk limits exceeded")
                return
            
            # Get risk parameters from hybrid pipeline
            # Handle both normal and scalp signals
            is_buy = signal.action in ["BUY", "SCALP_BUY"]
            is_sell = signal.action in ["SELL", "SCALP_SELL"]
            direction = 1 if is_buy else -1
            is_scalp = signal.action in ["SCALP_BUY", "SCALP_SELL"]
            row = features.iloc[-1]
            atr = row.get("atr", row.get("ATR", 0.0))
            
            # Use pipeline's calculated stop/target
            fallback_used = False
            # Get minimum stop from risk gate config
            risk_gate_cfg = getattr(self.settings, "risk_gate", None)
            min_stop_pts = getattr(risk_gate_cfg, "min_stop_points", 4.0) if risk_gate_cfg else 4.0
            
            # Strategy-First: prefer strategy's own SL/TP (absolute prices in metadata)
            # over the hybrid pipeline's generic ATR-multiplied offsets.
            strategy_sl = None
            strategy_tp = None
            sig_meta = getattr(signal, "metadata", None) or {}
            if isinstance(sig_meta, dict):
                strategy_sl = sig_meta.get("stop_loss")
                strategy_tp = sig_meta.get("take_profit")

            if strategy_sl and strategy_tp and strategy_sl > 0 and strategy_tp > 0:
                # Convert absolute prices → offsets
                stop_offset = abs(current_price - strategy_sl)
                target_offset = abs(strategy_tp - current_price)
                logger.info(
                    f"🎯 Using STRATEGY risk params: SL={stop_offset:.2f} "
                    f"(abs={strategy_sl:.2f}), TP={target_offset:.2f} (abs={strategy_tp:.2f})"
                )
            elif pipeline_result and pipeline_result.stop_loss > 0:
                stop_offset = pipeline_result.stop_loss
                target_offset = pipeline_result.take_profit
                logger.info(f"🎯 Using PIPELINE risk params: SL={stop_offset:.2f}, TP={target_offset:.2f}")
            else:
                # FEB 20 2026: Pass live VIX value so stop widens on event days
                _live_vix = self._get_live_vix_price()
                offsets = compute_protective_offsets(
                    atr_value=atr,
                    tick_size=self.settings.trading.tick_size,
                    scalper=is_scalp,
                    volatility=self.status.hybrid_volatility_regime,
                    current_price=current_price,
                    vix_value=_live_vix,
                )
                stop_offset = offsets.stop_offset
                target_offset = offsets.target_offset
                fallback_used = offsets.fallback_used
                if offsets.fallback_used:
                    logger.warning(
                        f"⚠️ Using ATR fallback offsets ({offsets.reason or 'fallback'}) "
                        f"SL={stop_offset:.2f}, TP={target_offset:.2f}"
                    )
                else:
                    label = "SCALP ATR" if is_scalp else "ATR"
                    logger.info(f"🎯 Using {label} offsets: SL={stop_offset:.2f}, TP={target_offset:.2f}")
            
            # Guardrails: ensure stop meets minimum from risk gate config
            min_guard_ticks = getattr(self.settings.trading, "min_distance_ticks", 4)
            min_tick_offset = self.settings.trading.tick_size * max(1, int(min_guard_ticks))
            # Use the larger of tick-based minimum or risk gate minimum
            min_guard_offset = max(min_tick_offset, min_stop_pts)
            
            if stop_offset < min_guard_offset:
                logger.info(
                    f"🛡️ Stop offset {stop_offset:.2f} too tight (< min {min_guard_offset:.2f}); "
                    f"expanding to {min_guard_offset:.2f}"
                )
                stop_offset = min_guard_offset
                # Also ensure target maintains reasonable R:R
                if target_offset < stop_offset * 1.5:
                    target_offset = stop_offset * 1.5
                    logger.info(f"🛡️ Adjusted target to {target_offset:.2f} to maintain R:R")
            if target_offset < min_tick_offset:
                logger.info(
                    f"🛡️ Target offset {target_offset:.2f} too tight (<{min_guard_ticks} ticks); "
                    f"expanding to {min_guard_offset:.2f}"
                )
                target_offset = min_guard_offset

            # Guardrail: clamp stop to max_stop_points from risk gate config
            # Prevents RiskGate STOP_EXCEEDS_CAP rejections when pipeline's
            # ATR multiplier × current ATR marginally exceeds the hard cap.
            max_stop_pts = getattr(risk_gate_cfg, "max_stop_points", 25.0) if risk_gate_cfg else 25.0
            if stop_offset > max_stop_pts:
                logger.info(
                    f"🛡️ Stop offset {stop_offset:.2f} exceeds max {max_stop_pts:.2f}; "
                    f"clamping to {max_stop_pts:.2f}"
                )
                stop_offset = max_stop_pts
    
            entry_price = current_price
            stop_loss = entry_price - stop_offset if direction > 0 else entry_price + stop_offset
            take_profit = entry_price + target_offset if direction > 0 else entry_price - target_offset

            # Enhanced debug of protective offsets (explicit formatting to avoid placeholder bleed)
            atr_value = row.get("ATR_14", None)
            atr_str = f"{atr_value:.4f}" if atr_value is not None else "N/A"
            debug_msg = (
                "🔍 Stop-Loss Calculation Debug:\n"
                f"   Action: {signal.action}\n"
                f"   Current Price: {current_price:.2f}\n"
                f"   Entry Price: {entry_price:.2f}\n"
                f"   ATR Value: {atr_str}\n"
                f"   Stop Offset: {stop_offset:.2f}\n"
                f"   Target Offset: {target_offset:.2f}\n"
                f"   Stop-Loss: {stop_loss:.2f} ({'above' if direction < 0 else 'below'} entry)\n"
                f"   Take-Profit: {take_profit:.2f} ({'below' if direction < 0 else 'above'} entry)\n"
                f"   Fallback Used: {'yes' if fallback_used else 'no'} ({offsets.reason if fallback_used else 'ok'})"
            )
            logger.info(debug_msg)
            
            # Build metadata from hybrid pipeline
            base_metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
            metadata = self._prepare_order_metadata(base_metadata, current_price, "hybrid")
            metadata["hybrid_pipeline"] = True
            metadata["hybrid_reasoning"] = metadata.get("hybrid_reasoning", "")
            metadata["market_trend"] = self.status.hybrid_market_trend
            metadata["volatility_regime"] = self.status.hybrid_volatility_regime
            metadata["atr_fallback_used"] = fallback_used
            
            # Add score breakdown for Telegram decision reasoning (JAN 8 2026)
            if pipeline_result and hasattr(pipeline_result, "rule_engine"):
                rule_indicators = getattr(pipeline_result.rule_engine, "indicators", {})
                score_breakdown = rule_indicators.get("score_breakdown")
                if score_breakdown:
                    metadata["score_breakdown"] = score_breakdown
                # Also add session info for Telegram
                session = rule_indicators.get("current_session")
                if session:
                    metadata["session"] = session

            allowed, reason, signal_key = await self.order_coordinator.enforce_entry_gates(
                signal.action,
                metadata,
            )
            if not allowed:
                logger.info("Hybrid entry blocked by gate: {}", reason)
                self._add_reason_code(reason)
                return
            
            # Prepare market data for trade logging
            market_data = {
                "close": float(row.get("close", current_price)),
                "rsi": float(row.get("RSI_14", 50)),
                "macd_hist": float(row.get("MACD", 0)),
                "atr": float(row.get("ATR_14", 0)),
                "ema_9": float(row.get("EMA_9", current_price)),
                "ema_20": float(row.get("EMA_20", current_price)),
                "pdh": float(row.get("PDH", 0)),
                "pdl": float(row.get("PDL", 0)),
            }
            
            # Add scoring data if available
            if "score_breakdown" in metadata:
                market_data["score_breakdown"] = metadata["score_breakdown"]
            
            # Add confidence tracking data
            if hasattr(signal, "confidence"):
                confidence_data = {
                    "final": signal.confidence,
                    "original": signal.confidence,  # Will be overridden if we have modifiers
                }
                
                # Extract confidence modifiers from metadata
                if "sentiment_modifier" in metadata:
                    confidence_data["sentiment_mult"] = metadata["sentiment_modifier"]
                    # Back-calculate original confidence
                    sentiment_mult = metadata.get("sentiment_modifier", 1.0)
                    lowvol_mult = metadata.get("low_volume_penalty", 1.0)
                    total_mult = sentiment_mult * lowvol_mult
                    if total_mult != 0:
                        confidence_data["original"] = signal.confidence / total_mult
                
                if "low_volume_penalty" in metadata:
                    confidence_data["lowvol_mult"] = metadata["low_volume_penalty"]
                
                market_data["confidence_data"] = confidence_data
            
            # Add session data
            session_data = {}
            current_session = metadata.get("session", "")
            if current_session:
                session_data["session"] = current_session
                
                # Get session-specific thresholds from config
                if hasattr(self, "config") and self.config:
                    if current_session == "RTH":
                        session_data["threshold_full"] = self.config.get("scoring_full_size_threshold", 65.0)
                        session_data["threshold_half"] = self.config.get("scoring_half_size_threshold", 50.0)
                        session_data["sentiment_threshold"] = self.config.get("sentiment_block_threshold_rth", 0.4)
                    elif current_session in ["EVENING", "OVERNIGHT"]:
                        session_data["threshold_full"] = self.config.get("scoring_evening_full_threshold", 70.0)
                        session_data["threshold_half"] = self.config.get("scoring_evening_half_threshold", 55.0)
                        session_data["sentiment_threshold"] = self.config.get("sentiment_block_threshold_overnight", 0.25)
                
                market_data["session_data"] = session_data
            
            # Add sentiment data
            if "combined_sentiment" in metadata or "source_breakdown" in metadata:
                sentiment_data = {
                    "combined": metadata.get("combined_sentiment", 0.0),
                    "reason": metadata.get("sentiment_reason", ""),
                }
                
                # Extract individual source scores from breakdown
                source_breakdown = metadata.get("source_breakdown", {})
                if source_breakdown:
                    st_val = source_breakdown.get("stocktwits", 0.0)
                    rd_val = source_breakdown.get("reddit", 0.0)
                    sentiment_data["stocktwits"] = st_val.get("score", 0.0) if isinstance(st_val, dict) else float(st_val)
                    sentiment_data["reddit"] = rd_val.get("score", 0.0) if isinstance(rd_val, dict) else float(rd_val)
                
                # Determine sentiment decision
                if metadata.get("sentiment_blocked"):
                    sentiment_data["decision"] = "BLOCK"
                elif metadata.get("sentiment_modifier", 1.0) < 1.0:
                    sentiment_data["decision"] = "REDUCE_SIZE"
                else:
                    sentiment_data["decision"] = "PROCEED"
                
                market_data["sentiment_data"] = sentiment_data
            
            # Broadcast order intent
            await self._broadcast_order_update({
                "status": "placing",
                "action": signal.action,
                "quantity": qty,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "hybrid_pipeline": True,
                "market_trend": self.status.hybrid_market_trend,
            })
            
            logger.info(
                "📊 Order telemetry | qty={qty} position={position} lock={locked} entry={entry:.2f} SL={sl:.2f} TP={tp:.2f} fallback={fallback}",
                qty=qty,
                position=self.status.current_position,
                locked=self.executor.is_order_locked() if self.executor else False,
                entry=current_price,
                sl=stop_loss,
                tp=take_profit,
                fallback="yes" if fallback_used else "no",
            )

            allowed_gate, gate_levels = await self._enforce_risk_gate(
                signal.action,
                qty,
                current_price,
                float(row.get("ATR_14", 0.0)),
                stop_loss,
                take_profit,
            )
            if not allowed_gate:
                return
            if gate_levels:
                stop_loss = gate_levels.get("stop_loss", stop_loss)
                take_profit = gate_levels.get("take_profit", take_profit)
                metadata.update(gate_levels)

            # HARD GUARDRAIL: Always validate entry guard (not just when feature flag is on)
            if not self._validate_entry_guard(
                current_price, stop_loss, take_profit, qty, signal.action
            ):
                self._add_reason_code("RISK_BLOCKED_INVALID_PROTECTION")
                log_structured_event(
                    agent="live_manager",
                    event_type="risk.entry_blocked",
                    message="Entry blocked by hard guardrails",
                    payload={"trade_cycle_id": self._current_cycle_id},
                )
                return
            
            # APR 8 2026: Price-confirms-breakout guard for OR_BREAK signals.
            # On Apr 8 13:15, candle closed at 6829.25 (above OR_H=6824.25)
            # but the live market price was 6815.75 — 14 pts BELOW the
            # breakout level.  The breakout was a spike that already retraced.
            # Validate that the LIVE price confirms the breakout direction
            # before placing the order.
            signal_reason_for_or = metadata.get("reason", "")
            if "OR_BREAK" in signal_reason_for_or:
                try:
                    live_price = await self.executor.get_current_price()
                    if live_price and live_price > 0:
                        # Extract OR_H / OR_L from signal metadata
                        or_h = metadata.get("or_high")
                        or_l = metadata.get("or_low")
                        # Also try parsing from reason string: "OR_H=6824.25"
                        if not or_h:
                            import re as _re
                            _m = _re.search(r"OR_H=([\d.]+)", signal_reason_for_or)
                            if _m:
                                or_h = float(_m.group(1))
                        if not or_l:
                            import re as _re
                            _m = _re.search(r"OR_L=([\d.]+)", signal_reason_for_or)
                            if _m:
                                or_l = float(_m.group(1))

                        block_entry = False
                        if is_buy and or_h:
                            or_h = float(or_h)
                            if live_price < or_h:
                                logger.warning(
                                    f"🚫 OR_BREAK_PRICE_REJECT: BUY but live_price "
                                    f"{live_price:.2f} < OR_H {or_h:.2f} — "
                                    f"breakout not confirmed by market"
                                )
                                block_entry = True
                        elif is_sell and or_l:
                            or_l = float(or_l)
                            if live_price > or_l:
                                logger.warning(
                                    f"🚫 OR_BREAK_PRICE_REJECT: SELL but live_price "
                                    f"{live_price:.2f} > OR_L {or_l:.2f} — "
                                    f"breakdown not confirmed by market"
                                )
                                block_entry = True

                        if block_entry:
                            self._add_reason_code("OR_BREAK_PRICE_NOT_CONFIRMED")
                            return
                except Exception as _exc:
                    logger.debug(f"OR_BREAK price confirmation check skipped: {_exc}")

            # === Simulation mode check ===
            if self.simulation_mode:
                logger.warning(f"🔶 SIMULATION: Would place HYBRID {signal.action} order for {qty} contracts @ {current_price:.2f}")
                logger.warning(f"   SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
                self._record_submission_timestamp()
                self.order_coordinator.record_signal_key(signal_key)
                
                # 🔄 MTF GATE: Notify position opened (transition to IN_POSITION state) - simulation
                position_direction = "long" if is_buy else "short"
                self._notify_position_opened(position_direction)
                
                # Log simulated trade entry
                if self.hybrid_pipeline:
                    self.hybrid_pipeline.log_trade_entry(
                        action=signal.action,
                        entry_price=current_price,
                        quantity=qty,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        market_data=market_data,
                        pipeline_result=pipeline_result,
                    )
                
                await self._broadcast_order_update({
                    "status": "SIMULATED",
                    "action": signal.action,
                    "quantity": qty,
                    "fill_price": current_price,
                    "order_id": f"SIM-HYBRID-{now_cst().strftime('%H%M%S')}"
                })
                return
            
            # Place real order
            try:
                current_pos = await self.executor.get_current_position()
                if current_pos and getattr(current_pos, "quantity", 0) != 0:
                    logger.info("Position changed before HYBRID submit (qty={}); skipping entry", current_pos.quantity)
                    return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Position recheck failed; skipping HYBRID entry: {}", exc)
                return
            result = await self.executor.place_order(
                action=signal.action,
                quantity=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
                rationale=metadata,
                features=market_data,
                market_regime=self.status.hybrid_market_trend,
                entry_price=current_price,
            )
            
            # Broadcast result
            await self._broadcast_order_update({
                "status": result.status,
                "action": signal.action,
                "quantity": qty,
                "fill_price": result.fill_price,
                "filled_quantity": result.filled_quantity,
                "order_id": result.trade.order.orderId if result.trade else None,
                "hybrid_pipeline": True,
            })
            
            if result.status not in {"Cancelled", "Inactive"}:
                self.order_coordinator.record_signal_key(signal_key)
                if result.fill_price or result.filled_quantity:
                    self._record_last_trade_timestamp()
                    if self.hybrid_pipeline:
                        self.hybrid_pipeline.record_trade_for_cooldown()
                    logger.info("⏱️ HYBRID trade fill - cooldown activated")
                    
                    # 🔄 MTF GATE: Notify position opened (transition to IN_POSITION state)
                    position_direction = "long" if is_buy else "short"
                    self._notify_position_opened(position_direction)
                else:
                    self._record_submission_timestamp()
                    logger.info("⏱️ HYBRID submission recorded (no fill yet)")
                
                self.risk.register_trade()
                
                if result.fill_price:
                    self.tracker.record_trade(
                        action=signal.action,
                        price=result.fill_price,
                        quantity=qty
                    )
                    self._update_status_from_tracker()
                
                # Log trade entry through hybrid pipeline
                if self.hybrid_pipeline:
                    self.hybrid_pipeline.log_trade_entry(
                        action=signal.action,
                        entry_price=result.fill_price or current_price,
                        quantity=qty,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        market_data=market_data,
                        pipeline_result=pipeline_result,
                    )
                    logger.info("✅ Trade logged to Hybrid RAG system")
                if result.fill_price:
                    self._register_trade_entry(
                        cycle_id=self._current_cycle_id,
                        action=signal.action,
                        quantity=qty,
                        entry_price=result.fill_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata=metadata,
                    )

        except Exception as e:
            logger.error(f"❌ CRITICAL: _place_hybrid_order failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            await self._broadcast_error(f"Hybrid order placement failed: {e}")
    
    async def _place_order(self, signal, current_price: float, features):
        """Place an order based on signal."""
        return await self.order_coordinator.execute_trade_with_risk_checks(
            signal,
            current_price,
            features,
        )

    async def _get_account_state_snapshot(self) -> Optional[dict]:
        """Collect account funds and realized PnL; fail safe on errors."""
        if not self.executor:
            return None
        try:
            balances = await self.executor.get_account_liquidity()
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ Unable to read account summary: {}", exc)
            return None
        balances = balances or {}
        realized_pnl = 0.0
        try:
            if self.tracker:
                realized_pnl = float(getattr(self.tracker, "daily_pnl", 0.0))
        except Exception:
            realized_pnl = 0.0
        equity = (
            balances.get("net_liquidation")
            or balances.get("equity")
            or balances.get("available_funds")
            or balances.get("excess_liquidity")
        )
        if equity is not None:
            balances["account_equity"] = equity
        balances["realized_pnl_today"] = realized_pnl
        return balances

    def _get_daily_pnl_for_persist(self) -> float:
        """Return current daily realized P&L for bot_state persistence.

        Fix #6 MAR 16 2026: Used by save_bot_state() calls so the $250 daily
        loss cap survives bot restarts.
        """
        try:
            if self.tracker:
                return float(getattr(self.tracker, "daily_pnl", 0.0))
        except Exception:
            pass
        return 0.0

    def _validate_bracket_prices(self, action: str, entry_price: float, stop_loss: float, take_profit: float) -> bool:
        """Validate that bracket order prices are logically correct."""
        act = action.upper()
        if act in ("BUY", "SCALP_BUY"):
            if stop_loss >= entry_price:
                logger.error("❌ BUY order: Stop-loss {:.4f} must be below entry {:.4f}", stop_loss, entry_price)
                return False
            if take_profit <= entry_price:
                logger.error("❌ BUY order: Take-profit {:.4f} must be above entry {:.4f}", take_profit, entry_price)
                return False
        elif act in ("SELL", "SCALP_SELL"):
            if stop_loss <= entry_price:
                logger.error("❌ SELL order: Stop-loss {:.4f} must be above entry {:.4f}", stop_loss, entry_price)
                return False
            if take_profit >= entry_price:
                logger.error("❌ SELL order: Take-profit {:.4f} must be below entry {:.4f}", take_profit, entry_price)
                return False
        logger.debug("✅ Bracket prices validated: {} @ {:.4f}, SL={:.4f}, TP={:.4f}", action, entry_price, stop_loss, take_profit)
        return True

    def _validate_entry_guard(
        self,
        entry_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        quantity: int,
        action: str = "BUY",
    ) -> bool:
        """Ensure we have sane protective levels and risk.
        
        HARD GUARDRAILS (always enforced):
        - Stop & target must exist and be > 0
        - BUY: stop < entry < target
        - SELL: target < entry < stop
        - Minimum distance in ticks (configurable, default 4 ticks)
        - Dollar risk must not exceed max_loss_per_trade
        """
        # Quick bracket direction sanity check
        if not self._validate_bracket_prices(action, entry_price, stop_loss, take_profit):
            self._add_reason_code("INVALID_BRACKET_DIRECTION")
            log_structured_event(
                agent="live_manager",
                event_type="risk.invalid_levels",
                message="Bracket direction invalid",
                payload={
                    "trade_cycle_id": self._current_cycle_id,
                    "action": action,
                    "entry": entry_price,
                    "stop": stop_loss,
                    "target": take_profit,
                },
            )
            return False

        # Check 1: Must have both stop and target
        if stop_loss is None or take_profit is None:
            logger.warning("⚠️ Protective levels missing - rejecting trade")
            self._add_reason_code("MISSING_PROTECTION")
            log_structured_event(
                agent="live_manager",
                event_type="risk.invalid_levels",
                message="Missing stop/target",
                payload={"trade_cycle_id": self._current_cycle_id},
            )
            return False
        
        # Check 2: Validate values are finite
        for label, value in (("entry_price", entry_price), ("stop_loss", stop_loss), ("take_profit", take_profit)):
            if value is None or not math.isfinite(value):
                logger.warning(f"⚠️ Protective level {label} is non-finite - rejecting trade")
                self._add_reason_code("INVALID_PROTECTION")
                log_structured_event(
                    agent="live_manager",
                    event_type="risk.invalid_levels",
                    message=f"Non-finite {label}",
                    payload={
                        "trade_cycle_id": self._current_cycle_id,
                        label: value,
                    },
                )
                return False
        
        # Check 3: Must be positive
        if stop_loss <= 0 or take_profit <= 0:
            logger.warning("⚠️ Protective levels invalid (non-positive) - rejecting trade")
            self._add_reason_code("INVALID_PROTECTION")
            log_structured_event(
                agent="live_manager",
                event_type="risk.invalid_levels",
                message="Non-positive stop/target",
                payload={
                    "trade_cycle_id": self._current_cycle_id,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                },
            )
            return False
        
        # Check 4: Bracket orientation (BUY: stop < entry < target, SELL: target < entry < stop)
        is_buy = action.upper() in ("BUY", "SCALP_BUY")
        if is_buy:
            if stop_loss >= entry_price:
                logger.warning(f"⚠️ Invalid bracket: BUY stop_loss ({stop_loss}) >= entry ({entry_price})")
                self._add_reason_code("INVALID_BRACKET")
                return False
            if take_profit <= entry_price:
                logger.warning(f"⚠️ Invalid bracket: BUY take_profit ({take_profit}) <= entry ({entry_price})")
                self._add_reason_code("INVALID_BRACKET")
                return False
        else:  # SELL
            if stop_loss <= entry_price:
                logger.warning(f"⚠️ Invalid bracket: SELL stop_loss ({stop_loss}) <= entry ({entry_price})")
                self._add_reason_code("INVALID_BRACKET")
                return False
            if take_profit >= entry_price:
                logger.warning(f"⚠️ Invalid bracket: SELL take_profit ({take_profit}) >= entry ({entry_price})")
                self._add_reason_code("INVALID_BRACKET")
                return False
        
        # Check 5: Minimum distance in ticks (configurable)
        min_distance_ticks = max(1, self._min_stop_distance_ticks)
        min_distance = self._min_stop_distance
        sl_distance = abs(entry_price - stop_loss)
        tp_distance = abs(take_profit - entry_price)
        if sl_distance < min_distance:
            logger.warning(
                f"⚠️ Stop loss too close: {sl_distance:.2f} < {min_distance:.2f} "
                f"(minimum {min_distance_ticks} ticks)"
            )
            self._add_reason_code("INSUFFICIENT_DISTANCE")
            return False
        if tp_distance < min_distance:
            logger.warning(
                f"⚠️ Take profit too close: {tp_distance:.2f} < {min_distance:.2f} "
                f"(minimum {min_distance_ticks} ticks)"
            )
            self._add_reason_code("INSUFFICIENT_DISTANCE")
            return False

        # Check 7: Enforce minimum risk/reward ratio
        # FEB 18 2026: Fixed — field was missing from TradingConfig dataclass,
        # so getattr always returned 1.5 fallback, blocking all OR signals.
        # Now reads from config.yaml via TradingConfig.min_risk_reward_ratio (default=1.0)
        min_rr_ratio = getattr(self.settings.trading, "min_risk_reward_ratio", 1.0)
        _, reward_points, rr_ratio = compute_risk_reward(
            entry_price,
            stop_loss,
            take_profit,
            action,
        )
        if rr_ratio < min_rr_ratio - 1e-6:
            logger.warning(
                f"⚠️ Risk/Reward {rr_ratio:.2f} < {min_rr_ratio:.1f} requirement (reward={reward_points:.2f} pts)"
            )
            self._add_reason_code("POOR_RISK_REWARD")
            log_structured_event(
                agent="live_manager",
                event_type="risk.rr_reject",
                message=f"Risk/reward below minimum {min_rr_ratio:.1f}:1",
                payload={
                    "trade_cycle_id": self._current_cycle_id,
                    "risk_reward": rr_ratio,
                    "min_required": min_rr_ratio,
                },
            )
            return False

        # Check 8: Minimum viable take-profit in live mode
        tp_ok, min_tp_points = enforce_min_take_profit(
            entry_price,
            take_profit,
            self.contract_spec,
            self.trading_mode,
            action,
        )
        if not tp_ok:
            logger.warning(
                f"⚠️ Take profit {abs(take_profit - entry_price):.2f} pts < "
                f"{min_tp_points:.2f} pts live minimum for {self.contract_spec.root_symbol}"
            )
            self._add_reason_code("MIN_TP")
            log_structured_event(
                agent="live_manager",
                event_type="risk.min_tp_reject",
                message="Take profit below live minimum",
                payload={
                    "trade_cycle_id": self._current_cycle_id,
                    "min_points": min_tp_points,
                },
            )
            return False

        # Check 9: Net payoff after commissions must be positive
        expected = expected_target_outcome(
            entry_price,
            take_profit,
            quantity,
            self.contract_spec,
            self.trading_mode,
            self._commission_per_side,
        )
        if expected.net_pnl <= 0:
            logger.warning(
                f"⚠️ Trade blocked: net PnL ${expected.net_pnl:.2f} <= 0 "
                f"(gross ${expected.gross_pnl:.2f}, commission ${expected.commission:.2f})"
            )
            self._add_reason_code("NEGATIVE_NET")
            log_structured_event(
                agent="live_manager",
                event_type="risk.commission_reject",
                message="Projected net PnL is non-positive after commissions",
                payload={
                    "trade_cycle_id": self._current_cycle_id,
                    "gross": expected.gross_pnl,
                    "commission": expected.commission,
                },
            )
            return False
        
        # Check 6: Dollar risk must not exceed max_loss_per_trade
        dollar_risk = compute_trade_risk_dollars(
            entry_price,
            stop_loss,
            self.contract_spec.point_value,
        ) * max(1, quantity)
        if dollar_risk > self.settings.trading.max_loss_per_trade:
            logger.warning(
                f"⛔ Estimated risk ${dollar_risk:.2f} exceeds per-trade cap "
                f"${self.settings.trading.max_loss_per_trade:.2f}"
            )
            self._add_reason_code("MAX_LOSS_CAP")
            log_structured_event(
                agent="live_manager",
                event_type="risk.trade_rejected",
                message="Estimated loss exceeds cap",
                payload={
                    "trade_cycle_id": self._current_cycle_id,
                    "risk_dollars": dollar_risk,
                    "cap": self.settings.trading.max_loss_per_trade,
                },
            )
            return False
        
        return True

    async def _enforce_risk_gate(
        self,
        action: str,
        quantity: int,
        entry_price: float,
        atr: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> tuple[bool, dict]:
        """Run hard margin/risk gate; returns (allowed, levels)."""
        account_state = await self._get_account_state_snapshot()
        if not account_state:
            log_structured_event(
                agent="live_manager",
                event_type="risk.entry_blocked",
                message="Account state unavailable",
                payload={"trade_cycle_id": self._current_cycle_id},
            )
            return False, {}
        now_ts = now_cst()
        current_pos = getattr(self.status, "current_position", 0) or 0
        result = self.risk_gate.evaluate_entry(
            action=action,
            quantity=quantity,
            entry_price=entry_price,
            atr=atr,
            account_state=account_state,
            current_position=current_pos,
            now=now_ts,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        if not result.allowed:
            logger.warning(f"🚫 RiskGate blocked entry: {result.reason}")
            log_structured_event(
                agent="live_manager",
                event_type="risk.entry_blocked",
                message=result.reason,
                payload={
                    "trade_cycle_id": self._current_cycle_id,
                    "account_state": account_state,
                    "levels": result.levels,
                },
            )
        else:
            logger.info(
                "✅ RiskGate pass: margin={avail:.2f} required={req:.2f} stopPts={stop_pts:.2f}",
                avail=account_state.get("available_funds") or account_state.get("excess_liquidity", 0.0),
                req=result.levels.get("required_margin", 0.0),
                stop_pts=result.levels.get("stop_points", 0.0),
            )
        return result.allowed, result.levels
    
    def _prepare_order_metadata(
        self,
        base_metadata: Optional[Dict[str, Any]],
        entry_price: float,
        strategy_name: str,
    ) -> Dict[str, Any]:
        """Attach trade-cycle context so downstream guardrails have consistent data."""
        return self.order_coordinator.prepare_order_metadata(base_metadata, entry_price, strategy_name)

    def _bucket_entry_price(self, price: float) -> float:
        """Quantize entry intent price to reduce idempotency noise."""
        return self.order_coordinator.bucket_entry_price(price)

    def _detect_trading_mode(self, settings: Settings) -> TradingMode:
        """Infer trading mode from CLI arguments and IBKR connectivity."""
        if self.simulation_mode:
            return "paper"
        ib_port = getattr(settings.data, "ibkr_port", 4001)
        if ib_port in (4001, 7496):
            return "live"
        if ib_port in (4002, 7497):
            return "paper"
        # Default to live on unknown ports to avoid under-estimating commissions
        return "live"

    def _record_last_trade_timestamp(self, timestamp: Optional[datetime] = None) -> None:
        """Persist the last trade time so cooldown survives restarts."""
        ts = timestamp or datetime.now(timezone.utc)
        self.cooldown_manager.record_last_trade_timestamp(ts)
        self._track_trade_timestamp(ts)

    def _track_trade_timestamp(self, ts: datetime) -> None:
        """Track trade timestamps for hourly/daily rate limits."""
        try:
            now_utc = datetime.now(timezone.utc)
            day_cutoff = now_utc - timedelta(hours=24)
            self._trade_timestamps = [t for t in self._trade_timestamps if t >= day_cutoff]
            self._trade_timestamps.append(ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))
        except Exception:
            return

    def _record_submission_timestamp(self) -> None:
        """Track last submission to prevent rapid resubmits before fill."""
        with self._trade_time_lock:
            self._last_submission_time = datetime.now(timezone.utc)

    def _load_persistent_cooldown_state(self) -> None:
        """Restore last trade timestamp from order tracker at startup."""
        self.cooldown_manager.load_persistent_cooldown_state()

    def _validate_persisted_trade_time(self, timestamp: Optional[datetime]) -> Optional[datetime]:
        """Reject persisted cooldown timestamps that are implausible."""
        return self.cooldown_manager.validate_persisted_trade_time(timestamp)

    def _apply_manual_state_reset(self) -> None:
        """Clear cooldown + lock state based on operator override."""
        self.cooldown_manager.apply_manual_state_reset()

    def _register_trade_entry(
        self,
        cycle_id: Optional[str],
        action: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store trade context for learning + telemetry.
        
        CRITICAL: Stores entry trade_cycle_id so exit orders can correlate back to entry.
        """
        self.order_coordinator.register_trade_entry(
            cycle_id,
            action,
            quantity,
            entry_price,
            stop_loss,
            take_profit,
            metadata,
        )

    async def _finalize_trade(
        self,
        exit_price: float,
        exit_time: datetime,
        realized_pnl: float,
    ) -> None:
        """Persist trade outcome + history snapshots."""
        await self.order_coordinator.finalize_trade(exit_price, exit_time, realized_pnl)

    def _add_reason_code(self, code: str) -> None:
        self.order_coordinator.add_reason_code(code)
    
    async def stop(self):
        """Stop the trading session."""
        await self.trading_session_manager.stop()
    
    async def force_order_reconciliation(self):
        """Force reconciliation of active orders with IBKR."""
        await self.trading_session_manager.force_order_reconciliation()
    
    # Broadcasting methods
    async def _broadcast_status(self):
        """Broadcast status update."""
        await self.status_broadcaster.broadcast_status()
    
    async def _broadcast_signal(self, signal, price: float):
        """Broadcast signal generated."""
        await self.status_broadcaster.broadcast_signal(signal, price)
    
    async def _broadcast_order_update(self, order_data: Dict):
        """Broadcast order update."""
        await self.status_broadcaster.broadcast_order_update(order_data)
    
    async def _broadcast_error(self, error_msg: str):
        """Broadcast error."""
        await self.status_broadcaster.broadcast_error(error_msg)
    
    def get_performance_snapshot(self) -> Dict:
        """Get current performance snapshot."""
        if not self.tracker:
            return {}
        
        snapshot = self.tracker.get_snapshot()
        return {
            "total_pnl": snapshot.total_pnl,
            "total_return": (snapshot.equity / self.tracker.initial_capital - 1) * 100,
            "sharpe_ratio": snapshot.sharpe_ratio,
            "max_drawdown": snapshot.max_drawdown,
            "win_rate": snapshot.win_rate,
            "total_trades": snapshot.trade_count,
            "winning_trades": snapshot.winning_trades,
            "losing_trades": snapshot.losing_trades,
        }

    def save_snapshot(self, row, buckets: Dict):
        """Save market snapshot to RAG storage."""
        if not self.rag_storage:
            return
            
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Construct data package for RAGStorage
            data = {
                "ohlcv": {
                    "open": float(row.get("open", 0.0)),
                    "high": float(row.get("high", 0.0)),
                    "low": float(row.get("low", 0.0)),
                    "close": float(row["close"]),
                    "volume": int(row.get("volume", 0))
                },
                "vwap": float(row.get("vwap", 0.0)), # Assuming vwap is in row if available
                "volatility": float(row.get("volatility_5m", 0.0)),
                "indicators": {
                    "atr_14": float(row.get("ATR_14", 0.0)),
                    "rsi_14": float(row.get("RSI_14", 0.0)),
                    "macd": float(row.get("MACD", 0.0)),
                    "bb_upper": float(row.get("BB_upper", 0.0)),
                    "bb_lower": float(row.get("BB_lower", 0.0)),
                    "sma_20": float(row.get("SMA_20", 0.0)),
                    "sma_50": float(row.get("SMA_50", 0.0)),
                    "buckets": buckets # Store buckets in indicators for now if useful
                }
            }
            
            self.rag_storage.save_snapshot(timestamp, data)
                
        except Exception as e:
            logger.error(f"Failed to save market snapshot: {e}")
