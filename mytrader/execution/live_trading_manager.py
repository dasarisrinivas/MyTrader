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
from ..strategies.momentum_reversal import MomentumReversalStrategy
from ..strategies.momentum_reversal import MomentumReversalStrategy
from ..strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
from ..strategies.market_regime import detect_market_regime, get_regime_parameters, MarketRegime
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


class LiveTradingManager:
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
    """Manages live trading session with WebSocket broadcasting.
    
    ENHANCED with:
    - Trade cooldown period (configurable, default 5 minutes)
    - Candle close validation (wait for candle to close before entry)
    - Higher-timeframe levels (PDH/PDL, WH/WL, PWH/PWL)
    - Trend confirmation filters
    - Simulation mode for testing without real orders
    """
    
    # === CONFIGURATION CONSTANTS ===
    DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes between trades
    MIN_CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence to place trade
    POLL_INTERVAL_SECONDS = 5  # How often to poll price (when NOT waiting for candle)
    CANDLE_PERIOD_SECONDS = 60  # 1-minute candles - only evaluate at candle close
    MIN_COOLDOWN_MINUTES = 1
    MAX_COOLDOWN_MINUTES = 60
    COOLDOWN_WARNING_MINUTES = 30
    PERSISTED_COOLDOWN_MAX_AGE = timedelta(days=7)
    FUTURE_COOLDOWN_TOLERANCE_SECONDS = 120
    
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
        self.price_history: List[Dict] = []
        self.running = False
        self.stop_requested = False
        self._trade_time_lock = threading.Lock()
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
        self.current_trade_features: Optional[Dict] = None
        self._current_entry_cycle_id: Optional[str] = None  # Entry cycle ID for exit correlation
        self.current_trade_buckets: Optional[Dict] = None
        self.current_trade_rationale: Optional[Dict] = None
        self._open_trade_context: Optional[Dict[str, Any]] = None
        self._current_cycle_id: Optional[str] = None
        self._cycle_context: Dict[str, Dict[str, Any]] = {}
        self._active_reason_codes: Set[str] = set()
        
        # NEW: Cooldown tracking
        self._last_trade_time: Optional[datetime] = None
        raw_cooldown_minutes = getattr(
            settings.trading,
            'trade_cooldown_minutes',
            self.DEFAULT_COOLDOWN_SECONDS // 60,
        )
        sanitized_minutes = self.cooldown_manager.sanitize_cooldown_minutes(raw_cooldown_minutes)
        self.settings.trading.trade_cooldown_minutes = sanitized_minutes
        self._cooldown_seconds = sanitized_minutes * 60  # Convert minutes to seconds
        
        # NEW: Candle tracking for proper candle-close validation
        self._last_candle_processed: Optional[datetime] = None
        wait_for_close = True
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
        """Handle execution details to track trade exits for RAG/learning."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            loop.create_task(self.order_coordinator.handle_order_fill(trade, fill))
        else:
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
                    
                    logger.info(f"✅ Historical context loaded:")
                    logger.info(f"   📈 Previous Day: High={pdh:.2f}, Low={pdl:.2f}, Close={prev_close:.2f}")
                    if self._historical_context.get('today'):
                        th = self._historical_context['today'].get('high', 0)
                        tl = self._historical_context['today'].get('low', 0)
                        logger.info(f"   📊 Today: High={th:.2f}, Low={tl:.2f}")
                    logger.info(f"   📅 Weekly Range: {self._historical_context['weekly']['low']:.2f} - {self._historical_context['weekly']['high']:.2f}")
                    
                    # Store in RAG for agents to query
                    await self._store_historical_context_in_rag()
                    
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
        duration_seconds = max(min_bars * self.CANDLE_PERIOD_SECONDS, 900)
        try:
            bars = await self.executor.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr=f"{duration_seconds} S",
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=2,
            )
        except AttributeError:
            bars = self.executor.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=f"{duration_seconds} S",
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=2,
            )
        except Exception as exc:
            logger.warning(f"Failed to bootstrap minute bars: {exc}")
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
            self.price_history = history[-500:]
            self.status.bars_collected = len(self.price_history)
            logger.info(
                f"📚 Bootstrapped {len(self.price_history)} historical 1-min bars for structural context"
            )

    async def start(self):
        """Start the live trading loop."""
        return await self.trading_session_manager.start()
    
    async def _process_trading_cycle(self, current_price: float):
        """Check position state first, then handle exits or entries."""
        position = None
        if self.executor:
            try:
                position = await self.executor.get_current_position()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Unable to fetch position for trading cycle: {exc}")
        qty = getattr(position, "quantity", 0) if position else 0

        if qty:
            logger.info(f"📊 Current position detected: {qty} contracts")
            exit_handled = await self._check_position_exit_signals(current_price)
            if exit_handled:
                return
            logger.debug("📊 Holding position (%s); skipping new entries", qty)
            return

        # Entry signals only - flat position
        return await self.signal_processor.process_trading_cycle(current_price)

    async def _check_position_exit_signals(self, current_price: Optional[float]) -> bool:
        """Check if existing position should be closed."""
        if not self.executor:
            return False
        
        position = await self.executor.get_current_position()
        if not position or position.quantity == 0:
            return False
        
        price = current_price
        if price is None:
            try:
                price = await self.executor.get_current_price()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"⚠️ Could not fetch price for exit check: {exc}")
                return False
        if price is None:
            logger.debug("No price available for exit checks; skipping exit evaluation")
            return False
        
        is_short = position.quantity < 0
        exit_signal = (
            self._generate_exit_signal_for_short(price, position)
            if is_short
            else self._generate_exit_signal_for_long(price, position)
        )
        if not exit_signal:
            exit_signal = await self._check_position_exit_logic(position)
        if exit_signal:
            active_orders = self.executor.get_active_order_count(sync=True)
            if active_orders > 0:
                logger.info(
                    "Exit signal detected but {active} active orders are still open; skipping duplicate exit",
                    active=active_orders,
                )
                return True
            qty = abs(position.quantity)
            direction = "SHORT" if is_short else "LONG"
            logger.info(f"🔄 Exit signal for {direction} position: {exit_signal}")
            # If custom exit signal includes an action, honor it
            if isinstance(exit_signal, dict) and "action" in exit_signal:
                await self._execute_position_exit(exit_signal, position, price)
            else:
                await self._place_exit_order("BUY" if is_short else "SELL", qty, price)
            return True
        
        return False

    async def _check_position_exit_logic(self, position) -> Optional[Dict[str, Any]]:
        """Check if existing position should be exited based on simple P&L bands."""
        if not self.executor:
            return None
        try:
            current_price = await self.executor.get_current_price()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"⚠️ Could not fetch price for position exit logic: {exc}")
            return None
        if not current_price:
            return None

        qty = int(getattr(position, "quantity", 0) or 0)
        if qty == 0:
            return None

        contracts = abs(qty)
        entry_price_raw = getattr(position, "avg_cost", 0.0) or 0.0
        entry_price = self._normalize_entry_price(entry_price_raw, current_price)
        multiplier = getattr(self.contract_spec, "point_value", 5) or 5

        pnl_per_contract = (current_price - entry_price) * multiplier
        # For shorts, invert PnL sign
        if qty < 0:
            pnl_per_contract = (entry_price - current_price) * multiplier
        total_pnl = pnl_per_contract * contracts

        logger.info(
            f"📊 Position P&L check -> entry={entry_price:.2f} price={current_price:.2f} "
            f"pnl/ct={pnl_per_contract:.2f} total={total_pnl:.2f}"
        )

        # Take profit
        if total_pnl >= 200:
            action = "SELL" if qty > 0 else "BUY"
            return {"reason": "PROFIT_TARGET", "action": action, "quantity": contracts, "pnl": total_pnl}
        # Stop loss
        if total_pnl <= -100:
            action = "SELL" if qty > 0 else "BUY"
            return {"reason": "STOP_LOSS", "action": action, "quantity": contracts, "pnl": total_pnl}

        return None

    async def _execute_position_exit(self, exit_signal: Dict[str, Any], position, current_price: Optional[float] = None) -> bool:
        """Execute position exit order honoring exit signal payload."""
        if not self.executor:
            return False
        action = exit_signal.get("action")
        quantity = int(exit_signal.get("quantity", 0))
        reason = exit_signal.get("reason", "EXIT")
        pnl = exit_signal.get("pnl", 0.0)
        if quantity <= 0 or action not in {"BUY", "SELL"}:
            logger.warning("Invalid exit signal payload: %s", exit_signal)
            return False
        price = current_price
        if price is None:
            try:
                price = await self.executor.get_current_price()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"⚠️ Could not fetch price for exit execution: {exc}")
                return False
        logger.info("🔄 Executing position exit: %s %s (reason=%s, pnl=%.2f)", action, quantity, reason, pnl)
        try:
            await self.executor.place_order(
                action=action,
                quantity=quantity,
                limit_price=price,
                stop_loss=None,
                take_profit=None,
                reduce_only=True,
                entry_price=price,
                metadata={"exit_reason": reason, "pnl": pnl, "position_exit": True},
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ Error executing position exit: {exc}")
            return False

    def monitor_current_position(self) -> bool:
        """Monitor the existing 2-contract LONG position and trigger forced exit on large loss."""
        current_position = 2  # From logs
        current_price = 6979.75  # From logs
        entry_price = 34896.87 / (2 * 5)  # 3489.687 per contract

        pnl_per_contract = (current_price - entry_price) * 5
        total_pnl = pnl_per_contract * current_position

        logger.info("🔍 Current Position Monitor:")
        logger.info("   Position: +%s LONG", current_position)
        logger.info("   Entry: %.2f", entry_price)
        logger.info("   Current: %.2f", current_price)
        logger.info("   Total P&L: $%.2f", total_pnl)

        if total_pnl <= -150:
            logger.warning("🚨 Significant loss detected, forcing position exit")
            return True

        return False

    def _normalize_entry_price(self, entry_price: float, current_price: float) -> float:
        """Normalize notional entry costs (e.g., futures multiplier embedded)."""
        if current_price <= 0:
            return entry_price
        ratio = entry_price / current_price
        configured_trading = getattr(getattr(self, "settings", None), "trading", None)
        custom_multipliers = getattr(configured_trading, "notional_multipliers", None) if configured_trading else None
        multipliers = tuple(custom_multipliers) if custom_multipliers else (5, 10, 20, 25, 50, 100, 200)
        for multiplier in multipliers:
            if abs(ratio - multiplier) <= 0.25:
                return entry_price / multiplier
        return entry_price

    def _get_max_exit_price_gap(self, current_price: float) -> float:
        """Return configurable/relative gap threshold for exit sanity checks."""
        configured_trading = getattr(getattr(self, "settings", None), "trading", None)
        custom_gap = getattr(configured_trading, "max_exit_price_gap", None) if configured_trading else None
        if custom_gap is not None:
            return float(custom_gap)
        return max(1000.0, abs(current_price) * 0.10)

    def _generate_exit_signal_for_short(self, current_price: float, position) -> Optional[Dict[str, float]]:
        """Generate exit signals for short positions."""
        entry_price = float(getattr(position, "avg_cost", current_price) or current_price)
        normalized_entry = self._normalize_entry_price(entry_price, current_price)
        if normalized_entry != entry_price:
            logger.debug(
                "Normalized entry price from notional value: raw={raw} normalized={normalized} current={current}",
                raw=entry_price,
                normalized=normalized_entry,
                current=current_price,
            )
        entry_price = normalized_entry
        if abs(entry_price - current_price) > self._get_max_exit_price_gap(current_price):
            logger.warning(
                "Skipping exit checks: entry/current price gap too large (entry={entry}, current={current})",
                entry=entry_price,
                current=current_price,
            )
            return None
        
        profit_points = entry_price - current_price
        if profit_points >= 20:
            return {"reason": "PROFIT_TARGET", "profit_points": profit_points}
        
        loss_points = current_price - entry_price
        if loss_points >= 10:
            return {"reason": "STOP_LOSS", "loss_points": loss_points}
        
        age_hours = self._get_position_age_hours(position)
        if age_hours is not None and age_hours >= 4:
            return {"reason": "TIME_EXIT", "hours_held": age_hours}
        
        return None

    def _generate_exit_signal_for_long(self, current_price: float, position) -> Optional[Dict[str, float]]:
        """Generate exit signals for long positions."""
        entry_price = float(getattr(position, "avg_cost", current_price) or current_price)
        normalized_entry = self._normalize_entry_price(entry_price, current_price)
        if normalized_entry != entry_price:
            logger.debug(
                "Normalized entry price from notional value: raw={raw} normalized={normalized} current={current}",
                raw=entry_price,
                normalized=normalized_entry,
                current=current_price,
            )
        entry_price = normalized_entry
        if abs(entry_price - current_price) > self._get_max_exit_price_gap(current_price):
            logger.warning(
                "Skipping exit checks: entry/current price gap too large (entry={entry}, current={current})",
                entry=entry_price,
                current=current_price,
            )
            return None
        
        profit_points = current_price - entry_price
        if profit_points >= 20:
            return {"reason": "PROFIT_TARGET", "profit_points": profit_points}
        
        loss_points = entry_price - current_price
        if loss_points >= 10:
            return {"reason": "STOP_LOSS", "loss_points": loss_points}
        
        age_hours = self._get_position_age_hours(position)
        if age_hours is not None and age_hours >= 4:
            return {"reason": "TIME_EXIT", "hours_held": age_hours}
        
        return None

    def _get_position_age_hours(self, position) -> Optional[float]:
        """Return position age (hours) if timestamp is available."""
        ts = getattr(position, "timestamp", None)
        if not ts:
            return None
        try:
            now = datetime.utcnow() if ts.tzinfo is None else datetime.now(tz=ts.tzinfo)
            return max(0.0, (now - ts).total_seconds() / 3600.0)
        except Exception:
            return None
    
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
        
        # Broadcast signal
        await self._broadcast_signal(signal, current_price)
        
        # Get current position
        current_position = await self.executor.get_current_position()
        self.status.current_position = current_position.quantity if current_position else 0
        self.status.active_orders = self.executor.get_active_order_count()
        
        if current_position:
            self.status.unrealized_pnl = await self.executor.get_unrealized_pnl()
            self.tracker.update_equity(current_price, realized_pnl=0.0)
            
            # Update trailing stops
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
        
        # If AWS agents rejected the trade, convert to HOLD
        if not aws_approved:
            logger.info(f"  🛑 Trade blocked by AWS Risk Agent - converting to HOLD")
            return

        # CRITICAL: Check for active orders FIRST
        active_orders = self.executor.get_active_order_count(sync=True)
        if active_orders > 0:
            logger.info(f"  ↳ {active_orders} active orders pending, waiting for completion")
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
                    self.hybrid_pipeline.log_trade_exit(
                        exit_price=current_price,
                        exit_reason="SIGNAL_EXIT",
                    )
                return
            else:
                logger.info(f"  ↳ Position open (qty={current_position.quantity}), same direction, no action")
                return
        
        # Place order using hybrid pipeline's risk parameters
        logger.info(f"  ↳ Placing HYBRID order: {signal.action}")
        await self._place_hybrid_order(signal, pipeline_result, current_price, features)
    
    async def _place_exit_order(self, action: str, quantity: int, exit_price: Optional[float] = None):
        """Place a market order to exit existing position.
        
        SAFETY: Always uses reduce_only=True to prevent accidental position opening.
        If exit order would open a new position, it is blocked.
        """
        price = exit_price
        if price is None:
            try:
                price = await self.executor.get_current_price()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"⚠️ Exit order: could not fetch price, using 0. Error: {exc}")
                price = 0.0
        logger.info(f"🚪 Placing EXIT order request: {action} {quantity} contracts @ ~{float(price):.2f}")
        current_position = await self.executor.get_current_position() if self.executor else None
        if not current_position or current_position.quantity == 0:
            logger.warning("⚠️ Exit requested but no open position - skipping")
            return
        
        # Determine correct exit action based on current position (not signal action)
        exit_action = "SELL" if current_position.quantity > 0 else "BUY"
        exit_qty = min(quantity, abs(current_position.quantity))
        
        # SAFETY CHECK: If exit action would open opposite position, block it
        if (current_position.quantity > 0 and exit_action == "BUY") or \
           (current_position.quantity < 0 and exit_action == "SELL"):
            logger.error(
                f"🚨 EXIT ORDER BLOCKED: Would open position. "
                f"Current: {current_position.quantity}, Exit action: {exit_action}"
            )
            self._add_reason_code("EXIT_WOULD_OPEN_POSITION")
            log_structured_event(
                agent="live_manager",
                event_type="risk.exit_blocked",
                message="Exit order would open new position",
                payload={
                    "trade_cycle_id": self._current_cycle_id,
                    "current_position": current_position.quantity,
                    "exit_action": exit_action,
                },
            )
            return
        
        logger.info(f"🚪 Placing EXIT order: {exit_action} {exit_qty} contracts (reduce_only=True)")

        # Get entry trade_cycle_id from position metadata if available
        entry_cycle_id = getattr(current_position, 'entry_trade_cycle_id', None) or \
                        getattr(self, '_current_entry_cycle_id', None) or \
                        self._current_cycle_id

        try:
            # Place market order to flatten position - ALWAYS reduce_only=True
            order_id = await self.executor.place_order(
                action=exit_action,  # BUY to close SHORT, SELL to close LONG
                quantity=exit_qty,
                limit_price=price,
                stop_loss=None,  # No stops for exit orders
                take_profit=None,
                reduce_only=True,  # HARD REQUIREMENT: Always reduce_only for exits
                entry_price=price,
                metadata={
                    "trade_cycle_id": entry_cycle_id,  # Use entry cycle ID, not current
                    "exit_order": True,
                    "allow_duplicate_exit": True,
                    "original_position": current_position.quantity,
                },
            )
            
            logger.info(f"✅ Exit order placed: ID={order_id}")
            
            # Broadcast exit
            await self._broadcast_order_update({
                "type": "EXIT",
                "action": exit_action,
                "quantity": exit_qty,
                "price": price,
                "order_id": order_id
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to place exit order: {e}")
            await self._broadcast_error(f"Exit order failed: {e}")
    
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
        decision = self.aws_agent_invoker.get_trading_decision(
            market_snapshot=snapshot,
            account_metrics=account_metrics,
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
                logger.info("AWS agent entry blocked by gate: %s", reason)
                self._add_reason_code(reason)
                return

            if self.simulation_mode:
                # Simulation mode - just log, don't place real order
                logger.info(f"🔶 [SIMULATION] Would place: {action} {quantity} @ {current_price:.2f}")
                self._record_last_trade_timestamp()
                return
            
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
                    logger.info("Position changed before AWS submit (qty=%s); skipping entry", current_pos.quantity)
                    return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Position recheck failed; skipping AWS entry: %s", exc)
                return
            order_id = await self.executor.place_order(
                action=action,
                quantity=quantity,
                limit_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_price=current_price,
                metadata=metadata,
            )
            
            logger.info(f"✅ AWS Agent order placed: ID={order_id}")
            
            # Track trade for RAG updates
            self.current_trade_id = str(uuid.uuid4())
            self.current_trade_entry_time = now_cst().isoformat()
            self.current_trade_entry_price = current_price
            self.current_trade_features = decision.get('decision_details', {}) if decision else {}
            self.current_trade_rationale = {
                'source': 'aws_agents',
                'decision': decision,
            }
            
            # Update counters
            self._record_last_trade_timestamp()
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
            
            # Use pipeline's calculated stop/target
            fallback_used = False
            if pipeline_result and pipeline_result.stop_loss > 0:
                stop_offset = pipeline_result.stop_loss
                target_offset = pipeline_result.take_profit
                # Tighter stops for scalps
                if is_scalp:
                    stop_offset = stop_offset * 0.6  # 60% of normal stop
                    target_offset = target_offset * 0.5  # 50% of normal target
                    logger.info(f"🎯 Using SCALP risk params: SL={stop_offset:.2f}, TP={target_offset:.2f}")
                else:
                    logger.info(f"🎯 Using HYBRID risk params: SL={stop_offset:.2f}, TP={target_offset:.2f}")
            else:
                atr = float(row.get("ATR_14", 0.0))
                offsets = compute_protective_offsets(
                    atr_value=atr,
                    tick_size=self.settings.trading.tick_size,
                    scalper=is_scalp,
                    volatility=self.status.hybrid_volatility_regime,
                    current_price=current_price,
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
            
            # Guardrails require a minimum of four ticks of distance, so normalize offsets here
            min_guard_ticks = getattr(self.settings.trading, "min_distance_ticks", 4)
            min_guard_offset = self.settings.trading.tick_size * max(1, int(min_guard_ticks))
            if stop_offset < min_guard_offset:
                logger.info(
                    f"🛡️ Stop offset {stop_offset:.2f} too tight (<{min_guard_ticks} ticks); "
                    f"expanding to {min_guard_offset:.2f}"
                )
                stop_offset = min_guard_offset
            if target_offset < min_guard_offset:
                logger.info(
                    f"🛡️ Target offset {target_offset:.2f} too tight (<{min_guard_ticks} ticks); "
                    f"expanding to {min_guard_offset:.2f}"
                )
                target_offset = min_guard_offset
    
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

            allowed, reason, signal_key = await self.order_coordinator.enforce_entry_gates(
                signal.action,
                metadata,
            )
            if not allowed:
                logger.info("Hybrid entry blocked by gate: %s", reason)
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
            
            # === Simulation mode check ===
            if self.simulation_mode:
                logger.warning(f"🔶 SIMULATION: Would place HYBRID {signal.action} order for {qty} contracts @ {current_price:.2f}")
                logger.warning(f"   SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
                self._record_last_trade_timestamp()
                self.order_coordinator.record_signal_key(signal_key)
                
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
                    logger.info("Position changed before HYBRID submit (qty=%s); skipping entry", current_pos.quantity)
                    return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Position recheck failed; skipping HYBRID entry: %s", exc)
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
                # Update cooldown
                self._record_last_trade_timestamp()
                if self.hybrid_pipeline:
                    self.hybrid_pipeline.record_trade_for_cooldown()
                logger.info(f"⏱️ HYBRID trade placed - cooldown activated")
                
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

    def _validate_bracket_prices(self, action: str, entry_price: float, stop_loss: float, take_profit: float) -> bool:
        """Validate that bracket order prices are logically correct."""
        act = action.upper()
        if act in ("BUY", "SCALP_BUY"):
            if stop_loss >= entry_price:
                logger.error("❌ BUY order: Stop-loss %.4f must be below entry %.4f", stop_loss, entry_price)
                return False
            if take_profit <= entry_price:
                logger.error("❌ BUY order: Take-profit %.4f must be above entry %.4f", take_profit, entry_price)
                return False
        elif act in ("SELL", "SCALP_SELL"):
            if stop_loss <= entry_price:
                logger.error("❌ SELL order: Stop-loss %.4f must be above entry %.4f", stop_loss, entry_price)
                return False
            if take_profit >= entry_price:
                logger.error("❌ SELL order: Take-profit %.4f must be below entry %.4f", take_profit, entry_price)
                return False
        logger.debug("✅ Bracket prices validated: %s @ %.4f, SL=%.4f, TP=%.4f", action, entry_price, stop_loss, take_profit)
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
        _, reward_points, rr_ratio = compute_risk_reward(
            entry_price,
            stop_loss,
            take_profit,
            action,
        )
        if rr_ratio < 1.0 - 1e-6:
            logger.warning(
                f"⚠️ Risk/Reward {rr_ratio:.2f} < 1.0 requirement (reward={reward_points:.2f} pts)"
            )
            self._add_reason_code("POOR_RISK_REWARD")
            log_structured_event(
                agent="live_manager",
                event_type="risk.rr_reject",
                message="Risk/reward below minimum 1:1",
                payload={
                    "trade_cycle_id": self._current_cycle_id,
                    "risk_reward": rr_ratio,
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
        ib_port = getattr(settings.data, "ibkr_port", 4002)
        if ib_port in (4001, 7496):
            return "live"
        if ib_port in (4002, 7497):
            return "paper"
        # Default to live on unknown ports to avoid under-estimating commissions
        return "live"

    def _record_last_trade_timestamp(self, timestamp: Optional[datetime] = None) -> None:
        """Persist the last trade time so cooldown survives restarts."""
        self.cooldown_manager.record_last_trade_timestamp(timestamp)

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
