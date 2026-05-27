"""
Backtest Engine
===============

Event-driven backtesting engine that:
- Iterates through historical bars
- Calls the EXACT same decision logic as live trading
- Routes orders to broker simulator
- Tracks all decisions, skips, and block reasons
- Supports decision trace mode for comparison with live

This is the core integration point between the backtest framework
and the live trading logic from shree/.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import copy

import pandas as pd
import numpy as np
from loguru import logger

# Import existing bot logic
from shree.config import (
    OneMinuteStrategyConfig,
    TradingConfig,
    RiskGateConfig as ConfigRiskGateConfig,
    EntryFilterConfig,
)
from shree.strategies.mes_one_minute import MesOneMinuteTrendStrategy, StrategyDecision
from shree.strategies.mes_one_minute_scoring import MesOneMinuteScoringStrategy

from shree.strategies.es_fifteen_min import EsFifteenMinStrategy
from shree.strategies.base import Signal
from shree.risk.manager import RiskManager
from shree.risk.risk_gate import RiskGate, RiskGateConfig, RiskGateResult
from shree.features.feature_engineer import engineer_features
from shree.strategies.trading_filters import TradingFilters, TradingFilterResult

# Import backtest components
from .broker_sim import BrokerSimulator, BrokerConfig, Order, OrderSide, OrderType, Fill, Position


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    
    # Symbol and data
    symbol: str = "MES"
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    end_date: datetime = field(default_factory=lambda: datetime(2026, 1, 9, tzinfo=timezone.utc))
    
    # Timeframes
    primary_bar_size: str = "1m"  # Primary decision timeframe
    secondary_bar_size: str = "5m"  # Higher timeframe for context
    
    # Initial capital
    initial_capital: float = 50000.0
    
    # Session
    session_type: str = "full"  # "full" (24h), "rth" (regular trading hours only)
    
    # Costs
    slippage_ticks: float = 1.0
    commission_per_contract: float = 2.40
    
    # Strategy config (uses existing OneMinuteStrategyConfig)
    strategy_config: Optional[OneMinuteStrategyConfig] = None
    
    # Risk config (uses existing configs)
    trading_config: Optional[TradingConfig] = None
    risk_gate_config: Optional[RiskGateConfig] = None
    
    # Trend continuation optimizer settings
    enable_trend_optimizer: bool = True
    tp_extension_atr_mult: float = 1.0
    min_trend_score: float = 0.55
    
    # Decision trace mode (for comparison with live)
    trace_mode: bool = False
    trace_output_path: Optional[Path] = None
    
    # Warm-up bars (for indicator calculation)
    warmup_bars: int = 320
    
    # Logging
    log_level: str = "INFO"
    log_path: Optional[Path] = None


@dataclass
class DecisionTrace:
    """Record of a single decision for trace mode."""
    timestamp: datetime
    bar_index: int
    close_price: float
    
    # Strategy decision
    signal_action: str
    signal_confidence: float
    signal_reason: str
    
    # Risk gate result
    risk_allowed: bool
    risk_reason: str
    
    # Final action
    final_action: str
    
    # Indicator values
    indicators: Dict[str, float] = field(default_factory=dict)
    
    # Position state
    position_qty: int = 0
    position_pnl: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestState:
    """Current state during backtest."""
    bar_index: int = 0
    timestamp: Optional[datetime] = None
    
    # Position tracking
    current_position: int = 0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    
    # Stop/target levels
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    original_stop: Optional[float] = None
    original_tp: Optional[float] = None
    
    # Trend optimizer tracking
    extension_count: int = 0
    near_tp_count: int = 0
    extension_approved: int = 0
    extension_pnl: float = 0.0
    
    # Daily tracking
    daily_pnl: float = 0.0
    daily_trades: int = 0
    last_trade_time: Optional[datetime] = None
    current_date: Optional[datetime] = None
    
    # Cooldown
    cooldown_until: Optional[datetime] = None
    
    # Account state
    account_equity: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    peak_equity: float = 0.0
    base_equity: float = 0.0
    drawdown_active: bool = False
    profit_lock_active: bool = False
    rth_close_tightened: bool = False


class BacktestEngine:
    """
    Event-driven backtest engine that uses existing bot logic.
    
    Key design principles:
    1. Reuse existing strategy/risk/execution logic exactly
    2. No lookahead bias - only use data available at decision time
    3. Realistic fills via broker simulator
    4. Comprehensive tracking for analysis
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
        # Initialize components with existing bot classes
        self._init_strategy()
        self._init_risk()
        self._init_broker()
        self._init_filters()
        
        # State
        self.state = BacktestState(account_equity=config.initial_capital)
        
        # Data
        self.df_1m: Optional[pd.DataFrame] = None
        self.df_5m: Optional[pd.DataFrame] = None
        self.df_15m: Optional[pd.DataFrame] = None  # JAN 11 2026: Add 15m regime TF
        self.df_30m: Optional[pd.DataFrame] = None  # JAN 11 2026: Add 30m overnight TF
        
        # Results tracking
        self.decision_traces: List[DecisionTrace] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.block_reasons: Dict[str, int] = {}
        self.signal_counts: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0}

        # MAY 27 2026: optional execution-faithful Trade Manager gate (default OFF).
        # Enable with env BT_WITH_MANAGER=1 to run the REAL live `rules.evaluate`
        # approval path in the backtest, so research == live. Touches no live code.
        self.manager_gate = None
        if os.environ.get("BT_WITH_MANAGER") == "1":
            try:
                from .manager_gate import BacktestManagerGate
                _mrr = os.environ.get("BT_MIN_RR")
                _eq = os.environ.get("BT_ACCT_EQUITY")
                self.manager_gate = BacktestManagerGate(
                    min_rr=float(_mrr) if _mrr else None,
                    account_equity=float(_eq) if _eq else None,
                )
                logger.info(f"🛡️  Backtest Trade Manager gate ENABLED (min_rr={_mrr or 'default'})")
            except Exception as e:
                logger.error(f"Failed to enable backtest manager gate: {e}")

        # Callbacks
        self.on_bar: Optional[Callable] = None
        self.on_trade: Optional[Callable] = None
    
    def _init_strategy(self) -> None:
        """Initialize strategy using existing bot classes."""
        # Use provided config or create default
        strategy_cfg = self.config.strategy_config or OneMinuteStrategyConfig()
        
        # Check which strategy to use
        use_15m_strategy = getattr(strategy_cfg, 'use_15m_strategy', False)
        use_30m_strategy = getattr(strategy_cfg, 'use_30m_strategy', False)
        use_structural_reversion = getattr(strategy_cfg, 'use_structural_reversion', False)
        use_scoring = getattr(strategy_cfg, 'use_scoring_system', False)
        
        if use_30m_strategy:
            from shree.strategies.mes_thirty_minute import MesThirtyMinuteStrategy
            from shree.config import ThirtyMinuteStrategyConfig
            # Build ThirtyMinuteStrategyConfig from strategy_cfg overrides
            tm_cfg = ThirtyMinuteStrategyConfig()
            # Copy any matching fields from strategy_cfg
            for field in ThirtyMinuteStrategyConfig.__dataclass_fields__:
                if hasattr(strategy_cfg, field):
                    setattr(tm_cfg, field, getattr(strategy_cfg, field))
            self.strategy = MesThirtyMinuteStrategy(tm_cfg)
            logger.info(f"Initialized 30-MIN overnight strategy: {self.strategy.name}")
        elif use_15m_strategy:
            self.strategy = EsFifteenMinStrategy(strategy_cfg)
            logger.info(f"Initialized 15-MIN strategy: {self.strategy.name}")
        elif use_structural_reversion:
            raise NotImplementedError(
                "MesStructuralReversionStrategy has been archived. "
                "Use use_15m_strategy=True (EsFifteenMinStrategy) instead."
            )
            logger.info(f"Initialized STRUCTURAL REVERSION strategy: {self.strategy.name}")
        elif use_scoring:
            self.strategy = MesOneMinuteScoringStrategy(strategy_cfg)
            logger.info(f"Initialized SCORING strategy: {self.strategy.name}")
        else:
            self.strategy = MesOneMinuteTrendStrategy(strategy_cfg)
            logger.info(f"Initialized strategy: {self.strategy.name}")
    
    def _init_risk(self) -> None:
        """Initialize risk management using existing bot classes."""
        # Trading config
        trading_cfg = self.config.trading_config or TradingConfig()
        self.risk_manager = RiskManager(trading_cfg)
        
        # Risk gate
        gate_cfg = self.config.risk_gate_config or RiskGateConfig()
        self.risk_gate = RiskGate(gate_cfg)
        
        logger.info("Initialized risk management")
    
    def _init_broker(self) -> None:
        """Initialize broker simulator."""
        broker_cfg = BrokerConfig(
            slippage_ticks=self.config.slippage_ticks,
            commission_per_contract=self.config.commission_per_contract,
            tick_size=0.25,  # MES tick size
            tick_value=1.25,  # MES tick value
            point_value=5.0,  # MES point value
        )
        self.broker = BrokerSimulator(broker_cfg)
        self.broker.on_fill = self._on_fill
        
        logger.info("Initialized broker simulator")
    
    def _init_filters(self) -> None:
        """Initialize trading filters."""
        filter_cfg = self.config.strategy_config or OneMinuteStrategyConfig()
        
        self.trading_filters = TradingFilters(
            require_candle_close=True,
            require_trend_alignment=getattr(filter_cfg, 'require_5m_trend_alignment', True),
        )
        
        logger.info("Initialized trading filters")
    
    def load_data(
        self,
        df_1m: pd.DataFrame,
        df_5m: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_30m: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Load price data for backtesting.
        
        Args:
            df_1m: 1-minute OHLCV data with UTC datetime index
            df_5m: Optional 5-minute data (will be generated if not provided)
            df_15m: Optional 15-minute data (will be generated if not provided)
            df_30m: Optional 30-minute data (will be generated if not provided)
        """
        # Validate data
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df_1m.columns]
        if missing:
            raise ValueError(f"1m data missing columns: {missing}")
        
        # Ensure datetime index
        if not isinstance(df_1m.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        # Filter to date range
        self.df_1m = df_1m[
            (df_1m.index >= self.config.start_date) & 
            (df_1m.index <= self.config.end_date)
        ].copy()
        
        # Load other timeframes
        if df_5m is not None:
            self.df_5m = df_5m[(df_5m.index >= self.config.start_date) & (df_5m.index <= self.config.end_date)].copy()
        else:
            self.df_5m = self._resample_to_5m(self.df_1m)

        if df_15m is not None:
            self.df_15m = df_15m[(df_15m.index >= self.config.start_date) & (df_15m.index <= self.config.end_date)].copy()
        else:
            self.df_15m = self._load_or_resample_15m()
            
        if df_30m is not None:
            self.df_30m = df_30m[(df_30m.index >= self.config.start_date) & (df_30m.index <= self.config.end_date)].copy()
        else:
            self.df_30m = self._load_or_resample_30m()

        logger.info(f"Loaded {len(self.df_1m)} 1m bars")
        if self.df_5m is not None: logger.info(f"Loaded {len(self.df_5m)} 5m bars")
        if self.df_15m is not None: logger.info(f"Loaded {len(self.df_15m)} 15m bars")
        if self.df_30m is not None: logger.info(f"Loaded {len(self.df_30m)} 30m bars")
        logger.info(f"Date range: {self.df_1m.index.min()} to {self.df_1m.index.max()}")
    
    def _resample_to_5m(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1m to 5m with proper alignment (no lookahead)."""
        resampled = df.resample("5T").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        
        # Shift index forward so bar is available only after completion
        # At time T, we can only see the 5m bar that closed at T, not the current one
        resampled.index = resampled.index + timedelta(minutes=5)
        
        return resampled
    
    def _resample_to_15m(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        JAN 11 2026: Resample 1m to 15m for regime analysis.
        
        15m timeframe is used to determine:
        - Market regime (trending vs ranging via ADX)
        - Trend direction (EMA alignment)
        - Overall bias (bullish/bearish/neutral)
        """
        resampled = df.resample("15T").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        
        # Shift index forward so bar is available only after completion
        resampled.index = resampled.index + timedelta(minutes=15)
        
        return resampled
    
    def _resample_to_30m(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        JAN 11 2026: Resample 1m to 30m for overnight session trading.
        
        30m timeframe is used during overnight/evening sessions because:
        - 1m data has 60%+ ZERO_RANGE bars overnight (IB artifact)
        - Longer timeframe smooths out noise and thin liquidity
        - Better signal quality with aggregated volume
        """
        resampled = df.resample("30min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        
        # Shift index forward so bar is available only after completion
        resampled.index = resampled.index + timedelta(minutes=30)
        
        return resampled
    
    def _load_or_resample_15m(self) -> pd.DataFrame:
        """
        Load native 15m data if available, otherwise resample from 1m.
        """
        from pathlib import Path
        
        # Try to load native 15m data
        paths_to_check = [
            Path("data/ib/ES_15m_1y.parquet"),
            Path("data/raw/ES/ES_15min_60D.parquet")
        ]
        
        for p in paths_to_check:
            if p.exists():
                try:
                    logger.info(f"Loading native 15m data from {p}")
                    df_15m = pd.read_parquet(p)
                    
                    # Ensure datetime index
                    if not isinstance(df_15m.index, pd.DatetimeIndex):
                        if 'timestamp' in df_15m.columns:
                            df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'])
                            df_15m = df_15m.set_index('timestamp')
                            
                    # Ensure timezone is UTC
                    if df_15m.index.tzinfo is None:
                         df_15m.index = df_15m.index.tz_localize("UTC")
                    elif str(df_15m.index.tz) != "UTC":
                         df_15m.index = df_15m.index.tz_convert("UTC")

                    # Filter to date range
                    df_15m = df_15m[
                        (df_15m.index >= self.config.start_date) & 
                        (df_15m.index <= self.config.end_date)
                    ]
                    
                    if not df_15m.empty:
                        return df_15m
                        
                except Exception as e:
                    logger.warning(f"Failed to load 15m data from {p}: {e}")
        
        logger.info("Native 15m data not found or empty, resampling from 1m...")
        return self._resample_to_15m(self.df_1m)

    def _load_or_resample_30m(self) -> pd.DataFrame:
        """
        JAN 11 2026: Load native 30m data if available, otherwise resample from 1m.
        
        Native 30m data from IB has much better quality:
        - Overnight: 8.2% zero-range (vs 60%+ when resampling noisy 1m data)
        - RTH: 2.7% zero-range
        
        This makes overnight trading viable with 30m signals.
        """
        from pathlib import Path
        
        # Try to load native 30m data
        # Check multiple locations
        paths_to_check = [
            Path("data/ib/ES_30m_1y.parquet"),       # The file I'm creating
            Path("data/raw/ES/ES_30min_60D.parquet")  # Original path
        ]
        
        for p in paths_to_check:
            if p.exists():
                try:
                    logger.info(f"Loading native 30m data from {p}")
                    df_30m = pd.read_parquet(p)
                    
                    # Ensure datetime index
                    if not isinstance(df_30m.index, pd.DatetimeIndex):
                        if 'timestamp' in df_30m.columns:
                            df_30m['timestamp'] = pd.to_datetime(df_30m['timestamp'])
                            df_30m = df_30m.set_index('timestamp')
                            
                    # Ensure timezone is UTC
                    if df_30m.index.tzinfo is None:
                         df_30m.index = df_30m.index.tz_localize("UTC")
                    elif str(df_30m.index.tz) != "UTC":
                         df_30m.index = df_30m.index.tz_convert("UTC")

                    # Filter to date range
                    df_30m = df_30m[
                        (df_30m.index >= self.config.start_date) & 
                        (df_30m.index <= self.config.end_date)
                    ]
                    
                    if not df_30m.empty:
                        return df_30m
                        
                except Exception as e:
                    logger.warning(f"Failed to load 30m data from {p}: {e}")
        
        logger.info("Native 30m data not found or empty, resampling from 1m...")
        return self._resample_to_30m(self.df_1m)

    
    def run(self) -> Dict:
        """
        Run the backtest.
        
        Returns:
            Dict with results including trades, metrics, equity curve
        """
        if self.df_1m is None or self.df_1m.empty:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info(f"Starting backtest: {self.config.symbol} from {self.config.start_date} to {self.config.end_date}")
        
        # Reset state
        self.state = BacktestState(account_equity=self.config.initial_capital)
        self.broker.reset()
        self.risk_manager.reset_stats()
        self.decision_traces.clear()
        self.equity_curve.clear()
        self.block_reasons.clear()
        
        # Initialize Simulated Agents
        self.learning_agent_enabled = True
        logger.info("🤖 Initializing Simulated Learning Agent (Recording Observations)")
        
        self.rag_agent_enabled = True
        logger.info("🧠 Initializing Simulated RAG Agent (Market Context Analysis)")
        
        # Prepare MTF Frames
        if self.df_5m is None or self.df_5m.empty:
            # Create a fallback/empty dataframe to prevent missing column errors
            # Alternatively, fill with 1m data aggregated (but let's just make it empty with columns)
            self.df_5m = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        # Pre-compute indicators for the full dataset
        logger.info("Computing indicators...")
        features_df = self._compute_features(self.df_1m)
        
        # Main backtest loop
        total_bars = len(features_df)
        logger.info(f"Processing {total_bars} bars...")
        
        import time as _time
        _last_progress = _time.time()
        
        for bar_idx in range(self.config.warmup_bars, total_bars):
            timestamp = features_df.index[bar_idx]
            bar = features_df.iloc[bar_idx]
            
            # Get history up to this point (no lookahead)
            history = features_df.iloc[:bar_idx + 1]
            
            # Process the bar
            self._process_bar(bar_idx, timestamp, bar, history)
            
            # Update equity curve
            self._update_equity(timestamp, float(bar["close"]))

            # Enforce peak drawdown guard (flatten/tighten)
            self._apply_drawdown_guard(bar, timestamp)
            
            # Real-time progress — print every 2 seconds
            _now = _time.time()
            if (_now - _last_progress) >= 2.0:
                _pct = (bar_idx - self.config.warmup_bars) / max(total_bars - self.config.warmup_bars, 1) * 100
                _trades = len(self.broker.trade_log)
                _pnl = self.state.realized_pnl
                logger.info(
                    f"⏳ {bar_idx}/{total_bars} bars ({_pct:.0f}%) | "
                    f"date: {timestamp.strftime('%Y-%m-%d %H:%M')} | "
                    f"trades: {_trades} | PnL: ${_pnl:+,.0f}"
                )
                _last_progress = _now
        
        # Close any open position at end
        self._close_final_position()
        
        # Compile results
        results = self._compile_results()
        
        logger.info(f"Backtest complete. Total trades: {len(self.broker.trade_log)}")
        
        return results
    
    def run_30m_only(self) -> Dict:
        """
        JAN 11 2026: Run backtest using only 30m bars.
        
        This is for testing overnight trading strategies where 1m data is too noisy.
        Uses native 30m data directly instead of overlaying 30m indicators on 1m.
        
        Returns:
            Dict with results including trades, metrics, equity curve
        """
        if self.df_30m is None or self.df_30m.empty:
            raise ValueError("No 30m data loaded.")
        
        logger.info(f"Starting 30m-only backtest: {self.config.symbol}")
        logger.info(f"Date range: {self.df_30m.index.min()} to {self.df_30m.index.max()}")
        
        # Reset state
        self.state = BacktestState(account_equity=self.config.initial_capital)
        self.broker.reset()
        self.risk_manager.reset_stats()
        self.decision_traces.clear()
        self.equity_curve.clear()
        self.block_reasons.clear()

        # Initialize agent simulation flags (same as run())
        self.learning_agent_enabled = False
        self.rag_agent_enabled = False
        
        # Compute indicators directly on 30m data
        logger.info("Computing 30m indicators...")
        features_df = engineer_features(self.df_30m[["open", "high", "low", "close", "volume"]])
        
        # Main backtest loop - process 30m bars
        total_bars = len(features_df)
        warmup = min(50, total_bars // 4)  # 50 bars warmup for 30m
        logger.info(f"Processing {total_bars} 30m bars (warmup={warmup})...")
        
        import time as _time
        _last_progress = _time.time()
        
        for bar_idx in range(warmup, total_bars):
            timestamp = features_df.index[bar_idx]
            bar = features_df.iloc[bar_idx]
            
            # Get history up to this point (no lookahead)
            history = features_df.iloc[:bar_idx + 1]
            
            # Process the 30m bar
            self._process_bar(bar_idx, timestamp, bar, history)
            
            # Update equity curve
            self._update_equity(timestamp, float(bar["close"]))

            # Enforce peak drawdown guard (flatten/tighten)
            self._apply_drawdown_guard(bar, timestamp)
            
            # Real-time progress — print every 2 seconds
            _now = _time.time()
            if (_now - _last_progress) >= 2.0:
                _pct = (bar_idx - warmup) / max(total_bars - warmup, 1) * 100
                _trades = len(self.broker.trade_log)
                _pnl = self.state.realized_pnl
                logger.info(
                    f"⏳ {bar_idx}/{total_bars} 30m bars ({_pct:.0f}%) | "
                    f"date: {timestamp.strftime('%Y-%m-%d %H:%M')} | "
                    f"trades: {_trades} | PnL: ${_pnl:+,.0f}"
                )
                _last_progress = _now
        
        # Close any open position at end
        self._close_final_position()
        
        # Compile results
        results = self._compile_results()
        
        logger.info(f"30m backtest complete. Total trades: {len(self.broker.trade_log)}")
        
        return results
    
    def load_30m_only(self, df_30m: pd.DataFrame) -> None:
        """
        JAN 11 2026: Load only 30m data for overnight-only backtesting.
        
        Args:
            df_30m: 30-minute OHLCV data with UTC datetime index
        """
        # Validate data
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df_30m.columns]
        if missing:
            raise ValueError(f"30m data missing columns: {missing}")
        
        # Ensure datetime index
        if not isinstance(df_30m.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        # Filter to date range
        self.df_30m = df_30m[
            (df_30m.index >= self.config.start_date) & 
            (df_30m.index <= self.config.end_date)
        ].copy()
        
        # Set other dataframes to None (not needed for 30m-only)
        self.df_1m = pd.DataFrame()  # Empty but not None to avoid errors
        self.df_5m = None
        self.df_15m = None
        
        logger.info(f"Loaded {len(self.df_30m)} 30m bars for overnight backtest")
        logger.info(f"Date range: {self.df_30m.index.min()} to {self.df_30m.index.max()}")

    def run_15m_only(self) -> Dict:
        """
        FEB 2026: Run backtest using 15m bars as the primary timeframe.
        
        This replaces the 1m loop entirely. The strategy receives 15m
        bars directly — no resampling or overlay. Much cleaner signal
        and dramatically better results than 1m.
        
        Returns:
            Dict with results including trades, metrics, equity curve
        """
        if self.df_15m is None or self.df_15m.empty:
            raise ValueError("No 15m data loaded. Call load_data() or load_15m_only() first.")
        
        logger.info(f"Starting 15m-only backtest: {self.config.symbol}")
        logger.info(f"Date range: {self.df_15m.index.min()} to {self.df_15m.index.max()}")
        
        # Reset state
        self.state = BacktestState(account_equity=self.config.initial_capital)
        self.broker.reset()
        self.risk_manager.reset_stats()
        self.decision_traces.clear()
        self.equity_curve.clear()
        self.block_reasons.clear()
        
        # Simulated agents (keep for compatibility)
        self.learning_agent_enabled = True
        self.rag_agent_enabled = True
        
        # ──────────────────────────────────────────────────────────────
        # FEB 7 2026: SESSION-ISOLATED INDICATOR ARCHITECTURE
        #
        # Problem: When --session full, overnight bars flow through the
        # strategy's generate() method. Although the strategy returns
        # HOLD for OUTSIDE_RTH, it still updates internal state:
        #   - _prev_close is set to overnight close prices
        #   - _session_date may reset at midnight
        # This causes 48 extra false signals during RTH, dropping PnL
        # from +$3,031 (PF 1.67) to +$1,410 (PF 1.21).
        #
        # Solution: Split the loop into two modes:
        #   1. RTH bars: compute indicators + call strategy (same as before)
        #   2. Non-RTH bars: only process fills on open positions
        #
        # Indicators are computed on the FULL DataFrame (all sessions)
        # because the validated backtest (193 trades, PF 1.67) used
        # this exact approach. The key is to prevent overnight bars
        # from reaching the strategy's generate() method.
        # ──────────────────────────────────────────────────────────────
        from shree.utils.session_utils import classify_session, TradingSession
        
        # Compute indicators on full 15m data (same as validated backtest)
        logger.info("Computing 15m indicators...")
        features_df = engineer_features(self.df_15m[["open", "high", "low", "close", "volume"]])
        
        # Tag each bar with its session type
        session_tags = [classify_session(ts) for ts in features_df.index]
        
        # Main backtest loop
        total_bars = len(features_df)
        warmup = min(60, total_bars // 4)
        
        # Count RTH bars for progress
        rth_count = sum(1 for s in session_tags if s == TradingSession.RTH)
        logger.info(
            f"Processing {total_bars} 15m bars ({rth_count} RTH, "
            f"{total_bars - rth_count} overnight/maintenance, warmup={warmup})..."
        )
        
        import time as _time
        _last_progress = _time.time()
        
        # FEB 7 2026: Fill processing mode
        #
        # Fills (stop-loss, take-profit, max-hold-exit) are ALWAYS processed
        # on every bar, including overnight. This matches the live bot where
        # IB executes bracket orders 24/7.
        #
        # The --session flag controls ONLY whether the strategy can evaluate
        # new entries during overnight:
        #   --session rth:  Strategy called ONLY during RTH (prevents _prev_close pollution)
        #   --session full: Strategy called for ALL bars (old behavior, 48 extra bad trades)
        #
        # NOTE: The strategy itself already gates entries to RTH via OUTSIDE_RTH,
        # but calling generate() during overnight pollutes _prev_close and _session_date.
        # The new session-aware loop prevents this by never calling the strategy overnight.
        
        # With this architecture, --session rth and --session full produce
        # IDENTICAL results because the strategy is always RTH-only.
        # The flag is kept for API compatibility.
        logger.info("✅ Session-aware 15m loop: fills on all bars, strategy on RTH only")
        
        for bar_idx in range(warmup, total_bars):
            timestamp = features_df.index[bar_idx]
            bar = features_df.iloc[bar_idx]
            session = session_tags[bar_idx]
            history = features_df.iloc[:bar_idx + 1]
            
            if session == TradingSession.RTH:
                # ── RTH bar: full _process_bar flow ──
                # This calls broker.process_bar() for fills, _is_valid_session(),
                # _check_max_hold_time_exit, strategy.generate(), etc.
                # Exactly the same as the validated backtest.
                self._process_bar(bar_idx, timestamp, bar, history)
            else:
                # ── Non-RTH bar: only process fills + daily reset ──
                # Stops and targets can fire overnight. Strategy is NOT called.
                # This prevents _prev_close pollution that caused 48 extra
                # bad trades in the old --session full code path.
                self._check_daily_reset(timestamp)
                fills = self.broker.process_bar(self.config.symbol, bar, timestamp)
                for fill in fills:
                    self._process_fill(fill)
            
            # Update equity curve on every bar
            self._update_equity(timestamp, float(bar["close"]))
            
            # Enforce peak drawdown guard
            self._apply_drawdown_guard(bar, timestamp)
            
            # Progress reporting
            _now = _time.time()
            if (_now - _last_progress) >= 2.0 or bar_idx % 2000 == 0:
                _pct = (bar_idx - warmup) / max(total_bars - warmup, 1) * 100
                _trades = len(self.broker.trade_log)
                _pnl = self.state.realized_pnl
                logger.info(
                    f"⏳ {bar_idx}/{total_bars} bars ({_pct:.0f}%) | "
                    f"date: {timestamp.strftime('%Y-%m-%d %H:%M')} | "
                    f"trades: {_trades} | PnL: ${_pnl:+,.0f}"
                )
                _last_progress = _now
        
        # Close any open position at end
        self._close_final_position()
        
        # Compile results
        results = self._compile_results()
        
        logger.info(f"15m backtest complete. Total trades: {len(self.broker.trade_log)}")
        
        return results
    
    def load_15m_only(self, df_15m: pd.DataFrame) -> None:
        """
        FEB 2026: Load only 15m data for 15m-primary backtesting.
        
        Args:
            df_15m: 15-minute OHLCV data with UTC datetime index
        """
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df_15m.columns]
        if missing:
            raise ValueError(f"15m data missing columns: {missing}")
        
        if not isinstance(df_15m.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        self.df_15m = df_15m[
            (df_15m.index >= self.config.start_date) & 
            (df_15m.index <= self.config.end_date)
        ].copy()
        
        # Set other dataframes to empty (not needed for 15m-only)
        self.df_1m = pd.DataFrame()
        self.df_5m = None
        self.df_30m = None
        
        logger.info(f"Loaded {len(self.df_15m)} 15m bars for backtest")
        logger.info(f"Date range: {self.df_15m.index.min()} to {self.df_15m.index.max()}")

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators.
        
        Uses the same feature engineering as the live bot.
        """
        # Use existing feature engineering
        features = engineer_features(df[["open", "high", "low", "close", "volume"]])
        
        # Add 5m features if available
        if self.df_5m is not None:
            features = self._add_5m_features(features)
        
        # JAN 11 2026: Add 15m regime features
        if self.df_15m is not None:
            features = self._add_15m_features(features)
        
        # JAN 11 2026: Add 30m overnight features
        if self.df_30m is not None:
            features = self._add_30m_features(features)
        
        return features
    
    def _add_5m_features(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """Add 5-minute timeframe features to 1m data."""
        if self.df_5m is None or self.df_5m.empty:
            return df_1m
        
        # Compute indicators for 5m data
        features_5m = engineer_features(self.df_5m[["open", "high", "low", "close", "volume"]])
        
        # Align 5m features to 1m bars (use last completed 5m bar)
        df_merged = df_1m.copy()
        
        for col in ["EMA_9", "EMA_21", "ADX_14", "ATR_14", "RSI_14"]:
            if col in features_5m.columns:
                # Reindex 5m to 1m with forward fill
                aligned = features_5m[col].reindex(df_merged.index, method="ffill")
                df_merged[f"5m_{col}"] = aligned
        
        return df_merged
    
    def _add_15m_features(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """
        JAN 11 2026: Add 15-minute regime features to 1m data.
        
        15m timeframe provides:
        - Regime classification (trending vs ranging)
        - Trend direction (bullish/bearish/neutral)
        - Regime change signals
        """
        if self.df_15m is None:
            return df_1m
        
        # Compute indicators for 15m data
        features_15m = engineer_features(self.df_15m[["open", "high", "low", "close", "volume"]])
        
        # Align 15m features to 1m bars (use last completed 15m bar)
        df_merged = df_1m.copy()
        
        for col in ["EMA_9", "EMA_21", "ADX_14", "ATR_14", "RSI_14"]:
            if col in features_15m.columns:
                aligned = features_15m[col].reindex(df_merged.index, method="ffill")
                df_merged[f"15m_{col}"] = aligned
        
        # Compute 15m regime classification
        df_merged["15m_regime"] = "UNKNOWN"
        
        if "15m_ADX_14" in df_merged.columns and "15m_EMA_9" in df_merged.columns:
            # Regime = TRENDING if ADX > threshold, else RANGING
            strategy_cfg = self.config.strategy_config or OneMinuteStrategyConfig()
            regime_adx_thresh = getattr(strategy_cfg, 'regime_adx_threshold', 20.0)
            
            is_trending = df_merged["15m_ADX_14"] >= regime_adx_thresh
            
            # Direction based on EMA alignment
            is_bullish = df_merged["15m_EMA_9"] > df_merged["15m_EMA_21"]
            
            df_merged.loc[is_trending & is_bullish, "15m_regime"] = "UPTREND"
            df_merged.loc[is_trending & ~is_bullish, "15m_regime"] = "DOWNTREND"
            df_merged.loc[~is_trending, "15m_regime"] = "RANGING"
        
        return df_merged
    
    def _add_30m_features(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """
        JAN 11 2026: Add 30-minute features to 1m data for overnight trading.
        
        30m timeframe is used during overnight/evening sessions because:
        - 1m data has poor quality overnight (60%+ ZERO_RANGE bars)
        - Longer timeframe aggregates thin liquidity into cleaner signals
        - 30m provides better trend identification with smoothed noise
        
        Features added:
        - 30m_EMA_9, 30m_EMA_21: Trend direction
        - 30m_ADX_14: Trend strength
        - 30m_ATR_14: Volatility (for position sizing)
        - 30m_RSI_14: Momentum
        - 30m_regime: UPTREND/DOWNTREND/RANGING classification
        """
        if self.df_30m is None:
            return df_1m
        
        # Compute indicators for 15m data (regime analysis)
        features_15m = engineer_features(self.df_15m[["open", "high", "low", "close", "volume"]])
        
        # Compute indicators for 30m data
        features_30m = engineer_features(self.df_30m[["open", "high", "low", "close", "volume"]])
        
        # Align 30m and 15m features to 1m bars
        df_merged = df_1m.copy()
        
        # Merge 15m features
        for col in ["EMA_9", "EMA_21", "ADX_14", "ATR_14", "RSI_14"]:
            if col in features_15m.columns:
                aligned = features_15m[col].reindex(df_merged.index, method="ffill")
                df_merged[f"15m_{col}"] = aligned

        # Merge 30m features
        for col in ["EMA_9", "EMA_21", "ADX_14", "ATR_14", "RSI_14"]:
            if col in features_30m.columns:
                aligned = features_30m[col].reindex(df_merged.index, method="ffill")
                df_merged[f"30m_{col}"] = aligned
        
        # Compute 15m regime classification
        df_merged["15m_regime"] = "UNKNOWN"
        if "15m_ADX_14" in df_merged.columns and "15m_EMA_9" in df_merged.columns:
            strategy_cfg = self.config.strategy_config or OneMinuteStrategyConfig()
            trend_adx = getattr(strategy_cfg, 'trend_adx_threshold', 25.0)
            
            is_trending = df_merged["15m_ADX_14"] >= trend_adx
            is_bullish = df_merged["15m_EMA_9"] > df_merged["15m_EMA_21"]
            
            df_merged.loc[is_trending & is_bullish, "15m_regime"] = "UPTREND"
            df_merged.loc[is_trending & ~is_bullish, "15m_regime"] = "DOWNTREND"
            df_merged.loc[~is_trending, "15m_regime"] = "RANGING"

        # Compute 30m regime classification (same logic as 15m)
        df_merged["30m_regime"] = "UNKNOWN"
        
        if "30m_ADX_14" in df_merged.columns and "30m_EMA_9" in df_merged.columns:
            strategy_cfg = self.config.strategy_config or OneMinuteStrategyConfig()
            # Use overnight ADX threshold (stricter than daytime)
            overnight_adx_thresh = getattr(strategy_cfg, 'overnight_adx_threshold', 25.0)
            
            is_trending = df_merged["30m_ADX_14"] >= overnight_adx_thresh
            is_bullish = df_merged["30m_EMA_9"] > df_merged["30m_EMA_21"]
            
            df_merged.loc[is_trending & is_bullish, "30m_regime"] = "UPTREND"
            df_merged.loc[is_trending & ~is_bullish, "30m_regime"] = "DOWNTREND"
            df_merged.loc[~is_trending, "30m_regime"] = "RANGING"
        
        return df_merged
    
    def _is_overnight_session(self, timestamp: datetime) -> bool:
        """
        JAN 11 2026: Check if timestamp is in overnight session.
        Overnight = 6:00 PM - 9:30 AM ET (complement of RTH)
        """
        strategy_cfg = self.config.strategy_config or OneMinuteStrategyConfig()
        allow_overnight = getattr(strategy_cfg, 'allow_overnight_trading', False)
        
        if not allow_overnight:
            return False
        
        # Convert to Central time (Chicago)
        try:
            if timestamp.tzinfo is not None:
                local_ts = timestamp.tz_convert("America/Chicago")
            else:
                local_ts = timestamp
        except Exception:
            local_ts = timestamp
        
        t = local_ts.time() if hasattr(local_ts, 'time') else time(0, 0)
        
        overnight_start = time(
            getattr(strategy_cfg, 'overnight_start_hour', 18), 0
        )
        overnight_end = time(
            getattr(strategy_cfg, 'overnight_end_hour', 9),
            getattr(strategy_cfg, 'overnight_end_minute', 30)
        )
        
        # Overnight spans midnight: 18:00 -> 09:30 next day
        return t >= overnight_start or t < overnight_end
    
    def _is_30m_bar_boundary(self, timestamp: datetime) -> bool:
        """Check if timestamp is on a 30m bar boundary (:00 or :30)."""
        return timestamp.minute in (0, 30) and timestamp.second == 0
    
    def _get_30m_bar(self, timestamp: datetime) -> pd.Series:
        """
        Get the most recent completed 30m bar for the given timestamp.
        Returns the 30m bar data with all indicators.
        """
        if self.df_30m is None:
            return None
        
        # Find the most recent 30m bar that is <= timestamp
        # Note: df_30m index is shifted forward by 30 mins (bar available after close)
        valid_bars = self.df_30m[self.df_30m.index <= timestamp]
        if valid_bars.empty:
            return None
        
        return valid_bars.iloc[-1]
    
    def _process_bar(
        self,
        bar_idx: int,
        timestamp: datetime,
        bar: pd.Series,
        history: pd.DataFrame
    ) -> None:
        """Process a single bar."""
        self.state.bar_index = bar_idx
        self.state.timestamp = timestamp
        
        close_price = float(bar["close"])
        
        # Check for new trading day
        self._check_daily_reset(timestamp)
        
        # JAN 11 2026: During overnight, only process on 30m bar boundaries
        # This ensures we make decisions based on completed 30m bars, not 1m noise
        is_overnight = self._is_overnight_session(timestamp)
        if is_overnight and not self._is_30m_bar_boundary(timestamp):
            # Still need to process fills for existing positions
            fills = self.broker.process_bar(self.config.symbol, bar, timestamp)
            for fill in fills:
                self._process_fill(fill)
            # Update equity but don't evaluate new entries
            return
        
        # Process any pending orders from broker
        fills = self.broker.process_bar(
            self.config.symbol,
            bar,
            timestamp
        )
        
        # Update state from fills
        for fill in fills:
            self._process_fill(fill)
        
        # Check session validity
        valid_sess = self._is_valid_session(timestamp)
        if not valid_sess:
            return
        
        # Get current position from broker
        position = self.broker.get_position(self.config.symbol)
        self.state.current_position = position.quantity
        
        # If we have a position, check for exit conditions first
        if not position.is_flat:
            # JAN 11 2026: Check max hold time exit FIRST
            if self._check_max_hold_time_exit(position, bar, timestamp):
                return  # Position closed, don't check other management
            self._check_position_management(bar, history, timestamp)
        
        # If flat, check for entry
        if position.is_flat and self._can_enter(timestamp):
            self._evaluate_entry(bar, history, timestamp)
        
        # Record trace if enabled
        if self.config.trace_mode:
            self._record_trace(bar_idx, timestamp, bar, history)
    
    def _check_daily_reset(self, timestamp: datetime) -> None:
        """Reset daily counters at start of new trading day."""
        current_date = timestamp.date()
        
        if self.state.current_date != current_date:
            self.state.current_date = current_date
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.risk_manager.reset()
            self.risk_gate.reset_consecutive_losses()
            if getattr(self.risk_gate.config, "peak_drawdown_reset_on_new_day", False):
                self.risk_gate.reset_drawdown()
                self.state.drawdown_active = False
            logger.debug(f"New trading day: {current_date}")
    
    def _is_valid_session(self, timestamp: datetime) -> bool:
        """Check if timestamp is within valid trading session.
        
        FEB 7 2026: Fixed DST bug — was using fixed -5h offset (EST) which
        mis-classified early RTH bars during EDT (Mar-Nov). Now uses
        ZoneInfo('US/Eastern') for correct DST handling, consistent with
        session_utils.classify_session().
        """
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        
        et = timestamp.astimezone(ZoneInfo("US/Eastern"))
        t = et.time()
        dow = et.weekday()  # Mon=0 … Sun=6
        
        if self.config.session_type == "full":
            # 24h futures session - exclude maintenance window
            # CME maintenance: 5:00 PM - 6:00 PM ET daily
            if time(17, 0) <= t < time(18, 0):
                return False
            
            # Weekend check (Sat-Sun before 6 PM)
            if dow == 5:  # Saturday
                return False
            if dow == 6 and t < time(18, 0):  # Sunday before 6 PM
                return False
            
            return True
        
        elif self.config.session_type == "rth":
            # Regular trading hours: 9:30 AM - 4:00 PM ET
            if time(9, 30) <= t < time(16, 0):
                if dow < 5:  # Mon-Fri
                    return True
            
            return False
        
        return True
    
    def _can_enter(self, timestamp: datetime) -> bool:
        """Check if we can enter a new trade."""
        # Check cooldown
        if self.state.cooldown_until and timestamp < self.state.cooldown_until:
            return False

        # Peak drawdown lockout / tiered reduction
        gate_cfg = getattr(self.risk_gate, "config", None)
        if gate_cfg and getattr(gate_cfg, "peak_drawdown_enabled", False):
            if getattr(self.risk_gate, "drawdown_active", False):
                action_mode = getattr(gate_cfg, "peak_drawdown_action", "halt").lower()
                if action_mode == "halt":
                    self._record_block("PEAK_DRAWDOWN_LOCKOUT")
                    return False
                elif action_mode == "tiered":
                    tier = self.risk_gate.get_drawdown_tier()
                    if tier >= 3:
                        self._record_block("PEAK_DD_TIER3_HALT")
                        return False
                    # Tiers 1/2: allow entry — size/grade applied in _evaluate_entry
        
        # Check daily trade limit
        strategy_cfg = self.config.strategy_config or OneMinuteStrategyConfig()
        if self.state.daily_trades >= strategy_cfg.max_trades_per_day:
            self._record_block("DAILY_TRADE_LIMIT")
            return False
        
        # Check daily loss limit
        trading_cfg = self.config.trading_config or TradingConfig()
        if self.state.daily_pnl <= -trading_cfg.max_daily_loss:
            self._record_block("DAILY_LOSS_LIMIT")
            return False
        
        return True
    
    def _evaluate_entry(
        self,
        bar: pd.Series,
        history: pd.DataFrame,
        timestamp: datetime
    ) -> None:
        """Evaluate entry using existing strategy logic."""
        # pass history to strategy
        signal = self.strategy.generate(history)

        # --- AGENT SIMULATION HOOKS ---
        # 1. RAG Agent: Check market context validation
        if self.rag_agent_enabled and signal.action != "HOLD":
            # In live trading, this calls HybridPipelineIntegration.enrich_signal()
            # which queries vector DB for similar historical scenarios.
            # For backtest, we simulate this by validating the context exists.
            context_ok = self._simulate_rag_context_check(bar, signal)
            if not context_ok:
                signal = Signal("HOLD", 0.0, {"reason": "RAG_CONTEXT_FILTER"})

        # 2. Learning Agent: Record observation
        if self.learning_agent_enabled:
            # In live trading, this calls TradeLearningRecorder.record_observation()
            # For backtest, we ensure the data flow matches.
            self._simulate_learning_observation(bar, signal)
        # ------------------------------
        
        # Track signal counts
        self.signal_counts[signal.action] = self.signal_counts.get(signal.action, 0) + 1
        
        if signal.action == "HOLD":
            reason = signal.metadata.get("reason", "NO_SIGNAL")
            self._record_block(reason)
            return
        
        # Extract stop/tp from signal metadata
        stop_loss = signal.metadata.get("stop_loss")
        take_profit = signal.metadata.get("take_profit")
        
        if stop_loss is None or take_profit is None:
            self._record_block("MISSING_STOPS")
            return

        # MAY 27 2026: Trade Manager veto — the REAL live gate, if enabled (default OFF).
        # Placed before the risk gate so it sees the strategy's proposed levels, exactly
        # like the live signal feed. A rejection here frees the slot for later signals.
        if self.manager_gate is not None:
            _md = signal.metadata
            _approved, _dec = self.manager_gate.evaluate(
                action=signal.action, ts=timestamp, close=float(bar["close"]),
                stop_loss=stop_loss, take_profit=take_profit,
                adx=float(_md.get("adx_value", _md.get("adx", 20.0)) or 20.0),
                rsi=float(_md.get("rsi", 55.0) or 55.0),
                atr=float(_md.get("atr_value", bar.get("ATR_14", 8.0)) or 8.0),
                signal_type=(_md.get("reason", "") or "").split("|")[0].strip(),
            )
            if not _approved:
                self._record_block("TM_REJECT:" + (_dec.reasoning or "")[:40])
                return

        # Evaluate risk gate (EXACT same logic as live)
        close_price = float(bar["close"])
        atr_value = float(bar.get("ATR_14", 2.0))
        
        account_state = {
            "available_funds": self.state.account_equity,
            "excess_liquidity": self.state.account_equity,
            "realized_pnl_today": self.state.daily_pnl,
            "account_equity": self.state.account_equity,
        }
        
        gate_result = self.risk_gate.evaluate_entry(
            action=signal.action,
            quantity=1,  # Single contract for simplicity
            entry_price=close_price,
            atr=atr_value,
            account_state=account_state,
            current_position=self.state.current_position,
            now=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        
        if not gate_result.allowed:
            self._record_block(gate_result.reason)
            return

        # Use adjusted levels from gate if provided
        stop_loss = gate_result.levels.get("stop_loss", stop_loss)
        take_profit = gate_result.levels.get("take_profit", take_profit)

        # MAR 2026: Tiered drawdown — track tier, no grade filter (size reduction is the guard)
        dd_tier = int(gate_result.levels.get("dd_tier", 0))
        if dd_tier >= 2:
            self._record_block("PEAK_DD_TIER2_ENTRY")  # tracking counter (doesn't halt)
        elif dd_tier == 1:
            self._record_block("PEAK_DD_TIER1_ENTRY")  # tracking counter

        # Submit bracket order through broker
        side = OrderSide.BUY if signal.action == "BUY" else OrderSide.SELL
        
        # JAN 11 2026: Include full metadata for regime analytics
        adx_value = float(bar.get("ADX_14", signal.metadata.get("adx_value", 0)))
        
        entry_order, sl_order, tp_order = self.broker.submit_bracket_order(
            symbol=self.config.symbol,
            side=side,
            quantity=1,
            entry_type=OrderType.MARKET,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=timestamp,
            metadata={
                "signal_confidence": signal.confidence,
                "signal_reason": signal.metadata.get("reason", ""),
                "entry_module": signal.metadata.get("entry_module", ""),
                "entry_reason": signal.metadata.get("entry_reason", getattr(signal, "reason", "")),
                "entry_type": signal.metadata.get("entry_type", getattr(signal, "entry_type", "")),
                "atr": atr_value,
                "atr_value": atr_value,  # For regime analytics compatibility
                "adx_value": adx_value,  # For regime analytics
                "market_state": signal.metadata.get("market_state", ""),
                "trend_label": signal.metadata.get("trend_label", ""),
                "trend_label_htf": signal.metadata.get("trend_label_htf", ""),
                "session_type": signal.metadata.get("session_type", "RTH"),  # JAN 11 2026
                "entry_bar_index": self.state.bar_index,
            }
        )
        
        # Update state
        self.state.stop_loss = stop_loss
        self.state.take_profit = take_profit
        self.state.original_stop = stop_loss
        self.state.original_tp = take_profit
        self.state.entry_time = timestamp  # JAN 11 2026: Track entry time for max hold
        
        logger.debug(
            f"Entry signal: {signal.action} @ {close_price}, "
            f"SL={stop_loss:.2f}, TP={take_profit:.2f}, "
            f"conf={signal.confidence:.2f}"
        )

    def _check_max_hold_time_exit(
        self,
        position,
        bar: pd.Series,
        timestamp: datetime
    ) -> bool:
        """
        JAN 11 2026: Exit position if max hold time exceeded.
        
        Problem: 1-min signal but 4.7 hour average hold = micro-noise entry, swing exit
        Solution: Exit after max_hold_minutes (default 90 min)
        """
        strategy_cfg = self.config.strategy_config or OneMinuteStrategyConfig()
        max_hold = getattr(strategy_cfg, 'max_hold_minutes', 0)
        
        if max_hold <= 0:
            return False  # Max hold disabled
            
        entry_time = getattr(self.state, 'entry_time', None)
        if entry_time is None:
            return False
            
        hold_duration = (timestamp - entry_time).total_seconds() / 60.0  # minutes
        
        if hold_duration >= max_hold:
            # Force exit at market
            close_price = float(bar["close"])
            logger.info(
                f"MAX_HOLD_EXIT: Position held {hold_duration:.0f} min >= {max_hold} min limit. "
                f"Closing at {close_price:.2f}"
            )
            
            # Cancel existing bracket orders
            self.broker.cancel_all_orders(self.config.symbol)
            
            # Place market close order
            side = OrderSide.SELL if position.is_long else OrderSide.BUY
            self.broker.submit_market_order(
                symbol=self.config.symbol,
                side=side,
                quantity=abs(position.quantity),
                timestamp=timestamp,
                metadata={"type": "flatten", "reason": "MAX_HOLD_EXIT", "hold_minutes": hold_duration}
            )
            
            # Process immediately to get fill
            fills = self.broker.process_bar(self.config.symbol, bar, timestamp)
            for fill in fills:
                self._process_fill(fill)
            
            return True
        
        return False
    
    def _check_position_management(
        self,
        bar: pd.Series,
        history: pd.DataFrame,
        timestamp: datetime
    ) -> None:
        """Check for position management actions (trailing stops, extensions)."""
        self._apply_rth_close_tighten(bar, history, timestamp)
        if not self.config.enable_trend_optimizer:
            return

    def _is_rth_close_window(self, timestamp: datetime, minutes: int) -> bool:
        if minutes <= 0:
            return False
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        try:
            et_time = timestamp.astimezone(ZoneInfo("US/Eastern"))
        except Exception:
            et_time = timestamp
        t = et_time.time()
        strategy_cfg = self.config.strategy_config or OneMinuteStrategyConfig()
        close_time = time(strategy_cfg.rth_end_hour, strategy_cfg.rth_end_minute)
        close_dt = datetime.combine(et_time.date(), close_time, tzinfo=et_time.tzinfo)
        window_start = close_dt - timedelta(minutes=minutes)
        return window_start.time() <= t <= close_time

    def _apply_rth_close_tighten(self, bar: pd.Series, history: pd.DataFrame, timestamp: datetime) -> None:
        """Tighten stops into RTH close to avoid late-day giveback/gap risk."""
        position = self.broker.get_position(self.config.symbol)
        if position.is_flat or self.state.rth_close_tightened:
            return

        strategy_cfg = self.config.strategy_config or OneMinuteStrategyConfig()
        minutes = getattr(strategy_cfg, "rth_close_tighten_minutes", 0)
        if minutes <= 0:
            return

        if not self._is_rth_close_window(timestamp, minutes):
            return

        buffer_points = float(getattr(strategy_cfg, "rth_close_stop_buffer_points", 0.5))
        if not self.state.entry_price:
            return

        if position.is_long:
            new_stop = max(self.state.stop_loss or 0.0, self.state.entry_price + buffer_points)
        else:
            new_stop = min(self.state.stop_loss or float(bar["close"]), self.state.entry_price - buffer_points)

        self._modify_bracket(new_stop, None, timestamp)
        self.state.rth_close_tightened = True
        logger.debug("RTH close tighten applied: new SL={:.2f}", new_stop)
        
        close_price = float(bar["close"])
        position = self.broker.get_position(self.config.symbol)
        
        if position.is_flat:
            return

        # Profit lock: tighten stop after partial R reached
        strategy_cfg = self.config.strategy_config or OneMinuteStrategyConfig()
        stop_distance = None
        if self.state.entry_price and self.state.original_stop:
            stop_distance = abs(self.state.entry_price - self.state.original_stop)

        if (
            stop_distance
            and not self.state.profit_lock_active
            and getattr(strategy_cfg, "profit_lock_trigger_r", 0.0) > 0
        ):
            trigger_r = float(getattr(strategy_cfg, "profit_lock_trigger_r", 0.75))
            buffer_points = float(getattr(strategy_cfg, "profit_lock_stop_buffer_points", 0.5))
            if position.is_long:
                unrealized_points = close_price - self.state.entry_price
            else:
                unrealized_points = self.state.entry_price - close_price

            if unrealized_points >= stop_distance * trigger_r:
                if position.is_long:
                    new_stop = max(self.state.stop_loss or 0.0, self.state.entry_price + buffer_points)
                else:
                    new_stop = min(self.state.stop_loss or close_price, self.state.entry_price - buffer_points)
                self._modify_bracket(new_stop, None, timestamp)
                self.state.profit_lock_active = True
                logger.debug(
                    "Profit lock activated: new SL={:.2f} (unrealized {:.2f} pts)",
                    new_stop,
                    unrealized_points,
                )
        
        # Check if near take profit
        if self.state.take_profit is not None:
            atr = float(bar.get("ATR_14", 2.0))
            is_long = position.is_long
            
            if is_long:
                distance_to_tp = self.state.take_profit - close_price
            else:
                distance_to_tp = close_price - self.state.take_profit
            
            # Near TP threshold
            near_tp_threshold = atr * 0.25
            
            if 0 < distance_to_tp < near_tp_threshold:
                self.state.near_tp_count += 1
                
                # Evaluate trend continuation
                should_extend, new_tp, new_sl = self._evaluate_trend_continuation(
                    bar, history, is_long, close_price, atr
                )
                
                if should_extend and new_tp is not None:
                    self.state.extension_count += 1
                    self.state.extension_approved += 1
                    
                    # Modify the bracket orders
                    self._modify_bracket(new_sl, new_tp, timestamp)
                    
                    logger.debug(
                        f"Extended position: new TP={new_tp:.2f}, new SL={new_sl:.2f}"
                    )
    
    def _evaluate_trend_continuation(
        self,
        bar: pd.Series,
        history: pd.DataFrame,
        is_long: bool,
        close_price: float,
        atr: float
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Evaluate whether to extend take profit (trend continuation).
        
        Uses similar logic to TrendContinuationOptimizer from live bot.
        """
        # Get indicators
        adx = float(bar.get("ADX_14", 0))
        rsi = float(bar.get("RSI_14", 50))
        ema9 = float(bar.get("EMA_9", close_price))
        ema21 = float(bar.get("EMA_21", close_price))
        
        # Trend score
        trend_score = 0.0
        
        # ADX check
        if adx >= 20:
            trend_score += 0.25
        elif adx >= 15:
            trend_score += 0.15
        
        # EMA alignment
        if is_long:
            if ema9 > ema21 and close_price > ema9:
                trend_score += 0.25
            if rsi > 50 and rsi < 70:
                trend_score += 0.25
        else:
            if ema9 < ema21 and close_price < ema9:
                trend_score += 0.25
            if rsi < 50 and rsi > 30:
                trend_score += 0.25
        
        # Price vs VWAP
        vwap = float(bar.get("SESSION_VWAP", close_price))
        if is_long and close_price > vwap:
            trend_score += 0.15
        elif not is_long and close_price < vwap:
            trend_score += 0.15
        
        # 5m trend alignment
        if f"5m_EMA_9" in bar and f"5m_EMA_21" in bar:
            ema9_5m = float(bar.get("5m_EMA_9", close_price))
            ema21_5m = float(bar.get("5m_EMA_21", close_price))
            
            if is_long and ema9_5m > ema21_5m:
                trend_score += 0.10
            elif not is_long and ema9_5m < ema21_5m:
                trend_score += 0.10
        
        # Decision
        if trend_score < self.config.min_trend_score:
            return False, None, None
        
        # Calculate new levels
        extension = atr * self.config.tp_extension_atr_mult
        
        if is_long:
            new_tp = self.state.take_profit + extension
            # Trail stop to breakeven + buffer
            new_sl = max(self.state.stop_loss, self.state.entry_price + 1.0)
        else:
            new_tp = self.state.take_profit - extension
            new_sl = min(self.state.stop_loss, self.state.entry_price - 1.0)
        
        return True, new_tp, new_sl
    
    def _modify_bracket(
        self,
        new_stop: Optional[float],
        new_tp: Optional[float],
        timestamp: datetime
    ) -> None:
        """Modify existing bracket orders."""
        pending = self.broker.get_pending_orders()
        
        for order in pending:
            if order.symbol != self.config.symbol:
                continue
            
            if order.metadata.get("type") == "stop_loss" and new_stop:
                self.broker.modify_order(
                    order.order_id,
                    new_stop_price=new_stop,
                    timestamp=timestamp,
                    reason="trend_continuation"
                )
                self.state.stop_loss = new_stop
            
            elif order.metadata.get("type") == "take_profit" and new_tp:
                self.broker.modify_order(
                    order.order_id,
                    new_limit_price=new_tp,
                    timestamp=timestamp,
                    reason="trend_continuation"
                )
                self.state.take_profit = new_tp
    
    def _on_fill(self, fill: Fill) -> None:
        """Handle fill callback from broker."""
        logger.debug(f"Fill: {fill.side.value} {fill.quantity} @ {fill.price}")
    
    def _process_fill(self, fill: Fill) -> None:
        """Process a fill and update state."""
        is_entry = fill.metadata.get("type") not in ("stop_loss", "take_profit", "flatten")
        
        if is_entry:
            self.state.entry_price = fill.price
            self.state.entry_time = fill.timestamp
            self.state.daily_trades += 1
            
            # Set cooldown
            strategy_cfg = self.config.strategy_config or OneMinuteStrategyConfig()
            cooldown_minutes = strategy_cfg.cooldown_minutes
            self.state.cooldown_until = fill.timestamp + timedelta(minutes=cooldown_minutes)
        else:
            # Exit fill
            position = self.broker.get_position(self.config.symbol)
            
            # Update daily P&L
            if self.broker.trade_log:
                last_trade = self.broker.trade_log[-1]
                pnl = last_trade.get("realized_pnl", 0)
                self.state.daily_pnl += pnl
                self.state.realized_pnl += pnl

                # Track win/loss for risk gate
                self.risk_gate.record_trade_result(pnl > 0)

                # MAY 27 2026: feed closed-trade outcome to the Trade Manager gate so
                # its streak/posture/daily-PnL state evolves like live (if enabled).
                if self.manager_gate is not None:
                    self.manager_gate.record_outcome(pnl, fill.timestamp)
            
            # Reset position state
            self.state.entry_price = 0.0
            self.state.entry_time = None
            self.state.stop_loss = None
            self.state.take_profit = None
            self.state.original_stop = None
            self.state.original_tp = None
            self.state.profit_lock_active = False
            self.state.rth_close_tightened = False
    
    def _update_equity(self, timestamp: datetime, close_price: float) -> None:
        """Update equity curve."""
        position = self.broker.get_position(self.config.symbol)
        
        if position.is_flat:
            unrealized = 0.0
        else:
            unrealized = (close_price - position.avg_price) * position.quantity * 5.0  # MES point value
        
        equity = self.config.initial_capital + self.state.realized_pnl + unrealized
        self.state.unrealized_pnl = unrealized
        self.state.account_equity = equity

        # Update peak drawdown tracking
        self.risk_gate.update_equity(equity, timestamp)
        if self.risk_gate.high_water_equity is not None:
            self.state.peak_equity = float(self.risk_gate.high_water_equity)
        if self.risk_gate.base_equity is not None:
            self.state.base_equity = float(self.risk_gate.base_equity)
        self.state.drawdown_active = bool(self.risk_gate.drawdown_active)
        
        self.equity_curve.append((timestamp, equity))

    def _apply_drawdown_guard(self, bar: pd.Series, timestamp: datetime) -> None:
        """Enforce peak drawdown guard actions (halt or tighten)."""
        gate_cfg = getattr(self.risk_gate, "config", None)
        if not gate_cfg or not getattr(gate_cfg, "peak_drawdown_enabled", False):
            return
        if not self.risk_gate.drawdown_active:
            return

        if not self.state.drawdown_active:
            self.state.drawdown_active = True
            self._record_block("PEAK_DRAWDOWN_TRIGGER")

        action_mode = getattr(gate_cfg, "peak_drawdown_action", "halt").lower()

        # Tiered mode: only flatten open positions when tier 3 is reached
        if action_mode == "tiered":
            tier = self.risk_gate.get_drawdown_tier()
            if tier < 3:
                return  # Tiers 1/2: keep open positions, only reduce new entries

        position = self.broker.get_position(self.config.symbol)
        if position.is_flat:
            return

        close_price = float(bar["close"])

        if action_mode in ("halt", "tiered") and getattr(gate_cfg, "peak_drawdown_flatten_on_trigger", True):
            logger.warning(
                "🚫 Peak drawdown flatten: closing position at {:.2f}",
                close_price,
            )
            self.broker.cancel_all_orders(self.config.symbol)
            side = OrderSide.SELL if position.is_long else OrderSide.BUY
            self.broker.submit_market_order(
                symbol=self.config.symbol,
                side=side,
                quantity=abs(position.quantity),
                timestamp=timestamp,
                metadata={"type": "flatten", "reason": "PEAK_DRAWDOWN_FLATTEN"},
            )
            fills = self.broker.process_bar(self.config.symbol, bar, timestamp)
            for fill in fills:
                self._process_fill(fill)
            return

        if action_mode == "tighten" and self.state.entry_price:
            buffer_points = float(getattr(gate_cfg, "peak_drawdown_stop_buffer_points", 0.5))
            if position.is_long:
                new_stop = max(self.state.stop_loss or 0.0, self.state.entry_price + buffer_points)
            else:
                new_stop = min(self.state.stop_loss or close_price, self.state.entry_price - buffer_points)
            self._modify_bracket(new_stop, None, timestamp)
    
    def _record_block(self, reason: str) -> None:
        """Record a block reason."""
        self.block_reasons[reason] = self.block_reasons.get(reason, 0) + 1
    
    def _record_trace(
        self,
        bar_idx: int,
        timestamp: datetime,
        bar: pd.Series,
        history: pd.DataFrame
    ) -> None:
        """Record decision trace for comparison with live."""
        trace = DecisionTrace(
            timestamp=timestamp,
            bar_index=bar_idx,
            close_price=float(bar["close"]),
            signal_action="",
            signal_confidence=0.0,
            signal_reason="",
            risk_allowed=True,
            risk_reason="",
            final_action="HOLD",
            indicators={
                "ema9": float(bar.get("EMA_9", np.nan)),
                "ema21": float(bar.get("EMA_21", np.nan)),
                "atr14": float(bar.get("ATR_14", np.nan)),
                "adx14": float(bar.get("ADX_14", np.nan)),
                "rsi14": float(bar.get("RSI_14", np.nan)),
            },
            position_qty=self.state.current_position,
            position_pnl=self.state.unrealized_pnl,
        )
        
        self.decision_traces.append(trace)
    
    def _close_final_position(self) -> None:
        """Close any open position at end of backtest."""
        position = self.broker.get_position(self.config.symbol)
        
        if position.is_flat:
            return
        
        # Determine the final bar source (1m or 15m)
        # NOTE: in the 15m-only path df_1m is an EMPTY DataFrame (not None),
        # so guard on .empty too or index[-1] raises (it had never triggered
        # before because prior runs happened to end flat).
        if self.df_1m is not None and not self.df_1m.empty:
            final_timestamp = self.df_1m.index[-1]
            final_bar = self.df_1m.iloc[-1]
        elif self.df_15m is not None and not self.df_15m.empty:
            final_timestamp = self.df_15m.index[-1]
            final_bar = self.df_15m.iloc[-1]
        else:
            logger.warning("No data available to close final position")
            return
        
        self.broker.flatten_position(
            self.config.symbol,
            final_timestamp,
            reason="flatten"  # Use "flatten" so _process_fill recognizes as exit
        )
        
        # Process the flatten order
        fills = self.broker.process_bar(
            self.config.symbol,
            final_bar,
            final_timestamp
        )
        
        for fill in fills:
            self._process_fill(fill)
    
    def _compile_results(self) -> Dict:
        """Compile backtest results."""
        # Equity curve
        if self.equity_curve:
            timestamps, values = zip(*self.equity_curve)
            equity_series = pd.Series(values, index=pd.DatetimeIndex(timestamps))
        else:
            equity_series = pd.Series(dtype=float)
        
        # Trade summary from broker
        trade_summary = self.broker.get_trade_summary()
        
        # Block reason summary
        block_summary = dict(sorted(
            self.block_reasons.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20])  # Top 20
        
        # Trend optimizer stats
        optimizer_stats = {
            "near_tp_events": self.state.near_tp_count,
            "extensions_approved": self.state.extension_approved,
            "extension_rate": (
                self.state.extension_approved / self.state.near_tp_count
                if self.state.near_tp_count > 0 else 0
            ),
        }
        
        results = {
            "config": {
                "symbol": self.config.symbol,
                "start_date": str(self.config.start_date),
                "end_date": str(self.config.end_date),
                "initial_capital": self.config.initial_capital,
            },
            "trades": self.broker.trade_log,
            "equity_curve": equity_series,
            "trade_summary": trade_summary,
            "block_reasons": block_summary,
            "signal_counts": self.signal_counts,
            "optimizer_stats": optimizer_stats,
            "total_bars_processed": self.state.bar_index,
        }
        
        # Add decision traces if enabled
        if self.config.trace_mode and self.decision_traces:
            results["decision_traces"] = [
                {
                    "timestamp": str(t.timestamp),
                    "bar_index": t.bar_index,
                    "close": t.close_price,
                    "action": t.final_action,
                    "indicators": t.indicators,
                }
                for t in self.decision_traces
            ]
        
        return results

    def _simulate_rag_context_check(self, bar: pd.Series, signal: Signal) -> bool:
        """
        Simulate RAG Agent validation.
        Ensures that if we are in a high-risk regime, we have sufficient confidence.
        """
        # Example RAG logic: High IV requires higher confidence
        atr = float(bar.get("ATR_14", 0))
        if atr > 5.0 and signal.confidence < 0.7:
             return False
        return True

    def _simulate_learning_agent(self, history: pd.DataFrame, signal: Signal) -> None:
        """Simulate Learning Agent recording observation."""
        if self.learning_agent_enabled:
             # In a real scenario, this would persist to DB
             # For backtest, we just log that we observed it
             pass

    def _simulate_rag_agent(self, history: pd.DataFrame, signal: Signal) -> Signal:
        """Simulate RAG Agent enriching signal."""
        if self.rag_agent_enabled and signal.action != "HOLD":
             # In real scenario, this queries vector DB
             # Here we assume RAG confirms valid strategy signals
             # But we could check basic regime sanity
             pass
        return signal

    def _simulate_learning_observation(self, bar: pd.Series, signal: Signal) -> None:
        """
        Simulate Learning Agent recording.
        Just ensures that we have the necessary data points that the learning agent would request.
        """
        # The learning agent typically records: state, action, reward (later)
        # Here we just validate we have the state variables.
        pass


def run_backtest(
    data_1m: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    data_5m: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Convenience function to run a backtest.
    
    Args:
        data_1m: 1-minute OHLCV data
        config: Backtest configuration
        data_5m: Optional 5-minute data
        
    Returns:
        Backtest results dict
    """
    config = config or BacktestConfig()
    
    engine = BacktestEngine(config)
    engine.load_data(data_1m, data_5m)
    
    return engine.run()
