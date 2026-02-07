"""Application configuration and settings management."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
from datetime import time


@dataclass
class DataSourceConfig:
    tradingview_webhook_url: Optional[str] = field(default_factory=lambda: os.environ.get("TRADINGVIEW_WEBHOOK_URL"))
    tradingview_symbol: str = field(default_factory=lambda: os.environ.get("TRADINGVIEW_SYMBOL", "MES"))
    tradingview_interval: str = field(default_factory=lambda: os.environ.get("TRADINGVIEW_INTERVAL", "1m"))

    ibkr_host: str = field(default_factory=lambda: os.environ.get("IBKR_HOST", "127.0.0.1"))
    ibkr_port: int = field(default_factory=lambda: int(os.environ.get("IBKR_PORT", "4002")))
    ibkr_client_id: int = field(default_factory=lambda: int(os.environ.get("IBKR_CLIENT_ID", "1")))
    ibkr_symbol: str = field(default_factory=lambda: os.environ.get("IBKR_SYMBOL", "MES"))
    ibkr_exchange: str = field(default_factory=lambda: os.environ.get("IBKR_EXCHANGE", "CME"))
    ibkr_currency: str = field(default_factory=lambda: os.environ.get("IBKR_CURRENCY", "USD"))

    twitter_bearer_token: Optional[str] = field(default_factory=lambda: os.environ.get("TWITTER_BEARER_TOKEN"))
    news_api_keys: List[str] = field(default_factory=lambda: os.environ.get("NEWS_API_KEYS", "").split(",") if os.environ.get("NEWS_API_KEYS") else [])
    sentiment_refresh_interval: int = 60  # seconds


@dataclass
class EntryFilterConfig:
    """Configuration for optional entry filters used by TradingFilters."""
    wait_for_candle_close: bool = True
    require_trend_alignment: bool = True
    allow_counter_trend: bool = False
    ema_fast_period: int = 9
    ema_slow_period: int = 20
    ema_alignment_tolerance_pct: float = 0.0002  # 0.02% of price
    counter_trend_penalty: float = 0.10
    # ADX trend strength filter (Jan 2026)
    require_adx_confirmation: bool = True
    min_adx_threshold: float = 15.0  # Minimum ADX for trend trades
    min_adx_threshold_low_volume: Optional[float] = None  # Optional override for evening/overnight
    adx_period: int = 14
    # ATR volatility filters
    atr_period: int = 14
    min_atr_threshold: float = 0.5
    min_atr_percentile: Optional[float] = None  # e.g. 10 = 10th percentile
    atr_percentile_lookback: int = 120
    low_atr_penalty_mode: bool = False
    low_atr_penalty: float = 0.10
    max_atr_threshold: float = 5.0
    chop_zone_buffer_pct: float = 0.25
    sr_proximity_ticks: int = 8
    candle_period_seconds: int = 60


@dataclass
class OneMinuteStrategyConfig:
    """Configuration for the MES 1-minute close strategy."""

    enabled: bool = True
    warmup_bars: int = 800  # >= 720 (12 hours) per user request to stabilize indicators
    use_eth_session: bool = False  # False = RTH VWAP reset, True = ETH
    breakout_enabled: bool = False
    breakout_strength_filter: bool = True
    pullback_lookback: int = 3
    require_pullback_confirmation: bool = False
    # JAN 11 2026 FIX: Relaxed ATR/candle filters - were blocking 99%+ of bars
    # Old values: 0.10, 0.90, 0.4 - too strict for 1-min data
    atr_percentile_low: float = 0.05   # Was 0.10 - allow more low-volatility periods
    atr_percentile_high: float = 0.95  # Was 0.90 - allow more high-volatility periods  
    tiny_candle_atr_factor: float = 0.15  # Was 0.4 - 1-min bars often have small ranges
    atr_bounds_blocking: bool = True
    tiny_candle_blocking: bool = True
    trend_close_tolerance_pct: float = 0.0002
    cooldown_minutes: int = 3
    max_trades_per_hour: int = 3
    max_trades_per_day: int = 8
    # JAN 11 2026 FIX: Increased stop multiplier to pass RiskGate min_stop_points=6
    # JAN 18 2026 UPDATE: Tuning for RTH Volatility (Backtest Verified)
    # Lowered stop multiplier to 3.0 (from 6.5) to catch normal RTH moves while staying under new 50pt cap
    stop_atr_multiplier: float = 3.0  # Was 6.5. Optimized for RTH trading.
    take_profit_multiple: float = 1.0  # Was 2.0. Quick scalps in RTH.
    trailing_atr_multiple: float = 1.0
    profit_lock_trigger_r: float = 0.75  # Tighten stop after 0.75R in favor
    profit_lock_stop_buffer_points: float = 0.5  # Buffer beyond breakeven when locking profits
    breakout_adx_threshold: float = 18.0
    trend_adx_threshold: float = 18.0 # Was 20.0. Lower threshold to enter trends earlier.
    allow_runner: bool = False
    runner_take_profit_multiple: float = 1.0
    dry_run: bool = False
    allow_add_on: bool = False
    trend_flip_exit: bool = True
    breakout_use_or_levels: bool = False
    window_bars: int = 400
    # JAN 8 2026: Multi-timeframe trend confirmation
    require_5m_trend_alignment: bool = True  # Check 5-min trend before 1-min entry
    mtf_gate_enabled: bool = True  # Enable 15m/30m MTF gate
    mtf_ema_period: int = 20  # EMA period for 5-min trend calculation
    
    # JAN 11 2026: Enhanced MTF structure - 15m regime, 5m setup, 1m execution
    # 15m determines market regime/bias (trend vs chop, direction)
    # 5m confirms setup (pullback structure, support/resistance)
    # 1m provides precise entry trigger
    use_mtf_regime: bool = True  # Enable full MTF structure
    regime_timeframe: str = "15m"  # Timeframe for regime determination
    regime_adx_threshold: float = 20.0  # ADX threshold on regime timeframe
    regime_ema_fast: int = 9
    # JAN 18 2026: Relaxed filters to increase RTH trade frequency per user request
    # Previous settings (Require Trend + High ATR) were too restrictive (~1 trade/day)
    require_regime_trend: bool = False  # Allow trading in 15m ranging markets (uses 1m trend)
    
    # JAN 11 2026: Session gating - only trade RTH to avoid bad overnight data
    # IB 1-min data has 60%+ ZERO_RANGE bars overnight (no real trading)
    rth_only: bool = True  # Only trade during RTH hours
    rth_start_hour: int = 9   # RTH start hour (9:30 AM ET)
    rth_start_minute: int = 30
    rth_end_hour: int = 16    # RTH end hour (4:00 PM ET)
    rth_end_minute: int = 0
    
    # JAN 11 2026: Volume filter - skip bars with no real trading activity
    # IB ZERO_RANGE bars have volume=0 or 1, real bars have 1000+ volume
    min_bar_volume: int = 10  # Minimum volume to consider bar valid for entry
    
    # JAN 11 2026: Max hold time - prevent 4.7 hour holds on 1-min signals
    max_hold_minutes: int = 90  # Exit if trade exceeds this duration
    
    # JAN 11 2026: Disable trend extensions until sample size is meaningful
    enable_trend_extensions: bool = False  # Was causing 0% win rate on extensions
    
    # JAN 18 2026: Disabled High ATR requirement to allow normal RTH trading
    # Backtest showed: High ATR (top 33%) is safer, but user wants more frequency
    # We rely on risk management (stops) to handle lower volatility periods
    require_high_atr: bool = False  # Was True. Disabled to increase valid RTH trades.
    high_atr_percentile: float = 0.20  # Lowered from 0.67 to 0.20 (unused if require_high_atr=False)
    high_atr_lookback: int = 200  # Bars to use for ATR percentile calculation

    # RTH session-specific overrides
    rth_min_bar_volume: Optional[int] = None
    rth_trend_adx_threshold: Optional[float] = None
    rth_stop_atr_multiplier: Optional[float] = None
    rth_take_profit_multiple: Optional[float] = None
    rth_require_high_atr: Optional[bool] = None
    rth_high_atr_percentile: Optional[float] = None
    rth_rsi_long_min: Optional[float] = None
    rth_rsi_long_max: Optional[float] = None
    rth_rsi_short_min: Optional[float] = None
    rth_rsi_short_max: Optional[float] = None

    # RTH open (first N minutes) mean-reversion/momentum capture
    rth_open_minutes: int = 90
    rth_open_mean_reversion_enabled: bool = False
    rth_open_rsi_oversold: float = 35.0
    rth_open_rsi_overbought: float = 65.0
    rth_open_vwap_atr_mult: float = 0.40
    rth_open_max_adx: Optional[float] = 18.0
    rth_open_min_volume: Optional[int] = None
    rth_require_pullback_confirmation: Optional[bool] = None
    rth_pullback_adx_bypass: Optional[float] = 22.0

    # RTH close protection
    rth_close_tighten_minutes: int = 20
    rth_close_stop_buffer_points: float = 0.5

    # RTH midday no-trade window (optional)
    rth_no_trade_start_hour: Optional[int] = None
    rth_no_trade_start_minute: Optional[int] = None
    rth_no_trade_end_hour: Optional[int] = None
    rth_no_trade_end_minute: Optional[int] = None
    
    # JAN 11 2026: Overnight/Evening session support with 30m timeframe
    # 1m data is noisy overnight - use 30m for cleaner signals
    # Overnight = 6:00 PM - 9:30 AM ET (ES futures globex session)
    # NOTE: Disabled until we have matching 1m+30m data with same date range
    # Native 30m data quality: RTH 2.7% zero-range, Overnight 8.2% (vs 60%+ resampled)
    allow_overnight_trading: bool = False  # DISABLED - need matching data
    overnight_timeframe: str = "30m"  # Use 30m bars for overnight decisions
    overnight_start_hour: int = 18  # Overnight starts 6:00 PM ET
    overnight_end_hour: int = 9     # Overnight ends before RTH (9:30 AM)
    overnight_end_minute: int = 30
    overnight_min_volume: int = 100  # Stricter volume threshold for overnight
    overnight_adx_threshold: float = 30.0  # Much stronger trend required overnight
    overnight_atr_percentile: float = 0.80  # Top 20% volatility only overnight

    # FEB 2026: Scoring-based entry system parameters
    # Replaces hard filters with weighted scoring for increased trade frequency
    use_scoring_system: bool = False  # Enable scoring-based entry (experimental)
    scoring_full_size_threshold: float = 60.0  # Score >= 60 → full position
    scoring_half_size_threshold: float = 45.0  # Score >= 45 → half position
    # Score < 45 → no trade
    # Risk gates (max loss, daily loss, open risk) remain HARD regardless of score


@dataclass
class ThirtyMinuteStrategyConfig:
    """Configuration for 30-minute timeframe strategy.
    
    JAN 11 2026: Designed specifically for 30m bars based on data analysis:
    - ATR: 6-8 pts (vs 1-2 pts for 1m)
    - ADX: 47.6 mean, 91.5% > 20 (stronger trends than 1m)
    - Overnight: 77% of 30m data is overnight/globex session
    - Bar range: ~6-7 pts typical
    
    Key differences from 1m strategy:
    1. Wider stops (ATR-based, 1-1.5x ATR = 6-12 pts)
    2. Higher ADX threshold (trends are clearer on 30m)
    3. Longer hold times (4-8 hours typical)
    4. Focus on trend continuation, not scalping
    """
    
    enabled: bool = True
    warmup_bars: int = 50  # 50 bars * 30min = 25 hours warmup
    
    # EMA settings - standard periods work well on 30m
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50  # Longer EMA for trend bias
    
    # ATR settings - 30m ATR is 6-8 pts typical
    atr_period: int = 14
    stop_atr_multiplier: float = 1.0  # 1x ATR = ~6-8 pts stop
    take_profit_multiplier: float = 2.0  # 2R target
    max_stop_points: float = 15.0  # Hard cap on stop distance
    min_stop_points: float = 4.0  # Minimum stop distance
    
    # ADX settings - 30m shows strong trends (mean 47.6)
    adx_period: int = 14
    adx_trend_threshold: float = 30.0  # Raised from 25 - need stronger trend
    adx_strong_trend: float = 40.0  # Strong trend for aggressive entries
    
    # RSI settings for overbought/oversold
    rsi_period: int = 14
    rsi_oversold: float = 35.0  # Long entry zone
    rsi_overbought: float = 65.0  # Short entry zone
    
    # Signal types enabled
    enable_ema_reclaim: bool = False  # Disabled - 28% WR, -$176 in backtest
    enable_trend_continuation: bool = True  # 38% WR, +$76 - the profitable signal
    enable_breakout: bool = False  # Disabled - 30m breakouts are risky
    
    # Pullback parameters
    pullback_bars: int = 3  # Bars to look back for pullback
    min_pullback_pct: float = 0.3  # Min pullback as % of recent swing
    
    # Trade management
    max_trades_per_day: int = 4  # Fewer trades on 30m
    cooldown_bars: int = 2  # 2 bars = 1 hour cooldown
    max_hold_bars: int = 16  # 16 bars = 8 hours max hold
    
    # Session settings - 30m can trade overnight
    trade_all_sessions: bool = True  # Trade RTH + Globex
    avoid_first_bar: bool = True  # Skip first bar of session (gap risk)
    avoid_last_bar: bool = True  # Skip last bar before close
    
    # Risk settings
    require_trend_alignment: bool = True  # EMA9 > EMA21 for longs
    require_higher_tf_trend: bool = False  # No higher TF for 30m
    
    # Filter settings
    min_bar_range: float = 1.0  # Skip bars with range < 1 pt
    skip_zero_range: bool = True  # Skip zero-range bars
    
    # Confidence thresholds
    min_entry_confidence: float = 0.60  # Minimum confidence to enter
    high_confidence_threshold: float = 0.75  # High confidence for larger size


@dataclass
class RiskGateConfig:
    """Hard risk/margin gate for MES."""

    # JAN 18 2026: Updated RiskGate limits to support RTH Volatility Strategy
    # Increased caps to allow wider stops (up to 50 pts) for volatile sessions
    risk_contracts_env: str = os.environ.get("MAX_MES_CONTRACTS", "1")
    max_contracts: int = field(default_factory=lambda: int(os.environ.get("MAX_MES_CONTRACTS", "1")))
    
    # Base risk per trade raised to $250 to accommodate 50pt stops (1 contract * $5 * 50pts)
    risk_per_trade_usd: float = field(default_factory=lambda: float(os.environ.get("RISK_PER_TRADE_USD", "250")))
    risk_per_trade_min: float = 25.0
    risk_per_trade_max: float = 250.0 # Increased from 75 to 250
    min_stop_points: float = 3.0      # Slightly increased floor
    
    # Daily loss raised to $750 (3 max loss trades) to prevent instant lockout
    daily_max_loss_usd: float = field(default_factory=lambda: float(os.environ.get("DAILY_MAX_LOSS_USD", "750")))
    margin_buffer_usd: float = field(default_factory=lambda: float(os.environ.get("MARGIN_BUFFER_USD", "1000")))
    initial_margin_long: float = field(default_factory=lambda: float(os.environ.get("MES_INITIAL_MARGIN_LONG", "2464")))
    initial_margin_short: float = field(default_factory=lambda: float(os.environ.get("MES_INITIAL_MARGIN_SHORT", "2305.6")))
    avoid_close_window_minutes: int = field(default_factory=lambda: int(os.environ.get("AVOID_CLOSE_WINDOW_MINUTES", "20")))
    avoid_close_enabled: bool = field(
        default_factory=lambda: os.environ.get("AVOID_CLOSE_WINDOW_ENABLED", "true").lower() not in {"0", "false", "no"}
    )
    intraday_close_time: time = time(16, 0)  # CME close (4 PM CT)
    maintenance_start: time = time(16, 0)  # CME maintenance start CT
    maintenance_end: time = time(17, 0)    # CME maintenance end CT
    tick_size: float = 0.25
    max_stop_points: float = 50.0  # Increased from 12.0 to 50.0 for RTH volatility
    max_consecutive_losses: int = 3
    peak_drawdown_enabled: bool = False
    peak_drawdown_pct: float = 4.0
    peak_drawdown_action: str = "halt"  # "halt" or "tighten"
    peak_drawdown_tighten_multiplier: float = 0.5
    peak_drawdown_stop_buffer_points: float = 0.5
    peak_drawdown_flatten_on_trigger: bool = True
    peak_drawdown_reset_on_new_day: bool = False

    def bounded_risk_usd(self) -> float:
        """Clamp risk-per-trade to a safe range."""
        raw = self.risk_per_trade_usd
        return min(self.risk_per_trade_max, max(self.risk_per_trade_min, raw))


@dataclass
class TradingConfig:
    max_position_size: int = field(default_factory=lambda: int(os.environ.get("MAX_POSITION_SIZE", "5")))
    contracts_per_order: int = 1
    max_daily_loss: float = field(default_factory=lambda: float(os.environ.get("MAX_DAILY_LOSS", "2000.0")))
    max_loss_per_trade: float = field(default_factory=lambda: float(os.environ.get("MAX_LOSS_PER_TRADE", "1250.0")))
    max_daily_trades: int = 20
    initial_capital: float = field(default_factory=lambda: float(os.environ.get("INITIAL_CAPITAL", "100000.0")))
    contract_multiplier: float = 50.0
    stop_loss_ticks: float = 10.0
    take_profit_ticks: float = 20.0
    tick_size: float = 0.25
    tick_value: float = 12.5
    commission_per_contract: float = 2.4
    
    # Position sizing method: "fixed_fraction" or "kelly"
    position_sizing_method: str = "fixed_fraction"
    # Risk percentage per trade for fixed fractional sizing (0.005 = 0.5%, 0.01 = 1%)
    risk_per_trade_pct: float = 0.005
    
    # Safety parameters
    disaster_stop_pct: float = 0.007
    max_trade_duration_minutes: int = 60
    trade_cooldown_minutes: int = 5
    order_lock_timeout_seconds: int = 300
    pending_order_timeout_seconds: int = 180
    enable_ib_api_debug_log: bool = field(
        default_factory=lambda: os.environ.get("IB_API_DEBUG_LOG", "False").lower() in {"1", "true", "yes"}
    )
    ib_api_log_path: str = field(
        default_factory=lambda: os.environ.get("IB_API_LOG_PATH", "logs/ib_api_wire.log")
    )
    reset_state_on_start: bool = False
    
    # Margin safety (used by PositionManager)
    initial_margin_long: float = field(default_factory=lambda: float(os.environ.get("MES_INITIAL_MARGIN_LONG", "2464")))
    initial_margin_short: float = field(default_factory=lambda: float(os.environ.get("MES_INITIAL_MARGIN_SHORT", "2305.6")))

    def get_validated_cooldown(self, min_val: int = 1, max_val: int = 60) -> int:
        """Return cooldown bounded to [min_val, max_val], log warning if out of bounds."""
        val = getattr(self, 'trade_cooldown_minutes', 5)
        if val < min_val:
            import logging
            logging.warning(f"trade_cooldown_minutes {val} too low; using {min_val}")
            return min_val
        if val > max_val:
            import logging
            logging.warning(f"trade_cooldown_minutes {val} too high; using {max_val}")
            return max_val
        if val > 30:
            import logging
            logging.warning(f"trade_cooldown_minutes unusually high: {val}")
        return val
    
    # Indicator warm-up
    min_bars_for_signals: int = 200
    
    # Market regime filter thresholds
    min_atr_threshold: float = 0.5
    max_spread_ticks: int = 1
    max_loop_latency_seconds: float = 3.0
    
    # Weighted voting thresholds
    min_weighted_confidence: float = 0.70
    confidence_threshold: float = field(default_factory=lambda: float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7")))
    min_confidence_for_trade: float = 0.60
    min_stop_distance_ticks: int = 4

    # Startup entry gating (prevents immediate post-restart entries)
    # 0 disables. These gates should only affect NEW entries; exits remain allowed.
    startup_grace_period_seconds: int = 0
    startup_min_completed_bars: int = 0

    # Hard Safety Constraints
    max_contracts_limit: int = field(default_factory=lambda: int(os.environ.get("MAX_CONTRACTS", "5")))
    margin_limit_pct: float = field(default_factory=lambda: float(os.environ.get("MARGIN_LIMIT_PCT", "0.80")))
    decision_min_interval_seconds: int = field(default_factory=lambda: int(os.environ.get("DECISION_MIN_INTERVAL_SECONDS", "30")))
    order_retry_limit: int = field(default_factory=lambda: int(os.environ.get("ORDER_RETRY_LIMIT", "3")))
    contract_month_offset: int = field(default_factory=lambda: int(os.environ.get("CONTRACT_MONTH_OFFSET", "0")))
    allow_naked_orders: bool = field(
        default_factory=lambda: os.environ.get("ALLOW_NAKED_ORDERS", "False").lower()
        in {"1", "true", "yes"}
    )
    enforce_market_hours: bool = field(
        default_factory=lambda: os.environ.get("ENFORCE_MARKET_HOURS", "True").lower()
        in {"1", "true", "yes"}
    )
    cancel_orders_on_startup: bool = field(
        default_factory=lambda: os.environ.get("CANCEL_ORDERS_ON_STARTUP", "False").lower()
        in {"1", "true", "yes"}
    )
    
    # Optional entry filter tuning
    entry_filters: EntryFilterConfig = field(default_factory=EntryFilterConfig)


@dataclass
class BacktestConfig:
    data_path: Path = Path("data/historical_spy_es.parquet")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 100_000.0
    slippage: float = 0.25
    risk_free_rate: float = 0.02


@dataclass
class OptimizationConfig:
    window_length: int = 5000
    retrain_interval: int = 60 * 60
    parameter_grid: dict = field(default_factory=lambda: {
        "rsi_period": [14, 21, 28],
        "macd_fast": [12, 16, 20],
        "macd_slow": [26, 30, 35],
        "sentiment_threshold": [0.4, 0.5, 0.6],
        "momentum_lookback": [20, 30, 40],
    })
    enable_daily_optimization: bool = True
    optimized_params_path: str = "data/optimized_params.json"
    optimization_hour: int = 16


@dataclass
class StrategyConfig:
    name: str = "rsi_macd_sentiment"
    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for AWS Bedrock LLM integration."""
    enabled: bool = field(default_factory=lambda: os.environ.get("LLM_ENABLED", "False").lower() == "true")
    model_id: str = field(default_factory=lambda: os.environ.get("LLM_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"))
    region_name: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    max_tokens: int = 2048
    temperature: float = 0.3
    min_confidence_threshold: float = 0.7
    override_mode: bool = False
    call_interval_seconds: int = 60
    enable_sentiment: bool = True
    sentiment_region: str = "us-east-1"
    s3_bucket: str = field(default_factory=lambda: os.environ.get("S3_BUCKET", ""))
    s3_prefix: str = "llm-training-data"
    retrain_interval_days: int = 7
    min_training_trades: int = 100
    trade_log_db_path: str = "data/llm_trades.db"
    
    use_background_thread: bool = True
    cache_timeout_seconds: int = 300


@dataclass
class RAGConfig:
    """Configuration for Retrieval-Augmented Generation (RAG)."""
    enabled: bool = field(default_factory=lambda: os.environ.get("RAG_ENABLED", "False").lower() == "true")
    backend: str = field(default_factory=lambda: os.environ.get("RAG_BACKEND", "local_faiss").lower())
    opensearch_enabled: bool = field(default_factory=lambda: os.environ.get("OPENSEARCH_ENABLED", "False").lower() in {"1", "true", "yes"})
    embedding_model_id: str = "amazon.titan-embed-text-v1"
    region_name: str = "us-east-1"
    vector_store_path: str = "data/rag_index"
    embedding_dimension: int = 1536
    top_k_results: int = 3
    score_threshold: float = 0.5
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    batch_size: int = 10
    knowledge_base_path: str = "data/knowledge_base"
    local_store_path: str = "rag_data/local_kb/local_kb.sqlite"
    kb_cache_ttl_seconds: int = 120
    min_similar_trades: int = field(default_factory=lambda: int(os.environ.get("MIN_SIMILAR_TRADES", "2")))
    min_win_rate: float = field(default_factory=lambda: float(os.environ.get("MIN_WIN_RATE", "0.15")))
    min_weighted_win_rate: float = field(default_factory=lambda: float(os.environ.get("MIN_WEIGHTED_WIN_RATE", "0.45")))
    min_weighted_win_rate_soft_floor: float = field(
        default_factory=lambda: float(
            os.environ.get(
                "MIN_WEIGHTED_WIN_RATE_SOFT_FLOOR",
                os.environ.get("MIN_WEIGHTED_WIN_RATE", "0.45"),
            )
        )
    )
    min_similar_trades_for_full_threshold: int = field(
        default_factory=lambda: int(os.environ.get("MIN_SIMILAR_TRADES_FOR_FULL_THRESHOLD", "0"))
    )
    min_sample_for_hard_block: int = field(
        default_factory=lambda: int(os.environ.get("RAG_MIN_SAMPLE_FOR_HARD_BLOCK", "30"))
    )
    soft_penalty_when_below: float = field(
        default_factory=lambda: float(os.environ.get("RAG_SOFT_PENALTY_WHEN_BELOW", "0.10"))
    )
    hard_block_when_below: bool = field(
        default_factory=lambda: os.environ.get("RAG_HARD_BLOCK_WHEN_BELOW", "False").lower() in {"1", "true", "yes"}
    )
    regime_mode: str = field(default_factory=lambda: os.environ.get("RAG_REGIME_MODE", "relaxed"))

    def __post_init__(self) -> None:
        self.backend = (self.backend or "off").lower()
        if self.min_weighted_win_rate_soft_floor > self.min_weighted_win_rate:
            self.min_weighted_win_rate_soft_floor = self.min_weighted_win_rate


@dataclass
class StockwitsSentimentConfig:
    """Configuration for Stocktwits sentiment integration (legacy single-source)."""
    enabled: bool = field(
        default_factory=lambda: os.environ.get("STOCKTWITS_SENTIMENT_ENABLED", "True").lower() in {"1", "true", "yes"}
    )
    # Symbols to fetch sentiment for
    symbols: List[str] = field(default_factory=lambda: ["ES_F", "SPY"])
    # Cache refresh interval (minimum time between API calls)
    refresh_interval_seconds: int = field(
        default_factory=lambda: int(os.environ.get("STOCKTWITS_REFRESH_INTERVAL_SECONDS", "300"))
    )
    # HTTP request timeout
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("STOCKTWITS_REQUEST_TIMEOUT", "3.0"))
    )
    # ============================================================
    # RTH (Regular Trading Hours) thresholds - 8:30 AM - 3:00 PM CT
    # Higher liquidity = more relaxed thresholds
    # ============================================================
    # Sentiment threshold to block contradictory entries
    # e.g., block long if sentiment < -0.4, block short if sentiment > 0.4
    entry_block_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_ENTRY_BLOCK_THRESHOLD", "0.4"))
    )
    # Weaker threshold for reducing confidence (but not blocking)
    weak_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_WEAK_THRESHOLD", "0.2"))
    )
    # Threshold to trigger protective actions for existing positions
    protect_position_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_PROTECT_POSITION_THRESHOLD", "0.6"))
    )
    # ============================================================
    # Low Volume Session thresholds (Evening & Overnight)
    # Lower liquidity = STRICTER thresholds to avoid bad fills
    # Evening: 5:00 PM - 11:00 PM CT
    # Overnight: 11:00 PM - 3:00 AM CT
    # ============================================================
    low_volume_entry_block_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_LOW_VOLUME_ENTRY_BLOCK_THRESHOLD", "0.25"))
    )
    low_volume_weak_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_LOW_VOLUME_WEAK_THRESHOLD", "0.10"))
    )
    low_volume_protect_position_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_LOW_VOLUME_PROTECT_POSITION_THRESHOLD", "0.40"))
    )
    # ============================================================
    # Confidence modifiers
    # ============================================================
    # Confidence modifier when sentiment is mildly contradictory
    mild_contradiction_confidence_mult: float = 0.8  # 20% reduction
    # Confidence boost when sentiment agrees with signal
    agreement_confidence_mult: float = 1.1  # 10% boost
    # Additional penalty for low volume sessions (stacks with above)
    low_volume_confidence_penalty: float = 0.9  # Extra 10% reduction in low volume


@dataclass
class MultiSourceSentimentConfig:
    """Configuration for multi-source sentiment aggregation (Stocktwits + Reddit + Twitter).
    
    This replaces StockwitsSentimentConfig with a more comprehensive multi-source approach.
    """
    enabled: bool = field(
        default_factory=lambda: os.environ.get("MULTI_SOURCE_SENTIMENT_ENABLED", "True").lower() in {"1", "true", "yes"}
    )
    
    # ============================================================
    # Source weights (must sum to 1.0)
    # ============================================================
    stocktwits_weight: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_STOCKTWITS_WEIGHT", "0.4"))
    )
    reddit_weight: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_REDDIT_WEIGHT", "0.3"))
    )
    twitter_weight: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_TWITTER_WEIGHT", "0.3"))
    )
    
    # ============================================================
    # API Configuration
    # ============================================================
    refresh_interval_seconds: int = field(
        default_factory=lambda: int(os.environ.get("SENTIMENT_REFRESH_INTERVAL_SECONDS", "300"))
    )
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_REQUEST_TIMEOUT", "3.0"))
    )
    
    # Stocktwits symbols
    stocktwits_symbols: List[str] = field(default_factory=lambda: ["ES_F", "SPY"])
    
    # Reddit configuration
    reddit_subreddits: List[str] = field(
        default_factory=lambda: ["wallstreetbets", "stocks", "options"]
    )
    reddit_search_terms: List[str] = field(
        default_factory=lambda: ["SPY", "SPX", "ES", "S&P 500", "futures"]
    )
    
    # Twitter search terms
    twitter_search_terms: List[str] = field(
        default_factory=lambda: ["$SPY", "SPX", "ES futures", "S&P 500"]
    )
    
    # ============================================================
    # RTH Thresholds (Regular Trading Hours: 8:30 AM - 3:00 PM CT)
    # ============================================================
    entry_block_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_ENTRY_BLOCK_THRESHOLD", "0.4"))
    )
    weak_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_WEAK_THRESHOLD", "0.2"))
    )
    strong_exit_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_STRONG_EXIT_THRESHOLD", "0.6"))
    )
    
    # ============================================================
    # Low Volume Session Thresholds (Evening & Overnight)
    # ============================================================
    low_volume_entry_block_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_LOW_VOLUME_ENTRY_BLOCK_THRESHOLD", "0.25"))
    )
    low_volume_weak_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SENTIMENT_LOW_VOLUME_WEAK_THRESHOLD", "0.10"))
    )
    
    # ============================================================
    # Confidence modifiers
    # ============================================================
    mild_contradiction_confidence_mult: float = 0.7  # 30% reduction for reduce_size
    agreement_confidence_mult: float = 1.1  # 10% boost
    low_volume_confidence_penalty: float = 0.9  # Extra 10% penalty


@dataclass
class VixFeedThresholds:
    """Thresholds for VX-based volatility multiplier."""
    extreme: float = 30.0  # VX >= 30: 0.4x multiplier
    elevated: float = 20.0  # VX >= 20: 0.7x multiplier


@dataclass
class VixFeedConfig:
    """Configuration for VX Futures Feed from IBKR.
    
    Provides real-time VIX futures data for volatility-based position sizing.
    When VIX is elevated, position sizes are automatically reduced.
    """
    enabled: bool = field(
        default_factory=lambda: os.environ.get("VIX_FEED_ENABLED", "True").lower() in {"1", "true", "yes"}
    )
    
    # IBKR Connection settings
    ib_host: str = field(
        default_factory=lambda: os.environ.get("VIX_FEED_IB_HOST", "127.0.0.1")
    )
    ib_port: int = field(
        default_factory=lambda: int(os.environ.get("VIX_FEED_IB_PORT", "7497"))
    )
    client_id: int = field(
        default_factory=lambda: int(os.environ.get("VIX_FEED_CLIENT_ID", "71"))
    )
    market_data_type: int = field(
        default_factory=lambda: int(os.environ.get("VIX_FEED_MARKET_DATA_TYPE", "1"))  # 1=live, 3=delayed
    )
    
    # Stale data handling
    stale_seconds: int = field(
        default_factory=lambda: int(os.environ.get("VIX_FEED_STALE_SECONDS", "120"))
    )
    conservative_on_stale: bool = field(
        default_factory=lambda: os.environ.get("VIX_FEED_CONSERVATIVE_ON_STALE", "False").lower() in {"1", "true", "yes"}
    )
    
    # Volatility multiplier thresholds
    thresholds: VixFeedThresholds = field(default_factory=VixFeedThresholds)
    
    # Reconnection settings
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("VIX_FEED_MAX_RETRIES", "5"))
    )
    base_delay: float = field(
        default_factory=lambda: float(os.environ.get("VIX_FEED_BASE_DELAY", "1.0"))
    )
    max_delay: float = field(
        default_factory=lambda: float(os.environ.get("VIX_FEED_MAX_DELAY", "60.0"))
    )


@dataclass
class TelegramConfig:
    """Configuration for Telegram notifications."""
    enabled: bool = field(default_factory=lambda: os.environ.get("TELEGRAM_ENABLED", "False").lower() == "true")
    bot_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""))
    notify_on_trade: bool = True
    notify_on_signal: bool = False
    notify_on_error: bool = True


@dataclass
class HybridConfig:
    """Configuration for Hybrid RAG + LLM Pipeline (3-layer decision system)."""
    # Master enable/disable
    enabled: bool = field(default_factory=lambda: os.environ.get("HYBRID_ENABLED", "True").lower() == "true")
    # Level confirmation gate
    level_confirmation_enabled: bool = True
    level_confirm_proximity_pct: float = 0.30
    level_confirm_buffer_atr_mult: float = 0.10
    level_confirm_min_buffer_points: float = 0.0
    level_confirm_max_wait_candles: int = 6
    level_confirm_timeout_mode: str = "SOFT_PENALTY"  # or "DISABLE"
    level_confirm_timeout_penalty: float = 0.12
    
    # D-Engine (Deterministic Rules) settings
    candidate_threshold: float = 0.55  # Minimum D-engine score to proceed to RAG
    atr_min: float = 0.15  # Minimum ATR threshold (lowered for low-vol markets)
    atr_max: float = 20.0  # Maximum ATR threshold (Increased for ES volatility)
    chop_ema_spread_min_pct: float = 0.0005  # EMA spread % threshold for CHOP_RANGE filter
    
    # H-Engine (LLM + RAG) settings  
    max_calls_per_hour: int = 10
    min_interval_seconds: int = 60
    top_k: int = 5  # RAG retrieval count
    cache_ttl_seconds: int = 300
    cooldown_minutes: int = 15
    allow_legacy_fallback: bool = False
    llm_uncertainty_band_low: float = 0.35
    llm_uncertainty_band_high: float = 0.65
    llm_call_cooldown_seconds: int = 60
    llm_response_cache_ttl_seconds: int = 900
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.60
    signal_threshold: int = 40
    oversold_extension_rsi_min: float = 40.0
    no_signal_allow_weak_signals: bool = False
    no_signal_allow_chop_bias: bool = False
    min_confidence_for_trade: int = 25  # ADDED: Minimum confidence % for trade execution (25 = 25%)
    
    # RAG data paths
    rag_data_path: str = "rag_data"


@dataclass
class AWSAgentsConfig:
    """Configuration for AWS Bedrock Agents (multi-agent decision system)."""
    # Master enable/disable
    enabled: bool = field(default_factory=lambda: os.environ.get("AWS_AGENTS_ENABLED", "False").lower() == "true")
    block_on_wait: bool = True
    wait_override_confidence: float = 0.75
    
    # Configuration source
    use_deployed_config: bool = True  # Auto-load from deployed_resources.yaml
    config_path: str = "aws/config/deployed_resources.yaml"
    
    # Manual configuration (used if use_deployed_config=false)
    region_name: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    
    # Agent IDs (filled automatically from deployed_resources.yaml)
    data_agent_id: str = ""
    data_agent_alias: str = ""
    decision_agent_id: str = ""
    decision_agent_alias: str = ""
    risk_agent_id: str = ""
    risk_agent_alias: str = ""
    learning_agent_id: str = ""
    learning_agent_alias: str = ""
    
    # Knowledge Base
    knowledge_base_id: str = ""


@dataclass
class LearningConfig:
    """Local learning/ingestion hooks."""
    enabled: bool = True
    outcomes_dir: str = "rag_data/training/trade_outcomes"
    history_dir: str = "rag_data/history_snapshots"
    history_days: int = 45
    ingest_window_days: int = 60
    reason_code_retention_days: int = 90


@dataclass
class FeatureFlagsConfig:
    """Feature flags to safely roll out guardrails."""
    enforce_entry_risk_checks: bool = field(default_factory=lambda: os.environ.get("FF_ENTRY_RISK_GUARDS", "true").lower() not in {"0", "false", "no"})
    enforce_wait_blocking: bool = field(default_factory=lambda: os.environ.get("FF_WAIT_BLOCKING", "true").lower() not in {"0", "false", "no"})
    enforce_reduce_only_exits: bool = field(default_factory=lambda: os.environ.get("FF_EXIT_GUARDS", "true").lower() not in {"0", "false", "no"})
    enable_learning_hooks: bool = field(default_factory=lambda: os.environ.get("FF_LEARNING_HOOKS", "true").lower() not in {"0", "false", "no"})


@dataclass
class ObservabilityConfig:
    """Observability settings (Prometheus exporter)."""
    prometheus_enabled: bool = field(default_factory=lambda: os.environ.get("PROMETHEUS_ENABLED", "False").lower() in {"1", "true", "yes"})
    prometheus_addr: str = field(default_factory=lambda: os.environ.get("PROMETHEUS_ADDR", "0.0.0.0"))
    prometheus_port: int = field(default_factory=lambda: int(os.environ.get("PROMETHEUS_PORT", "8000")))
    env_label: str = field(default_factory=lambda: os.environ.get("DEPLOY_ENV", "local"))


@dataclass
class Settings:
    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    one_minute: OneMinuteStrategyConfig = field(default_factory=OneMinuteStrategyConfig)
    risk_gate: RiskGateConfig = field(default_factory=RiskGateConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    strategies: List[StrategyConfig] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    stocktwits_sentiment: StockwitsSentimentConfig = field(default_factory=StockwitsSentimentConfig)
    multi_source_sentiment: MultiSourceSentimentConfig = field(default_factory=MultiSourceSentimentConfig)
    vix_feed: VixFeedConfig = field(default_factory=VixFeedConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    aws_agents: AWSAgentsConfig = field(default_factory=AWSAgentsConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    features: FeatureFlagsConfig = field(default_factory=FeatureFlagsConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    def validate(self) -> None:
        import logging
        logger = logging.getLogger(__name__)
        
        if self.trading.initial_capital <= 0:
            raise ValueError("initial capital must be positive")
        if self.trading.max_position_size <= 0:
            raise ValueError("max position size must be positive")
        if self.trading.tick_size <= 0:
            raise ValueError("tick size must be positive")
        if self.backtest.slippage < 0:
            raise ValueError("slippage cannot be negative")
        if self.trading.max_contracts_limit > 5:
             # Enforce hard cap in code even if env var tries to override
             self.trading.max_contracts_limit = 5
        
        # ============================================================
        # CRITICAL: Consolidate risk limits to use MOST CONSERVATIVE values
        # This prevents dangerous inconsistencies between RiskGateConfig 
        # and TradingConfig that could allow excessive risk.
        # See review.md: "Inconsistent Risk Limits"
        # ============================================================
        
        # Max contracts: use minimum of all sources
        risk_gate_contracts = self.risk_gate.max_contracts
        trading_max_pos = self.trading.max_position_size
        trading_contracts_limit = self.trading.max_contracts_limit
        
        conservative_max_contracts = min(
            risk_gate_contracts,
            trading_max_pos, 
            trading_contracts_limit
        )
        
        if conservative_max_contracts != trading_max_pos or conservative_max_contracts != trading_contracts_limit:
            logger.warning(
                f"RISK CONSOLIDATION: Max contracts mismatch detected. "
                f"RiskGate={risk_gate_contracts}, TradingConfig.max_position_size={trading_max_pos}, "
                f"TradingConfig.max_contracts_limit={trading_contracts_limit}. "
                f"Using most conservative: {conservative_max_contracts}"
            )
            self.trading.max_position_size = conservative_max_contracts
            self.trading.max_contracts_limit = conservative_max_contracts
        
        # Daily loss limit: use minimum (more conservative) value
        risk_gate_daily_loss = self.risk_gate.daily_max_loss_usd
        trading_daily_loss = self.trading.max_daily_loss
        
        conservative_daily_loss = min(risk_gate_daily_loss, trading_daily_loss)
        
        if risk_gate_daily_loss != trading_daily_loss:
            logger.warning(
                f"RISK CONSOLIDATION: Daily loss limit mismatch detected. "
                f"RiskGate=${risk_gate_daily_loss}, TradingConfig=${trading_daily_loss}. "
                f"Using most conservative: ${conservative_daily_loss}"
            )
            self.trading.max_daily_loss = conservative_daily_loss
        
        # Log final consolidated values for audit trail
        logger.info(
            f"RISK LIMITS CONSOLIDATED: max_contracts={conservative_max_contracts}, "
            f"daily_max_loss=${conservative_daily_loss}"
        )
