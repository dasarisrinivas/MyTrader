"""Strategy configuration dataclasses — entry filters, 1m, 30m, and generic strategy config."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
from datetime import time

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

    # FEB 2026: RTH forced flatten — hard exit if position exists at this time (ET)
    # Defense-in-depth backstop. Normal exits (brackets, time-stop) should fire first.
    rth_flatten_time_et: str = "15:50"

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

    # FEB 2026: Structural reversion strategy (replaces scoring system)
    # Data-driven redesign: mean reversion from VWAP/PDH/PDL after opening range
    use_structural_reversion: bool = False
    sr_min_atr: float = 4.0           # Minimum ATR to trade (88% of losses were ATR < 4)
    sr_max_trades: int = 2            # Max trades per session (data: 3rd+ trades lose money)
    sr_cooldown_minutes: int = 15     # Minutes between trades
    sr_target_mode: str = "vwap"      # 'vwap' = target VWAP, 'atr' = fixed ATR multiple
    sr_target_atr_mult: float = 1.5   # If target_mode='atr'
    sr_time_stop_minutes: int = 45    # Max hold time (data: 10-45 min is profitable window)
    sr_rsi_long_max: float = 38.0     # RSI must be below this for longs
    sr_rsi_short_min: float = 62.0    # RSI must be above this for shorts
    sr_min_volume: int = 100          # Minimum volume on entry bar
    sr_or_minutes: int = 30           # Opening range duration (minutes after RTH open)
    sr_overextension_atr_mult: float = 0.5  # How far from VWAP = "overextended"
    sr_breakeven_r: float = 1.0       # Move stop to breakeven after this R-multiple
    sr_trade_start_hour: int = 10     # Trade window start (CST)
    sr_trade_start_minute: int = 0
    sr_trade_end_hour: int = 11       # Trade window end (CST)
    sr_trade_end_minute: int = 30

    # FEB 2026: 15-minute strategy — NOW THE DEFAULT
    # Replaces all 1m approaches (scoring, structural reversion, trend)
    # Data-driven: EMA21 pullback + EMA9 pullback + OR breakout, long-only
    # v3 backtest (FEB 8): PF 2.18, Sharpe 24.73, +$4,129 on 174 trades
    use_15m_strategy: bool = True  # DEFAULT ON — 1m strategies are sunset
    use_30m_strategy: bool = False  # 30m overnight/Globex strategy
    ft_pb_stop_mult: float = 1.5      # Pullback stop = 1.5 × ATR_14
    ft_pb_target_mult: float = 1.0    # Pullback target = 1.0 × ATR_14 (v3: swept 0.8-2.0)
    ft_or_target_r: float = 1.0       # OR breakout target = 1.0 × risk (v3: swept 0.8-1.3)
    ft_adx_min: float = 20.0          # Minimum ADX for any entry
    ft_adx_max: float = 35.0          # Maximum ADX (cap exhaustion moves, v3: 35+ loses money)
    ft_ema_touch_pct: float = 0.001   # How close low must be to EMA21 (0.1%)
    ft_or_minutes: int = 30           # Opening range window (minutes)
    ft_max_hold_bars: int = 8         # Max hold = 8 × 15m = 120 minutes (v3: tested 10→worse)

    # FEB 8 2026: EMA9 Pullback (Signal C) — faster trend capture
    ft_ema9_pb_enabled: bool = True    # v3: enabled, +$519 on 43 trades, 60.5% WR
    ft_ema9_pb_stop_mult: float = 1.2  # Tighter stop for shallow pullbacks
    ft_ema9_pb_target_mult: float = 1.5
    ft_ema9_touch_pct: float = 0.0015  # EMA9 touch tolerance (0.15%)

    # FEB 8 2026: Entry time filter (ET) — v3 optimized window
    # 10:30-14:59 ET optimal (swept 11:00, 13:59, 14:29 cutoffs)
    ft_entry_start_hour: int = 10     # Earliest entry hour (ET)
    ft_entry_start_minute: int = 30
    ft_entry_end_hour: int = 15       # Latest entry hour (ET, exclusive)
    ft_entry_end_minute: int = 0


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
class StrategyConfig:
    name: str = "rsi_macd_sentiment"
    enabled: bool = True
    params: dict = field(default_factory=dict)



