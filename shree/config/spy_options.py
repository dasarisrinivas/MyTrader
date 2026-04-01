"""SPY Options signal-only bot configuration.

Signal-only: no orders placed. Signals sent via Telegram.
Uses ib_insync connecting to the same IB Gateway as the MES/Gold bots (port 4001).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpyOptionsIBConfig:
    """IB Gateway connection settings (same gateway as MES/Gold bots)."""

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4001            # Live IB Gateway (4002 = paper)
    ibkr_client_id: int = 5          # Separate from MES(1), VIX(2), Gold(3)

    # How long to wait for snapshot price data after reqMktData(snapshot=True)
    snapshot_wait_s: float = 3.0

    # Extra wait for modelGreeks to populate (Greeks arrive after price data)
    greeks_wait_s: float = 4.0

    # Maximum options to subscribe simultaneously (IB allows ~100 lines)
    max_subscriptions: int = 60


@dataclass
class SpyOptionsChainConfig:
    """Option chain construction settings."""

    # Strikes fetched: all within ±strike_pct_range of current SPY price
    strike_pct_range: float = 0.04    # ±4% ATM window

    # Hard cap per expiry (calls + puts combined)
    max_strikes_per_expiry: int = 30

    # How many near-term expiries to track simultaneously
    num_expiries: int = 2

    # IB exchange
    exchange: str = "SMART"

    # Delay (seconds) between successive option contract qualifications
    conid_resolve_delay_s: float = 0.1

    # Liquidity filters — applied per contract before signal generation
    liquidity_min_oi: int = 1000          # Minimum open interest
    liquidity_max_spread_pct: float = 8.0 # Max bid/ask spread as % of mid
    liquidity_min_volume: int = 500       # Minimum daily volume


@dataclass
class SpyOptionsSignalConfig:
    """Signal generation thresholds and rules."""

    # Volume spike: poll-increment > spike_mult × rolling avg of past increments
    volume_spike_mult: float = 4.0

    # Absolute floor: ignore strikes with total session volume below this
    min_volume_for_signal: int = 500

    # Minimum contracts added in a single poll window to flag as a sweep
    sweep_poll_volume_threshold: int = 300

    # VIX regime thresholds
    vix_low: float = 16.0            # Below → low IV → debit spreads preferred
    vix_high: float = 26.0           # Above → high IV → premium selling preferred

    # Put/Call volume ratio extremes (chain-level)
    pc_ratio_bearish: float = 1.8    # P/C > 1.8 → strong bearish skew
    pc_ratio_bullish: float = 0.5    # P/C < 0.5 → heavy call bias

    # Bid/ask size imbalance to infer directional pressure
    bid_ask_imbalance_threshold: float = 3.0

    # Straddle: both call AND put must spike by at least this multiple
    straddle_spike_mult: float = 3.0

    # Weighted confidence thresholds (replaces simple 0.55 threshold)
    min_confidence: float = 0.70          # Drop signals below this
    confidence_tier_high: float = 0.80    # HIGH tier starts here
    confidence_tier_extreme: float = 0.90 # EXTREME tier starts here

    # Suppress re-sending same (type, expiry, strike, right) within this window
    dedup_window_minutes: int = 90

    # Repeat sweep detection window — same strike flagged N× within this boosts score
    sweep_window_minutes: int = 15


@dataclass
class SpyOptionsSessionConfig:
    """Market session gates."""

    rth_only: bool = True

    # America/New_York (ET)
    rth_start_et: str = "09:35"    # Skip first 5 min open noise
    rth_stop_et: str = "15:45"     # Stop 15 min before close

    # Poll interval in seconds
    poll_interval_s: int = 60


@dataclass
class SpyOptionsAnalyticsConfig:
    """Signal analytics persistence (SQLite)."""

    enabled: bool = True
    db_path: str = "data/spy_options_signals.db"


@dataclass
class SpyOptionsConfig:
    """Top-level SPY Options signal bot configuration.

    Signal-only — no orders are ever placed.
    Uses ib_insync connecting to IB Gateway (same as MES/Gold bots).
    """

    enabled: bool = False

    ib: SpyOptionsIBConfig = field(default_factory=SpyOptionsIBConfig)
    chain: SpyOptionsChainConfig = field(default_factory=SpyOptionsChainConfig)
    signals: SpyOptionsSignalConfig = field(default_factory=SpyOptionsSignalConfig)
    session: SpyOptionsSessionConfig = field(default_factory=SpyOptionsSessionConfig)
    analytics: SpyOptionsAnalyticsConfig = field(default_factory=SpyOptionsAnalyticsConfig)

    log_file: str = "logs/spy_options.log"
