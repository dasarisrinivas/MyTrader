"""SPY Options signal-only bot configuration.

Signal-only: no orders are placed. Signals are sent via Telegram.
Uses IB Client Portal REST API (port 5000) — separate from the TWS Gateway
used by the MES/Gold bots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SpyOptionsIBConfig:
    """IB Client Portal REST API connection settings."""

    host: str = "127.0.0.1"
    port: int = 5000               # Client Portal default; some use 5001
    use_https: bool = True
    verify_ssl: bool = False       # IB uses a self-signed cert — must be False
    request_timeout_s: float = 10.0

    # Keep-alive tickle (IB requires POST /tickle every ~1 min)
    tickle_interval_s: float = 55.0

    # Retry settings for transient errors
    max_retries: int = 3
    retry_delay_s: float = 2.0

    # First snapshot call subscribes but returns no data; second returns data.
    # Wait this many seconds between pre-flight subscribe and actual read.
    snapshot_preflight_delay_s: float = 1.5

    # VIX index conid on IBKR (used to fetch VIX for IV regime detection).
    # Set to 0 to disable VIX fetching.
    vix_conid: int = 13455763


@dataclass
class SpyOptionsChainConfig:
    """Option chain construction settings."""

    # Strikes fetched = all strikes within ±strike_pct_range of SPY price
    strike_pct_range: float = 0.04    # ±4% ATM window

    # Hard cap: maximum strikes to subscribe per expiry (calls + puts combined)
    max_strikes_per_expiry: int = 30

    # How many near-term monthly expiries to track simultaneously
    num_expiries: int = 2

    # IB option exchange
    exchange: str = "SMART"

    # Inter-request delay when resolving option conids (stay under 10 req/s limit)
    conid_resolve_delay_s: float = 0.15


@dataclass
class SpyOptionsSignalConfig:
    """Signal generation thresholds and rules."""

    # Volume spike: poll-increment > spike_mult × rolling-avg of past increments
    volume_spike_mult: float = 4.0

    # Absolute floor: ignore strikes with total session volume below this
    min_volume_for_signal: int = 500

    # Sweep: minimum contracts added in a single poll window to flag as sweep
    sweep_poll_volume_threshold: int = 300

    # VIX proxy for IV regime
    vix_low: float = 16.0          # VIX < this → low IV → debit spreads preferred
    vix_high: float = 26.0         # VIX > this → high IV → premium selling preferred

    # Put/Call volume ratio extremes (chain-level)
    pc_ratio_bearish: float = 1.8  # P/C > 1.8 → strong bearish skew
    pc_ratio_bullish: float = 0.5  # P/C < 0.5 → heavy call bias / complacency

    # Bid/ask size imbalance to infer directional pressure
    # bid_size / ask_size > threshold → aggressive call buying
    bid_ask_imbalance_threshold: float = 3.0

    # Straddle: call AND put must both spike by at least this multiple
    straddle_spike_mult: float = 3.0

    # Don't send a signal unless confidence >= this
    min_confidence: float = 0.55

    # Deduplication: suppress re-sending the same (type, expiry, strike, right)
    # for this many minutes after the first alert
    dedup_window_minutes: int = 90


@dataclass
class SpyOptionsSessionConfig:
    """Market session gates."""

    rth_only: bool = True

    # All times in America/New_York (ET)
    rth_start_et: str = "09:35"    # Skip first 5 min open noise
    rth_stop_et: str = "15:45"     # Stop 15 min before close

    # How often to poll IB for new snapshots
    poll_interval_s: int = 60


@dataclass
class SpyOptionsConfig:
    """Top-level SPY Options signal bot configuration.

    Signal-only — no orders are ever placed.
    Signals are always sent via Telegram when enabled.
    Requires IB Client Portal Gateway to be running and authenticated.
    """

    enabled: bool = False

    ib: SpyOptionsIBConfig = field(default_factory=SpyOptionsIBConfig)
    chain: SpyOptionsChainConfig = field(default_factory=SpyOptionsChainConfig)
    signals: SpyOptionsSignalConfig = field(default_factory=SpyOptionsSignalConfig)
    session: SpyOptionsSessionConfig = field(default_factory=SpyOptionsSessionConfig)

    log_file: str = "logs/spy_options.log"
