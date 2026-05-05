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

    # Minimum volume required for BOTH puts AND calls before computing P/C ratio.
    # Prevents nonsense ratios like 46:1 from tiny denominators.
    pc_ratio_min_denom_volume: int = 200

    # After sending a PC_RATIO signal in one direction, suppress the opposite
    # direction for this many minutes (prevents whipsaw flip-flop alerts).
    pc_ratio_flip_cooldown_minutes: int = 30

    # Suppress re-sending same (type, expiry, strike, right) within this window
    dedup_window_minutes: int = 90

    # Maximum number of signals dispatched per trading day.
    # After this limit, all further signals are suppressed until daily reset.
    # Prevents signal flooding (e.g. 55 signals in 3 days).
    max_signals_per_day: int = 10

    # Repeat sweep detection window — same strike flagged N× within this boosts score
    sweep_window_minutes: int = 15

    # ── Edge Reality additions (May 2026 — institutional-audit features) ──
    # When enabled, every dispatched signal is augmented with:
    #   - regime-adjusted historical win rate
    #   - break-even win rate (transaction-cost-aware)
    #   - net edge margin (green / amber / red)
    #   - hourly theta $/hr by session phase
    #   - effective gamma multiplier near close (×1.5 / 2.5 / 4.0)
    #   - IV-adjusted premium-stop %
    #   - SPY index put-skew warning on bearish signals
    # Additive only — never blocks a signal. Default ON because all changes
    # are pure display + an additional advisory exit trigger; existing
    # confidence + tier logic is unchanged.
    edge_reality_enabled: bool = True

    # IV-adjusted premium stop: when enabled, an additional exit alert fires
    # when the *estimated option premium drawdown* exceeds the IVR-conditional
    # cap from edge_reality.iv_adjusted_stop_pct (15/20/25 by IVR band).
    # The estimate uses delta × SPY_dollar_move / entry_premium as a proxy
    # because the manager does not re-snapshot Greeks per poll.
    iv_adjusted_premium_stop_enabled: bool = True


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
class SpyOptionsExternalConfig:
    """External signal sources (economic calendar, news, social, macro, flow)."""

    enabled: bool = True

    # Economic calendar (Forex Factory JSON — free, no key)
    calendar_enabled: bool = True
    event_risk_window_minutes: int = 30   # Flag event risk within ±N min

    # RSS news sentiment (VADER scoring — no API key)
    news_enabled: bool = True
    news_ttl_minutes: float = 10.0

    # StockTwits retail sentiment (free, no auth)
    stocktwits_enabled: bool = True
    stocktwits_ttl_minutes: float = 10.0

    # Reddit enhanced sentiment (asyncpraw — full body+comments; disabled by default)
    reddit_enabled: bool = False
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_ttl_minutes: float = 15.0

    # Macro signals via yfinance — daily EOD + 15-min intraday refresh
    macro_enabled: bool = True

    # CBOE P/C ratio daily CSV (free, no auth)
    cboe_enabled: bool = True

    # Options flow confirmation (yfinance options chain, GEX, dark pool proxy, gamma walls)
    flow_enabled: bool = True
    flow_ttl_minutes: float = 10.0
    flow_barchart_enabled: bool = True   # Attempt Barchart scrape (fails gracefully)
    flow_dark_pool_enabled: bool = True  # Alpha Query dark pool + premium-skew proxy

    # Market breadth (11 sector ETFs above-open ratio, SPY up/down vol — yfinance, 10-min TTL)
    breadth_enabled: bool = True
    breadth_ttl_minutes: float = 10.0

    # Sector leadership (XLK, XLF, SMH, IWM, QQQ, XLE, XLI vs day-open — 10-min TTL)
    sector_enabled: bool = True
    sector_ttl_minutes: float = 10.0

    # Volatility term structure (VIX/VXV ratio, VVIX — 15-min TTL)
    vol_structure_enabled: bool = True
    vol_structure_ttl_minutes: float = 15.0

    # OPEX calendar (pure date math, zero latency)
    opex_enabled: bool = True

    # External composite score impact on weighted confidence (max ±5%)
    composite_confidence_boost: float = 0.05


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
    external: SpyOptionsExternalConfig = field(default_factory=SpyOptionsExternalConfig)

    # Regime-first, structure-based rules layer (added Apr 22 2026 after Apr 21 postmortem).
    # Defaults to enabled=False — turning this on replaces the legacy directional
    # throttle and ORB/PC_RATIO gating with the v2 pipeline in
    # shree/spy_options/rules_v2/.
    rules_v2: "SpyOptionsRulesV2Config" = field(
        default_factory=lambda: SpyOptionsRulesV2Config()
    )

    log_file: str = "logs/spy_options.log"


# ───────────────────────────────────────────────────────────────────────────
# Rules-v2 config is defined in shree/spy_options/rules_v2/config.py.
# This thin wrapper re-exports it under the settings tree so existing
# YAML-loading machinery can populate it without importing from the strategy
# module at config-load time.
# ───────────────────────────────────────────────────────────────────────────
try:
    from ..spy_options.rules_v2.config import RulesV2Config as SpyOptionsRulesV2Config
except Exception:  # pragma: no cover — config module must load even if rules_v2 missing
    @dataclass
    class SpyOptionsRulesV2Config:  # type: ignore[no-redef]
        enabled: bool = False
