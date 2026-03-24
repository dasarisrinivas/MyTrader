"""Gold futures strategy configuration.

Isolated config namespace — no MES fields live here.
All defaults are conservative (MGC paper-trading safe).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GoldSessionConfig:
    """COMEX Gold intraday session windows (all times in ET)."""

    # Primary COMEX liquid window
    session_open_et: str = "08:20"      # COMEX open
    session_close_et: str = "13:30"     # Pit session close
    flatten_before_close_minutes: int = 10  # Flatten at 13:20 ET

    # IBKR daily maintenance (CME/COMEX) — hard block, no trading
    maintenance_start_ct: str = "16:00"  # 4 PM CT
    maintenance_end_ct: str = "17:00"    # 5 PM CT

    # Opening range window (first N minutes after session_open_et)
    opening_range_minutes: int = 15

    # Blackout after session open — skip noisy first bars
    no_trade_open_minutes: int = 3

    # Optional news/economic-release lockout windows (ET, inclusive)
    # Format: [["08:25", "08:35"], ["10:00", "10:05"]]
    news_lockout_windows_et: List[List[str]] = field(default_factory=list)

    # Extended / overnight session (6 PM – 8:20 AM ET, Sunday–Friday)
    extended_hours_enabled: bool = False
    extended_session_open_et: str = "18:00"   # COMEX overnight open (6 PM ET)


@dataclass
class GoldIndicatorConfig:
    """Indicator parameters for the Gold intraday strategy."""

    # VWAP
    vwap_session_anchor_et: str = "08:20"   # Reset VWAP at COMEX open

    # EMAs
    ema_fast: int = 9
    ema_slow: int = 21

    # ATR — volatility-aware stops
    atr_period: int = 14

    # ADX — trend strength
    adx_period: int = 14

    # Warmup bars required before any signal can fire
    warmup_bars: int = 60   # 60 bars on 1-min = 1 hour warmup

    # Multi-timeframe confirmation gate
    # When enabled, signals require higher-TF ADX strength + EMA alignment
    mtf_enabled: bool = False              # Gate entries on HTF ADX/EMA alignment
    mtf_timeframe_minutes: int = 15        # Resample to this TF for HTF check
    mtf_adx_min: float = 18.0             # HTF ADX must be ≥ this (slightly looser than 1m)
    mtf_ema_alignment_required: bool = True  # HTF EMA9>EMA21 must agree with 1m regime

    # ── Phase 2: Regime quality gates ─────────────────────────────────────────
    # EMA spread — reject "trending" label when EMAs are nearly flat / overlapping
    ema_spread_min_ratio: float = 0.00015  # |ema9 - ema21| / close must exceed this
    # EMA slope — require the fast EMA to actually be *moving* in the trend direction
    ema_slope_enabled: bool = True
    ema_slope_lookback_bars: int = 3       # Slope = (ema9[now] - ema9[now-N]) / N
    ema_slope_min_per_bar: float = 0.02    # Min absolute slope per bar (points)
    # Price structure — optional higher-high/higher-low (bull) or lower-high/lower-low (bear)
    price_structure_enabled: bool = True
    price_structure_lookback_bars: int = 5  # How many bars back to check swing structure


@dataclass
class GoldEntryConfig:
    """Entry trigger parameters."""

    # Regime filters
    adx_trend_min: float = 20.0     # Minimum ADX to confirm trending regime
    adx_trend_max: float = 55.0     # Maximum ADX — above this = too chaotic / no-trade
    atr_min_ratio: float = 0.0002   # Min ATR / price (avoids flat/illiquid market)
    atr_max_ratio: float = 0.006    # Max ATR / price (avoids chaotic spike)

    # VWAP pullback trigger — how close price must be to VWAP
    vwap_touch_pct: float = 0.002   # 0.20% of price

    # EMA touch trigger
    ema_touch_pct: float = 0.0025   # 0.25% of price

    # Confirmation: how many consecutive bars must close in the pullback zone
    # before the break-of-bar-high/low entry fires
    confirmation_bars: int = 1

    # Pullback confirmation quality
    pullback_reclaim_required: bool = True   # Bar must reclaim the level after probing it
    pullback_min_body_fraction: float = 0.25  # Body / range must meet this threshold
    pullback_confirm_with_bar_direction: bool = True  # Long requires bullish close; short bearish

    # Anti-chase / extension guards (ATR-based)
    pullback_max_vwap_extension_atr: float = 1.5   # Block pullbacks too far from VWAP
    pullback_max_ema_extension_atr: float = 1.25   # Block pullbacks too far from EMA21

    # Overnight / extended-hours strictness multiplier for extension guards.
    # Values < 1.0 tighten thresholds (e.g. 0.7 reduces max extension by 30%).
    # Set to 1.0 to disable overnight tightening.
    extended_hours_extension_strictness_mult: float = 0.7

    # Opening range breakout — minimum OR size to qualify
    orb_enabled: bool = True
    orb_min_range_ratio: float = 0.001   # OR range / price ≥ 0.1%
    orb_breakout_min_atr_fraction: float = 0.10   # Require breakout beyond OR by ATR fraction
    orb_volume_lookback_bars: int = 20
    orb_volume_min_multiple: float = 1.20
    orb_max_breakout_candle_atr: float = 1.25
    orb_max_extension_atr: float = 0.75
    orb_max_vwap_extension_atr: float = 1.5   # Block ORB when price too far from VWAP

    # Minimum bar volume (skip zero-range / no-data bars)
    min_bar_volume: int = 5

    # Block same-direction entries for N bars after a stop-loss exit
    post_loss_cooldown_bars: int = 3

    # Direction-aware cooldown: after a loss, block the *same* signal family
    # for a longer cooldown while allowing opposite-direction entries sooner.
    # "Same family" = same signal type prefix (VWAP_PB, EMA_PB, ORB).
    # Set to 0 to disable direction-aware logic (falls back to uniform cooldown).
    post_loss_same_family_cooldown_bars: int = 6  # Longer cooldown for revenge trades
    post_loss_opposite_cooldown_bars: int = 1      # Minimal cooldown for opposite direction


@dataclass
class GoldExitConfig:
    """Exit parameters — all ATR-based to adapt to gold volatility."""

    # Stop loss
    atr_sl_multiplier: float = 1.5      # SL = entry ± ATR × 1.5
    sl_floor_points: float = 2.0        # Minimum SL distance (pts) — avoids tiny stops
    sl_ceiling_points: float = 30.0     # Maximum SL distance (pts) — avoids oversized risk

    # Take profit
    atr_tp_multiplier: float = 2.5      # TP = entry ± ATR × 2.5  →  R:R ≈ 1.67

    # Trailing stop (activates only after minimum favorable excursion)
    trailing_stop_enabled: bool = True
    trailing_activation_r: float = 1.0  # Activate trail after 1R in favor
    trailing_atr_mult: float = 1.0      # Trail by ATR × 1.0

    # Time stop — exit if no meaningful progress within N bars
    time_stop_enabled: bool = True
    time_stop_bars: int = 60            # 60 min on 1-min bars; 0 = disabled (hard cap)

    # Progress-aware staged time stop (evaluated before the hard time_stop_bars cap)
    # Stage 1: after N bars, require at least X×R unrealized progress or exit
    time_stop_stage_1_bars: int = 20
    time_stop_stage_1_min_progress_r: float = 0.25  # 0.25R = 25% of SL distance
    # Stage 2: after M bars, require break-even or better
    time_stop_stage_2_bars: int = 40
    time_stop_stage_2_min_progress_r: float = 0.0   # 0.0R = break-even

    # End-of-session flatten — hard close before maintenance
    # (uses GoldSessionConfig.flatten_before_close_minutes)

    # Extended/overnight session — wider multipliers to absorb overnight vol
    extended_atr_sl_multiplier: float = 2.5   # RTH default: 1.5
    extended_atr_tp_multiplier: float = 4.0   # RTH default: 2.5

    # Minimum R:R to accept a trade
    min_rr_ratio: float = 1.2


@dataclass
class GoldRiskConfig:
    """Position sizing and daily guardrails."""

    # Per-trade risk
    max_risk_per_trade_usd: float = 50.0    # Max $ at risk per trade (conservative default)

    # Daily guardrails
    daily_loss_limit_usd: float = 200.0     # Halt trading for the day after this loss
    max_trades_per_day: int = 5

    # Consecutive-loss cooldown
    max_consecutive_losses: int = 3
    post_loss_cooldown_minutes: int = 30

    # Position cap
    max_concurrent_positions: int = 1       # Never hold 2 gold positions simultaneously

    # Safety: hard override on contract count regardless of position sizing math
    max_contracts_hard_cap: int = 2         # Absolute maximum for any single order

    # Live GC validation gate (used by validate_for_live_gc)
    # Minimum MGC paper-trading track record before GC live is allowed
    gc_min_paper_trades: int = 50
    gc_min_win_rate: float = 0.45           # 45 % win rate on paper MGC
    # When going live on GC (10× risk vs MGC), risk is automatically scaled
    # down by this factor so dollar exposure stays comparable to MGC paper trading.
    gc_risk_scale: float = 0.1             # GC point_value=100 vs MGC=10 → 0.1× default


@dataclass
class GoldStrategyConfig:
    """Top-level Gold futures strategy configuration.

    Feature-flagged: ``enabled: false`` by default so adding this config to
    an existing MES-only config file has zero runtime impact.

    Instrument defaults to MGC (Micro Gold) for paper trading safety.
    GC (Full Gold) requires explicit ``allow_gc: true``.
    """

    # ── Feature flag ──────────────────────────────────────────────────────────
    enabled: bool = False               # Must be explicitly enabled

    # ── Instrument ───────────────────────────────────────────────────────────
    symbol: str = "MGC"                 # "MGC" = Micro Gold (default); "GC" = Full Gold
    exchange: str = "COMEX"
    currency: str = "USD"
    bar_size: str = "1 min"             # IB bar size spec

    # Safety: standard Gold requires explicit opt-in to prevent accidental use
    allow_gc: bool = False              # Set true to permit GC (100 oz, $100/pt)

    # ── IBKR connectivity ─────────────────────────────────────────────────────
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002               # Default to paper (4002) until explicitly changed
    ibkr_client_id: int = 3             # Separate from MES (1) and VIX feed (2)

    # ── Mode ─────────────────────────────────────────────────────────────────
    simulation: bool = True             # Dry-run by default until explicitly disabled

    # ── Sub-configs ──────────────────────────────────────────────────────────
    session: GoldSessionConfig = field(default_factory=GoldSessionConfig)
    indicators: GoldIndicatorConfig = field(default_factory=GoldIndicatorConfig)
    entry: GoldEntryConfig = field(default_factory=GoldEntryConfig)
    exit: GoldExitConfig = field(default_factory=GoldExitConfig)
    risk: GoldRiskConfig = field(default_factory=GoldRiskConfig)

    # ── Persistence paths ─────────────────────────────────────────────────────
    state_file: str = "data/gold_state.json"
    orders_db: str = "data/gold_orders.db"
    journal_dir: str = "data/gold_journal"
    log_file: str = "logs/gold_trading.log"

    def validate(self) -> None:
        """Raise ValueError for obviously unsafe configurations."""
        if self.symbol == "GC" and not self.allow_gc:
            raise ValueError(
                "GC (Full Gold, 100 oz, $100/pt) requires 'gold.allow_gc: true' "
                "to prevent accidental use. Set it explicitly when you intend to "
                "trade full-size gold futures."
            )
        if self.risk.max_risk_per_trade_usd <= 0:
            raise ValueError("gold.risk.max_risk_per_trade_usd must be positive")
        if self.risk.daily_loss_limit_usd <= 0:
            raise ValueError("gold.risk.daily_loss_limit_usd must be positive")
        if self.risk.max_contracts_hard_cap < 1:
            raise ValueError("gold.risk.max_contracts_hard_cap must be >= 1")
        if self.exit.atr_sl_multiplier <= 0:
            raise ValueError("gold.exit.atr_sl_multiplier must be positive")
        if self.exit.atr_tp_multiplier <= self.exit.atr_sl_multiplier:
            raise ValueError(
                "gold.exit.atr_tp_multiplier must exceed atr_sl_multiplier "
                f"(got tp={self.exit.atr_tp_multiplier} sl={self.exit.atr_sl_multiplier})"
            )
        if self.ibkr_client_id in (1, 2):
            raise ValueError(
                f"gold.ibkr_client_id={self.ibkr_client_id} conflicts with MES (1) "
                "or VIX feed (2). Use a different client ID (e.g. 3)."
            )

    def validate_for_live_gc(
        self,
        paper_trade_count: int,
        paper_win_rate: float,
    ) -> None:
        """Gate live GC trading on a validated MGC paper-trading track record.

        Call this *before* switching ``simulation=False`` on a GC config.
        Raises ``ValueError`` if the minimum track record isn't met.

        Args:
            paper_trade_count: Number of completed MGC paper trades.
            paper_win_rate:    Win rate on those paper trades (0.0–1.0).
        """
        if self.symbol != "GC":
            return   # Not applicable
        if self.simulation:
            return   # Still in paper mode — no gate needed

        min_trades = self.risk.gc_min_paper_trades
        min_wr = self.risk.gc_min_win_rate

        if paper_trade_count < min_trades:
            raise ValueError(
                f"Live GC requires at least {min_trades} validated MGC paper trades "
                f"(current: {paper_trade_count}).  Keep paper-trading MGC."
            )
        if paper_win_rate < min_wr:
            raise ValueError(
                f"Live GC requires a {min_wr * 100:.0f}%+ win rate on paper MGC "
                f"(current: {paper_win_rate * 100:.1f}%).  Review strategy before going live."
            )

    def gc_adjusted_risk_usd(self) -> float:
        """Return the per-trade risk in USD, scaled down for GC's 10× point value.

        GC has ``point_value=100`` vs MGC's ``10``.  To keep dollar risk at
        comparable levels, the default ``gc_risk_scale=0.1`` reduces the raw
        ``max_risk_per_trade_usd`` by 10×.  Override ``gc_risk_scale`` in config
        to calibrate.

        Only relevant when ``symbol == "GC"``; returns unmodified value otherwise.
        """
        if self.symbol != "GC":
            return self.risk.max_risk_per_trade_usd
        return self.risk.max_risk_per_trade_usd * self.risk.gc_risk_scale
