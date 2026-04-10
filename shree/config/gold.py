"""Gold futures strategy configuration.

Isolated config namespace — no MES fields live here.
All defaults are conservative (MGC paper-trading safe).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class GoldSessionBucket(str, Enum):
    """Named intraday session buckets for Gold futures.

    Each bucket has a distinct liquidity/volatility personality.
    All boundary times are in ET (America/New_York).
    """

    OVERNIGHT = "OVERNIGHT"           # 18:00–03:00 ET (Asia session, thinnest liquidity)
    LONDON_OPEN = "LONDON_OPEN"       # 03:00–05:00 ET (London open, volatility spike)
    PRE_COMEX_LATE = "PRE_COMEX_LATE" # 05:00–08:20 ET (post-London, quieting pre-COMEX)
    # Legacy alias preserved for backward-compat with older config.yaml overrides
    PRE_COMEX = "PRE_COMEX"           # Deprecated alias → maps to LONDON_OPEN behavior
    COMEX_OPEN = "COMEX_OPEN"         # 08:20–10:30 ET (COMEX floor open, peak liquidity)
    MIDDAY = "MIDDAY"                 # 10:30–12:00 ET (lunch chop, reduced aggressiveness)
    PRE_CLOSE = "PRE_CLOSE"           # 12:00–13:30 ET (winding down, tightest entries)
    MAINTENANCE = "MAINTENANCE"       # 16:00–17:00 CT (hard block, no trading)
    UNKNOWN = "UNKNOWN"               # Fallback (pre-session gap between 13:30-18:00 ET)


@dataclass
class GoldSessionBucketConfig:
    """Per-bucket behavior modifiers (Phase 5).

    Multipliers are applied on top of base config values:
    - 1.0 = no change
    - < 1.0 = tighter (e.g. 0.7 = 30% tighter)
    - > 1.0 = looser
    - 0.0 = disabled (for boolean-like knobs)
    """

    # Signal family enable/disable
    orb_enabled: bool = True            # ORB signals allowed in this bucket
    pullback_enabled: bool = True       # Pullback signals allowed in this bucket

    # Confidence adjustment (additive, applied after base confidence calc)
    confidence_offset: float = 0.0      # e.g. -0.10 = reduce confidence by 10%

    # ADX minimum multiplier (applied to entry.adx_trend_min)
    adx_min_mult: float = 1.0

    # ATR minimum ratio multiplier (applied to entry.atr_min_ratio)
    atr_min_ratio_mult: float = 1.0

    # Volume minimum multiplier (applied to entry.min_bar_volume)
    volume_min_mult: float = 1.0

    # Extension guard tightening (applied to pullback_max_vwap/ema_extension_atr)
    extension_strictness_mult: float = 1.0

    # SL/TP ATR multiplier overrides (Phase 2)
    # Applied on top of the base exit.atr_sl_multiplier / exit.atr_tp_multiplier.
    # 1.0 = no change (use base config value).
    sl_mult: float = 1.0
    tp_mult: float = 1.0


# ── Default bucket configs ────────────────────────────────────────────────
# These are the out-of-box defaults per session bucket.  Override in config.yaml
# under ``session.buckets.<BUCKET_NAME>`` to tune.

def _default_session_buckets() -> Dict[str, GoldSessionBucketConfig]:
    return {
        GoldSessionBucket.OVERNIGHT.value: GoldSessionBucketConfig(
            orb_enabled=False,          # No OR formed yet; ORB meaningless
            pullback_enabled=True,      # Allow pullbacks but strict
            confidence_offset=-0.10,    # Lower conviction in thin markets
            adx_min_mult=1.25,          # Require stronger ADX signal
            atr_min_ratio_mult=1.5,     # Require higher volatility to trade
            volume_min_mult=2.0,        # Double the volume floor
            extension_strictness_mult=0.6,  # 40% tighter extension guards
            sl_mult=1.8,                # Wide stops — overnight gaps happen
            tp_mult=2.5,                # Need larger target to justify the risk
        ),
        GoldSessionBucket.LONDON_OPEN.value: GoldSessionBucketConfig(
            orb_enabled=False,          # OR not yet formed
            pullback_enabled=True,      # London pullbacks are prime setups
            confidence_offset=0.02,     # Slight bonus — London open is a clean session
            adx_min_mult=1.05,          # Mildly stricter — fast moves, need real trend
            atr_min_ratio_mult=1.1,     # Need genuine volatility
            volume_min_mult=1.2,        # London has good volume
            extension_strictness_mult=0.85,
            sl_mult=1.3,                # Wider stops — London gaps and news
            tp_mult=2.0,                # Generous target; strong directional moves
        ),
        GoldSessionBucket.PRE_COMEX_LATE.value: GoldSessionBucketConfig(
            orb_enabled=False,          # OR not yet formed
            pullback_enabled=True,      # Valid but less energetic
            confidence_offset=-0.05,    # Slight penalty vs London open
            adx_min_mult=1.1,           # Slightly stricter ADX
            atr_min_ratio_mult=1.2,     # Need some volatility
            volume_min_mult=1.5,        # Moderate volume floor bump
            extension_strictness_mult=0.8,
            sl_mult=1.2,                # Narrower than London open
            tp_mult=1.8,                # Good R:R but quieter
        ),
        # PRE_COMEX kept as deprecated alias; config.yaml overrides using "PRE_COMEX"
        # will still work — bucket classifier now routes 03:00-05:00 to LONDON_OPEN.
        GoldSessionBucket.PRE_COMEX.value: GoldSessionBucketConfig(
            orb_enabled=False,
            pullback_enabled=True,
            confidence_offset=-0.05,
            adx_min_mult=1.1,
            atr_min_ratio_mult=1.2,
            volume_min_mult=1.5,
            extension_strictness_mult=0.8,
            sl_mult=1.3,
            tp_mult=2.0,
        ),
        GoldSessionBucket.COMEX_OPEN.value: GoldSessionBucketConfig(
            orb_enabled=True,           # Peak ORB window
            pullback_enabled=True,      # Full signal menu
            confidence_offset=0.0,      # No adjustment — base conditions
            adx_min_mult=1.0,
            atr_min_ratio_mult=1.0,
            volume_min_mult=1.0,
            extension_strictness_mult=1.0,
            sl_mult=1.0,                # Base config values — tightest fills
            tp_mult=1.5,                # Slightly stretched target, strong momentum
        ),
        GoldSessionBucket.MIDDAY.value: GoldSessionBucketConfig(
            orb_enabled=True,           # OR still valid, but choppier
            pullback_enabled=True,
            confidence_offset=-0.08,    # Reduced conviction in chop zone
            adx_min_mult=1.2,           # Require more trend strength
            atr_min_ratio_mult=1.0,
            volume_min_mult=1.3,        # Need more volume to confirm
            extension_strictness_mult=0.85,  # Slightly tighter
            sl_mult=0.8,                # Tighter stops — low volatility chop
            tp_mult=1.2,                # Conservative target in lunch lull
        ),
        GoldSessionBucket.PRE_CLOSE.value: GoldSessionBucketConfig(
            orb_enabled=False,          # Too late for ORB
            pullback_enabled=True,      # Only high-conviction pullbacks
            confidence_offset=-0.12,    # Significant penalty near close
            adx_min_mult=1.3,           # Must be strongly trending
            atr_min_ratio_mult=1.0,
            volume_min_mult=1.5,
            extension_strictness_mult=0.7,  # 30% tighter
            sl_mult=1.2,                # Slightly wider — end-of-day noise
            tp_mult=1.0,                # Conservative target, short time window
        ),
    }


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

    # ── Phase 3: Post-news momentum mode ─────────────────────────────────────
    # After a news lockout window ends, relax regime gates for this many bars
    # so the bot can catch the initial directional impulse.
    post_news_momentum_bars: int = 15

    # ── Phase 5: Session bucket boundaries (ET) ──────────────────────────────
    # Customize where each bucket starts.  Order matters: each bucket runs
    # from its start time to the next bucket's start time.
    bucket_overnight_start_et: str = "18:00"       # COMEX overnight open
    bucket_london_open_start_et: str = "03:00"     # London open (Phase 2)
    bucket_pre_comex_late_start_et: str = "05:00"  # Post-London, pre-COMEX (Phase 2)
    bucket_comex_open_start_et: str = "08:20"      # COMEX floor open
    bucket_midday_start_et: str = "10:30"          # Post-morning session
    bucket_pre_close_start_et: str = "12:00"       # Approaching close

    # Per-bucket behavior modifiers (populated from defaults + config overrides)
    buckets: Dict[str, GoldSessionBucketConfig] = field(
        default_factory=_default_session_buckets,
    )


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

    # RSI — exhaustion filter
    rsi_period: int = 14

    # Bollinger Bands — ORB conviction filter (BBW)
    bb_period: int = 20
    bb_std: float = 2.0

    # Warmup bars required before any signal can fire
    warmup_bars: int = 60   # 60 bars on 1-min = 1 hour warmup

    # Multi-timeframe confirmation gate
    # When enabled, signals require higher-TF ADX strength + EMA alignment
    mtf_enabled: bool = False              # Gate entries on HTF ADX/EMA alignment
    mtf_timeframe_minutes: int = 15        # Resample to this TF for HTF check
    mtf_adx_min: float = 18.0             # HTF ADX must be ≥ this (slightly looser than 1m)
    mtf_ema_alignment_required: bool = True  # HTF EMA9>EMA21 must agree with 1m regime

    # ── Phase 2: Adaptive ADX threshold ──────────────────────────────────────
    # Effective ADX floor = max(15, min(rolling_median + 0.5*std, adx_trend_min))
    # Adapts to low-volatility sessions without hard-coding a static threshold.
    adx_adaptive_window: int = 50          # Bars used to compute rolling ADX stats

    # ── Phase 2: Regime quality gates ─────────────────────────────────────────
    # EMA spread — reject "trending" label when EMAs are nearly flat / overlapping
    ema_spread_min_ratio: float = 0.00015  # |ema9 - ema21| / close must exceed this
    # EMA slope — require the fast EMA to actually be *moving* in the trend direction
    ema_slope_enabled: bool = True
    ema_slope_lookback_bars: int = 3       # Slope = (ema9[now] - ema9[now-N]) / N
    ema_slope_min_per_bar: float = 0.02    # Min absolute slope per bar (points)
    # Price structure — monotonic swing check (Phase 3 fix)
    # Require >50% of consecutive bar-pairs within the window to show HH/HL (bull)
    # or LH/LL (bear).  Needs at least 4 bars to form meaningful pairs.
    price_structure_enabled: bool = True
    price_structure_lookback_bars: int = 6  # Bump to 6 for 5 meaningful pairs

    # ── Phase 4: Incremental indicator computation ────────────────────────────
    # Truncate the DataFrame to this many bars before computing indicators.
    # EWM converges quickly — 500 bars is more than enough for any indicator.
    indicator_max_lookback_bars: int = 500

    # ── Phase 3: Keltner Channels — midday mean-reversion ────────────────────
    keltner_period: int = 20        # EMA period for Keltner midline and ATR
    keltner_mult: float = 1.5       # Band width in ATR multiples


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

    # ── Phase 3: Keltner mean-reversion (midday RANGING) ─────────────────────
    keltner_mr_enabled: bool = True
    # How close to the Keltner band the close must be (in ATR units)
    keltner_touch_atr_mult: float = 0.20     # e.g. 0.20 * ATR above lower band
    # Max extension from VWAP before the mean-reversion setup is invalidated
    keltner_max_vwap_extension_atr: float = 1.0

    # ORB retest: allow a second-chance entry on pullback to OR level after
    # the initial breakout bar, for up to this many bars.  Set to 0 to disable.
    orb_retest_max_bars: int = 10

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

    # Partial exit — exit a fraction of the position at a given R multiple.
    # Only fires when contracts >= 2 (can't split 1 contract).
    # When move_sl_to_be is True the stop is also moved to break-even at the
    # same trigger, protecting the locked-in partial profit.
    partial_exit_enabled: bool = True
    partial_exit_r: float = 1.0           # Trigger after 1R of favorable move
    partial_exit_fraction: float = 0.5    # Exit 50% of the position
    partial_exit_move_sl_to_be: bool = True  # Move SL to break-even on partial


@dataclass
class GoldRiskConfig:
    """Position sizing and daily guardrails."""

    # Per-trade risk — 1 MGC contract with ATR-based SL typically risks $120-$180.
    # 50.0 was too conservative (bypassed by the min-1-contract floor in risk.py).
    # Default raised to 175.0 to reflect realistic single-contract paper risk.
    max_risk_per_trade_usd: float = 175.0

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
