"""SPY Options bot configuration.

Signals are sent via Telegram. When ``execution.enabled`` is true, signals
passing the strict quality gate are ALSO executed as IB bracket orders
(limit entry + attached stop-loss + take-profit) — see
SpyOptionsExecutionConfig. Otherwise the bot remains signal-only.

Uses ib_insync connecting to the same IB Gateway as the MES/Gold bots (port 4001).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


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

    # Which expiry to use within each month (JUL 2 2026):
    #   "nearest" — earliest non-past expiry (0-2 DTE via SPY dailies/weeklies;
    #               required for the execution gate to ever see tradeable DTE)
    #   "monthly" — legacy end-of-month expiry (13-29 DTE signals)
    expiry_selection: str = "nearest"

    # Minimum DTE for TREND_CONTINUATION swing entries (JUL 7 2026). These are
    # held minutes-to-hours, so 0DTE afternoon theta erodes the backtested 1.5R
    # (and the theta-burn gate blocks many 0DTE afternoon entries outright).
    # 1 => enrich continuation from the nearest ≥1DTE expiry (gentler theta,
    # holds through the afternoon). 0 => use the 0DTE chain like other signals.
    # Falls back to 0DTE if no ≥min_dte expiry is listed.
    # 2 (JUL 8 2026): a minutes-to-hours swing hold wants the lowest theta the
    # executor allows (max_dte=2). Live evidence — a valid TREND_DOWN
    # continuation (TM-approved, HIGH tier) was blocked at the theta gate on
    # 1DTE (6.4%/hr). 2DTE roughly halves the theta rate while staying a same-
    # day intraday instrument (never held to expiry).
    continuation_min_dte: int = 2

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
    # ── PC_AFTERNOON_FLOW pilot (2026-07-19, log-mined backtest) ──────────
    # Evidence: 9 sessions of poll logs (2,331 prints), P/C-extreme events with
    # 45-min cooldown, ±0.5% barrier walk (same semantics as the shadow sim):
    #   MIDDAY P/C extremes: NO edge (40% WR puts, −0.07%/event) — rejected.
    #   ≥14:00 ET extreme-P/C PUTS: 7W/1L/2S (88% of decided), +0.34% avg,
    #   wins spread over 6 distinct days. Flow-follow into the close.
    # Complements TREND_CONTINUATION: different information source (chain flow
    # vs price structure), different window (TC clusters 10:30–13:30), PUT-only.
    # Pilot discipline: 1 contract, 1/day, executor rolling auto-kill.
    pcaf_enabled: bool = True
    pcaf_min_pc: float = 2.0            # chain P/C ratio trigger (backtest range 1.8–52)
    pcaf_window_start_et: str = "14:00"
    pcaf_window_end_et: str = "14:55"   # entries hard-stop at 15:00 ET anyway
    pcaf_confidence: float = 0.80       # from measured WR — sets HIGH tier honestly
    pcaf_max_per_day: int = 1
    # Stress test 2026-07-19 (6wk, 24 events): 16W/5L/3S, Wilson-lo 54.9%,
    # positive in all 3 regimes and both VIX bands 15-18/18-22. VIX>22 has
    # n=1 → UNPROVEN, not proven-bad: hard guard until data exists.
    pcaf_max_vix: float = 25.0

    # ── VWAP_REVERSION range engine — SHADOW-INCUBATING (2026-07-19) ──────
    # Log-mined (9 sessions): morning 2SD-stretch + RSI-extreme fades resolved
    # 3W/1L (+0.15% avg) in 10:30–13:30 ET, but n=4 → Wilson-lo 30% < ~40%
    # breakeven ⇒ NOT live-qualified. Emitted at confidence 0.60 (honest,
    # sub-threshold) so the shadow book tracks every occurrence with simulated
    # exits; scorecard promotes it via Wilson-lo > breakeven at n≥30. The one
    # late-day fade in the sample lost −0.98% → hard window stop at 13:30 ET.
    vrev_enabled: bool = True
    vrev_window_start_et: str = "10:30"
    vrev_window_end_et: str = "13:30"
    vrev_rsi_low: int = 32              # BELOW_2SD + RSI ≤ this → fade UP (call)
    vrev_rsi_high: int = 68             # ABOVE_2SD + RSI ≥ this → fade DOWN (put)
    vrev_confidence: float = 0.60       # sub-threshold BY DESIGN → shadow-only
    vrev_cooldown_min: int = 45

    # ── Single-authority regime gating (strategy audit 2026-07-17) ────────
    # Legacy engine-internal blocks duplicated rules_v2 gates using a DIFFERENT
    # regime classifier — PC_RATIO survived only when the two classifiers
    # disagreed (audit CRIT), and ORB paid two time-gates in two files with
    # different cutoffs (engine 11:30 hard vs rules_v2 window). Default OFF:
    # rules_v2 (pc_ratio_alignment / orb_gate) is the sole authority.
    # Set True ONLY if running with rules_v2 disabled.
    pc_structural_block_enabled: bool = False
    orb_engine_time_gate_enabled: bool = False

    # Dispatch right='BOTH' (straddle) signals? OFF by default — the executor
    # structurally rejects them (51 dispatched all-time, 0 tradeable, 0 decided
    # outcomes) so they only consumed the daily signal cap. Re-enable for
    # alert-only straddle ideas. (strategy audit 2026-07-17)
    dispatch_non_directional: bool = False

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

    # ── Edge-aware dispatch gates (May 2026 — feedback fix) ──────────────
    # The Edge Reality system was honest enough to compute a -3.2% net edge
    # but the dispatch path still emitted the signal at EXTREME tier. These
    # flags align the *action* with the *math*:
    #
    #   suppress_red_edge: drop the signal entirely when edge_color == "red"
    #     (i.e. regime-adjusted WR < break-even WR). Equivalent to the doc's
    #     "Grade D = do not trade" rule. Default ON.
    #
    #   cap_tier_on_amber_edge: prevent EXTREME tier when edge_color is
    #     "amber" (thin edge, 0–5%) and HIGH/EXTREME when red. The numeric
    #     confidence is left untouched so the audit trail is preserved —
    #     only the displayed *tier* is capped.  Default ON.
    suppress_red_edge: bool = True
    cap_tier_on_amber_edge: bool = True

    # ── Exit-monitor timing (JUL 2 2026 profitability review) ──────────────
    # Apr-Jun data: 44 resolved signals, exits were ~exclusively time_stop /
    # vwap_reversion firing into losses; profit_target fired ONCE. The 30/60
    # min stops cut positions before the thesis could play out while theta
    # was already paid. Widen the windows and make them tunable.
    exit_time_stop_0dte_min: int = 45     # was hard-coded 30
    exit_time_stop_swing_min: int = 90    # was hard-coded 60
    exit_profit_target_pct: float = 0.5   # favorable SPY move → take-profit alert

    # Long-straddle gate: block LONG_STRADDLE generation in RANGE_BOUND /
    # TRANSITION / LOW_VOL regimes with IVR < 30 unless a high-impact
    # catalyst is within 60 min. Long straddles need realised-vol expansion
    # — exactly the wrong setup in compressed-vol regimes.  Default ON.
    block_long_straddle_in_range_low_iv: bool = True


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
class SpyOptionsExecutionConfig:
    """Live/paper order execution for SPY options signals (JUL 2 2026).

    When enabled, signals that pass the *strict quality gate* below are
    executed as IB bracket orders (limit entry + attached stop + take-profit)
    on a DEDICATED ib_insync connection — separate host/port/client_id from
    the market-data connection so paper execution (port 4002) can run against
    live data (port 4001).

    The Telegram signal feed is unchanged: every signal still alerts; only
    the subset passing this gate trades. This preserves the long-standing
    design that feed-level quality gates are advisory (premium/$ never
    rejects a *signal*) while execution applies hard gates.
    """

    # JUL 5 2026: the bot is a TRADING bot, not an advisory bot — execution
    # defaults ON against the PAPER port (4002). Telegram remains the audit
    # trail. Switch ibkr_port to 4001 only when paper results earn it.
    enabled: bool = True

    # ── Order connection (separate from data connection) ──────────────────
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002            # 4002 = PAPER (default), 4001 = LIVE
    ibkr_client_id: int = 7          # Separate from data client (6)
    account: str = ""                # Optional explicit IB account id (e.g. DU1234567)

    # ── Sizing: fixed dollar risk per trade ────────────────────────────────
    # contracts = floor(risk_per_trade_usd / (entry_mid × 100 × stop_pct))
    # A signal whose 1-contract stop-risk exceeds the budget is NOT traded.
    risk_per_trade_usd: float = 150.0
    max_contracts: int = 5           # Absolute cap regardless of budget math

    # ── Portfolio guards ───────────────────────────────────────────────────
    max_open_positions: int = 2
    max_trades_per_day: int = 4
    daily_loss_limit_usd: float = 300.0   # Realized; halts new entries for the day
    max_consecutive_stopouts: int = 2     # Halts new entries for the day

    # ── Bracket geometry ───────────────────────────────────────────────────
    # Stop % comes from the signal's IV-adjusted stop (15/20/25 by IVR band);
    # stop_pct_fallback is used when the signal carries none.
    stop_pct_fallback: float = 25.0
    take_profit_pct: float = 40.0    # vs stop 15-25% → ~1.6-2.6 : 1 reward:risk
    stop_type: str = "stop_limit"    # "stop_limit" (default) or "stop" (market)
    stop_limit_buffer_pct: float = 10.0   # limit = stop_price × (1 − buffer)

    # ── Structure-based bracket (JUL 7 2026) ───────────────────────────────
    # For entries carrying a structural_stop (SPY level) — TREND_CONTINUATION —
    # derive the option premium stop from that level via delta and set the
    # target at continuation_target_r × the stop (the backtested 1.5R sweet
    # spot: PF 1.52 on 60d). The derived premium stop is clamped to
    # [structural_stop_pct_min, structural_stop_pct_max] so a razor-thin
    # structural stop isn't noise-stopped and a huge one doesn't over-risk;
    # the target scales with the clamped stop to hold the R:R.
    use_structural_bracket: bool = True
    continuation_target_r: float = 1.5
    structural_stop_pct_min: float = 12.0
    structural_stop_pct_max: float = 35.0

    # ── Strict quality gate ────────────────────────────────────────────────
    allowed_tiers: List[str] = field(default_factory=lambda: ["HIGH", "EXTREME"])
    require_green_edge: bool = True  # edge_margin > 0 after costs — hard gate
    max_dte: int = 3                 # 0-3 DTE (3 covers Fri→Mon weekend gap): tightest spreads, real gamma
    min_abs_delta: float = 0.30      # avoid lottery tickets
    max_abs_delta: float = 0.70      # avoid deep-ITM (poor % leverage per $)
    max_entry_spread_pct: float = 5.0    # bid/ask spread as % of mid — cost gate
    min_premium: float = 0.30        # avoid junk contracts (spread noise dominates)
    max_premium: float = 8.0
    skip_event_risk: bool = True     # No entries within the event-risk window

    # ── Greeks gates (JUL 5 2026) — theta/IV/gamma as HARD entry filters ──
    # Theta: refuse entries whose premium decays faster than the strategy can
    # realistically outrun. Burn is computed at the CURRENT session pace
    # (edge_reality hourly theta fractions), so the same option passes at
    # 10:00 and fails at 14:30.
    # 8.0 (JUL 8 2026): calibrated from live evidence. 6.0 was a conservative
    # placeholder; it blocked a validated 1-2DTE swing continuation at 6.4%/hr.
    # For a swing targeting 1.5R over ~1-2h, ≤8%/hr theta (≤~12-16% decay over
    # the hold) is tolerable — the directional edge clears it. Still blocks the
    # 0DTE afternoon traps (11-38%/hr) this gate was built for.
    max_theta_burn_pct_per_hour: float = 8.0
    # Required SPY drift (%/hr) merely to offset decay: |theta_hr| / ($delta).
    # SPY sustains ~0.10-0.20%/hr on trend days; needing more than 0.15%/hr
    # just to break even on theta means the trade is renting a melting asset.
    max_breakeven_drift_pct_per_hour: float = 0.15
    # IV crush: no naked-long entries at extreme IV rank — direction can be
    # right and the trade still loses when vol mean-reverts.
    max_ivr_naked_long: float = 75.0
    # Gamma bomb: no fresh 0DTE entries once effective gamma ≥ this multiple
    # (2.5x kicks in <60 min to close; backstops no_new_entries_after_et).
    max_0dte_gamma_accel: float = 2.5

    # ── Cross-asset veto (JUL 5 2026) ──────────────────────────────────────
    # Hard veto: no calls on an active bearish QQQ non-confirmation (SPY new
    # session high that QQQ didn't confirm), no puts on a bullish one, and no
    # entries against strong opposing QQQ relative strength.
    require_cross_asset_confirm: bool = True
    max_opposed_qqq_rs: float = 0.35   # pct points of opposing QQQ-vs-SPY RS

    # ── Timing guards ──────────────────────────────────────────────────────
    no_new_entries_after_et: str = "15:00"   # theta-kill zone
    flatten_0dte_at_et: str = "15:50"        # force-close 0DTE before the bell
    entry_timeout_s: int = 180       # cancel unfilled entry limit after this
    max_hold_minutes: int = 90       # position time stop (bracket may exit earlier)

    # ── Entry fill: fresh-quote + bounded chase (JUL 13 2026) ──────────────
    # Root cause of chronic no-fills (753C 2026-07-10; 749P/748P 2026-07-13, all
    # placed marketably yet never filled): the entry limit was priced off the
    # STALE signal-enrichment quote and never repriced. In the fast directional
    # move the strategy targets, the live ask runs away from the snapshot within
    # seconds, so the limit sits dead until an exit trigger cancels it. Fix:
    #   (a) re-quote the LIVE bid/ask immediately before pricing the order, and
    #   (b) chase — re-post toward the live ask every few seconds while unfilled,
    #       bounded by count and a hard % cap above the original entry.
    entry_requote_at_placement: bool = True  # fetch a fresh IB quote before pricing
    entry_cross_frac: float = 0.25    # cross buffer = min(spread*frac, cross_max) beyond ask
    entry_cross_max: float = 0.03     # hard cap on the cross buffer ($/share)
    entry_reprice_interval_s: float = 8.0  # re-post toward the live ask this often while unfilled
    entry_max_reprices: int = 8       # bounded chase attempts (failures count too — 2026-07-21)
    entry_chase_max_pct: float = 6.0  # never chase the limit >this% above the ORIGINAL entry

    # ── Evidence-tier overrides (2026-07-19) ───────────────────────────────
    # Feature flags for the family→tier map without a code deploy, e.g.:
    #   family_tier_overrides:
    #     TREND_CONTINUATION: "pilot"        # further demote
    #     PC_AFTERNOON_FLOW: "probation"     # ramp after 5 live fills EV>0
    # Valid tiers: experimental(0) / pilot(1) / probation(2) / production(3) / core.
    # Defaults live in executor._FAMILY_TIERS; this map wins where set.
    family_tier_overrides: Dict[str, str] = field(default_factory=dict)


@dataclass
class SpyOptionsExitEngineConfig:
    """Exit Engine v2 (JUL 14 2026) — multi-stage, confirmation-based exits.

    Replaces the legacy 7-trigger any-one-fires-full-close exit logic with a
    scored, confirmed, prioritized engine (docs/SPY_EXIT_ENGINE_DESIGN.md).
    Ships with shadow_mode=True: the engine evaluates and LOGS its decision
    every poll alongside the legacy triggers, but the legacy triggers keep
    acting. Flip shadow_mode=False only after the shadow log shows the engine
    making better calls on real fills.
    """

    enabled: bool = True
    shadow_mode: bool = True          # log-only until validated on live fills

    # Grace period — soft exits suppressed after entry; hard exits stay live.
    grace_min_0dte: int = 6           # ≥ one full closed 5m bar + slack
    grace_min_swing: int = 10

    # Exit-confidence scoring.
    base_threshold: int = 55
    one_shot_margin: int = 15         # score ≥ thr+margin → exit without 2nd eval
    deescalate_margin: int = 10       # score < thr−margin → leave EXIT_PENDING

    # Confirmation counts (closed 5m bars / consecutive evaluations).
    vwap_cross_closes: int = 2        # closes across VWAP itself
    vwap_band_decay_closes: int = 3   # closes with entry band lost
    ema9_cross_closes: int = 2
    regime_confirm_evals: int = 3

    # Profit ladder.
    partial_at_r: float = 1.0         # first scale-out level (R vs premium stop)
    partial_fraction: float = 0.5
    trail_giveback_r: float = 0.5     # after BE-lock, give back ≤ this from HWM
    trail_lock_frac: float = 0.5      # trail locks this fraction of HWM gain

    # Catastrophic overrides (grace does NOT shield these).
    catastrophic_adverse_spy_pct: float = 1.0
    catastrophic_premium_buffer_pct: float = 10.0   # premium loss ≥ iv_stop+buffer

    # Factor weights (binary × weight, capped at 100).
    w_vwap_full_reversion: int = 25
    w_vwap_band_decay: int = 10
    w_ema9_cross: int = 15
    w_ema21_break: int = 20
    w_regime_flip: int = 20
    w_momentum_stall: int = 10
    w_rsi_reversal: int = 10
    w_tape_flip: int = 10
    w_atr_adverse_expansion: int = 15
    w_delta_decay: int = 10


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
class SpyOptionsRealFlowConfig:
    """Real order-flow data: SPY L2 depth + tick-by-tick time & sales.

    Depth requires paid entitlements (NASDAQ TotalView / NYSE ArcaBook);
    both feeds degrade gracefully to unavailable if IB rejects them.
    """

    enabled: bool = True
    tape_enabled: bool = True          # reqTickByTickData(SPY, 'AllLast')
    depth_enabled: bool = True         # reqMktDepth(SPY, isSmartDepth=True)
    depth_levels: int = 5              # book levels per side for imbalance
    tape_window_minutes: int = 5       # rolling tape aggregation window
    large_print_shares: int = 10_000   # block-trade threshold for large-print bias


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
    execution: SpyOptionsExecutionConfig = field(default_factory=SpyOptionsExecutionConfig)
    real_flow: SpyOptionsRealFlowConfig = field(default_factory=SpyOptionsRealFlowConfig)
    exit_engine: SpyOptionsExitEngineConfig = field(default_factory=SpyOptionsExitEngineConfig)

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
