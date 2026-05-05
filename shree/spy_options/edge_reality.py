"""Edge Reality System — institutional-audit additions for SPY signals.

Implements the five Tier-1 features missing from the SPY options bot vs the
OptionsEdge architecture doc (v3.0):

  1. Per-pattern base win rates (PATTERN_WIN_RATE table)
  2. Regime-conditioned win rate adjustment (REGIME_WIN_RATE_MULT)
  3. Break-even win rate calculator (transaction-cost-aware)
  4. Net edge margin = regimeWR − breakevenWR  (green / amber / red)
  5. Hourly theta acceleration display ($/hr by session)
  6. Gamma convexity acceleration multiplier (×1.5/2.5/4.0 near close)
  7. IV-adjusted premium stop %  (15/20/25 by IVR)
  8. SPY index put-skew warning  (3–5 IV pts structural premium)

Pure-functions module — no IB calls, no I/O, no global state. Safe to import
anywhere.  All callers should treat output as advisory: these are heuristics
calibrated against the doc, not statistically-derived probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional, Tuple
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Per-pattern base win rates
# ─────────────────────────────────────────────────────────────────────────────
#
# Mapped from the doc's 13 named patterns → ShreeBot's SignalType enum. The
# bot's signal taxonomy is flow- and structure-based rather than pattern-pure,
# so the mapping is approximate. Numbers align to the doc's regime-NEUTRAL
# baselines and should be treated as priors, not historical truth — they will
# be replaced with bot-empirical win rates from analytics_db once enough
# closed-trade data accumulates (see win_rate_by_signal_type()).

# ── Source attribution ───────────────────────────────────────────────────
# These constants tag where the displayed numbers came from. Surfaced into
# the Telegram block so the trader cannot misread a doc-prior estimate as a
# bot-empirical measurement.
WR_SOURCE_DOC_PRIOR: str = "doc_prior"     # Hand-calibrated from the architecture doc
WR_SOURCE_EMPIRICAL: str = "empirical"     # Derived from analytics_db (≥ 30 closed trades / type)

RT_SOURCE_LIVE_QUOTE: str = "live_quote"   # Computed from the contract's live bid/ask
RT_SOURCE_DEFAULT: str = "default"         # Fallback to SPY_ROUND_TRIP_COST_PCT constant


# Default by signal type. Keyed by SignalType.value (string) so this module
# does not need to import SignalType (avoids circular import).
PATTERN_WIN_RATE: dict[str, int] = {
    # Directional flow signals — call/put sweeps with bid/ask imbalance
    "CALL_SWEEP":         74,   # ≈ Gap Fill Long / VWAP Reclaim band
    "PUT_SWEEP":          74,   # ≈ Gap Fill Short / VWAP Rejection band
    # Debit spreads on low-IV — same directional thesis with vega-protection
    "BULL_CALL_SPREAD":   72,
    "BEAR_PUT_SPREAD":    72,
    # ORB family — closely matches doc's "ORB Long/Short" baseline
    "ORB_BREAKOUT":       78,
    # Trend continuation = pullback-to-anchor rejection in trend
    # Mirrors doc's "PDL Bounce / PDH Reclaim" thesis
    "TREND_CONTINUATION": 76,
    # Long straddle = vol play, no directional edge → mid-tier
    "LONG_STRADDLE":      55,
    # Informational — never used as primary trade rationale
    "HIGH_IV_ALERT":      50,
    "PC_RATIO_EXTREME":   62,
}


def base_win_rate_for(signal_type: str) -> int:
    """Return the base (regime-neutral) historical win rate for a signal type.

    Falls back to 60 — a conservative neutral prior — for unknown types so
    downstream math always has a value to work with.
    """
    return PATTERN_WIN_RATE.get(signal_type, 60)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Regime-conditioned win rate adjustment
# ─────────────────────────────────────────────────────────────────────────────
#
# Doc § 11.1 — historical win rates are averaged across all market regimes.
# In practice the same pattern performs vastly differently by regime. The
# multipliers below are taken directly from the doc.
#
# ShreeBot regime taxonomy is richer than the doc's 4-regime model. Mapping:
#   doc TREND DAY  → bot TREND_UP / TREND_DOWN
#   doc RANGE DAY  → bot RANGE_BOUND
#   doc CHOPPY     → bot TRANSITION / NEWS_DRIVEN-without-direction
#   doc HIGH VOL   → bot HIGH_VOL
#   bot LOW_VOL    → effectively TREND DAY conditions for buyers (no penalty)
#
# Direction-aware: if the signal direction *opposes* the trend regime, the
# multiplier degrades because we are fighting the regime. The doc handles
# this via Layer 3 (alignment multiplier) rather than per-regime; here we
# fold it in so a bearish CALL in TREND_DOWN does not get TREND_DAY credit.

REGIME_WIN_RATE_MULT_ALIGNED: dict[str, float] = {
    "TREND_UP":     1.00,
    "TREND_DOWN":   1.00,
    "RANGE_BOUND":  0.82,
    "TRANSITION":   0.60,
    "NEWS_DRIVEN":  0.60,   # treat as CHOPPY-equivalent for win-rate purposes
    "HIGH_VOL":     0.72,
    "LOW_VOL":      0.92,   # debit buyers love low IV; near-trend treatment
}

REGIME_WIN_RATE_MULT_COUNTER: dict[str, float] = {
    # Counter-trend in a confirmed trend regime — heavy penalty
    "TREND_UP":     0.55,
    "TREND_DOWN":   0.55,
    # Range / transition / vol regimes have no defined "with vs against"
    # so the aligned multiplier is reused.
    "RANGE_BOUND":  0.82,
    "TRANSITION":   0.60,
    "NEWS_DRIVEN":  0.60,
    "HIGH_VOL":     0.72,
    "LOW_VOL":      0.92,
}


def _is_aligned(regime: str, direction: str) -> bool:
    """True if the signal direction agrees with the regime's directional bias.

    direction: "C" (bullish), "P" (bearish), or anything else (treated as
    aligned — neutral patterns are not penalised here).
    """
    if regime == "TREND_UP":
        return direction != "P"
    if regime == "TREND_DOWN":
        return direction != "C"
    return True


def regime_adjusted_wr(
    signal_type: str,
    regime: str,
    direction: str,
    base_wr: Optional[int] = None,
) -> int:
    """Return the regime-conditioned win rate as an integer 0–100.

    Args:
        signal_type: SignalType.value string.
        regime:      Current market regime (from RegimeContext.regime).
        direction:   "C" | "P" | "BOTH" — used to detect counter-trend signals.
        base_wr:     Override base WR (for testing / future empirical data).
    """
    base = base_wr if base_wr is not None else base_win_rate_for(signal_type)
    table = (
        REGIME_WIN_RATE_MULT_ALIGNED
        if _is_aligned(regime, direction)
        else REGIME_WIN_RATE_MULT_COUNTER
    )
    mult = table.get(regime, 0.85)   # unknown regime → mild haircut
    return int(round(base * mult))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Round-trip transaction cost
# ─────────────────────────────────────────────────────────────────────────────
#
# Doc § 14.1 — the bid-ask spread is paid twice (entry + exit). For SPY the
# round-trip cost averages 4.8% of premium under typical conditions.
#
# This module is SPY-only; constants below are from the doc's per-ticker table.
# If you ever extend this to other tickers, add them here.

SPY_ROUND_TRIP_COST_PCT: float = 4.8


def round_trip_cost_pct_from_quote(
    bid: Optional[float],
    ask: Optional[float],
) -> Tuple[float, str]:
    """Compute round-trip cost from live quote, falling back to the SPY default.

    Returns ``(cost_pct, source)`` where ``source`` is one of:
        - ``RT_SOURCE_LIVE_QUOTE`` — cost derived from the actual bid/ask
        - ``RT_SOURCE_DEFAULT``    — fell back to ``SPY_ROUND_TRIP_COST_PCT``
                                     because bid/ask were missing/invalid.
                                     Caller should log a DEBUG line and
                                     surface the estimate to the user.

    Formula: (bid_ask_spread / mid) × 200   — 100 to convert to %, ×2 for
    both legs (entry + exit).
    """
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask <= bid:
        return SPY_ROUND_TRIP_COST_PCT, RT_SOURCE_DEFAULT
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return SPY_ROUND_TRIP_COST_PCT, RT_SOURCE_DEFAULT
    spread = ask - bid
    rt = (spread / mid) * 200.0
    # Sanity clamp: round-trip should be 1–25% in any reasonable market.
    rt = max(1.0, min(25.0, round(rt, 2)))
    return rt, RT_SOURCE_LIVE_QUOTE


# ─────────────────────────────────────────────────────────────────────────────
# 4. Break-even win rate + net edge margin
# ─────────────────────────────────────────────────────────────────────────────
#
# Doc § 11.2 — every trade has a minimum win rate below which transaction
# costs eliminate the edge even with correct directional calls.
#
#   netWin  = targetPct − roundTripCostPct
#   netLoss = stopPct   + roundTripCostPct
#   breakeven = netLoss / (netWin + netLoss) × 100

@dataclass
class EdgeReality:
    historical_wr: int          # base win rate (regime-neutral)
    regime_wr: int              # base × regime multiplier
    breakeven_wr: float         # break-even % given target / stop / cost
    edge_margin: float          # regime_wr − breakeven_wr
    edge_color: str             # "green" | "amber" | "red"
    target_pct: float
    stop_pct: float
    round_trip_pct: float
    # Source attribution — surfaced into the Telegram block so the trader
    # can tell at a glance whether they are looking at a hand-calibrated
    # prior or a live measurement.
    wr_source: str = WR_SOURCE_DOC_PRIOR
    rt_source: str = RT_SOURCE_DEFAULT


def breakeven_win_rate(
    target_pct: float,
    stop_pct: float,
    round_trip_pct: float,
) -> float:
    """Break-even win rate as a percentage (0–100).

    All inputs are positive percentages. ``stop_pct`` is the magnitude of the
    stop-loss (e.g. 20 for a 20% stop), not a signed value.
    """
    net_win = max(0.1, target_pct - round_trip_pct)
    net_loss = stop_pct + round_trip_pct
    if (net_win + net_loss) <= 0:
        return 100.0
    return round(net_loss / (net_win + net_loss) * 100.0, 1)


def edge_margin_color(margin_pct: float) -> str:
    """Doc § 11.3 — green > +5%, amber 0–5%, red ≤ 0%."""
    if margin_pct > 5.0:
        return "green"
    if margin_pct > 0.0:
        return "amber"
    return "red"


def compute_edge_reality(
    signal_type: str,
    regime: str,
    direction: str,
    target_pct: float,
    stop_pct: float,
    round_trip_pct: float = SPY_ROUND_TRIP_COST_PCT,
    base_wr: Optional[int] = None,
    wr_source: str = WR_SOURCE_DOC_PRIOR,
    rt_source: str = RT_SOURCE_DEFAULT,
) -> EdgeReality:
    """All-in-one helper: returns the full EdgeReality bundle for a signal.

    ``wr_source`` and ``rt_source`` are propagated through to the returned
    dataclass so the Telegram formatter can flag estimates vs measurements.
    Default sources are the conservative/cautious ones (doc_prior + default)
    — callers should override when they have live data.
    """
    historical = base_wr if base_wr is not None else base_win_rate_for(signal_type)
    regime_wr_val = regime_adjusted_wr(signal_type, regime, direction, base_wr=historical)
    be = breakeven_win_rate(target_pct, stop_pct, round_trip_pct)
    margin = round(regime_wr_val - be, 1)
    return EdgeReality(
        historical_wr=historical,
        regime_wr=regime_wr_val,
        breakeven_wr=be,
        edge_margin=margin,
        edge_color=edge_margin_color(margin),
        target_pct=target_pct,
        stop_pct=stop_pct,
        round_trip_pct=round_trip_pct,
        wr_source=wr_source,
        rt_source=rt_source,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. IV-adjusted premium stop
# ─────────────────────────────────────────────────────────────────────────────
#
# Doc § 8.4 — a fixed -20% stop is wrong: high-IV options move violently per
# point of underlying. The mapping below caps premium drawdown by IVR band.

def iv_adjusted_stop_pct(iv_rank: float) -> float:
    """Return the stop-loss % (positive number) by IV-rank band."""
    if iv_rank > 65:
        return 15.0
    if iv_rank > 40:
        return 20.0
    return 25.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Hourly theta acceleration
# ─────────────────────────────────────────────────────────────────────────────
#
# Doc § 7.2 — daily theta is not experienced evenly. The fractions below match
# the doc's table and translate a "$0.45/day" theta into a real $/hr cost
# during the current session phase.
#
# Time buckets here align with manager._format / dynamic_confidence buckets,
# but are independent so this module can stand alone.

_SESSION_THETA_FRACTION: list[Tuple[time, float]] = [
    (time(9, 30),  0.05),    # opening danger 09:30–09:45 → 5%/15min
    (time(9, 45),  0.09),    # prime time     09:45–11:30 → 9%/hr
    (time(11, 30), 0.09),    # lunch chop     11:30–13:00 → 9%/hr (still ticks)
    (time(13, 0),  0.14),    # afternoon      13:00–15:00 → 14%/hr
    (time(15, 0),  0.38),    # power hour     15:00–16:00 → 38%/hr
]


def hourly_theta_fraction(now_et: Optional[datetime] = None) -> float:
    """Fraction of the daily theta consumed per hour at the current ET time.

    Returns 0.0 outside RTH (no premium decay tracked outside session).
    """
    if now_et is None:
        now_et = datetime.now(ET)
    t = now_et.time()
    if t < time(9, 30) or t >= time(16, 0):
        return 0.0
    last = 0.0
    for cutoff, frac in _SESSION_THETA_FRACTION:
        if t >= cutoff:
            last = frac
        else:
            break
    return last


def hourly_theta_dollars(daily_theta_per_share: float, now_et: Optional[datetime] = None) -> float:
    """Per-contract $/hr cost.

    ``daily_theta_per_share`` is the IB-reported theta (negative for long
    options).  Returns the absolute $ decay rate per hour per contract
    (×100 shares).
    """
    frac = hourly_theta_fraction(now_et)
    return abs(daily_theta_per_share) * 100.0 * frac


# ─────────────────────────────────────────────────────────────────────────────
# 7. Gamma convexity acceleration
# ─────────────────────────────────────────────────────────────────────────────
#
# Doc § 11.4 — static gamma is dangerously underestimated near 0DTE expiry.
# The empirical multipliers below translate the IB-reported gamma into the
# *effective* gamma a trader actually experiences in the last hour.

def _minutes_to_close(now_et: Optional[datetime] = None) -> int:
    """Whole minutes from now until 16:00 ET. Negative after the close."""
    if now_et is None:
        now_et = datetime.now(ET)
    close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    delta = close - now_et
    return int(delta.total_seconds() // 60)


def gamma_accel_mult(
    dte: int,
    now_et: Optional[datetime] = None,
) -> float:
    """Return the effective-gamma multiplier for the current minutes-to-close.

    Only applies to 0DTE positions (multiplier collapses to 1.0 for >0 DTE).
    """
    if dte != 0:
        return 1.0
    mins = _minutes_to_close(now_et)
    if mins < 30:
        return 4.0
    if mins < 60:
        return 2.5
    if mins < 120:
        return 1.5
    return 1.0


def gamma_warning_active(dte: int, now_et: Optional[datetime] = None) -> bool:
    """True when the gamma-bomb warning should be shown to the trader."""
    return gamma_accel_mult(dte, now_et) > 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. SPY index put skew warning
# ─────────────────────────────────────────────────────────────────────────────
#
# Doc § 8.7 — institutional portfolio managers perpetually buy index puts for
# protection, creating structural demand. SPY puts trade at 3–5 extra IV
# points vs equivalent calls — a hidden cost that makes BEAR_PUT_SPREAD and
# PUT_SWEEP signals systematically more expensive than they appear.

SPY_INDEX_PUT_SKEW_IV_POINTS: float = 4.0   # midpoint of the 3–5 doc range

_PUT_LIKE_TYPES = {"PUT_SWEEP", "BEAR_PUT_SPREAD"}


def skew_warning(signal_type: str, right: str) -> str:
    """Return a display-ready skew warning, or empty string if N/A."""
    is_put_signal = (
        signal_type in _PUT_LIKE_TYPES
        or right == "P"
    )
    if not is_put_signal:
        return ""
    return (
        f"SPY puts trade with ~{SPY_INDEX_PUT_SKEW_IV_POINTS:.0f} extra IV "
        f"points (institutional protection demand). Premium is structurally "
        f"richer than an equivalent-distance call — assume mid-fills are "
        f"closer to the ask on entry."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Default target-pct picker
# ─────────────────────────────────────────────────────────────────────────────
#
# The doc uses 35% target / IV-adjusted stop in its worked examples. ShreeBot
# does not currently set per-signal profit targets — the manager exits on
# +0.5% SPY move. We use the doc's 35% premium target for the break-even
# calculation regardless, since that is what realistic 0DTE traders aim for.
# This is a heuristic and should be revisited once analytics_db has enough
# closed-trade premium-side data to pick a calibrated target per signal type.

DEFAULT_TARGET_PCT_BY_REGIME: dict[str, float] = {
    "TREND_UP":     45.0,
    "TREND_DOWN":   45.0,
    "LOW_VOL":      40.0,
    "RANGE_BOUND":  30.0,
    "TRANSITION":   25.0,
    "NEWS_DRIVEN":  50.0,
    "HIGH_VOL":     50.0,   # bigger payoff required when premium is rich
}


def default_target_pct(regime: str) -> float:
    return DEFAULT_TARGET_PCT_BY_REGIME.get(regime, 35.0)
