"""Signal engine — determines strategy direction using 9 filters.

Filter order (all must pass to enter):
  1. Event Risk     — block entry around FOMC / CPI / NFP / SPY ex-div dates
  2. VIX gate       — VIX must be within [MIN_VIX, MAX_VIX]
  2b. VIX spike     — VIX not >1.25× its 5-day average (early panic guard)
  3. IV Rank        — VIX must be elevated vs its own 52-week range
  3b. Large move    — SPY not moved >2% from prior close (gap guard)
  4. SPY Trend      — SMA20 determines put / call / strangle bias
  5. Support/Res.   — block puts near 52-week low, calls near 52-week high
  6. Skew           — skip puts if calls are significantly more expensive
  7. PDT            — max 3 round-trips per rolling week
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import TYPE_CHECKING

from ib_insync import IB, Index, Stock

from logger import logger
import config
from guard_tracker import GuardTracker

if TYPE_CHECKING:
    from pdt_tracker import PDTTracker


# ---------------------------------------------------------------------------
# Known high-impact event dates — block entry on these days and the day before
# Update annually.
# ---------------------------------------------------------------------------
_EVENT_DATES: set[date] = {
    # FOMC 2026
    date(2026, 1, 29), date(2026, 3, 19), date(2026, 5, 7),
    date(2026, 6, 18), date(2026, 7, 30), date(2026, 9, 17),
    date(2026, 10, 29), date(2026, 12, 10),
    # CPI 2026 (approximate — 2nd Wed of each month)
    date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
    date(2026, 4, 10), date(2026, 5, 13), date(2026, 6, 10),
    date(2026, 7, 14), date(2026, 8, 12), date(2026, 9, 11),
    date(2026, 10, 13), date(2026, 11, 12), date(2026, 12, 10),
    # NFP 2026 (first Friday of each month)
    date(2026, 1, 9), date(2026, 2, 6), date(2026, 3, 6),
    date(2026, 4, 3), date(2026, 5, 1), date(2026, 6, 5),
    date(2026, 7, 10), date(2026, 8, 7), date(2026, 9, 4),
    date(2026, 10, 2), date(2026, 11, 6), date(2026, 12, 4),
    # SPY Ex-Dividend dates 2026 — short call positions risk early assignment
    # the day before ex-div as call holders exercise to capture the dividend.
    # Verify exact dates at etf.com before each quarter.
    date(2026, 3, 20),   # Q1 ex-div (approximate mid-March)
    date(2026, 6, 19),   # Q2 ex-div (approximate mid-June)
    date(2026, 9, 18),   # Q3 ex-div (approximate mid-September)
    date(2026, 12, 18),  # Q4 ex-div (approximate mid-December)
}


class Strategy(str, Enum):
    SELL_PUT = "sell_put"           # Bullish / uptrend
    SELL_CALL = "sell_call"         # Bearish / downtrend
    SELL_STRANGLE = "sell_strangle" # Neutral
    NO_TRADE = "no_trade"


@dataclass
class SignalResult:
    strategy: Strategy
    vix: float | None
    spy_price: float | None
    sma20: float | None
    trend: str                      # 'up', 'down', 'neutral', 'unknown'
    reason: str
    iv_rank: float | None = None    # 0.0–1.0; None if unavailable
    skew: float | None = None       # put_iv - call_iv estimate; None if unavailable
    week_high: float | None = None  # SPY 52-week high
    week_low: float | None = None   # SPY 52-week low


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_strategy(trend: str, pdt: "PDTTracker | None" = None) -> Strategy | None:
    """Map trend + PDT slots to a concrete strategy.

    Returns None if PDT limit is reached.
    """
    slots = pdt.slots_remaining() if pdt is not None else config.MAX_WEEKLY_TRADES

    if slots <= 0:
        logger.warning("PDT limit reached — no new trades this week")
        return None

    if trend == "up":
        return Strategy.SELL_PUT

    if trend == "down":
        return Strategy.SELL_CALL

    if trend == "neutral":
        if slots >= 2:
            return Strategy.SELL_STRANGLE
        logger.warning(
            f"Only {slots}/3 PDT slot(s) left — "
            "downgrading strangle to sell_put to preserve PDT headroom"
        )
        return Strategy.SELL_PUT

    return None


async def evaluate_signals(
    ib: IB,
    pdt: "PDTTracker | None" = None,
    guards: GuardTracker | None = None,
) -> SignalResult:
    """Run all filters and return a PDT-aware trading signal."""

    # 1. Event risk — fastest check, no IBKR call needed
    event_block, event_reason = _check_event_risk()
    if event_block:
        logger.warning(f"Event risk block: {event_reason}")
        return SignalResult(Strategy.NO_TRADE, None, None, None, "unknown", event_reason)

    # Guard tracker: check persistent cooldown from prior spike/move
    if guards is not None and guards.is_blocked():
        msg = f"Guard cooldown active: {guards.block_reason()}"
        logger.warning(msg)
        return SignalResult(Strategy.NO_TRADE, None, None, None, "unknown", msg)

    # 2. Fetch market data in parallel
    vix, (spy_price, sma20, week_high, week_low, prev_close, spy_open), iv_rank = \
        await asyncio.gather(
            _get_vix(ib),
            _get_spy_data(ib),
            _get_iv_rank(ib),
        )

    # 3. VIX gate
    if vix is None:
        msg = "VIX data unavailable — blocking entry"
        logger.warning(msg)
        return SignalResult(Strategy.NO_TRADE, None, spy_price, sma20, "unknown", msg,
                            iv_rank, None, week_high, week_low)

    if vix < config.MIN_VIX:
        msg = f"VIX={vix:.2f} < {config.MIN_VIX} minimum — premium too thin"
        logger.info(msg)
        return SignalResult(Strategy.NO_TRADE, vix, spy_price, sma20, "unknown", msg,
                            iv_rank, None, week_high, week_low)

    if vix > config.MAX_VIX:
        msg = f"VIX={vix:.2f} > {config.MAX_VIX} circuit breaker — panic spike"
        logger.warning(msg)
        return SignalResult(Strategy.NO_TRADE, vix, spy_price, sma20, "unknown", msg,
                            iv_rank, None, week_high, week_low)

    # 3b. VIX spike guard — VIX accelerating >1.25× its own 5-day average
    vix_5d_avg = await _get_vix_5day_avg(ib)
    if vix_5d_avg is not None and vix > vix_5d_avg * config.VIX_SPIKE_MULTIPLIER:
        msg = (
            f"VIX spike guard: VIX={vix:.2f} > "
            f"{config.VIX_SPIKE_MULTIPLIER}× 5-day avg {vix_5d_avg:.2f} — "
            f"pausing {config.VIX_SPIKE_SKIP_DAYS} days"
        )
        logger.warning(msg)
        if guards is not None:
            guards.set_vix_skip(config.VIX_SPIKE_SKIP_DAYS)
        return SignalResult(Strategy.NO_TRADE, vix, spy_price, sma20, "unknown", msg,
                            iv_rank, None, week_high, week_low)

    # 4. IV Rank gate
    if iv_rank is not None and iv_rank < config.MIN_IV_RANK:
        msg = (
            f"IV Rank={iv_rank:.2f} < {config.MIN_IV_RANK} minimum — "
            "VIX historically low, premium not worth the risk"
        )
        logger.info(msg)
        return SignalResult(Strategy.NO_TRADE, vix, spy_price, sma20, "unknown", msg,
                            iv_rank, None, week_high, week_low)

    if iv_rank is not None:
        logger.info(f"IV Rank: {iv_rank:.2f} (VIX={vix:.2f}) — premium environment acceptable")

    # 4b. Large move guard — SPY gapped or moved >2% from prior close
    if spy_open is not None and prev_close is not None and prev_close > 0:
        move_pct = abs(spy_open - prev_close) / prev_close
        if move_pct > config.LARGE_MOVE_PCT:
            msg = (
                f"Large move guard: SPY moved {move_pct:.1%} from prior close ${prev_close:.2f} "
                f"to open ${spy_open:.2f} — pausing {config.LARGE_MOVE_SKIP_DAYS} days"
            )
            logger.warning(msg)
            if guards is not None:
                guards.set_move_skip(config.LARGE_MOVE_SKIP_DAYS)
            return SignalResult(Strategy.NO_TRADE, vix, spy_price, sma20, "unknown", msg,
                                iv_rank, None, week_high, week_low)

    # 5. SPY trend
    if spy_price is None or sma20 is None:
        trend = "neutral"
        trend_reason = "SPY trend unavailable — defaulting to neutral"
        logger.warning(trend_reason)
    elif spy_price > sma20 * 1.002:
        trend = "up"
        trend_reason = f"SPY ${spy_price:.2f} > SMA20 ${sma20:.2f} → uptrend"
    elif spy_price < sma20 * 0.998:
        trend = "down"
        trend_reason = f"SPY ${spy_price:.2f} < SMA20 ${sma20:.2f} → downtrend"
    else:
        trend = "neutral"
        trend_reason = f"SPY ${spy_price:.2f} ≈ SMA20 ${sma20:.2f} → neutral"

    logger.info(f"Trend: {trend_reason}")

    # 6. Support / resistance gate
    sr_block, sr_reason, blocked_side = _check_support_resistance(spy_price, week_high, week_low)
    if sr_block and blocked_side == "both":
        return SignalResult(Strategy.NO_TRADE, vix, spy_price, sma20, trend, sr_reason,
                            iv_rank, None, week_high, week_low)

    # Adjust trend if one side is blocked by S/R
    if sr_block and blocked_side == "put" and trend in ("up", "neutral"):
        logger.warning(f"S/R block on puts: {sr_reason} — downgrading to sell_call")
        trend = "down"
    elif sr_block and blocked_side == "call" and trend in ("down", "neutral"):
        logger.warning(f"S/R block on calls: {sr_reason} — downgrading to sell_put")
        trend = "up"

    # 7. Skew estimate (uses VIX term structure as proxy — live chain skew computed in option_chain)
    skew = _estimate_skew(vix)
    if skew is not None and skew < config.MIN_PUT_CALL_SKEW and trend in ("up", "neutral"):
        msg = (
            f"Skew={skew:.3f} inverted (calls more expensive than puts) — "
            "skipping put-side entry"
        )
        logger.warning(msg)
        if trend == "up":
            trend = "down"   # Flip to sell call instead
        else:
            trend = "down"

    # 8. PDT-aware strategy selection
    strategy = select_strategy(trend, pdt)
    if strategy is None:
        msg = f"PDT limit reached | trend={trend}"
        return SignalResult(Strategy.NO_TRADE, vix, spy_price, sma20, trend, msg,
                            iv_rank, skew, week_high, week_low)

    reason = f"{trend_reason} → {strategy.value}"
    logger.info(
        f"Signal: {strategy.value} | VIX={vix:.2f} | IVR={iv_rank:.2f if iv_rank else 'n/a'} "
        f"| skew={skew:.3f if skew else 'n/a'} | {reason}"
    )
    return SignalResult(strategy, vix, spy_price, sma20, trend, reason,
                        iv_rank, skew, week_high, week_low)


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _check_event_risk() -> tuple[bool, str]:
    """Return (blocked, reason). Blocks entry on event day and the day before."""
    today = date.today()
    for event_date in _EVENT_DATES:
        for offset in range(config.EVENT_BLACKOUT_DAYS + 1):
            if today == event_date - timedelta(days=offset):
                days_away = offset
                label = "today" if days_away == 0 else f"in {days_away} day(s)"
                return True, f"Event risk blackout — major event {label} ({event_date})"
    return False, ""


def _check_support_resistance(
    spy_price: float | None,
    week_high: float | None,
    week_low: float | None,
) -> tuple[bool, str, str]:
    """Return (blocked, reason, side).

    side is 'put', 'call', or 'both'.
    """
    if spy_price is None or week_high is None or week_low is None:
        return False, "", ""

    put_threshold = week_low * (1 + config.SR_LOW_BUFFER_PCT)
    call_threshold = week_high * (1 - config.SR_HIGH_BUFFER_PCT)

    near_low = spy_price <= put_threshold
    near_high = spy_price >= call_threshold

    if near_low and near_high:
        return True, (
            f"SPY ${spy_price:.2f} compressed between 52w low ${week_low:.2f} "
            f"and high ${week_high:.2f} — no entry"
        ), "both"

    if near_low:
        return True, (
            f"SPY ${spy_price:.2f} within {config.SR_LOW_BUFFER_PCT*100:.0f}% "
            f"of 52-week low ${week_low:.2f} — skipping puts"
        ), "put"

    if near_high:
        return True, (
            f"SPY ${spy_price:.2f} within {config.SR_HIGH_BUFFER_PCT*100:.0f}% "
            f"of 52-week high ${week_high:.2f} — skipping calls"
        ), "call"

    logger.debug(
        f"S/R clear — SPY ${spy_price:.2f} | 52w range ${week_low:.2f}–${week_high:.2f}"
    )
    return False, "", ""


def _estimate_skew(vix: float) -> float | None:
    """Estimate put-call skew from VIX term structure.

    Real skew requires ATM put IV minus ATM call IV from the option chain.
    As a fast proxy: if VIX is elevated the put skew is typically positive
    (puts more expensive). We return a rough signed estimate.

    Returns None if skew cannot be estimated.
    Note: option_chain.py computes precise skew from actual chain data and logs it.
    """
    # When VIX > 20, put skew is almost always positive — normal condition.
    # When VIX < 14, skew often compresses or inverts in low-fear environments.
    # This is a directional hint only; precise skew is logged per chain fetch.
    if vix >= 20:
        return 0.04   # Normal positive put skew
    if vix >= 14:
        return 0.01   # Mild skew
    return -0.01      # Low-vol environment, mild inversion possible


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

async def _get_vix(ib: IB) -> float | None:
    """Fetch current VIX level."""
    try:
        vix_contract = Index("VIX", "CBOE", "USD")
        qualified = await ib.qualifyContractsAsync(vix_contract)
        if not qualified:
            return None
        ticker = ib.reqMktData(qualified[0], "", snapshot=True)
        await asyncio.sleep(2)
        ib.cancelMktData(qualified[0])
        price = ticker.last or ticker.close
        if price and price > 0:
            logger.debug(f"VIX: {price:.2f}")
            return float(price)
    except Exception as exc:
        logger.error(f"Error fetching VIX: {exc}")
    return None


async def _get_iv_rank(ib: IB) -> float | None:
    """Compute IV Rank from 1-year VIX daily bars.

    IV Rank = (current_vix - 52w_low) / (52w_high - 52w_low)
    Range: 0.0 (VIX at yearly low) → 1.0 (VIX at yearly high).
    """
    try:
        vix_contract = Index("VIX", "CBOE", "USD")
        qualified = await ib.qualifyContractsAsync(vix_contract)
        if not qualified:
            return None

        bars = await ib.reqHistoricalDataAsync(
            qualified[0],
            endDateTime="",
            durationStr="1 Y",
            barSizeSetting="1 day",
            whatToShow="MIDPOINT",
            useRTH=True,
            formatDate=1,
        )
        if len(bars) < 20:
            logger.warning(f"Insufficient VIX history for IV Rank: {len(bars)} bars")
            return None

        closes = [b.close for b in bars]
        current = closes[-1]
        low_52w = min(closes)
        high_52w = max(closes)

        if high_52w == low_52w:
            return None

        iv_rank = (current - low_52w) / (high_52w - low_52w)
        logger.debug(
            f"IV Rank: {iv_rank:.2f} | VIX={current:.2f} | "
            f"52w range {low_52w:.2f}–{high_52w:.2f}"
        )
        return round(iv_rank, 3)

    except Exception as exc:
        logger.error(f"Error computing IV Rank: {exc}")
        return None


async def _get_vix_5day_avg(ib: IB) -> float | None:
    """Fetch 5-day VIX close average for spike guard."""
    try:
        vix_contract = Index("VIX", "CBOE", "USD")
        qualified = await ib.qualifyContractsAsync(vix_contract)
        if not qualified:
            return None
        bars = await ib.reqHistoricalDataAsync(
            qualified[0],
            endDateTime="",
            durationStr="10 D",
            barSizeSetting="1 day",
            whatToShow="MIDPOINT",
            useRTH=True,
            formatDate=1,
        )
        if len(bars) < 5:
            return None
        avg = sum(b.close for b in bars[-5:]) / 5
        logger.debug(f"VIX 5-day avg: {avg:.2f}")
        return round(avg, 2)
    except Exception as exc:
        logger.error(f"Error fetching VIX 5-day avg: {exc}")
        return None


async def _get_spy_data(ib: IB) -> tuple[
    float | None, float | None, float | None, float | None, float | None, float | None
]:
    """Fetch SPY price, SMA20, 52-week high, 52-week low, prev_close, today_open.

    Returns (spy_price, sma20, week_high, week_low, prev_close, spy_open).
    prev_close and spy_open are used for the large move guard.
    """
    try:
        spy = Stock(config.SYMBOL, config.EXCHANGE, config.CURRENCY)
        qualified = await ib.qualifyContractsAsync(spy)
        if not qualified:
            return None, None, None, None, None, None
        spy_q = qualified[0]

        ticker = ib.reqMktData(spy_q, "", snapshot=True)
        await asyncio.sleep(2)
        ib.cancelMktData(spy_q)
        spy_price = ticker.last or ticker.close
        if not spy_price or spy_price <= 0:
            return None, None, None, None, None, None

        bars = await ib.reqHistoricalDataAsync(
            spy_q,
            endDateTime="",
            durationStr="1 Y",
            barSizeSetting="1 day",
            whatToShow="MIDPOINT",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            return float(spy_price), None, None, None, None, None

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        opens = [b.open for b in bars]

        sma20 = round(sum(closes[-config.TREND_SMA_DAYS:]) / config.TREND_SMA_DAYS, 2) \
            if len(closes) >= config.TREND_SMA_DAYS else None

        week_high = round(max(highs), 2)
        week_low = round(min(lows), 2)

        # For large move guard: today's open vs yesterday's close
        prev_close = round(closes[-2], 2) if len(closes) >= 2 else None
        spy_open = round(opens[-1], 2) if opens else None

        logger.debug(
            f"SPY ${spy_price:.2f} | open ${spy_open} | prev_close ${prev_close} | "
            f"SMA20 ${sma20} | 52w {week_low:.2f}–{week_high:.2f}"
        )
        return float(spy_price), sma20, week_high, week_low, prev_close, spy_open

    except Exception as exc:
        logger.error(f"Error fetching SPY data: {exc}")
        return None, None, None, None, None, None
