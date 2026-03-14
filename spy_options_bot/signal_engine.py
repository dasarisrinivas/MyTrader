"""Signal engine — determines strategy direction based on VIX and SPY trend.

Outputs one of: 'sell_put', 'sell_call', 'sell_strangle', or 'no_trade'.

PDT-aware strategy selection:
- Strangle requires 2 available PDT slots; degrades to sell_put if only 1 slot left.
- Pass a PDTTracker to evaluate_signals() to enable this gate.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ib_insync import IB, Index, Stock

from logger import logger
import config

if TYPE_CHECKING:
    from pdt_tracker import PDTTracker


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
    trend: str           # 'up', 'down', 'neutral', 'unknown'
    reason: str


def select_strategy(trend: str, pdt: "PDTTracker | None" = None) -> Strategy | None:
    """Map trend + PDT slots to a concrete strategy, with graceful degradation.

    Returns None if PDT limit is reached and no trade should be placed.

    Args:
        trend:  'up', 'down', or 'neutral'
        pdt:    Optional PDTTracker; if None, slot checks are skipped.
    """
    slots = pdt.slots_remaining() if pdt is not None else config.MAX_WEEKLY_TRADES

    if slots <= 0:
        logger.warning("PDT limit reached — no new trades this week")
        return None

    if trend == "up":
        return Strategy.SELL_PUT

    if trend == "down":
        return Strategy.SELL_CALL

    # Neutral trend → strangle if 2+ slots available, else degrade to sell_put
    if trend == "neutral":
        if slots >= 2:
            return Strategy.SELL_STRANGLE
        else:
            logger.warning(
                f"Only {slots}/3 PDT slot(s) left — "
                "downgrading strangle to sell_put to preserve PDT headroom"
            )
            return Strategy.SELL_PUT

    return None  # Unreachable for valid trend values


async def evaluate_signals(
    ib: IB,
    pdt: "PDTTracker | None" = None,
) -> SignalResult:
    """Evaluate VIX level, SPY trend, and return a PDT-aware trading signal.

    Args:
        ib:  Connected IB instance.
        pdt: Optional PDTTracker for strangle slot gating.
    """
    vix = await _get_vix(ib)
    spy_price, sma20 = await _get_spy_trend(ib)

    # VIX gate
    if vix is None:
        logger.warning("VIX data unavailable — blocking new entries")
        return SignalResult(Strategy.NO_TRADE, vix, spy_price, sma20, "unknown",
                            "VIX data unavailable")

    if vix < config.MIN_VIX:
        msg = f"VIX={vix:.2f} < {config.MIN_VIX} minimum — not selling premium in low-vol"
        logger.info(msg)
        return SignalResult(Strategy.NO_TRADE, vix, spy_price, sma20, "unknown", msg)

    if vix > config.MAX_VIX:
        msg = f"VIX={vix:.2f} > {config.MAX_VIX} circuit breaker — skipping entry on panic-spike day"
        logger.warning(msg)
        return SignalResult(Strategy.NO_TRADE, vix, spy_price, sma20, "unknown", msg)

    # Determine raw trend
    if spy_price is None or sma20 is None:
        logger.warning("SPY trend data unavailable — defaulting to neutral")
        trend = "neutral"
        trend_reason = "SPY trend unavailable; defaulting to neutral"
    elif spy_price > sma20:
        trend = "up"
        trend_reason = f"SPY ${spy_price:.2f} > SMA20 ${sma20:.2f} → uptrend"
    elif spy_price < sma20:
        trend = "down"
        trend_reason = f"SPY ${spy_price:.2f} < SMA20 ${sma20:.2f} → downtrend"
    else:
        trend = "neutral"
        trend_reason = f"SPY ${spy_price:.2f} ≈ SMA20 ${sma20:.2f} → neutral"

    # PDT-aware strategy selection
    strategy = select_strategy(trend, pdt)
    if strategy is None:
        msg = f"PDT limit reached | trend={trend} | {trend_reason}"
        return SignalResult(Strategy.NO_TRADE, vix, spy_price, sma20, trend, msg)

    reason = f"{trend_reason} → {strategy.value}"
    logger.info(f"Signal: {strategy.value} | VIX={vix:.2f} | {reason}")
    return SignalResult(strategy, vix, spy_price, sma20, trend, reason)


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


async def _get_spy_trend(ib: IB) -> tuple[float | None, float | None]:
    """Fetch SPY current price and 20-day SMA.

    Returns (current_price, sma20).
    """
    try:
        spy = Stock(config.SYMBOL, config.EXCHANGE, config.CURRENCY)
        qualified = await ib.qualifyContractsAsync(spy)
        if not qualified:
            return None, None
        spy_q = qualified[0]

        ticker = ib.reqMktData(spy_q, "", snapshot=True)
        await asyncio.sleep(2)
        ib.cancelMktData(spy_q)
        spy_price = ticker.last or ticker.close
        if not spy_price or spy_price <= 0:
            return None, None

        bars = await ib.reqHistoricalDataAsync(
            spy_q,
            endDateTime="",
            durationStr=f"{config.TREND_SMA_DAYS + 10} D",
            barSizeSetting="1 day",
            whatToShow="MIDPOINT",
            useRTH=True,
            formatDate=1,
        )
        if len(bars) < config.TREND_SMA_DAYS:
            logger.warning(f"Insufficient historical data for SMA: {len(bars)} bars")
            return float(spy_price), None

        closes = [b.close for b in bars[-config.TREND_SMA_DAYS:]]
        sma20 = sum(closes) / len(closes)
        logger.debug(f"SPY: ${spy_price:.2f} | SMA20: ${sma20:.2f}")
        return float(spy_price), round(sma20, 2)

    except Exception as exc:
        logger.error(f"Error fetching SPY trend: {exc}")
        return None, None
