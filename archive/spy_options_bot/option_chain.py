"""SPY option chain fetching and contract selection.

Responsible for:
- Fetching available expirations and strikes from IBKR
- Requesting market data (Greeks, bid/ask, OI, volume) for candidates
- Filtering contracts by: delta, expected move, theta/delta efficiency,
  liquidity (volume + OI), spread, and skew
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, timedelta

from ib_insync import IB, Option, Stock

from logger import logger
import config

_GREEKS_MAX_RETRIES = 3
_GREEKS_RETRY_DELAY = 2.0  # seconds between retries


def _greeks_valid(ticker) -> bool:
    """Return True only if all required Greeks are present, numeric, and non-zero."""
    greek = ticker.modelGreeks
    if greek is None:
        return False
    required = [greek.delta, greek.theta, greek.vega, greek.impliedVol]
    if any(v is None or v != v for v in required):  # NaN check via v != v
        return False
    if greek.impliedVol <= 0 or greek.vega <= 0:
        return False
    return True


@dataclass
class OptionCandidate:
    contract: Option
    expiry: str          # YYYYMMDD
    strike: float
    right: str           # 'P' or 'C'
    delta: float
    theta: float
    vega: float
    iv: float
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    spread_pct: float    # (ask - bid) / mid
    # Credit spread fields — populated by fetch_hedge_leg(), None if not yet fetched
    hedge_contract: Option | None = None
    hedge_strike: float = 0.0
    hedge_mid: float = 0.0
    net_credit: float = 0.0   # mid - hedge_mid (net credit collected)


def _next_friday() -> date:
    """Return the date of the next (or current) Friday."""
    today = date.today()
    days_ahead = 4 - today.weekday()  # Friday = weekday 4
    if days_ahead <= 0:
        days_ahead += 7
    return today + timedelta(days=days_ahead)


def _trading_days_to(target: date) -> int:
    """Approximate DTE in trading days."""
    today = date.today()
    count = 0
    current = today
    while current < target:
        if current.weekday() < 5:  # Mon–Fri
            count += 1
        current += timedelta(days=1)
    return count


async def get_spy_conid(ib: IB) -> int:
    """Qualify SPY stock contract and return its conId."""
    spy = Stock(config.SYMBOL, config.EXCHANGE, config.CURRENCY)
    contracts = await ib.qualifyContractsAsync(spy)
    if not contracts:
        raise ValueError("Could not qualify SPY stock contract")
    return contracts[0].conId


async def fetch_option_chain(ib: IB) -> list[OptionCandidate]:
    """Fetch SPY weekly option chain and return filtered candidates.

    Steps:
      1. Qualify SPY stock to get conId
      2. Call reqSecDefOptParams to get expirations & strikes
      3. Filter to weekly expiries within DTE window
      4. Request market data for OTM candidates near target delta
      5. Apply all filters and return qualifying OptionCandidate objects
    """
    spy_conid = await get_spy_conid(ib)

    # Step 2: Get chain structure
    logger.info("Requesting SPY option chain parameters...")
    chains = await ib.reqSecDefOptParamsAsync(
        underlyingSymbol=config.SYMBOL,
        futFopExchange="",
        underlyingSecType="STK",
        underlyingConId=spy_conid,
    )

    if not chains:
        logger.error("No option chain returned from IBKR")
        return []

    # Use SMART exchange chain
    chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
    logger.debug(
        f"Chain: {len(chain.expirations)} expirations, {len(chain.strikes)} strikes"
    )

    # Step 3: Identify target expiry (nearest Friday within DTE window)
    target_friday = _next_friday()
    dte = _trading_days_to(target_friday)

    if not (config.MIN_DTE <= dte <= config.MAX_DTE):
        logger.warning(
            f"Next Friday DTE={dte} is outside [{config.MIN_DTE}, {config.MAX_DTE}] window. "
            "Skipping chain fetch."
        )
        return []

    expiry_str = target_friday.strftime("%Y%m%d")
    if expiry_str not in chain.expirations:
        logger.warning(f"Target expiry {expiry_str} not found in chain expirations")
        return []

    logger.info(f"Target expiry: {expiry_str} (DTE={dte})")

    # Step 4: Get current SPY price for OTM strike selection
    spy_stock = Stock(config.SYMBOL, config.EXCHANGE, config.CURRENCY)
    ib.qualifyContracts(spy_stock)
    spy_ticker = ib.reqMktData(spy_stock, "", snapshot=True)
    await asyncio.sleep(2)
    spy_price = spy_ticker.last or spy_ticker.close or spy_ticker.marketPrice()
    ib.cancelMktData(spy_stock)

    if not spy_price or spy_price <= 0:
        logger.error("Could not determine SPY price — skipping chain fetch")
        return []

    logger.info(f"SPY current price: ${spy_price:.2f}")

    # Build candidate strikes: OTM puts (below spot) and calls (above spot)
    # Aim for roughly 10–15% OTM range to bracket the target delta zone
    put_strikes = [s for s in chain.strikes if spy_price * 0.88 <= s <= spy_price * 0.97]
    call_strikes = [s for s in chain.strikes if spy_price * 1.03 <= s <= spy_price * 1.12]

    candidates: list[OptionCandidate] = []

    for right, strikes in (("P", put_strikes), ("C", call_strikes)):
        logger.debug(f"Scanning {len(strikes)} {right} strikes for {expiry_str}")
        for strike in strikes:
            opt = Option(
                config.SYMBOL,
                expiry_str,
                strike,
                right,
                config.EXCHANGE,
                multiplier="100",
                currency=config.CURRENCY,
            )
            try:
                qualified = await ib.qualifyContractsAsync(opt)
                if not qualified:
                    continue
                opt = qualified[0]
            except Exception as exc:
                logger.debug(f"Could not qualify {right}{strike}: {exc}")
                continue

            # Request market data with Greeks — retry up to 3× if Greeks missing
            ticker = ib.reqMktData(opt, genericTickList="100,101", snapshot=False)
            greeks_populated = False
            for attempt in range(_GREEKS_MAX_RETRIES):
                await asyncio.sleep(_GREEKS_RETRY_DELAY)
                if _greeks_valid(ticker):
                    greeks_populated = True
                    break
                if attempt < _GREEKS_MAX_RETRIES - 1:
                    logger.debug(
                        f"Greeks not yet populated for {right}{strike} {expiry_str} "
                        f"(attempt {attempt + 1}/{_GREEKS_MAX_RETRIES}) — retrying"
                    )

            if not greeks_populated:
                logger.warning(
                    f"Skipping {right}{strike} {expiry_str}: Greeks unavailable after "
                    f"{_GREEKS_MAX_RETRIES} attempts "
                    f"(modelGreeks={ticker.modelGreeks})"
                )
                ib.cancelMktData(opt)
                continue

            try:
                candidate = _extract_candidate(opt, ticker, expiry_str, strike, right)
                if candidate:
                    candidates.append(candidate)
            finally:
                ib.cancelMktData(opt)

    logger.info(f"Raw candidates before filtering: {len(candidates)}")
    filtered = _apply_filters(candidates, spy_price=spy_price, dte=dte)
    _log_skew(filtered)
    logger.info(f"Candidates after filtering: {len(filtered)}")
    return filtered


def _extract_candidate(
    contract: Option,
    ticker,
    expiry: str,
    strike: float,
    right: str,
) -> OptionCandidate | None:
    """Extract Greeks and market data from ticker; return None if data missing.

    Callers must have already confirmed _greeks_valid(ticker) before calling this.
    """
    greeks = ticker.modelGreeks  # Guaranteed non-None by _greeks_valid check upstream

    bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
    ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0

    if bid <= 0 or ask <= 0:
        return None

    mid = (bid + ask) / 2
    spread = ask - bid
    spread_pct = spread / mid if mid > 0 else 999.0

    volume = ticker.volume or 0
    oi = ticker.optionOpenInterest or 0

    return OptionCandidate(
        contract=contract,
        expiry=expiry,
        strike=strike,
        right=right,
        delta=greeks.delta,
        theta=greeks.theta,
        vega=greeks.vega,
        iv=greeks.impliedVol,
        bid=bid,
        ask=ask,
        mid=mid,
        volume=int(volume),
        open_interest=int(oi),
        spread_pct=spread_pct,
    )


def _expected_move(spy_price: float, iv: float, dte_trading_days: int) -> float:
    """1-sigma expected move in dollars.

    Uses actual IV from ATM options if available, else falls back to the
    candidate's own IV. Formula: price × iv × sqrt(dte / 365).
    """
    import math
    dte_years = dte_trading_days / 252  # trading-day convention
    return spy_price * iv * math.sqrt(dte_years)


def _apply_filters(
    candidates: list[OptionCandidate],
    spy_price: float,
    dte: int,
) -> list[OptionCandidate]:
    """Apply all selection criteria filters."""
    # ATM IV estimate: median IV of all raw candidates (before delta filter)
    if candidates:
        atm_iv = sorted(c.iv for c in candidates)[len(candidates) // 2]
    else:
        atm_iv = 0.15  # fallback

    em = _expected_move(spy_price, atm_iv, dte)
    logger.info(
        f"Expected move: ±${em:.2f} (SPY=${spy_price:.2f}, ATM_IV={atm_iv:.3f}, DTE={dte})"
    )

    result = []
    for c in candidates:
        # 1. Delta filter
        if c.right == "P":
            target = config.TARGET_DELTA_PUT
            if not (target - config.DELTA_TOLERANCE <= c.delta <= target + config.DELTA_TOLERANCE):
                continue
        else:
            target = config.TARGET_DELTA_CALL
            if not (target - config.DELTA_TOLERANCE <= c.delta <= target + config.DELTA_TOLERANCE):
                continue

        # 2. Expected move — strike must be at or beyond EXPECTED_MOVE_BUFFER × EM from spot
        min_distance = em * config.EXPECTED_MOVE_BUFFER
        actual_distance = abs(c.strike - spy_price)
        if actual_distance < min_distance:
            logger.debug(
                f"Skip {c.right}{c.strike}: distance ${actual_distance:.2f} "
                f"< expected move ${min_distance:.2f}"
            )
            continue

        # 3. Theta filter — must be decaying (negative theta)
        if c.theta > config.MIN_THETA:
            continue

        # 4. Theta / delta efficiency ratio
        #    theta is negative, delta is positive for calls / negative for puts
        #    ratio = |theta| / |delta| — higher = more decay per unit of directional risk
        if c.delta != 0:
            ratio = abs(c.theta) / abs(c.delta)
            if ratio < config.MIN_THETA_DELTA_RATIO:
                logger.debug(
                    f"Skip {c.right}{c.strike}: theta/delta={ratio:.3f} "
                    f"< {config.MIN_THETA_DELTA_RATIO} minimum"
                )
                continue

        # 4b. Vega filter — cap IV-expansion exposure on the short leg.
        #     vega × 100 = dollars lost per 1-point VIX increase per contract.
        #     The hedge partially offsets this, but short leg drives the exposure.
        vega_loss = c.vega * 100.0
        if vega_loss > config.MAX_VEGA_LOSS_PER_VIX_POINT:
            logger.debug(
                f"Skip {c.right}{c.strike}: vega loss ${vega_loss:.2f}/VIX pt "
                f"> ${config.MAX_VEGA_LOSS_PER_VIX_POINT:.0f} maximum"
            )
            continue

        # 5. Liquidity filters
        if c.open_interest < config.MIN_OPEN_INTEREST:
            continue
        if c.volume < config.MIN_VOLUME:
            continue

        # 6. Spread filter
        if c.spread_pct > config.MAX_SPREAD_PCT:
            continue
        if (c.ask - c.bid) > config.MAX_SPREAD_ABS:
            continue

        result.append(c)

    return result


def _log_skew(candidates: list[OptionCandidate]) -> None:
    """Log put-call IV skew from filtered candidates."""
    puts = [c for c in candidates if c.right == "P"]
    calls = [c for c in candidates if c.right == "C"]
    if not puts or not calls:
        return
    avg_put_iv = sum(c.iv for c in puts) / len(puts)
    avg_call_iv = sum(c.iv for c in calls) / len(calls)
    skew = avg_put_iv - avg_call_iv
    direction = "normal (puts richer)" if skew > 0 else "inverted (calls richer)"
    logger.info(
        f"Chain skew: {skew:+.4f} — {direction} "
        f"(put_iv={avg_put_iv:.4f}, call_iv={avg_call_iv:.4f})"
    )


def select_best_put(candidates: list[OptionCandidate]) -> OptionCandidate | None:
    """Select the put with best theta/delta efficiency closest to target delta."""
    puts = [c for c in candidates if c.right == "P"]
    if not puts:
        return None
    return min(
        puts,
        key=lambda c: (
            abs(c.delta - config.TARGET_DELTA_PUT),
            -(abs(c.theta) / abs(c.delta)) if c.delta != 0 else 0,
        ),
    )


def select_best_call(candidates: list[OptionCandidate]) -> OptionCandidate | None:
    """Select the call with best theta/delta efficiency closest to target delta."""
    calls = [c for c in candidates if c.right == "C"]
    if not calls:
        return None
    return min(
        calls,
        key=lambda c: (
            abs(c.delta - config.TARGET_DELTA_CALL),
            -(abs(c.theta) / abs(c.delta)) if c.delta != 0 else 0,
        ),
    )


async def fetch_hedge_leg(ib: IB, candidate: OptionCandidate) -> OptionCandidate:
    """Find the protective hedge leg for a credit spread and attach it to the candidate.

    For a put spread: buy a put SPREAD_WIDTH strikes below the short put.
    For a call spread: buy a call SPREAD_WIDTH strikes above the short call.

    Returns the candidate with hedge fields populated. If the hedge cannot be
    found, returns candidate unchanged (net_credit = mid = naked position).
    """
    width = int(config.SPREAD_WIDTH)
    if candidate.right == "P":
        hedge_strike = round(candidate.strike - width)
    else:
        hedge_strike = round(candidate.strike + width)

    opt = Option(
        config.SYMBOL,
        candidate.expiry,
        hedge_strike,
        candidate.right,
        config.EXCHANGE,
        multiplier="100",
        currency=config.CURRENCY,
    )
    try:
        qualified = await ib.qualifyContractsAsync(opt)
        if not qualified:
            logger.warning(f"Could not qualify hedge leg {candidate.right}{hedge_strike}")
            candidate.net_credit = candidate.mid
            return candidate

        hedge_contract = qualified[0]
        ticker = ib.reqMktData(hedge_contract, genericTickList="100,101", snapshot=False)

        for attempt in range(_GREEKS_MAX_RETRIES):
            await asyncio.sleep(_GREEKS_RETRY_DELAY)
            bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0
            if bid > 0 and ask > 0:
                break

        ib.cancelMktData(hedge_contract)

        bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0.0
        ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0.0
        if bid <= 0 or ask <= 0:
            logger.warning(f"No valid market data for hedge {candidate.right}{hedge_strike}")
            candidate.net_credit = candidate.mid
            return candidate

        hedge_mid = (bid + ask) / 2
        net_credit = round(candidate.mid - hedge_mid, 4)

        candidate.hedge_contract = hedge_contract
        candidate.hedge_strike = float(hedge_strike)
        candidate.hedge_mid = round(hedge_mid, 4)
        candidate.net_credit = net_credit

        logger.info(
            f"Spread: SELL {candidate.right}{candidate.strike} ${candidate.mid:.2f} | "
            f"BUY {candidate.right}{hedge_strike} ${hedge_mid:.2f} | "
            f"net credit ${net_credit:.2f}"
        )

        # Apply minimum net credit floor
        if net_credit < config.MIN_NET_CREDIT:
            logger.info(
                f"Net credit ${net_credit:.2f} < floor ${config.MIN_NET_CREDIT:.2f} — "
                "skipping this week"
            )
            return None  # type: ignore[return-value]

    except Exception as exc:
        logger.error(f"Error fetching hedge leg: {exc}")
        candidate.net_credit = candidate.mid

    return candidate
