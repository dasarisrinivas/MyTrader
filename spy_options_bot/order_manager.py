"""Order placement and management for SPY options bot.

Handles:
- Placing limit sell orders at mid-price
- Adjusting price after fill timeout
- Recording positions to disk with full entry/close detail
- Marking positions closed (never deletes — full audit trail)
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from ib_insync import IB, LimitOrder, Option, Trade, Contract, ComboLeg

from logger import logger
import config
from option_chain import OptionCandidate

if TYPE_CHECKING:
    from notifier import Notifier

ET = ZoneInfo("America/New_York")
POSITION_LOG = Path("spy_options_bot/open_positions.json")

# Valid close reason values
CLOSE_REASONS = frozenset({
    "profit_target",
    "loss_stop",
    "delta_stop",
    "thursday_eod",
    "emergency_gamma",
    "manual",
})


@dataclass
class OpenPosition:
    position_id: str       # e.g. "SPY_PUT_20240119_480"
    symbol: str
    right: str             # 'P' or 'C'
    strike: float
    expiry: str            # YYYYMMDD
    quantity: int          # Negative = short
    entry_time: str        # ISO datetime with TZ
    entry_premium: float   # Credit received per share at fill
    entry_delta: float
    entry_theta: float
    entry_iv: float
    profit_target_price: float  # entry_premium * PROFIT_TARGET_PCT — computed once at entry
    stop_loss_price: float      # entry_premium * MAX_LOSS_MULTIPLE — computed once at entry
    strategy_type: str          # "single" or "strangle"
    conid: int
    order_id: int
    status: str = "open"
    close_time: str | None = None
    close_premium: float | None = None
    pnl: float | None = None
    close_reason: str | None = None
    # Credit spread fields (populated when hedge leg is used)
    is_spread: bool = False
    hedge_conid: int = 0
    hedge_strike: float = 0.0
    hedge_premium: float = 0.0   # Premium paid for the long leg
    net_credit: float = 0.0      # entry_premium - hedge_premium
    max_spread_loss: float = 0.0 # (SPREAD_WIDTH - net_credit) × 100


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_all_raw() -> list[dict]:
    """Load the raw JSON list (open + closed) from disk."""
    if not POSITION_LOG.exists():
        return []
    try:
        return json.loads(POSITION_LOG.read_text())
    except Exception as exc:
        logger.warning(f"Could not read position log: {exc}")
        return []


def _save_all_raw(records: list[dict]) -> None:
    POSITION_LOG.parent.mkdir(parents=True, exist_ok=True)
    POSITION_LOG.write_text(json.dumps(records, indent=2))


def load_open_positions() -> list[OpenPosition]:
    """Return only positions with status=='open'."""
    all_records = _load_all_raw()
    result = []
    for r in all_records:
        if r.get("status", "open") == "open":
            try:
                result.append(OpenPosition(**r))
            except Exception as exc:
                logger.warning(f"Skipping malformed position record: {exc} | {r}")
    return result


def _append_position(pos: OpenPosition) -> None:
    records = _load_all_raw()
    records.append(asdict(pos))
    _save_all_raw(records)


def mark_position_closed(
    position_id: str,
    close_premium: float,
    close_reason: str,
    qty: int,
    entry_premium: float,
) -> None:
    """Update position record in-place: set status=closed and fill close fields."""
    if close_reason not in CLOSE_REASONS:
        logger.warning(f"Unknown close_reason '{close_reason}' — storing as-is")

    records = _load_all_raw()
    for rec in records:
        if rec.get("position_id") == position_id:
            rec["status"] = "closed"
            rec["close_time"] = datetime.now(ET).isoformat()
            rec["close_premium"] = round(close_premium, 4)
            rec["pnl"] = round((entry_premium - close_premium) * 100 * abs(qty), 2)
            rec["close_reason"] = close_reason
            break
    _save_all_raw(records)


# ---------------------------------------------------------------------------
# Order manager
# ---------------------------------------------------------------------------

class OrderManager:
    def __init__(self, ib: IB, dry_run: bool = False, notifier: "Notifier | None" = None) -> None:
        self.ib = ib
        self.dry_run = dry_run
        self.notifier = notifier

    # ------------------------------------------------------------------
    # Entry orders
    # ------------------------------------------------------------------

    async def sell_option(
        self,
        candidate: OptionCandidate,
        qty: int = 1,
        strategy_type: str = "single",
    ) -> OpenPosition | None:
        """Sell a single option contract at mid-price.

        Tries limit at mid; if unfilled after ORDER_FILL_TIMEOUT seconds,
        adjusts by ORDER_ADJUST_STEP and retries once.
        """
        # Hard ceiling — never exceed MAX_CONTRACTS regardless of caller
        if qty > config.MAX_CONTRACTS:
            logger.warning(
                f"qty={qty} exceeds MAX_CONTRACTS={config.MAX_CONTRACTS} — capping"
            )
            qty = config.MAX_CONTRACTS

        contract = candidate.contract
        mid = candidate.mid
        side = f"SELL {'PUT' if candidate.right == 'P' else 'CALL'}"
        label = f"{config.SYMBOL} {candidate.expiry} {candidate.right}{candidate.strike}"

        is_spread = candidate.hedge_contract is not None and candidate.net_credit > 0
        net_credit = candidate.net_credit if is_spread else mid

        if self.dry_run:
            spread_note = f" [spread net credit ${net_credit:.2f}]" if is_spread else ""
            logger.info(f"[DRY RUN] Would place: {side} {qty}x {label} @ ${mid:.2f}{spread_note}")
            return None

        if is_spread:
            trade = await self._place_spread(candidate, qty)
            fill_price = net_credit   # For spreads, fill price = net credit received
        else:
            logger.info(f"Placing {side} {qty}x {label} @ ${mid:.2f} limit (naked)")
            trade = await self._place_limit(contract, "SELL", qty, mid)

        if trade is None:
            return None

        if is_spread:
            filled = await self._wait_for_fill(trade, None, "SELL", qty, initial_mid=net_credit)
        else:
            filled = await self._wait_for_fill(trade, contract, "SELL", qty, initial_mid=mid)

        if not filled:
            logger.warning(f"Order for {label} could not be filled — cancelling")
            self.ib.cancelOrder(trade.order)
            return None

        fill_price = trade.orderStatus.avgFillPrice or fill_price
        logger.info(
            f"Filled: {label} @ ${fill_price:.2f} | "
            f"{'spread net credit' if is_spread else 'premium received'}: "
            f"${fill_price * 100:.2f} | strategy: {strategy_type}"
        )

        position_id = f"{config.SYMBOL}_{candidate.right}_{candidate.expiry}_{int(candidate.strike)}"

        # For spreads: exits based on net credit (spread value), not raw premium
        entry_ref = fill_price if is_spread else fill_price
        max_loss_price = min(
            round(entry_ref * config.MAX_LOSS_MULTIPLE, 4),
            round(config.SPREAD_WIDTH - entry_ref, 4),
        ) if is_spread else round(entry_ref * config.MAX_LOSS_MULTIPLE, 4)

        position = OpenPosition(
            position_id=position_id,
            symbol=config.SYMBOL,
            right=candidate.right,
            strike=candidate.strike,
            expiry=candidate.expiry,
            quantity=-qty,
            entry_time=datetime.now(ET).isoformat(),
            entry_premium=fill_price,
            entry_delta=candidate.delta,
            entry_theta=candidate.theta,
            entry_iv=candidate.iv,
            profit_target_price=round(fill_price * config.PROFIT_TARGET_PCT, 4),
            stop_loss_price=max_loss_price,
            strategy_type=strategy_type,
            conid=contract.conId,
            order_id=trade.order.orderId,
            is_spread=is_spread,
            hedge_conid=candidate.hedge_contract.conId if is_spread else 0,
            hedge_strike=candidate.hedge_strike if is_spread else 0.0,
            hedge_premium=candidate.hedge_mid if is_spread else 0.0,
            net_credit=net_credit if is_spread else 0.0,
            max_spread_loss=round((config.SPREAD_WIDTH - net_credit) * 100, 2) if is_spread else 0.0,
        )

        _append_position(position)

        if self.notifier:
            self.notifier.on_trade_open(asdict(position))

        return position

    # ------------------------------------------------------------------
    # Exit orders
    # ------------------------------------------------------------------

    async def close_position(self, position: OpenPosition, reason: str) -> bool:
        """Buy back a short option to close the position.

        Returns True if successfully closed.
        """
        opt = Option(
            position.symbol,
            position.expiry,
            position.strike,
            position.right,
            config.EXCHANGE,
            multiplier="100",
            currency=config.CURRENCY,
        )
        try:
            qualified = await self.ib.qualifyContractsAsync(opt)
            if not qualified:
                logger.error(f"Could not qualify closing contract for {position.position_id}")
                return False
            contract = qualified[0]
        except Exception as exc:
            logger.error(f"Error qualifying closing contract: {exc}")
            return False

        # Get current mid price
        ticker = self.ib.reqMktData(contract, "", snapshot=True)
        await asyncio.sleep(2)
        self.ib.cancelMktData(contract)

        bid = ticker.bid or 0.0
        ask = ticker.ask or 0.0
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else None

        if mid is None or mid <= 0:
            mid = ticker.last or ticker.close or 0.0
        if mid <= 0:
            logger.error(f"Cannot determine closing price for {position.position_id}")
            return False

        qty = abs(position.quantity)
        label = f"{position.symbol} {position.expiry} {position.right}{position.strike}"

        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would close: BUY {qty}x {label} @ ${mid:.2f} | reason: {reason}"
            )
            return True

        logger.info(f"Closing position: BUY {qty}x {label} @ ${mid:.2f} | reason: {reason}")
        trade = await self._place_limit(contract, "BUY", qty, mid)
        if trade is None:
            return False

        filled = await self._wait_for_fill(trade, contract, "BUY", qty, initial_mid=mid)
        if filled:
            fill_price = trade.orderStatus.avgFillPrice
            pnl = (position.entry_premium - fill_price) * 100 * qty
            logger.info(
                f"Closed: {label} @ ${fill_price:.2f} | "
                f"P&L: ${pnl:+.2f} | reason: {reason}"
            )
            mark_position_closed(
                position_id=position.position_id,
                close_premium=fill_price,
                close_reason=reason,
                qty=qty,
                entry_premium=position.entry_premium,
            )
            if self.notifier:
                closed_data = asdict(position)
                closed_data.update({
                    "close_premium": fill_price,
                    "pnl": round(pnl, 2),
                    "close_reason": reason,
                })
                self.notifier.on_trade_close(closed_data)
            return True
        else:
            logger.warning(
                f"Close order unfilled for {label} — manual intervention may be needed"
            )
            self.ib.cancelOrder(trade.order)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _place_spread(
        self, candidate: "OptionCandidate", qty: int
    ) -> "Trade | None":
        """Place a credit spread as a BAG (combo) order."""
        from option_chain import OptionCandidate as OC  # avoid circular at module level

        bag = Contract()
        bag.symbol = config.SYMBOL
        bag.secType = "BAG"
        bag.currency = config.CURRENCY
        bag.exchange = config.EXCHANGE

        sell_leg = ComboLeg()
        sell_leg.conId = candidate.contract.conId
        sell_leg.ratio = 1
        sell_leg.action = "SELL"
        sell_leg.exchange = config.EXCHANGE

        buy_leg = ComboLeg()
        buy_leg.conId = candidate.hedge_contract.conId
        buy_leg.ratio = 1
        buy_leg.action = "BUY"
        buy_leg.exchange = config.EXCHANGE

        bag.comboLegs = [sell_leg, buy_leg]

        label = (
            f"SELL {candidate.right}{candidate.strike} / "
            f"BUY {candidate.right}{candidate.hedge_strike} "
            f"net credit ${candidate.net_credit:.2f}"
        )
        logger.info(f"Placing spread (BAG): {label}")
        # For credit spreads placed as BAG, action="SELL" means net credit
        order = LimitOrder("SELL", qty, round(candidate.net_credit, 2), tif="DAY")
        try:
            trade = self.ib.placeOrder(bag, order)
            await asyncio.sleep(0.5)
            return trade
        except Exception as exc:
            logger.error(f"BAG placeOrder error: {exc}")
            return None

    async def _place_limit(
        self, contract, action: str, qty: int, price: float
    ) -> "Trade | None":
        order = LimitOrder(action, qty, round(price, 2), tif="DAY")
        try:
            trade = self.ib.placeOrder(contract, order)
            await asyncio.sleep(0.5)
            return trade
        except Exception as exc:
            logger.error(f"placeOrder error: {exc}")
            return None

    async def _wait_for_fill(
        self,
        trade: Trade,
        contract,
        action: str,
        qty: int,
        initial_mid: float,
    ) -> bool:
        """Wait for fill; adjust price once after timeout. Returns True if filled."""
        deadline = asyncio.get_event_loop().time() + config.ORDER_FILL_TIMEOUT
        while asyncio.get_event_loop().time() < deadline:
            if trade.isDone():
                return trade.orderStatus.status == "Filled"
            await asyncio.sleep(2)

        if trade.isDone():
            return trade.orderStatus.status == "Filled"

        # Adjust price once
        adjusted = round(initial_mid - config.ORDER_ADJUST_STEP, 2)
        logger.info(f"Fill timeout — adjusting limit price to ${adjusted:.2f}")
        self.ib.cancelOrder(trade.order)
        await asyncio.sleep(1)

        adjusted_trade = await self._place_limit(contract, action, qty, adjusted)
        if adjusted_trade is None:
            return False

        deadline2 = asyncio.get_event_loop().time() + config.ORDER_FILL_TIMEOUT
        while asyncio.get_event_loop().time() < deadline2:
            if adjusted_trade.isDone():
                if adjusted_trade.orderStatus.status == "Filled":
                    # Propagate fill price back to caller's trade reference
                    trade.orderStatus.avgFillPrice = adjusted_trade.orderStatus.avgFillPrice
                    trade.orderStatus.status = adjusted_trade.orderStatus.status
                    return True
                return False
            await asyncio.sleep(2)

        return False
