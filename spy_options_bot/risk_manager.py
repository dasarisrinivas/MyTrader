"""Risk management and position monitoring for SPY options bot.

Exit conditions checked each poll cycle:
  1. Profit target: buy back at profit_target_price (pre-computed at entry)
  2. Max loss stop: buy back at stop_loss_price (pre-computed at entry)
  3. Delta stop: close if |delta| >= DELTA_STOP (gone deeply ITM)
  4. Thursday EOD hard close: always close by EOD_CLOSE_HOUR:EOD_CLOSE_MINUTE ET
  5. Thursday emergency gamma: close immediately if SPY moves >1.5% in a 5-min bar

Portfolio-level entry checks:
  - Block if open positions already exist
  - Block if daily NLV drop > DAILY_LOSS_LIMIT_PCT
  - Block if margin headroom < MAX_ACCOUNT_RISK_PCT of NLV

Poll interval is dynamic:
  - Thursday:    60s  (high gamma risk)
  - DTE ≤ 2:    120s
  - Otherwise:  300s  (config.POLL_INTERVAL)
"""
from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from ib_insync import IB, Option, Stock
import asyncio

from logger import logger
import config
from order_manager import OpenPosition, OrderManager, load_open_positions

if TYPE_CHECKING:
    from notifier import Notifier

ET = ZoneInfo("America/New_York")

_EMERGENCY_MOVE_THRESHOLD = 0.015  # 1.5% in a single 5-min bar


def _now_et() -> datetime:
    return datetime.now(ET)


class RiskManager:
    def __init__(
        self,
        ib: IB,
        order_manager: OrderManager,
        notifier: "Notifier | None" = None,
    ) -> None:
        self.ib = ib
        self.order_manager = order_manager
        self.notifier = notifier
        self._session_start_nlv: float | None = None

    # ------------------------------------------------------------------
    # Dynamic poll interval
    # ------------------------------------------------------------------

    def get_poll_interval(self) -> int:
        """Return polling interval in seconds based on day-of-week and DTE."""
        now = _now_et()
        dte = self._get_dte()

        if now.weekday() == 3:  # Thursday
            interval = 60
            label = "Thursday gamma mode"
        elif dte is not None and dte <= 2:
            interval = 120
            label = f"DTE={dte} ≤ 2"
        else:
            interval = config.POLL_INTERVAL
            label = "standard"

        logger.info(f"[RiskManager] Poll interval: {interval}s ({label})")
        return interval

    def _get_dte(self) -> int | None:
        """Return calendar days to nearest open-position expiry, or None."""
        positions = load_open_positions()
        if not positions:
            return None
        today = date.today()
        dtes = []
        for pos in positions:
            try:
                expiry_date = datetime.strptime(pos.expiry, "%Y%m%d").date()
                dtes.append((expiry_date - today).days)
            except ValueError:
                pass
        return min(dtes) if dtes else None

    # ------------------------------------------------------------------
    # Pre-entry checks
    # ------------------------------------------------------------------

    async def can_enter(self) -> tuple[bool, str]:
        """Return (allowed, reason) for pre-entry portfolio-level checks."""
        open_positions = load_open_positions()
        if open_positions:
            return False, f"Already have {len(open_positions)} open position(s)"

        nlv = await self._get_nlv()
        if nlv is None:
            return False, "Could not retrieve account NLV"

        if self._session_start_nlv is None:
            self._session_start_nlv = nlv
            logger.info(f"Session start NLV: ${nlv:,.2f}")

        daily_drop = (self._session_start_nlv - nlv) / self._session_start_nlv
        if daily_drop > config.DAILY_LOSS_LIMIT_PCT:
            return False, (
                f"Daily loss limit reached: account down {daily_drop:.1%} "
                f"(limit {config.DAILY_LOSS_LIMIT_PCT:.0%})"
            )

        margin_available = await self._get_available_margin()
        if margin_available is None:
            return False, "Could not retrieve margin data"

        max_margin_use = nlv * config.MAX_ACCOUNT_RISK_PCT
        if margin_available < max_margin_use:
            return False, (
                f"Insufficient margin: available ${margin_available:,.0f} "
                f"< required ${max_margin_use:,.0f}"
            )

        return True, "OK"

    # ------------------------------------------------------------------
    # Position monitoring loop
    # ------------------------------------------------------------------

    async def monitor_positions(self) -> None:
        """Check exit conditions for all open positions."""
        positions = load_open_positions()
        if not positions:
            return

        now = _now_et()
        is_thursday = now.weekday() == 3
        past_eod_close = (
            now.hour > config.EOD_CLOSE_HOUR
            or (now.hour == config.EOD_CLOSE_HOUR and now.minute >= config.EOD_CLOSE_MINUTE)
        )

        # Thursday emergency gamma check — fires before individual position checks
        if is_thursday and await self._check_thursday_emergency():
            logger.warning(
                "Emergency gamma close: closing all positions due to rapid SPY move on Thursday"
            )
            if self.notifier:
                self.notifier.on_emergency_close(
                    {}, "1.5%+ SPY move in 5-min bar on Thursday"
                )
            for pos in positions:
                await self.order_manager.close_position(pos, "emergency_gamma")
            return

        for pos in positions:
            await self._check_position(pos, is_thursday and past_eod_close)

    async def _check_thursday_emergency(self) -> bool:
        """Return True if SPY moved > 1.5% in the last 5-minute bar (Thursday only)."""
        try:
            spy = Stock(config.SYMBOL, config.EXCHANGE, config.CURRENCY)
            qualified = await self.ib.qualifyContractsAsync(spy)
            if not qualified:
                return False

            bars = await self.ib.reqHistoricalDataAsync(
                qualified[0],
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="5 mins",
                whatToShow="MIDPOINT",
                useRTH=True,
                formatDate=1,
            )
            if len(bars) < 2:
                return False

            last_bar = bars[-1]
            prev_bar = bars[-2]
            if prev_bar.close <= 0:
                return False

            change_pct = abs((last_bar.close - prev_bar.close) / prev_bar.close)
            if change_pct > _EMERGENCY_MOVE_THRESHOLD:
                logger.warning(
                    f"SPY 5-min bar move: {change_pct:.2%} "
                    f"(prev close={prev_bar.close:.2f}, last close={last_bar.close:.2f})"
                )
                return True
        except Exception as exc:
            logger.error(f"Error checking Thursday emergency: {exc}")
        return False

    async def _check_position(self, pos: OpenPosition, force_close: bool) -> None:
        """Evaluate exit conditions for a single position."""
        if force_close:
            logger.info(
                f"Thursday EOD hard close: {pos.right}{pos.strike} {pos.expiry}"
            )
            await self.order_manager.close_position(pos, "thursday_eod")
            return

        # Get current market data
        opt = Option(
            pos.symbol, pos.expiry, pos.strike, pos.right,
            config.EXCHANGE, multiplier="100", currency=config.CURRENCY,
        )
        try:
            qualified = await self.ib.qualifyContractsAsync(opt)
            if not qualified:
                logger.warning(f"Could not qualify position contract: {pos.position_id}")
                return
            contract = qualified[0]
        except Exception as exc:
            logger.error(f"Error qualifying position {pos.position_id}: {exc}")
            return

        ticker = self.ib.reqMktData(contract, "", snapshot=True)
        await asyncio.sleep(2)
        self.ib.cancelMktData(contract)

        bid = ticker.bid or 0.0
        ask = ticker.ask or 0.0
        current_price = (
            (bid + ask) / 2 if bid > 0 and ask > 0
            else (ticker.last or ticker.close or 0.0)
        )
        greeks = ticker.modelGreeks

        if current_price <= 0:
            logger.warning(f"Cannot get current price for {pos.position_id}")
            return

        # pnl_per_share: positive = profit (we sold high, now lower)
        pnl_per_share = pos.entry_premium - current_price
        delta = greeks.delta if greeks else None

        label = f"{pos.symbol} {pos.expiry} {pos.right}{pos.strike}"
        logger.debug(
            f"Monitor {label}: current=${current_price:.2f} | "
            f"entry=${pos.entry_premium:.2f} | pnl/share=${pnl_per_share:+.2f} | "
            f"target=${pos.profit_target_price:.2f} | stop=${pos.stop_loss_price:.2f} | "
            f"delta={f'{delta:.3f}' if delta is not None else 'N/A'}"
        )

        # Profit target — use pre-computed value, never recompute
        if current_price <= pos.profit_target_price:
            logger.info(
                f"Profit target hit: current=${current_price:.2f} <= "
                f"target=${pos.profit_target_price:.2f} (50% of ${pos.entry_premium:.2f})"
            )
            await self.order_manager.close_position(pos, "profit_target")
            return

        # Max loss stop — use pre-computed value
        if current_price >= pos.stop_loss_price:
            logger.warning(
                f"Max loss stop: current=${current_price:.2f} >= "
                f"stop=${pos.stop_loss_price:.2f} ({config.MAX_LOSS_MULTIPLE}× premium)"
            )
            await self.order_manager.close_position(pos, "loss_stop")
            if self.notifier:
                self.notifier.on_risk_stop({"label": label}, "loss_stop")
            return

        # Roll trigger — close early and let the bot re-enter next week.
        # Fires when delta crosses ROLL_DELTA_TRIGGER (0.40) before DELTA_STOP (0.50).
        # Only active on Mon–Wed so a new spread can be opened in the same cycle.
        if delta is not None and config.ROLL_DELTA_TRIGGER <= abs(delta) < config.DELTA_STOP:
            now = _now_et()
            if now.weekday() in config.ENTRY_DAYS:
                logger.info(
                    f"Roll trigger: |delta|={abs(delta):.3f} >= {config.ROLL_DELTA_TRIGGER} "
                    f"(below hard stop {config.DELTA_STOP}) — closing to re-enter next week"
                )
                await self.order_manager.close_position(pos, "roll_close")
                if self.notifier:
                    self.notifier.on_risk_stop({"label": label}, "roll_close")
                return

        # Delta stop — hard exit when position has gone deeply ITM
        if delta is not None and abs(delta) >= config.DELTA_STOP:
            logger.warning(
                f"Delta stop: |delta|={abs(delta):.3f} >= {config.DELTA_STOP}"
            )
            await self.order_manager.close_position(pos, "delta_stop")
            if self.notifier:
                self.notifier.on_risk_stop({"label": label}, "delta_stop")
            return

    # ------------------------------------------------------------------
    # Account data helpers
    # ------------------------------------------------------------------

    async def _get_nlv(self) -> float | None:
        try:
            account_values = self.ib.accountValues()
            nlv_vals = [
                v for v in account_values
                if v.tag == "NetLiquidation" and v.currency == "USD"
            ]
            if nlv_vals:
                return float(nlv_vals[0].value)
        except Exception as exc:
            logger.error(f"Error fetching NLV: {exc}")
        return None

    async def _get_available_margin(self) -> float | None:
        try:
            account_values = self.ib.accountValues()
            avail = [
                v for v in account_values
                if v.tag == "AvailableFunds" and v.currency == "USD"
            ]
            if avail:
                return float(avail[0].value)
        except Exception as exc:
            logger.error(f"Error fetching available funds: {exc}")
        return None
