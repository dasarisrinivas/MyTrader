"""Exit logic for position management.

Extracted from ``LiveTradingManager`` to reduce its line count.
All methods previously lived on the manager as ``self._check_position_exit_*``
and related helpers.  ``ExitManager`` receives a *manager* reference so it can
reach ``executor``, ``status``, ``settings``, ``price_history``, etc.

Usage inside ``LiveTradingManager.__init__``::

    from .components.exit_manager import ExitManager
    self._exit_mgr = ExitManager(self)

Then each old method becomes a thin delegator::

    async def _check_position_exit_signals(self, current_price):
        return await self._exit_mgr.check_position_exit_signals(current_price)
"""

from __future__ import annotations

from datetime import datetime, time, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")

from ...utils.logger import logger
from ...utils.structured_logging import log_structured_event
from ...utils.timezone_utils import now_cst

if TYPE_CHECKING:
    from ..live_trading_manager import LiveTradingManager

__all__ = ["ExitManager"]


class ExitManager:
    """Encapsulates all position-exit evaluation & execution logic.

    The *manager* back-reference gives access to the full trading context
    without duplicating state.
    """

    def __init__(self, manager: "LiveTradingManager") -> None:
        self._m = manager
        self._last_pnl_alert_time: Optional[datetime] = None
        self._last_known_qty: int = 0  # tracks flat→open transitions
        self._thesis_reversal_state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Convenience accessors (keep method bodies short & readable)
    # ------------------------------------------------------------------

    @property
    def executor(self):
        return self._m.executor

    @property
    def status(self):
        return self._m.status

    @property
    def settings(self):
        return self._m.settings

    @property
    def one_minute_cfg(self):
        return self._m.one_minute_cfg

    @property
    def price_history(self) -> List[Dict]:
        return self._m.price_history

    @property
    def contract_spec(self):
        return self._m.contract_spec

    @property
    def signal_processor(self):
        return self._m.signal_processor

    # ------------------------------------------------------------------
    # Top-level entry points
    # ------------------------------------------------------------------

    async def check_position_exit_signals(self, current_price: Optional[float]) -> bool:
        """Check if existing position should be closed.

        Returns *True* if an exit was triggered (even if just a stop update).
        """
        if not self.executor:
            return False

        position = await self.executor.get_current_position()
        if not position or position.quantity == 0:
            self._last_known_qty = 0
            return False

        # Reset P&L alert timer on new position entry (flat → open)
        if self._last_known_qty == 0 and position.quantity != 0:
            self._last_pnl_alert_time = None
            self._thesis_reversal_state = {}
        self._last_known_qty = position.quantity

        price = current_price
        if price is None:
            try:
                price = await self.executor.get_current_price()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"⚠️ Could not fetch price for exit check: {exc}")
                return False
        if price is None:
            logger.debug("No price available for exit checks; skipping exit evaluation")
            return False

        # Trend-flip exit (optional)
        if self.one_minute_cfg and getattr(self.one_minute_cfg, "trend_flip_exit", False):
            trend_label = getattr(self.status, "last_trend", "") or getattr(
                self.status, "hybrid_market_trend", ""
            )
            adx_value = float(getattr(self.status, "last_adx", 0.0) or 0.0)
            if adx_value >= getattr(self.one_minute_cfg, "trend_adx_threshold", 20.0):
                if position.quantity > 0 and trend_label == "DOWNTREND":
                    logger.info("🔄 Trend flip detected against LONG; exiting at market")
                    await self.place_exit_order("SELL", abs(position.quantity), price)
                    return True
                if position.quantity < 0 and trend_label == "UPTREND":
                    logger.info("🔄 Trend flip detected against SHORT; exiting at market")
                    await self.place_exit_order("BUY", abs(position.quantity), price)
                    return True

        # ──────────────────────────────────────────────────────────────
        # FEB 7 2026: 15m STRATEGY EXIT PARITY
        #
        # The 15m strategy uses ONLY bracket orders (ATR-based TP/SL)
        # and ft_max_hold_bars (time-stop). The legacy point-based and
        # dollar-based exit layers (generate_exit_signal_for_long/short,
        # trend-change exit) do NOT exist in the backtest.
        #
        # For 15m: skip legacy exit generators → go straight to
        # check_position_exit_logic which handles bracket protection,
        # ft_max_hold_minutes, and profit protection.
        #
        # Legacy exits are only used for 1m strategies.
        # ──────────────────────────────────────────────────────────────
        is_15m = getattr(self._m, "_active_timeframe", "1m") == "15m"

        is_short = position.quantity < 0
        exit_signal = None
        if not is_15m:
            exit_signal = (
                self.generate_exit_signal_for_short(price, position)
                if is_short
                else self.generate_exit_signal_for_long(price, position)
            )

        # Update trailing stops while holding
        try:
            atr_val = float(getattr(self.status, "last_atr", 0.0) or 0.0)
            if atr_val == 0.0 and self.price_history:
                latest_bar = self.price_history[-1]
                atr_val = float(latest_bar.get("ATR_14", latest_bar.get("atr", 0.0)) or 0.0)
            await self.executor.update_trailing_stops(price, atr_val)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Trailing stop update skipped: {exc}")

        if not exit_signal:
            exit_signal = await self.check_position_exit_logic(position)

        if exit_signal:
            active_orders = self.executor.get_active_order_count(sync=True)
            exit_reason = exit_signal.get("reason") if isinstance(exit_signal, dict) else None
            if active_orders > 0 and exit_reason != "MAX_HOLD_EXIT":
                logger.info(
                    "Exit signal detected but {active} active orders are still open; skipping duplicate exit",
                    active=active_orders,
                )
                return True
            if active_orders > 0 and exit_reason == "MAX_HOLD_EXIT":
                logger.warning(
                    "⏳ MAX_HOLD_EXIT override: {active} active bracket orders still open; forcing market flatten",
                    active=active_orders,
                )
            qty = abs(position.quantity)
            direction = "SHORT" if is_short else "LONG"
            logger.info(f"🔄 Exit signal for {direction} position: {exit_signal}")
            # If custom exit signal includes an action, honor it
            if isinstance(exit_signal, dict) and "action" in exit_signal:
                await self.execute_position_exit(exit_signal, position, price)
            else:
                await self.place_exit_order("BUY" if is_short else "SELL", qty, price)
            return True

        return False

    # ------------------------------------------------------------------
    # P&L-band / bracket-aware exit logic
    # ------------------------------------------------------------------

    async def check_position_exit_logic(self, position) -> Optional[Dict[str, Any]]:
        """Check if existing position should be exited based on simple P&L bands."""
        if not self.executor:
            return None
        try:
            current_price = await self.executor.get_current_price()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"⚠️ Could not fetch price for position exit logic: {exc}")
            return None
        if not current_price:
            return None

        qty = int(getattr(position, "quantity", 0) or 0)
        if qty == 0:
            return None

        contracts = abs(qty)

        # ──────────────────────────────────────────────────────────────
        # FEB 2026: RTH FORCED FLATTEN — HIGHEST PRIORITY EXIT
        #
        # Defense-in-depth: if we still hold a position at 15:50 ET
        # (10 min before RTH close), force a market exit regardless of
        # bracket status, P&L, or any other condition.
        #
        # Normal operation: brackets or ft_max_hold_bars fire well
        # before this. The flatten should trigger 0 times/month.
        # Any trigger is a notable event requiring investigation.
        #
        # This prevents overnight position leaks caused by:
        #   - IB bracket order glitch / non-fill
        #   - max_hold timer race condition
        #   - Unexpected bot restart during RTH close window
        # ──────────────────────────────────────────────────────────────
        rth_flatten_time = getattr(
            self.one_minute_cfg, "rth_flatten_time_et", "15:50"
        ) if self.one_minute_cfg else "15:50"
        try:
            parts = str(rth_flatten_time).split(":")
            flatten_hour, flatten_minute = int(parts[0]), int(parts[1])
            now_et = datetime.now(ET)
            flatten_et = time(flatten_hour, flatten_minute)
            rth_close_et = time(16, 0)
            current_time_et = now_et.time()
            # Only flatten during the window: flatten_time <= now < 16:00 ET
            # (After 16:00, CME maintenance starts — separate concern)
            if flatten_et <= current_time_et < rth_close_et:
                action = "SELL" if qty > 0 else "BUY"
                logger.warning(
                    f"🚨 RTH_FORCED_FLATTEN triggered at {now_et.strftime('%H:%M:%S')} ET "
                    f"({contracts} contracts). This is a safety backstop — "
                    f"investigate why brackets/time-stop did not fire."
                )
                try:
                    log_structured_event(
                        agent="exit_manager",
                        event_type="RTH_FORCED_FLATTEN",
                        message=f"Forced flatten at {now_et.strftime('%H:%M')} ET",
                        payload={
                            "contracts": contracts,
                            "direction": "LONG" if qty > 0 else "SHORT",
                            "current_price": current_price,
                            "flatten_time": str(rth_flatten_time),
                        },
                    )
                except Exception:
                    pass
                return {
                    "reason": "RTH_FORCED_FLATTEN",
                    "action": action,
                    "quantity": contracts,
                    "pnl": 0.0,  # Will be computed on actual fill
                }
        except Exception as exc:
            logger.debug(f"RTH flatten time check skipped: {exc}")

        entry_price_raw = getattr(position, "avg_cost", 0.0) or 0.0
        entry_price = self._m._normalize_entry_price(entry_price_raw, current_price)
        multiplier = getattr(self.contract_spec, "point_value", 5) or 5

        pnl_per_contract = (current_price - entry_price) * multiplier
        # For shorts, invert PnL sign
        if qty < 0:
            pnl_per_contract = (entry_price - current_price) * multiplier
        total_pnl = pnl_per_contract * contracts

        logger.info(
            f"📊 Position P&L check -> entry={entry_price:.2f} price={current_price:.2f} "
            f"pnl/ct={pnl_per_contract:.2f} total={total_pnl:.2f}"
        )

        thesis_reversal = self._check_thesis_reversal_exit(
            qty=qty,
            current_price=current_price,
            entry_price=entry_price,
            total_pnl=total_pnl,
            contracts=contracts,
        )
        if thesis_reversal:
            return thesis_reversal

        # ── Periodic Telegram P&L alert (every 30 min while position open) ──
        telegram = getattr(self._m, "telegram", None)
        if telegram and getattr(telegram, "enabled", False):
            now = now_cst()
            _interval = timedelta(minutes=30)
            if self._last_pnl_alert_time is None or (now - self._last_pnl_alert_time) >= _interval:
                symbol = getattr(self._m, "symbol", "ES")
                stop_loss = getattr(position, "stop_loss", None)
                take_profit = getattr(position, "take_profit", None)
                telegram.send_position_pnl_background(
                    symbol=symbol,
                    quantity=qty,
                    entry_price=entry_price,
                    current_price=current_price,
                    pnl_per_contract=pnl_per_contract,
                    total_pnl=total_pnl,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
                self._last_pnl_alert_time = now

        # ──────────────────────────────────────────────────────────────
        # FEB 20 2026: STRUCTURAL SUPPORT CIRCUIT BREAKER
        #
        # If price breaks below a configured structural support floor,
        # immediately flatten any LONG position at market.  For shorts,
        # tighten the trailing stop aggressively (support-break is in
        # our favour, but we lock profit quickly to avoid the snap-back).
        #
        # Config: ``structural_support_floor`` in config.yaml or env
        # ``STRUCTURAL_SUPPORT_FLOOR``.  Set to 0 or omit to disable.
        # ──────────────────────────────────────────────────────────────
        support_floor = self._get_support_floor()
        if support_floor is not None and support_floor > 0:
            if qty > 0 and current_price < support_floor:
                # LONG below structural support → immediate market exit
                logger.warning(
                    f"🚨 SUPPORT_BREAK_FLATTEN: Price {current_price:.2f} < "
                    f"floor {support_floor:.2f}. Flattening {contracts} LONG contracts."
                )
                try:
                    log_structured_event(
                        agent="exit_manager",
                        event_type="SUPPORT_BREAK_FLATTEN",
                        message=f"Price {current_price:.2f} broke structural support {support_floor:.2f}",
                        payload={
                            "contracts": contracts,
                            "direction": "LONG",
                            "current_price": current_price,
                            "support_floor": support_floor,
                            "pnl": total_pnl,
                        },
                    )
                except Exception:
                    pass
                return {
                    "reason": "SUPPORT_BREAK_FLATTEN",
                    "action": "SELL",
                    "quantity": contracts,
                    "pnl": total_pnl,
                }
            if qty < 0 and current_price < support_floor - 5.0:
                # SHORT and price is well below support → tighten trailing stop
                # to lock in profits before a potential snap-back rally
                if self.executor and hasattr(self.executor, "get_active_bracket_levels"):
                    try:
                        bracket_levels = self.executor.get_active_bracket_levels()
                        if bracket_levels:
                            current_stop = bracket_levels.get("stop_loss")
                            if current_stop is not None:
                                # Tighten stop to 4 pts above current price
                                tight_stop = current_price + 4.0
                                if tight_stop < current_stop:
                                    logger.warning(
                                        f"📉 SUPPORT_BREAK_TIGHTEN: Price {current_price:.2f} broke "
                                        f"floor {support_floor:.2f}. Tightening SHORT stop "
                                        f"{current_stop:.2f} → {tight_stop:.2f}"
                                    )
                                    return {
                                        "reason": "SUPPORT_BREAK_TIGHTEN",
                                        "action": None,
                                        "new_stop_loss": tight_stop,
                                        "quantity": 0,
                                    }
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(f"Support-break tighten skipped: {exc}")

        # PROFIT PROTECTION: Check if price has reached TP level and force market exit
        # This works around IB Paper Trading LIMIT order fill issues
        if self.executor and hasattr(self.executor, "get_active_bracket_levels"):
            try:
                bracket_levels = self.executor.get_active_bracket_levels()
                if bracket_levels:
                    take_profit = bracket_levels.get("take_profit")
                    stop_loss = bracket_levels.get("stop_loss")

                    if take_profit is not None:
                        # For long positions: check if current price >= take profit
                        if qty > 0 and current_price >= take_profit:
                            logger.warning(
                                f"🎯 PROFIT PROTECTION TRIGGERED: Price {current_price:.2f} >= TP {take_profit:.2f} "
                                f"(+${total_pnl:.2f}). Forcing MARKET exit to capture profit."
                            )
                            return {
                                "reason": "PROFIT_PROTECTION",
                                "action": "SELL",
                                "quantity": contracts,
                                "pnl": total_pnl,
                            }
                        # For short positions: check if current price <= take profit
                        elif qty < 0 and current_price <= take_profit:
                            logger.warning(
                                f"🎯 PROFIT PROTECTION TRIGGERED: Price {current_price:.2f} <= TP {take_profit:.2f} "
                                f"(+${total_pnl:.2f}). Forcing MARKET exit to capture profit."
                            )
                            return {
                                "reason": "PROFIT_PROTECTION",
                                "action": "BUY",
                                "quantity": contracts,
                                "pnl": total_pnl,
                            }

                    # BREAKEVEN PROTECTION: Move stop to breakeven when price moves
                    # ft_breakeven_trigger_pts in favour (default 3 pts / $15 for 1 contract).
                    #
                    # MAR 11 2026 Fix #13: Replaced the old "60% to TP AND pnl > $50" gate with a
                    # simpler point-based trigger so 1-contract overnight trades get protection
                    # earlier.  Trade 7 (Mar 10) reached +$25 (5 pts) but the old gate required
                    # pnl > $50 (10 pts) — so it never fired and the full gain reversed to a loss.
                    #
                    # Thresholds (configurable via config.yaml):
                    #   ft_breakeven_trigger_pts:         3.0  → RTH: move SL to entry+1 tick once +3 pts
                    #   ft_overnight_breakeven_trigger_pts: 2.0 → Fix #15: tighter trigger outside RTH
                    # Both thresholds work the same way for a 1-contract position: move SL to BE.
                    # Partial profit taking (2+ contracts) keeps the original 60%-to-TP logic.
                    _trading_cfg = getattr(getattr(self._m, "settings", None), "trading", None)
                    _rth_be_pts = float(getattr(_trading_cfg, "ft_breakeven_trigger_pts", None) or 3.0)
                    # Fix #15 (MAR 12 2026): use tighter trigger during overnight/evening sessions
                    try:
                        _et_now_t = datetime.now(ET).time()
                        _is_rth = time(9, 30) <= _et_now_t < time(16, 0)
                    except Exception:
                        _is_rth = True
                    if _is_rth:
                        _be_trigger_pts = _rth_be_pts
                    else:
                        _be_trigger_pts = float(
                            getattr(_trading_cfg, "ft_overnight_breakeven_trigger_pts", None) or 2.0
                        )
                    # MES: $5/pt, so convert pts → dollars for comparison with total_pnl
                    _be_trigger_usd = _be_trigger_pts * 5.0

                    if take_profit is not None and stop_loss is not None:
                        # Calculate distance to TP (still used for partial-profit multi-contract path)
                        if qty > 0:  # Long position
                            tp_distance = take_profit - entry_price
                            current_distance = current_price - entry_price
                            percent_to_tp = (current_distance / tp_distance) * 100 if tp_distance > 0 else 0

                            # Point-based breakeven trigger (Fix #13) OR 60%-to-TP (existing)
                            _breakeven_triggered = (
                                (total_pnl >= _be_trigger_usd)           # new: simpler point-based
                                or (60 <= percent_to_tp < 100 and total_pnl > 50)  # keep legacy for safety
                            )
                            if _breakeven_triggered:
                                if not hasattr(self._m, "_breakeven_stop_set") or not self._m._breakeven_stop_set:
                                    # For 2+ contracts: Partial profit taking (unchanged)
                                    if contracts >= 2:
                                        partial_qty = contracts // 2  # Exit 50%
                                        logger.warning(
                                            f"💰 PARTIAL PROFIT TAKING: {percent_to_tp:.1f}% to TP "
                                            f"({current_price:.2f}/{take_profit:.2f}). "
                                            f"Exiting {partial_qty} contracts, keeping {contracts - partial_qty}."
                                        )
                                        self._m._breakeven_stop_set = True
                                        return {
                                            "reason": "PARTIAL_PROFIT",
                                            "action": "SELL",
                                            "quantity": partial_qty,
                                            "pnl": (total_pnl / contracts) * partial_qty,
                                            "move_stop_to_breakeven": True,
                                        }
                                    # For 1 contract: Just move stop to breakeven
                                    else:
                                        breakeven_stop = entry_price + 1.0  # +1 point buffer
                                        if breakeven_stop > stop_loss:
                                            logger.warning(
                                                f"🔒 BREAKEVEN PROTECTION (Fix#13): P&L=${total_pnl:.2f} "
                                                f"(trigger=${_be_trigger_usd:.0f} / {_be_trigger_pts:.1f}pts). "
                                                f"Moving stop to breakeven +1: {breakeven_stop:.2f}"
                                            )
                                            self._m._breakeven_stop_set = True
                                            return {
                                                "reason": "BREAKEVEN_STOP_UPDATE",
                                                "action": None,
                                                "new_stop_loss": breakeven_stop,
                                                "quantity": 0,
                                            }

                        elif qty < 0:  # Short position
                            tp_distance = entry_price - take_profit
                            current_distance = entry_price - current_price
                            percent_to_tp = (current_distance / tp_distance) * 100 if tp_distance > 0 else 0

                            _breakeven_triggered = (
                                (total_pnl >= _be_trigger_usd)
                                or (60 <= percent_to_tp < 100 and total_pnl > 50)
                            )
                            if _breakeven_triggered:
                                if not hasattr(self._m, "_breakeven_stop_set") or not self._m._breakeven_stop_set:
                                    # For 2+ contracts: Partial profit taking (unchanged)
                                    if contracts >= 2:
                                        partial_qty = contracts // 2
                                        logger.warning(
                                            f"💰 PARTIAL PROFIT TAKING: {percent_to_tp:.1f}% to TP "
                                            f"({current_price:.2f}/{take_profit:.2f}). "
                                            f"Exiting {partial_qty} contracts, keeping {contracts - partial_qty}."
                                        )
                                        self._m._breakeven_stop_set = True
                                        return {
                                            "reason": "PARTIAL_PROFIT",
                                            "action": "BUY",
                                            "quantity": partial_qty,
                                            "pnl": (total_pnl / contracts) * partial_qty,
                                            "move_stop_to_breakeven": True,
                                        }
                                    # For 1 contract: Just move stop to breakeven
                                    else:
                                        breakeven_stop = entry_price - 1.0  # -1 point buffer
                                        if breakeven_stop < stop_loss:
                                            logger.warning(
                                                f"🔒 BREAKEVEN PROTECTION (Fix#13): P&L=${total_pnl:.2f} "
                                                f"(trigger=${_be_trigger_usd:.0f} / {_be_trigger_pts:.1f}pts). "
                                                f"Moving stop to breakeven -1: {breakeven_stop:.2f}"
                                            )
                                            self._m._breakeven_stop_set = True
                                            return {
                                                "reason": "BREAKEVEN_STOP_UPDATE",
                                                "action": None,
                                                "new_stop_loss": breakeven_stop,
                                                "quantity": 0,
                                            }

                    # TRAILING STOP: ATR-based trailing once solidly profitable
                    if stop_loss is not None and total_pnl > 50:
                        # Get current ATR for adaptive trailing
                        _trail_atr = 0.0
                        if self.price_history:
                            _trail_atr = float(self.price_history[-1].get("ATR_14", 0.0) or 0.0)
                        # Trail at 1.5x ATR, minimum 6 points, maximum 12 points
                        trailing_distance = max(6.0, min(12.0, _trail_atr * 1.5)) if _trail_atr > 0 else 8.0

                        if qty > 0:  # Long position
                            # Calculate new trailing stop
                            new_stop = current_price - trailing_distance

                            # Only move stop up (never down)
                            if new_stop > stop_loss and new_stop > entry_price:
                                logger.info(
                                    f"📈 TRAILING STOP: Moving stop from {stop_loss:.2f} → {new_stop:.2f} "
                                    f"(price={current_price:.2f}, P&L=${total_pnl:.2f}, trail={trailing_distance:.1f}pts)"
                                )
                                return {
                                    "reason": "TRAILING_STOP_UPDATE",
                                    "action": None,
                                    "new_stop_loss": new_stop,
                                    "quantity": 0,
                                }

                        elif qty < 0:  # Short position
                            # Calculate new trailing stop
                            new_stop = current_price + trailing_distance

                            # Only move stop down (never up)
                            if new_stop < stop_loss and new_stop < entry_price:
                                logger.info(
                                    f"📉 TRAILING STOP: Moving stop from {stop_loss:.2f} → {new_stop:.2f} "
                                    f"(price={current_price:.2f}, P&L=${total_pnl:.2f}, trail={trailing_distance:.1f}pts)"
                                )
                                return {
                                    "reason": "TRAILING_STOP_UPDATE",
                                    "action": None,
                                    "new_stop_loss": new_stop,
                                    "quantity": 0,
                                }

            except Exception as e:
                logger.debug(f"Profit protection/partial profit check skipped: {e}")

        # ──────────────────────────────────────────────────────────────
        # FEB 7 2026: 15m strategy uses ONLY bracket orders + time-stop.
        # Skip trend-change, legacy time-based, and dollar-based exits
        # to maintain parity with the validated backtest.
        # ──────────────────────────────────────────────────────────────
        is_15m = getattr(self._m, "_active_timeframe", "1m") == "15m"

        # ──────────────────────────────────────────────────────────────
        # MAR 12 2026: 15m SIGNAL FLIP EXIT
        #
        # When a position is open, evaluate the current EMA9/EMA21
        # crossover and MACD histogram to detect a directional reversal.
        # If indicators flip against the position and we have minimum
        # open profit, exit at market instead of waiting for full
        # trend reversal to hit the stop loss.
        #
        # Config:
        #   ft_trend_flip_exit_enabled: true/false (default: false)
        #   ft_trend_flip_exit_min_profit_pts: float (default: 2.0)
        # ──────────────────────────────────────────────────────────────
        if is_15m:
            flip_exit = self._check_15m_signal_flip_exit(qty, current_price, total_pnl, contracts)
            if flip_exit:
                return flip_exit

        if not is_15m:
            # Trend-based exit: if trend flips against position
            trend = getattr(self.status, "hybrid_market_trend", "") or getattr(self.status, "last_trend", "")
            trend = str(trend).upper()
            if trend:
                logger.info(f"Trend-based exit check: current trend={trend} qty={qty}")
            else:
                # Compute trend from price history as fallback — use 40 bars for stability
                closes = [float(bar.get("close", 0.0)) for bar in self.price_history[-40:] if isinstance(bar, dict)]
                if len(closes) >= 10:
                    slope = np.polyfit(np.arange(len(closes)), closes, 1)[0]
                    # Require a steeper slope to declare trend change
                    trend = "DOWNTREND" if slope < -0.03 else "UPTREND" if slope > 0.03 else "NEUTRAL"
                    self.status.hybrid_market_trend = trend
                    logger.info(f"Trend-based exit: computed from price slope={slope:.5f} -> {trend}")
                else:
                    logger.info("Trend-based exit check: insufficient price history for trend calculation")
                    trend = ""
            if qty > 0 and trend == "DOWNTREND":
                return {"reason": "TREND_CHANGE", "action": "SELL", "quantity": contracts, "pnl": total_pnl}
            if qty < 0 and trend == "UPTREND":
                return {"reason": "TREND_CHANGE", "action": "BUY", "quantity": contracts, "pnl": total_pnl}

        # FEB 2026: 15m strategy max-hold time-stop
        ft_max_hold_min = getattr(self._m, "_ft_max_hold_minutes", 0)
        if ft_max_hold_min and ft_max_hold_min > 0:
            entry_ts_ft = getattr(position, "timestamp", None)
            if entry_ts_ft:
                try:
                    now_utc = datetime.now(timezone.utc)
                    entry_utc = entry_ts_ft if entry_ts_ft.tzinfo else entry_ts_ft.replace(tzinfo=timezone.utc)
                    hold_minutes = (now_utc - entry_utc).total_seconds() / 60.0
                    if hold_minutes >= ft_max_hold_min:
                        action = "SELL" if qty > 0 else "BUY"
                        logger.warning(
                            f"⏳ 15m MAX_HOLD_EXIT triggered: held {hold_minutes:.1f} min >= {ft_max_hold_min} min"
                        )
                        return {"reason": "MAX_HOLD_EXIT", "action": action, "quantity": contracts, "pnl": total_pnl}
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"15m max-hold check skipped: {exc}")

        # For 15m strategy: bracket + time-stop are the only exit mechanisms.
        # Skip legacy time-based and dollar-based exits.
        if is_15m:
            return None

        # Time-based exit check
        max_hold_hours = getattr(getattr(self.settings, "trading", None), "position_exit", None)
        max_hold_hours = getattr(max_hold_hours, "max_hold_time_hours", None) if max_hold_hours else None
        entry_ts = getattr(position, "timestamp", None)
        if entry_ts and max_hold_hours:
            try:
                duration = datetime.now(timezone.utc) - (
                    entry_ts if entry_ts.tzinfo else entry_ts.replace(tzinfo=timezone.utc)
                )
                if duration.total_seconds() >= max_hold_hours * 3600:
                    action = "SELL" if qty > 0 else "BUY"
                    logger.warning(
                        "⏳ Time-based exit triggered (hold {:.2f} hrs >= {:.2f} hrs)",
                        duration.total_seconds() / 3600,
                        max_hold_hours,
                    )
                    return {"reason": "TIME_EXIT", "action": action, "quantity": contracts, "pnl": total_pnl}
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Time-based exit check skipped: {exc}")

        # Dollar-based exits from config
        exit_cfg = getattr(getattr(self.settings, "trading", None), "position_exit", None)
        tp_dollars = getattr(exit_cfg, "profit_target_dollars", None) if exit_cfg else None
        sl_dollars = getattr(exit_cfg, "stop_loss_dollars", None) if exit_cfg else None
        # Fallback to legacy thresholds if not configured
        tp_dollars = tp_dollars if tp_dollars is not None else 200
        sl_dollars = sl_dollars if sl_dollars is not None else 100

        if tp_dollars is not None and total_pnl >= tp_dollars:
            action = "SELL" if qty > 0 else "BUY"
            return {"reason": "PROFIT_TARGET", "action": action, "quantity": contracts, "pnl": total_pnl}
        if sl_dollars is not None:
            sl_threshold = abs(sl_dollars)
            if sl_dollars < 0:
                logger.warning("STOP_LOSS dollars configured negative; using absolute value {:.2f}", sl_threshold)
            if sl_threshold > 0 and total_pnl <= -sl_threshold:
                action = "SELL" if qty > 0 else "BUY"
                return {"reason": "STOP_LOSS", "action": action, "quantity": contracts, "pnl": total_pnl}

        return None

    def _check_thesis_reversal_exit(
        self,
        qty: int,
        current_price: float,
        entry_price: float,
        total_pnl: float,
        contracts: int,
    ) -> Optional[Dict[str, Any]]:
        """Continuously flatten when the original entry thesis becomes invalid.

        This guard should *not* guess based on raw price wobble alone. Instead,
        it re-checks a small set of conditions implied by the original signal
        family using stored trade context and current indicators.
        """
        _trading_cfg = getattr(getattr(self._m, "settings", None), "trading", None)
        if not getattr(_trading_cfg, "ft_thesis_reversal_exit_enabled", True):
            return None

        if qty == 0 or contracts <= 0:
            return None

        open_ctx = getattr(self._m, "_open_trade_context", None) or {}
        if not open_ctx:
            return None

        stop_loss = open_ctx.get("stop_loss")
        take_profit = open_ctx.get("take_profit")
        if stop_loss is None or take_profit is None:
            return None

        try:
            stop_distance = abs(float(entry_price) - float(stop_loss))
        except Exception:
            return None
        if stop_distance <= 0:
            return None

        features = open_ctx.get("features") or getattr(self._m, "current_trade_features", {}) or {}
        signal_type = str(open_ctx.get("signal_type") or open_ctx.get("action") or "").upper()
        atr_raw = (
            features.get("atr")
            or features.get("ATR_14")
            or (self.price_history[-1].get("ATR_14") if self.price_history else 0.0)
            or 0.0
        )
        try:
            atr_value = float(atr_raw or 0.0)
        except Exception:
            atr_value = 0.0

        adverse_points = max(0.0, (entry_price - current_price) if qty > 0 else (current_price - entry_price))
        favorable_points = max(0.0, (current_price - entry_price) if qty > 0 else (entry_price - current_price))

        state_key = str(open_ctx.get("cycle_id") or "active")
        state = self._thesis_reversal_state.setdefault(
            state_key,
            {"best_favorable_points": 0.0, "armed": False},
        )
        state["best_favorable_points"] = max(float(state.get("best_favorable_points", 0.0)), favorable_points)

        arm_profit_pts = float(getattr(_trading_cfg, "ft_thesis_reversal_arm_profit_pts", 2.0) or 2.0)
        arm_fraction = float(getattr(_trading_cfg, "ft_thesis_reversal_arm_stop_fraction", 0.35) or 0.35)
        arm_threshold = max(arm_profit_pts, stop_distance * arm_fraction)
        if state["best_favorable_points"] >= arm_threshold:
            state["armed"] = True

        if not state.get("armed"):
            return None

        invalidation = self._evaluate_thesis_invalidation(
            signal_type=signal_type,
            qty=qty,
            current_price=current_price,
            entry_price=entry_price,
            stop_distance=stop_distance,
            atr_value=atr_value,
            adverse_points=adverse_points,
        )
        if not invalidation:
            return None

        action = "SELL" if qty > 0 else "BUY"
        direction = "LONG" if qty > 0 else "SHORT"
        logger.warning(
            "🚨 THESIS_REVERSAL_EXIT: {} {} invalidated | reason={} | best_favorable={:.2f}pts "
            "adverse={:.2f}pts | ATR={:.2f}",
            direction,
            signal_type or "UNKNOWN_SIGNAL",
            invalidation.get("reason"),
            float(state["best_favorable_points"]),
            adverse_points,
            atr_value,
        )
        try:
            log_structured_event(
                agent="exit_manager",
                event_type="THESIS_REVERSAL_EXIT",
                message=f"{direction} thesis invalidated; flattening early",
                payload={
                    "signal_type": signal_type,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "best_favorable_points": float(state["best_favorable_points"]),
                    "adverse_points": adverse_points,
                    "atr": atr_value,
                    "contracts": contracts,
                    "pnl": total_pnl,
                    "invalidation": invalidation,
                },
            )
        except Exception:
            pass

        return {
            "reason": "THESIS_REVERSAL_EXIT",
            "action": action,
            "quantity": contracts,
            "pnl": total_pnl,
        }

    def _evaluate_thesis_invalidation(
        self,
        signal_type: str,
        qty: int,
        current_price: float,
        entry_price: float,
        stop_distance: float,
        atr_value: float,
        adverse_points: float,
    ) -> Optional[Dict[str, Any]]:
        """Return a reason dict when current conditions invalidate the entry thesis."""
        features = getattr(self._m, "current_trade_features", {}) or {}
        ema9 = float(features.get("ema_9") or features.get("EMA_9") or 0.0)
        ema21 = float(features.get("ema_21") or features.get("EMA_21") or 0.0)
        macd_h = float(features.get("macd_hist") or features.get("MACDhist_12_26_9") or 0.0)
        adx = float(features.get("adx") or features.get("ADX_14") or 0.0)
        rsi = float(features.get("rsi") or features.get("RSI_14") or 0.0)

        min_adverse = max(stop_distance * 0.35, 1.5 if atr_value <= 0 else min(atr_value * 0.35, stop_distance * 0.6))
        if adverse_points < min_adverse:
            return None

        if "TREND_CONT_LONG" in signal_type:
            if ema9 and ema21 and ema9 < ema21 and macd_h < 0:
                return {"reason": "ema9_below_ema21_and_macd_negative", "signal_family": "trend_cont_long"}
            if adx and adx < 20.0 and macd_h < 0:
                return {"reason": "trend_strength_lost", "signal_family": "trend_cont_long"}

        if "TREND_CONT_SHORT" in signal_type:
            if ema9 and ema21 and ema9 > ema21 and macd_h > 0:
                return {"reason": "ema9_above_ema21_and_macd_positive", "signal_family": "trend_cont_short"}
            if adx and adx < 20.0 and macd_h > 0:
                return {"reason": "trend_strength_lost", "signal_family": "trend_cont_short"}

        if "EMA21_PB_LONG" in signal_type:
            if ema21 and current_price < ema21 and macd_h < 0:
                return {"reason": "lost_ema21_support_and_macd_negative", "signal_family": "pullback_long"}
            if rsi and rsi < 45 and adverse_points >= stop_distance * 0.45:
                return {"reason": "momentum_failed_after_pullback_long", "signal_family": "pullback_long"}

        if "EMA21_PB_SHORT" in signal_type:
            if ema21 and current_price > ema21 and macd_h > 0:
                return {"reason": "reclaimed_ema21_and_macd_positive", "signal_family": "pullback_short"}
            if rsi and rsi > 55 and adverse_points >= stop_distance * 0.45:
                return {"reason": "momentum_failed_after_pullback_short", "signal_family": "pullback_short"}

        return None

    # ------------------------------------------------------------------
    # FEB 20 2026: Structural support floor helper
    # ------------------------------------------------------------------

    def _get_support_floor(self) -> Optional[float]:
        """Read the structural support floor from the dynamic floor or env.

        MAR 2026: Prefers DynamicSupportFloor (auto-computed from PDL /
        weekly low / OR low).  Falls back to static env override.
        """
        # Prefer the dynamic floor
        dsf = getattr(self._m, "dynamic_support_floor", None)
        if dsf is not None:
            floor = dsf.get_floor()
            if floor is not None:
                return float(floor)
        # Fallback: static override on order coordinator
        coordinator = getattr(self._m, "order_coordinator", None)
        if coordinator is not None:
            floor = getattr(coordinator, "_static_floor_override", None)
            if floor is not None:
                return float(floor)
        # Fallback: env var
        import os as _os
        raw = _os.environ.get("STRUCTURAL_SUPPORT_FLOOR")
        if raw:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass
        return None

    # ------------------------------------------------------------------
    # Exit execution
    # ------------------------------------------------------------------

    async def execute_position_exit(
        self, exit_signal: Dict[str, Any], position, current_price: Optional[float] = None
    ) -> bool:
        """Execute position exit order honoring exit signal payload."""
        if not self.executor:
            return False

        reason = exit_signal.get("reason", "EXIT")

        # Handle trailing stop updates and breakeven stop updates (no exit, just update stop)
        if reason in ("TRAILING_STOP_UPDATE", "BREAKEVEN_STOP_UPDATE"):
            new_stop = exit_signal.get("new_stop_loss")
            if new_stop and hasattr(self.executor, "update_bracket_stop_loss"):
                try:
                    await self.executor.update_bracket_stop_loss(new_stop)
                    action_name = "Trailing stop" if reason == "TRAILING_STOP_UPDATE" else "Breakeven stop"
                    logger.info(f"✅ {action_name} updated to {new_stop:.2f}")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update stop: {e}")
                    return False
            return False

        action = exit_signal.get("action")
        quantity = int(exit_signal.get("quantity", 0))
        pnl = exit_signal.get("pnl", 0.0)
        move_stop_to_breakeven = exit_signal.get("move_stop_to_breakeven", False)

        if quantity <= 0 or action not in {"BUY", "SELL"}:
            logger.warning("Invalid exit signal payload: {}", exit_signal)
            return False

        price = current_price
        if price is None:
            try:
                price = await self.executor.get_current_price()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"⚠️ Could not fetch price for exit execution: {exc}")
                return False
        logger.info("🔄 Executing position exit: {} {} (reason={}, pnl={:.2f})", action, quantity, reason, pnl)
        try:
            # Cancel bracket orders before market exit (prevent SL/TP racing the exit fill)
            if reason in ("PROFIT_PROTECTION", "SIGNAL_FLIP_EXIT", "THESIS_REVERSAL_EXIT", "MAX_HOLD_EXIT") and self.executor:
                try:
                    logger.info("🚫 Cancelling bracket orders before {} market exit", reason)
                    await self.executor.cancel_all_orders()
                except Exception as cancel_exc:
                    logger.warning(f"⚠️ Error cancelling bracket orders: {cancel_exc}")

            # Determine position direction (LONG if qty > 0, SHORT if qty < 0)
            position_direction = "LONG" if position.quantity > 0 else "SHORT"

            # Best-effort: store the intended reason so the later position->flat transition
            # can finalize trade_outcomes with a specific exit_reason.
            trade_cycle_id = (
                getattr(self._m, "_current_entry_cycle_id", None)
                or getattr(self._m, "_current_cycle_id", None)
                or getattr(self._m, "current_trade_id", None)
            )
            self._m._note_pending_exit_reason(trade_cycle_id, reason)

            await self.executor.place_order(
                action=action,
                quantity=quantity,
                limit_price=price,
                stop_loss=None,
                take_profit=None,
                reduce_only=True,
                entry_price=price,
                metadata={
                    "exit_reason": reason,
                    "pnl": pnl,
                    "position_exit": True,
                    "trade_cycle_id": trade_cycle_id,
                },
            )

            # If partial profit taking, move stop to breakeven
            if move_stop_to_breakeven and hasattr(self.executor, "get_active_bracket_levels"):
                try:
                    logger.info("🔒 Moving stop to breakeven after partial profit exit")
                    entry_price_val = getattr(position, "avg_cost", 0.0) or 0.0
                    entry_normalized = self._m._normalize_entry_price(entry_price_val, price)

                    # Add small buffer (1-2 points) to ensure breakeven + small profit
                    buffer = 1.0
                    breakeven_stop = entry_normalized + buffer if position.quantity > 0 else entry_normalized - buffer

                    if hasattr(self.executor, "update_bracket_stop_loss"):
                        await self.executor.update_bracket_stop_loss(breakeven_stop)
                        logger.info(f"✅ Stop moved to breakeven: {breakeven_stop:.2f} (entry={entry_normalized:.2f})")
                except Exception as be_exc:
                    logger.warning(f"⚠️ Failed to move stop to breakeven: {be_exc}")

            # Notify MTF gate about position close
            self._m._notify_position_closed(
                close_reason=reason,
                direction=position_direction,
                pnl=pnl,
            )

            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(f"❌ Error executing position exit: {exc}")
            return False

    def _check_15m_signal_flip_exit(
        self,
        qty: int,
        current_price: float,
        total_pnl: float,
        contracts: int,
    ) -> Optional[Dict[str, Any]]:
        """Check if 15m indicators signal a reversal against the current position.

        Uses EMA9/EMA21 crossover + MACD histogram sign to detect a trend flip.
        Returns an exit dict if flip is confirmed and minimum profit condition is met.

        Config (under trading:):
          ft_trend_flip_exit_enabled: true/false
          ft_trend_flip_exit_min_profit_pts: float (MES points, default 2.0)
        """
        _trading_cfg = getattr(getattr(self._m, "settings", None), "trading", None)
        if not getattr(_trading_cfg, "ft_trend_flip_exit_enabled", False):
            return None

        _min_profit_pts = float(getattr(_trading_cfg, "ft_trend_flip_exit_min_profit_pts", 2.0) or 2.0)
        _min_profit_usd = _min_profit_pts * 5.0  # MES $5/point
        if total_pnl < _min_profit_usd:
            return None

        if len(self.price_history) < 30:
            return None

        try:
            import pandas as pd
            from ...features.feature_engineer import engineer_features

            df = pd.DataFrame(self.price_history)
            df.set_index("timestamp", inplace=True)
            features = engineer_features(df[["open", "high", "low", "close", "volume"]], None)
            if features.empty:
                return None
            latest = features.iloc[-1]
            ema9 = float(latest.get("EMA_9", 0.0) or 0.0)
            ema21 = float(latest.get("EMA_21", 0.0) or 0.0)
            macd_h = float(latest.get("MACDhist_12_26_9", 0.0) or 0.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Trend-flip exit: indicator computation skipped: {exc}")
            return None

        if ema9 == 0.0 or ema21 == 0.0:
            return None

        is_bearish_flip = (ema9 < ema21) and (macd_h < 0)
        is_bullish_flip = (ema9 > ema21) and (macd_h > 0)

        if qty > 0 and is_bearish_flip:
            logger.warning(
                f"🔄 SIGNAL_FLIP_EXIT (LONG→FLAT): EMA9={ema9:.2f} < EMA21={ema21:.2f}, "
                f"MACD_H={macd_h:.3f} — bearish flip detected, exiting {contracts} contracts "
                f"(open P&L=${total_pnl:.2f})"
            )
            try:
                log_structured_event(
                    agent="exit_manager",
                    event_type="SIGNAL_FLIP_EXIT",
                    message="Bearish flip: exiting LONG position",
                    payload={
                        "ema9": ema9,
                        "ema21": ema21,
                        "macd_h": macd_h,
                        "pnl": total_pnl,
                        "current_price": current_price,
                    },
                )
            except Exception:
                pass
            return {"reason": "SIGNAL_FLIP_EXIT", "action": "SELL", "quantity": contracts, "pnl": total_pnl}

        if qty < 0 and is_bullish_flip:
            logger.warning(
                f"🔄 SIGNAL_FLIP_EXIT (SHORT→FLAT): EMA9={ema9:.2f} > EMA21={ema21:.2f}, "
                f"MACD_H={macd_h:.3f} — bullish flip detected, exiting {contracts} contracts "
                f"(open P&L=${total_pnl:.2f})"
            )
            try:
                log_structured_event(
                    agent="exit_manager",
                    event_type="SIGNAL_FLIP_EXIT",
                    message="Bullish flip: exiting SHORT position",
                    payload={
                        "ema9": ema9,
                        "ema21": ema21,
                        "macd_h": macd_h,
                        "pnl": total_pnl,
                        "current_price": current_price,
                    },
                )
            except Exception:
                pass
            return {"reason": "SIGNAL_FLIP_EXIT", "action": "BUY", "quantity": contracts, "pnl": total_pnl}

        return None

    async def place_exit_order(
        self, action: str, quantity: int, exit_price: Optional[float] = None
    ) -> None:
        """Place a market order to exit existing position.

        SAFETY: Always uses reduce_only=True to prevent accidental position opening.
        """
        price = exit_price
        if price is None:
            try:
                price = await self.executor.get_current_price()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"⚠️ Exit order: could not fetch price, using 0. Error: {exc}")
                price = 0.0
        logger.info(f"🚪 Placing EXIT order request: {action} {quantity} contracts @ ~{float(price):.2f}")
        current_position = await self.executor.get_current_position() if self.executor else None
        if not current_position or current_position.quantity == 0:
            logger.warning("⚠️ Exit requested but no open position - skipping")
            return

        # Determine correct exit action based on current position (not signal action)
        exit_action = "SELL" if current_position.quantity > 0 else "BUY"
        exit_qty = min(quantity, abs(current_position.quantity))

        # SAFETY CHECK: If exit action would open opposite position, block it
        if (current_position.quantity > 0 and exit_action == "BUY") or (
            current_position.quantity < 0 and exit_action == "SELL"
        ):
            logger.error(
                f"🚨 EXIT ORDER BLOCKED: Would open position. "
                f"Current: {current_position.quantity}, Exit action: {exit_action}"
            )
            self._m._add_reason_code("EXIT_WOULD_OPEN_POSITION")
            log_structured_event(
                agent="live_manager",
                event_type="risk.exit_blocked",
                message="Exit order would open new position",
                payload={
                    "trade_cycle_id": self._m._current_cycle_id,
                    "current_position": current_position.quantity,
                    "exit_action": exit_action,
                },
            )
            return

        logger.info(f"🚪 Placing EXIT order: {exit_action} {exit_qty} contracts (reduce_only=True)")

        # Get entry trade_cycle_id from position metadata if available
        entry_cycle_id = (
            getattr(current_position, "entry_trade_cycle_id", None)
            or getattr(self._m, "_current_entry_cycle_id", None)
            or self._m._current_cycle_id
        )

        # Determine position direction for close notification
        position_direction = "LONG" if current_position.quantity > 0 else "SHORT"

        try:
            # Place market order to flatten position - ALWAYS reduce_only=True
            order_id = await self.executor.place_order(
                action=exit_action,
                quantity=exit_qty,
                limit_price=price,
                stop_loss=None,
                take_profit=None,
                reduce_only=True,
                entry_price=price,
                metadata={
                    "trade_cycle_id": entry_cycle_id,
                    "exit_order": True,
                    "allow_duplicate_exit": True,
                    "original_position": current_position.quantity,
                },
            )

            logger.info(f"✅ Exit order placed: ID={order_id}")

            # Notify MTF gate about position close
            self._m._notify_position_closed(
                close_reason="SIGNAL_EXIT",
                direction=position_direction,
                pnl=0.0,
            )

            # Broadcast exit
            await self._m._broadcast_order_update(
                {
                    "type": "EXIT",
                    "action": exit_action,
                    "quantity": exit_qty,
                    "price": price,
                    "order_id": order_id,
                }
            )

        except Exception as e:
            logger.error(f"❌ Failed to place exit order: {e}")
            await self._m._broadcast_error(f"Exit order failed: {e}")

    # ------------------------------------------------------------------
    # Helper / threshold methods
    # ------------------------------------------------------------------

    def get_exit_thresholds(self) -> Tuple[float, float, float]:
        """Return (profit_points, loss_points, max_hold_hours) with sensible fallbacks."""
        trading_cfg = getattr(self.settings, "trading", None) if self.settings else None
        point_value = getattr(self.contract_spec, "point_value", 1) or 1

        profit_points = None
        loss_points = None
        max_hold_hours = None

        pos_exit = getattr(trading_cfg, "position_exit", None) if trading_cfg else None
        if pos_exit:
            profit_dollars = getattr(pos_exit, "profit_target_dollars", None)
            loss_dollars = getattr(pos_exit, "stop_loss_dollars", None)
            max_hold_hours = getattr(pos_exit, "max_hold_time_hours", None)
            if profit_dollars:
                profit_points = float(profit_dollars) / point_value
            if loss_dollars:
                loss_points = float(loss_dollars) / point_value

        pm_cfg = getattr(trading_cfg, "position_management", None) if trading_cfg else None
        if profit_points is None and pm_cfg:
            profit_points = getattr(pm_cfg, "profit_target_points", None)
        if loss_points is None and pm_cfg:
            loss_points = getattr(pm_cfg, "stop_loss_points", None)
        if max_hold_hours is None and pm_cfg:
            max_hold_hours = getattr(pm_cfg, "force_exits_after_hours", None) or getattr(
                pm_cfg, "max_position_duration_hours", None
            )

        # Hard fallbacks to legacy constants
        if profit_points is None:
            profit_points = 20
        if loss_points is None:
            loss_points = 10
        if max_hold_hours is None:
            max_hold_hours = 4
        return float(profit_points), float(loss_points), float(max_hold_hours)

    def generate_exit_signal_for_short(
        self, current_price: float, position
    ) -> Optional[Dict[str, float]]:
        """Generate exit signals for short positions."""
        entry_price = float(getattr(position, "avg_cost", current_price) or current_price)
        normalized_entry = self._m._normalize_entry_price(entry_price, current_price)
        if normalized_entry != entry_price:
            logger.debug(
                "Normalized entry price from notional value: raw={raw} normalized={normalized} current={current}",
                raw=entry_price,
                normalized=normalized_entry,
                current=current_price,
            )
        entry_price = normalized_entry
        if abs(entry_price - current_price) > self.get_max_exit_price_gap(current_price):
            logger.warning(
                "Skipping exit checks: entry/current price gap too large (entry={entry}, current={current})",
                entry=entry_price,
                current=current_price,
            )
            return None

        profit_points_target, loss_points_limit, max_hold_hours = self.get_exit_thresholds()
        profit_points = entry_price - current_price
        if profit_points >= profit_points_target:
            return {"reason": "PROFIT_TARGET", "profit_points": profit_points}

        loss_points = current_price - entry_price
        if loss_points >= loss_points_limit:
            return {"reason": "STOP_LOSS", "loss_points": loss_points}

        age_hours = self.get_position_age_hours(position)
        if age_hours is not None and age_hours >= max_hold_hours:
            return {"reason": "TIME_EXIT", "hours_held": age_hours}

        return None

    def generate_exit_signal_for_long(
        self, current_price: float, position
    ) -> Optional[Dict[str, float]]:
        """Generate exit signals for long positions."""
        entry_price = float(getattr(position, "avg_cost", current_price) or current_price)
        normalized_entry = self._m._normalize_entry_price(entry_price, current_price)
        if normalized_entry != entry_price:
            logger.debug(
                "Normalized entry price from notional value: raw={raw} normalized={normalized} current={current}",
                raw=entry_price,
                normalized=normalized_entry,
                current=current_price,
            )
        entry_price = normalized_entry
        if abs(entry_price - current_price) > self.get_max_exit_price_gap(current_price):
            logger.warning(
                "Skipping exit checks: entry/current price gap too large (entry={entry}, current={current})",
                entry=entry_price,
                current=current_price,
            )
            return None

        profit_points_target, loss_points_limit, max_hold_hours = self.get_exit_thresholds()
        profit_points = current_price - entry_price
        if profit_points >= profit_points_target:
            return {"reason": "PROFIT_TARGET", "profit_points": profit_points}

        loss_points = entry_price - current_price
        if loss_points >= loss_points_limit:
            return {"reason": "STOP_LOSS", "loss_points": loss_points}

        age_hours = self.get_position_age_hours(position)
        if age_hours is not None and age_hours >= max_hold_hours:
            return {"reason": "TIME_EXIT", "hours_held": age_hours}

        return None

    def get_position_age_hours(self, position) -> Optional[float]:
        """Return position age (hours) if timestamp is available."""
        ts = getattr(position, "timestamp", None)
        if not ts:
            return None
        try:
            now = datetime.utcnow() if ts.tzinfo is None else datetime.now(tz=ts.tzinfo)
            return max(0.0, (now - ts).total_seconds() / 3600.0)
        except Exception:
            return None

    def get_max_exit_price_gap(self, current_price: float) -> float:
        """Return configurable/relative gap threshold for exit sanity checks."""
        configured_trading = getattr(self.settings, "trading", None) if self.settings else None
        custom_gap = getattr(configured_trading, "max_exit_price_gap", None) if configured_trading else None
        if custom_gap is not None:
            return float(custom_gap)
        return max(1000.0, abs(current_price) * 0.10)
