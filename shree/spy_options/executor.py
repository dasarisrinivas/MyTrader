"""SPY Options order executor — IB bracket orders via ib_insync (JUL 2 2026).

Turns the signal-only SPY bot into a trading bot. Design contract:

  * DEDICATED ib_insync connection (own host/port/client_id) so paper
    execution (4002) can run against the live-data connection (4001).
  * Every entry is a BRACKET: limit entry + attached stop-loss + take-profit
    (OCA pair). The stop lives AT IB — if this process or the gateway box
    dies, the position is still protected.
  * Stop distance = the signal's IV-adjusted premium stop (15/20/25% by IVR
    band, doc § 8.4); take-profit = +40% premium by default → ~1.6-2.6 : 1
    reward:risk, which lowers the breakeven win rate to ~38-45%.
  * STRICT QUALITY GATE — only directional (C/P), naked-long-structure,
    HIGH/EXTREME tier, green-edge, 0-2 DTE, tight-spread, sane-delta signals
    trade. Everything else stays Telegram-only. (Feed-level gates remain
    advisory by design; execution gates are hard.)
  * Fixed dollar risk per trade: contracts = risk_budget / (mid × 100 × stop%).
  * Portfolio guards: max open positions, max trades/day, daily realized loss
    limit, consecutive-stopout halt, no entries after 15:00 ET, 0DTE flatten
    at 15:50 ET, unfilled-entry timeout, position time stop.

The manager calls:  start() / maybe_execute(sig) / on_poll(spy_price) /
close_position(dedup_key, reason) / daily_reset() / close().
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from ib_insync import IB, LimitOrder, Option, Order, StopLimitOrder, StopOrder, Trade

from ..config.spy_options import SpyOptionsExecutionConfig
from ..utils.logger import logger
from ..utils.telegram_notifier import TelegramNotifier
from .edge_reality import gamma_accel_mult, hourly_theta_dollars
from .signal_engine import SpySignal

ET = ZoneInfo("America/New_York")


def _parse_et(hhmm: str, default: dtime) -> dtime:
    try:
        h, m = map(int, hhmm.split(":"))
        return dtime(h, m)
    except Exception:
        return default


def _round_tick(price: float) -> float:
    """SPY options trade in $0.01 ticks."""
    return max(0.01, round(price + 1e-9, 2))


@dataclass
class LivePosition:
    """One bracket-managed position keyed by the signal dedup_key."""

    key: str
    signal: SpySignal
    contract: Option
    qty: int
    entry_mid: float
    stop_pct: float
    placed_at: datetime                      # UTC
    parent: Trade
    take_profit: Trade
    stop_loss: Trade
    entry_filled: bool = False
    closed: bool = False
    close_reason: str = ""
    realized_pnl: float = 0.0
    exit_reported: bool = False


class SpyOptionsExecutor:
    """Bracket-order executor for gated SPY options signals."""

    def __init__(
        self,
        cfg: SpyOptionsExecutionConfig,
        telegram: Optional[TelegramNotifier] = None,
        analytics: Optional["object"] = None,   # AnalyticsDB — duck-typed to avoid import cycle
    ) -> None:
        self._cfg = cfg
        self._telegram = telegram
        # JUL 2 2026 (audit item #2): REAL fills feed the analytics DB so the
        # empirical WR loop can learn from actual option P&L instead of
        # delta×spot estimates.
        self._analytics = analytics
        self._ib = IB()
        self._connected = False
        self._reconnecting = False
        self._keepalive_task = None

        # Research log: full feature vector + realized P&L, one row per closed
        # trade. Accumulates the production dataset for feature-importance
        # analysis (backtest can't see live-only features). Never affects trading.
        from .research_log import TradeResearchLog
        self._research = TradeResearchLog()

        self._positions: Dict[str, LivePosition] = {}

        # Daily counters (reset by daily_reset())
        self._trades_today = 0
        self._realized_pnl_today = 0.0
        self._consecutive_stopouts = 0
        self._halted_reason: str = ""

        self._no_entry_after = _parse_et(cfg.no_new_entries_after_et, dtime(15, 0))
        self._flatten_0dte_at = _parse_et(cfg.flatten_0dte_at_et, dtime(15, 50))

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not self._cfg.enabled:
            return
        mode = "PAPER" if self._cfg.ibkr_port in (4002, 7497) else "LIVE"
        try:
            await self._ib.connectAsync(
                self._cfg.ibkr_host,
                self._cfg.ibkr_port,
                clientId=self._cfg.ibkr_client_id,
                timeout=20,
            )
            self._connected = True
            accounts = self._ib.managedAccounts()
            logger.info(
                "SPY EXECUTOR connected [{}] {}:{} clientId={} accounts={}",
                mode, self._cfg.ibkr_host, self._cfg.ibkr_port,
                self._cfg.ibkr_client_id, accounts,
            )
            # Safety: refuse silent live trading if user thinks this is paper.
            if mode == "LIVE":
                logger.warning(
                    "⚠️ SPY EXECUTOR is connected to a LIVE port ({}) — real "
                    "orders will be placed.", self._cfg.ibkr_port,
                )
            # Auto-reconnect: the IB Gateway restarts nightly (~midnight ET) for
            # maintenance, dropping this order connection. Without this the
            # executor stays stale-connected and every order fails at placement
            # (root cause of a "signal but no trade" morning). Mirror the data
            # client: keepalive ping + disconnect handler + backoff reconnect.
            if self._keepalive_task is None:
                self._keepalive_task = asyncio.ensure_future(self._keepalive_loop())
            self._ib.disconnectedEvent += self._on_disconnect
            await self._notify(
                f"🤖 <b>SPY Executor ONLINE</b> [{mode}]\n"
                f"Risk/trade: ${self._cfg.risk_per_trade_usd:.0f} · "
                f"TP +{self._cfg.take_profit_pct:.0f}% · "
                f"Stop {self._cfg.stop_pct_fallback:.0f}% (IV-adj) · "
                f"Max {self._cfg.max_trades_per_day}/day · "
                f"Daily loss limit ${self._cfg.daily_loss_limit_usd:.0f}"
            )
        except Exception as exc:
            self._connected = False
            logger.error(
                "SPY EXECUTOR could not connect to {}:{} — running SIGNAL-ONLY. "
                "({}) Is the {} gateway running?",
                self._cfg.ibkr_host, self._cfg.ibkr_port, exc, mode,
            )
            await self._notify(
                f"⚠️ <b>SPY Executor OFFLINE</b> — could not reach IB "
                f"{self._cfg.ibkr_host}:{self._cfg.ibkr_port} ({mode}). "
                "Bot continues signal-only."
            )

    async def _keepalive_loop(self) -> None:
        """Ping the order gateway every 30s; reconnect if the socket dropped."""
        while True:
            await asyncio.sleep(30)
            if not self._cfg.enabled:
                continue
            try:
                if self._ib.isConnected():
                    self._ib.reqCurrentTime()
                    if not self._connected:      # socket back but flag stale
                        self._connected = True
                elif not self._reconnecting:
                    logger.warning("EXEC keepalive: order gateway down — reconnecting")
                    await self._reconnect()
            except Exception as exc:
                logger.debug("EXEC keepalive ping failed: {}", exc)

    def _on_disconnect(self) -> None:
        """Handle an unexpected order-gateway disconnect (nightly IB restart)."""
        self._connected = False
        if not self._reconnecting:
            logger.warning("EXEC order gateway disconnected — scheduling reconnect")
            asyncio.ensure_future(self._reconnect())

    async def _reconnect(self) -> None:
        """Reconnect the order connection with backoff. Positions are held in
        memory and brackets rest at IB, so both survive the reconnect."""
        if self._reconnecting:
            return
        self._reconnecting = True
        try:
            for attempt in range(1, 6):
                try:
                    if self._ib.isConnected():
                        self._ib.disconnect()
                    await asyncio.sleep(min(attempt * 5, 30))
                    await self._ib.connectAsync(
                        self._cfg.ibkr_host, self._cfg.ibkr_port,
                        clientId=self._cfg.ibkr_client_id, timeout=30,
                    )
                    self._connected = True
                    logger.info("SPY EXECUTOR reconnected (attempt {}/5)", attempt)
                    return
                except Exception as exc:
                    logger.warning("EXEC reconnect attempt {}/5 failed: {}", attempt, exc)
            logger.error("EXEC: all 5 reconnect attempts failed — retry on next keepalive")
        finally:
            self._reconnecting = False

    async def close(self) -> None:
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        if self._connected:
            try:
                self._ib.disconnect()
            except Exception:
                pass
        self._connected = False

    def daily_reset(self) -> None:
        self._trades_today = 0
        self._realized_pnl_today = 0.0
        self._consecutive_stopouts = 0
        self._halted_reason = ""
        # Positions from a previous day should have been flattened/expired;
        # drop closed ones, keep any stragglers visible.
        self._positions = {k: p for k, p in self._positions.items() if not p.closed}

    # ── Quality gate ─────────────────────────────────────────────────────────

    def _gate(self, sig: SpySignal) -> Optional[str]:
        """Return a rejection reason, or None if the signal may trade."""
        c = self._cfg
        if not (self._cfg.enabled and self._connected):
            return "executor offline"
        if self._halted_reason:
            return f"halted for the day: {self._halted_reason}"
        if sig.right not in ("C", "P"):
            return "non-directional (straddle/BOTH)"
        if sig.structure not in ("", "LONG"):
            return f"structure {sig.structure} not supported (naked long only)"
        if sig.confidence_tier not in c.allowed_tiers:
            return f"tier {sig.confidence_tier} not in {c.allowed_tiers}"
        if c.require_green_edge and sig.edge_color != "green":
            return f"edge {sig.edge_color or 'unknown'} (need green after costs)"
        if sig.dte is None or sig.dte > c.max_dte:
            return f"DTE {sig.dte} > max {c.max_dte}"
        if not sig.expiry_date:
            return "no expiry_date on signal"
        if c.skip_event_risk and sig.event_risk:
            return f"event risk in {sig.event_minutes:.0f}m ({sig.next_event_title})"

        now_t = datetime.now(ET).time()
        if now_t >= self._no_entry_after:
            return f"after {c.no_new_entries_after_et} ET (theta-kill zone)"

        d = abs(sig.delta or 0.0)
        if d and not (c.min_abs_delta <= d <= c.max_abs_delta):
            return f"|delta| {d:.2f} outside [{c.min_abs_delta}, {c.max_abs_delta}]"

        if not (sig.bid and sig.ask and sig.bid > 0 and sig.ask > 0):
            return "no live bid/ask"
        mid = (sig.bid + sig.ask) / 2.0
        if mid < c.min_premium or mid > c.max_premium:
            return f"premium ${mid:.2f} outside [{c.min_premium}, {c.max_premium}]"
        spread_pct = (sig.ask - sig.bid) / mid * 100.0
        if spread_pct > c.max_entry_spread_pct:
            return f"spread {spread_pct:.1f}% > {c.max_entry_spread_pct}%"

        # ── Greeks gates (JUL 5 2026) ─────────────────────────────────────
        # Theta: premium burn per hour at the CURRENT session decay pace.
        # The same contract can pass at 10:00 and fail at 14:30.
        if sig.theta:
            theta_hr = hourly_theta_dollars(sig.theta)      # $/contract/hr
            contract_cost = mid * 100.0
            burn_pct_hr = theta_hr / contract_cost * 100.0
            max_burn = getattr(c, "max_theta_burn_pct_per_hour", 6.0)
            if burn_pct_hr > max_burn:
                return (
                    f"theta burn {burn_pct_hr:.1f}%/hr of premium "
                    f"> {max_burn:.1f}%/hr at this session pace"
                )
            # Required SPY drift merely to offset decay: theta $/hr vs $delta
            if sig.delta and sig.spy_price:
                dollar_delta = abs(sig.delta) * 100.0 * sig.spy_price  # $/contract per 100% SPY move
                drift_pct_hr = theta_hr / dollar_delta * 100.0         # SPY %/hr needed
                max_drift = getattr(c, "max_breakeven_drift_pct_per_hour", 0.15)
                if drift_pct_hr > max_drift:
                    return (
                        f"needs {drift_pct_hr:.2f}%/hr SPY drift just to pay theta "
                        f"(max {max_drift:.2f}%/hr)"
                    )

        # IV crush: naked longs at extreme IV rank lose even when direction
        # is right, once vol mean-reverts. (Straddles never reach this gate —
        # non-directional signals are rejected above.)
        max_ivr = getattr(c, "max_ivr_naked_long", 75.0)
        if sig.iv_rank and sig.iv_rank > max_ivr:
            return (
                f"IV rank {sig.iv_rank:.0f} > {max_ivr:.0f} — long premium at "
                f"rich vol (crush risk)"
            )

        # Gamma bomb: no fresh 0DTE entries once effective gamma has
        # accelerated past the configured multiple (2.5x = <60 min to close).
        if sig.dte == 0:
            accel = gamma_accel_mult(0)
            max_accel = getattr(c, "max_0dte_gamma_accel", 2.5)
            if accel >= max_accel:
                return f"0DTE gamma zone (effective gamma {accel:.1f}x ≥ {max_accel:.1f}x)"

        # ── Cross-asset veto (JUL 5 2026) ─────────────────────────────────
        # Never buy a SPY breakout the rest of the tape refuses to confirm.
        # Active session-extreme divergence (SPY new high, QQQ didn't → calls
        # blocked; SPY new low, QQQ held → puts blocked). Strong opposing QQQ
        # relative strength also vetoes.
        if getattr(c, "require_cross_asset_confirm", True):
            div = getattr(sig, "cross_asset_divergence", "NONE")
            if div == "BEARISH_NONCONFIRM" and sig.right == "C":
                return "QQQ did not confirm SPY session high (bearish non-confirm)"
            if div == "BULLISH_NONCONFIRM" and sig.right == "P":
                return "QQQ held while SPY made session low (bullish non-confirm)"
            qqq_rs = getattr(sig, "qqq_rs", 0.0)
            max_opposed = getattr(c, "max_opposed_qqq_rs", 0.35)
            if sig.right == "C" and qqq_rs <= -max_opposed:
                return f"QQQ lagging SPY by {abs(qqq_rs):.2f}pp — narrow rally, no call entries"
            if sig.right == "P" and qqq_rs >= max_opposed:
                return f"QQQ leading SPY by {qqq_rs:.2f}pp — tech bid, no put entries"

        open_count = sum(1 for p in self._positions.values() if not p.closed)
        if open_count >= c.max_open_positions:
            return f"max open positions ({open_count})"
        if self._trades_today >= c.max_trades_per_day:
            return f"max trades/day ({self._trades_today})"
        if self._realized_pnl_today <= -abs(c.daily_loss_limit_usd):
            self._halted_reason = (
                f"daily loss limit ${abs(self._realized_pnl_today):.0f}"
            )
            return self._halted_reason
        if sig.dedup_key in self._positions and not self._positions[sig.dedup_key].closed:
            return "already holding this signal"
        return None

    def _size(self, mid: float, stop_pct: float) -> int:
        risk_per_ct = mid * 100.0 * (stop_pct / 100.0)
        if risk_per_ct <= 0:
            return 0
        qty = int(self._cfg.risk_per_trade_usd // risk_per_ct)
        return max(0, min(qty, self._cfg.max_contracts))

    # ── Entry ────────────────────────────────────────────────────────────────

    async def maybe_execute(self, sig: SpySignal) -> bool:
        """Gate → size → place bracket. Returns True when an order was placed."""
        reason = self._gate(sig)
        if reason is not None:
            if self._cfg.enabled:
                logger.info(
                    "EXEC gate: {} {}{} not traded — {}",
                    sig.signal_type.value, sig.strike, sig.right, reason,
                )
            return False

        mid = _round_tick((sig.bid + sig.ask) / 2.0)

        # Bracket geometry. Default: fixed premium % (IV-adjusted stop, 40% TP).
        # Structure-based (TREND_CONTINUATION with a structural_stop): translate
        # the SPY invalidation level into a premium stop via delta and target
        # continuation_target_r × it — anchored to where the thesis is right/wrong.
        stop_pct = float(sig.iv_adjusted_stop_pct or self._cfg.stop_pct_fallback)
        tp_pct = float(self._cfg.take_profit_pct)
        bracket_basis = "premium"
        if (
            getattr(self._cfg, "use_structural_bracket", False)
            and getattr(sig, "structural_stop", 0.0)
            and sig.delta and abs(sig.delta) > 0.05
            and sig.spy_price and mid > 0
        ):
            stop_dist_spy = abs(sig.spy_price - sig.structural_stop)
            prem_stop_dist = abs(sig.delta) * stop_dist_spy      # $ premium move to stop
            raw_stop_pct = prem_stop_dist / mid * 100.0
            clamped = max(
                self._cfg.structural_stop_pct_min,
                min(self._cfg.structural_stop_pct_max, raw_stop_pct),
            )
            if clamped > 0:
                stop_pct = clamped
                tp_pct = clamped * self._cfg.continuation_target_r
                bracket_basis = "structural"

        qty = self._size(mid, stop_pct)
        if qty < 1:
            logger.info(
                "EXEC gate: {} {}{} not traded — 1-contract stop-risk "
                "${:.0f} exceeds budget ${:.0f}",
                sig.signal_type.value, sig.strike, sig.right,
                mid * 100 * stop_pct / 100.0, self._cfg.risk_per_trade_usd,
            )
            return False

        contract = Option(
            symbol="SPY",
            lastTradeDateOrContractMonth=sig.expiry_date,
            strike=float(sig.strike),
            right=sig.right,
            exchange="SMART",
            currency="USD",
            multiplier="100",
            # Disambiguate from "2SPY" (corporate-action-adjusted class) —
            # without this, qualifyContracts returns Ambiguous and fails.
            # Caught by scripts/test_spy_execution.py on JUL 2 2026.
            tradingClass="SPY",
        )
        try:
            qualified = await self._ib.qualifyContractsAsync(contract)
            if not qualified:
                logger.error("EXEC: could not qualify {}", contract)
                return False
            contract = qualified[0]
        except Exception as exc:
            logger.error("EXEC: qualify failed for {}: {}", contract, exc)
            return False

        # Bracket prices — entry capped at the ask (never chase above it).
        entry_limit = _round_tick(min(sig.ask, mid + 0.02))
        tp_price = _round_tick(entry_limit * (1 + tp_pct / 100.0))
        sl_stop = _round_tick(entry_limit * (1 - stop_pct / 100.0))
        if bracket_basis == "structural":
            logger.info(
                "EXEC structural bracket: {} {}{} SPY_stop=${:.2f} (Δ={:.2f}) "
                "→ prem stop {:.0f}% / TP {:.0f}% ({:.1f}R)",
                sig.signal_type.value, sig.strike, sig.right,
                sig.structural_stop, sig.delta, stop_pct, tp_pct,
                self._cfg.continuation_target_r,
            )

        parent = LimitOrder("BUY", qty, entry_limit)
        parent.tif = "DAY"
        parent.transmit = False
        if self._cfg.account:
            parent.account = self._cfg.account

        tp = LimitOrder("SELL", qty, tp_price)
        tp.tif = "DAY"
        tp.transmit = False

        if self._cfg.stop_type == "stop":
            sl: Order = StopOrder("SELL", qty, sl_stop)
        else:
            sl_limit = _round_tick(
                sl_stop * (1 - self._cfg.stop_limit_buffer_pct / 100.0)
            )
            sl = StopLimitOrder("SELL", qty, sl_limit, sl_stop)
        sl.tif = "DAY"
        sl.transmit = True   # transmitting the last child releases the bracket

        try:
            parent_trade = self._ib.placeOrder(contract, parent)
            oca = f"spy_{parent_trade.order.orderId}"
            for child in (tp, sl):
                child.parentId = parent_trade.order.orderId
                child.ocaGroup = oca
                child.ocaType = 1
                if self._cfg.account:
                    child.account = self._cfg.account
            tp_trade = self._ib.placeOrder(contract, tp)
            sl_trade = self._ib.placeOrder(contract, sl)
        except Exception as exc:
            logger.opt(exception=True).error("EXEC: bracket placement failed: {}", exc)
            return False

        pos = LivePosition(
            key=sig.dedup_key,
            signal=sig,
            contract=contract,
            qty=qty,
            entry_mid=entry_limit,
            stop_pct=stop_pct,
            placed_at=datetime.utcnow(),
            parent=parent_trade,
            take_profit=tp_trade,
            stop_loss=sl_trade,
        )
        self._positions[sig.dedup_key] = pos
        self._trades_today += 1

        logger.info(
            "🟢 EXEC ORDER: BUY {}x SPY {} {}{} @{:.2f} LMT  "
            "TP {:.2f} (+{:.0f}%)  SL {:.2f} (-{:.0f}%{})  risk≈${:.0f}",
            qty, sig.expiry_date, sig.strike, sig.right, entry_limit,
            tp_price, self._cfg.take_profit_pct, sl_stop, stop_pct,
            "" if self._cfg.stop_type == "stop" else ", stop-limit",
            qty * entry_limit * 100 * stop_pct / 100.0,
        )
        await self._notify(
            f"🟢 <b>ORDER PLACED</b> — {sig.signal_type.value}\n"
            f"BUY {qty}× SPY {sig.expiry_date} {sig.strike:.0f}{sig.right} "
            f"@ ${entry_limit:.2f} LMT\n"
            f"🎯 TP ${tp_price:.2f} (+{self._cfg.take_profit_pct:.0f}%)  "
            f"🛑 SL ${sl_stop:.2f} (−{stop_pct:.0f}%)\n"
            f"Risk ≈ ${qty * entry_limit * 100 * stop_pct / 100.0:.0f} · "
            f"Trade {self._trades_today}/{self._cfg.max_trades_per_day} today"
        )
        return True

    # ── Poll-cycle maintenance ───────────────────────────────────────────────

    async def on_poll(self) -> None:
        """Entry timeouts, fill detection, time stops, 0DTE flatten, P&L."""
        if not (self._cfg.enabled and self._connected):
            return
        now_utc = datetime.utcnow()
        now_et = datetime.now(ET).time()

        for pos in list(self._positions.values()):
            if pos.closed:
                continue
            st = pos.parent.orderStatus.status
            filled_qty = int(pos.parent.orderStatus.filled or 0)

            # 1. Entry fill detection / notification
            if not pos.entry_filled and filled_qty > 0:
                pos.entry_filled = True
                avg = pos.parent.orderStatus.avgFillPrice or pos.entry_mid
                logger.info(
                    "✅ EXEC FILL: {}x {} @{:.2f}", filled_qty, pos.contract.localSymbol, avg,
                )
                if self._analytics is not None:
                    try:
                        self._analytics.record_fill_entry(pos.signal, avg, filled_qty)
                    except Exception as exc:
                        logger.warning("record_fill_entry failed: {}", exc)
                await self._notify(
                    f"✅ <b>FILLED</b> {filled_qty}× {pos.contract.localSymbol} "
                    f"@ ${avg:.2f} — bracket active (TP/SL at IB)"
                )

            # 2. Unfilled entry timeout → cancel bracket
            age_s = (now_utc - pos.placed_at).total_seconds()
            if not pos.entry_filled and age_s > self._cfg.entry_timeout_s:
                if st not in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
                    logger.info(
                        "⏳ EXEC: entry unfilled {}s — cancelling {}",
                        int(age_s), pos.contract.localSymbol,
                    )
                    try:
                        self._ib.cancelOrder(pos.parent.order)
                    except Exception as exc:
                        logger.warning("EXEC cancel failed: {}", exc)
                    pos.closed = True
                    pos.close_reason = "entry timeout (never filled)"
                    self._trades_today = max(0, self._trades_today - 1)  # give the slot back
                    await self._notify(
                        f"⏳ <b>CANCELLED</b> {pos.contract.localSymbol} — "
                        f"entry not filled in {self._cfg.entry_timeout_s}s"
                    )
                continue

            # 3. Bracket exit detection (TP or SL filled)
            if pos.entry_filled:
                exit_trade = None
                exit_label = ""
                if int(pos.take_profit.orderStatus.filled or 0) >= pos.qty:
                    exit_trade, exit_label = pos.take_profit, "take_profit"
                elif int(pos.stop_loss.orderStatus.filled or 0) >= pos.qty:
                    exit_trade, exit_label = pos.stop_loss, "stop_loss"
                if exit_trade is not None:
                    entry_px = pos.parent.orderStatus.avgFillPrice or pos.entry_mid
                    exit_px = exit_trade.orderStatus.avgFillPrice or 0.0
                    pnl = (exit_px - entry_px) * pos.qty * 100.0
                    self._register_close(pos, exit_label, pnl, exit_premium=exit_px)
                    await self._notify(
                        f"{'🎯' if exit_label == 'take_profit' else '🛑'} "
                        f"<b>{exit_label.upper()}</b> {pos.contract.localSymbol}\n"
                        f"${entry_px:.2f} → ${exit_px:.2f}  "
                        f"P&L <b>${pnl:+.0f}</b>  (day: ${self._realized_pnl_today:+.0f})"
                    )
                    continue

                # 4. Position time stop
                held_min = age_s / 60.0
                if held_min >= self._cfg.max_hold_minutes:
                    await self.close_position(
                        pos.key, f"time stop {self._cfg.max_hold_minutes}min"
                    )
                    continue

                # 5. 0DTE end-of-day flatten
                if (pos.signal.dte or 0) == 0 and now_et >= self._flatten_0dte_at:
                    await self.close_position(pos.key, "0DTE EOD flatten")

        # Purge long-dead entries to keep the dict small
        self._positions = {
            k: p for k, p in self._positions.items()
            if not (p.closed and (now_utc - p.placed_at).total_seconds() > 6 * 3600)
        }

    # ── Exits ────────────────────────────────────────────────────────────────

    async def close_position(self, key: str, reason: str) -> bool:
        """Cancel the bracket children and market-close any filled quantity."""
        pos = self._positions.get(key)
        if pos is None or pos.closed:
            return False

        filled_qty = int(pos.parent.orderStatus.filled or 0)
        try:
            # Cancel whichever orders are still working
            for tr in (pos.parent, pos.take_profit, pos.stop_loss):
                if tr.orderStatus.status not in (
                    "Filled", "Cancelled", "ApiCancelled", "Inactive",
                ):
                    try:
                        self._ib.cancelOrder(tr.order)
                    except Exception:
                        pass

            if filled_qty > 0:
                # How much has already been sold by a partially-filled child?
                sold = int(pos.take_profit.orderStatus.filled or 0) + int(
                    pos.stop_loss.orderStatus.filled or 0
                )
                remaining = filled_qty - sold
                if remaining > 0:
                    mkt = Order(action="SELL", orderType="MKT", totalQuantity=remaining)
                    mkt.tif = "DAY"
                    if self._cfg.account:
                        mkt.account = self._cfg.account
                    exit_trade = self._ib.placeOrder(pos.contract, mkt)
                    try:
                        await asyncio.wait_for(exit_trade.fillEvent, timeout=15)
                    except Exception:
                        pass
                    # Capture the REAL exit fill. avgFillPrice populates async, so
                    # retry a few times before giving up — never silently record a
                    # false breakeven from entry_px.
                    entry_px = pos.parent.orderStatus.avgFillPrice or pos.entry_mid
                    exit_px = 0.0
                    for _ in range(6):
                        exit_px = exit_trade.orderStatus.avgFillPrice or 0.0
                        if exit_px > 0:
                            break
                        await asyncio.sleep(1.0)
                    if exit_px > 0:
                        pnl = (exit_px - entry_px) * remaining * 100.0
                        self._register_close(
                            pos, f"bot exit: {reason}", pnl, exit_premium=exit_px
                        )
                        logger.info(
                            "EXEC CLOSE {} @ ${:.2f} (entry ${:.2f})  P&L ${:+.0f} — {} "
                            "(day ${:+.0f})",
                            pos.contract.localSymbol, exit_px, entry_px, pnl, reason,
                            self._realized_pnl_today,
                        )
                        await self._notify(
                            f"🔻 <b>CLOSED</b> {pos.contract.localSymbol} — {reason}\n"
                            f"${entry_px:.2f} → ${exit_px:.2f}  "
                            f"P&L <b>${pnl:+.0f}</b>  (day: ${self._realized_pnl_today:+.0f})"
                        )
                    else:
                        # Exit fill never confirmed after retries — mark closed to
                        # avoid re-sending the close, but do NOT record a fabricated
                        # breakeven. Flag loudly so the real fill is reconciled from
                        # the IB account rather than trusting a bad record.
                        pos.closed = True
                        pos.close_reason = f"{reason} (exit fill unconfirmed)"
                        logger.warning(
                            "EXEC CLOSE {} — {} — exit fill UNCONFIRMED after retries; "
                            "P&L NOT recorded (reconcile from IB account)",
                            pos.contract.localSymbol, reason,
                        )
                else:
                    # Already exited via a bracket child; that fill's P&L is
                    # recorded by on_poll's fill detection.
                    self._register_close(pos, reason, 0.0)
                    logger.info(
                        "EXEC CLOSE {} — {} (exited via bracket child)",
                        pos.contract.localSymbol, reason,
                    )
            else:
                # Entry order NEVER filled — this is a CANCEL, not a position
                # close. No fill, no P&L, and it must not look like a trade.
                pos.closed = True
                pos.close_reason = f"unfilled entry cancelled: {reason}"
                logger.info(
                    "EXEC CANCEL (entry never filled): {} — {}",
                    pos.contract.localSymbol, reason,
                )
            return True
        except Exception as exc:
            logger.opt(exception=True).error(
                "EXEC close_position({}) failed: {}", key, exc,
            )
            return False

    def has_open_position(self, key: str) -> bool:
        pos = self._positions.get(key)
        return bool(pos and not pos.closed)

    # ── Internals ────────────────────────────────────────────────────────────

    def _register_close(
        self,
        pos: LivePosition,
        reason: str,
        pnl: float,
        exit_premium: Optional[float] = None,
    ) -> None:
        pos.closed = True
        pos.close_reason = reason
        pos.realized_pnl = pnl
        self._realized_pnl_today += pnl

        # REAL fill → analytics (JUL 2 2026, audit item #2). Ground truth for
        # the empirical WR loop — never estimated.
        if self._analytics is not None and exit_premium is not None:
            try:
                self._analytics.record_fill_exit(
                    pos.signal, exit_premium, pnl, reason
                )
            except Exception as exc:
                logger.warning("record_fill_exit failed: {}", exc)

        # Research log: full feature vector at entry + realized option P&L.
        try:
            s = pos.signal
            entry = pos.entry_mid or 0.0
            exitp = exit_premium if exit_premium is not None else 0.0
            risk_frac = (pos.stop_pct or 0.0) / 100.0
            pnl_r = (exitp - entry) / (entry * risk_frac) if (entry and risk_frac) else 0.0
            hold_min = (datetime.utcnow() - pos.placed_at).total_seconds() / 60.0
            sig_fields = {
                k: v for k, v in s.__dict__.items()
                if k != "research_ctx" and not k.startswith("_")
            }
            self._research.log_trade({
                "trade_id": pos.key,
                "entry_at": pos.placed_at.isoformat(),
                "closed_at": datetime.utcnow().isoformat(),
                "signal_type": s.signal_type.value,
                "right": s.right, "strike": s.strike, "dte": s.dte, "qty": pos.qty,
                "entry_premium": entry, "exit_premium": exitp,
                "pnl_usd": pnl, "pnl_r": round(pnl_r, 3),
                "hold_min": round(hold_min, 1),
                "exit_reason": reason,
                "outcome": "win" if pnl > 0 else "loss",
                "signal_fields": sig_fields,
                "ext_ctx": getattr(s, "research_ctx", {}) or {},
            })
        except Exception as exc:
            logger.debug("research log write skipped: {}", exc)
        if reason == "stop_loss":
            self._consecutive_stopouts += 1
        elif pnl > 0:
            self._consecutive_stopouts = 0
        if (
            self._consecutive_stopouts >= self._cfg.max_consecutive_stopouts
            and not self._halted_reason
        ):
            self._halted_reason = (
                f"{self._consecutive_stopouts} consecutive stop-outs"
            )
            logger.warning("EXEC HALT for the day: {}", self._halted_reason)
        if (
            self._realized_pnl_today <= -abs(self._cfg.daily_loss_limit_usd)
            and not self._halted_reason
        ):
            self._halted_reason = (
                f"daily loss limit (${self._realized_pnl_today:+.0f})"
            )
            logger.warning("EXEC HALT for the day: {}", self._halted_reason)

    async def _notify(self, msg: str) -> None:
        if self._telegram is None:
            return
        try:
            await self._telegram.send_message(msg)
        except Exception as exc:
            logger.warning("EXEC telegram notify failed: {}", exc)
