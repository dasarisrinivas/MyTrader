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


# Families in PILOT status: minimum size (1 contract), rolling auto-kill on
# negative EV, independent metrics via signal_type in analytics/research logs.
# Promotion out of pilot = scorecard evidence (see scripts/strategy_scorecard.py).
_PILOT_FAMILIES = {"PC_AFTERNOON_FLOW"}


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
    # Deferred exit-fill confirmations (JUL 17 2026): closes whose
    # avgFillPrice hadn't populated inside the synchronous retry window.
    # Confirmed on later polls so the P&L is never silently lost.
    # Each: {"trade","label","qty","entry_px","attempts","kind"}.
    pending_exits: List[dict] = field(default_factory=list)
    # Entry-chase state (JUL 13 2026): bump the resting entry limit toward the
    # live ask while unfilled, bounded by count and a hard % cap.
    orig_entry_limit: float = 0.0            # original entry limit — chase cap anchor
    reprice_count: int = 0
    last_reprice_at: Optional[datetime] = None   # UTC


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
            # Startup reconciliation: surface any SPY option positions/orders
            # left at IB by a crashed prior process (audit 2026-07-13, P0-2).
            await self._reconcile_open_state("startup")
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
                    # AWAIT the async variant — the sync reqCurrentTime() re-enters
                    # the running loop and always raises "event loop already running"
                    # (was swallowed at DEBUG, so the ping never completed and the
                    # reconnect path never fired). Audit 2026-07-13.
                    await self._ib.reqCurrentTimeAsync()
                    if not self._connected:      # socket back but flag stale
                        self._connected = True
                elif not self._reconnecting:
                    logger.warning("EXEC keepalive: order gateway down — reconnecting")
                    await self._reconnect()
            except Exception as exc:
                logger.warning("EXEC keepalive ping failed: {}", exc)

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
                    # Rebind stale Trade objects to the reconnected session so
                    # open brackets stay tracked (audit 2026-07-13, P0-2).
                    await self._reconcile_open_state("reconnect")
                    return
                except Exception as exc:
                    logger.warning("EXEC reconnect attempt {}/5 failed: {}", attempt, exc)
            logger.error("EXEC: all 5 reconnect attempts failed — retry on next keepalive")
        finally:
            self._reconnecting = False

    async def _reconcile_open_state(self, context: str) -> None:
        """After a (re)connect, rebind in-memory LivePosition Trade objects to
        the reconnected IB's live Trade objects (matched by orderId), and ALERT
        on anything that can't be reconciled.

        The old Trade objects are frozen at their pre-disconnect state, so
        without this on_poll reads stale order status and a live bracket becomes
        untracked after any reconnect (audit 2026-07-13). This is deliberately
        NON-destructive: it only rebinds references and surfaces mismatches for
        manual review — it never cancels or flattens anything, because guessing
        wrong on live positions is worse than an operator alert.
        """
        if not self._connected:
            return
        try:
            await self._ib.reqOpenOrdersAsync()
        except Exception as exc:
            logger.warning("EXEC reconcile: reqOpenOrders failed: {}", exc)
        try:
            by_id = {
                t.order.orderId: t for t in self._ib.trades()
                if getattr(t, "order", None) is not None
            }
        except Exception as exc:
            logger.warning("EXEC reconcile: trades() failed: {}", exc)
            by_id = {}

        rebound = 0
        unresolved: List[str] = []
        for pos in self._positions.values():
            if pos.closed:
                continue
            for attr in ("parent", "take_profit", "stop_loss"):
                tr = getattr(pos, attr, None)
                if tr is None or getattr(tr, "order", None) is None:
                    continue
                fresh = by_id.get(tr.order.orderId)
                if fresh is not None and fresh is not tr:
                    setattr(pos, attr, fresh)
                    rebound += 1
            pid = getattr(pos.parent.order, "orderId", None) if pos.parent else None
            if pid is not None and pid not in by_id and not pos.entry_filled:
                unresolved.append(pos.contract.localSymbol)

        # Orphans AT IB: open SPY OPTION positions the bot isn't tracking.
        orphans: List[str] = []
        try:
            tracked = {p.contract.localSymbol for p in self._positions.values() if not p.closed}
            for ibpos in self._ib.positions():
                c = ibpos.contract
                if (getattr(c, "secType", "") == "OPT"
                        and getattr(c, "symbol", "") == "SPY"
                        and ibpos.position != 0
                        and c.localSymbol not in tracked):
                    orphans.append(f"{c.localSymbol} x{ibpos.position:g}")
        except Exception as exc:
            logger.warning("EXEC reconcile: positions() failed: {}", exc)

        tracked_n = sum(1 for p in self._positions.values() if not p.closed)
        logger.info(
            "EXEC reconcile ({}): rebound {} order ref(s), {} tracked position(s), "
            "{} unresolved, {} orphan(s) at IB",
            context, rebound, tracked_n, len(unresolved), len(orphans),
        )
        if unresolved or orphans:
            lines = [f"⚠️ <b>EXEC reconcile ({context})</b> — manual review:"]
            if unresolved:
                lines.append(f"• {len(unresolved)} tracked entr(y/ies) not found live: "
                             + ", ".join(unresolved[:5]))
            if orphans:
                lines.append(f"• {len(orphans)} untracked SPY option position(s) at IB: "
                             + ", ".join(orphans[:5]))
            await self._notify("\n".join(lines))

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
        # max_dte is a TRADING-session budget. Prefer the manager-computed
        # trading-day count (weekends/NYSE holidays excluded) so a Thu→Mon
        # contract (4 calendar / 2 trading) passes while a genuine 4-session
        # contract is still rejected. Calendar fallback when unset.
        dte_eff = getattr(sig, "trading_dte", None)
        if dte_eff is None:
            dte_eff = sig.dte
        if dte_eff is None or dte_eff > c.max_dte:
            return (
                f"DTE {sig.dte} cal / {dte_eff} trading > max {c.max_dte}"
                if dte_eff is not None else "DTE unknown"
            )
        if not sig.expiry_date:
            return "no expiry_date on signal"
        if c.skip_event_risk and sig.event_risk:
            return f"event risk in {sig.event_minutes:.0f}m ({sig.next_event_title})"

        now_t = datetime.now(ET).time()
        if now_t >= self._no_entry_after:
            return f"after {c.no_new_entries_after_et} ET (theta-kill zone)"

        # Greeks must be present for a directional single-leg trade. A missing
        # delta (0.0 — IB delayed-data mode zeroes every greek) must FAIL CLOSED:
        # the old `if d and ...` SKIPPED the band check when delta was 0, so a
        # phantom zero-greek contract passed the delta gate (and, downstream, the
        # theta gate's `if sig.theta:` skipped too). (audit 2026-07-13)
        d = abs(sig.delta or 0.0)
        if d < 0.01:
            return "missing/zero delta — no valid greeks (fail-closed)"
        if not (c.min_abs_delta <= d <= c.max_abs_delta):
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

        # Pilot auto-kill (strategy audit 2026-07-19): a pilot family with
        # rolling negative realized EV stops trading — automatic demotion,
        # no human in the loop. Re-enable = clear history via restart after
        # review, or promote out of pilot status in code.
        if sig.signal_type.value in _PILOT_FAMILIES:
            hist = getattr(self, "_family_pnls", {}).get(sig.signal_type.value, [])
            recent = hist[-6:]
            if len(recent) >= 4 and sum(recent) < 0:
                return (
                    f"pilot {sig.signal_type.value} auto-disabled: rolling EV "
                    f"${sum(recent):+.0f} over last {len(recent)} trades"
                )
        return None

    def _size(self, mid: float, stop_pct: float) -> int:
        risk_per_ct = mid * 100.0 * (stop_pct / 100.0)
        if risk_per_ct <= 0:
            return 0
        qty = int(self._cfg.risk_per_trade_usd // risk_per_ct)
        return max(0, min(qty, self._cfg.max_contracts))

    async def _live_quote(self, contract: Option) -> tuple[float, float]:
        """Fetch a FRESH (bid, ask) for `contract` on the executor's own IB
        connection. Returns (0.0, 0.0) on any failure — callers MUST fall back
        to the signal's snapshot quote and never price an order off zeros.

        This is the fix for the stale-quote no-fill: sig.bid/sig.ask are captured
        at signal-enrichment time (often minutes before placement); in a fast
        move the live ask has already run away, so a limit priced off the
        snapshot never crosses. We re-quote right before pricing and on each
        chase step.
        """
        try:
            tickers = await asyncio.wait_for(
                self._ib.reqTickersAsync(contract), timeout=3.0
            )
        except Exception as exc:
            logger.warning("EXEC live-quote failed for {}: {}",
                           getattr(contract, "localSymbol", contract), exc)
            return 0.0, 0.0
        if not tickers:
            return 0.0, 0.0
        t = tickers[0]
        bid = float(t.bid) if (t.bid and t.bid > 0 and not math.isnan(t.bid)) else 0.0
        ask = float(t.ask) if (t.ask and t.ask > 0 and not math.isnan(t.ask)) else 0.0
        return bid, ask

    def _entry_limit_from(self, bid: float, ask: float) -> float:
        """Marketable entry: cross to the ask + a small, bounded buffer."""
        spread = max(0.0, ask - bid)
        cross = min(spread * self._cfg.entry_cross_frac, self._cfg.entry_cross_max)
        return _round_tick(ask + cross)

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
        # Pilot families trade minimum size until promoted (strategy audit
        # 2026-07-19): capped at 1 contract regardless of risk budget.
        if sig.signal_type.value in _PILOT_FAMILIES:
            qty = min(qty, 1)
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

        # Entry must be MARKETABLE, priced off a FRESH quote. Cross the spread
        # (ask + small bounded buffer) so a 1–2 tick uptick during routing still
        # fills. Critically, re-quote the LIVE bid/ask here rather than trusting
        # sig.bid/sig.ask (captured at signal-enrichment time, often minutes ago):
        # in the fast move the strategy targets, the live ask has already run
        # away from the snapshot, so a limit priced off the stale quote sits
        # dead. Fall back to the snapshot only if the live quote is unavailable.
        #   Chronic no-fills this fixes: 753C 2026-07-10; 749P/748P 2026-07-13
        #   (all placed marketably off a stale ask, none filled).
        q_bid, q_ask = sig.bid, sig.ask
        q_src = "snapshot"
        if self._cfg.entry_requote_at_placement:
            live_bid, live_ask = await self._live_quote(contract)
            if live_ask > 0:
                q_bid, q_ask, q_src = live_bid or q_bid, live_ask, "live"
        entry_limit = self._entry_limit_from(q_bid, q_ask)
        tp_price = _round_tick(entry_limit * (1 + tp_pct / 100.0))
        sl_stop = _round_tick(entry_limit * (1 - stop_pct / 100.0))
        if q_src == "live" and abs(q_ask - sig.ask) >= 0.02:
            logger.info(
                "EXEC re-quote: {}{} snapshot ask ${:.2f} → live ask ${:.2f} "
                "(entry {:.2f})", sig.strike, sig.right, sig.ask, q_ask, entry_limit,
            )
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
            orig_entry_limit=entry_limit,
        )
        self._positions[sig.dedup_key] = pos
        self._trades_today += 1

        # Fast, bounded entry chase — independent of the ~75s poll so a fast
        # directional move doesn't leave the limit dead. Self-terminates on
        # fill/close/max-reprices (≤ entry_max_reprices × interval lifetime).
        if self._cfg.entry_max_reprices > 0:
            asyncio.ensure_future(self._chase_loop(pos))

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

    # ── Entry chase ────────────────────────────────────────────────────────

    async def _chase_loop(self, pos: LivePosition) -> None:
        """Bounded background task: re-post the entry limit toward the live ask
        every `entry_reprice_interval_s` while it's unfilled, up to
        `entry_max_reprices`. Runs independently of on_poll so it can react in
        seconds, not the ~75s poll cadence. Self-terminates on fill / close /
        max-reprices, so its lifetime is bounded (≤ interval × max_reprices)."""
        try:
            while (not pos.closed and not pos.entry_filled
                   and pos.reprice_count < self._cfg.entry_max_reprices):
                await asyncio.sleep(self._cfg.entry_reprice_interval_s)
                if pos.closed or pos.entry_filled:
                    return
                if int(pos.parent.orderStatus.filled or 0) > 0:
                    return  # on_poll will record the fill
                await self._chase_entry(pos, datetime.utcnow())
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("EXEC chase loop error ({}): {}", pos.key, exc)

    async def _chase_entry(self, pos: LivePosition, now_utc: datetime) -> None:
        """One chase step: bump the resting entry limit toward the live ask,
        capped at `entry_chase_max_pct` above the ORIGINAL entry. Chases UP only.
        Modifies the parent limit in place (children keep their parentId link;
        their prices drift by at most the bounded chase % and only matter after
        fill)."""
        st = pos.parent.orderStatus.status
        if st not in ("Submitted", "PreSubmitted", "PendingSubmit", "ApiPending"):
            return
        live_bid, live_ask = await self._live_quote(pos.contract)
        if live_ask <= 0:
            return
        new_limit = self._entry_limit_from(live_bid, live_ask)
        cap = _round_tick(pos.orig_entry_limit * (1 + self._cfg.entry_chase_max_pct / 100.0))
        new_limit = min(new_limit, cap)
        cur = float(pos.parent.order.lmtPrice or pos.orig_entry_limit)
        if new_limit <= cur + 1e-9:
            return  # already at/above the live ask, or capped out — nothing to chase
        try:
            pos.parent.order.lmtPrice = new_limit
            pos.parent.order.transmit = True
            self._ib.placeOrder(pos.contract, pos.parent.order)
        except Exception as exc:
            logger.warning("EXEC entry chase failed ({}): {}",
                           pos.contract.localSymbol, exc)
            return
        pos.reprice_count += 1
        pos.last_reprice_at = now_utc
        pos.entry_mid = new_limit
        logger.info(
            "🐎 EXEC CHASE {}/{}: {} entry ${:.2f}→${:.2f} (live ask ${:.2f}, cap ${:.2f})",
            pos.reprice_count, self._cfg.entry_max_reprices,
            pos.contract.localSymbol, cur, new_limit, live_ask, cap,
        )

    # ── Poll-cycle maintenance ───────────────────────────────────────────────

    async def on_poll(self) -> None:
        """Entry timeouts, fill detection, time stops, 0DTE flatten, P&L."""
        if not (self._cfg.enabled and self._connected):
            return
        now_utc = datetime.utcnow()
        now_et = datetime.now(ET).time()

        # 0. Deferred exit-fill confirmation (JUL 17 2026). A close whose fill
        #    price hadn't populated in the synchronous retry window is confirmed
        #    here on later polls instead of losing its P&L forever (2-week audit:
        #    4 of 11 real fills had no recorded exit; 2 closes UNCONFIRMED on
        #    Jul 17 alone). Runs for CLOSED positions too — the main loop below
        #    deliberately skips them.
        for pos in list(self._positions.values()):
            for pe in list(pos.pending_exits):
                px = float(pe["trade"].orderStatus.avgFillPrice or 0.0)
                pe["attempts"] += 1
                if px > 0:
                    pnl = (px - pe["entry_px"]) * pe["qty"] * 100.0
                    if pe["kind"] == "partial":
                        self._realized_pnl_today += pnl
                        logger.info(
                            "💵 EXEC PARTIAL CONFIRMED (deferred): {}x {} @${:.2f} "
                            "P&L ${:+.0f} (day ${:+.0f})",
                            pe["qty"], pos.contract.localSymbol, px, pnl,
                            self._realized_pnl_today,
                        )
                    else:
                        self._register_close(pos, pe["label"], pnl, exit_premium=px)
                        logger.info(
                            "💵 EXEC EXIT CONFIRMED (deferred): {} {} @${:.2f} "
                            "P&L ${:+.0f} (day ${:+.0f})",
                            pos.contract.localSymbol, pe["label"], px, pnl,
                            self._realized_pnl_today,
                        )
                    pos.pending_exits.remove(pe)
                elif pe["attempts"] > 40:   # ~50 min of polls — give up loudly
                    logger.warning(
                        "EXEC: exit fill for {} ({}) STILL unconfirmed after {} "
                        "checks — abandoning; reconcile P&L from IB statement",
                        pos.contract.localSymbol, pe["label"], pe["attempts"],
                    )
                    pos.pending_exits.remove(pe)

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

            # 1.5 Entry died (rejected / externally cancelled) before filling —
            #     release the slot IMMEDIATELY rather than holding it until the
            #     timeout. Without this a rejected order pins a position slot and
            #     a trades-today count for up to entry_timeout_s (or forever, per
            #     the step-2 nesting bug fixed below). (audit 2026-07-13)
            if not pos.entry_filled and st in ("Cancelled", "ApiCancelled"):
                logger.warning(
                    "EXEC: entry {} reached terminal status {} without filling — "
                    "releasing slot", pos.contract.localSymbol, st,
                )
                for child in (pos.take_profit, pos.stop_loss):
                    try:
                        if child.orderStatus.status not in (
                            "Filled", "Cancelled", "ApiCancelled", "Inactive"
                        ):
                            self._ib.cancelOrder(child.order)
                    except Exception:
                        pass
                pos.closed = True
                pos.close_reason = f"entry {st.lower()} (never filled)"
                self._trades_today = max(0, self._trades_today - 1)
                await self._notify(
                    f"⚠️ <b>ENTRY {st.upper()}</b> {pos.contract.localSymbol} — "
                    "did not fill; slot released"
                )
                continue

            # 2. Unfilled entry timeout → give up and RELEASE THE SLOT.
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
                # Release the slot REGARDLESS of terminal/non-terminal status.
                # Previously this was nested under the non-terminal branch, so a
                # terminal-but-unfilled parent (rejected/Inactive) skipped the
                # release and leaked the slot forever → open_count stayed high and
                # the bot silently stopped entering. (audit 2026-07-13)
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
                    if exit_px <= 0:
                        # Child filled but price not yet propagated — defer, do
                        # NOT record a fabricated $0-exit (audit: rows 203/212).
                        pos.closed = True
                        pos.close_reason = f"{exit_label} (fill price pending)"
                        pos.pending_exits.append({
                            "trade": exit_trade, "label": exit_label,
                            "qty": pos.qty, "entry_px": entry_px,
                            "attempts": 0, "kind": "full",
                        })
                        logger.info(
                            "EXEC {}: {} filled at IB, price pending — deferred "
                            "confirmation armed", exit_label.upper(),
                            pos.contract.localSymbol,
                        )
                        continue
                    pnl = (exit_px - entry_px) * pos.qty * 100.0
                    self._register_close(pos, exit_label, pnl, exit_premium=exit_px)
                    logger.info(
                        "{} EXEC {}: {} ${:.2f} → ${:.2f}  P&L ${:+.0f} (day ${:+.0f})",
                        "🎯" if exit_label == "take_profit" else "🛑",
                        exit_label.upper(), pos.contract.localSymbol,
                        entry_px, exit_px, pnl, self._realized_pnl_today,
                    )
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

    # ── Exit-engine order primitives (JUL 14 2026) ───────────────────────────

    async def move_stop(self, key: str, new_stop: float) -> bool:
        """Tighten the resting stop-loss child to `new_stop` (premium $).

        Modify-in-place (same orderId re-place). NEVER widens: for our long
        positions the SL is a SELL stop, so tightening = raising. Returns True
        when a modify was sent.
        """
        pos = self._positions.get(key)
        if pos is None or pos.closed or not pos.entry_filled:
            return False
        sl = pos.stop_loss.order
        new_stop = _round_tick(new_stop)
        cur = float(sl.auxPrice or 0.0)
        if new_stop <= cur + 1e-9:
            return False                      # would widen or no-op
        try:
            sl.auxPrice = new_stop
            if self._cfg.stop_type != "stop":  # stop-limit: keep the buffer
                sl.lmtPrice = _round_tick(
                    new_stop * (1 - self._cfg.stop_limit_buffer_pct / 100.0)
                )
            self._ib.placeOrder(pos.contract, sl)
        except Exception as exc:
            logger.warning("EXEC move_stop failed ({}): {}", key, exc)
            return False
        logger.info("🔒 EXEC STOP MOVED: {} SL ${:.2f} → ${:.2f}",
                    pos.contract.localSymbol, cur, new_stop)
        return True

    async def partial_close(self, key: str, fraction: float, reason: str) -> bool:
        """Scale out `fraction` of the position at market; shrink the bracket
        children FIRST so the remaining TP/SL quantities always match the
        remaining position (no oversell window). Full-close when the fraction
        would leave nothing meaningful."""
        pos = self._positions.get(key)
        if pos is None or pos.closed or not pos.entry_filled:
            return False
        qty_out = max(1, int(pos.qty * fraction))
        if qty_out >= pos.qty:
            return await self.close_position(key, reason)
        remaining = pos.qty - qty_out
        try:
            # 1) Shrink both bracket children to the remaining quantity.
            for child in (pos.take_profit, pos.stop_loss):
                if child.orderStatus.status not in ("Filled", "Cancelled",
                                                    "ApiCancelled", "Inactive"):
                    child.order.totalQuantity = remaining
                    self._ib.placeOrder(pos.contract, child.order)
            # 2) Market-sell the freed quantity (standalone — not in the OCA).
            mkt = Order(action="SELL", orderType="MKT", totalQuantity=qty_out)
            if self._cfg.account:
                mkt.account = self._cfg.account
            sell_trade = self._ib.placeOrder(pos.contract, mkt)
        except Exception as exc:
            logger.opt(exception=True).error("EXEC partial_close failed ({}): {}", key, exc)
            return False

        pos.qty = remaining
        # Best-effort realized P&L on the scale-out fill.
        exit_px = 0.0
        for _ in range(4):
            await asyncio.sleep(1.5)
            exit_px = float(sell_trade.orderStatus.avgFillPrice or 0.0)
            if exit_px > 0:
                break
        entry_px = float(pos.parent.orderStatus.avgFillPrice or pos.entry_mid)
        if exit_px > 0:
            pnl = (exit_px - entry_px) * qty_out * 100.0
            self._realized_pnl_today += pnl
        else:
            # Scale-out fill price pending — defer so the partial's P&L is
            # confirmed on a later poll instead of silently booked as $0.
            pnl = 0.0
            pos.pending_exits.append({
                "trade": sell_trade, "label": "partial", "qty": qty_out,
                "entry_px": entry_px, "attempts": 0, "kind": "partial",
            })
        logger.info(
            "💰 EXEC PARTIAL: SOLD {}x {} @{} ({}) — {} remain, P&L ${:+.0f}",
            qty_out, pos.contract.localSymbol,
            f"${exit_px:.2f}" if exit_px > 0 else "pending-fill",
            reason, remaining, pnl,
        )
        await self._notify(
            f"💰 <b>PARTIAL EXIT</b> {qty_out}× {pos.contract.localSymbol} "
            f"@ {'$%.2f' % exit_px if exit_px > 0 else 'MKT'} — {reason}\n"
            f"{remaining} remain · bracket resized · P&L ${pnl:+.0f}"
        )
        return True

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
                        # Exit fill not confirmed inside the retry window — mark
                        # closed (never re-send), arm DEFERRED confirmation so the
                        # real P&L lands on a later poll instead of being lost.
                        pos.closed = True
                        pos.close_reason = f"{reason} (exit fill pending)"
                        pos.pending_exits.append({
                            "trade": exit_trade, "label": f"bot exit: {reason}",
                            "qty": remaining, "entry_px": entry_px,
                            "attempts": 0, "kind": "full",
                        })
                        logger.warning(
                            "EXEC CLOSE {} — {} — exit fill unconfirmed in window; "
                            "deferred confirmation armed",
                            pos.contract.localSymbol, reason,
                        )
                else:
                    # Already exited via a bracket child. Record the CHILD's real
                    # fill — previously this registered $0.00 with no exit premium
                    # and on_poll skips closed positions, so the bracket P&L was
                    # lost forever (audit: row 233, 750P Jul 15). Find which child
                    # filled and register (or defer) its actual price.
                    child, label = None, ""
                    if int(pos.take_profit.orderStatus.filled or 0) > 0:
                        child, label = pos.take_profit, "take_profit"
                    elif int(pos.stop_loss.orderStatus.filled or 0) > 0:
                        child, label = pos.stop_loss, "stop_loss"
                    entry_px = pos.parent.orderStatus.avgFillPrice or pos.entry_mid
                    child_px = float(child.orderStatus.avgFillPrice or 0.0) if child else 0.0
                    if child is not None and child_px > 0:
                        pnl = (child_px - entry_px) * remaining * 100.0
                        self._register_close(pos, label, pnl, exit_premium=child_px)
                        logger.info(
                            "EXEC CLOSE {} — {} (bracket {} @${:.2f}, P&L ${:+.0f})",
                            pos.contract.localSymbol, reason, label, child_px, pnl,
                        )
                    elif child is not None:
                        pos.closed = True
                        pos.close_reason = f"{label} (fill price pending)"
                        pos.pending_exits.append({
                            "trade": child, "label": label, "qty": remaining,
                            "entry_px": entry_px, "attempts": 0, "kind": "full",
                        })
                        logger.info(
                            "EXEC CLOSE {} — {} (bracket {} filled, price pending — "
                            "deferred)", pos.contract.localSymbol, reason, label,
                        )
                    else:
                        self._register_close(pos, reason, 0.0)
                        logger.warning(
                            "EXEC CLOSE {} — {} (bracket-exit inferred but no child "
                            "shows fills — P&L unknown, reconcile from IB)",
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

        # Per-family realized P&L history — feeds the pilot auto-kill gate.
        if not hasattr(self, "_family_pnls"):
            self._family_pnls: Dict[str, List[float]] = {}
        fam = pos.signal.signal_type.value
        self._family_pnls.setdefault(fam, []).append(pnl)
        self._family_pnls[fam] = self._family_pnls[fam][-20:]

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
