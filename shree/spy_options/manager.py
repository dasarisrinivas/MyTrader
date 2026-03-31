"""SPY Options signal bot — main async manager.

Orchestration loop:
  1. Determine near-term SPY option expiry months
  2. Fetch SPY price + VIX from IB Client Portal
  3. Build option chain snapshot (strikes near ATM)
  4. Run SignalEngine to produce SpySignal objects
  5. Deduplicate against recently sent signals
  6. Format and send Telegram alerts

No orders are ever placed.
"""
from __future__ import annotations

import asyncio
import html as _html
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

from ..config.integrations import TelegramConfig
from ..config.spy_options import SpyOptionsConfig
from ..utils.logger import logger
from ..utils.telegram_notifier import TelegramNotifier
from .chain_builder import ChainSnapshot, OptionQuote, VolumeTracker
from .ib_gateway_client import IBGatewayOptionsClient
from .signal_engine import SignalEngine, SignalType, SpySignal

ET = ZoneInfo("America/New_York")


def _ib_month(dt: Optional[datetime] = None) -> str:
    """Convert datetime → IB month string, e.g. "APR26"."""
    if dt is None:
        dt = datetime.now(ET)
    return dt.strftime("%b%y").upper()


def _near_strikes(
    all_strikes: List[float],
    spy_price: float,
    pct_range: float,
    max_n: int,
) -> List[float]:
    """Filter strikes within ±pct_range of spy_price, capped at max_n."""
    lo = spy_price * (1 - pct_range)
    hi = spy_price * (1 + pct_range)
    filtered = sorted(s for s in all_strikes if lo <= s <= hi)
    if len(filtered) > max_n:
        step = max(1, len(filtered) // max_n)
        filtered = filtered[::step][:max_n]
    return filtered


class SpyOptionsManager:
    """Signal-only SPY options bot.

    Polls IB Client Portal REST API on a configurable interval, generates
    rule-based signals, and sends Telegram alerts.
    """

    def __init__(
        self,
        cfg: SpyOptionsConfig,
        telegram_cfg: Optional[TelegramConfig] = None,
    ) -> None:
        self._cfg = cfg
        self._running = False


        self._ib = IBGatewayOptionsClient(cfg.ib)
        self._tracker = VolumeTracker()
        self._engine = SignalEngine(cfg.signals, self._tracker)

        if telegram_cfg and telegram_cfg.enabled:
            self._telegram: TelegramNotifier = TelegramNotifier(
                bot_token=telegram_cfg.bot_token,
                chat_id=telegram_cfg.chat_id,
                enabled=True,
            )
        else:
            self._telegram = TelegramNotifier("", "", enabled=False)

        # Deduplication state
        self._sent_times: Dict[str, datetime] = {}
        self._last_reset_date: Optional[str] = None

        # conid resolution cache: (expiry_month, strike, right) → conid
        self._conid_map: Dict[tuple, int] = {}
        # reverse: conid → (expiry_month, strike, right)
        self._conid_details: Dict[int, tuple] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        logger.info("SpyOptionsManager stop requested")
        self._running = False

    def start(self) -> None:
        """Run the polling loop until stop() is called."""
        logger.info("=== SPY Options Bot starting ===")
        connected = self._ib.connect()
        if not connected:
            logger.error(
                "Failed to connect to IB Gateway — is IB Gateway running and authenticated?"
            )
            return

        spy_conid = self._ib.resolve_spy()
        if not spy_conid:
            logger.error(
                "Failed to resolve SPY conid — is IB Gateway running and authenticated?"
            )
            self._ib.disconnect()
            return

        logger.info("SPY conid={}", spy_conid)
        self._running = True

        try:
            import time
            while self._running:
                try:
                    # Replace with synchronous polling logic as needed
                    pass
                except Exception as exc:
                    logger.opt(exception=True).error("Poll error: {}", exc)
                time.sleep(self._cfg.session.poll_interval_s)
        finally:
            self._ib.disconnect()
            self._telegram.close()
            logger.info("=== SPY Options Bot stopped ===")

    # ── Session gate ──────────────────────────────────────────────────────────

    def _market_open(self) -> bool:
        now = datetime.now(ET)
        if now.weekday() >= 5:          # Saturday / Sunday
            return False
        if self._cfg.session.rth_only:
            h0, m0 = map(int, self._cfg.session.rth_start_et.split(":"))
            h1, m1 = map(int, self._cfg.session.rth_stop_et.split(":"))
            t = now.time().replace(second=0, microsecond=0)
            if not (time(h0, m0) <= t <= time(h1, m1)):
                return False
        return True

    def _daily_reset_if_needed(self) -> None:
        today = datetime.now(ET).strftime("%Y-%m-%d")
        if self._last_reset_date != today:
            self._last_reset_date = today
            self._sent_times.clear()
            self._tracker.reset()
            self._conid_map.clear()
            self._conid_details.clear()
            logger.info("New day {} — signal dedup + tracker reset", today)

    # ── Chain building ────────────────────────────────────────────────────────

    async def _build_chain(
        self,
        spy_conid: int,
        spy_price: float,
        expiry_month: str,
    ) -> Optional[ChainSnapshot]:
        """Fetch strikes, resolve conids, snapshot market data → ChainSnapshot."""
        cfg_c = self._cfg.chain

        strikes_data = await self._ib.get_strikes(spy_conid, expiry_month, cfg_c.exchange)
        if not strikes_data:
            logger.warning("No strikes returned for SPY {} from IB", expiry_month)
            return None

        call_strikes = strikes_data.get("call", [])
        put_strikes  = strikes_data.get("put", [])

        half_cap = cfg_c.max_strikes_per_expiry // 2
        near_calls = _near_strikes(call_strikes, spy_price, cfg_c.strike_pct_range, half_cap)
        near_puts  = _near_strikes(put_strikes,  spy_price, cfg_c.strike_pct_range, half_cap)

        # Resolve any missing option conids
        async def _resolve(strike: float, right: str) -> None:
            key = (expiry_month, strike, right)
            if key in self._conid_map:
                return
            conid = await self._ib.get_option_conid(
                spy_conid, expiry_month, strike, right, cfg_c.exchange
            )
            if conid:
                self._conid_map[key] = conid
                self._conid_details[conid] = key

        for strike in near_calls:
            await _resolve(strike, "C")
            await asyncio.sleep(cfg_c.conid_resolve_delay_s)
        for strike in near_puts:
            await _resolve(strike, "P")
            await asyncio.sleep(cfg_c.conid_resolve_delay_s)

        call_conids = [
            self._conid_map[(expiry_month, s, "C")]
            for s in near_calls
            if (expiry_month, s, "C") in self._conid_map
        ]
        put_conids = [
            self._conid_map[(expiry_month, s, "P")]
            for s in near_puts
            if (expiry_month, s, "P") in self._conid_map
        ]
        all_conids = call_conids + put_conids

        if not all_conids:
            logger.warning("No option conids resolved for SPY {}", expiry_month)
            return None

        fields = "31,84,85,86,87,88"
        await self._ib.subscribe_snapshot(all_conids, fields)
        snaps = await self._ib.get_snapshot(all_conids, fields)

        chain = ChainSnapshot(expiry_month)

        for conid, snap in snaps.items():
            details = self._conid_details.get(conid)
            if not details:
                continue
            _exp, strike, right = details

            def _f(k: str) -> float:
                try:
                    return float(str(snap.get(k, "0")).replace(",", "").strip())
                except (ValueError, TypeError):
                    return 0.0

            def _i(k: str) -> int:
                try:
                    return int(float(str(snap.get(k, "0")).replace(",", "").strip()))
                except (ValueError, TypeError):
                    return 0

            quote = OptionQuote(
                conid=conid,
                symbol=f"SPY {expiry_month} {strike:.0f}{right}",
                strike=strike,
                right=right,
                expiry_month=expiry_month,
                bid=_f("84"),
                ask=_f("86"),
                last=_f("31"),
                bid_size=_i("85"),
                ask_size=_i("88"),
                volume=_i("87"),
            )
            if right == "C":
                chain.calls.append(quote)
            else:
                chain.puts.append(quote)

        return chain

    # ── Main poll cycle ───────────────────────────────────────────────────────

    async def _poll(self, spy_conid: int) -> None:
        self._daily_reset_if_needed()

        if not self._market_open():
            logger.debug("Market closed — skipping poll")
            return

        spy_price = await self._ib.get_spy_price(spy_conid)
        if not spy_price:
            logger.warning("Could not fetch SPY price — skipping poll cycle")
            return

        vix = await self._ib.get_vix(self._cfg.ib.vix_conid) if self._cfg.ib.vix_conid else None
        logger.info(
            "Poll: SPY={:.2f}  VIX={}",
            spy_price,
            f"{vix:.1f}" if vix is not None else "n/a",
        )

        # Determine expiry months (current + next)
        now_et = datetime.now(ET)
        expiry_months = []
        for i in range(self._cfg.chain.num_expiries):
            # Advance by ~32 days per step to roll over month boundary
            month_dt = (now_et.replace(day=1) + timedelta(days=32 * i))
            expiry_months.append(_ib_month(month_dt))

        all_signals: List[SpySignal] = []
        for expiry in expiry_months:
            chain = await self._build_chain(spy_conid, spy_price, expiry)
            if chain:
                logger.debug(
                    "Chain {}: {} calls vol={:,}  {} puts vol={:,}  P/C={}",
                    expiry,
                    len(chain.calls), chain.total_call_volume,
                    len(chain.puts), chain.total_put_volume,
                    f"{chain.put_call_ratio:.2f}" if chain.put_call_ratio else "n/a",
                )
                sigs = self._engine.evaluate(chain, spy_price, vix)
                all_signals.extend(sigs)

        await self._dispatch_signals(all_signals)

    async def _dispatch_signals(self, signals: List[SpySignal]) -> None:
        """Send new signals, skipping duplicates within the dedup window."""
        dedup_td = timedelta(minutes=self._cfg.signals.dedup_window_minutes)
        now = datetime.utcnow()

        for sig in signals:
            key = sig.dedup_key
            last_sent = self._sent_times.get(key)
            if last_sent and (now - last_sent) < dedup_td:
                logger.debug("Dedup suppress: {}", key)
                continue

            await self._send_signal(sig)
            self._sent_times[key] = now

    # ── Signal formatting ─────────────────────────────────────────────────────

    _TYPE_EMOJI = {
        SignalType.CALL_SWEEP:       "🔥",
        SignalType.PUT_SWEEP:        "🐻",
        SignalType.BULL_CALL_SPREAD: "🐂",
        SignalType.BEAR_PUT_SPREAD:  "🔻",
        SignalType.LONG_STRADDLE:    "⚡",
        SignalType.HIGH_IV_ALERT:    "📈",
        SignalType.PC_RATIO_EXTREME: "⚖️",
    }

    def _format(self, sig: SpySignal) -> str:
        emoji = self._TYPE_EMOJI.get(sig.signal_type, "📊")
        conf_pct = int(sig.confidence * 100)
        now_et = datetime.now(ET).strftime("%H:%M ET")

        lines = [
            f"{emoji} <b>SPY OPTIONS — {sig.signal_type.value}</b>",
            "",
            f"📌 <b>{sig.strike:.0f}{sig.right}</b>  exp <b>{sig.expiry}</b>",
            f"💰 SPY: <b>${sig.spy_price:.2f}</b>",
        ]

        if sig.vix is not None:
            iv_tag = "LOW IV" if sig.vix < self._cfg.signals.vix_low else (
                "HIGH IV" if sig.vix > self._cfg.signals.vix_high else "MID IV"
            )
            lines.append(f"📊 VIX: <b>{sig.vix:.1f}</b> ({iv_tag})")

        if sig.volume > 0:
            lines.append(f"📦 Session volume: <b>{sig.volume:,}</b>")
        if sig.volume_spike_mult > 0:
            lines.append(f"🔥 Spike: <b>{sig.volume_spike_mult:.1f}×</b> rolling avg")
        if sig.bid_size or sig.ask_size:
            lines.append(f"📐 Bid/Ask size: {sig.bid_size:,} / {sig.ask_size:,}")

        lines.append(f"🎯 Confidence: <b>{conf_pct}%</b>")
        lines.append(f"🕐 {now_et}")

        if sig.reasoning:
            lines += ["", "<b>🧠 Analysis:</b>"]
            for r in sig.reasoning[:4]:
                lines.append(f"  • {_html.escape(str(r))}")

        if sig.suggested_trade:
            lines += ["", "<b>💡 Signal Idea:</b>"]
            for part in sig.suggested_trade.split("\n"):
                lines.append(f"  {_html.escape(part)}")

        lines += [
            "",
            "<i>⚠️ For informational purposes only. Not financial advice. "
            "Options carry significant risk of loss.</i>",
            "#SPY #Options #ShreeBot",
        ]
        return "\n".join(lines)

    async def _send_signal(self, sig: SpySignal) -> None:
        msg = self._format(sig)
        logger.info(
            "Sending signal: {} {}{}  exp={}  conf={:.0f}%",
            sig.signal_type.value,
            sig.strike,
            sig.right,
            sig.expiry,
            sig.confidence * 100,
        )
        await self._telegram.send_message(msg)
