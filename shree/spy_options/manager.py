"""SPY Options signal bot — main async manager.

Orchestration loop (every 60 seconds during RTH):
  1. Fetch SPY price + VIX from IB Gateway
  2. Fetch SPY 5-minute bars → classify market regime
  3. Score sentiment from VIX trend + VWAP position + EMA slope
  4. Compute IV rank from VIX 52-week range (cached at startup)
  5. Build option chain snapshot (strikes near ATM) with Greeks
  6. Apply liquidity filters per contract
  7. Run SignalEngine (weighted confidence model)
  8. Deduplicate → send Telegram alerts → persist to SQLite

No orders are ever placed.
"""
from __future__ import annotations

import asyncio
import html as _html
from collections import deque
from datetime import datetime, time, timedelta
from typing import Dict, Deque, List, Optional, Set
from zoneinfo import ZoneInfo

from ..config.integrations import TelegramConfig
from ..config.spy_options import SpyOptionsConfig
from ..utils.logger import logger
from ..utils.telegram_notifier import TelegramNotifier
from .analytics_db import AnalyticsDB
from .chain_builder import ChainSnapshot, OptionQuote, VolumeTracker, passes_liquidity
from .ib_client import IBOptionsClient
from .regime_detector import RegimeContext, RegimeDetector
from .sentiment_engine import SentimentContext, SentimentEngine
from .signal_engine import SignalContext, SignalEngine, SignalType, SpySignal
from .sweep_tracker import SweepTracker

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

    Polls IB Gateway (ib_insync) on a configurable interval, generates
    Greek-scored signals, and sends Telegram alerts.
    """

    def __init__(
        self,
        cfg: SpyOptionsConfig,
        telegram_cfg: Optional[TelegramConfig] = None,
    ) -> None:
        self._cfg = cfg
        self._running = False

        self._ib = IBOptionsClient(cfg.ib)
        self._tracker = VolumeTracker()
        self._engine = SignalEngine(cfg.signals, self._tracker)

        self._regime_detector = RegimeDetector()
        self._sentiment_engine = SentimentEngine()
        self._sweep_tracker = SweepTracker(window_minutes=cfg.signals.sweep_window_minutes)

        if cfg.analytics.enabled:
            self._analytics: Optional[AnalyticsDB] = AnalyticsDB(cfg.analytics.db_path)
        else:
            self._analytics = None

        if telegram_cfg and telegram_cfg.enabled:
            self._telegram = TelegramNotifier(
                bot_token=telegram_cfg.bot_token,
                chat_id=telegram_cfg.chat_id,
                enabled=True,
            )
        else:
            self._telegram = TelegramNotifier("", "", enabled=False)

        # Deduplication: dedup_key → last sent datetime
        self._sent_times: Dict[str, datetime] = {}
        self._last_reset_date: Optional[str] = None

        # Option conid resolution cache: (expiry_month, strike, right) → conid
        self._conid_map: Dict[tuple, int] = {}
        self._conid_details: Dict[int, tuple] = {}

        # VIX history for sentiment engine (rolling, newest last)
        self._vix_history: Deque[float] = deque(maxlen=10)

        # VIX 52-week range for IV rank (fetched once at startup)
        self._vix_52w_low: Optional[float] = None
        self._vix_52w_high: Optional[float] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        logger.info("SpyOptionsManager stop requested")
        self._running = False

    async def start(self) -> None:
        """Run the async polling loop until stop() is called."""
        logger.info("=== SPY Options Bot starting ===")
        await self._ib.start()

        spy_conid = await self._ib.get_spy_conid()
        if not spy_conid:
            logger.error(
                "Failed to resolve SPY conid — "
                "is IB Gateway running and authenticated on port {}?",
                self._cfg.ib.ibkr_port,
            )
            await self._ib.close()
            return

        logger.info("SPY conid={}", spy_conid)

        # Fetch VIX 52-week range once at startup for IV rank computation
        result = await self._ib.get_vix_52w_range()
        if result:
            self._vix_52w_low, self._vix_52w_high = result
            logger.info(
                "VIX 52w range: low={:.1f}  high={:.1f}",
                self._vix_52w_low, self._vix_52w_high,
            )
        else:
            logger.warning("Could not fetch VIX 52w range — IV rank will default to 50")

        self._running = True
        try:
            while self._running:
                try:
                    await self._poll(spy_conid)
                except Exception as exc:
                    logger.opt(exception=True).error("Poll error: {}", exc)
                await asyncio.sleep(self._cfg.session.poll_interval_s)
        finally:
            await self._ib.close()
            await self._telegram.close()
            if self._analytics:
                self._analytics.close()
            logger.info("=== SPY Options Bot stopped ===")

    # ── Session gate ──────────────────────────────────────────────────────────

    def _market_open(self) -> bool:
        now = datetime.now(ET)
        if now.weekday() >= 5:
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
            self._sweep_tracker.reset()
            self._conid_map.clear()
            self._conid_details.clear()
            self._vix_history.clear()
            logger.info("New day {} — signal dedup + tracker reset", today)

    # ── IV rank ───────────────────────────────────────────────────────────────

    def _compute_iv_rank(self, vix: Optional[float]) -> float:
        """Compute IV rank (0-100) from cached 52-week VIX range."""
        if (
            vix is None
            or self._vix_52w_low is None
            or self._vix_52w_high is None
            or (self._vix_52w_high - self._vix_52w_low) <= 0
        ):
            return 50.0  # neutral default
        return max(0.0, min(100.0,
            (vix - self._vix_52w_low) / (self._vix_52w_high - self._vix_52w_low) * 100.0
        ))

    # ── Chain building ────────────────────────────────────────────────────────

    async def _build_chain(
        self,
        spy_conid: int,
        spy_price: float,
        expiry_month: str,
    ) -> Optional[ChainSnapshot]:
        """Fetch strikes, resolve conids, snapshot Greeks → ChainSnapshot."""
        cfg_c = self._cfg.chain

        strikes_data = await self._ib.get_strikes(spy_conid, expiry_month, cfg_c.exchange)
        if not strikes_data:
            logger.warning("No strikes returned for SPY {}", expiry_month)
            return None

        call_strikes = strikes_data.get("call", [])
        put_strikes  = strikes_data.get("put", [])

        half_cap = cfg_c.max_strikes_per_expiry // 2
        near_calls = _near_strikes(call_strikes, spy_price, cfg_c.strike_pct_range, half_cap)
        near_puts  = _near_strikes(put_strikes,  spy_price, cfg_c.strike_pct_range, half_cap)

        # Resolve missing option conids
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

        call_conids = [self._conid_map[(expiry_month, s, "C")] for s in near_calls if (expiry_month, s, "C") in self._conid_map]
        put_conids  = [self._conid_map[(expiry_month, s, "P")] for s in near_puts  if (expiry_month, s, "P") in self._conid_map]
        all_conids  = call_conids + put_conids

        if not all_conids:
            logger.warning("No option conids resolved for SPY {}", expiry_month)
            return None

        # Fetch price + Greeks (snapshot=False + explicit cancel)
        snaps = await self._ib.get_snapshot_with_greeks(all_conids)

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
                bid=_f("84"), ask=_f("86"), last=_f("31"),
                bid_size=_i("85"), ask_size=_i("88"),
                volume=_i("87"),
                delta=_f("delta"),
                gamma=_f("gamma"),
                theta=_f("theta"),
                vega=_f("vega"),
                impl_vol=_f("impl_vol"),
                open_interest=_i("open_interest"),
            )

            # Apply liquidity filter before adding to chain
            if not passes_liquidity(
                quote,
                min_oi=cfg_c.liquidity_min_oi,
                max_spread_pct=cfg_c.liquidity_max_spread_pct,
                min_volume=cfg_c.liquidity_min_volume,
            ):
                logger.debug(
                    "Liquidity filter drop: {} OI={} spread={:.1f}% vol={}",
                    quote.symbol, quote.open_interest, quote.spread_pct, quote.volume,
                )
                continue

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

        vix = await self._ib.get_vix()
        if vix is not None:
            self._vix_history.append(vix)

        iv_rank = self._compute_iv_rank(vix)

        # Fetch 5-min bars for regime + sentiment
        bars_5m = await self._ib.get_spy_bars_5m()
        regime_ctx = self._regime_detector.classify(
            bars_5m, spy_price, vix,
            vix_high=self._cfg.signals.vix_high,
            vix_low=16.0,
        )
        sentiment_ctx = self._sentiment_engine.score(
            vix=vix or 20.0,
            vix_history=list(self._vix_history),
            regime=regime_ctx,
        )

        logger.info(
            "Poll: SPY={:.2f}  VIX={}  IVRank={:.0f}  Regime={}  Sentiment={:+.0f}({})",
            spy_price,
            f"{vix:.1f}" if vix is not None else "n/a",
            iv_rank,
            regime_ctx.regime,
            sentiment_ctx.score,
            sentiment_ctx.label,
        )

        ctx = SignalContext(
            regime=regime_ctx,
            sentiment=sentiment_ctx,
            iv_rank=iv_rank,
            vix=vix,
            spy_price=spy_price,
        )

        now_et = datetime.now(ET)
        expiry_months = [
            _ib_month(now_et.replace(day=1) + timedelta(days=32 * i))
            for i in range(self._cfg.chain.num_expiries)
        ]

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
                all_signals.extend(self._engine.evaluate(chain, ctx, self._sweep_tracker))

        await self._dispatch_signals(all_signals)

    async def _dispatch_signals(self, signals: List[SpySignal]) -> None:
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
            if self._analytics:
                self._analytics.insert(sig)

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

    _TIER_LABEL = {
        "MEDIUM":  "MEDIUM",
        "HIGH":    "★ HIGH",
        "EXTREME": "★★ EXTREME ★★",
    }

    def _format(self, sig: SpySignal) -> str:
        emoji = self._TYPE_EMOJI.get(sig.signal_type, "📊")
        conf_pct = int(sig.confidence * 100)
        tier_label = self._TIER_LABEL.get(sig.confidence_tier, sig.confidence_tier)
        now_et = datetime.now(ET).strftime("%H:%M ET")

        lines = [
            f"{emoji} <b>SPY OPTIONS — {sig.signal_type.value}</b>",
            "",
            f"📌 <b>{sig.strike:.0f}{sig.right}</b>  exp <b>{sig.expiry}</b>",
            f"💰 SPY: <b>${sig.spy_price:.2f}</b>",
        ]

        if sig.vix is not None:
            iv_tag = (
                "LOW IV" if sig.vix < self._cfg.signals.vix_low else
                "HIGH IV" if sig.vix > self._cfg.signals.vix_high else
                "MID IV"
            )
            lines.append(f"📊 VIX: <b>{sig.vix:.1f}</b> ({iv_tag})")

        # IV rank and regime
        lines.append(f"🌡 IV Rank: <b>{sig.iv_rank:.0f}/100</b>")
        lines.append(f"🌍 Regime: <b>{sig.regime}</b>")

        # Greeks block (only if any are populated)
        if sig.delta != 0.0 or sig.gamma != 0.0:
            greek_parts = []
            if sig.delta != 0.0:
                greek_parts.append(f"Δ {sig.delta:+.3f}")
            if sig.gamma != 0.0:
                greek_parts.append(f"Γ {sig.gamma:.4f}")
            if sig.theta != 0.0:
                greek_parts.append(f"Θ {sig.theta:.3f}")
            if sig.vega != 0.0:
                greek_parts.append(f"V {sig.vega:.3f}")
            if sig.impl_vol != 0.0:
                greek_parts.append(f"IV {sig.impl_vol:.1%}")
            lines.append(f"📐 Greeks: {' | '.join(greek_parts)}")

        # Volume and flow
        if sig.volume > 0:
            lines.append(f"📦 Volume: <b>{sig.volume:,}</b>")
        if sig.volume_spike_mult > 0:
            lines.append(f"🔥 Spike: <b>{sig.volume_spike_mult:.1f}×</b> rolling avg")

        # Liquidity
        if sig.open_interest > 0:
            lines.append(f"🏦 Open Interest: {sig.open_interest:,}")
        if sig.bid_size or sig.ask_size:
            lines.append(f"↔ Bid/Ask size: {sig.bid_size:,} / {sig.ask_size:,}")
        if sig.bid > 0 or sig.ask > 0:
            lines.append(f"💵 Bid/Ask: ${sig.bid:.2f} / ${sig.ask:.2f}")
        if sig.spread_pct > 0:
            lines.append(f"↔ Spread: {sig.spread_pct:.1f}%")

        # Sentiment
        lines.append(
            f"🧭 Sentiment: <b>{sig.sentiment_score:+.0f}</b> ({sig.sentiment_label})"
        )

        # Confidence
        lines.append(f"🎯 Confidence: <b>{conf_pct}%</b> [{tier_label}]")
        lines.append(f"🕐 {now_et}")

        # Reasoning
        if sig.reasoning:
            lines += ["", "<b>🧠 Analysis:</b>"]
            for r in sig.reasoning[:5]:
                lines.append(f"  • {_html.escape(str(r))}")

        # Trade idea
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
            "Sending signal: {} {}{}  exp={}  conf={:.0f}%  tier={}  regime={}",
            sig.signal_type.value, sig.strike, sig.right,
            sig.expiry, sig.confidence * 100,
            sig.confidence_tier, sig.regime,
        )
        await self._telegram.send_message(msg)
