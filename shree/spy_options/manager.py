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
import math
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
from .edge_reality import (
    compute_edge_reality,
    default_target_pct,
    gamma_accel_mult,
    gamma_warning_active,
    hourly_theta_dollars,
    iv_adjusted_stop_pct,
    round_trip_cost_pct_from_quote,
    skew_warning,
)
from .external import ExternalDataManager
from .ib_client import IBOptionsClient
from .regime_detector import RegimeContext, RegimeDetector
from .rules_v2.engine import EngineInputs, RulesV2Engine
from .sentiment_engine import SentimentContext, SentimentEngine
from .signal_engine import SignalContext, SignalEngine, SignalType, SpySignal
from .sweep_tracker import SweepTracker
from .technical_levels import TechnicalLevelsTracker, compute_max_pain

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
        self._engine = SignalEngine(
            cfg.signals,
            self._tracker,
            composite_confidence_boost=cfg.external.composite_confidence_boost,
        )

        self._regime_detector = RegimeDetector()
        self._sentiment_engine = SentimentEngine()
        self._sweep_tracker = SweepTracker(window_minutes=cfg.signals.sweep_window_minutes)

        if cfg.analytics.enabled:
            self._analytics: Optional[AnalyticsDB] = AnalyticsDB(cfg.analytics.db_path)
        else:
            self._analytics = None

        # External signals (news, social, macro, flow, economic calendar).
        # Each sub-source can be individually disabled via its config flag so
        # ExternalDataManager never fetches data for disabled sources.
        ext_cfg = cfg.external
        if ext_cfg.enabled:
            self._external: Optional[ExternalDataManager] = ExternalDataManager(
                reddit_enabled=ext_cfg.reddit_enabled,
                reddit_client_id=ext_cfg.reddit_client_id,
                reddit_client_secret=ext_cfg.reddit_client_secret,
                news_ttl_minutes=ext_cfg.news_ttl_minutes,
                reddit_ttl_minutes=ext_cfg.reddit_ttl_minutes,
                stocktwits_ttl_minutes=ext_cfg.stocktwits_ttl_minutes,
                event_risk_window_minutes=ext_cfg.event_risk_window_minutes,
                flow_ttl_minutes=ext_cfg.flow_ttl_minutes,
                flow_barchart_enabled=ext_cfg.flow_barchart_enabled,
                flow_dark_pool_enabled=ext_cfg.flow_dark_pool_enabled,
                # Per-source enable flags — wired from SpyOptionsExternalConfig
                calendar_enabled=ext_cfg.calendar_enabled,
                news_enabled=ext_cfg.news_enabled,
                stocktwits_enabled=ext_cfg.stocktwits_enabled,
                macro_enabled=ext_cfg.macro_enabled,
                cboe_enabled=ext_cfg.cboe_enabled,
                flow_enabled=ext_cfg.flow_enabled,
                breadth_enabled=ext_cfg.breadth_enabled,
                sector_enabled=ext_cfg.sector_enabled,
                vol_structure_enabled=ext_cfg.vol_structure_enabled,
                opex_enabled=ext_cfg.opex_enabled,
            )
        else:
            self._external = None

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

        # Active signal tracking for exit alerts
        # dedup_key → {signal, entry_price, entry_regime, sent_at}
        self._active_signals: Dict[str, Dict] = {}
        self._exit_sent: Set[str] = set()  # dedup_keys for which exit was already sent

        # Option conid resolution cache: (expiry_month, strike, right) → conid
        self._conid_map: Dict[tuple, int] = {}
        self._conid_details: Dict[int, tuple] = {}

        # Technical levels tracker (ORB, VWAP bands, pivots, EDR, RSI)
        self._tech_tracker = TechnicalLevelsTracker()

        # Last-poll technical levels cache (read by Telegram formatter)
        self._last_orb_status: str = "BUILDING"
        self._last_orb_high: Optional[float] = None
        self._last_orb_low: Optional[float] = None
        self._last_vwap_band: str = "INSIDE_1SD"
        self._last_edr_pct: float = 0.0
        self._last_rsi_5m: float = 50.0
        self._last_rsi_div: str = "NONE"
        self._last_pivot_nearest: Optional[str] = None
        self._last_pivot_bias: Optional[str] = None
        self._last_max_pain: Optional[float] = None
        self._last_near_max_pain: bool = False

        # Direction-flip cooldown: track the last PC_RATIO direction sent
        # to suppress opposite-direction signals within the cooldown window.
        self._last_pc_ratio_direction: Optional[str] = None   # "C" or "P"
        self._last_pc_ratio_sent: Optional[datetime] = None

        # Same-direction throttle: after N directional signals in a window,
        # suppress further same-direction alerts to prevent signal flooding.
        # Root cause #3: 9 PUT signals in 3.5 hours — user overexposed.
        self._dir_signal_times: Dict[str, List[datetime]] = {"C": [], "P": []}
        self._max_same_direction_signals: int = 3   # max signals per direction per window
        self._same_direction_window = timedelta(minutes=90)  # sliding window

        # Daily signal cap: prevent signal flooding (e.g. 55 signals in 3 days)
        self._daily_signal_count: int = 0

        # VIX history for sentiment engine (rolling, newest last)
        self._vix_history: Deque[float] = deque(maxlen=10)

        # VIX 52-week range for IV rank (fetched once at startup)
        self._vix_52w_low: Optional[float] = None
        self._vix_52w_high: Optional[float] = None

        # Rules-v2 engine (regime-first, structure-based rules layer).
        # Only instantiated when the feature flag is on; otherwise the legacy
        # pipeline is entirely untouched. Added Apr 22 2026 after Apr 21
        # post-mortem.
        if cfg.rules_v2.enabled:
            self._rules_v2: Optional[RulesV2Engine] = RulesV2Engine(cfg.rules_v2)
            self._rules_v2.begin_session()
            logger.info(
                "rules_v2 ENABLED — regime classifier + continuation "
                "+ structure throttle active"
            )
        else:
            self._rules_v2 = None

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
            self._tech_tracker.reset()
            self._conid_map.clear()
            self._conid_details.clear()
            self._vix_history.clear()
            self._active_signals.clear()
            self._exit_sent.clear()
            self._last_pc_ratio_direction = None
            self._last_pc_ratio_sent = None
            self._dir_signal_times = {"C": [], "P": []}
            self._daily_signal_count = 0
            if self._rules_v2 is not None:
                self._rules_v2.begin_session()
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

        # Resolve actual expiry date (YYYYMMDD) for this month
        expiry_date = self._ib._best_expiry(expiry_month) or ""

        chain = ChainSnapshot(expiry_month, expiry_date=expiry_date)
        filtered_count = 0

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
                filtered_count += 1
                if filtered_count <= 3:  # Log first 3 drops at INFO
                    logger.info(
                        "Liquidity drop: {} bid={:.2f} ask={:.2f} OI={} vol={} delta={:.3f}",
                        quote.symbol, quote.bid, quote.ask, quote.open_interest, quote.volume, quote.delta,
                    )
                continue

            if right == "C":
                chain.calls.append(quote)
            else:
                chain.puts.append(quote)

        if filtered_count > 0:
            logger.info(
                "Chain {}: {}/{} options filtered by liquidity",
                expiry_month, filtered_count, len(snaps),
            )

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

        # Fallback: IB Gateway doesn't serve VIX index via reqMktData snapshot —
        # get_vix() returns nan (a float) rather than None in that case.
        # Pull from yfinance via the MacroSignals module if available.
        if (vix is None or (isinstance(vix, float) and math.isnan(vix))) and self._external is not None:
            try:
                macro_vix = self._external._macro.state.vix
                if macro_vix is not None and macro_vix > 0:
                    vix = macro_vix
                    logger.debug("VIX from MacroSignals fallback: {:.1f}", vix)
            except Exception:
                pass

        if vix is not None and not (isinstance(vix, float) and math.isnan(vix)):
            self._vix_history.append(vix)
        else:
            vix = None  # normalise nan → None for all downstream consumers

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

        # Compute intraday technical levels from bars (ORB, VWAP bands, pivots, EDR, RSI)
        tech_levels = self._tech_tracker.update(bars_5m, spy_price, vix)

        # Refresh external signals (TTL-gated; most sources won't re-fetch every 60s)
        ext_ctx = None
        if self._external is not None:
            try:
                await self._external.refresh_if_stale()
                ext_ctx = self._external.context
            except Exception as exc:
                logger.warning("External data refresh failed: {}", exc)

        # Inject technical levels into external context (create a stub if needed)
        if ext_ctx is None:
            from .external.composite import ExternalContext
            ext_ctx = ExternalContext()
        ext_ctx.orb_high              = tech_levels.orb_high
        ext_ctx.orb_low               = tech_levels.orb_low
        ext_ctx.orb_established       = tech_levels.orb_established
        ext_ctx.orb_width_pct         = tech_levels.orb_width_pct
        ext_ctx.orb_status            = tech_levels.orb_status
        ext_ctx.orb_breakout_confirmed = tech_levels.orb_breakout_confirmed
        ext_ctx.vwap_1sd_upper        = tech_levels.vwap_1sd_upper
        ext_ctx.vwap_1sd_lower        = tech_levels.vwap_1sd_lower
        ext_ctx.vwap_2sd_upper        = tech_levels.vwap_2sd_upper
        ext_ctx.vwap_2sd_lower        = tech_levels.vwap_2sd_lower
        ext_ctx.vwap_band_position    = tech_levels.vwap_band_position
        ext_ctx.pivot_pp              = tech_levels.pivot_pp
        ext_ctx.pivot_r1              = tech_levels.pivot_r1
        ext_ctx.pivot_r2              = tech_levels.pivot_r2
        ext_ctx.pivot_s1              = tech_levels.pivot_s1
        ext_ctx.pivot_s2              = tech_levels.pivot_s2
        ext_ctx.near_pivot            = tech_levels.near_pivot
        ext_ctx.pivot_nearest         = tech_levels.pivot_nearest
        ext_ctx.pivot_bias            = tech_levels.pivot_bias
        ext_ctx.edr_points            = tech_levels.edr_points
        ext_ctx.edr_used_pct          = tech_levels.edr_used_pct
        ext_ctx.edr_exhausted         = tech_levels.edr_exhausted
        ext_ctx.rsi_5m                = tech_levels.rsi_5m
        ext_ctx.rsi_overbought        = tech_levels.rsi_overbought
        ext_ctx.rsi_oversold          = tech_levels.rsi_oversold
        ext_ctx.rsi_divergence        = tech_levels.rsi_divergence

        # Cache for Telegram formatter (which only receives the signal, not ext_ctx)
        self._last_orb_status     = tech_levels.orb_status
        self._last_orb_high       = tech_levels.orb_high
        self._last_orb_low        = tech_levels.orb_low
        self._last_vwap_band      = tech_levels.vwap_band_position
        self._last_edr_pct        = tech_levels.edr_used_pct
        self._last_rsi_5m         = tech_levels.rsi_5m
        self._last_rsi_div        = tech_levels.rsi_divergence
        self._last_pivot_nearest  = tech_levels.pivot_nearest if tech_levels.near_pivot else None
        self._last_pivot_bias     = tech_levels.pivot_bias if tech_levels.near_pivot else None
        self._last_max_pain       = None        # will be set after chain build
        self._last_near_max_pain  = False

        logger.info(
            "Poll: SPY={:.2f}  VIX={}  IVRank={:.0f}  Regime={}  Sentiment={:+.0f}({})"
            "{}{}",
            spy_price,
            f"{vix:.1f}" if vix is not None else "n/a",
            iv_rank,
            regime_ctx.regime,
            sentiment_ctx.score,
            sentiment_ctx.label,
            f"  Ext={ext_ctx.composite_score:+.2f}" if ext_ctx else "",
            f"  Flow={ext_ctx.flow_score:+.0f}  DP={ext_ctx.flow_dark_pool}"
            if ext_ctx else "",
        )
        # Log technical levels summary
        _orb_tag = (
            f"ORB={tech_levels.orb_status}"
            + (f"({tech_levels.orb_high:.2f}/{tech_levels.orb_low:.2f})"
               if tech_levels.orb_high else "")
        )
        _vwap_tag = f"VWAP_band={tech_levels.vwap_band_position}"
        _edr_tag  = f"EDR={tech_levels.edr_used_pct:.0f}%{'[EXHAUSTED]' if tech_levels.edr_exhausted else ''}"
        _rsi_tag  = f"RSI5m={tech_levels.rsi_5m:.0f}{'' if tech_levels.rsi_divergence == 'NONE' else f'[{tech_levels.rsi_divergence}]'}"
        _pvt_tag  = f"Pivot={tech_levels.pivot_nearest or 'none'}" if tech_levels.near_pivot else ""
        logger.info(
            "TechLevels: {}  {}  {}  {}  {}",
            _orb_tag, _vwap_tag, _edr_tag, _rsi_tag, _pvt_tag,
        )

        ctx = SignalContext(
            regime=regime_ctx,
            sentiment=sentiment_ctx,
            iv_rank=iv_rank,
            vix=vix,
            spy_price=spy_price,
            external=ext_ctx,
        )

        now_et = datetime.now(ET)
        expiry_months = [
            _ib_month(now_et.replace(day=1) + timedelta(days=32 * i))
            for i in range(self._cfg.chain.num_expiries)
        ]

        all_signals: List[SpySignal] = []
        max_pain_computed = False   # compute once from the first available chain
        for expiry in expiry_months:
            chain = await self._build_chain(spy_conid, spy_price, expiry)
            if chain:
                logger.info(
                    "Chain {}: {} calls vol={:,}  {} puts vol={:,}  P/C={}",
                    expiry,
                    len(chain.calls), chain.total_call_volume,
                    len(chain.puts), chain.total_put_volume,
                    f"{chain.put_call_ratio:.2f}" if chain.put_call_ratio else "n/a",
                )

                # Compute max pain from the first available chain and inject into ctx
                if not max_pain_computed:
                    call_oi = {q.strike: q.open_interest for q in chain.calls if q.open_interest > 0}
                    put_oi  = {q.strike: q.open_interest for q in chain.puts  if q.open_interest > 0}
                    strikes = sorted(set(list(call_oi) + list(put_oi)))
                    mp = compute_max_pain(strikes, call_oi, put_oi)
                    if mp is not None:
                        ctx.external.max_pain_strike = mp
                        dist = abs(spy_price - mp)
                        ctx.external.max_pain_distance = round(dist, 2)
                        ctx.external.near_max_pain = dist <= 1.50
                        self._last_max_pain      = mp
                        self._last_near_max_pain = ctx.external.near_max_pain
                        logger.info(
                            "Max pain: ${:.2f}  SPY=${:.2f}  dist={:.2f}{}",
                            mp, spy_price, dist,
                            "  [NEAR MAX PAIN]" if ctx.external.near_max_pain else "",
                        )
                    max_pain_computed = True

                all_signals.extend(self._engine.evaluate(chain, ctx, self._sweep_tracker))
            else:
                logger.info("Chain {}: empty (all options filtered out or no conids)", expiry)

        if all_signals:
            logger.info("Signal engine produced {} signal(s)", len(all_signals))
        else:
            logger.info("Signal engine: no signals this cycle")

        # ── Cross-expiry directional conflict filter ──────────────────────────
        # After collecting signals from all expiries, drop the weaker direction
        # when both CALL and PUT signals are present simultaneously.
        if all_signals:
            ce_calls = [s for s in all_signals if s.right == "C"]
            ce_puts  = [s for s in all_signals if s.right == "P"]
            ce_both  = [s for s in all_signals if s.right == "BOTH"]
            if ce_calls and ce_puts:
                best_call = max(s.confidence for s in ce_calls)
                best_put  = max(s.confidence for s in ce_puts)
                if best_call >= best_put:
                    logger.warning(
                        "Cross-expiry conflict: {} CALL + {} PUT signals across expiries — "
                        "keeping CALL (best={:.0f}% vs PUT best={:.0f}%)",
                        len(ce_calls), len(ce_puts), best_call * 100, best_put * 100,
                    )
                    all_signals = ce_calls + ce_both
                else:
                    logger.warning(
                        "Cross-expiry conflict: {} CALL + {} PUT signals across expiries — "
                        "keeping PUT (best={:.0f}% vs CALL best={:.0f}%)",
                        len(ce_calls), len(ce_puts), best_put * 100, best_call * 100,
                    )
                    all_signals = ce_puts + ce_both

        # ── rules_v2 layer: regime-first filter + continuation generator ─────
        # Feature-flagged. When disabled, all_signals flows straight through to
        # dispatch unchanged.
        if self._rules_v2 is not None:
            all_signals = self._apply_rules_v2(
                signals=all_signals,
                bars_5m=bars_5m,
                spy_price=spy_price,
                vix=vix,
                iv_rank=iv_rank,
                orb_high=tech_levels.orb_high,
                orb_low=tech_levels.orb_low,
                rsi_5m=tech_levels.rsi_5m,
                expiry=expiry_months[0] if expiry_months else "",
            )

        await self._dispatch_signals(all_signals)

        # Check whether any previously-sent signals now warrant an EXIT alert
        await self._check_exit_conditions(spy_price, regime_ctx)

    # ── rules_v2 layer ─────────────────────────────────────────────────────────
    def _apply_rules_v2(
        self,
        signals: List[SpySignal],
        bars_5m: List[Dict],
        spy_price: float,
        vix: Optional[float],
        iv_rank: Optional[float],
        orb_high: Optional[float],
        orb_low: Optional[float],
        rsi_5m: Optional[float],
        expiry: str,
    ) -> List[SpySignal]:
        """Apply the rules_v2 filter + continuation generator.

        Responsibilities:
          1. Classify the v2 regime from bars seen so far.
          2. Route every legacy signal through ``RulesV2Engine.filter`` —
             drop any that get blocked by a v2 gate.
          3. Generate any NEW TREND_CONTINUATION candidates that the legacy
             engine does not produce, convert them to ``SpySignal`` and
             append.
          4. Commit each allowed entry to the v2 structure throttle so
             subsequent signals see the updated leg state.

        Called only when ``self._rules_v2`` is not None (feature flag on).
        """
        engine = self._rules_v2
        assert engine is not None
        now = datetime.now(ET)

        regime = engine.classify_regime(bars_5m, spy_price)

        inputs = EngineInputs(
            bars=bars_5m,
            spy_price=spy_price,
            vix=vix,
            iv_rank=iv_rank,
            orb_high=orb_high,
            orb_low=orb_low,
            rsi_5m=rsi_5m,
            vwap=regime.vwap,
            option_quotes=None,
            now=now,
        )

        # ── Filter legacy signals ──────────────────────────────────────────
        kept: List[SpySignal] = []
        for sig in signals:
            # Skip BOTH-direction signals (straddles) — v2 is directional
            if sig.right not in ("C", "P"):
                kept.append(sig)
                continue
            # Per-leg pricing for the expected-move gate.
            # bid/ask are populated by signal_engine when greeks resolve;
            # impl_vol comes from IB greeks. Either may be 0/None when the
            # quote was missing — pass None and let the EM gate skip cleanly.
            _leg_mid = (
                ((sig.bid + sig.ask) / 2.0)
                if (sig.bid and sig.ask and sig.bid > 0 and sig.ask > 0)
                else None
            )
            _leg_iv = sig.impl_vol if (sig.impl_vol and sig.impl_vol > 0) else None
            decision = engine.filter(
                signal_type=sig.signal_type.value,
                direction=sig.right,
                price=spy_price,
                confidence=sig.confidence,
                regime=regime,
                inputs=inputs,
                leg_mid=_leg_mid,
                leg_iv=_leg_iv,
                dte=sig.dte,
            )
            if decision.allowed:
                kept.append(sig)
                engine.commit(sig.right, spy_price, now=now)
            else:
                logger.info(
                    "rules_v2 BLOCK {} {} @{:.2f}  rule={}  reason={}",
                    sig.signal_type.value, sig.right, spy_price,
                    decision.rule, decision.reason,
                )

        # ── Generate TREND_CONTINUATION candidates (not emitted by legacy) ─
        for cand in engine.generate_additional(bars_5m, regime, spy_price, now=now):
            decision = engine.filter(
                signal_type="TREND_CONTINUATION",
                direction=cand.direction,
                price=spy_price,
                confidence=cand.confidence,
                regime=regime,
                inputs=inputs,
            )
            if not decision.allowed:
                logger.info(
                    "rules_v2 CONTINUATION BLOCK {} @{:.2f}  rule={}  reason={}",
                    cand.direction, spy_price, decision.rule, decision.reason,
                )
                continue
            engine.commit(cand.direction, spy_price, now=now)
            kept.append(
                SpySignal(
                    signal_type=SignalType.TREND_CONTINUATION,
                    strike=round(cand.trigger_price),   # nearest whole $ — manager will delta-pick later
                    expiry=expiry,
                    right=cand.direction,
                    confidence=cand.confidence,
                    spy_price=spy_price,
                    vix=vix,
                    volume=0,
                    volume_spike_mult=0.0,
                    bid_size=0,
                    ask_size=0,
                    reasoning=list(cand.reasons) + [
                        f"trigger={cand.trigger_price:.2f} stop={cand.stop_price:.2f}",
                        f"regime={regime.regime} vwap={regime.vwap:.2f}",
                    ],
                    suggested_trade=(
                        f"BUY {'CALL' if cand.direction == 'C' else 'PUT'} on break of "
                        f"{cand.trigger_price:.2f} — stop {cand.stop_price:.2f}"
                    ),
                    # TREND_CONTINUATION is a naked long call/put entry with a
                    # structural stop; size with the long-leg heuristic.
                    structure="LONG",
                )
            )
            logger.info(
                "rules_v2 CONTINUATION ALLOW {} @{:.2f}  trigger={:.2f} stop={:.2f}",
                cand.direction, spy_price, cand.trigger_price, cand.stop_price,
            )

        return kept

    async def _dispatch_signals(self, signals: List[SpySignal]) -> None:
        dedup_td  = timedelta(minutes=self._cfg.signals.dedup_window_minutes)
        flip_td   = timedelta(minutes=self._cfg.signals.pc_ratio_flip_cooldown_minutes)
        daily_cap = self._cfg.signals.max_signals_per_day
        now = datetime.utcnow()
        seen_keys: set[str] = set()          # batch-level dedup (cross-expiry)
        for sig in signals:
            key = sig.dedup_key
            # ── Daily signal cap ──────────────────────────────────────────────
            if self._daily_signal_count >= daily_cap:
                logger.info(
                    "Daily signal cap reached ({}/{}) — suppressing {}",
                    self._daily_signal_count, daily_cap, key,
                )
                break   # no more signals today
            # ── Batch dedup: same key already dispatched this cycle ───────────
            if key in seen_keys:
                logger.debug("Batch dedup suppress (cross-expiry): {}", key)
                continue
            last_sent = self._sent_times.get(key)
            if last_sent and (now - last_sent) < dedup_td:
                logger.debug("Dedup suppress: {}", key)
                continue

            # ── PC_RATIO direction-flip cooldown ──────────────────────────────
            if sig.signal_type == SignalType.PC_RATIO_EXTREME:
                opposite = "P" if sig.right == "C" else "C"
                if (
                    self._last_pc_ratio_direction == opposite
                    and self._last_pc_ratio_sent is not None
                    and (now - self._last_pc_ratio_sent) < flip_td
                ):
                    remaining = (flip_td - (now - self._last_pc_ratio_sent)).seconds // 60
                    logger.info(
                        "PC_RATIO flip suppressed: {} after {} — cooldown {}min remaining",
                        sig.right, opposite, remaining,
                    )
                    continue

            # ── Same-direction throttle ───────────────────────────────────────
            # After N signals in the same direction within a sliding window,
            # suppress further same-direction alerts to prevent signal flooding.
            # Root cause #3: 9 PUT signals in 3.5h → user overexposed.
            #
            # When rules_v2 is enabled, the structure-based throttle
            # (zone-lock + leg cap + regime flip reset) has already ruled
            # and this cap is redundant. Skip it to avoid double-gating.
            if self._rules_v2 is None and sig.right in ("C", "P"):
                dir_times = self._dir_signal_times[sig.right]
                # Prune stale entries outside the window
                dir_times[:] = [t for t in dir_times if (now - t) < self._same_direction_window]
                if len(dir_times) >= self._max_same_direction_signals:
                    logger.info(
                        "Directional throttle: {} {} signals already sent in last {}min — suppressing {}",
                        len(dir_times), sig.right,
                        int(self._same_direction_window.total_seconds() // 60),
                        key,
                    )
                    continue

            # Edge Reality: populate regime-adjusted WR / breakeven / margin
            # / hourly theta / gamma multiplier / put-skew warning before
            # both Telegram dispatch and analytics persist. Strict no-op
            # when the feature flag is off — sig.* fields remain at defaults.
            self._apply_edge_reality(sig)

            # Red-edge suppression (May 2026 review fix). When net edge after
            # costs is negative the signal is informational at best —
            # emitting it at any confidence tier creates pressure to trade
            # an edge-negative setup. The doc's Grade D rule says "do not
            # trade" and we honor that. Toggle via cfg.signals.suppress_red_edge.
            if (
                sig.edge_color == "red"
                and getattr(self._cfg.signals, "suppress_red_edge", True)
            ):
                logger.info(
                    "Red-edge suppression: skipping {} {}{} "
                    "(regime_wr={}, breakeven_wr={:.1f}, margin={:+.1f}%)",
                    sig.signal_type.value, sig.strike, sig.right,
                    sig.regime_wr, sig.breakeven_wr, sig.edge_margin,
                )
                continue

            # MAY 4 2026: Trading Manager veto for SPY options.
            # We write the candidate signal to logs/spy_signals.jsonl, wait up
            # to ~2 seconds for the TM daemon to evaluate and write a verdict
            # to logs/manager_decisions.jsonl, then proceed unless the verdict
            # is REJECT. Fail-open: if the TM is not running, we log a
            # warning and proceed.
            try:
                if not await self._tm_check_and_publish(sig):
                    logger.warning(
                        "🛡️  TRADING MANAGER VETO (SPY): {} {}{} {} — skipping dispatch",
                        sig.signal_type.value, sig.right, sig.strike, sig.expiry,
                    )
                    continue
            except Exception as _tm_err:
                # Never let the veto hook block trading on its own bug.
                logger.error("Trading Manager SPY veto hook error (fail-open): {}", _tm_err)

            await self._send_signal(sig)
            self._sent_times[key] = now
            seen_keys.add(key)
            self._daily_signal_count += 1

            # Record directional send time for throttle
            if sig.right in ("C", "P"):
                self._dir_signal_times[sig.right].append(now)

            if sig.signal_type == SignalType.PC_RATIO_EXTREME:
                self._last_pc_ratio_direction = sig.right
                self._last_pc_ratio_sent = now
            if self._analytics:
                self._analytics.insert(sig)

            # Track directional signals for exit monitoring.
            # Store entry-time context so exit triggers can compare against
            # current conditions (VWAP reversion, time elapsed, etc).
            if sig.signal_type in {
                SignalType.CALL_SWEEP, SignalType.PUT_SWEEP,
                SignalType.BULL_CALL_SPREAD, SignalType.BEAR_PUT_SPREAD,
                SignalType.PC_RATIO_EXTREME, SignalType.ORB_BREAKOUT,
                SignalType.TREND_CONTINUATION,
            }:
                ext_ctx_now = self._external.context if self._external else None
                self._active_signals[key] = {
                    "signal": sig,
                    "entry_price": sig.spy_price,
                    "entry_regime": sig.regime,
                    "entry_vwap_band": (
                        getattr(ext_ctx_now, "vwap_band_position", "INSIDE_1SD")
                        if ext_ctx_now else "INSIDE_1SD"
                    ),
                    "entry_dte": getattr(sig, "dte", 1),
                    "sent_at": now,
                }

    # ── Edge Reality population ───────────────────────────────────────────────

    def _apply_edge_reality(self, sig: SpySignal) -> None:
        """Populate the Edge Reality fields on the signal in place.

        Strict no-op when ``cfg.signals.edge_reality_enabled`` is False — all
        SpySignal Edge Reality fields stay at their dataclass defaults so
        downstream Telegram + analytics code paths render nothing.

        Computes:
          - historical_wr / regime_wr (per-pattern × regime)
          - round_trip_cost_pct (from live bid/ask, falling back to SPY default)
          - target_pct_assumed + breakeven_wr + edge_margin + edge_color
          - iv_adjusted_stop_pct (15/20/25 by IVR band)
          - hourly_theta_dollars (scales daily theta by current session phase)
          - gamma_accel_mult + effective_gamma + gamma_warning_active (0DTE only)
          - skew_warning (SPY put-skew advisory on bearish signals)
        """
        if not getattr(self._cfg.signals, "edge_reality_enabled", True):
            return

        now_et = datetime.now(ET)
        signal_type_str = sig.signal_type.value
        regime = sig.regime or "RANGE_BOUND"
        right = sig.right or ""

        # Pull config-conditional bits up-front
        ivr = float(sig.iv_rank) if sig.iv_rank is not None else 50.0
        sig.iv_adjusted_stop_pct = iv_adjusted_stop_pct(ivr)

        # Round-trip cost: prefer live bid/ask, log+tag when we fall back to
        # the SPY default constant (4.8%). Without this debug line a wide-
        # spread strike with no live quote would silently look like a tight
        # 4.8% liquidity profile.
        sig.round_trip_cost_pct, sig.rt_source = round_trip_cost_pct_from_quote(
            sig.bid, sig.ask
        )
        if sig.rt_source != "live_quote":
            logger.debug(
                "Edge Reality: round-trip cost fell back to SPY default "
                "{:.1f}% for {} {}{} (bid={} ask={}) — figure shown to "
                "trader is an estimate",
                sig.round_trip_cost_pct,
                sig.signal_type.value, sig.strike, sig.right,
                sig.bid, sig.ask,
            )

        # Edge Reality bundle: regime-adjusted WR + breakeven + margin.
        # WR source stays at the doc-prior default until analytics_db has
        # ≥ 30 closed trades per signal type — at which point a future
        # _wr_source_for(signal_type) helper can override it to "empirical".
        target = default_target_pct(regime)
        sig.target_pct_assumed = target
        edge = compute_edge_reality(
            signal_type=signal_type_str,
            regime=regime,
            direction=right,
            target_pct=target,
            stop_pct=sig.iv_adjusted_stop_pct,
            round_trip_pct=sig.round_trip_cost_pct,
            rt_source=sig.rt_source,
        )
        sig.historical_wr = edge.historical_wr
        sig.regime_wr = edge.regime_wr
        sig.breakeven_wr = edge.breakeven_wr
        sig.edge_margin = edge.edge_margin
        sig.edge_color = edge.edge_color
        sig.wr_source = edge.wr_source

        # Hourly theta + gamma convexity acceleration (Power-Hour-relevant)
        if sig.theta:
            sig.hourly_theta_dollars = round(
                hourly_theta_dollars(sig.theta, now_et=now_et), 2
            )
        gam_mult = gamma_accel_mult(int(sig.dte or 0), now_et=now_et)
        sig.gamma_accel_mult = gam_mult
        sig.effective_gamma = round((sig.gamma or 0.0) * gam_mult, 5)
        sig.gamma_warning_active = gamma_warning_active(int(sig.dte or 0), now_et=now_et)

        # SPY put-skew warning on bearish-direction signals
        sig.skew_warning = skew_warning(signal_type_str, right)

        # ── Edge-aware tier cap ───────────────────────────────────────────
        # The reviewer caught a real flaw: 83% EXTREME on a -3.2% red-edge
        # signal. Confidence-tier label and edge-margin must agree. We never
        # change ``sig.confidence`` — the numeric value preserves the model
        # audit trail — only the displayed tier label is downgraded.
        if getattr(self._cfg.signals, "cap_tier_on_amber_edge", True):
            prior_tier = sig.confidence_tier
            if sig.edge_color == "red" and sig.confidence_tier in ("HIGH", "EXTREME"):
                sig.confidence_tier = "MEDIUM"
            elif sig.edge_color == "amber" and sig.confidence_tier == "EXTREME":
                sig.confidence_tier = "HIGH"
            if sig.confidence_tier != prior_tier:
                logger.info(
                    "Edge-aware tier cap: {} {}{} downgraded {} → {} "
                    "(edge_color={}, margin={:+.1f}%)",
                    sig.signal_type.value, sig.strike, sig.right,
                    prior_tier, sig.confidence_tier,
                    sig.edge_color, sig.edge_margin,
                )

    # ── Exit monitoring ──────────────────────────────────────────────────────

    # Bullish signals expect SPY to go up; bearish signals expect SPY to go down
    _BULLISH_SIGNALS = {
        SignalType.CALL_SWEEP, SignalType.BULL_CALL_SPREAD,
    }
    _BEARISH_SIGNALS = {
        SignalType.PUT_SWEEP, SignalType.BEAR_PUT_SPREAD,
    }
    # ORB_BREAKOUT direction is determined by the .right field (C=bullish, P=bearish)

    @staticmethod
    def _signal_direction(sig: SpySignal) -> str:
        """Return 'BULLISH', 'BEARISH', or 'NEUTRAL' for a signal."""
        if sig.signal_type in SpyOptionsManager._BULLISH_SIGNALS:
            return "BULLISH"
        if sig.signal_type in SpyOptionsManager._BEARISH_SIGNALS:
            return "BEARISH"
        # PC_RATIO_EXTREME, ORB_BREAKOUT, and TREND_CONTINUATION: direction from .right field
        if sig.signal_type in (
            SignalType.PC_RATIO_EXTREME,
            SignalType.ORB_BREAKOUT,
            SignalType.TREND_CONTINUATION,
        ):
            return "BEARISH" if sig.right == "P" else "BULLISH"
        return "NEUTRAL"

    async def _check_exit_conditions(
        self,
        spy_price: float,
        regime_ctx: RegimeContext,
    ) -> None:
        """Check active signals and send EXIT alerts when conditions reverse.

        Seven independent triggers (any one fires the exit alert):
          1. Price adverse ≥ 0.5%       — SPY moved against signal direction
          2. Regime flip                 — TREND_UP ↔ TREND_DOWN
          3. Large adverse move ≥ 1.0%  — urgent stop regardless of regime
          4. Time stop                   — 0DTE: 30 min | swing: 60 min
          5. Profit target hit           — +0.5% favorable SPY move (take profits)
          6. VWAP reversion              — SPY crossed back through VWAP vs entry side
          7. IV-adjusted premium stop    — estimated option drawdown ≥ IVR-cap
                                           (15/20/25% by IVR band, doc § 8.4).
                                           Uses delta × SPY_move / entry_premium
                                           as proxy because we don't re-snap
                                           Greeks per poll. Layered ON TOP of
                                           structural triggers — never replaces.
        """
        if not self._active_signals:
            return

        ext_ctx = self._external.context if self._external else None
        expired_keys: List[str] = []
        now = datetime.utcnow()

        for key, entry in list(self._active_signals.items()):
            # Auto-expire: 0DTE signals expire after 2h; swing after 6h.
            entry_dte: int = entry.get("entry_dte", 1)
            max_age = timedelta(hours=2 if entry_dte == 0 else 6)
            if (now - entry["sent_at"]) > max_age:
                expired_keys.append(key)
                continue

            # Already sent exit for this signal
            if key in self._exit_sent:
                continue

            sig: SpySignal = entry["signal"]
            entry_price: float = entry["entry_price"]
            entry_regime: str = entry["entry_regime"]
            entry_vwap_band: str = entry.get("entry_vwap_band", "INSIDE_1SD")
            direction = self._signal_direction(sig)

            if direction == "NEUTRAL":
                continue

            price_chg_pct = (spy_price - entry_price) / entry_price * 100.0
            reasons: List[str] = []

            # ── Trigger 1: price adverse ≥ 0.5% ──────────────────────────
            if direction == "BULLISH" and price_chg_pct <= -0.5:
                reasons.append(
                    f"SPY dropped {abs(price_chg_pct):.2f}% since entry "
                    f"(${entry_price:.2f} → ${spy_price:.2f})"
                )
            elif direction == "BEARISH" and price_chg_pct >= 0.5:
                reasons.append(
                    f"SPY rallied {price_chg_pct:.2f}% since entry "
                    f"(${entry_price:.2f} → ${spy_price:.2f})"
                )

            # ── Trigger 2: regime flip ────────────────────────────────────
            cur_regime = regime_ctx.regime
            if direction == "BULLISH" and entry_regime == "TREND_UP" and cur_regime == "TREND_DOWN":
                reasons.append(f"Regime flipped: {entry_regime} → {cur_regime}")
            elif direction == "BEARISH" and entry_regime == "TREND_DOWN" and cur_regime == "TREND_UP":
                reasons.append(f"Regime flipped: {entry_regime} → {cur_regime}")

            # ── Trigger 3: large adverse move ≥ 1.0% (urgent) ────────────
            if direction == "BULLISH" and price_chg_pct <= -1.0:
                if not any("dropped" in r for r in reasons):
                    reasons.append(
                        f"⚠️ URGENT — SPY dropped {abs(price_chg_pct):.2f}% "
                        f"(${entry_price:.2f} → ${spy_price:.2f})"
                    )
            elif direction == "BEARISH" and price_chg_pct >= 1.0:
                if not any("rallied" in r for r in reasons):
                    reasons.append(
                        f"⚠️ URGENT — SPY rallied {price_chg_pct:.2f}% "
                        f"(${entry_price:.2f} → ${spy_price:.2f})"
                    )

            # ── Trigger 4: time stop ──────────────────────────────────────
            # 0DTE options lose value exponentially — hard cap at 30 min.
            # Swing setups (1+ DTE) allow 60 min before staleness forces exit.
            minutes_held = (now - entry["sent_at"]).total_seconds() / 60.0
            time_stop_min = 30 if entry_dte == 0 else 60
            if minutes_held >= time_stop_min:
                reasons.append(
                    f"⏱ Time stop: held {minutes_held:.0f} min "
                    f"(limit {time_stop_min} min for {'0DTE' if entry_dte == 0 else 'swing'})"
                )

            # ── Trigger 5: profit target hit (+0.5% favorable move) ───────
            # Advisory exit — "take profits here" rather than "stop loss".
            _PROFIT_TARGET_PCT = 0.5
            if direction == "BULLISH" and price_chg_pct >= _PROFIT_TARGET_PCT:
                reasons.append(
                    f"✅ Profit target: SPY +{price_chg_pct:.2f}% since entry "
                    f"(${entry_price:.2f} → ${spy_price:.2f}) — consider taking profits"
                )
            elif direction == "BEARISH" and price_chg_pct <= -_PROFIT_TARGET_PCT:
                reasons.append(
                    f"✅ Profit target: SPY {price_chg_pct:.2f}% since entry "
                    f"(${entry_price:.2f} → ${spy_price:.2f}) — consider taking profits"
                )

            # ── Trigger 6: VWAP reversion ─────────────────────────────────
            # If the signal was entered with SPY on one side of VWAP and SPY
            # has since crossed back through, the directional thesis is
            # weakened — the mean-reversion has already begun.
            if ext_ctx is not None:
                cur_vwap_band = getattr(ext_ctx, "vwap_band_position", "INSIDE_1SD")
                _bullish_side = {"ABOVE_1SD", "ABOVE_2SD"}
                _bearish_side = {"BELOW_1SD", "BELOW_2SD"}
                if direction == "BULLISH" and entry_vwap_band in _bullish_side:
                    if cur_vwap_band not in _bullish_side:
                        reasons.append(
                            f"🔄 VWAP reversion: SPY was {entry_vwap_band} at entry, "
                            f"now {cur_vwap_band} — bullish thesis weakened"
                        )
                elif direction == "BEARISH" and entry_vwap_band in _bearish_side:
                    if cur_vwap_band not in _bearish_side:
                        reasons.append(
                            f"🔄 VWAP reversion: SPY was {entry_vwap_band} at entry, "
                            f"now {cur_vwap_band} — bearish thesis weakened"
                        )

            # ── Trigger 7: IV-adjusted premium stop ───────────────────────
            # Doc § 8.4 — high-IV options move violently per point of
            # underlying. A fixed -0.5% SPY adverse trigger is too loose at
            # high IVR (option may already be down 30%+) and too tight at
            # low IVR (cheap premium can absorb the move). Cap *estimated*
            # option drawdown using the IVR-conditional table:
            #     IVR > 65 → 15% | IVR 41–65 → 20% | IVR ≤ 40 → 25%.
            # Estimate uses delta × adverse_SPY / entry_premium. This is an
            # approximation (gamma + IV crush ignored) — for high-precision
            # stops, re-snapshot the option premium.
            if (
                getattr(self._cfg.signals, "iv_adjusted_premium_stop_enabled", True)
                and sig.iv_adjusted_stop_pct > 0
                and sig.delta
                and entry_price > 0
            ):
                # Adverse SPY $-move (always positive when against the signal)
                if direction == "BULLISH":
                    adverse_dollar = max(0.0, entry_price - spy_price)
                elif direction == "BEARISH":
                    adverse_dollar = max(0.0, spy_price - entry_price)
                else:
                    adverse_dollar = 0.0

                # Estimated premium drawdown %  (delta is signed; use abs)
                entry_mid = (
                    (sig.bid + sig.ask) / 2.0
                    if sig.bid and sig.ask else (sig.ask or sig.bid or 0.0)
                )
                if entry_mid > 0 and adverse_dollar > 0:
                    # Linear delta approximation breaks down near 0DTE expiry
                    # because gamma dominates — a delta-only estimate can be
                    # 2–3× too low in the last 30 minutes. Multiply by the
                    # gamma_accel_mult that _apply_edge_reality already
                    # computed (×1.0 / 1.5 / 2.5 / 4.0 by minutes-to-close)
                    # so the IV-stop tightens automatically when it matters
                    # most.
                    gam_mult = max(1.0, float(sig.gamma_accel_mult or 1.0))
                    est_premium_loss_pct = (
                        abs(sig.delta) * adverse_dollar / entry_mid * 100.0
                    ) * gam_mult
                    if est_premium_loss_pct >= sig.iv_adjusted_stop_pct:
                        gam_tag = (
                            f" ×{gam_mult:.1f}γ" if gam_mult > 1.0 else ""
                        )
                        reasons.append(
                            f"💸 IV-adjusted premium stop: est. option drawdown "
                            f"{est_premium_loss_pct:.0f}% ≥ {sig.iv_adjusted_stop_pct:.0f}% "
                            f"cap (IVR {sig.iv_rank:.0f}) — "
                            f"|Δ|={abs(sig.delta):.2f} × ${adverse_dollar:.2f} "
                            f"adverse / ${entry_mid:.2f} mid{gam_tag}"
                        )

            if reasons:
                logger.info(
                    "EXIT trigger for {}: {}", key, " | ".join(reasons),
                )
                await self._send_exit_alert(sig, spy_price, entry_price, reasons, regime_ctx.regime)
                self._exit_sent.add(key)

                # ── Record outcome in analytics DB ─────────────────────────
                if self._analytics:
                    # Determine which trigger type fired (first reason wins)
                    first = reasons[0] if reasons else ""
                    if "Time stop" in first:
                        trigger_label = "time_stop"
                    elif "Profit target" in first:
                        trigger_label = "profit_target"
                    elif "URGENT" in first or "dropped" in first or "rallied" in first:
                        trigger_label = "adverse_move"
                    elif "Regime" in first:
                        trigger_label = "regime_flip"
                    elif "VWAP reversion" in first:
                        trigger_label = "vwap_reversion"
                    elif "IV-adjusted premium stop" in first:
                        trigger_label = "iv_premium_stop"
                    else:
                        trigger_label = "manual"

                    # Classify win/loss/scratch from SPY move
                    raw_pct = (spy_price - entry_price) / entry_price * 100.0
                    if direction == "BULLISH":
                        fav_pct = raw_pct
                    elif direction == "BEARISH":
                        fav_pct = -raw_pct
                    else:
                        fav_pct = 0.0
                    if fav_pct > 0.1:
                        outcome_label = "win"
                    elif fav_pct < -0.1:
                        outcome_label = "loss"
                    else:
                        outcome_label = "scratch"

                    # Look up the analytics row id from sent_at + dedup_key
                    db_id = self._analytics.find_signal_id(sig)
                    if db_id is not None:
                        self._analytics.record_outcome(
                            signal_id=db_id,
                            outcome=outcome_label,
                            spy_price_exit=spy_price,
                            exit_trigger=trigger_label,
                            direction=direction,
                            entry_price=entry_price,
                        )

        # Clean up expired signals
        for k in expired_keys:
            self._active_signals.pop(k, None)
            self._exit_sent.discard(k)

    async def _send_exit_alert(
        self,
        sig: SpySignal,
        current_price: float,
        entry_price: float,
        reasons: List[str],
        current_regime: str,
    ) -> None:
        """Format and send an EXIT alert via Telegram."""
        msg = self._format_exit(sig, current_price, entry_price, reasons, current_regime)
        logger.info(
            "Sending EXIT alert: {} {}{}  entry=${:.2f} now=${:.2f}",
            sig.signal_type.value, sig.strike, sig.right,
            entry_price, current_price,
        )
        await self._telegram.send_message(msg)

    def _format_exit(
        self,
        sig: SpySignal,
        current_price: float,
        entry_price: float,
        reasons: List[str],
        current_regime: str,
    ) -> str:
        """Build the EXIT alert Telegram message."""
        price_chg = current_price - entry_price
        price_chg_pct = price_chg / entry_price * 100.0
        direction = self._signal_direction(sig)
        arrow = "📈" if price_chg > 0 else "📉"

        # Contract display (same logic as entry format)
        if sig.expiry_date:
            try:
                exp_dt = datetime.strptime(sig.expiry_date, "%Y%m%d")
                exp_display = exp_dt.strftime("%b %d, %Y")
            except ValueError:
                exp_display = sig.expiry
        else:
            exp_display = sig.expiry
        dte_tag = f"  ({sig.dte} DTE)" if sig.dte > 0 else ""

        now_et = datetime.now(ET).strftime("%H:%M ET")

        lines = [
            f"🚨 <b>EXIT ALERT — {sig.signal_type.value}</b> 🚨",
            "",
            f"📌 <b>SPY {exp_display} {sig.strike:.0f}{sig.right}</b>{dte_tag}",
            "",
            f"Original signal was <b>{direction}</b>",
            f"🔹 Entry SPY: <b>${entry_price:.2f}</b>",
            f"{arrow} Current SPY: <b>${current_price:.2f}</b>  "
            f"({price_chg:+.2f}, {price_chg_pct:+.2f}%)",
            "",
            "<b>⚠️ Exit Reason(s):</b>",
        ]
        for r in reasons:
            lines.append(f"  • {_html.escape(r)}")

        lines += [
            "",
            f"🌍 Current Regime: <b>{current_regime}</b>",
            f"🕐 {now_et}",
            "",
            "<i>⚠️ Consider closing or hedging this position. "
            "Not financial advice.</i>",
            "#SPY #Options #EXIT #ShreeBot",
        ]
        return "\n".join(lines)

    # ── Signal formatting ─────────────────────────────────────────────────────

    _TYPE_EMOJI = {
        SignalType.CALL_SWEEP:       "🔥",
        SignalType.PUT_SWEEP:        "🐻",
        SignalType.BULL_CALL_SPREAD: "🐂",
        SignalType.BEAR_PUT_SPREAD:  "🔻",
        SignalType.LONG_STRADDLE:    "⚡",
        SignalType.HIGH_IV_ALERT:    "📈",
        SignalType.PC_RATIO_EXTREME: "⚖️",
        SignalType.ORB_BREAKOUT:          "📐",
        SignalType.TREND_CONTINUATION:    "🔄",
    }

    # ── OptionsEdge-inspired named pattern classifier ──────────────────────────
    @staticmethod
    def _classify_named_pattern(
        signal_type: SignalType,
        right: str,
        orb_status: str,
        vwap_band: str,
        regime: str,
    ) -> str:
        """Map live signal context to one of the 13 named price patterns.

        Pattern names match OptionsEdge taxonomy (PDH Reclaim, ORB Long, etc.)
        so Telegram readers have an immediately recognisable setup label.
        Pure price-structure inference from already-computed state — no extra
        data fetches required.
        """
        # ORB / TREND_CONTINUATION — always ORB-family
        if signal_type in (SignalType.ORB_BREAKOUT, SignalType.TREND_CONTINUATION):
            return "ORB Long" if right == "C" else "ORB Short"

        is_bullish = right == "C" or signal_type in (
            SignalType.CALL_SWEEP, SignalType.BULL_CALL_SPREAD
        )
        is_bearish = right == "P" or signal_type in (
            SignalType.PUT_SWEEP, SignalType.BEAR_PUT_SPREAD
        )

        if is_bullish:
            # Price is already above the opening range → ORB continuation
            if orb_status == "ABOVE_ORB":
                return "ORB Long"
            # Price was below VWAP and is now recovering → VWAP Reclaim
            if vwap_band in ("BELOW_1SD", "BELOW_2SD"):
                return "VWAP Reclaim"
            # Price is extended above VWAP in trend regime → PDH Reclaim
            if vwap_band in ("ABOVE_1SD", "ABOVE_2SD") and "TREND" in regime:
                return "PDH Reclaim"
            # Default bullish sweep at/near VWAP
            return "VWAP Reclaim"

        if is_bearish:
            # Price broke below the opening range → ORB Short
            if orb_status == "BELOW_ORB":
                return "ORB Short"
            # Price was above VWAP and is now failing → VWAP Rejection
            if vwap_band in ("ABOVE_1SD", "ABOVE_2SD"):
                return "VWAP Rejection"
            # Price below VWAP in downtrend → PDL Breakdown
            if vwap_band in ("BELOW_1SD", "BELOW_2SD") and "TREND" in regime:
                return "PDL Breakdown"
            # Default bearish sweep at/near VWAP
            return "VWAP Rejection"

        return ""

    # ── DTE recommendation match helper ───────────────────────────────────────
    @staticmethod
    def _dte_rec_matches_actual(dte_rec_label: str, actual_dte: int) -> bool:
        """True when the actual contract DTE falls inside the recommended band.

        Parses labels like "0 DTE", "0-1 DTE", "1-2 DTE", "0-1 DTE" and
        returns True iff ``actual_dte`` is within the parsed inclusive range.
        Empty / unparseable labels return True (don't false-alarm).
        """
        import re
        if not dte_rec_label or actual_dte < 0:
            return True
        nums = [int(n) for n in re.findall(r"\d+", dte_rec_label)]
        if not nums:
            return True
        lo, hi = min(nums), max(nums)
        return lo <= actual_dte <= hi

    # ── DTE recommendation (mirrors OptionsEdge computeDTERec) ────────────────
    @staticmethod
    def _dte_recommendation(
        iv_rank: float,
        now_et: Optional[datetime] = None,
    ) -> "tuple[str, str]":
        """Return (recommended_dte_label, reason_str) for the current environment.

        Factors: session phase (time of day) and IV Rank.
        Matches OptionsEdge Layer 5 DTE optimisation logic.
        """
        if now_et is None:
            now_et = datetime.now(ET)
        t = now_et.time()

        # Power hour: 0DTE decays exponentially after 3 PM — step up to 1 DTE
        if time(15, 0) <= t < time(16, 0):
            return (
                "1-2 DTE",
                "After 3 PM — 0DTE exponential theta kill. 1 DTE preserves value overnight.",
            )
        # Very high IV: premium is severely overpriced → shorter DTE to cut vega
        if iv_rank > 65:
            return (
                "1-2 DTE",
                f"IV Rank {iv_rank:.0f} — premium overpriced. 1-2 DTE cuts vega exposure ~40%.",
            )
        # Elevated IV in afternoon: 1 DTE if target is far
        if iv_rank > 40 and time(13, 0) <= t < time(15, 0):
            return (
                "0-1 DTE",
                f"Mid-session + elevated IV Rank {iv_rank:.0f}. Use 1 DTE if target > 1 ATR away.",
            )
        # Prime-time with normal IV: 0DTE is ideal for directional leverage
        if time(9, 45) <= t < time(11, 30):
            return (
                "0 DTE",
                "Prime-time + normal IV — 0DTE maximises directional leverage.",
            )
        return (
            "0-1 DTE",
            "Standard session window — 0DTE if within 2 hrs of entry, else 1 DTE.",
        )

    # ── Event-specific action guidance ───────────────────────────────────────
    @staticmethod
    def _event_action_guidance(
        event_title: str,
        event_minutes: float,
        right: str,
    ) -> str:
        """Return specific, actionable text for a known high-impact event type.

        Replaces the generic 'EVENT RISK in X min' with the same kind of
        explicit pre-trade instruction a professional desk would issue —
        matching the OptionsEdge TODAY_EVENTS guidance approach.
        """
        title_up = event_title.upper()
        direction_word = "CALL" if right == "C" else "PUT" if right == "P" else "option"
        close_min = max(0, int(event_minutes - 30))

        if "FOMC" in title_up or "INTEREST RATE" in title_up or "FED DECISION" in title_up:
            return (
                f"IV will CRUSH immediately post-announcement — direction irrelevant. "
                f"Close ALL option longs within {close_min} min (30 min before event). "
                "Long vega into FOMC = donating premium."
            )
        if any(k in title_up for k in ("CPI", "INFLATION", "PPI", "PCE")):
            return (
                "Expect IV spike into release then immediate crush post-print. "
                "Spread structures preferred over naked options today."
            )
        if any(k in title_up for k in ("NFP", "NON-FARM", "PAYROLL", "JOBS REPORT")):
            return (
                "Gap risk is highest on NFP days. "
                "Reduce size 50% or use debit spread to cap IV-crush exposure."
            )
        if any(k in title_up for k in ("EARNINGS", "EPS", "RESULTS")):
            return (
                f"Earnings IV will crush 40-60%% post-release. "
                f"Buying {direction_word}s today = owning IV that evaporates at the close. "
                "This event may OVERRIDE the signal — reconsider before entering."
            )
        if any(k in title_up for k in ("OPEX", "EXPIR", "TRIPLE WITCH", "QUAD WITCH")):
            return (
                "Options expiration day — max pain pinning is strongest. "
                "Gamma risk is extreme on 0DTE near the max pain strike."
            )
        # Generic high-impact fallback
        return (
            f"High-impact event in {event_minutes:.0f} min — "
            "consider reducing size by 50% or waiting for post-event price clarity."
        )

    _TIER_LABEL = {
        "MEDIUM":  "MEDIUM",
        "HIGH":    "★ HIGH",
        "EXTREME": "★★ EXTREME ★★",
    }

    def _format(self, sig: SpySignal) -> str:
        emoji = self._TYPE_EMOJI.get(sig.signal_type, "📊")
        conf_pct = int(sig.confidence * 100)
        tier_label = self._TIER_LABEL.get(sig.confidence_tier, sig.confidence_tier)
        now_et_dt = datetime.now(ET)
        now_et = now_et_dt.strftime("%H:%M ET")

        # ── OptionsEdge-inspired context helpers ───────────────────────────
        named_pattern = self._classify_named_pattern(
            sig.signal_type,
            sig.right,
            getattr(self, "_last_orb_status", "BUILDING"),
            getattr(self, "_last_vwap_band", "INSIDE_1SD"),
            sig.regime,
        )
        dte_rec, dte_reason = self._dte_recommendation(sig.iv_rank, now_et_dt)

        # Power Hour 0DTE size warning flag
        _is_power_hour_0dte = (
            sig.dte == 0
            and time(15, 0) <= now_et_dt.time() < time(16, 0)
        )

        # Format expiry: "Apr 17, 2026 (16 DTE)" if date available, else "APR26"
        if sig.expiry_date:
            try:
                exp_dt = datetime.strptime(sig.expiry_date, "%Y%m%d")
                exp_display = exp_dt.strftime("%b %d, %Y")
            except ValueError:
                exp_display = sig.expiry
        else:
            exp_display = sig.expiry
        dte_tag = f"  ({sig.dte} DTE)" if sig.dte > 0 else ""

        # Contract line: "SPY Apr 17, 2026 659P (16 DTE)"
        contract_line = f"📌 <b>SPY {exp_display} {sig.strike:.0f}{sig.right}</b>{dte_tag}"

        lines = [
            f"{emoji} <b>SPY OPTIONS — {sig.signal_type.value}</b>",
            "",
            contract_line,
            f"💰 SPY: <b>${sig.spy_price:.2f}</b>",
        ]

        # Named pattern (OptionsEdge taxonomy) + DTE recommendation
        if named_pattern:
            lines.append(f"📋 Pattern: <b>{named_pattern}</b>")

        # ── DTE recommendation vs actual contract DTE ──────────────────────
        # Reviewer caught a real bug: signal showed "Rec DTE: 0 DTE" while
        # the actual contract was 23 DTE. They are different trades — 0DTE
        # is a high-gamma intraday breakout play, 23DTE is a vol-expansion
        # play with much slower theta. Flag the mismatch loudly so the
        # trader knows the pricing model assumed in the recommendation does
        # not match the contract that triggered.
        dte_rec_match = self._dte_rec_matches_actual(dte_rec, sig.dte)
        if dte_rec_match or sig.dte == 0:
            # Match (or 0DTE-on-0DTE) — show as before.
            lines.append(
                f"📅 Rec DTE: <b>{dte_rec}</b>  "
                f"<i>({_html.escape(dte_reason)})</i>"
            )
        else:
            lines.append(
                f"⚠️ <b>DTE mismatch</b>: contract is "
                f"<b>{sig.dte} DTE</b> but optimal is <b>{dte_rec}</b>  "
                f"<i>({_html.escape(dte_reason)})</i>"
            )
            lines.append(
                "  <i>This is a different trade type than the model assumes — "
                "longer-dated = lower gamma but slower theta and bigger move "
                "needed; size accordingly.</i>"
            )

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

        # ── Edge Reality block (May 2026 — institutional-audit additions) ──
        # Only rendered when _apply_edge_reality has populated the fields
        # (regime_wr > 0). Off when the feature flag is disabled.
        #
        # Source attribution rules:
        #   - WR figures get a "[doc-prior]" tag until analytics_db has
        #     enough closed trades to switch to empirical WRs. Prevents the
        #     trader from over-trusting a green margin built on assumed WRs.
        #   - Round-trip cost gets a "~" prefix when the function fell back
        #     to the SPY 4.8% default (no live bid/ask available). On a wide-
        #     spread strike that fallback can flip a real-red signal to
        #     apparent-green, so the marker is loud.
        if sig.regime_wr > 0:
            color_icon = (
                "🟢" if sig.edge_color == "green"
                else "🟠" if sig.edge_color == "amber"
                else "🔴"
            )
            margin_sign = "+" if sig.edge_margin >= 0 else ""
            wr_tag = (
                "  <i>[doc-prior]</i>" if sig.wr_source == "doc_prior" else ""
            )
            rt_prefix = "~" if sig.rt_source != "live_quote" else ""
            lines += [
                "",
                "<b>🎯 Edge Reality:</b>",
                f"  📈 Regime WR: <b>{sig.regime_wr}%</b>"
                f"  <i>(historical {sig.historical_wr}%)</i>{wr_tag}",
                f"  ⚖️ Break-even WR: <b>{sig.breakeven_wr:.1f}%</b>"
                f"  <i>(target {sig.target_pct_assumed:.0f}%, "
                f"stop {sig.iv_adjusted_stop_pct:.0f}%, "
                f"round-trip {rt_prefix}{sig.round_trip_cost_pct:.1f}%)</i>",
                f"  {color_icon} Net edge: <b>{margin_sign}{sig.edge_margin:.1f}%</b>",
            ]
            if sig.rt_source != "live_quote":
                lines.append(
                    "  <i>ℹ️ Round-trip cost is a SPY default — no live bid/ask "
                    "for this strike; a wider real spread would compress the edge.</i>"
                )
            if sig.edge_color == "red":
                lines.append(
                    "  <i>⚠️ No edge after costs — limit orders critical "
                    "or skip this trade.</i>"
                )
            elif sig.edge_color == "amber":
                lines.append(
                    "  <i>⚠️ Thin edge — one bad fill eliminates profitability.</i>"
                )

        # ── Power Hour gamma + theta warning ──────────────────────────────
        # Only fires when gamma_accel_mult > 1.0 (i.e. 0DTE inside last 2 hrs)
        # OR when hourly theta is consequential (>$5/hr per contract).
        if sig.gamma_warning_active or sig.hourly_theta_dollars > 5.0 or _is_power_hour_0dte:
            lines += ["", "<b>⏳ Power Hour Risk:</b>"]
            if _is_power_hour_0dte:
                lines.append(
                    "  🚨 <b>0DTE POWER HOUR — REDUCE SIZE 50%.</b> "
                    "Gamma convexity is extreme: small SPY moves cause outsized premium swings."
                )
            if sig.gamma_warning_active:
                lines.append(
                    f"  💥 Effective Γ: <b>{sig.effective_gamma:.4f}</b> "
                    f"(×{sig.gamma_accel_mult:.1f} static) — gamma-bomb territory"
                )
            if sig.hourly_theta_dollars > 0:
                lines.append(
                    f"  ⏱ Hourly theta cost: <b>≈${sig.hourly_theta_dollars:.2f}/hr</b> "
                    f"per contract"
                )

        # ── SPY put-skew warning (bearish signals only) ───────────────────
        if sig.skew_warning:
            lines.append(f"⚠️ <b>Skew:</b> <i>{_html.escape(sig.skew_warning)}</i>")

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

        # Contract cost estimate
        _ask_px = sig.ask if sig.ask > 0 else 0.0
        _bid_px = sig.bid if sig.bid > 0 else 0.0
        if _ask_px > 0:
            _mid = (_bid_px + _ask_px) / 2 if _bid_px > 0 else _ask_px
            if sig.right == "BOTH":
                # bid/ask already = call+put combined premium; × 100 shares = total cost
                _total = _mid * 100
                lines.append(
                    f"💲 Straddle cost (1C + 1P): <b>≈${_total:,.0f}</b>"
                    f"<i> (combined mid ${_mid:.2f} × 100 shares)</i>"
                )
            else:
                _cost1 = _mid * 100
                lines.append(
                    f"💲 1 contract costs <b>≈${_cost1:,.0f}</b>"
                    f"<i> (mid ${_mid:.2f} × 100)</i>"
                )

        # Sentiment (IB-based)
        lines.append(
            f"🧭 Sentiment: <b>{sig.sentiment_score:+.0f}</b> ({sig.sentiment_label})"
        )

        # ── External signals block ─────────────────────────────────────────────
        has_ext = (
            sig.external_composite != 0.0 or sig.news_score != 0.0
            or sig.flow_confirmation_score != 0.0
        )
        if has_ext:
            ext_label = (
                "BULLISH" if sig.external_composite > 0.15 else
                "BEARISH" if sig.external_composite < -0.15 else
                "NEUTRAL"
            )
            lines.append(
                f"🌐 Ext Composite: <b>{sig.external_composite:+.2f}</b> ({ext_label})"
            )

            # Options flow
            if sig.flow_confirmation_score != 0.0:
                flow_arrow = "🟢" if sig.flow_confirmation_score > 10 else "🔴" if sig.flow_confirmation_score < -10 else "🟡"
                lines.append(
                    f"  {flow_arrow} Flow: <b>{sig.flow_confirmation_score:+.0f}</b>"
                    f"  GEX: {sig.gex_bias.replace('_', ' ')}"
                )
                if sig.dark_pool_bias != "NEUTRAL":
                    dp_icon = "🏦↑" if sig.dark_pool_bias == "ACCUMULATION" else "🏦↓"
                    lines.append(f"  {dp_icon} Dark Pool: <b>{sig.dark_pool_bias}</b>")
                if sig.intraday_pc_ratio is not None:
                    lines.append(f"  ⚖️ Intraday P/C: {sig.intraday_pc_ratio:.2f}")

            # Macro
            if sig.macro_headwind != 0.0 or sig.macro_label != "NEUTRAL":
                hw_icon = "🔴" if "HEADWIND" in sig.macro_label else "🟢" if "TAILWIND" in sig.macro_label else "🟡"
                lines.append(
                    f"  {hw_icon} Macro: <b>{sig.macro_label}</b> "
                    f"| TNX {sig.tnx_trend} | DXY {sig.dxy_trend}"
                )

            # News / social
            if sig.news_score != 0.0:
                news_icon = "📰+" if sig.news_score > 0.05 else "📰-" if sig.news_score < -0.05 else "📰~"
                lines.append(f"  {news_icon} News: {sig.news_score:+.3f}")
            if sig.retail_score != 0.0:
                lines.append(f"  📱 StockTwits: {sig.retail_score:+.2f}")
            if sig.equity_pc is not None:
                lines.append(f"  ⚖️ CBOE Equity P/C: {sig.equity_pc:.2f}")

        # Technical levels block (injected from TechnicalLevelsTracker)
        _orb_s = getattr(self, "_last_orb_status", None)
        _orb_h = getattr(self, "_last_orb_high", None)
        _orb_l = getattr(self, "_last_orb_low", None)
        _vwap_band = getattr(self, "_last_vwap_band", None)
        _edr_pct = getattr(self, "_last_edr_pct", None)
        _rsi = getattr(self, "_last_rsi_5m", None)
        _rsi_div = getattr(self, "_last_rsi_div", None)
        _pvt_n = getattr(self, "_last_pivot_nearest", None)
        _pvt_b = getattr(self, "_last_pivot_bias", None)
        _mp = getattr(self, "_last_max_pain", None)
        _near_mp = getattr(self, "_last_near_max_pain", False)

        tech_lines = []
        if _orb_s and _orb_s != "BUILDING":
            orb_icon = "✅" if _orb_s in ("ABOVE_ORB", "BELOW_ORB") else "⏸"
            orb_range = f" ({_orb_h:.2f}–{_orb_l:.2f})" if _orb_h and _orb_l else ""
            tech_lines.append(f"  {orb_icon} ORB: <b>{_orb_s}</b>{orb_range}")
        if _vwap_band and _vwap_band != "INSIDE_1SD":
            band_icon = "🔴" if "ABOVE" in _vwap_band else "🟢"
            tech_lines.append(f"  {band_icon} VWAP Band: <b>{_vwap_band.replace('_', ' ')}</b>")
        if _edr_pct is not None and _edr_pct > 0:
            edr_icon = "⚠️" if _edr_pct >= 85 else "📏"
            tech_lines.append(f"  {edr_icon} EDR used: <b>{_edr_pct:.0f}%</b>")
        if _rsi is not None:
            rsi_div_tag = f" [{_rsi_div.replace('_', ' ')}]" if _rsi_div and _rsi_div != "NONE" else ""
            tech_lines.append(f"  📊 RSI (5m): {_rsi:.0f}{rsi_div_tag}")
        if _pvt_n:
            tech_lines.append(f"  📍 Near {_pvt_n} pivot ({_pvt_b.replace('_', ' ')})")
        if _mp is not None:
            mp_tag = "  [NEAR PIN]" if _near_mp else ""
            tech_lines.append(f"  📌 Max Pain: ${_mp:.2f}{mp_tag}")
        if tech_lines:
            lines += ["", "<b>📐 Tech Levels:</b>"] + tech_lines

        # Dynamic confidence note (show when adjustment is meaningful)
        if abs(sig.dynamic_confidence_delta) >= 0.03:
            sign = "+" if sig.dynamic_confidence_delta > 0 else ""
            lines.append(
                f"⚙️ Dyn adj: <b>{sign}{sig.dynamic_confidence_delta:.0%}</b> "
                f"[{sig.confidence_time_bucket} | {sig.confidence_dte_rule}]"
            )
        if sig.conflict_detected:
            lines.append("⚡ <b>Conflicting signals detected</b> — reduced conviction")

        # Event risk — specific actionable guidance per event type
        if sig.event_risk:
            event_guidance = self._event_action_guidance(
                sig.next_event_title, sig.event_minutes, sig.right
            )
            lines += [
                "",
                f"🚨 <b>EVENT RISK: {_html.escape(sig.next_event_title)}</b> "
                f"in <b>{sig.event_minutes:.0f} min</b>",
                f"  ⚡ {_html.escape(event_guidance)}",
            ]
        elif sig.next_event_title and sig.event_minutes < 120:
            lines.append(
                f"📅 Next event: {_html.escape(sig.next_event_title)} "
                f"({sig.event_minutes:.0f} min)"
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

        # Exit trigger levels (only for directional signals)
        direction = self._signal_direction(sig)
        if direction in ("BULLISH", "BEARISH") and sig.spy_price:
            entry_px = sig.spy_price
            if direction == "BULLISH":
                exit_05 = entry_px * (1 - 0.005)
                exit_10 = entry_px * (1 - 0.010)
                lines += [
                    "",
                    f"⚠️ Exit trigger: SPY ≤ ~${exit_05:,.2f} "
                    f"(−0.5% from ${entry_px:.2f})",
                    f"🔴 Urgent exit: SPY ≤ ~${exit_10:,.2f} (−1.0%)",
                ]
            else:  # BEARISH
                exit_05 = entry_px * (1 + 0.005)
                exit_10 = entry_px * (1 + 0.010)
                lines += [
                    "",
                    f"⚠️ Exit trigger: SPY ≥ ~${exit_05:,.2f} "
                    f"(+0.5% from ${entry_px:.2f})",
                    f"🔴 Urgent exit: SPY ≥ ~${exit_10:,.2f} (+1.0%)",
                ]

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
            "Sending signal: {} {}{}  exp={} ({})  dte={}  conf={:.0f}%  tier={}  regime={}",
            sig.signal_type.value, sig.strike, sig.right,
            sig.expiry_date or sig.expiry, sig.expiry,
            sig.dte, sig.confidence * 100,
            sig.confidence_tier, sig.regime,
        )
        await self._telegram.send_message(msg)

    async def _tm_check_and_publish(self, sig: SpySignal) -> bool:
        """Publish candidate to logs/spy_signals.jsonl and check the Trading
        Manager's verdict from logs/manager_decisions.jsonl.

        Returns:
            True if the manager APPROVED, MODIFIED, or could not be reached
                (fail-open).
            False if the manager explicitly REJECTED.
        """
        import asyncio
        import json
        import os
        from datetime import datetime, timezone

        # Build a stable signal_id that the manager uses to write its verdict.
        ts_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        signal_id = f"{sig.dedup_key}:{sig.expiry}:{ts_iso}"

        spy_jsonl = os.environ.get("TM_SPY_SIGNALS_FILE", "logs/spy_signals.jsonl")
        decisions_jsonl = os.environ.get("TM_MANAGER_FILE", "logs/manager_decisions.jsonl")

        rec = {
            "ts": ts_iso,
            "kind": "spy_signal",
            "signal_id": signal_id,
            "signal_type": sig.signal_type.value,
            "strike": float(sig.strike),
            "right": sig.right,
            "expiry": sig.expiry,
            "expiry_date": sig.expiry_date,
            "dte": int(sig.dte or 0),
            "confidence": float(sig.confidence or 0.0),
            "confidence_tier": sig.confidence_tier,
            "spy_price": float(sig.spy_price or 0.0),
            "vix": float(sig.vix or 0.0) if sig.vix is not None else 0.0,
            "iv_rank": float(sig.iv_rank or 0.0),
            "regime": sig.regime,
            "delta": float(sig.delta or 0.0),
            "gamma": float(sig.gamma or 0.0),
            "theta": float(sig.theta or 0.0),
            "vega": float(sig.vega or 0.0),
            "impl_vol": float(sig.impl_vol or 0.0),
            "bid": float(sig.bid or 0.0),
            "ask": float(sig.ask or 0.0),
            "spread_pct": float(sig.spread_pct or 0.0),
            "volume": int(sig.volume or 0),
            "open_interest": int(sig.open_interest or 0),
            "sentiment_label": sig.sentiment_label,
            "sentiment_score": float(sig.sentiment_score or 0.0),
            "reasoning": list(sig.reasoning or []),
            "suggested_trade": sig.suggested_trade,
            # MAY 19 2026 — structure tag lets the Trading Manager size
            # actual per-trade risk correctly for defined-risk spreads.
            # Older readers ignore unknown keys, so this is back-compatible.
            "structure": getattr(sig, "structure", "") or "",
            "short_strike": float(getattr(sig, "short_strike", 0.0) or 0.0),
            "short_bid": float(getattr(sig, "short_bid", 0.0) or 0.0),
            "short_ask": float(getattr(sig, "short_ask", 0.0) or 0.0),
        }

        try:
            os.makedirs(os.path.dirname(spy_jsonl) or ".", exist_ok=True)
            with open(spy_jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as exc:
            logger.warning("TM publish failed (fail-open): {}", exc)
            return True

        # Poll for the manager verdict for up to 2.5s (TM polls every 1s).
        try:
            from ..trading_manager.decision_log import latest_decision_by_signal_id
        except ImportError as exc:
            logger.warning("TM module not importable (fail-open): {}", exc)
            return True

        deadline = asyncio.get_event_loop().time() + 2.5
        verdict = None
        while asyncio.get_event_loop().time() < deadline:
            verdict = latest_decision_by_signal_id(decisions_jsonl, signal_id)
            if verdict is not None:
                break
            await asyncio.sleep(0.2)

        if verdict is None:
            logger.warning(
                "⚠️  Trading Manager decision not found for {} — proceeding fail-open. "
                "Verify the TM daemon is running.",
                signal_id,
            )
            return True

        decision = (verdict.get("decision") or "").upper()
        reasoning = str(verdict.get("reasoning", ""))[:200]
        if decision == "REJECT":
            logger.warning(
                "🛡️  TM REJECT: {} | {}",
                signal_id, reasoning,
            )
            return False
        if decision == "MODIFY":
            logger.warning(
                "🛡️  TM MODIFY (size-down handled by TM in future): {} | {}",
                signal_id, reasoning,
            )
            return True
        # APPROVE or unknown → proceed
        logger.info("🛡️  TM {}: {}", decision or "APPROVE", reasoning)
        return True
