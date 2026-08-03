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
  9. When execution.enabled: signals passing the strict quality gate are
     executed as IB bracket orders (limit entry + stop-loss + take-profit)
     via SpyOptionsExecutor. Otherwise signal-only.
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
from .exit_engine import (
    FULL_EXIT,
    HOLD,
    PARTIAL_EXIT,
    TRAIL_STOP,
    ExitDecision,
    ExitEngine,
    ExitSnapshot,
    PositionExitState,
)
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
from .executor import SpyOptionsExecutor
from .external import ExternalDataManager
from .ib_client import IBOptionsClient
from .regime_detector import RegimeContext, RegimeDetector
from .rules_v2.engine import EngineInputs, RulesV2Engine
from .sentiment_engine import SentimentContext, SentimentEngine
from .signal_engine import log_blocked_signal, SignalContext, SignalEngine, SignalType, SpySignal, _tier
from .sweep_tracker import SweepTracker
from .technical_levels import TechnicalLevelsTracker, compute_max_pain
from .real_flow import RealFlowFeed, RealFlowState
from .cross_asset import CrossAssetFeed, CrossAssetState
from . import v2_shadow_gate

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
        # Expiry selection: "nearest" (0-2 DTE dailies/weeklies, tradeable by
        # the executor) vs "monthly" (legacy end-of-month). JUL 2 2026.
        self._ib.expiry_mode = getattr(cfg.chain, "expiry_selection", "nearest")
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

        # Order executor (JUL 2 2026) — bracket orders at IB for signals that
        # pass the strict quality gate. None when execution.enabled is false,
        # in which case the bot behaves exactly as the legacy signal-only feed.
        if cfg.execution.enabled:
            self._executor: Optional[SpyOptionsExecutor] = SpyOptionsExecutor(
                cfg.execution, telegram=self._telegram, analytics=self._analytics,
            )
            logger.info(
                "SPY EXECUTION ENABLED — port={} risk/trade=${:.0f} "
                "TP+{:.0f}% IV-adj stops, strict gate (green edge, {}, ≤{} DTE)",
                cfg.execution.ibkr_port, cfg.execution.risk_per_trade_usd,
                cfg.execution.take_profit_pct,
                "/".join(cfg.execution.allowed_tiers), cfg.execution.max_dte,
            )
        else:
            self._executor = None

        # Deduplication: dedup_key → last sent datetime
        self._sent_times: Dict[str, datetime] = {}
        self._last_reset_date: Optional[str] = None

        # Active signal tracking for exit alerts
        # dedup_key → {signal, entry_price, entry_regime, sent_at}
        self._active_signals: Dict[str, Dict] = {}
        self._exit_sent: Set[str] = set()  # dedup_keys for which exit was already sent

        # Exit Engine v2 (JUL 14 2026, docs/SPY_EXIT_ENGINE_DESIGN.md).
        # shadow_mode=True → evaluates+logs only; legacy triggers keep acting.
        _ee_cfg = getattr(cfg, "exit_engine", None)
        self._exit_engine: Optional[ExitEngine] = (
            ExitEngine(_ee_cfg) if _ee_cfg is not None and _ee_cfg.enabled else None
        )
        self._exit_states: Dict[str, PositionExitState] = {}

        # Shadow book (strategy audit 2026-07-17): BLOCKED signals tracked with
        # the same simulated-exit semantics as dispatched ones, persisted with
        # blocked_gate set — the counterfactual data for strategy selection.
        self._shadow_signals: Dict[str, Dict] = {}
        self._shadow_cooldown: Dict[str, datetime] = {}
        # Per-poll snapshot inputs for the exit engine (stashed in _poll).
        self._last_bars_5m: List[Dict] = []
        self._last_rsi_5m: float = 50.0
        self._last_vix: Optional[float] = None

        # Option conid resolution cache: (expiry_month, strike, right) → conid
        self._conid_map: Dict[tuple, int] = {}
        self._conid_details: Dict[int, tuple] = {}

        # Technical levels tracker (ORB, VWAP bands, pivots, EDR, RSI)
        self._tech_tracker = TechnicalLevelsTracker()

        # Real order-flow feed (L2 depth + tape) — created in start() once the
        # SPY contract is qualified; None when disabled or unavailable
        self._real_flow: Optional[RealFlowFeed] = None

        # Cross-asset confirmation feed (live QQQ/IWM bars via IB)
        self._cross_asset: Optional[CrossAssetFeed] = None

        # Set by the daily reset; consumed on the first market-open poll to
        # re-subscribe the RealFlow tape/depth streams (they go stale overnight).
        self._feeds_need_resubscribe: bool = False

        # Latest ExternalContext, snapshotted onto signals for the research log.
        self._last_ext_ctx = None

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

        if self._executor is not None:
            await self._executor.start()

        # Real order-flow feeds (L2 depth + tick-by-tick tape) on the shared
        # ib_insync connection. Feeds degrade to unavailable on entitlement errors.
        rf_cfg = getattr(self._cfg, "real_flow", None)
        if rf_cfg is not None and rf_cfg.enabled and self._ib.spy_contract is not None:
            try:
                self._real_flow = RealFlowFeed(
                    self._ib.ib,
                    self._ib.spy_contract,
                    tape_enabled=rf_cfg.tape_enabled,
                    depth_enabled=rf_cfg.depth_enabled,
                    depth_levels=rf_cfg.depth_levels,
                    tape_window_minutes=rf_cfg.tape_window_minutes,
                    large_print_shares=rf_cfg.large_print_shares,
                )
                await self._real_flow.start()
            except Exception as exc:
                logger.warning("RealFlowFeed init failed — continuing without: {}", exc)
                self._real_flow = None

        # Cross-asset confirmation: live QQQ/IWM on the shared connection
        try:
            self._cross_asset = CrossAssetFeed(self._ib.ib)
            await self._cross_asset.start()
        except Exception as exc:
            logger.warning("CrossAssetFeed init failed — continuing without: {}", exc)
            self._cross_asset = None

        self._running = True
        try:
            while self._running:
                try:
                    # An IB Gateway drop mid-poll cancels the in-flight request
                    # future → CancelledError. That is a BaseException, so the
                    # broad `except Exception` below NEVER caught it: it
                    # propagated out of the loop into the finally and KILLED the
                    # whole bot (reproduced daily 08:01 CST, Jul 20/22/23 — the
                    # Gateway's auto-restart landed ~60s after the 08:00 start).
                    # The reconnect the disconnect handler scheduled never ran
                    # because asyncio.run then cancelled every pending task.
                    # Now: reconnect BEFORE polling a dead socket, and treat a
                    # transient CancelledError as recoverable (only a real
                    # stop() — which sets _running False — ends the bot).
                    if not self._ib.ib.isConnected():
                        logger.warning(
                            "SPY data socket down — reconnecting before poll")
                        await self._ib.reconnect_now()
                        if not self._ib.ib.isConnected():
                            await asyncio.sleep(10)
                            continue
                    await self._poll(spy_conid)
                except asyncio.CancelledError:
                    if not self._running:
                        raise                       # genuine shutdown — honour it
                    logger.warning(
                        "SPY poll cancelled (IB disconnect) — surviving; will "
                        "reconnect on next loop")
                except Exception as exc:
                    logger.opt(exception=True).error("Poll error: {}", exc)
                try:
                    await asyncio.sleep(self._cfg.session.poll_interval_s)
                except asyncio.CancelledError:
                    if not self._running:
                        raise
        finally:
            if self._real_flow is not None:
                self._real_flow.stop()
            if self._executor is not None:
                await self._executor.close()
            await self._ib.close()
            await self._telegram.close()
            if self._analytics:
                self._analytics.close()
            logger.info("=== SPY Options Bot stopped ===")

    # ── Session gate ──────────────────────────────────────────────────────────

    # NYSE full-day market holidays (YYYY-MM-DD). Observed dates included.
    _MARKET_HOLIDAYS = {
        # 2026
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
        "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
        "2026-11-26", "2026-12-25",
        # 2027
        "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
        "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06",
        "2027-11-25", "2027-12-24",
    }

    def _trading_dte(self, expiry_yyyymmdd: str) -> Optional[int]:
        """Days-to-expiry counted in TRADING sessions (Mon–Fri minus NYSE
        holidays), from today (ET) exclusive to expiry inclusive.

        Thu → Mon expiry = 2 (Fri, Mon); Fri → Mon = 1; same-day 0DTE = 0.
        Returns None when the expiry string is missing/invalid so callers can
        fall back to the calendar count."""
        try:
            expiry = datetime.strptime(expiry_yyyymmdd, "%Y%m%d").date()
        except (TypeError, ValueError):
            return None
        d = datetime.now(ET).date()
        if expiry <= d:
            return 0
        n = 0
        while d < expiry:
            d += timedelta(days=1)
            if d.weekday() < 5 and d.strftime("%Y-%m-%d") not in self._MARKET_HOLIDAYS:
                n += 1
        return n

    def _register_shadow(self, sig: SpySignal, gate: str) -> None:
        """Shadow book: give a BLOCKED signal the same simulated-exit tracking
        as a dispatched one, persisted with blocked_gate set. 45-min cooldown
        per (gate, dedup_key) — repeated per-poll kills of the same setup are
        one sample, not seventy. Best-effort; never blocks the signal path."""
        try:
            if sig.right not in ("C", "P") or not sig.spy_price:
                return
            direction = self._signal_direction(sig)
            if direction not in ("BULLISH", "BEARISH"):
                return
            key = f"{gate}:{sig.dedup_key}"
            now = datetime.utcnow()
            if key in self._shadow_signals:
                return
            cd = self._shadow_cooldown.get(key)
            if cd and (now - cd) < timedelta(minutes=45):
                return
            if len(self._shadow_signals) >= 60:      # runaway guard
                return
            db_id = None
            if self._analytics:
                self._analytics.insert(sig)
                db_id = self._analytics.find_signal_id(sig)
                if db_id:
                    self._analytics.mark_blocked(db_id, gate)
            self._shadow_cooldown[key] = now
            self._shadow_signals[key] = {
                "db_id": db_id,
                "entry_price": float(sig.spy_price),
                "direction": direction,
                "dte": int(sig.dte or 1),
                "sent_at": now,
            }
        except Exception as exc:
            logger.debug("shadow register failed: {}", exc)

    def _simulate_shadow_exits(self, spy_price: float) -> None:
        """Resolve shadow-book entries with the SAME trigger semantics as the
        dispatched-signal simulator (profit +0.5% / adverse −0.5%,−1.0% /
        time-stop 45|90min, win/loss/scratch at ±0.1%) so blocked-vs-dispatched
        expectancy is apples-to-apples."""
        if not self._shadow_signals or spy_price <= 0:
            return
        now = datetime.utcnow()
        for key, se in list(self._shadow_signals.items()):
            age_min = (now - se["sent_at"]).total_seconds() / 60.0
            chg = (spy_price - se["entry_price"]) / se["entry_price"] * 100.0
            fav = chg if se["direction"] == "BULLISH" else -chg
            trigger = None
            outcome = "scratch"
            if fav >= 0.5:
                trigger, outcome = "profit_target", "win"
            elif fav <= -1.0:
                trigger, outcome = "adverse_move", "loss"
            elif fav <= -0.5:
                trigger, outcome = "adverse_move", "loss"
            elif age_min >= (45 if se["dte"] == 0 else 90):
                trigger = "time_stop"
                outcome = ("win" if fav > 0.1 else
                           "loss" if fav < -0.1 else "scratch")
            if trigger:
                if self._analytics and se.get("db_id"):
                    try:
                        self._analytics.record_outcome(
                            signal_id=se["db_id"], outcome=outcome,
                            spy_price_exit=spy_price, exit_trigger=trigger,
                            direction=se["direction"],
                            entry_price=se["entry_price"],
                        )
                    except Exception as exc:
                        logger.debug("shadow outcome failed: {}", exc)
                self._shadow_signals.pop(key, None)
            elif age_min > 360:                      # stale — drop unresolved
                self._shadow_signals.pop(key, None)

    def _market_open(self) -> bool:
        now = datetime.now(ET)
        if now.weekday() >= 5:
            return False
        if now.strftime("%Y-%m-%d") in self._MARKET_HOLIDAYS:
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
            self._exit_states.clear()
            self._shadow_signals.clear()
            self._shadow_cooldown.clear()
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
            if self._executor is not None:
                self._executor.daily_reset()
            # RealFlow tape/depth subscriptions die across the overnight session
            # boundary. Flag a re-subscribe for the first market-open poll (doing
            # it now, at ET-midnight with the market closed, would just re-stale).
            self._feeds_need_resubscribe = True
            logger.info("New day {} — signal dedup + tracker reset", today)

    # ── IV rank ───────────────────────────────────────────────────────────────

    def _compute_iv_rank(self, vix: Optional[float]) -> float:
        """NORMALIZED VIX REGIME SCORE (0-100) — NOT option IV rank/percentile.

        MISNOMER WARNING (do not trust the name in research): this is VIX's
        position within its own trailing 52-week range, i.e. a coarse
        vol-regime score derived purely from the VIX index. It is NOT the
        implied-volatility rank/percentile of the specific option being traded
        (that would require per-strike IV history, e.g. from ThetaData, and was
        shown to be non-predictive so is deliberately not computed). The field
        and DB column are still called `iv_rank` for backward compatibility;
        read them as `vix_regime_score`. A true rename is a tracked migration.
        """
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
        expiry_date: Optional[str] = None,
        liquidity_override: Optional[dict] = None,
    ) -> Optional[ChainSnapshot]:
        """Fetch strikes, resolve conids, snapshot Greeks → ChainSnapshot.

        ``expiry_date`` (YYYYMMDD), when given, pins the exact expiration so a
        1DTE continuation chain can be built distinct from the 0DTE chain in the
        same month. The conid cache is keyed by the resolved expiry DATE (not the
        month) so 0DTE and 1DTE contracts never collide.

        ``liquidity_override`` (min_oi/max_spread_pct/min_volume) relaxes the
        default filter — used for the continuation swing chain so the ATM
        ~0.50Δ strikes (lower volume 2-3 DTE out) survive for enrichment. The
        executor's own gates (spread, delta) remain the final arbiter.
        """
        cfg_c = self._cfg.chain
        _liq_min_oi = (liquidity_override or {}).get("min_oi", cfg_c.liquidity_min_oi)
        _liq_max_spread = (liquidity_override or {}).get("max_spread_pct", cfg_c.liquidity_max_spread_pct)
        _liq_min_vol = (liquidity_override or {}).get("min_volume", cfg_c.liquidity_min_volume)

        strikes_data = await self._ib.get_strikes(spy_conid, expiry_month, cfg_c.exchange)
        if not strikes_data:
            logger.warning("No strikes returned for SPY {}", expiry_month)
            return None

        call_strikes = strikes_data.get("call", [])
        put_strikes  = strikes_data.get("put", [])

        half_cap = cfg_c.max_strikes_per_expiry // 2
        near_calls = _near_strikes(call_strikes, spy_price, cfg_c.strike_pct_range, half_cap)
        near_puts  = _near_strikes(put_strikes,  spy_price, cfg_c.strike_pct_range, half_cap)

        # Resolve the exact expiration up front; the conid cache keys on it.
        resolved_expiry = expiry_date or self._ib._best_expiry(expiry_month) or ""
        if not resolved_expiry:
            logger.warning("No expiry resolved for SPY {}", expiry_month)
            return None

        # Resolve missing option conids
        async def _resolve(strike: float, right: str) -> None:
            key = (resolved_expiry, strike, right)
            if key in self._conid_map:
                return
            conid = await self._ib.get_option_conid(
                spy_conid, expiry_month, strike, right, cfg_c.exchange,
                expiry_date=resolved_expiry,
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

        call_conids = [self._conid_map[(resolved_expiry, s, "C")] for s in near_calls if (resolved_expiry, s, "C") in self._conid_map]
        put_conids  = [self._conid_map[(resolved_expiry, s, "P")] for s in near_puts  if (resolved_expiry, s, "P") in self._conid_map]
        all_conids  = call_conids + put_conids

        if not all_conids:
            logger.warning("No option conids resolved for SPY {} exp {}", expiry_month, resolved_expiry)
            return None

        # Fetch price + Greeks (snapshot=False + explicit cancel)
        snaps = await self._ib.get_snapshot_with_greeks(all_conids)

        expiry_date = resolved_expiry
        chain = ChainSnapshot(expiry_month, expiry_date=expiry_date)
        filtered_count = 0
        live_count = 0   # contracts with a real two-sided quote (dead-feed detector)

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
            if quote.bid > 0 and quote.ask > 0:
                live_count += 1

            # Apply liquidity filter before adding to chain
            if not passes_liquidity(
                quote,
                min_oi=_liq_min_oi,
                max_spread_pct=_liq_max_spread,
                min_volume=_liq_min_vol,
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

        # Dead/delayed-feed alarm: snapshot returned contracts but NONE has a
        # live two-sided quote → the options feed is down or in IB delayed-data
        # mode (every field reads 0). passes_liquidity now drops these so nothing
        # trades on a phantom, but the operator must KNOW — otherwise the bot
        # looks merely "quiet" while it's actually blind. (audit 2026-07-13)
        if snaps and live_count == 0:
            logger.error(
                "⚠️ DEAD OPTIONS FEED: SPY {} — {} contracts, ZERO live quotes "
                "(all bid/ask=0). Likely IB delayed-data (err 10089) or a feed "
                "outage. No trades can fire until quotes return.",
                expiry_month, len(snaps),
            )
            now_alert = datetime.now(ET)
            last_alert = getattr(self, "_last_dead_feed_alert", None)
            if last_alert is None or (now_alert - last_alert).total_seconds() > 600:
                self._last_dead_feed_alert = now_alert
                try:
                    await self._telegram.send_message(
                        f"⚠️ <b>DEAD OPTIONS FEED</b> — SPY {expiry_month}: "
                        f"{len(snaps)} contracts, zero live quotes. Feed outage "
                        f"or IB delayed-data mode — the bot is blind and no "
                        f"trades will fire. Check the IB Gateway market-data "
                        f"subscription."
                    )
                except Exception as exc:
                    logger.warning("dead-feed alert send failed: {}", exc)

        return chain

    # ── Main poll cycle ───────────────────────────────────────────────────────

    async def _executor_maintenance(self) -> None:
        """Position/exit management: fill detection, entry timeouts, position
        time stops, 0DTE EOD flatten, bracket-exit P&L.

        MUST run on every poll tick — including the early-return paths below —
        because it manages OPEN positions on the executor's own IB connection
        (clientId 7), which is independent of the data feed. Previously it was
        only reached at the end of a fully-successful poll, so:
          • outside the entry window (rth_stop 15:45 < flatten_0dte 15:50) the
            0DTE EOD flatten and time-stops never fired, and
          • a data-feed-only outage (get_spy_price fails → early return) silently
            suspended ALL position risk management while the execution socket
            was perfectly healthy.
        (audit 2026-07-13: P0-6 + P0-7)"""
        if self._executor is not None:
            try:
                await self._executor.on_poll()
            except Exception as exc:
                logger.opt(exception=True).error("Executor on_poll error: {}", exc)

    async def _poll(self, spy_conid: int) -> None:
        self._daily_reset_if_needed()

        if not self._market_open():
            logger.debug("Outside entry window — running position management only")
            await self._executor_maintenance()   # exits/flatten must still run
            return

        # First market-open poll of a new session: re-establish the RealFlow
        # tape/depth streams, which go stale across the overnight boundary.
        if self._feeds_need_resubscribe:
            self._feeds_need_resubscribe = False
            if self._real_flow is not None:
                try:
                    await self._real_flow.resubscribe()
                    logger.info("RealFlow: re-subscribed tape + depth for new session")
                except Exception as exc:
                    logger.warning("RealFlow resubscribe failed: {}", exc)

        spy_price = await self._ib.get_spy_price(spy_conid)
        if not spy_price:
            logger.warning("Could not fetch SPY price — skipping signal generation "
                           "this cycle (position management still runs)")
            await self._executor_maintenance()   # data feed down ≠ stop managing positions
            return

        vix = await self._ib.get_vix()

        # Fallback: if VX futures price is unavailable for any reason,
        # pull from MacroSignals module if available.
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

        # Compute intraday technical levels from bars (ORB, VWAP bands, pivots, EDR, RSI).
        # Feed the authoritative prior-day OHLC (from get_opening_context's real
        # daily bars, cached on self._opening_ctx) so floor-trader pivots actually
        # populate — the trailing 5-min window never contains a full prior session.
        # (audit 2026-07-13, P1-14). Available from the 2nd poll of the day; the
        # opening context for THIS poll is fetched a few lines below.
        _oc = getattr(self, "_opening_ctx", None)
        _prev_ohlc = None
        if _oc and _oc.get("pdh") and _oc.get("pdl") and _oc.get("pdc"):
            _prev_ohlc = (_oc["pdh"], _oc["pdl"], _oc["pdc"])
        tech_levels = self._tech_tracker.update(bars_5m, spy_price, vix, prev_ohlc=_prev_ohlc)
        # Stash per-poll inputs for the exit engine (closed bars only — the
        # forming bar is already excluded by get_spy_bars_5m).
        self._last_bars_5m = bars_5m
        self._last_rsi_5m = float(getattr(tech_levels, "rsi_5m", 50.0) or 50.0)
        self._last_vix = vix

        # Refresh external signals (TTL-gated; most sources won't re-fetch every 60s)
        ext_ctx = None
        if self._external is not None:
            # Feed the flow source the IBKR chain from the PREVIOUS poll (real
            # bid/ask/size → real directional flow_score). yfinance can't supply
            # bid/ask, so this is the working directional input. 1-poll lag is
            # immaterial for flow. Falls back to yfinance if no quotes yet.
            prev_quotes = getattr(self, "_last_ibkr_flow_quotes", None)
            if prev_quotes:
                try:
                    self._external.set_ibkr_flow_quotes(prev_quotes)
                except Exception as exc:
                    logger.warning("IBKR flow injection failed: {}", exc)
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

        # Cache the full context for the trade research log (snapshotted onto
        # each signal at dispatch so closed trades carry every live feature).
        self._last_ext_ctx = ext_ctx

        # ── Real order flow (L2 depth + tape) ──────────────────────────────
        if self._real_flow is not None:
            rf = self._real_flow.snapshot()
            ext_ctx.tape_available   = rf.tape_available
            ext_ctx.tape_score       = rf.tape_score
            ext_ctx.tape_buy_vol     = rf.tape_buy_vol
            ext_ctx.tape_sell_vol    = rf.tape_sell_vol
            ext_ctx.tape_large_bias  = rf.tape_large_bias
            ext_ctx.depth_available  = rf.depth_available
            ext_ctx.depth_imbalance  = rf.depth_imbalance
            ext_ctx.depth_bid_qty    = rf.depth_bid_qty
            ext_ctx.depth_ask_qty    = rf.depth_ask_qty
            if rf.tape_available or rf.depth_available:
                logger.info(
                    "RealFlow: tape={:+.0f} ({}k buy / {}k sell{}){}",
                    rf.tape_score,
                    rf.tape_buy_vol // 1000, rf.tape_sell_vol // 1000,
                    f", blocks={rf.tape_large_bias}" if rf.tape_large_bias != "NEUTRAL" else "",
                    f"  depth={rf.depth_imbalance:+.2f} ({rf.depth_bid_qty}/{rf.depth_ask_qty})"
                    if rf.depth_available else "",
                )

        # ── Cross-asset confirmation (live QQQ/IWM) ─────────────────────────
        if self._cross_asset is not None:
            try:
                ca = await self._cross_asset.snapshot(bars_5m)
            except Exception as exc:
                logger.warning("CrossAsset snapshot failed: {}", exc)
                ca = CrossAssetState()
            ext_ctx.cross_asset_available   = ca.available
            ext_ctx.qqq_rs                  = ca.qqq_rs
            ext_ctx.iwm_rs                  = ca.iwm_rs
            ext_ctx.qqq_trend               = ca.qqq_trend
            ext_ctx.cross_asset_divergence  = ca.cross_asset_divergence
            ext_ctx.cross_asset_bias        = ca.cross_asset_bias
            if ca.available:
                logger.info(
                    "CrossAsset: QQQ_RS={:+.2f}  IWM_RS={:+.2f}  QQQ={}  bias={}{}",
                    ca.qqq_rs, ca.iwm_rs, ca.qqq_trend, ca.cross_asset_bias,
                    f"  ⚠ {ca.cross_asset_divergence}"
                    if ca.cross_asset_divergence != "NONE" else "",
                )

        # ── Opening context (JUL 2 2026, audit item #5) ───────────────────
        # Fetched once per session day from IB: prior-day H/L/C + TRUE
        # overnight range. Overrides the mislabeled sector_signals flags
        # (which were today's RTH high/low) with real overnight levels.
        oc = getattr(self, "_opening_ctx", None)
        oc_day = getattr(self, "_opening_ctx_day", None)
        today_str = datetime.now(ET).strftime("%Y-%m-%d")
        if oc_day != today_str:
            try:
                oc = await self._ib.get_opening_context()
            except Exception as exc:
                logger.warning("Opening context fetch failed: {}", exc)
                oc = None
            self._opening_ctx = oc
            self._opening_ctx_day = today_str
            if oc:
                # Gap vs prior close from today's first RTH bar
                gap_pct, gap_type = 0.0, "NONE"
                if oc.get("pdc") and bars_5m:
                    today_open = bars_5m[0].get("open") or spy_price
                    gap_pct = (today_open - oc["pdc"]) / oc["pdc"] * 100.0
                    gap_type = (
                        "GAP_UP" if gap_pct >= 0.30
                        else "GAP_DOWN" if gap_pct <= -0.30
                        else "FLAT"
                    )
                oc["gap_pct"], oc["gap_type"] = round(gap_pct, 2), gap_type
                logger.info(
                    "OPENING CONTEXT: PDH={} PDL={} PDC={}  ON_H={} ON_L={}  "
                    "gap={:+.2f}% ({})",
                    *(f"{oc[k]:.2f}" if oc.get(k) else "n/a"
                      for k in ("pdh", "pdl", "pdc", "overnight_high", "overnight_low")),
                    oc["gap_pct"], oc["gap_type"],
                )
        if oc:
            ext_ctx.pdh            = oc.get("pdh")
            ext_ctx.pdl            = oc.get("pdl")
            ext_ctx.pdc            = oc.get("pdc")
            ext_ctx.overnight_high = oc.get("overnight_high")
            ext_ctx.overnight_low  = oc.get("overnight_low")
            ext_ctx.gap_pct        = oc.get("gap_pct", 0.0)
            ext_ctx.gap_type       = oc.get("gap_type", "NONE")
            # Correct the legacy flags with TRUE overnight levels
            if oc.get("overnight_high"):
                ext_ctx.above_overnight_high = spy_price >= oc["overnight_high"] * 0.9995
            if oc.get("overnight_low"):
                ext_ctx.below_overnight_low = spy_price <= oc["overnight_low"] * 1.0005

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
        # ── Flow-feed health alarm (Tier-1 fix, MAY 30 2026) ──
        # The CBOE flow feed returning HTTP 403 leaves flow_score pinned at 0
        # for the entire session with no operator-visible error.  flow_score
        # contributes up to +0.10 confidence per directional signal, so a
        # silent zero degrades every signal.  Alarm when flow has been exactly
        # 0 for many consecutive polls so a dead feed is surfaced, not silently
        # absorbed.  Pure observability — no effect on signal/trade logic.
        if ext_ctx is not None:
            if abs(ext_ctx.flow_score) < 1e-9:
                self._flow_zero_streak = getattr(self, "_flow_zero_streak", 0) + 1
                if self._flow_zero_streak == 10 or self._flow_zero_streak % 60 == 0:
                    logger.error(
                        "⚠️ FLOW FEED HEALTH: flow_score == 0 for {} consecutive "
                        "polls — CBOE/flow source likely dead (HTTP 403). "
                        "Directional signals are losing their flow-alignment "
                        "boost (up to +0.10 conf). Verify flow feed "
                        "credentials/endpoint.",
                        self._flow_zero_streak,
                    )
            else:
                if getattr(self, "_flow_zero_streak", 0) >= 10:
                    logger.info(
                        "✅ FLOW FEED HEALTH: flow_score recovered "
                        "(was 0 for {} polls)",
                        self._flow_zero_streak,
                    )
                self._flow_zero_streak = 0
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

        # True session high/low from today's bars — restart-proof inputs for
        # the engine's chop-day gate and move-exhaustion penalty (JUL 2 2026).
        _day_high = max((b["high"] for b in bars_5m), default=None) if bars_5m else None
        _day_low = min((b["low"] for b in bars_5m), default=None) if bars_5m else None

        ctx = SignalContext(
            regime=regime_ctx,
            sentiment=sentiment_ctx,
            iv_rank=iv_rank,
            vix=vix,
            spy_price=spy_price,
            external=ext_ctx,
            day_high=_day_high,
            day_low=_day_low,
        )

        now_et = datetime.now(ET)
        expiry_months = [
            _ib_month(now_et.replace(day=1) + timedelta(days=32 * i))
            for i in range(self._cfg.chain.num_expiries)
        ]

        all_signals: List[SpySignal] = []
        max_pain_computed = False   # compute once from the first available chain
        ibkr_flow_quotes: List[dict] = []   # collected for next poll's flow source
        chains_by_expiry: Dict[str, ChainSnapshot] = {}   # retained for continuation enrichment
        for expiry in expiry_months:
            chain = await self._build_chain(spy_conid, spy_price, expiry)
            if chain:
                chains_by_expiry[expiry] = chain
                # Snapshot quotes (real bid/ask/size) for the flow source.
                try:
                    _edt = datetime.strptime(chain.expiry_date, "%Y%m%d").date()
                    _dte = max(0, (_edt - datetime.now().date()).days)
                except Exception:
                    _dte = 7
                for _q in list(chain.calls) + list(chain.puts):
                    ibkr_flow_quotes.append({
                        "strike": _q.strike, "right": _q.right,
                        "expiry": chain.expiry_month,
                        "bid": _q.bid, "ask": _q.ask,
                        "bid_size": _q.bid_size, "ask_size": _q.ask_size,
                        "volume": _q.volume, "open_interest": _q.open_interest,
                        "delta": _q.delta, "gamma": _q.gamma, "dte": _dte,
                    })
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
                # Shadow book: engine-level kills (quality gate / threshold).
                for _bsig, _bgate in getattr(self._engine, "last_blocked", []):
                    self._register_shadow(_bsig, _bgate)
            else:
                logger.info("Chain {}: empty (all options filtered out or no conids)", expiry)

        # Stash this poll's chain quotes; next poll feeds them to the flow source.
        self._last_ibkr_flow_quotes = ibkr_flow_quotes

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

        # Enrich TREND_CONTINUATION signals with a real, delta-selected option
        # quote. The generator only produces a trigger strike + structural stop;
        # without bid/ask/greeks the executor rejects them at "no live bid/ask".
        # Continuation is a minutes-to-hours swing hold, so prefer a ≥1DTE
        # expiry (theta protection) — built lazily and reused across signals,
        # falling back to the already-built 0DTE chain if 1DTE isn't listed.
        _swing_chain = None           # lazily resolved swing chain (≥1DTE, theta-friendly)
        _swing_resolved = False
        # The already-built 0DTE (nearest) chain — liquid fallback when the thin
        # swing chain has no enrichable strike.
        _zero_chain = (
            chains_by_expiry.get(expiry_months[0]) if expiry_months else None
        ) or (next(iter(chains_by_expiry.values()), None))
        for sig in all_signals:
            if sig.signal_type == SignalType.TREND_CONTINUATION and not (sig.bid and sig.ask):
                if not _swing_resolved:
                    _swing_resolved = True
                    min_dte = getattr(self._cfg.chain, "continuation_min_dte", 0)
                    if min_dte >= 1:
                        swing_exp = self._ib.resolve_expiry_min_dte(min_dte)
                        already = {c.expiry_date for c in chains_by_expiry.values()}
                        if swing_exp and swing_exp not in already:
                            try:
                                # Relaxed liquidity so the ATM ~0.50Δ swing
                                # strikes (lower volume 2-3 DTE out) survive to
                                # be enriched; executor gates make the final call.
                                _swing_chain = await self._build_chain(
                                    spy_conid, spy_price,
                                    expiry_months[0] if expiry_months else "",
                                    expiry_date=swing_exp,
                                    liquidity_override={
                                        "min_oi": 100, "min_volume": 0, "max_spread_pct": 15.0,
                                    },
                                )
                                if _swing_chain:
                                    logger.info(
                                        "Continuation swing chain built: exp {} "
                                        "({} calls / {} puts)",
                                        swing_exp, len(_swing_chain.calls), len(_swing_chain.puts),
                                    )
                            except Exception as exc:
                                logger.warning("Swing chain build failed, using 0DTE: {}", exc)
                                _swing_chain = None
                # Prefer the theta-friendly swing chain; if it has no liquid
                # enrichable strike, fall back to the liquid 0DTE chain so a
                # valid signal still reaches the executor (better a 0DTE trade
                # the theta gate can judge than a dropped signal). Try both.
                if _swing_chain is not None:
                    self._enrich_continuation_quote(sig, _swing_chain)
                if not (sig.bid and sig.ask) and _zero_chain is not None and _zero_chain is not _swing_chain:
                    self._enrich_continuation_quote(sig, _zero_chain)

        await self._dispatch_signals(all_signals)

        # Check whether any previously-sent signals now warrant an EXIT alert
        await self._check_exit_conditions(spy_price, regime_ctx)

        # Resolve shadow-book (blocked-signal) simulated exits.
        self._simulate_shadow_exits(spy_price)

        # Executor maintenance: fill detection, entry timeouts, position time
        # stops, 0DTE EOD flatten, bracket-exit P&L accounting. Same helper is
        # invoked on the early-return paths above so exits never stall.
        await self._executor_maintenance()

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

        # Observability (JUL 2 2026): the v2 classifier requires ATR expansion
        # for a trend call, so it drops to TRANSITION on grind days while the
        # legacy detector still says TREND_* — which silently disables the
        # continuation generator. Log v2 regime changes so the divergence is
        # visible instead of inferred.
        _prev_v2 = getattr(self, "_last_v2_regime", None)
        if regime.regime != _prev_v2:
            logger.info(
                "rules_v2 regime: {} → {}  ({})",
                _prev_v2 or "—", regime.regime,
                "; ".join(regime.reasons[:2]) if regime.reasons else "no reasons",
            )
            self._last_v2_regime = regime.regime

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
                log_blocked_signal(sig, f"rules_v2:{decision.rule}", decision.reason)
                self._register_shadow(sig, f"rules_v2:{decision.rule}")
                logger.info(
                    "rules_v2 BLOCK {} {} @{:.2f}  rule={}  reason={}",
                    sig.signal_type.value, sig.right, spy_price,
                    decision.rule, decision.reason,
                )

        # ── Generate TREND_CONTINUATION candidates (not emitted by legacy) ─
        _continuation_cands = engine.generate_additional(bars_5m, regime, spy_price, now=now)
        # Observability (JUL 2 2026): in a confirmed trend with no candidate,
        # say WHY — the generator was silent for a -1.25% trend day and the
        # logs couldn't distinguish "wrong v2 regime" from "no pullback setup".
        if not _continuation_cands and regime.regime in ("TREND_UP", "TREND_DOWN"):
            logger.info(
                "rules_v2 CONTINUATION none ({}): {}",
                regime.regime, engine.continuation_skip_reason or "unknown",
            )
        for cand in _continuation_cands:
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
                    # BUG FIX (JUL 7 2026): tier was never computed → defaulted to
                    # "MEDIUM", so the executor (HIGH/EXTREME only) rejected EVERY
                    # continuation signal regardless of confidence. Compute it.
                    confidence_tier=_tier(cand.confidence),
                    # BUG FIX (JUL 7 2026): regime was never set → defaulted to
                    # "RANGE_BOUND", so the Trading Manager vetoed every
                    # continuation ("TREND_CONTINUATION in RANGE_BOUND — regime
                    # fit failed") pre-dispatch. Carry the v2 trend regime that
                    # actually generated the signal.
                    regime=regime.regime,
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
                    # Structural invalidation in SPY terms — the executor turns
                    # this into the option premium stop/target via delta (1.5R).
                    structural_stop=cand.stop_price,
                )
            )
            # ── TC×DynamicConfidence SIDE-BY-SIDE (2026-07-20, log-only) ──
            # Validation phase for wiring TC into the confidence pipeline.
            # Frozen-model replay over 20 decided TC trades: flat-0.82 book
            # PF 0.72 vs dyn-filtered PF 1.39 @0.77 (0 winners lost, 9/9
            # filtered were losers, p≈0.07). This logs what dyn WOULD say per
            # live TC signal — behavior unchanged until side-by-side confirms.
            try:
                _ext = getattr(self, "_last_ext_ctx", None)
                _adj = self._engine._dyn.adjust(
                    base=cand.confidence, right=cand.direction, dte=2,
                    ext_ctx=_ext, ib_sentiment_score=0.0, vix=vix,
                    regime=regime.regime,
                )
                _top = sorted(_adj.breakdown.items(), key=lambda kv: -abs(kv[1]))[:4]
                logger.info(
                    "🔬 TC-DYN side-by-side {} @{:.2f}: flat={:.2f} dyn={:.2f} "
                    "(Δ{:+.2f}) [{}] → dyn verdict: {}",
                    cand.direction, spy_price, cand.confidence, _adj.final,
                    _adj.final - cand.confidence,
                    " ".join(f"{k}:{v:+.2f}" for k, v in _top if k != "base"),
                    ("DISPATCH" if _adj.final >= 0.77 else "FILTER<0.77")
                    + ("/HIGH" if _adj.final >= 0.80 else "/tier-fail"),
                )
            except Exception as _sx:
                logger.debug("TC-DYN side-by-side failed: {}", _sx)
            logger.info(
                "rules_v2 CONTINUATION ALLOW {} @{:.2f}  trigger={:.2f} stop={:.2f}",
                cand.direction, spy_price, cand.trigger_price, cand.stop_price,
            )

        return kept

    def _enrich_continuation_quote(
        self, sig: SpySignal, chain: ChainSnapshot, target_delta: float = 0.48
    ) -> None:
        """Attach a real delta-selected option quote to a continuation signal.

        Picks the option on the signal's side whose |delta| is closest to
        target_delta within [0.30, 0.65] (directional-swing sweet spot), and
        copies its strike, quote, and greeks onto the signal so the executor
        can gate, size, and bracket it. Leaves the signal unquoted (bid/ask 0 →
        executor skips it) if no suitable liquid strike exists.
        """
        quotes = chain.calls if sig.right == "C" else chain.puts
        # Delta band matches the executor's gate [0.30, 0.70] (was [0.30, 0.65])
        # so a slightly-ITM swing strike is enrichable. Pick |Δ| closest to
        # target among liquid strikes.
        best = None
        best_err = 1e9
        for q in quotes:
            d = abs(q.delta or 0.0)
            if d < 0.30 or d > 0.70:
                continue
            if not (q.bid and q.ask and q.bid > 0 and q.ask > 0):
                continue
            err = abs(d - target_delta)
            if err < best_err:
                best_err, best = err, q
        if best is None:
            logger.info(
                "CONTINUATION {}{}: no liquid 0.30–0.70Δ strike to enrich — stays alert-only",
                sig.strike, sig.right,
            )
            return
        sig.strike = best.strike
        sig.bid = best.bid
        sig.ask = best.ask
        sig.bid_size = best.bid_size
        sig.ask_size = best.ask_size
        sig.delta = best.delta
        sig.gamma = best.gamma
        sig.theta = best.theta
        sig.vega = best.vega
        sig.impl_vol = best.impl_vol
        sig.open_interest = best.open_interest
        sig.spread_pct = best.spread_pct
        sig.volume = best.volume
        sig.expiry_date = chain.expiry_date
        if chain.expiry_date:
            try:
                exp_dt = datetime.strptime(chain.expiry_date, "%Y%m%d").date()
                sig.dte = max(0, (exp_dt - datetime.now().date()).days)
            except ValueError:
                pass
        logger.info(
            "CONTINUATION enriched: {}{} Δ={:.2f} bid/ask={:.2f}/{:.2f} dte={} "
            "(structural_stop=${:.2f})",
            sig.strike, sig.right, sig.delta, sig.bid, sig.ask, sig.dte,
            sig.structural_stop,
        )

    async def _dispatch_signals(self, signals: List[SpySignal]) -> None:
        dedup_td  = timedelta(minutes=self._cfg.signals.dedup_window_minutes)
        flip_td   = timedelta(minutes=self._cfg.signals.pc_ratio_flip_cooldown_minutes)
        daily_cap = self._cfg.signals.max_signals_per_day
        now = datetime.utcnow()
        seen_keys: set[str] = set()          # batch-level dedup (cross-expiry)

        def _v2_rollback(s: SpySignal) -> None:
            # JUL 2 2026: rules_v2 commits at filter time, so any signal
            # suppressed AFTER that point but BEFORE dispatch must be rolled
            # back — otherwise phantom entries consume the leg cap and
            # zone-lock real setups (live bug: TM-vetoed 10:21/10:25
            # continuation puts blocked the 744-745 zone on a trend day).
            if self._rules_v2 is not None and s.right in ("C", "P"):
                try:
                    if self._rules_v2.uncommit(s.right):
                        logger.info(
                            "rules_v2 ROLLBACK: uncommitted {} {} (suppressed pre-dispatch)",
                            s.signal_type.value, s.right,
                        )
                except Exception as _rb_err:
                    logger.warning("rules_v2 rollback error: {}", _rb_err)

        for sig in signals:
            key = sig.dedup_key
            # ── Non-directional structures never trade — stop burning slots ──
            # LONG_STRADDLE (right='BOTH'): 51 dispatched all-time, 0 tradeable
            # (executor structurally rejects right∉{C,P}), 0 decided shadow
            # outcomes — pure noise consuming the max_signals_per_day cap (9 of
            # the last 2 weeks' budget). Alert-only value can be re-enabled via
            # signals.dispatch_non_directional. (strategy audit 2026-07-17)
            if sig.right not in ("C", "P") and not getattr(
                self._cfg.signals, "dispatch_non_directional", False
            ):
                log_blocked_signal(sig, "non_directional_skip",
                                   "right=BOTH untradeable — dispatch disabled")
                continue
            # ── Daily signal cap ──────────────────────────────────────────────
            if self._daily_signal_count >= daily_cap:
                logger.info(
                    "Daily signal cap reached ({}/{}) — suppressing {}",
                    self._daily_signal_count, daily_cap, key,
                )
                _v2_rollback(sig)
                break   # no more signals today
            # ── Batch dedup: same key already dispatched this cycle ───────────
            if key in seen_keys:
                logger.debug("Batch dedup suppress (cross-expiry): {}", key)
                _v2_rollback(sig)
                continue
            last_sent = self._sent_times.get(key)
            if last_sent and (now - last_sent) < dedup_td:
                logger.debug("Dedup suppress: {}", key)
                _v2_rollback(sig)
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
                    _v2_rollback(sig)
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
                _v2_rollback(sig)
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
                    _v2_rollback(sig)
                    continue
            except Exception as _tm_err:
                # The veto hook itself errored. Don't let its own bug block
                # trading in general — but still FAIL CLOSED on a hard kill-switch
                # read from the TM state file. (audit 2026-07-13, P0-5)
                _local = self._tm_local_kill_check()
                if _local is not None:
                    logger.error(
                        "🛡️  TM veto hook error AND local kill-switch ACTIVE — "
                        "skipping dispatch: {} | {}", _local, _tm_err,
                    )
                    _v2_rollback(sig)
                    continue
                logger.error("Trading Manager SPY veto hook error (fail-open): {}", _tm_err)

            await self._send_signal(sig)
            self._sent_times[key] = now
            seen_keys.add(key)
            self._daily_signal_count += 1

            # ── Order execution (JUL 2 2026) ──────────────────────────────
            # Strict quality gate lives inside the executor; a rejected
            # signal stays Telegram-only. Never let an execution bug break
            # the signal feed.
            if self._executor is not None:
                # Snapshot the full external context onto the signal so a
                # closed trade carries every live feature for research.
                try:
                    ec = getattr(self, "_last_ext_ctx", None)
                    if ec is not None:
                        from dataclasses import asdict as _asdict
                        sig.research_ctx = _asdict(ec)
                except Exception:
                    pass
                # Trading-session DTE for the executor's max_dte gate — a
                # Thu→Mon contract is 4 calendar but only 2 TRADING days out;
                # the calendar count made the gate reject every Thursday swing
                # entry (2026-07-16: "DTE 4 > max 3" ×2). Weekend/holiday-
                # invariant by construction.
                sig.trading_dte = self._trading_dte(sig.expiry_date)
                try:
                    await self._executor.maybe_execute(sig)
                except Exception as exc:
                    logger.opt(exception=True).error(
                        "Executor error for {} (signal feed unaffected): {}",
                        key, exc,
                    )

            # Record directional send time for throttle
            if sig.right in ("C", "P"):
                self._dir_signal_times[sig.right].append(now)

            if sig.signal_type == SignalType.PC_RATIO_EXTREME:
                self._last_pc_ratio_direction = sig.right
                self._last_pc_ratio_sent = now
            if self._analytics:
                self._analytics.insert(sig)

            # V2 directional-composite gate — SHADOW ONLY (2026-08-03).
            # Logs what the frozen Variant-2 gate WOULD have decided. It never
            # gates, sizes, prices, or blocks anything — pure observation, and
            # it swallows all exceptions. Hooked here (immediately after the
            # spy_signals insert) so the shadow population matches the study
            # population by construction. See docs/V2_SHADOW_GATE.md.
            if sig.signal_type == SignalType.CALL_SWEEP:
                v2_shadow_gate.record(sig)

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
                    "entry_rsi": self._last_rsi_5m,
                    "sent_at": now,
                }

    # ── Empirical WR cache (JUL 2 2026, audit item #1) ───────────────────────

    def _empirical_wr_for(self, signal_type: str) -> Optional[Dict]:
        """Cached empirical win rate from REAL fills, or None (use prior).

        Cache is rebuilt lazily once per session day; real fills accumulate
        at a few per day at most, so intraday staleness is immaterial.
        """
        if self._analytics is None:
            return None
        cache_day = getattr(self, "_emp_wr_cache_day", None)
        today = datetime.now(ET).strftime("%Y-%m-%d")
        if cache_day != today:
            self._emp_wr_cache: Dict[str, Optional[Dict]] = {}
            self._emp_wr_cache_day = today
        if signal_type not in self._emp_wr_cache:
            try:
                res = self._analytics.empirical_win_rate(signal_type, min_n=30)
            except Exception as exc:
                logger.warning("empirical_win_rate lookup failed: {}", exc)
                res = None
            self._emp_wr_cache[signal_type] = res
            if res is not None:
                logger.info(
                    "Edge Reality: EMPIRICAL WR active for {} — {}% over {} "
                    "real fills (avg ${:+.2f}/trade). Doc prior retired.",
                    signal_type, res["wr"], res["n"], res["avg_pnl_usd"],
                )
        return self._emp_wr_cache[signal_type]

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
        # JUL 2 2026 (audit item #1): the empirical WR loop is now LIVE.
        # Once analytics_db holds ≥30 REAL executor-fill outcomes for a
        # signal type, its measured win rate replaces the doc prior and
        # wr_source flips to "empirical". Cached per-day (refreshed on the
        # daily reset) — fills accrue slowly, no need to re-query per signal.
        # Judge the edge against the target the EXECUTOR will actually aim for,
        # not a regime guess disconnected from the trade. The old
        # default_target_pct(regime) (30–50%) rarely matched the executor's real
        # take-profit, so breakeven_wr/edge_margin scored a trade that never
        # happens. (audit 2026-07-13, P1-12)
        _exc = getattr(self._cfg, "execution", None)
        if _exc is None:
            target = default_target_pct(regime)
        elif (getattr(_exc, "use_structural_bracket", False)
              and getattr(sig, "structural_stop", 0.0)
              and signal_type_str == "TREND_CONTINUATION"):
            # Structural bracket: TP = continuation_target_r × the premium stop.
            _stop = float(sig.iv_adjusted_stop_pct or _exc.stop_pct_fallback)
            target = _stop * float(_exc.continuation_target_r)
        else:
            target = float(_exc.take_profit_pct)   # premium bracket default
        sig.target_pct_assumed = target

        _emp_wr: Optional[int] = None
        _wr_src = "doc_prior"
        emp = self._empirical_wr_for(signal_type_str)
        if emp is not None:
            _emp_wr = emp["wr"]
            _wr_src = "empirical"

        edge = compute_edge_reality(
            signal_type=signal_type_str,
            regime=regime,
            direction=right,
            target_pct=target,
            stop_pct=sig.iv_adjusted_stop_pct,
            round_trip_pct=sig.round_trip_cost_pct,
            rt_source=sig.rt_source,
            base_wr=_emp_wr,
            wr_source=_wr_src,
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

            # ── Trigger 5 (checked FIRST — JUL 2 2026 profitability fix):
            # profit target hit. Apr-Jun data: profit_target fired once in
            # 44 resolved exits because time_stop/vwap labels always won the
            # "first reason" slot. Winners must be labelled as winners so
            # analytics can measure the profit side.
            _profit_target_pct = float(
                getattr(self._cfg.signals, "exit_profit_target_pct", 0.5)
            )
            if direction == "BULLISH" and price_chg_pct >= _profit_target_pct:
                reasons.append(
                    f"✅ Profit target: SPY +{price_chg_pct:.2f}% since entry "
                    f"(${entry_price:.2f} → ${spy_price:.2f}) — consider taking profits"
                )
            elif direction == "BEARISH" and price_chg_pct <= -_profit_target_pct:
                reasons.append(
                    f"✅ Profit target: SPY {price_chg_pct:.2f}% since entry "
                    f"(${entry_price:.2f} → ${spy_price:.2f}) — consider taking profits"
                )

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
            # Swing setups (1+ DTE) allow a longer leash before staleness
            # forces exit. Configurable since JUL 2 2026 — the hard-coded
            # 30/60 min stops were cutting positions before targets while
            # theta was already paid (Apr-Jun: ~all time_stop exits negative).
            minutes_held = (now - entry["sent_at"]).total_seconds() / 60.0
            time_stop_min = (
                getattr(self._cfg.signals, "exit_time_stop_0dte_min", 45)
                if entry_dte == 0
                else getattr(self._cfg.signals, "exit_time_stop_swing_min", 90)
            )
            if minutes_held >= time_stop_min:
                reasons.append(
                    f"⏱ Time stop: held {minutes_held:.0f} min "
                    f"(limit {time_stop_min} min for {'0DTE' if entry_dte == 0 else 'swing'})"
                )

            # (Trigger 5 — profit target — is evaluated FIRST, above Trigger 1,
            #  so winning exits are labelled profit_target in analytics.)

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

            # ── Exit Engine v2 (JUL 14 2026) ──────────────────────────────
            # Shadow mode: evaluate + log next to the legacy decision, then
            # let the legacy triggers act. Active mode: the engine decision
            # REPLACES the legacy triggers for this signal entirely.
            ee_dec: Optional[ExitDecision] = None
            if self._exit_engine is not None:
                snap = self._build_exit_snapshot(
                    key, entry, sig, direction, spy_price, entry_price,
                    price_chg_pct, minutes_held, regime_ctx, ext_ctx,
                )
                if snap is not None:
                    st = self._exit_states.get(key)
                    if st is None:
                        st = PositionExitState()
                        self._exit_states[key] = st
                    ee_dec = self._exit_engine.evaluate(st, snap)
                    _shadow = getattr(self._cfg.exit_engine, "shadow_mode", True)
                    if ee_dec.action != HOLD or reasons or ee_dec.score >= 25:
                        logger.info(
                            "🧭 ExitEngine[{}] {}: {} score={}/{} stage={} "
                            "[{}] — {}{}",
                            "SHADOW" if _shadow else "ACTIVE", key,
                            ee_dec.action, ee_dec.score, ee_dec.threshold,
                            ee_dec.stage, ee_dec.factors_str(), ee_dec.reason,
                            (f" | legacy={'EXIT: ' + reasons[0][:60] if reasons else 'hold'}"
                             if _shadow else ""),
                        )
                    if not _shadow:
                        await self._apply_exit_decision(
                            key, sig, entry, ee_dec, spy_price, entry_price,
                            regime_ctx,
                        )
                        continue   # legacy triggers fully replaced

            if reasons:
                logger.info(
                    "EXIT trigger for {}: {}", key, " | ".join(reasons),
                )
                await self._send_exit_alert(sig, spy_price, entry_price, reasons, regime_ctx.regime)
                self._exit_sent.add(key)

                # If the executor holds this position, close it now — the
                # bot-managed exit fires earlier than the resting bracket
                # would (regime flips, VWAP reversion, time stops). The IB
                # bracket remains the fail-safe if this path errors.
                if self._executor is not None and self._executor.has_open_position(key):
                    try:
                        await self._executor.close_position(
                            key, reasons[0][:120]
                        )
                    except Exception as exc:
                        logger.opt(exception=True).error(
                            "Executor close on exit-trigger failed for {}: {}",
                            key, exc,
                        )

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
            self._exit_states.pop(k, None)

    # ── Exit Engine v2 adapter ────────────────────────────────────────────────

    def _build_exit_snapshot(
        self,
        key: str,
        entry: Dict,
        sig: SpySignal,
        direction: str,
        spy_price: float,
        entry_price: float,
        price_chg_pct: float,
        minutes_held: float,
        regime_ctx: RegimeContext,
        ext_ctx,
    ) -> Optional[ExitSnapshot]:
        """Assemble the engine's inputs from per-poll data. Returns None when
        essential inputs (closed bars / regime indicators) are unavailable —
        the engine then simply doesn't evaluate this poll."""
        bars = self._last_bars_5m
        if not bars or not regime_ctx or regime_ctx.ema9 <= 0:
            return None
        last_bar = bars[-1]
        bull = direction == "BULLISH"

        favorable_pct = price_chg_pct if bull else -price_chg_pct
        spy_adverse_pct = max(0.0, -favorable_pct)

        # Premium economics: executor fill when held, else signal quote.
        entry_mid = 0.0
        stop_pct = float(sig.iv_adjusted_stop_pct or 0.0)
        pos = None
        if self._executor is not None:
            pos = self._executor._positions.get(key)  # noqa: SLF001 — same package
        if pos is not None and not pos.closed and pos.entry_filled:
            entry_mid = float(pos.parent.orderStatus.avgFillPrice or pos.entry_mid)
            stop_pct = float(pos.stop_pct or stop_pct)
        elif sig.bid and sig.ask:
            entry_mid = (sig.bid + sig.ask) / 2.0

        unrealized_r: Optional[float] = None
        premium_loss_pct_est = 0.0
        if entry_mid > 0 and sig.delta:
            spy_move = spy_price - entry_price
            est_chg = abs(sig.delta) * (spy_move if bull else -spy_move)
            if stop_pct > 0:
                unrealized_r = est_chg / (entry_mid * stop_pct / 100.0)
            if est_chg < 0:
                gam = max(1.0, float(sig.gamma_accel_mult or 1.0))
                premium_loss_pct_est = -est_chg / entry_mid * 100.0 * gam

        now_et = datetime.now(ET)
        time_stop_min = (
            getattr(self._cfg.signals, "exit_time_stop_0dte_min", 45)
            if entry.get("entry_dte", 1) == 0
            else getattr(self._cfg.signals, "exit_time_stop_swing_min", 90)
        )
        tape = None
        if ext_ctx is not None and getattr(ext_ctx, "tape_available", False):
            tape = getattr(ext_ctx, "tape_score", None)

        bar_ts = last_bar.get("date")
        return ExitSnapshot(
            direction=direction,
            spy_price=spy_price,
            entry_spy=entry_price,
            last_bar_ts=str(bar_ts),
            last_close=float(last_bar.get("close", spy_price)),
            last_bar_range=float(last_bar.get("high", 0.0)) - float(last_bar.get("low", 0.0)),
            vwap=float(regime_ctx.vwap or 0.0),
            ema9=float(regime_ctx.ema9 or 0.0),
            ema21=float(regime_ctx.ema21 or 0.0),
            ema9_slope=float(regime_ctx.ema_slope or 0.0),
            rsi_5m=self._last_rsi_5m,
            atr14=float(regime_ctx.atr14 or 0.0),
            vwap_band=(
                getattr(ext_ctx, "vwap_band_position", "INSIDE_1SD")
                if ext_ctx is not None else "INSIDE_1SD"
            ),
            regime=regime_ctx.regime,
            minutes_held=minutes_held,
            max_hold_min=float(time_stop_min),
            dte=int(entry.get("entry_dte", 1)),
            is_late_0dte=(entry.get("entry_dte", 1) == 0 and now_et.time() >= time(14, 0)),
            vix=self._last_vix,
            tape_score=tape,
            delta_now=None,   # live per-position greeks not refreshed; factor inert
            unrealized_r=unrealized_r,
            premium_loss_pct_est=premium_loss_pct_est,
            spy_adverse_pct=spy_adverse_pct,
            entry_vwap_band=entry.get("entry_vwap_band", "INSIDE_1SD"),
            entry_regime=entry.get("entry_regime", ""),
            entry_tier=str(getattr(sig, "confidence_tier", "") or ""),
            entry_confidence=float(getattr(sig, "confidence", 0.0) or 0.0),
            entry_rsi=entry.get("entry_rsi"),
            entry_delta=float(sig.delta or 0.0) or None,
            iv_stop_pct=stop_pct,
        )

    async def _apply_exit_decision(
        self,
        key: str,
        sig: SpySignal,
        entry: Dict,
        dec: ExitDecision,
        spy_price: float,
        entry_price: float,
        regime_ctx: RegimeContext,
    ) -> None:
        """ACTIVE mode: execute the engine's decision with the same
        bookkeeping the legacy trigger block performs (alert, _exit_sent,
        executor close, analytics outcome)."""
        held = (
            self._executor is not None
            and self._executor.has_open_position(key)
        )
        if dec.action == HOLD:
            return

        if dec.action in (PARTIAL_EXIT, TRAIL_STOP):
            if not held:
                return   # alert-only signal: ladder actions are meaningless
            pos = self._executor._positions.get(key)  # noqa: SLF001
            entry_mid = float(pos.parent.orderStatus.avgFillPrice or pos.entry_mid)
            stop_pct = float(pos.stop_pct or sig.iv_adjusted_stop_pct or 15.0)
            if dec.new_stop_r is not None:
                new_stop = entry_mid * (1 + dec.new_stop_r * stop_pct / 100.0)
                await self._executor.move_stop(key, new_stop)
            if dec.action == PARTIAL_EXIT and dec.fraction > 0:
                await self._executor.partial_close(key, dec.fraction, dec.reason)
            return

        # FULL_EXIT — mirror the legacy action block.
        reason_line = (
            f"🧭 ExitEngine: {dec.reason} "
            f"(score {dec.score}/{dec.threshold}: {dec.factors_str()})"
        )
        logger.info("EXIT (engine) for {}: {}", key, reason_line)
        await self._send_exit_alert(
            sig, spy_price, entry_price, [reason_line], regime_ctx.regime
        )
        self._exit_sent.add(key)
        if held:
            try:
                await self._executor.close_position(key, dec.reason[:120])
            except Exception as exc:
                logger.opt(exception=True).error(
                    "Executor close on engine exit failed for {}: {}", key, exc,
                )
        if self._analytics:
            _r = dec.reason
            if "max hold" in _r:
                label = "time_stop"
            elif "catastrophic" in _r:
                label = "adverse_move"
            else:
                label = "exit_confidence"
            # Same win/loss/scratch classification as the legacy block.
            direction = self._signal_direction(sig)
            raw_pct = (spy_price - entry_price) / entry_price * 100.0
            fav_pct = raw_pct if direction == "BULLISH" else (
                -raw_pct if direction == "BEARISH" else 0.0
            )
            outcome_label = (
                "win" if fav_pct > 0.1 else "loss" if fav_pct < -0.1 else "scratch"
            )
            try:
                db_id = self._analytics.find_signal_id(sig)
                if db_id is not None:
                    self._analytics.record_outcome(
                        signal_id=db_id,
                        outcome=outcome_label,
                        spy_price_exit=spy_price,
                        exit_trigger=label,
                        direction=direction,
                        entry_price=entry_price,
                    )
            except Exception as exc:
                logger.warning("Engine exit record_outcome failed: {}", exc)

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

    def _tm_local_kill_check(self) -> Optional[str]:
        """Fallback kill-switch read DIRECTLY from the TM state file, for when
        the TM daemon's per-signal verdict never lands (daemon down/slow/crashed)
        or the veto hook itself errors. Returns a rejection reason if a HARD stop
        is active for the SPY track, else None.

        This makes the veto FAIL CLOSED on the safety-critical states
        (POSTURE_KILLED / LOCKED / SPY daily-loss breach) instead of blindly
        proceeding — the prior behaviour disabled every multi-day/daily
        kill-switch exactly when the daemon was most likely to be down.
        It does NOT halt on a merely-slow daemon when no hard stop is active:
        the executor's own local guards (daily-loss, stopout, max-positions)
        remain the second layer. (audit 2026-07-13, P0-5)"""
        try:
            import json as _json
            import os as _os
            from ..trading_manager.config import CONFIG as _TMCFG
            path = _TMCFG.state_file
            if not path or not _os.path.exists(path):
                return None
            with open(path, "r") as fh:
                st = _json.load(fh)
        except Exception:
            return None  # can't read state → don't fabricate a block
        posture = str(st.get("spy_posture") or st.get("posture") or "").upper()
        if posture in ("KILLED", "LOCKED"):
            return f"TM state: SPY posture={posture} (daemon silent → fail-closed)"
        try:
            spy_pnl = float(st.get("spy_realized_pnl_today", 0.0) or 0.0)
            hard = float(getattr(_TMCFG, "daily_loss_hard_dollars", 0.0) or 0.0)
            if hard > 0 and spy_pnl <= -hard:
                return (f"TM state: SPY daily loss {spy_pnl:+.0f} <= -{hard:.0f} "
                        f"(daemon silent → fail-closed)")
        except Exception:
            pass
        return None

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
            local = self._tm_local_kill_check()
            if local is not None:
                logger.error(
                    "🛡️  TM verdict missing for {} AND local kill-switch ACTIVE — "
                    "REJECT: {}", signal_id, local,
                )
                return False
            logger.warning(
                "⚠️  Trading Manager decision not found for {} — daemon may be down. "
                "No hard kill-switch active in the TM state file; proceeding under "
                "the executor's own local risk guards. Verify the TM daemon.",
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
