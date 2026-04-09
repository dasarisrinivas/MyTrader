"""
ExternalDataManager — aggregates all external signal sources into a single
ExternalContext consumed by the signal engine and DynamicConfidence.

Sources and default weights (normalised dynamically if source unavailable):
  News RSS sentiment         0.20
  StockTwits retail          0.15
  Macro headwind             0.20
  CBOE P/C contrarian        0.15
  External flow score        0.20  (options flow, GEX, dark pool)
  Reddit enhanced            0.10  (opt-in)

Additional context sources (not weighted — used for conf adjustments only):
  Market breadth   (BreadthSignals)
  Sector leadership (SectorSignals)
  Vol term structure (VolStructure)
  OPEX calendar   (OpexCalendar)
  Gamma walls      (via FlowState)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import List, Optional

from loguru import logger

from .economic_calendar import EconomicCalendar
from .news_fetcher import NewsFetcher
from .reddit_enhanced import RedditEnhanced
from .stocktwits_client import StockTwitsClient
from .macro_signals import MacroSignals
from .cboe_flow import CboeFlow
from .flow_confirmation import ExternalFlowConfirmation
from .breadth_signals import BreadthSignals
from .sector_signals import SectorSignals
from .vol_structure import VolStructure
from .opex_calendar import OpexCalendar


@dataclass
class ExternalContext:
    composite_score: float = 0.0
    event_risk: bool = False
    event_minutes: float = 999.0
    next_event_title: str = ""

    # Individual weighted source scores
    news_score: float = 0.0
    retail_score: float = 0.0          # StockTwits -1..+1
    reddit_score: float = 0.0          # -100..+100
    macro_headwind: float = 0.0        # -1..+1
    macro_label: str = "NEUTRAL"
    cboe_bias: float = 0.0             # contrarian -1..+1
    flow_score: float = 0.0            # options flow -100..+100

    # Flow detail
    flow_dark_pool: str = "NEUTRAL"
    flow_gex_bias: str = "NEUTRAL"
    flow_pc_ratio: Optional[float] = None
    flow_bullish_premium: float = 0.0
    flow_bearish_premium: float = 0.0
    flow_unusual_strikes: List[str] = field(default_factory=list)

    # Gamma walls (from FlowState)
    call_wall_strike: Optional[float] = None
    put_wall_strike: Optional[float] = None
    gamma_flip_level: Optional[float] = None
    at_call_wall: bool = False
    at_put_wall: bool = False

    # Macro detail
    tnx_trend: str = "FLAT"
    dxy_trend: str = "FLAT"
    oil_trend: str = "FLAT"
    vix_intraday: str = "FLAT"
    spy_vs_open_pct: float = 0.0
    vix_change_pct: float = 0.0

    # StockTwits detail
    bullish_pct: float = 0.0
    bearish_pct: float = 0.0

    # News
    news_headline_sample: List[str] = field(default_factory=list)

    # CBOE
    equity_pc: Optional[float] = None

    # Reddit
    reddit_contrarian: str = "NONE"

    # Market breadth
    breadth_ratio: float = 0.5
    breadth_label: str = "NEUTRAL"
    up_vol_ratio: float = 0.5
    breadth_sector_count_up: int = 0
    breadth_sector_count_down: int = 0
    tick_value: Optional[float] = None    # NYSE TICK snapshot (None if unavailable)
    tick_available: bool = False

    # Sector leadership
    sector_bull_count: int = 0
    sector_bear_count: int = 0
    sector_label: str = "NEUTRAL"
    qqq_vs_spy_pct: float = 0.0
    iwm_vs_spy_pct: float = 0.0
    es_premium: float = 0.0
    gap_pct: float = 0.0                  # today's open gap vs prior close
    above_overnight_high: bool = False
    below_overnight_low: bool = False
    overnight_range_pct: float = 0.0
    usdjpy_trend: str = "NEUTRAL"         # RISK_ON / RISK_OFF / NEUTRAL

    # Volatility term structure
    vix_vxv_ratio: Optional[float] = None
    vol_structure: str = "FLAT"           # STEEP_CONTANGO / CONTANGO / FLAT / BACKWARDATION / STEEP_BACKWARDATION
    vvix: Optional[float] = None
    vvix_elevated: bool = False

    # OPEX calendar
    opex_type: str = "MONTHLY"            # MONTHLY / TRIPLE_WITCHING
    days_to_opex: int = 99
    is_opex_week: bool = False
    is_opex_day: bool = False
    gamma_environment: str = "NEUTRAL"    # PINNING / EXPANSIVE / NEUTRAL

    sources_available: int = 0

    # ── Technical levels (from TechnicalLevelsTracker — injected by manager) ──
    # Opening Range Breakout: the single most-watched SPY intraday pattern
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_established: bool = False
    orb_width_pct: float = 0.0
    orb_status: str = "BUILDING"          # BUILDING | INSIDE | ABOVE_ORB | BELOW_ORB
    orb_breakout_confirmed: bool = False

    # VWAP standard-deviation bands
    vwap_1sd_upper: Optional[float] = None
    vwap_1sd_lower: Optional[float] = None
    vwap_2sd_upper: Optional[float] = None
    vwap_2sd_lower: Optional[float] = None
    vwap_band_position: str = "INSIDE_1SD"
    # ABOVE_2SD | ABOVE_1SD | INSIDE_1SD | BELOW_1SD | BELOW_2SD

    # Daily pivot points (Floor-Trader formula from prior session H/L/C)
    pivot_pp: Optional[float] = None
    pivot_r1: Optional[float] = None
    pivot_r2: Optional[float] = None
    pivot_s1: Optional[float] = None
    pivot_s2: Optional[float] = None
    near_pivot: bool = False
    pivot_nearest: str = ""               # "PP" | "R1" | "R2" | "S1" | "S2"
    pivot_bias: str = "NEUTRAL"           # AT_RESISTANCE | AT_SUPPORT | AT_PIVOT | NEUTRAL

    # Expected Daily Range exhaustion (VIX-implied intraday range)
    edr_points: float = 0.0
    edr_used_pct: float = 0.0
    edr_exhausted: bool = False

    # RSI (5-min) — overbought/oversold and divergence detection
    rsi_5m: float = 50.0
    rsi_overbought: bool = False
    rsi_oversold: bool = False
    rsi_divergence: str = "NONE"          # BULLISH_DIV | BEARISH_DIV | NONE

    # Max pain (computed from IB chain OI — injected by manager after chain build)
    max_pain_strike: Optional[float] = None
    near_max_pain: bool = False           # price within $1.50 of max pain
    max_pain_distance: float = 999.0      # |SPY − max_pain| in points


class ExternalDataManager:
    """
    Manages all external data sources. Call `refresh_if_stale()` once per
    poll cycle; then read `.context` for the latest ExternalContext.
    """

    def __init__(
        self,
        reddit_enabled: bool = False,
        reddit_client_id: str = "",
        reddit_client_secret: str = "",
        news_ttl_minutes: float = 10.0,
        reddit_ttl_minutes: float = 15.0,
        stocktwits_ttl_minutes: float = 10.0,
        event_risk_window_minutes: int = 30,
        flow_ttl_minutes: float = 10.0,
        flow_barchart_enabled: bool = True,
        flow_dark_pool_enabled: bool = True,
    ):
        self._calendar = EconomicCalendar()
        self._news = NewsFetcher(ttl_minutes=news_ttl_minutes)
        self._reddit = RedditEnhanced(
            enabled=reddit_enabled,
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            ttl_minutes=reddit_ttl_minutes,
        )
        self._stocktwits = StockTwitsClient(ttl_minutes=stocktwits_ttl_minutes)
        self._macro = MacroSignals()
        self._cboe = CboeFlow()
        self._flow = ExternalFlowConfirmation(
            ttl_minutes=flow_ttl_minutes,
            barchart_enabled=flow_barchart_enabled,
            dark_pool_enabled=flow_dark_pool_enabled,
        )
        self._breadth = BreadthSignals()
        self._sector = SectorSignals()
        self._vol_structure = VolStructure()
        self._opex = OpexCalendar()
        self._event_window = event_risk_window_minutes
        self._context = ExternalContext()

    async def refresh_if_stale(self) -> None:
        await asyncio.gather(
            self._calendar.refresh_if_stale(),
            self._news.refresh_if_stale(),
            self._reddit.refresh_if_stale(),
            self._stocktwits.refresh_if_stale(),
            self._macro.refresh_if_stale(),
            self._cboe.refresh_if_stale(),
            self._flow.refresh_if_stale(),
            self._breadth.refresh_if_stale(),
            self._sector.refresh_if_stale(),
            self._vol_structure.refresh_if_stale(),
            return_exceptions=True,
        )
        self._context = self._build_context()

    def _build_context(self) -> ExternalContext:
        cal    = self._calendar.state
        news   = self._news.state
        reddit = self._reddit.state
        st     = self._stocktwits.state
        macro  = self._macro.state
        cboe   = self._cboe.state
        flow   = self._flow.state
        brd    = self._breadth.state
        sec    = self._sector.state
        vol    = self._vol_structure.state

        # ── Event risk ────────────────────────────────────────────────────
        now = datetime.now(timezone.utc)
        event_risk = False
        event_minutes = 999.0
        next_event_title = ""

        nearby = cal.high_impact_within(self._event_window)
        if nearby:
            event_risk = True
            nearest = min(nearby, key=lambda e: abs((e.event_dt - now).total_seconds()))
            event_minutes = abs((nearest.event_dt - now).total_seconds()) / 60.0
            next_event_title = nearest.title
        else:
            nxt = cal.next_high_impact()
            if nxt:
                event_minutes = (nxt.event_dt - now).total_seconds() / 60.0
                next_event_title = nxt.title

        # ── Component scores ─────────────────────────────────────────────
        components: dict[str, tuple[float, float]] = {}

        if news.article_count > 0:
            components["news"] = (news.score, 0.20)

        if st.available and st.message_count > 0:
            components["stocktwits"] = (st.score, 0.15)

        if macro.available:
            components["macro"] = (macro.spy_headwind, 0.20)

        if cboe.available:
            components["cboe"] = (cboe.sentiment_bias, 0.15)

        if flow.available:
            # flow.aggregate_score is -100..+100; normalise to -1..+1
            components["flow"] = (flow.aggregate_score / 100.0, 0.20)

        if reddit.available and reddit.post_count > 0:
            components["reddit"] = (reddit.score / 100.0, 0.10)

        # Normalise weights
        if components:
            total_w = sum(w for _, w in components.values())
            composite = sum(s * w / total_w for s, w in components.values())
        else:
            composite = 0.0

        # ── OPEX (no refresh needed — pure date math) ─────────────────────
        opex_state = self._opex.compute(today=date.today(), net_gex=flow.net_gex)

        return ExternalContext(
            composite_score=round(composite, 4),
            event_risk=event_risk,
            event_minutes=round(event_minutes, 1),
            next_event_title=next_event_title,

            # Weighted sources
            news_score=round(news.score, 4),
            retail_score=round(st.score, 4),
            reddit_score=round(reddit.score, 2),
            macro_headwind=round(macro.spy_headwind, 4),
            macro_label=macro.macro_label,
            cboe_bias=round(cboe.sentiment_bias, 4),
            flow_score=round(flow.aggregate_score, 2),

            # Flow detail
            flow_dark_pool=flow.dark_pool_bias,
            flow_gex_bias=flow.gamma_exposure_bias,
            flow_pc_ratio=flow.pc_ratio,
            flow_bullish_premium=round(flow.bullish_premium, 0),
            flow_bearish_premium=round(flow.bearish_premium, 0),
            flow_unusual_strikes=flow.unusual_strikes,

            # Gamma walls
            call_wall_strike=flow.call_wall_strike,
            put_wall_strike=flow.put_wall_strike,
            gamma_flip_level=flow.gamma_flip_level,
            at_call_wall=flow.at_call_wall,
            at_put_wall=flow.at_put_wall,

            # Macro detail
            tnx_trend=macro.tnx_trend,
            dxy_trend=macro.dxy_trend,
            oil_trend=macro.oil_trend,
            vix_intraday=macro.vix_intraday,
            spy_vs_open_pct=macro.spy_vs_open_pct,
            vix_change_pct=macro.vix_change_pct,

            # StockTwits
            bullish_pct=round(st.bullish_pct, 1),
            bearish_pct=round(st.bearish_pct, 1),

            # News
            news_headline_sample=news.headline_sample,

            # CBOE
            equity_pc=cboe.equity_pc,

            # Reddit
            reddit_contrarian=reddit.contrarian_signal,

            # Market breadth
            breadth_ratio=round(brd.breadth_ratio, 3),
            breadth_label=brd.breadth_label,
            up_vol_ratio=round(brd.up_vol_ratio, 3),
            breadth_sector_count_up=brd.sector_count_up,
            breadth_sector_count_down=brd.sector_count_down,
            tick_value=brd.tick_value,
            tick_available=brd.tick_available,

            # Sector leadership
            sector_bull_count=sec.sector_bull_count,
            sector_bear_count=sec.sector_bear_count,
            sector_label=sec.sector_label,
            qqq_vs_spy_pct=round(sec.qqq_vs_spy_pct, 3),
            iwm_vs_spy_pct=round(sec.iwm_vs_spy_pct, 3),
            es_premium=round(sec.es_premium, 2),
            gap_pct=round(sec.gap_pct, 3),
            above_overnight_high=sec.above_overnight_high,
            below_overnight_low=sec.below_overnight_low,
            overnight_range_pct=round(sec.overnight_range_pct, 3),
            usdjpy_trend=sec.usdjpy_trend,

            # Vol structure
            vix_vxv_ratio=vol.vix_vxv_ratio,
            vol_structure=vol.vol_structure,
            vvix=vol.vvix,
            vvix_elevated=vol.vvix_elevated,

            # OPEX
            opex_type=opex_state.opex_type,
            days_to_opex=opex_state.days_to_opex,
            is_opex_week=opex_state.is_opex_week,
            is_opex_day=opex_state.is_opex_day,
            gamma_environment=opex_state.gamma_environment,

            sources_available=len(components),
        )

    @property
    def context(self) -> ExternalContext:
        return self._context
