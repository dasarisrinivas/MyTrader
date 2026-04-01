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

ExternalContext fields:
  composite_score      float  -1.0..+1.0
  event_risk           bool   True if US High-impact event within ±window
  event_minutes        float  Minutes to/from nearest event
  next_event_title     str
  news_score           float  -1.0..+1.0
  retail_score         float  StockTwits -1..+1
  reddit_score         float  -100..+100 (enhanced, 0 if disabled)
  macro_headwind       float  -1.0..+1.0
  macro_label          str    STRONG_HEADWIND|HEADWIND|NEUTRAL|TAILWIND|STRONG_TAILWIND
  cboe_bias            float  -1.0..+1.0 (contrarian)
  flow_score           float  -100..+100 (ExternalFlowConfirmation)
  flow_dark_pool       str    ACCUMULATION|DISTRIBUTION|NEUTRAL
  flow_gex_bias        str    SUPPORTIVE_UPSIDE|SUPPORTIVE_DOWNSIDE|NEUTRAL
  flow_pc_ratio        float  intraday P/C ratio (or None)
  flow_unusual_strikes list   top unusual activity labels
  tnx_trend            str
  dxy_trend            str
  oil_trend            str
  vix_intraday         str    intraday VIX direction
  spy_vs_open_pct      float
  bullish_pct          float  StockTwits bullish %
  bearish_pct          float
  news_headline_sample list
  equity_pc            float  CBOE P/C (or None)
  reddit_contrarian    str    CONTRARIAN_BULLISH|CONTRARIAN_BEARISH|NONE
  sources_available    int
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

from .economic_calendar import EconomicCalendar
from .news_fetcher import NewsFetcher
from .reddit_enhanced import RedditEnhanced
from .stocktwits_client import StockTwitsClient
from .macro_signals import MacroSignals
from .cboe_flow import CboeFlow
from .flow_confirmation import ExternalFlowConfirmation


@dataclass
class ExternalContext:
    composite_score: float = 0.0
    event_risk: bool = False
    event_minutes: float = 999.0
    next_event_title: str = ""

    # Individual source scores
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

    sources_available: int = 0


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
            return_exceptions=True,
        )
        self._context = self._build_context()

    def _build_context(self) -> ExternalContext:
        cal      = self._calendar.state
        news     = self._news.state
        reddit   = self._reddit.state
        st       = self._stocktwits.state
        macro    = self._macro.state
        cboe     = self._cboe.state
        flow     = self._flow.state

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
        # name: (score_−1_to_+1, weight)
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
            # reddit.score is -100..+100
            components["reddit"] = (reddit.score / 100.0, 0.10)

        # Normalise weights
        if components:
            total_w = sum(w for _, w in components.values())
            composite = sum(s * w / total_w for s, w in components.values())
        else:
            composite = 0.0

        return ExternalContext(
            composite_score=round(composite, 4),
            event_risk=event_risk,
            event_minutes=round(event_minutes, 1),
            next_event_title=next_event_title,
            news_score=round(news.score, 4),
            retail_score=round(st.score, 4),
            reddit_score=round(reddit.score, 2),
            macro_headwind=round(macro.spy_headwind, 4),
            macro_label=macro.macro_label,
            cboe_bias=round(cboe.sentiment_bias, 4),
            flow_score=round(flow.aggregate_score, 2),
            flow_dark_pool=flow.dark_pool_bias,
            flow_gex_bias=flow.gamma_exposure_bias,
            flow_pc_ratio=flow.pc_ratio,
            flow_bullish_premium=round(flow.bullish_premium, 0),
            flow_bearish_premium=round(flow.bearish_premium, 0),
            flow_unusual_strikes=flow.unusual_strikes,
            tnx_trend=macro.tnx_trend,
            dxy_trend=macro.dxy_trend,
            oil_trend=macro.oil_trend,
            vix_intraday=macro.vix_intraday,
            spy_vs_open_pct=macro.spy_vs_open_pct,
            vix_change_pct=macro.vix_change_pct,
            bullish_pct=round(st.bullish_pct, 1),
            bearish_pct=round(st.bearish_pct, 1),
            news_headline_sample=news.headline_sample,
            equity_pc=cboe.equity_pc,
            reddit_contrarian=reddit.contrarian_signal,
            sources_available=len(components),
        )

    @property
    def context(self) -> ExternalContext:
        return self._context
