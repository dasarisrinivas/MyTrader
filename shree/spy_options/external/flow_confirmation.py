"""
ExternalFlowConfirmation — aggregates free SPY options flow data into a
scored signal between -100 (strongly bearish) and +100 (strongly bullish).

Primary source (always available):
  yfinance options chain → compute unusual volume/OI ratios, net premium,
  sweep proxies, gamma exposure (GEX), and directional flow.

Secondary sources (attempted, silently skipped on failure):
  Barchart unusual options  — HTML scrape with anti-bot fallback
  Alpha Query dark pool     — public JSON endpoint
  CBOE intraday P/C        — intraday estimate from CBOE public data

Scoring formula per flow record:
  contribution = base × dte_mult × prem_mult × bias_mult × voi_mult
                      × delta_mult × type_mult × time_decay

  base:        +10 BULLISH, -10 BEARISH
  dte_mult:    2.0 (0DTE) → 1.5 (1DTE) → 1.0 (2-7DTE) → 0.5 (8+DTE)
  prem_mult:   log10(premium / $100K) + 1, clamped 0.5–3.0
  bias_mult:   1.3 ASK execution | 1.0 MID | 0.7 BID
  voi_mult:    1.5 (vol/OI > 3) | 1.2 (vol/OI > 1)
  delta_mult:  1.0 (ATM: |delta| 0.25–0.75) | 0.6 (far OTM)
  type_mult:   1.2 if sweep or block
  time_decay:  exp(-age_min / 120)    half-life ~83 min

GEX (Gamma Exposure):
  net_gex = Σ(call_gamma × call_oi) − Σ(put_gamma × put_oi)  per strike
  Positive: dealers sell rallies / buy dips → dampening (range-bound signal)
  Negative: dealers buy rallies / sell dips → amplifying (trend signal)

Dark pool proxy:
  Inferred from large block premium accumulation skew on bid vs ask.
  True dark pool data (FINRA) is only published weekly — real-time is
  estimated from institutional-size ($1M+) options flow bias.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import aiohttp
import yfinance as yf
from loguru import logger


# ── Schema ────────────────────────────────────────────────────────────────────

@dataclass
class FlowRecord:
    timestamp: datetime
    source: str                   # "yfinance" | "barchart" | "computed"
    symbol: str = "SPY"
    strike: float = 0.0
    expiry: str = ""
    right: str = "C"              # "C" or "P"
    premium_size: float = 0.0     # total premium in USD (price × vol × 100)
    is_sweep: bool = False        # large single-poll volume burst
    is_block: bool = False        # single trade > $500K premium
    bid_ask_bias: str = "MID"     # "ASK" | "MID" | "BID"
    volume: int = 0
    open_interest: int = 0
    vol_oi_ratio: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    dte: int = 0
    direction: str = "NEUTRAL"   # "BULLISH" | "BEARISH" | "NEUTRAL"
    confidence: float = 0.5
    dark_pool_bias: str = "NEUTRAL"  # "ACCUMULATION" | "DISTRIBUTION" | "NEUTRAL"


@dataclass
class FlowState:
    records: List[FlowRecord] = field(default_factory=list)
    aggregate_score: float = 0.0    # -100 to +100
    bullish_premium: float = 0.0    # total USD in bullish flow
    bearish_premium: float = 0.0
    call_volume: int = 0
    put_volume: int = 0
    unusual_strikes: List[str] = field(default_factory=list)  # "565C APR26" format
    dark_pool_bias: str = "NEUTRAL"
    gamma_exposure_bias: str = "NEUTRAL"  # "SUPPORTIVE_UPSIDE" | "SUPPORTIVE_DOWNSIDE"
    net_gex: float = 0.0
    pc_ratio: Optional[float] = None  # intraday put/call vol ratio
    sources_used: List[str] = field(default_factory=list)
    fetched_at: float = 0.0
    available: bool = False

    def is_stale(self, ttl_s: float) -> bool:
        return (time.monotonic() - self.fetched_at) > ttl_s


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dte_from_expiry(expiry_str: str) -> int:
    """Parse yfinance expiry string 'YYYY-MM-DD' → days to expiry."""
    try:
        exp = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0, (exp.date() - now.date()).days)
    except Exception:
        return 30


def _score_record(record: FlowRecord, now: datetime) -> float:
    """Compute the signed contribution of a single flow record."""
    base = 10.0 if record.direction == "BULLISH" else -10.0 if record.direction == "BEARISH" else 0.0
    if base == 0.0:
        return 0.0

    # DTE multiplier: 0DTE is most relevant
    dte = record.dte
    if dte == 0:
        dte_mult = 2.0
    elif dte == 1:
        dte_mult = 1.5
    elif dte <= 7:
        dte_mult = 1.0
    else:
        dte_mult = 0.5

    # Premium size (log scale)
    prem = max(1.0, record.premium_size)
    prem_mult = max(0.5, min(3.0, math.log10(prem / 100_000) + 1))

    # Execution bias
    bias_map = {"ASK": 1.3, "MID": 1.0, "BID": 0.7}
    bias_mult = bias_map.get(record.bid_ask_bias, 1.0)

    # Volume / Open Interest
    if record.vol_oi_ratio > 3.0:
        voi_mult = 1.5
    elif record.vol_oi_ratio > 1.0:
        voi_mult = 1.2
    else:
        voi_mult = 1.0

    # Delta quality: ATM options carry more signal weight
    d = abs(record.delta)
    delta_mult = 1.0 if 0.25 <= d <= 0.75 else 0.6

    # Sweep / block premium
    type_mult = 1.2 if (record.is_sweep or record.is_block) else 1.0

    # Time decay — half-life ~83 min
    age_min = (now - record.timestamp).total_seconds() / 60.0
    time_decay = math.exp(-age_min / 120.0)

    return base * dte_mult * prem_mult * bias_mult * voi_mult * delta_mult * type_mult * time_decay


def _normalise_score(raw: float) -> float:
    """Sigmoid-normalise raw score → -100 to +100."""
    if raw == 0:
        return 0.0
    # Use a scaling factor; 200 raw = ±80 normalised
    clamped = max(-300.0, min(300.0, raw))
    return round(clamped / 3.0, 2)


# ── Main class ────────────────────────────────────────────────────────────────

class ExternalFlowConfirmation:
    """
    Aggregates SPY options flow from multiple free sources.
    Call `refresh_if_stale()` once per poll; read `.state` afterward.
    """

    def __init__(
        self,
        ttl_minutes: float = 10.0,
        min_volume_for_unusual: int = 1000,
        min_vol_oi_ratio: float = 2.0,
        min_block_premium: float = 500_000,
        barchart_enabled: bool = True,
        dark_pool_enabled: bool = True,
        timeout_s: float = 10.0,
    ):
        self._ttl_s = ttl_minutes * 60.0
        self._min_vol_unusual = min_volume_for_unusual
        self._min_voi = min_vol_oi_ratio
        self._min_block = min_block_premium
        self._barchart_enabled = barchart_enabled
        self._dark_pool_enabled = dark_pool_enabled
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._state = FlowState()
        self._lock = asyncio.Lock()

    async def refresh_if_stale(self) -> None:
        if not self._state.is_stale(self._ttl_s):
            return
        async with self._lock:
            if not self._state.is_stale(self._ttl_s):
                return
            await self._fetch()

    async def _fetch(self) -> None:
        now = datetime.now(timezone.utc)
        all_records: List[FlowRecord] = []
        sources: List[str] = []

        # ── 1. yfinance options chain (primary, always attempted) ──────────
        yf_records = await self._fetch_yfinance(now)
        if yf_records:
            all_records.extend(yf_records)
            sources.append("yfinance")

        # ── 2. Barchart unusual options (HTML scrape, optional) ───────────
        if self._barchart_enabled:
            bc_records = await self._try_barchart(now)
            if bc_records:
                all_records.extend(bc_records)
                sources.append("barchart")

        # ── 3. Dark pool proxy via Alpha Query ────────────────────────────
        dp_bias = "NEUTRAL"
        if self._dark_pool_enabled:
            dp_bias = await self._try_dark_pool()

        # ── Aggregate ─────────────────────────────────────────────────────
        if not all_records:
            self._state = FlowState(
                fetched_at=time.monotonic(), available=False
            )
            return

        raw_score = sum(_score_record(r, now) for r in all_records)
        agg_score = _normalise_score(raw_score)

        bullish_prem = sum(r.premium_size for r in all_records if r.direction == "BULLISH")
        bearish_prem = sum(r.premium_size for r in all_records if r.direction == "BEARISH")
        call_vol = sum(r.volume for r in all_records if r.right == "C")
        put_vol = sum(r.volume for r in all_records if r.right == "P")
        pc = put_vol / call_vol if call_vol > 0 else None

        unusual = [
            f"{r.strike:.0f}{r.right} {r.expiry}"
            for r in all_records
            if r.vol_oi_ratio >= self._min_voi and r.volume >= self._min_vol_unusual
        ][:10]

        # ── GEX from yfinance records ──────────────────────────────────────
        net_gex, gex_bias = self._compute_gex(yf_records)

        self._state = FlowState(
            records=all_records[-200:],  # keep last 200
            aggregate_score=agg_score,
            bullish_premium=bullish_prem,
            bearish_premium=bearish_prem,
            call_volume=call_vol,
            put_volume=put_vol,
            unusual_strikes=unusual,
            dark_pool_bias=dp_bias,
            gamma_exposure_bias=gex_bias,
            net_gex=round(net_gex, 2),
            pc_ratio=round(pc, 3) if pc else None,
            sources_used=sources,
            fetched_at=time.monotonic(),
            available=True,
        )
        logger.info(
            "[FlowConfirm] score={:+.0f}  bull_prem=${:.0f}K  bear_prem=${:.0f}K  "
            "P/C={:.2f}  GEX={:.1f}({})  dp={}  sources={}",
            agg_score,
            bullish_prem / 1000,
            bearish_prem / 1000,
            pc or 0,
            net_gex,
            gex_bias,
            dp_bias,
            sources,
        )

    # ── yfinance options chain ────────────────────────────────────────────────

    async def _fetch_yfinance(self, now: datetime) -> List[FlowRecord]:
        loop = asyncio.get_event_loop()
        try:
            records = await loop.run_in_executor(None, self._yfinance_sync, now)
            return records
        except Exception as exc:
            logger.debug(f"[FlowConfirm] yfinance error: {exc}")
            return []

    def _yfinance_sync(self, now: datetime) -> List[FlowRecord]:
        spy = yf.Ticker("SPY")
        spy_price = spy.fast_info.get("last_price") or 0.0

        all_dates = spy.options  # tuple of expiry strings 'YYYY-MM-DD'
        if not all_dates:
            return []

        records: List[FlowRecord] = []
        # Process up to first 4 expiries (0DTE through ~1 week)
        for exp_str in all_dates[:4]:
            dte = _dte_from_expiry(exp_str)
            try:
                chain = spy.option_chain(exp_str)
            except Exception:
                continue

            for df, right in [(chain.calls, "C"), (chain.puts, "P")]:
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    volume = int(row.get("volume") or 0)
                    oi = int(row.get("openInterest") or 0)
                    if volume < 100:
                        continue

                    strike = float(row.get("strike") or 0)
                    bid = float(row.get("bid") or 0)
                    ask = float(row.get("ask") or 0)
                    last = float(row.get("lastPrice") or 0)
                    delta = float(row.get("delta") or 0)  # may be 0 if not in chain
                    gamma = float(row.get("gamma") or 0)

                    # Estimate delta from strike if not provided
                    if delta == 0.0 and spy_price > 0:
                        moneyness = (spy_price - strike) / spy_price
                        if right == "C":
                            delta = max(0.05, min(0.95, 0.5 + moneyness * 10))
                        else:
                            delta = max(-0.95, min(-0.05, -(0.5 - moneyness * 10)))

                    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last
                    premium_size = mid * volume * 100

                    # Execution bias
                    spread = ask - bid
                    if spread > 0.01 and last > 0:
                        bias_raw = (last - bid) / spread
                        if bias_raw > 0.75:
                            bias = "ASK"
                        elif bias_raw < 0.25:
                            bias = "BID"
                        else:
                            bias = "MID"
                    else:
                        bias = "MID"

                    vol_oi = volume / max(1, oi)
                    is_unusual = vol_oi >= self._min_voi and volume >= self._min_vol_unusual
                    is_block = premium_size >= self._min_block

                    # Direction: calls near ask = bullish, puts near ask = bearish
                    if right == "C":
                        direction = "BULLISH" if bias == "ASK" else "NEUTRAL"
                    else:
                        direction = "BEARISH" if bias == "ASK" else "NEUTRAL"

                    # Only store if unusual or block
                    if not (is_unusual or is_block):
                        continue

                    records.append(FlowRecord(
                        timestamp=now,
                        source="yfinance",
                        strike=strike,
                        expiry=exp_str,
                        right=right,
                        premium_size=premium_size,
                        is_sweep=is_unusual,
                        is_block=is_block,
                        bid_ask_bias=bias,
                        volume=volume,
                        open_interest=oi,
                        vol_oi_ratio=round(vol_oi, 2),
                        delta=delta,
                        gamma=gamma,
                        dte=dte,
                        direction=direction,
                        confidence=0.6 if is_block else 0.45,
                    ))

        return records

    # ── GEX computation ───────────────────────────────────────────────────────

    def _compute_gex(self, records: List[FlowRecord]) -> Tuple[float, str]:
        """Compute net gamma exposure from flow records.

        GEX = Σ(call_gamma × call_OI) − Σ(put_gamma × put_OI)
        Positive GEX → dealers long gamma → dampening → range signal
        Negative GEX → dealers short gamma → amplifying → trend signal
        """
        call_gex = sum(r.gamma * r.open_interest for r in records if r.right == "C" and r.gamma > 0)
        put_gex  = sum(r.gamma * r.open_interest for r in records if r.right == "P" and r.gamma > 0)
        net = call_gex - put_gex

        threshold = 100_000  # arbitrary scale for normalisation
        if net > threshold:
            bias = "SUPPORTIVE_UPSIDE"
        elif net < -threshold:
            bias = "SUPPORTIVE_DOWNSIDE"
        else:
            bias = "NEUTRAL"

        return net, bias

    # ── Barchart scraper (optional, best-effort) ──────────────────────────────

    async def _try_barchart(self, now: datetime) -> List[FlowRecord]:
        """
        Attempt to scrape Barchart unusual options activity.
        Returns empty list on any failure (anti-bot, rate limit, schema change).
        """
        url = "https://www.barchart.com/options/unusual-activity/stocks?page=1&orderBy=tradeValue&orderDir=desc"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.barchart.com/",
        }
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=8.0)
            ) as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        logger.debug(f"[Barchart] HTTP {resp.status} — skipping")
                        return []
                    html = await resp.text()
            return self._parse_barchart_html(html, now)
        except Exception as exc:
            logger.debug(f"[Barchart] Scrape failed: {exc}")
            return []

    def _parse_barchart_html(self, html: str, now: datetime) -> List[FlowRecord]:
        """
        Parse Barchart unusual options table rows.
        Barchart renders rows as: Symbol | Exp | Strike | Type | Vol | OI | Vol/OI | Value | ...
        This parser is best-effort; returns [] if schema doesn't match.
        """
        records: List[FlowRecord] = []
        try:
            # Look for SPY rows only
            rows = html.split("SPY")
            for row_fragment in rows[1:21]:  # at most 20 SPY rows
                # Extract strike
                import re
                strike_m = re.search(r"\$?(\d{3,4}(?:\.\d{0,2})?)", row_fragment[:200])
                if not strike_m:
                    continue
                strike = float(strike_m.group(1))
                if strike < 400 or strike > 700:
                    continue

                right = "C" if "Call" in row_fragment[:300] or "CALL" in row_fragment[:300] else "P"

                # Look for volume/value
                numbers = re.findall(r"[\d,]+(?:\.\d+)?[KMB]?", row_fragment[:500])
                if len(numbers) < 3:
                    continue

                def _parse_num(s: str) -> float:
                    s = s.replace(",", "")
                    if s.endswith("B"):
                        return float(s[:-1]) * 1e9
                    if s.endswith("M"):
                        return float(s[:-1]) * 1e6
                    if s.endswith("K"):
                        return float(s[:-1]) * 1e3
                    try:
                        return float(s)
                    except ValueError:
                        return 0.0

                direction = "BULLISH" if right == "C" else "BEARISH"
                records.append(FlowRecord(
                    timestamp=now,
                    source="barchart",
                    strike=strike,
                    right=right,
                    premium_size=0.0,  # not reliably parsed
                    is_sweep=True,
                    bid_ask_bias="ASK",  # Barchart shows sweeps by default
                    volume=0,
                    dte=0,
                    direction=direction,
                    confidence=0.4,
                ))
        except Exception as exc:
            logger.debug(f"[Barchart] Parse error: {exc}")
        return records

    # ── Dark pool proxy ───────────────────────────────────────────────────────

    async def _try_dark_pool(self) -> str:
        """
        Attempt Alpha Query dark pool endpoint.
        Returns ACCUMULATION | DISTRIBUTION | NEUTRAL.
        Falls back to computing from flow record premium skew.
        """
        url = "https://alphaquery.com/stock/SPY/volatility-option-statistics/30-day/dark-pool-data"
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=6.0)
            ) as session:
                headers = {"Accept": "application/json, text/javascript, */*"}
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        return self._dark_pool_from_premium_skew()
                    data = await resp.json(content_type=None)

            # Alpha Query returns JSON with darkPoolIndex field
            dp_index = data.get("darkPoolIndex") or data.get("dark_pool_index") or 0.0
            dp_float = float(dp_index)
            if dp_float > 55:
                return "ACCUMULATION"
            elif dp_float < 45:
                return "DISTRIBUTION"
            return "NEUTRAL"
        except Exception:
            return self._dark_pool_from_premium_skew()

    def _dark_pool_from_premium_skew(self) -> str:
        """
        Proxy: if recent large block trades ($1M+) are heavily skewed toward
        one direction, infer dark pool bias.
        """
        records = self._state.records
        blocks = [r for r in records if r.premium_size >= 1_000_000]
        if not blocks:
            return "NEUTRAL"
        bull_block = sum(r.premium_size for r in blocks if r.direction == "BULLISH")
        bear_block = sum(r.premium_size for r in blocks if r.direction == "BEARISH")
        total = bull_block + bear_block
        if total == 0:
            return "NEUTRAL"
        bull_ratio = bull_block / total
        if bull_ratio > 0.65:
            return "ACCUMULATION"
        if bull_ratio < 0.35:
            return "DISTRIBUTION"
        return "NEUTRAL"

    @property
    def state(self) -> FlowState:
        return self._state
