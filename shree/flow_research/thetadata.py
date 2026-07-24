"""ThetaData v3 adapter — real OPRA option prints for the flow research layer.

Talks to the LOCAL Theta Terminal REST server (default http://127.0.0.1:25503),
v3 API, CSV responses. Read-only market data; imports nothing from production.

Confirmed empirically 2026-07-23 against a running terminal:
  * REST v3 on :25503, CSV responses (header row + rows).
  * Endpoints (option): /v3/option/history/trade_quote (STANDARD tier),
      /v3/option/history/trade (STANDARD), /v3/option/history/quote (VALUE),
      /v3/option/list/expirations (FREE), /v3/option/list/strikes (FREE).
      /v3/stock/history/eod (FREE) for the underlying close.
  * Query params: symbol, expiration=YYYYMMDD, strike, right=C|P,
      start_date=YYYYMMDD, end_date=YYYYMMDD. (list output dates are YYYY-MM-DD.)
  * The EOD/quote schema exposes bid/ask + bid_exchange/ask_exchange +
      bid_condition/ask_condition -> venue + condition codes ARE available
      (real sweep detection + spread-leg filtering, unlike IB).

TIMEZONE: ThetaData reports ET (America/New_York) — CORROBORATED 2026-07-23 on
free EOD data: max `last_trade` hour across the chain = 16 (ET close), not 20
(would be UTC). `source_tz` stays configurable; still worth a final glance on the
first Standard tick pull. Our flow prints store ts_et in ET, so no conversion.

BULK: omitting `strike` (and optionally `right`) returns the WHOLE chain for an
expiration+date in one call (verified on free EOD: 335 rows calls+puts). Cuts an
Apr-Jul pull from ~5000 per-contract requests to ~240 whole-chain calls. This
adapter defaults to per-contract near-ATM (bounded payloads, requests are
unlimited/local on Standard); switch to chain pulls if request count ever bites.

The exact `trade_quote` column names are gated behind the Standard tier, so the
row->Print mapper is HEADER-DRIVEN with alias resolution and logs any unmapped
columns on first use — pin the real names when the subscription is live.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Sequence
from zoneinfo import ZoneInfo

import requests

from .models import Print
from .ingest import PrintSource
from .greeks import bs_delta, implied_vol

logger = logging.getLogger("flow_research.thetadata")

DEFAULT_BASE = "http://127.0.0.1:25503"
ET = ZoneInfo("America/New_York")
TRADING_DAY_YEAR = 252.0


class ThetaError(RuntimeError):
    pass


class ThetaSubscriptionError(ThetaError):
    """Raised on HTTP 403 — the endpoint needs a higher tier than logged in."""


# ── column alias resolution (header-driven) ──────────────────────────────────

# Confirmed against the live Standard trade_quote schema (2026-07-24):
#   trade_timestamp, quote_timestamp, sequence, ext_condition1..4, condition,
#   size, exchange, price, bid_size, bid_exchange, bid, bid_condition,
#   ask_size, ask_exchange, ask, ask_condition
# NOTE: `condition`/`exchange` are NUMERIC OPRA codes (e.g. condition=125,
# exchange=22), not strings — so the string-based spread/auction condition
# filter (is_clean_print) currently passes everything. That only affects NET
# measures (may include spread legs = noise, biasing toward FAIL — the safe
# direction). Aggressor classification (Lee-Ready on bid/ask) is unaffected.
# Mapping numeric OPRA condition codes is a documented refinement.
_ALIASES = {
    "timestamp": ["trade_timestamp", "timestamp", "datetime", "trade_time",
                  "time", "last_trade", "created"],
    "ms_of_day": ["ms_of_day", "trade_ms", "ms"],
    "date": ["date", "trade_date"],
    "price": ["price", "trade_price", "last_trade_price"],
    "size": ["size", "trade_size"],
    "exchange": ["exchange", "trade_exchange"],
    "condition": ["condition", "conditions", "trade_condition"],
    "bid": ["bid", "bid_price"],
    "ask": ["ask", "ask_price"],
    "bid_size": ["bid_size"],
    "ask_size": ["ask_size"],
    "bid_exchange": ["bid_exchange"],
    "ask_exchange": ["ask_exchange"],
    "bid_condition": ["bid_condition"],
    "ask_condition": ["ask_condition"],
}


def _resolve(header: Sequence[str]) -> Dict[str, str]:
    """Map canonical name -> actual header column (first alias that appears)."""
    lower = {h.lower(): h for h in header}
    out: Dict[str, str] = {}
    for canon, aliases in _ALIASES.items():
        for a in aliases:
            if a in lower:
                out[canon] = lower[a]
                break
    return out


# ── HTTP client ──────────────────────────────────────────────────────────────

class ThetaClient:
    def __init__(self, base_url: str = DEFAULT_BASE, timeout: int = 30):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def get_csv(self, path: str, params: Dict) -> List[Dict[str, str]]:
        url = f"{self.base}{path}"
        r = self.session.get(url, params=params, timeout=self.timeout)
        if r.status_code == 403:
            raise ThetaSubscriptionError(r.text.strip())
        if r.status_code == 404:
            raise ThetaError(f"404 path not found: {path}")
        if r.status_code != 200:
            raise ThetaError(f"HTTP {r.status_code} on {path}: {r.text[:200]}")
        reader = csv.DictReader(io.StringIO(r.text))
        return list(reader)

    # discovery (FREE tier) --------------------------------------------------
    def list_expirations(self, symbol: str) -> List[date]:
        rows = self.get_csv("/v3/option/list/expirations", {"symbol": symbol})
        out = []
        for row in rows:
            v = (row.get("expiration") or "").strip('"')
            d = _parse_date(v)
            if d:
                out.append(d)
        return sorted(set(out))

    def list_strikes(self, symbol: str, expiration: date) -> List[float]:
        rows = self.get_csv("/v3/option/list/strikes",
                            {"symbol": symbol, "expiration": _ymd(expiration)})
        out = []
        for row in rows:
            try:
                out.append(float(row.get("strike")))
            except (TypeError, ValueError):
                continue
        return sorted(set(out))

    def stock_eod_close(self, symbol: str, start: date, end: date) -> Dict[date, float]:
        """SPY daily close by session (FREE) — coarse underlying for greeks."""
        rows = self.get_csv("/v3/stock/history/eod",
                            {"symbol": symbol,
                             "start_date": _ymd(start), "end_date": _ymd(end)})
        out: Dict[date, float] = {}
        for row in rows:
            d = _parse_date((row.get("last_trade") or row.get("created") or "")[:10])
            try:
                close = float(row.get("close"))
            except (TypeError, ValueError):
                continue
            if d:
                out[d] = close
        return out

    # option history (SUBSCRIPTION) ------------------------------------------
    def option_trade_quote(self, symbol: str, expiration: date, strike: float,
                           right: str, start: date, end: date) -> List[Dict[str, str]]:
        return self.get_csv("/v3/option/history/trade_quote", {
            "symbol": symbol, "expiration": _ymd(expiration),
            "strike": _strike(strike), "right": right.upper()[:1],
            "start_date": _ymd(start), "end_date": _ymd(end),
        })


# ── source ───────────────────────────────────────────────────────────────────

class ThetaDataSource(PrintSource):
    """Iterates SPY 0..dte_max option contracts near ATM over a date range and
    yields classified-ready Print rows from ThetaData trade_quote.

    Observation only. Requires the Standard tier for trade_quote (raises
    ThetaSubscriptionError otherwise — caught by the driver so a free run still
    exercises discovery + parsing).
    """

    def __init__(self, client: ThetaClient, start: date, end: date,
                 symbol: str = "SPY", dte_max: int = 2,
                 strikes_around_atm: int = 5, source_tz: ZoneInfo = ET,
                 compute_greeks: bool = True):
        self.c = client
        self.symbol = symbol
        self.start = start
        self.end = end
        self.dte_max = dte_max
        self.n = strikes_around_atm
        self.tz = source_tz
        self.compute_greeks = compute_greeks
        self._colmap: Optional[Dict[str, str]] = None
        self._warned_unmapped = False

    def prints(self) -> Iterator[Print]:
        exps = self.c.list_expirations(self.symbol)
        sessions = [e for e in exps if self.start <= e <= self.end]
        closes = self.c.stock_eod_close(self.symbol, self.start, self.end)
        for i, S in enumerate(sessions):
            atm = closes.get(S)
            # 0..dte_max DTE contracts = this expiry + the next dte_max listed ones
            targets = [e for e in exps if e >= S][: self.dte_max + 1]
            for exp in targets:
                strikes = self._near_atm(exp, atm)
                dte = _trading_dte(S, exp, exps)
                for strike in strikes:
                    for right in ("C", "P"):
                        try:
                            rows = self.c.option_trade_quote(
                                self.symbol, exp, strike, right, S, S)
                        except ThetaSubscriptionError:
                            raise  # surfaced to driver; nothing else to do free
                        for r in rows:
                            p = self._row_to_print(r, exp, strike, right, S,
                                                   atm, dte)
                            if p is not None:
                                yield p

    def _near_atm(self, exp: date, atm: Optional[float]) -> List[float]:
        strikes = self.c.list_strikes(self.symbol, exp)
        if not strikes:
            return []
        if atm is None:
            return strikes
        strikes.sort(key=lambda k: abs(k - atm))
        return sorted(strikes[: 2 * self.n + 1])

    def _row_to_print(self, row: Dict[str, str], exp: date, strike: float,
                      right: str, session: date, atm: Optional[float],
                      dte: int) -> Optional[Print]:
        if self._colmap is None:
            self._colmap = _resolve(list(row.keys()))
            self._log_unmapped(row)
        cm = self._colmap

        ts = self._parse_ts(row, cm, session)
        if ts is None:
            return None
        px = _f(row.get(cm.get("price", "")))
        sz = _i(row.get(cm.get("size", "")))
        if not px or not sz or px <= 0 or sz <= 0:
            return None
        bid = _f(row.get(cm.get("bid", "")))
        ask = _f(row.get(cm.get("ask", "")))
        exch = row.get(cm.get("exchange", "")) or None
        cond_raw = row.get(cm.get("condition", "")) or ""
        conds = [c for c in cond_raw.replace("|", ",").split(",") if c.strip()]

        delta = iv = None
        gsrc = None
        if self.compute_greeks and atm:
            T = max(dte, 0) / TRADING_DAY_YEAR
            if T == 0:
                T = 0.5 / TRADING_DAY_YEAR  # intraday 0DTE: fraction of a day
            iv = implied_vol(px, atm, strike, T, right)
            delta = bs_delta(atm, strike, T, iv, right) if iv else None
            gsrc = "COMPUTED" if delta is not None else None

        ts_et = ts.astimezone(ET)
        return Print(
            ts_utc=ts.astimezone(timezone.utc).isoformat(timespec="milliseconds"),
            ts_et=ts_et.replace(tzinfo=None).isoformat(timespec="milliseconds"),
            session_date=ts_et.date().isoformat(),
            root=self.symbol, expiry=exp.isoformat(),
            strike=float(strike), right=right.upper()[:1],
            trade_px=px, size=sz, exchange=exch, condition_codes=conds,
            underlying_px=atm, dte=dte, bid=bid, ask=ask,
            delta=delta, iv=iv, greeks_src=gsrc,
            data_source="thetadata",
        )

    def _parse_ts(self, row: Dict[str, str], cm: Dict[str, str],
                  session: date) -> Optional[datetime]:
        # preferred: a full ISO timestamp column
        raw = row.get(cm.get("timestamp", ""))
        if raw:
            dt = _parse_iso(raw)
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=self.tz)
                return dt
        # fallback: date + ms_of_day
        ms = _i(row.get(cm.get("ms_of_day", "")))
        if ms is not None:
            d = _parse_date(row.get(cm.get("date", "")) or "") or session
            return datetime(d.year, d.month, d.day, tzinfo=self.tz) + \
                timedelta(milliseconds=ms)
        return None

    def _log_unmapped(self, row: Dict[str, str]) -> None:
        if self._warned_unmapped:
            return
        self._warned_unmapped = True
        mapped = set(self._colmap.values()) if self._colmap else set()
        unmapped = [k for k in row.keys() if k not in mapped]
        logger.info("thetadata trade_quote header=%s", list(row.keys()))
        if unmapped:
            logger.info("thetadata UNMAPPED columns (pin aliases): %s", unmapped)
        for need in ("timestamp", "price", "size", "bid", "ask"):
            if need not in (self._colmap or {}):
                logger.warning("thetadata: could not map '%s' — check header", need)


# ── small parsers ─────────────────────────────────────────────────────────────

def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _strike(k: float) -> str:
    # v3 accepts plain strike; send integer when whole to match listing style
    return str(int(k)) if float(k).is_integer() else str(k)


def _parse_date(s: str) -> Optional[date]:
    s = (s or "").strip().strip('"')
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s[:10] if fmt == "%Y-%m-%d" else s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_iso(s: str) -> Optional[datetime]:
    s = (s or "").strip().strip('"')
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _trading_dte(session: date, expiry: date, all_exps: Sequence[date]) -> int:
    """Trading-day DTE using the SPY expiration calendar (daily expiries ≈ every
    trading day) as the session calendar — holiday-safe."""
    cal = [e for e in all_exps if session <= e <= expiry]
    return max(len(cal) - 1, 0)


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v) -> Optional[int]:
    f = _f(v)
    return int(f) if f is not None else None
