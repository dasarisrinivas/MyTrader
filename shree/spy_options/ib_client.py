"""IB Gateway (ib_insync) client for SPY options market data.

Connects to IB Gateway via the TWS socket API (same port as MES/Gold bots,
port 4001 live / 4002 paper). Uses ib_insync for async market data,
option chain lookup, Greeks, and VIX — no IB Client Portal REST API required.

No orders are ever placed — this is a read-only feed for signal generation.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ib_insync import IB, Index, Option, Stock

from ..config.spy_options import SpyOptionsIBConfig
from ..utils.logger import logger


def _safe_float(v: Any) -> float:
    """Convert value to float, returning 0.0 for None/NaN/invalid."""
    try:
        f = float(v)
        return 0.0 if (f != f) else f  # NaN check: nan != nan
    except (TypeError, ValueError):
        return 0.0


def _safe_int(v: Any) -> int:
    """Convert value to int, returning 0 for None/NaN/invalid."""
    try:
        f = float(v)
        return 0 if (f != f) else int(f)
    except (TypeError, ValueError):
        return 0


class IBOptionsClient:
    """ib_insync client for SPY option chain market data and Greeks.

    Connects to IB Gateway (port 4001/4002) — the same gateway used by
    MES and Gold bots. Client-id 5 keeps this connection separate.
    """

    def __init__(self, cfg: SpyOptionsIBConfig) -> None:
        self._cfg = cfg
        self._ib = IB()
        self._contract_cache: Dict[int, Any] = {}
        self._chain_params: Optional[Dict] = None
        self._spy_contract: Optional[Any] = None
        self._vix_contract: Optional[Any] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to IB Gateway and set live market data type."""
        try:
            await self._ib.connectAsync(
                self._cfg.ibkr_host,
                self._cfg.ibkr_port,
                clientId=self._cfg.ibkr_client_id,
                timeout=30,
            )
        except Exception as exc:
            logger.error(
                "Failed to connect to IB Gateway at {}:{} — {}",
                self._cfg.ibkr_host,
                self._cfg.ibkr_port,
                exc,
            )
            raise

        self._ib.reqMarketDataType(1)  # 1=live, 2=frozen, 3=delayed
        logger.info(
            "IBOptionsClient connected → IB Gateway {}:{} (client_id={})",
            self._cfg.ibkr_host,
            self._cfg.ibkr_port,
            self._cfg.ibkr_client_id,
        )

    async def close(self) -> None:
        """Disconnect from IB Gateway."""
        if self._ib.isConnected():
            self._ib.disconnect()
        logger.info("IBOptionsClient disconnected")

    # ── SPY contract ──────────────────────────────────────────────────────────

    async def get_spy_conid(self) -> Optional[int]:
        """Qualify SPY stock contract and return its conId."""
        spy = Stock("SPY", "SMART", "USD")
        try:
            qualified = await self._ib.qualifyContractsAsync(spy)
        except Exception as exc:
            logger.error("Could not qualify SPY stock contract: {}", exc)
            return None

        if not qualified:
            return None

        self._spy_contract = qualified[0]
        self._contract_cache[qualified[0].conId] = qualified[0]
        logger.info("SPY qualified: conId={}", qualified[0].conId)
        return qualified[0].conId

    # ── Option chain parameters ───────────────────────────────────────────────

    async def _load_chain_params(self, spy_conid: int, exchange: str) -> None:
        """Fetch and cache reqSecDefOptParams (called once per session)."""
        if self._chain_params is not None:
            return

        logger.info("Fetching SPY option chain parameters from IB Gateway...")
        try:
            chains = await self._ib.reqSecDefOptParamsAsync(
                underlyingSymbol="SPY",
                futFopExchange="",
                underlyingSecType="STK",
                underlyingConId=spy_conid,
            )
        except Exception as exc:
            logger.error("reqSecDefOptParams failed: {}", exc)
            self._chain_params = {"expirations": [], "strikes": []}
            return

        if not chains:
            logger.error("reqSecDefOptParams returned empty chain list")
            self._chain_params = {"expirations": [], "strikes": []}
            return

        chain = (
            next((c for c in chains if c.exchange == exchange), None)
            or next((c for c in chains if c.exchange == "SMART"), None)
            or chains[0]
        )
        self._chain_params = {
            "expirations": sorted(chain.expirations),
            "strikes": sorted(chain.strikes),
        }
        logger.info(
            "Chain params loaded: {} expirations, {} strikes",
            len(self._chain_params["expirations"]),
            len(self._chain_params["strikes"]),
        )

    def _month_to_prefix(self, month: str) -> Optional[str]:
        """Convert IB month string "APR26" → YYYYMM prefix "202604"."""
        try:
            return datetime.strptime(month, "%b%y").strftime("%Y%m")
        except ValueError:
            logger.warning("Unrecognised IB month format: {}", month)
            return None

    def _best_expiry(self, month: str) -> Optional[str]:
        """Return the latest monthly expiry YYYYMMDD for a given month string."""
        if self._chain_params is None:
            return None
        prefix = self._month_to_prefix(month)
        if not prefix:
            return None
        candidates = [e for e in self._chain_params["expirations"] if e.startswith(prefix)]
        return max(candidates) if candidates else None

    async def get_strikes(
        self,
        spy_conid: int,
        month: str,
        exchange: str = "SMART",
    ) -> Dict[str, List[float]]:
        """Return available strikes for an expiry month."""
        await self._load_chain_params(spy_conid, exchange)
        if not self._chain_params or not self._best_expiry(month):
            logger.warning("No expirations found for month {} in IB chain", month)
            return {"call": [], "put": []}
        return {"call": self._chain_params["strikes"], "put": self._chain_params["strikes"]}

    async def get_option_conid(
        self,
        spy_conid: int,
        month: str,
        strike: float,
        right: str,
        exchange: str = "SMART",
    ) -> Optional[int]:
        """Qualify an option contract and cache it for later market data requests."""
        await self._load_chain_params(spy_conid, exchange)
        expiry = self._best_expiry(month)
        if not expiry:
            return None

        opt = Option("SPY", expiry, strike, right, exchange, multiplier="100", currency="USD")
        try:
            qualified = await self._ib.qualifyContractsAsync(opt)
        except Exception as exc:
            logger.debug("qualify failed for SPY {} {} {} {}: {}", month, expiry, strike, right, exc)
            return None

        if not qualified:
            return None

        contract = qualified[0]
        self._contract_cache[contract.conId] = contract
        return contract.conId

    # ── Market data: price snapshot (snapshot=True, no Greeks) ───────────────

    async def subscribe_snapshot(self, conids: List[int], fields: str = "") -> None:
        """No-op — ib_insync handles subscriptions via reqMktData internally."""

    async def get_snapshot(
        self,
        conids: List[int],
        fields: str = "",
    ) -> Dict[int, Dict]:
        """Price-only snapshot for SPY stock. Uses snapshot=True (fast, no Greeks).

        Field keys: "31"=last, "84"=bid, "85"=bid_size, "86"=ask, "87"=volume, "88"=ask_size
        """
        contracts = [self._contract_cache[c] for c in conids if c in self._contract_cache]
        if not contracts:
            return {}

        tickers = {
            c.conId: self._ib.reqMktData(c, genericTickList="", snapshot=True)
            for c in contracts
        }
        await asyncio.sleep(self._cfg.snapshot_wait_s)

        return {
            conid: {
                "31": _safe_float(t.last),
                "84": _safe_float(t.bid),
                "85": _safe_int(t.bidSize),
                "86": _safe_float(t.ask),
                "87": _safe_int(t.volume),
                "88": _safe_int(t.askSize),
            }
            for conid, t in tickers.items()
        }

    # ── Market data: Greeks snapshot (snapshot=False + explicit cancel) ───────

    async def get_snapshot_with_greeks(
        self,
        conids: List[int],
    ) -> Dict[int, Dict]:
        """Fetch price + Greeks for option contracts.

        Uses snapshot=False with genericTickList="100,101" to get modelGreeks
        and open interest. Explicitly cancels all subscriptions after reading
        to avoid exhausting IB's ~100 concurrent data line limit.

        Returns dict keyed by conid with fields:
          "31","84","85","86","87","88" — price/size (same as get_snapshot)
          "delta","gamma","theta","vega","impl_vol","open_interest" — Greeks
        """
        contracts = [self._contract_cache[c] for c in conids if c in self._contract_cache]
        if not contracts:
            return {}

        # Subscribe to live feed (snapshot=False) with Greek tick types
        tickers: Dict[int, Any] = {}
        for contract in contracts:
            tickers[contract.conId] = self._ib.reqMktData(
                contract,
                genericTickList="100,101",  # 100=option vol, 101=OI
                snapshot=False,
                regulatorySnapshot=False,
            )

        # Greeks arrive asynchronously from IB's option model — needs extra time
        await asyncio.sleep(self._cfg.greeks_wait_s)

        result: Dict[int, Dict] = {}
        for conid, ticker in tickers.items():
            greeks = ticker.modelGreeks  # OptionComputation or None
            oi = ticker.optionOpenInterest

            result[conid] = {
                "31": _safe_float(ticker.last),
                "84": _safe_float(ticker.bid),
                "85": _safe_int(ticker.bidSize),
                "86": _safe_float(ticker.ask),
                "87": _safe_int(ticker.volume),
                "88": _safe_int(ticker.askSize),
                "delta":         _safe_float(greeks.delta    if greeks else None),
                "gamma":         _safe_float(greeks.gamma    if greeks else None),
                "theta":         _safe_float(greeks.theta    if greeks else None),
                "vega":          _safe_float(greeks.vega     if greeks else None),
                "impl_vol":      _safe_float(greeks.impliedVol if greeks else None),
                "open_interest": _safe_int(oi),
            }

        # CRITICAL: cancel all live subscriptions to free data lines
        for contract in contracts:
            try:
                self._ib.cancelMktData(contract)
            except Exception:
                pass

        return result

    # ── SPY price ──────────────────────────────────────────────────────────────

    async def get_spy_price(self, spy_conid: int) -> Optional[float]:
        """Return current SPY price (last, close, or bid/ask mid)."""
        if not self._spy_contract:
            await self.get_spy_conid()
        if not self._spy_contract:
            return None

        ticker = self._ib.reqMktData(self._spy_contract, genericTickList="", snapshot=True)
        await asyncio.sleep(self._cfg.snapshot_wait_s)

        for val in (
            ticker.last,
            ticker.close,
            (ticker.bid + ticker.ask) / 2 if (ticker.bid and ticker.ask) else None,
        ):
            if val is not None:
                f = _safe_float(val)
                if f > 0:
                    return f
        return None

    # ── VIX ───────────────────────────────────────────────────────────────────

    async def _get_vix_contract(self) -> Optional[Any]:
        """Qualify and cache the VIX Index contract."""
        if self._vix_contract is not None:
            return self._vix_contract
        try:
            qualified = await self._ib.qualifyContractsAsync(Index("VIX", "CBOE"))
            if qualified:
                self._vix_contract = qualified[0]
        except Exception as exc:
            logger.debug("VIX contract qualify error: {}", exc)
        return self._vix_contract

    async def get_vix(self) -> Optional[float]:
        """Return VIX index level via IB Gateway."""
        contract = await self._get_vix_contract()
        if not contract:
            return None
        try:
            ticker = self._ib.reqMktData(contract, genericTickList="", snapshot=True)
            await asyncio.sleep(self._cfg.snapshot_wait_s)
            for val in (ticker.last, ticker.close):
                f = _safe_float(val)
                if f > 0:
                    return f
        except Exception as exc:
            logger.debug("VIX fetch error: {}", exc)
        return None

    async def get_vix_52w_range(self) -> Optional[Tuple[float, float]]:
        """Fetch 1 year of daily VIX bars and return (52w_low, 52w_high).

        Called once at startup and cached by the manager for IV rank computation.
        """
        contract = await self._get_vix_contract()
        if not contract:
            logger.warning("Cannot fetch VIX 52w range — VIX contract unavailable")
            return None
        try:
            bars = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr="1 Y",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                keepUpToDate=False,
                timeout=30,
            )
            if not bars:
                return None
            closes = [b.close for b in bars if b.close and b.close > 0]
            if not closes:
                return None
            return float(min(closes)), float(max(closes))
        except Exception as exc:
            logger.warning("VIX 52w range fetch failed: {}", exc)
            return None

    # ── SPY 5-minute bars ─────────────────────────────────────────────────────

    async def get_spy_bars_5m(self) -> List[Dict[str, Any]]:
        """Return today's 5-minute OHLCV bars for SPY (newest last).

        Used by RegimeDetector for EMA, ATR, and VWAP computation.
        Returns [] on failure — caller should handle gracefully.
        """
        if not self._spy_contract:
            return []
        try:
            bars = await self._ib.reqHistoricalDataAsync(
                self._spy_contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=True,
                keepUpToDate=False,
                timeout=15,
            )
            if not bars:
                return []
            return [
                {
                    "date":   b.date if isinstance(b.date, datetime) else datetime.utcnow(),
                    "open":   float(b.open),
                    "high":   float(b.high),
                    "low":    float(b.low),
                    "close":  float(b.close),
                    "volume": int(b.volume),
                }
                for b in bars
            ]
        except Exception as exc:
            logger.warning("SPY 5m bars fetch failed: {}", exc)
            return []
