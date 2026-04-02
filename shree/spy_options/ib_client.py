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
        self._keepalive_task: Optional[asyncio.Task] = None
        self._reconnecting = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to IB Gateway and set market data type.

        Tries live data (type 1) first. If the account lacks the required
        ARCA/TOP subscription for SPY stock, IB returns error 10089. We
        register an error handler that automatically falls back to delayed
        data (type 3, 15-min delay) so the bot can still operate.
        """
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

        # Register error handler to detect missing market data subscriptions
        self._tried_delayed_fallback = False

        def _on_error(reqId, errorCode, errorString, *args):
            if errorCode == 10089 and not self._tried_delayed_fallback:
                self._tried_delayed_fallback = True
                logger.warning(
                    "Error 10089: live market data not subscribed for SPY. "
                    "Falling back to delayed data (15-min delay). "
                    "To fix: IB Account → Settings → Market Data Subscriptions → "
                    "1) Subscribe to 'US Securities Snapshot and Futures Value Bundle' "
                    "2) Add 'US Equity and Options Add-On Streaming Bundle' for live data",
                )
                self._ib.reqMarketDataType(3)  # 3=delayed

        self._ib.errorEvent += _on_error

        self._ib.reqMarketDataType(1)  # 1=live, 2=frozen, 3=delayed
        logger.info(
            "IBOptionsClient connected → IB Gateway {}:{} (client_id={})",
            self._cfg.ibkr_host,
            self._cfg.ibkr_port,
            self._cfg.ibkr_client_id,
        )

        # Start keepalive task to prevent idle disconnects
        self._keepalive_task = asyncio.ensure_future(self._keepalive_loop())

        # Auto-reconnect on unexpected disconnect
        self._ib.disconnectedEvent += self._on_disconnect

    async def _keepalive_loop(self) -> None:
        """Ping IB Gateway every 30s to keep the connection alive.

        IB Gateway drops idle connections. The MES bot has a similar
        keepalive — without it, the SPY Options bot disconnects between polls.
        """
        while True:
            await asyncio.sleep(30)
            try:
                if self._ib.isConnected():
                    # reqCurrentTime is a lightweight ping that keeps the socket alive
                    self._ib.reqCurrentTime()
                else:
                    logger.warning("Keepalive: IB not connected, triggering reconnect")
                    await self._reconnect()
            except Exception as exc:
                logger.debug("Keepalive ping failed: {}", exc)

    def _on_disconnect(self) -> None:
        """Handle unexpected IB Gateway disconnection."""
        if not self._reconnecting:
            logger.warning("IB Gateway disconnected — scheduling reconnect")
            asyncio.ensure_future(self._reconnect())

    async def _reconnect(self) -> None:
        """Reconnect to IB Gateway and reset cached state."""
        if self._reconnecting:
            return
        self._reconnecting = True
        try:
            # Clear stale state so fresh data is fetched after reconnect
            self._chain_params = None
            self._spy_ticker = None

            for attempt in range(1, 6):  # up to 5 retries
                try:
                    if self._ib.isConnected():
                        self._ib.disconnect()
                    await asyncio.sleep(min(attempt * 5, 30))  # backoff: 5s, 10s, 15s, 20s, 25s
                    await self._ib.connectAsync(
                        self._cfg.ibkr_host,
                        self._cfg.ibkr_port,
                        clientId=self._cfg.ibkr_client_id,
                        timeout=30,
                    )
                    self._ib.reqMarketDataType(1)
                    logger.info(
                        "IBOptionsClient reconnected (attempt {}/5)", attempt,
                    )
                    return
                except Exception as exc:
                    logger.warning(
                        "Reconnect attempt {}/5 failed: {}", attempt, exc,
                    )
            logger.error("All 5 reconnect attempts failed — bot will retry on next keepalive")
        finally:
            self._reconnecting = False

    async def close(self) -> None:
        """Disconnect from IB Gateway, cancelling any active subscriptions."""
        # Cancel keepalive task
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None

        if self._ib.isConnected():
            # Cancel persistent SPY streaming subscription
            if hasattr(self, "_spy_ticker") and self._spy_ticker is not None:
                try:
                    self._ib.cancelMktData(self._spy_contract)
                except Exception:
                    pass
                self._spy_ticker = None
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

        # IB returns MULTIPLE entries per exchange (e.g. two "SMART" entries:
        # one with 1 expiration/strike, another with 36/436). Pick the one
        # with the most expirations for the target exchange.
        candidates = [c for c in chains if c.exchange == exchange]
        if not candidates:
            candidates = [c for c in chains if c.exchange == "SMART"]
        if not candidates:
            candidates = chains
        chain = max(candidates, key=lambda c: len(c.expirations))
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
        if not self._ib.isConnected():
            logger.warning("get_snapshot_with_greeks: not connected, triggering reconnect")
            await self._reconnect()
            if not self._ib.isConnected():
                return {}
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
            # ib_insync uses callOpenInterest / putOpenInterest (not optionOpenInterest)
            oi = getattr(ticker, "callOpenInterest", None) or getattr(ticker, "putOpenInterest", None)

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

    # ── SPY price (persistent streaming subscription) ───────────────────────

    async def get_spy_price(self, spy_conid: int) -> Optional[float]:
        """Return current SPY price from a persistent streaming subscription.

        On first call, subscribes to live streaming data for SPY (snapshot=False).
        Subsequent calls just read the latest value from the ticker — no new
        IB requests needed. This is more efficient than re-requesting a snapshot
        every 60s and ensures the bot always has the freshest price.

        If live data isn't available (error 10089), falls back to delayed data.
        """
        if not self._ib.isConnected():
            logger.warning("get_spy_price: not connected, triggering reconnect")
            await self._reconnect()
            if not self._ib.isConnected():
                return None
        if not self._spy_contract:
            await self.get_spy_conid()
        if not self._spy_contract:
            return None

        # Start persistent subscription on first call
        if not hasattr(self, "_spy_ticker") or self._spy_ticker is None:
            self._spy_ticker = self._ib.reqMktData(
                self._spy_contract, genericTickList="", snapshot=False,
            )
            # Wait for first data to arrive
            await asyncio.sleep(self._cfg.snapshot_wait_s)

        ticker = self._spy_ticker

        # Try live fields first
        for val in (
            ticker.last,
            ticker.close,
            (ticker.bid + ticker.ask) / 2 if (ticker.bid and ticker.ask) else None,
        ):
            if val is not None:
                f = _safe_float(val)
                if f > 0:
                    return f

        # If live failed, try delayed fields (populated after 10089 fallback)
        for val in (
            getattr(ticker, "delayedLast", None),
            getattr(ticker, "delayedClose", None),
        ):
            if val is not None:
                f = _safe_float(val)
                if f > 0:
                    logger.debug("SPY price from delayed data: {:.2f}", f)
                    return f

        # If still nothing and we haven't tried delayed mode yet, switch
        if not self._tried_delayed_fallback:
            self._tried_delayed_fallback = True
            logger.warning(
                "SPY streaming returned no price — switching to delayed data and resubscribing"
            )
            try:
                self._ib.cancelMktData(self._spy_contract)
            except Exception:
                pass
            self._ib.reqMarketDataType(3)
            self._spy_ticker = self._ib.reqMktData(
                self._spy_contract, genericTickList="", snapshot=False,
            )
            await asyncio.sleep(self._cfg.snapshot_wait_s + 2)

            ticker = self._spy_ticker
            for val in (
                ticker.last, ticker.close,
                getattr(ticker, "delayedLast", None),
                getattr(ticker, "delayedClose", None),
                (ticker.bid + ticker.ask) / 2 if (ticker.bid and ticker.ask) else None,
            ):
                if val is not None:
                    f = _safe_float(val)
                    if f > 0:
                        logger.info("SPY price recovered via delayed streaming: {:.2f}", f)
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
        """Return VIX index level via IB Gateway.

        VIX is a CBOE index — it has no 'last' trade price. The primary
        value comes from ``ticker.marketPrice()`` (IB's best effort), with
        fallbacks to ``last``, ``close``, and bid/ask midpoint.
        """
        contract = await self._get_vix_contract()
        if not contract:
            logger.warning("VIX contract unavailable — cannot fetch spot VIX")
            return None
        ticker = None
        try:
            ticker = self._ib.reqMktData(contract, genericTickList="", snapshot=True)
            await asyncio.sleep(self._cfg.snapshot_wait_s)

            # Primary: IB's computed market price (works best for indices)
            mp = _safe_float(ticker.marketPrice())
            if mp > 0:
                return mp

            # Fallback chain: last → close → bid/ask midpoint
            for val in (ticker.last, ticker.close):
                f = _safe_float(val)
                if f > 0:
                    return f

            bid = _safe_float(ticker.bid)
            ask = _safe_float(ticker.ask)
            if bid > 0 and ask > 0:
                return round((bid + ask) / 2, 2)

            # All attempts failed — log diagnostics
            logger.warning(
                "VIX snapshot empty: marketPrice={} last={} close={} bid={} ask={}",
                ticker.marketPrice(), ticker.last, ticker.close, ticker.bid, ticker.ask,
            )

        except Exception as exc:
            logger.warning("VIX fetch error: {}", exc)
        finally:
            # Always cancel the snapshot subscription to avoid stale tickers
            if ticker is not None:
                try:
                    self._ib.cancelMktData(contract)
                except Exception:
                    pass
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
