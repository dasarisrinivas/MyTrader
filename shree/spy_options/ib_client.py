"""IB Client Portal REST API wrapper for SPY options data.

Requires IB Client Portal Gateway running on localhost (port 5000 by default).
The gateway must be authenticated via browser login before this client can work.
Self-signed SSL certificate is expected — verify_ssl must be False in config.

Key endpoints used:
  POST /v1/api/iserver/secdef/search   — resolve SPY underlying conid
  GET  /v1/api/iserver/secdef/strikes  — list strikes for an expiry month
  GET  /v1/api/iserver/secdef/info     — get option contract conid
  GET  /v1/api/iserver/marketdata/snapshot — live bid/ask/last/volume
  POST /v1/api/tickle                  — session keep-alive

IB rate limit: 10 requests/second globally. The client throttles accordingly.
"""
from __future__ import annotations

import asyncio
import ssl
from typing import Any, Dict, List, Optional

import aiohttp

from ..config.spy_options import SpyOptionsIBConfig
from ..utils.logger import logger


class IBOptionsClient:
    """Async IB Client Portal REST API client for SPY options data."""

    def __init__(self, cfg: SpyOptionsIBConfig) -> None:
        self._cfg = cfg
        schema = "https" if cfg.use_https else "http"
        self._base = f"{schema}://{cfg.host}:{cfg.port}/v1/api"
        self._session: Optional[aiohttp.ClientSession] = None
        self._spy_conid: Optional[int] = None
        self._tickle_task: Optional[asyncio.Task] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Open HTTP session and start keep-alive tickle loop."""
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        if not self._cfg.verify_ssl:
            ssl_ctx.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        timeout = aiohttp.ClientTimeout(total=self._cfg.request_timeout_s)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        ok = await self._tickle()
        if not ok:
            logger.warning(
                "IB Client Portal tickle failed on startup — is the gateway "
                "running and authenticated at {}?",
                self._base,
            )

        self._tickle_task = asyncio.create_task(self._tickle_loop())
        logger.info("IBOptionsClient started → {}", self._base)

    async def close(self) -> None:
        """Shut down background tasks and HTTP session."""
        if self._tickle_task and not self._tickle_task.done():
            self._tickle_task.cancel()
            try:
                await self._tickle_task
            except asyncio.CancelledError:
                pass
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Keep-alive ────────────────────────────────────────────────────────────

    async def _tickle(self) -> bool:
        """POST /tickle to keep the IB session alive. Returns True on success."""
        try:
            async with self._session.post(f"{self._base}/tickle") as r:
                return r.status == 200
        except Exception as exc:
            logger.debug("Tickle error: {}", exc)
            return False

    async def _tickle_loop(self) -> None:
        """Background task: POST /tickle every tickle_interval_s seconds."""
        while True:
            await asyncio.sleep(self._cfg.tickle_interval_s)
            await self._tickle()

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    async def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        """GET request with retry. Returns parsed JSON or None on failure."""
        url = f"{self._base}{path}"
        for attempt in range(self._cfg.max_retries + 1):
            try:
                async with self._session.get(url, params=params) as r:
                    if r.status == 200:
                        return await r.json(content_type=None)
                    text = await r.text()
                    logger.warning("GET {} → HTTP {} (attempt {}): {}", path, r.status, attempt + 1, text[:120])
            except asyncio.TimeoutError:
                logger.warning("GET {} timed out (attempt {})", path, attempt + 1)
            except Exception as exc:
                logger.warning("GET {} error (attempt {}): {}", path, attempt + 1, exc)
            if attempt < self._cfg.max_retries:
                await asyncio.sleep(self._cfg.retry_delay_s)
        return None

    async def _post(self, path: str, body: Dict) -> Any:
        """POST request with retry. Returns parsed JSON or None on failure."""
        url = f"{self._base}{path}"
        for attempt in range(self._cfg.max_retries + 1):
            try:
                async with self._session.post(url, json=body) as r:
                    if r.status == 200:
                        return await r.json(content_type=None)
                    text = await r.text()
                    logger.warning("POST {} → HTTP {} (attempt {}): {}", path, r.status, attempt + 1, text[:120])
            except asyncio.TimeoutError:
                logger.warning("POST {} timed out (attempt {})", path, attempt + 1)
            except Exception as exc:
                logger.warning("POST {} error (attempt {}): {}", path, attempt + 1, exc)
            if attempt < self._cfg.max_retries:
                await asyncio.sleep(self._cfg.retry_delay_s)
        return None

    # ── Contract lookup ───────────────────────────────────────────────────────

    async def get_spy_conid(self) -> Optional[int]:
        """Resolve and cache SPY's underlying conid via secdef/search."""
        if self._spy_conid:
            return self._spy_conid
        data = await self._post(
            "/iserver/secdef/search",
            {"symbol": "SPY", "secType": "STK"},
        )
        if not data or not isinstance(data, list):
            return None
        conid = data[0].get("conid")
        if conid:
            self._spy_conid = int(conid)
            logger.info("SPY underlying conid resolved: {}", self._spy_conid)
        return self._spy_conid

    async def get_strikes(
        self,
        spy_conid: int,
        month: str,
        exchange: str = "SMART",
    ) -> Dict[str, List[float]]:
        """Return available strikes for a given expiry month.

        Args:
            month: IB month format, e.g. "APR26" for April 2026.

        Returns:
            {"call": [strike, ...], "put": [strike, ...]}
        """
        data = await self._get(
            "/iserver/secdef/strikes",
            params={
                "conid": spy_conid,
                "secType": "OPT",
                "month": month,
                "exchange": exchange,
            },
        )
        if not data:
            return {"call": [], "put": []}
        return data

    async def get_option_conid(
        self,
        spy_conid: int,
        month: str,
        strike: float,
        right: str,
        exchange: str = "SMART",
    ) -> Optional[int]:
        """Return conid for a specific option contract (strike + right + expiry).

        Args:
            right: "C" for call, "P" for put.
        """
        data = await self._get(
            "/iserver/secdef/info",
            params={
                "conid": spy_conid,
                "secType": "OPT",
                "month": month,
                "strike": strike,
                "right": right,
                "exchange": exchange,
            },
        )
        if not data:
            return None
        if isinstance(data, list) and data:
            return int(data[0]["conid"]) if data[0].get("conid") else None
        if isinstance(data, dict):
            return int(data["conid"]) if data.get("conid") else None
        return None

    # ── Market data ───────────────────────────────────────────────────────────

    async def subscribe_snapshot(
        self,
        conids: List[int],
        fields: str = "31,84,85,86,87,88",
    ) -> None:
        """Pre-flight call to initialise the IB market-data stream.

        IB requires the first snapshot call to "subscribe" before data flows.
        This call's response is ignored; wait snapshot_preflight_delay_s before
        calling get_snapshot() to receive actual data.
        """
        if not conids:
            return
        conid_str = ",".join(str(c) for c in conids)
        await self._get(
            "/iserver/marketdata/snapshot",
            params={"conids": conid_str, "fields": fields},
        )
        await asyncio.sleep(self._cfg.snapshot_preflight_delay_s)

    async def get_snapshot(
        self,
        conids: List[int],
        fields: str = "31,84,85,86,87,88",
    ) -> Dict[int, Dict]:
        """Return live snapshot data for the given conids.

        Field codes:
          31  = Last price
          84  = Bid
          85  = Bid size
          86  = Ask
          87  = Day volume
          88  = Ask size

        Returns:
            {conid: {field_code_str: value, ...}, ...}
        """
        if not conids:
            return {}

        results: Dict[int, Dict] = {}
        # IB snapshot endpoint handles ~50 conids per call safely
        for chunk_start in range(0, len(conids), 50):
            chunk = conids[chunk_start : chunk_start + 50]
            conid_str = ",".join(str(c) for c in chunk)
            data = await self._get(
                "/iserver/marketdata/snapshot",
                params={"conids": conid_str},
            )
            if data and isinstance(data, list):
                for item in data:
                    cid = item.get("conid")
                    if cid is not None:
                        results[int(cid)] = item

        return results

    async def get_spy_price(self, spy_conid: int) -> Optional[float]:
        """Return current SPY last/bid/ask mid price."""
        await self.subscribe_snapshot([spy_conid], fields="31,84,86")
        snap = await self.get_snapshot([spy_conid], fields="31,84,86")
        item = snap.get(spy_conid, {})
        for field_id in ("31", "84", "86"):
            val = item.get(field_id)
            if val:
                try:
                    return float(str(val).replace(",", "").strip())
                except (ValueError, TypeError):
                    continue
        return None

    async def get_vix(self, vix_conid: int) -> Optional[float]:
        """Return VIX index value from IB snapshot.

        Args:
            vix_conid: IB conid for the VIX index (configurable; default 13455763).
        """
        if not vix_conid:
            return None
        try:
            snap = await self.get_snapshot([vix_conid], fields="31,84,86")
            item = snap.get(vix_conid, {})
            for fid in ("31", "84", "86"):
                v = item.get(fid)
                if v:
                    return float(str(v).replace(",", "").strip())
        except Exception as exc:
            logger.debug("VIX fetch error: {}", exc)
        return None
