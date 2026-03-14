"""IBKR connection manager for SPY options bot.

Uses ib_insync (same library as the MES bot).
Handles initial connection and automatic reconnection with exponential backoff.
"""
from __future__ import annotations

import asyncio
from typing import Callable

from ib_insync import IB

from logger import logger


class IBKRConnection:
    """Wraps ib_insync IB instance with reconnect logic."""

    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        max_retries: int = 10,
        base_delay: float = 2.0,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.max_retries = max_retries
        self.base_delay = base_delay

        self.ib = IB()
        self._reconnect_callbacks: list[Callable] = []
        self._disconnect_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to TWS / IB Gateway with retry logic."""
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Connecting to IBKR {self.host}:{self.port} "
                    f"clientId={self.client_id} (attempt {attempt + 1}/{self.max_retries})"
                )
                await self.ib.connectAsync(
                    host=self.host,
                    port=self.port,
                    clientId=self.client_id,
                    timeout=20,
                )
                logger.info("IBKR connected successfully")
                self._register_disconnect_handler()
                return
            except TimeoutError:
                delay = self.base_delay * (2 ** attempt)
                logger.warning(
                    f"Connection timed out. Retrying in {delay:.1f}s... "
                    f"(Is TWS/Gateway running on port {self.port}?)"
                )
                await asyncio.sleep(delay)
            except ConnectionRefusedError:
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"Connection refused on port {self.port}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            except Exception as exc:
                delay = self.base_delay * (2 ** attempt)
                logger.error(f"Connection error: {exc}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

        raise ConnectionError(
            f"Failed to connect to IBKR after {self.max_retries} attempts. "
            f"Verify TWS/Gateway is running on {self.host}:{self.port} "
            f"and clientId={self.client_id} is not already in use."
        )

    async def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    def is_connected(self) -> bool:
        return self.ib.isConnected()

    # ------------------------------------------------------------------
    # Reconnect
    # ------------------------------------------------------------------

    def _register_disconnect_handler(self) -> None:
        """Register an event handler to auto-reconnect on disconnect."""
        # Avoid duplicate registration
        try:
            self.ib.disconnectedEvent -= self._on_disconnected
        except Exception:
            pass
        self.ib.disconnectedEvent += self._on_disconnected

    def _on_disconnected(self) -> None:
        logger.warning("IBKR connection lost — scheduling reconnect")
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self._reconnect())

    async def _reconnect(self) -> None:
        logger.info("Starting reconnection sequence...")
        await asyncio.sleep(5)  # Brief pause before attempting reconnect
        try:
            await self.connect()
            for cb in self._reconnect_callbacks:
                try:
                    await cb()
                except Exception as exc:
                    logger.error(f"Reconnect callback error: {exc}")
        except ConnectionError as exc:
            logger.error(f"Reconnect failed: {exc}")

    def on_reconnect(self, callback: Callable) -> None:
        """Register a coroutine to be called after successful reconnect."""
        self._reconnect_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Convenience pass-throughs
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying IB instance."""
        return getattr(self.ib, name)
