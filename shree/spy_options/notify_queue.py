"""Non-blocking Telegram delivery — keeps notifications OFF the order path.

AUG 3 2026 (dispatch-cap audit). `_send_signal()` was awaited at
manager.py:1720, immediately BEFORE `maybe_execute()` at :1746. Telegram was
therefore in the critical path of every live entry: `send_message` has a 10s
aiohttp timeout and no rate limiting, and Telegram allows ~20 msg/min per chat,
so a burst of alerts could delay order submission by seconds per signal.

Fix: this class duck-types `TelegramNotifier.send_message`, so it can be
substituted anywhere one is used. `send_message` formats nothing and touches no
socket — it appends to an in-memory FIFO and returns. A single background
consumer performs the real network sends.

Ordering guarantee: ONE consumer draining ONE FIFO queue. Messages are
delivered in exactly the order they were enqueued, which is the order the old
awaited code sent them in. Enqueue happens at the same call sites and in the
same sequence as before, so observable message order is unchanged.

The queue is deliberately UNBOUNDED: dropping a message would mean a missing
alert, and the dispatch cap already bounds volume per session. Depth is logged
so a stuck consumer is visible.

The consumer can never die: every iteration is individually guarded, and a
failed send is logged and dropped rather than retried, so one bad message can
never head-of-line block the rest.
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from ..utils.logger import logger

# Log a warning once the backlog passes this — indicates Telegram is degraded.
_DEPTH_WARN = 25


class NotifyQueue:
    """FIFO, non-blocking Telegram front-end. Substitutable for TelegramNotifier."""

    def __init__(self, telegram: Any) -> None:
        self._tg = telegram
        self._q: "asyncio.Queue[str]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._sent = 0
        self._failed = 0
        self._warned_depth = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._drain())

    async def close(self, flush_timeout: float = 10.0) -> None:
        """Drain what is queued, then stop. Best-effort — never raises."""
        try:
            await asyncio.wait_for(self._q.join(), timeout=flush_timeout)
        except Exception:
            logger.warning(
                "NotifyQueue: {} message(s) undelivered at shutdown",
                self._q.qsize(),
            )
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # ── producer side (never blocks, never raises) ───────────────────────────

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Enqueue and return. Signature matches TelegramNotifier.send_message.

        `async` only so existing `await notifier.send_message(...)` call sites
        work untouched — this yields at most one event-loop tick and performs
        no I/O.
        """
        self.enqueue(text)
        return True

    def enqueue(self, text: str) -> None:
        try:
            self._q.put_nowait(text)
            depth = self._q.qsize()
            if depth > _DEPTH_WARN and not self._warned_depth:
                self._warned_depth = True
                logger.warning(
                    "NotifyQueue backlog {} messages — Telegram degraded; "
                    "trading is UNAFFECTED", depth,
                )
            elif depth <= 1:
                self._warned_depth = False
        except Exception as exc:                     # pragma: no cover
            logger.warning("NotifyQueue enqueue failed (alert dropped): {}", exc)

    # ── consumer side ────────────────────────────────────────────────────────

    async def _drain(self) -> None:
        while True:
            text = await self._q.get()
            try:
                await self._tg.send_message(text)
                self._sent += 1
            except asyncio.CancelledError:
                self._q.task_done()
                raise
            except Exception as exc:
                # Swallow: a failed alert must never stall the queue behind it.
                self._failed += 1
                logger.warning("NotifyQueue send failed (dropped): {}", exc)
            finally:
                try:
                    self._q.task_done()
                except Exception:
                    pass

    # ── introspection (tests / diagnostics) ──────────────────────────────────

    @property
    def depth(self) -> int:
        return self._q.qsize()

    @property
    def stats(self) -> dict:
        return {"sent": self._sent, "failed": self._failed, "depth": self.depth}

    async def drain_now(self, timeout: float = 5.0) -> bool:
        """Wait until the backlog is delivered. Test/diagnostic helper only."""
        try:
            await asyncio.wait_for(self._q.join(), timeout=timeout)
            return True
        except Exception:
            return False

    async def close_underlying(self) -> None:
        close = getattr(self._tg, "close", None)
        if close is not None:
            await close()

    def pending(self) -> List[str]:                  # pragma: no cover
        return list(self._q._queue)  # noqa: SLF001 — diagnostics only
