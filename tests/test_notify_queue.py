"""Telegram must never sit in the order path (AUG 3 2026).

The dispatch-cap audit found `_send_signal()` awaited at manager.py:1720,
immediately before `maybe_execute()` at :1746 — so a slow or rate-limited
Telegram delayed live entries. These tests pin the fix: enqueue is O(1) and
non-blocking under every Telegram failure mode, ordering is preserved, and no
alert is duplicated or lost.
"""
from __future__ import annotations

import asyncio

import pytest

from shree.spy_options.notify_queue import NotifyQueue

pytestmark = pytest.mark.asyncio


# ── fake Telegram back-ends, one per failure mode ────────────────────────────

class Recorder:
    """Healthy notifier that timestamps each delivery."""

    def __init__(self):
        self.messages = []

    async def send_message(self, text, parse_mode="HTML"):
        self.messages.append((text, asyncio.get_event_loop().time()))
        return True


class Hangs(Recorder):
    """Telegram stalls — models the 10s aiohttp ClientTimeout."""

    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    async def send_message(self, text, parse_mode="HTML"):
        await asyncio.sleep(self.delay)
        return await super().send_message(text, parse_mode)


class Raises(Recorder):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc
        self.attempts = 0

    async def send_message(self, text, parse_mode="HTML"):
        self.attempts += 1
        raise self.exc


class RateLimited(Recorder):
    """HTTP 429 — TelegramNotifier logs and returns False, does not raise."""

    def __init__(self):
        super().__init__()
        self.attempts = 0

    async def send_message(self, text, parse_mode="HTML"):
        self.attempts += 1
        return False


async def _elapsed(coro):
    t0 = asyncio.get_event_loop().time()
    await coro
    return asyncio.get_event_loop().time() - t0


# ── 1. enqueue never blocks, under every failure mode ────────────────────────

@pytest.mark.parametrize("backend", [
    Hangs(30.0),                                    # 30s stall
    Hangs(10.0),                                    # full aiohttp timeout
    Raises(asyncio.TimeoutError()),                 # timeout exception
    Raises(ConnectionError("network unreachable")), # network failure
    Raises(RuntimeError("boom")),                   # generic exception
    RateLimited(),                                  # HTTP 429
])
async def test_enqueue_never_blocks(backend):
    q = NotifyQueue(backend)
    q.start()
    took = await _elapsed(q.send_message("alert"))
    assert took < 0.05, f"enqueue blocked for {took:.3f}s"
    q._task.cancel()


async def test_order_submission_is_not_delayed_by_a_30s_telegram_stall():
    """The regression under test: a 30s Telegram hang must not push out the
    simulated order submission that follows the alert."""
    q = NotifyQueue(Hangs(30.0))
    q.start()
    submitted_at = None

    async def dispatch_one():
        nonlocal submitted_at
        await q.send_message("SIGNAL")          # was: await telegram.send_message
        submitted_at = asyncio.get_event_loop().time()   # maybe_execute()

    t0 = asyncio.get_event_loop().time()
    await asyncio.wait_for(dispatch_one(), timeout=1.0)
    assert submitted_at - t0 < 0.05
    q._task.cancel()


# ── 2. failures cannot prevent execution or stall the queue ──────────────────

async def test_failed_send_does_not_head_of_line_block():
    """One bad message must not stop the messages behind it."""
    class FailsFirst(Recorder):
        async def send_message(self, text, parse_mode="HTML"):
            if text == "bad":
                raise RuntimeError("nope")
            return await super().send_message(text, parse_mode)

    be = FailsFirst()
    q = NotifyQueue(be)
    q.start()
    for m in ("bad", "good1", "good2"):
        await q.send_message(m)
    assert await q.drain_now(timeout=2.0)
    assert [m for m, _ in be.messages] == ["good1", "good2"]
    assert q.stats["failed"] == 1
    q._task.cancel()


async def test_consumer_survives_every_exception_type():
    q = NotifyQueue(Raises(RuntimeError("x")))
    q.start()
    for i in range(5):
        await q.send_message(f"m{i}")
    assert await q.drain_now(timeout=2.0)
    assert not q._task.done(), "consumer died — later alerts would be lost"
    q._task.cancel()


# ── 3. ordering, no duplicates, no losses ────────────────────────────────────

async def test_delivery_order_matches_enqueue_order():
    be = Recorder()
    q = NotifyQueue(be)
    q.start()
    expected = [f"msg{i}" for i in range(50)]
    for m in expected:
        await q.send_message(m)
    assert await q.drain_now(timeout=5.0)
    assert [m for m, _ in be.messages] == expected
    q._task.cancel()


async def test_signal_alert_precedes_order_placed_alert():
    """Manager enqueues the signal alert, then executor enqueues ORDER PLACED
    through the same queue. Delivery order must match that sequence."""
    be = Recorder()
    q = NotifyQueue(be)
    q.start()
    await q.send_message("SIGNAL CALL_SWEEP")     # manager._send_signal
    await q.send_message("ORDER PLACED")          # executor._notify
    assert await q.drain_now(timeout=2.0)
    assert [m for m, _ in be.messages] == ["SIGNAL CALL_SWEEP", "ORDER PLACED"]
    q._task.cancel()


async def test_no_duplicates_and_none_missing():
    be = Recorder()
    q = NotifyQueue(be)
    q.start()
    for i in range(100):
        await q.send_message(f"m{i}")
    assert await q.drain_now(timeout=5.0)
    got = [m for m, _ in be.messages]
    assert len(got) == 100 and len(set(got)) == 100
    assert q.stats["sent"] == 100
    q._task.cancel()


async def test_close_flushes_pending_alerts():
    be = Hangs(0.01)
    q = NotifyQueue(be)
    q.start()
    for i in range(10):
        await q.send_message(f"m{i}")
    await q.close(flush_timeout=5.0)
    assert len(be.messages) == 10, "shutdown dropped queued alerts"


# ── 4. it is substitutable for TelegramNotifier ──────────────────────────────

async def test_executor_is_wired_to_the_queue_not_the_raw_notifier():
    """executor.py still reads `await self._telegram.send_message(msg)`, so the
    non-blocking guarantee depends entirely on the manager INJECTING the queue.
    If someone reverts that argument, Telegram silently re-enters the order
    path with no other test failing."""
    import inspect
    from shree.spy_options import manager as mgr
    src = inspect.getsource(mgr.SpyOptionsManager.__init__)
    assert "telegram=self._notify_q" in src, \
        "executor must be constructed with the NotifyQueue, not self._telegram"


async def test_executor_notify_through_queue_does_not_block():
    """Drive the executor's real _notify() with a stalling backend behind the
    queue: it must return immediately."""
    from shree.spy_options.executor import SpyOptionsExecutor
    q = NotifyQueue(Hangs(30.0))
    q.start()
    ex = SpyOptionsExecutor.__new__(SpyOptionsExecutor)   # no IB connection
    ex._telegram = q
    took = await _elapsed(ex._notify("ORDER PLACED"))
    assert took < 0.05, f"executor._notify blocked for {took:.3f}s"
    q._task.cancel()


async def test_duck_types_telegram_notifier():
    """executor.py is unmodified — it calls `await self._telegram.send_message`
    on whatever it was handed, so the signature must match exactly."""
    import inspect
    from shree.utils.telegram_notifier import TelegramNotifier
    real = inspect.signature(TelegramNotifier.send_message).parameters
    ours = inspect.signature(NotifyQueue.send_message).parameters
    assert list(ours) == list(real)
    assert asyncio.iscoroutinefunction(NotifyQueue.send_message)
