"""Regression locks for the three 2026-08-06 production fixes.

Each test names the incident it prevents recurring.
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import time as dtime

import pytest

from shree.spy_options import manager as mgr
from shree.spy_options import signal_engine as se


# ── FIX 1: watchdog must stop stall-checking before the bot goes quiet ──────

def test_stall_window_ends_before_bot_stops_polling():
    """2026-08-06 14:51:16 — the watchdog SIGKILLed a healthy bot because its
    stall window (to 14:55 CDT) outlived the bot's rth_stop_et (14:45 CDT)."""
    import sys, os
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import spy_watchdog as w
    from shree.utils.settings_loader import load_settings
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rth_stop = load_settings(os.path.join(root, "config.yaml")
                             ).spy_options.session.rth_stop_et       # ET
    h, m = map(int, rth_stop.split(":"))
    stop_cdt = dtime((h - 1) % 24, m)          # machine is Central, config is ET
    assert w.STALL_CHECK_END < stop_cdt, (
        f"stall window ends {w.STALL_CHECK_END} but bot polls until "
        f"{stop_cdt} CDT — post-RTH idle would be misread as a hang")


def test_stall_check_uses_the_new_window():
    import sys, os
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import spy_watchdog as w
    src = inspect.getsource(w.stalled)
    assert "STALL_CHECK_END" in src and "SESSION_END" not in src


def test_restart_window_unchanged():
    """Only stall DETECTION narrowed. A genuinely absent bot must still be
    restarted right up to the daily-stop job."""
    import sys, os
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import spy_watchdog as w
    assert w.SESSION_END == dtime(14, 55)


# ── FIX 2: Telegram must not silently drop an alert on an HTML parse error ──

class _Resp:
    def __init__(self, status, text):
        self.status, self._t = status, text
    async def text(self):
        return self._t
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False


class _Session:
    """First POST 400s on parse entities; a retry without parse_mode succeeds."""
    def __init__(self):
        self.posts = []
    def post(self, url, json=None):
        self.posts.append(json)
        if "parse_mode" in json:
            return _Resp(400, '{"ok":false,"error_code":400,"description":'
                              '"Bad Request: can\'t parse entities: '
                              'Unsupported start tag \\"=\\" at byte offset 132"}')
        return _Resp(200, '{"ok":true}')


@pytest.mark.asyncio
async def test_html_parse_failure_falls_back_to_plain_text():
    """2026-08-06 09:49:18 — both alerts for the FIRST live fill were dropped
    with 400 'can't parse entities' because a reason string contained '<'."""
    from shree.utils.telegram_notifier import TelegramNotifier
    n = TelegramNotifier("tok", "chat", enabled=True)
    sess = _Session()
    n._get_session = lambda: asyncio.sleep(0, result=sess)
    ok = await n.send_message("flow +0 < adaptive req ±25")
    assert ok is True, "alert was dropped instead of retried"
    assert len(sess.posts) == 2
    assert "parse_mode" in sess.posts[0]
    assert "parse_mode" not in sess.posts[1]


@pytest.mark.asyncio
async def test_non_parse_400_is_not_retried():
    """Only parse errors get the fallback — don't mask real API failures."""
    from shree.utils.telegram_notifier import TelegramNotifier

    class Always400:
        def __init__(self): self.posts = []
        def post(self, url, json=None):
            self.posts.append(json)
            return _Resp(400, '{"description":"Bad Request: chat not found"}')

    n = TelegramNotifier("tok", "chat", enabled=True)
    sess = Always400()
    n._get_session = lambda: asyncio.sleep(0, result=sess)
    ok = await n.send_message("hello")
    assert ok is False
    assert len(sess.posts) == 1, "must not retry a non-parse 400"


# ── FIX 3: V2 recorder upstream of every directional filter ────────────────

def test_v2_recorder_lives_in_evaluate_before_conflict_filter():
    """2026-08-05: 7 CALL_SWEEP survived quality but only 1 reached the V2
    ledger — the conflict filter removed 6 before the manager recorded them."""
    src = inspect.getsource(se.SignalEngine.evaluate)
    rec = src.find("_v2.record(_s)")
    conflict = src.find("Cross-signal directional conflict filter")
    assert rec != -1, "V2 recorder missing from evaluate()"
    assert conflict != -1
    assert rec < conflict, "recorder must run BEFORE the conflict filter"


def test_recorder_runs_after_quality_and_confidence_gates():
    """'Qualified' means post-quality — the Phase 3 directive's wording."""
    src = inspect.getsource(se.SignalEngine.evaluate)
    assert src.find("Quality gate BLOCKED") < src.find("_v2.record(_s)")
    assert src.find("Confidence gate") < src.find("_v2.record(_s)")


def test_no_duplicate_recorder_in_manager():
    """Two recorders would double-log every CALL_SWEEP."""
    assert "v2_shadow_gate.record(" not in inspect.getsource(mgr.SpyOptionsManager)


def test_recorder_is_observation_only():
    """It must not be able to change `filtered`."""
    src = inspect.getsource(se.SignalEngine.evaluate)
    i = src.find("_v2.record(_s)")
    block = src[max(0, i - 400):i + 200]
    assert "filtered =" not in block.split("_v2.record(_s)")[1]
