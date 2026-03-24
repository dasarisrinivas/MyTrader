"""Economic-release calendar integration.

Fetches upcoming high-impact USD events and converts them into
``session.news_lockout_windows_et`` entries compatible with
``GoldSessionConfig``.

Primary source: ForexFactory public JSON calendar feed.
Falls back to an empty list on any network/parse error so the bot
continues trading without a stale lockout list.

Usage — one-time refresh at session start:
    from shree.utils.news_calendar import fetch_lockout_windows
    windows = fetch_lockout_windows(date.today())
    cfg.session.news_lockout_windows_et = windows

The returned windows are conservative: ``pre_minutes`` before the
release and ``post_minutes`` after, to avoid entering into a spike.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from datetime import date, datetime, timedelta
from typing import List, Optional
import pytz

from ..utils.logger import logger


_ET = pytz.timezone("America/New_York")

# ForexFactory weekly calendar — public JSON, no API key required.
# Covers the current week (Mon–Sun) from Sunday 00:00 UTC.
_FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Impact levels and currencies we care about for Gold trading
_TRADEABLE_CURRENCIES = {"USD"}
_HIGH_IMPACT_LEVELS = {"High"}


def fetch_lockout_windows(
    target_date: date,
    pre_minutes: int = 5,
    post_minutes: int = 10,
    timeout_sec: int = 8,
    url: str = _FF_CALENDAR_URL,
) -> List[List[str]]:
    """Return ``[[start_et, end_et], ...]`` lockout windows for ``target_date``.

    Args:
        target_date:   The trading date to filter events for.
        pre_minutes:   How many minutes before the event to start the blackout.
        post_minutes:  How many minutes after the event to end the blackout.
        timeout_sec:   HTTP request timeout.
        url:           Override the calendar feed URL (useful in tests).

    Returns:
        List of ``["HH:MM", "HH:MM"]`` pairs in ET, ready to assign to
        ``GoldSessionConfig.news_lockout_windows_et``.  Empty list if
        no events found or if the fetch fails.
    """
    try:
        events = _fetch_events(url, timeout_sec)
    except Exception as exc:
        logger.warning("news_calendar: fetch failed (%s) — no lockout windows applied", exc)
        return []

    windows: List[List[str]] = []
    for event in events:
        try:
            window = _event_to_window(event, target_date, pre_minutes, post_minutes)
        except Exception as exc:
            logger.debug("news_calendar: skipping malformed event %s — %s", event, exc)
            continue
        if window is not None:
            windows.append(window)

    windows = _merge_overlapping(windows)
    logger.info(
        "news_calendar: %d lockout window(s) for %s — %s",
        len(windows),
        target_date,
        windows or "none",
    )
    return windows


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_events(url: str, timeout: int) -> List[dict]:
    """Download and parse the JSON calendar feed."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ShreeBot/1.0 economic-calendar"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _event_to_window(
    event: dict,
    target_date: date,
    pre_minutes: int,
    post_minutes: int,
) -> Optional[List[str]]:
    """Convert a single calendar event to a ``[start_et, end_et]`` window.

    Returns None if the event should be skipped (wrong date, currency, or impact).
    """
    # Filter: currency and impact
    currency = str(event.get("country", "")).upper()
    impact = str(event.get("impact", "")).capitalize()
    if currency not in _TRADEABLE_CURRENCIES or impact not in _HIGH_IMPACT_LEVELS:
        return None

    # Parse event timestamp — ForexFactory provides ISO 8601 UTC
    date_str: str = event.get("date", "")
    if not date_str:
        return None

    # Try both ISO and legacy formats
    event_utc: Optional[datetime] = None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M%z"):
        try:
            event_utc = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue
    if event_utc is None:
        # Try fromisoformat (Python 3.7+)
        try:
            event_utc = datetime.fromisoformat(date_str)
        except ValueError:
            return None

    # Convert to ET and match date
    event_et = event_utc.astimezone(_ET)
    if event_et.date() != target_date:
        return None

    # Build lockout window
    start_et = event_et - timedelta(minutes=pre_minutes)
    end_et = event_et + timedelta(minutes=post_minutes)

    return [start_et.strftime("%H:%M"), end_et.strftime("%H:%M")]


def _merge_overlapping(windows: List[List[str]]) -> List[List[str]]:
    """Merge adjacent or overlapping ``[start, end]`` windows."""
    if not windows:
        return windows

    def _hm(t: str) -> int:
        h, m = t.split(":")
        return int(h) * 60 + int(m)

    def _fmt(mins: int) -> str:
        return f"{mins // 60:02d}:{mins % 60:02d}"

    parsed = sorted((_hm(w[0]), _hm(w[1])) for w in windows)
    merged: List[tuple] = [parsed[0]]
    for start, end in parsed[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return [[_fmt(s), _fmt(e)] for s, e in merged]


def is_in_lockout(
    bar_ts: "pd.Timestamp",  # noqa: F821
    windows: List[List[str]],
) -> bool:
    """Return True if ``bar_ts`` falls within any lockout window.

    Args:
        bar_ts:   Timestamp of the completed bar (any timezone).
        windows:  List of ``["HH:MM", "HH:MM"]`` ET window pairs.
    """
    if not windows:
        return False
    try:
        local = bar_ts.tz_convert("America/New_York")
    except Exception:
        local = bar_ts
    hm = local.hour * 60 + local.minute
    for start_str, end_str in windows:
        s = int(start_str.split(":")[0]) * 60 + int(start_str.split(":")[1])
        e = int(end_str.split(":")[0]) * 60 + int(end_str.split(":")[1])
        if s <= hm <= e:
            return True
    return False
