"""Tests for shree/utils/news_calendar.py (no network I/O needed)."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone, timedelta
from typing import List
from unittest.mock import patch, MagicMock
import io

import pytest
import pytz

from shree.utils.news_calendar import (
    _event_to_window,
    _merge_overlapping,
    fetch_lockout_windows,
    is_in_lockout,
)

_ET = pytz.timezone("America/New_York")


def _usd_high_event(et_time: str, title: str = "CPI") -> dict:
    """Build a ForexFactory-like event dict with a UTC ISO timestamp."""
    naive = datetime.strptime(f"2026-03-24 {et_time}", "%Y-%m-%d %H:%M")
    et_dt = _ET.localize(naive)
    utc_str = et_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return {"title": title, "country": "USD", "impact": "High", "date": utc_str}


class TestEventToWindow:
    def test_basic_window(self) -> None:
        event = _usd_high_event("08:30")
        w = _event_to_window(event, date(2026, 3, 24), pre_minutes=5, post_minutes=10)
        assert w is not None
        assert w[0] == "08:25"
        assert w[1] == "08:40"

    def test_wrong_currency_skipped(self) -> None:
        event = _usd_high_event("08:30")
        event["country"] = "EUR"
        assert _event_to_window(event, date(2026, 3, 24), 5, 10) is None

    def test_non_high_impact_skipped(self) -> None:
        event = _usd_high_event("08:30")
        event["impact"] = "Low"
        assert _event_to_window(event, date(2026, 3, 24), 5, 10) is None

    def test_wrong_date_skipped(self) -> None:
        event = _usd_high_event("08:30")
        # Event is 2026-03-24 in ET; query for a different date
        assert _event_to_window(event, date(2026, 3, 25), 5, 10) is None

    def test_missing_date_field_returns_none(self) -> None:
        event = {"title": "x", "country": "USD", "impact": "High", "date": ""}
        assert _event_to_window(event, date(2026, 3, 24), 5, 10) is None

    def test_malformed_date_returns_none(self) -> None:
        event = {"title": "x", "country": "USD", "impact": "High", "date": "not-a-date"}
        assert _event_to_window(event, date(2026, 3, 24), 5, 10) is None

    def test_window_crosses_hour_boundary(self) -> None:
        event = _usd_high_event("10:00")
        w = _event_to_window(event, date(2026, 3, 24), pre_minutes=5, post_minutes=10)
        assert w == ["09:55", "10:10"]


class TestMergeOverlapping:
    def test_empty(self) -> None:
        assert _merge_overlapping([]) == []

    def test_no_overlap(self) -> None:
        windows = [["08:25", "08:35"], ["10:00", "10:10"]]
        assert _merge_overlapping(windows) == [["08:25", "08:35"], ["10:00", "10:10"]]

    def test_overlapping_merged(self) -> None:
        windows = [["08:25", "08:40"], ["08:35", "08:50"]]
        result = _merge_overlapping(windows)
        assert result == [["08:25", "08:50"]]

    def test_adjacent_merged(self) -> None:
        windows = [["08:25", "08:35"], ["08:35", "08:45"]]
        result = _merge_overlapping(windows)
        assert result == [["08:25", "08:45"]]

    def test_three_windows_two_merged(self) -> None:
        windows = [["08:00", "08:10"], ["08:08", "08:20"], ["10:00", "10:10"]]
        result = _merge_overlapping(windows)
        assert len(result) == 2
        assert result[0] == ["08:00", "08:20"]
        assert result[1] == ["10:00", "10:10"]


class TestFetchLockoutWindows:
    def _fake_feed(self, events: List[dict]) -> bytes:
        return json.dumps(events).encode()

    def test_returns_windows_for_target_date(self) -> None:
        events = [_usd_high_event("08:30"), _usd_high_event("14:00")]
        feed = self._fake_feed(events)

        with patch("urllib.request.urlopen") as mock_open:
            ctx = MagicMock()
            ctx.__enter__ = lambda s: MagicMock(read=lambda: feed)
            ctx.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = ctx
            windows = fetch_lockout_windows(date(2026, 3, 24), pre_minutes=5, post_minutes=10)

        assert len(windows) == 2

    def test_network_error_returns_empty(self) -> None:
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            windows = fetch_lockout_windows(date(2026, 3, 24))
        assert windows == []

    def test_non_usd_events_filtered(self) -> None:
        event = _usd_high_event("08:30")
        event["country"] = "EUR"
        feed = self._fake_feed([event])

        with patch("urllib.request.urlopen") as mock_open:
            ctx = MagicMock()
            ctx.__enter__ = lambda s: MagicMock(read=lambda: feed)
            ctx.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = ctx
            windows = fetch_lockout_windows(date(2026, 3, 24))

        assert windows == []

    def test_overlapping_events_merged(self) -> None:
        # Two events at 08:30 and 08:35 — windows will overlap
        events = [_usd_high_event("08:30"), _usd_high_event("08:35")]
        feed = self._fake_feed(events)

        with patch("urllib.request.urlopen") as mock_open:
            ctx = MagicMock()
            ctx.__enter__ = lambda s: MagicMock(read=lambda: feed)
            ctx.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = ctx
            windows = fetch_lockout_windows(
                date(2026, 3, 24), pre_minutes=5, post_minutes=10
            )

        assert len(windows) == 1   # Merged into one


class TestIsInLockout:
    def _ts(self, h: int, m: int) -> "pd.Timestamp":
        import pandas as pd
        return pd.Timestamp(f"2026-03-24 {h:02d}:{m:02d}:00", tz="America/New_York")

    def test_inside_window(self) -> None:
        assert is_in_lockout(self._ts(8, 30), [["08:25", "08:40"]])

    def test_outside_window(self) -> None:
        assert not is_in_lockout(self._ts(9, 0), [["08:25", "08:40"]])

    def test_empty_windows_never_locked(self) -> None:
        assert not is_in_lockout(self._ts(8, 30), [])

    def test_boundary_start(self) -> None:
        assert is_in_lockout(self._ts(8, 25), [["08:25", "08:40"]])

    def test_boundary_end(self) -> None:
        assert is_in_lockout(self._ts(8, 40), [["08:25", "08:40"]])

    def test_multiple_windows(self) -> None:
        windows = [["08:25", "08:35"], ["10:00", "10:10"]]
        assert is_in_lockout(self._ts(10, 5), windows)
        assert not is_in_lockout(self._ts(9, 0), windows)
