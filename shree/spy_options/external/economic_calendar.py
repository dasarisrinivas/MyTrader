"""
Economic calendar via Forex Factory public JSON feed.
Caches results for the full trading day; refreshes once per day.

Timezone note: Forex Factory event times are published in **US Eastern Time**
(ET, America/New_York).  The raw JSON has no timezone indicator, so we parse
them as naive datetimes and then attach the Eastern timezone before converting
to UTC.  This corrects a prior bug where all times were treated as UTC, which
shifted true event proximity by 4–5 hours depending on DST.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional
from zoneinfo import ZoneInfo
import aiohttp
from loguru import logger

_ET = ZoneInfo("America/New_York")

_FF_URL = "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json"
_HIGH_IMPACT = {"High"}
_MEDIUM_IMPACT = {"Medium", "High"}


@dataclass
class EconomicEvent:
    title: str
    impact: str          # "High" | "Medium" | "Low" | "Holiday"
    country: str
    event_dt: datetime   # UTC-aware


@dataclass
class CalendarState:
    events: List[EconomicEvent] = field(default_factory=list)
    fetched_date: Optional[date] = None   # calendar date when fetched

    def is_stale(self) -> bool:
        today = datetime.now(timezone.utc).date()
        return self.fetched_date is None or self.fetched_date < today

    def high_impact_within(self, minutes: int = 30) -> List[EconomicEvent]:
        """Return US High-impact events within ±minutes of now."""
        now = datetime.now(timezone.utc)
        window = timedelta(minutes=minutes)
        return [
            e for e in self.events
            if e.country.upper() == "USD"
            and e.impact in _HIGH_IMPACT
            and abs((e.event_dt - now).total_seconds()) <= window.total_seconds()
        ]

    def next_high_impact(self) -> Optional[EconomicEvent]:
        now = datetime.now(timezone.utc)
        upcoming = [
            e for e in self.events
            if e.country.upper() == "USD"
            and e.impact in _HIGH_IMPACT
            and e.event_dt > now
        ]
        return min(upcoming, key=lambda e: e.event_dt) if upcoming else None


class EconomicCalendar:
    def __init__(self, timeout_s: float = 10.0):
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._state = CalendarState()
        self._lock = asyncio.Lock()

    async def refresh_if_stale(self) -> None:
        if not self._state.is_stale():
            return
        async with self._lock:
            if not self._state.is_stale():  # double-check after lock
                return
            await self._fetch()

    async def _fetch(self) -> None:
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.get(_FF_URL) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
            events: List[EconomicEvent] = []
            for item in data:
                dt_str = item.get("date", "")
                try:
                    # FF uses "MMM DD, YYYY HH:MMam/pm" format and publishes in
                    # US Eastern Time (ET).  Attach the ET timezone then convert
                    # to UTC so proximity comparisons against datetime.utcnow()
                    # are accurate regardless of DST offset (−4 EDT / −5 EST).
                    event_dt_naive = datetime.strptime(dt_str, "%b %d, %Y %I:%M%p")
                    event_dt = event_dt_naive.replace(tzinfo=_ET).astimezone(timezone.utc)
                except ValueError:
                    continue
                events.append(EconomicEvent(
                    title=item.get("title", ""),
                    impact=item.get("impact", "Low"),
                    country=item.get("country", ""),
                    event_dt=event_dt,
                ))
            self._state = CalendarState(
                events=events,
                fetched_date=datetime.now(timezone.utc).date(),
            )
            logger.info(f"[EconCalendar] Loaded {len(events)} events")
        except Exception as exc:
            logger.warning(f"[EconCalendar] Fetch failed: {exc}")

    @property
    def state(self) -> CalendarState:
        return self._state
