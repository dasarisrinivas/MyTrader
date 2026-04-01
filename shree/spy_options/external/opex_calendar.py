"""OPEX / Expiration calendar — pure Python, no external data needed.

Computes:
  - Monthly OPEX: 3rd Friday of each month
  - Quarterly triple witching: 3rd Friday of March, June, September, December
  - Days to nearest OPEX
  - is_opex_week, is_opex_day flags
  - gamma_environment: PINNING / EXPANSIVE / NEUTRAL

Gamma environment logic:
  PINNING  — within opex week AND positive net GEX (dealers long gamma → pin)
  EXPANSIVE — last ≥ 3 days before opex OR negative net GEX (dealers short gamma)
  NEUTRAL  — otherwise

No network I/O required — all calendar math is pure datetime.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional


@dataclass
class OpexState:
    next_opex: date = date.today()
    days_to_opex: int = 0
    is_opex_week: bool = False
    is_opex_day: bool = False
    is_triple_witching: bool = False
    opex_type: str = "MONTHLY"          # MONTHLY / TRIPLE_WITCHING
    gamma_environment: str = "NEUTRAL"  # PINNING / EXPANSIVE / NEUTRAL


class OpexCalendar:
    """Pure-Python OPEX calendar. Call compute() each poll cycle — zero latency."""

    def compute(self, today: Optional[date] = None, net_gex: float = 0.0) -> OpexState:
        """Return OpexState for *today* (defaults to date.today()).

        Args:
            today:   override for testing / after-hours simulation
            net_gex: net dealer GEX from FlowState (positive = dealers long gamma)
        """
        if today is None:
            today = date.today()

        state = OpexState()
        state.next_opex = _next_opex(today)
        state.days_to_opex = (state.next_opex - today).days
        state.is_opex_day = state.next_opex == today
        state.is_opex_week = state.days_to_opex <= 4  # Mon–Fri of opex week

        tw_month = state.next_opex.month
        state.is_triple_witching = tw_month in (3, 6, 9, 12)
        state.opex_type = "TRIPLE_WITCHING" if state.is_triple_witching else "MONTHLY"

        state.gamma_environment = _gamma_env(
            days_to_opex=state.days_to_opex,
            is_opex_week=state.is_opex_week,
            net_gex=net_gex,
        )
        return state


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given year/month."""
    c = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in c if week[calendar.FRIDAY] != 0]
    return date(year, month, fridays[2])  # 0-indexed: 3rd = index 2


def _next_opex(today: date) -> date:
    """Return the nearest upcoming (or today's) monthly OPEX (3rd Friday)."""
    # Try this month's OPEX first
    opex_this = _third_friday(today.year, today.month)
    if opex_this >= today:
        return opex_this
    # Otherwise next month
    if today.month == 12:
        return _third_friday(today.year + 1, 1)
    return _third_friday(today.year, today.month + 1)


def _gamma_env(days_to_opex: int, is_opex_week: bool, net_gex: float) -> str:
    """Classify gamma environment.

    PINNING:
      - Dealers are net long gamma (net_gex > 0)
      - AND we are in opex week (1–4 days away)
      → dealers hedge by selling rallies / buying dips → price pinning

    EXPANSIVE:
      - Dealers net short gamma (net_gex < 0)
      - OR more than 3 days from OPEX (pin force weak)
      → dealers chase price → moves accelerate

    NEUTRAL otherwise.
    """
    if is_opex_week and net_gex > 0:
        return "PINNING"
    if not is_opex_week or net_gex < 0:
        return "EXPANSIVE"
    return "NEUTRAL"
