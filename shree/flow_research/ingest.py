"""Print sources — where raw OPRA prints come from.

Vendor-agnostic. The pipeline (classify -> features -> snapshot) never knows or
cares which source produced the prints, so the POC runs on synthetic or CSV
replay data with ZERO paid subscription, and a real vendor is a drop-in later.

Sources implemented:
  * SyntheticSource  — generated prints, for unit tests & smoke runs (free)
  * CsvReplaySource  — reads a CSV export of historical prints (free once you
                       have a one-off historical file / flat-file download)
  * ThetaDataSource  — STUB. Real adapter to fill in when a subscription exists.

None of these connect to the live trading system.
"""
from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from .models import Print


class PrintSource(ABC):
    """Yields raw Print rows (unclassified). Ordered by ts_utc ascending."""

    @abstractmethod
    def prints(self) -> Iterator[Print]:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# CSV replay — the cheapest real-data path
# ─────────────────────────────────────────────────────────────────────────────

class CsvReplaySource(PrintSource):
    """Read historical prints from a CSV export.

    Expected columns (extra columns ignored, missing optional ones tolerated):
      ts_utc, ts_et, session_date, root, expiry, strike, right,
      trade_px, size, exchange, condition_codes, underlying_px, dte,
      bid, ask, delta, gamma, iv, greeks_src, data_source

    condition_codes may be a '|' or ',' separated string.
    This is what a ThetaData / Databento / Polygon flat-file dump maps onto.
    """

    def __init__(self, path: str, default_source: str = "csv_replay"):
        self.path = path
        self.default_source = default_source

    def prints(self) -> Iterator[Print]:
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield self._row_to_print(row)

    def _row_to_print(self, row: dict) -> Print:
        def _f(k) -> Optional[float]:
            v = row.get(k)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except ValueError:
                return None

        def _i(k) -> Optional[int]:
            v = _f(k)
            return int(v) if v is not None else None

        cond_raw = (row.get("condition_codes") or "").replace("|", ",")
        conds = [c.strip() for c in cond_raw.split(",") if c.strip()]

        return Print(
            ts_utc=row.get("ts_utc") or row.get("ts_et") or "",
            ts_et=row.get("ts_et") or row.get("ts_utc") or "",
            session_date=row.get("session_date") or "",
            root=(row.get("root") or "SPY").upper(),
            expiry=row.get("expiry") or "",
            strike=_f("strike") or 0.0,
            right=(row.get("right") or "C").upper(),
            trade_px=_f("trade_px") or 0.0,
            size=_i("size") or 0,
            exchange=row.get("exchange") or None,
            condition_codes=conds,
            underlying_px=_f("underlying_px"),
            dte=_i("dte"),
            bid=_f("bid"),
            ask=_f("ask"),
            delta=_f("delta"),
            gamma=_f("gamma"),
            iv=_f("iv"),
            greeks_src=row.get("greeks_src") or None,
            data_source=row.get("data_source") or self.default_source,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic — free data for tests and plumbing smoke runs
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticSource(PrintSource):
    """Deterministic generated prints. NOT market data — plumbing only.

    Accepts a pre-built list so callers control the scenario (and tests stay
    deterministic without any RNG).
    """

    def __init__(self, prints: List[Print]):
        self._prints = list(prints)

    def prints(self) -> Iterator[Print]:
        yield from self._prints


# ─────────────────────────────────────────────────────────────────────────────
# ThetaData — REAL adapter stub (fill in only after subscribing)
# ─────────────────────────────────────────────────────────────────────────────

class ThetaDataSource(PrintSource):
    """Placeholder for the real ThetaData historical/real-time adapter.

    Intentionally not implemented: the POC must run and be validated on CSV
    replay BEFORE any subscription is bought (design doc section 6). When the
    historical POC shows lift, implement `prints()` here to pull SPY 0-2DTE
    trades + NBBO + greeks from ThetaData's REST/stream and map to Print.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def prints(self) -> Iterator[Print]:
        raise NotImplementedError(
            "ThetaDataSource is a stub. Validate the CSV-replay POC first "
            "(docs/FLOW_RESEARCH_DESIGN.md s6) before implementing a paid feed."
        )
