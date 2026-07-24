"""Unit tests for the ThetaData v3 adapter — network-free.

Discovery + live 403 handling are proven by the smoke run against the terminal;
here we lock the header-driven row->Print mapper, alias resolution, greeks, and
the date/DTE helpers so a schema change is caught, not silently mis-parsed.
"""
from __future__ import annotations

import datetime as dt

from shree.flow_research.thetadata import (
    ThetaDataSource, _resolve, _parse_date, _parse_iso, _trading_dte,
    _ymd, _strike,
)
from shree.flow_research.greeks import bs_delta, implied_vol


def _src():
    # source with no client calls exercised (we call _row_to_print directly)
    return ThetaDataSource(client=None, start=dt.date(2026, 7, 23),
                           end=dt.date(2026, 7, 23), strikes_around_atm=1)


def test_alias_resolution():
    header = ["timestamp", "price", "size", "exchange", "condition",
              "bid_size", "bid_exchange", "bid", "bid_condition",
              "ask_size", "ask_exchange", "ask", "ask_condition"]
    cm = _resolve(header)
    assert cm["timestamp"] == "timestamp"
    assert cm["price"] == "price"
    assert cm["size"] == "size"
    assert cm["bid"] == "bid" and cm["ask"] == "ask"
    assert cm["exchange"] == "exchange" and cm["condition"] == "condition"


def test_row_to_print_iso_timestamp_and_greeks():
    src = _src()
    row = {
        "timestamp": "2026-07-23T14:30:00.000",  # ET
        "price": "0.55", "size": "40", "exchange": "CBOE",
        "condition": "REGULAR",
        "bid": "0.53", "ask": "0.57", "bid_size": "10", "ask_size": "12",
    }
    p = src._row_to_print(row, exp=dt.date(2026, 7, 23), strike=738.0,
                          right="C", session=dt.date(2026, 7, 23),
                          atm=738.3, dte=0)
    assert p is not None
    assert p.ts_et.startswith("2026-07-23T14:30:00")
    assert p.session_date == "2026-07-23"
    assert p.trade_px == 0.55 and p.size == 40
    assert p.bid == 0.53 and p.ask == 0.57
    assert p.exchange == "CBOE" and p.condition_codes == ["REGULAR"]
    assert p.root == "SPY" and p.right == "C"
    # 0DTE ATM call: computed delta present and in (0,1)
    assert p.delta is not None and 0.0 < p.delta < 1.0
    assert p.greeks_src == "COMPUTED"


def test_row_to_print_date_plus_ms_fallback():
    src = _src()
    # no ISO timestamp; date + ms_of_day (ms since ET midnight). 14:30 ET.
    ms = (14 * 3600 + 30 * 60) * 1000
    row = {"date": "20260723", "ms_of_day": str(ms), "price": "1.00",
           "size": "5", "bid": "0.95", "ask": "1.05"}
    p = src._row_to_print(row, exp=dt.date(2026, 7, 23), strike=738.0,
                          right="P", session=dt.date(2026, 7, 23),
                          atm=738.3, dte=0)
    assert p is not None and p.ts_et.startswith("2026-07-23T14:30:00")
    assert p.right == "P"


def test_row_to_print_rejects_bad_rows():
    src = _src()
    base = {"timestamp": "2026-07-23T14:30:00", "bid": "0.5", "ask": "0.6"}
    assert src._row_to_print({**base, "price": "0", "size": "5"},
                             dt.date(2026, 7, 23), 738.0, "C",
                             dt.date(2026, 7, 23), 738.3, 0) is None
    assert src._row_to_print({**base, "price": "0.5", "size": "0"},
                             dt.date(2026, 7, 23), 738.0, "C",
                             dt.date(2026, 7, 23), 738.3, 0) is None


def test_greeks_sanity():
    # ATM, short T: call delta ~0.5, put delta ~-0.5
    iv = 0.15
    T = 1 / 252.0
    dc = bs_delta(738.0, 738.0, T, iv, "C")
    dp = bs_delta(738.0, 738.0, T, iv, "P")
    assert 0.45 < dc < 0.6
    assert -0.6 < dp < -0.4
    # IV inversion round-trips roughly
    from shree.flow_research.greeks import bs_price
    px = bs_price(738.0, 738.0, T, 0.20, "C")
    got = implied_vol(px, 738.0, 738.0, T, "C")
    assert got is not None and abs(got - 0.20) < 0.02


def test_date_and_dte_helpers():
    assert _parse_date("2026-07-23") == dt.date(2026, 7, 23)
    assert _parse_date("20260723") == dt.date(2026, 7, 23)
    assert _parse_iso("2026-07-23T14:30:00.000").hour == 14
    assert _ymd(dt.date(2026, 7, 23)) == "20260723"
    assert _strike(738.0) == "738" and _strike(738.5) == "738.5"
    # daily expiries as calendar: dte from 07-21 to 07-23 = 2 trading days
    cal = [dt.date(2026, 7, d) for d in (20, 21, 22, 23, 24)]
    assert _trading_dte(dt.date(2026, 7, 21), dt.date(2026, 7, 23), cal) == 2
    assert _trading_dte(dt.date(2026, 7, 23), dt.date(2026, 7, 23), cal) == 0
