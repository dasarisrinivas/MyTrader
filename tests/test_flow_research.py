"""Tests for the flow-research layer. Deterministic; no paid data, no RNG in
fixtures. Proves classification, feature math, DB round-trip, the snapshotter
join, and the shuffle control's false-positive resistance.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import pytest

from shree.flow_research.models import (
    Print, BUY, SELL, MID, UNKNOWN, SRC_QUOTE, SRC_NONE,
)
from shree.flow_research.classify import (
    classify_aggressor, classify_all, mark_blocks, mark_sweeps, is_clean_print,
)
from shree.flow_research.features import compute_features
from shree.flow_research import schema
from shree.flow_research.snapshotter import PrintStore, snapshot_signals
from shree.flow_research import validate as V


def _p(strike=500.0, right="C", px=1.00, size=10, bid=0.95, ask=1.05,
       ts="2026-07-23T10:30:00", exch="CBOE", conds=None, delta=0.5,
       iv=0.15, underlying=500.0, root="SPY", expiry="2026-07-23"):
    return Print(
        ts_utc=ts, ts_et=ts, session_date=ts[:10], root=root, expiry=expiry,
        strike=strike, right=right, trade_px=px, size=size, exchange=exch,
        condition_codes=conds or [], underlying_px=underlying, dte=0,
        bid=bid, ask=ask, delta=delta, iv=iv,
    )


# ── classification ─────────────────────────────────────────────────────────

def test_aggressor_at_ask_is_buy():
    p = classify_aggressor(_p(px=1.05, bid=0.95, ask=1.05))
    assert p.aggressor == BUY and p.aggressor_src == SRC_QUOTE


def test_aggressor_at_bid_is_sell():
    p = classify_aggressor(_p(px=0.95, bid=0.95, ask=1.05))
    assert p.aggressor == SELL and p.aggressor_src == SRC_QUOTE


def test_aggressor_mid_is_ambiguous():
    p = classify_aggressor(_p(px=1.00, bid=0.95, ask=1.05))
    assert p.aggressor == MID


def test_aggressor_no_quote_is_unknown():
    p = classify_aggressor(_p(px=1.00, bid=None, ask=None))
    assert p.aggressor == UNKNOWN and p.aggressor_src == SRC_NONE


def test_crossed_quote_rejected():
    # ask < bid -> unusable
    p = classify_aggressor(_p(px=1.0, bid=1.10, ask=0.90))
    assert p.aggressor == UNKNOWN


# ── condition filtering ──────────────────────────────────────────────────────

def test_spread_leg_excluded_from_clean():
    assert not is_clean_print(_p(conds=["SPREAD"]))
    assert is_clean_print(_p(conds=["REGULAR"]))


# ── blocks & sweeps ──────────────────────────────────────────────────────────

def test_block_threshold():
    ps = [_p(size=300), _p(size=10)]
    mark_blocks(ps)
    assert ps[0].is_block and not ps[1].is_block


def test_sweep_multi_venue():
    ps = [
        _p(px=1.05, bid=0.95, ask=1.05, exch="CBOE", ts="2026-07-23T10:30:00.100"),
        _p(px=1.05, bid=0.95, ask=1.05, exch="ISE",  ts="2026-07-23T10:30:00.200"),
        _p(px=1.05, bid=0.95, ask=1.05, exch="PHLX", ts="2026-07-23T10:30:00.300"),
    ]
    classify_all(ps)
    mark_sweeps(ps, window_ms=500, min_venues=2)
    assert all(p.is_sweep for p in ps)


def test_no_sweep_single_venue():
    ps = [
        _p(px=1.05, bid=0.95, ask=1.05, exch="CBOE", ts="2026-07-23T10:30:00.100"),
        _p(px=1.05, bid=0.95, ask=1.05, exch="CBOE", ts="2026-07-23T10:30:00.200"),
    ]
    classify_all(ps)
    mark_sweeps(ps, window_ms=500, min_venues=2)
    assert not any(p.is_sweep for p in ps)


# ── features ─────────────────────────────────────────────────────────────────

def test_net_call_premium_positive_when_calls_bought():
    ps = [_p(right="C", px=1.05, bid=0.95, ask=1.05, size=10) for _ in range(3)]
    classify_all(ps)
    snap = compute_features(ps, snapshot_kind="INTERVAL",
                            session_date="2026-07-23", ts_et="2026-07-23T10:30:00",
                            window_s=1800)
    assert snap.net_call_prem > 0
    assert snap.pc_prem_imbalance > 0
    assert snap.n_prints_used == 3


def test_buying_puts_is_bearish_delta_flow():
    # put delta negative; buying puts -> negative dw_flow (bearish)
    ps = [_p(right="P", px=1.05, bid=0.95, ask=1.05, delta=-0.5, size=10)
          for _ in range(3)]
    classify_all(ps)
    snap = compute_features(ps, snapshot_kind="INTERVAL",
                            session_date="2026-07-23", ts_et="2026-07-23T10:30:00",
                            window_s=1800)
    assert snap.net_put_prem > 0        # premium bought
    assert snap.dw_flow < 0             # but directional exposure is bearish


def test_mid_and_spread_prints_excluded_from_net():
    ps = [
        _p(px=1.00, bid=0.95, ask=1.05),                  # MID -> excluded
        _p(px=1.05, bid=0.95, ask=1.05, conds=["SPREAD"]),  # spread -> excluded
        _p(px=1.05, bid=0.95, ask=1.05),                  # clean BUY -> used
    ]
    classify_all(ps)
    snap = compute_features(ps, snapshot_kind="INTERVAL",
                            session_date="2026-07-23", ts_et="2026-07-23T10:30:00",
                            window_s=1800)
    assert snap.n_prints == 3 and snap.n_prints_used == 1


def test_empty_window_is_safe():
    snap = compute_features([], snapshot_kind="INTERVAL",
                            session_date="2026-07-23", ts_et="2026-07-23T10:30:00",
                            window_s=1800)
    assert snap.n_prints_used == 0 and snap.pc_prem_imbalance == 0.0


# ── DB round-trip + snapshotter join ─────────────────────────────────────────

def test_db_roundtrip_and_signal_snapshot(tmp_path):
    flow_db = str(tmp_path / "flow.db")
    spy_db = str(tmp_path / "spy.db")

    # raw prints in one session window
    conn = schema.open_db(flow_db)
    ps = [_p(right="C", px=1.05, bid=0.95, ask=1.05, size=50,
             ts=f"2026-07-23T10:{m:02d}:00") for m in range(20, 35)]
    schema.insert_prints(conn, ps)

    # a fake production spy_signals with a signal at 10:35
    sc = sqlite3.connect(spy_db)
    cols = ("id INTEGER, sent_at TEXT, signal_type TEXT, spy_price REAL, "
            "spy_price_exit REAL, pnl_pct REAL, outcome TEXT, "
            + ", ".join(f"{f} REAL" for f in V.EXISTING_FEATURES))
    sc.execute(f"CREATE TABLE spy_signals ({cols})")
    # sent_at is UTC (see snapshotter._to_et_iso); 14:35 UTC = 10:35 ET, which
    # falls at the end of the 10:20-10:34 ET print window.
    sc.execute(
        "INSERT INTO spy_signals (id, sent_at, signal_type, spy_price, "
        "spy_price_exit, pnl_pct, outcome) VALUES (1,'2026-07-23T14:35:00',"
        "'TREND_CONTINUATION',500.0,502.0,12.5,'WIN')")
    sc.commit(); sc.close()

    n = snapshot_signals(PrintStore(conn), conn, spy_db, window_s=1800)
    assert n == 1
    row = conn.execute(
        "SELECT snapshot_kind, signal_id, n_prints_used, net_call_prem "
        "FROM shadow_flow").fetchone()
    conn.close()
    assert row["snapshot_kind"] == "SIGNAL"
    assert row["signal_id"] == 1
    assert row["n_prints_used"] == 15
    assert row["net_call_prem"] > 0


def test_sent_at_utc_converts_to_et():
    # sent_at is stored UTC; the join must convert to ET (verified 2026-07-23).
    from shree.flow_research.snapshotter import _to_et_iso
    # 18:35 UTC in July (EDT, -4) -> 14:35 ET
    assert _to_et_iso("2026-07-23T18:35:25.660676") == "2026-07-23T14:35:25"
    # explicit 'Z' handled identically
    assert _to_et_iso("2026-07-23T18:35:25Z") == "2026-07-23T14:35:25"
    # a 13:30 UTC open maps to 09:30 ET
    assert _to_et_iso("2026-07-23T13:30:00") == "2026-07-23T09:30:00"


def test_readonly_open_does_not_write(tmp_path):
    # snapshotter must open production strictly read-only
    spy_db = str(tmp_path / "spy_ro.db")
    sc = sqlite3.connect(spy_db)
    sc.execute("CREATE TABLE spy_signals (id INTEGER, sent_at TEXT)")
    sc.execute("INSERT INTO spy_signals VALUES (1,'2026-07-23T10:35:00')")
    sc.commit(); sc.close()
    before = os.path.getmtime(spy_db)
    from shree.flow_research.snapshotter import _open_readonly
    ro = _open_readonly(spy_db)
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("INSERT INTO spy_signals VALUES (2,'x')")
    ro.close()
    assert os.path.getmtime(spy_db) == before


# ── validation controls ──────────────────────────────────────────────────────

def test_shuffle_control_no_false_edge_on_random_data():
    # pure noise: measure independent of outcome -> shuffle p should be large
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "pc_prem_imbalance": rng.normal(size=400),
        "opt_pnl": rng.normal(size=400),
    })
    res = V.shuffle_control(df, "pc_prem_imbalance", n_shuffles=500)
    assert res["status"] == "OK"
    assert res["p_value"] > 0.05          # no spurious edge
    assert res["passes"] is False


def test_shuffle_control_detects_real_edge():
    # planted monotone relation -> shuffle p should be small
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    y = x * 2.0 + rng.normal(scale=0.5, size=400)   # strong dependence
    df = pd.DataFrame({"dw_flow": x, "opt_pnl": y})
    res = V.shuffle_control(df, "dw_flow", n_shuffles=500)
    assert res["p_value"] < 0.05 and res["passes"] is True


def test_direction_confound_kills_beta_only_effect():
    # measure correlates with outcome ONLY through the underlying move (beta)
    rng = np.random.default_rng(11)
    move = rng.normal(size=500)
    measure = move + rng.normal(scale=0.1, size=500)   # measure ~ move
    outcome = move + rng.normal(scale=0.1, size=500)   # outcome ~ move
    df = pd.DataFrame({"dw_flow": measure, "opt_pnl": outcome,
                       "underlying_move": move})
    res = V.direction_confound(df, "dw_flow")
    assert res["raw_corr"] > 0.5                 # looks predictive raw
    assert res["survives_confound"] is False     # but it's just beta
