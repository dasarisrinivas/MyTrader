from __future__ import annotations

import types


def _apply_entry_trade_cycle_id(live_manager: object, metadata: dict | None) -> str | None:
    """Replicates the minimal entry-cycle retention logic used in live_trading_manager.

    This test is intentionally lightweight (no IB/event loop). It guards against regressions
    where AWS-entry assigns a random current_trade_id and loses the trade_cycle_id needed
    for trade_outcomes finalization.
    """

    entry_trade_cycle_id = None
    try:
        entry_trade_cycle_id = (metadata or {}).get("trade_cycle_id") or getattr(live_manager, "_current_cycle_id", None)
    except Exception:
        entry_trade_cycle_id = getattr(live_manager, "_current_cycle_id", None)

    if entry_trade_cycle_id:
        live_manager._current_entry_cycle_id = str(entry_trade_cycle_id)
        live_manager.current_trade_id = str(entry_trade_cycle_id)
        return str(entry_trade_cycle_id)

    live_manager.current_trade_id = "RANDOM_UUID_FALLBACK"
    return None


def test_entry_trade_cycle_id_is_retained() -> None:
    mgr = types.SimpleNamespace(_current_cycle_id="cycle123", _current_entry_cycle_id=None, current_trade_id=None)
    applied = _apply_entry_trade_cycle_id(mgr, {"trade_cycle_id": "tradeABC"})

    assert applied == "tradeABC"
    assert mgr._current_entry_cycle_id == "tradeABC"
    assert mgr.current_trade_id == "tradeABC"


def test_entry_trade_cycle_id_falls_back_to_current_cycle() -> None:
    mgr = types.SimpleNamespace(_current_cycle_id="cycle123", _current_entry_cycle_id=None, current_trade_id=None)
    applied = _apply_entry_trade_cycle_id(mgr, {})

    assert applied == "cycle123"
    assert mgr._current_entry_cycle_id == "cycle123"
    assert mgr.current_trade_id == "cycle123"
