from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from mytrader.risk.risk_gate import RiskGate, RiskGateConfig


def _base_gate():
    cfg = RiskGateConfig()
    cfg.tick_size = 0.25
    cfg.min_stop_points = 2.0
    cfg.risk_per_trade_usd = 50.0
    gate = RiskGate(cfg)
    account = {"available_funds": 5000.0, "realized_pnl_today": 0.0}
    return gate, account


def test_blocks_when_position_at_cap():
    gate, account = _base_gate()
    result = gate.evaluate_entry(
        action="BUY",
        quantity=1,
        entry_price=5000.0,
        atr=5.0,
        account_state=account,
        current_position=1,
        now=datetime.now(timezone.utc),
        stop_loss=4990.0,
        take_profit=5010.0,
    )
    assert not result.allowed
    assert "POSITION_LIMIT" in result.reason


def test_blocks_invalid_bracket():
    gate, account = _base_gate()
    result = gate.evaluate_entry(
        action="SELL",
        quantity=1,
        entry_price=100.0,
        atr=5.0,
        account_state=account,
        current_position=0,
        now=datetime.now(timezone.utc),
        stop_loss=99.0,  # wrong side for SELL
        take_profit=101.0,
    )
    assert not result.allowed
    assert result.reason == "BRACKET_DIRECTION"


def test_blocks_when_stop_too_tight_for_risk():
    gate, account = _base_gate()
    result = gate.evaluate_entry(
        action="BUY",
        quantity=1,
        entry_price=100.0,
        atr=5.0,
        account_state=account,
        current_position=0,
        now=datetime.now(timezone.utc),
        stop_loss=99.0,  # 1 point < required 10 point risk distance
        take_profit=105.0,
    )
    assert not result.allowed
    assert result.reason == "STOP_TOO_TIGHT"


def test_blocks_on_margin_buffer():
    gate, account = _base_gate()
    account["available_funds"] = 100.0
    result = gate.evaluate_entry(
        action="BUY",
        quantity=1,
        entry_price=100.0,
        atr=5.0,
        account_state=account,
        current_position=0,
        now=datetime.now(timezone.utc),
        stop_loss=90.0,
        take_profit=110.0,
    )
    assert not result.allowed
    assert "INSUFFICIENT_MARGIN" in result.reason


def test_blocks_on_daily_loss():
    gate, account = _base_gate()
    account["realized_pnl_today"] = -200.0
    result = gate.evaluate_entry(
        action="BUY",
        quantity=1,
        entry_price=100.0,
        atr=5.0,
        account_state=account,
        current_position=0,
        now=datetime.now(timezone.utc),
        stop_loss=90.0,
        take_profit=110.0,
    )
    assert not result.allowed
    assert result.reason == "DAILY_LOSS_LIMIT"


def test_passes_when_all_guards_ok():
    gate, account = _base_gate()
    # Use a stop within the allowed range (min 2.0, max = risk_budget/5 = 50/5 = 10)
    result = gate.evaluate_entry(
        action="BUY",
        quantity=1,
        entry_price=100.0,
        atr=5.0,
        account_state=account,
        current_position=0,
        now=datetime.now(timezone.utc),
        stop_loss=95.0,  # 5 point stop (within 2-10 range)
        take_profit=110.0,
    )
    assert result.allowed
    assert result.reason == "OK"
    assert result.levels["actual_stop_points"] == pytest.approx(5.0)


def test_peak_drawdown_halt_blocks_entry():
    gate, account = _base_gate()
    gate.config.peak_drawdown_enabled = True
    gate.config.peak_drawdown_pct = 2.0
    gate.config.peak_drawdown_action = "halt"
    gate.update_equity(10000.0, datetime.now(timezone.utc))
    gate.update_equity(9700.0, datetime.now(timezone.utc))  # -3% drawdown
    account["account_equity"] = 9700.0
    result = gate.evaluate_entry(
        action="BUY",
        quantity=1,
        entry_price=100.0,
        atr=5.0,
        account_state=account,
        current_position=0,
        now=datetime.now(timezone.utc),
        stop_loss=95.0,
        take_profit=110.0,
    )
    assert not result.allowed
    assert result.reason == "PEAK_DRAWDOWN_LOCKOUT"


def test_peak_drawdown_tighten_adjusts_stop():
    gate, account = _base_gate()
    gate.config.peak_drawdown_enabled = True
    gate.config.peak_drawdown_pct = 2.0
    gate.config.peak_drawdown_action = "tighten"
    gate.config.peak_drawdown_tighten_multiplier = 0.5
    gate.config.min_stop_points = 2.0
    gate.update_equity(10000.0, datetime.now(timezone.utc))
    gate.update_equity(9700.0, datetime.now(timezone.utc))
    account["account_equity"] = 9700.0
    result = gate.evaluate_entry(
        action="BUY",
        quantity=1,
        entry_price=100.0,
        atr=5.0,
        account_state=account,
        current_position=0,
        now=datetime.now(timezone.utc),
        stop_loss=90.0,
        take_profit=110.0,
    )
    assert result.allowed
    assert result.levels["stop_loss"] > 90.0
