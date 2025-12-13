import math

from mytrader.execution.order_builder import validate_bracket_prices
from mytrader.execution.guards import WaitDecisionContext, should_block_on_wait
from mytrader.risk.atr_module import compute_protective_offsets


def test_buy_bracket_rejects_take_profit_below_entry():
    result = validate_bracket_prices(
        action="BUY",
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=99.5,
        tick_size=0.25,
    )
    assert not result.valid
    assert "Take-profit" in (result.reason or "")


def test_sell_bracket_rejects_take_profit_above_entry():
    result = validate_bracket_prices(
        action="SELL",
        entry_price=100.0,
        stop_loss=101.5,
        take_profit=100.5,
        tick_size=0.25,
    )
    assert not result.valid
    assert "Take-profit" in (result.reason or "")


def test_bracket_clamps_to_minimum_tick_distance():
    result = validate_bracket_prices(
        action="BUY",
        entry_price=100.0,
        stop_loss=99.8,
        take_profit=100.1,
        tick_size=0.5,
    )
    assert result.valid
    assert math.isclose(result.take_profit, 100.5)
    assert math.isclose(result.stop_loss, 99.5)
    assert "take_profit" in result.adjusted_fields or "stop_loss" in result.adjusted_fields


def test_buy_bracket_rejects_stop_above_entry():
    result = validate_bracket_prices(
        action="BUY",
        entry_price=100.0,
        stop_loss=101.0,
        take_profit=102.0,
        tick_size=0.25,
    )
    assert not result.valid
    assert "Stop-loss" in (result.reason or "")


def test_atr_fallback_produces_positive_offsets():
    offsets = compute_protective_offsets(
        atr_value=0.0,
        tick_size=0.25,
        scalper=True,
        volatility="LOW",
    )
    assert offsets.fallback_used
    assert offsets.stop_offset > 0
    assert offsets.target_offset > offsets.stop_offset


def test_wait_guard_blocks_when_required():
    wait = WaitDecisionContext(decision="WAIT", advisory_only=False, confidence=0.5)
    assert should_block_on_wait(wait, block_on_wait=True, override_confidence=0.7, signal_confidence=0.6)


def test_wait_guard_allows_high_confidence_override():
    wait = WaitDecisionContext(decision="WAIT", advisory_only=True, confidence=0.5)
    assert not should_block_on_wait(wait, block_on_wait=True, override_confidence=0.7, signal_confidence=0.8)
