import math
import pytest
from unittest.mock import Mock, patch

from mytrader.execution.order_builder import validate_bracket_prices
from mytrader.execution.guards import WaitDecisionContext, should_block_on_wait, compute_trade_risk_dollars
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


def test_bracket_validation_enforces_minimum_distance_ticks():
    """Test that bracket validation enforces minimum distance in ticks (default 4 ticks)."""
    # For ES: tick_size=0.25, min_distance_ticks=4, so min_distance=1.0
    result = validate_bracket_prices(
        action="BUY",
        entry_price=100.0,
        stop_loss=99.5,  # Only 0.5 away (2 ticks) - should be adjusted
        take_profit=100.5,  # Only 0.5 away (2 ticks) - should be adjusted
        tick_size=0.25,
        min_distance_ticks=4,  # 4 ticks = 1.0 point
    )
    assert result.valid
    # Should be adjusted to at least 1.0 away
    assert abs(result.stop_loss - 100.0) >= 1.0
    assert abs(result.take_profit - 100.0) >= 1.0


def test_bracket_validation_rejects_immediate_fill_distances():
    """Test that brackets too close to entry are rejected/adjusted."""
    result = validate_bracket_prices(
        action="BUY",
        entry_price=6890.50,
        stop_loss=6890.25,  # Only 0.25 away (1 tick) - too close
        take_profit=6890.75,  # Only 0.25 away (1 tick) - too close
        tick_size=0.25,
        min_distance_ticks=4,
    )
    assert result.valid  # Should be adjusted, not rejected
    assert abs(result.stop_loss - 6890.50) >= 1.0  # At least 4 ticks away
    assert abs(result.take_profit - 6890.50) >= 1.0


def test_validate_entry_guard_blocks_missing_protection():
    """Test that _validate_entry_guard blocks trades with None SL/TP."""
    from mytrader.execution.live_trading_manager import LiveTradingManager
    from mytrader.config import Settings
    
    settings = Settings()
    manager = LiveTradingManager(settings, simulation_mode=True)
    
    # Missing stop loss
    assert not manager._validate_entry_guard(
        entry_price=100.0,
        stop_loss=None,
        take_profit=102.0,
        quantity=1,
        action="BUY",
    )
    
    # Missing take profit
    assert not manager._validate_entry_guard(
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=None,
        quantity=1,
        action="BUY",
    )


def test_validate_entry_guard_blocks_invalid_bracket_orientation():
    """Test that _validate_entry_guard blocks invalid bracket orientation."""
    from mytrader.execution.live_trading_manager import LiveTradingManager
    from mytrader.config import Settings
    
    settings = Settings()
    manager = LiveTradingManager(settings, simulation_mode=True)
    
    # BUY with stop above entry (invalid)
    assert not manager._validate_entry_guard(
        entry_price=100.0,
        stop_loss=101.0,  # Wrong side for BUY
        take_profit=102.0,
        quantity=1,
        action="BUY",
    )
    
    # BUY with target below entry (invalid)
    assert not manager._validate_entry_guard(
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=99.5,  # Wrong side for BUY
        quantity=1,
        action="BUY",
    )
    
    # SELL with stop below entry (invalid)
    assert not manager._validate_entry_guard(
        entry_price=100.0,
        stop_loss=99.0,  # Wrong side for SELL
        take_profit=98.0,
        quantity=1,
        action="SELL",
    )
    
    # SELL with target above entry (invalid)
    assert not manager._validate_entry_guard(
        entry_price=100.0,
        stop_loss=101.0,
        take_profit=101.5,  # Wrong side for SELL
        quantity=1,
        action="SELL",
    )


def test_validate_entry_guard_blocks_insufficient_distance():
    """Test that _validate_entry_guard blocks trades with insufficient distance."""
    from mytrader.execution.live_trading_manager import LiveTradingManager
    from mytrader.config import Settings
    
    settings = Settings()
    settings.trading.tick_size = 0.25
    manager = LiveTradingManager(settings, simulation_mode=True)
    
    # Stop too close (less than 4 ticks = 1.0 point)
    assert not manager._validate_entry_guard(
        entry_price=100.0,
        stop_loss=99.5,  # Only 0.5 away (2 ticks) - insufficient
        take_profit=102.0,
        quantity=1,
        action="BUY",
    )
    
    # Target too close
    assert not manager._validate_entry_guard(
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=100.5,  # Only 0.5 away - insufficient
        quantity=1,
        action="BUY",
    )


def test_validate_entry_guard_blocks_excessive_risk():
    """Test that _validate_entry_guard blocks trades exceeding max_loss_per_trade."""
    from mytrader.execution.live_trading_manager import LiveTradingManager
    from mytrader.config import Settings
    
    settings = Settings()
    settings.trading.max_loss_per_trade = 1250.0
    settings.trading.contract_multiplier = 50.0
    manager = LiveTradingManager(settings, simulation_mode=True)
    
    # Risk = (100.0 - 90.0) * 50.0 * 2 = 1000.0 (OK)
    assert manager._validate_entry_guard(
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=110.0,
        quantity=2,
        action="BUY",
    )
    
    # Risk = (100.0 - 75.0) * 50.0 * 1 = 1250.0 (at limit, should pass)
    assert manager._validate_entry_guard(
        entry_price=100.0,
        stop_loss=75.0,
        take_profit=125.0,
        quantity=1,
        action="BUY",
    )
    
    # Risk = (100.0 - 74.0) * 50.0 * 1 = 1300.0 (exceeds limit)
    assert not manager._validate_entry_guard(
        entry_price=100.0,
        stop_loss=74.0,
        take_profit=126.0,
        quantity=1,
        action="BUY",
    )


def test_validate_entry_guard_allows_valid_brackets():
    """Test that _validate_entry_guard allows valid brackets."""
    from mytrader.execution.live_trading_manager import LiveTradingManager
    from mytrader.config import Settings
    
    settings = Settings()
    settings.trading.max_loss_per_trade = 1250.0
    settings.trading.contract_multiplier = 50.0
    settings.trading.tick_size = 0.25
    manager = LiveTradingManager(settings, simulation_mode=True)
    
    # Valid BUY bracket: stop < entry < target, sufficient distance
    assert manager._validate_entry_guard(
        entry_price=6890.50,
        stop_loss=6889.50,  # 1.0 point below (4 ticks)
        take_profit=6892.50,  # 2.0 points above (8 ticks)
        quantity=1,
        action="BUY",
    )
    
    # Valid SELL bracket: target < entry < stop, sufficient distance
    assert manager._validate_entry_guard(
        entry_price=6890.50,
        stop_loss=6891.50,  # 1.0 point above (4 ticks)
        take_profit=6888.50,  # 2.0 points below (8 ticks)
        quantity=1,
        action="SELL",
    )


def test_wait_guard_always_blocks_non_advisory_wait():
    """Test that WAIT always blocks when not advisory-only, regardless of signal confidence."""
    wait = WaitDecisionContext(decision="WAIT", advisory_only=False, confidence=0.5)
    # Should block even with high signal confidence
    assert should_block_on_wait(wait, block_on_wait=True, override_confidence=0.7, signal_confidence=0.9)


def test_wait_guard_respects_block_on_wait_flag():
    """Test that WAIT only blocks if block_on_wait=True."""
    wait = WaitDecisionContext(decision="WAIT", advisory_only=False, confidence=0.5)
    # Should not block if block_on_wait=False
    assert not should_block_on_wait(wait, block_on_wait=False, override_confidence=0.7, signal_confidence=0.6)


def test_compute_trade_risk_dollars():
    """Test trade risk calculation."""
    # BUY: entry=100, stop=99, multiplier=50, risk = 1.0 * 50 = 50
    risk = compute_trade_risk_dollars(100.0, 99.0, 50.0)
    assert risk == 50.0
    
    # SELL: entry=100, stop=101, multiplier=50, risk = 1.0 * 50 = 50
    risk = compute_trade_risk_dollars(100.0, 101.0, 50.0)
    assert risk == 50.0


def test_bracket_validation_handles_none_values():
    """Test that bracket validation handles None values correctly."""
    # None stop_loss should be allowed if take_profit is provided
    result = validate_bracket_prices(
        action="BUY",
        entry_price=100.0,
        stop_loss=None,
        take_profit=102.0,
        tick_size=0.25,
    )
    # Should be valid (only validates what's provided)
    assert result.valid
    assert result.stop_loss is None
    assert result.take_profit == 102.0


def test_bracket_validation_rejects_negative_values():
    """Test that bracket validation rejects negative/zero values."""
    result = validate_bracket_prices(
        action="BUY",
        entry_price=100.0,
        stop_loss=-1.0,  # Negative
        take_profit=102.0,
        tick_size=0.25,
    )
    assert not result.valid
    assert "stop_loss must be positive" in (result.reason or "")
    
    result = validate_bracket_prices(
        action="BUY",
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=0.0,  # Zero
        tick_size=0.25,
    )
    assert not result.valid
    assert "take_profit must be positive" in (result.reason or "")
