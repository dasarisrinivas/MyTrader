"""Unit tests for risk management."""
import pytest
import numpy as np
from mytrader.risk.manager import RiskManager
from mytrader.config import TradingConfig


@pytest.fixture
def default_config():
    """Create default trading configuration."""
    return TradingConfig(
        initial_capital=100000.0,
        max_position_size=4,
        max_daily_loss=2000.0,
        max_daily_trades=20,
        stop_loss_ticks=10.0,
        take_profit_ticks=20.0,
        tick_size=0.25,
        tick_value=12.5,
        commission_per_contract=2.4,
        contract_multiplier=50.0
    )


class TestRiskManager:
    """Test RiskManager functionality."""
    
    def test_initialization(self, default_config):
        """Test risk manager initialization."""
        risk = RiskManager(default_config, position_sizing_method="fixed_fraction")
        assert risk.daily_loss == 0.0
        assert risk.trade_count == 0
        assert risk.position_sizing_method == "fixed_fraction"
        
    def test_can_trade_limits(self, default_config):
        """Test trade limit checks."""
        risk = RiskManager(default_config)
        
        # Should be able to trade initially
        assert risk.can_trade(2) is True
        
        # Exceed max position size
        assert risk.can_trade(10) is False
        
        # Exceed daily trades
        for _ in range(20):
            risk.register_trade()
        assert risk.can_trade(1) is False
        
        # Exceed daily loss
        risk.reset()
        risk.daily_loss = 2500.0
        assert risk.can_trade(1) is False
        
    def test_update_pnl_tracking(self, default_config):
        """Test PnL tracking updates."""
        risk = RiskManager(default_config)
        
        # Record winning trade
        risk.update_pnl(500.0)
        assert risk.winning_trades == 1
        assert risk.total_wins == 500.0
        assert risk.daily_loss == -500.0  # Negative because profit reduces loss
        
        # Record losing trade
        risk.update_pnl(-250.0)
        assert risk.losing_trades == 1
        assert risk.total_losses == 250.0
        assert risk.daily_loss == -250.0
        
    def test_fixed_fraction_position_sizing(self, default_config):
        """Test fixed fraction position sizing."""
        risk = RiskManager(default_config, position_sizing_method="fixed_fraction")
        
        # High confidence
        qty_high = risk.position_size(100000, 0.9)
        assert 1 <= qty_high <= default_config.max_position_size
        
        # Low confidence
        qty_low = risk.position_size(100000, 0.3)
        assert 1 <= qty_low <= default_config.max_position_size
        assert qty_low <= qty_high  # Lower confidence should result in smaller position
        
    def test_kelly_criterion_position_sizing(self, default_config):
        """Test Kelly Criterion position sizing."""
        risk = RiskManager(default_config, position_sizing_method="kelly")
        
        # With historical data
        qty = risk.position_size(
            100000, 0.8,
            win_rate=0.6,
            avg_win=500.0,
            avg_loss=250.0
        )
        assert 1 <= qty <= default_config.max_position_size
        
        # Without historical data (uses defaults)
        qty_default = risk.position_size(100000, 0.8)
        assert 1 <= qty_default <= default_config.max_position_size
        
    def test_calculate_dynamic_stops(self, default_config):
        """Test ATR-based dynamic stops."""
        risk = RiskManager(default_config)
        
        entry_price = 4500.0
        current_atr = 15.0
        
        # Long position
        stop_long, target_long = risk.calculate_dynamic_stops(
            entry_price, current_atr, "long", atr_multiplier=2.0
        )
        assert stop_long < entry_price
        assert target_long > entry_price
        assert (target_long - entry_price) > (entry_price - stop_long)  # Risk/reward ratio
        
        # Short position
        stop_short, target_short = risk.calculate_dynamic_stops(
            entry_price, current_atr, "short", atr_multiplier=2.0
        )
        assert stop_short > entry_price
        assert target_short < entry_price
        
    def test_calculate_trailing_stop(self, default_config):
        """Test trailing stop calculation."""
        risk = RiskManager(default_config)
        
        entry_price = 4500.0
        current_atr = 15.0
        
        # Long position - price moved up
        trailing_stop = risk.calculate_trailing_stop(
            entry_price, 4550.0, "long",
            current_atr=current_atr, atr_multiplier=1.5
        )
        assert trailing_stop > entry_price - current_atr * 1.5
        assert trailing_stop < 4550.0
        
        # Short position - price moved down
        trailing_stop_short = risk.calculate_trailing_stop(
            entry_price, 4450.0, "short",
            current_atr=current_atr, atr_multiplier=1.5
        )
        assert trailing_stop_short < entry_price + current_atr * 1.5
        assert trailing_stop_short > 4450.0
        
    def test_get_statistics(self, default_config):
        """Test statistics calculation."""
        risk = RiskManager(default_config)
        
        # Simulate some trades
        risk.update_pnl(500.0)
        risk.update_pnl(300.0)
        risk.update_pnl(-200.0)
        risk.update_pnl(-150.0)
        risk.register_trade()
        risk.register_trade()
        risk.register_trade()
        risk.register_trade()
        
        stats = risk.get_statistics()
        
        assert stats["total_trades"] == 4
        assert stats["winning_trades"] == 2
        assert stats["losing_trades"] == 2
        assert stats["win_rate"] == 0.5
        assert stats["avg_win"] == 400.0
        assert stats["avg_loss"] == 175.0
        assert stats["profit_factor"] > 1.0
        
    def test_reset_functionality(self, default_config):
        """Test reset methods."""
        risk = RiskManager(default_config)
        
        # Add some activity
        risk.update_pnl(500.0)
        risk.register_trade()
        risk.register_trade()
        
        # Reset daily counters
        risk.reset()
        assert risk.daily_loss == 0.0
        assert risk.trade_count == 0
        assert risk.winning_trades == 1  # Stats preserved
        
        # Reset all stats
        risk.reset_stats()
        assert risk.winning_trades == 0
        assert risk.total_wins == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
