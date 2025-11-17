"""
Unit tests for Risk Management Configuration
Tests validation, serialization, and risk parameter calculations
"""
import pytest
import json
import tempfile
from pathlib import Path

from mytrader.risk.config import (
    StopLossConfig,
    TakeProfitConfig,
    PositionSizingConfig,
    RiskLimitsConfig,
    RiskManagementConfig,
    DEFAULT_RISK_CONFIG
)


class TestStopLossConfig:
    """Test stop-loss configuration"""
    
    def test_default_config(self):
        """Test default stop-loss configuration"""
        config = StopLossConfig()
        
        assert config.type == "atr_based"
        assert config.atr_multiplier == 2.0
        assert config.use_trailing is True
    
    def test_fixed_ticks_stop_distance(self):
        """Test fixed ticks stop distance calculation"""
        config = StopLossConfig(type="fixed_ticks", fixed_ticks=10.0)
        
        distance = config.get_stop_distance(tick_size=0.25)
        assert distance == 2.5  # 10 * 0.25
    
    def test_atr_based_stop_distance(self):
        """Test ATR-based stop distance calculation"""
        config = StopLossConfig(type="atr_based", atr_multiplier=2.0)
        
        distance = config.get_stop_distance(atr=1.5)
        assert distance == 3.0  # 2.0 * 1.5
    
    def test_atr_based_fallback_on_zero_atr(self):
        """Test fallback to fixed ticks when ATR is zero"""
        config = StopLossConfig(
            type="atr_based",
            atr_multiplier=2.0,
            fixed_ticks=10.0
        )
        
        distance = config.get_stop_distance(atr=0.0, tick_size=0.25)
        assert distance == 2.5  # Falls back to fixed ticks


class TestTakeProfitConfig:
    """Test take-profit configuration"""
    
    def test_default_config(self):
        """Test default take-profit configuration"""
        config = TakeProfitConfig()
        
        assert config.type == "risk_reward_ratio"
        assert config.risk_reward_ratio == 2.0
    
    def test_fixed_ticks_take_profit(self):
        """Test fixed ticks take-profit calculation"""
        config = TakeProfitConfig(type="fixed_ticks", fixed_ticks=20.0)
        
        distance = config.get_take_profit_distance(stop_distance=0, tick_size=0.25)
        assert distance == 5.0  # 20 * 0.25
    
    def test_risk_reward_ratio_take_profit(self):
        """Test risk:reward ratio take-profit calculation"""
        config = TakeProfitConfig(
            type="risk_reward_ratio",
            risk_reward_ratio=2.0
        )
        
        stop_distance = 3.0
        distance = config.get_take_profit_distance(stop_distance=stop_distance)
        assert distance == 6.0  # 3.0 * 2.0
    
    def test_atr_based_take_profit(self):
        """Test ATR-based take-profit calculation"""
        config = TakeProfitConfig(type="atr_based", atr_multiplier=3.0)
        
        distance = config.get_take_profit_distance(stop_distance=0, atr=1.5)
        assert distance == 4.5  # 3.0 * 1.5
    
    def test_partial_exits_configuration(self):
        """Test partial exits configuration"""
        config = TakeProfitConfig(
            use_partial_exits=True,
            partial_exit_levels=[0.5, 0.75],
            partial_exit_sizes=[0.3, 0.3]
        )
        
        assert config.use_partial_exits is True
        assert len(config.partial_exit_levels) == 2
        assert len(config.partial_exit_sizes) == 2


class TestPositionSizingConfig:
    """Test position sizing configuration"""
    
    def test_default_config(self):
        """Test default position sizing configuration"""
        config = PositionSizingConfig()
        
        assert config.method == "kelly"
        assert config.kelly_fraction == 0.5
        assert config.scale_by_confidence is True
    
    def test_fixed_sizing(self):
        """Test fixed position sizing"""
        config = PositionSizingConfig(method="fixed", fixed_size=2)
        
        assert config.method == "fixed"
        assert config.fixed_size == 2
    
    def test_risk_per_trade(self):
        """Test risk per trade configuration"""
        config = PositionSizingConfig(risk_per_trade_pct=1.5)
        
        assert config.risk_per_trade_pct == 1.5


class TestRiskLimitsConfig:
    """Test risk limits configuration"""
    
    def test_default_limits(self):
        """Test default risk limits"""
        config = RiskLimitsConfig()
        
        assert config.max_daily_loss == 2000.0
        assert config.max_daily_trades == 20
        assert config.enable_circuit_breaker is True
    
    def test_circuit_breaker(self):
        """Test circuit breaker configuration"""
        config = RiskLimitsConfig(
            enable_circuit_breaker=True,
            circuit_breaker_loss_pct=5.0
        )
        
        assert config.enable_circuit_breaker is True
        assert config.circuit_breaker_loss_pct == 5.0
    
    def test_time_restrictions(self):
        """Test time-based trading restrictions"""
        config = RiskLimitsConfig(
            no_trade_hours=[12, 13],  # No trading during lunch
            no_trade_days=["Monday"]
        )
        
        assert 12 in config.no_trade_hours
        assert "Monday" in config.no_trade_days


class TestRiskManagementConfig:
    """Test complete risk management configuration"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = RiskManagementConfig()
        
        assert isinstance(config.stop_loss, StopLossConfig)
        assert isinstance(config.take_profit, TakeProfitConfig)
        assert isinstance(config.position_sizing, PositionSizingConfig)
        assert isinstance(config.limits, RiskLimitsConfig)
    
    def test_validation_success(self):
        """Test validation passes with valid config"""
        config = RiskManagementConfig()
        
        assert config.validate() is True
    
    def test_validation_fails_invalid_kelly_fraction(self):
        """Test validation fails with invalid Kelly fraction"""
        config = RiskManagementConfig()
        config.position_sizing.kelly_fraction = 1.5  # > 1.0, invalid
        
        assert config.validate() is False
    
    def test_validation_fails_invalid_risk_reward(self):
        """Test validation fails with invalid risk:reward ratio"""
        config = RiskManagementConfig()
        config.take_profit.risk_reward_ratio = -1.0  # Negative, invalid
        
        assert config.validate() is False
    
    def test_validation_fails_invalid_atr_multiplier(self):
        """Test validation fails with invalid ATR multiplier"""
        config = RiskManagementConfig()
        config.stop_loss.atr_multiplier = -0.5  # Negative, invalid
        
        assert config.validate() is False
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary"""
        config = RiskManagementConfig()
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "stop_loss" in config_dict
        assert "take_profit" in config_dict
        assert "position_sizing" in config_dict
        assert "limits" in config_dict
    
    def test_from_dict_conversion(self):
        """Test creation from dictionary"""
        config_dict = {
            "stop_loss": {
                "type": "atr_based",
                "atr_multiplier": 2.5
            },
            "take_profit": {
                "type": "risk_reward_ratio",
                "risk_reward_ratio": 3.0
            },
            "position_sizing": {
                "method": "kelly",
                "kelly_fraction": 0.25
            },
            "limits": {
                "max_daily_loss": 1500.0
            }
        }
        
        config = RiskManagementConfig.from_dict(config_dict)
        
        assert config.stop_loss.atr_multiplier == 2.5
        assert config.take_profit.risk_reward_ratio == 3.0
        assert config.position_sizing.kelly_fraction == 0.25
        assert config.limits.max_daily_loss == 1500.0


class TestConfigSerialization:
    """Test JSON serialization and deserialization"""
    
    def test_save_and_load_json(self):
        """Test saving and loading configuration from JSON"""
        config = RiskManagementConfig()
        config.stop_loss.atr_multiplier = 2.5
        config.limits.max_daily_loss = 3000.0
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            config.save_to_json(temp_path)
            
            # Load
            loaded_config = RiskManagementConfig.from_json(temp_path)
            
            assert loaded_config.stop_loss.atr_multiplier == 2.5
            assert loaded_config.limits.max_daily_loss == 3000.0
        finally:
            Path(temp_path).unlink()
    
    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file returns default config"""
        config = RiskManagementConfig.from_json("/nonexistent/path.json")
        
        # Should return default config without crashing
        assert isinstance(config, RiskManagementConfig)
    
    def test_roundtrip_serialization(self):
        """Test that config survives save/load roundtrip"""
        original = RiskManagementConfig()
        original.stop_loss.type = "percentage"
        original.stop_loss.percentage = 1.5
        original.take_profit.use_partial_exits = True
        original.position_sizing.method = "risk_parity"
        original.limits.no_trade_hours = [9, 10, 11]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            original.save_to_json(temp_path)
            loaded = RiskManagementConfig.from_json(temp_path)
            
            assert loaded.stop_loss.type == original.stop_loss.type
            assert loaded.stop_loss.percentage == original.stop_loss.percentage
            assert loaded.take_profit.use_partial_exits == original.take_profit.use_partial_exits
            assert loaded.position_sizing.method == original.position_sizing.method
            assert loaded.limits.no_trade_hours == original.limits.no_trade_hours
        finally:
            Path(temp_path).unlink()


class TestDefaultConfig:
    """Test default configuration constant"""
    
    def test_default_config_exists(self):
        """Test that default config constant exists"""
        assert DEFAULT_RISK_CONFIG is not None
        assert isinstance(DEFAULT_RISK_CONFIG, RiskManagementConfig)
    
    def test_default_config_is_valid(self):
        """Test that default config passes validation"""
        assert DEFAULT_RISK_CONFIG.validate() is True


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_zero_risk_per_trade(self):
        """Test validation with zero risk per trade"""
        config = RiskManagementConfig()
        config.position_sizing.risk_per_trade_pct = 0.0
        
        assert config.validate() is False
    
    def test_excessive_risk_per_trade(self):
        """Test validation with excessive risk per trade"""
        config = RiskManagementConfig()
        config.position_sizing.risk_per_trade_pct = 15.0  # > 10%
        
        assert config.validate() is False
    
    def test_negative_daily_loss_limit(self):
        """Test validation with negative daily loss limit"""
        config = RiskManagementConfig()
        config.limits.max_daily_loss = -100.0
        
        assert config.validate() is False
    
    def test_zero_daily_trades_limit(self):
        """Test validation with zero daily trades limit"""
        config = RiskManagementConfig()
        config.limits.max_daily_trades = 0
        
        assert config.validate() is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
