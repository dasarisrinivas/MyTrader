import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from mytrader.backtesting.engine import BacktestingEngine
from mytrader.config import TradingConfig, BacktestConfig
from mytrader.strategies.base import BaseStrategy, Signal

class AggressiveStrategy(BaseStrategy):
    def __init__(self, config):
        self.config = config
        self.name = "Aggressive"
        
    def generate(self, features: pd.DataFrame) -> Signal:
        # Always buy with max confidence
        return Signal(action="BUY", confidence=1.0, metadata={"position_scaler": 10.0})

@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="5min")
    data = pd.DataFrame({
        "open": np.linspace(100, 110, 100),
        "high": np.linspace(101, 111, 100),
        "low": np.linspace(99, 109, 100),
        "close": np.linspace(100, 110, 100),
        "volume": np.random.randint(100, 1000, 100),
        "ATR_14": [1.0] * 100
    }, index=dates)
    return data

def test_backtest_max_contracts_enforcement(sample_data):
    # Configure with strict limit
    trading_config = TradingConfig()
    trading_config.max_contracts_limit = 5
    trading_config.max_position_size = 5
    trading_config.initial_capital = 1_000_000 # Plenty of capital
    
    backtest_config = BacktestConfig()
    backtest_config.initial_capital = 1_000_000
    
    strategy = AggressiveStrategy(trading_config)
    
    engine = BacktestingEngine([strategy], trading_config, backtest_config)
    
    # Mock feature engineering to just return data + ATR
    with pytest.MonkeyPatch.context() as m:
        m.setattr("mytrader.backtesting.engine.engineer_features", lambda d, s=None: d.assign(ATR_14=1.0))
        
        result = engine.run(sample_data)
        
    # Assertions
    assert result.metrics["max_concurrent_contracts"] <= 5
    assert result.metrics["max_concurrent_contracts"] > 0 # Should have traded
    
    # Verify individual trades don't exceed 5
    for trade in result.trades:
        if trade["action"] in ["BUY", "SELL"]:
            assert abs(trade["qty"]) <= 5

def test_backtest_respects_lower_limit(sample_data):
    # Test with lower limit
    trading_config = TradingConfig()
    trading_config.max_contracts_limit = 2
    trading_config.max_position_size = 2 # Should be synced
    trading_config.initial_capital = 1_000_000
    
    backtest_config = BacktestConfig()
    
    strategy = AggressiveStrategy(trading_config)
    
    engine = BacktestingEngine([strategy], trading_config, backtest_config)
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr("mytrader.backtesting.engine.engineer_features", lambda d, s=None: d.assign(ATR_14=1.0))
        
        result = engine.run(sample_data)
        
    assert result.metrics["max_concurrent_contracts"] <= 2
