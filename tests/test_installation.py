#!/usr/bin/env python3
"""Test script to validate MyTrader installation and run basic tests."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Testing module imports...")
    print("=" * 60)
    
    try:
        # Core modules
        from mytrader.config import Settings
        print("✓ Config module")
        
        from mytrader.data.base import DataCollector
        print("✓ Data base module")
        
        from mytrader.features.feature_engineer import engineer_features
        print("✓ Feature engineering module")
        
        from mytrader.strategies.base import BaseStrategy, Signal
        print("✓ Strategy base module")
        
        from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
        print("✓ RSI MACD Sentiment strategy")
        
        from mytrader.strategies.momentum_reversal import MomentumReversalStrategy
        print("✓ Momentum Reversal strategy")
        
        from mytrader.risk.manager import RiskManager
        print("✓ Risk manager module")
        
        from mytrader.backtesting.engine import BacktestingEngine
        print("✓ Backtesting engine")
        
        from mytrader.backtesting.performance import summarize_performance
        print("✓ Performance analytics")
        
        from mytrader.optimization.optimizer import ParameterOptimizer
        print("✓ Parameter optimizer")
        
        from mytrader.optimization.ml_optimizer import MLParameterOptimizer
        print("✓ ML optimizer (Optuna)")
        
        from mytrader.monitoring.live_tracker import LivePerformanceTracker
        print("✓ Live performance tracker")
        
        from mytrader.utils.logger import logger
        print("✓ Logger utilities")
        
        from mytrader.utils.settings_loader import load_settings
        print("✓ Settings loader")
        
        print("\n✅ All imports successful!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineering():
    """Test feature engineering with sample data."""
    print("=" * 60)
    print("Testing feature engineering...")
    print("=" * 60)
    
    try:
        import pandas as pd
        from mytrader.features.feature_engineer import engineer_features
        
        # Create sample data
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 101.5] * 25,
            'high': [101.0, 102.0, 103.0, 102.5] * 25,
            'low': [99.0, 100.0, 101.0, 100.5] * 25,
            'close': [100.5, 101.5, 102.5, 101.0] * 25,
            'volume': [1000, 1100, 1200, 1150] * 25
        })
        data.index = pd.date_range('2024-01-01', periods=len(data), freq='1min')
        
        # Add sentiment
        sentiment = pd.DataFrame({
            'sentiment': [0.1, 0.2, -0.1, 0.0] * 25
        })
        sentiment.index = data.index
        
        # Engineer features
        features = engineer_features(data, sentiment)
        
        print(f"  Input rows: {len(data)}")
        print(f"  Output rows: {len(features)}")
        print(f"  Features generated: {len(features.columns)}")
        print(f"  Sample indicators: {', '.join(list(features.columns)[:10])}")
        
        # Check for key indicators
        expected_indicators = ['RSI_14', 'MACD_12_26_9', 'ATR_14', 'ADX_14', 'STOCH_K']
        for indicator in expected_indicators:
            if indicator in features.columns:
                print(f"  ✓ {indicator}")
            else:
                print(f"  ✗ {indicator} - MISSING!")
        
        print("\n✅ Feature engineering test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Feature engineering test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_strategy():
    """Test strategy signal generation."""
    print("=" * 60)
    print("Testing strategy signal generation...")
    print("=" * 60)
    
    try:
        import pandas as pd
        from mytrader.strategies.rsi_macd_sentiment import RsiMacdSentimentStrategy
        from mytrader.features.feature_engineer import engineer_features
        
        # Create sample data
        data = pd.DataFrame({
            'open': [100.0 + i*0.5 for i in range(100)],
            'high': [101.0 + i*0.5 for i in range(100)],
            'low': [99.0 + i*0.5 for i in range(100)],
            'close': [100.5 + i*0.5 for i in range(100)],
            'volume': [1000 + i*10 for i in range(100)]
        })
        data.index = pd.date_range('2024-01-01', periods=len(data), freq='1min')
        
        # Engineer features
        features = engineer_features(data, None)
        
        # Create strategy
        strategy = RsiMacdSentimentStrategy()
        
        # Generate signal
        signal = strategy.generate(features)
        
        print(f"  Strategy: {strategy.name}")
        print(f"  Signal: {signal.action}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Metadata: {signal.metadata}")
        
        print("\n✅ Strategy test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Strategy test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_risk_manager():
    """Test risk manager."""
    print("=" * 60)
    print("Testing risk manager...")
    print("=" * 60)
    
    try:
        from mytrader.risk.manager import RiskManager
        from mytrader.config import TradingConfig
        
        config = TradingConfig()
        
        # Test fixed fraction
        risk_fixed = RiskManager(config, position_sizing_method="fixed_fraction")
        qty_fixed = risk_fixed.position_size(100000, 0.8)
        print(f"  Fixed fraction sizing: {qty_fixed} contracts")
        
        # Test Kelly Criterion
        risk_kelly = RiskManager(config, position_sizing_method="kelly")
        qty_kelly = risk_kelly.position_size(
            100000, 0.8, 
            win_rate=0.55, 
            avg_win=500.0, 
            avg_loss=250.0
        )
        print(f"  Kelly criterion sizing: {qty_kelly} contracts")
        
        # Test ATR-based stops
        stop, target = risk_kelly.calculate_dynamic_stops(
            entry_price=4500.0,
            current_atr=15.0,
            direction="long",
            atr_multiplier=2.0
        )
        print(f"  ATR stop: {stop:.2f}, Target: {target:.2f}")
        
        # Test statistics
        risk_kelly.update_pnl(500.0)
        risk_kelly.update_pnl(-250.0)
        risk_kelly.update_pnl(300.0)
        stats = risk_kelly.get_statistics()
        print(f"  Win rate: {stats['win_rate']:.1%}")
        print(f"  Profit factor: {stats['profit_factor']:.2f}")
        
        print("\n✅ Risk manager test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Risk manager test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_performance_tracker():
    """Test live performance tracker."""
    print("=" * 60)
    print("Testing live performance tracker...")
    print("=" * 60)
    
    try:
        from mytrader.monitoring.live_tracker import LivePerformanceTracker
        
        tracker = LivePerformanceTracker(initial_capital=100000)
        
        # Simulate some trades
        tracker.record_trade("BUY", 4500.0, 2)
        tracker.update_equity(4510.0, realized_pnl=0)
        
        tracker.record_trade("SELL", 4510.0, 2)
        tracker.update_equity(4510.0, realized_pnl=1000.0)
        
        tracker.record_trade("BUY", 4505.0, 2)
        tracker.update_equity(4500.0, realized_pnl=-500.0)
        
        # Get snapshot
        snapshot = tracker.get_snapshot()
        print(f"  Equity: ${snapshot.equity:,.2f}")
        print(f"  Total PnL: ${snapshot.total_pnl:,.2f}")
        print(f"  Trade count: {snapshot.trade_count}")
        print(f"  Win rate: {snapshot.win_rate:.1%}")
        
        print("\n✅ Performance tracker test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Performance tracker test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MyTrader Installation & Unit Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_imports,
        test_feature_engineering,
        test_strategy,
        test_risk_manager,
        test_performance_tracker
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed! System is ready.\n")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
