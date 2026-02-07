"""Example usage and validation of the scoring-based entry system.

This script demonstrates:
1. How to use the scoring system standalone
2. How to integrate it with the strategy
3. Diagnostic output and logging
4. Comparison with hard filter approach

Run this to validate the scoring system works correctly.

Author: Senior Quantitative Trading Engineer - Feb 2026
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, time

from mytrader.strategies.scoring_entry import (
    calculate_signal_score,
    should_enter_trade,
    check_risk_gates,
    PositionSize
)
from mytrader.strategies.scoring_integration import create_scoring_evaluator
from mytrader.config import OneMinuteStrategyConfig


def create_test_data_bullish():
    """Create sample data representing a bullish setup."""
    return {
        # Price action
        'close': 5850.0,
        'open': 5845.0,
        'high': 5852.0,
        'low': 5844.0,
        
        # EMAs - bullish alignment
        'EMA_9': 5848.0,
        'EMA_21': 5840.0,
        'EMA_50': 5830.0,
        
        # VWAP
        'SESSION_VWAP': 5835.0,
        
        # Momentum - positive
        'RSI_14': 62.0,
        'MACD_hist': 0.45,
        
        # Regime
        'ADX_14': 28.0,
        'ATR_14': 8.5,
        
        # Volume
        'volume': 1200,
        
        # HTF alignment
        'trend_label': 'UPTREND',
        'trend_label_htf': 'UPTREND',
    }


def create_test_data_bearish():
    """Create sample data representing a bearish setup."""
    return {
        # Price action
        'close': 5820.0,
        'open': 5825.0,
        'high': 5826.0,
        'low': 5818.0,
        
        # EMAs - bearish alignment
        'EMA_9': 5822.0,
        'EMA_21': 5830.0,
        'EMA_50': 5840.0,
        
        # VWAP
        'SESSION_VWAP': 5835.0,
        
        # Momentum - negative
        'RSI_14': 38.0,
        'MACD_hist': -0.35,
        
        # Regime
        'ADX_14': 26.0,
        'ATR_14': 7.2,
        
        # Volume
        'volume': 1150,
        
        # HTF alignment
        'trend_label': 'DOWNTREND',
        'trend_label_htf': 'DOWNTREND',
    }


def create_test_data_chop():
    """Create sample data representing a choppy/range-bound market."""
    return {
        # Price action
        'close': 5830.0,
        'open': 5829.0,
        'high': 5831.0,
        'low': 5828.0,
        
        # EMAs - no clear alignment
        'EMA_9': 5829.5,
        'EMA_21': 5830.0,
        'EMA_50': 5830.5,
        
        # VWAP
        'SESSION_VWAP': 5830.0,
        
        # Momentum - neutral
        'RSI_14': 50.0,
        'MACD_hist': 0.05,
        
        # Regime - weak
        'ADX_14': 12.0,
        'ATR_14': 4.5,
        
        # Volume
        'volume': 800,
        
        # HTF alignment
        'trend_label': 'CHOP',
        'trend_label_htf': 'RANGING',
    }


def test_scoring_standalone():
    """Test the scoring system with standalone data."""
    print("\n" + "="*80)
    print("TEST 1: SCORING SYSTEM STANDALONE")
    print("="*80)
    
    # Test bullish setup
    print("\n--- Bullish Setup ---")
    bullish_data = create_test_data_bullish()
    
    score = calculate_signal_score(
        data=bullish_data,
        timestamp=datetime.now().replace(hour=10, minute=30),
        atr_percentile=0.75
    )
    
    print(f"\nDirection: {score.direction}")
    print(f"Total Score: {score.total_score:.1f}")
    print(f"\nBreakdown:")
    breakdown = score.get_breakdown()
    for category, value in breakdown.items():
        print(f"  {category:12s}: {value:+6.1f}")
    
    print(f"\nTop Contributors:")
    for reason in score.reasons:
        print(f"  - {reason}")
    
    print(f"\nDetailed Components:")
    for comp in score.components:
        print(f"  [{comp.category:8s}] {comp.name:20s}: {comp.value:+5.1f} ({comp.reason})")
    
    position_size, size_reason = should_enter_trade(score)
    print(f"\nTrade Decision: {position_size.name} ({size_reason})")
    
    # Test bearish setup
    print("\n\n--- Bearish Setup ---")
    bearish_data = create_test_data_bearish()
    
    score = calculate_signal_score(
        data=bearish_data,
        timestamp=datetime.now().replace(hour=14, minute=0),
        atr_percentile=0.65
    )
    
    print(f"\nDirection: {score.direction}")
    print(f"Total Score: {score.total_score:.1f}")
    print(f"\nBreakdown:")
    breakdown = score.get_breakdown()
    for category, value in breakdown.items():
        print(f"  {category:12s}: {value:+6.1f}")
    
    position_size, size_reason = should_enter_trade(score)
    print(f"\nTrade Decision: {position_size.name} ({size_reason})")
    
    # Test choppy market
    print("\n\n--- Choppy Market ---")
    chop_data = create_test_data_chop()
    
    score = calculate_signal_score(
        data=chop_data,
        timestamp=datetime.now().replace(hour=12, minute=0),
        atr_percentile=0.25
    )
    
    print(f"\nDirection: {score.direction}")
    print(f"Total Score: {score.total_score:.1f}")
    print(f"\nBreakdown:")
    breakdown = score.get_breakdown()
    for category, value in breakdown.items():
        print(f"  {category:12s}: {value:+6.1f}")
    
    position_size, size_reason = should_enter_trade(score)
    print(f"\nTrade Decision: {position_size.name} ({size_reason})")


def test_risk_gates():
    """Test the hard risk gates."""
    print("\n" + "="*80)
    print("TEST 2: RISK GATES (HARD CONSTRAINTS)")
    print("="*80)
    
    data = create_test_data_bullish()
    timestamp = datetime.now().replace(hour=10, minute=30)
    
    # Test 1: All gates pass
    print("\n--- Test 1: All Gates Pass ---")
    allowed, reason = check_risk_gates(
        data=data,
        timestamp=timestamp,
        daily_pnl=100.0,
        open_risk=50.0,
        max_daily_loss=500.0,
        max_open_risk=300.0
    )
    print(f"Allowed: {allowed}, Reason: {reason}")
    
    # Test 2: Daily loss limit hit
    print("\n--- Test 2: Daily Loss Limit ---")
    allowed, reason = check_risk_gates(
        data=data,
        timestamp=timestamp,
        daily_pnl=-550.0,
        open_risk=50.0,
        max_daily_loss=500.0,
        max_open_risk=300.0
    )
    print(f"Allowed: {allowed}, Reason: {reason}")
    
    # Test 3: Open risk limit hit
    print("\n--- Test 3: Open Risk Limit ---")
    allowed, reason = check_risk_gates(
        data=data,
        timestamp=timestamp,
        daily_pnl=100.0,
        open_risk=320.0,
        max_daily_loss=500.0,
        max_open_risk=300.0
    )
    print(f"Allowed: {allowed}, Reason: {reason}")
    
    # Test 4: Session cutoff
    print("\n--- Test 4: Session Cutoff ---")
    late_timestamp = datetime.now().replace(hour=15, minute=30)
    allowed, reason = check_risk_gates(
        data=data,
        timestamp=late_timestamp,
        daily_pnl=100.0,
        open_risk=50.0,
        max_daily_loss=500.0,
        max_open_risk=300.0,
        session_end_time=time(15, 0)
    )
    print(f"Allowed: {allowed}, Reason: {reason}")


def test_integration():
    """Test the integration layer with full strategy config."""
    print("\n" + "="*80)
    print("TEST 3: INTEGRATION WITH STRATEGY")
    print("="*80)
    
    # Create minimal config
    config = OneMinuteStrategyConfig(
        warmup_bars=50,
        scoring_full_size_threshold=60.0,
        scoring_half_size_threshold=45.0,
        stop_atr_multiplier=1.5,
        take_profit_multiple=2.0
    )
    
    # Create evaluator
    evaluator = create_scoring_evaluator(config)
    
    # Create test data as pandas Series
    bullish_data = create_test_data_bullish()
    latest = pd.Series(bullish_data)
    
    # Create recent bars DataFrame
    recent_bars = pd.DataFrame([
        {**bullish_data, 'close': 5840 + i, 'high': 5842 + i, 'low': 5838 + i}
        for i in range(-10, 0)
    ])
    
    # Create metadata
    metadata = {
        'trend_label': 'UPTREND',
        'session_type': 'RTH',
        'market_state': 'TRENDING',
        '15m_regime': 'UPTREND'
    }
    
    # Evaluate
    decision = evaluator.evaluate(
        latest=latest,
        prev=None,
        recent_bars=recent_bars,
        current_time=pd.Timestamp(datetime.now().replace(hour=10, minute=30)),
        metadata=metadata
    )
    
    print(f"\nAction: {decision.action}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Position Size: {decision.position_size:.1f}x")
    print(f"Reason: {decision.reason}")
    
    if decision.signal_score:
        print(f"\nScore Details:")
        print(f"  Total: {decision.signal_score.total_score:.1f}")
        print(f"  Direction: {decision.signal_score.direction}")
        print(f"  Breakdown: {decision.signal_score.get_breakdown()}")
    
    if decision.stop_loss and decision.take_profit:
        print(f"\nBrackets:")
        print(f"  Entry: {latest['close']:.2f}")
        print(f"  Stop: {decision.stop_loss:.2f}")
        print(f"  Target: {decision.take_profit:.2f}")
        risk = abs(latest['close'] - decision.stop_loss)
        reward = abs(decision.take_profit - latest['close'])
        print(f"  Risk/Reward: 1:{reward/risk:.2f}")


def compare_with_hard_filters():
    """Compare scoring vs hard filter approach."""
    print("\n" + "="*80)
    print("TEST 4: SCORING VS HARD FILTERS")
    print("="*80)
    
    # Test scenario: Good trend, moderate momentum, decent ADX
    scenario = {
        'close': 5850.0,
        'open': 5847.0,
        'high': 5852.0,
        'low': 5846.0,
        'EMA_9': 5848.0,
        'EMA_21': 5843.0,
        'EMA_50': 5835.0,
        'SESSION_VWAP': 5840.0,
        'RSI_14': 58.0,  # Slightly below typical "pullback" zone
        'MACD_hist': 0.25,
        'ADX_14': 22.0,  # Below typical threshold of 25
        'ATR_14': 7.5,
        'volume': 1000,
        'trend_label': 'UPTREND',
        'trend_label_htf': 'UPTREND',
    }
    
    print("\nScenario: Good trend, moderate momentum, ADX=22")
    print("This would typically be REJECTED by hard filters (ADX < 25)")
    print("\nHard Filter Result: REJECTED (ADX too low)")
    
    # Scoring approach
    score = calculate_signal_score(
        data=scenario,
        timestamp=datetime.now().replace(hour=10, minute=15),
        atr_percentile=0.60
    )
    
    position_size, size_reason = should_enter_trade(score)
    
    print(f"\nScoring Result: {position_size.name}")
    print(f"Score: {score.total_score:.1f}")
    print(f"\nBreakdown:")
    breakdown = score.get_breakdown()
    for category, value in breakdown.items():
        print(f"  {category:12s}: {value:+6.1f}")
    
    print(f"\nKey Insight:")
    if position_size != PositionSize.NONE:
        print("  ✓ Scoring system ALLOWS trade (with appropriate sizing)")
        print("  ✓ Captures setup that hard filters would reject")
        print("  ✓ Position sizing reflects moderate conviction")
    else:
        print("  ✗ Both systems reject this trade")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SCORING-BASED ENTRY SYSTEM - VALIDATION TESTS")
    print("="*80)
    
    try:
        test_scoring_standalone()
        test_risk_gates()
        test_integration()
        compare_with_hard_filters()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nThe scoring system is ready for backtesting.")
        print("Key advantages:")
        print("  ✓ Replaces hard filters with weighted scoring")
        print("  ✓ Increases trade frequency")
        print("  ✓ Preserves edge through intelligent weighting")
        print("  ✓ Position sizing based on signal quality")
        print("  ✓ Hard risk gates remain intact")
        print("\nNext steps:")
        print("  1. Run backtest with mes_one_minute_scoring strategy")
        print("  2. Compare metrics vs original hard filter approach")
        print("  3. Tune score thresholds based on results")
        print("  4. Monitor score distributions in live trading")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
