"""Test script for TradingFilters - validates the new filter implementation."""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, '/Users/svss/Documents/code/MyTrader')

from mytrader.strategies.trading_filters import (
    TradingFilters,
    PriceLevels,
    TradingFilterResult,
    calculate_enhanced_confidence
)


def create_test_dataframe(num_bars: int = 100, start_price: float = 6000.0) -> pd.DataFrame:
    """Create a test DataFrame with OHLCV data and indicators."""
    np.random.seed(42)
    
    # Generate realistic price data
    prices = [start_price]
    for i in range(num_bars - 1):
        change = np.random.normal(0, 2)  # ~2 point average move
        prices.append(prices[-1] + change)
    
    # Create DataFrame with timestamps spanning multiple days
    timestamps = [datetime.now(timezone.utc) - timedelta(minutes=num_bars-i) for i in range(num_bars)]
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.normal(0, 1)) for p in prices],
        'low': [p - abs(np.random.normal(0, 1)) for p in prices],
        'close': [p + np.random.normal(0, 0.5) for p in prices],
        'volume': [int(np.random.uniform(100, 1000)) for _ in prices]
    }, index=pd.DatetimeIndex(timestamps))
    
    # Add indicators
    df['EMA_9'] = df['close'].ewm(span=9).mean()
    df['EMA_20'] = df['close'].ewm(span=20).mean()
    df['ATR_14'] = calculate_atr(df, 14)
    df['RSI_14'] = calculate_rsi(df['close'], 14)
    
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def test_filter_initialization():
    """Test that TradingFilters initializes correctly."""
    print("\n=== Test: Filter Initialization ===")
    filters = TradingFilters()
    
    assert filters is not None, "Failed to create TradingFilters instance"
    print("✅ TradingFilters initialized successfully")


def test_level_calculation():
    """Test PDH/PDL/WH/WL level calculation."""
    print("\n=== Test: Level Calculation ===")
    
    # Create data spanning multiple days for level calculation
    df = create_test_dataframe(2880, 6000.0)  # ~48 hours of 1-min data
    filters = TradingFilters()
    
    filters.set_historical_data(df)
    levels = filters.get_levels()
    
    if levels:
        print(f"   PDH: {levels.pdh}")
        print(f"   PDL: {levels.pdl}")
        print(f"   WH:  {levels.wh}")
        print(f"   WL:  {levels.wl}")
        print(f"   PWH: {levels.pwh}")
        print(f"   PWL: {levels.pwl}")
        
        # Basic sanity checks
        assert levels.pdh is None or levels.pdh > 0, "PDH should be positive if set"
        assert levels.pdl is None or levels.pdl > 0, "PDL should be positive if set"
    else:
        print("   (No levels calculated - expected for limited test data)")
    
    print("✅ Level calculation completed")


def test_trend_filter_bullish():
    """Test trend filter with bullish EMA alignment."""
    print("\n=== Test: Trend Filter (Bullish) ===")
    
    df = create_test_dataframe(100)
    # Force bullish trend: EMA_9 > EMA_20
    df['EMA_9'] = df['close'] + 5
    df['EMA_20'] = df['close'] - 5
    
    filters = TradingFilters(require_candle_close=False)  # Disable candle check for testing
    
    current_price = float(df['close'].iloc[-1])
    result = filters.evaluate(
        current_price=current_price,
        proposed_action="BUY",
        features=df
    )
    
    print(f"   Can trade: {result.can_trade}")
    print(f"   Reasons: {result.reasons}")
    
    # BUY should pass with bullish trend
    assert result.can_trade or 'trend' not in str(result.reasons).lower(), "BUY should pass with bullish trend (unless blocked by other filter)"
    print("✅ Bullish trend filter working")


def test_trend_filter_bearish():
    """Test trend filter with bearish EMA alignment."""
    print("\n=== Test: Trend Filter (Bearish) ===")
    
    df = create_test_dataframe(100)
    # Force bearish trend: EMA_9 < EMA_20
    df['EMA_9'] = df['close'] - 5
    df['EMA_20'] = df['close'] + 5
    
    filters = TradingFilters(require_candle_close=False)
    
    current_price = float(df['close'].iloc[-1])
    result = filters.evaluate(
        current_price=current_price,
        proposed_action="SELL",
        features=df
    )
    
    print(f"   Can trade: {result.can_trade}")
    print(f"   Reasons: {result.reasons}")
    
    # SELL should pass with bearish trend
    assert result.can_trade or 'trend' not in str(result.reasons).lower(), "SELL should pass with bearish trend"
    print("✅ Bearish trend filter working")


def test_counter_trend_blocked():
    """Test that counter-trend trades are blocked."""
    print("\n=== Test: Counter-Trend Blocked ===")
    
    df = create_test_dataframe(100)
    # Force bearish trend
    df['EMA_9'] = df['close'] - 5
    df['EMA_20'] = df['close'] + 5
    
    filters = TradingFilters(require_candle_close=False)
    
    current_price = float(df['close'].iloc[-1])
    result = filters.evaluate(
        current_price=current_price,
        proposed_action="BUY",  # BUY in bearish trend = counter-trend
        features=df
    )
    
    print(f"   Can trade: {result.can_trade}")
    print(f"   Reasons: {result.reasons}")
    
    # BUY should be blocked in bearish trend
    assert not result.can_trade, "Counter-trend BUY should be blocked"
    assert any('trend' in r.lower() for r in result.reasons), "Should have trend-related reason"
    print("✅ Counter-trend blocking works")


def test_volatility_filter():
    """Test volatility filter with extreme ATR."""
    print("\n=== Test: Volatility Filter ===")
    
    df = create_test_dataframe(100)
    df['EMA_9'] = df['close'] + 5  # Ensure trend is aligned
    df['EMA_20'] = df['close'] - 5
    
    # Test with very low ATR (should block)
    df['ATR_14'] = 0.1
    filters = TradingFilters(require_candle_close=False, min_atr_threshold=0.5)
    result = filters.evaluate(
        current_price=6000.0,
        proposed_action="BUY",
        features=df
    )
    print(f"   Low ATR (0.1): can_trade={result.can_trade}, reasons={result.reasons}")
    assert not result.can_trade, "Should block on low ATR"
    
    # Test with normal ATR (should pass)
    df['ATR_14'] = 2.0
    result = filters.evaluate(
        current_price=6000.0,
        proposed_action="BUY",
        features=df
    )
    print(f"   Normal ATR (2.0): can_trade={result.can_trade}, reasons={result.reasons}")
    # May still be blocked by other filters, but not ATR
    
    # Test with very high ATR (should block)
    df['ATR_14'] = 100.0
    filters2 = TradingFilters(require_candle_close=False, max_atr_threshold=5.0)
    result = filters2.evaluate(
        current_price=6000.0,
        proposed_action="BUY",
        features=df
    )
    print(f"   High ATR (100.0): can_trade={result.can_trade}, reasons={result.reasons}")
    assert not result.can_trade, "Should block on high ATR"
    
    print("✅ Volatility filter working")


def test_enhanced_confidence():
    """Test enhanced confidence calculation."""
    print("\n=== Test: Enhanced Confidence ===")
    
    df = create_test_dataframe(100)
    df['RSI_14'] = 35  # Moderately oversold - good for buys
    
    base_confidence = 0.55
    enhanced, reasons = calculate_enhanced_confidence(
        base_confidence=base_confidence,
        features=df,
        action="BUY",
        price_levels=None,
        current_price=6000.0
    )
    
    print(f"   Base confidence: {base_confidence:.3f}")
    print(f"   Enhanced confidence: {enhanced:.3f}")
    print(f"   Reasons: {reasons}")
    
    print("✅ Enhanced confidence calculation working")


def test_enhanced_confidence_blocked():
    """Test enhanced confidence when factors are negative."""
    print("\n=== Test: Enhanced Confidence (Negative Factors) ===")
    
    df = create_test_dataframe(100)
    df['RSI_14'] = 80  # Overbought - bad for buys
    df['EMA_9'] = df['close'] - 10  # Bearish trend
    df['EMA_20'] = df['close'] + 10
    
    base_confidence = 0.70
    enhanced, reasons = calculate_enhanced_confidence(
        base_confidence=base_confidence,
        features=df,
        action="BUY",  # BUY when overbought and bearish = bad
        price_levels=None,
        current_price=6000.0
    )
    
    print(f"   Base confidence: {base_confidence:.3f}")
    print(f"   Enhanced confidence: {enhanced:.3f}")
    print(f"   Reasons: {reasons}")
    
    assert enhanced < base_confidence, "Enhanced confidence should be reduced with negative factors"
    print("✅ Negative factor confidence reduction working")


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("TRADING FILTERS TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_filter_initialization,
        test_level_calculation,
        test_trend_filter_bullish,
        test_trend_filter_bearish,
        test_counter_trend_blocked,
        test_volatility_filter,
        test_enhanced_confidence,
        test_enhanced_confidence_blocked,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
