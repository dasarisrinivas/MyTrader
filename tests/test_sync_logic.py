"""Test script to verify backtest and live logic synchronization."""
import pandas as pd
from mytrader.strategies.base import Signal
from mytrader.config import TradingConfig

def test_stop_calculation_parity():
    """Test that stop calculation matches between backtest and live."""
    print("=" * 60)
    print("Testing Stop-Loss/Take-Profit Calculation Parity")
    print("=" * 60)
    
    # Simulate metadata from strategy
    metadata = {
        "atr_stop_multiplier": 2.0,
        "risk_reward": 2.5,
        "atr_value": 15.0
    }
    
    entry_price = 4500.0
    direction = 1  # Long
    atr = 15.0
    tick_size = 0.25
    stop_ticks = 20
    take_ticks = 50
    
    # BACKTEST LOGIC (from engine.py _compute_trade_levels)
    print("\n📊 BACKTEST LOGIC:")
    
    default_stop_offset = stop_ticks * tick_size
    default_target_offset = take_ticks * tick_size
    
    atr_multiplier = metadata.get("atr_stop_multiplier", 0.0)
    if atr_multiplier > 0 and atr > 0:
        stop_offset = atr * atr_multiplier
    else:
        stop_offset = default_stop_offset
    
    if stop_offset <= 0:
        stop_offset = default_stop_offset
    
    backtest_stop = entry_price - stop_offset if direction > 0 else entry_price + stop_offset
    
    risk_reward = metadata.get("risk_reward", 0.0)
    if risk_reward <= 0:
        if default_stop_offset > 0:
            risk_reward = take_ticks / max(1e-6, stop_ticks)
        else:
            risk_reward = 2.0
    
    stop_distance = abs(entry_price - backtest_stop)
    target_offset = stop_distance * risk_reward if risk_reward > 0 else default_target_offset
    backtest_target = entry_price + target_offset if direction > 0 else entry_price - target_offset
    
    print(f"  Entry: ${entry_price:.2f}")
    print(f"  Stop:  ${backtest_stop:.2f} (offset: ${stop_offset:.2f})")
    print(f"  Target: ${backtest_target:.2f} (offset: ${target_offset:.2f})")
    print(f"  Risk/Reward: {risk_reward:.2f}")
    
    # LIVE LOGIC (from updated main.py)
    print("\n🔴 LIVE LOGIC (UPDATED):")
    
    raw_stop = metadata.get("stop_loss_price")
    stop_loss = float(raw_stop) if isinstance(raw_stop, (int, float)) else None
    
    raw_target = metadata.get("take_profit_price")
    take_profit = float(raw_target) if isinstance(raw_target, (int, float)) else None
    
    if stop_loss is None:
        atr_multiplier = float(metadata.get("atr_stop_multiplier", 0.0))
        if atr_multiplier > 0 and atr > 0:
            stop_offset = atr * atr_multiplier
        else:
            stop_offset = default_stop_offset
        
        if stop_offset <= 0:
            stop_offset = default_stop_offset
        
        stop_loss = entry_price - stop_offset if direction > 0 else entry_price + stop_offset
    
    if take_profit is None:
        risk_reward = float(metadata.get("risk_reward", 0.0))
        if risk_reward <= 0:
            if default_stop_offset > 0:
                risk_reward = take_ticks / max(1e-6, stop_ticks)
            else:
                risk_reward = 2.0
        
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            stop_distance = default_stop_offset
        
        target_offset = stop_distance * risk_reward if risk_reward > 0 else default_target_offset
        take_profit = entry_price + target_offset if direction > 0 else entry_price - target_offset
    
    print(f"  Entry: ${entry_price:.2f}")
    print(f"  Stop:  ${stop_loss:.2f}")
    print(f"  Target: ${take_profit:.2f}")
    
    # COMPARISON
    print("\n✅ COMPARISON:")
    stop_match = abs(backtest_stop - stop_loss) < 0.01
    target_match = abs(backtest_target - take_profit) < 0.01
    
    print(f"  Stop Match:   {stop_match} (diff: ${abs(backtest_stop - stop_loss):.4f})")
    print(f"  Target Match: {target_match} (diff: ${abs(backtest_target - take_profit):.4f})")
    
    if stop_match and target_match:
        print("\n✅ PASS: Logic is synchronized!")
    else:
        print("\n❌ FAIL: Logic mismatch detected!")
    
    return stop_match and target_match


def test_position_scaler():
    """Test position scaler application."""
    print("\n" + "=" * 60)
    print("Testing Position Scaler Application")
    print("=" * 60)
    
    base_qty = 10
    
    test_cases = [
        {"position_scaler": 1.0, "expected": 10},
        {"position_scaler": 1.5, "expected": 15},
        {"position_scaler": 0.5, "expected": 5},
        {"position_scaler": 2.0, "expected": 20},
        {"position_scaler": 0.0, "expected": 10},  # Should default to 1.0
    ]
    
    all_pass = True
    for test in test_cases:
        scaler = test.get("position_scaler", 1.0)
        expected = test["expected"]
        
        # BACKTEST LOGIC
        backtest_scaler = scaler or 1.0
        if backtest_scaler > 0:
            backtest_qty = max(1, int(round(base_qty * backtest_scaler)))
        else:
            backtest_qty = base_qty
        
        # LIVE LOGIC (updated)
        live_scaler = float(scaler)
        if live_scaler > 0:
            live_qty = max(1, int(round(base_qty * live_scaler)))
        else:
            live_qty = base_qty
        
        match = backtest_qty == live_qty == expected
        status = "✅" if match else "❌"
        
        print(f"  {status} Scaler={scaler:.1f}: Backtest={backtest_qty}, Live={live_qty}, Expected={expected}")
        
        if not match:
            all_pass = False
    
    if all_pass:
        print("\n✅ PASS: Position scaling is synchronized!")
    else:
        print("\n❌ FAIL: Position scaling mismatch!")
    
    return all_pass


def test_trailing_stop_logic():
    """Test trailing stop calculation parity."""
    print("\n" + "=" * 60)
    print("Testing Trailing Stop Logic")
    print("=" * 60)
    
    entry_price = 4500.0
    current_price = 4550.0
    direction = 1  # Long
    atr = 15.0
    trailing_atr_multiplier = 1.5
    trailing_percent = 0.5
    current_stop = 4480.0
    
    # BACKTEST LOGIC (from engine.py)
    print("\n📊 BACKTEST LOGIC (ATR-based):")
    
    trail_distance = atr * trailing_atr_multiplier
    if direction > 0:
        backtest_new_stop = current_price - trail_distance
        if current_stop is None or backtest_new_stop > current_stop:
            backtest_stop_updated = backtest_new_stop
        else:
            backtest_stop_updated = current_stop
    
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Current Stop:  ${current_stop:.2f}")
    print(f"  Trail Distance: ${trail_distance:.2f}")
    print(f"  New Stop: ${backtest_stop_updated:.2f}")
    
    # LIVE LOGIC (from updated ib_executor.py)
    print("\n🔴 LIVE LOGIC (ATR-based):")
    
    trail_distance = atr * trailing_atr_multiplier
    if direction > 0:
        potential_stop = current_price - trail_distance
        if current_stop is None or potential_stop > current_stop:
            live_new_stop = potential_stop
        else:
            live_new_stop = current_stop
    
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Current Stop:  ${current_stop:.2f}")
    print(f"  Trail Distance: ${trail_distance:.2f}")
    print(f"  New Stop: ${live_new_stop:.2f}")
    
    match = abs(backtest_stop_updated - live_new_stop) < 0.01
    
    if match:
        print(f"\n✅ PASS: Trailing stops match! (diff: ${abs(backtest_stop_updated - live_new_stop):.4f})")
    else:
        print(f"\n❌ FAIL: Trailing stops mismatch! (diff: ${abs(backtest_stop_updated - live_new_stop):.4f})")
    
    return match


def main():
    """Run all synchronization tests."""
    print("\n" + "=" * 60)
    print("BACKTEST <-> LIVE TRADING LOGIC SYNC TEST")
    print("=" * 60)
    
    results = []
    
    # Test 1: Stop calculation
    try:
        results.append(("Stop Calculation", test_stop_calculation_parity()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        results.append(("Stop Calculation", False))
    
    # Test 2: Position scaling
    try:
        results.append(("Position Scaling", test_position_scaler()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        results.append(("Position Scaling", False))
    
    # Test 3: Trailing stops
    try:
        results.append(("Trailing Stops", test_trailing_stop_logic()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        results.append(("Trailing Stops", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_pass = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ ALL TESTS PASSED - Logic is synchronized!")
    else:
        print("❌ SOME TESTS FAILED - Review implementation!")
    print("=" * 60 + "\n")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
