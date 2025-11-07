#!/usr/bin/env python3
"""Analyze trading logs to extract detailed trade history."""
import re
from datetime import datetime
from collections import defaultdict

log_file = "/Users/svss/Documents/code/MyTrader/logs/live_trading.log"

# Parse the log file
trades = []
current_position = 0
position_history = []

with open(log_file, 'r') as f:
    lines = f.readlines()

trade_data = {}
for i, line in enumerate(lines):
    try:
        # Extract timestamp
        if '|' not in line:
            continue
        timestamp_str = line.split('|')[0].strip()
        timestamp = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
        
        # Track position changes
        if 'Current position:' in line and 'contracts' in line:
            parts = line.split('Current position:')[1].split('contracts')[0].strip()
            new_position = int(parts)
            
            if new_position != current_position:
                position_history.append({
                    'timestamp': timestamp,
                    'old_position': current_position,
                    'new_position': new_position,
                    'change': new_position - current_position
                })
                current_position = new_position
        
        # Track signals
        if 'SIGNAL GENERATED:' in line:
            if 'BUY' in line:
                action = 'BUY'
            elif 'SELL' in line:
                action = 'SELL'
            else:
                continue
            
            # Get confidence
            confidence = 0.0
            if 'confidence=' in line:
                conf_str = line.split('confidence=')[1].split()[0].replace(',', '')
                confidence = float(conf_str)
            
            # Look ahead for price, stop loss, take profit
            price = None
            stop_loss = None
            take_profit = None
            atr = None
            
            for j in range(max(0, i-5), min(i+15, len(lines))):
                next_line = lines[j]
                if 'Got last price:' in next_line:
                    try:
                        price = float(next_line.split('Got last price:')[1].strip())
                    except:
                        pass
                if 'Stop Loss:' in next_line:
                    try:
                        stop_loss = float(next_line.split('Stop Loss:')[1].strip().split()[0])
                    except:
                        pass
                if 'Take Profit:' in next_line:
                    try:
                        take_profit = float(next_line.split('Take Profit:')[1].strip().split()[0])
                    except:
                        pass
                if 'ATR:' in next_line and 'ATR:' not in next_line.split('|')[0]:
                    try:
                        atr = float(next_line.split('ATR:')[1].strip().split()[0])
                    except:
                        pass
            
            trades.append({
                'timestamp': timestamp,
                'action': action,
                'price': price,
                'confidence': confidence,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr
            })
    except Exception as e:
        continue

# Print analysis
print("=" * 80)
print("TRADING ACTIVITY ANALYSIS")
print("=" * 80)
print(f"\nTotal Signals: {len(trades)}")
print(f"Total Position Changes: {len(position_history)}")

print("\n" + "=" * 80)
print("DETAILED TRADE LOG")
print("=" * 80)

for i, trade in enumerate(trades, 1):
    print(f"\n--- Trade #{i} ---")
    print(f"Time:       {trade['timestamp'].strftime('%H:%M:%S')}")
    print(f"Action:     {trade['action']}")
    print(f"Price:      ${trade['price']:.2f}" if trade['price'] else "Price:      N/A")
    print(f"Confidence: {trade['confidence']:.2f}")
    print(f"Stop Loss:  ${trade['stop_loss']:.2f}" if trade['stop_loss'] else "Stop Loss:  N/A")
    print(f"Take Profit: ${trade['take_profit']:.2f}" if trade['take_profit'] else "Take Profit: N/A")
    print(f"ATR:        ${trade['atr']:.2f}" if trade['atr'] else "ATR:        N/A")
    
    # Calculate risk/reward
    if trade['price'] and trade['stop_loss'] and trade['take_profit']:
        if trade['action'] == 'BUY':
            risk = trade['price'] - trade['stop_loss']
            reward = trade['take_profit'] - trade['price']
        else:  # SELL
            risk = trade['stop_loss'] - trade['price']
            reward = trade['price'] - trade['take_profit']
        
        if risk > 0:
            rr_ratio = reward / risk
            print(f"Risk:       ${risk:.2f}")
            print(f"Reward:     ${reward:.2f}")
            print(f"R:R Ratio:  {rr_ratio:.2f}:1")

print("\n" + "=" * 80)
print("POSITION CHANGES")
print("=" * 80)

for i, change in enumerate(position_history, 1):
    print(f"\n{i}. {change['timestamp'].strftime('%H:%M:%S')}")
    print(f"   {change['old_position']:+4d} → {change['new_position']:+4d} (change: {change['change']:+4d})")

# Estimate P&L based on position changes
print("\n" + "=" * 80)
print("ESTIMATED P&L BY TRADE")
print("=" * 80)

total_pnl = 0.0
entry_price = None
entry_qty = 0

print("\nNote: This is estimated based on position changes and prices.")
print("Actual P&L should be checked in IBKR account.\n")

# Match trades with position changes
for i, (trade, pos_change) in enumerate(zip(trades, position_history), 1):
    print(f"Trade #{i} @ {trade['timestamp'].strftime('%H:%M:%S')}")
    print(f"  Signal: {trade['action']} at ${trade['price']:.2f}")
    print(f"  Position: {pos_change['old_position']:+4d} → {pos_change['new_position']:+4d}")
    
    # Estimate P&L for closing trades
    if abs(pos_change['new_position']) < abs(pos_change['old_position']):
        print(f"  *** CLOSING TRADE ***")
        if entry_price and trade['price']:
            # Closed some or all position
            if pos_change['old_position'] > 0:  # Was LONG
                pnl_per_contract = trade['price'] - entry_price
            else:  # Was SHORT
                pnl_per_contract = entry_price - trade['price']
            
            contracts_closed = abs(pos_change['change'])
            trade_pnl = pnl_per_contract * contracts_closed * 50  # ES multiplier = $50
            total_pnl += trade_pnl
            
            print(f"  Entry Price: ${entry_price:.2f}")
            print(f"  Exit Price:  ${trade['price']:.2f}")
            print(f"  P&L/Contract: ${pnl_per_contract:.2f}")
            print(f"  Contracts: {contracts_closed}")
            print(f"  Trade P&L: ${trade_pnl:,.2f}")
            print(f"  Running Total: ${total_pnl:,.2f}")
    
    # Update entry for new position
    if pos_change['new_position'] != 0:
        entry_price = trade['price']
        entry_qty = abs(pos_change['new_position'])

print("\n" + "=" * 80)
print(f"ESTIMATED TOTAL P&L: ${total_pnl:,.2f}")
print("=" * 80)
print("\nIMPORTANT: This is an estimate. Check your IBKR account for actual P&L!")
print("The actual loss of -$116,000 suggests high leverage or many more trades.")
