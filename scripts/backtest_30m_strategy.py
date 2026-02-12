#!/usr/bin/env python3
"""
Backtest the 30-minute strategy with native 30m data.

JAN 11 2026: This script tests the purpose-built 30m strategy which is
designed for longer timeframe trading and works during overnight sessions.

Usage:
    python scripts/backtest_30m_strategy.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shree.config import ThirtyMinuteStrategyConfig
from shree.strategies.mes_thirty_minute import MesThirtyMinuteStrategy, ThirtyMinuteSignal
from loguru import logger


def run_30m_strategy_backtest():
    """Run backtest using the 30m-specific strategy."""
    
    # Load native 30m data
    data_path = Path("data/raw/ES/ES_30min_60D.parquet")
    if not data_path.exists():
        logger.error(f"30m data not found: {data_path}")
        logger.info("Run: python scripts/download_30m_data.py")
        return None
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} 30m bars")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Check data quality
    zero_range = ((df['high'] - df['low']) == 0).sum()
    logger.info(f"Zero-range bars: {zero_range} ({zero_range/len(df)*100:.1f}%)")
    
    # Create strategy with default config
    config = ThirtyMinuteStrategyConfig()
    strategy = MesThirtyMinuteStrategy(config)
    
    # Calculate indicators once
    logger.info("Calculating indicators...")
    df = strategy._calculate_indicators(df)
    
    # Run backtest
    logger.info("Running 30m strategy backtest...")
    
    trades: List[Dict[str, Any]] = []
    block_reasons: Dict[str, int] = {}
    
    # Position tracking
    position = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    entry_bar = 0
    stop_loss = 0.0
    take_profit = 0.0
    entry_signal = None
    
    # Equity tracking
    initial_capital = 5000.0
    equity = initial_capital
    commission = 2.40  # Per contract per side
    point_value = 5.0  # MES = $5 per point
    
    for i in range(len(df)):
        row = df.iloc[i]
        timestamp = df.index[i]
        
        # If in position, check for exit
        if position != 0:
            direction = "LONG" if position > 0 else "SHORT"
            should_exit, exit_reason, exit_price = strategy.should_exit(
                df, i, entry_bar, direction, entry_price, stop_loss, take_profit
            )
            
            if should_exit:
                # Calculate P&L
                if direction == "LONG":
                    pnl_points = exit_price - entry_price
                else:
                    pnl_points = entry_price - exit_price
                    
                pnl_usd = (pnl_points * point_value) - (commission * 2)  # Entry + exit commission
                equity += pnl_usd
                
                # Record trade
                trades.append({
                    'entry_time': df.index[entry_bar],
                    'exit_time': timestamp,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_points': pnl_points,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'bars_held': i - entry_bar,
                    'signal_type': entry_signal.signal_type if entry_signal else "UNKNOWN",
                    'entry_atr': entry_signal.metadata.get('atr', 0) if entry_signal else 0,
                    'entry_adx': entry_signal.metadata.get('adx', 0) if entry_signal else 0,
                })
                
                # Reset position
                position = 0
                entry_price = 0.0
                entry_bar = 0
                stop_loss = 0.0
                take_profit = 0.0
                entry_signal = None
                
        # If flat, check for entry
        if position == 0:
            signal = strategy.generate_signal(df, i, current_position=0)
            
            # Track block reasons
            if signal.action == "HOLD" and signal.signal_type != "NO_SIGNAL":
                reason = signal.signal_type
                block_reasons[reason] = block_reasons.get(reason, 0) + 1
                
            # Enter if we get a signal
            if signal.action in ("BUY", "SELL"):
                position = 1 if signal.action == "BUY" else -1
                entry_price = signal.entry_price
                entry_bar = i
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit
                entry_signal = signal
                
        # Progress logging
        if (i + 1) % 200 == 0:
            logger.info(f"Processed {i+1}/{len(df)} bars, {len(trades)} trades")
            
    # Close any open position at end
    if position != 0:
        direction = "LONG" if position > 0 else "SHORT"
        exit_price = df.iloc[-1]['close']
        if direction == "LONG":
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price
        pnl_usd = (pnl_points * point_value) - (commission * 2)
        equity += pnl_usd
        
        trades.append({
            'entry_time': df.index[entry_bar],
            'exit_time': df.index[-1],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_points': pnl_points,
            'pnl_usd': pnl_usd,
            'exit_reason': 'END_OF_DATA',
            'bars_held': len(df) - 1 - entry_bar,
            'signal_type': entry_signal.signal_type if entry_signal else "UNKNOWN",
            'entry_atr': entry_signal.metadata.get('atr', 0) if entry_signal else 0,
            'entry_adx': entry_signal.metadata.get('adx', 0) if entry_signal else 0,
        })
        
    # Print results
    print("\n" + "="*70)
    print("30M STRATEGY BACKTEST RESULTS")
    print("="*70)
    
    if trades:
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades)
        winners = trades_df[trades_df['pnl_usd'] > 0]
        losers = trades_df[trades_df['pnl_usd'] < 0]
        
        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        total_pnl = trades_df['pnl_usd'].sum()
        gross_profit = winners['pnl_usd'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl_usd'].sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_win = winners['pnl_usd'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl_usd'].mean() if len(losers) > 0 else 0
        avg_bars_held = trades_df['bars_held'].mean()
        
        print(f"Total Trades:      {total_trades}")
        print(f"Winners:           {len(winners)}")
        print(f"Losers:            {len(losers)}")
        print(f"Win Rate:          {win_rate*100:.1f}%")
        print(f"Total P&L:         ${total_pnl:.2f}")
        print(f"Gross Profit:      ${gross_profit:.2f}")
        print(f"Gross Loss:        ${gross_loss:.2f}")
        print(f"Profit Factor:     {profit_factor:.2f}")
        print(f"Avg Win:           ${avg_win:.2f}")
        print(f"Avg Loss:          ${avg_loss:.2f}")
        print(f"Avg Bars Held:     {avg_bars_held:.1f} ({avg_bars_held*30:.0f} min)")
        print(f"Final Equity:      ${equity:.2f}")
        print(f"Return:            {(equity - initial_capital) / initial_capital * 100:.2f}%")
        
        # By signal type
        print("\n" + "-"*40)
        print("BY SIGNAL TYPE:")
        for signal_type in trades_df['signal_type'].unique():
            subset = trades_df[trades_df['signal_type'] == signal_type]
            sw = len(subset[subset['pnl_usd'] > 0])
            print(f"  {signal_type}: {len(subset)} trades, {sw}/{len(subset)} wins "
                  f"({sw/len(subset)*100:.0f}%), ${subset['pnl_usd'].sum():.2f}")
        
        # By exit reason
        print("\n" + "-"*40)
        print("BY EXIT REASON:")
        for reason in trades_df['exit_reason'].unique():
            subset = trades_df[trades_df['exit_reason'] == reason]
            sw = len(subset[subset['pnl_usd'] > 0])
            print(f"  {reason}: {len(subset)} trades, ${subset['pnl_usd'].sum():.2f}")
            
        # Sample trades
        print("\n" + "-"*40)
        print("SAMPLE TRADES (first 10):")
        display_cols = ['entry_time', 'direction', 'signal_type', 'entry_price', 
                        'exit_price', 'pnl_usd', 'exit_reason', 'bars_held']
        print(trades_df[display_cols].head(10).to_string())
        
    else:
        print("No trades generated!")
        
    # Block reasons
    print("\n" + "-"*40)
    print("BLOCK REASONS (top 10):")
    for reason, count in sorted(block_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"  {reason}: {count}")
        
    print("="*70)
    
    return {
        'trades': trades,
        'block_reasons': block_reasons,
        'equity': equity,
        'initial_capital': initial_capital,
    }


if __name__ == "__main__":
    results = run_30m_strategy_backtest()
