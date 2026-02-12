#!/usr/bin/env python3
"""
Quick backtest for Scoring Strategy on ES data
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from shree.strategies.scoring_entry import calculate_signal_score, should_enter_trade

logger.remove()
logger.add(sys.stdout, level="INFO")

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators needed for scoring."""
    df = df.copy()
    
    # EMAs
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # ADX
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()
    
    # Simple ADX calculation
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / df['ATR_14'])
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / df['ATR_14'])
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX_14'] = dx.rolling(window=14).mean()
    
    # VWAP (daily reset)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp_dt'].dt.date
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap_num'] = (df['typical_price'] * df['volume']).groupby(df['date']).cumsum()
    df['vwap_den'] = df['volume'].groupby(df['date']).cumsum()
    df['VWAP'] = df['vwap_num'] / df['vwap_den']
    
    return df

def run_backtest(data_file: str, full_size_threshold: float = 60.0, half_size_threshold: float = 45.0, 
                 start_date: str = None, end_date: str = None):
    """Run backtest with scoring strategy."""
    logger.info(f"Loading data from {data_file}")
    
    # Load data (support both CSV and Parquet)
    if data_file.endswith('.parquet'):
        df = pd.read_parquet(data_file)
        # Reset index if datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={'index': 'timestamp'})
    else:
        df = pd.read_csv(data_file)
    
    # Filter by date range if specified
    if start_date or end_date:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]
    
    logger.info(f"Calculating indicators...")
    df = calculate_indicators(df)
    
    # Drop NaN rows from indicator calculation
    df = df.dropna()
    
    logger.info(f"Running backtest on {len(df)} bars...")
    logger.info(f"Date range: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
    
    trades = []
    equity = [50000.0]  # Starting capital
    current_position = None
    
    for i in range(50, len(df)):  # Start after warmup period
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        recent_bars = df.iloc[i-20:i]
        
        current_price = current_bar['close']
        
        # Prepare data dict for scoring
        data = {
            'close': current_bar['close'],
            'high': current_bar['high'],
            'low': current_bar['low'],
            'open': current_bar['open'],
            'volume': current_bar['volume'],
            'EMA_9': current_bar['EMA_9'],
            'EMA_21': current_bar['EMA_21'],
            'EMA_50': current_bar['EMA_50'],
            'MACD': current_bar['MACD'],
            'MACD_signal': current_bar['MACD_signal'],
            'MACD_hist': current_bar['MACD_hist'],
            'RSI_14': current_bar['RSI_14'],
            'ADX_14': current_bar['ADX_14'],
            'ATR_14': current_bar['ATR_14'],
            'VWAP': current_bar['VWAP'],
        }
        
        prev_data = {
            'ADX_14': prev_bar['ADX_14'],
            'MACD': prev_bar['MACD'],
            'RSI_14': prev_bar['RSI_14'],
        }
        
        # Check for exit if in position
        if current_position is not None:
            direction = current_position['direction']
            entry_price = current_position['entry_price']
            stop_loss = current_position['stop_loss']
            take_profit = current_position['take_profit']
            
            if direction == 'LONG':
                if current_price <= stop_loss:
                    pnl = (stop_loss - entry_price) * 5  # MES multiplier
                    trades.append({
                        'entry_time': current_position['entry_time'],
                        'exit_time': current_bar['timestamp'],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'pnl': pnl,
                        'exit_reason': 'stop_loss',
                        'score': current_position['score']
                    })
                    equity.append(equity[-1] + pnl)
                    current_position = None
                elif current_price >= take_profit:
                    pnl = (take_profit - entry_price) * 5
                    trades.append({
                        'entry_time': current_position['entry_time'],
                        'exit_time': current_bar['timestamp'],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'pnl': pnl,
                        'exit_reason': 'take_profit',
                        'score': current_position['score']
                    })
                    equity.append(equity[-1] + pnl)
                    current_position = None
            else:  # SHORT
                if current_price >= stop_loss:
                    pnl = (entry_price - stop_loss) * 5
                    trades.append({
                        'entry_time': current_position['entry_time'],
                        'exit_time': current_bar['timestamp'],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'pnl': pnl,
                        'exit_reason': 'stop_loss',
                        'score': current_position['score']
                    })
                    equity.append(equity[-1] + pnl)
                    current_position = None
                elif current_price <= take_profit:
                    pnl = (entry_price - take_profit) * 5
                    trades.append({
                        'entry_time': current_position['entry_time'],
                        'exit_time': current_bar['timestamp'],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'pnl': pnl,
                        'exit_reason': 'take_profit',
                        'score': current_position['score']
                    })
                    equity.append(equity[-1] + pnl)
                    current_position = None
        
        # Check for new entry if not in position
        if current_position is None:
            try:
                # Calculate signal score
                score = calculate_signal_score(
                    data=data,
                    prev_data=prev_data,
                    recent_bars=recent_bars,
                    timestamp=None,
                    atr_percentile=None
                )
                
                # Check if should enter trade
                position_size, reason = should_enter_trade(
                    score=score,
                    min_full_size_score=full_size_threshold,
                    min_half_size_score=half_size_threshold
                )
                
                # Only trade if score meets threshold
                from shree.strategies.scoring_entry import PositionSize
                if position_size != PositionSize.NONE:
                    direction = score.direction  # LONG or SHORT
                    atr = data['ATR_14']
                    
                    if direction == 'LONG':
                        entry_price = current_price
                        stop_loss = entry_price - (2 * atr)
                        take_profit = entry_price + (3 * atr)
                    else:  # SHORT
                        entry_price = current_price
                        stop_loss = entry_price + (2 * atr)
                        take_profit = entry_price - (3 * atr)
                    
                    current_position = {
                        'direction': direction,
                        'entry_price': entry_price,
                        'entry_time': current_bar['timestamp'],
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'score': score.total_score
                    }
            except Exception as e:
                # Skip bars with scoring errors
                pass
    
    # Close any open position at end
    if current_position is not None:
        final_bar = df.iloc[-1]
        direction = current_position['direction']
        entry_price = current_position['entry_price']
        exit_price = final_bar['close']
        
        if direction == 'LONG':
            pnl = (exit_price - entry_price) * 5
        else:
            pnl = (entry_price - exit_price) * 5
        
        trades.append({
            'entry_time': current_position['entry_time'],
            'exit_time': final_bar['timestamp'],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': 'end_of_data',
            'score': current_position['score']
        })
        equity.append(equity[-1] + pnl)
    
    # Calculate statistics
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        logger.error("❌ No trades generated!")
        logger.info("This suggests thresholds are too high or market conditions poor")
        return
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = winning_trades / total_trades * 100
    
    total_pnl = trades_df['pnl'].sum()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    
    # Score distribution
    avg_score = trades_df['score'].mean()
    min_score = trades_df['score'].min()
    max_score = trades_df['score'].max()
    
    # Print results
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS - SCORING STRATEGY")
    logger.info("=" * 60)
    logger.info(f"Thresholds: Full={full_size_threshold}, Half={half_size_threshold}")
    logger.info(f"")
    logger.info(f"Total Trades:     {total_trades}")
    logger.info(f"Winning Trades:   {winning_trades} ({win_rate:.1f}%)")
    logger.info(f"Losing Trades:    {losing_trades}")
    logger.info(f"")
    logger.info(f"Total P&L:        ${total_pnl:,.2f}")
    logger.info(f"Average Win:      ${avg_win:.2f}")
    logger.info(f"Average Loss:     ${avg_loss:.2f}")
    logger.info(f"Profit Factor:    {abs(avg_win / avg_loss) if avg_loss != 0 else 0:.2f}")
    logger.info(f"")
    logger.info(f"Score Stats:")
    logger.info(f"  Average:        {avg_score:.1f}")
    logger.info(f"  Range:          {min_score:.1f} - {max_score:.1f}")
    logger.info(f"")
    logger.info(f"Final Equity:     ${equity[-1]:,.2f}")
    logger.info(f"Return:           {((equity[-1] - 50000) / 50000 * 100):.2f}%")
    logger.info("=" * 60)
    
    # Save trades to CSV
    output_file = f"reports/backtest_scoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    Path("reports").mkdir(exist_ok=True)
    trades_df.to_csv(output_file, index=False)
    logger.info(f"✅ Trades saved to: {output_file}")

if __name__ == "__main__":
    import sys
    
    data_file = "data/ib/ES_1m_1y.parquet"
    full_threshold = 60.0
    half_threshold = 45.0
    start_date = None
    end_date = None
    
    if len(sys.argv) > 1:
        full_threshold = float(sys.argv[1])
    if len(sys.argv) > 2:
        half_threshold = float(sys.argv[2])
    if len(sys.argv) > 3:
        start_date = sys.argv[3]
    if len(sys.argv) > 4:
        end_date = sys.argv[4]
    
    logger.info(f"Running backtest with thresholds: full={full_threshold}, half={half_threshold}")
    if start_date or end_date:
        logger.info(f"Date range: {start_date or 'START'} to {end_date or 'END'}")
    run_backtest(data_file, full_threshold, half_threshold, start_date, end_date)
