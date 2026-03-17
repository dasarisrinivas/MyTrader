import sqlite3
import pandas as pd
import json
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

db_path = 'data/orders.db'

def get_trades():
    conn = sqlite3.connect(db_path)
    
    # Query trade_outcomes joined with orders to get regime and features
    query = """
    SELECT 
        t.trade_cycle_id,
        t.symbol,
        t.entry_time,
        t.exit_time,
        t.entry_price,
        t.exit_price,
        t.quantity,
        t.exit_reason,
        t.realized_pnl,
        t.net_pnl,
        t.commission,
        t.extra_json,
        o.market_regime,
        o.rationale,
        o.features,
        o.action as direction,
        o.stop_loss,
        o.take_profit,
        o.atr
    FROM trade_outcomes t
    LEFT JOIN orders o ON t.root_order_id = o.order_id
    WHERE t.entry_time >= '2026-01-14'
    ORDER BY t.entry_time ASC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

df_trades = get_trades()

print(f"Total trades found: {len(df_trades)}")

if len(df_trades) == 0:
    print("No trades found in the last 2 days.")
    exit()

# Parse timestamps
df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
df_trades['holding_time'] = (df_trades['exit_time'] - df_trades['entry_time']).dt.total_seconds() / 60.0 # minutes

# Add metrics
df_trades['is_win'] = df_trades['net_pnl'] > 0
df_trades['is_loss'] = df_trades['net_pnl'] <= 0

# RTH vs Overnight (Approximation: RTH 9:30 - 16:00 ET, which is 8:30 - 15:00 CST)
# Assuming log timestamps are UTC or system time. The logs show CST.
# Let's inspect time zone. The logs say CST. '2026-01-16 12:50:06 CST'
# The DB timestamps seem to be ISO strings. '2026-01-16T15:45:13.692722+00:00'. So they are UTC.
# RTH in UTC (winter): 14:30 - 21:00 UTC.
df_trades['hour'] = df_trades['entry_time'].dt.hour
df_trades['is_rth'] = df_trades['entry_time'].apply(lambda x: 14 <= x.hour < 21) # Crude approximation, refining later if needed
# Better check: 9:30 EST = 14:30 UTC. 16:00 EST = 21:00 UTC.
def is_rth(dt):
    # Convert to minutes from midnight
    minutes = dt.hour * 60 + dt.minute
    return (14 * 60 + 30) <= minutes < (21 * 60)

df_trades['is_rth'] = df_trades['entry_time'].apply(is_rth)
df_trades['session'] = df_trades['is_rth'].map({True: 'RTH', False: 'Overnight'})

print("\n--- Loss Rate Analysis ---")
print(f"Total Trades: {len(df_trades)}")
print(f"Win Rate: {df_trades['is_win'].mean():.2%}")
print(f"Loss Rate: {df_trades['is_loss'].mean():.2%}")

print("\n--- By Session ---")
print(df_trades.groupby('session')[['net_pnl', 'is_win', 'is_loss']].agg(
    {'net_pnl': 'sum', 'is_win': 'mean', 'is_loss': 'mean', 'is_loss': 'count'}
))

print("\n--- By Direction ---")
print(df_trades.groupby('direction')[['net_pnl', 'is_win', 'is_loss']].agg(
    {'net_pnl': 'sum', 'is_win': 'mean', 'is_loss': 'mean'}
).to_string())

print("\n--- Win/Loss Stats ---")
wins = df_trades[df_trades['is_win']]
losses = df_trades[df_trades['is_loss']]
print(f"Avg Win: {wins['net_pnl'].mean():.2f}")
print(f"Avg Loss: {losses['net_pnl'].mean():.2f}")
print(f"Avg Holding Time (Win): {wins['holding_time'].mean():.2f} min")
print(f"Avg Holding Time (Loss): {losses['holding_time'].mean():.2f} min")

print("\n--- Market Regime ---")
if 'market_regime' in df_trades.columns:
    print(df_trades.groupby('market_regime')[['net_pnl', 'is_win']].mean())

print("\n--- Sample Losing Trades ---")
print(losses[['entry_time', 'direction', 'entry_price', 'exit_price', 'net_pnl', 'exit_reason', 'market_regime', 'holding_time']].tail(5))

# Check for consecutive losses
df_trades['prev_loss'] = df_trades['is_loss'].shift(1)
df_trades['time_since_prev'] = (df_trades['entry_time'] - df_trades['exit_time'].shift(1)).dt.total_seconds() / 60.0
print("\n--- Churn / Revenge Trading Check ---")
print("Trades opened < 5 mins after previous exit:")
churn_trades = df_trades[df_trades['time_since_prev'] < 5]
print(churn_trades[['entry_time', 'time_since_prev', 'is_loss', 'net_pnl']])

# Save to CSV for manual inspection if needed
df_trades.to_csv('recent_trades_analysis.csv', index=False)
