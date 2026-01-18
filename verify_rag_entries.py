import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from datetime import datetime
print("Imports starting...")
from mytrader.rag.hybrid_rag_pipeline import RuleEngine, TradeAction
from mytrader.utils.session_manager import TradingSession
print("Imports done.")

# Configuration matching the environment
config = {
    "rule_engine": {
        "atr_min": 0.15,
        "atr_max": 20.0,
        "trend_weight": 30
    },
    "sessions": {
        "rth_start": "08:30",
        "rth_end": "15:00",
        "timezone": "America/Chicago"
    }
}

def load_data():
    # Load the Parquet file identified in previous logs
    try:
        # Load CSV data
        df = pd.read_csv("data/es_historical.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Ensure UTC if needed, but the file has offset. 
        # If timestamp is already timezone-aware, we might need to convert to UTC for consistent processing
        if df['timestamp'].dt.tz is not None:
             df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        
        df.set_index('timestamp', inplace=True)
        print(f"Loaded {len(df)} rows from CSV.")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def calculate_indicators(df):
    # Minimal indicator calculation needed for the RuleEngine
    # Requires: ADX, RSI, EMA_9, EMA_50, EMA_20, BB_UPPER, BB_LOWER, ATR
    
    df = df.copy()
    close_prices = df['close']
    
    # EMA
    df['EMA_9'] = close_prices.ewm(span=9).mean()
    df['EMA_20'] = close_prices.ewm(span=20).mean()
    df['EMA_50'] = close_prices.ewm(span=50).mean()
    
    # RSI 14
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # ADX 14 (Simplified)
    # Using ATR for ADX
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR_14'] = true_range.rolling(14).mean()
    df['atr_20_avg'] = df['ATR_14'].rolling(20).mean()
    
    # Directional Movement
    # Calculating ADX properly is complex, approximating or using library if available
    # For now, simple approximation or assume needed columns exist if using raw data
    # But this is raw data.
    # Let's simple placeholder ADX or implement basic one
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Smooth
    tr14 = true_range.rolling(14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / tr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / tr14)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX_14'] = dx.rolling(14).mean()
    
    # Bollinger Bands (20, 2)
    sma20 = close_prices.rolling(20).mean()
    std20 = close_prices.rolling(20).std()
    df['BB_UPPER'] = sma20 + (2 * std20)
    df['BB_LOWER'] = sma20 - (2 * std20)
    
    # VWAP (Session VWAP is ideal, but simple cumulative for snippet)
    # Resetting daily would be better
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

    # Fill NaNs
    df.fillna(method='bfill', inplace=True)
    return df

def run_simulation():
    df = load_data()
    if df is None:
        return

    df = calculate_indicators(df)
    
    engine = RuleEngine(config["rule_engine"])
    
    trades = []
    daily_trade_count = {}
    daily_loss_count = {}
    
    print("Starting simulation...")
    print(f"Sample timestamp: {df.index[0]}")
    
    # DEBUG: Count potential candidates
    cnt_model1_candidates = 0
    cnt_model2_candidates = 0

    for i, row in df.iterrows():
        # Condition Check Debug
        # Check raw conditions for Model 1
        # timestamp is row.name
        # Assuming timestamps are naive, we might need to handle TZ.
        # But for now, let's just look at the indicators.
        
        # Model 1 Logic:
        # if (adx > 25 and 
        #          ema_9 > ema_50 and 
        #          abs(price - ema_9) / ema_9 <= 0.0005 and 
        #          45 <= rsi <= 60 and
        #          price < bb_upper)
        
        adx = row['ADX_14']
        rsi = row['RSI_14']
        ema_9 = row['EMA_9']
        ema_50 = row['EMA_50']
        price = row['close']
        bb_upper = row['BB_UPPER']
        bb_lower = row['BB_LOWER']
        vwap = row['vwap']
        
        cond_adx = adx > 25
        cond_trend = ema_9 > ema_50
        cond_pullback = abs(price - ema_9) / ema_9 <= 0.0005
        cond_rsi = 45 <= rsi <= 60
        cond_bb = price < bb_upper
        
        is_rth_time = False
        t_utc = row.name.time()
        # Simple RTH check (UTC 14:30 - 21:00)
        from datetime import time
        if (time(14, 30) <= t_utc <= time(21, 00)):
            is_rth_time = True

        if cond_adx and cond_trend and cond_pullback and cond_rsi and cond_bb:
            cnt_model1_candidates += 1
            if is_rth_time:
                print(f"✅ Model 1 RTH CANDIDATE at {row.name}")
            else:
                print(f"⚠️ Model 1 Non-RTH Candidate at {row.name}")

        # Model 2 Logic
        # if (price <= bb_lower and
        #         rsi < 30 and
        #         dist_to_vwap > 0.004):
        dist_to_vwap = abs(price - vwap) / vwap if vwap > 0 else 0
        
        cond_bb_low = price <= bb_lower
        cond_rsi_low = rsi < 30
        cond_vwap = dist_to_vwap > 0.004
        
        if cond_bb_low and cond_rsi_low and cond_vwap:
             cnt_model2_candidates += 1
             print(f"Model 2 Candidate at {row.name}: RSI={rsi:.1f}, DistVWAP={dist_to_vwap:.4f}")
             
             # Debug why engine rejected it
             if result.signal != TradeAction.BUY:
                 print(f"    ❌ Engine Rejected: Signal={result.signal}, Blocked={result.filters_blocked}, Score={result.score}")
                 print(f"    Explain: {result.indicators.get('score_breakdown', [])}")

        # Construct market_data dict expected by evaluate
        market_data = {
            "price": row['close'],
            "high": row['high'],
            "low": row['low'],
            "volume": row['volume'],
            "time": row.name, # Index is timestamp
            "atr": row['ATR_14'],
            "atr_20_avg": row['atr_20_avg'],
            "adx": row['ADX_14'],
            "rsi": row['RSI_14'],
            "ema_9": row['EMA_9'],
            "ema_20": row['EMA_20'],
            "ema_50": row['EMA_50'],
            "bb_upper": row['BB_UPPER'],
            "bb_lower": row['BB_LOWER'],
            "vwap": row['vwap'],
            
            # Simulated Risk State
            "trades_today": daily_trade_count.get(row.name.date(), 0),
            "daily_losses": daily_loss_count.get(row.name.date(), 0),
            "current_position_qty": 0,
            
            # VX
            "vx_price": 0 # Default
        }
        
        # Determine Session (Simplified)
        # Using simple hour check for now, matching engine logic roughly
        # Engine uses CST. Assuming DF index is UTC or similar.
        # Ideally use proper session manager, but for quick check:
        # RTH: 8:30 - 15:00 CST -> 14:30 - 21:00 UTC (roughly)
        
        try:
            result = engine.evaluate(market_data)
        except Exception as e:
            # print(f"Error evaluating: {e}")
            continue

        # Check for our specific models
        passed_filters = result.filters_passed
        score_details = result.indicators.get("score_breakdown", [])
        
        is_model_1 = any("ENTRY_MODEL_1" in s for s in score_details)
        is_model_2 = any("ENTRY_MODEL_2" in s for s in score_details)
        
        if (is_model_1 or is_model_2) and result.signal == TradeAction.BUY:
            
            # Check Stop Loss Constraint
            price = row['close']
            
            # Model 2 SL Calculation: 0.2%
            sl_pct = 0.002
            sl_pts = price * sl_pct
            
            risk_gate_passed = True
            if sl_pts > 25.0:
                risk_gate_passed = False
                fail_reason = f"SL {sl_pts:.2f} > 25"
            else:
                fail_reason = ""
                
            entry_type = "MODEL_1 (RTH)" if is_model_1 else "MODEL_2 (O/N)"
            
            print(f"[{row.name}] SIGNAL {entry_type} | Price: {price:.2f} | Implied SL: {sl_pts:.2f} pts | RiskGate: {'PASS' if risk_gate_passed else 'FAIL ' + fail_reason}")
            
            if risk_gate_passed:
                trades.append({
                    "time": row.name,
                    "type": entry_type,
                    "price": price,
                    "sl_pts": sl_pts
                })
                # Simulate trade count increment
                d = row.name.date()
                daily_trade_count[d] = daily_trade_count.get(d, 0) + 1

    print(f"\n--- Analysis Results ---")
    print(f"Total Rows Processed: {len(df)}")
    print(f"Model 1 (Trend Pullback) Raw Candidates: {cnt_model1_candidates}")
    print(f"Model 2 (Reversion) Raw Candidates: {cnt_model2_candidates}")

    print(f"\nTotal Valid Trades Found by Engine: {len(trades)}")
    for t in trades:
        print(f"  - {t['time']} {t['type']} @ {t['price']}")

if __name__ == "__main__":
    run_simulation()
