#!/usr/bin/env python3
"""
Download 1 year of ES futures data from IBKR using Continuous Futures.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from ib_insync import IB, Future, util
import nest_asyncio

# Apply nest_asyncio
nest_asyncio.apply()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shree.utils.logger import configure_logging, logger
from shree.utils.settings_loader import load_settings


def download_historical_data(ib: IB, contract, duration_str: str, bar_size: str, what_to_show: str, use_rth: bool) -> pd.DataFrame:
    """Download historical data in chunks explicitly using expired contracts."""
    
    end = datetime.now(timezone.utc)
    target_start_date = end - timedelta(days=365)
    
    chunks = []
    
    # We must iterate backwards month by month finding the specific contract for that period
    # because ContFuture doesn't support endDateTime hacking well.
    
    current_date = end
    
    while current_date > target_start_date:
        # Find the contract that was active 'current_date'
        # For ES, front month expires Mar(H), Jun(M), Sep(U), Dec(Z)
        # We need to find the specific contract that was the front month at 'current_date'
        
        # Helper to get contract month code
        # Logic: ES contracts expire 3rd Friday of Month. 
        # But rolling happens ~8 days prior.
        # Simplification: Use the contract expiring in the current or next quarter.
        
        # Actually, let's use a simpler approach if we want 1 year of data:
        # Just request the specific Quarterly contracts passed:
        # current date -> identify contract.
        
        # Better approach for downloading 1 year of consistent 1m data without ContFuture complexity:
        # Manually iterate firmly known contracts:
        # ESH6 (Mar 26), ESZ5 (Dec 25), ESU5 (Sep 25), ESM5 (Jun 25), ESH5 (Mar 25)
        
        pass 
        break # Breaking to switch strategy above
        
    # Manual List Strategy
    # Construct list of last 5 quarterly contracts
    # ES Futures symbols: H (Mar), M (Jun), U (Sep), Z (Dec)
    # Current is Jan 2026 -> Front is H6 (Mar 2026)
    # Previous: Z5 (Dec 2025), U5 (Sep 2025), M5 (Jun 2025), H5 (Mar 2025)
    
    contracts_specs = [
        ("202603", "20251212"), # H6
        ("202512", "20250912"), # Z5
        ("202509", "20250613"), # U5
        ("202506", "20250314"), # M5
        ("202503", "20241213"), # H5
    ]
    
    # We need to fetch data for each of these specific contracts for their active period
    
    for expiry, roll_date_str in contracts_specs:
        # roll_dt needs to be timezone aware
        roll_dt = pd.to_datetime(roll_date_str).replace(tzinfo=timezone.utc)
        
        # Define the specific contract - use CME or GLOBEX depending on setup, try CME first as per config
        # IBKR often maps GLOBEX to CME or vice versa, but being explicit helps.
        # Also use just YYYYMM for lastTradeDateOrContractMonth to match easier.
        contract_details = Future(symbol='ES', lastTradeDateOrContractMonth=expiry, exchange='CME', currency='USD', includeExpired=True)
        qualified_list = ib.qualifyContracts(contract_details)
        
        if not qualified_list:
            # Fallback to GLOBEX if CME fails
            contract_details.exchange = 'GLOBEX'
            qualified_list = ib.qualifyContracts(contract_details)
            
        if not qualified_list:
            logger.warning(f"Could not qualify ES expiring {expiry}")
            continue
            
        real_contract = qualified_list[0]
        expiry_dt = pd.to_datetime(real_contract.lastTradeDateOrContractMonth).replace(tzinfo=timezone.utc)
        
        logger.info(f"Downloading {real_contract.localSymbol} (exp: {real_contract.lastTradeDateOrContractMonth})")
        
        # Download this specific contract in chunks
        # End time for this contract is min(now, expiry)
        contract_end = min(end, expiry_dt.replace(tzinfo=timezone.utc))
        contract_start = roll_dt
        
        curr = contract_end
        
        while curr > contract_start:
            # Chunk size
            chunk_days = 5 # 5 days is safe for 1 min
            
            # Start of this chunk
            chunk_start_limit = max(contract_start, curr - timedelta(days=chunk_days))
            
            # Calculate duration string
            # It's better to just ask for "5 D" ending at curr, and filter later if needed, 
            # but safe to just overlap slightly.
            
            duration = f"{chunk_days} D"
            
            logger.info(f"  Fetching {real_contract.localSymbol} ending {curr}, duration {duration}")
            
            try:
                bars = ib.reqHistoricalData(
                    real_contract,
                    endDateTime=curr,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=1,
                    keepUpToDate=False,
                )
                
                if bars:
                    df = util.df(bars)
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        if df['date'].dt.tz is None:
                            df['date'] = df['date'].dt.tz_localize('UTC')
                        
                        # Filter to ensure we don't get data before our roll date (to avoid overlap duplicates with prev contract)
                        df = df[df['date'] >= contract_start]
                        
                        if not df.empty:
                            chunks.append(df)
                            logger.info(f"    ✅ Got {len(df)} bars. Min date: {df['date'].min()}")
                    else:
                        logger.warning("    ⚠️ Empty DataFrame")
                else:
                    logger.warning("    ⚠️ No bars")
                    
            except Exception as e:
                logger.error(f"    ❌ Error: {e}")
            
            # Move back
            curr = curr - timedelta(days=chunk_days)
            ib.sleep(2.0)
            
    if not chunks:
        raise ValueError("No bars returned from IB.")
    
    # Combine all chunks
    logger.info("Combining chunks...")
    full_df = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    
    return full_df


def main():
    parser = argparse.ArgumentParser(description="Download 1y continuous futures data")
    parser.add_argument("--symbol", default="ES", help="Symbol (default: ES)")
    parser.add_argument("--out", required=True, help="Output parquet path")
    parser.add_argument("--bar-size", default="1 min", help="Bar size (default: 1 min)")
    
    args = parser.parse_args()
    configure_logging(level="INFO")
    
    # Settings / Defaults
    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PORT", "4001"))  # 4001=live gateway, 4002=paper gateway
    client_id = 15 # Different ID to avoid conflict
    
    ib = IB()
    try:
        logger.info(f"Connecting to IBKR {host}:{port}...")
        ib.connect(host, port, clientId=client_id)
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return 1
        
    try:
        # Use Manual splicing of specific contracts instead of ContFuture
        # because ContFuture doesn't support 'endDateTime'
        
        # contract = ContFuture(symbol=args.symbol, exchange='GLOBEX', currency='USD')
        # logger.info(f"Contract: {contract}")
        
        df = download_historical_data(ib, None, "1 Y", args.bar_size, "TRADES", False)
        
        # Save
         # Convert to Shree format (timestamp, open, high, low, close, volume)
        df_clean = pd.DataFrame({
            'timestamp': df['date'],
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': df['volume']
        })
        df_clean.set_index('timestamp', inplace=True)
        
        # Ensure dir
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        
        df_clean.to_parquet(args.out)
        logger.info(f"Saved {len(df_clean)} rows to {args.out}")
        
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return 1
    finally:
        ib.disconnect()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
