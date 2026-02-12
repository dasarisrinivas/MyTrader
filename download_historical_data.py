#!/usr/bin/env python3
"""
Download historical data from IB for backtesting
"""
import sys
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from shree.execution.ib_executor import IBExecutor
from ib_insync import IB, Contract, util

logger.remove()
logger.add(sys.stdout, level="INFO")

async def download_data(start_date: str, end_date: str, output_file: str):
    """Download historical 1-minute data from IB."""
    logger.info(f"Connecting to IB...")
    
    ib = IB()
    await ib.connectAsync('127.0.0.1', 4002, clientId=999, timeout=60)
    
    logger.info("✅ Connected to IB")
    
    # Create MES contract
    contract = Contract()
    contract.symbol = "MES"
    contract.secType = "CONTFUTURE"
    contract.exchange = "CME"
    contract.currency = "USD"
    
    # Qualify contract
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        logger.error("Failed to qualify MES contract")
        return
    
    contract = qualified[0]
    logger.info(f"Using contract: {contract.localSymbol} (exp: {contract.lastTradeDateOrContractMonth})")
    
    # Convert dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    logger.info(f"Downloading data from {start_date} to {end_date}")
    
    all_bars = []
    current_end = end_dt
    
    # Download in chunks (IB limits to ~2000 bars per request for 1-min data)
    # 2000 bars = ~33 hours, so we'll download in 7-day chunks
    while current_end > start_dt:
        chunk_start = max(current_end - timedelta(days=7), start_dt)
        
        logger.info(f"  Downloading: {chunk_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
        
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=current_end,
                durationStr='7 D',
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False,  # Include extended hours
                formatDate=1
            )
            
            if bars:
                df_chunk = util.df(bars)
                df_chunk = df_chunk.rename(columns={'date': 'timestamp'})
                all_bars.append(df_chunk)
                logger.info(f"    ✓ Got {len(bars)} bars")
            else:
                logger.warning(f"    ⚠️  No bars returned")
            
            # Move to next chunk
            current_end = chunk_start
            
            # Rate limiting
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"    ❌ Error: {e}")
            break
    
    if not all_bars:
        logger.error("No data downloaded!")
        ib.disconnect()
        return
    
    # Combine all chunks
    logger.info("Combining data...")
    df = pd.concat(all_bars, ignore_index=True)
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'])
    
    # Filter to requested date range
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
    
    logger.info(f"Total bars: {len(df)}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Saved to: {output_file}")
    
    ib.disconnect()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python download_historical_data.py START_DATE END_DATE OUTPUT_FILE")
        print("Example: python download_historical_data.py 2025-01-01 2025-12-31 data/es_2025_full.csv")
        sys.exit(1)
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    output_file = sys.argv[3]
    
    asyncio.run(download_data(start_date, end_date, output_file))
