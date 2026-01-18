#!/usr/bin/env python3
"""
Download 30-minute ES futures data from Interactive Brokers.

This downloads native 30m bars (not resampled from 1m) for proper overnight backtesting.
IB provides 30m bars with correct OHLCV aggregation.

Usage:
    python scripts/download_30m_data.py
    
Requires:
    - IB Gateway running on port 4002
    - ib_insync installed
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.data.ib_downloader import IBHistoricalDownloader, DownloadConfig
from loguru import logger


async def download_30m_es_data():
    """Download 30-minute ES futures data from IB."""
    
    # Configure download
    config = DownloadConfig(
        host="127.0.0.1",
        port=4002,
        client_id=98,  # Different from main bot
        cache_dir=Path("data/raw"),
    )
    
    downloader = IBHistoricalDownloader(config)
    
    try:
        # Connect to IB
        await downloader.connect()
        
        # Download 60 days of 30m bars (same range as 1m data)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=60)
        
        logger.info(f"Downloading ES 30m data from {start.date()} to {end.date()}")
        
        # Download 30m bars - IB bar size string is "30 mins"
        df = await downloader.download(
            symbol="ES",
            start=start,
            end=end,
            bar_size="30 mins",  # Native 30-minute bars
            use_cache=True,
        )
        
        if df.empty:
            logger.error("No data downloaded!")
            return None
        
        logger.info(f"Downloaded {len(df)} 30m bars")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
        
        # Check data quality
        zero_range_count = (df['high'] == df['low']).sum()
        zero_range_pct = zero_range_count / len(df) * 100
        logger.info(f"Zero-range bars: {zero_range_count} ({zero_range_pct:.1f}%)")
        
        # Save to parquet
        output_path = Path("data/raw/ES/ES_30min_60D.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, engine="pyarrow")
        logger.info(f"Saved to {output_path}")
        
        # Also analyze overnight vs RTH quality
        if df.index.tz is not None:
            df_local = df.copy()
            df_local.index = df_local.index.tz_convert("America/Chicago")
        else:
            df_local = df
        
        # RTH: 9:30 AM - 4:00 PM CT
        rth_mask = (df_local.index.hour >= 9) & (df_local.index.hour < 16)
        overnight_mask = ~rth_mask
        
        rth_zero_range = (df_local[rth_mask]['high'] == df_local[rth_mask]['low']).sum()
        overnight_zero_range = (df_local[overnight_mask]['high'] == df_local[overnight_mask]['low']).sum()
        
        rth_count = rth_mask.sum()
        overnight_count = overnight_mask.sum()
        
        logger.info(f"\nData Quality by Session:")
        logger.info(f"  RTH bars: {rth_count}, Zero-range: {rth_zero_range} ({rth_zero_range/rth_count*100:.1f}%)")
        logger.info(f"  Overnight bars: {overnight_count}, Zero-range: {overnight_zero_range} ({overnight_zero_range/overnight_count*100:.1f}%)")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise
    finally:
        downloader.disconnect()


if __name__ == "__main__":
    logger.info("Starting 30m ES data download from IB...")
    df = asyncio.run(download_30m_es_data())
    
    if df is not None:
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
        print(f"Total bars: {len(df)}")
        print(f"File: data/raw/ES/ES_30min_60D.parquet")
        print("\nSample data:")
        print(df.tail(10))
