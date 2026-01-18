#!/usr/bin/env python3
"""
Download fresh 1-minute ES futures data from Interactive Brokers.

This ensures 1m data matches the date range of our 30m data for proper backtesting.

Usage:
    python scripts/download_fresh_1m_data.py
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.data.ib_downloader import IBHistoricalDownloader, DownloadConfig
from loguru import logger


async def download_1m_es_data():
    """Download fresh 1-minute ES futures data from IB."""
    
    # Configure download
    config = DownloadConfig(
        host="127.0.0.1",
        port=4002,
        client_id=97,  # Different from other downloads
        cache_dir=Path("data/raw"),
    )
    
    downloader = IBHistoricalDownloader(config)
    
    try:
        # Connect to IB
        await downloader.connect()
        
        # Download 60 days of 1m bars (same as 30m data range)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=60)
        
        logger.info(f"Downloading ES 1m data from {start.date()} to {end.date()}")
        
        # Download 1m bars
        df = await downloader.download(
            symbol="ES",
            start=start,
            end=end,
            bar_size="1 min",
            use_cache=False,  # Force fresh download
        )
        
        if df.empty:
            logger.error("No data downloaded!")
            return None
        
        logger.info(f"Downloaded {len(df)} 1m bars")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Save to parquet (overwrite old file)
        output_path = Path("data/raw/ES/ES_1min_60D.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, engine="pyarrow")
        logger.info(f"Saved to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise
    finally:
        downloader.disconnect()


if __name__ == "__main__":
    logger.info("Starting fresh 1m ES data download from IB...")
    df = asyncio.run(download_1m_es_data())
    
    if df is not None:
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
        print(f"Total bars: {len(df)}")
        print(f"File: data/raw/ES/ES_1min_60D.parquet")
