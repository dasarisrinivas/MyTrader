#!/usr/bin/env python3
"""
Download real ES futures data from Yahoo Finance for backtesting.

Downloads 1-minute bars for ES futures for a specified date range.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mytrader.utils.logger import logger


def download_es_data(start_date: str, end_date: str, output_file: str = None) -> pd.DataFrame:
    """
    Download ES futures data from Yahoo Finance.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_file: Optional path to save CSV file
    
    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Downloading ES futures data from {start_date} to {end_date}")
    
    # ES=F is the E-mini S&P 500 futures ticker on Yahoo Finance
    ticker = "ES=F"
    
    try:
        # Download data with 1-minute intervals
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1m",
            progress=False,
            auto_adjust=True
        )
        
        if df.empty:
            logger.warning("No data downloaded. Market might be closed or invalid date range.")
            return df
        
        # Flatten MultiIndex columns if present (yfinance sometimes returns MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Rename columns to match our format
        df.columns = df.columns.str.lower()
        df = df.rename(columns={
            'adj close': 'adj_close'
        })
        
        # Keep only necessary columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # Reset index to make timestamp a column
        df.reset_index(inplace=True)
        df.rename(columns={'Datetime': 'timestamp'}, inplace=True)
        
        # Add synthetic sentiment columns (neutral for now)
        df['sentiment_twitter'] = 0.0
        df['sentiment_news'] = 0.0
        
        logger.info(f"Downloaded {len(df)} bars")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved data to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise


def get_this_friday() -> str:
    """Get the date of this Friday in YYYY-MM-DD format."""
    today = datetime.now()
    # Friday is weekday 4 (Monday is 0)
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0 and today.hour >= 16:  # After market close
        days_until_friday = 7  # Next Friday
    friday = today + timedelta(days=days_until_friday)
    return friday.strftime("%Y-%m-%d")


def get_last_friday() -> str:
    """Get the date of last Friday in YYYY-MM-DD format."""
    today = datetime.now()
    # Calculate days since last Friday
    days_since_friday = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=days_since_friday)
    return last_friday.strftime("%Y-%m-%d")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ES futures data for backtesting")
    parser.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD). Default: last Friday",
        default=None
    )
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD). Default: this Friday",
        default=None
    )
    parser.add_argument(
        "--output",
        help="Output CSV file path",
        default="data/es_friday_data.csv"
    )
    parser.add_argument(
        "--last-week",
        action="store_true",
        help="Download last week's data (Mon-Fri)"
    )
    
    args = parser.parse_args()
    
    # Determine date range
    if args.last_week:
        # Get last week Monday to Friday
        today = datetime.now()
        days_since_monday = (today.weekday() - 0) % 7
        last_monday = today - timedelta(days=days_since_monday + 7)
        last_friday = last_monday + timedelta(days=4)
        start_date = last_monday.strftime("%Y-%m-%d")
        end_date = (last_friday + timedelta(days=1)).strftime("%Y-%m-%d")  # +1 for inclusive
    else:
        start_date = args.start or get_last_friday()
        end_date = args.end or get_this_friday()
        # Add one day to end_date to make it inclusive
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        end_date = end_date_obj.strftime("%Y-%m-%d")
    
    print(f"\n📊 Downloading ES Futures Data")
    print(f"   Start: {start_date}")
    print(f"   End: {end_date}")
    print(f"   Output: {args.output}\n")
    
    # Download data
    df = download_es_data(start_date, end_date, args.output)
    
    if not df.empty:
        print(f"\n✅ Success!")
        print(f"   Downloaded: {len(df)} bars")
        print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"   Saved to: {args.output}\n")
    else:
        print("\n⚠️  No data downloaded. Market might be closed or invalid date range.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
