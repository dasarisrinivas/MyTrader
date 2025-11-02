#!/usr/bin/env python3
"""
Download October 2025 data in 7-day chunks (Yahoo Finance 1m data limit is 8 days).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.download_data import download_es_data
from mytrader.utils.logger import logger

def download_month_data(year: int, month: int, output_file: str):
    """
    Download a full month of data in 7-day chunks.
    
    Args:
        year: Year (e.g., 2025)
        month: Month (1-12)
        output_file: Path to save combined CSV
    """
    # Get the first and last day of the month
    first_day = datetime(year, month, 1)
    if month == 12:
        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)
    
    print(f"\n📊 Downloading {first_day.strftime('%B %Y')} ES Futures Data")
    print(f"   Period: {first_day.date()} to {last_day.date()}")
    print(f"   Strategy: 7-day chunks (Yahoo Finance limit)\n")
    
    all_data = []
    current_date = first_day
    chunk_num = 1
    
    while current_date <= last_day:
        # Calculate chunk end date (7 days or end of month)
        chunk_end = min(current_date + timedelta(days=6), last_day)
        # Add 1 day to make it inclusive for the API
        api_end = chunk_end + timedelta(days=1)
        
        start_str = current_date.strftime("%Y-%m-%d")
        end_str = api_end.strftime("%Y-%m-%d")
        
        print(f"📦 Chunk {chunk_num}: {current_date.date()} to {chunk_end.date()}")
        
        try:
            df = download_es_data(start_str, end_str, output_file=None)
            
            if not df.empty:
                all_data.append(df)
                print(f"   ✅ Downloaded {len(df)} bars\n")
            else:
                print(f"   ⚠️  No data (market closed or no trading)\n")
        
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
        
        # Move to next chunk
        current_date = chunk_end + timedelta(days=1)
        chunk_num += 1
    
    # Combine all chunks
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates (in case of overlapping chunks)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'])
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS!")
        print(f"{'='*60}")
        print(f"Total bars downloaded: {len(combined_df)}")
        print(f"Time range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        print(f"Price range: ${combined_df['close'].min():.2f} - ${combined_df['close'].max():.2f}")
        print(f"Saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return combined_df
    else:
        print("\n❌ No data downloaded for the entire month!")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download a full month of ES futures data")
    parser.add_argument("--year", type=int, default=2025, help="Year (default: 2025)")
    parser.add_argument("--month", type=int, default=10, help="Month (1-12, default: 10)")
    parser.add_argument("--output", default="data/es_october_2025.csv", help="Output file")
    
    args = parser.parse_args()
    
    df = download_month_data(args.year, args.month, args.output)
    
    if df is None:
        sys.exit(1)
