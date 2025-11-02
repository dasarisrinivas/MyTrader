#!/usr/bin/env python3
"""
IBKR Connection Setup and Data Download Script

This script helps you:
1. Test your IBKR connection
2. Download historical ES futures data
3. Validate data quality
4. Prepare data for backtesting

Requirements:
- IB Gateway or TWS running on port 4002 (paper trading) or 7496 (live)
- IBKR account with market data subscriptions
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from ib_insync import IB, Future, util

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mytrader.utils.logger import configure_logging, logger


def test_connection(host='127.0.0.1', port=4002, client_id=1):
    """Test IBKR connection."""
    print("\n" + "=" * 80)
    print("Testing IBKR Connection")
    print("=" * 80)
    
    ib = IB()
    
    try:
        print(f"\n🔌 Connecting to IBKR at {host}:{port}...")
        ib.connect(host, port, clientId=client_id, timeout=10)
        
        print("✅ Connected successfully!")
        
        # Get account info
        accounts = ib.managedAccounts()
        print(f"\n📊 Managed Accounts: {accounts}")
        
        # Get account summary
        if accounts:
            account = accounts[0]
            summary = ib.accountSummary(account)
            
            print(f"\n💰 Account Summary for {account}:")
            print("-" * 80)
            for item in summary[:10]:  # Show first 10 items
                print(f"   {item.tag:30s}: {item.value:>20s} {item.currency}")
        
        return ib
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure IB Gateway or TWS is running")
        print("2. Check that API connections are enabled in TWS settings")
        print("3. Verify the port number (4002 for paper, 7496 for live)")
        print("4. Check that 'localhost' connections are allowed")
        print("5. Disable 'Read-Only API' in Global Configuration → API → Settings")
        return None


def download_es_futures(ib, duration='30 D', bar_size='1 min', save_path='data/es_historical.csv'):
    """Download ES futures historical data."""
    print("\n" + "=" * 80)
    print("Downloading ES Futures Data")
    print("=" * 80)
    
    try:
        # Define ES futures contract
        # Use current front month (December 2025 for November 2025)
        contract = Future('ES', '202512', 'CME')  # December 2025 contract
        
        print(f"\n📋 Note: Using ES December 2025 contract (front month)")
        print(f"   If this fails, the current front month may be March 2026 (202603)")
        
        print(f"\n📡 Requesting contract details for {contract.symbol}...")
        ib.qualifyContracts(contract)
        print(f"✅ Contract qualified: {contract}")
        
        # Request historical data
        print(f"\n📥 Downloading {duration} of {bar_size} bars...")
        print("   This may take a few minutes...")
        
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=False,  # Include after-hours data
            formatDate=1
        )
        
        if not bars:
            print("❌ No data received")
            return None
        
        # Convert to DataFrame
        df = util.df(bars)
        print(f"\n✅ Downloaded {len(df)} bars")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # Prepare for MyTrader format
        df_clean = pd.DataFrame({
            'timestamp': pd.to_datetime(df['date']),
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': df['volume']
        })
        df_clean.set_index('timestamp', inplace=True)
        
        # Save to CSV
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(save_path)
        
        print(f"\n💾 Data saved to: {save_path}")
        print(f"   File size: {save_path.stat().st_size / 1024:.1f} KB")
        
        # Show data quality metrics
        print("\n📊 Data Quality:")
        print(f"   Missing values: {df_clean.isnull().sum().sum()}")
        print(f"   Duplicate timestamps: {df_clean.index.duplicated().sum()}")
        print(f"   Zero volume bars: {(df_clean['volume'] == 0).sum()}")
        
        # Show sample data
        print("\n📋 Sample Data (first 5 rows):")
        print(df_clean.head())
        
        return df_clean
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nCommon issues:")
        print("1. Market data subscription required for ES futures")
        print("2. Contract month may need adjustment (check current front month)")
        print("3. Too much data requested - try shorter duration")
        return None


def download_multiple_timeframes(ib):
    """Download ES data in multiple timeframes for multi-timeframe analysis."""
    print("\n" + "=" * 80)
    print("Downloading Multiple Timeframes")
    print("=" * 80)
    
    timeframes = [
        ('5 D', '1 min', 'data/es_5d_1min.csv'),
        ('30 D', '5 mins', 'data/es_30d_5min.csv'),
        ('60 D', '15 mins', 'data/es_60d_15min.csv'),
        ('6 M', '1 hour', 'data/es_6m_1hour.csv'),
    ]
    
    results = {}
    
    for duration, bar_size, save_path in timeframes:
        print(f"\n📥 Downloading {duration} of {bar_size} bars...")
        df = download_es_futures(ib, duration, bar_size, save_path)
        if df is not None:
            results[bar_size] = df
            print(f"✅ Saved to {save_path}")
        else:
            print(f"⚠️  Failed to download {bar_size} data")
        
        # Wait between requests to avoid rate limiting
        import time
        time.sleep(2)
    
    return results


def validate_data(csv_path):
    """Validate downloaded data for use in backtesting."""
    print("\n" + "=" * 80)
    print("Validating Data")
    print("=" * 80)
    
    try:
        df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
        
        print(f"\n📊 Data loaded from: {csv_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        
        # Check for issues
        issues = []
        
        if df.isnull().sum().sum() > 0:
            issues.append(f"❌ Missing values: {df.isnull().sum().sum()}")
        else:
            print("✅ No missing values")
        
        if df.index.duplicated().sum() > 0:
            issues.append(f"❌ Duplicate timestamps: {df.index.duplicated().sum()}")
        else:
            print("✅ No duplicate timestamps")
        
        if (df['volume'] == 0).sum() > 0:
            issues.append(f"⚠️  Zero volume bars: {(df['volume'] == 0).sum()}")
        else:
            print("✅ No zero volume bars")
        
        # Check for data gaps
        time_diff = df.index.to_series().diff()
        median_diff = time_diff.median()
        gaps = time_diff[time_diff > median_diff * 2]
        if len(gaps) > 0:
            issues.append(f"⚠️  Data gaps detected: {len(gaps)}")
            print(f"⚠️  {len(gaps)} data gaps detected (> 2x median interval)")
        else:
            print("✅ No significant data gaps")
        
        # Check OHLC validity
        invalid_ohlc = (df['high'] < df['low']) | (df['close'] > df['high']) | (df['close'] < df['low'])
        if invalid_ohlc.sum() > 0:
            issues.append(f"❌ Invalid OHLC bars: {invalid_ohlc.sum()}")
        else:
            print("✅ Valid OHLC data")
        
        if issues:
            print("\n⚠️  Issues found:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✅ Data validation passed!")
        
        return len(issues) == 0
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return False


def main():
    """Main setup workflow."""
    configure_logging(level="INFO")
    
    print("\n" + "=" * 80)
    print("MyTrader - IBKR Setup and Data Download")
    print("=" * 80)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Setup IBKR connection and download data')
    parser.add_argument('--host', default='127.0.0.1', help='IBKR host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=4002, help='IBKR port (default: 4002 for paper)')
    parser.add_argument('--duration', default='30 D', help='Historical data duration (default: 30 D)')
    parser.add_argument('--bar-size', default='1 min', help='Bar size (default: 1 min)')
    parser.add_argument('--output', default='data/es_historical.csv', help='Output file path')
    parser.add_argument('--validate', help='Validate existing CSV file')
    parser.add_argument('--multiple', action='store_true', help='Download multiple timeframes')
    
    args = parser.parse_args()
    
    # Validation mode
    if args.validate:
        validate_data(args.validate)
        return
    
    # Test connection
    ib = test_connection(args.host, args.port)
    
    if ib is None:
        print("\n❌ Setup failed - could not connect to IBKR")
        return
    
    try:
        # Download data
        if args.multiple:
            download_multiple_timeframes(ib)
        else:
            df = download_es_futures(ib, args.duration, args.bar_size, args.output)
            
            if df is not None:
                # Validate
                validate_data(args.output)
                
                print("\n" + "=" * 80)
                print("✅ Setup Complete!")
                print("=" * 80)
                print(f"\nYou can now run backtests with this data:")
                print(f"python main.py backtest --data {args.output}")
        
    finally:
        ib.disconnect()
        print("\n🔌 Disconnected from IBKR")


if __name__ == "__main__":
    main()
