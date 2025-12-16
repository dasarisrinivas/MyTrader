#!/usr/bin/env python3
"""
Download last 30 days of ES futures data from IBKR.

Reuses the repo's existing IB connection and contract qualification logic.
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

# Apply nest_asyncio to allow nested event loops (required for ib_insync)
nest_asyncio.apply()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mytrader.utils.logger import configure_logging, logger
from mytrader.utils.settings_loader import load_settings


def get_qualified_contract(ib: IB, symbol: str, exchange: str, currency: str) -> Future:
    """
    Get qualified front month contract using repo's logic.
    
    This mirrors the logic from TradeExecutor.get_qualified_contract()
    """
    contract = Future(symbol=symbol, exchange=exchange, currency=currency)
    
    # Request contract details to find front month (synchronous)
    details = ib.reqContractDetails(contract)
    if not details:
        raise ValueError(
            f"Could not find any contracts for {symbol} on {exchange}. "
            f"Check symbol, exchange, and market data permissions."
        )
    
    # Sort by expiration date to get the front month (earliest expiration)
    details.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
    front_month = details[0].contract
    
    logger.info(f"✅ Qualified contract: {front_month.localSymbol} (exp: {front_month.lastTradeDateOrContractMonth})")
    
    return front_month


def download_historical_data(
    ib: IB,
    contract: Future,
    duration_days: int,
    bar_size: str = "1 min",
    what_to_show: str = "TRADES",
    use_rth: bool = False,
) -> pd.DataFrame:
    """
    Download historical data in chunks to avoid rate limits.
    
    Args:
        ib: IB connection
        contract: Qualified contract
        duration_days: Number of days to download
        bar_size: Bar size (e.g., "1 min")
        what_to_show: "TRADES", "MIDPOINT", etc.
        use_rth: Regular trading hours only
    
    Returns:
        DataFrame with historical bars
    """
    end = datetime.now(timezone.utc)
    chunk_days = 7  # Download in 7-day chunks to avoid rate limits
    remaining = duration_days
    chunks = []
    
    logger.info(f"📥 Downloading {duration_days} days of {bar_size} bars...")
    
    while remaining > 0:
        chunk_size = min(chunk_days, remaining)
        duration_str = f"{chunk_size} D"
        
        logger.info(f"  Downloading chunk: {chunk_size} days (remaining: {remaining} days)")
        
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end.strftime("%Y%m%d %H:%M:%S"),
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
                keepUpToDate=False,
            )
            
            if bars:
                df = util.df(bars)
                if not df.empty:
                    chunks.append(df)
                    # Update end time to the earliest bar in this chunk
                    end = df["date"].min().to_pydatetime().replace(tzinfo=timezone.utc)
                    logger.info(f"    ✅ Downloaded {len(df)} bars (from {df['date'].min()} to {df['date'].max()})")
                else:
                    logger.warning(f"    ⚠️  Empty chunk received")
            else:
                logger.warning(f"    ⚠️  No bars returned for this chunk")
            
        except Exception as e:
            logger.error(f"    ❌ Error downloading chunk: {e}")
            # Continue with next chunk
        
        remaining -= chunk_size
        
        # Pacing: sleep between chunks to avoid rate limits
        if remaining > 0:
            sleep_time = 2.0
            logger.debug(f"    ⏳ Sleeping {sleep_time}s before next chunk...")
            ib.sleep(sleep_time)  # Use IB's sleep method which respects the event loop
    
    if not chunks:
        raise ValueError("No bars returned from IB. Check contract permissions/subscription.")
    
    # Combine all chunks
    full_df = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    
    logger.info(f"✅ Total bars downloaded: {len(full_df)}")
    logger.info(f"   Date range: {full_df['date'].min()} to {full_df['date'].max()}")
    
    return full_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download last 30 days of ES futures data from IBKR"
    )
    parser.add_argument("--symbol", default="ES", help="Futures symbol (default: ES)")
    parser.add_argument("--exchange", default="", help="Exchange (default: from config or GLOBEX)")
    parser.add_argument("--currency", default="USD", help="Currency (default: USD)")
    parser.add_argument("--bar-size", default="1 min", help="Bar size (default: 1 min)")
    parser.add_argument("--duration-days", type=int, default=30, help="Days to download (default: 30)")
    parser.add_argument("--what-to-show", default="TRADES", help="What to show (default: TRADES)")
    parser.add_argument("--use-rth", action="store_true", help="Regular trading hours only")
    parser.add_argument("--out", required=True, help="Output parquet file path")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(level="INFO", serialize=False)
    
    # Load settings for IB connection
    try:
        settings = load_settings(args.config)
        host = getattr(settings.data, "ibkr_host", "127.0.0.1")
        port = getattr(settings.data, "ibkr_port", 7497)
        client_id = getattr(settings.data, "ibkr_client_id", 11)
        exchange = args.exchange or getattr(settings.data, "ibkr_exchange", "GLOBEX")
        currency = args.currency or getattr(settings.data, "ibkr_currency", "USD")
    except Exception as e:
        logger.warning(f"Could not load config: {e}, using defaults")
        host = os.getenv("IBKR_HOST", "127.0.0.1")
        port = int(os.getenv("IBKR_PORT", "7497"))
        client_id = int(os.getenv("IBKR_CLIENT_ID", "11"))
        exchange = args.exchange or "GLOBEX"
        currency = args.currency or "USD"
    
    # Create output directory
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Meta path will be updated if we fall back to CSV
    meta_path = out_path.with_suffix(".meta.json")
    
    # Connect to IB (synchronous)
    ib = IB()
    logger.info(f"🔌 Connecting to IBKR at {host}:{port} (client_id={client_id})...")
    
    try:
        ib.connect(host, port, clientId=client_id, timeout=30)
        logger.info("✅ Connected to IBKR")
    except Exception as e:
        logger.error(f"❌ Failed to connect to IBKR: {e}")
        logger.error("   Make sure IB Gateway/TWS is running and configured correctly")
        return 1
    
    try:
        # Get qualified contract (front month)
        contract = get_qualified_contract(ib, args.symbol, exchange, currency)
        
        # Download historical data
        df = download_historical_data(
            ib,
            contract,
            args.duration_days,
            args.bar_size,
            args.what_to_show,
            args.use_rth,
        )
        
        # Convert to MyTrader format (timestamp, open, high, low, close, volume)
        df_clean = pd.DataFrame({
            'timestamp': pd.to_datetime(df['date']),
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': df['volume']
        })
        df_clean.set_index('timestamp', inplace=True)
        
        # Save to parquet (fallback to CSV if pyarrow not available)
        try:
            df_clean.to_parquet(out_path, index=True)
            logger.info(f"💾 Saved data to: {out_path} ({len(df_clean)} rows)")
        except (ImportError, ValueError) as e:
            # Fallback to CSV if parquet not available
            error_str = str(e).lower()
            if 'parquet' in error_str or 'pyarrow' in error_str or 'fastparquet' in error_str:
                logger.warning("⚠️  Parquet support not available, saving as CSV instead")
                logger.warning("   Install pyarrow for better performance: pip install pyarrow")
                csv_path = out_path.with_suffix('.csv')
                df_clean.to_csv(csv_path, index=True)
                logger.info(f"💾 Saved data to: {csv_path} ({len(df_clean)} rows)")
                # Update paths for metadata
                out_path = csv_path
                meta_path = csv_path.with_suffix('.meta.json')
            else:
                raise
        
        # Save metadata (update path if we fell back to CSV)
        meta = {
            "symbol": args.symbol,
            "exchange": exchange,
            "currency": currency,
            "contract_local_symbol": contract.localSymbol,
            "contract_expiry": contract.lastTradeDateOrContractMonth,
            "bar_size": args.bar_size,
            "what_to_show": args.what_to_show,
            "use_rth": args.use_rth,
            "rows": int(len(df_clean)),
            "start": str(df_clean.index.min()),
            "end": str(df_clean.index.max()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "out": str(out_path),
            "format": "parquet" if out_path.suffix == ".parquet" else "csv",
        }
        
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"💾 Saved metadata to: {meta_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1
        
    finally:
        ib.disconnect()
        logger.info("🔌 Disconnected from IBKR")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
