#!/usr/bin/env python3
"""Backfill S3 RAG storage with historical trades from orders.db.

This script reads completed trades from the orders database and
uploads them to S3 RAG storage for historical analysis.

Usage:
    python scripts/backfill_s3_rag.py [--start-date YYYY-MM-DD] [--dry-run]
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import uuid

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mytrader.rag.rag_storage_manager import RAGStorageManager, TradeRecord
from mytrader.utils.timezone_utils import now_cst


def get_completed_trades(db_path: str, start_date: Optional[str] = None) -> List[Dict]:
    """Query completed round-trip trades from orders.db."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Query filled orders with execution details
    query = """
    SELECT 
        o.order_id,
        o.timestamp,
        o.action,
        o.quantity,
        o.status,
        o.avg_fill_price as fill_price,
        o.filled_quantity,
        o.realized_pnl,
        o.commission,
        o.confidence,
        o.atr,
        o.stop_loss,
        o.take_profit,
        o.market_regime,
        date(o.timestamp) as trade_date
    FROM orders o
    WHERE o.status = 'Filled'
      AND o.avg_fill_price IS NOT NULL
    """
    
    if start_date:
        query += f" AND date(o.timestamp) >= '{start_date}'"
    
    query += " ORDER BY o.timestamp"
    
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def match_trades_to_roundtrips(trades: List[Dict]) -> List[Dict]:
    """Match BUY and SELL orders into round-trip trades."""
    roundtrips = []
    pending_entries = []  # Stack of entry orders
    
    for trade in trades:
        action = trade['action'].upper()
        
        if action == 'BUY':
            # Entry order
            pending_entries.append(trade)
        elif action == 'SELL' and pending_entries:
            # Exit order - match with oldest entry
            entry = pending_entries.pop(0)
            
            # Calculate P&L
            entry_price = entry['fill_price'] or 0
            exit_price = trade['fill_price'] or 0
            qty = entry['filled_quantity'] or entry['quantity'] or 1
            
            # For MES: $5 per point
            pnl = (exit_price - entry_price) * qty * 5.0
            
            # Use realized_pnl if available from exit order
            if trade.get('realized_pnl') and abs(trade['realized_pnl']) > 0.001:
                pnl = trade['realized_pnl']
            
            roundtrips.append({
                'trade_id': str(uuid.uuid4())[:8],
                'entry_time': entry['timestamp'],
                'exit_time': trade['timestamp'],
                'action': 'BUY',  # Entry action
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': qty,
                'realized_pnl': pnl,
                'result': 'WIN' if pnl > 0 else 'LOSS' if pnl < 0 else 'SCRATCH',
                'trade_date': entry['trade_date'],
                'confidence': entry.get('confidence', 0.0),
                'atr': entry.get('atr', 0.0),
                'stop_loss': entry.get('stop_loss', 0.0),
                'take_profit': entry.get('take_profit', 0.0),
                'market_regime': entry.get('market_regime', ''),
            })
    
    return roundtrips


def create_trade_record(trade: Dict) -> TradeRecord:
    """Convert a matched trade to RAGStorageManager TradeRecord."""
    return TradeRecord(
        trade_id=trade['trade_id'],
        timestamp=trade['entry_time'],
        action=trade['action'],
        entry_price=trade['entry_price'],
        exit_price=trade['exit_price'],
        quantity=trade['quantity'],
        result=trade['result'],
        pnl=trade['realized_pnl'],  # TradeRecord uses 'pnl' not 'realized_pnl'
        stop_loss=trade.get('stop_loss', 0.0),
        take_profit=trade.get('take_profit', 0.0),
        atr=trade.get('atr', 0.0),
        llm_confidence=trade.get('confidence', 0.0),
        market_trend=trade.get('market_regime', ''),
    )


def main():
    parser = argparse.ArgumentParser(description='Backfill S3 RAG storage with historical trades')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be uploaded without uploading')
    parser.add_argument('--db-path', type=str, default='data/orders.db', help='Path to orders database')
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)
    
    print(f"📂 Reading trades from: {db_path}")
    
    # Get all completed trades
    trades = get_completed_trades(str(db_path), args.start_date)
    print(f"📊 Found {len(trades)} filled orders")
    
    # Match into round-trips
    roundtrips = match_trades_to_roundtrips(trades)
    print(f"🔄 Matched {len(roundtrips)} round-trip trades")
    
    if not roundtrips:
        print("⚠️ No round-trip trades found to backfill")
        return
    
    # Group by date for summary
    by_date = {}
    for rt in roundtrips:
        date = rt['trade_date']
        if date not in by_date:
            by_date[date] = []
        by_date[date].append(rt)
    
    print("\n📅 Trades by date:")
    for date in sorted(by_date.keys()):
        trades_on_date = by_date[date]
        total_pnl = sum(t['realized_pnl'] for t in trades_on_date)
        wins = sum(1 for t in trades_on_date if t['result'] == 'WIN')
        print(f"   {date}: {len(trades_on_date)} trades, {wins} wins, P&L: ${total_pnl:.2f}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - no uploads performed")
        return
    
    # Initialize S3 storage
    print("\n🚀 Uploading to S3...")
    try:
        storage = RAGStorageManager()
    except Exception as e:
        print(f"❌ Failed to initialize S3 storage: {e}")
        sys.exit(1)
    
    # Upload each trade
    uploaded = 0
    errors = 0
    for rt in roundtrips:
        try:
            record = create_trade_record(rt)
            key = storage.save_trade(record)
            uploaded += 1
        except Exception as e:
            print(f"❌ Failed to upload trade {rt['trade_id']}: {e}")
            errors += 1
    
    print(f"\n✅ Backfill complete!")
    print(f"   Uploaded: {uploaded}")
    print(f"   Errors: {errors}")
    print(f"   Total P&L: ${sum(rt['realized_pnl'] for rt in roundtrips):.2f}")


if __name__ == '__main__':
    main()
