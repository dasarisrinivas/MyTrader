#!/usr/bin/env python3
"""
Upload Historical Trade Data to S3 for Knowledge Base

This script:
1. Reads trade data from local backups/CSV files
2. Transforms it into the Knowledge Base document format
3. Uploads to S3 structured/ and kb/ folders
4. Triggers Knowledge Base sync
"""

import boto3
import json
import csv
import os
from datetime import datetime
from pathlib import Path

# Configuration
S3_BUCKET = "trading-bot-data-897729113303"
REGION = "us-east-1"
KB_ID = "Z0EPG8YT8F"

# Initialize clients
s3_client = boto3.client('s3', region_name=REGION)
bedrock_agent = boto3.client('bedrock-agent', region_name=REGION)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def read_orders_csv(filepath: str) -> list:
    """Read orders from CSV file."""
    trades = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only include filled orders (actual trades)
            if row.get('status') == 'Filled' and row.get('action') in ['BUY', 'SELL']:
                trades.append(row)
    return trades


def read_executions_csv(filepath: str) -> list:
    """Read executions from CSV file."""
    executions = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            executions.append(row)
    return executions


def transform_to_kb_documents(orders: list, executions: list) -> list:
    """Transform trade data into Knowledge Base document format."""
    
    # Create execution lookup by order_id
    exec_by_order = {}
    for ex in executions:
        order_id = ex.get('order_id')
        if order_id:
            if order_id not in exec_by_order:
                exec_by_order[order_id] = []
            exec_by_order[order_id].append(ex)
    
    documents = []
    
    for order in orders:
        order_id = order.get('order_id', '')
        symbol = order.get('symbol', 'ES')
        action = order.get('action', '')
        quantity = order.get('quantity', '1')
        avg_fill_price = order.get('avg_fill_price', '')
        stop_loss = order.get('stop_loss', '')
        take_profit = order.get('take_profit', '')
        confidence = order.get('confidence', '')
        atr = order.get('atr', '')
        realized_pnl = order.get('realized_pnl', '0')
        timestamp = order.get('timestamp', '')
        rationale = order.get('rationale', '')
        market_regime = order.get('market_regime', '')
        
        # Determine outcome
        try:
            pnl = float(realized_pnl) if realized_pnl else 0
            outcome = 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BREAKEVEN')
        except:
            outcome = 'UNKNOWN'
            pnl = 0
        
        # Extract time of day
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = dt.hour
            if hour < 10:
                time_of_day = 'MORNING'
            elif hour < 12:
                time_of_day = 'MIDDAY'
            elif hour < 15:
                time_of_day = 'AFTERNOON'
            else:
                time_of_day = 'CLOSE'
        except:
            time_of_day = 'UNKNOWN'
        
        # Create document text for RAG
        doc_text = f"""
Trade Record: {order_id}
Symbol: {symbol}
Action: {action}
Quantity: {quantity}
Entry Price: {avg_fill_price}
Stop Loss: {stop_loss}
Take Profit: {take_profit}
Outcome: {outcome}
P&L: ${pnl:.2f}
Confidence: {confidence}
ATR: {atr}
Time of Day: {time_of_day}
Market Regime: {market_regime}
Timestamp: {timestamp}
Rationale: {rationale}

This {action} trade on {symbol} at price {avg_fill_price} resulted in a {outcome} with P&L of ${pnl:.2f}. 
The trade was executed during the {time_of_day} session with ATR of {atr} and confidence of {confidence}.
"""
        
        # Create structured document
        doc = {
            "trade_id": order_id,
            "symbol": symbol,
            "action": action,
            "quantity": int(quantity) if quantity else 1,
            "entry_price": float(avg_fill_price) if avg_fill_price else 0,
            "stop_loss": float(stop_loss) if stop_loss else None,
            "take_profit": float(take_profit) if take_profit else None,
            "outcome": outcome,
            "pnl": pnl,
            "confidence": float(confidence) if confidence else None,
            "atr": float(atr) if atr else None,
            "time_of_day": time_of_day,
            "market_regime": market_regime,
            "timestamp": timestamp,
            "rationale": rationale,
            "text_content": doc_text.strip()
        }
        
        documents.append(doc)
    
    return documents


def upload_to_s3(documents: list):
    """Upload documents to S3."""
    
    print(f"\nUploading {len(documents)} trade documents to S3...")
    
    # Group by date
    docs_by_date = {}
    for doc in documents:
        try:
            date_str = doc['timestamp'][:10]  # YYYY-MM-DD
        except:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        if date_str not in docs_by_date:
            docs_by_date[date_str] = []
        docs_by_date[date_str].append(doc)
    
    uploaded_keys = []
    
    for date_str, docs in docs_by_date.items():
        # Upload to structured/ folder (JSON format)
        structured_key = f"structured/{date_str}/trades.json"
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=structured_key,
            Body=json.dumps(docs, indent=2),
            ContentType='application/json'
        )
        print(f"  ✅ Uploaded: s3://{S3_BUCKET}/{structured_key} ({len(docs)} trades)")
        uploaded_keys.append(structured_key)
        
        # Upload to kb/ folder (text format for better RAG)
        kb_docs = []
        for doc in docs:
            kb_docs.append({
                "content": doc['text_content'],
                "metadata": {
                    "trade_id": doc['trade_id'],
                    "symbol": doc['symbol'],
                    "action": doc['action'],
                    "outcome": doc['outcome'],
                    "pnl": doc['pnl']
                }
            })
        
        kb_key = f"kb/{date_str}/trade_patterns.json"
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=kb_key,
            Body=json.dumps(kb_docs, indent=2),
            ContentType='application/json'
        )
        print(f"  ✅ Uploaded: s3://{S3_BUCKET}/{kb_key}")
        uploaded_keys.append(kb_key)
    
    # Also create a combined summary document
    summary = create_trading_summary(documents)
    summary_key = "kb/trading_patterns_summary.txt"
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=summary_key,
        Body=summary,
        ContentType='text/plain'
    )
    print(f"  ✅ Uploaded: s3://{S3_BUCKET}/{summary_key}")
    
    return uploaded_keys


def create_trading_summary(documents: list) -> str:
    """Create a summary document of trading patterns."""
    
    total = len(documents)
    wins = [d for d in documents if d['outcome'] == 'WIN']
    losses = [d for d in documents if d['outcome'] == 'LOSS']
    
    buys = [d for d in documents if d['action'] == 'BUY']
    sells = [d for d in documents if d['action'] == 'SELL']
    
    buy_wins = [d for d in buys if d['outcome'] == 'WIN']
    sell_wins = [d for d in sells if d['outcome'] == 'WIN']
    
    total_pnl = sum(d['pnl'] for d in documents)
    
    # Time of day analysis
    tod_stats = {}
    for doc in documents:
        tod = doc.get('time_of_day', 'UNKNOWN')
        if tod not in tod_stats:
            tod_stats[tod] = {'total': 0, 'wins': 0, 'pnl': 0}
        tod_stats[tod]['total'] += 1
        if doc['outcome'] == 'WIN':
            tod_stats[tod]['wins'] += 1
        tod_stats[tod]['pnl'] += doc['pnl']
    
    summary = f"""
TRADING PATTERNS SUMMARY
========================
Generated: {datetime.now().isoformat()}

OVERALL STATISTICS
-----------------
Total Trades: {total}
Wins: {len(wins)} ({len(wins)/total*100:.1f}%)
Losses: {len(losses)} ({len(losses)/total*100:.1f}%)
Total P&L: ${total_pnl:.2f}
Win Rate: {len(wins)/total*100:.1f}%

BUY TRADES
----------
Total BUY Trades: {len(buys)}
BUY Win Rate: {len(buy_wins)/len(buys)*100:.1f}% ({len(buy_wins)}/{len(buys)})
Average BUY P&L: ${sum(d['pnl'] for d in buys)/len(buys):.2f}

SELL TRADES
-----------
Total SELL Trades: {len(sells)}
SELL Win Rate: {len(sell_wins)/len(sells)*100:.1f}% ({len(sell_wins)}/{len(sells)}) 
Average SELL P&L: ${sum(d['pnl'] for d in sells)/len(sells):.2f}

TIME OF DAY ANALYSIS
--------------------
"""
    
    for tod, stats in sorted(tod_stats.items()):
        wr = stats['wins']/stats['total']*100 if stats['total'] > 0 else 0
        summary += f"{tod}: {stats['total']} trades, {wr:.1f}% win rate, ${stats['pnl']:.2f} P&L\n"
    
    summary += """

TRADING RULES DERIVED FROM PATTERNS
-----------------------------------
1. Best time to trade: """ + max(tod_stats.keys(), key=lambda k: tod_stats[k]['wins']/tod_stats[k]['total'] if tod_stats[k]['total'] > 0 else 0) + """
2. Preferred action: """ + ('BUY' if len(buy_wins)/len(buys) > len(sell_wins)/len(sells) else 'SELL') + """
3. Risk management: Stop losses should be set based on ATR values from historical trades

KEY INSIGHTS
------------
- Historical win rate provides baseline for confidence calibration
- Time of day significantly impacts trade outcomes
- Position sizing should account for recent win/loss streaks
"""
    
    return summary


def sync_knowledge_base():
    """Trigger Knowledge Base sync to index new documents."""
    
    print("\nTriggering Knowledge Base sync...")
    
    # Get data sources for the knowledge base
    try:
        response = bedrock_agent.list_data_sources(
            knowledgeBaseId=KB_ID
        )
        
        data_sources = response.get('dataSourceSummaries', [])
        print(f"Found {len(data_sources)} data sources")
        
        for ds in data_sources:
            ds_id = ds['dataSourceId']
            ds_name = ds.get('name', ds_id)
            
            print(f"  Starting ingestion job for: {ds_name}...")
            
            try:
                job_response = bedrock_agent.start_ingestion_job(
                    knowledgeBaseId=KB_ID,
                    dataSourceId=ds_id
                )
                job_id = job_response.get('ingestionJob', {}).get('ingestionJobId', 'unknown')
                print(f"    ✅ Started job: {job_id}")
            except Exception as e:
                print(f"    ⚠️ Could not start job: {str(e)[:50]}")
        
    except Exception as e:
        print(f"⚠️ Could not sync Knowledge Base: {e}")
        print("You can manually sync from AWS Console > Bedrock > Knowledge Bases")


def main():
    print("="*60)
    print("Upload Historical Trade Data to Knowledge Base")
    print("="*60)
    
    # Find data files
    orders_file = PROJECT_ROOT / "backups" / "orders_20251207_123909.csv"
    executions_file = PROJECT_ROOT / "backups" / "executions_20251207_123909.csv"
    
    if not orders_file.exists():
        print(f"❌ Orders file not found: {orders_file}")
        return
    
    print(f"\n📂 Reading data from:")
    print(f"   Orders: {orders_file}")
    print(f"   Executions: {executions_file}")
    
    # Read data
    orders = read_orders_csv(str(orders_file))
    executions = read_executions_csv(str(executions_file)) if executions_file.exists() else []
    
    print(f"\n📊 Found {len(orders)} filled orders, {len(executions)} executions")
    
    # Transform to KB format
    documents = transform_to_kb_documents(orders, executions)
    print(f"📝 Created {len(documents)} trade documents")
    
    # Show sample
    if documents:
        print(f"\n📄 Sample document:")
        sample = documents[0]
        print(f"   Trade ID: {sample['trade_id']}")
        print(f"   Action: {sample['action']}")
        print(f"   Outcome: {sample['outcome']}")
        print(f"   P&L: ${sample['pnl']:.2f}")
    
    # Upload to S3
    uploaded = upload_to_s3(documents)
    
    # Sync Knowledge Base
    sync_knowledge_base()
    
    print("\n" + "="*60)
    print("✅ UPLOAD COMPLETE!")
    print("="*60)
    print(f"\nUploaded {len(documents)} trade documents to S3")
    print(f"Knowledge Base sync initiated")
    print("\nThe agents can now use this historical data for:")
    print("  - Pattern matching (Decision Engine)")
    print("  - Win rate analysis (Calculate Winrate Lambda)")
    print("  - Loss pattern detection (Learning Agent)")


if __name__ == '__main__':
    main()
