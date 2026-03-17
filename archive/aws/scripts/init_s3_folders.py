#!/usr/bin/env python3
"""
Initialize S3 Folder Structure for Trading Bot

This script creates the required folder structure in S3:
- /raw/          - Raw trade logs and market data
- /structured/   - Parquet and JSON feature files
- /kb/           - Knowledge Base documents
- /pnl/          - P&L tracking data
- /strategy/     - Strategy rules and configurations
- /bad_patterns/ - Patterns to avoid
- /features/     - Computed features

Usage:
    python init_s3_folders.py --bucket <bucket-name> --region <region>
"""

import argparse
import json
from datetime import datetime

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("Error: boto3 is required. Install with: pip install boto3")
    exit(1)


# S3 folder structure
FOLDERS = [
    'raw/',
    'structured/',
    'kb/',
    'pnl/',
    'strategy/',
    'bad_patterns/',
    'features/',
    'lambda-code/',
    'lambda-layers/',
]

# Initial files to create
INITIAL_FILES = {
    'strategy/rules.json': {
        'version': '1.0',
        'created_at': datetime.utcnow().isoformat(),
        'min_confidence': 0.60,
        'blocked_conditions': [],
        'regime_preferences': {
            'UPTREND': {'allowed_actions': ['BUY', 'SELL'], 'confidence_boost': 0.05},
            'DOWNTREND': {'allowed_actions': ['BUY', 'SELL'], 'confidence_boost': 0.0},
            'RANGE': {'allowed_actions': ['BUY', 'SELL'], 'confidence_boost': -0.05}
        },
        'volatility_adjustments': {
            'HIGH': {'size_multiplier': 0.6, 'min_confidence': 0.70},
            'MED': {'size_multiplier': 1.0, 'min_confidence': 0.60},
            'LOW': {'size_multiplier': 1.0, 'min_confidence': 0.55}
        },
        'time_restrictions': {},
        'indicator_thresholds': {
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
    },
    'bad_patterns/patterns.json': [],
    'pnl/risk_state.json': {
        'date': datetime.utcnow().strftime('%Y-%m-%d'),
        'daily_pnl': 0,
        'daily_trades': 0,
        'losing_streak': 0,
        'winning_streak': 0,
        'peak_balance': 100000,
        'drawdown': 0,
        'last_trade_time': None,
        'trade_history': []
    },
    'kb/README.md': """# Trading Bot Knowledge Base

This folder contains documents for the Bedrock Knowledge Base.

## Document Types

- **trade_patterns.json** - Daily trade pattern summaries
- **strategy_docs.md** - Strategy documentation
- **market_insights.json** - Market regime insights

## Auto-Ingestion

Documents placed here will be automatically ingested by the
Bedrock Knowledge Base data source.

## Schema

Trade pattern documents should follow this schema:

```json
{
    "date": "2024-01-15",
    "document_type": "daily_trade_patterns",
    "summary": "...",
    "features": {...},
    "trades": [...]
}
```
""",
}


def init_s3_folders(bucket_name: str, region: str) -> None:
    """Initialize S3 folder structure."""
    print(f"Initializing S3 bucket: {bucket_name}")
    
    s3 = boto3.client('s3', region_name=region)
    
    # Create folder placeholders
    for folder in FOLDERS:
        try:
            # Check if folder exists
            response = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=folder,
                MaxKeys=1
            )
            
            if response.get('KeyCount', 0) == 0:
                # Create empty folder marker
                s3.put_object(
                    Bucket=bucket_name,
                    Key=folder,
                    Body=b''
                )
                print(f"  Created: {folder}")
            else:
                print(f"  Exists:  {folder}")
                
        except ClientError as e:
            print(f"  Error creating {folder}: {e}")
    
    # Create initial files
    print("\nCreating initial files...")
    for key, content in INITIAL_FILES.items():
        try:
            # Check if file exists
            try:
                s3.head_object(Bucket=bucket_name, Key=key)
                print(f"  Exists:  {key}")
                continue
            except ClientError:
                pass  # File doesn't exist, create it
            
            # Determine content type and body
            if key.endswith('.json'):
                body = json.dumps(content, indent=2).encode('utf-8')
                content_type = 'application/json'
            elif key.endswith('.md'):
                body = content.encode('utf-8')
                content_type = 'text/markdown'
            else:
                body = str(content).encode('utf-8')
                content_type = 'text/plain'
            
            s3.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=body,
                ContentType=content_type
            )
            print(f"  Created: {key}")
            
        except ClientError as e:
            print(f"  Error creating {key}: {e}")
    
    print("\nS3 initialization complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Initialize S3 folder structure for Trading Bot'
    )
    parser.add_argument(
        '--bucket',
        required=True,
        help='S3 bucket name'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    
    args = parser.parse_args()
    
    init_s3_folders(args.bucket, args.region)


if __name__ == '__main__':
    main()
