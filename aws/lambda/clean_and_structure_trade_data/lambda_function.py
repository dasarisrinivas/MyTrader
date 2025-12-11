"""
Lambda Function: Clean and Structure Trade Data
Agent 1 - Data Ingestion & Feature Builder

This function normalizes and prepares market/trading logs.
It converts raw data into structured Parquet and JSON feature sets.
"""

import json
import os
import io
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

# Import pyarrow for Parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Initialize AWS clients
s3_client = boto3.client('s3')

# Environment variables
S3_BUCKET = os.environ.get('S3_BUCKET', 'trading-bot-data')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'prod')
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')


# ============================================
# TRADE DATA SCHEMA
# ============================================
TRADE_SCHEMA = {
    'required_fields': [
        'trade_id', 'timestamp', 'symbol', 'action', 'price', 'quantity'
    ],
    'optional_fields': [
        'regime', 'volatility', 'PDH_delta', 'PDL_delta', 'ADR', 'ATR',
        'time_of_day', 'outcome', 'pnl', 'trade_duration', 'confidence',
        'stop_loss', 'take_profit', 'entry_reason', 'exit_reason',
        'rsi', 'macd_histogram', 'ema_9', 'ema_20', 'volume'
    ]
}

FEATURE_SCHEMA = pa.schema([
    ('trade_id', pa.string()),
    ('timestamp', pa.timestamp('ms')),
    ('symbol', pa.string()),
    ('action', pa.string()),  # BUY/SELL
    ('price', pa.float64()),
    ('quantity', pa.int32()),
    ('regime', pa.string()),  # UPTREND/DOWNTREND/RANGE
    ('volatility', pa.string()),  # HIGH/MED/LOW
    ('PDH_delta', pa.float64()),
    ('PDL_delta', pa.float64()),
    ('ADR', pa.float64()),
    ('ATR', pa.float64()),
    ('time_of_day', pa.string()),  # MORNING/MIDDAY/AFTERNOON/CLOSE
    ('outcome', pa.string()),  # WIN/LOSS
    ('pnl', pa.float64()),
    ('trade_duration', pa.int32()),  # seconds
    ('confidence', pa.float64()),
    ('rsi', pa.float64()),
    ('macd_histogram', pa.float64()),
    ('ema_9', pa.float64()),
    ('ema_20', pa.float64()),
    ('volume', pa.int64()),
])


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for cleaning and structuring trade data.
    
    This function is invoked by Bedrock Agent as an action group.
    """
    print(f"[INFO] Received event: {json.dumps(event, default=str)[:1000]}")
    
    # Handle Bedrock Agent invocation format
    if 'actionGroup' in event:
        return _handle_bedrock_agent_request(event, context)
    
    # Handle direct invocation (for testing)
    return _handle_direct_request(event, context)


def _handle_bedrock_agent_request(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle request from Bedrock Agent action group."""
    action_group = event.get('actionGroup', '')
    function_name = event.get('function', '')
    parameters = event.get('parameters', [])
    
    # Convert parameters list to dict
    params = {p['name']: p['value'] for p in parameters} if parameters else {}
    
    print(f"[INFO] Bedrock Agent call - Action: {action_group}, Function: {function_name}, Params: {params}")
    
    try:
        source_prefix = params.get('source_prefix', 'raw/')
        date_filter = params.get('date_filter', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        
        # List files in the source prefix
        result = {
            'status': 'success',
            'message': f'Data ingestion initiated for {source_prefix}',
            'source_prefix': source_prefix,
            'date_filter': date_filter,
            'note': 'Upload raw trade data to S3 raw/ folder for processing'
        }
        
        return {
            'messageVersion': '1.0',
            'response': {
                'actionGroup': action_group,
                'function': function_name,
                'functionResponse': {
                    'responseBody': {
                        'TEXT': {
                            'body': json.dumps(result)
                        }
                    }
                }
            }
        }
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {
            'messageVersion': '1.0',
            'response': {
                'actionGroup': action_group,
                'function': function_name,
                'functionResponse': {
                    'responseBody': {
                        'TEXT': {
                            'body': json.dumps({'error': str(e), 'status': 'error'})
                        }
                    }
                }
            }
        }


def _handle_direct_request(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle direct Lambda invocation.
    
    Input Event:
    {
        "source": "s3" | "direct",
        "raw_data": [...] | null,  # Direct input
        "s3_key": "raw/2024-01-15/trades.json" | null,  # S3 input
        "date": "2024-01-15"  # Processing date
    }
    """
    print(f"[INFO] Direct invocation: {json.dumps(event, default=str)[:500]}")
    
    try:
        # Validate input
        source = event.get('source', 'direct')
        date_str = event.get('date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        
        # Get raw data
        if source == 's3':
            s3_key = event.get('s3_key')
            if not s3_key:
                raise ValueError("s3_key is required when source is 's3'")
            raw_data = _read_s3_json(s3_key)
        else:
            raw_data = event.get('raw_data', [])
        
        if not raw_data:
            return {
                'status': 'success',
                'processed_count': 0,
                'message': 'No data to process',
                'structured_key': None,
                'features_key': None,
                'errors': []
            }
        
        # Clean and validate trades
        cleaned_trades, errors = _clean_and_validate_trades(raw_data)
        
        if not cleaned_trades:
            return {
                'status': 'error',
                'processed_count': 0,
                'message': 'No valid trades after cleaning',
                'errors': errors
            }
        
        # Extract features
        features = _extract_features(cleaned_trades)
        
        # Write to S3
        structured_key = f"structured/{date_str}/trades.parquet"
        features_key = f"features/{date_str}/features.json"
        kb_key = f"kb/{date_str}/trade_patterns.json"
        
        # Write Parquet file
        if PYARROW_AVAILABLE:
            _write_parquet_to_s3(cleaned_trades, structured_key)
        else:
            # Fallback to JSON
            structured_key = f"structured/{date_str}/trades.json"
            _write_json_to_s3(cleaned_trades, structured_key)
        
        # Write features JSON
        _write_json_to_s3(features, features_key)
        
        # Write KB-ready document for Bedrock Knowledge Base
        kb_document = _create_kb_document(cleaned_trades, features, date_str)
        _write_json_to_s3(kb_document, kb_key)
        
        return {
            'status': 'success',
            'processed_count': len(cleaned_trades),
            'structured_key': structured_key,
            'features_key': features_key,
            'kb_key': kb_key,
            'errors': errors,
            'summary': {
                'total_trades': len(cleaned_trades),
                'wins': sum(1 for t in cleaned_trades if t.get('outcome') == 'WIN'),
                'losses': sum(1 for t in cleaned_trades if t.get('outcome') == 'LOSS'),
                'total_pnl': sum(t.get('pnl', 0) for t in cleaned_trades)
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to process trade data: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'processed_count': 0,
            'errors': [str(e)]
        }


def _read_s3_json(s3_key: str) -> List[Dict]:
    """Read JSON data from S3."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except ClientError as e:
        print(f"[ERROR] Failed to read S3 object {s3_key}: {e}")
        raise


def _write_json_to_s3(data: Any, s3_key: str) -> None:
    """Write JSON data to S3."""
    try:
        json_content = json.dumps(data, default=str, indent=2)
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json_content.encode('utf-8'),
            ContentType='application/json'
        )
        print(f"[INFO] Wrote JSON to s3://{S3_BUCKET}/{s3_key}")
    except ClientError as e:
        print(f"[ERROR] Failed to write S3 object {s3_key}: {e}")
        raise


def _write_parquet_to_s3(trades: List[Dict], s3_key: str) -> None:
    """Write trades as Parquet to S3."""
    try:
        # Create PyArrow table
        table = pa.Table.from_pylist(trades, schema=FEATURE_SCHEMA)
        
        # Write to buffer
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression='snappy')
        buffer.seek(0)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType='application/octet-stream'
        )
        print(f"[INFO] Wrote Parquet to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"[ERROR] Failed to write Parquet: {e}")
        raise


def _clean_and_validate_trades(raw_trades: List[Dict]) -> tuple[List[Dict], List[str]]:
    """Clean and validate trade records."""
    cleaned = []
    errors = []
    
    for i, trade in enumerate(raw_trades):
        try:
            # Validate required fields
            missing_fields = [
                f for f in TRADE_SCHEMA['required_fields']
                if f not in trade or trade[f] is None
            ]
            if missing_fields:
                errors.append(f"Trade {i}: Missing required fields: {missing_fields}")
                continue
            
            # Generate trade_id if not present (idempotency)
            if 'trade_id' not in trade:
                trade['trade_id'] = _generate_trade_id(trade)
            
            # Normalize and clean data
            cleaned_trade = _normalize_trade(trade)
            cleaned.append(cleaned_trade)
            
        except Exception as e:
            errors.append(f"Trade {i}: Validation error - {str(e)}")
    
    return cleaned, errors


def _generate_trade_id(trade: Dict) -> str:
    """Generate deterministic trade ID for idempotency."""
    key_string = f"{trade.get('timestamp')}_{trade.get('symbol')}_{trade.get('action')}_{trade.get('price')}"
    return hashlib.md5(key_string.encode()).hexdigest()[:16]


def _normalize_trade(trade: Dict) -> Dict:
    """Normalize trade data to standard format."""
    normalized = {
        'trade_id': trade['trade_id'],
        'timestamp': _parse_timestamp(trade['timestamp']),
        'symbol': str(trade['symbol']).upper(),
        'action': str(trade['action']).upper(),
        'price': float(trade['price']),
        'quantity': int(trade['quantity']),
    }
    
    # Add optional fields with defaults
    normalized['regime'] = str(trade.get('regime', 'UNKNOWN')).upper()
    normalized['volatility'] = _classify_volatility(trade.get('volatility'), trade.get('ATR'))
    normalized['PDH_delta'] = float(trade.get('PDH_delta', 0))
    normalized['PDL_delta'] = float(trade.get('PDL_delta', 0))
    normalized['ADR'] = float(trade.get('ADR', 0))
    normalized['ATR'] = float(trade.get('ATR', 0))
    normalized['time_of_day'] = _classify_time_of_day(trade.get('timestamp'))
    normalized['outcome'] = str(trade.get('outcome', 'PENDING')).upper()
    normalized['pnl'] = float(trade.get('pnl', 0))
    normalized['trade_duration'] = int(trade.get('trade_duration', 0))
    normalized['confidence'] = float(trade.get('confidence', 0))
    normalized['rsi'] = float(trade.get('rsi', 50))
    normalized['macd_histogram'] = float(trade.get('macd_histogram', 0))
    normalized['ema_9'] = float(trade.get('ema_9', 0))
    normalized['ema_20'] = float(trade.get('ema_20', 0))
    normalized['volume'] = int(trade.get('volume', 0))
    
    return normalized


def _parse_timestamp(ts: Any) -> str:
    """Parse and normalize timestamp to ISO format."""
    if isinstance(ts, str):
        # Try to parse various formats
        for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S']:
            try:
                dt = datetime.strptime(ts, fmt)
                return dt.isoformat()
            except ValueError:
                continue
        return ts  # Return as-is if parsing fails
    elif isinstance(ts, (int, float)):
        # Unix timestamp
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return str(ts)


def _classify_volatility(volatility: Optional[str], atr: Optional[float]) -> str:
    """Classify volatility as HIGH/MED/LOW."""
    if volatility:
        vol_upper = str(volatility).upper()
        if vol_upper in ['HIGH', 'MED', 'MEDIUM', 'LOW']:
            return 'MED' if vol_upper == 'MEDIUM' else vol_upper
    
    # Classify based on ATR if provided
    if atr is not None:
        if atr > 2.0:
            return 'HIGH'
        elif atr > 1.0:
            return 'MED'
        else:
            return 'LOW'
    
    return 'MED'  # Default


def _classify_time_of_day(timestamp: Any) -> str:
    """Classify time of day based on market hours (CST)."""
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        else:
            return 'UNKNOWN'
        
        # Convert to CST (UTC-6)
        hour = (dt.hour - 6) % 24
        
        if 8 <= hour < 10:
            return 'MORNING'
        elif 10 <= hour < 12:
            return 'MIDDAY'
        elif 12 <= hour < 14:
            return 'AFTERNOON'
        elif 14 <= hour < 16:
            return 'CLOSE'
        else:
            return 'EXTENDED'
            
    except Exception:
        return 'UNKNOWN'


def _extract_features(trades: List[Dict]) -> Dict[str, Any]:
    """Extract aggregated features from trades."""
    if not trades:
        return {}
    
    # Calculate statistics
    wins = [t for t in trades if t.get('outcome') == 'WIN']
    losses = [t for t in trades if t.get('outcome') == 'LOSS']
    
    features = {
        'date': trades[0].get('timestamp', '')[:10],
        'total_trades': len(trades),
        'win_count': len(wins),
        'loss_count': len(losses),
        'win_rate': len(wins) / len(trades) if trades else 0,
        'total_pnl': sum(t.get('pnl', 0) for t in trades),
        'avg_pnl': sum(t.get('pnl', 0) for t in trades) / len(trades) if trades else 0,
        'avg_win': sum(t.get('pnl', 0) for t in wins) / len(wins) if wins else 0,
        'avg_loss': sum(t.get('pnl', 0) for t in losses) / len(losses) if losses else 0,
        
        # Regime distribution
        'regime_distribution': _count_by_field(trades, 'regime'),
        'volatility_distribution': _count_by_field(trades, 'volatility'),
        'time_of_day_distribution': _count_by_field(trades, 'time_of_day'),
        'action_distribution': _count_by_field(trades, 'action'),
        
        # Performance by regime
        'win_rate_by_regime': _win_rate_by_field(trades, 'regime'),
        'win_rate_by_volatility': _win_rate_by_field(trades, 'volatility'),
        'win_rate_by_time': _win_rate_by_field(trades, 'time_of_day'),
        
        # Indicator averages
        'avg_rsi': sum(t.get('rsi', 50) for t in trades) / len(trades),
        'avg_atr': sum(t.get('ATR', 0) for t in trades) / len(trades),
        'avg_confidence': sum(t.get('confidence', 0) for t in trades) / len(trades),
    }
    
    return features


def _count_by_field(trades: List[Dict], field: str) -> Dict[str, int]:
    """Count trades by field value."""
    counts = {}
    for trade in trades:
        value = trade.get(field, 'UNKNOWN')
        counts[value] = counts.get(value, 0) + 1
    return counts


def _win_rate_by_field(trades: List[Dict], field: str) -> Dict[str, float]:
    """Calculate win rate grouped by field value."""
    groups = {}
    for trade in trades:
        value = trade.get(field, 'UNKNOWN')
        if value not in groups:
            groups[value] = {'wins': 0, 'total': 0}
        groups[value]['total'] += 1
        if trade.get('outcome') == 'WIN':
            groups[value]['wins'] += 1
    
    return {k: v['wins'] / v['total'] if v['total'] > 0 else 0 for k, v in groups.items()}


def _create_kb_document(trades: List[Dict], features: Dict, date_str: str) -> Dict:
    """Create a document suitable for Bedrock Knowledge Base ingestion."""
    # Create text summaries for each trade pattern
    trade_summaries = []
    
    for trade in trades:
        summary = (
            f"Trade on {trade['timestamp']}: {trade['action']} {trade['symbol']} "
            f"at ${trade['price']:.2f}. Regime: {trade['regime']}, "
            f"Volatility: {trade['volatility']}, Time: {trade['time_of_day']}. "
            f"PDH delta: {trade['PDH_delta']:.3f}, PDL delta: {trade['PDL_delta']:.3f}. "
            f"RSI: {trade['rsi']:.1f}, ATR: {trade['ATR']:.3f}. "
            f"Outcome: {trade['outcome']}, PnL: ${trade['pnl']:.2f}"
        )
        trade_summaries.append({
            'trade_id': trade['trade_id'],
            'text': summary,
            'metadata': {
                'action': trade['action'],
                'regime': trade['regime'],
                'volatility': trade['volatility'],
                'outcome': trade['outcome'],
                'pnl': trade['pnl'],
                'confidence': trade['confidence']
            }
        })
    
    return {
        'date': date_str,
        'document_type': 'daily_trade_patterns',
        'summary': f"Trading activity for {date_str}: {features['total_trades']} trades, "
                   f"{features['win_rate']*100:.1f}% win rate, ${features['total_pnl']:.2f} P&L",
        'features': features,
        'trades': trade_summaries
    }
