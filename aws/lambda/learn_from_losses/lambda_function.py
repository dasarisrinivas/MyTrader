"""
Lambda Function: Learn From Losses
Agent 4 - Strategy Optimization & Learning Agent

This function analyzes losing trades, identifies repeating loss patterns,
and updates strategy rules to avoid similar mistakes in the future.
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError

from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit
from pydantic import BaseModel, ValidationError

# Initialize AWS clients
s3_client = boto3.client('s3')
cloudwatch_client = boto3.client('cloudwatch')

# Powertools
logger = Logger()
metrics = Metrics(namespace="TradingBot")
tracer = Tracer()

# Environment variables
S3_BUCKET = os.environ.get('S3_BUCKET', 'trading-bot-data')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'prod')
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
METRIC_NAMESPACE = 'MyTrader/LearningAgent'

# Analysis configuration
MIN_PATTERN_OCCURRENCES = 3
LOOKBACK_DAYS = 30
CONFIDENCE_DECAY = 0.95  # How much old patterns decay in importance


class LearningRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    min_loss_threshold: Optional[float] = None


class DailyLossSummaryBuilder:
    """Builds a per-day loss summary from S3 structured data."""

    def __init__(self, bucket: str):
        self.bucket = bucket

    def build(self, target_date: str) -> Dict[str, Any]:
        """Load trades and pnl data for a target date."""
        trades, source_key = self._load_structured_trades(target_date)
        losing_trades = [t for t in trades if _is_losing_trade(t)]
        pnl_doc = self._load_json(f'pnl/pnl_{target_date}.json')

        summary = {
            'date': target_date,
            'total_trades': len(trades),
            'losing_trades': losing_trades,
            'loss_count': len(losing_trades),
            'loss_pnl': sum(t.get('pnl', 0) for t in losing_trades),
            'trade_source_key': source_key,
            'pnl_document': pnl_doc,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

        return summary

    def _load_structured_trades(self, target_date: str) -> Tuple[List[Dict[str, Any]], str]:
        key = f'structured/{target_date}/trades.json'
        try:
            response = s3_client.get_object(Bucket=self.bucket, Key=key)
            trades = json.loads(response['Body'].read().decode('utf-8'))
            return trades, key
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return [], key
            raise

    def _load_json(self, key: str) -> Dict[str, Any]:
        try:
            response = s3_client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return {}
            raise


def _log_event(event_type: str, payload: Dict[str, Any]) -> None:
    """Emit structured CloudWatch log entry."""
    entry = {
        'event': event_type,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        **payload
    }
    logger.info(json.dumps(entry, default=str))


def _publish_metrics(metrics: List[Dict[str, Any]]) -> None:
    """Publish CloudWatch metrics with shared dimensions."""
    if not metrics:
        return

    timestamp = datetime.now(timezone.utc)
    dimensions = [
        {'Name': 'Environment', 'Value': ENVIRONMENT},
        {'Name': 'Agent', 'Value': 'Agent4'}
    ]

    metric_payload = []
    for metric in metrics:
        metric_payload.append({
            'MetricName': metric['name'],
            'Timestamp': timestamp,
            'Value': metric['value'],
            'Unit': metric.get('unit', 'Count'),
            'Dimensions': dimensions
        })

    try:
        cloudwatch_client.put_metric_data(
            Namespace=METRIC_NAMESPACE,
            MetricData=metric_payload
        )
    except ClientError as e:
        logger.warning(f"Failed to push CloudWatch metrics: {e}")


def _resolve_target_date(value: Optional[str]) -> str:
    """Normalize incoming date/timestamp to YYYY-MM-DD."""
    if not value:
        return datetime.now(timezone.utc).strftime('%Y-%m-%d')

    try:
        if 'T' in value:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        else:
            parsed = datetime.strptime(value, '%Y-%m-%d')
        return parsed.date().isoformat()
    except Exception:
        return datetime.now(timezone.utc).strftime('%Y-%m-%d')


def _build_analysis_range(target_date: str, lookback: int = LOOKBACK_DAYS) -> Dict[str, str]:
    """Build analysis date range inclusive of target date."""
    end_dt = datetime.strptime(target_date, '%Y-%m-%d')
    start_dt = end_dt - timedelta(days=lookback)
    return {
        'start': start_dt.strftime('%Y-%m-%d'),
        'end': end_dt.strftime('%Y-%m-%d')
    }


def _persist_learning_outputs(
    target_date: str,
    run_id: str,
    mode: str,
    source: str,
    schedule_type: str,
    daily_summary: Dict[str, Any],
    learning_payload: Dict[str, Any],
    dry_run: bool = False
) -> int:
    """Persist learning artifacts to S3 (idempotent writes)."""
    if dry_run:
        return 0

    learning_key = f'learning/{target_date}/agent4_learning_update.json'
    state_key = 'learning/strategy_state.json'
    artifacts_written = 0

    learning_document = {
        'run_id': run_id,
        'mode': mode,
        'source': source,
        'schedule_type': schedule_type,
        'date': target_date
    }
    learning_document.update(learning_payload)
    if 'daily_summary' not in learning_document:
        learning_document['daily_summary'] = daily_summary

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=learning_key,
        Body=json.dumps(learning_document, indent=2).encode('utf-8'),
        ContentType='application/json'
    )
    artifacts_written += 1

    state_document = {
        'last_run_id': run_id,
        'last_run_date': target_date,
        'mode': mode,
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'summary': {
            'patterns_identified': len(learning_payload.get('patterns_identified', [])),
            'rules_updated': len(learning_payload.get('rules_updated', [])),
            'bad_patterns_added': len(learning_payload.get('bad_patterns_added', [])),
            'loss_count': daily_summary.get('loss_count', 0),
            'loss_pnl': daily_summary.get('loss_pnl', 0)
        }
    }

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=state_key,
        Body=json.dumps(state_document, indent=2).encode('utf-8'),
        ContentType='application/json'
    )
    artifacts_written += 1

    logger.info(f"Persisted learning artifacts to s3://{S3_BUCKET}/{learning_key} and {state_key}")
    return artifacts_written


def _is_losing_trade(trade: Dict[str, Any]) -> bool:
    """Determine whether a trade is a loss."""
    if trade.get('outcome') == 'LOSS':
        return True
    pnl = trade.get('pnl')
    try:
        return pnl is not None and float(pnl) < 0
    except (TypeError, ValueError):
        return False


@tracer.capture_lambda_handler
@logger.inject_lambda_context
@metrics.log_metrics
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Analyze losing trades and update strategy rules.
    
    This function is invoked by Bedrock Agent as an action group.
    """
    logger.info(f"Received event: {json.dumps(event, default=str)[:1000]}")
    metrics.add_metric(name="RequestsReceived", unit=MetricUnit.Count, value=1)
    
    # Handle Bedrock Agent invocation format
    if 'actionGroup' in event:
        return _handle_bedrock_agent_request(event, context)
    
    # Handle direct invocation (Step Functions, API, manual)
    try:
        validated = LearningRequest(**event)
    except ValidationError as exc:
        logger.error(f"Input validation failed: {exc}")
        metrics.add_metric(name="ValidationErrors", unit=MetricUnit.Count, value=1)
        raise
    merged_event = {**event, **validated.dict()}
    return _handle_direct_request(merged_event, context)


def _handle_bedrock_agent_request(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle request from Bedrock Agent action group."""
    action_group = event.get('actionGroup', '')
    function_name = event.get('function', '')
    parameters = event.get('parameters', [])
    
    # Convert parameters list to dict
    params = {p['name']: p['value'] for p in parameters} if parameters else {}
    
    logger.info(f"Bedrock Agent call - Action: {action_group}, Function: {function_name}, Params: {params}")
    metrics.add_metric(name="BedrockRequests", unit=MetricUnit.Count, value=1)
    
    try:
        start_date = params.get('start_date', (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d'))
        end_date = params.get('end_date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        min_loss = float(params.get('min_loss_threshold', 0))
        
        # Generate learning analysis response
        result = {
            'status': 'success',
            'analysis_period': f'{start_date} to {end_date}',
            'min_loss_threshold': min_loss,
            'patterns_identified': [],
            'rules_updated': [],
            'recommendations': [
                'Upload historical trade data to S3 for pattern analysis',
                'Ensure trades have outcome and pnl fields for loss detection'
            ],
            'message': 'Learning analysis complete. Upload trade history for detailed pattern detection.'
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
        logger.error(f"Learning Bedrock action error: {str(e)}")
        metrics.add_metric(name="ProcessingErrors", unit=MetricUnit.Count, value=1)
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
    """Handle direct Lambda invocation."""
    start_time = time.time()
    run_id = str(event.get('run_id') or (getattr(context, 'aws_request_id', None)) or uuid4())
    schedule_type = event.get('schedule_type', 'nightly')
    source = event.get('source', 'direct')
    mode = event.get('mode') or ('backtest' if event.get('backtest_date') else 'live')
    dry_run = bool(event.get('dry_run', False))
    analysis_type = event.get('analysis_type', 'daily')

    target_date = _resolve_target_date(
        event.get('date') or event.get('backtest_date') or event.get('date_range', {}).get('end')
    )
    analysis_range = event.get('date_range') or _build_analysis_range(target_date)

    run_metadata = {
        'run_id': run_id,
        'target_date': target_date,
        'mode': mode,
        'source': source,
        'schedule_type': schedule_type,
        'analysis_type': analysis_type,
        'dry_run': dry_run,
        'analysis_range': analysis_range
    }
    _log_event('LEARNING_AGENT_START', run_metadata)

    try:
        summary_builder = DailyLossSummaryBuilder(S3_BUCKET)
        daily_summary = summary_builder.build(target_date)
        provided_losses = event.get('losing_trades')

        if provided_losses is not None:
            losing_trades = provided_losses
        else:
            losing_trades = _fetch_losing_trades(analysis_range)
            if not losing_trades:
                losing_trades = daily_summary.get('losing_trades', [])

        if not losing_trades:
            result = {
                'status': 'success',
                'message': 'No losing trades to analyze',
                'patterns_identified': [],
                'rules_updated': [],
                'bad_patterns_added': [],
                'run_id': run_id,
                'date': target_date,
                'analysis_range': analysis_range,
                'daily_summary': daily_summary
            }
            duration_ms = int((time.time() - start_time) * 1000)
            _publish_metrics([
                {'name': 'Agent4Invocations', 'value': 1},
                {'name': 'Agent4Success', 'value': 1},
                {'name': 'Agent4DurationMs', 'value': duration_ms, 'unit': 'Milliseconds'},
                {'name': 'Agent4UpdatesWritten', 'value': 0}
            ])
            _log_event('LEARNING_AGENT_END', {**run_metadata, 'status': 'SKIPPED', 'duration_ms': duration_ms,
                                              'updates_written': 0})
            return result

        patterns = _identify_loss_patterns(losing_trades)
        current_rules = _load_strategy_rules()
        bad_patterns = _load_bad_patterns()

        rules_updated, new_bad_patterns = _generate_rule_updates(
            patterns, current_rules, bad_patterns
        )

        artifacts_written = 0
        if rules_updated and not dry_run:
            updated_rules = _apply_rule_updates(current_rules, rules_updated)
            _save_strategy_rules(updated_rules)

        if new_bad_patterns and not dry_run:
            bad_patterns.extend(new_bad_patterns)
            _save_bad_patterns(bad_patterns)

        summary = _generate_analysis_summary(
            losing_trades, patterns, rules_updated, new_bad_patterns, analysis_range
        )

        result_payload = {
            'status': 'success',
            'patterns_identified': patterns,
            'rules_updated': rules_updated,
            'bad_patterns_added': new_bad_patterns,
            'summary': summary,
            'run_id': run_id,
            'date': target_date,
            'analysis_range': analysis_range,
            'analysis_type': analysis_type,
            'dry_run': dry_run,
            'daily_summary': daily_summary,
            'analyzed_at': datetime.now(timezone.utc).isoformat()
        }

        artifacts_written = _persist_learning_outputs(
            target_date=target_date,
            run_id=run_id,
            mode=mode,
            source=source,
            schedule_type=schedule_type,
            daily_summary=daily_summary,
            learning_payload=result_payload,
            dry_run=dry_run
        )

        duration_ms = int((time.time() - start_time) * 1000)
        _publish_metrics([
            {'name': 'Agent4Invocations', 'value': 1},
            {'name': 'Agent4Success', 'value': 1},
            {'name': 'Agent4DurationMs', 'value': duration_ms, 'unit': 'Milliseconds'},
            {'name': 'Agent4UpdatesWritten', 'value': artifacts_written}
        ])
        _log_event('LEARNING_AGENT_END', {**run_metadata, 'status': 'SUCCESS', 'duration_ms': duration_ms,
                                          'updates_written': artifacts_written})

        return result_payload

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        _publish_metrics([
            {'name': 'Agent4Invocations', 'value': 1},
            {'name': 'Agent4Success', 'value': 0},
            {'name': 'Agent4DurationMs', 'value': duration_ms, 'unit': 'Milliseconds'},
            {'name': 'Agent4UpdatesWritten', 'value': 0}
        ])
        _log_event('LEARNING_AGENT_END', {**run_metadata, 'status': 'FAILED', 'error': str(e),
                                          'duration_ms': duration_ms})
        logger.error(f"Learning analysis failed: {str(e)}")
        metrics.add_metric(name="ProcessingErrors", unit=MetricUnit.Count, value=1)
        return {
            'status': 'error',
            'message': str(e),
            'patterns_identified': [],
            'rules_updated': [],
            'bad_patterns_added': [],
            'run_id': run_id,
            'date': target_date
        }



def _fetch_losing_trades(date_range: Dict) -> List[Dict]:
    """Fetch losing trades from S3 structured data."""
    losing_trades = []
    
    start_date = datetime.strptime(date_range['start'], '%Y-%m-%d')
    end_date = datetime.strptime(date_range['end'], '%Y-%m-%d')
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        try:
            # Try to read structured trades for this date
            response = s3_client.get_object(
                Bucket=S3_BUCKET,
                Key=f'structured/{date_str}/trades.json'
            )
            trades = json.loads(response['Body'].read().decode('utf-8'))
            
            # Filter for losses
            for trade in trades:
                if _is_losing_trade(trade):
                    trade['date'] = date_str
                    losing_trades.append(trade)
                    
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                logger.warning(f"Error reading trades for {date_str}: {e}")
        
        current_date += timedelta(days=1)
    
    return losing_trades


def _identify_loss_patterns(trades: List[Dict]) -> List[Dict]:
    """Identify common patterns in losing trades."""
    patterns = []
    
    # Pattern 1: Regime + Volatility + Action combinations
    regime_vol_patterns = _analyze_regime_volatility_patterns(trades)
    patterns.extend(regime_vol_patterns)
    
    # Pattern 2: Time of day patterns
    time_patterns = _analyze_time_patterns(trades)
    patterns.extend(time_patterns)
    
    # Pattern 3: Indicator value patterns (RSI, etc.)
    indicator_patterns = _analyze_indicator_patterns(trades)
    patterns.extend(indicator_patterns)
    
    # Pattern 4: PDH/PDL proximity patterns
    level_patterns = _analyze_level_patterns(trades)
    patterns.extend(level_patterns)
    
    # Pattern 5: Confidence level patterns
    confidence_patterns = _analyze_confidence_patterns(trades)
    patterns.extend(confidence_patterns)
    
    # Filter patterns that meet minimum occurrence threshold
    significant_patterns = [
        p for p in patterns 
        if p.get('occurrences', 0) >= MIN_PATTERN_OCCURRENCES
    ]
    
    # Sort by impact (average loss * occurrences)
    significant_patterns.sort(
        key=lambda p: abs(p.get('avg_loss', 0)) * p.get('occurrences', 0),
        reverse=True
    )
    
    return significant_patterns[:10]  # Top 10 patterns


def _analyze_regime_volatility_patterns(trades: List[Dict]) -> List[Dict]:
    """Analyze regime + volatility + action patterns."""
    combos = defaultdict(lambda: {'trades': [], 'total_loss': 0})
    
    for trade in trades:
        key = (
            trade.get('regime', 'UNKNOWN'),
            trade.get('volatility', 'MED'),
            trade.get('action', 'UNKNOWN')
        )
        combos[key]['trades'].append(trade)
        combos[key]['total_loss'] += trade.get('pnl', 0)
    
    patterns = []
    for (regime, volatility, action), data in combos.items():
        if len(data['trades']) >= 2:
            patterns.append({
                'type': 'REGIME_VOLATILITY_ACTION',
                'pattern': {
                    'regime': regime,
                    'volatility': volatility,
                    'action': action
                },
                'occurrences': len(data['trades']),
                'total_loss': data['total_loss'],
                'avg_loss': data['total_loss'] / len(data['trades']),
                'description': f"Losing {action} trades in {regime} regime with {volatility} volatility",
                'recommendation': f"Avoid {action} in {regime}/{volatility} conditions"
            })
    
    return patterns


def _analyze_time_patterns(trades: List[Dict]) -> List[Dict]:
    """Analyze time-of-day patterns."""
    time_groups = defaultdict(lambda: {'trades': [], 'total_loss': 0})
    
    for trade in trades:
        time_of_day = trade.get('time_of_day', 'UNKNOWN')
        time_groups[time_of_day]['trades'].append(trade)
        time_groups[time_of_day]['total_loss'] += trade.get('pnl', 0)
    
    patterns = []
    for time_of_day, data in time_groups.items():
        if len(data['trades']) >= 2:
            patterns.append({
                'type': 'TIME_OF_DAY',
                'pattern': {'time_of_day': time_of_day},
                'occurrences': len(data['trades']),
                'total_loss': data['total_loss'],
                'avg_loss': data['total_loss'] / len(data['trades']),
                'description': f"Losing trades during {time_of_day}",
                'recommendation': f"Reduce trading during {time_of_day}"
            })
    
    return patterns


def _analyze_indicator_patterns(trades: List[Dict]) -> List[Dict]:
    """Analyze indicator value patterns (RSI extremes, etc.)."""
    patterns = []
    
    # RSI extremes
    rsi_extreme_trades = [
        t for t in trades 
        if t.get('rsi', 50) < 30 or t.get('rsi', 50) > 70
    ]
    
    if len(rsi_extreme_trades) >= 2:
        oversold = [t for t in rsi_extreme_trades if t.get('rsi', 50) < 30]
        overbought = [t for t in rsi_extreme_trades if t.get('rsi', 50) > 70]
        
        if oversold:
            patterns.append({
                'type': 'RSI_EXTREME',
                'pattern': {'rsi_condition': 'OVERSOLD', 'threshold': 30},
                'occurrences': len(oversold),
                'total_loss': sum(t.get('pnl', 0) for t in oversold),
                'avg_loss': sum(t.get('pnl', 0) for t in oversold) / len(oversold),
                'description': 'Losses when RSI < 30 (oversold)',
                'recommendation': 'Wait for RSI to recover above 35 before entering'
            })
        
        if overbought:
            patterns.append({
                'type': 'RSI_EXTREME',
                'pattern': {'rsi_condition': 'OVERBOUGHT', 'threshold': 70},
                'occurrences': len(overbought),
                'total_loss': sum(t.get('pnl', 0) for t in overbought),
                'avg_loss': sum(t.get('pnl', 0) for t in overbought) / len(overbought),
                'description': 'Losses when RSI > 70 (overbought)',
                'recommendation': 'Wait for RSI to pull back below 65 before entering'
            })
    
    return patterns


def _analyze_level_patterns(trades: List[Dict]) -> List[Dict]:
    """Analyze PDH/PDL proximity patterns."""
    patterns = []
    
    # Near PDH losses
    near_pdh = [
        t for t in trades 
        if abs(t.get('PDH_delta', 1)) < 0.2  # Within 0.2%
    ]
    
    if len(near_pdh) >= 2:
        patterns.append({
            'type': 'LEVEL_PROXIMITY',
            'pattern': {'level': 'PDH', 'proximity': 0.2},
            'occurrences': len(near_pdh),
            'total_loss': sum(t.get('pnl', 0) for t in near_pdh),
            'avg_loss': sum(t.get('pnl', 0) for t in near_pdh) / len(near_pdh),
            'description': 'Losses when entering near Previous Day High',
            'recommendation': 'Avoid entries within 0.2% of PDH'
        })
    
    # Near PDL losses
    near_pdl = [
        t for t in trades 
        if abs(t.get('PDL_delta', 1)) < 0.2
    ]
    
    if len(near_pdl) >= 2:
        patterns.append({
            'type': 'LEVEL_PROXIMITY',
            'pattern': {'level': 'PDL', 'proximity': 0.2},
            'occurrences': len(near_pdl),
            'total_loss': sum(t.get('pnl', 0) for t in near_pdl),
            'avg_loss': sum(t.get('pnl', 0) for t in near_pdl) / len(near_pdl),
            'description': 'Losses when entering near Previous Day Low',
            'recommendation': 'Avoid entries within 0.2% of PDL'
        })
    
    return patterns


def _analyze_confidence_patterns(trades: List[Dict]) -> List[Dict]:
    """Analyze confidence level patterns."""
    patterns = []
    
    low_confidence = [t for t in trades if t.get('confidence', 0) < 0.6]
    
    if len(low_confidence) >= 2:
        patterns.append({
            'type': 'LOW_CONFIDENCE',
            'pattern': {'confidence_threshold': 0.6},
            'occurrences': len(low_confidence),
            'total_loss': sum(t.get('pnl', 0) for t in low_confidence),
            'avg_loss': sum(t.get('pnl', 0) for t in low_confidence) / len(low_confidence),
            'description': 'Losses on low-confidence trades (< 60%)',
            'recommendation': 'Increase minimum confidence threshold to 0.65'
        })
    
    return patterns


def _load_strategy_rules() -> Dict:
    """Load current strategy rules from S3."""
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key='strategy/rules.json'
        )
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return _default_strategy_rules()
        raise


def _default_strategy_rules() -> Dict:
    """Return default strategy rules."""
    return {
        'version': '1.0',
        'last_updated': datetime.now(timezone.utc).isoformat(),
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
    }


def _load_bad_patterns() -> List[Dict]:
    """Load bad patterns to avoid from S3."""
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key='bad_patterns/patterns.json'
        )
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return []
        raise


def _save_strategy_rules(rules: Dict) -> None:
    """Save strategy rules to S3."""
    rules['last_updated'] = datetime.now(timezone.utc).isoformat()
    
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key='strategy/rules.json',
        Body=json.dumps(rules, indent=2).encode('utf-8'),
        ContentType='application/json'
    )
    logger.info("Saved updated strategy rules")


def _save_bad_patterns(patterns: List[Dict]) -> None:
    """Save bad patterns to S3."""
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key='bad_patterns/patterns.json',
        Body=json.dumps(patterns, indent=2).encode('utf-8'),
        ContentType='application/json'
    )
    logger.info(f"Saved {len(patterns)} bad patterns")


def _generate_rule_updates(
    patterns: List[Dict],
    current_rules: Dict,
    existing_bad_patterns: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """Generate rule updates based on identified patterns."""
    rule_updates = []
    new_bad_patterns = []
    
    for pattern in patterns:
        pattern_type = pattern.get('type')
        
        # Skip if already in bad patterns
        if _pattern_exists(pattern, existing_bad_patterns):
            continue
        
        if pattern_type == 'REGIME_VOLATILITY_ACTION':
            # Add blocked condition
            condition = pattern['pattern']
            rule_updates.append({
                'rule_type': 'blocked_condition',
                'action': 'add',
                'value': condition,
                'reason': pattern['description']
            })
            
        elif pattern_type == 'TIME_OF_DAY':
            # Add time restriction
            time_of_day = pattern['pattern']['time_of_day']
            rule_updates.append({
                'rule_type': 'time_restriction',
                'action': 'add',
                'value': {time_of_day: {'allowed': False}},
                'reason': pattern['description']
            })
            
        elif pattern_type == 'LOW_CONFIDENCE':
            # Increase minimum confidence
            current_min = current_rules.get('min_confidence', 0.60)
            new_min = min(current_min + 0.05, 0.80)
            rule_updates.append({
                'rule_type': 'min_confidence',
                'action': 'update',
                'value': new_min,
                'reason': pattern['description']
            })
        
        # Add to bad patterns
        new_bad_patterns.append({
            'pattern': pattern['pattern'],
            'type': pattern_type,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'occurrences': pattern['occurrences'],
            'avg_loss': pattern['avg_loss'],
            'description': pattern['description']
        })
    
    return rule_updates, new_bad_patterns


def _pattern_exists(pattern: Dict, existing: List[Dict]) -> bool:
    """Check if pattern already exists in bad patterns."""
    for existing_pattern in existing:
        if (existing_pattern.get('type') == pattern.get('type') and 
            existing_pattern.get('pattern') == pattern.get('pattern')):
            return True
    return False


def _apply_rule_updates(rules: Dict, updates: List[Dict]) -> Dict:
    """Apply rule updates to strategy rules."""
    for update in updates:
        rule_type = update['rule_type']
        action = update['action']
        value = update['value']
        
        if rule_type == 'blocked_condition' and action == 'add':
            if 'blocked_conditions' not in rules:
                rules['blocked_conditions'] = []
            if value not in rules['blocked_conditions']:
                rules['blocked_conditions'].append(value)
                
        elif rule_type == 'time_restriction' and action == 'add':
            if 'time_restrictions' not in rules:
                rules['time_restrictions'] = {}
            rules['time_restrictions'].update(value)
            
        elif rule_type == 'min_confidence' and action == 'update':
            rules['min_confidence'] = value
    
    return rules


def _generate_analysis_summary(
    trades: List[Dict],
    patterns: List[Dict],
    rules_updated: List[Dict],
    bad_patterns: List[Dict],
    analysis_range: Optional[Dict[str, str]] = None
) -> Dict:
    """Generate summary of the analysis."""
    return {
        'trades_analyzed': len(trades),
        'total_loss': sum(t.get('pnl', 0) for t in trades),
        'patterns_found': len(patterns),
        'rules_updated_count': len(rules_updated),
        'new_bad_patterns_count': len(bad_patterns),
        'top_pattern': patterns[0] if patterns else None,
        'analysis_window': analysis_range,
        'recommendations': [
            p.get('recommendation') for p in patterns[:5]
        ]
    }
