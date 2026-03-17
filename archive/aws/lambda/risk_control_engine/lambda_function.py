"""
Lambda Function: Risk Control Engine
Agent 3 - Risk & Position Sizing Agent

This function evaluates if a trade is allowed based on:
- Volatility conditions
- Losing streak analysis
- Recent P&L drawdown
- Current exposure
- Daily/weekly limits

Returns allowed_to_trade flag and size_multiplier.
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit
from pydantic import BaseModel, ValidationError

# Initialize AWS clients
s3_client = boto3.client('s3')

# Powertools
logger = Logger()
metrics = Metrics(namespace="TradingBot")
tracer = Tracer()

# Environment variables
S3_BUCKET = os.environ.get('S3_BUCKET', 'trading-bot-data')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'prod')
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# Risk configuration from environment (with defaults)
MAX_DAILY_LOSS = float(os.environ.get('MAX_DAILY_LOSS', '1500'))
MAX_POSITION_SIZE = int(os.environ.get('MAX_POSITION_SIZE', '5'))
MAX_LOSING_STREAK = int(os.environ.get('MAX_LOSING_STREAK', '3'))
MAX_DRAWDOWN_PCT = float(os.environ.get('MAX_DRAWDOWN_PCT', '5.0'))
MAX_DAILY_TRADES = int(os.environ.get('MAX_DAILY_TRADES', '20'))


class TradeDecision(BaseModel):
    action: str
    confidence: float
    symbol: str = "ES"
    proposed_size: int = 0


class AccountMetrics(BaseModel):
    current_pnl_today: float
    account_balance: float
    losing_streak: int = 0
    trades_today: int = 0
    current_position: int = 0
    open_risk: float = 0


class MarketConditions(BaseModel):
    volatility: str
    regime: str
    atr: float
    vix: float


class TradeRequest(BaseModel):
    trade_decision: TradeDecision
    account_metrics: AccountMetrics
    market_conditions: MarketConditions


@tracer.capture_lambda_handler
@logger.inject_lambda_context
@metrics.log_metrics
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Evaluate trade risk and determine if trading is allowed.
    
    This function is invoked by Bedrock Agent as an action group.
    """
    start_time = time.time()
    logger.info(f"Received event: {json.dumps(event, default=str)[:1000]}")
    metrics.add_metric(name="RequestsReceived", unit=MetricUnit.Count, value=1)
    
    # Handle Bedrock Agent invocation format
    if 'actionGroup' in event:
        response = _handle_bedrock_agent_request(event, context)
        duration_ms = int((time.time() - start_time) * 1000)
        metrics.add_metric(name="ResponseTime", unit=MetricUnit.Milliseconds, value=duration_ms)
        metrics.add_metric(name="CacheHits", unit=MetricUnit.Count, value=1 if event.get('cached') else 0)
        metrics.add_metric(name="ConfidenceScore", unit=MetricUnit.Percent, value=float(event.get('confidence', 0)) * 100)
        return response
    
    try:
        request = TradeRequest(**event)
    except ValidationError as exc:
        logger.error(f"Input validation failed: {exc}")
        metrics.add_metric(name="ValidationErrors", unit=MetricUnit.Count, value=1)
        raise

    # Handle direct invocation (for testing) with validated payload
    merged_event = {**event, **request.dict()}
    response = _handle_direct_request(merged_event, context)
    duration_ms = int((time.time() - start_time) * 1000)
    metrics.add_metric(name="ResponseTime", unit=MetricUnit.Milliseconds, value=duration_ms)
    metrics.add_metric(name="CacheHits", unit=MetricUnit.Count, value=1 if event.get('cached') else 0)
    confidence = merged_event.get('trade_decision', {}).get('confidence', 0.0)
    metrics.add_metric(name="ConfidenceScore", unit=MetricUnit.Percent, value=float(confidence) * 100)
    return response


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
        # Extract parameters
        action = params.get('action', 'BUY')
        entry_price = float(params.get('entry_price', 0))
        stop_loss = float(params.get('stop_loss', 0))
        account_balance = float(params.get('account_balance', 50000))
        
        # Calculate risk
        if entry_price > 0 and stop_loss > 0:
            risk_per_contract = abs(entry_price - stop_loss) * 50  # ES multiplier
            risk_pct = (risk_per_contract / account_balance) * 100
            max_contracts = int((account_balance * 0.02) / risk_per_contract) if risk_per_contract > 0 else 1
        else:
            risk_pct = 0
            max_contracts = 1
        
        # Determine approval
        approved = risk_pct <= 2.0  # 2% max risk per trade
        
        result = {
            'approved': approved,
            'position_size': max(1, min(max_contracts, 3)),
            'risk_percentage': round(risk_pct, 2),
            'concerns': [] if approved else [f'Risk {risk_pct:.1f}% exceeds 2% limit'],
            'recommendation': f"{'Approved' if approved else 'Rejected'}: {action} with {max_contracts} contracts at {risk_pct:.1f}% risk"
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
        logger.error(f"Bedrock action error: {str(e)}")
        metrics.add_metric(name="ProcessingErrors", unit=MetricUnit.Count, value=1)
        return {
            'messageVersion': '1.0',
            'response': {
                'actionGroup': action_group,
                'function': function_name,
                'functionResponse': {
                    'responseBody': {
                        'TEXT': {
                            'body': json.dumps({'error': str(e), 'approved': False})
                        }
                    }
                }
            }
        }


def _handle_direct_request(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle direct Lambda invocation.
    
    Input Event:
    {
        "trade_decision": {
            "action": "BUY",
            "confidence": 0.78,
            "symbol": "ES",
            "proposed_size": 1
        },
        "account_metrics": {
            "current_pnl_today": -250.50,
            "current_position": 2,
            "losing_streak": 1,
            "trades_today": 5,
            "account_balance": 50000,
            "open_risk": 500
        },
        "market_conditions": {
            "volatility": "HIGH",
            "regime": "DOWNTREND",
            "atr": 2.5,
            "vix": 22.5
        }
    }
    """
    logger.info(f"Direct invocation: {json.dumps(event, default=str)[:500]}")
    
    try:
        trade_decision = event.get('trade_decision', {})
        account_metrics = event.get('account_metrics', {})
        market_conditions = event.get('market_conditions', {})
        
        # Validate inputs
        if not trade_decision or not trade_decision.get('action'):
            return _create_response(
                allowed=False,
                multiplier=0,
                size=0,
                flags=['MISSING_TRADE_DECISION'],
                reason='No trade decision provided'
            )
        
        # Load current risk state from S3
        risk_state = _load_risk_state()
        
        # Update risk state with account metrics
        risk_state = _update_risk_state(risk_state, account_metrics)
        
        # Run all risk checks
        risk_flags = []
        size_multiplier = 1.0
        
        # Check 1: Daily P&L limit
        pnl_check = _check_daily_pnl_limit(account_metrics)
        if not pnl_check['passed']:
            risk_flags.append(pnl_check['flag'])
            if pnl_check['hard_stop']:
                return _create_response(
                    allowed=False,
                    multiplier=0,
                    size=0,
                    flags=risk_flags,
                    reason=pnl_check['reason']
                )
            size_multiplier *= pnl_check['multiplier']
        
        # Check 2: Losing streak
        streak_check = _check_losing_streak(account_metrics)
        if not streak_check['passed']:
            risk_flags.append(streak_check['flag'])
            if streak_check['hard_stop']:
                return _create_response(
                    allowed=False,
                    multiplier=0,
                    size=0,
                    flags=risk_flags,
                    reason=streak_check['reason']
                )
            size_multiplier *= streak_check['multiplier']
        
        # Check 3: Daily trade count
        trades_check = _check_daily_trades(account_metrics)
        if not trades_check['passed']:
            risk_flags.append(trades_check['flag'])
            if trades_check['hard_stop']:
                return _create_response(
                    allowed=False,
                    multiplier=0,
                    size=0,
                    flags=risk_flags,
                    reason=trades_check['reason']
                )
        
        # Check 4: Position exposure
        exposure_check = _check_position_exposure(account_metrics, trade_decision)
        if not exposure_check['passed']:
            risk_flags.append(exposure_check['flag'])
            if exposure_check['hard_stop']:
                return _create_response(
                    allowed=False,
                    multiplier=0,
                    size=0,
                    flags=risk_flags,
                    reason=exposure_check['reason']
                )
            size_multiplier *= exposure_check['multiplier']
        
        # Check 5: Volatility conditions
        vol_check = _check_volatility(market_conditions)
        if not vol_check['passed']:
            risk_flags.append(vol_check['flag'])
            size_multiplier *= vol_check['multiplier']
        
        # Check 6: Confidence threshold
        confidence_check = _check_confidence(trade_decision)
        if not confidence_check['passed']:
            risk_flags.append(confidence_check['flag'])
            size_multiplier *= confidence_check['multiplier']
        
        # Calculate final position size
        proposed_size = trade_decision.get('proposed_size', 1)
        adjusted_size = max(1, int(proposed_size * size_multiplier))
        adjusted_size = min(adjusted_size, MAX_POSITION_SIZE - account_metrics.get('current_position', 0))
        
        # Determine if trading is allowed
        allowed = size_multiplier > 0.3 and adjusted_size > 0
        
        # Generate reason
        if allowed:
            if risk_flags:
                reason = f"Trade allowed with caution. Risk factors: {', '.join(risk_flags)}. Size reduced to {adjusted_size}."
            else:
                reason = "Trade approved. All risk checks passed."
        else:
            reason = f"Trade blocked. Risk factors: {', '.join(risk_flags) if risk_flags else 'Size reduced to zero'}."
        
        # Save updated risk state
        _save_risk_state(risk_state)
        
        return _create_response(
            allowed=allowed,
            multiplier=size_multiplier,
            size=adjusted_size,
            flags=risk_flags,
            reason=reason
        )
        
    except Exception as e:
        logger.error(f"Risk evaluation failed: {str(e)}")
        metrics.add_metric(name="ProcessingErrors", unit=MetricUnit.Count, value=1)
        return _create_response(
            allowed=False,
            multiplier=0,
            size=0,
            flags=['SYSTEM_ERROR'],
            reason=f'Risk system error: {str(e)}. Trading blocked for safety.'
        )


def _create_response(
    allowed: bool,
    multiplier: float,
    size: int,
    flags: List[str],
    reason: str
) -> Dict[str, Any]:
    """Create standardized response."""
    return {
        'status': 'success',
        'allowed_to_trade': allowed,
        'size_multiplier': round(multiplier, 2),
        'adjusted_size': size,
        'risk_flags': flags,
        'reason': reason,
        'evaluated_at': datetime.now(timezone.utc).isoformat()
    }


def _load_risk_state() -> Dict:
    """Load current risk state from S3."""
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key='pnl/risk_state.json'
        )
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return _default_risk_state()
        raise


def _save_risk_state(state: Dict) -> None:
    """Save risk state to S3."""
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key='pnl/risk_state.json',
            Body=json.dumps(state, default=str).encode('utf-8'),
            ContentType='application/json'
        )
    except ClientError as e:
        logger.warning(f"Failed to save risk state: {e}")


def _default_risk_state() -> Dict:
    """Return default risk state."""
    return {
        'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'daily_pnl': 0,
        'daily_trades': 0,
        'losing_streak': 0,
        'winning_streak': 0,
        'peak_balance': 0,
        'drawdown': 0,
        'last_trade_time': None,
        'trade_history': []
    }


def _update_risk_state(state: Dict, metrics: Dict) -> Dict:
    """Update risk state with current metrics."""
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    # Reset if new day
    if state.get('date') != today:
        state = _default_risk_state()
        state['date'] = today
    
    # Update from account metrics
    state['daily_pnl'] = metrics.get('current_pnl_today', state.get('daily_pnl', 0))
    state['daily_trades'] = metrics.get('trades_today', state.get('daily_trades', 0))
    state['losing_streak'] = metrics.get('losing_streak', state.get('losing_streak', 0))
    
    return state


def _check_daily_pnl_limit(metrics: Dict) -> Dict:
    """Check if daily P&L limit has been reached."""
    current_pnl = metrics.get('current_pnl_today', 0)
    
    if current_pnl <= -MAX_DAILY_LOSS:
        return {
            'passed': False,
            'hard_stop': True,
            'flag': 'DAILY_LOSS_LIMIT_REACHED',
            'reason': f'Daily loss limit reached (${current_pnl:.2f} <= -${MAX_DAILY_LOSS:.2f}). Trading halted.',
            'multiplier': 0
        }
    
    # Reduce size if approaching limit
    loss_pct = abs(current_pnl) / MAX_DAILY_LOSS if current_pnl < 0 else 0
    
    if loss_pct > 0.8:
        return {
            'passed': False,
            'hard_stop': False,
            'flag': 'APPROACHING_DAILY_LIMIT',
            'reason': f'Approaching daily loss limit ({loss_pct*100:.0f}%)',
            'multiplier': 0.5
        }
    elif loss_pct > 0.5:
        return {
            'passed': False,
            'hard_stop': False,
            'flag': 'PARTIAL_DAILY_LOSS',
            'reason': f'Partial daily loss ({loss_pct*100:.0f}% of limit)',
            'multiplier': 0.7
        }
    
    return {'passed': True, 'hard_stop': False, 'flag': None, 'reason': None, 'multiplier': 1.0}


def _check_losing_streak(metrics: Dict) -> Dict:
    """Check losing streak status."""
    streak = metrics.get('losing_streak', 0)
    
    if streak >= MAX_LOSING_STREAK:
        return {
            'passed': False,
            'hard_stop': True,
            'flag': 'MAX_LOSING_STREAK',
            'reason': f'Maximum losing streak reached ({streak} consecutive losses). Take a break.',
            'multiplier': 0
        }
    
    if streak >= 2:
        return {
            'passed': False,
            'hard_stop': False,
            'flag': 'LOSING_STREAK_WARNING',
            'reason': f'On {streak} losing streak',
            'multiplier': 0.8 - (streak * 0.1)
        }
    
    return {'passed': True, 'hard_stop': False, 'flag': None, 'reason': None, 'multiplier': 1.0}


def _check_daily_trades(metrics: Dict) -> Dict:
    """Check daily trade count."""
    trades_today = metrics.get('trades_today', 0)
    
    if trades_today >= MAX_DAILY_TRADES:
        return {
            'passed': False,
            'hard_stop': True,
            'flag': 'MAX_DAILY_TRADES',
            'reason': f'Maximum daily trades reached ({trades_today}/{MAX_DAILY_TRADES})',
            'multiplier': 0
        }
    
    if trades_today >= MAX_DAILY_TRADES * 0.8:
        return {
            'passed': False,
            'hard_stop': False,
            'flag': 'APPROACHING_MAX_TRADES',
            'reason': f'Approaching daily trade limit ({trades_today}/{MAX_DAILY_TRADES})',
            'multiplier': 0.9
        }
    
    return {'passed': True, 'hard_stop': False, 'flag': None, 'reason': None, 'multiplier': 1.0}


def _check_position_exposure(metrics: Dict, trade: Dict) -> Dict:
    """Check current position exposure."""
    current_position = metrics.get('current_position', 0)
    proposed_size = trade.get('proposed_size', 1)
    open_risk = metrics.get('open_risk', 0)
    account_balance = metrics.get('account_balance', 50000)
    
    # Check if adding to position would exceed limit
    new_position = current_position + proposed_size
    if new_position > MAX_POSITION_SIZE:
        return {
            'passed': False,
            'hard_stop': True,
            'flag': 'MAX_POSITION_SIZE',
            'reason': f'Would exceed max position size ({new_position} > {MAX_POSITION_SIZE})',
            'multiplier': 0
        }
    
    # Check open risk as % of account
    risk_pct = (open_risk / account_balance) * 100 if account_balance > 0 else 0
    if risk_pct > MAX_DRAWDOWN_PCT:
        return {
            'passed': False,
            'hard_stop': False,
            'flag': 'HIGH_OPEN_RISK',
            'reason': f'High open risk ({risk_pct:.1f}% of account)',
            'multiplier': 0.6
        }
    
    return {'passed': True, 'hard_stop': False, 'flag': None, 'reason': None, 'multiplier': 1.0}


def _check_volatility(conditions: Dict) -> Dict:
    """Check market volatility conditions."""
    volatility = conditions.get('volatility', 'MED')
    atr = conditions.get('atr', 1.0)
    vix = conditions.get('vix', 15)
    
    # High volatility = reduce size
    if volatility == 'HIGH' or vix > 25 or atr > 3.0:
        return {
            'passed': False,
            'hard_stop': False,
            'flag': 'HIGH_VOLATILITY',
            'reason': f'High volatility conditions (VIX: {vix}, ATR: {atr})',
            'multiplier': 0.6
        }
    
    # Very high volatility
    if vix > 35 or atr > 5.0:
        return {
            'passed': False,
            'hard_stop': False,
            'flag': 'EXTREME_VOLATILITY',
            'reason': f'Extreme volatility (VIX: {vix}, ATR: {atr}). Reduce exposure.',
            'multiplier': 0.3
        }
    
    return {'passed': True, 'hard_stop': False, 'flag': None, 'reason': None, 'multiplier': 1.0}


def _check_confidence(trade: Dict) -> Dict:
    """Check trade confidence level."""
    confidence = trade.get('confidence', 0)
    
    if confidence < 0.55:
        return {
            'passed': False,
            'hard_stop': False,
            'flag': 'LOW_CONFIDENCE',
            'reason': f'Low trade confidence ({confidence*100:.0f}%)',
            'multiplier': 0.5
        }
    
    if confidence < 0.65:
        return {
            'passed': False,
            'hard_stop': False,
            'flag': 'MODERATE_CONFIDENCE',
            'reason': f'Moderate trade confidence ({confidence*100:.0f}%)',
            'multiplier': 0.8
        }
    
    return {'passed': True, 'hard_stop': False, 'flag': None, 'reason': None, 'multiplier': 1.0}
