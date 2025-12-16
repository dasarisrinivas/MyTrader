"""
Lambda Function: Calculate Winrate Statistics
Agent 2 - RAG + Similarity Search Decision Engine

This function calculates win-rate percentage, best action, and PnL statistics
from a list of similar historical trades retrieved via RAG.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from statistics import mean, stdev

import boto3
from botocore.exceptions import ClientError

# Initialize AWS clients
s3_client = boto3.client('s3')

# Environment variables
S3_BUCKET = os.environ.get('S3_BUCKET', 'trading-bot-data')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'prod')
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# Configuration
MIN_SAMPLES_FOR_CONFIDENCE = 10
HIGH_CONFIDENCE_THRESHOLD = 0.70
MEDIUM_CONFIDENCE_THRESHOLD = 0.55


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Calculate win-rate and PnL statistics from similar trades.
    
    This function is invoked by Bedrock Agent as an action group.
    The event format from Bedrock Agent is different from direct invocation.
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
    
    # Process the request
    try:
        # For now, return a helpful response since we don't have actual trade data
        result = {
            'decision': 'WAIT',
            'confidence': 0.5,
            'reason': f'Analysis requested for pattern_type={params.get("pattern_type", "unknown")}. No historical data in knowledge base yet.',
            'statistics': {
                'total_matches': 0,
                'win_rate': 0,
                'message': 'Upload historical trade data to S3 structured/ folder for RAG analysis'
            }
        }
        
        response_body = {
            'TEXT': {
                'body': json.dumps(result)
            }
        }
        
        return {
            'messageVersion': '1.0',
            'response': {
                'actionGroup': action_group,
                'function': function_name,
                'functionResponse': {
                    'responseBody': response_body
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
                            'body': json.dumps({'error': str(e), 'decision': 'WAIT'})
                        }
                    }
                }
            }
        }


def _handle_direct_request(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle direct Lambda invocation (for testing or Step Functions).
    
    Input Event:
    {
        "similar_trades": [
            {
                "trade_id": "abc123",
                "action": "BUY",
                "outcome": "WIN",
                "pnl": 125.50,
                "confidence": 0.75,
                "regime": "UPTREND",
                "volatility": "LOW",
                "similarity_score": 0.92
            },
            ...
        ],
        "current_context": {
            "trend": "UPTREND",
            "volatility": "LOW",
            "PDH_delta": 0.23,
            "PDL_delta": 1.03,
            "rsi": 45.2,
            "atr": 1.5
        }
    }
    """
    try:
        similar_trades = event.get('similar_trades', [])
        current_context = event.get('current_context', {})
        
        if not similar_trades:
            return {
                'status': 'success',
                'decision': 'WAIT',
                'confidence': 0.0,
                'reason': 'No similar historical patterns found. Insufficient data for decision.',
                'statistics': _empty_statistics(),
                'stop_loss': None,
                'take_profit': None
            }
        
        # Calculate comprehensive statistics
        statistics = _calculate_statistics(similar_trades)
        
        # Determine best action based on analysis
        decision, confidence, reason = _make_decision(statistics, current_context)
        
        # Calculate stop-loss and take-profit recommendations
        stop_loss, take_profit = _calculate_exit_levels(
            statistics, current_context, decision
        )
        
        return {
            'status': 'success',
            'decision': decision,
            'confidence': confidence,
            'reason': reason,
            'statistics': statistics,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'metadata': {
                'processed_at': datetime.now(timezone.utc).isoformat(),
                'sample_size': len(similar_trades),
                'context': current_context
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to calculate statistics: {str(e)}")
        return {
            'status': 'error',
            'decision': 'WAIT',
            'confidence': 0.0,
            'reason': f'Error calculating statistics: {str(e)}',
            'statistics': _empty_statistics(),
            'stop_loss': None,
            'take_profit': None
        }


def _calculate_statistics(trades: List[Dict]) -> Dict[str, Any]:
    """Calculate comprehensive trading statistics."""
    total = len(trades)
    
    # Separate by outcome
    wins = [t for t in trades if t.get('outcome') == 'WIN']
    losses = [t for t in trades if t.get('outcome') == 'LOSS']
    
    # Separate by action
    buys = [t for t in trades if t.get('action') == 'BUY']
    sells = [t for t in trades if t.get('action') == 'SELL']
    
    # Calculate P&L metrics
    all_pnl = [t.get('pnl', 0) for t in trades]
    win_pnl = [t.get('pnl', 0) for t in wins]
    loss_pnl = [t.get('pnl', 0) for t in losses]
    
    # Buy statistics
    buy_wins = [t for t in buys if t.get('outcome') == 'WIN']
    buy_pnl = [t.get('pnl', 0) for t in buys]
    
    # Sell statistics  
    sell_wins = [t for t in sells if t.get('outcome') == 'WIN']
    sell_pnl = [t.get('pnl', 0) for t in sells]
    
    # Weighted by similarity score
    weighted_scores = []
    for t in trades:
        score = t.get('similarity_score', 0.5)
        outcome = 1 if t.get('outcome') == 'WIN' else 0
        weighted_scores.append(score * outcome)
    
    # Regime breakdown
    regime_stats = _breakdown_by_field(trades, 'regime')
    volatility_stats = _breakdown_by_field(trades, 'volatility')
    
    statistics = {
        'total_matches': total,
        'win_count': len(wins),
        'loss_count': len(losses),
        'win_rate': len(wins) / total if total > 0 else 0,
        
        # P&L metrics
        'total_pnl': sum(all_pnl),
        'avg_pnl': mean(all_pnl) if all_pnl else 0,
        'pnl_std': stdev(all_pnl) if len(all_pnl) > 1 else 0,
        'avg_win_pnl': mean(win_pnl) if win_pnl else 0,
        'avg_loss_pnl': mean(loss_pnl) if loss_pnl else 0,
        'profit_factor': (sum(win_pnl) / abs(sum(loss_pnl))) if loss_pnl and sum(loss_pnl) != 0 else float('inf'),
        
        # Buy analysis
        'buy_count': len(buys),
        'buy_win_rate': len(buy_wins) / len(buys) if buys else 0,
        'buy_avg_pnl': mean(buy_pnl) if buy_pnl else 0,
        
        # Sell analysis
        'sell_count': len(sells),
        'sell_win_rate': len(sell_wins) / len(sells) if sells else 0,
        'sell_avg_pnl': mean(sell_pnl) if sell_pnl else 0,
        
        # Best action determination
        'best_action': _determine_best_action(buys, sells, buy_wins, sell_wins),
        
        # Weighted confidence
        'weighted_win_rate': sum(weighted_scores) / sum(t.get('similarity_score', 0.5) for t in trades) if trades else 0,
        'avg_similarity': mean([t.get('similarity_score', 0.5) for t in trades]) if trades else 0,
        
        # Breakdowns
        'by_regime': regime_stats,
        'by_volatility': volatility_stats,
        
        # Risk metrics
        'max_drawdown': min(loss_pnl) if loss_pnl else 0,
        'best_trade': max(win_pnl) if win_pnl else 0,
        'win_loss_ratio': (mean(win_pnl) / abs(mean(loss_pnl))) if loss_pnl and mean(loss_pnl) != 0 else float('inf'),
    }
    
    return statistics


def _breakdown_by_field(trades: List[Dict], field: str) -> Dict[str, Dict]:
    """Break down statistics by a specific field."""
    groups = {}
    
    for trade in trades:
        value = trade.get(field, 'UNKNOWN')
        if value not in groups:
            groups[value] = {'wins': 0, 'total': 0, 'pnl': []}
        
        groups[value]['total'] += 1
        groups[value]['pnl'].append(trade.get('pnl', 0))
        if trade.get('outcome') == 'WIN':
            groups[value]['wins'] += 1
    
    # Calculate win rates and avg P&L
    result = {}
    for key, data in groups.items():
        result[key] = {
            'count': data['total'],
            'win_rate': data['wins'] / data['total'] if data['total'] > 0 else 0,
            'avg_pnl': mean(data['pnl']) if data['pnl'] else 0
        }
    
    return result


def _determine_best_action(buys: List, sells: List, buy_wins: List, sell_wins: List) -> str:
    """Determine the best action based on historical performance."""
    buy_win_rate = len(buy_wins) / len(buys) if buys else 0
    sell_win_rate = len(sell_wins) / len(sells) if sells else 0
    
    buy_pnl = sum(t.get('pnl', 0) for t in buys)
    sell_pnl = sum(t.get('pnl', 0) for t in sells)
    
    # Combined score: win rate * 0.6 + normalized P&L * 0.4
    max_pnl = max(abs(buy_pnl), abs(sell_pnl), 1)
    
    buy_score = buy_win_rate * 0.6 + (buy_pnl / max_pnl) * 0.4
    sell_score = sell_win_rate * 0.6 + (sell_pnl / max_pnl) * 0.4
    
    if buy_score > sell_score + 0.1:
        return 'BUY'
    elif sell_score > buy_score + 0.1:
        return 'SELL'
    else:
        return 'NEUTRAL'


def _make_decision(statistics: Dict, context: Dict) -> tuple[str, float, str]:
    """Make trading decision based on statistics and context."""
    total_matches = statistics.get('total_matches', 0)
    win_rate = statistics.get('win_rate', 0)
    best_action = statistics.get('best_action', 'NEUTRAL')
    avg_similarity = statistics.get('avg_similarity', 0)
    profit_factor = statistics.get('profit_factor', 0)
    
    # Calculate confidence score
    confidence = _calculate_confidence(statistics)
    
    # Decision logic
    if total_matches < MIN_SAMPLES_FOR_CONFIDENCE:
        return (
            'WAIT',
            confidence * 0.5,
            f"Insufficient historical data. Only {total_matches} similar patterns found (need {MIN_SAMPLES_FOR_CONFIDENCE})."
        )
    
    if win_rate < 0.45:  # Below breakeven after costs
        return (
            'WAIT',
            confidence,
            f"Historical win rate too low ({win_rate*100:.1f}%). Pattern shows negative expectancy."
        )
    
    if confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        return (
            'WAIT',
            confidence,
            f"Confidence too low ({confidence*100:.1f}%). Waiting for clearer setup."
        )
    
    # Make directional decision
    if best_action == 'NEUTRAL':
        return (
            'WAIT',
            confidence,
            f"No clear directional bias. BUY and SELL have similar performance."
        )
    
    regime = context.get('trend', 'UNKNOWN')
    volatility = context.get('volatility', 'MED')
    
    reason = (
        f"Matched {total_matches} similar patterns in {regime} regime with {volatility} volatility. "
        f"Win rate: {win_rate*100:.1f}%, Profit factor: {profit_factor:.2f}. "
        f"Best action: {best_action} (avg similarity: {avg_similarity*100:.1f}%)."
    )
    
    return (best_action, confidence, reason)


def _calculate_confidence(statistics: Dict) -> float:
    """Calculate overall confidence score (0-1)."""
    # Components of confidence
    sample_confidence = min(statistics.get('total_matches', 0) / 50, 1.0) * 0.2
    win_rate_confidence = statistics.get('win_rate', 0) * 0.3
    similarity_confidence = statistics.get('avg_similarity', 0) * 0.2
    
    # Profit factor contribution (capped at 2.0)
    pf = min(statistics.get('profit_factor', 0), 2.0)
    pf_confidence = (pf / 2.0) * 0.2
    
    # Consistency (low std deviation is good)
    pnl_std = statistics.get('pnl_std', 100)
    avg_pnl = abs(statistics.get('avg_pnl', 1))
    consistency = max(0, 1 - (pnl_std / (avg_pnl + 1))) * 0.1
    
    total_confidence = sample_confidence + win_rate_confidence + similarity_confidence + pf_confidence + consistency
    
    return min(max(total_confidence, 0), 1.0)


def _calculate_exit_levels(statistics: Dict, context: Dict, decision: str) -> tuple[Optional[str], Optional[str]]:
    """Calculate stop-loss and take-profit levels."""
    if decision == 'WAIT':
        return None, None
    
    pdh_delta = context.get('PDH_delta', 0)
    pdl_delta = context.get('PDL_delta', 0)
    atr = context.get('atr', 1.0)
    volatility = context.get('volatility', 'MED')
    
    # Adjust multipliers based on volatility
    vol_multipliers = {'LOW': 0.8, 'MED': 1.0, 'HIGH': 1.3}
    vol_mult = vol_multipliers.get(volatility, 1.0)
    
    # Calculate based on average win/loss and ATR
    avg_win = statistics.get('avg_win_pnl', 100)
    avg_loss = abs(statistics.get('avg_loss_pnl', 50))
    
    # Target 2:1 reward/risk ratio minimum
    risk_points = max(atr * 2 * vol_mult, avg_loss / 5)  # Assuming $5/point for ES micro
    reward_points = max(atr * 3 * vol_mult, avg_win / 5)
    
    if decision == 'BUY':
        stop_loss = f"Entry - {risk_points:.2f} pts ({atr * 2:.2f} ATR)"
        take_profit = f"Entry + {reward_points:.2f} pts (PDH + {pdh_delta:.2f}%)" if pdh_delta < 0.5 else f"Entry + {reward_points:.2f} pts"
    else:  # SELL
        stop_loss = f"Entry + {risk_points:.2f} pts ({atr * 2:.2f} ATR)"
        take_profit = f"Entry - {reward_points:.2f} pts (PDL - {pdl_delta:.2f}%)" if pdl_delta < 0.5 else f"Entry - {reward_points:.2f} pts"
    
    return stop_loss, take_profit


def _empty_statistics() -> Dict:
    """Return empty statistics structure."""
    return {
        'total_matches': 0,
        'win_count': 0,
        'loss_count': 0,
        'win_rate': 0,
        'total_pnl': 0,
        'avg_pnl': 0,
        'pnl_std': 0,
        'avg_win_pnl': 0,
        'avg_loss_pnl': 0,
        'profit_factor': 0,
        'buy_count': 0,
        'buy_win_rate': 0,
        'buy_avg_pnl': 0,
        'sell_count': 0,
        'sell_win_rate': 0,
        'sell_avg_pnl': 0,
        'best_action': 'NEUTRAL',
        'weighted_win_rate': 0,
        'avg_similarity': 0,
        'by_regime': {},
        'by_volatility': {},
        'max_drawdown': 0,
        'best_trade': 0,
        'win_loss_ratio': 0
    }
