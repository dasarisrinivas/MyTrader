"""
Local wrappers for AWS Lambda functions.

These wrappers allow the backtest to run offline by providing local implementations
of the Lambda functions used by the four agents. They preserve the same function
signatures as the AWS Lambda handlers for compatibility.
"""
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

from ..learning.strategy_state import StrategyStateManager

# Import Lambda function logic directly
try:
    import sys
    lambda_path = Path(__file__).parent.parent.parent / "aws" / "lambda"
    sys.path.insert(0, str(lambda_path))
    
    from clean_and_structure_trade_data.lambda_function import (
        _handle_direct_request as clean_data_handler,
        _clean_and_validate_trades,
        _extract_features,
        _create_kb_document,
    )
    from calculate_winrate_statistics.lambda_function import (
        _handle_direct_request as winrate_handler,
        _calculate_statistics,
        _make_decision,
        _calculate_exit_levels,
    )
    from risk_control_engine.lambda_function import (
        _handle_direct_request as risk_handler,
        _check_daily_pnl_limit,
        _check_losing_streak,
        _check_daily_trades,
        _check_position_exposure,
        _check_volatility,
        _check_confidence,
        _create_response,
    )
    from learn_from_losses.lambda_function import (
        _handle_direct_request as learning_handler,
        _identify_loss_patterns,
        _generate_rule_updates,
        _apply_rule_updates,
    )
    LAMBDA_IMPORTS_AVAILABLE = True
except Exception:
    LAMBDA_IMPORTS_AVAILABLE = False


class LocalLambdaWrapper:
    """Base class for local Lambda wrappers."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def _resolve_artifact_date(
        self,
        event: Optional[Dict[str, Any]] = None,
        default: Optional[str] = None
    ) -> str:
        """Figure out which logical trading day an artifact belongs to."""
        if default is None:
            default = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if not event:
            return default
        
        for key in ('backtest_date', 'date', 'target_date', 'event_date'):
            candidate = event.get(key) if isinstance(event, dict) else None
            if candidate:
                return str(candidate)
        
        date_range = event.get('date_range') if isinstance(event, dict) else None
        if isinstance(date_range, dict):
            for key in ('end', 'start'):
                candidate = date_range.get(key)
                if candidate:
                    return str(candidate)
        
        context = event.get('current_context') if isinstance(event, dict) else None
        if isinstance(context, dict):
            candidate = context.get('date')
            if candidate:
                return str(candidate)
        
        env_override = os.environ.get('FF_BACKTEST_TARGET_DATE')
        if env_override:
            return env_override
        return default
    
    def _save_artifact(self, date: str, filename: str, data: Any) -> Path:
        """Save artifact to date-specific directory."""
        date_dir = self.artifacts_dir / date
        date_dir.mkdir(parents=True, exist_ok=True)
        filepath = date_dir / filename
        
        if filename.endswith('.ndjson'):
            # Append to NDJSON file
            with open(filepath, 'a') as f:
                f.write(json.dumps(data, default=str) + '\n')
        else:
            # Write JSON file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        return filepath


class Agent1DataIngestionWrapper(LocalLambdaWrapper):
    """Local wrapper for Agent 1: Data Ingestion & Feature Builder."""
    
    def invoke(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw trade data and generate features.
        
        Args:
            event: {
                "source": "direct",
                "raw_data": [...],  # List of trade records
                "date": "2024-11-15"
            }
        
        Returns:
            {
                "status": "success",
                "processed_count": 10,
                "structured_key": "...",
                "features_key": "...",
                "summary": {...}
            }
        """
        if LAMBDA_IMPORTS_AVAILABLE:
            result = clean_data_handler(event, None)
        else:
            # Fallback implementation
            result = self._fallback_process(event)
        
        # Save manifest
        date = event.get('date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        manifest = {
            'date': date,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'processed_count': result.get('processed_count', 0),
            'structured_key': result.get('structured_key'),
            'features_key': result.get('features_key'),
            'summary': result.get('summary', {}),
            'schema_version': '1.0'
        }
        self._save_artifact(date, 'agent1_features_manifest.json', manifest)
        
        return result
    
    def _fallback_process(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback processing if Lambda imports unavailable."""
        raw_data = event.get('raw_data', [])
        date = event.get('date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        
        if not raw_data:
            return {
                'status': 'success',
                'processed_count': 0,
                'message': 'No data to process',
                'structured_key': None,
                'features_key': None,
            }
        
        # Simple feature extraction
        features = {
            'date': date,
            'total_trades': len(raw_data),
            'win_count': sum(1 for t in raw_data if t.get('outcome') == 'WIN'),
            'loss_count': sum(1 for t in raw_data if t.get('outcome') == 'LOSS'),
            'total_pnl': sum(t.get('pnl', 0) for t in raw_data),
        }
        
        return {
            'status': 'success',
            'processed_count': len(raw_data),
            'structured_key': f'structured/{date}/trades.json',
            'features_key': f'features/{date}/features.json',
            'summary': features
        }


class Agent2DecisionEngineWrapper(LocalLambdaWrapper):
    """Local wrapper for Agent 2: RAG + Similarity Search Decision Engine."""
    
    def invoke(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate winrate statistics and make trading decision.
        
        Args:
            event: {
                "similar_trades": [...],
                "current_context": {
                    "trend": "UPTREND",
                    "volatility": "LOW",
                    ...
                }
            }
        
        Returns:
            {
                "status": "success",
                "decision": "BUY" | "SELL" | "WAIT",
                "confidence": 0.0-1.0,
                "reason": "...",
                "statistics": {...},
                "stop_loss": "...",
                "take_profit": "..."
            }
        """
        if LAMBDA_IMPORTS_AVAILABLE:
            result = winrate_handler(event, None)
        else:
            result = self._fallback_decision(event)
        
        # Log decision
        date = self._resolve_artifact_date(event)
        decision_log = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'decision': result.get('decision'),
            'confidence': result.get('confidence'),
            'reason': result.get('reason'),
            'statistics': result.get('statistics', {}),
            'stop_loss': result.get('stop_loss'),
            'take_profit': result.get('take_profit'),
            'similar_trades_count': len(event.get('similar_trades', [])),
            'placeholder_reason': event.get('placeholder_reason'),
        }
        self._save_artifact(date, 'agent2_decisions.ndjson', decision_log)
        
        return result
    
    def _fallback_decision(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback decision logic if Lambda imports unavailable."""
        similar_trades = event.get('similar_trades', [])
        current_context = event.get('current_context', {})
        strategy_params = event.get('strategy_params', {})
        
        if similar_trades:
            wins = [t for t in similar_trades if t.get('outcome') == 'WIN']
            win_rate = len(wins) / len(similar_trades) if similar_trades else 0
            
            if win_rate >= 0.5 and len(similar_trades) >= 5:
                buys = [t for t in similar_trades if t.get('action') == 'BUY']
                sells = [t for t in similar_trades if t.get('action') == 'SELL']
                buy_win_rate = len([t for t in buys if t.get('outcome') == 'WIN']) / len(buys) if buys else 0
                sell_win_rate = len([t for t in sells if t.get('outcome') == 'WIN']) / len(sells) if sells else 0
                
                if buy_win_rate > sell_win_rate + 0.05:
                    decision = 'BUY'
                    confidence = min(0.85, max(win_rate, buy_win_rate))
                elif sell_win_rate > buy_win_rate + 0.05:
                    decision = 'SELL'
                    confidence = min(0.85, max(win_rate, sell_win_rate))
                else:
                    decision = 'WAIT'
                    confidence = win_rate * 0.6
            else:
                decision = 'WAIT'
                confidence = max(0.2, win_rate * 0.5)
            
            result = {
                'status': 'success',
                'decision': decision,
                'confidence': confidence,
                'reason': f'Pattern win rate {win_rate*100:.1f}% from {len(similar_trades)} matches',
                'statistics': {
                    'total_matches': len(similar_trades),
                    'win_rate': win_rate,
                    'win_count': len(wins),
                    'loss_count': len(similar_trades) - len(wins),
                    'mode': 'retrieval',
                },
                'stop_loss': 'Entry - 2.0 ATR' if decision != 'WAIT' else None,
                'take_profit': 'Entry + 3.0 ATR' if decision != 'WAIT' else None
            }
            if decision != 'WAIT':
                return result
        
        # Fall back to heuristic decisioning when retrieval could not trigger a trade
        return self._heuristic_decision(current_context, strategy_params, event)

    def _heuristic_decision(
        self,
        current_context: Dict[str, Any],
        strategy_params: Dict[str, Any],
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a deterministic trade signal when retrieval data is sparse."""
        if not current_context:
            return {
                'status': 'success',
                'decision': 'WAIT',
                'confidence': 0.0,
                'reason': 'No market context provided',
                'statistics': {'mode': 'heuristic', 'detail': 'missing_context'},
                'stop_loss': None,
                'take_profit': None
            }
        
        trend = current_context.get('trend', 'RANGE')
        rsi = float(current_context.get('rsi', 50))
        macd = float(current_context.get('macd_histogram', 0))
        atr = float(current_context.get('atr', 1.0))
        volatility = current_context.get('volatility', 'MED')
        pdh_delta = float(current_context.get('PDH_delta', 0) or 0)
        pdl_delta = float(current_context.get('PDL_delta', 0) or 0)
        threshold = strategy_params.get('decision_threshold', 0.58)
        min_confidence = strategy_params.get('min_confidence', 0.55)
        exploration_rate = strategy_params.get('exploration_rate', 0.0)
        
        buy_score = 0.0
        sell_score = 0.0
        
        if trend == 'UPTREND':
            buy_score += 0.35
        elif trend == 'DOWNTREND':
            sell_score += 0.35
        else:
            buy_score += 0.1
            sell_score += 0.1
        
        buy_score += max(0.0, (rsi - 52) / 100.0)
        sell_score += max(0.0, (48 - rsi) / 100.0)
        
        if macd > 0:
            buy_score += min(0.2, macd / 4.0)
        elif macd < 0:
            sell_score += min(0.2, abs(macd) / 4.0)
        
        if pdh_delta > 0:
            buy_score += 0.05
        if pdl_delta < 0:
            sell_score += 0.05
        
        if volatility == 'HIGH':
            buy_score -= 0.05
            sell_score -= 0.05
        
        buy_score = max(0.0, min(1.0, buy_score))
        sell_score = max(0.0, min(1.0, sell_score))
        
        action = 'BUY' if buy_score >= sell_score else 'SELL'
        signal_strength = max(buy_score, sell_score)
        
        if signal_strength + exploration_rate < threshold:
            return {
                'status': 'success',
                'decision': 'WAIT',
                'confidence': signal_strength,
                'reason': (
                    f'Heuristic signal {signal_strength:.2f} below threshold '
                    f'{threshold:.2f}'
                ),
                'statistics': {
                    'mode': 'heuristic',
                    'buy_score': buy_score,
                    'sell_score': sell_score,
                    'threshold': threshold,
                },
                'stop_loss': None,
                'take_profit': None
            }
        
        confidence = max(min_confidence, min(0.95, signal_strength + exploration_rate / 2))
        reason = (
            f"Heuristic {action} signal strength {signal_strength:.2f} "
            f"(threshold {threshold:.2f}, volatility {volatility})"
        )
        
        return {
            'status': 'success',
            'decision': action,
            'confidence': confidence,
            'reason': reason,
            'statistics': {
                'mode': 'heuristic',
                'buy_score': buy_score,
                'sell_score': sell_score,
                'threshold': threshold,
                'exploration': exploration_rate,
            },
            'stop_loss': f'Entry - {2.0 * atr:.2f} ATR' if action == 'BUY' else f'Entry + {2.0 * atr:.2f} ATR',
            'take_profit': f'Entry + {3.0 * atr:.2f} ATR' if action == 'BUY' else f'Entry - {3.0 * atr:.2f} ATR'
        }


class Agent3RiskControlWrapper(LocalLambdaWrapper):
    """Local wrapper for Agent 3: Risk & Position Sizing Agent."""
    
    def invoke(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate trade risk and determine position sizing.
        
        Args:
            event: {
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
        
        Returns:
            {
                "status": "success",
                "allowed_to_trade": True,
                "size_multiplier": 0.8,
                "adjusted_size": 1,
                "risk_flags": [...],
                "reason": "..."
            }
        """
        if LAMBDA_IMPORTS_AVAILABLE:
            result = risk_handler(event, None)
        else:
            result = self._fallback_risk_check(event)
        
        # Log risk evaluation
        date = self._resolve_artifact_date(event)
        risk_log = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'trade_decision': event.get('trade_decision', {}),
            'allowed': result.get('allowed_to_trade', False),
            'size_multiplier': result.get('size_multiplier', 0),
            'adjusted_size': result.get('adjusted_size', 0),
            'risk_flags': result.get('risk_flags', []),
            'reason': result.get('reason', ''),
            'placeholder_reason': event.get('placeholder_reason'),
        }
        self._save_artifact(date, 'agent3_risk.ndjson', risk_log)
        
        return result
    
    def _fallback_risk_check(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback risk check if Lambda imports unavailable."""
        trade_decision = event.get('trade_decision', {})
        account_metrics = event.get('account_metrics', {})
        market_conditions = event.get('market_conditions', {})
        strategy_params = event.get('strategy_params', {})
        
        placeholder_reason = event.get('placeholder_reason')
        if placeholder_reason:
            return _create_response(
                allowed=False,
                multiplier=0.0,
                size=0,
                flags=['PLACEHOLDER'],
                reason=placeholder_reason
            )
        
        if not trade_decision or trade_decision.get('action') in (None, 'WAIT'):
            return _create_response(
                allowed=False,
                multiplier=0.0,
                size=0,
                flags=['NO_SIGNAL'],
                reason='No actionable decision supplied'
            )
        
        # Simple risk checks
        risk_flags = []
        size_multiplier = 1.0
        
        # Daily P&L check
        daily_pnl = account_metrics.get('current_pnl_today', 0)
        if daily_pnl <= -1500:
            return _create_response(
                allowed=False,
                multiplier=0,
                size=0,
                flags=['DAILY_LOSS_LIMIT'],
                reason='Daily loss limit reached'
            )
        elif daily_pnl < -1000:
            size_multiplier *= 0.5
            risk_flags.append('APPROACHING_DAILY_LIMIT')
        
        # Losing streak check
        losing_streak = account_metrics.get('losing_streak', 0)
        if losing_streak >= 3:
            return _create_response(
                allowed=False,
                multiplier=0,
                size=0,
                flags=['MAX_LOSING_STREAK'],
                reason='Maximum losing streak reached'
            )
        elif losing_streak >= 2:
            size_multiplier *= 0.8
            risk_flags.append('LOSING_STREAK_WARNING')
        
        max_trades_per_day = strategy_params.get('max_trades_per_day')
        if max_trades_per_day is not None:
            if account_metrics.get('trades_today', 0) >= max_trades_per_day:
                return _create_response(
                    allowed=False,
                    multiplier=0.0,
                    size=0,
                    flags=['DAILY_TRADE_CAP'],
                    reason='Adaptive cap on number of trades reached'
                )
        
        # Volatility check
        volatility = market_conditions.get('volatility', 'MED')
        if volatility == 'HIGH':
            size_multiplier *= 0.6
            risk_flags.append('HIGH_VOLATILITY')
        
        # Confidence check
        confidence = trade_decision.get('confidence', 0)
        if confidence < 0.6:
            size_multiplier *= 0.5
            risk_flags.append('LOW_CONFIDENCE')
        
        size_multiplier *= max(0.2, float(strategy_params.get('risk_multiplier', 1.0)))
        
        proposed_size = trade_decision.get('proposed_size', 1)
        adjusted_size = max(1, int(proposed_size * size_multiplier))
        
        allowed = size_multiplier > 0.3 and adjusted_size > 0
        
        return _create_response(
            allowed=allowed,
            multiplier=size_multiplier,
            size=adjusted_size,
            flags=risk_flags,
            reason='Risk check passed' if allowed else 'Risk check failed'
        )


class Agent4LearningWrapper(LocalLambdaWrapper):
    """Local wrapper for Agent 4: Strategy Optimization & Learning Agent."""
    
    def __init__(self, artifacts_dir: Path):
        super().__init__(artifacts_dir)
        self.strategy_manager = StrategyStateManager(artifacts_dir / 'strategy_state.json')
    
    def invoke(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze losing trades and update strategy rules.
        
        Args:
            event: {
                "analysis_type": "daily",
                "date_range": {
                    "start": "2024-11-15",
                    "end": "2024-11-15"
                },
                "losing_trades": [...]  # Optional
            }
        
        Returns:
            {
                "status": "success",
                "patterns_identified": [...],
                "rules_updated": [...],
                "bad_patterns_added": [...],
                "summary": {...}
            }
        """
        if LAMBDA_IMPORTS_AVAILABLE:
            result = learning_handler(event, None)
        else:
            result = self._fallback_learning(event)
        
        # Save learning update
        date = event.get('date_range', {}).get('end', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        learning_update = {
            'date': date,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'patterns_identified': result.get('patterns_identified', []),
            'rules_updated': result.get('rules_updated', []),
            'bad_patterns_added': result.get('bad_patterns_added', []),
            'summary': result.get('summary', {}),
        }
        self._save_artifact(date, 'agent4_learning_update.json', learning_update)
        
        return result
    
    def _fallback_learning(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback learning analysis if Lambda imports unavailable."""
        losing_trades = event.get('losing_trades', [])
        date = event.get('date_range', {}).get('end', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        daily_summary = event.get('daily_summary', {})
        current_state = self.strategy_manager.load_state()
        
        patterns = []
        if losing_trades:
            regime_counts = defaultdict(int)
            for trade in losing_trades:
                regime = trade.get('regime', 'UNKNOWN')
                regime_counts[regime] += 1
            
            for regime, count in regime_counts.items():
                if count >= 2:
                    patterns.append({
                        'type': 'REGIME_PATTERN',
                        'pattern': {'regime': regime},
                        'occurrences': count,
                        'description': f'Multiple losses in {regime} regime',
                        'recommendation': f'Reduce trading in {regime} conditions'
                    })
        
        updates, notes = self._derive_state_updates(current_state, daily_summary, losing_trades)
        updated_state = current_state
        if updates:
            updated_state = self.strategy_manager.update_state(updates)
            self.strategy_manager.append_adjustment_record({
                'date': date,
                'updates': updates,
                'notes': notes,
                'pnl': daily_summary.get('pnl'),
                'trades_taken': daily_summary.get('trades_taken'),
            })
        
        rules_updated = []
        if updates:
            for key, value in updates.items():
                rules_updated.append({
                    'parameter': key,
                    'new_value': updated_state.get(key),
                    'reason': '; '.join(notes) or 'Adaptive tuning',
                })
        
        summary_payload = {
            'trades_analyzed': len(losing_trades),
            'patterns_found': len(patterns),
            'total_loss': sum(t.get('pnl', 0) for t in losing_trades),
            'daily_summary': daily_summary,
            'strategy_state': updated_state,
            'adjustment_notes': notes,
        }
        
        if not losing_trades:
            summary_payload['message'] = 'No losing trades to analyze'
        
        return {
            'status': 'success',
            'patterns_identified': patterns,
            'rules_updated': rules_updated,
            'bad_patterns_added': patterns[:5],
            'summary': summary_payload
        }
    
    def _derive_state_updates(
        self,
        current_state: Dict[str, Any],
        daily_summary: Dict[str, Any],
        losing_trades: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Translate daily outcomes into strategy parameter adjustments."""
        updates: Dict[str, Any] = {}
        notes: List[str] = []
        
        trades_taken = daily_summary.get('trades_taken', 0)
        pnl = daily_summary.get('pnl', 0.0) or 0.0
        decision_events = daily_summary.get('decision_events', 0)
        missed_opportunity = abs(daily_summary.get('missed_opportunity', 0.0) or 0.0)
        
        if trades_taken == 0 and decision_events == 0:
            updates['decision_threshold'] = max(
                0.45, current_state.get('decision_threshold', 0.58) - 0.02
            )
            updates['exploration_rate'] = min(
                0.3, current_state.get('exploration_rate', 0.1) + 0.02
            )
            notes.append('Loosened signal threshold due to zero activity')
        
        if trades_taken == 0 and missed_opportunity > 0:
            updates['target_daily_trades'] = min(
                4, current_state.get('target_daily_trades', 2) + 1
            )
            notes.append('Raising target trades after missed move')
        
        if pnl < 0 and losing_trades:
            updates['risk_multiplier'] = max(
                0.5, current_state.get('risk_multiplier', 1.0) - 0.05
            )
            updates['decision_threshold'] = min(
                0.9, current_state.get('decision_threshold', 0.58) + 0.01
            )
            notes.append('Reduced risk budget after losing day')
        elif pnl > 0 and trades_taken > 0:
            updates['risk_multiplier'] = min(
                1.5, current_state.get('risk_multiplier', 1.0) + 0.02
            )
            notes.append('Rewarding profitable day with higher risk allowance')
        
        max_trades = current_state.get('max_trades_per_day', 6)
        target_trades = current_state.get('target_daily_trades', 2)
        if trades_taken > target_trades and max_trades < 8:
            updates['max_trades_per_day'] = max_trades + 1
            notes.append('Allowing more trades after exceeding target')
        elif trades_taken == 0 and max_trades > 2:
            updates['max_trades_per_day'] = max(2, max_trades - 1)
            notes.append('Reducing trade cap after inactivity')
        
        return updates, notes


def _create_response(
    allowed: bool,
    multiplier: float,
    size: int,
    flags: List[str],
    reason: str
) -> Dict[str, Any]:
    """Create standardized response (from risk_control_engine)."""
    return {
        'status': 'success',
        'allowed_to_trade': allowed,
        'size_multiplier': round(multiplier, 2),
        'adjusted_size': size,
        'risk_flags': flags,
        'reason': reason,
        'evaluated_at': datetime.now(timezone.utc).isoformat()
    }
