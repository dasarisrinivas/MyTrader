"""
AWS Bedrock Agent Client for Local Bot Integration

This module provides a client for invoking Bedrock Agents and
processing their responses for the trading bot.
"""

import json
import time
import uuid
import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from loguru import logger


class BedrockAgentClientError(Exception):
    """Custom exception for Bedrock Agent client errors."""
    pass


class BedrockAgentClient:
    """Client for invoking AWS Bedrock Agents.
    
    This client provides:
    - Agent invocation with retry logic
    - Session management
    - Response parsing and validation
    - Rate limiting
    """
    
    # Default agent configuration
    DEFAULT_CONFIG = {
        'region_name': 'us-east-1',
        'max_retries': 3,
        'retry_delay': 2.0,
        'timeout': 30,
        'max_requests_per_minute': 20,
    }
    
    def __init__(
        self,
        region_name: str = None,
        agent_ids: Dict[str, str] = None,
        agent_alias_ids: Dict[str, str] = None,
        max_retries: int = None,
        timeout: int = None,
    ):
        """Initialize Bedrock Agent client.
        
        Args:
            region_name: AWS region for Bedrock
            agent_ids: Dictionary mapping agent names to IDs
                      {'decision': 'AGENT_ID_1', 'risk': 'AGENT_ID_2', ...}
            agent_alias_ids: Dictionary mapping agent names to alias IDs
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        if not BOTO3_AVAILABLE:
            raise BedrockAgentClientError(
                "boto3 is required. Install with: pip install boto3"
            )
        
        self.region_name = region_name or self.DEFAULT_CONFIG['region_name']
        self.max_retries = max_retries or self.DEFAULT_CONFIG['max_retries']
        self.timeout = timeout or self.DEFAULT_CONFIG['timeout']
        
        # Agent configuration
        self.agent_ids = agent_ids or {}
        self.agent_alias_ids = agent_alias_ids or {}

        # Optional DynamoDB response cache
        self.cache_table_name = os.environ.get('AGENT_CACHE_TABLE')
        self.cache_ttl_seconds = int(os.environ.get('AGENT_CACHE_TTL_SECONDS', '900'))
        self._dynamo = None
        
        # Session management
        self._sessions: Dict[str, str] = {}
        
        # Rate limiting
        self._request_times: List[datetime] = []
        self._max_rpm = self.DEFAULT_CONFIG['max_requests_per_minute']
        
        # Initialize boto3 client
        try:
            self.client = boto3.client(
                'bedrock-agent-runtime',
                region_name=self.region_name
            )
            if self.cache_table_name:
                self._dynamo = boto3.resource('dynamodb', region_name=self.region_name)
            logger.info(f"Initialized Bedrock Agent client in {self.region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock Agent client: {e}")
            raise BedrockAgentClientError(f"Client initialization failed: {e}")
    
    def invoke_agent(
        self,
        agent_name: str,
        input_text: str,
        session_id: str = None,
        enable_trace: bool = False,
    ) -> Dict[str, Any]:
        """Invoke a Bedrock Agent.
        
        Args:
            agent_name: Name of agent to invoke ('decision', 'risk', etc.)
            input_text: Input prompt for the agent
            session_id: Optional session ID for conversation context
            enable_trace: Enable trace information in response
            
        Returns:
            Dictionary with agent response and metadata
        """
        # Get agent configuration
        agent_id = self.agent_ids.get(agent_name)
        alias_id = self.agent_alias_ids.get(agent_name)
        
        if not agent_id:
            raise BedrockAgentClientError(f"Unknown agent: {agent_name}")
        
        # Get or create session
        if not session_id:
            session_id = self._get_session(agent_name)
        
        # Enforce rate limiting
        self._enforce_rate_limit()

        cache_key = None
        if self._dynamo and self.cache_table_name:
            cache_key = self._build_cache_key(agent_name, input_text)
            cached_response = self._read_cache(cache_key)
            if cached_response:
                if not cached_response.get('session_id'):
                    cached_response['session_id'] = session_id
                cached_response['cached'] = True
                return cached_response
        
        # Invoke with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Invoking agent {agent_name} (attempt {attempt + 1})")
                
                response = self.client.invoke_agent(
                    agentId=agent_id,
                    agentAliasId=alias_id or 'TSTALIASID',
                    sessionId=session_id,
                    inputText=input_text,
                    enableTrace=enable_trace,
                )
                
                # Process streaming response
                result = self._process_response(response, agent_name)
                result['session_id'] = session_id
                result['cached'] = False

                if cache_key:
                    self._write_cache(cache_key, result)
                
                return result
                
            except ClientError as e:
                last_error = e
                error_code = e.response.get('Error', {}).get('Code', '')
                
                if error_code in ['ThrottlingException', 'ServiceUnavailableException']:
                    wait_time = self.DEFAULT_CONFIG['retry_delay'] * (2 ** attempt)
                    logger.warning(f"Agent {agent_name} throttled, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Agent invocation error: {e}")
                    raise BedrockAgentClientError(f"Agent invocation failed: {e}")
                    
            except BotoCoreError as e:
                last_error = e
                logger.warning(f"Agent {agent_name} error (attempt {attempt + 1}): {e}")
                time.sleep(self.DEFAULT_CONFIG['retry_delay'])
        
        raise BedrockAgentClientError(
            f"Agent {agent_name} invocation failed after {self.max_retries} attempts: {last_error}"
        )
    
    def invoke_decision_agent(
        self,
        market_snapshot: Dict[str, Any],
        session_id: str = None,
    ) -> Dict[str, Any]:
        """Invoke the Decision Engine Agent (Agent 2).
        
        Args:
            market_snapshot: Market data and indicators
            session_id: Optional session ID
            
        Returns:
            Trading decision with confidence and reasoning
        """
        prompt = f"""Analyze this market snapshot and provide a trading decision:

Market Snapshot:
- Symbol: {market_snapshot.get('symbol', 'ES')}
- Current Price: {market_snapshot.get('price', 0)}
- Trend/Regime: {market_snapshot.get('trend', 'UNKNOWN')}
- Volatility: {market_snapshot.get('volatility', 'MED')}
- RSI: {market_snapshot.get('rsi', 50)}
- ATR: {market_snapshot.get('atr', 0)}
- PDH Delta: {market_snapshot.get('PDH_delta', 0):.3f}
- PDL Delta: {market_snapshot.get('PDL_delta', 0):.3f}
- EMA 9: {market_snapshot.get('ema_9', 0)}
- EMA 20: {market_snapshot.get('ema_20', 0)}
- Volume: {market_snapshot.get('volume', 0)}

Please search the knowledge base for similar historical patterns and provide:
1. Your trading decision (BUY/SELL/WAIT)
2. Confidence level (0-100%)
3. Reasoning based on historical patterns
4. Recommended stop-loss and take-profit levels

Return your response as structured JSON."""

        response = self.invoke_agent('decision', prompt, session_id)
        
        # Parse and validate decision response
        return self._parse_decision_response(response)
    
    def invoke_risk_agent(
        self,
        trade_decision: Dict[str, Any],
        account_metrics: Dict[str, Any],
        market_conditions: Dict[str, Any],
        session_id: str = None,
    ) -> Dict[str, Any]:
        """Invoke the Risk Control Agent (Agent 3).
        
        Args:
            trade_decision: Decision from Agent 2
            account_metrics: Current account state
            market_conditions: Current market conditions
            session_id: Optional session ID
            
        Returns:
            Risk evaluation with position sizing
        """
        prompt = f"""Evaluate this trade decision for risk:

Trade Decision:
- Action: {trade_decision.get('decision', 'WAIT')}
- Confidence: {trade_decision.get('confidence', 0)}
- Symbol: {trade_decision.get('symbol', 'ES')}
- Proposed Size: {trade_decision.get('proposed_size', 1)}

Account Metrics:
- Today's P&L: ${account_metrics.get('current_pnl_today', 0):.2f}
- Current Position: {account_metrics.get('current_position', 0)} contracts
- Losing Streak: {account_metrics.get('losing_streak', 0)}
- Trades Today: {account_metrics.get('trades_today', 0)}
- Account Balance: ${account_metrics.get('account_balance', 0):,.2f}
- Open Risk: ${account_metrics.get('open_risk', 0):.2f}

Market Conditions:
- Volatility: {market_conditions.get('volatility', 'MED')}
- Regime: {market_conditions.get('regime', 'UNKNOWN')}
- ATR: {market_conditions.get('atr', 0)}
- VIX: {market_conditions.get('vix', 15)}

Please evaluate and provide:
1. Whether this trade is allowed (true/false)
2. Position size multiplier (0-1)
3. Adjusted position size
4. Any risk flags or warnings
5. Reasoning for your decision

Return your response as structured JSON."""

        response = self.invoke_agent('risk', prompt, session_id)
        
        # Parse and validate risk response
        return self._parse_risk_response(response)
    
    def _process_response(
        self,
        response: Dict,
        agent_name: str,
    ) -> Dict[str, Any]:
        """Process streaming response from Bedrock Agent."""
        completion = ""
        trace_data = []
        
        # Process event stream
        event_stream = response.get('completion', [])
        
        for event in event_stream:
            if 'chunk' in event:
                chunk_data = event['chunk']
                if 'bytes' in chunk_data:
                    completion += chunk_data['bytes'].decode('utf-8')
            
            if 'trace' in event:
                trace_data.append(event['trace'])
        
        return {
            'agent': agent_name,
            'completion': completion,
            'trace': trace_data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
    
    def _parse_decision_response(self, response: Dict) -> Dict[str, Any]:
        """Parse decision agent response into structured format."""
        completion = response.get('completion', '')
        
        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            json_start = completion.find('{')
            json_end = completion.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = completion[json_start:json_end]
                parsed = json.loads(json_str)
                
                return {
                    'decision': parsed.get('decision', 'WAIT').upper(),
                    'confidence': float(parsed.get('confidence', 0)) / 100 if parsed.get('confidence', 0) > 1 else float(parsed.get('confidence', 0)),
                    'reason': parsed.get('reason', parsed.get('reasoning', '')),
                    'stop_loss': parsed.get('stop_loss'),
                    'take_profit': parsed.get('take_profit'),
                    'similar_patterns': parsed.get('similar_patterns', 0),
                    'raw_response': completion,
                    'session_id': response.get('session_id'),
                }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse decision JSON: {e}")
        
        # Fallback: try to extract decision from text
        completion_upper = completion.upper()
        decision = 'WAIT'
        if 'BUY' in completion_upper and 'SELL' not in completion_upper:
            decision = 'BUY'
        elif 'SELL' in completion_upper and 'BUY' not in completion_upper:
            decision = 'SELL'
        
        return {
            'decision': decision,
            'confidence': 0.5,
            'reason': completion[:500] if completion else 'No response from agent',
            'stop_loss': None,
            'take_profit': None,
            'raw_response': completion,
            'session_id': response.get('session_id'),
        }
    
    def _parse_risk_response(self, response: Dict) -> Dict[str, Any]:
        """Parse risk agent response into structured format."""
        completion = response.get('completion', '')
        
        # Try to extract JSON from response
        try:
            json_start = completion.find('{')
            json_end = completion.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = completion[json_start:json_end]
                parsed = json.loads(json_str)
                
                risk_flags = parsed.get('risk_flags', [])
                if isinstance(risk_flags, str):
                    risk_flags = [risk_flags]
                allowed = parsed.get('allowed_to_trade')
                if allowed is None:
                    allowed = not bool(risk_flags)
                return {
                    'allowed_to_trade': bool(allowed),
                    'size_multiplier': float(parsed.get('size_multiplier', 0)),
                    'adjusted_size': int(parsed.get('adjusted_size', 0)),
                    'risk_flags': risk_flags,
                    'reason': parsed.get('reason', ''),
                    'raw_response': completion,
                    'session_id': response.get('session_id'),
                }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse risk JSON: {e}")
        
        # Fallback: conservative response
        return {
            'allowed_to_trade': False,
            'size_multiplier': 0,
            'adjusted_size': 0,
            'risk_flags': ['PARSE_ERROR'],
            'reason': 'Failed to parse risk response, blocking trade for safety',
            'raw_response': completion,
            'session_id': response.get('session_id'),
        }

    def _build_cache_key(self, agent_name: str, input_text: str) -> str:
        digest = hashlib.sha256(input_text.encode('utf-8')).hexdigest()
        return f"{agent_name}#{digest}"

    def _read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self._dynamo or not self.cache_table_name:
            return None
        try:
            table = self._dynamo.Table(self.cache_table_name)
            response = table.get_item(Key={"pk": cache_key})
            item = response.get('Item')
            if not item:
                return None
            ttl = item.get('ttl')
            if ttl and int(ttl) < int(time.time()):
                return None
            payload = item.get('payload')
            return json.loads(payload) if payload else None
        except Exception as exc:
            logger.debug(f"Cache read failed for {cache_key}: {exc}")
            return None

    def _write_cache(self, cache_key: str, payload: Dict[str, Any]) -> None:
        if not self._dynamo or not self.cache_table_name:
            return
        try:
            table = self._dynamo.Table(self.cache_table_name)
            ttl = int(time.time()) + self.cache_ttl_seconds
            table.put_item(
                Item={
                    "pk": cache_key,
                    "ttl": ttl,
                    "payload": json.dumps(payload),
                }
            )
        except Exception as exc:
            logger.debug(f"Cache write failed for {cache_key}: {exc}")
    
    def _get_session(self, agent_name: str) -> str:
        """Get or create session ID for an agent."""
        if agent_name not in self._sessions:
            self._sessions[agent_name] = str(uuid.uuid4())
        return self._sessions[agent_name]
    
    def reset_session(self, agent_name: str = None) -> None:
        """Reset session(s) to clear conversation context."""
        if agent_name:
            self._sessions.pop(agent_name, None)
        else:
            self._sessions.clear()
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting for API calls."""
        now = datetime.now(timezone.utc)
        
        # Remove requests older than 1 minute
        self._request_times = [
            t for t in self._request_times
            if (now - t).total_seconds() < 60
        ]
        
        # Wait if at capacity
        if len(self._request_times) >= self._max_rpm:
            oldest = self._request_times[0]
            wait_time = 60 - (now - oldest).total_seconds()
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        
        self._request_times.append(now)
