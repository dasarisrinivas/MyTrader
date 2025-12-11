"""
Agent Invoker for Trading Bot

This module provides high-level orchestration of Bedrock Agents
for the trading signal flow.

Usage:
    # Auto-load configuration from deployed_resources.yaml
    invoker = AgentInvoker.from_deployed_config()
    
    # Or with explicit config
    invoker = AgentInvoker(config={
        'agent_ids': {...},
        'agent_alias_ids': {...},
    })
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from loguru import logger

from .bedrock_agent_client import BedrockAgentClient, BedrockAgentClientError


class AgentInvoker:
    """High-level orchestrator for Bedrock Agent invocations.
    
    This class manages the signal flow:
    1. Market snapshot → Decision Agent (Agent 2)
    2. Decision → Risk Agent (Agent 3)
    3. Final trade decision
    """
    
    @classmethod
    def from_deployed_config(
        cls,
        config_path: Optional[str] = None,
        use_step_functions: bool = False,
    ) -> "AgentInvoker":
        """Create AgentInvoker from deployed_resources.yaml configuration.
        
        Args:
            config_path: Path to deployed_resources.yaml (uses default if None)
            use_step_functions: Use Step Functions for orchestration
            
        Returns:
            Configured AgentInvoker instance
        """
        from .config_loader import load_aws_config
        config = load_aws_config(config_path)
        return cls(config=config, use_step_functions=use_step_functions)
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        use_step_functions: bool = False,
    ):
        """Initialize Agent Invoker.
        
        Args:
            config: Configuration dictionary with agent IDs and settings.
                   Can be in either old format (agent1_id, agent2_id, etc.)
                   or new format (agent_ids dict, agent_alias_ids dict)
            use_step_functions: Use Step Functions for orchestration instead of direct calls
        """
        self.config = config or {}
        self.use_step_functions = use_step_functions
        
        # Support both old and new config formats
        agent_ids = self._get_agent_ids(config)
        agent_alias_ids = self._get_agent_alias_ids(config)
        
        # Initialize Bedrock Agent client
        self.agent_client = BedrockAgentClient(
            region_name=config.get('region_name', 'us-east-1'),
            agent_ids=agent_ids,
            agent_alias_ids=agent_alias_ids,
        )
        
        # Step Functions client (if enabled)
        if use_step_functions and BOTO3_AVAILABLE:
            self.sfn_client = boto3.client(
                'stepfunctions',
                region_name=config.get('region_name', 'us-east-1')
            )
            self.signal_flow_arn = config.get('signal_flow_arn')
        else:
            self.sfn_client = None
            self.signal_flow_arn = None
        
        # Metrics
        self._invocation_count = 0
        self._last_invocation_time = None
        
        logger.info(f"Initialized AgentInvoker with agents: {list(agent_ids.keys())}")
    
    def _get_agent_ids(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Extract agent IDs from config (supports old and new formats)."""
        # New format: agent_ids dict
        if 'agent_ids' in config:
            return config['agent_ids']
        
        # Old format: agent1_id, agent2_id, etc.
        return {
            'data': config.get('agent1_id', ''),
            'decision': config.get('agent2_id', ''),
            'risk': config.get('agent3_id', ''),
            'learning': config.get('agent4_id', ''),
        }
    
    def _get_agent_alias_ids(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Extract agent alias IDs from config (supports old and new formats)."""
        # New format: agent_alias_ids dict
        if 'agent_alias_ids' in config:
            return config['agent_alias_ids']
        
        # Old format: agent1_alias_id, agent2_alias_id, etc.
        return {
            'data': config.get('agent1_alias_id', ''),
            'decision': config.get('agent2_alias_id', ''),
            'risk': config.get('agent3_alias_id', ''),
            'learning': config.get('agent4_alias_id', ''),
        }
    
    def get_trading_decision(
        self,
        market_snapshot: Dict[str, Any],
        account_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get trading decision through the agent pipeline.
        
        This is the main entry point for the signal flow:
        Market Snapshot → Decision Agent → Risk Agent → Final Decision
        
        Args:
            market_snapshot: Current market data and indicators
            account_metrics: Current account state
            
        Returns:
            Final trading decision with all metadata
        """
        start_time = time.time()
        
        try:
            if self.use_step_functions and self.sfn_client:
                result = self._invoke_via_step_functions(
                    market_snapshot, account_metrics
                )
            else:
                result = self._invoke_direct(market_snapshot, account_metrics)
            
            # Add timing metadata
            result['latency_ms'] = int((time.time() - start_time) * 1000)
            result['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            self._invocation_count += 1
            self._last_invocation_time = datetime.now(timezone.utc)
            
            return result
            
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}")
            return self._create_error_response(str(e), start_time)
    
    def _invoke_direct(
        self,
        market_snapshot: Dict[str, Any],
        account_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Invoke agents directly without Step Functions."""
        
        # Step 1: Get decision from Decision Agent (Agent 2)
        logger.info("Invoking Decision Agent...")
        decision_response = self.agent_client.invoke_decision_agent(
            market_snapshot=market_snapshot,
        )
        
        logger.info(
            f"Decision Agent response: {decision_response.get('decision')} "
            f"(confidence: {decision_response.get('confidence', 0):.2%})"
        )
        
        # If WAIT, no need to check risk
        if decision_response.get('decision') == 'WAIT':
            return {
                'decision': 'WAIT',
                'confidence': decision_response.get('confidence', 0),
                'reason': decision_response.get('reason', 'No clear signal'),
                'allowed_to_trade': False,
                'size_multiplier': 0,
                'adjusted_size': 0,
                'risk_flags': [],
                'stop_loss': None,
                'take_profit': None,
                'decision_details': decision_response,
                'risk_details': None,
            }
        
        # Step 2: Get risk evaluation from Risk Agent (Agent 3)
        logger.info("Invoking Risk Agent...")
        
        # Extract market conditions from snapshot
        market_conditions = {
            'volatility': market_snapshot.get('volatility', 'MED'),
            'regime': market_snapshot.get('trend', 'UNKNOWN'),
            'atr': market_snapshot.get('atr', 1.0),
            'vix': market_snapshot.get('vix', 15),
        }
        
        # Build trade decision for risk agent
        trade_decision = {
            'decision': decision_response.get('decision'),
            'confidence': decision_response.get('confidence', 0),
            'symbol': market_snapshot.get('symbol', 'ES'),
            'proposed_size': 1,  # Default to 1 contract
        }
        
        risk_response = self.agent_client.invoke_risk_agent(
            trade_decision=trade_decision,
            account_metrics=account_metrics,
            market_conditions=market_conditions,
        )
        
        logger.info(
            f"Risk Agent response: allowed={risk_response.get('allowed_to_trade')}, "
            f"size_multiplier={risk_response.get('size_multiplier', 0):.2f}"
        )
        
        # Build final response
        return {
            'decision': decision_response.get('decision'),
            'confidence': decision_response.get('confidence', 0),
            'reason': decision_response.get('reason', ''),
            'allowed_to_trade': risk_response.get('allowed_to_trade', False),
            'size_multiplier': risk_response.get('size_multiplier', 0),
            'adjusted_size': risk_response.get('adjusted_size', 0),
            'risk_flags': risk_response.get('risk_flags', []),
            'stop_loss': decision_response.get('stop_loss'),
            'take_profit': decision_response.get('take_profit'),
            'decision_details': decision_response,
            'risk_details': risk_response,
        }
    
    def _invoke_via_step_functions(
        self,
        market_snapshot: Dict[str, Any],
        account_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Invoke the signal flow via AWS Step Functions."""
        
        if not self.signal_flow_arn:
            raise BedrockAgentClientError("Signal flow ARN not configured")
        
        import uuid
        
        # Start execution
        execution_input = json.dumps({
            'market_snapshot': market_snapshot,
            'account_metrics': account_metrics,
            'session_id': str(uuid.uuid4()),
        })
        
        response = self.sfn_client.start_sync_execution(
            stateMachineArn=self.signal_flow_arn,
            input=execution_input,
        )
        
        # Check status
        status = response.get('status')
        
        if status == 'SUCCEEDED':
            output = json.loads(response.get('output', '{}'))
            return output
        else:
            error = response.get('error', 'Unknown error')
            cause = response.get('cause', '')
            raise BedrockAgentClientError(f"Step Function failed: {error} - {cause}")
    
    def _create_error_response(self, error: str, start_time: float) -> Dict[str, Any]:
        """Create error response for failed invocations."""
        return {
            'decision': 'WAIT',
            'confidence': 0,
            'reason': f'Agent invocation failed: {error}',
            'allowed_to_trade': False,
            'size_multiplier': 0,
            'adjusted_size': 0,
            'risk_flags': ['AGENT_ERROR'],
            'stop_loss': None,
            'take_profit': None,
            'error': error,
            'latency_ms': int((time.time() - start_time) * 1000),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
    
    def upload_trade_data(
        self,
        trades: list,
        date: str = None,
    ) -> Dict[str, Any]:
        """Upload trade data for nightly processing.
        
        Args:
            trades: List of trade records
            date: Date string (YYYY-MM-DD), defaults to today
            
        Returns:
            Upload result
        """
        if not date:
            date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        # Upload to S3
        if BOTO3_AVAILABLE:
            s3_client = boto3.client('s3')
            bucket = self.config.get('s3_bucket', 'trading-bot-data')
            key = f"raw/{date}/trades.json"
            
            try:
                s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=json.dumps(trades, default=str).encode('utf-8'),
                    ContentType='application/json',
                )
                
                logger.info(f"Uploaded {len(trades)} trades to s3://{bucket}/{key}")
                
                return {
                    'status': 'success',
                    'bucket': bucket,
                    'key': key,
                    'trade_count': len(trades),
                }
            except Exception as e:
                logger.error(f"Failed to upload trades: {e}")
                return {
                    'status': 'error',
                    'error': str(e),
                }
        else:
            return {
                'status': 'error',
                'error': 'boto3 not available',
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get invoker statistics."""
        return {
            'invocation_count': self._invocation_count,
            'last_invocation': self._last_invocation_time.isoformat() if self._last_invocation_time else None,
            'use_step_functions': self.use_step_functions,
        }
