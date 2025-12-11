"""
AWS Multi-Agent Integration Module

This module provides local bot integration with AWS Bedrock Agents.

Quick Start:
    from mytrader.aws import AgentInvoker
    
    # Auto-load from deployed_resources.yaml
    invoker = AgentInvoker.from_deployed_config()
    
    # Get trading decision
    decision = invoker.get_trading_decision(market_snapshot, account_metrics)
"""

from .bedrock_agent_client import BedrockAgentClient, BedrockAgentClientError
from .agent_invoker import AgentInvoker
from .market_snapshot import MarketSnapshotBuilder
from .pnl_updater import PnLUpdater
from .config_loader import AWSConfigLoader, load_aws_config, get_aws_config

__all__ = [
    'BedrockAgentClient',
    'BedrockAgentClientError',
    'AgentInvoker',
    'MarketSnapshotBuilder',
    'PnLUpdater',
    'AWSConfigLoader',
    'load_aws_config',
    'get_aws_config',
]
