#!/usr/bin/env python3
"""
Start AWS Trading Bot Agents

This script:
1. Loads deployed resource configuration
2. Initializes the Bedrock Agent client
3. Tests connectivity to all agents
4. Optionally starts the trading bot with AWS integration
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 is required. Install with: pip3 install boto3")
    sys.exit(1)


def load_config(config_path: str = None) -> dict:
    """Load deployed resources configuration."""
    if config_path is None:
        config_path = PROJECT_ROOT / "aws" / "config" / "deployed_resources.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_agent(agent_runtime_client, agent_id: str, alias_id: str, agent_name: str) -> bool:
    """Test an agent with a simple ping."""
    try:
        print(f"  Testing {agent_name} (ID: {agent_id})...", end=" ")
        
        response = agent_runtime_client.invoke_agent(
            agentId=agent_id,
            agentAliasId=alias_id,
            sessionId=f"test-{agent_id}",
            inputText="Hello, please respond with OK if you're working."
        )
        
        # Read the streaming response
        completion = ""
        for event in response.get('completion', []):
            if 'chunk' in event:
                chunk_data = event['chunk'].get('bytes', b'')
                completion += chunk_data.decode('utf-8')
        
        print(f"✅ OK (response: {completion[:50]}...)" if len(completion) > 50 else f"✅ OK")
        return True
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        print(f"❌ FAILED ({error_code})")
        return False
    except Exception as e:
        print(f"❌ FAILED ({str(e)[:50]})")
        return False


def test_knowledge_base(bedrock_agent_client, kb_id: str) -> bool:
    """Test knowledge base connectivity."""
    try:
        print(f"  Testing Knowledge Base (ID: {kb_id})...", end=" ")
        
        response = bedrock_agent_client.get_knowledge_base(
            knowledgeBaseId=kb_id
        )
        
        status = response.get('knowledgeBase', {}).get('status', 'UNKNOWN')
        print(f"✅ Status: {status}")
        return status == 'ACTIVE'
        
    except Exception as e:
        print(f"❌ FAILED ({str(e)[:50]})")
        return False


def test_step_function(sfn_client, state_machine_arn: str, name: str) -> bool:
    """Test Step Function exists and is active."""
    try:
        print(f"  Testing {name}...", end=" ")
        
        response = sfn_client.describe_state_machine(
            stateMachineArn=state_machine_arn
        )
        
        status = response.get('status', 'UNKNOWN')
        print(f"✅ Status: {status}")
        return status == 'ACTIVE'
        
    except Exception as e:
        print(f"❌ FAILED ({str(e)[:50]})")
        return False


def main():
    parser = argparse.ArgumentParser(description='Start AWS Trading Bot')
    parser.add_argument('--config', '-c', help='Path to deployed_resources.yaml')
    parser.add_argument('--test-only', action='store_true', help='Only test connectivity')
    parser.add_argument('--start-bot', action='store_true', help='Start the trading bot')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    args = parser.parse_args()
    
    print("=" * 60)
    print("AWS Trading Bot - Agent Startup")
    print("=" * 60)
    print()
    
    # Load configuration
    print("Loading configuration...")
    try:
        config = load_config(args.config)
        print(f"  Environment: {config['environment']}")
        print(f"  Region: {config['region']}")
        print(f"  Account: {config['account_id']}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)
    
    # Initialize AWS clients
    print("Initializing AWS clients...")
    region = args.region or config['region']
    
    try:
        agent_runtime = boto3.client('bedrock-agent-runtime', region_name=region)
        bedrock_agent = boto3.client('bedrock-agent', region_name=region)
        sfn_client = boto3.client('stepfunctions', region_name=region)
        print("  ✅ AWS clients initialized")
        print()
    except Exception as e:
        print(f"ERROR: Failed to initialize AWS clients: {e}")
        sys.exit(1)
    
    # Test Knowledge Base
    print("Testing Knowledge Base...")
    kb_ok = test_knowledge_base(bedrock_agent, config['knowledge_base']['id'])
    print()
    
    # Test all agents
    print("Testing Bedrock Agents...")
    agents = config['agents']
    agent_results = {}
    
    for agent_key, agent_config in agents.items():
        agent_results[agent_key] = test_agent(
            agent_runtime,
            agent_config['id'],
            agent_config['alias_id'],
            agent_config['name']
        )
    print()
    
    # Test Step Functions
    print("Testing Step Functions...")
    sf_config = config['step_functions']
    signal_ok = test_step_function(sfn_client, sf_config['signal_flow'], "Signal Flow")
    nightly_ok = test_step_function(sfn_client, sf_config['nightly_flow'], "Nightly Flow")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = kb_ok and all(agent_results.values()) and signal_ok and nightly_ok
    
    print(f"  Knowledge Base: {'✅' if kb_ok else '❌'}")
    for agent_key, ok in agent_results.items():
        print(f"  Agent ({agent_key}): {'✅' if ok else '❌'}")
    print(f"  Signal Flow: {'✅' if signal_ok else '❌'}")
    print(f"  Nightly Flow: {'✅' if nightly_ok else '❌'}")
    print()
    
    if all_ok:
        print("✅ All systems operational!")
    else:
        print("⚠️  Some components have issues. Check logs above.")
    print()
    
    if args.test_only:
        sys.exit(0 if all_ok else 1)
    
    # Start bot if requested
    if args.start_bot:
        print("Starting trading bot with AWS integration...")
        print("-" * 60)
        
        # Import and configure the bot
        from mytrader.aws.bedrock_agent_client import BedrockAgentClient
        from mytrader.aws.agent_invoker import TradingAgentOrchestrator
        
        # Create client with deployed config
        client = BedrockAgentClient(
            region_name=region,
            agent_ids={
                'data': agents['data_ingestion']['id'],
                'decision': agents['decision_engine']['id'],
                'risk': agents['risk_control']['id'],
                'learning': agents['learning']['id'],
            },
            agent_alias_ids={
                'data': agents['data_ingestion']['alias_id'],
                'decision': agents['decision_engine']['alias_id'],
                'risk': agents['risk_control']['alias_id'],
                'learning': agents['learning']['alias_id'],
            }
        )
        
        # Create orchestrator
        orchestrator = TradingAgentOrchestrator(
            bedrock_client=client,
            knowledge_base_id=config['knowledge_base']['id']
        )
        
        print("✅ Trading bot initialized with AWS agents!")
        print()
        print("Available commands:")
        print("  orchestrator.get_trading_decision(market_data)")
        print("  orchestrator.run_nightly_analysis()")
        print()
        
        # Return orchestrator for interactive use
        return orchestrator
    
    return None


if __name__ == '__main__':
    main()
