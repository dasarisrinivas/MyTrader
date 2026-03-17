#!/usr/bin/env python3
"""
Test Bedrock Agents

This script tests the Bedrock Agent invocations with sample data.

Usage:
    python test_agents.py --config <config-file>
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("Error: boto3 is required. Install with: pip install boto3")
    sys.exit(1)


# Sample market snapshot for testing
SAMPLE_MARKET_SNAPSHOT = {
    "symbol": "ES",
    "price": 5950.25,
    "bid": 5950.00,
    "ask": 5950.50,
    "volume": 15000,
    "trend": "UPTREND",
    "volatility": "MED",
    "rsi": 55.3,
    "macd": 2.5,
    "macd_histogram": 0.8,
    "ema_9": 5945.50,
    "ema_20": 5940.25,
    "atr": 1.8,
    "PDH_delta": 0.15,
    "PDL_delta": 0.85,
    "time_of_day": "MORNING"
}

# Sample account metrics
SAMPLE_ACCOUNT_METRICS = {
    "current_pnl_today": 125.50,
    "current_position": 1,
    "losing_streak": 0,
    "trades_today": 3,
    "account_balance": 50000,
    "open_risk": 100
}


def test_agent(
    agent_id: str,
    agent_alias_id: str,
    input_text: str,
    region: str = 'us-east-1'
) -> dict:
    """Test a single Bedrock Agent."""
    client = boto3.client('bedrock-agent-runtime', region_name=region)
    
    try:
        print(f"\n{'='*60}")
        print(f"Testing Agent: {agent_id}")
        print(f"Input: {input_text[:100]}...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id or 'TSTALIASID',
            sessionId=f"test-{int(time.time())}",
            inputText=input_text,
            enableTrace=True
        )
        
        # Process streaming response
        completion = ""
        for event in response.get('completion', []):
            if 'chunk' in event:
                chunk_data = event['chunk']
                if 'bytes' in chunk_data:
                    completion += chunk_data['bytes'].decode('utf-8')
        
        elapsed = time.time() - start_time
        
        print(f"\nResponse ({elapsed:.2f}s):")
        print("-" * 40)
        print(completion[:500] if len(completion) > 500 else completion)
        if len(completion) > 500:
            print("...")
        print("-" * 40)
        
        return {
            'status': 'success',
            'agent_id': agent_id,
            'latency_ms': int(elapsed * 1000),
            'response_length': len(completion),
            'response': completion
        }
        
    except ClientError as e:
        error_msg = str(e)
        print(f"\n❌ Error: {error_msg}")
        return {
            'status': 'error',
            'agent_id': agent_id,
            'error': error_msg
        }


def test_decision_agent(config: dict, region: str) -> dict:
    """Test the Decision Engine Agent (Agent 2)."""
    agent_id = config.get('agent2_id')
    agent_alias = config.get('agent2_alias_id')
    
    if not agent_id:
        return {'status': 'skipped', 'reason': 'agent2_id not configured'}
    
    prompt = f"""Analyze this market snapshot and provide a trading decision:

Market Snapshot:
{json.dumps(SAMPLE_MARKET_SNAPSHOT, indent=2)}

Please:
1. Search the knowledge base for similar historical patterns
2. Calculate win-rate from similar trades
3. Provide a BUY/SELL/WAIT decision
4. Include confidence level and reasoning
5. Suggest stop-loss and take-profit levels

Return as structured JSON."""

    return test_agent(agent_id, agent_alias, prompt, region)


def test_risk_agent(config: dict, region: str) -> dict:
    """Test the Risk Control Agent (Agent 3)."""
    agent_id = config.get('agent3_id')
    agent_alias = config.get('agent3_alias_id')
    
    if not agent_id:
        return {'status': 'skipped', 'reason': 'agent3_id not configured'}
    
    trade_decision = {
        "action": "BUY",
        "confidence": 0.72,
        "symbol": "ES",
        "proposed_size": 1
    }
    
    market_conditions = {
        "volatility": "MED",
        "regime": "UPTREND",
        "atr": 1.8,
        "vix": 18.5
    }
    
    prompt = f"""Evaluate this trade decision for risk:

Trade Decision:
{json.dumps(trade_decision, indent=2)}

Account Metrics:
{json.dumps(SAMPLE_ACCOUNT_METRICS, indent=2)}

Market Conditions:
{json.dumps(market_conditions, indent=2)}

Please evaluate and provide:
1. Whether this trade is allowed (true/false)
2. Position size multiplier (0-1)
3. Adjusted position size
4. Any risk flags or warnings
5. Reasoning

Return as structured JSON."""

    return test_agent(agent_id, agent_alias, prompt, region)


def test_learning_agent(config: dict, region: str) -> dict:
    """Test the Learning Agent (Agent 4)."""
    agent_id = config.get('agent4_id')
    agent_alias = config.get('agent4_alias_id')
    
    if not agent_id:
        return {'status': 'skipped', 'reason': 'agent4_id not configured'}
    
    sample_losses = [
        {
            "trade_id": "loss-001",
            "action": "BUY",
            "regime": "DOWNTREND",
            "volatility": "HIGH",
            "pnl": -75.50,
            "rsi": 28,
            "time_of_day": "MORNING"
        },
        {
            "trade_id": "loss-002",
            "action": "BUY",
            "regime": "DOWNTREND",
            "volatility": "HIGH",
            "pnl": -62.25,
            "rsi": 32,
            "time_of_day": "MORNING"
        },
        {
            "trade_id": "loss-003",
            "action": "BUY",
            "regime": "DOWNTREND",
            "volatility": "MED",
            "pnl": -45.00,
            "rsi": 35,
            "time_of_day": "MIDDAY"
        }
    ]
    
    prompt = f"""Analyze these losing trades and identify patterns:

Losing Trades:
{json.dumps(sample_losses, indent=2)}

Please:
1. Identify common patterns in these losses
2. Determine what conditions led to losses
3. Suggest rule updates to avoid these patterns
4. Add to bad patterns list if warranted

Return as structured JSON with:
- patterns_identified
- rules_updated
- bad_patterns_added
- recommendations"""

    return test_agent(agent_id, agent_alias, prompt, region)


def main():
    parser = argparse.ArgumentParser(description='Test Bedrock Agents')
    parser.add_argument(
        '--config',
        help='Path to config file with agent IDs'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region'
    )
    parser.add_argument(
        '--agent2-id',
        help='Decision Agent ID'
    )
    parser.add_argument(
        '--agent3-id',
        help='Risk Agent ID'
    )
    parser.add_argument(
        '--agent4-id',
        help='Learning Agent ID'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                    import yaml
                    config = yaml.safe_load(f).get('aws_agents', {})
                else:
                    config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    # Override with command line args
    if args.agent2_id:
        config['agent2_id'] = args.agent2_id
    if args.agent3_id:
        config['agent3_id'] = args.agent3_id
    if args.agent4_id:
        config['agent4_id'] = args.agent4_id
    
    print("=" * 60)
    print("Trading Bot Agent Tests")
    print("=" * 60)
    print(f"Region: {args.region}")
    print(f"Config: {config}")
    
    results = []
    
    # Test Decision Agent
    print("\n\n🎯 Testing Decision Agent (Agent 2)...")
    result = test_decision_agent(config, args.region)
    results.append(('Decision Agent', result))
    
    # Test Risk Agent
    print("\n\n🛡️ Testing Risk Agent (Agent 3)...")
    result = test_risk_agent(config, args.region)
    results.append(('Risk Agent', result))
    
    # Test Learning Agent
    print("\n\n📚 Testing Learning Agent (Agent 4)...")
    result = test_learning_agent(config, args.region)
    results.append(('Learning Agent', result))
    
    # Summary
    print("\n\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, result in results:
        status = result.get('status', 'unknown')
        if status == 'success':
            latency = result.get('latency_ms', 0)
            print(f"✅ {name}: SUCCESS ({latency}ms)")
        elif status == 'skipped':
            reason = result.get('reason', '')
            print(f"⏭️  {name}: SKIPPED ({reason})")
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ {name}: FAILED ({error[:50]})")


if __name__ == '__main__':
    main()
