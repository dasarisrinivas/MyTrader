#!/usr/bin/env python3
"""
Demo: AWS Trading Bot Agents in Action

This script demonstrates invoking the trading bot agents
for a sample trading decision workflow.
"""

import boto3
import json
import yaml
from pathlib import Path
from datetime import datetime

# Load configuration
config_path = Path(__file__).parent.parent / "config" / "deployed_resources.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Initialize client
agent_runtime = boto3.client('bedrock-agent-runtime', region_name=config['region'])

def invoke_agent(agent_key: str, prompt: str) -> str:
    """Invoke a Bedrock agent and return the response."""
    agent_config = config['agents'][agent_key]
    agent_id = agent_config['id']
    alias_id = agent_config['alias_id']
    
    print(f"\n{'='*60}")
    print(f"Agent: {agent_config['name']}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"{'='*60}")
    
    response = agent_runtime.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=f"demo-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        inputText=prompt
    )
    
    # Read streaming response
    completion = ""
    for event in response.get('completion', []):
        if 'chunk' in event:
            chunk_data = event['chunk'].get('bytes', b'')
            completion += chunk_data.decode('utf-8')
    
    print(f"\nResponse:\n{completion}")
    return completion


def demo_trading_workflow():
    """Demonstrate a full trading decision workflow."""
    
    print("\n" + "="*70)
    print("   AWS TRADING BOT - DEMO WORKFLOW")
    print("="*70)
    
    # Sample market data
    market_data = {
        "symbol": "ES",
        "timestamp": datetime.now().isoformat(),
        "price": 5950.25,
        "volume": 125000,
        "rsi": 65,
        "macd_signal": "bullish_crossover",
        "support": 5920,
        "resistance": 5980,
        "vix": 14.5,
        "trend": "upward"
    }
    
    print(f"\n📊 Market Data:")
    print(json.dumps(market_data, indent=2))
    
    # Step 1: Decision Engine - Analyze and recommend
    print("\n" + "-"*70)
    print("STEP 1: Decision Engine Analysis")
    print("-"*70)
    
    decision_prompt = f"""
    Analyze the following market data for ES futures and provide a trading recommendation:
    
    {json.dumps(market_data, indent=2)}
    
    Based on this data:
    1. Should we BUY, SELL, or WAIT?
    2. What is your confidence level (0-100)?
    3. What are the key factors supporting this decision?
    
    Respond in JSON format with: action, confidence, reasoning
    """
    
    decision_response = invoke_agent('decision_engine', decision_prompt)
    
    # Step 2: Risk Control - Validate the decision
    print("\n" + "-"*70)
    print("STEP 2: Risk Control Validation")
    print("-"*70)
    
    risk_prompt = f"""
    Evaluate the following trade proposal for risk:
    
    Market Data: {json.dumps(market_data, indent=2)}
    
    Proposed Trade:
    - Action: BUY (based on decision engine)
    - Entry: {market_data['price']}
    - Stop Loss: {market_data['support']} 
    - Take Profit: {market_data['resistance']}
    - Account Balance: $50,000
    
    Please evaluate:
    1. Is this trade within our 2% risk per trade limit?
    2. What position size is appropriate?
    3. Are there any risk concerns?
    
    Respond with: approved (true/false), position_size, risk_percentage, concerns
    """
    
    risk_response = invoke_agent('risk_control', risk_prompt)
    
    # Summary
    print("\n" + "="*70)
    print("   WORKFLOW COMPLETE")
    print("="*70)
    print(f"\n✅ Decision Engine analyzed market conditions")
    print(f"✅ Risk Control validated the trade proposal")
    print(f"\nThe full multi-agent system is operational!")
    

def demo_learning_agent():
    """Demonstrate the learning agent analyzing past trades."""
    
    print("\n" + "="*70)
    print("   LEARNING AGENT DEMO")
    print("="*70)
    
    # Sample losing trade for analysis
    losing_trade = {
        "trade_id": "T-2025-1234",
        "symbol": "ES",
        "action": "BUY",
        "entry_price": 5950,
        "exit_price": 5920,
        "pnl": -750,
        "entry_time": "2025-12-10T09:30:00",
        "exit_time": "2025-12-10T10:15:00",
        "indicators_at_entry": {
            "rsi": 72,
            "vix": 18.5,
            "volume": "below_average",
            "time_of_day": "market_open"
        }
    }
    
    learning_prompt = f"""
    Analyze this losing trade and identify patterns to avoid:
    
    {json.dumps(losing_trade, indent=2)}
    
    Please provide:
    1. What went wrong with this trade?
    2. What warning signs should we look for?
    3. What rule should we add to avoid similar losses?
    
    Format the rule as a JSON object that can be added to our rules.json file.
    """
    
    invoke_agent('learning', learning_prompt)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--learning':
        demo_learning_agent()
    else:
        demo_trading_workflow()
        
    print("\n🎉 Demo complete!")
