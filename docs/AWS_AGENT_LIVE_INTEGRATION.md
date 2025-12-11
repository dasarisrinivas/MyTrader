# AWS Agent Live Trading Integration

This document describes how to connect the deployed AWS Bedrock Agents to the live trading bot.

## Quick Start

### 1. Test Agent Integration

First, verify the agents are properly connected:

```bash
python3 aws/scripts/test_live_integration.py
```

This will run 6 tests:
1. Config Loader - Loads deployed_resources.yaml
2. AgentInvoker Initialization - Creates the agent client
3. Market Snapshot Builder - Builds market data for agents
4. Decision Agent Invocation - Tests the Decision Engine
5. Risk Agent Invocation - Tests the Risk Control agent
6. Full Trading Decision Flow - End-to-end test

### 2. Run in Test Mode (Offline)

Test the integration without connecting to IBKR:

```bash
python3 bin/run_aws_agent_trading.py --test
```

### 3. Run with IBKR Simulation

Connect to IBKR but don't place real orders:

```bash
python3 bin/run_aws_agent_trading.py --simulation
```

### 4. Enable in Main Trading Bot

To enable AWS agents in the main trading bot:

1. Edit `config.yaml`:

```yaml
aws_agents:
  enabled: true
  use_deployed_config: true
```

2. Start the bot:

```bash
python3 run_bot.py --simulation
```

The bot will automatically use AWS agents for trading decisions when enabled.

## Configuration

### config.yaml Options

```yaml
aws_agents:
  # Master enable/disable
  enabled: false  # Set to true to enable
  
  # Auto-load from deployed_resources.yaml
  use_deployed_config: true
  
  # Orchestration
  use_step_functions: false  # Use Step Functions vs direct calls
  
  # Confidence thresholds
  min_decision_confidence: 0.60
  min_risk_approval_threshold: 0.50
  
  # Trade execution
  apply_agent_decision: false  # Apply to actual trading
  require_risk_approval: true
```

### deployed_resources.yaml

The agent IDs and other AWS resources are auto-loaded from:
`aws/config/deployed_resources.yaml`

This file is generated during CloudFormation deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Live Trading Bot                          │
│                                                              │
│  ┌────────────┐    ┌──────────────────────────────────────┐ │
│  │ Market     │───▶│ AWS Agent Invoker                    │ │
│  │ Snapshot   │    │                                      │ │
│  │ Builder    │    │  ┌─────────────┐   ┌─────────────┐   │ │
│  └────────────┘    │  │ Decision    │──▶│ Risk        │   │ │
│                    │  │ Agent       │   │ Agent       │   │ │
│                    │  │ (Agent 2)   │   │ (Agent 3)   │   │ │
│                    │  └─────────────┘   └─────────────┘   │ │
│                    │         │                 │          │ │
│                    │         ▼                 ▼          │ │
│                    │  ┌─────────────────────────────────┐ │ │
│                    │  │ Knowledge Base (282 trades)     │ │ │
│                    │  │ OpenSearch Vectors              │ │ │
│                    │  └─────────────────────────────────┘ │ │
│                    └──────────────────────────────────────┘ │
│                                     │                        │
│                                     ▼                        │
│  ┌────────────┐    ┌──────────────────────────────────────┐ │
│  │ Trade      │◀───│ Final Decision                       │ │
│  │ Executor   │    │ (BUY/SELL/WAIT, Size, SL/TP)        │ │
│  └────────────┘    └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Agent Decision Flow

1. **Market Snapshot Creation**
   - Current price, RSI, ATR, EMAs, volume
   - PDH/PDL levels
   - Trend and volatility classification

2. **Decision Agent (Agent 2)**
   - Analyzes market conditions
   - Searches Knowledge Base for similar patterns
   - Returns: BUY/SELL/WAIT with confidence and reasoning

3. **Risk Agent (Agent 3)** (if not WAIT)
   - Evaluates account metrics (P&L, position, losing streak)
   - Applies position sizing rules
   - Returns: allowed_to_trade, size_multiplier, risk_flags

4. **Trade Execution**
   - Places order via IBKR if approved
   - Applies stop loss and take profit levels
   - Uploads trade data to S3 for learning

## Deployed Resources

| Resource | ID | Status |
|----------|-----|--------|
| Data Agent | VOVTP6XIZF | ✅ PREPARED |
| Decision Agent | NCZKO9CMEE | ✅ PREPARED |
| Risk Agent | YNQJHCMEUO | ✅ PREPARED |
| Learning Agent | KVXVWKKJBX | ✅ PREPARED |
| Knowledge Base | Z0EPG8YT8F | ✅ ACTIVE (13 docs) |
| S3 Bucket | trading-bot-data-897729113303 | ✅ Active |
| Signal Flow | trading-bot-dev-signal-flow | ✅ Active |

## Troubleshooting

### Agent Invocation Errors

1. Check AWS credentials:
```bash
aws sts get-caller-identity
```

2. Verify agent status:
```bash
python3 aws/scripts/start_agents.py
```

### Slow Response Times

- Decision + Risk agents typically take 5-10 seconds
- Consider enabling Step Functions for async processing
- Check CloudWatch for agent latency metrics

### Knowledge Base Not Finding Patterns

1. Verify KB status:
```bash
aws bedrock-agent get-knowledge-base --knowledge-base-id Z0EPG8YT8F
```

2. Check indexed documents:
```bash
python3 aws/scripts/demo_agents.py
```

## Files

| File | Purpose |
|------|---------|
| `mytrader/aws/agent_invoker.py` | High-level agent orchestration |
| `mytrader/aws/bedrock_agent_client.py` | Low-level Bedrock API client |
| `mytrader/aws/config_loader.py` | Loads deployed resource config |
| `mytrader/aws/market_snapshot.py` | Builds market data for agents |
| `bin/run_aws_agent_trading.py` | Standalone AWS agent trading script |
| `aws/scripts/test_live_integration.py` | Integration test suite |
| `aws/config/deployed_resources.yaml` | Deployed resource IDs |
