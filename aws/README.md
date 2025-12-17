# AWS Multi-Agent Trading Architecture

This directory contains the AWS infrastructure for the SPY Futures Trading Bot's multi-agent system using AWS Bedrock Agents, Bedrock Knowledge Base, S3, and Lambda.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LOCAL TRADING BOT                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  bot.py → bedrock_client.py → agent_invoker.py → IB Gateway         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ HTTPS (boto3)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AWS CLOUD                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    STEP FUNCTIONS ORCHESTRATOR                         │  │
│  │  Signal Flow: Agent2 → Agent3 → Response                              │  │
│  │  Nightly Flow: Agent1 → KB Sync → Agent4 → Agent3                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                  │                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │ AGENT 1  │  │ AGENT 2  │  │ AGENT 3  │  │ AGENT 4  │                     │
│  │ Data     │  │ Decision │  │ Risk     │  │ Learning │                     │
│  │ Ingestion│  │ Engine   │  │ Control  │  │ Engine   │                     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                     │
│       │             │             │             │                            │
│       ▼             ▼             ▼             ▼                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         LAMBDA FUNCTIONS                              │   │
│  │  • clean_and_structure_trade_data                                     │   │
│  │  • calculate_winrate_statistics                                       │   │
│  │  • risk_control_engine                                                │   │
│  │  • learn_from_losses                                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                  │                                           │
│  ┌───────────────────────────────┴───────────────────────────────────────┐  │
│  │                    BEDROCK KNOWLEDGE BASE                              │  │
│  │  • Titan Embeddings v2                                                 │  │
│  │  • Vector Index (FAISS)                                                │  │
│  │  • Auto-chunking & Auto-ingestion                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                  │                                           │
│  ┌───────────────────────────────┴───────────────────────────────────────┐  │
│  │                           S3 BUCKET                                    │  │
│  │  /raw/          - Raw trade logs                                       │  │
│  │  /structured/   - Parquet + JSON feature files                         │  │
│  │  /kb/           - Knowledge Base documents                             │  │
│  │  /pnl/          - P&L tracking                                         │  │
│  │  /strategy/     - Strategy rules & configurations                      │  │
│  │  /bad_patterns/ - Patterns to avoid                                    │  │
│  │  /features/     - Computed features                                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
aws/
├── README.md                           # This file
├── cloudformation/
│   ├── main.yaml                       # Main CloudFormation stack
│   ├── s3-bucket.yaml                  # S3 bucket and folder structure
│   ├── iam-roles.yaml                  # IAM roles and policies
│   ├── bedrock-kb.yaml                 # Bedrock Knowledge Base
│   ├── bedrock-agents.yaml             # Bedrock Agents (4 agents)
│   ├── lambda-functions.yaml           # Lambda function definitions
│   ├── step-functions.yaml             # Step Functions state machines
│   └── eventbridge.yaml                # EventBridge rules for triggers
├── lambda/
│   ├── clean_and_structure_trade_data/
│   │   ├── lambda_function.py
│   │   └── requirements.txt
│   ├── calculate_winrate_statistics/
│   │   ├── lambda_function.py
│   │   └── requirements.txt
│   ├── risk_control_engine/
│   │   ├── lambda_function.py
│   │   └── requirements.txt
│   └── learn_from_losses/
│       ├── lambda_function.py
│       └── requirements.txt
├── bedrock/
│   ├── agents/
│   │   ├── agent1_data_ingestion.json
│   │   ├── agent2_decision_engine.json
│   │   ├── agent3_risk_control.json
│   │   └── agent4_learning.json
│   └── knowledge_base/
│       └── kb_config.json
├── step_functions/
│   ├── signal_flow.asl.json            # Signal processing workflow
│   └── nightly_flow.asl.json           # Nightly learning workflow
└── scripts/
    ├── deploy.sh                       # Deployment script
    ├── init_s3_folders.py              # Initialize S3 folder structure
    └── test_agents.py                  # Test agent invocations
```

## Deployment

### Prerequisites

1. AWS CLI configured with appropriate credentials
2. Python 3.12+
3. boto3, pyarrow packages

### Deploy Infrastructure

```bash
# Deploy all stacks
cd aws/scripts
./deploy.sh

# Or deploy individual stacks
aws cloudformation deploy --template-file cloudformation/main.yaml --stack-name trading-bot-infra
```

### Initialize S3 Structure

```bash
python scripts/init_s3_folders.py
```

## Agent Descriptions

### Agent 1: Data Ingestion & Feature Builder
- **Purpose**: Normalize and prepare market/trading logs
- **Trigger**: EventBridge nightly schedule
- **Tools**: clean_and_structure_trade_data Lambda, S3 write

### Agent 2: RAG + Similarity Search Decision Engine
- **Purpose**: Generate trading decisions based on historical patterns
- **Trigger**: Real-time API call from local bot
- **Tools**: Bedrock KB vector search, calculate_winrate_statistics Lambda

### Agent 3: Risk & Position Sizing Agent
- **Purpose**: Evaluate trade safety and determine position sizing
- **Trigger**: After Agent 2 decision
- **Tools**: risk_control_engine Lambda, S3 read/write

### Agent 4: Strategy Optimization & Learning Agent
- **Purpose**: Learn from mistakes and update trading rules
- **Trigger**: EventBridge nightly at 11 PM CST
- **Tools**: learn_from_losses Lambda, KB search

## Cost Estimation (Monthly)

| Service | Estimated Cost |
|---------|----------------|
| S3 Storage (10GB) | $0.23 |
| Lambda (100K invocations) | $0.20 |
| Bedrock Agents (Claude 3 Haiku) | $15-20 |
| Bedrock KB (Titan Embeddings) | $5-8 |
| Step Functions (10K executions) | $0.25 |
| EventBridge | Free tier |
| **Total** | **~$20-30/month** |

## AWS Runtime Entry Points

- **Lambda Functions** (`aws/cloudformation/lambda-functions.yaml`):
  - `clean_and_structure_trade_data` (`lambda_function.lambda_handler`) – invoked by Bedrock Agent 1 for nightly structuring.
  - `calculate_winrate_statistics` – Agent 2 action group.
  - `risk_control_engine` – Agent 3 action group and Step Functions updates.
  - `learn_from_losses` – Agent 4 Lambda now wired into Step Functions and EventBridge for nightly learning.
- **Step Functions** (`aws/cloudformation/step-functions.yaml`):
  - `signal-flow` Express machine orchestrates Agent 2 ➜ Agent 3 for live trades.
  - `nightly-flow` Standard machine now runs `Agent1 → KB wait → learn_from_losses Lambda → Agent3 risk updates`.
- **EventBridge Rules** (`aws/cloudformation/eventbridge.yaml`):
  - Nightly rule (`cron(55 5 ? * MON-FRI *)`) triggers the nightly Step Function with payload `{mode:"live", source:"eventbridge", dry_run:false}`.
  - Weekly review rule reuses the same state machine with `analysis_type:"weekly"`.
  - Raw S3 upload rule fans into the nightly state machine when new `/raw/` objects arrive.

## Agent 4 Nightly Learning Flow

1. **Schedule** – EventBridge submits the nightly payload to the `nightly-flow` state machine around 23:55 CST.
2. **Data Prep** – Step Functions invokes Bedrock Agent 1 to structure data, waits 60s for the Knowledge Base sync, then calls the `learn_from_losses` Lambda.
3. **DailyLossSummaryBuilder** – The Lambda reads `structured/YYYY-MM-DD/trades.json` and `pnl/pnl_YYYY-MM-DD.json`, filters losing trades, and builds a daily summary.
4. **Learning Analysis** – Up to 30 days of losses are scanned to extract repeated patterns, rule updates, and new “bad patterns”.
5. **Persistence** – Results are written idempotently to `s3://<bucket>/learning/YYYY-MM-DD/agent4_learning_update.json` plus `learning/strategy_state.json`.
6. **Risk Updates** – When rules were updated the state machine forwards the insights to the Agent 3 Bedrock action to refresh downstream risk controls.
7. **Telemetry** – The Lambda emits `LEARNING_AGENT_START/END` structured logs and CloudWatch metrics (`Agent4Invocations`, `Agent4Success`, `Agent4DurationMs`, `Agent4UpdatesWritten`). Two alarms track missing runs and repeated no-op updates.

## Telemetry & Verification Runbook

1. **CloudWatch Logs Insights**
   ```sql
   fields @timestamp, event, run_id, target_date, updates_written
   | filter event in ['LEARNING_AGENT_START','LEARNING_AGENT_END']
   | sort @timestamp desc
   | limit 40
   ```
   - Expect matching START/END pairs per date/run_id. Missing END events or statuses other than `SUCCESS/SKIPPED` indicate failures.

2. **CloudWatch Metrics / Alarms**
   - Namespace `MyTrader/LearningAgent` holds the four metrics. The alarms defined in `lambda-functions.yaml` raise SNS notifications when no success is recorded in 24h or when invocations repeatedly write zero updates.
   - Use the console or `aws cloudwatch get-metric-statistics --namespace MyTrader/LearningAgent --metric-name Agent4Success ...` for ad-hoc checks.

3. **S3 Validation**
   ```bash
   aws s3 ls s3://$DATA_BUCKET/learning/2025-01-16/agent4_learning_update.json
   aws s3 ls s3://$DATA_BUCKET/learning/strategy_state.json
   ```
   - Each trading day should have a dated folder with `agent4_learning_update.json`.
   - `strategy_state.json`’s `LastModified` timestamp must advance with each successful run; compare `ETag`s to confirm updates.

4. **Safe Replay / Dry Run**
   ```bash
   aws lambda invoke \
     --function-name trading-bot-dev-learn-from-losses \
     --payload '{"date":"2025-01-16","mode":"live","dry_run":true,"source":"verification","schedule_type":"manual"}' \
     out.json
   ```
   - Dry runs read the same data but skip writes, so artifacts and strategy files remain untouched.
   - Review `out.json` for identified patterns; check CloudWatch logs/metrics for a `SKIPPED` or `SUCCESS` end-state with `updates_written:0`.

## Local Integration

The `mytrader/aws/` module provides Python integration with the local trading bot:

```python
from mytrader.aws import TradingAgentOrchestrator, MarketSnapshotBuilder, PnLUpdater

# Initialize orchestrator
orchestrator = TradingAgentOrchestrator(
    region='us-east-1',
    agent_ids={
        'data_ingestion': 'AGENT1_ID',
        'decision_engine': 'AGENT2_ID',
        'risk_control': 'AGENT3_ID',
        'learning': 'AGENT4_ID'
    }
)

# Build market snapshot from IB data
snapshot_builder = MarketSnapshotBuilder(ib_client)
market_data = await snapshot_builder.build_snapshot()

# Get trading decision via multi-agent flow
decision = await orchestrator.get_trading_decision(
    market_snapshot=market_data,
    account_state=account_info
)

# Execute if approved
if decision['approved']:
    # Trade via IB Gateway
    order = create_order(decision['action'], decision['position_size'])
```

## Configuration

Add to your `config.yaml`:

```yaml
aws:
  region: us-east-1
  bedrock:
    agents:
      data_ingestion: "AGENT1_ID"
      decision_engine: "AGENT2_ID"
      risk_control: "AGENT3_ID"
      learning: "AGENT4_ID"
    knowledge_base_id: "KB_ID"
  s3:
    bucket: "trading-bot-prod-123456789012"
  step_functions:
    signal_flow_arn: "arn:aws:states:..."
    nightly_batch_arn: "arn:aws:states:..."
```

### Regime & Risk Controls

| Setting | Description | Default / Override |
|---------|-------------|--------------------|
| `rag.min_similar_trades` | Minimum number of similar historical trades required before the LLM stage is called. If the retrieval count is below this threshold the Hybrid pipeline holds the trade to avoid acting in unfamiliar regimes. | `3` (override with `MIN_SIMILAR_TRADES`) |
| `rag.min_weighted_win_rate` | Lowest acceptable weighted win rate from retrieved trades. Falling below this ratio triggers a HOLD with a detailed reasoning string so you know the regime guard tripped. | `0.5` (override with `MIN_WEIGHTED_WIN_RATE`) |
| `trading.confidence_threshold` | Confidence score needed to take full position size. RiskManager automatically halves the contracts below this value, so reducing it makes the bot more aggressive while increasing it enforces smaller feeler positions until conviction rises. | `0.7` (override with `CONFIDENCE_THRESHOLD`) |

These knobs power the new regime filter, confirmation trigger, and dynamic risk sizing features. The defaults live in `config.yaml`, but you can tune them per session using environment variables before running `. start_bot.sh`. The start script now forwards `MIN_SIMILAR_TRADES`, `MIN_WEIGHTED_WIN_RATE`, and `CONFIDENCE_THRESHOLD` into the Python process so `load_settings()` and `HybridRAGPipeline` stay in sync with your launch parameters.

## Signal Flow (Real-time)

```
Market Data ──▶ Agent 2 (Decision) ──▶ Agent 3 (Risk) ──▶ Execute Trade
                     │                      │
                     ▼                      ▼
              Query Knowledge Base    Validate Against
              for Similar Patterns    Risk Rules
```

## Nightly Batch Flow

```
8 PM EST
    │
    ▼
┌───────────────────┐
│     Agent 1       │
│  Data Ingestion   │
│  • Process raw    │
│  • Build features │
│  • Update KB      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│     Agent 4       │
│    Learning       │
│  • Analyze losses │
│  • Find patterns  │
│  • Update rules   │
└───────────────────┘
```

## Troubleshooting

### Agent not responding
1. Check CloudWatch logs for errors
2. Verify agent is in PREPARED status
3. Ensure Lambda permissions are correct

### Knowledge Base not finding patterns
1. Verify `/structured/` has Parquet files
2. Check data source sync status
3. Validate embedding model is working

### Risk agent blocking all trades
1. Check daily P&L in `/pnl/`
2. Verify losing streak count
3. Review `rules.json` for overly restrictive rules

## Security Notes

- All IAM roles follow least-privilege principle
- S3 bucket has server-side encryption enabled
- Lambda functions run in VPC (optional)
- Bedrock model access restricted to specific models
- CloudTrail logging for audit
- No credentials stored in code
