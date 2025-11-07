# AWS Bedrock LLM Integration Guide

## Overview

MyTrader now includes AWS Bedrock LLM integration that enhances trading decisions with artificial intelligence. The LLM analyzes market conditions, technical indicators, and sentiment data to provide intelligent trade recommendations.

## Features

### 🤖 Intelligent Trade Recommendations
- **Real-time Analysis**: LLM evaluates market conditions before each trade
- **Multi-factor Reasoning**: Considers technical indicators, sentiment, and risk metrics
- **Confidence Scoring**: Each recommendation includes a confidence level (0.0-1.0)
- **Structured Responses**: JSON-formatted recommendations with reasoning and key factors

### 📊 Continuous Learning
- **Trade Logging**: All trades stored with LLM predictions and actual outcomes
- **Performance Tracking**: Win rate, profit factor, and accuracy metrics
- **Training Pipeline**: Periodic fine-tuning using historical trade data
- **S3 Integration**: Automated data storage for model retraining

### 🎯 Multiple Operating Modes
1. **Consensus Mode** (Default): LLM and traditional signals must agree
2. **Override Mode**: LLM can override traditional strategy signals
3. **Advisory Mode**: LLM provides recommendations without affecting execution
4. **Disabled**: Traditional strategies work unchanged

### 📈 Enhanced Risk Management
- **Position Sizing**: LLM suggests optimal position sizes
- **Dynamic Stops**: Recommended stop-loss and take-profit levels
- **Risk Assessment**: Detailed risk analysis for each trade
- **Confidence Filtering**: Only execute trades above confidence threshold

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Trading System Flow                          │
└─────────────────────────────────────────────────────────────────┘

    Market Data → Feature Engineering → Traditional Strategy
                                              ↓
                                        Signal Generated
                                              ↓
                    ┌─────────────────────────┴─────────────────────┐
                    │                                               │
                    ▼                                               ▼
          [LLM Enhancement Disabled]                    [LLM Enhancement Enabled]
                    │                                               │
                    │                                               ▼
                    │                              ┌────────────────────────────┐
                    │                              │   Build Trading Context    │
                    │                              │  - Technical Indicators    │
                    │                              │  - Sentiment Data          │
                    │                              │  - Risk Metrics            │
                    │                              └────────────┬───────────────┘
                    │                                           │
                    │                                           ▼
                    │                              ┌────────────────────────────┐
                    │                              │  Query AWS Bedrock LLM     │
                    │                              │  - Claude 3 / Titan        │
                    │                              │  - Get Recommendation      │
                    │                              └────────────┬───────────────┘
                    │                                           │
                    │                                           ▼
                    │                              ┌────────────────────────────┐
                    │                              │   Trade Advisor Logic      │
                    │                              │  - Consensus / Override    │
                    │                              │  - Confidence Filtering    │
                    │                              └────────────┬───────────────┘
                    │                                           │
                    └───────────────────┬───────────────────────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │   Enhanced Signal     │
                            │   with LLM Metadata   │
                            └───────────┬───────────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │   Risk Management     │
                            └───────────┬───────────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │   Execute Trade       │
                            └───────────┬───────────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │   Log to Database     │
                            │  - Trade Details      │
                            │  - LLM Prediction     │
                            │  - Actual Outcome     │
                            └───────────────────────┘
```

## Installation

### 1. Install Dependencies

```bash
pip install boto3 botocore
```

Dependencies are already included in `requirements.txt`:
```
boto3>=1.34.0
botocore>=1.34.0
```

### 2. Configure AWS Credentials

Set up AWS credentials for Bedrock access:

```bash
# Option 1: Using AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### 3. Request Bedrock Model Access

**Good News!** As of 2024, AWS Bedrock automatically enables access to all serverless foundation models. No manual approval needed!

**Available Models:**
- **Claude 3 Sonnet**: `anthropic.claude-3-sonnet-20240229-v1:0` (recommended)
- **Claude 3 Haiku**: `anthropic.claude-3-haiku-20240307-v1:0` (faster/cheaper)
- **Claude 3.5 Sonnet**: `anthropic.claude-3-5-sonnet-20240620-v1:0` (most advanced)

**First-time Anthropic Users:**
- Some users may need to submit use case details on first access
- Visit the [Bedrock Model Catalog](https://console.aws.amazon.com/bedrock/home#/model-catalog)
- Select a model and open it in the playground
- Complete any required use case information

**Optional Access Control:**
- Use IAM policies to restrict model access as needed
- Configure Service Control Policies for organization-wide restrictions
- See [IAM policies documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html)

## Configuration

### Basic Configuration

Add to your `config.yaml`:

```yaml
llm:
  # Enable/disable LLM enhancement
  enabled: true
  
  # AWS Bedrock settings
  model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
  region_name: "us-east-1"
  max_tokens: 2048
  temperature: 0.3  # Lower = more deterministic
  
  # Trade advisor settings
  min_confidence_threshold: 0.7  # Only execute if confidence >= 0.7
  override_mode: false  # false = consensus, true = LLM override
  
  # Training settings
  s3_bucket: "your-trading-data-bucket"
  s3_prefix: "llm-training-data"
  retrain_interval_days: 7
```

### Strategy Configuration

Use the LLM-enhanced strategy in your strategy list:

```yaml
strategies:
  - name: llm_enhanced
    enabled: true
    params:
      enable_llm: true
      min_llm_confidence: 0.7
      llm_override_mode: false
      # Base strategy will use RsiMacdSentimentStrategy by default
```

## Usage Examples

### Example 1: Basic LLM Enhancement (Consensus Mode)

```python
from mytrader.strategies.llm_enhanced_strategy import LLMEnhancedStrategy

# Create LLM-enhanced strategy
strategy = LLMEnhancedStrategy(
    enable_llm=True,
    min_llm_confidence=0.7,
    llm_override_mode=False,  # Consensus mode
)

# Generate signal with LLM enhancement
signal = strategy.generate(features_df)

print(f"Action: {signal.action}")
print(f"Confidence: {signal.confidence:.2f}")
print(f"LLM Reasoning: {signal.metadata.get('llm_reasoning')}")
```

### Example 2: LLM Override Mode

```python
# LLM can override traditional signals
strategy = LLMEnhancedStrategy(
    enable_llm=True,
    min_llm_confidence=0.75,
    llm_override_mode=True,  # LLM has final say
)

signal = strategy.generate(features_df)
```

### Example 3: Direct LLM Query

```python
from mytrader.llm.bedrock_client import BedrockClient
from mytrader.llm.data_schema import TradingContext
from datetime import datetime

# Initialize client
client = BedrockClient(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)

# Build context
context = TradingContext(
    symbol="ES",
    current_price=4950.0,
    timestamp=datetime.now(),
    rsi=35.0,
    macd=0.5,
    macd_signal=0.3,
    macd_hist=0.2,
    atr=10.0,
    sentiment_score=0.2,
)

# Get recommendation
recommendation = client.get_trade_recommendation(context)

print(f"Decision: {recommendation.trade_decision}")
print(f"Confidence: {recommendation.confidence:.2f}")
print(f"Reasoning: {recommendation.reasoning}")
print(f"Key Factors: {recommendation.key_factors}")
```

### Example 4: Logging Trades for Training

```python
from mytrader.llm.trade_logger import TradeLogger
from mytrader.llm.data_schema import TradeOutcome

# Initialize logger
logger = TradeLogger()

# Log trade entry
outcome = TradeOutcome(
    order_id=12345,
    symbol="ES",
    timestamp=datetime.now(),
    action="BUY",
    quantity=2,
    entry_price=4950.0,
    entry_context=context,
)

trade_id = logger.log_trade_entry(outcome, llm_recommendation)

# Later... log trade exit
logger.update_trade_exit(
    order_id=12345,
    exit_price=4960.0,
    realized_pnl=500.0,
    trade_duration_minutes=15.0,
    outcome="WIN",
)

# View performance
summary = logger.get_performance_summary(days=30)
print(f"Win Rate: {summary['win_rate']:.2%}")
print(f"Total P&L: ${summary['total_pnl']:.2f}")
```

## Training Pipeline

### Running the Training Pipeline

The training pipeline collects trade data and prepares it for model fine-tuning:

```bash
# Basic usage - export training data
python scripts/train_llm.py \
    --s3-bucket your-trading-data-bucket \
    --days 30

# Full pipeline - export and create fine-tuning job
python scripts/train_llm.py \
    --s3-bucket your-trading-data-bucket \
    --days 30 \
    --create-job \
    --base-model anthropic.claude-3-sonnet-20240229-v1:0
```

### Automated Retraining Schedule

Use cron or AWS EventBridge to schedule periodic retraining:

```bash
# Add to crontab for weekly retraining
0 2 * * 0 cd /path/to/MyTrader && python scripts/train_llm.py --s3-bucket your-bucket --days 30 >> logs/training.log 2>&1
```

### Training Data Format

The training pipeline exports data in JSONL format:

```json
{
  "input": {
    "market_data": {"symbol": "ES", "price": 4950.0, "timestamp": "2024-01-01T12:00:00"},
    "technical_indicators": {"rsi": 35.0, "macd": 0.5, "atr": 10.0},
    "sentiment": {"score": 0.2, "sources": {}},
    "risk_metrics": {"portfolio_heat": 0.15, "daily_pnl": 250.0}
  },
  "llm_prediction": {
    "trade_decision": "BUY",
    "confidence": 0.85,
    "reasoning": "Strong oversold signal"
  },
  "actual_outcome": {
    "outcome": "WIN",
    "realized_pnl": 500.0,
    "exit_price": 4960.0
  },
  "label": {
    "correct_prediction": true,
    "pnl": 500.0
  }
}
```

## Sentiment Integration

### AWS Comprehend Integration

```python
from mytrader.llm.sentiment_aggregator import SentimentAggregator

# Initialize aggregator
aggregator = SentimentAggregator(
    region_name="us-east-1",
    enable_comprehend=True
)

# Analyze sentiment from multiple sources
sentiment_score = aggregator.aggregate_sentiment(
    news_headlines=[
        "S&P 500 rallies on strong earnings",
        "Fed signals pause in rate hikes"
    ],
    social_media_posts=[
        "Bullish on tech stocks",
        "Market looking strong"
    ],
    existing_sentiment=0.2
)

print(f"Aggregated Sentiment: {sentiment_score:.3f}")
```

### Adding Custom Sentiment Sources

Extend the sentiment aggregator to include additional sources:

```python
# Example: Integrate with NewsAPI
def get_news_sentiment(symbol: str, api_key: str) -> float:
    aggregator = SentimentAggregator()
    
    # Fetch news (pseudocode)
    headlines = fetch_news_headlines(symbol, api_key)
    
    # Aggregate sentiment
    return aggregator.aggregate_sentiment(news_headlines=headlines)
```

## Performance Monitoring

### Viewing LLM Performance

```python
from mytrader.llm.trade_logger import TradeLogger

logger = TradeLogger()

# Get performance summary
summary = logger.get_performance_summary(days=30)

print("=== LLM Performance (Last 30 Days) ===")
print(f"Total Trades: {summary['total_trades']}")
print(f"Win Rate: {summary['win_rate']:.2%}")
print(f"Total P&L: ${summary['total_pnl']:.2f}")
print(f"Avg Win: ${summary['avg_win']:.2f}")
print(f"Avg Loss: ${summary['avg_loss']:.2f}")
print(f"Profit Factor: {summary['profit_factor']:.2f}")

# Get recent trades
recent_trades = logger.get_recent_trades(limit=10)
for trade in recent_trades:
    print(f"{trade['timestamp']}: {trade['action']} @ ${trade['entry_price']:.2f}")
    print(f"  LLM Decision: {trade['trade_decision']} (confidence: {trade['confidence']:.2f})")
    print(f"  Outcome: {trade['outcome']} | P&L: ${trade['realized_pnl']:.2f}")
```

### Exporting Data for Analysis

```python
# Export training data for external analysis
logger.export_training_data("data/llm_training_export.json")
```

## Cost Optimization

### Bedrock Pricing (Approximate)

**Claude 3 Sonnet:**
- Input: $0.003 per 1K tokens
- Output: $0.015 per 1K tokens

**Typical Trade Analysis:**
- Input: ~1,500 tokens (prompt with context)
- Output: ~500 tokens (recommendation)
- Cost per query: ~$0.012

**Daily Trading (20 trades/day):**
- 20 trades × $0.012 = $0.24/day
- Monthly cost: ~$7.20

### Cost Reduction Strategies

1. **Batch Processing**: Query LLM less frequently during low-volatility periods
2. **Use Haiku Model**: Claude 3 Haiku is 60% cheaper for less critical decisions
3. **Confidence Caching**: Skip LLM if traditional signal has very high/low confidence
4. **Async Processing**: Process multiple signals in parallel

```python
# Example: Conditional LLM usage
if abs(traditional_signal.confidence - 0.5) < 0.2:
    # Signal is uncertain - consult LLM
    enhanced_signal = advisor.enhance_signal(traditional_signal, context)
else:
    # Signal is strong - use traditional
    enhanced_signal = traditional_signal
```

## Troubleshooting

### Issue: "boto3 not installed"
```bash
pip install boto3 botocore
```

### Issue: "Access denied to model"
- ✅ **Good news:** Model access is now automatic for all Bedrock models
- First-time Anthropic users may need to submit use case details
- Visit [Bedrock Model Catalog](https://console.aws.amazon.com/bedrock/home#/model-catalog)
- Select your model and complete any required information
- Verify IAM permissions include `bedrock:InvokeModel` action

### Issue: "LLM returning None"
- Check AWS credentials are configured correctly
- Verify model ID is correct
- Check CloudWatch logs for API errors
- Ensure sufficient permissions on IAM role

### Issue: "Low confidence recommendations"
- Temperature too high → reduce to 0.2-0.3
- Insufficient market data → ensure all indicators calculated
- Model needs fine-tuning → run training pipeline

### Issue: "Slow response times"
- Use Claude 3 Haiku instead of Sonnet
- Reduce max_tokens parameter
- Consider caching recent recommendations

## Best Practices

### 1. Start Conservative
- Begin with `llm_override_mode=False` (consensus mode)
- Set high confidence threshold (0.75-0.80)
- Monitor performance for 1-2 weeks before loosening

### 2. Regular Fine-Tuning
- Retrain weekly or bi-weekly
- Use at least 100 completed trades for training
- Monitor accuracy trends over time

### 3. Risk Management
- Never bypass traditional risk limits
- Use LLM confidence as additional filter, not replacement
- Monitor portfolio heat and daily loss limits

### 4. Testing
- Test in paper trading first
- Compare LLM vs non-LLM performance
- Track both accuracy and profitability

### 5. Monitoring
- Log all LLM responses
- Track reasoning patterns
- Alert on low confidence periods
- Monitor API costs

## Advanced Features

### Custom Base Strategies

Wrap any existing strategy with LLM enhancement:

```python
from mytrader.strategies.momentum_reversal import MomentumReversalStrategy

custom_strategy = LLMEnhancedStrategy(
    base_strategy=MomentumReversalStrategy(),
    enable_llm=True,
    min_llm_confidence=0.75
)
```

### Dynamic Configuration

Update LLM settings at runtime:

```python
strategy.update_config(
    min_llm_confidence=0.8,  # Raise threshold
    llm_override_mode=True,   # Enable override
    enable_llm=True           # Ensure enabled
)
```

### Multiple LLM Models

Compare different models:

```python
# Sonnet for important decisions
sonnet = BedrockClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")

# Haiku for quick checks
haiku = BedrockClient(model_id="anthropic.claude-3-haiku-20240307-v1:0")

# Use based on market conditions
if volatility > threshold:
    client = sonnet  # More sophisticated analysis
else:
    client = haiku   # Faster, cheaper
```

## Support

For issues or questions:
1. Check logs in `data/llm_trades.db` for trade history
2. Review CloudWatch logs for AWS API errors
3. Test with `enable_llm=False` to isolate issues
4. Verify AWS Bedrock permissions

## References

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Claude 3 Model Guide](https://docs.anthropic.com/claude/docs)
- [Boto3 Bedrock Reference](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime.html)

---

**Last Updated**: November 6, 2024
