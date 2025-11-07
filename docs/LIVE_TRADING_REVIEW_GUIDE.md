# Live Paper Trading Review System Guide

## Overview

The Live Paper Trading Review System automatically analyzes your IBKR paper trading performance, generates AI-powered insights, and produces comprehensive reports for human review. This system is designed to run daily after market close, providing actionable recommendations without automatically applying changes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LIVE TRADING REVIEW SYSTEM                  │
└─────────────────────────────────────────────────────────────────┘

1. DATA COLLECTION
   ├── IBKR Paper Trading → Trade Logger
   ├── SQLite Database (llm_trades.db)
   └── CSV Logs (logs/trades.csv)

2. ANALYSIS PIPELINE
   ├── LiveTradeAnalyzer
   │   ├── Load trades (last N days)
   │   ├── Calculate metrics (win rate, P&L, Sharpe, etc.)
   │   └── Breakdown by signal, hour, sentiment
   │
   └── AIInsightGenerator
       ├── Generate comprehensive analysis prompt
       ├── Call AWS Bedrock Claude 3 Sonnet
       └── Parse structured insights (JSON)

3. REPORTING
   ├── JSON Report (machine-readable)
   ├── Markdown Report (human-readable)
   └── Combined Summary Report

4. HUMAN REVIEW
   ├── Review insights and recommendations
   ├── Evaluate warnings and risks
   └── Manually apply parameter changes (if desired)
```

## Key Components

### 1. LiveTradeAnalyzer (`mytrader/llm/trade_analyzer.py`)

**Purpose**: Load and analyze recent paper trading performance

**Key Methods**:
- `load_trades_from_db()` - Load trades from SQLite database
- `load_trades_from_csv()` - Load trades from CSV logs
- `compute_summary()` - Calculate comprehensive metrics
- `analyze_recent_performance()` - Main analysis entry point

**Metrics Calculated**:
- Total trades, open/closed positions
- Win rate, profit factor, max drawdown
- Average profit/loss per trade
- Sharpe ratio, Sortino ratio
- Breakdown by signal type
- Breakdown by trading hour
- LLM signal accuracy (if available)

### 2. AIInsightGenerator (`mytrader/llm/ai_insights.py`)

**Purpose**: Generate AI-powered insights using AWS Bedrock

**Key Methods**:
- `generate_analysis_prompt()` - Create comprehensive LLM prompt
- `parse_llm_response()` - Extract structured insights from JSON
- `generate_insights()` - Full insight generation pipeline
- `save_report()` - Save JSON and Markdown reports

**Insight Types**:
- **Observations**: Patterns detected in trading behavior
- **Behavioral Patterns**: Overtrading, timing issues, signal biases
- **Market Trends**: Market condition analysis
- **Recommendations**: Specific parameter adjustments
- **Warnings**: Risk alerts and concerns

### 3. DailyReviewOrchestrator (`mytrader/llm/daily_review.py`)

**Purpose**: Coordinate complete daily review workflow

**Workflow**:
1. Analyze recent performance (LiveTradeAnalyzer)
2. Generate AI insights (AIInsightGenerator)
3. Save reports (JSON + Markdown)
4. Create combined summary
5. Log results

## Setup Instructions

### 1. Configure IBKR Trade Logging

During live paper trading, ensure trades are logged to the database:

```python
from mytrader.llm.trade_logger import TradeLogger

# Initialize logger
trade_logger = TradeLogger()

# Log each trade
trade_logger.log_trade(
    timestamp=datetime.now(),
    symbol="ES",
    action="BUY",
    quantity=2,
    price=4500.25,
    signal_type="rsi_macd_sentiment",
    llm_confidence=0.75,
    llm_reasoning="Strong bullish sentiment with oversold RSI",
    position_id="POS_123"
)

# Log trade exit
trade_logger.log_trade_exit(
    position_id="POS_123",
    exit_timestamp=datetime.now(),
    exit_price=4510.50,
    pnl=512.50,
    exit_reason="Take profit hit"
)
```

### 2. Configure Settings in `config.yaml`

```yaml
# Live Paper Trading Review Configuration
live_review:
  enabled: true
  run_time: "18:00"  # Daily at 6:00 PM (after market close)
  analysis_days: 3  # Analyze last 3 days
  
  use_database: true  # Use SQLite database
  csv_log_path: "logs/trades.csv"
  
  generate_json: true
  generate_markdown: true
  reports_dir: "reports/daily_reviews"
  
  enable_ai_insights: true
  insight_confidence_threshold: 0.6
```

### 3. Manual Execution

#### Run Daily Review

```bash
# Basic review (last 3 days)
python run_daily_review.py

# Review last 7 days
python run_daily_review.py --days 7

# Use CSV logs instead of database
python run_daily_review.py --csv

# JSON report only (skip markdown)
python run_daily_review.py --no-markdown
```

#### Command Options

```
--days N              Number of days to analyze (default: 3)
--csv                 Use CSV logs instead of database
--no-json             Skip JSON report generation
--no-markdown         Skip Markdown report generation
--reports-dir PATH    Custom directory for reports
--verbose, -v         Enable verbose output
```

### 4. Scheduled Execution

#### Linux/macOS (Cron)

```bash
# Edit crontab
crontab -e

# Add line for daily execution at 6:00 PM
0 18 * * * cd /path/to/MyTrader && /path/to/python run_daily_review.py

# Example with virtual environment
0 18 * * * cd /path/to/MyTrader && /path/to/venv/bin/python run_daily_review.py
```

#### Windows (Task Scheduler)

1. Open Task Scheduler
2. Create Basic Task
3. Name: "MyTrader Daily Review"
4. Trigger: Daily at 6:00 PM
5. Action: Start a program
   - Program: `python.exe` (or full path to Python)
   - Arguments: `run_daily_review.py`
   - Start in: `C:\path\to\MyTrader`
6. Finish and test

## Report Formats

### JSON Report (`ai_insights_YYYY-MM-DD.json`)

Machine-readable structured format:

```json
{
  "timestamp": "2025-01-15T18:05:23.123456",
  "analysis_period": {
    "start": "2025-01-12",
    "end": "2025-01-15",
    "days": 3
  },
  "summary_text": "Analysis of 45 trades over 3 days...",
  "observations": [
    {
      "type": "pattern",
      "category": "timing",
      "description": "High win rate between 10:00-12:00 EST",
      "severity": "info",
      "confidence": 0.85,
      "reasoning": "Observed 12/15 winning trades during this window"
    }
  ],
  "recommendations": [
    {
      "parameter": "rsi_buy",
      "current_value": 35.84,
      "suggested_value": 32.0,
      "reasoning": "RSI buy signal too late, missing early entries",
      "expected_impact": "Improve entry timing and win rate",
      "confidence": 0.75,
      "estimated_risk": "low"
    }
  ],
  "warnings": [
    "Low win rate during 14:00-16:00 EST - consider avoiding late afternoon trades"
  ]
}
```

### Markdown Report (`daily_review_YYYY-MM-DD.md`)

Human-readable format with sections:

1. **Executive Summary**
   - Period analyzed
   - Total trades
   - Win rate, P&L, profit factor
   - Quick assessment

2. **Key Observations**
   - Pattern detection
   - Signal performance analysis
   - Timing analysis

3. **Behavioral Patterns**
   - Overtrading detection
   - Emotional trading patterns
   - Consistency issues

4. **Market Trends**
   - Market condition assessment
   - Volatility analysis
   - Trend alignment

5. **Recommendations**
   - Parameter adjustments
   - Strategy modifications
   - Risk management updates

6. **Warnings**
   - Critical issues
   - Risk alerts
   - Performance concerns

7. **Next Steps**
   - Action items
   - Testing recommendations
   - Review schedule

### Combined Report (`daily_review_combined_YYYY-MM-DD.json`)

Includes both performance summary and AI insights:

```json
{
  "timestamp": "2025-01-15T18:05:23.123456",
  "period": {
    "start": "2025-01-12",
    "end": "2025-01-15",
    "days": 3
  },
  "performance_summary": {
    "total_trades": 45,
    "win_rate": 0.667,
    "total_pnl": 2450.75,
    "profit_factor": 2.34,
    "max_drawdown": -675.00,
    "sharpe_ratio": 1.82,
    "trades_by_signal": {
      "rsi_macd_sentiment": 40,
      "llm_enhanced": 5
    },
    "pnl_by_signal": {
      "rsi_macd_sentiment": 2100.50,
      "llm_enhanced": 350.25
    }
  },
  "ai_insights": { /* ... */ },
  "report_files": [
    "reports/ai_insights/ai_insights_2025-01-15.json",
    "reports/ai_insights/daily_review_2025-01-15.md"
  ]
}
```

## Workflow Example

### Daily Workflow (Automated)

```
16:00 EST - Market closes
18:00 EST - Scheduled review runs
          ↓
1. System loads last 3 days of trades
2. Calculates performance metrics
3. Generates AI insights via AWS Bedrock
4. Saves JSON + Markdown reports
5. Logs summary to console/logs
          ↓
Evening   - Review reports
          - Evaluate recommendations
          - Plan parameter adjustments
          ↓
Next Day  - Manually apply approved changes
          - Monitor impact
```

### Manual Review Process

1. **Check Daily Summary** (`logs/trading.log` or console output)
   ```
   📊 Performance Summary:
     Trades: 45 (42 closed)
     Win Rate: 66.7%
     Total P&L: $2,450.75
     Profit Factor: 2.34
   ```

2. **Open Markdown Report** (`reports/daily_reviews/daily_review_2025-01-15.md`)
   - Read executive summary
   - Review key observations
   - Evaluate recommendations

3. **Examine JSON for Details** (if needed)
   - Detailed metrics
   - Confidence scores
   - Full reasoning

4. **Evaluate Recommendations**
   - Check current vs suggested values
   - Read reasoning
   - Assess risk level
   - Verify confidence scores

5. **Apply Changes** (if approved)
   ```bash
   # Manually edit config.yaml
   vim config.yaml
   
   # Or use configuration manager
   python -c "
   from mytrader.llm.config_manager import ConfigurationManager
   cm = ConfigurationManager()
   cm.apply_adjustments([
       {'parameter': 'strategies[0].params.rsi_buy', 'value': 32.0}
   ])
   "
   ```

## Integration with Autonomous System

The live review system can optionally feed insights into the autonomous learning system:

### Option 1: Manual Review Only (Default)

```yaml
live_review:
  feed_to_autonomous: false  # No automatic integration
  auto_apply_recommendations: false
```

**Workflow**: Human reviews reports → Manual parameter updates

### Option 2: Feed to Autonomous (Semi-Automatic)

```yaml
live_review:
  feed_to_autonomous: true  # Feed insights to autonomous system
  auto_apply_recommendations: false  # Still requires approval

autonomous:
  require_human_approval: true
```

**Workflow**: Live review → Autonomous system → Human approval → Apply

### Option 3: Fully Autonomous (Advanced)

```yaml
live_review:
  feed_to_autonomous: true
  auto_apply_recommendations: true

autonomous:
  require_human_approval: false
  auto_approve_threshold: 0.85  # Only high-confidence changes
```

**Workflow**: Live review → Autonomous system → Auto-apply (if confidence ≥ 0.85)

⚠️ **Warning**: Option 3 should only be used after extensive testing and validation.

## Troubleshooting

### No Trades Found

**Symptom**: "No trading data available"

**Solutions**:
1. Verify trade logging is active during paper trading
2. Check database file exists: `data/llm_trades.db`
3. Try CSV mode: `python run_daily_review.py --csv`
4. Verify CSV log path in config.yaml

### LLM Call Failures

**Symptom**: "Failed to generate insights"

**Solutions**:
1. Check AWS credentials configured
2. Verify Bedrock service available in region
3. Check `llm.enabled: true` in config.yaml
4. Review error logs in `logs/trading.log`

### Missing Reports

**Symptom**: Reports not generated

**Solutions**:
1. Check write permissions on `reports/` directory
2. Verify reports_dir setting in config.yaml
3. Check disk space
4. Review error logs

### Low Confidence Recommendations

**Symptom**: AI confidence scores consistently low

**Possible Causes**:
1. Insufficient trade data (increase --days)
2. Inconsistent trading patterns
3. Mixed signal performance
4. High market volatility

**Solutions**:
1. Analyze more days: `--days 7`
2. Focus on single strategy first
3. Wait for more data accumulation

## Performance Metrics Reference

### Key Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| **Win Rate** | Percentage of winning trades | > 55% |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 |
| **Sharpe Ratio** | Risk-adjusted returns | > 1.0 |
| **Sortino Ratio** | Downside risk-adjusted returns | > 1.5 |
| **Max Drawdown** | Largest peak-to-trough decline | < 10% of capital |
| **Average Win** | Average profit per winning trade | > 1.5x avg loss |
| **Average Loss** | Average loss per losing trade | < $300 |

### Signal Performance

Breakdown of trades and P&L by signal type:
- `rsi_macd_sentiment` - Traditional technical + sentiment
- `llm_enhanced` - LLM-confirmed signals
- `manual` - Human-initiated trades

### Timing Analysis

Win rate breakdown by hour (EST):
- **09:30-11:00**: Opening session volatility
- **11:00-14:00**: Mid-day consolidation
- **14:00-16:00**: Afternoon trends
- **High win rate hours**: Focus trading effort
- **Low win rate hours**: Consider avoiding

## Best Practices

### 1. Gradual Implementation

- Start with manual review only
- Build confidence over 2-4 weeks
- Gradually increase automation

### 2. Regular Monitoring

- Review daily reports consistently
- Track recommendation accuracy
- Monitor parameter drift

### 3. Conservative Parameter Changes

- Apply one change at a time
- Monitor impact for 2-3 days
- Rollback if performance degrades

### 4. Maintain Audit Trail

- Keep all reports for 90 days
- Document manual changes
- Track reasoning for decisions

### 5. Safety First

- Never apply all recommendations at once
- Prioritize low-risk changes
- Maintain stop-loss discipline
- Respect position size limits

## Next Steps

1. **Initial Setup**
   - Configure trade logging
   - Test manual execution
   - Review sample reports

2. **Validation Phase** (2 weeks)
   - Run daily reviews
   - Evaluate recommendation quality
   - Build confidence in system

3. **Semi-Autonomous Phase** (4 weeks)
   - Apply selected recommendations
   - Monitor performance impact
   - Refine parameter ranges

4. **Full Integration** (8+ weeks)
   - Consider autonomous integration
   - Implement email notifications
   - Optimize scheduling

## Support

For issues or questions:
1. Check logs: `logs/trading.log`
2. Review documentation: `docs/`
3. Test with sample data: `python demo_backtest.py`
4. File issue with error details

---

**System Version**: 1.0  
**Last Updated**: 2025-01-15  
**Components**: LiveTradeAnalyzer, AIInsightGenerator, DailyReviewOrchestrator
