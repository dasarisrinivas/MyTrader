# SPY Futures Daily Review System - Complete Guide

## Overview

The SPY Futures Daily Review System analyzes paper trading performance for SPY Futures (ES/MES), generates AI-powered insights using AWS Bedrock Claude, and pushes structured results to the React dashboard for visualization.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│           SPY FUTURES DAILY REVIEW & DASHBOARD SYSTEM           │
└─────────────────────────────────────────────────────────────────┘

1. DATA COLLECTION (SPY Futures Only)
   ├── IBKR Paper Trading (ES/MES)
   ├── Trade Logger → llm_trades.db
   └── CSV Logs → logs/trades.csv

2. ANALYSIS PIPELINE
   ├── SPYFuturesAnalyzer
   │   ├── Load SPY trades (ES, MES, SPY filter)
   │   ├── Calculate metrics (20+ metrics)
   │   └── Breakdown by signal, hour, holding time
   │
   └── SPYFuturesInsightGenerator
       ├── Generate comprehensive prompt
       ├── Call AWS Bedrock Claude 3 Sonnet
       └── Parse structured JSON insights

3. ORCHESTRATION
   └── SPYFuturesDailyOrchestrator
       ├── Coordinate analysis → insights
       ├── Save reports locally
       └── Push to dashboard API

4. DASHBOARD INTEGRATION
   ├── FastAPI Backend
   │   ├── POST /api/trading-summary
   │   ├── GET /api/spy-futures/latest-summary
   │   ├── GET /api/spy-futures/summary-history
   │   └── POST /api/spy-futures/run-analysis
   │
   └── React Frontend
       └── SPYFuturesInsights Component
           ├── Real-time metrics display
           ├── AI observations & recommendations
           ├── Warnings & alerts
           └── Manual analysis trigger
```

## Key Components

### 1. SPYFuturesAnalyzer (`mytrader/llm/spy_futures_analyzer.py`)

**Purpose**: Load and analyze SPY Futures trades only

**Key Features**:
- Symbol filtering: ES (E-mini S&P 500), MES (Micro E-mini), SPY
- Database and CSV support
- Comprehensive metrics calculation
- Holding time analysis
- Hour-based performance breakdown
- LLM enhancement tracking

**Metrics Computed**:
- Total trades, open/closed positions
- Win rate, profit factor, max drawdown
- Gross profit/loss, average win/loss
- Largest win/loss
- Holding time analysis
- Hourly performance breakdown
- Signal type performance
- LLM enhancement analysis

### 2. SPYFuturesInsightGenerator (`mytrader/llm/spy_futures_insights.py`)

**Purpose**: Generate AI insights using AWS Bedrock

**Structured Output**:
```json
{
  "date": "2025-11-06",
  "symbol": "ES",
  "performance": {
    "total_trades": 12,
    "win_rate": 67,
    "profit_loss": 1450,
    "max_drawdown": 300,
    "profit_factor": 2.1,
    "average_win": 185.50,
    "average_loss": 125.00,
    "holding_time_avg": 45.3
  },
  "observations": [
    "High frequency trades during low volatility periods",
    "Sentiment-only trades show lower accuracy"
  ],
  "insights": [
    {
      "type": "timing",
      "category": "entry_optimization",
      "description": "Best performance between 10:00-12:00 EST",
      "severity": "info",
      "confidence": 0.85,
      "reasoning": "12 of 15 trades profitable during this window"
    }
  ],
  "suggestions": {
    "trade_frequency_limit": "reduce to 3/hour",
    "sentiment_confidence_threshold": 0.75,
    "preferred_trading_hours": [10, 11, 14],
    "position_sizing_adjustment": "maintain",
    "stop_loss_adjustment": "+5 ticks",
    "take_profit_adjustment": "+10 ticks"
  },
  "warnings": [
    "Low win rate during 14:00-16:00 EST - avoid late afternoon trades"
  ],
  "profitable_patterns": [
    "RSI oversold + positive sentiment",
    "Morning breakouts with volume confirmation"
  ],
  "losing_patterns": [
    "Counter-trend trades in strong momentum",
    "Late afternoon mean reversion"
  ],
  "market_conditions": "Moderate volatility with trending bias",
  "volatility_assessment": "VIX 15-18 range, suitable for directional trades"
}
```

### 3. SPYFuturesDailyOrchestrator (`mytrader/llm/spy_futures_orchestrator.py`)

**Purpose**: Coordinate complete daily review workflow

**Workflow**:
1. Load SPY Futures trades (DB or CSV)
2. Compute performance metrics
3. Generate AI insights via LLM
4. Save reports locally
5. Push to dashboard API
6. Broadcast via WebSocket (optional)

### 4. Dashboard API Endpoints (`dashboard/backend/dashboard_api.py`)

#### POST `/api/trading-summary`
Receive daily SPY Futures summary and broadcast to clients.

**Request Body**:
```json
{
  "date": "2025-11-06",
  "performance": { /* metrics */ },
  "observations": [ /* strings */ ],
  "suggestions": { /* key-value pairs */ },
  "warnings": [ /* strings */ ],
  "insights": [ /* insight objects */ ]
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Trading summary received for 2025-11-06",
  "saved_to": "dashboard/backend/reports/spy_summary_2025-11-06.json"
}
```

#### GET `/api/spy-futures/latest-summary`
Get most recent SPY Futures summary.

**Response**:
```json
{
  "status": "success",
  "data": { /* complete summary */ },
  "file": "path/to/file.json",
  "timestamp": "2025-11-06T18:05:23"
}
```

#### GET `/api/spy-futures/summary-history?days=7`
Get last N days of summaries.

#### POST `/api/spy-futures/run-analysis?days=1`
Trigger analysis manually from dashboard.

### 5. React Component (`dashboard/frontend/src/components/SPYFuturesInsights.jsx`)

**Features**:
- Real-time performance metrics display
- AI observations and recommendations
- Critical warnings with visual alerts
- Detailed insights with confidence scores
- Manual analysis trigger button
- Auto-refresh every 30 seconds
- Responsive grid layout

## Setup Instructions

### Prerequisites

1. **Python Environment**
   ```bash
   python >= 3.8
   pip install -r requirements.txt
   ```

2. **AWS Credentials**
   ```bash
   export AWS_ACCESS_KEY_ID="your-key"
   export AWS_SECRET_ACCESS_KEY="your-secret"
   export AWS_DEFAULT_REGION="us-east-1"
   ```

3. **Dashboard Running**
   ```bash
   # Backend
   cd dashboard/backend
   python dashboard_api.py
   
   # Frontend (separate terminal)
   cd dashboard/frontend
   npm install
   npm run dev
   ```

### Configuration

Update `config.yaml`:

```yaml
# LLM settings
llm:
  enabled: true
  model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
  region_name: "us-east-1"
  temperature: 0.3

# Live review settings
live_review:
  enabled: true
  run_time: "18:00"
  analysis_days: 1
  use_database: true
```

### Trade Logging

During paper trading, log SPY Futures trades:

```python
from mytrader.llm.trade_logger import TradeLogger

logger = TradeLogger()

# Log ES trade entry
logger.log_trade(
    timestamp=datetime.now(),
    symbol="ES",  # or "MES" for Micro E-mini
    action="BUY",
    quantity=2,
    price=4500.25,
    signal_type="rsi_macd_sentiment",
    llm_confidence=0.75,
    position_id="POS_123"
)

# Log exit
logger.log_trade_exit(
    position_id="POS_123",
    exit_timestamp=datetime.now(),
    exit_price=4510.50,
    pnl=512.50,
    exit_reason="Take profit"
)
```

## Usage Examples

### Manual Execution

```bash
# Basic review (ES, last day, push to dashboard)
python run_spy_futures_review.py

# Review last 3 days
python run_spy_futures_review.py --days 3

# Analyze MES instead of ES
python run_spy_futures_review.py --symbol MES

# Skip dashboard push (local only)
python run_spy_futures_review.py --no-dashboard

# Use CSV logs instead of database
python run_spy_futures_review.py --csv

# Custom dashboard URL
python run_spy_futures_review.py --dashboard-url http://192.168.1.100:8000

# Full custom
python run_spy_futures_review.py --days 7 --symbol ES --csv --verbose
```

### Scheduled Execution

#### Linux/macOS (Cron)

```bash
# Edit crontab
crontab -e

# Add line for daily 6 PM execution
0 18 * * * cd /path/to/MyTrader && /path/to/python run_spy_futures_review.py

# With virtual environment
0 18 * * * cd /path/to/MyTrader && /path/to/venv/bin/python run_spy_futures_review.py >> logs/spy_review_cron.log 2>&1
```

#### Windows (Task Scheduler)

See `docs/WINDOWS_TASK_SCHEDULER_SETUP.md` for detailed setup.

Quick setup:
1. Open Task Scheduler
2. Create Basic Task: "SPY Futures Daily Review"
3. Trigger: Daily at 6:00 PM
4. Action: `python.exe run_spy_futures_review.py`
5. Start in: `C:\path\to\MyTrader`

### API Usage

#### From Python

```python
import requests

# Post summary to dashboard
response = requests.post(
    'http://localhost:8000/api/trading-summary',
    json={
        "date": "2025-11-06",
        "performance": {...},
        "observations": [...],
        "suggestions": {...}
    }
)

# Get latest summary
response = requests.get('http://localhost:8000/api/spy-futures/latest-summary')
data = response.json()
```

#### From Dashboard (JavaScript)

```javascript
// Fetch latest summary
const response = await fetch('http://localhost:8000/api/spy-futures/latest-summary');
const data = await response.json();

// Trigger analysis
await fetch('http://localhost:8000/api/spy-futures/run-analysis', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ days: 1 })
});
```

## Dashboard Integration

### Add Component to Dashboard

Edit `dashboard/frontend/src/App.jsx` or `Dashboard.jsx`:

```jsx
import SPYFuturesInsights from './components/SPYFuturesInsights';

function Dashboard() {
  return (
    <div className="dashboard">
      {/* Existing components */}
      
      {/* Add SPY Futures Insights */}
      <SPYFuturesInsights apiUrl="http://localhost:8000" />
    </div>
  );
}
```

### WebSocket Updates (Optional)

For real-time updates, the dashboard API broadcasts via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'spy_trading_summary') {
    // Update UI with new summary
    updateSPYInsights(data.data);
  }
};
```

## Report Files

### Local Reports

```
MyTrader/
├── reports/
│   ├── spy_futures_daily/
│   │   └── spy_futures_report_2025-11-06.json
│   └── dashboard_ready/
│       └── spy_futures_2025-11-06.json
│
└── dashboard/backend/reports/
    └── spy_summary_2025-11-06.json
```

### Report Structure

**spy_futures_report_YYYY-MM-DD.json**:
- Complete analysis results
- All insights with full details
- Raw performance data
- For audit and review

**spy_summary_YYYY-MM-DD.json** (dashboard):
- Dashboard-optimized format
- Flattened structure
- Ready for UI consumption

## Troubleshooting

### No Trades Found

**Symptom**: "No SPY Futures trading data available"

**Solutions**:
1. Verify SPY Futures trades logged (ES, MES, SPY symbols only)
2. Check database: `ls -lh data/llm_trades.db`
3. Try CSV mode: `--csv`
4. Verify symbol filter: `--symbol ES`

### Dashboard Not Updating

**Symptom**: Dashboard shows old data or errors

**Solutions**:
1. Check dashboard backend running: `ps aux | grep dashboard_api`
2. Verify API accessible: `curl http://localhost:8000/api/status`
3. Check firewall/port 8000 open
4. Review backend logs: `tail -f dashboard/backend/logs/*.log`

### LLM Analysis Fails

**Symptom**: "LLM analysis failed" in logs

**Solutions**:
1. Check AWS credentials: `aws sts get-caller-identity`
2. Verify Bedrock enabled in region
3. Check `llm.enabled: true` in config.yaml
4. Review temperature/token settings
5. Check network connectivity

### Dashboard Push Failed

**Symptom**: "Dashboard push failed" message

**Solutions**:
1. Start dashboard: `cd dashboard/backend && python dashboard_api.py`
2. Check URL: `--dashboard-url http://localhost:8000`
3. Verify network connectivity
4. Check CORS settings in dashboard_api.py
5. Review dashboard logs

## Best Practices

### 1. SPY Futures Only

- ✅ **Trade only ES, MES, or SPY**
- ❌ Do not mix with other instruments
- ❌ System designed specifically for SPY Futures
- ❌ LLM trained to analyze SPY patterns only

### 2. Data Quality

- Log ALL trades (entries and exits)
- Include LLM confidence when available
- Record exit reasons for pattern analysis
- Maintain consistent timestamp format

### 3. Review Schedule

- Run after market close (6:00 PM ET recommended)
- Allow minimum 10-20 trades per day for meaningful analysis
- Weekly cumulative reviews on weekends
- Compare against baseline metrics

### 4. Dashboard Monitoring

- Check dashboard daily after review runs
- Prioritize high-severity warnings
- Track recommendation accuracy over time
- Document parameter changes and impacts

### 5. Safety Rules

- ✅ Review AI suggestions before applying
- ✅ Test one parameter change at a time
- ✅ Monitor impact for 2-3 days
- ✅ Maintain trade log for audit
- ❌ Never apply all suggestions at once
- ❌ Don't override risk limits based on AI alone

## Performance Metrics Reference

| Metric | Description | Good Target |
|--------|-------------|-------------|
| **Win Rate** | % of profitable trades | > 55% |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 |
| **Max Drawdown** | Largest peak-to-trough decline | < $1,000 |
| **Avg Hold Time** | Average minutes per trade | 30-90 min |
| **Avg Win** | Average profit per win | > 1.5x avg loss |
| **Avg Loss** | Average loss per trade | < $250 |

### SPY Futures Specifics

- **ES** (E-mini S&P 500): $50 per point, ~$12.50 per tick
- **MES** (Micro E-mini): $5 per point, ~$1.25 per tick
- **Trading Hours**: Nearly 24/5 (Sun 6PM - Fri 5PM ET)
- **Regular Hours**: 9:30 AM - 4:00 PM ET
- **Extended Hours**: 6:00 PM - 9:30 AM ET

## Example Workflow

### Daily Workflow (Automated)

```
16:00 ET - Market closes
18:00 ET - Scheduled review runs
          ↓
1. System loads SPY trades from last 24h
2. Calculates 20+ performance metrics  
3. Generates AI insights via Bedrock
4. Saves JSON reports locally
5. Pushes to dashboard API
6. Dashboard updates automatically
          ↓
Evening  - Review dashboard insights
         - Evaluate AI recommendations
         - Check warnings and alerts
         - Plan parameter adjustments
          ↓
Next Day - Monitor live trading
         - Observe impact of changes
```

### Manual Review Process

1. **Open Dashboard** (`http://localhost:8000`)
2. **Review Performance Metrics** (win rate, P&L, drawdown)
3. **Read AI Observations** (patterns identified)
4. **Evaluate Suggestions** (parameter adjustments)
5. **Check Warnings** (critical issues)
6. **Profitable vs Losing Patterns** (what works, what doesn't)
7. **Market Context** (volatility, conditions)
8. **Take Action** (manual parameter updates if approved)

## API Reference

### POST /api/trading-summary

**Request**:
```json
{
  "date": "string (YYYY-MM-DD)",
  "performance": {
    "total_trades": "integer",
    "win_rate": "float",
    "profit_loss": "float",
    "max_drawdown": "float",
    ...
  },
  "observations": ["string"],
  "suggestions": {"key": "value"},
  "warnings": ["string"],
  "insights": [{"type": "string", ...}]
}
```

**Response**: `200 OK`
```json
{
  "status": "success",
  "message": "Trading summary received for 2025-11-06",
  "saved_to": "path/to/file.json"
}
```

### GET /api/spy-futures/latest-summary

**Response**: `200 OK`
```json
{
  "status": "success",
  "data": { /* complete summary */ },
  "file": "string",
  "timestamp": "ISO 8601"
}
```

### GET /api/spy-futures/summary-history?days=7

**Query Parameters**:
- `days`: Number of days (default: 7)

**Response**: `200 OK`
```json
{
  "status": "success",
  "count": "integer",
  "summaries": [ /* array of summaries */ ]
}
```

### POST /api/spy-futures/run-analysis?days=1

**Query Parameters**:
- `days`: Number of days to analyze (default: 1)

**Response**: `200 OK`
```json
{
  "status": "success",
  "message": "SPY Futures analysis completed",
  "result": { /* complete analysis result */ }
}
```

## Next Steps

1. **Initial Setup**
   - Configure trade logging for SPY Futures
   - Test manual execution
   - Verify dashboard integration

2. **Validation Phase** (1-2 weeks)
   - Run daily reviews
   - Evaluate AI recommendation quality
   - Build confidence in system

3. **Optimization** (2-4 weeks)
   - Apply selected recommendations
   - Track recommendation accuracy
   - Fine-tune parameters

4. **Production** (4+ weeks)
   - Setup scheduled execution
   - Monitor automated reviews
   - Continuous improvement

## Support

- **Logs**: `logs/trading.log`, `dashboard/backend/logs/`
- **Documentation**: This guide, inline code comments
- **Test**: `python run_spy_futures_review.py -v`
- **Dashboard**: http://localhost:8000

---

**System Version**: 1.0  
**Last Updated**: November 6, 2025  
**Instrument**: SPY Futures (ES/MES) Only  
**Status**: Production Ready ✅
