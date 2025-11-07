# SPY Futures Daily Review - Quick Reference

## Quick Start

```bash
# Run SPY Futures review
python run_spy_futures_review.py

# Start dashboard (separate terminal)
cd dashboard/backend && python dashboard_api.py

# View dashboard
# Open browser: http://localhost:8000
```

## Common Commands

```bash
# ES (E-mini), last day, push to dashboard
python run_spy_futures_review.py

# MES (Micro E-mini), last 3 days
python run_spy_futures_review.py --symbol MES --days 3

# Local only (no dashboard push)
python run_spy_futures_review.py --no-dashboard

# Use CSV logs
python run_spy_futures_review.py --csv

# Verbose output
python run_spy_futures_review.py -v
```

## File Locations

```
Reports:     reports/spy_futures_daily/
Dashboard:   dashboard/backend/reports/
Database:    data/llm_trades.db
CSV Logs:    logs/trades.csv
Config:      config.yaml
```

## Configuration (config.yaml)

```yaml
llm:
  enabled: true
  model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
  temperature: 0.3

live_review:
  enabled: true
  run_time: "18:00"
  analysis_days: 1
  use_database: true
```

## Trade Logging (During Paper Trading)

```python
from mytrader.llm.trade_logger import TradeLogger

logger = TradeLogger()

# Log ES trade
logger.log_trade(
    timestamp=datetime.now(),
    symbol="ES",  # or "MES"
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
    pnl=512.50
)
```

## Dashboard API Endpoints

```bash
# Post summary
POST http://localhost:8000/api/trading-summary

# Get latest
GET http://localhost:8000/api/spy-futures/latest-summary

# Get history
GET http://localhost:8000/api/spy-futures/summary-history?days=7

# Trigger analysis
POST http://localhost:8000/api/spy-futures/run-analysis?days=1
```

## Report Structure (JSON)

```json
{
  "date": "2025-11-06",
  "symbol": "ES",
  "performance": {
    "total_trades": 12,
    "win_rate": 67,
    "profit_loss": 1450,
    "max_drawdown": 300
  },
  "observations": [
    "High frequency during low volatility",
    "Best performance 10-12 EST"
  ],
  "suggestions": {
    "trade_frequency_limit": "3/hour",
    "sentiment_confidence_threshold": 0.75,
    "preferred_trading_hours": [10, 11, 14]
  },
  "warnings": [
    "Low win rate 14-16 EST"
  ]
}
```

## Scheduled Execution

### Linux/macOS (Cron)
```bash
# Daily at 6 PM
crontab -e
# Add: 0 18 * * * cd /path/to/MyTrader && python run_spy_futures_review.py
```

### Windows (Task Scheduler)
```
1. Open Task Scheduler
2. Create task: "SPY Futures Review"
3. Trigger: Daily 6:00 PM
4. Action: python.exe run_spy_futures_review.py
5. Start in: C:\path\to\MyTrader
```

## Dashboard Component (React)

```jsx
import SPYFuturesInsights from './components/SPYFuturesInsights';

function Dashboard() {
  return (
    <div>
      <SPYFuturesInsights apiUrl="http://localhost:8000" />
    </div>
  );
}
```

## Key Metrics

| Metric | Good Target |
|--------|-------------|
| Win Rate | > 55% |
| Profit Factor | > 1.5 |
| Max Drawdown | < $1,000 |
| Avg Hold Time | 30-90 min |

## SPY Futures Specs

- **ES**: $50/point, $12.50/tick
- **MES**: $5/point, $1.25/tick  
- **Hours**: Nearly 24/5 (Sun 6PM - Fri 5PM ET)
- **Regular**: 9:30 AM - 4:00 PM ET

## Troubleshooting

### No trades found
```bash
# Check database
ls -lh data/llm_trades.db

# Try CSV
python run_spy_futures_review.py --csv

# Check logs
tail -f logs/trading.log
```

### Dashboard not updating
```bash
# Check backend running
ps aux | grep dashboard_api

# Test API
curl http://localhost:8000/api/status

# Start dashboard
cd dashboard/backend && python dashboard_api.py
```

### LLM analysis fails
```bash
# Check AWS credentials
aws sts get-caller-identity

# Verify config
grep "llm:" config.yaml

# Test with verbose
python run_spy_futures_review.py -v
```

## Workflow

1. **6:00 PM** - Market closes, review runs
2. **Evening** - Check dashboard for insights
3. **Review** - Observations, suggestions, warnings
4. **Plan** - Parameter adjustments (if any)
5. **Next Day** - Monitor impact

## Safety Rules

✅ SPY Futures (ES/MES) only  
✅ Review AI suggestions before applying  
✅ One change at a time  
✅ Monitor 2-3 days  
✅ Maintain audit trail  
❌ Never apply all suggestions at once  

## Quick Health Check

```bash
# Test system
python run_spy_futures_review.py --days 1 -v

# Check latest report
cat reports/spy_futures_daily/spy_futures_report_$(date +%Y-%m-%d).json | jq .

# View dashboard report
cat dashboard/backend/reports/spy_summary_$(date +%Y-%m-%d).json | jq .

# Check scheduled task
crontab -l | grep spy_futures  # Linux/macOS
Get-ScheduledTask | Where-Object {$_.TaskName -like "*SPY*"}  # Windows
```

## Documentation

- **Full Guide**: `docs/SPY_FUTURES_REVIEW_GUIDE.md`
- **General Review**: `docs/LIVE_TRADING_REVIEW_GUIDE.md`
- **Windows Setup**: `docs/WINDOWS_TASK_SCHEDULER_SETUP.md`

## Support

```bash
# View help
python run_spy_futures_review.py --help

# Test mode
python run_spy_futures_review.py --no-dashboard -v

# Check logs
tail -f logs/trading.log
tail -f dashboard/backend/logs/*.log
```

---

**Quick Help**: `python run_spy_futures_review.py --help`  
**Dashboard**: http://localhost:8000  
**Version**: 1.0  
**Status**: Production Ready ✅  
**Instrument**: SPY Futures (ES/MES) Only
