# Live Paper Trading Review - Quick Reference

## Quick Start

```bash
# Run review
python run_daily_review.py

# Setup scheduled execution
./scripts/setup_cron.sh           # Linux/macOS
.\scripts\setup_task_scheduler.ps1  # Windows
```

## Common Commands

```bash
# Last 3 days (default)
python run_daily_review.py

# Last 7 days
python run_daily_review.py --days 7

# Use CSV logs
python run_daily_review.py --csv

# JSON only
python run_daily_review.py --no-markdown

# Verbose output
python run_daily_review.py -v
```

## File Locations

```
Reports:     reports/daily_reviews/
             reports/ai_insights/

Database:    data/llm_trades.db
CSV Logs:    logs/trades.csv
System Logs: logs/trading.log
Cron Logs:   logs/cron_review.log

Config:      config.yaml (live_review section)
```

## Configuration (config.yaml)

```yaml
live_review:
  enabled: true
  run_time: "18:00"           # Daily at 6:00 PM
  analysis_days: 3            # Last 3 days
  use_database: true          # SQLite vs CSV
  generate_json: true
  generate_markdown: true
  enable_ai_insights: true
```

## Trade Logging (During Live Trading)

```python
from mytrader.llm.trade_logger import TradeLogger

logger = TradeLogger()

# Log entry
logger.log_trade(
    timestamp=datetime.now(),
    symbol="ES",
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

## Scheduled Execution

### Linux/macOS
```bash
# View cron jobs
crontab -l

# Edit cron jobs
crontab -e

# Example entry (daily 6 PM)
0 18 * * * cd /path/to/MyTrader && python run_daily_review.py
```

### Windows
```powershell
# Check task
Get-ScheduledTask -TaskName "MyTrader Daily Review"

# Run task manually
Start-ScheduledTask -TaskName "MyTrader Daily Review"

# View task history
Get-ScheduledTaskInfo -TaskName "MyTrader Daily Review"
```

## Report Structure

### JSON Report (ai_insights_YYYY-MM-DD.json)
```json
{
  "observations": [...],      // Detected patterns
  "recommendations": [...],   // Parameter suggestions
  "warnings": [...],          // Risk alerts
  "behavioral_patterns": [...],
  "market_trends": [...]
}
```

### Markdown Report (daily_review_YYYY-MM-DD.md)
- Executive Summary
- Key Observations
- Recommendations
- Warnings
- Next Steps

## Key Metrics

| Metric | Good Range |
|--------|-----------|
| Win Rate | > 55% |
| Profit Factor | > 1.5 |
| Sharpe Ratio | > 1.0 |
| Max Drawdown | < 10% capital |

## Troubleshooting

### No trades found
```bash
# Check database
ls -lh data/llm_trades.db

# Try CSV mode
python run_daily_review.py --csv

# Check CSV exists
cat logs/trades.csv
```

### LLM failures
```bash
# Check AWS credentials
aws sts get-caller-identity

# Check config
grep "llm:" config.yaml

# View logs
tail -f logs/trading.log
```

### Missing reports
```bash
# Check permissions
ls -ld reports/daily_reviews/

# Create directories
mkdir -p reports/daily_reviews reports/ai_insights

# Run verbose
python run_daily_review.py -v
```

### Scheduled execution not working
```bash
# Linux/macOS: Check cron
grep "run_daily_review" /var/log/syslog
tail -f logs/cron_review.log

# Windows: Check Task Scheduler
eventvwr.msc  # Event Viewer → Windows Logs → Application
```

## Workflow

1. **6:00 PM** - Scheduled review runs
2. **Evening** - Review reports
3. **Evaluate** - Assess recommendations
4. **Next Day** - Apply approved changes
5. **Monitor** - Track impact 2-3 days

## Safety Rules

✅ Human review required  
✅ One change at a time  
✅ Monitor impact 2-3 days  
✅ Rollback if performance degrades  
✅ Never apply all recommendations at once  

## Integration Modes

**1. Manual Review (Default)**
```yaml
feed_to_autonomous: false
auto_apply_recommendations: false
```

**2. Semi-Autonomous**
```yaml
feed_to_autonomous: true
auto_apply_recommendations: false
autonomous:
  require_human_approval: true
```

**3. Fully Autonomous (Advanced)**
```yaml
feed_to_autonomous: true
auto_apply_recommendations: true
autonomous:
  require_human_approval: false
  auto_approve_threshold: 0.85
```

## Documentation

- **Full Guide**: `docs/LIVE_TRADING_REVIEW_GUIDE.md`
- **Windows Setup**: `docs/WINDOWS_TASK_SCHEDULER_SETUP.md`
- **Implementation**: `docs/LIVE_REVIEW_IMPLEMENTATION_COMPLETE.md`
- **Autonomous System**: `docs/AUTONOMOUS_TRADING_GUIDE.md`

## Support

Check logs: `logs/trading.log`  
Test script: `python run_daily_review.py -v`  
Verify setup: `./scripts/setup_live_review.sh`  

## Quick Health Check

```bash
# Test system
python run_daily_review.py --days 1

# Check last 5 reports
ls -lt reports/daily_reviews/ | head -6

# View latest report
cat reports/ai_insights/daily_review_$(date +%Y-%m-%d).md

# Check scheduled task
crontab -l | grep run_daily_review  # Linux/macOS
Get-ScheduledTask | Where-Object {$_.TaskName -like "*MyTrader*"}  # Windows
```

---

**Quick Help**: `python run_daily_review.py --help`  
**Version**: 1.0  
**Status**: Production Ready ✅
