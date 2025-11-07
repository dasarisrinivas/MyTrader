# MyTrader - AI-Powered SPY Futures Trading System

Production-ready SPY Futures (ES/MES) paper trading system with AI-powered insights, autonomous learning, and real-time dashboard.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure AWS credentials
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"

# 3. Start dashboard
./scripts/start_dashboard.sh

# 4. Run SPY Futures review
python run_spy_futures_review.py

# View dashboard: http://localhost:8000
```

## 📋 System Overview

MyTrader is a comprehensive trading system with three main components:

### 1. SPY Futures Daily Review (Production)
- **Purpose**: Analyze ES/MES paper trading performance with AI insights
- **Command**: `python run_spy_futures_review.py`
- **Features**:
  - 20+ performance metrics
  - AI pattern detection
  - Dashboard integration
  - Structured recommendations
- **Docs**: `docs/SPY_FUTURES_REVIEW_GUIDE.md`

### 2. Live Trading Review (General)
- **Purpose**: Multi-instrument trading analysis with LLM insights
- **Command**: `python run_daily_review.py`
- **Features**:
  - CSV/Database support
  - Signal performance analysis
  - Timing optimization
  - JSON/Markdown reports
- **Docs**: `docs/LIVE_TRADING_REVIEW_GUIDE.md`

### 3. Autonomous Learning System (Advanced)
- **Purpose**: Self-optimizing parameter adjustment
- **Command**: `python run_autonomous_trading.py daily`
- **Features**:
  - Daily performance analysis
  - Automatic parameter tuning
  - Safety constraints
  - Rollback mechanism
- **Docs**: `docs/AUTONOMOUS_TRADING_GUIDE.md`

## 📁 Project Structure

```
MyTrader/
├── run_spy_futures_review.py       # SPY Futures daily review (MAIN)
├── run_daily_review.py             # General live trading review
├── run_autonomous_trading.py       # Autonomous learning system
├── main.py                         # Legacy main entry point
│
├── mytrader/                       # Core package
│   ├── llm/                        # LLM & analysis modules
│   │   ├── spy_futures_analyzer.py         # SPY-specific analysis
│   │   ├── spy_futures_insights.py         # SPY LLM insights
│   │   ├── spy_futures_orchestrator.py     # SPY orchestration
│   │   ├── trade_analyzer.py               # General trade analysis
│   │   ├── ai_insights.py                  # General LLM insights
│   │   ├── daily_review.py                 # General orchestration
│   │   ├── autonomous_orchestrator.py      # Autonomous system
│   │   ├── performance_analyzer.py         # Performance metrics
│   │   ├── adaptive_engine.py              # Learning engine
│   │   ├── config_manager.py               # Config management
│   │   └── ...
│   ├── execution/                  # Trade execution
│   ├── strategies/                 # Trading strategies
│   ├── risk/                       # Risk management
│   └── utils/                      # Utilities
│
├── dashboard/                      # React dashboard
│   ├── backend/                    # FastAPI backend
│   │   └── dashboard_api.py        # API endpoints
│   └── frontend/                   # React UI
│       └── src/components/
│           └── SPYFuturesInsights.jsx
│
├── docs/                           # Documentation
│   ├── SPY_FUTURES_REVIEW_GUIDE.md         # SPY system guide
│   ├── LIVE_TRADING_REVIEW_GUIDE.md        # General review guide
│   ├── AUTONOMOUS_TRADING_GUIDE.md         # Autonomous system guide
│   ├── QUICK_REFERENCE.md                  # General quick ref
│   ├── SPY_FUTURES_QUICK_REF.md            # SPY quick ref
│   └── ...
│
├── scripts/                        # Utility scripts
│   ├── start_dashboard.sh          # Start dashboard
│   ├── setup_live_review.sh        # Setup review system
│   └── setup_cron.sh               # Schedule reviews
│
├── data/                           # Data directory
│   └── llm_trades.db               # Trade database
│
├── logs/                           # Log files
│   └── trades.csv                  # CSV trade logs
│
└── reports/                        # Generated reports
    ├── spy_futures_daily/          # SPY reports
    ├── daily_reviews/              # General reports
    └── autonomous/                 # Autonomous reports
```

## 🎯 Main Commands

### SPY Futures Review (Recommended)
```bash
# Daily review with dashboard push
python run_spy_futures_review.py

# Last 3 days, MES symbol
python run_spy_futures_review.py --days 3 --symbol MES

# Local only (no dashboard)
python run_spy_futures_review.py --no-dashboard

# Use CSV logs
python run_spy_futures_review.py --csv

# Help
python run_spy_futures_review.py --help
```

### General Live Review
```bash
# Review last 3 days
python run_daily_review.py

# Custom period
python run_daily_review.py --days 7

# CSV mode
python run_daily_review.py --csv
```

### Autonomous System
```bash
# Daily analysis
python run_autonomous_trading.py daily

# Weekly review
python run_autonomous_trading.py weekly

# System status
python run_autonomous_trading.py status

# Rollback changes
python run_autonomous_trading.py rollback
```

### Dashboard
```bash
# Start backend + frontend
./scripts/start_dashboard.sh

# Or manually:
cd dashboard/backend && python dashboard_api.py
cd dashboard/frontend && npm run dev

# View: http://localhost:8000
```

## 📊 Dashboard API Endpoints

```
POST   /api/trading-summary                    # Post SPY summary
GET    /api/spy-futures/latest-summary         # Get latest SPY data
GET    /api/spy-futures/summary-history        # Historical summaries
POST   /api/spy-futures/run-analysis           # Trigger analysis
GET    /api/status                             # System status
GET    /api/performance                        # Performance metrics
```

## ⚙️ Configuration

Edit `config.yaml`:

```yaml
# LLM Settings
llm:
  enabled: true
  model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
  temperature: 0.3

# Live Review
live_review:
  enabled: true
  run_time: "18:00"
  analysis_days: 1
  use_database: true

# Autonomous System
autonomous:
  enabled: false  # Set true after testing
  auto_apply_changes: false
  require_human_approval: true
```

## 🔐 Safety Rules

✅ **SPY Futures Focus**: System optimized for ES/MES  
✅ **Human Review**: All AI recommendations require approval  
✅ **Audit Trail**: Complete logging of all decisions  
✅ **One Change at a Time**: Incremental parameter updates  
✅ **Rollback Ready**: Quick reversion if needed  
❌ **No Blind Auto-Apply**: Never apply all suggestions at once  

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| `docs/SPY_FUTURES_REVIEW_GUIDE.md` | Complete SPY system guide |
| `docs/SPY_FUTURES_QUICK_REF.md` | SPY quick reference |
| `docs/LIVE_TRADING_REVIEW_GUIDE.md` | General review guide |
| `docs/QUICK_REFERENCE.md` | General quick reference |
| `docs/AUTONOMOUS_TRADING_GUIDE.md` | Autonomous system guide |
| `docs/HOW_TO_START.md` | Getting started |
| `docs/WINDOWS_TASK_SCHEDULER_SETUP.md` | Windows scheduling |

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- Node.js 16+ (for dashboard)
- AWS account with Bedrock access
- IBKR paper trading account

### Installation
```bash
# 1. Clone repository
git clone <repository-url>
cd MyTrader

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure settings
cp config.example.yaml config.yaml
# Edit config.yaml with your settings

# 5. Setup AWS credentials
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"

# 6. Install dashboard dependencies
cd dashboard/frontend
npm install
cd ../..

# 7. Test setup
python run_spy_futures_review.py --help
```

## 📅 Scheduled Execution

### Linux/macOS (Cron)
```bash
# Edit crontab
crontab -e

# Add daily 6 PM execution
0 18 * * * cd /path/to/MyTrader && python run_spy_futures_review.py
```

### Windows (Task Scheduler)
```
1. Open Task Scheduler
2. Create Task: "SPY Futures Review"
3. Trigger: Daily 6:00 PM
4. Action: python.exe run_spy_futures_review.py
5. Start in: C:\path\to\MyTrader
```

See `docs/WINDOWS_TASK_SCHEDULER_SETUP.md` for details.

## 🧪 Testing

```bash
# Test SPY review
python run_spy_futures_review.py --days 1 -v

# Test dashboard connection
curl http://localhost:8000/api/status

# Run test suite
python -m pytest tests/

# Verify setup
python verify_optimization_setup.py
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No trades found | Check DB exists, verify symbol filter |
| LLM call fails | Check AWS credentials, verify Bedrock enabled |
| Dashboard not updating | Ensure backend running on port 8000 |
| Import errors | Activate virtual environment |

See individual guide docs for detailed troubleshooting.

## 📈 Performance Metrics

| Metric | Good Target |
|--------|-------------|
| Win Rate | > 55% |
| Profit Factor | > 1.5 |
| Max Drawdown | < $1,000 |
| Avg Hold Time | 30-90 min |

## 🔄 Workflow

```
1. Market Close (4:00 PM ET)
   ↓
2. Review Runs (6:00 PM ET)
   - Load trades
   - Calculate metrics
   - Generate AI insights
   - Push to dashboard
   ↓
3. Evening Review
   - Check dashboard
   - Review recommendations
   - Evaluate warnings
   ↓
4. Next Day
   - Monitor live trading
   - Observe impact
```

## 🤝 Contributing

This is a personal trading system. For questions or issues, refer to documentation in `docs/`.

## 📄 License

Private project - All rights reserved

## 🆘 Support

- **Logs**: `logs/trading.log`
- **Dashboard Logs**: `dashboard/backend/logs/`
- **Documentation**: `docs/` directory
- **Quick Help**: `python run_spy_futures_review.py --help`

---

**Version**: 1.0  
**Status**: Production Ready ✅  
**Focus**: SPY Futures (ES/MES)  
**Last Updated**: November 6, 2025
