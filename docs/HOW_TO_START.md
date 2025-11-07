# 🚀 How to Start MyTrader

## Quick Reference

MyTrader has **three main modes** of operation:

| Mode | Command | Purpose | IB Gateway Required? |
|------|---------|---------|---------------------|
| **Dashboard Only** | `./start.sh` | View metrics, monitor system | ⚠️ Optional (warns if missing) |
| **Dashboard** | `./start_dashboard.sh` | Dashboard with trading UI | ⚠️ Optional (warns if missing) |
| **Live Trading** | `./start_trading.sh` | Execute real trades | ✅ **REQUIRED** |

---

## 🎯 Recommended Startup Sequence

### Step 1: Start IB Gateway (If Trading)

**Only needed for live trading. Skip if just viewing dashboard.**

1. **Open IB Gateway** application
2. **Select mode:**
   - ✅ **Paper Trading** (recommended for testing)
   - ⚠️ Live Trading (real money)
3. **Configure API Settings:**
   - Go to: Edit → Global Configuration → API → Settings
   - ✅ Enable "ActiveX and Socket Clients"
   - ✅ Set Socket port to **4002** (paper) or **7497** (live)
   - ✅ Add `127.0.0.1` to Trusted IPs
   - ✅ **UNCHECK "Read-Only API"** (important!)
4. **Login** with your IBKR credentials

### Step 2: Choose Your Startup Mode

#### 🎨 **Option A: Dashboard Only** (Monitoring & Analysis)

```bash
./start.sh
```

**What it starts:**
- ✅ Backend API (FastAPI) on http://localhost:8000
- ✅ Frontend Dashboard (React) on http://localhost:5173
- ✅ Automatically opens in your browser

**Use this when:**
- You want to view performance metrics
- Analyze past trades and strategies
- Monitor system health
- No actual trading

---

#### 📊 **Option B: Dashboard with Trading UI** (Monitor + Trade)

```bash
./start_dashboard.sh
```

**What it starts:**
- ✅ Backend API with IB Gateway connection
- ✅ Frontend Dashboard with live trading features
- ✅ Real-time market data display
- ✅ Order management interface

**Use this when:**
- You want the full dashboard experience
- Monitor live trading if bot is running
- Manually place/cancel orders
- View real-time P&L

---

#### 🤖 **Option C: Live Trading Bot** (Automated Trading)

```bash
./start_trading.sh
```

**What it does:**
- ✅ Connects to IB Gateway
- ✅ Streams market data
- ✅ Executes trading strategies automatically
- ✅ Manages positions and risk
- ✅ Logs all trades

**⚠️ WARNING:** This executes REAL trades!

**Use this when:**
- You're ready for automated trading
- IB Gateway is running
- Strategies are tested and configured
- Risk management is set up

---

## 🔧 Configuration Files

### Main Config: `config.yaml`

Your current configuration:

```yaml
# LLM is ENABLED ✅
llm:
  enabled: true
  model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
  min_confidence_threshold: 0.7
  override_mode: false  # Consensus mode

# Trading Settings
trading:
  max_position_size: 2
  max_daily_loss: 1500.0
  max_daily_trades: 20

# IB Gateway Connection
data:
  ibkr_port: 4002  # Paper trading
  ibkr_symbol: "ES"
  ibkr_exchange: "CME"
```

**To modify:** Edit `config.yaml` and restart services

---

## 📝 Common Workflows

### Workflow 1: Test LLM Integration (No Trading)

```bash
# 1. Test AWS Bedrock connection
python3 test_aws_bedrock.py

# 2. Run LLM examples
python3 example_llm_integration.py

# 3. Start dashboard to view
./start.sh
```

### Workflow 2: Paper Trading with Dashboard

```bash
# 1. Start IB Gateway (Paper Trading mode, port 4002)
# 2. Start dashboard
./start_dashboard.sh

# 3. In another terminal, start trading bot
./start_trading.sh

# 4. Monitor in browser at http://localhost:5173
```

### Workflow 3: Backtest Before Live Trading

```bash
# 1. Run backtest on historical data
python3 demo_backtest.py

# 2. Review results in reports/ directory

# 3. If satisfied, proceed to paper trading
./start_trading.sh
```

### Workflow 4: Monitor Only (No Trading)

```bash
# Just view past performance and system stats
./start.sh
```

---

## 🛑 How to Stop

### Stop All Services

```bash
./stop.sh
```

This kills:
- Backend API
- Frontend dashboard
- Any background processes

### Stop Trading Bot

Press **Ctrl+C** in the terminal running `./start_trading.sh`

---

## 📊 Accessing the Dashboard

Once started, open your browser to:

- **Dashboard UI:** http://localhost:5173
- **API Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## 🐛 Troubleshooting

### Issue: "Port already in use"

```bash
# Kill processes on ports 8000 and 5173
./stop.sh

# Or manually:
lsof -ti:8000 | xargs kill -9
lsof -ti:5173 | xargs kill -9
```

### Issue: "IB Gateway not running"

```bash
# Check if IB Gateway is listening
lsof -Pi :4002 -sTCP:LISTEN

# If not, start IB Gateway application
```

### Issue: "AWS Bedrock errors"

```bash
# Test your AWS connection
python3 test_aws_bedrock.py

# Reconfigure if needed
aws configure
```

### Issue: "Dependencies missing"

```bash
# Reinstall Python dependencies
source .venv/bin/activate
pip install -r requirements.txt

# Reinstall Node dependencies
cd dashboard/frontend
npm install
```

### Issue: "Trading bot can't connect to IB"

1. Check IB Gateway is running on correct port
2. Verify API settings:
   - "Read-Only API" is **UNCHECKED**
   - Socket Clients enabled
   - Correct port (4002 for paper, 7497 for live)
3. Check firewall isn't blocking connections

---

## 📖 Detailed Documentation

- **LLM Integration:** `LLM_INTEGRATION.md`
- **Code Review:** `LLM_INTEGRATION_REVIEW.md`
- **Verification:** `VERIFICATION_COMPLETE.md`
- **README:** `README.md`
- **Restart Guide:** `RESTART_AND_PNL_GUIDE.md`

---

## 🎯 Recommended First Steps

### For New Users:

1. **Test without trading:**
   ```bash
   ./start.sh
   ```

2. **Explore the dashboard:**
   - View performance metrics
   - Check strategy parameters
   - Review historical trades

3. **Test AWS Bedrock:**
   ```bash
   python3 test_aws_bedrock.py
   python3 example_llm_integration.py
   ```

4. **Run a backtest:**
   ```bash
   python3 demo_backtest.py
   ```

5. **When ready, paper trade:**
   - Start IB Gateway (Paper mode)
   - Run `./start_trading.sh`
   - Monitor for a few days

6. **Only after success, consider live trading**

---

## ⚙️ Advanced Options

### Custom Config File

```bash
./start_trading.sh --config my_custom_config.yaml
```

### Start Without Browser

```bash
./start.sh --no-browser
```

### View Logs

```bash
# Live logs
tail -f logs/backend.log
tail -f logs/frontend.log

# Both at once
tail -f logs/*.log
```

---

## 🔐 Security Notes

- ✅ **AWS credentials** stored in `~/.aws/credentials`
- ✅ **IB credentials** entered in IB Gateway (not stored in code)
- ✅ **API keys** in `config.yaml` (add to `.gitignore`)
- ⚠️ Never commit credentials to git
- ⚠️ Always start with paper trading first

---

## 💡 Pro Tips

1. **Always test in paper trading first** - Even with LLM enabled
2. **Monitor for 1-2 weeks** before going live
3. **Start with small position sizes** - Increase gradually
4. **Review LLM recommendations** in logs before trusting fully
5. **Keep daily loss limits low** initially
6. **Use consensus mode** (not override) until confident
7. **Check logs regularly** - `tail -f logs/*.log`
8. **Have an exit plan** - Know when to stop the bot

---

## 📞 Quick Help

| Question | Answer |
|----------|--------|
| How do I just view the dashboard? | `./start.sh` |
| How do I actually trade? | `./start_trading.sh` (requires IB Gateway) |
| How do I stop everything? | `./stop.sh` |
| Where are the logs? | `logs/backend.log` and `logs/frontend.log` |
| Is LLM enabled? | Yes! Check `config.yaml` → `llm.enabled: true` |
| How much does LLM cost? | ~$7/month for 20 trades/day |
| Can I test without money? | Yes! Use Paper Trading mode in IB Gateway |

---

## 🎉 You're Ready!

Your system is configured with:
- ✅ AWS Bedrock LLM integration
- ✅ Claude 3 Sonnet model
- ✅ Consensus mode (safe)
- ✅ 0.7 confidence threshold
- ✅ Conservative risk limits

**Start with:**
```bash
./start.sh
```

Then explore the dashboard and decide when you're ready for live trading!
