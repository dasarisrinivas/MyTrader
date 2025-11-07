# 🚀 MyTrader - Complete System Startup

## Quick Start

To start **everything** (dashboard + trading bot) with one command:

```bash
./start_all.sh
```

This will start:
1. ✅ Dashboard Backend API (port 8000)
2. ✅ Dashboard Frontend (port 5173)
3. ✅ Trading Bot (live trading)

## Prerequisites

Before running `start_all.sh`, ensure:

1. **IB Gateway or TWS is running**
   - Open IB Gateway/TWS
   - Login with Paper Trading account
   - Configure API settings:
     - Port: 4002
     - Enable Socket Clients
     - Add 127.0.0.1 to Trusted IPs
     - **UNCHECK "Read-Only API"**

2. **Virtual environment is set up**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Config file exists**
   ```bash
   cp config.example.yaml config.yaml
   # Edit config.yaml with your settings
   ```

## Available Scripts

### Start Everything
```bash
./start_all.sh              # Interactive mode (asks for confirmation)
./start_all.sh --yes        # Skip all confirmations
./start_all.sh --config custom.yaml  # Use custom config file
```

### Start Components Individually

#### Dashboard Only
```bash
./start_dashboard.sh        # Just the dashboard (no trading bot)
```

#### Trading Bot Only
```bash
./start_trading.sh          # Just the bot (no dashboard)
```

### Stop Everything
```bash
./stop.sh                   # Stops all services gracefully
```

## What Gets Started

### 1. Dashboard Backend API
- **URL:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **WebSocket:** ws://localhost:8000/ws
- **Log:** `logs/backend.log`
- **PID File:** `logs/backend.pid`

### 2. Dashboard Frontend
- **URL:** http://localhost:5173
- **Log:** `logs/frontend.log`
- **PID File:** `logs/frontend.pid`

### 3. Trading Bot
- **Process:** `python3 main.py live`
- **Config:** `config.yaml` (or custom)
- **Logs:** 
  - `logs/bot.log` (main process)
  - `logs/live_trading.log` (trading activity)
- **PID File:** `logs/bot.pid`

## Monitoring

### View Live Logs

**Bot logs:**
```bash
tail -f logs/bot.log
tail -f logs/live_trading.log
```

**Backend logs:**
```bash
tail -f logs/backend.log
```

**All logs:**
```bash
tail -f logs/*.log
```

### Check Service Status
```bash
ps aux | grep -E 'dashboard_api|main.py live|vite'
```

### Check Specific Component
```bash
# Check if bot is running
pgrep -f "main.py live"

# Check if backend is running
curl http://localhost:8000/api/status

# Check if frontend is running
curl http://localhost:5173
```

## Typical Workflow

### 1. Morning Startup
```bash
# Start IB Gateway first
# Then start everything
./start_all.sh --yes
```

### 2. During Trading Hours
- Monitor dashboard at http://localhost:5173
- Check bot logs: `tail -f logs/live_trading.log`
- View AI decisions in "AI Intelligence" tab
- Track trades in "Trade Trail" tab

### 3. End of Day Shutdown
```bash
./stop.sh
# Then close IB Gateway
```

## Troubleshooting

### "IB Gateway not running" error
Start IB Gateway/TWS and ensure:
- It's logged in
- API settings are configured (port 4002)
- "Read-Only API" is UNCHECKED

### "Port already in use" error
```bash
# Stop all services
./stop.sh

# If that doesn't work, force kill:
lsof -ti:8000 | xargs kill -9
lsof -ti:5173 | xargs kill -9
pkill -9 -f "main.py live"
```

### "Virtual environment not found" error
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Bot not trading
Check these in order:
1. IB Gateway is running and logged in
2. API port 4002 is accessible
3. "Read-Only API" is UNCHECKED in IB settings
4. Bot logs show connection success: `grep "Connected to IB" logs/bot.log`
5. Check for errors: `tail -30 logs/bot.log`

### Dashboard shows no data
1. Wait 5-10 seconds for initial data load
2. Check backend is running: `curl http://localhost:8000/api/status`
3. Check WebSocket connection in browser console (F12)
4. Verify bot is running: `pgrep -f "main.py live"`

## Features of start_all.sh

✅ **Pre-flight checks**
- Verifies virtual environment
- Checks config file
- Tests IB Gateway connection
- Installs missing dependencies
- Checks for existing services

✅ **Graceful startup**
- Starts services in correct order
- Waits for each service to be ready
- Provides real-time status updates

✅ **Process management**
- Saves PID files for each service
- Logs all output to separate files
- Supports graceful shutdown

✅ **User-friendly**
- Color-coded output
- Clear status messages
- Interactive confirmations (can be skipped with --yes)
- Opens browser automatically

✅ **Safety features**
- Warns about live trading
- Confirms before stopping existing services
- Validates IB Gateway connection
- Provides fallback if services fail

## Advanced Usage

### Run in Background (No Monitoring)
```bash
./start_all.sh --yes && exit
```

### Custom Config File
```bash
./start_all.sh --config my_custom_config.yaml
```

### Automatic Startup (cron)
```bash
# Add to crontab (opens trading at 9:00 AM weekdays)
0 9 * * 1-5 cd /path/to/MyTrader && ./start_all.sh --yes
```

### Automatic Shutdown (cron)
```bash
# Add to crontab (closes trading at 4:00 PM weekdays)
0 16 * * 1-5 cd /path/to/MyTrader && ./stop.sh
```

## Summary of Files

| Script | Purpose | What It Starts |
|--------|---------|----------------|
| `start_all.sh` | **Start everything** | Backend + Frontend + Bot |
| `start_dashboard.sh` | Dashboard only | Backend + Frontend |
| `start_trading.sh` | Bot only | Trading Bot |
| `stop.sh` | Stop everything | Stops all services |

## Next Steps

After starting with `./start_all.sh`:

1. **Open Dashboard:** http://localhost:5173
2. **Check Bot Status:** Green "Bot Active" badge should appear
3. **Monitor Trades:** Go to "Trade Trail" tab
4. **View AI Decisions:** Go to "AI Intelligence" tab
5. **Watch Charts:** Go to "Analytics" tab

Enjoy automated trading! 🚀
