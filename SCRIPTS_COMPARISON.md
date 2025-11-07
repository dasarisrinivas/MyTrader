# MyTrader Startup Scripts Comparison

## 📋 Quick Reference

| Script | Dashboard Backend | Dashboard Frontend | Trading Bot | Use Case |
|--------|------------------|-------------------|-------------|----------|
| `./start_all.sh` | ✅ | ✅ | ✅ | **Production - Start everything** |
| `./start_dashboard.sh` | ✅ | ✅ | ❌ | Monitor existing bot / Development |
| `./start_trading.sh` | ❌ | ❌ | ✅ | Bot only (no dashboard) |
| `./stop.sh` | 🛑 | 🛑 | 🛑 | Stop all services |

## 🎯 Which Script Should I Use?

### Use `./start_all.sh` when:
- ✅ You want to start trading with full visibility
- ✅ First time running the system
- ✅ Beginning of trading day
- ✅ You want the complete experience

**What you get:**
- Dashboard to monitor everything
- Trading bot executing trades
- Real-time updates and AI insights
- All features working together

**Command:**
```bash
./start_all.sh
```

---

### Use `./start_dashboard.sh` when:
- 📊 Bot is already running elsewhere
- 👀 You just want to monitor/visualize
- 🛠️ You're developing dashboard features
- 🐛 You're debugging the frontend

**What you get:**
- Dashboard only
- Can view existing trades
- Can see bot status (if bot running separately)
- No trading functionality

**Command:**
```bash
./start_dashboard.sh
```

---

### Use `./start_trading.sh` when:
- 🤖 You only want the bot (no UI)
- 💻 Running on a headless server
- 📉 You prefer command-line monitoring
- 🔒 You want minimal resource usage

**What you get:**
- Trading bot only
- Console/log monitoring
- Lower memory footprint
- No browser needed

**Command:**
```bash
./start_trading.sh
```

---

### Use `./stop.sh` when:
- 🛑 End of trading day
- 🔄 Need to restart services
- 🐛 Troubleshooting issues
- 💤 Closing everything down

**What it does:**
- Gracefully stops all services
- Closes positions (if configured)
- Frees up ports 8000, 5173
- Cleans up PID files

**Command:**
```bash
./stop.sh
```

## 🔄 Common Workflows

### Morning Trading Routine
```bash
# 1. Start IB Gateway
# 2. Start everything
./start_all.sh --yes

# 3. Browser opens automatically to http://localhost:5173
```

### Development Mode (Frontend)
```bash
# Terminal 1: Start bot
./start_trading.sh

# Terminal 2: Start dashboard in dev mode
cd dashboard/frontend
npm run dev
```

### Headless Server
```bash
# Just run the bot (no dashboard)
./start_trading.sh

# Monitor via logs
tail -f logs/live_trading.log
```

### Restart Everything
```bash
# Stop all services
./stop.sh

# Wait a moment
sleep 3

# Start everything again
./start_all.sh --yes
```

## 📊 Resource Usage Comparison

| Configuration | CPU | RAM | Ports Used | Browser Required |
|--------------|-----|-----|------------|------------------|
| All Components | ~15% | ~500MB | 8000, 5173 | Yes |
| Dashboard Only | ~5% | ~200MB | 8000, 5173 | Yes |
| Bot Only | ~8% | ~150MB | None | No |

## 🎨 Visual Flow

```
┌─────────────────────────────────────────────────────────┐
│                    ./start_all.sh                       │
│                   (RECOMMENDED)                          │
└──────────────┬─────────────┬─────────────┬──────────────┘
               │             │             │
               ▼             ▼             ▼
         ┌─────────┐   ┌─────────┐   ┌─────────┐
         │ Backend │   │Frontend │   │   Bot   │
         │  :8000  │   │  :5173  │   │  Live   │
         └────┬────┘   └────┬────┘   └────┬────┘
              │             │             │
              └─────────────┼─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Browser    │
                    │ Dashboard UI │
                    └──────────────┘
```

```
┌─────────────────────────────────────────────────────────┐
│               ./start_dashboard.sh                       │
│            (Monitoring/Development)                      │
└──────────────┬─────────────┬──────────────────────────────┘
               │             │
               ▼             ▼
         ┌─────────┐   ┌─────────┐
         │ Backend │   │Frontend │
         │  :8000  │   │  :5173  │
         └────┬────┘   └────┬────┘
              │             │
              └─────────────┘
                     │
                     ▼
              ┌──────────────┐
              │   Browser    │
              │ Dashboard UI │
              └──────────────┘
```

```
┌─────────────────────────────────────────────────────────┐
│               ./start_trading.sh                         │
│              (Headless/Server)                           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
                 ┌─────────┐
                 │   Bot   │
                 │  Live   │
                 └────┬────┘
                      │
                      ▼
               ┌─────────────┐
               │     Logs    │
               │ tail -f ... │
               └─────────────┘
```

## 🔐 Safety Features

### start_all.sh Safety Checks
- ✅ Verifies IB Gateway is running
- ✅ Confirms you want to start trading
- ✅ Checks for existing services
- ✅ Validates config file
- ✅ Tests dependencies
- ✅ Warns about live vs paper trading

### start_trading.sh Safety Checks  
- ✅ Requires IB Gateway
- ✅ Multiple confirmations before trading
- ✅ Shows account mode (paper/live)
- ✅ Validates config

### start_dashboard.sh Safety Checks
- ✅ Can run without IB Gateway
- ✅ Installs missing dependencies
- ✅ Checks for port conflicts

## 📝 Examples

### Example 1: Full System Startup
```bash
# Start everything (asks for confirmation)
./start_all.sh

# Or skip confirmations
./start_all.sh --yes
```

**Output:**
```
╔════════════════════════════════════════════════════════════════╗
║         🚀 MyTrader - Complete System Startup 🚀           ║
╚════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    Step 1: Pre-flight Checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/6] Checking virtual environment... OK
[2/6] Checking config file... OK
[3/6] Checking IB Gateway/TWS... OK
[4/6] Checking Python dependencies... OK
[5/6] Checking Node dependencies... OK
[6/6] Checking for existing services... OK

✅ All pre-flight checks passed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    Step 2: Starting Services
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/3] Starting Dashboard Backend API...
      ✅ Backend started (PID: 12345)
      URL: http://localhost:8000
      Waiting for backend to be ready. OK

[2/3] Starting Dashboard Frontend...
      ✅ Frontend started (PID: 12346)
      URL: http://localhost:5173
      Waiting for frontend to be ready........ OK

[3/3] Starting Trading Bot...
      ✅ Bot started (PID: 12347)
      Config: /Users/svss/Documents/code/MyTrader/config.yaml
      Initializing trading bot..... OK

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  ✅ All Services Running! ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Service Status:
  📊 Dashboard Frontend:  http://localhost:5173
  🔌 Backend API:         http://localhost:8000
  📡 WebSocket:           ws://localhost:8000/ws
  📚 API Docs:            http://localhost:8000/docs
  🤖 Trading Bot:         Active (PID: 12347)
```

### Example 2: Dashboard Only
```bash
./start_dashboard.sh
```

**Use case:** Bot is already running, you just want to see the dashboard

### Example 3: Bot Only (Headless)
```bash
# Start bot only
./start_trading.sh

# Monitor in another terminal
tail -f logs/live_trading.log
```

**Use case:** Running on a VPS without GUI

### Example 4: Stop Everything
```bash
./stop.sh
```

**Output:**
```
🛑 Stopping MyTrader services...

✅ Stopped backend (PID: 12345)
✅ Stopped frontend (PID: 12346)
Gracefully stopping trading bot...
✅ Stopped bot (PID: 12347)

✨ All services stopped
Dashboard and trading bot are no longer running
```

## 🆘 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| "Port already in use" | Run `./stop.sh` then try again |
| "IB Gateway not running" | Start IB Gateway first |
| "Config not found" | Copy `config.example.yaml` to `config.yaml` |
| Dashboard shows no data | Wait 10 seconds, check bot is running |
| Bot not trading | Verify IB Gateway settings, check logs |

## 💡 Pro Tips

1. **Always use `./start_all.sh --yes` for automation**
2. **Check logs first when troubleshooting:** `tail -f logs/*.log`
3. **Use `./stop.sh` before restarting services**
4. **Monitor with:** `ps aux | grep -E 'dashboard_api|main.py live|vite'`
5. **For development, start components individually**

---

**Remember:** `./start_all.sh` is your one-stop-shop for complete system startup! 🚀
