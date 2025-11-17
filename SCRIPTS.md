# MyTrader Scripts

Three simple commands to control your trading system:

## 🤖 Start Bot Only
```bash
./start_bot.sh
```
Starts the trading bot connected to IB Gateway.
- Requires: IB Gateway/TWS running on port 4002
- Logs: `logs/bot.log`
- Does NOT start the dashboard

## 📊 Start Dashboard Only
```bash
./start_dashboard.sh
```
Starts the web dashboard (backend + frontend).
- Backend: http://localhost:8000
- Frontend: http://localhost:5173
- WebSocket: ws://localhost:8000/ws
- Does NOT start the bot (you can start it from the dashboard UI)

## 🛑 Stop Everything
```bash
./stop.sh
```
Stops all running services:
- Trading bot
- Dashboard backend
- Dashboard frontend
- Frees ports 8000 and 5173

---

## Quick Start

### Option 1: Bot + Dashboard (recommended)
```bash
# Terminal 1: Start dashboard
./start_dashboard.sh

# Terminal 2: Open browser to http://localhost:5173
# Click "Start Bot" button in the UI
```

### Option 2: Bot Only (no dashboard)
```bash
./start_bot.sh
tail -f logs/bot.log  # Watch logs
```

### Stop All Services
```bash
./stop.sh
```

---

## Logs

- **Bot**: `logs/bot.log`
- **Dashboard Backend**: `logs/backend.log`
- **Dashboard Frontend**: `logs/frontend.log`

View live logs:
```bash
tail -f logs/bot.log
tail -f logs/backend.log
```

---

## Troubleshooting

### Bot won't connect to IB Gateway
1. Check IB Gateway is running: `lsof -i:4002`
2. Verify API is enabled in IB Gateway settings
3. Check port in `config.yaml` matches IB Gateway (4002 for paper, 4001 for live)

### Dashboard shows "bot not running"
- If you started the bot with `./start_bot.sh`, the dashboard can monitor it but won't manage it
- To let dashboard manage the bot, click "Start Bot" in the dashboard UI instead

### Ports already in use
```bash
./stop.sh  # This will free all ports
```
