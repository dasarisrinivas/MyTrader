# Quick Start Guide - New Bot Analytics Dashboard

## 🚀 Starting the Dashboard

### Step 1: Start the Backend API
```bash
cd /Users/svss/Documents/code/MyTrader/dashboard/backend
python dashboard_api.py
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 2: Start the Frontend
```bash
cd /Users/svss/Documents/code/MyTrader/dashboard/frontend
npm run dev
```

Expected output:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Step 3: Open in Browser
Navigate to: **http://localhost:5173**

## 🎯 Using the Dashboard

### Overview Tab (Default)
**What you'll see:**
- Total P&L (green if positive, red if negative)
- Today's trade count
- Current open position
- Active orders
- Symbol information (ES Futures)

**Actions:**
- Monitor real-time performance
- See at-a-glance bot status

### AI Intelligence Tab
**What you'll see:**
- Current trading signal (BUY/SELL/HOLD)
- Confidence percentage with bar
- AI reasoning explanation
- Market sentiment gauge
- Last trade decision details

**Actions:**
- Understand why the bot made decisions
- Monitor sentiment shifts
- Review entry/exit logic

### Trade Trail Tab
**What you'll see:**
- Chronological list of all trades
- Time, action, quantity, price, P&L
- Click to expand for full details
- AI decision summary
- Technical indicators (confidence, sentiment)

**Actions:**
- Review trade history
- Analyze winning vs losing trades
- Understand patterns

### Analytics Tab
**What you'll see:**
Three interactive charts:
1. **Price Chart** - Shows entry/exit points as dots
2. **Sentiment Trend** - Area chart of sentiment over time
3. **Cumulative Profit** - Equity curve

**Actions:**
- Visualize trading patterns
- Track sentiment changes
- Monitor profit progression

### Backtest Tab
**What you'll see:**
- Same as before: historical backtesting tools

**Actions:**
- Run historical simulations
- Optimize parameters

## 🎮 Bot Controls

### Starting the Bot
1. Click the **green "Start Bot"** button in the top right
2. Wait for "Bot Active" indicator to appear
3. Monitor the Overview tab for first signals

### Stopping the Bot
1. Click the **red "Stop Bot"** button in the top right
2. Bot will gracefully shutdown
3. All positions will be closed (if configured)

## 🔍 Monitoring Bot Health

**Location:** Top of every page (Bot Health Indicator)

**Status Meanings:**
- 🟢 **Healthy** - Everything working perfectly
- 🟡 **WebSocket Offline** - API working, WebSocket down (slower updates)
- 🟠 **Stale Connection** - No updates received recently
- 🔴 **Disconnected** - Bot or API offline

**Metrics:**
- **Latency** - Response time (green < 100ms, yellow < 300ms, red > 300ms)
- **Last Update** - Time since last heartbeat

## 📊 Understanding the Data

### Signal Types
- **BUY** 🟢 - Bot wants to enter long position
- **SELL** 🔴 - Bot wants to enter short position
- **HOLD** 🟡 - Bot waiting for clearer signal

### Confidence Levels
- **>70%** - High confidence (strong signal)
- **50-70%** - Medium confidence (moderate signal)
- **<50%** - Low confidence (weak signal)

### Sentiment Scores
- **+0.3 to +1.0** - Bullish
- **-0.3 to +0.3** - Neutral
- **-1.0 to -0.3** - Bearish

### P&L Colors
- **Green** - Profitable
- **Red** - Loss
- **Gray** - Break-even

## 🐛 Troubleshooting

### Dashboard Won't Load
1. Check backend is running: http://localhost:8000/api/status
2. Check frontend terminal for errors
3. Clear browser cache and reload

### No Data Showing
1. Verify bot is running (green "Bot Active" badge)
2. Check Bot Health Indicator status
3. Wait 5-10 seconds for initial data load

### WebSocket Issues
- Indicator shows "WebSocket Offline"
- Dashboard will fall back to polling
- Try refreshing the page
- Check firewall/network settings

### Charts Not Updating
1. Wait for bot to collect minimum bars (usually 15-50)
2. Check Analytics tab for "collecting data" message
3. Verify trades are being executed (Trade Trail tab)

## 💡 Tips & Best Practices

### Monitoring
- Keep Overview tab open for at-a-glance status
- Check AI Intelligence tab before major decisions
- Review Trade Trail at end of day

### Performance
- Dashboard updates every 2-5 seconds automatically
- No need to manually refresh
- Multiple tabs can be open simultaneously

### Decision Making
- AI reasoning provides context for each signal
- Sentiment helps understand market mood
- Confidence indicates signal strength

## 🔧 Configuration

### Changing Update Frequency
Edit component files:
```javascript
// In each component's useEffect
const interval = setInterval(fetchData, 3000); // 3 seconds
```

### Adjusting Chart Time Window
```javascript
// In RealTimeCharts.jsx
setPriceData(mockPriceData.slice(-30)); // Last 30 points
// Change -30 to -60 for more history
```

### Customizing Colors
Edit Tailwind classes in components:
- Green: `text-green-400`, `bg-green-900/30`
- Red: `text-red-400`, `bg-red-900/30`
- Blue: `text-blue-400`, `bg-blue-900/30`

## 📞 Support

If issues persist:
1. Check main.py bot logs
2. Check dashboard/backend logs
3. Verify all dependencies installed
4. Ensure ports 8000 and 5173 are available

## 🎉 Enjoy Your New Dashboard!

You now have a professional, AI-focused trading dashboard that shows exactly what your bot is thinking and doing—without any manual trading distractions!
